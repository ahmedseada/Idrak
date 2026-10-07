// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Diagnostics;
using Idrak.Abstraction.Devices;
using Idrak.Generation.Abstractions;
using Idrak.Layers;
using Idrak.Layers.Abstractions;

namespace Idrak.Generation;

/// <summary>A piece of one prompt's continuation in <see cref="TextGenerator.StreamBatch"/>.</summary>
/// <param name="Index">The prompt's index in the batch.</param>
/// <param name="Text">Newly generated text (empty on the final chunk).</param>
/// <param name="Done">True for the prompt's final chunk.</param>
/// <param name="DoneReason">"stop" or "length" on the final chunk.</param>
/// <param name="Stats">Statistics, on the final chunk only.</param>
public sealed record BatchChunk(int Index, string Text, bool Done = false, string? DoneReason = null, GenerationStats? Stats = null);

/// <summary>
/// Autoregressive text generation for a causal language model built as a <see cref="Sequential"/> of
/// <see cref="ICachedModule"/>-capable layers (embedding, positional encoding, causal transformer blocks, norm, head).
/// Handles prompt truncation to the context window, KV-cache decoding with optional graph replay, on-device sampling
/// with every <see cref="GenerationOptions"/> filter and penalty, stop sequences and streaming.
/// </summary>
/// <example>
/// <code>
/// var generator = new TextGenerator(model, new CharTokenizer(vocabulary), contextLength: 64);
/// foreach (var chunk in generator.Stream("once upon a time", new GenerationOptions { NumPredict = 200 }))
///     Console.Write(chunk.Text);
/// </code>
/// </example>
public sealed class TextGenerator(Sequential model, ITokenizer tokenizer, int contextLength) : ITextModel
{
    /// <summary>Upper bound on generated tokens when <see cref="GenerationOptions.NumPredict"/> is negative.</summary>
    public const int MaxTokens = 4096;

    /// <summary>The language model.</summary>
    public Sequential Model { get; } = model;

    /// <summary>The tokenizer matching the model's vocabulary.</summary>
    public ITokenizer Tokenizer { get; } = tokenizer;

    /// <summary>The model's maximum context (its positional-encoding length).</summary>
    public int ContextLength { get; } = contextLength;

    /// <summary>The device the model's parameters live on.</summary>
    public Device Device => Model.Parameters().Concat(Model.Buffers()).First().Device;

    /// <summary>
    /// How the KV cache stores keys and values while generating: <see cref="KeyValueFormat.Int8"/> takes about a quarter
    /// of the memory (longer contexts, more parallel sequences) at a small cost in accuracy.
    /// </summary>
    public KeyValueFormat CacheFormat { get; init; } = KeyValueFormat.Float32;

    /// <summary>
    /// The KV cache's layout when set (a format of one's own, or a built-in one by name with <see cref="KeyValueLayouts.Get"/>);
    /// it takes precedence over <see cref="CacheFormat"/>. Default null: the layout of <see cref="CacheFormat"/>.
    /// </summary>
    public KeyValueLayout? CacheLayout { get; init; }

    // The layout caches are created with.
    private KeyValueLayout Layout => CacheLayout ?? KeyValueLayouts.For(CacheFormat);

    /// <summary>
    /// Creates the sampler each generation chooses tokens with (default <see cref="TokenSamplers.Create"/>: the sampler
    /// registered as "default", the built-in <see cref="TokenSampler"/> with temperature, top-k, top-p, min-p and penalties
    /// on the device, unless an app overrode it). Set it to plug in another <see cref="ITokenSampler"/> for this generator
    /// alone; one that is not <see cref="ITokenSampler.Recordable"/> turns off the CUDA graph of the decoding step.
    /// </summary>
    public Func<SamplerRequest, ITokenSampler> CreateSampler { get; set; } = TokenSamplers.Create;

    /// <summary>
    /// Keep the KV cache after each generation, so the next prompt that starts with the same tokens (the earlier turns of
    /// a conversation, a long system prompt or tool list) only processes what follows them. The cache stays allocated
    /// between calls; <see cref="ReleaseCache"/> frees it. Default true.
    /// </summary>
    public bool KeepCache { get; set; } = true;

    // Whether a decoding step can be recorded: not with experts, whose routing is read back to the host on every step.
    private bool StepsRecordable => _stepsRecordable ??= !Model.Descendants().Any(m => m is MixtureOfExperts);

    private bool? _stepsRecordable;

    private readonly Lock _keptLock = new();
    private DecodingContext? _kept;
    private List<int> _keptIds = [];

    /// <summary>Frees the KV cache kept between generations (see <see cref="KeepCache"/>).</summary>
    public void ReleaseCache()
    {
        lock (_keptLock)
        {
            _kept?.Dispose();
            _kept = null;
            _keptIds = [];
        }
    }

    // Takes the kept cache when it fits this generation (same capacity and format), with the ids it holds.
    private (DecodingContext? Context, List<int> Ids) TakeCache(int capacity)
    {
        lock (_keptLock)
        {
            var (context, ids) = (_kept, _keptIds);
            (_kept, _keptIds) = (null, []);
            if (context is not null && (context.Capacity != capacity || context.Layout != Layout || context.Device != Device))
            {
                context.Dispose();
                return (null, []);
            }

            return (context, ids);
        }
    }

    private void KeepCacheFor(DecodingContext context, List<int> ids)
    {
        lock (_keptLock)
        {
            _kept?.Dispose();
            (_kept, _keptIds) = (context, ids);
        }
    }

    /// <summary>Generates the whole continuation of <paramref name="prompt"/>.</summary>
    public (string Text, string DoneReason, GenerationStats Stats) Generate(string prompt, GenerationOptions options, CancellationToken cancellationToken = default)
    {
        var text = new System.Text.StringBuilder();
        foreach (var chunk in Stream(prompt, options, cancellationToken))
        {
            text.Append(chunk.Text);
            if (chunk.Done)
            {
                return (text.ToString(), chunk.DoneReason!, chunk.Stats!);
            }
        }

        throw new InvalidOperationException("Generation ended without a final chunk.");
    }

    /// <summary>
    /// Whether <see cref="GenerateBatch"/> decodes several prompts together on this model and device (attention from
    /// per-row starts: on CUDA, bfloat16 tensor cores with head size 64 or 128 and no attention layer with a sliding window
    /// that can mask or soft-capped scores); otherwise it runs them one by one.
    /// </summary>
    public bool SupportsBatches =>
        Model.Descendants().OfType<CausalSelfAttention>().ToList() is { Count: > 0 } attention
        && !Model.Descendants().Any(m => m is MultiHeadAttention or PositionalEncoding)
        && attention.All(a => a.SupportsSegmented(Device.Backend));

    /// <summary>
    /// Generates the continuations of several prompts together (one batch through the model per token, so the GPU
    /// reads each weight once for all of them): the prompts are padded on the left to end at the same position, and each
    /// row numbers its positions and attends from its own start, so every row gets what it would get alone. The same
    /// options apply to every prompt; repetition penalties are not supported (use <see cref="Generate"/>). Runs the
    /// prompts one by one when the model cannot batch them (<see cref="SupportsBatches"/>).
    /// </summary>
    public IReadOnlyList<(string Text, string DoneReason, GenerationStats Stats)> GenerateBatch(IReadOnlyList<string> prompts, GenerationOptions options,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(prompts);
        var results = new (string Text, string DoneReason, GenerationStats Stats)[prompts.Count];
        var texts = prompts.Select(_ => new System.Text.StringBuilder()).ToArray();
        foreach (var chunk in StreamBatch(prompts, options, cancellationToken))
        {
            texts[chunk.Index].Append(chunk.Text);
            if (chunk.Done)
            {
                results[chunk.Index] = (texts[chunk.Index].ToString(), chunk.DoneReason!, chunk.Stats!);
            }
        }

        return results;
    }

    /// <summary>
    /// <see cref="GenerateBatch"/> streamed: each prompt's text as it is generated (every <see cref="GenerationOptions.ChunkSize"/>
    /// tokens, pieces tagged with the prompt's index; a possible stop sequence is held back until it is ruled out), then one
    /// final chunk per prompt (<see cref="BatchChunk.Done"/>, with its reason and statistics) once all are done. The pieces
    /// of a prompt add up to the text <see cref="GenerateBatch"/> returns for it.
    /// </summary>
    public IEnumerable<BatchChunk> StreamBatch(IReadOnlyList<string> prompts, GenerationOptions options, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(prompts);
        if (prompts.Count == 0)
        {
            yield break;
        }

        bool penalties = options.RepeatPenalty != 1f || options.PresencePenalty != 0f || options.FrequencyPenalty != 0f;
        if (prompts.Count == 1 || penalties || !SupportsBatches)
        {
            for (int p = 0; p < prompts.Count; p++)
            {
                foreach (var chunk in Stream(prompts[p], options, cancellationToken))
                {
                    yield return new BatchChunk(p, chunk.Text, chunk.Done, chunk.DoneReason, chunk.Stats);
                }
            }

            yield break;
        }

        var total = Stopwatch.StartNew();
        int rows = prompts.Count;
        int context = Math.Clamp(options.NumCtx, 2, ContextLength);
        var tokens = prompts.Select(p =>
        {
            var ids = Tokenizer.Encode(p).ToList();
            if (ids.Count == 0)
            {
                ids.Add(0);
            }

            return ids.Count > context - 1 ? ids.GetRange(ids.Count - (context - 1), context - 1) : ids;
        }).ToList();
        int promptLength = tokens.Max(t => t.Count);
        int limit = Math.Min(options.NumPredict > 0 ? Math.Min(options.NumPredict, MaxTokens) : MaxTokens, context - promptLength);
        limit = Math.Max(1, limit);
        var stops = options.Stop.Where(x => x.Length > 0).ToArray();
        int holdBack = stops.Length == 0 ? 0 : stops.Max(x => x.Length) - 1;
        var stopValues = stops.Length == 0 ? null : SearchValues.Create(stops, StringComparison.Ordinal);
        var starts =tokens.Select(t => promptLength - t.Count).ToArray();
        var input = new float[rows * promptLength];
        for (int r = 0; r < rows; r++)
        {
            for (int i = 0; i < tokens[r].Count; i++)
            {
                input[r * promptLength + starts[r] + i] = tokens[r][i];
            }
        }

        using var sampler = CreateSampler(new SamplerRequest(Device, rows, Tokenizer.VocabularySize, limit + 1, 1, options));   // no penalties here
        // Rows of different lengths need a cache whose format can attend from per-row starts (float32 today).
        var batchLayout = Layout.RowStarts ? Layout : KeyValueLayouts.For(KeyValueFormat.Float32);
        using var decoding = new DecodingContext(Device, rows, promptLength + limit, batchLayout) { LastPositionOnly = true };
        decoding.SetRowStarts(starts);
        var generated = Enumerable.Range(0, rows).Select(_ => new List<int>()).ToList();
        var ends = new int?[rows];                                            // text length where a stop sequence begins
        var emitted = new int[rows];                                          // characters handed out so far
        var searched = Enumerable.Repeat("", rows).ToArray();                // the last text searched for stop sequences
        Model.Eval();
        TensorOffloading.StepBoundary(Device);                                     // with offloading: cold data out, or tensors back
        TimeSpan promptDuration;
        using (Autograd.NoGrad())
        using (var scope = new TensorScope())
        {
            sampler.Sample(Model.ForwardCached(Tensor.From(input, [rows, promptLength], Device), decoding));
        }

        promptDuration = total.Elapsed;
        int produced = 1, read = 0;
        var pieces = new List<BatchChunk>();

        // Reads the new tokens, finds stop sequences and collects each row's new text (the tail that could still become
        // a stop sequence, or ends in an incomplete character, is held back while the row goes on).
        bool Finished(bool final)
        {
            foreach (var step in sampler.Read(read, produced))
            {
                for (int r = 0; r < rows; r++)
                {
                    if (ends[r] is null)
                    {
                        generated[r].Add(step[r].Id);
                    }
                }
            }

            read = produced;
            for (int r = 0; r < rows; r++)
            {
                if (ends[r] is int stopped && emitted[r] >= stopped)
                {
                    continue;                                                  // stopped and handed out: nothing more to add
                }

                string text = Tokenizer.Decode(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(generated[r]));
                if (ends[r] is null && stopValues is not null)
                {
                    // The previous text held no stop, so a first match must end past the part it shares with this one
                    // (decoding is not always prefix-stable: bytes still to come can change the last characters).
                    int from = Math.Max(0, text.AsSpan().CommonPrefixLength(searched[r]) - holdBack);
                    int at = text.AsSpan(from).IndexOfAny(stopValues);
                    if (at >= 0)
                    {
                        ends[r] = from + at;
                    }

                    searched[r] = text;
                }

                int until = ends[r] ?? (final ? text.Length : Math.Max(0, text.Length - holdBack));
                if (ends[r] is null && !final && until > 0 && text[until - 1] == '\uFFFD')
                {
                    until--;                                                   // an incomplete character: wait for its other bytes
                }

                if (ends[r] is null && !final && until > emitted[r] && char.IsHighSurrogate(text[until - 1]))
                {
                    until--;                                                   // keep a character's two halves together
                }

                if (until > emitted[r])
                {
                    pieces.Add(new BatchChunk(r, text[emitted[r]..until]));
                    emitted[r] = until;
                }
            }

            return ends.All(e => e is not null);
        }

        while (produced < limit && !cancellationToken.IsCancellationRequested)
        {
            if (produced % Math.Max(1, options.ChunkSize) == 0)
            {
                bool done = Finished(final: false);
                foreach (var piece in pieces)
                {
                    yield return piece;
                }

                pieces.Clear();
                if (done)
                {
                    break;
                }
            }

            using (Autograd.NoGrad())
            using (var scope = new TensorScope())
            {
                sampler.Sample(Model.ForwardCached(sampler.Ids.Reshape(rows, 1), decoding));
            }

            produced++;
        }

        Finished(final: true);
        foreach (var piece in pieces)
        {
            yield return piece;
        }

        var elapsed = total.Elapsed;
        for (int r = 0; r < rows; r++)
        {
            var ids = generated[r];
            int count = ids.Count;
            if (ends[r] is int end)
            {
                // The tokens up to the one that completes the stop sequence.
                count = 1;
                while (count < ids.Count && Tokenizer.Decode(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(ids)[..count]).Length < end + 1)
                {
                    count++;
                }
            }

            yield return new BatchChunk(r, "", true, ends[r] is null ? "length" : "stop",
                new GenerationStats(tokens[r].Count, promptDuration, count, elapsed - promptDuration, elapsed, 0));
        }
    }

    /// <summary>
    /// <see cref="Stream"/> on a background thread, as an <c>await foreach</c> stream: the same chunks, and the calling
    /// thread (a UI or request thread) is never blocked by the model.
    /// </summary>
    public IAsyncEnumerable<GenerationChunk> StreamAsync(string prompt, GenerationOptions options, CancellationToken cancellationToken = default) =>
        BackgroundStream.Run(token => Stream(prompt, options, token), cancellationToken);

    /// <summary><see cref="Generate"/> on a background thread.</summary>
    public Task<(string Text, string DoneReason, GenerationStats Stats)> GenerateAsync(string prompt, GenerationOptions options,
        CancellationToken cancellationToken = default) =>
        Task.Run(() => Generate(prompt, options, cancellationToken), CancellationToken.None);

    /// <summary>Streams the continuation of <paramref name="prompt"/> in chunks of about <see cref="GenerationOptions.ChunkSize"/> tokens.</summary>
    public IEnumerable<GenerationChunk> Stream(string prompt, GenerationOptions options, CancellationToken cancellationToken = default)
    {
        var total = Stopwatch.StartNew();
        int context = Math.Clamp(options.NumCtx, 2, ContextLength);
        int limit = options.NumPredict > 0 ? Math.Min(options.NumPredict, MaxTokens) : MaxTokens;
        var history = new List<int>(Tokenizer.Encode(prompt));
        if (history.Count == 0)
        {
            history.Add(0);
        }

        if (history.Count > context - 1)
        {
            history.RemoveRange(0, history.Count - (context - 1));          // keep the most recent part of the prompt
        }

        int promptTokens = history.Count;
        var stops = options.Stop.Where(s => s.Length > 0).ToArray();
        int holdBack = stops.Length == 0 ? 0 : stops.Max(s => s.Length) - 1;
        var stopValues = stops.Length == 0 ? null : SearchValues.Create(stops, StringComparison.Ordinal);

        using var sampler = CreateSampler(new SamplerRequest(Device, 1, Tokenizer.VocabularySize, limit, Math.Max(1, options.RepeatLastN), options));
        sampler.SetHistory(history);

        var generated = new List<int>();
        var text = new ArrayBufferWriter<char>();
        int emitted = 0, read = 0, decoded = 0, decodedFrom = 0, taken = 0, lastProgress = 0, resets = 0;
        TimeSpan promptDuration = TimeSpan.Zero;
        string? doneReason = null;

        // Downloads sampled ids up to `produced`, extends the text, and checks stop sequences; returns the new text to emit.
        string Collect(int produced, bool final)
        {
            if (produced > read)
            {
                foreach (var step in sampler.Read(read, produced))
                {
                    generated.Add(step[0].Id);
                    history.Add(step[0].Id);
                }

                read = produced;
            }

            // The text is the decoding of the tokens from `decodedFrom` on, of which the first `taken` characters are in
            // `text` already. Byte-level tokenizers split a character's UTF-8 bytes across tokens (one token can end a
            // character and start the next), so a decoding can end in U+FFFD for bytes still to come: those characters
            // wait, unless four tokens bring nothing new (a character has at most 4 bytes: the bytes are invalid).
            // Decoding restarts a few tokens back (not at the new ones) now and then, since decoders may treat the start
            // of a text specially (SentencePiece decoders drop its leading space).
            if (generated.Count > decoded)
            {
                var ids = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(generated);
                string all = Tokenizer.Decode(ids[decodedFrom..]);
                int whole = all.Length;
                while (!final && whole > taken && all[whole - 1] == '\uFFFD')
                {
                    whole--;
                }

                if (whole == taken && generated.Count - lastProgress >= 4)
                {
                    whole = all.Length;
                }

                if (whole > taken)
                {
                    text.Write(all.AsSpan(taken, whole - taken));
                    taken = whole;
                    lastProgress = generated.Count;
                }

                decoded = generated.Count;
                if (taken == all.Length && decoded - decodedFrom >= 64)
                {
                    decodedFrom = decoded - 4;
                    taken = Tokenizer.Decode(ids[decodedFrom..decoded]).Length;
                }
            }

            var written = text.WrittenSpan;
            int end = written.Length;
            int searchFrom = Math.Max(0, emitted - holdBack);
            if (stopValues is not null && searchFrom < end && written[searchFrom..].IndexOfAny(stopValues) >= 0)
            {
                // Some stop occurs: find the end as stop by stop (each search stops at the end found so far).
                foreach (var stop in stops)
                {
                    // Only the text not yet searched (plus a stop's length of overlap) can hold a new match.
                    int from = Math.Max(0, emitted - stop.Length + 1);
                    int at = from < end ? written[from..end].IndexOf(stop, StringComparison.Ordinal) : -1;
                    at = at < 0 ? -1 : at + from;
                    if (at >= 0 && at < end)
                    {
                        end = at;
                        doneReason = "stop";
                    }
                }
            }

            int safe = doneReason is not null || final ? end : Math.Max(emitted, end - holdBack);
            if (doneReason is null && !final && safe > emitted && char.IsHighSurrogate(written[safe - 1]))
            {
                safe--;                                                        // keep a character's two halves together
            }

            string piece = new(written[emitted..safe]);
            emitted = safe;
            return piece;
        }

        Model.Eval();
        TensorOffloading.StepBoundary(Device);                                     // with offloading: cold data out, or tensors back
        {
            // NoGrad is entered per compute call, never held across a yield (it is thread-local state of the caller).
            var (kept, keptIds) = options.UseCache && KeepCache ? TakeCache(context) : (null, []);
            var decoding = kept ?? new DecodingContext(Device, 1, context, Layout);
            decoding.LastPositionOnly = true;                                   // the sampler reads the last position only
            bool keep = false, steppedOnce = false;
            ComputeGraph? graph = null;
            try
            {
                Tensor Window(int keep)
                {
                    var values = new float[keep];
                    var ids = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(history)[^keep..];
                    for (int i = 0; i < keep; i++)
                    {
                        values[i] = ids[i];
                    }

                    return Tensor.From(values, [1, keep], Device);
                }

                void Prefill(int keep)
                {
                    using var noGrad = Autograd.NoGrad();
                    using var scope = new TensorScope();
                    if (options.UseCache)
                    {
                        decoding.Reset();
                        sampler.Sample(Model.ForwardCached(Window(keep), decoding));
                    }
                    else
                    {
                        sampler.Sample(Model.Forward(Window(keep)));
                    }
                }

                void Step() => sampler.Sample(Model.ForwardCached(sampler.Ids.Reshape(1, 1), decoding));

                void RunStep()
                {
                    using var noGrad = Autograd.NoGrad();
                    using var scope = new TensorScope();
                    Step();
                }

                // Reuse the kept cache for the prompt's shared prefix (at least one token is fed, for the next logits).
                int shared = 0;
                while (shared < keptIds.Count && shared < history.Count && keptIds[shared] == history[shared])
                {
                    shared++;
                }

                shared = Math.Min(shared, history.Count - 1);
                if (kept is not null && shared > 0 && shared <= decoding.Length)
                {
                    using var noGrad = Autograd.NoGrad();
                    using var scope = new TensorScope();
                    decoding.Truncate(shared);
                    sampler.Sample(Model.ForwardCached(Window(history.Count - shared), decoding));
                }
                else
                {
                    Prefill(history.Count);
                }

                sampler.Read(0, 1);
                promptDuration = total.Elapsed;
                int produced = 1;
                while (doneReason is null && !cancellationToken.IsCancellationRequested)
                {
                    bool flush = produced % Math.Max(1, options.ChunkSize) == 0 || produced >= limit;
                    if (flush)
                    {
                        string piece = Collect(produced, final: false);
                        if (piece.Length > 0)
                        {
                            yield return new GenerationChunk(piece);
                        }
                    }

                    if (doneReason is not null || produced >= limit)
                    {
                        break;
                    }

                    if (!options.UseCache || decoding.Length >= context)
                    {
                        // Full recompute needs every id on the host; a full window is re-read from its last half.
                        string piece = Collect(produced, final: false);
                        if (piece.Length > 0)
                        {
                            yield return new GenerationChunk(piece);
                        }

                        if (doneReason is not null)
                        {
                            break;
                        }

                        if (options.UseCache)
                        {
                            Prefill(context / 2);
                            resets++;
                        }
                        else
                        {
                            Prefill(Math.Min(history.Count, context));
                        }
                    }
                    else
                    {
                        // The first step runs as it is: the device measures its card-dependent choices for these
                        // shapes (which it cannot while recording), and the graph recorded from the second step keeps them.
                        if (graph is null && options.UseGraph && steppedOnce && sampler.Recordable && decoding.Layout.Recordable && StepsRecordable)
                        {
                            graph = decoding.CaptureStep(Step);
                        }

                        steppedOnce = true;

                        if (graph is not null)
                        {
                            decoding.ReplayStep(graph);
                        }
                        else
                        {
                            RunStep();
                        }
                    }

                    produced++;
                }

                string rest = Collect(produced, final: true);
                if (rest.Length > 0)
                {
                    yield return new GenerationChunk(rest);
                }

                // The cache holds the keys and values of history[..Length] (the last sampled token was never fed).
                if (options.UseCache && KeepCache && resets == 0 && decoding.Length <= history.Count)
                {
                    KeepCacheFor(decoding, history.GetRange(0, decoding.Length));
                    keep = true;
                }
            }
            finally
            {
                graph?.Dispose();
                if (!keep)
                {
                    decoding.Dispose();
                }
            }
        }

        doneReason ??= "length";
        Device.Synchronize();
        var stats = new GenerationStats(promptTokens, promptDuration, generated.Count, total.Elapsed - promptDuration,
            total.Elapsed, resets);
        yield return new GenerationChunk("", Done: true, DoneReason: doneReason, Stats: stats);
    }
}

/// <summary>Runs a synchronous stream on a thread-pool thread and hands its items to an asynchronous reader.</summary>
internal static class BackgroundStream
{
    public static async IAsyncEnumerable<T> Run<T>(Func<CancellationToken, IEnumerable<T>> source,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken)
    {
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        var channel = System.Threading.Channels.Channel.CreateUnbounded<T>(
            new System.Threading.Channels.UnboundedChannelOptions { SingleReader = true, SingleWriter = true });
        var producer = Task.Run(() =>
        {
            try
            {
                foreach (var item in source(stop.Token))
                {
                    channel.Writer.TryWrite(item);
                }

                channel.Writer.TryComplete();
            }
            catch (Exception ex)
            {
                channel.Writer.TryComplete(ex);
            }
        }, CancellationToken.None);

        try
        {
            await foreach (var item in channel.Reader.ReadAllAsync(cancellationToken).ConfigureAwait(false))
            {
                yield return item;
            }
        }
        finally
        {
            stop.Cancel();
            await producer.ConfigureAwait(false);
        }
    }
}
