// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Diagnostics;
using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Diagnostics;
using Idrak.Layers;
using Idrak.Layers.Abstractions;

namespace Idrak.Generation;

/// <summary>A piece of one prompt's continuation in <c>TextGenerator.StreamBatch</c> (text or images).</summary>
/// <param name="Index">The prompt's index in the batch.</param>
/// <param name="Text">Newly generated text (empty on the final chunk).</param>
/// <param name="Done">True for the prompt's final chunk.</param>
/// <param name="DoneReason">"stop" or "length" on the final chunk.</param>
/// <param name="Stats">Statistics, on the final chunk only.</param>
public sealed record BatchChunk(int Index, string Text, bool Done = false, string? DoneReason = null, GenerationStats? Stats = null)
{
    /// <summary>The prompt's and the continuation's ids, on the final chunk when the options asked for them (<see cref="GenerationOptions.ReturnTokens"/>); null otherwise.</summary>
    public ChatTokens? Tokens { get; init; }
}

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
    public IKeyValueLayout? CacheLayout { get; init; }

    // The layout caches are created with.
    private IKeyValueLayout Layout => CacheLayout ?? KeyValueLayouts.For(CacheFormat);

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

    /// <summary>
    /// The arithmetic of a prompt's pass (<see cref="MixedPrecision.Use"/> around the prefill alone), or null for the
    /// caller's (<see cref="MixedPrecision.Current"/>). <see cref="MatMulPrecision.BFloat16"/> runs the prompt's matrix
    /// products on tensor cores where the device has them (a long prompt, or an image's tokens, several times faster); the
    /// decoding steps keep the caller's arithmetic. The cached keys and values come within bfloat16's rounding of float32:
    /// compare a model's answers before relying on it.
    /// </summary>
    public MatMulPrecision? PrefillPrecision { get; set; }

    // The scope a prompt's pass runs in (PrefillPrecision), or none.
    private MixedPrecision.Scope? PrefillScope() => PrefillPrecision is { } precision ? MixedPrecision.Use(precision) : null;

    // The modules a prompt's pieces before the last run through (ForwardPieces): up to the last cached (attention) layer,
    // the norm and head after it skipped, their logits never sampled; 0 when the model has no cached layer (one pass).
    private int CachedLayers => _cachedLayers ??= Model.ToList().FindLastIndex(m => m is ICachedModule) + 1;

    private int? _cachedLayers;

    // Floats one row's position holds at once in a prompt's pass, from the model's own dimensions: the widest module up to
    // the last cached layer, each linear output in it (a block's query, key and value, its attention output, its
    // feed-forward's hidden and output; a mixture's routed experts only) and its input twice (the residual and its
    // normalization). 0 when there is no linear layer to measure (one pass).
    private long PositionWidth => _positionWidth ??= Widest(Model.Take(CachedLayers));

    private long? _positionWidth;

    private static long Widest(IEnumerable<Module> modules) =>
        modules.Select(m => m is Sequential inner ? Widest(inner) : Outputs(m) + 2L * (m.Descendants().OfType<ILinearLayer>().FirstOrDefault()?.InFeatures ?? 0))
            .DefaultIfEmpty(0).Max();

    private static long Outputs(Module module) => module switch
    {
        ILinearLayer linear => linear.OutFeatures,
        MixtureOfExperts moe => moe.Router.OutFeatures + (moe.Experts.Count > 0 ? moe.ExpertsPerToken * Outputs(moe.Experts[0]) : 0)
                                + (moe.SharedExpert is { } shared ? Outputs(shared) : 0) + (moe.SharedExpertGate?.OutFeatures ?? 0),
        _ => module.Children().Sum(Outputs),
    };

    // Bytes the attention layers' key/value caches take once `decoding` makes them at its first step (none once made): what
    // the device's free memory holds besides the prompt's pass.
    private long CachesToCome(DecodingContext decoding)
    {
        if (decoding.CacheBytes > 0)
        {
            return 0;
        }

        var layout = decoding.Layout;
        long Bytes(long rows, int headDim) => rows * (2L * layout.RowWidth(headDim) + (layout is IScaledLayout ? 2 : 0)) * sizeof(float);
        long bytes = 0;
        foreach (var module in Model.Descendants())
        {
            if (module is CausalSelfAttention attention)
            {
                int slots = decoding.SlidingWindowSteps > 0 && attention.SlidingWindow is { } window
                    ? Math.Min(decoding.Capacity, window - 1 + decoding.SlidingWindowSteps) : decoding.Capacity;
                bytes += Bytes((long)decoding.Batch * attention.KvHeads * slots, attention.HeadDim);
            }
            else if (module is MultiHeadAttention heads)
            {
                bytes += Bytes((long)decoding.Batch * heads.Heads * decoding.Capacity, heads.HeadDim);
            }
        }

        return bytes;
    }

    // The positions per piece of a prompt's pass of `length` positions over `decoding`'s rows (ForwardPieces): the setting
    // (GenerationOptions.PrefillChunk, else IDRAK_PREFILL_CHUNK), else measured: as many positions as the workspace share
    // (ComputeResources.WorkspaceShare) of the memory the device reports available, less the caches still to be made,
    // holds at the model's width per position (PositionWidth); `length` (one pass) on a device that reports none or a
    // model without a width to measure.
    private int PrefillPiece(GenerationOptions options, DecodingContext decoding, int length)
    {
        int set = options.PrefillChunk > 0 ? options.PrefillChunk : PrefillChunkVariable();
        if (CachedLayers == 0)
        {
            return length;
        }

        if (set > 0)
        {
            return Math.Min(set, length);
        }

        if (length <= 1 || PositionWidth <= 0 || Device.Backend.AvailableMemory() is not { } free)
        {
            return length;
        }

        var usage = Device.Backend.GetMemoryUsage();
        if (usage.Limit is { } limit)
        {
            free = Math.Min(free, limit - usage.InUse);
        }

        long room = (long)((free - CachesToCome(decoding)) * ComputeResources.WorkspaceShare);
        int least = options.PrefillMinChunk > 0 ? options.PrefillMinChunk : PositionsVariable(GenerationOptions.PrefillMinChunkVariable) is int given and > 0 ? given
            : GenerationOptions.DefaultPrefillMinChunk;
        return PieceOf(room, decoding.Batch * PositionWidth * sizeof(float), least, length);
    }

    // The positions per piece that `room` bytes hold at `perPosition` bytes a position: never fewer than `least` (each piece
    // is a pass through every layer: a pass per few positions would make a long prompt as many passes as positions), never
    // more than the prompt's `length`.
    internal static int PieceOf(long room, long perPosition, int least, int length) =>
        (int)Math.Clamp(Math.Max(0, room) / Math.Max(1, perPosition), Math.Min(Math.Max(1, least), length), length);

    // IDRAK_PREFILL_CHUNK (GenerationOptions.PrefillChunk when that is 0): 0 when unset.
    private static int PrefillChunkVariable() => PositionsVariable(GenerationOptions.PrefillChunkVariable) ?? 0;

    // A count of positions from the environment variable `name`, null when unset.
    private static int? PositionsVariable(string name)
    {
        if (Environment.GetEnvironmentVariable(name) is not { Length: > 0 } text)
        {
            return null;
        }

        return int.TryParse(text, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out int value) && value >= 0
            ? value
            : throw new InvalidOperationException($"{name} is '{text}': give 0 (the default) or a number of positions.");
    }

    // A prompt's pass through the cached model in pieces of at most `piece` positions (as equal as they can be: few
    // shapes), the rows' ids [rows, length] read from the host: each piece's keys and values are cached before the next
    // piece attends to them, so the pass holds one piece's activations at a time instead of the whole prompt's. The pieces
    // before the last stop at the last cached layer (their logits are never sampled); the last one's logits are returned
    // as one pass returns them. The caller's arithmetic (PrefillScope) and gradient mode apply.
    internal Tensor ForwardPieces(ReadOnlySpan<float> ids, int rows, int length, DecodingContext decoding, int piece)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(piece);
        int layers = CachedLayers;
        var (_, size) = PieceShape(length, piece);
        float[] buffer = ArrayPool<float>.Shared.Rent(rows * size);
        try
        {
            for (int from = 0; ; from += size)
            {
                int count = Math.Min(size, length - from);
                for (int r = 0; r < rows; r++)
                {
                    ids.Slice(r * length + from, count).CopyTo(buffer.AsSpan(r * count, count));
                }

                using var part = Tensor.From(buffer.AsSpan(0, rows * count), [rows, count], Device);
                if (from + count >= length)
                {
                    return Model.ForwardCached(part, decoding);
                }

                using var scope = new TensorScope();
                Model.ForwardCached(part, decoding, layers);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
        }
    }

    // The pieces a prompt's pass of `length` positions goes through at most `piece` at a time (ForwardPieces) and the
    // positions of each (as equal as they can be: few shapes); one pass when nothing is cached between pieces.
    private (int Pieces, int Size) PieceShape(int length, int piece)
    {
        if (CachedLayers == 0 || piece >= length)
        {
            return (1, length);
        }

        int pieces = (length + piece - 1) / piece;
        return (pieces, (length + pieces - 1) / pieces);
    }

    // Whether a decoding step can be recorded: not with experts, whose routing is read back to the host on every step.
    // A pipeline's boundary sends each step to the other stages (plan 23, C7): a recorded step would replay without them.
    private bool StepsRecordable => _stepsRecordable ??= !Model.Descendants().Any(m => m is MixtureOfExperts or PipelineBoundary);

    private bool? _stepsRecordable;

    /// <summary>
    /// Keep the decoding context, the sampler and the recorded decoding step between generations that keep no attention
    /// cache (<see cref="KeepCache"/> false): the next generation of the same size (cache capacity and layout) takes the
    /// context, and with the same sampling options and a set seed (<see cref="GenerationOptions.Seed"/>; greedy image reads
    /// set one) the sampler and the recorded step too, each reset first, instead of allocating and clearing a cache, making
    /// a sampler and recording the step again. Nothing of the earlier prompt is reused (every generation starts at
    /// position 0, unlike <see cref="KeepCache"/>): the old keys and values left in the cache lie past every position a
    /// step reads, so the tokens are a fresh generation's. Default false; <see cref="ReleaseCache"/> frees them.
    /// </summary>
    public bool ReuseDecoding { get; set; }

    private readonly Lock _keptLock = new();
    private DecodingContext? _kept;
    private List<int> _keptIds = [];
    private Reusable? _reusable;

    /// <summary>Frees the KV cache kept between generations (see <see cref="KeepCache"/>) and what <see cref="ReuseDecoding"/> keeps.</summary>
    public void ReleaseCache()
    {
        lock (_keptLock)
        {
            _kept?.Dispose();
            _kept = null;
            _keptIds = [];
            _reusable?.Dispose();
            _reusable = null;
        }
    }

    // What ReuseDecoding keeps between generations: the context, the sampler with the request it was made for (null: none
    // kept, a generation without a seed), and the decoding step recorded with both.
    private sealed class Reusable : IDisposable
    {
        public DecodingContext? Context;
        public ITokenSampler? Sampler;
        public SamplerRequest? Request;
        public ComputeGraph? Graph;

        // Frees the sampler and the step recorded with it (the context stays).
        public void DropSampler()
        {
            Graph?.Dispose();
            Sampler?.Dispose();
            (Graph, Sampler, Request) = (null, null, null);
        }

        public void Dispose()
        {
            DropSampler();
            Context?.Dispose();
            Context = null;
        }
    }

    // What ReuseDecoding holds between generations (the tests compare them across generations).
    internal (DecodingContext? Context, ITokenSampler? Sampler, ComputeGraph? Step) HeldDecoding
    {
        get
        {
            lock (_keptLock)
            {
                return (_reusable?.Context, _reusable?.Sampler, _reusable?.Graph);
            }
        }
    }

    // Takes what ReuseDecoding kept (a generation at a time holds it), or a new empty holder.
    private Reusable TakeReusable()
    {
        lock (_keptLock)
        {
            var held = _reusable ?? new Reusable();
            _reusable = null;
            return held;
        }
    }

    // Keeps what `held` holds for the next generation (moved out of it, so disposing `held` frees nothing; what another
    // generation kept meanwhile is freed).
    private void KeepReusable(Reusable held)
    {
        var moved = new Reusable { Context = held.Context, Sampler = held.Sampler, Request = held.Request, Graph = held.Graph };
        (held.Context, held.Sampler, held.Request, held.Graph) = (null, null, null, null);
        lock (_keptLock)
        {
            _reusable?.Dispose();
            _reusable = moved;
        }
    }

    // Whether a sampler made for `kept` samples as one made for `request` would: the same shape and options (the stop
    // sequences, read by the generation only, aside), with a seed (one drawn at random per generation is not repeated).
    private static bool SameSampling(SamplerRequest kept, SamplerRequest request) =>
        request.Options.Seed is not null && kept.Device == request.Device && kept.Rows == request.Rows && kept.Vocabulary == request.Vocabulary
        && kept.MaxSteps == request.MaxSteps && kept.HistoryCapacity == request.HistoryCapacity
        && kept.Options with { Stop = NoStops } == request.Options with { Stop = NoStops };

    private static readonly string[] NoStops = [];

    // The positions a step may feed when sliding windows' caches hold their window (DecodingContext.SlidingWindowSteps):
    // the prompt's, and at least 128, so a window moves its last positions to the front at most once in 128 steps (the
    // move copies about two windows' rows, under 2% of what those steps' attention reads).
    private static int SlidingSteps(int prompt) => SlideWindows ? Math.Max(prompt, MinSlidingSteps) : 0;

    // Whether sliding windows' caches hold their window (tests compare with caches of the whole reply), and the least
    // steps between moves (tests make a small model's windows move).
    internal static bool SlideWindows { get; set; } = true;

    internal static int MinSlidingSteps { get; set; } = 128;

    /// <summary>
    /// Whether a key/value cache is kept from the last generation (<see cref="KeepCache"/>): a caller that runs out of
    /// memory can release it (<see cref="ReleaseCache"/>) and try again.
    /// </summary>
    public bool HoldsCache
    {
        get
        {
            lock (_keptLock)
            {
                return _kept is not null;
            }
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
    /// Whether <see cref="GenerateBatch"/> decodes prompts of different lengths together on this model and device
    /// (attention from per-row starts for every attention layer's head size, window and cap:
    /// <c>Backend.SupportsAttentionRows</c>); otherwise prompts of one length still decode together and the others alone.
    /// </summary>
    public bool SupportsBatches =>
        Model.Descendants().OfType<CausalSelfAttention>().ToList() is { Count: > 0 } attention
        && !Model.Descendants().Any(m => m is MultiHeadAttention or PositionalEncoding)
        && attention.All(a => a.SupportsRows(Device.Backend));

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
        if (prompts.Count == 1 || penalties)
        {
            for (int p = 0; p < prompts.Count; p++)
            {
                foreach (var chunk in Stream(prompts[p], options, cancellationToken))
                {
                    yield return new BatchChunk(p, chunk.Text, chunk.Done, chunk.DoneReason, chunk.Stats) { Tokens = chunk.Tokens };
                }
            }

            yield break;
        }

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

        // Rows of different lengths attend from their own starts where the model can (SupportsBatches); elsewhere rows of
        // one length still decode together (no starts needed), and a length alone runs alone.
        var groups = SupportsBatches ? [[.. Enumerable.Range(0, prompts.Count)]] : Groups(tokens);
        foreach (var group in groups)
        {
            if (group.Length == 1)
            {
                foreach (var chunk in Stream(prompts[group[0]], options, cancellationToken))
                {
                    yield return new BatchChunk(group[0], chunk.Text, chunk.Done, chunk.DoneReason, chunk.Stats) { Tokens = chunk.Tokens };
                }

                continue;
            }

            foreach (var chunk in Batch([.. group.Select(i => tokens[i])], group, null, options, cancellationToken))
            {
                yield return chunk;
            }
        }
    }

    /// <summary>
    /// The replies to several prompts with images (each prompt's image tokens taken by its own images, in order, as
    /// <see cref="Stream(string, int, IReadOnlyList{ImageFeatures}, IImageAttentionRule, GenerationOptions, CancellationToken)"/>
    /// takes them), decoded together: prompts of one length (pages read with one instruction) go through the model as one
    /// batch, one pass a token for all of them (the device reads each weight once for every row), and their decoding step
    /// is recorded once; prompts of different lengths too where every attention layer attends from per-row starts
    /// (<see cref="SupportsBatches"/>: each padded on the left, its steps not recorded), else a prompt of a length of its
    /// own runs alone. Each row stops on its own (its stop sequences, or a
    /// loop under <see cref="GenerationOptions.RepeatStop"/>) and gets what it would get alone, within the batch's rounding.
    /// Chunks as <see cref="StreamBatch(IReadOnlyList{string}, GenerationOptions, CancellationToken)"/>. Repetition
    /// penalties are not batched: such prompts run one by one.
    /// </summary>
    /// <exception cref="ArgumentException">The prompts and the image lists differ in number.</exception>
    /// <exception cref="InvalidOperationException">A prompt with its images does not fit the context window.</exception>
    public IEnumerable<BatchChunk> StreamBatch(IReadOnlyList<string> prompts, int imageToken, IReadOnlyList<IReadOnlyList<ImageFeatures>> images,
        IImageAttentionRule attention, GenerationOptions options, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(prompts);
        ArgumentNullException.ThrowIfNull(images);
        ArgumentNullException.ThrowIfNull(attention);
        if (images.Count != prompts.Count)
        {
            throw new ArgumentException($"{prompts.Count} prompts and {images.Count} lists of images.", nameof(images));
        }

        if (prompts.Count == 0)
        {
            yield break;
        }

        bool penalties = options.RepeatPenalty != 1f || options.PresencePenalty != 0f || options.FrequencyPenalty != 0f;
        var tokens = prompts.Select(p => Tokenizer.Encode(p).ToList()).ToList();
        // Pages of different lengths (another instruction, pan-and-scan crops) decode together where every attention layer
        // attends from per-row starts (SupportsBatches); elsewhere pages of one length still do, and a length alone runs alone.
        var groups = penalties ? [.. Enumerable.Range(0, prompts.Count).Select(i => new[] { i })]
            : SupportsBatches ? [[.. Enumerable.Range(0, prompts.Count)]] : Groups(tokens);
        foreach (var group in groups)
        {
            if (group.Length == 1)
            {
                int r = group[0];
                foreach (var chunk in Stream(prompts[r], (imageToken, images[r], attention), options, cancellationToken))
                {
                    yield return new BatchChunk(r, chunk.Text, chunk.Done, chunk.DoneReason, chunk.Stats) { Tokens = chunk.Tokens };
                }

                continue;
            }

            foreach (var chunk in Batch([.. group.Select(i => tokens[i])], group, (imageToken, [.. group.Select(i => images[i])], attention), options, cancellationToken))
            {
                yield return chunk;
            }
        }
    }

    // The prompts' indices grouped by token count, in the order each count first appears.
    private static List<int[]> Groups(List<List<int>> tokens) =>
        [.. Enumerable.Range(0, tokens.Count).GroupBy(i => tokens[i].Count).Select(g => g.ToArray())];

    // Decodes the rows `tokens` together (their prompts' indices `index`): padded on the left to the longest, each row
    // attending from its own start, when their lengths differ (SupportsBatches); with each row's images when given. Yields
    // each row's pieces as they come, then a final chunk per row.
    private IEnumerable<BatchChunk> Batch(List<List<int>> tokens, int[] index, (int Token, IReadOnlyList<IReadOnlyList<ImageFeatures>> Features, IImageAttentionRule Attention)? imageInput,
        GenerationOptions options, CancellationToken cancellationToken)
    {
        var total = Stopwatch.StartNew();
        int rows = tokens.Count;
        int context = Math.Clamp(options.NumCtx, 2, ContextLength);
        int promptLength = tokens.Max(t => t.Count);
        bool padded = tokens.Any(t => t.Count != promptLength);
        if (imageInput is not null && promptLength >= context)
        {
            throw new InvalidOperationException($"A prompt of {promptLength} tokens with its images does not fit the context window of {context}.");
        }

        int limit = Math.Min(options.NumPredict > 0 ? options.NumPredict : int.MaxValue, context - promptLength);   // no cap but the context window
        limit = Math.Max(1, limit);
        var stops = options.Stop.Where(x => x.Length > 0).ToArray();
        int holdBack = stops.Length == 0 ? 0 : stops.Max(x => x.Length) - 1;
        var stopValues = stops.Length == 0 ? null : SearchValues.Create(stops, StringComparison.Ordinal);
        var starts = tokens.Select(t => promptLength - t.Count).ToArray();
        var input = new float[rows * promptLength];
        for (int r = 0; r < rows; r++)
        {
            for (int i = 0; i < tokens[r].Count; i++)
            {
                input[r * promptLength + starts[r] + i] = tokens[r][i];
            }
        }

        // Each row's images, at their positions in its padded row (its prompt starts at starts[r]).
        IReadOnlyList<PromptImage> images = imageInput is { } given
            ? [.. Enumerable.Range(0, rows).SelectMany(r => ImagePrefill.Locate(tokens[r], given.Token, given.Features[r], sequence: r)
                .Select(i => i with { Position = i.Position + starts[r] }))]
            : [];
        using var sampler = CreateSampler(new SamplerRequest(Device, rows, Tokenizer.VocabularySize, limit + 1, 1, options));   // no penalties here
        // Rows of different lengths need a cache whose format attends from per-row starts (IRowStartsLayout: float32,
        // bfloat16); another format decodes them in float32.
        var batchLayout = !padded || Layout is IRowStartsLayout ? Layout : KeyValueLayouts.For(KeyValueFormat.Float32);
        using var decoding = new DecodingContext(Device, rows, promptLength + limit, batchLayout)
        {
            LastPositionOnly = true, Cancellation = cancellationToken, SlidingWindowSteps = padded ? 0 : SlidingSteps(promptLength),
        };
        if (padded)
        {
            decoding.SetRowStarts(starts);
        }

        var generated = Enumerable.Range(0, rows).Select(_ => new List<int>()).ToList();
        // The ids asked for on the final chunks (ReturnTokens): each sampled id's log-probability, each stop's length.
        bool returnTokens = options.ReturnTokens || options.ReturnLogProbabilities;
        var logProbabilities = options.ReturnLogProbabilities ? Enumerable.Range(0, rows).Select(_ => new List<float>()).ToList() : null;
        var stopTexts = new string?[rows];
        var ends = new int?[rows];                                            // text length where a stop sequence begins
        var reasons = new string?[rows];                                      // why a row ended ("stop", "repetition")
        var emitted = new int[rows];                                          // characters handed out so far
        var searched = Enumerable.Repeat("", rows).ToArray();                // the last text searched for stop sequences
        Model.Eval();
        TensorOffloading.BeforeInference(Device, Model);                          // with offloading: room for the pass on the device
        TimeSpan promptDuration;
        // A long prompt goes through in pieces (PrefillChunk); one with images in one pass, never cut inside an image.
        int prefillPiece = images.Count == 0 ? PrefillPiece(options, decoding, promptLength) : promptLength;
        long prefillStart = Telemetry.StartStage();
        using (Autograd.NoGrad())
        using (var scope = new TensorScope())
        {
            Tensor logits;
            using (PrefillScope())
            {
                if (prefillPiece < promptLength)
                {
                    logits = ForwardPieces(input, rows, promptLength, decoding, prefillPiece);
                }
                else
                {
                    using var ids = Tensor.From(input, [rows, promptLength], Device);
                    logits = images.Count == 0 ? Model.ForwardCached(ids, decoding) : Model.ForwardCached(ids, images, imageInput!.Value.Attention, decoding);
                }
            }

            sampler.Sample(logits);
        }

        if (prefillStart != 0)
        {
            // As the single prompt's pass, to its first tokens read back (only while someone listens: one read).
            sampler.Read(0, 1);
            var (prefillPieces, pieceSize) = PieceShape(promptLength, prefillPiece);
            Telemetry.InferenceStage(InferenceStageKind.Prefill, Device, prefillStart, (long)rows * promptLength, 0, rows, prefillPieces, pieceSize);
        }

        promptDuration = total.Elapsed;
        int produced = 1, read = 0, chunkFrom = 1;
        long chunkStart = Telemetry.StartStage();
        var pieces = new List<BatchChunk>();

        // Reads the new tokens, finds stop sequences and loops, and collects each row's new text (the tail that could still
        // become a stop sequence, or ends in an incomplete character, is held back while the row goes on).
        bool Finished(bool final)
        {
            foreach (var step in sampler.Read(read, produced))
            {
                for (int r = 0; r < rows; r++)
                {
                    if (ends[r] is null)
                    {
                        generated[r].Add(step[r].Id);
                        logProbabilities?[r].Add(MathF.Log(step[r].Probability));
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

                var ids = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(generated[r]);
                string text = Tokenizer.Decode(ids);
                if (ends[r] is null && stopValues is not null)
                {
                    // The previous text held no stop, so a first match must end past the part it shares with this one
                    // (decoding is not always prefix-stable: bytes still to come can change the last characters).
                    int from = Math.Max(0, text.AsSpan().CommonPrefixLength(searched[r]) - holdBack);
                    int at = text.AsSpan(from).IndexOfAny(stopValues);
                    if (at >= 0)
                    {
                        (ends[r], reasons[r]) = (from + at, "stop");
                        stopTexts[r] = StopAt(text, from + at, stops);
                    }

                    searched[r] = text;
                }

                if (ends[r] is null && options.RepeatStop >= 2 && Looping(ids, options.RepeatStop, options.RepeatStopMinTokens))
                {
                    (ends[r], reasons[r]) = (text.Length, "repetition");      // the loop is kept, as the single path keeps it
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
                    pieces.Add(new BatchChunk(index[r], text[emitted[r]..until]));
                    emitted[r] = until;
                }
            }

            return ends.All(e => e is not null);
        }

        // The decoding step: run as it is first (the device measures its choices for these shapes, which it cannot while
        // recording), then recorded once and replayed, rows of one length and rows with starts alike (a one-token step's
        // positions advance on the device).
        void Step() => sampler.Sample(Model.ForwardCached(sampler.Ids.Reshape(rows, 1), decoding));
        ComputeGraph? graph = null;
        bool steppedOnce = false;
        try
        {
            while (produced < limit && !cancellationToken.IsCancellationRequested)
            {
                if (produced % Math.Max(1, options.ChunkSize) == 0)
                {
                    bool done = Finished(final: false);
                    if (produced > chunkFrom)
                    {
                        Telemetry.InferenceStage(InferenceStageKind.DecodeChunk, Device, chunkStart, (long)(produced - chunkFrom) * rows, 0, rows);
                    }

                    (chunkFrom, chunkStart) = (produced, Telemetry.StartStage());
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

                if (graph is null && options.UseGraph && steppedOnce && sampler.Recordable && decoding.Recordable && StepsRecordable)
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
                    using (Autograd.NoGrad())
                    using (var scope = new TensorScope())
                    {
                        Step();
                    }
                }

                produced++;
            }
        }
        finally
        {
            graph?.Dispose();
        }

        Finished(final: true);
        if (produced > chunkFrom)
        {
            Telemetry.InferenceStage(InferenceStageKind.DecodeChunk, Device, chunkStart, (long)(produced - chunkFrom) * rows, 0, rows);
        }

        foreach (var piece in pieces)
        {
            yield return piece;
        }

        Device.Synchronize();
        var elapsed = total.Elapsed;
        for (int r = 0; r < rows; r++)
        {
            var ids = generated[r];
            int count = ids.Count;
            if (ends[r] is int end && reasons[r] == "stop")
            {
                // The tokens up to the one that completes the stop sequence.
                count = 1;
                while (count < ids.Count && Tokenizer.Decode(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(ids)[..count]).Length < end + 1)
                {
                    count++;
                }
            }

            yield return new BatchChunk(index[r], "", true, reasons[r] ?? "length",
                new GenerationStats(tokens[r].Count, promptDuration, count, elapsed - promptDuration, elapsed, 0))
            {
                Tokens = returnTokens ? Sampled([.. tokens[r]], ids, logProbabilities?[r], reasons[r] == "stop" ? stopTexts[r] : null) : null,
            };
        }
    }

    /// <summary>
    /// <see cref="Stream(string, GenerationOptions, CancellationToken)"/> on a background thread, as an <c>await foreach</c> stream: the same chunks, and the calling
    /// thread (a UI or request thread) is never blocked by the model.
    /// </summary>
    public IAsyncEnumerable<GenerationChunk> StreamAsync(string prompt, GenerationOptions options, CancellationToken cancellationToken = default) =>
        BackgroundStream.Run(token => Stream(prompt, options, token), cancellationToken);

    /// <summary><see cref="Generate"/> on a background thread.</summary>
    public Task<(string Text, string DoneReason, GenerationStats Stats)> GenerateAsync(string prompt, GenerationOptions options,
        CancellationToken cancellationToken = default) =>
        Task.Run(() => Generate(prompt, options, cancellationToken), CancellationToken.None);

    /// <summary>Streams the continuation of <paramref name="prompt"/> in chunks of about <see cref="GenerationOptions.ChunkSize"/> tokens.</summary>
    public IEnumerable<GenerationChunk> Stream(string prompt, GenerationOptions options, CancellationToken cancellationToken = default) =>
        Stream(prompt, null, options, cancellationToken);

    /// <summary>
    /// Streams the continuation of a prompt holding images (a vision-language model's): the image tokens
    /// (<paramref name="imageToken"/>) of the tokenized <paramref name="prompt"/> are taken in order by
    /// <paramref name="images"/> (each image's features from its family's encoder, on the model's device, read and not
    /// disposed), each its own count, whose rows replace those tokens' embeddings and attend by the family's
    /// <paramref name="attention"/> rule (<see cref="ImagePrefill"/>). The prompt is never cut inside an image: a prompt that
    /// does not fit the context window throws, a kept cache is reused only up to the first image, and a full window is
    /// re-read from after an image. Otherwise as <see cref="Stream(string, GenerationOptions, CancellationToken)"/>.
    /// </summary>
    /// <exception cref="ArgumentException">The prompt's image tokens and the images differ in number or length.</exception>
    /// <exception cref="InvalidOperationException">The prompt with its images does not fit the context window.</exception>
    public IEnumerable<GenerationChunk> Stream(string prompt, int imageToken, IReadOnlyList<ImageFeatures> images, IImageAttentionRule attention, GenerationOptions options,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(images);
        ArgumentNullException.ThrowIfNull(attention);
        return Stream(prompt, (imageToken, images, attention), options, cancellationToken);
    }

    private IEnumerable<GenerationChunk> Stream(string prompt, (int Token, IReadOnlyList<ImageFeatures> Features, IImageAttentionRule Attention)? imageInput, GenerationOptions options,
        CancellationToken cancellationToken)
    {
        var attention = imageInput?.Attention;                              // the family's rule, read only with images
        var total = Stopwatch.StartNew();
        int context = Math.Clamp(options.NumCtx, 2, ContextLength);
        int limit = options.NumPredict > 0 ? options.NumPredict : context;                              // no cap but the context window (the window slides)
        var history = new List<int>(Tokenizer.Encode(prompt));
        if (history.Count == 0)
        {
            history.Add(0);
        }

        // The prompt's images: each run of image tokens and its features (positions in `history`).
        IReadOnlyList<PromptImage> images = imageInput is { } input ? ImagePrefill.Locate(history, input.Token, input.Features) : [];
        if (images.Count > 0 && history.Count > context - 1)
        {
            throw new InvalidOperationException($"The prompt is {history.Count} tokens with its {images.Count} image{(images.Count == 1 ? "" : "s")} "
                                                + $"({images.Sum(i => i.Tokens)} image tokens); the context window holds {context - 1}: use a larger context window or fewer images.");
        }

        if (history.Count > context - 1)
        {
            history.RemoveRange(0, history.Count - (context - 1));          // keep the most recent part of the prompt
        }

        if (!options.SlideContext)
        {
            limit = Math.Min(limit, context - history.Count);               // the reply ends where the window does
        }

        // The images whose tokens lie from `start` on, numbered from there; `start` moved past an image it would cut.
        int SafeStart(int start)
        {
            foreach (var image in images)
            {
                if (image.Position < start && start < image.Position + image.Tokens)
                {
                    start = image.Position + image.Tokens;
                }
            }

            return Math.Min(start, history.Count - 1);
        }

        IReadOnlyList<PromptImage> ImagesFrom(int start) =>
            images.Count == 0 ? images : [.. images.Where(i => i.Position >= start).Select(i => i with { Position = i.Position - start })];

        // A prefill (cached when a decoding context is given) with these images, attending by the family's rule; none: the
        // plain one. In PrefillPrecision's arithmetic.
        Tensor WithImages(Tensor ids, IReadOnlyList<PromptImage> from, DecodingContext? cached)
        {
            using (PrefillScope())
            {
                return from.Count == 0 ? cached is null ? Model.Forward(ids) : Model.ForwardCached(ids, cached)
                    : cached is null ? Model.Forward(ids, from, attention!) : Model.ForwardCached(ids, from, attention!, cached);
            }
        }

        int promptTokens = history.Count;
        // The ids asked for on the final chunk (ReturnTokens): the prompt as fed, each sampled id's log-probability, where
        // a stop sequence begins and its length.
        int[]? promptIds = options.ReturnTokens || options.ReturnLogProbabilities ? [.. history] : null;
        var logProbabilities = options.ReturnLogProbabilities ? new List<float>() : null;
        string? stopText = null;
        var stops = options.Stop.Where(s => s.Length > 0).ToArray();
        int holdBack = stops.Length == 0 ? 0 : stops.Max(s => s.Length) - 1;
        var stopValues = stops.Length == 0 ? null : SearchValues.Create(stops, StringComparison.Ordinal);

        // ReuseDecoding: the context, sampler and recorded step the last generation left (one generation at a time holds
        // them); `held` frees whatever this generation does not hand back. A sampler made for other options is dropped
        // with the step recorded with it.
        bool reusing = ReuseDecoding && !KeepCache && options.UseCache;
        using var held = reusing ? TakeReusable() : new Reusable();
        var request = new SamplerRequest(Device, 1, Tokenizer.VocabularySize, limit, Math.Max(1, options.RepeatLastN), options);
        if (held.Request is { } keptRequest && !SameSampling(keptRequest, request))
        {
            held.DropSampler();
        }

        ITokenSampler sampler;
        if (held.Sampler is { } keptSampler)
        {
            keptSampler.Reset();
            sampler = keptSampler;
        }
        else
        {
            sampler = CreateSampler(request);
            (held.Sampler, held.Request) = (sampler, request);
        }

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
                    logProbabilities?.Add(MathF.Log(step[0].Probability));
                }

                read = produced;
                if (options.RepeatStop >= 2 && doneReason is null && Looping(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(generated), options.RepeatStop,
                        options.RepeatStopMinTokens))
                {
                    doneReason = "repetition";
                }
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
                        stopText = stop;
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
        TensorOffloading.BeforeInference(Device, Model);                          // with offloading: room for the pass on the device
        {
            // NoGrad is entered per compute call, never held across a yield (it is thread-local state of the caller).
            var (kept, keptIds) = options.UseCache && KeepCache ? TakeCache(context) : (null, []);
            // A cache kept for the next turn holds the whole window; otherwise it holds what this reply can reach (the
            // prompt and its most tokens, as the batched path), less memory to hold and to clear on every call. Only a
            // reply that can outgrow the window re-reads its last half, at the window's size as before.
            int capacity = options.UseCache && KeepCache ? context : Math.Min(context, history.Count + limit);

            // Sliding windows' caches hold their window, not the reply, when nothing re-reads the cache: no cache kept for
            // the next turn and a reply that cannot outgrow the window.
            int sliding = options.UseCache && !KeepCache && history.Count + limit <= context ? SlidingSteps(history.Count) : 0;

            // A kept context of this size is taken (reset by the prefill) with the step recorded on it; one of another size
            // goes, with its step. Both are this generation's until handed back.
            if (held.Context is { } keptContext && (keptContext.Capacity != capacity || keptContext.Layout != Layout || keptContext.Device != Device
                || keptContext.SlidingWindowSteps != sliding))
            {
                held.Graph?.Dispose();
                held.Context.Dispose();
                (held.Graph, held.Context) = (null, null);
            }

            var decoding = kept ?? held.Context ?? new DecodingContext(Device, 1, capacity, Layout) { SlidingWindowSteps = sliding };
            ComputeGraph? graph = options.UseGraph ? held.Graph : null;
            if (!options.UseGraph)
            {
                held.Graph?.Dispose();
            }

            (held.Context, held.Graph) = (null, null);
            decoding.LastPositionOnly = true;                                   // the sampler reads the last position only
            decoding.Cancellation = cancellationToken;                          // the prompt's pass stops between layers
            // A long prompt goes through the cache in pieces (PrefillChunk); one with images in one pass, never cut inside an image.
            int prefillPiece = options.UseCache && images.Count == 0 ? PrefillPiece(options, decoding, history.Count) : int.MaxValue;
            bool keep = false, steppedOnce = false;
            int fed = history.Count;                                            // the positions the prompt's pass fed (its pieces' telemetry)
            try
            {
                float[] WindowValues(int keep)
                {
                    var values = new float[keep];
                    var ids = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(history)[^keep..];
                    for (int i = 0; i < keep; i++)
                    {
                        values[i] = ids[i];
                    }

                    return values;
                }

                Tensor Window(int keep) => Tensor.From(WindowValues(keep), [1, keep], Device);

                // The cached pass over the last `keep` ids (from position `start`): in pieces when they are more than one.
                Tensor Feed(int keep, int start)
                {
                    fed = keep;
                    if (keep <= prefillPiece)
                    {
                        return WithImages(Window(keep), ImagesFrom(start), decoding);
                    }

                    using (PrefillScope())
                    {
                        return ForwardPieces(WindowValues(keep), 1, keep, decoding, prefillPiece);
                    }
                }

                void Prefill(int keep)
                {
                    using var noGrad = Autograd.NoGrad();
                    using var scope = new TensorScope();
                    int start = SafeStart(history.Count - keep);               // an image is read whole or not at all
                    keep = history.Count - start;
                    if (options.UseCache)
                    {
                        decoding.Reset();
                        sampler.Sample(Feed(keep, start));
                    }
                    else
                    {
                        fed = keep;
                        sampler.Sample(WithImages(Window(keep), ImagesFrom(start), null));
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
                long prefillStart = Telemetry.StartStage();
                int shared = 0;
                while (shared < keptIds.Count && shared < history.Count && keptIds[shared] == history[shared])
                {
                    shared++;
                }

                // Image tokens are the same ids whatever the image: the kept cache is reused up to the first image only.
                shared = Math.Min(Math.Min(shared, history.Count - 1), images.Count > 0 ? images.Min(i => i.Position) : int.MaxValue);
                if (kept is not null && shared > 0 && shared <= decoding.Length)
                {
                    using var noGrad = Autograd.NoGrad();
                    using var scope = new TensorScope();
                    decoding.Truncate(shared);
                    sampler.Sample(Feed(history.Count - shared, shared));
                }
                else
                {
                    Prefill(history.Count);
                }

                sampler.Read(0, 1);
                promptDuration = total.Elapsed;
                if (prefillStart != 0)
                {
                    var (prefillPieces, pieceSize) = PieceShape(fed, prefillPiece);
                    Telemetry.InferenceStage(InferenceStageKind.Prefill, Device, prefillStart, history.Count, 0, 1, prefillPieces, pieceSize);
                }
                int produced = 1, chunkFrom = 1;
                long chunkStart = Telemetry.StartStage();
                while (doneReason is null && !cancellationToken.IsCancellationRequested)
                {
                    bool flush = produced % Math.Max(1, options.ChunkSize) == 0 || produced >= limit;
                    if (flush)
                    {
                        string piece = Collect(produced, final: false);
                        Telemetry.InferenceStage(InferenceStageKind.DecodeChunk, Device, chunkStart, produced - chunkFrom);
                        (chunkFrom, chunkStart) = (produced, Telemetry.StartStage());
                        if (piece.Length > 0)
                        {
                            yield return new GenerationChunk(piece);
                        }
                    }

                    if (doneReason is not null || produced >= limit)
                    {
                        break;
                    }

                    if (!options.UseCache || decoding.Length >= decoding.Capacity)
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
                        if (graph is null && options.UseGraph && steppedOnce && sampler.Recordable && decoding.Recordable && StepsRecordable)
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
                else if (reusing)
                {
                    // Handed back: the context, and the sampler with the step recorded with it while they sample alike (a
                    // seed drawn at random is not repeated, so without one the sampler and the step go).
                    (held.Context, held.Graph) = (decoding, graph);
                    (keep, graph) = (true, null);
                    if (options.Seed is null)
                    {
                        held.DropSampler();
                    }

                    KeepReusable(held);
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
        yield return new GenerationChunk("", Done: true, DoneReason: doneReason, Stats: stats)
        {
            Tokens = promptIds is null ? null : Sampled(promptIds, generated, logProbabilities, doneReason == "stop" ? stopText : null),
        };
    }

    // A reply's ids as sampled (GenerationOptions.ReturnTokens): up to the token that completes its stop sequence `stop`
    // (the shortest run of ids whose decoding holds it, found by bisection: a decoding that holds it keeps it as ids are
    // added, while its length alone need not grow, a character's bytes still to come decoding as replacements), else
    // every id sampled; with their log-probabilities when recorded.
    private ChatTokens Sampled(int[] prompt, List<int> generated, List<float>? logProbabilities, string? stop)
    {
        var ids = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(generated);
        int count = ids.Length;
        if (stop is { Length: > 0 } && count > 1)
        {
            int low = 1, high = count;
            while (low < high)
            {
                int middle = (low + high) / 2;
                if (Tokenizer.Decode(ids[..middle]).Contains(stop, StringComparison.Ordinal))
                {
                    high = middle;
                }
                else
                {
                    low = middle + 1;
                }
            }

            count = low;
        }

        return new ChatTokens(prompt, ids[..count].ToArray(),
            logProbabilities is null ? null : System.Runtime.InteropServices.CollectionsMarshal.AsSpan(logProbabilities)[..count].ToArray());
    }

    // The stop sequence that begins at `at` of `text` (the first of `stops`, in their order, that does).
    private static string? StopAt(string text, int at, string[] stops)
    {
        foreach (var stop in stops)
        {
            if (text.AsSpan(at).StartsWith(stop, StringComparison.Ordinal))
            {
                return stop;
            }
        }

        return null;
    }

    /// <summary>
    /// Whether <paramref name="tokens"/> end with one run of <paramref name="minimum"/> to 64 tokens repeated
    /// <paramref name="times"/> times in a row (<see cref="GenerationOptions.RepeatStop"/>).
    /// </summary>
    internal static bool Looping(ReadOnlySpan<int> tokens, int times, int minimum)
    {
        int longest = Math.Min(64, tokens.Length / Math.Max(2, times));
        for (int period = Math.Max(1, minimum); period <= longest; period++)
        {
            var tail = tokens[^(period * times)..];
            if (tail[period..].SequenceEqual(tail[..^period]) && !ShortRun(tail[..period], minimum))   // each token equals the one a period before
            {
                return true;
            }
        }

        return false;

        // A run that is itself a shorter run repeated (a row of dots, a two-token rule) is text, not a phrase.
        static bool ShortRun(ReadOnlySpan<int> run, int minimum)
        {
            for (int q = 1; q < minimum && q < run.Length; q++)
            {
                if (run.Length % q == 0 && run[q..].SequenceEqual(run[..^q]))
                {
                    return true;
                }
            }

            return false;
        }
    }
}
