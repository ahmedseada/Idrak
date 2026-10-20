// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;

namespace Idrak.Generation;

/// <summary>
/// Generates text with any text encoder-decoder (<see cref="ISeq2SeqModel"/>): the prompt is the source, encoded once;
/// the answer is decoded token by token from the model's start token until its end token, a stop sequence or the length
/// limit, each step through the key/value cache of the decoder's self-attention and the cross-attentions' memory projected
/// once. Tokens are chosen by the library's samplers (<see cref="TokenSamplers"/>: temperature, top-k, top-p, min-p,
/// penalties; top-k 1 or temperature 0 is greedy), or, with <see cref="GenerationOptions.NumBeams"/> above 1, by beam
/// search (<see cref="BeamSearch.Decode"/>: the beams decoded together through one cache, the memory of the one source
/// serving every beam), so a family plug-in only registers its model (<c>PretrainedSeq2Seq</c>) and generation, streaming
/// and the inference engine's text-model contract (<see cref="ITextModel"/>) work with it. A pretrained tokenizer
/// encodes the prompt as transformers' tokenizer does (its special tokens added, <see cref="AddSpecialTokens"/>) and
/// decodes the answer without them (<see cref="SkipSpecialTokens"/>). One generation at a time per generator.
/// </summary>
/// <param name="model">The encoder-decoder (not owned).</param>
/// <param name="tokenizer">Turns the prompt into source ids and the target ids into text (the model's vocabulary).</param>
public sealed class Seq2SeqGenerator(ISeq2SeqModel model, ITokenizer tokenizer) : ITextModel
{
    /// <summary>The encoder-decoder.</summary>
    public ISeq2SeqModel Model { get; } = model ?? throw new ArgumentNullException(nameof(model));

    /// <summary>The tokenizer.</summary>
    public ITokenizer Tokenizer { get; } = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));

    /// <summary>How the decoder's keys and values are cached (null: float32).</summary>
    public KeyValueLayout? CacheLayout { get; init; }

    /// <summary>
    /// Whether a pretrained tokenizer (<see cref="IPretrainedTokenizer"/>) gives the source its post-processor's special
    /// tokens (true, the default: as transformers' <c>tokenizer(text)</c>, an end token after the text for most
    /// encoder-decoders); false encodes the text alone. Another tokenizer encodes the text alone either way.
    /// </summary>
    public bool AddSpecialTokens { get; init; } = true;

    /// <summary>
    /// Whether the source ends with the model's end token, appended when the tokenizer's encoding does not end with it
    /// (false by default): for a family trained so, whose tokenizer may come without a post-processor (a SentencePiece
    /// model file adds none). <c>PretrainedSeq2SeqModel.CreateGenerator</c> sets it from the family.
    /// </summary>
    public bool EndSource { get; init; }

    /// <summary>
    /// Whether a pretrained tokenizer leaves the special tokens out of the answer's text (true, the default: as
    /// transformers' <c>decode(skip_special_tokens=True)</c>); false writes them. Another tokenizer decodes every token.
    /// </summary>
    public bool SkipSpecialTokens { get; init; } = true;

    /// <summary>
    /// Makes the sampler each generation chooses tokens with (<see cref="TokenSamplers.Create"/>: the sampler registered as
    /// <see cref="TokenSamplers.DefaultName"/>, unless set).
    /// </summary>
    public Func<SamplerRequest, ITokenSampler> CreateSampler { get; set; } = TokenSamplers.Create;

    /// <inheritdoc />
    public IAsyncEnumerable<GenerationChunk> StreamAsync(string prompt, GenerationOptions options, CancellationToken cancellationToken = default) =>
        BackgroundStream.Run(token => Stream(prompt, options, token), cancellationToken);

    /// <summary>The whole answer to <paramref name="prompt"/>, why it ended ("stop": the end token or a stop sequence; "length") and the statistics.</summary>
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
    /// Streams the answer to <paramref name="prompt"/> in chunks of about <see cref="GenerationOptions.ChunkSize"/> tokens;
    /// the last chunk has <see cref="GenerationChunk.Done"/> set and carries the statistics. A prompt longer than the
    /// encoder takes keeps its first tokens; <see cref="GenerationOptions.NumPredict"/> (when positive) caps the answer,
    /// as the decoder's longest target does.
    /// </summary>
    public IEnumerable<GenerationChunk> Stream(string prompt, GenerationOptions options, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentNullException.ThrowIfNull(options);
        var total = Stopwatch.StartNew();
        var source = Source(prompt);
        var stops = options.Stop.Where(s => s.Length > 0).ToArray();
        int holdBack = stops.Length == 0 ? 0 : stops.Max(s => s.Length) - 1;
        var generated = new List<int>();
        int chunk = Math.Max(1, options.ChunkSize), emitted = 0;
        string text = "", reason = "length";
        bool cut = false;
        var encoding = TimeSpan.Zero;
        var decoding = new Stopwatch();
        foreach (int id in Tokens(source, options, cancellationToken, elapsed =>
                 {
                     encoding = elapsed;
                     decoding.Start();                                       // decoding from the first step (a beam search's whole search)
                 }))
        {
            if (id < 0)
            {
                reason = "stop";                                            // the end token
                break;
            }

            generated.Add(id);
            if (generated.Count % chunk != 0)
            {
                continue;
            }

            text = Text(generated);
            if (Stopped(text, stops) is { } end)
            {
                text = text[..end];
                reason = "stop";
                cut = true;
                break;
            }

            // Characters still to be completed by later tokens (U+FFFD at the end) and a possible stop's start wait.
            int safe = text.Length;
            while (safe > emitted && text[safe - 1] == '�')
            {
                safe--;
            }

            safe = Math.Max(emitted, safe - holdBack);
            if (safe > emitted && char.IsHighSurrogate(text[safe - 1]))
            {
                safe--;
            }

            if (safe > emitted)
            {
                yield return new GenerationChunk(text[emitted..safe]);
                emitted = safe;
            }
        }

        decoding.Stop();
        if (!cut)
        {
            text = Text(generated);
            if (Stopped(text, stops) is { } end)
            {
                text = text[..end];
                reason = "stop";
            }
        }

        var stats = new GenerationStats(source.Count, encoding, generated.Count, decoding.Elapsed, total.Elapsed, 0);
        yield return new GenerationChunk(emitted < text.Length ? text[emitted..] : "", Done: true, DoneReason: reason, Stats: stats);
    }

    /// <summary>
    /// The answer's token ids to the source ids <paramref name="source"/> (without the start and end tokens): the same
    /// tokens <see cref="Stream"/> decodes into text.
    /// </summary>
    public IReadOnlyList<int> GenerateIds(IReadOnlyList<int> source, GenerationOptions options, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(options);
        var ids = new List<int>();
        foreach (int id in Tokens(source, options, cancellationToken, _ => { }))
        {
            if (id < 0)
            {
                break;
            }

            ids.Add(id);
        }

        return ids;
    }

    /// <summary>
    /// The hypotheses of a beam search (<see cref="GenerationOptions.NumBeams"/> beams, at least 1) from the source ids
    /// <paramref name="source"/>, best first: their tokens without the start and end tokens, as many as finished or live at
    /// most the beams (<see cref="BeamSearch.Decode"/> with <see cref="GenerationOptions.LengthPenalty"/>,
    /// <see cref="GenerationOptions.EarlyStopping"/>, the end token, and the length limit of <see cref="Stream"/>). The best
    /// one's tokens are what <see cref="Stream"/> decodes with that many beams.
    /// </summary>
    public IReadOnlyList<BeamHypothesis> GenerateBeams(IReadOnlyList<int> source, GenerationOptions options, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(options);
        int limit = Limit(options);
        return limit < 1 ? [] : Search(source, options, limit, cancellationToken, _ => { });
    }

    // The prompt's source ids: a pretrained tokenizer's encoding with its special tokens (unless AddSpecialTokens is off),
    // the model's end token after them when EndSource asks and they do not end with it.
    private IReadOnlyList<int> Source(string prompt)
    {
        IReadOnlyList<int> ids = AddSpecialTokens && Tokenizer is IPretrainedTokenizer pretrained
            ? pretrained.Encode(prompt, null, SourceEncoding).Ids
            : Tokenizer.Encode(prompt);
        return EndSource && (ids.Count == 0 || ids[^1] != Model.EndToken) ? [.. ids, Model.EndToken] : ids;
    }

    // A pretrained tokenizer's ids of the source as transformers' tokenizer(text) gives them: special tokens, no padding.
    private static readonly TokenizerEncodeOptions SourceEncoding = new() { AddSpecialTokens = true, Padding = "none", Offsets = false };

    // The text of the answer's tokens (special tokens left out of a pretrained tokenizer's when SkipSpecialTokens).
    private string Text(List<int> ids)
    {
        var span = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(ids);
        return SkipSpecialTokens && Tokenizer is IPretrainedTokenizer pretrained ? pretrained.Decode(span, skipSpecialTokens: true) : Tokenizer.Decode(span);
    }

    // The longest answer: the decoder's longest target less the start token, capped by NumPredict when positive.
    private int Limit(GenerationOptions options)
    {
        int limit = Model.MaxTargetPositions - 1;
        return options.NumPredict > 0 ? Math.Min(limit, options.NumPredict) : limit;
    }

    // The source encoded once (no gradient, its tensors freed with the scope), its memory handed to the context.
    private void Encode(IReadOnlyList<int> source, DecodingContext context)
    {
        int length = Math.Clamp(source.Count, 1, Model.Encoder.MaxPositions);
        Model.Network.Eval();
        using (Autograd.NoGrad())
        using (new TensorScope())
        {
            var ids = new float[length];
            for (int i = 0; i < Math.Min(length, source.Count); i++)
            {
                ids[i] = source[i];                                          // an empty prompt: one token 0
            }

            var memory = Model.Encoder.Encode(Tensor.From(ids, [1, length], Model.Device));
            Model.SetMemory(memory, context);                               // projected once; the context holds it (one memory, every row)
        }
    }

    // A beam search of options.NumBeams beams over `limit` tokens at most: the source encoded once, its memory serving every
    // beam, the beams decoded together through one cache whose rows follow the beams kept.
    private IReadOnlyList<BeamHypothesis> Search(IReadOnlyList<int> source, GenerationOptions options, int limit, CancellationToken cancellationToken, Action<TimeSpan> encoded)
    {
        int beams = options.NumBeams;
        using var context = new DecodingContext(Model.Device, beams, limit, CacheLayout ?? KeyValueLayouts.For(KeyValueFormat.Float32))
        {
            LastPositionOnly = true,
            Cancellation = cancellationToken,
        };
        var clock = Stopwatch.StartNew();
        Encode(source, context);
        encoded(clock.Elapsed);
        var search = new BeamOptions
        {
            Beams = beams,
            MaxTokens = limit,
            EndToken = Model.EndToken,
            LengthPenalty = options.LengthPenalty,
            EarlyStopping = options.EarlyStopping,
        };
        var found = BeamSearch.Decode((ids, ctx) => Model.DecodeCached(ids, ctx), context, [Model.DecoderStartToken], search, cancellationToken: cancellationToken);
        cancellationToken.ThrowIfCancellationRequested();                   // a stopped search is not an answer
        return found;
    }

    // The end of the text before the first stop sequence in it, or null.
    private static int? Stopped(string text, string[] stops)
    {
        int? end = null;
        foreach (string stop in stops)
        {
            int at = text.IndexOf(stop, StringComparison.Ordinal);
            if (at >= 0 && (end is null || at < end))
            {
                end = at;
            }
        }

        return end;
    }

    // The answer's ids one by one, then -1 when it ended with the end token (nothing after the length limit): sampled, or
    // the best hypothesis of a beam search.
    private IEnumerable<int> Tokens(IReadOnlyList<int> source, GenerationOptions options, CancellationToken cancellationToken, Action<TimeSpan> encoded) =>
        options.NumBeams > 1 ? Beams(source, options, cancellationToken, encoded) : Sampled(source, options, cancellationToken, encoded);

    // The best hypothesis's tokens, then -1 when it ended with the end token (it is shorter than the limit then: the end
    // token took a step).
    private IEnumerable<int> Beams(IReadOnlyList<int> source, GenerationOptions options, CancellationToken cancellationToken, Action<TimeSpan> encoded)
    {
        int limit = Limit(options);
        if (limit < 1)
        {
            yield break;
        }

        var found = Search(source, options, limit, cancellationToken, encoded);
        if (found.Count == 0)
        {
            yield break;
        }

        foreach (int id in found[0].Tokens)
        {
            yield return id;
        }

        if (found[0].Tokens.Count < limit)
        {
            yield return -1;
        }
    }

    // The sampled ids one by one, then -1 when the end token is sampled (nothing after the length limit). The source is
    // encoded once and its memory handed to the decoding context; each step feeds the last token through the cache. No
    // gradient or tensor scope is held across a yield (both are per thread and per step).
    private IEnumerable<int> Sampled(IReadOnlyList<int> source, GenerationOptions options, CancellationToken cancellationToken, Action<TimeSpan> encoded)
    {
        var device = Model.Device;
        int limit = Limit(options);
        if (limit < 1)
        {
            yield break;
        }

        var context = new DecodingContext(device, 1, limit, CacheLayout ?? KeyValueLayouts.For(KeyValueFormat.Float32))
        {
            LastPositionOnly = true,
            Cancellation = cancellationToken,
        };
        ITokenSampler? sampler = null;
        try
        {
            var clock = Stopwatch.StartNew();
            Encode(source, context);
            encoded(clock.Elapsed);
            sampler = CreateSampler(new SamplerRequest(device, 1, Model.Vocabulary, limit, Math.Max(1, options.RepeatLastN), options));
            sampler.SetHistory([Model.DecoderStartToken]);
            var step = new float[1];
            int current = Model.DecoderStartToken;
            for (int i = 0; i < limit; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                using (Autograd.NoGrad())
                using (new TensorScope())
                {
                    step[0] = current;
                    var logits = Model.DecodeCached(Tensor.From(step, [1, 1], device), context);
                    sampler.Sample(logits);
                }

                current = sampler.Read(i, i + 1)[0][0].Id;
                if (current == Model.EndToken)
                {
                    yield return -1;
                    yield break;
                }

                yield return current;
            }
        }
        finally
        {
            sampler?.Dispose();
            context.Dispose();
        }
    }
}
