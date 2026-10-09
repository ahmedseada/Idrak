// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Layers;

/// <summary>
/// An image inside a prompt, for <see cref="ImagePrefill"/>: its features (the image tokens' embeddings, [tokens, dim] or
/// [1, tokens, dim], dim the decoder's width) replace the embeddings of the prompt's image tokens from
/// <paramref name="Position"/> on (one image block); how the block's rows attend is the family's
/// <see cref="IImageAttentionRule"/>, given to the prefill.
/// </summary>
/// <param name="Position">The column of the image's first token in the prompt's ids (counted within the ids given, not the cache).</param>
/// <param name="Features">The image's features, one row per image token, on the decoder's device; read, never disposed.</param>
public sealed record PromptImage(int Position, Tensor Features)
{
    /// <summary>The prompt (row of the [batch, steps] ids) the image belongs to; 0 for a single prompt.</summary>
    public int Sequence { get; init; }

    /// <summary>
    /// The image tokens' position ids on each axis of a multi-axis rotary embedding (M-RoPE), [axes, tokens], or null (the
    /// default): they take the prompt's next positions, as text does. A decoder without multi-axis positions refuses them.
    /// </summary>
    public Tensor? Positions { get; init; }

    /// <summary>Image tokens (rows of <see cref="Features"/>).</summary>
    public int Tokens => Features.Rank == 3 ? Features.Shape[1] : Features.Shape[0];
}

/// <summary>
/// A decoder's pass over a prompt holding images (any vision-language family): the token embeddings of each image's
/// tokens are replaced by the image's features (not multiplied by the embedding scale, as transformers merges them), and
/// every attention layer attends by the family's rule (<see cref="IImageAttentionRule"/>): a causal rule keeps the
/// decoder's causal kernels (only the embeddings change); another rule gives each row its range of keys (Gemma 3's image
/// blocks: a row inside an image also sees the rest of it, <see cref="KeySpans.ImageBlocks"/>), within the window on
/// sliding-window layers. The cached pass writes the KV cache as the plain prefill does, so decoding afterwards is the
/// usual one-token step (causal, windowed), and one pass without the cache over the prompt and the generated tokens gives
/// the same logits. Prompts without images take the plain path (the same kernels and speed).
/// </summary>
/// <example>
/// <code>
/// var vision = model.Vision!;                                                         // the family's registration
/// var images = ImagePrefill.Locate(ids, vision.PromptFormat.ImageToken, features);   // ImageFeatures from its encoder
/// var logits = model.Network.ForwardCached(Tensor.From(ids, [1, ids.Length], device), images, vision.Attention, context);
/// // then model.Network.ForwardCached(next, context) per new token
/// </code>
/// </example>
public static class ImagePrefill
{
    /// <summary>
    /// <see cref="Sequential.ForwardCached(Tensor, DecodingContext, int)"/> with images in the prompt: <paramref name="ids"/>
    /// [batch, steps] (the image tokens included, as the processor expands them), each image's block lying inside these
    /// steps (a prefill continuing a cached prefix sees the cached positions as usual). Rows of different lengths
    /// (<see cref="DecodingContext.RowStarts"/>) work too, with a float32 cache.
    /// </summary>
    /// <param name="decoder">A decoder built from a <see cref="DecoderSpec"/> (embedding, then decoder blocks).</param>
    /// <param name="ids">The prompts' token ids, [batch, steps].</param>
    /// <param name="images">The images, in any order (none: the plain prefill).</param>
    /// <param name="attention">How image tokens attend: the family's rule (<c>PretrainedVision.Attention</c>).</param>
    /// <param name="context">The decoding state.</param>
    /// <param name="layers">Run the first this many modules only (all when null).</param>
    /// <exception cref="ArgumentException">An image lies outside the prompt, overlaps another, or its features do not fit.</exception>
    public static Tensor ForwardCached(this Sequential decoder, Tensor ids, IReadOnlyList<PromptImage> images, IImageAttentionRule attention, DecodingContext context, int? layers = null)
    {
        ArgumentNullException.ThrowIfNull(decoder);
        ArgumentNullException.ThrowIfNull(ids);
        ArgumentNullException.ThrowIfNull(images);
        ArgumentNullException.ThrowIfNull(attention);
        ArgumentNullException.ThrowIfNull(context);
        int count = layers ?? decoder.Count;
        if (images.Count == 0)
        {
            return decoder.ForwardCached(ids, context, count);
        }

        if (ComputeGraph.IsCapturing)
        {
            throw new InvalidOperationException("A prefill with images is not recordable (its image blocks are uploaded from the host).");
        }

        var (blocks, first) = Prepare(decoder, ids, images, attention);
        using (blocks.Use())
        {
            return decoder.ForwardCached(ids, context, count, (i, x) => i == first ? blocks.Substitute(x) : x);
        }
    }

    /// <summary>
    /// The decoder's whole pass over <paramref name="ids"/> [batch, steps] with images, without a cache (logits of every
    /// position; with autograd on, gradients reach the features too), image tokens attending by <paramref name="attention"/>.
    /// </summary>
    /// <exception cref="ArgumentException">An image lies outside the prompt, overlaps another, or its features do not fit.</exception>
    public static Tensor Forward(this Sequential decoder, Tensor ids, IReadOnlyList<PromptImage> images, IImageAttentionRule attention)
    {
        ArgumentNullException.ThrowIfNull(decoder);
        ArgumentNullException.ThrowIfNull(ids);
        ArgumentNullException.ThrowIfNull(images);
        ArgumentNullException.ThrowIfNull(attention);
        if (images.Count == 0)
        {
            return decoder.Forward(ids);
        }

        if (PackedSequences.Current is { } packing && packing.Matches(ids.Shape[0], ids.Shape[1]))
        {
            throw new NotSupportedException("Images in packed sequences are not supported; pass each prompt as its own row.");
        }

        var (blocks, first) = Prepare(decoder, ids, images, attention);
        using (blocks.Use())
        {
            return decoder.Run(ids, decoder.Count, (i, x) => i == first ? blocks.Substitute(x) : x);
        }
    }

    /// <summary>
    /// The images of one prompt from its ids: the image tokens, in order, are taken by <paramref name="features"/> in
    /// order, each image its own count of consecutive image tokens (its features' rows), so images whose tokens touch (two
    /// LLaVA images side by side) are told apart by their counts.
    /// </summary>
    /// <exception cref="ArgumentException">The image tokens and the features differ in number, or an image's tokens are not consecutive.</exception>
    public static IReadOnlyList<PromptImage> Locate(IReadOnlyList<int> ids, int imageToken, IReadOnlyList<Tensor> features, int sequence = 0)
    {
        ArgumentNullException.ThrowIfNull(features);
        return Locate(ids, imageToken, features.Count, k => new PromptImage(0, features[k]), sequence, nameof(features));
    }

    /// <summary>
    /// The images of one prompt from its ids and the images' features from a vision encoder (<see cref="IVisionEncoder.Encode"/>),
    /// with their position ids when the family gives them; as <see cref="Locate(IReadOnlyList{int}, int, IReadOnlyList{Tensor}, int)"/>.
    /// </summary>
    /// <exception cref="ArgumentException">The image tokens and the features differ in number, or an image's tokens are not consecutive.</exception>
    public static IReadOnlyList<PromptImage> Locate(IReadOnlyList<int> ids, int imageToken, IReadOnlyList<ImageFeatures> features, int sequence = 0)
    {
        ArgumentNullException.ThrowIfNull(features);
        return Locate(ids, imageToken, features.Count, k => new PromptImage(0, features[k].Features) { Positions = features[k].Positions }, sequence, nameof(features));
    }

    private static List<PromptImage> Locate(IReadOnlyList<int> ids, int imageToken, int count, Func<int, PromptImage> image, int sequence, string parameter)
    {
        ArgumentNullException.ThrowIfNull(ids);
        var images = new List<PromptImage>();
        for (int i = 0; i < ids.Count; i++)
        {
            if (ids[i] != imageToken)
            {
                continue;
            }

            if (images.Count >= count)
            {
                throw new ArgumentException($"The prompt holds more image tokens than the {count} images given take (another at {i}).", parameter);
            }

            var next = image(images.Count) with { Position = i, Sequence = sequence };
            int end = i;
            while (end < ids.Count && end < i + next.Tokens && ids[end] == imageToken)
            {
                end++;
            }

            if (end - i != next.Tokens)
            {
                throw new ArgumentException($"Image {images.Count}: {end - i} image tokens at {i}, but its features have {next.Tokens} rows.", parameter);
            }

            images.Add(next);
            i = end - 1;
        }

        if (images.Count != count)
        {
            throw new ArgumentException($"{count} images given, {images.Count} in the prompt's image tokens.", parameter);
        }

        return images;
    }

    private static (ImageBlocks Blocks, int First) Prepare(Sequential decoder, Tensor ids, IReadOnlyList<PromptImage> images, IImageAttentionRule attention)
    {
        if (ids.Rank != 2)
        {
            throw new ArgumentException($"The ids are [batch, steps], got {Tensor.FormatShape(ids.Shape)}.", nameof(ids));
        }

        int first = -1;
        for (int i = 0; i < decoder.Count && first < 0; i++)
        {
            if (decoder[i] is DecoderBlock)
            {
                first = i;
            }
        }

        if (first < 0)
        {
            throw new ArgumentException("Images go into a decoder's embeddings before its first decoder block; this network has none.", nameof(decoder));
        }

        if (images.FirstOrDefault(i => i?.Positions is not null) is { } placed)
        {
            // The slot is in the contract (ImageFeatures.Positions) for M-RoPE families (Qwen2-VL, Qwen2.5-VL); this decoder
            // places each token by one position (plan 11, "Vision contracts": what it would need).
            throw new NotSupportedException($"The image at {placed.Position} has multi-axis position ids (M-RoPE); this decoder places each token by one position "
                + "and has no multi-axis rotary embedding.");
        }

        int dim = ((DecoderBlock)decoder[first]).Attention.Query.InFeatures;
        return (new ImageBlocks(ids.Shape[0], ids.Shape[1], dim, images, attention), first);
    }
}

/// <summary>
/// The image blocks of the step in progress on this thread (set by <see cref="ImagePrefill"/>): what the decoder's
/// attention layers attend by (the family's rule; nothing for a causal rule), and the features put into the embeddings.
/// </summary>
internal sealed class ImageBlocks
{
    [ThreadStatic]
    private static ImageBlocks? t_current;

    private readonly int _dim;
    private readonly List<PromptImage> _images;
    private readonly List<(int Start, int Length)>[] _blocks;
    private readonly IImageAttentionRule _rule;

    public ImageBlocks(int batch, int steps, int dim, IReadOnlyList<PromptImage> images, IImageAttentionRule rule)
    {
        (Batch, Steps, _dim, _rule) = (batch, steps, dim, rule);
        Attends = !rule.Causal;
        _images = [.. images];
        _blocks = [.. Enumerable.Range(0, batch).Select(_ => new List<(int, int)>())];
        foreach (var image in _images)
        {
            ArgumentNullException.ThrowIfNull(image, nameof(images));
            var f = image.Features;
            if (f.Rank is not (2 or 3) || f.Rank == 3 && f.Shape[0] != 1 || f.Shape[^1] != dim)
            {
                throw new ArgumentException($"An image's features are [tokens, {dim}] or [1, tokens, {dim}], got {Tensor.FormatShape(f.Shape)}.", nameof(images));
            }

            if (image.Sequence < 0 || image.Sequence >= batch)
            {
                throw new ArgumentException($"Image at {image.Position}: sequence {image.Sequence} is outside the batch of {batch}.", nameof(images));
            }

            if (image.Position < 0 || image.Tokens == 0 || image.Position + image.Tokens > steps)
            {
                throw new ArgumentException($"The image block [{image.Position}, {image.Position + image.Tokens}) lies outside the prompt's {steps} steps "
                    + "(a block must lie inside one prefill step).", nameof(images));
            }

            _blocks[image.Sequence].Add((image.Position, image.Tokens));
        }

        foreach (var blocks in _blocks)
        {
            _ = KeySpans.ImageBlocks(steps, blocks);                     // refuses overlapping blocks
        }
    }

    /// <summary>The blocks in effect on this thread, or null.</summary>
    public static ImageBlocks? Current => t_current;

    public int Batch { get; }

    public int Steps { get; }

    /// <summary>Whether the attention layers attend by <see cref="Spans"/> (false: the rule is causal and the usual kernels run).</summary>
    public bool Attends { get; }

    /// <summary>Whether an attention layer over [n, t] takes its key ranges from these blocks.</summary>
    public bool AttendsFor(int n, int t) => Attends && n == Batch && t == Steps;

    public Scope Use()
    {
        var previous = t_current;
        t_current = this;
        return new Scope(previous);
    }

    /// <summary>
    /// Each row's key range by the family's rule, [batch · steps] starts and ends (one table per sequence), for rows at
    /// positions [offset, offset + steps) of keys numbered from 0, row b seeing nothing before rowStarts[b].
    /// </summary>
    public (Tensor Starts, Tensor Ends) Spans(int window, int offset, IReadOnlyList<int>? rowStarts, Device device)
    {
        var starts = new float[Batch * Steps];
        var ends = new float[Batch * Steps];
        for (int b = 0; b < Batch; b++)
        {
            var spans = _rule.Spans(offset + Steps, [.. _blocks[b].Select(x => (x.Start + offset, x.Length))], window);
            if (spans.Starts.Count != offset + Steps)
            {
                throw new InvalidOperationException($"The image attention rule '{_rule.Name}' gave key ranges for {spans.Starts.Count} rows, not {offset + Steps}.");
            }

            int from = rowStarts?[b] ?? 0;
            for (int i = 0; i < Steps; i++)
            {
                starts[b * Steps + i] = Math.Max(spans.Starts[offset + i], from);
                ends[b * Steps + i] = spans.Ends[offset + i];
            }
        }

        return (Tensor.From(starts, device), Tensor.From(ends, device));
    }

    /// <summary>The embeddings [batch, steps, dim] with each image's rows replaced by its features (one gather).</summary>
    public Tensor Substitute(Tensor embeddings)
    {
        int n = Batch, t = Steps;
        if (embeddings.Rank != 3 || embeddings.Shape[0] != n || embeddings.Shape[1] != t || embeddings.Shape[2] != _dim)
        {
            throw new InvalidOperationException($"The embeddings before the first decoder block are {Tensor.FormatShape(embeddings.Shape)}, not [{n}, {t}, {_dim}].");
        }

        var rows = new float[n * t];
        for (int r = 0; r < rows.Length; r++)
        {
            rows[r] = r;
        }

        var parts = new List<Tensor> { embeddings.Reshape(n * t, _dim) };
        int next = n * t;
        foreach (var image in _images)
        {
            if (image.Features.Device != embeddings.Device)
            {
                throw new ArgumentException($"An image's features are on {image.Features.Device}, the decoder on {embeddings.Device}.");
            }

            parts.Add(image.Features.Reshape(image.Tokens, _dim));
            for (int i = 0; i < image.Tokens; i++)
            {
                rows[image.Sequence * t + image.Position + i] = next++;
            }
        }

        var table = Tensor.Concat(parts, 0);
        return table.EmbeddingLookup(Tensor.From(rows, embeddings.Device)).Reshape(n, t, _dim);
    }

    public readonly struct Scope(ImageBlocks? previous) : IDisposable
    {
        public void Dispose() => t_current = previous;
    }
}
