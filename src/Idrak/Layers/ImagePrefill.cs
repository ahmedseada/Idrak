// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Layers;

/// <summary>
/// An image inside a prompt, for <see cref="ImagePrefill"/>: its features (the projected image tokens, [tokens, dim] or
/// [1, tokens, dim], dim the decoder's width) replace the embeddings of the prompt's image tokens from
/// <paramref name="Position"/> on, and those tokens see each other in both directions (one image block).
/// </summary>
/// <param name="Position">The column of the image's first token in the prompt's ids (counted within the ids given, not the cache).</param>
/// <param name="Features">The image's features, one row per image token, on the decoder's device; read, never disposed.</param>
public sealed record PromptImage(int Position, Tensor Features)
{
    /// <summary>The prompt (row of the [batch, steps] ids) the image belongs to; 0 for a single prompt.</summary>
    public int Sequence { get; init; }

    /// <summary>Image tokens (rows of <see cref="Features"/>).</summary>
    public int Tokens => Features.Rank == 3 ? Features.Shape[1] : Features.Shape[0];
}

/// <summary>
/// A decoder's pass over a prompt holding images (Gemma 3's vision-language model): the token embeddings of each image's
/// tokens are replaced by the image's features (not multiplied by the embedding scale, as transformers merges them), and
/// in every attention layer a row inside an image block also sees the rest of its block (causal everywhere else, and
/// within the window on sliding-window layers: <see cref="KeySpans.ImageBlocks"/>). The cached pass writes the KV cache
/// as the plain prefill does, so decoding afterwards is the usual one-token step (causal, windowed), and one pass
/// without the cache over the prompt and the generated tokens gives the same logits. Prompts without images take the
/// plain path (the same kernels and speed).
/// </summary>
/// <example>
/// <code>
/// var images = ImagePrefill.Locate(ids, model.Vision!.ImageTokens.ImageToken, [features]);   // features [256, 2560]
/// var logits = model.Network.ForwardCached(Tensor.From(ids, [1, ids.Length], device), images, context);
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
    /// <param name="context">The decoding state.</param>
    /// <param name="layers">Run the first this many modules only (all when null).</param>
    /// <exception cref="ArgumentException">An image lies outside the prompt, overlaps another, or its features do not fit.</exception>
    public static Tensor ForwardCached(this Sequential decoder, Tensor ids, IReadOnlyList<PromptImage> images, DecodingContext context, int? layers = null)
    {
        ArgumentNullException.ThrowIfNull(decoder);
        ArgumentNullException.ThrowIfNull(ids);
        ArgumentNullException.ThrowIfNull(images);
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

        var (blocks, first) = Prepare(decoder, ids, images);
        using (blocks.Use())
        {
            return decoder.ForwardCached(ids, context, count, (i, x) => i == first ? blocks.Substitute(x) : x);
        }
    }

    /// <summary>
    /// The decoder's whole pass over <paramref name="ids"/> [batch, steps] with images, without a cache (logits of every
    /// position; with autograd on, gradients reach the features too).
    /// </summary>
    /// <exception cref="ArgumentException">An image lies outside the prompt, overlaps another, or its features do not fit.</exception>
    public static Tensor Forward(this Sequential decoder, Tensor ids, IReadOnlyList<PromptImage> images)
    {
        ArgumentNullException.ThrowIfNull(decoder);
        ArgumentNullException.ThrowIfNull(ids);
        ArgumentNullException.ThrowIfNull(images);
        if (images.Count == 0)
        {
            return decoder.Forward(ids);
        }

        if (PackedSequences.Current is { } packing && packing.Matches(ids.Shape[0], ids.Shape[1]))
        {
            throw new NotSupportedException("Images in packed sequences are not supported; pass each prompt as its own row.");
        }

        var (blocks, first) = Prepare(decoder, ids, images);
        using (blocks.Use())
        {
            return decoder.Run(ids, decoder.Count, (i, x) => i == first ? blocks.Substitute(x) : x);
        }
    }

    /// <summary>
    /// The images of one prompt from its ids: each run of <paramref name="imageToken"/> is the next image, paired in order
    /// with <paramref name="features"/> (whose token counts must match the runs).
    /// </summary>
    /// <exception cref="ArgumentException">The runs and the features differ in number or length.</exception>
    public static IReadOnlyList<PromptImage> Locate(IReadOnlyList<int> ids, int imageToken, IReadOnlyList<Tensor> features, int sequence = 0)
    {
        ArgumentNullException.ThrowIfNull(ids);
        ArgumentNullException.ThrowIfNull(features);
        var images = new List<PromptImage>();
        for (int i = 0; i < ids.Count; i++)
        {
            if (ids[i] != imageToken || i > 0 && ids[i - 1] == imageToken)
            {
                continue;
            }

            int end = i;
            while (end < ids.Count && ids[end] == imageToken)
            {
                end++;
            }

            if (images.Count >= features.Count)
            {
                throw new ArgumentException($"The prompt holds more images than the {features.Count} given (another run of image tokens at {i}).", nameof(features));
            }

            var image = new PromptImage(i, features[images.Count]) { Sequence = sequence };
            if (image.Tokens != end - i)
            {
                throw new ArgumentException($"Image {images.Count}: {end - i} image tokens at {i}, but its features have {image.Tokens} rows.", nameof(features));
            }

            images.Add(image);
        }

        if (images.Count != features.Count)
        {
            throw new ArgumentException($"{features.Count} images given, {images.Count} runs of image tokens in the prompt.", nameof(features));
        }

        return images;
    }

    private static (ImageBlocks Blocks, int First) Prepare(Sequential decoder, Tensor ids, IReadOnlyList<PromptImage> images)
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

        int dim = ((DecoderBlock)decoder[first]).Attention.Query.InFeatures;
        return (new ImageBlocks(ids.Shape[0], ids.Shape[1], dim, images), first);
    }
}

/// <summary>
/// The image blocks of the step in progress on this thread (set by <see cref="ImagePrefill"/>): what the decoder's
/// attention layers attend by, and the features put into the embeddings.
/// </summary>
internal sealed class ImageBlocks
{
    [ThreadStatic]
    private static ImageBlocks? t_current;

    private readonly int _dim;
    private readonly List<PromptImage> _images;
    private readonly List<(int Start, int Length)>[] _blocks;

    public ImageBlocks(int batch, int steps, int dim, IReadOnlyList<PromptImage> images)
    {
        (Batch, Steps, _dim) = (batch, steps, dim);
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

    public bool Matches(int n, int t) => n == Batch && t == Steps;

    public Scope Use()
    {
        var previous = t_current;
        t_current = this;
        return new Scope(previous);
    }

    /// <summary>
    /// Each row's key range, [batch · steps] starts and ends (one table per sequence), for rows at positions
    /// [offset, offset + steps) of keys numbered from 0, row b seeing nothing before rowStarts[b].
    /// </summary>
    public (Tensor Starts, Tensor Ends) Spans(int window, int offset, IReadOnlyList<int>? rowStarts, Device device)
    {
        var starts = new float[Batch * Steps];
        var ends = new float[Batch * Steps];
        for (int b = 0; b < Batch; b++)
        {
            var spans = KeySpans.ImageBlocks(offset + Steps, [.. _blocks[b].Select(x => (x.Start + offset, x.Length))], window);
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
