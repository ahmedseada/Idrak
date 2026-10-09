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

        var (blocks, first) = Prepare(decoder, ids, images, attention, segments: null);
        using (blocks.Use())
        {
            return decoder.ForwardCached(ids, context, count, (i, x) => i == first ? blocks.Substitute(x) : x);
        }
    }

    /// <summary>
    /// The decoder's whole pass over <paramref name="ids"/> [batch, steps] with images, without a cache (logits of every
    /// position; with autograd on, gradients reach the features too), image tokens attending by <paramref name="attention"/>.
    /// The rows may be packed sequences (<see cref="PackedSequences"/> in effect for this shape): each sequence then gets
    /// the logits it gets alone, its images at their columns in the row.
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

        var (blocks, first) = Prepare(decoder, ids, images, attention, Packing(ids));
        using (blocks.Use())
        {
            return decoder.Run(ids, decoder.Count, (i, x) => i == first ? blocks.Substitute(x) : x);
        }
    }

    /// <summary>
    /// The images of a pass over <paramref name="ids"/> [batch, steps] that the caller runs module by module (a fine-tuning
    /// step that checkpoints the decoder's blocks itself), without a cache: while the scope is open on this thread the
    /// decoder's attention layers attend by <paramref name="attention"/>, and the caller gives the module at
    /// <see cref="ImagePrefillScope.FirstBlock"/> the embeddings through <see cref="ImagePrefillScope.Substitute"/>. A block
    /// run through <c>ForwardCheckpointed</c> recomputes in the backward pass with these image blocks, whether the scope is
    /// still open then or not. With autograd on, gradients reach the features. Dispose the scope on the thread that opened it.
    /// </summary>
    /// <param name="decoder">A decoder built from a <see cref="DecoderSpec"/> (embedding, then decoder blocks).</param>
    /// <param name="ids">
    /// The prompts' token ids, [batch, steps]: padded rows, or packed rows (<see cref="PackedSequences"/> in effect for this
    /// shape: each image's <see cref="PromptImage.Position"/> is its column in the row, and each sequence attends to its own
    /// keys only, by the rule).
    /// </param>
    /// <param name="images">The images, each with its row (<see cref="PromptImage.Sequence"/>); at least one.</param>
    /// <param name="attention">How image tokens attend: the family's rule (<c>PretrainedVision.Attention</c>).</param>
    /// <exception cref="ArgumentException">An image lies outside its row (or its packed sequence), overlaps another, or its features do not fit.</exception>
    /// <exception cref="InvalidOperationException">A graph is being recorded (the image blocks are uploaded from the host; <see cref="ImagePrefillInputs"/> records).</exception>
    public static ImagePrefillScope Begin(this Sequential decoder, Tensor ids, IReadOnlyList<PromptImage> images, IImageAttentionRule attention)
    {
        ArgumentNullException.ThrowIfNull(decoder);
        ArgumentNullException.ThrowIfNull(ids);
        ArgumentNullException.ThrowIfNull(images);
        ArgumentNullException.ThrowIfNull(attention);
        if (images.Count == 0)
        {
            throw new ArgumentException("A scope of images needs at least one image; run the plain pass without one.", nameof(images));
        }

        if (ComputeGraph.IsCapturing)
        {
            throw new InvalidOperationException("A pass with images is not recordable (its image blocks are uploaded from the host).");
        }

        var (blocks, first) = Prepare(decoder, ids, images, attention, Packing(ids));
        return new ImagePrefillScope(blocks, first);
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

    // The packed rows' sequence lengths when the packing in effect lays out ids [batch, steps], else null (padded rows).
    private static IReadOnlyList<IReadOnlyList<int>>? Packing(Tensor ids) =>
        ids.Rank == 2 && PackedSequences.Current is { } packing && packing.Matches(ids.Shape[0], ids.Shape[1]) ? packing.Lengths : null;

    private static (ImageBlocks Blocks, int First) Prepare(Sequential decoder, Tensor ids, IReadOnlyList<PromptImage> images, IImageAttentionRule attention,
        IReadOnlyList<IReadOnlyList<int>>? segments)
    {
        if (ids.Rank != 2)
        {
            throw new ArgumentException($"The ids are [batch, steps], got {Tensor.FormatShape(ids.Shape)}.", nameof(ids));
        }

        var (first, dim) = FirstBlock(decoder);
        CheckPositions(images);
        return (new ImageBlocks(ids.Shape[0], ids.Shape[1], dim, images, attention, segments), first);
    }

    // The decoder's first decoder block (the module whose input takes the images) and the decoder's width.
    internal static (int First, int Dim) FirstBlock(Sequential decoder)
    {
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

        return (first, ((DecoderBlock)decoder[first]).Attention.Query.InFeatures);
    }

    private static void CheckPositions(IReadOnlyList<PromptImage> images)
    {
        if (images.FirstOrDefault(i => i?.Positions is not null) is { } placed)
        {
            // The slot is in the contract (ImageFeatures.Positions) for M-RoPE families (Qwen2-VL, Qwen2.5-VL); this decoder
            // places each token by one position (plan 11, "Vision contracts": what it would need).
            throw new NotSupportedException($"The image at {placed.Position} has multi-axis position ids (M-RoPE); this decoder places each token by one position "
                + "and has no multi-axis rotary embedding.");
        }
    }
}

/// <summary>
/// The images of a pass the caller runs module by module (<see cref="ImagePrefill.Begin"/>): while it is open on its
/// thread, the decoder's attention layers attend by the family's rule; the embeddings going into
/// <see cref="FirstBlock"/> take the images' features through <see cref="Substitute"/>. Checkpointed blocks keep the image
/// blocks for their recompute, so the backward pass may run after the scope is closed.
/// </summary>
public sealed class ImagePrefillScope : IDisposable
{
    private readonly ImageBlocks _blocks;
    private readonly ImageBlocks.Scope _scope;
    private bool _disposed;

    internal ImagePrefillScope(ImageBlocks blocks, int firstBlock)
    {
        _blocks = blocks;
        FirstBlock = firstBlock;
        _scope = blocks.Use();
    }

    /// <summary>The index in the decoder of its first decoder block: the module whose input takes the images.</summary>
    public int FirstBlock { get; }

    /// <summary>
    /// <paramref name="embeddings"/> [batch, steps, width] (the output of the modules before <see cref="FirstBlock"/>) with
    /// each image's rows replaced by its features (one gather; gradients reach the features).
    /// </summary>
    /// <exception cref="InvalidOperationException">The embeddings are not [batch, steps, width] of the ids the scope was opened for.</exception>
    public Tensor Substitute(Tensor embeddings)
    {
        ArgumentNullException.ThrowIfNull(embeddings);
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _blocks.Substitute(embeddings);
    }

    /// <summary>Ends the image attention on this thread (what was in effect before comes back).</summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _scope.Dispose();
        }
    }
}

/// <summary>
/// The images of a pass recorded as a graph and replayed for other batches (a fine-tuning step with images on a device
/// that records graphs): the images' features, where they go in the embeddings and each row's range of keys live in fixed
/// buffers on the decoder's device, which <see cref="Load"/> overwrites before each run, so a pass recorded through
/// <see cref="Begin"/> reads the batch loaded last when replayed. One set of inputs serves batches of its rows and steps
/// holding up to <see cref="ImageTokens"/> image tokens (the features' buffer); the features enter as values (no gradient
/// reaches them through these buffers). Without graphs it is the same pass as <see cref="ImagePrefill.Begin"/>.
/// </summary>
/// <example>
/// <code>
/// using var inputs = ImagePrefillInputs.Create(decoder, rows, steps, imageTokens, vision.Attention);
/// inputs.Load(images, packing);                         // before recording, and before every replay
/// using (var scope = inputs.Begin()) { /* the pass, module by module, scope.Substitute before scope.FirstBlock */ }
/// </code>
/// </example>
public sealed class ImagePrefillInputs : IDisposable
{
    private readonly ImageBlocks _blocks;
    private readonly int _first;
    private bool _loaded;

    private ImagePrefillInputs(ImageBlocks blocks, int first)
    {
        _blocks = blocks;
        _first = first;
    }

    /// <summary>
    /// Fixed inputs for passes of <paramref name="decoder"/> over [<paramref name="batch"/>, <paramref name="steps"/>] ids
    /// holding up to <paramref name="imageTokens"/> image tokens, attending by <paramref name="attention"/>, on the
    /// decoder's device.
    /// </summary>
    /// <exception cref="ArgumentException">The decoder has no decoder block, or a size is not positive.</exception>
    public static ImagePrefillInputs Create(Sequential decoder, int batch, int steps, int imageTokens, IImageAttentionRule attention)
    {
        ArgumentNullException.ThrowIfNull(decoder);
        ArgumentNullException.ThrowIfNull(attention);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batch);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(steps);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(imageTokens);
        var (first, dim) = ImagePrefill.FirstBlock(decoder);
        var device = decoder.WeightsDevice ?? Device.Default;

        // One pair of range tables per window the decoder's layers attend with (a sliding-window layer and a global one
        // see different keys), none for a causal rule (its layers keep their causal kernels).
        int[] windows = attention.Causal ? [] : [.. decoder.Descendants().OfType<CausalSelfAttention>().Select(a => a.SpanWindow).Distinct()];
        var buffers = new ImageBlocks.Buffers(Tensor.PersistentZeros([imageTokens, dim], device),
            Tensor.PersistentZeros([batch * steps], device),
            windows.ToDictionary(w => w, _ => (Tensor.PersistentZeros([batch * steps], device), Tensor.PersistentZeros([batch * steps], device))));
        return new ImagePrefillInputs(new ImageBlocks(batch, steps, dim, [], attention, null, buffers), first);
    }

    /// <summary>The rows of the batches these inputs serve.</summary>
    public int Batch => _blocks.Batch;

    /// <summary>The steps (positions per row) of the batches these inputs serve.</summary>
    public int Steps => _blocks.Steps;

    /// <summary>The most image tokens a batch may hold (the features' buffer).</summary>
    public int ImageTokens => _blocks.Fixed!.Features.Shape[0];

    /// <summary>The device memory the buffers hold, in bytes.</summary>
    public long Bytes => _blocks.Fixed!.Bytes;

    /// <summary>Whether a batch of <paramref name="batch"/> rows of <paramref name="steps"/> positions holding <paramref name="imageTokens"/> image tokens fits these inputs.</summary>
    public bool Fits(int batch, int steps, int imageTokens) => batch == Batch && steps == Steps && imageTokens <= ImageTokens;

    /// <summary>
    /// Writes a batch's images into the buffers (the features copied on the device; where they go and every window's key
    /// ranges uploaded): <paramref name="images"/> as for <see cref="ImagePrefill.Begin"/>, in the rows of
    /// <paramref name="packing"/>'s current layout when the rows are packed. Not while a graph is being recorded.
    /// </summary>
    /// <exception cref="ArgumentException">The images hold more tokens than <see cref="ImageTokens"/>, lie outside their rows (or sequences), or are on another device.</exception>
    /// <exception cref="InvalidOperationException">A graph is being recorded.</exception>
    public void Load(IReadOnlyList<PromptImage> images, PackedSequences? packing = null)
    {
        ArgumentNullException.ThrowIfNull(images);
        if (ComputeGraph.IsCapturing)
        {
            throw new InvalidOperationException("Load the images before recording or replaying the pass, not while a graph is being recorded.");
        }

        if (packing is not null && !packing.Matches(Batch, Steps))
        {
            throw new ArgumentException($"The packing lays out {packing.Rows} x {packing.Length} positions; these inputs serve {Batch} x {Steps}.", nameof(packing));
        }

        int tokens = images.Sum(i => i.Tokens);
        if (tokens > ImageTokens)
        {
            throw new ArgumentException($"The images hold {tokens} image tokens; these inputs hold {ImageTokens}.", nameof(images));
        }

        _blocks.Set(images, packing?.Lengths);
        _blocks.Write();
        _loaded = true;
    }

    /// <summary>
    /// Opens the images loaded last for a pass run module by module, as <see cref="ImagePrefill.Begin"/> does, but from the
    /// buffers: recordable (nothing is uploaded while the scope is open), and a replay reads what <see cref="Load"/> wrote.
    /// </summary>
    /// <exception cref="InvalidOperationException">Nothing was loaded yet.</exception>
    public ImagePrefillScope Begin()
    {
        if (!_loaded)
        {
            throw new InvalidOperationException("Load a batch's images before the pass.");
        }

        return new ImagePrefillScope(_blocks, _first);
    }

    /// <summary>Frees the buffers.</summary>
    public void Dispose() => _blocks.Fixed!.Dispose();
}

/// <summary>
/// The image blocks of the step in progress on this thread (set by <see cref="ImagePrefill"/>): what the decoder's
/// attention layers attend by (the family's rule; nothing for a causal rule), and the features put into the embeddings.
/// In packed rows (<see cref="PackedSequences"/>) each sequence gets its own ranges by the rule (it sees nothing of the
/// others). With fixed buffers (<see cref="ImagePrefillInputs"/>) the features, the rows they replace and the ranges are
/// read from them, so the pass is recordable.
/// </summary>
internal sealed class ImageBlocks
{
    [ThreadStatic]
    private static ImageBlocks? t_current;

    private readonly int _dim;
    private readonly IImageAttentionRule _rule;
    private List<PromptImage> _images = [];
    private List<(int Start, int Length)>[] _blocks = [];
    private IReadOnlyList<IReadOnlyList<int>>? _segments;

    public ImageBlocks(int batch, int steps, int dim, IReadOnlyList<PromptImage> images, IImageAttentionRule rule, IReadOnlyList<IReadOnlyList<int>>? segments,
        Buffers? buffers = null)
    {
        (Batch, Steps, _dim, _rule, Fixed) = (batch, steps, dim, rule, buffers);
        Attends = !rule.Causal;
        Set(images, segments);
    }

    /// <summary>The blocks in effect on this thread, or null.</summary>
    public static ImageBlocks? Current => t_current;

    public int Batch { get; }

    public int Steps { get; }

    /// <summary>Whether the rows are packed sequences (each sequence its own ranges).</summary>
    public bool Packed => _segments is not null;

    /// <summary>The fixed buffers the pass reads (a recordable pass), or null.</summary>
    public Buffers? Fixed { get; }

    /// <summary>Whether the attention layers attend by <see cref="Spans"/> (false: the rule is causal and the usual kernels run).</summary>
    public bool Attends { get; }

    /// <summary>Whether an attention layer over [n, t] takes its key ranges from these blocks.</summary>
    public bool AttendsFor(int n, int t) => Attends && n == Batch && t == Steps;

    // Takes the images (and, for packed rows, each row's sequence lengths; the rest of a row is padding), checking that each
    // lies inside its row, or inside one sequence of it, and that none overlap.
    public void Set(IReadOnlyList<PromptImage> images, IReadOnlyList<IReadOnlyList<int>>? segments)
    {
        if (segments is not null && (segments.Count != Batch || segments.Any(r => r.Any(l => l < 0) || r.Sum() > Steps)))
        {
            throw new ArgumentException($"The packing's {segments.Count} rows of sequences do not lay out {Batch} rows of {Steps} positions.", nameof(segments));
        }

        var blocks = Enumerable.Range(0, Batch).Select(_ => new List<(int Start, int Length)>()).ToArray();
        var list = new List<PromptImage>(images.Count);
        foreach (var image in images)
        {
            ArgumentNullException.ThrowIfNull(image, nameof(images));
            var f = image.Features;
            if (f.Rank is not (2 or 3) || f.Rank == 3 && f.Shape[0] != 1 || f.Shape[^1] != _dim)
            {
                throw new ArgumentException($"An image's features are [tokens, {_dim}] or [1, tokens, {_dim}], got {Tensor.FormatShape(f.Shape)}.", nameof(images));
            }

            if (image.Sequence < 0 || image.Sequence >= Batch)
            {
                throw new ArgumentException($"Image at {image.Position}: sequence {image.Sequence} is outside the batch of {Batch}.", nameof(images));
            }

            if (image.Position < 0 || image.Tokens == 0 || image.Position + image.Tokens > Steps)
            {
                throw new ArgumentException($"The image block [{image.Position}, {image.Position + image.Tokens}) lies outside the prompt's {Steps} steps "
                    + "(a block must lie inside one prefill step).", nameof(images));
            }

            if (segments is not null)
            {
                var (begin, length) = Segment(segments[image.Sequence], image.Position);
                if (length == 0 || image.Position + image.Tokens > begin + length)
                {
                    throw new ArgumentException($"The image block [{image.Position}, {image.Position + image.Tokens}) of row {image.Sequence} does not lie inside one of its "
                        + $"packed sequences ({string.Join(" + ", segments[image.Sequence])} positions).", nameof(images));
                }
            }

            blocks[image.Sequence].Add((image.Position, image.Tokens));
            list.Add(image);
        }

        foreach (var row in blocks)
        {
            _ = KeySpans.ImageBlocks(Steps, row);                        // refuses overlapping blocks
        }

        (_images, _blocks, _segments) = (list, blocks, segments);
    }

    // The packed sequence of a row (its first position and length) holding `position`; length 0 in the padding.
    private static (int Begin, int Length) Segment(IReadOnlyList<int> lengths, int position)
    {
        int begin = 0;
        foreach (int length in lengths)
        {
            if (position < begin + length)
            {
                return (begin, length);
            }

            begin += length;
        }

        return (begin, 0);
    }

    public Scope Use()
    {
        var previous = t_current;
        t_current = this;
        return new Scope(previous);
    }

    /// <summary>
    /// Each row's key range by the family's rule, [batch · steps] starts and ends (one table per sequence), for rows at
    /// positions [offset, offset + steps) of keys numbered from 0, row b seeing nothing before rowStarts[b]. Packed rows:
    /// each sequence's table at its columns (no offset, no row starts: a pass without a cache). From the fixed buffers
    /// when there are some.
    /// </summary>
    public (Tensor Starts, Tensor Ends) Spans(int window, int offset, IReadOnlyList<int>? rowStarts, Device device)
    {
        if (Fixed is not null)
        {
            if (offset != 0 || rowStarts is not null || !Fixed.Spans.TryGetValue(window, out var spans))
            {
                throw new InvalidOperationException($"The recorded image inputs hold the key ranges of a pass without a cache for windows {string.Join(", ", Fixed.Spans.Keys)}; "
                    + $"asked for window {window} at offset {offset}.");
            }

            return spans;
        }

        var (starts, ends) = SpanValues(window, offset, rowStarts);
        return (Tensor.From(starts, device), Tensor.From(ends, device));
    }

    private (float[] Starts, float[] Ends) SpanValues(int window, int offset, IReadOnlyList<int>? rowStarts)
    {
        if (_segments is not null && (offset != 0 || rowStarts is not null))
        {
            throw new InvalidOperationException("Packed rows with images attend in a pass without a cache (no offset, no row starts).");
        }

        var starts = new float[Batch * Steps];
        var ends = new float[Batch * Steps];
        for (int b = 0; b < Batch; b++)
        {
            if (_segments is not null)
            {
                // Each packed sequence (and the padding after them) its own table, at its columns: a row sees nothing of the
                // other sequences.
                int begin = 0;
                foreach (int length in _segments[b].Append(Steps - _segments[b].Sum()))
                {
                    if (length > 0)
                    {
                        var inside = _blocks[b].Where(x => x.Start >= begin && x.Start < begin + length).Select(x => (x.Start - begin, x.Length)).ToList();
                        var spans = RuleSpans(length, inside, window);
                        for (int i = 0; i < length; i++)
                        {
                            starts[b * Steps + begin + i] = spans.Starts[i] + begin;
                            ends[b * Steps + begin + i] = spans.Ends[i] + begin;
                        }
                    }

                    begin += length;
                }

                continue;
            }

            var table = RuleSpans(offset + Steps, [.. _blocks[b].Select(x => (x.Start + offset, x.Length))], window);
            int from = rowStarts?[b] ?? 0;
            for (int i = 0; i < Steps; i++)
            {
                starts[b * Steps + i] = Math.Max(table.Starts[offset + i], from);
                ends[b * Steps + i] = table.Ends[offset + i];
            }
        }

        return (starts, ends);
    }

    private KeySpans RuleSpans(int rows, IReadOnlyList<(int Start, int Length)> blocks, int window)
    {
        var spans = _rule.Spans(rows, blocks, window);
        if (spans.Starts.Count != rows)
        {
            throw new InvalidOperationException($"The image attention rule '{_rule.Name}' gave key ranges for {spans.Starts.Count} rows, not {rows}.");
        }

        return spans;
    }

    // The row of the table [batch · steps rows of embeddings, then the features] each position reads: its own embedding,
    // or its image's feature row (the features counted from `first`, image after image).
    private float[] Rows(int first)
    {
        int n = Batch, t = Steps;
        var rows = new float[n * t];
        for (int r = 0; r < rows.Length; r++)
        {
            rows[r] = r;
        }

        int next = first;
        foreach (var image in _images)
        {
            for (int i = 0; i < image.Tokens; i++)
            {
                rows[image.Sequence * t + image.Position + i] = next++;
            }
        }

        return rows;
    }

    // Writes the images into the fixed buffers: the features one after another, the rows they replace, every window's ranges.
    public void Write()
    {
        var buffers = Fixed!;
        var device = buffers.Features.Device;
        int at = 0;
        foreach (var image in _images)
        {
            var f = image.Features;
            if (f.Device != device)
            {
                throw new ArgumentException($"An image's features are on {f.Device}, the decoder on {device}.");
            }

            int count = image.Tokens * _dim;
            device.Backend.Copy2D(f.Storage, 0, count, buffers.Features.Storage, at * _dim, count, 1, count, accumulate: false);
            at += image.Tokens;
        }

        buffers.Rows.Load(Rows(Batch * Steps));
        foreach (var (window, (starts, ends)) in buffers.Spans)
        {
            var (s, e) = SpanValues(window, 0, null);
            starts.Load(s);
            ends.Load(e);
        }
    }

    /// <summary>The embeddings [batch, steps, dim] with each image's rows replaced by its features (one gather).</summary>
    public Tensor Substitute(Tensor embeddings)
    {
        int n = Batch, t = Steps;
        if (embeddings.Rank != 3 || embeddings.Shape[0] != n || embeddings.Shape[1] != t || embeddings.Shape[2] != _dim)
        {
            throw new InvalidOperationException($"The embeddings before the first decoder block are {Tensor.FormatShape(embeddings.Shape)}, not [{n}, {t}, {_dim}].");
        }

        if (Fixed is { } buffers)
        {
            // Recordable: the features' buffer after the embeddings, the rows read from their buffer.
            if (buffers.Features.Device != embeddings.Device)
            {
                throw new ArgumentException($"The images' buffers are on {buffers.Features.Device}, the decoder on {embeddings.Device}.");
            }

            var joined = Tensor.Concat([embeddings.Reshape(n * t, _dim), buffers.Features], 0);
            return joined.EmbeddingLookup(buffers.Rows).Reshape(n, t, _dim);
        }

        var parts = new List<Tensor> { embeddings.Reshape(n * t, _dim) };
        foreach (var image in _images)
        {
            if (image.Features.Device != embeddings.Device)
            {
                throw new ArgumentException($"An image's features are on {image.Features.Device}, the decoder on {embeddings.Device}.");
            }

            parts.Add(image.Features.Reshape(image.Tokens, _dim));
        }

        var table = Tensor.Concat(parts, 0);
        return table.EmbeddingLookup(Tensor.From(Rows(n * t), embeddings.Device)).Reshape(n, t, _dim);
    }

    public readonly struct Scope(ImageBlocks? previous) : IDisposable
    {
        public void Dispose() => t_current = previous;
    }

    /// <summary>The fixed buffers of a recordable pass: the features [capacity, dim], the rows [batch · steps], each window's ranges.</summary>
    internal sealed class Buffers(Tensor features, Tensor rows, Dictionary<int, (Tensor Starts, Tensor Ends)> spans) : IDisposable
    {
        public Tensor Features { get; } = features;

        public Tensor Rows { get; } = rows;

        public Dictionary<int, (Tensor Starts, Tensor Ends)> Spans { get; } = spans;

        public long Bytes => 4L * (Features.Size + Rows.Size + Spans.Values.Sum(s => (long)s.Starts.Size + s.Ends.Size));

        public void Dispose()
        {
            Features.Dispose();
            Rows.Dispose();
            foreach (var (starts, ends) in Spans.Values)
            {
                starts.Dispose();
                ends.Dispose();
            }
        }
    }
}
