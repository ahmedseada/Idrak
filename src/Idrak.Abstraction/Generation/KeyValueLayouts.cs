// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using Idrak.Abstraction.Devices;

namespace Idrak.Abstraction.Generation;

/// <summary>
/// How a <see cref="KeyValueCache"/> stores keys and values and attends over them: the attention layers write and read
/// a
/// cache through its layout instead of branching on the format. Chosen once when the cache is created (decoding steps
/// recorded as CUDA graphs keep the same kernels). The built-in format's layout comes from
/// <see cref="KeyValueLayouts.For"/>. <see cref="KeyValueLayout"/> is a base that attends through basic operations every
/// backend has; what a format can do beyond that is an interface of its own, tested once when the cache is made
/// (<see cref="IScaledLayout"/>, <see cref="IRecordableLayout"/>, <see cref="ISlidingWindowLayout"/>,
/// <see cref="IFusedWriteLayout"/>, <see cref="IRowStartsLayout"/>).
/// </summary>
public interface IKeyValueLayout
{
    /// <summary>
    /// Elements (four bytes each) per cached row of <paramref name="headDim"/> values: <see cref="KeyValueCache.Keys"/> and
    /// <see cref="KeyValueCache.Values"/> are [rows, capacity, RowWidth(headDim)].
    /// </summary>
    int RowWidth(int headDim);

    /// <summary>
    /// Writes keys and values [rows, steps, headDim] (float32) into <paramref name="cache"/> at the position held in the
    /// one-element device tensor <paramref name="position"/> (the positions already decoded).
    /// </summary>
    void Write(Tensor keys, Tensor values, KeyValueCache cache, Tensor position);

    /// <summary>
    /// The cached keys (<paramref name="keys"/> true) or values as float32, [rows, capacity, headDim]. Slots not written yet
    /// may hold anything: attention masks them out. The result may be the cache's own tensor (the float32 format), so it is
    /// only read, never changed or disposed.
    /// </summary>
    Tensor Expand(KeyValueCache cache, bool keys);

    /// <summary>
    /// The cached keys (<paramref name="keys"/> true) or values of the first <paramref name="positions"/> slots as float32,
    /// [rows, at least <paramref name="positions"/>, headDim]: what attention over a prefix of the cache reads (an image
    /// prefill step, whose key ranges end at the step's last position), without a float32 copy of every slot. A format
    /// with nothing better returns <see cref="Expand"/> (every slot). Only read, never changed or disposed (it may be the
    /// cache's own tensor).
    /// </summary>
    /// <param name="cache">The cache.</param>
    /// <param name="keys">Keys (true) or values.</param>
    /// <param name="positions">How many slots from the first are read (clamped to [1, capacity]).</param>
    Tensor ExpandPrefix(KeyValueCache cache, bool keys, int positions);

    /// <summary>
    /// Attention of q [rows, queries, headDim] over the cached positions up to each query's own → [rows, queries, headDim].
    /// A decoder's query heads sharing a key/value head are stacked along the queries (queries = group ·
    /// <paramref name="steps"/>; the mask rows repeat).
    /// </summary>
    /// <param name="q">The queries.</param>
    /// <param name="cache">The cache, already holding this step's keys and values.</param>
    /// <param name="context">The decoding state (position, mask).</param>
    /// <param name="steps">New positions in this step.</param>
    /// <param name="scale">The scale of the scores (1 / √headDim).</param>
    /// <param name="decoderKernels">
    /// True from decoder models (<c>DecoderSpec</c>), whose built-in formats use the decoding and tiled attention
    /// kernels; false from the multi-head attention layer. A format of one's own may ignore it.
    /// </param>
    Tensor Attend(Tensor q, KeyValueCache cache, DecodingContext context, int steps, float scale, bool decoderKernels);
}

/// <summary>
/// A key/value cache format whose rows each have a scale: the cache then holds <see cref="KeyValueCache.KeyScales"/> and
/// <see cref="KeyValueCache.ValueScales"/>, [rows, capacity], which the format's <see cref="IKeyValueLayout.Write"/> fills
/// and its reads use.
/// </summary>
public interface IScaledLayout : IKeyValueLayout;

/// <summary>
/// A key/value cache format whose <see cref="IKeyValueLayout.Write"/> and <see cref="IKeyValueLayout.Attend"/> only queue
/// work on the device (no reads back to the host, no uploads), so a decoding step can be recorded once as a CUDA graph
/// and replayed. A format that is not runs every step as it is. The built-in formats are recordable.
/// </summary>
public interface IRecordableLayout : IKeyValueLayout, IRecordable;

/// <summary>
/// A key/value cache format for decoder layers with a sliding window or soft-capped scores: a sliding window's cache may
/// hold only its window (<see cref="DecodingContext.SlidingWindowSteps"/>), since the format writes and attends from the
/// cache's own position and mask (<see cref="DecodingContext.PositionOf"/>, <see cref="DecodingContext.MaskOf"/>), which
/// differ from the context's once the window's positions have moved to its front; and it attends with its own kernels
/// (<see cref="AttendVariant"/>). A format that is not keeps every cache at the context's capacity and attends through
/// basic operations over every cached slot. The built-in formats are.
/// </summary>
public interface ISlidingWindowLayout : IKeyValueLayout
{
    /// <summary>
    /// <see cref="IKeyValueLayout.Attend"/> for a decoder layer with a sliding window or soft-capped scores
    /// (<paramref name="variant"/>), through the format's own kernels, which start each query at its window and cap the
    /// scores; null when the format has none for this head size (the layer then attends through basic operations over
    /// every cached slot).
    /// </summary>
    Tensor? AttendVariant(Tensor q, KeyValueCache cache, DecodingContext context, int steps, float scale, AttentionVariant variant);
}

/// <summary>
/// A key/value cache format whose attention can write its output token-major: [sequences, steps, heads·headDim], the rows
/// an output projection reads, so no copy rearranges the heads after attention (the heads' results otherwise come as
/// <see cref="IKeyValueLayout.Attend"/> returns them, [rows, queries, headDim], and the caller merges them). Tested once,
/// when the cache is made (<see cref="KeyValueCache.MergedHeadsLayout"/>). The built-in formats are, on devices whose tiled
/// attention writes so (<see cref="Backend.MergesAttention"/>).
/// </summary>
public interface IMergedHeadsLayout : IKeyValueLayout
{
    /// <summary>
    /// <see cref="IKeyValueLayout.Attend"/> (with a <paramref name="variant"/>: <see cref="ISlidingWindowLayout.AttendVariant"/>)
    /// of q [sequences · <paramref name="kvHeads"/>, group · steps, headDim] written token-major: [sequences, steps,
    /// kvHeads · group · headDim], query head kv · group + g of step s at row s, columns (kv · group + g) · headDim, the
    /// same values bit for bit. Null when the format or device has no such pass here (the caller merges the heads itself).
    /// </summary>
    /// <param name="q">The queries.</param>
    /// <param name="cache">The cache, already holding this step's keys and values.</param>
    /// <param name="context">The decoding state (position, mask).</param>
    /// <param name="steps">New positions in this step.</param>
    /// <param name="scale">The scale of the scores (1 / √headDim).</param>
    /// <param name="kvHeads">Key/value heads per sequence (the cache's rows per sequence).</param>
    /// <param name="variant">A sliding window or soft-capped scores (default: plain causal attention).</param>
    Tensor? AttendMerged(Tensor q, KeyValueCache cache, DecodingContext context, int steps, float scale, int kvHeads, AttentionVariant variant);
}

/// <summary>
/// A key/value cache format the fused projection kernel (norms, rotary positions, head layout) may write keys and values
/// straight into. The built-in float32 format is.
/// </summary>
public interface IFusedWriteLayout : IKeyValueLayout;

/// <summary>
/// A base for a key/value cache format of one's own (<see cref="IKeyValueLayout"/>): it implements <see cref="RowWidth"/>, <see cref="Write"/> and <see cref="Expand"/>, and attends through the default <see cref="Attend"/> (keys and values expanded to float32, then masked products), which runs on every backend. The
/// built-in formats (<see cref="KeyValueFormat"/>) use their own kernels. A format of one's own is passed as its layout; what it can do beyond the base is the interface of that capability (<see cref="IScaledLayout"/> when its rows have scales, <see cref="IRecordableLayout"/> when it only queues device work).
/// </summary>
/// <example>
/// <code> var layout = new MyLayout(); // its Name: "my-format" using var context = new
/// DecodingContext(device, batch: 1, capacity: 512, layout); var generator = new
/// TextGenerator(model, tokenizer, 512) { CacheLayout = layout };
/// </code>
/// </example>
public abstract class KeyValueLayout : IKeyValueLayout
{
    /// <summary>The format's name (the built-in: "float32").</summary>
    public abstract string Name { get; }

    /// <summary>One line for listings: the name unless the format says more.</summary>
    public virtual string Summary => Name;

    /// <inheritdoc />
    public abstract int RowWidth(int headDim);

    /// <summary>
    /// Writes keys and values [rows, steps, headDim] (float32) into <paramref name="cache"/> at the position held in the
    /// one-element device tensor <paramref name="position"/> (the positions already decoded).
    /// </summary>
    public abstract void Write(Tensor keys, Tensor values, KeyValueCache cache, Tensor position);

    /// <summary>
    /// The cached keys (<paramref name="keys"/> true) or values as float32, [rows, capacity, headDim]. Slots not written yet
    /// may hold anything: attention masks them out. The result may be the cache's own tensor (the float32 format), so it is
    /// only read, never changed or disposed.
    /// </summary>
    public abstract Tensor Expand(KeyValueCache cache, bool keys);

    /// <summary>
    /// The cached keys (<paramref name="keys"/> true) or values of the first <paramref name="positions"/> slots as float32,
    /// [rows, at least <paramref name="positions"/>, headDim]: what attention over a prefix of the cache reads (an image
    /// prefill step, whose key ranges end at the step's last position), without a float32 copy of every slot. The
    /// default is <see cref="Expand"/> (every slot); the built-in float32 cache is read in place. Only read, never changed or disposed (it may be the cache's own tensor).
    /// </summary>
    /// <param name="cache">The cache.</param>
    /// <param name="keys">Keys (true) or values.</param>
    /// <param name="positions">How many slots from the first are read (clamped to [1, capacity]).</param>
    public virtual Tensor ExpandPrefix(KeyValueCache cache, bool keys, int positions) => Expand(cache, keys);

    /// <summary>
    /// Attention of q [rows, queries, headDim] over the cached positions up to each query's own → [rows, queries, headDim].
    /// The default expands keys and values (<see cref="Expand"/>) and computes
    /// softmax(scale · q·Kᵀ + mask)·V with the context's causal mask (<see cref="DecodingContext.Mask"/>), from basic
    /// operations every backend has. A decoder's query heads sharing a key/value head are stacked along the queries
    /// (queries = group · <paramref name="steps"/>; the mask rows repeat).
    /// </summary>
    /// <param name="q">The queries.</param>
    /// <param name="cache">The cache, already holding this step's keys and values.</param>
    /// <param name="context">The decoding state (position, mask).</param>
    /// <param name="steps">New positions in this step.</param>
    /// <param name="scale">The scale of the scores (1 / √headDim).</param>
    /// <param name="decoderKernels">
    /// True from decoder models (<c>DecoderSpec</c>), whose built-in formats use the decoding and tiled attention
    /// kernels; false from the multi-head attention layer. A format of one's own may ignore it.
    /// </param>
    public virtual Tensor Attend(Tensor q, KeyValueCache cache, DecodingContext context, int steps, float scale, bool decoderKernels)
    {
        var keys = Expand(cache, keys: true);
        var values = Expand(cache, keys: false);
        var weights = q.MatMul(keys, transposeB: true).ScaleMaskSoftmax(scale, context.MaskOf(cache));
        return weights.MatMul(values);
    }

    /// <summary>
    /// Copies <paramref name="rows"/> [n, steps, width] (four-byte elements: floats, or values packed into them) into
    /// <paramref name="target"/> [n, capacity, width] at the position held in the one-element device tensor
    /// <paramref name="position"/>, on the device (recordable): what a format of one's own writes its rows with in
    /// <see cref="Write"/>. Per-row scales [n, steps] go into [n, capacity] reshaped with a width of 1.
    /// </summary>
    protected static void WriteRows(Tensor rows, Tensor target, Tensor position)
    {
        ArgumentNullException.ThrowIfNull(rows);
        ArgumentNullException.ThrowIfNull(target);
        ArgumentNullException.ThrowIfNull(position);
        if (rows.Rank != 3 || target.Rank != 3 || rows.Shape[0] != target.Shape[0] || rows.Shape[2] != target.Shape[2]
            || rows.Shape[1] > target.Shape[1] || rows.Device != target.Device)
        {
            throw new ArgumentException($"Rows [n, steps, width] go into a target [n, capacity, width] on the same device (got {Tensor.FormatShape(rows.Shape)} on {rows.Device}, {Tensor.FormatShape(target.Shape)} on {target.Device}).");
        }

        Tensor.WriteKeyValues(rows, target, position);
    }

    /// <inheritdoc />
    public override string ToString() => Name;
}

/// <summary>
/// The contract of a key/value cache format that decodes rows of different lengths together (prompts of several lengths
/// padded on the left, <see cref="DecodingContext.RowStarts"/>): a layout implements it to take part in batched
/// generation of such rows (<see cref="KeyValueCache.RowStartsLayout"/> holds it); a layout that does not is replaced by
/// float32 for those batches. The rows' keys and values reach the cache through the layout's own
/// <see cref="IKeyValueLayout.Write"/>; this contract reads them back. The built-in float32 format implements it
/// (through <c>Backend.AttentionRows</c>).
/// </summary>
public interface IRowStartsLayout : IKeyValueLayout
{
    /// <summary>
    /// Attention of q [rows·kvHeads, group·steps, headDim] for rows of different lengths decoded together: query i of
    /// cache row h sees the cached positions from its row's start, <paramref name="starts"/>[(h / <paramref name="headsPerRow"/>)·steps + i % steps],
    /// to its own, <paramref name="position"/>[0] + i % steps (a row with none, padding, gets zeros). Null when the device
    /// has no such pass for this head size or variant.
    /// </summary>
    /// <param name="q">The queries.</param>
    /// <param name="cache">The cache, already holding this step's keys and values (<see cref="IKeyValueLayout.Write"/>).</param>
    /// <param name="position">The one-element device tensor of the positions already decoded.</param>
    /// <param name="steps">New positions in this step.</param>
    /// <param name="scale">The scale of the scores (1 / √headDim).</param>
    /// <param name="starts">Each new token's row start, [rows · steps] (<see cref="DecodingContext.TokenStarts"/>).</param>
    /// <param name="headsPerRow">Cache rows per sequence (the key/value heads).</param>
    /// <param name="variant">A sliding window or soft-capped scores.</param>
    Tensor? AttendRows(Tensor q, KeyValueCache cache, Tensor position, int steps, float scale, Tensor starts, int headsPerRow, AttentionVariant variant);
}

/// <summary></summary>
public static class KeyValueLayouts
{
    // The built-ins, indexed by their format.
    private static readonly IKeyValueLayout[] BuiltIn = [new Float32Layout()];

    /// <summary>The built-in layout storing <paramref name="format"/>.</summary>
    public static IKeyValueLayout For(KeyValueFormat format) =>
        format == KeyValueFormat.Float32 ? BuiltIn[(int)format]
            : throw new ArgumentException($"{format} is not a built-in format; pass the layout itself (or its name to KeyValueLayouts.Get).", nameof(format));

    /// <summary>The built-in format <paramref name="layout"/> stores, or <see cref="KeyValueFormat.Custom"/> for a format of one's own.</summary>
    public static KeyValueFormat FormatOf(IKeyValueLayout layout)
    {
        ArgumentNullException.ThrowIfNull(layout);
        int index = Array.IndexOf(BuiltIn, layout);
        return index < 0 ? KeyValueFormat.Custom : (KeyValueFormat)index;
    }

    // Whether the built-in formats' tiled pass over `cache` writes q's output token-major on q's device: q [rows, group ·
    // steps, dim] over the cache's rows, kvHeads of them per sequence.
    private static bool Merges(Tensor q, Operations.Operation operation, KeyValueCache cache, int kvHeads, AttentionVariant variant) =>
        kvHeads > 0 && q.Rank == 3 && q.Shape[0] % kvHeads == 0 && q.Shape[0] == cache.Keys.Shape[0]
        && q.Backend.MergesAttention(operation, cache.HeadDim, variant);

    private sealed class Float32Layout : KeyValueLayout, IRowStartsLayout, IRecordableLayout, ISlidingWindowLayout, IFusedWriteLayout, IMergedHeadsLayout
    {
        public override string Name => "float32";

        public override string Summary => "32-bit floats (exact)";

        public override int RowWidth(int headDim) => headDim;

        public override void Write(Tensor keys, Tensor values, KeyValueCache cache, Tensor position)
        {
            Tensor.WriteKeyValues(keys, cache.Keys, position);
            Tensor.WriteKeyValues(values, cache.Values, position);
        }

        public override Tensor Expand(KeyValueCache cache, bool keys) => keys ? cache.Keys : cache.Values;

        public override Tensor Attend(Tensor q, KeyValueCache cache, DecodingContext context, int steps, float scale, bool decoderKernels)
        {
            if (decoderKernels && Kernels(q, cache, context, steps, scale, default) is { } attended)
            {
                return attended;
            }

            // Every slot of the cache, the unwritten ones masked out.
            var weights = q.MatMul(cache.Keys, transposeB: true).ScaleMaskSoftmax(scale, context.MaskOf(cache));
            return weights.MatMul(cache.Values);
        }

        public Tensor? AttendVariant(Tensor q, KeyValueCache cache, DecodingContext context, int steps, float scale, AttentionVariant variant) =>
            Kernels(q, cache, context, steps, scale, variant);

        public Tensor? AttendRows(Tensor q, KeyValueCache cache, Tensor position, int steps, float scale, Tensor starts, int headsPerRow,
            AttentionVariant variant) =>
            Tensor.AttentionRows(q, cache.Keys, cache.Values, position, steps, scale, starts, headsPerRow, variant);

        // A prompt's tiled pass, its output token-major where the device writes so.
        public Tensor? AttendMerged(Tensor q, KeyValueCache cache, DecodingContext context, int steps, float scale, int kvHeads, AttentionVariant variant) =>
            steps >= 8 && cache.HeadDim <= q.Backend.Capabilities.TiledAttentionHeadDim && Merges(q, Operations.Ops.AttentionTiled, cache, kvHeads, variant)
                ? Tensor.AttentionTiled(q, cache.Keys, cache.Values, context.PositionOf(cache), steps, scale, variant with { Merged = kvHeads })
                : null;

        private static Tensor? Kernels(Tensor q, KeyValueCache cache, DecodingContext context, int steps, float scale, AttentionVariant variant)
        {
            var capabilities = q.Backend.Capabilities;
            if (steps >= 8 && cache.HeadDim <= capabilities.TiledAttentionHeadDim)
            {
                return Tensor.AttentionTiled(q, cache.Keys, cache.Values, context.PositionOf(cache), steps, scale, variant);   // a prompt: tiled
            }

            return cache.HeadDim <= capabilities.DecodeAttentionHeadDim
                ? Tensor.AttentionDecode(q, cache, context.PositionOf(cache), steps, scale, variant)                        // only the filled positions
                : null;
        }
    }
}
