// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using Idrak.Abstraction.Devices;

namespace Idrak.Abstraction.Generation;

/// <summary>
/// How a <see cref="KeyValueCache"/> stores keys and values and attends over them: the attention layers write and read a
/// cache through its layout instead of branching on the format. Chosen once when the cache is created (decoding steps
/// recorded as CUDA graphs keep the same kernels). Registered by its <see cref="IKind.Name"/> in
/// <see cref="KeyValueLayouts"/>. <see cref="KeyValueLayout"/> is a base that attends through basic operations every
/// backend has; what a format can do beyond that is an interface of its own, tested once when the cache is made
/// (<see cref="IScaledLayout"/>, <see cref="IRecordableLayout"/>, <see cref="ISlidingWindowLayout"/>,
/// <see cref="IFusedWriteLayout"/>, <see cref="IRowStartsLayout"/>).
/// </summary>
public interface IKeyValueLayout : IKind
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
/// and its reads use. The built-in int8 format is one.
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
/// A key/value cache format the fused projection kernel (norms, rotary positions, head layout) may write keys and values
/// straight into. The built-in float32 and bfloat16 formats are.
/// </summary>
public interface IFusedWriteLayout : IKeyValueLayout
{
    /// <summary>Whether values are stored as bfloat16 pairs (two per element), which the kernel then writes.</summary>
    bool HalfWords { get; }
}

/// <summary>
/// A base for a key/value cache format of one's own (<see cref="IKeyValueLayout"/>): it implements
/// <see cref="RowWidth"/>, <see cref="Write"/> and <see cref="Expand"/>, and attends through the default
/// <see cref="Attend"/> (keys and values expanded to float32, then masked products), which runs on every backend. The
/// built-in formats (<see cref="KeyValueFormat"/>) use their own kernels. Register a format by its name with
/// <see cref="KeyValueLayouts.Register"/>; what it can do beyond the base is the interface of that capability
/// (<see cref="IScaledLayout"/> when its rows have scales, <see cref="IRecordableLayout"/> when it only queues device work).
/// </summary>
/// <example>
/// <code>
/// KeyValueLayouts.Register(new MyLayout());                                 // its Name: "my-format"
/// using var context = new DecodingContext(device, batch: 1, capacity: 512, KeyValueLayouts.Get("my-format"));
/// var generator = new TextGenerator(model, tokenizer, 512) { CacheLayout = KeyValueLayouts.Get("my-format") };
/// </code>
/// </example>
public abstract class KeyValueLayout : IKeyValueLayout
{
    /// <summary>The format's name (the built-ins: "float32", "int8", "bfloat16").</summary>
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
    /// default is <see cref="Expand"/> (every slot); the built-in float32 cache is read in place, bfloat16 and int8 caches
    /// expand only those slots. Only read, never changed or disposed (it may be the cache's own tensor).
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
/// <see cref="IKeyValueLayout.Write"/>; this contract reads them back. The built-in float32 and bfloat16 formats
/// implement it (through <c>Backend.AttentionRows</c> and <c>Backend.AttentionRowsBFloat16</c>).
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

/// <summary>
/// The key/value cache formats by name: "float32", "int8" and "bfloat16" (the built-in <see cref="KeyValueFormat"/>s) are
/// registered; add formats of one's own with <see cref="Register"/>, so options and tools can choose them by name.
/// </summary>
public static class KeyValueLayouts
{
    // The built-ins, indexed by their format.
    private static readonly IKeyValueLayout[] BuiltIn = [new Float32Layout(), new Int8Layout(), new BFloat16Layout()];

    private static readonly KindTable<IKeyValueLayout> Registry = Layouts();

    private static KindTable<IKeyValueLayout> Layouts()
    {
        var table = new KindTable<IKeyValueLayout>(nameof(KeyValueLayouts), "Key/value cache format", "add it with KeyValueLayouts.Register",
            comparer: StringComparer.OrdinalIgnoreCase, unguarded: "a layout holds a cache's rows through a whole sequence, so it cannot change half-way");
        foreach (var layout in BuiltIn)
        {
            table.RegisterDefault(layout);
        }

        return table;
    }

    /// <summary>The built-in layout storing <paramref name="format"/>.</summary>
    public static IKeyValueLayout For(KeyValueFormat format) =>
        format is >= KeyValueFormat.Float32 and <= KeyValueFormat.BFloat16 ? BuiltIn[(int)format]
            : throw new ArgumentException($"{format} is not a built-in format; pass the layout itself (or its name to KeyValueLayouts.Get).", nameof(format));

    /// <summary>The built-in format <paramref name="layout"/> stores, or <see cref="KeyValueFormat.Custom"/> for a format of one's own.</summary>
    public static KeyValueFormat FormatOf(IKeyValueLayout layout)
    {
        ArgumentNullException.ThrowIfNull(layout);
        int index = Array.IndexOf(BuiltIn, layout);
        return index < 0 ? KeyValueFormat.Custom : (KeyValueFormat)index;
    }

    /// <summary>
    /// Registers <paramref name="layout"/> under its <see cref="IKind.Name"/> (names ignore case); under a built-in name it
    /// overrides the library's layout until <see cref="Unregister"/>. A cache keeps the layout it was made with through
    /// its sequence, so a layout does not fall back to the library's when it fails. Choosing a built-in
    /// <see cref="KeyValueFormat"/> always uses the built-in layout, whatever is registered under its name.
    /// </summary>
    /// <param name="layout">How the format stores and attends.</param>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IKeyValueLayout layout) => Registry.Register(layout, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's format <paramref name="name"/> (a built-in name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered format names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Names;

    /// <summary>The layout registered as <paramref name="name"/> (any case).</summary>
    public static IKeyValueLayout Get(ReadOnlySpan<char> name) => Registry.Find(name)
        ?? throw new NotRegisteredException(nameof(KeyValueLayouts), name.ToString(), Registry.Names,
            $"No key/value cache format '{name}' is registered ({string.Join(", ", Registry.Names)}); add it with KeyValueLayouts.Register.");

    /// <summary>The library's layout <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IKeyValueLayout? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the layout <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    private sealed class Float32Layout : KeyValueLayout, IRowStartsLayout, IRecordableLayout, ISlidingWindowLayout, IFusedWriteLayout
    {
        public override string Name => "float32";

        public override string Summary => "32-bit floats (exact)";

        public override int RowWidth(int headDim) => headDim;

        public bool HalfWords => false;

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

    private sealed class Int8Layout : KeyValueLayout, IScaledLayout, IRecordableLayout, ISlidingWindowLayout
    {
        public override string Name => "int8";

        public override string Summary => "one signed byte per value and a float scale per cached row";

        public override int RowWidth(int headDim) => (headDim + 3) / 4;

        public override void Write(Tensor keys, Tensor values, KeyValueCache cache, Tensor position)
        {
            Tensor.WriteKeyValuesInt8(keys, cache.Keys, cache.KeyScales!, position);
            Tensor.WriteKeyValuesInt8(values, cache.Values, cache.ValueScales!, position);
        }

        // The bytes as floats (each row's bytes are a weight row with unit column scales), times each row's scale.
        public override Tensor Expand(KeyValueCache cache, bool keys) =>
            Expanded(keys ? cache.Keys : cache.Values, keys ? cache.KeyScales! : cache.ValueScales!, cache.HeadDim);

        // Only the first slots: the packed rows and their scales narrowed first (a copy of the prefix's bytes).
        public override Tensor ExpandPrefix(KeyValueCache cache, bool keys, int positions)
        {
            var packed = keys ? cache.Keys : cache.Values;
            positions = Math.Clamp(positions, 1, packed.Shape[1]);
            if (positions == packed.Shape[1])
            {
                return Expand(cache, keys);
            }

            using var prefix = packed.Narrow(1, 0, positions);
            using var scales = (keys ? cache.KeyScales! : cache.ValueScales!).Narrow(1, 0, positions);
            return Expanded(prefix, scales, cache.HeadDim);
        }

        private static Tensor Expanded(Tensor packed, Tensor scales, int dim)
        {
            int rows = packed.Shape[0], capacity = packed.Shape[1];
            using var ones = Tensor.Ones([dim], packed.Device);
            using var bytes = Tensor.Empty([rows, capacity, dim], packed.Device);
            using var wide = Tensor.Empty([rows, capacity, dim], packed.Device, zeroed: true);
            packed.Backend.Int8Dequantize(packed.Storage, ones.Storage, bytes.Storage, rows * capacity, dim);
            packed.Backend.BroadcastAxis(scales.Storage, wide.Storage, rows * capacity, dim, 1, 1f);   // each row's scale along it
            return bytes * wide;
        }

        public override Tensor Attend(Tensor q, KeyValueCache cache, DecodingContext context, int steps, float scale, bool decoderKernels)
        {
            if (cache.HeadDim <= q.Backend.Capabilities.DecodeAttentionHeadDim)
            {
                return Tensor.AttentionInt8(q, cache, context.PositionOf(cache), steps, scale, tiled: steps >= 8);   // only the filled positions
            }

            var weights = Tensor.AttentionScoresInt8(q, cache).ScaleMaskSoftmax(scale, context.MaskOf(cache));
            return Tensor.AttentionContextInt8(weights, cache);
        }

        public Tensor? AttendVariant(Tensor q, KeyValueCache cache, DecodingContext context, int steps, float scale, AttentionVariant variant) =>
            cache.HeadDim <= q.Backend.Capabilities.DecodeAttentionHeadDim
                ? Tensor.AttentionInt8(q, cache, context.PositionOf(cache), steps, scale, tiled: steps >= 8, variant)
                : null;
    }

    private sealed class BFloat16Layout : KeyValueLayout, IRowStartsLayout, IRecordableLayout, ISlidingWindowLayout, IFusedWriteLayout
    {
        public override string Name => "bfloat16";

        public override string Summary => "bfloat16 values, half the memory of float32";

        public override int RowWidth(int headDim) => (headDim + 1) / 2;

        public bool HalfWords => true;

        public override void Write(Tensor keys, Tensor values, KeyValueCache cache, Tensor position)
        {
            Tensor.WriteKeyValuesBFloat16(keys, cache.Keys, position, cache.HeadDim);
            Tensor.WriteKeyValuesBFloat16(values, cache.Values, position, cache.HeadDim);
        }

        // Each cached row is a row of bfloat16 pairs, as packed weights hold them.
        public override Tensor Expand(KeyValueCache cache, bool keys)
        {
            var packed = keys ? cache.Keys : cache.Values;
            int rows = packed.Shape[0], capacity = packed.Shape[1];
            var y = Tensor.Empty([rows, capacity, cache.HeadDim], packed.Device);
            packed.Backend.BFloat16Dequantize(packed.Storage, y.Storage, rows * capacity, cache.HeadDim);
            return y;
        }

        // Only the first slots: the packed words of the prefix copied out, then expanded.
        public override Tensor ExpandPrefix(KeyValueCache cache, bool keys, int positions)
        {
            var packed = keys ? cache.Keys : cache.Values;
            int rows = packed.Shape[0];
            positions = Math.Clamp(positions, 1, packed.Shape[1]);
            if (positions == packed.Shape[1])
            {
                return Expand(cache, keys);
            }

            using var prefix = packed.Narrow(1, 0, positions);
            var y = Tensor.Empty([rows, positions, cache.HeadDim], packed.Device);
            packed.Backend.BFloat16Dequantize(prefix.Storage, y.Storage, rows * positions, cache.HeadDim);
            return y;
        }

        // Decoder models read the halves directly; the multi-head attention layer attends through the expanded values.
        public override Tensor Attend(Tensor q, KeyValueCache cache, DecodingContext context, int steps, float scale, bool decoderKernels) =>
            decoderKernels
                ? Tensor.AttentionBFloat16(q, cache, context.PositionOf(cache), steps, scale, tiled: steps >= 8)   // only the filled positions
                : base.Attend(q, cache, context, steps, scale, decoderKernels);

        public Tensor? AttendVariant(Tensor q, KeyValueCache cache, DecodingContext context, int steps, float scale, AttentionVariant variant) =>
            Tensor.AttentionBFloat16(q, cache, context.PositionOf(cache), steps, scale, tiled: steps >= 8, variant);

        public Tensor? AttendRows(Tensor q, KeyValueCache cache, Tensor position, int steps, float scale, Tensor starts, int headsPerRow,
            AttentionVariant variant) =>
            Tensor.AttentionRowsBFloat16(q, cache, position, steps, scale, starts, headsPerRow, variant);
    }

}
