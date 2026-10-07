// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;

namespace Idrak.Abstraction.Generation;

/// <summary>
/// How a <see cref="KeyValueCache"/> stores keys and values and attends over them: the attention layers write and read a
/// cache through its layout instead of branching on the format. Chosen once when the cache is created (decoding steps
/// recorded as CUDA graphs keep the same kernels). The built-in formats (<see cref="KeyValueFormat"/>) use their own
/// kernels; a format of one's own derives from this class, implements <see cref="RowWidth"/>, <see cref="Write"/> and
/// <see cref="Expand"/>, and attends through the default <see cref="Attend"/> (keys and values expanded to float32, then
/// masked products), which runs on every backend. Register it by name with <see cref="KeyValueLayouts.Register"/>.
/// </summary>
/// <example>
/// <code>
/// KeyValueLayouts.Register("my-format", new MyLayout());
/// using var context = new DecodingContext(device, batch: 1, capacity: 512, KeyValueLayouts.Get("my-format"));
/// var generator = new TextGenerator(model, tokenizer, 512) { CacheLayout = KeyValueLayouts.Get("my-format") };
/// </code>
/// </example>
public abstract class KeyValueLayout
{
    /// <summary>The format's name (the built-ins: "float32", "int8", "bfloat16").</summary>
    public abstract string Name { get; }

    /// <summary>The built-in format this layout stores, or <see cref="KeyValueFormat.Custom"/> for a format of one's own.</summary>
    public KeyValueFormat Format => BuiltInFormat;

    /// <summary>
    /// Elements (four bytes each) per cached row of <paramref name="headDim"/> values: <see cref="KeyValueCache.Keys"/> and
    /// <see cref="KeyValueCache.Values"/> are [rows, capacity, RowWidth(headDim)].
    /// </summary>
    public abstract int RowWidth(int headDim);

    /// <summary>
    /// Whether each cached row has a scale: the cache then holds <see cref="KeyValueCache.KeyScales"/> and
    /// <see cref="KeyValueCache.ValueScales"/>, [rows, capacity]. Default false.
    /// </summary>
    public virtual bool HasScales => false;

    /// <summary>
    /// True when <see cref="Write"/> and <see cref="Attend"/> only queue work on the device (no reads back to the host, no
    /// uploads), so a decoding step can be recorded once as a CUDA graph and replayed. Default false (generation then runs
    /// every step as it is); the built-in formats are recordable.
    /// </summary>
    public virtual bool Recordable => false;

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
        var weights = q.MatMul(keys, transposeB: true).ScaleMaskSoftmax(scale, context.Mask);
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

    /// <summary>The built-ins' own format (only they override it).</summary>
    internal virtual KeyValueFormat BuiltInFormat => KeyValueFormat.Custom;

    /// <summary>
    /// Whether the fused projection kernel (norms, rotary positions, head layout) may write keys and values straight into
    /// the cache; <see cref="HalfWords"/> tells it to write bfloat16 pairs.
    /// </summary>
    internal virtual bool FusedWrite => false;

    /// <summary>Whether values are stored as bfloat16 pairs (two per element).</summary>
    internal virtual bool HalfWords => false;

    /// <summary>Whether rows of different lengths (per-row starts, batched prompts) can attend from this cache.</summary>
    internal virtual bool RowStarts => false;

    /// <summary>
    /// <see cref="Attend"/> for a decoder layer with a sliding window or soft-capped scores (<paramref name="variant"/>),
    /// through the format's own kernels, which start each query at its window and cap the scores; null when the format
    /// has none for this head size (the layer then attends through basic operations over every cached slot).
    /// </summary>
    internal virtual Tensor? AttendVariant(Tensor q, KeyValueCache cache, DecodingContext context, int steps, float scale, AttentionVariant variant) => null;

    /// <inheritdoc />
    public override string ToString() => Name;
}

/// <summary>
/// The key/value cache formats by name: "float32", "int8" and "bfloat16" (the built-in <see cref="KeyValueFormat"/>s) are
/// registered; add formats of one's own with <see cref="Register"/>, so options and tools can choose them by name.
/// </summary>
public static class KeyValueLayouts
{
    // The built-ins, indexed by their format.
    private static readonly KeyValueLayout[] BuiltIn = [new Float32Layout(), new Int8Layout(), new BFloat16Layout()];

    private static readonly Dictionary<string, KeyValueLayout> Registry = new(StringComparer.OrdinalIgnoreCase)
    {
        ["float32"] = BuiltIn[0],
        ["int8"] = BuiltIn[1],
        ["bfloat16"] = BuiltIn[2],
    };

    /// <summary>The built-in layout storing <paramref name="format"/>.</summary>
    public static KeyValueLayout For(KeyValueFormat format) =>
        format is >= KeyValueFormat.Float32 and <= KeyValueFormat.BFloat16 ? BuiltIn[(int)format]
            : throw new ArgumentException($"{format} is not a built-in format; pass the KeyValueLayout itself (or its name to KeyValueLayouts.Get).", nameof(format));

    /// <summary>
    /// Registers (or replaces) the format <paramref name="name"/> (names ignore case). Choosing a built-in
    /// <see cref="KeyValueFormat"/> always uses the built-in layout, whatever is registered under its name.
    /// </summary>
    /// <param name="name">The name to choose the format by.</param>
    /// <param name="layout">How the format stores and attends.</param>
    public static void Register(string name, KeyValueLayout layout)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(layout);
        lock (Registry)
        {
            Registry[name] = layout;
        }
    }

    /// <summary>The registered format names.</summary>
    public static IReadOnlyCollection<string> Names
    {
        get
        {
            lock (Registry)
            {
                return [.. Registry.Keys];
            }
        }
    }

    /// <summary>The layout registered as <paramref name="name"/> (any case).</summary>
    public static KeyValueLayout Get(string name)
    {
        lock (Registry)
        {
            return Registry.TryGetValue(name, out var layout) ? layout
                : throw new NotSupportedException($"No key/value cache format '{name}' is registered ({string.Join(", ", Registry.Keys)}); add it with KeyValueLayouts.Register.");
        }
    }

    // Removes a name (tests restore the registry with it).
    internal static void Unregister(string name)
    {
        lock (Registry)
        {
            Registry.Remove(name);
        }
    }

    private sealed class Float32Layout : KeyValueLayout
    {
        public override string Name => "float32";

        internal override KeyValueFormat BuiltInFormat => KeyValueFormat.Float32;

        public override int RowWidth(int headDim) => headDim;

        public override bool Recordable => true;

        internal override bool FusedWrite => true;

        internal override bool RowStarts => true;

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
            var weights = q.MatMul(cache.Keys, transposeB: true).ScaleMaskSoftmax(scale, context.Mask);
            return weights.MatMul(cache.Values);
        }

        internal override Tensor? AttendVariant(Tensor q, KeyValueCache cache, DecodingContext context, int steps, float scale, AttentionVariant variant) =>
            Kernels(q, cache, context, steps, scale, variant);

        private static Tensor? Kernels(Tensor q, KeyValueCache cache, DecodingContext context, int steps, float scale, AttentionVariant variant)
        {
            var capabilities = q.Backend.Capabilities;
            if (steps >= 8 && cache.HeadDim <= capabilities.TiledAttentionHeadDim)
            {
                return Tensor.AttentionTiled(q, cache.Keys, cache.Values, context.Position, steps, scale, variant);   // a prompt: tiled
            }

            return cache.HeadDim <= capabilities.DecodeAttentionHeadDim
                ? Tensor.AttentionDecode(q, cache, context.Position, steps, scale, variant)                        // only the filled positions
                : null;
        }
    }

    private sealed class Int8Layout : KeyValueLayout
    {
        public override string Name => "int8";

        internal override KeyValueFormat BuiltInFormat => KeyValueFormat.Int8;

        public override int RowWidth(int headDim) => (headDim + 3) / 4;

        public override bool HasScales => true;

        public override bool Recordable => true;

        public override void Write(Tensor keys, Tensor values, KeyValueCache cache, Tensor position)
        {
            Tensor.WriteKeyValuesInt8(keys, cache.Keys, cache.KeyScales!, position);
            Tensor.WriteKeyValuesInt8(values, cache.Values, cache.ValueScales!, position);
        }

        // The bytes as floats (each row's bytes are a weight row with unit column scales), times each row's scale.
        public override Tensor Expand(KeyValueCache cache, bool keys)
        {
            var packed = keys ? cache.Keys : cache.Values;
            var scales = keys ? cache.KeyScales! : cache.ValueScales!;
            int rows = packed.Shape[0], capacity = packed.Shape[1], dim = cache.HeadDim;
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
                return Tensor.AttentionInt8(q, cache, context.Position, steps, scale, tiled: steps >= 8);   // only the filled positions
            }

            var weights = Tensor.AttentionScoresInt8(q, cache).ScaleMaskSoftmax(scale, context.Mask);
            return Tensor.AttentionContextInt8(weights, cache);
        }

        internal override Tensor? AttendVariant(Tensor q, KeyValueCache cache, DecodingContext context, int steps, float scale, AttentionVariant variant) =>
            cache.HeadDim <= q.Backend.Capabilities.DecodeAttentionHeadDim
                ? Tensor.AttentionInt8(q, cache, context.Position, steps, scale, tiled: steps >= 8, variant)
                : null;
    }

    private sealed class BFloat16Layout : KeyValueLayout
    {
        public override string Name => "bfloat16";

        internal override KeyValueFormat BuiltInFormat => KeyValueFormat.BFloat16;

        public override int RowWidth(int headDim) => (headDim + 1) / 2;

        public override bool Recordable => true;

        internal override bool FusedWrite => true;

        internal override bool HalfWords => true;

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

        // Decoder models read the halves directly; the multi-head attention layer attends through the expanded values.
        public override Tensor Attend(Tensor q, KeyValueCache cache, DecodingContext context, int steps, float scale, bool decoderKernels) =>
            decoderKernels
                ? Tensor.AttentionBFloat16(q, cache, context.Position, steps, scale, tiled: steps >= 8)   // only the filled positions
                : base.Attend(q, cache, context, steps, scale, decoderKernels);

        internal override Tensor? AttendVariant(Tensor q, KeyValueCache cache, DecodingContext context, int steps, float scale, AttentionVariant variant) =>
            Tensor.AttentionBFloat16(q, cache, context.Position, steps, scale, tiled: steps >= 8, variant);
    }

}
