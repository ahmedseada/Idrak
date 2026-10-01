// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Layers;

/// <summary>
/// How one <see cref="KeyValueFormat"/> stores keys and values and attends over them: the attention layers write and read
/// a cache through its layout instead of branching on the format. Chosen once when the cache is created (decoding
/// steps recorded as CUDA graphs keep the same kernels); a new format is a new layout and an entry in
/// <see cref="KeyValueLayouts"/>.
/// </summary>
internal interface IKeyValueLayout
{
    /// <summary>The format this layout stores.</summary>
    KeyValueFormat Format { get; }

    /// <summary>Elements (four bytes each) per cached row of <paramref name="headDim"/> values.</summary>
    int RowWidth(int headDim);

    /// <summary>Whether each cached row has a scale (<see cref="KeyValueCache.KeyScales"/>, <see cref="KeyValueCache.ValueScales"/>).</summary>
    bool HasScales { get; }

    /// <summary>
    /// Whether the fused projection kernel (norms, rotary positions, head layout) may write keys and values straight into
    /// the cache; <see cref="HalfWords"/> tells it to write bfloat16 pairs.
    /// </summary>
    bool FusedWrite { get; }

    /// <summary>Whether values are stored as bfloat16 pairs (two per element).</summary>
    bool HalfWords { get; }

    /// <summary>Whether rows of different lengths (per-row starts, batched prompts) can attend from this cache.</summary>
    bool RowStarts { get; }

    /// <summary>Writes keys and values [rows, steps, headDim] at the device-side position.</summary>
    void Write(Tensor keys, Tensor values, KeyValueCache cache, Tensor position);

    /// <summary>
    /// Attention of q [rows, steps, headDim] over the filled positions → [rows, steps, headDim]. With
    /// <paramref name="decoderKernels"/>, the layout may use the decoding and tiled attention kernels (decoder models);
    /// without, a float32 cache is read with plain products behind the mask (the multi-head attention layer).
    /// </summary>
    Tensor Attend(Tensor q, KeyValueCache cache, DecodingContext context, int steps, float scale, bool decoderKernels);
}

/// <summary>The layout of each <see cref="KeyValueFormat"/>, indexed by the format.</summary>
internal static class KeyValueLayouts
{
    private static readonly IKeyValueLayout[] Layouts = [new Float32Layout(), new Int8Layout(), new BFloat16Layout()];

    /// <summary>The layout storing <paramref name="format"/>.</summary>
    public static IKeyValueLayout For(KeyValueFormat format) => Layouts[(int)format];

    private sealed class Float32Layout : IKeyValueLayout
    {
        public KeyValueFormat Format => KeyValueFormat.Float32;

        public int RowWidth(int headDim) => headDim;

        public bool HasScales => false;

        public bool FusedWrite => true;

        public bool HalfWords => false;

        public bool RowStarts => true;

        public void Write(Tensor keys, Tensor values, KeyValueCache cache, Tensor position)
        {
            Tensor.WriteKeyValues(keys, cache.Keys, position);
            Tensor.WriteKeyValues(values, cache.Values, position);
        }

        public Tensor Attend(Tensor q, KeyValueCache cache, DecodingContext context, int steps, float scale, bool decoderKernels)
        {
            var capabilities = q.Backend.Capabilities;
            if (decoderKernels && steps >= 8 && cache.HeadDim <= capabilities.TiledAttentionHeadDim)
            {
                return Tensor.AttentionTiled(q, cache.Keys, cache.Values, context.Position, steps, scale);   // a prompt: tiled
            }

            if (decoderKernels && cache.HeadDim <= capabilities.DecodeAttentionHeadDim)
            {
                return Tensor.AttentionDecode(q, cache, context.Position, steps, scale);      // only the filled positions
            }

            // Every slot of the cache, the unwritten ones masked out.
            var weights = q.MatMul(cache.Keys, transposeB: true).ScaleMaskSoftmax(scale, context.Mask);
            return weights.MatMul(cache.Values);
        }
    }

    private sealed class Int8Layout : IKeyValueLayout
    {
        public KeyValueFormat Format => KeyValueFormat.Int8;

        public int RowWidth(int headDim) => (headDim + 3) / 4;

        public bool HasScales => true;

        public bool FusedWrite => false;

        public bool HalfWords => false;

        public bool RowStarts => false;

        public void Write(Tensor keys, Tensor values, KeyValueCache cache, Tensor position)
        {
            Tensor.WriteKeyValuesInt8(keys, cache.Keys, cache.KeyScales!, position);
            Tensor.WriteKeyValuesInt8(values, cache.Values, cache.ValueScales!, position);
        }

        public Tensor Attend(Tensor q, KeyValueCache cache, DecodingContext context, int steps, float scale, bool decoderKernels)
        {
            if (cache.HeadDim <= q.Backend.Capabilities.DecodeAttentionHeadDim)
            {
                return Tensor.AttentionInt8(q, cache, context.Position, steps, scale, tiled: steps >= 8);   // only the filled positions
            }

            var weights = Tensor.AttentionScoresInt8(q, cache).ScaleMaskSoftmax(scale, context.Mask);
            return Tensor.AttentionContextInt8(weights, cache);
        }
    }

    private sealed class BFloat16Layout : IKeyValueLayout
    {
        public KeyValueFormat Format => KeyValueFormat.BFloat16;

        public int RowWidth(int headDim) => (headDim + 1) / 2;

        public bool HasScales => false;

        public bool FusedWrite => true;

        public bool HalfWords => true;

        public bool RowStarts => false;

        public void Write(Tensor keys, Tensor values, KeyValueCache cache, Tensor position)
        {
            Tensor.WriteKeyValuesBFloat16(keys, cache.Keys, position, cache.HeadDim);
            Tensor.WriteKeyValuesBFloat16(values, cache.Values, position, cache.HeadDim);
        }

        public Tensor Attend(Tensor q, KeyValueCache cache, DecodingContext context, int steps, float scale, bool decoderKernels) =>
            decoderKernels
                ? Tensor.AttentionBFloat16(q, cache, context.Position, steps, scale, tiled: steps >= 8)   // only the filled positions
                : throw new NotSupportedException("A bfloat16 KV cache is supported by decoder models (DecoderSpec); use Float32 or Int8 here.");
    }
}
