// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using Idrak.Backends;

namespace Idrak.Layers;

/// <summary>How <see cref="Module.Save(string, WeightFormat)"/> stores floating-point values in a weights file.</summary>
public enum WeightFormat
{
    /// <summary>32-bit floats: exact (4 bytes per value).</summary>
    Float32,

    /// <summary>IEEE half precision (2 bytes per value): about 3 significant digits, range ±65504.</summary>
    Float16,

    /// <summary>bfloat16 (2 bytes per value): the range of float32 with about 2–3 significant digits.</summary>
    BFloat16,
}

/// <summary>The packed formats a <see cref="Linear"/> layer's weights can be held in (see <see cref="PackedWeight"/>).</summary>
public enum PackedFormat
{
    /// <summary>A signed byte per weight with one scale per column (<see cref="Int8Weight"/>).</summary>
    Int8,

    /// <summary>A signed nibble per weight with one scale per group of 32 rows of a column (<see cref="Int4Weight"/>).</summary>
    Int4,

    /// <summary>bfloat16 values (<see cref="BFloat16Weight"/>).</summary>
    BFloat16,
}

/// <summary>
/// Weights of a <see cref="Linear"/> layer held in a packed format (<see cref="Int8Weight"/>, <see cref="Int4Weight"/>,
/// <see cref="BFloat16Weight"/>). The layer and the fused products go through this type instead of checking which
/// format a layer holds: each format says how it multiplies, expands and moves, and which fused paths it takes.
/// </summary>
public abstract class PackedWeight : IDisposable
{
    private protected PackedWeight()
    {
    }

    /// <summary>The format.</summary>
    public abstract PackedFormat Format { get; }

    /// <summary>Input features (rows of the [rows, columns] weight).</summary>
    public abstract int Rows { get; }

    /// <summary>Output features (columns).</summary>
    public abstract int Columns { get; }

    /// <summary>Device memory used, in bytes.</summary>
    public abstract long Bytes { get; }

    /// <summary>The float weights, [rows, columns], on the same device.</summary>
    public abstract Tensor Dequantize();

    /// <inheritdoc />
    public abstract void Dispose();

    // The packed words, and the scales where the format has them (the storages the packed products and offloading read).
    internal abstract Tensor PackedValues { get; }

    internal virtual Tensor? ScaleValues => null;

    internal IEnumerable<Tensor> Buffers() => ScaleValues is { } scales ? [PackedValues, scales] : [PackedValues];

    internal abstract void MoveTo(Device device, Func<Tensor, Device, Tensor> move);

    // Writes the float weights [rows, columns] into `destination` (on the same device).
    internal abstract void DequantizeInto(Storage destination);

    // input [..., rows] · W → [..., columns], reading the packed words.
    internal abstract Tensor MatMul(Tensor input);

    // Whether the LoRA products read this format packed (Tensor.LoraProducts); otherwise the base product runs on its own.
    internal virtual bool LowRankProducts => false;

    // Whether an FP8 copy may be made for training products (Linear.AttachFloat8).
    internal virtual bool Float8Copy => true;

    // dx (+)= g · Wᵀ (+ dt · Aᵀ) reading W as stored; false when the format or the device has no such product.
    internal virtual bool TransposedProduct(Tensor g, Storage dx, int m, int k, int n, float beta, Storage? dt, Storage? a, int rank) => false;

    // Whether a gated feed-forward block applies the activation as the down projection reads its input (4-bit: eight
    // columns per word repay that work), rather than in the gate/up product.
    internal virtual bool ActivationInDownProjection => false;

    /// <summary>Packs [rows, columns] float values in <paramref name="format"/> on <paramref name="device"/>.</summary>
    public static PackedWeight FromValues(PackedFormat format, ReadOnlySpan<float> values, int rows, int columns, Device device) => format switch
    {
        PackedFormat.Int8 => Int8Weight.Quantize(values, rows, columns, device),
        PackedFormat.Int4 => Int4Weight.Quantize(values, rows, columns, device),
        _ => BFloat16Weight.FromValues(values, rows, columns, device),
    };

    // For messages: "int8", "4-bit", "bfloat16"; ToString's short name; and how to get float weights back.
    internal abstract string Description { get; }

    internal abstract string ShortName { get; }

    internal virtual string FloatMethod => "ToFloat32()";
}

/// <summary>
/// The int8 weights of a quantized <see cref="Linear"/> layer: each weight is a signed byte times its output column's
/// scale (symmetric, per-column: scale = max |w| / 127). A quarter of the float32 memory; created by
/// <see cref="ModuleExtensions.QuantizeInt8"/>. Inputs, outputs, biases and LoRA adapters stay float32.
/// </summary>
public sealed class Int8Weight : PackedWeight
{
    /// <inheritdoc />
    public override PackedFormat Format => PackedFormat.Int8;

    internal override Tensor PackedValues => Packed;

    internal override Tensor? ScaleValues => Scales;

    internal override Tensor MatMul(Tensor input) => input.MatMulInt8(this);

    internal override bool Float8Copy => false;

    internal override string Description => "int8";

    internal override string ShortName => "int8";

    internal override string FloatMethod => "DequantizeInt8()";

    private Int8Weight(Tensor packed, Tensor scales, int rows, int columns)
    {
        Packed = packed;
        Scales = scales;
        Rows = rows;
        Columns = columns;
    }

    /// <summary>Input features (rows of the weight matrix).</summary>
    public override int Rows { get; }

    /// <summary>Output features (columns).</summary>
    public override int Columns { get; }

    /// <summary>Device memory used, in bytes (weights and scales).</summary>
    public override long Bytes => 4L * (Packed.Size + Scales.Size);

    /// <summary>The bytes, packed four per element along each row (rows padded to a multiple of four columns).</summary>
    internal Tensor Packed { get; private set; }

    /// <summary>One scale per column.</summary>
    internal Tensor Scales { get; private set; }

    /// <summary>Quantizes a float weight matrix [rows, columns] (on its device).</summary>
    public static Int8Weight Quantize(Tensor weight)
    {
        if (weight.Rank != 2)
        {
            throw new ArgumentException($"Int8 quantization needs a matrix, got {Tensor.FormatShape(weight.Shape)}.", nameof(weight));
        }

        return Quantize(weight.ToArray(), weight.Shape[0], weight.Shape[1], weight.Device);
    }

    /// <summary>
    /// Quantizes weights given as host values [rows, columns] and uploads only the bytes to <paramref name="device"/>
    /// (large models never exist as float32 on the device).
    /// </summary>
    public static unsafe Int8Weight Quantize(ReadOnlySpan<float> values, int rows, int columns, Device device)
    {
        if (values.Length != rows * columns)
        {
            throw new ArgumentException($"{values.Length} values do not fill [{rows}, {columns}].", nameof(values));
        }

        // The worker lambdas cannot capture a span: they read the pinned values through a pointer (no copy).
        fixed (float* pinned = values)
        {
            return Quantize(pinned, rows, columns, device);
        }
    }

    /// <summary>
    /// Packed bytes and scales for [rows, columns] allocated on <paramref name="device"/> without computing anything, for
    /// loading stored values into (<see cref="Module.Load(string)"/>).
    /// </summary>
    internal static Int8Weight Empty(int rows, int columns, Device device)
    {
        int stride = (columns + 3) / 4 * 4;
        return new Int8Weight(
            Tensor.PersistentZeros([rows * stride / 4], device),
            Tensor.PersistentZeros([columns], device),
            rows, columns);
    }

    private static unsafe Int8Weight Quantize(float* source, int rows, int columns, Device device)
    {
        int stride = (columns + 3) / 4 * 4;

        // Column maxima per chunk of rows (all cores), then combined.
        var scales = new float[columns];
        var gate = new Lock();
        HostParallel.For(rows, Math.Max(1, (1 << 16) / Math.Max(1, columns)), (first, last) =>
        {
            var local = new float[columns];
            for (int r = first; r < last; r++)
            {
                for (int j = 0; j < columns; j++)
                {
                    local[j] = MathF.Max(local[j], MathF.Abs(source[r * columns + j]));
                }
            }

            lock (gate)
            {
                for (int j = 0; j < columns; j++)
                {
                    scales[j] = MathF.Max(scales[j], local[j]);
                }
            }
        });
        for (int j = 0; j < columns; j++)
        {
            scales[j] = scales[j] > 0f ? scales[j] / 127f : 1f;
        }

        var bytes = new sbyte[rows * stride];
        HostParallel.For(rows, Math.Max(1, (1 << 16) / Math.Max(1, columns)), (first, last) =>
        {
            for (int r = first; r < last; r++)
            {
                for (int j = 0; j < columns; j++)
                {
                    bytes[r * stride + j] = (sbyte)Math.Clamp(MathF.Round(source[r * columns + j] / scales[j]), -127f, 127f);
                }
            }
        });

        var packed = MemoryMarshal.Cast<sbyte, float>(bytes);
        return new Int8Weight(
            Tensor.Persistent(packed, [packed.Length], device, requiresGrad: false),
            Tensor.Persistent(scales, [columns], device, requiresGrad: false),
            rows, columns);
    }

    /// <summary>The float weights these bytes stand for, [rows, columns], on the same device.</summary>
    public override Tensor Dequantize()
    {
        var w = Tensor.PersistentZeros([Rows, Columns], Packed.Device);
        DequantizeInto(w.Storage);
        return w;
    }

    internal override void DequantizeInto(Storage destination) => Packed.Backend.Int8Dequantize(Packed.Storage, Scales.Storage, destination, Rows, Columns);

    internal override void MoveTo(Device device, Func<Tensor, Device, Tensor> move)
    {
        Packed = move(Packed, device);
        Scales = move(Scales, device);
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        Packed.Dispose();
        Scales.Dispose();
    }
}

/// <summary>
/// The 4-bit weights of a <see cref="Linear"/> layer: each group of 32 input rows of a column shares one float scale and
/// every weight is a signed nibble in -8..7 (the scale is searched per group for the least squared error), about
/// 5 bits per weight; decoding reads 8× less than float32. Created by <see cref="ModuleExtensions.QuantizeInt4"/> or when
/// loading with 4-bit weights; inputs, outputs, biases and LoRA adapters stay float32 (QLoRA-style fine-tuning).
/// </summary>
public sealed class Int4Weight : PackedWeight
{
    /// <inheritdoc />
    public override PackedFormat Format => PackedFormat.Int4;

    internal override Tensor PackedValues => Packed;

    internal override Tensor? ScaleValues => Scales;

    internal override Tensor MatMul(Tensor input) => input.MatMulInt4(this);

    internal override bool LowRankProducts => true;

    internal override bool ActivationInDownProjection => true;

    internal override string Description => "4-bit";

    internal override string ShortName => "int4";

    /// <summary>Weight rows that share one scale per column.</summary>
    public const int GroupSize = 32;

    private Int4Weight(Tensor packed, Tensor scales, int rows, int columns)
    {
        Packed = packed;
        Scales = scales;
        Rows = rows;
        Columns = columns;
    }

    /// <summary>Input features (rows of the weight matrix).</summary>
    public override int Rows { get; }

    /// <summary>Output features (columns).</summary>
    public override int Columns { get; }

    /// <summary>Device memory used, in bytes (weights and scales).</summary>
    public override long Bytes => 4L * (Packed.Size + Scales.Size);

    /// <summary>The nibbles, eight per element along each row (nibble c of element w is column 8w + c).</summary>
    internal Tensor Packed { get; private set; }

    /// <summary>One scale per group of <see cref="GroupSize"/> rows and column: [⌈rows / 32⌉, 8·⌈columns / 8⌉].</summary>
    internal Tensor Scales { get; private set; }

    /// <summary>Quantizes a float weight matrix [rows, columns] (on its device).</summary>
    public static Int4Weight Quantize(Tensor weight)
    {
        if (weight.Rank != 2)
        {
            throw new ArgumentException($"Int4 quantization needs a matrix, got {Tensor.FormatShape(weight.Shape)}.", nameof(weight));
        }

        return Quantize(weight.ToArray(), weight.Shape[0], weight.Shape[1], weight.Device);
    }

    /// <summary>Quantizes host values [rows, columns] and uploads only the nibbles and scales to <paramref name="device"/>.</summary>
    public static unsafe Int4Weight Quantize(ReadOnlySpan<float> values, int rows, int columns, Device device)
    {
        if (values.Length != rows * columns)
        {
            throw new ArgumentException($"{values.Length} values do not fill [{rows}, {columns}].", nameof(values));
        }

        // The worker lambdas cannot capture a span: they read the pinned values through a pointer (no copy).
        fixed (float* pinned = values)
        {
            return Quantize(pinned, rows, columns, device);
        }
    }

    /// <summary>
    /// Nibbles and scales for [rows, columns] allocated on <paramref name="device"/> without computing anything, for
    /// loading stored values into (<see cref="Module.Load(string)"/>).
    /// </summary>
    internal static Int4Weight Empty(int rows, int columns, Device device)
    {
        int words = (columns + 7) / 8, groups = (rows + GroupSize - 1) / GroupSize;
        return new Int4Weight(
            Tensor.PersistentZeros([rows * words], device),
            Tensor.PersistentZeros([groups * words * 8], device),
            rows, columns);
    }

    private static unsafe Int4Weight Quantize(float* source, int rows, int columns, Device device)
    {
        int words = (columns + 7) / 8, groups = (rows + GroupSize - 1) / GroupSize;
        var packed = new uint[rows * words];
        var scales = new float[groups * words * 8];
        HostParallel.For(groups, Math.Max(1, (1 << 12) / Math.Max(1, columns)), (first, last) =>
        {
            Span<float> w = stackalloc float[GroupSize];
            Span<sbyte> q = stackalloc sbyte[GroupSize];
            Span<sbyte> best = stackalloc sbyte[GroupSize];
            for (int g = first; g < last; g++)
            {
                int r0 = g * GroupSize, count = Math.Min(rows, r0 + GroupSize) - r0;
                for (int j = 0; j < columns; j++)
                {
                    float extreme = 0f;
                    for (int i = 0; i < count; i++)
                    {
                        w[i] = source[(r0 + i) * columns + j];
                        extreme = MathF.Abs(w[i]) > MathF.Abs(extreme) ? w[i] : extreme;
                    }

                    float scale = 0f;
                    if (extreme != 0f)
                    {
                        // Divisors t around -8 (the group's extreme maps near -8); for each, the least-squares scale of
                        // the rounded values; keep the smallest squared error (t = -8 is the plain rule, never worse).
                        double bestError = double.MaxValue;
                        for (int step = -10; step <= 10; step++)
                        {
                            float t = -8f + 0.1f * step, inverse = t / extreme;
                            double wq = 0, qq = 0;
                            for (int i = 0; i < count; i++)
                            {
                                q[i] = (sbyte)Math.Clamp((int)MathF.Round(w[i] * inverse), -8, 7);
                                wq += w[i] * q[i];
                                qq += q[i] * q[i];
                            }

                            if (qq == 0)
                            {
                                continue;
                            }

                            float d = (float)(wq / qq);
                            double error = 0;
                            for (int i = 0; i < count; i++)
                            {
                                double e = w[i] - d * q[i];
                                error += e * e;
                            }

                            if (error < bestError)
                            {
                                (bestError, scale) = (error, d);
                                q[..count].CopyTo(best);
                            }
                        }
                    }
                    else
                    {
                        best.Clear();
                    }

                    scales[g * words * 8 + j] = scale;
                    for (int i = 0; i < count; i++)
                    {
                        packed[(r0 + i) * words + (j >> 3)] |= (uint)(best[i] & 15) << (4 * (j & 7));
                    }
                }
            }
        });

        return new Int4Weight(
            Tensor.Persistent(MemoryMarshal.Cast<uint, float>(packed), [packed.Length], device, requiresGrad: false),
            Tensor.Persistent(scales, [scales.Length], device, requiresGrad: false),
            rows, columns);
    }

    /// <summary>The float weights these nibbles stand for, [rows, columns], on the same device.</summary>
    public override Tensor Dequantize()
    {
        var w = Tensor.PersistentZeros([Rows, Columns], Packed.Device);
        DequantizeInto(w.Storage);
        return w;
    }

    internal override void DequantizeInto(Storage destination) => Packed.Backend.Int4Dequantize(Packed.Storage, Scales.Storage, destination, Rows, Columns);

    internal override void MoveTo(Device device, Func<Tensor, Device, Tensor> move)
    {
        Packed = move(Packed, device);
        Scales = move(Scales, device);
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        Packed.Dispose();
        Scales.Dispose();
    }
}

/// <summary>
/// The bfloat16 weights of a <see cref="Linear"/> layer: each weight keeps float32's range with an 8-bit mantissa (about
/// 3 significant digits), in half the memory; decoding reads half the bytes of float32. Hugging Face checkpoints are
/// usually stored this way, so loading them as bfloat16 is exact. Created by <see cref="ModuleExtensions.ToBFloat16"/>
/// or when loading with bfloat16 weights; inputs, outputs, biases and LoRA adapters stay float32.
/// </summary>
public sealed class BFloat16Weight : PackedWeight
{
    /// <inheritdoc />
    public override PackedFormat Format => PackedFormat.BFloat16;

    internal override Tensor PackedValues => Packed;

    internal override Tensor MatMul(Tensor input) => input.MatMulBFloat16(this);

    internal override bool LowRankProducts => true;

    internal override bool TransposedProduct(Tensor g, Storage dx, int m, int k, int n, float beta, Storage? dt, Storage? a, int rank) =>
        g.Backend.BFloat16TransposedMatMul(g.Storage, Packed.Storage, dx, m, k, n, beta, dt, a, rank);

    internal override string Description => "bfloat16";

    internal override string ShortName => "bf16";

    private BFloat16Weight(Tensor packed, int rows, int columns)
    {
        Packed = packed;
        Rows = rows;
        Columns = columns;
    }

    /// <summary>Input features (rows of the weight matrix).</summary>
    public override int Rows { get; }

    /// <summary>Output features (columns).</summary>
    public override int Columns { get; }

    /// <summary>Device memory used, in bytes.</summary>
    public override long Bytes => 4L * Packed.Size;

    /// <summary>The values, two per element along each row (rows padded to an even number of columns).</summary>
    internal Tensor Packed { get; private set; }

    /// <summary>Rounds a float weight matrix [rows, columns] (on its device) to bfloat16.</summary>
    public static BFloat16Weight Convert(Tensor weight)
    {
        if (weight.Rank != 2)
        {
            throw new ArgumentException($"bfloat16 weights need a matrix, got {Tensor.FormatShape(weight.Shape)}.", nameof(weight));
        }

        return FromValues(weight.ToArray(), weight.Shape[0], weight.Shape[1], weight.Device);
    }

    /// <summary>Rounds host values [rows, columns] to bfloat16 (to nearest, ties to even) and uploads only those.</summary>
    public static unsafe BFloat16Weight FromValues(ReadOnlySpan<float> values, int rows, int columns, Device device)
    {
        if (values.Length != rows * columns)
        {
            throw new ArgumentException($"{values.Length} values do not fill [{rows}, {columns}].", nameof(values));
        }

        // The worker lambdas cannot capture a span: they read the pinned values through a pointer (no copy).
        fixed (float* pinned = values)
        {
            return FromValues(pinned, rows, columns, device);
        }
    }

    /// <summary>
    /// Room for bfloat16 values [rows, columns] on <paramref name="device"/>, allocated without computing anything, for
    /// loading stored values into (<see cref="Module.Load(string)"/>).
    /// </summary>
    internal static BFloat16Weight Empty(int rows, int columns, Device device)
    {
        int stride = (columns + 1) / 2 * 2;
        return new BFloat16Weight(Tensor.PersistentZeros([rows * stride / 2], device), rows, columns);
    }

    private static unsafe BFloat16Weight FromValues(float* source, int rows, int columns, Device device)
    {
        int stride = (columns + 1) / 2 * 2;
        var halves = new ushort[rows * stride];
        HostParallel.For(rows, Math.Max(1, (1 << 16) / Math.Max(1, columns)), (first, last) =>
        {
            for (int r = first; r < last; r++)
            {
                for (int j = 0; j < columns; j++)
                {
                    halves[r * stride + j] = Round(source[r * columns + j]);
                }
            }
        });

        var packed = MemoryMarshal.Cast<ushort, float>(halves);
        return new BFloat16Weight(Tensor.Persistent(packed, [packed.Length], device, requiresGrad: false), rows, columns);
    }

    /// <summary>A value's bfloat16 bits (round to nearest, ties to even; NaN stays NaN).</summary>
    internal static ushort Round(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        if (float.IsNaN(value))
        {
            return (ushort)((bits >> 16) | 0x40);
        }

        return (ushort)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
    }

    /// <summary>The float weights, [rows, columns], on the same device.</summary>
    public override Tensor Dequantize()
    {
        var w = Tensor.PersistentZeros([Rows, Columns], Packed.Device);
        DequantizeInto(w.Storage);
        return w;
    }

    internal override void DequantizeInto(Storage destination) => Packed.Backend.BFloat16Dequantize(Packed.Storage, destination, Rows, Columns);

    internal override void MoveTo(Device device, Func<Tensor, Device, Tensor> move) => Packed = move(Packed, device);

    /// <inheritdoc />
    public override void Dispose() => Packed.Dispose();
}
