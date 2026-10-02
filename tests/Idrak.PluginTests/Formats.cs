// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Layers;

namespace Idrak.PluginTests;

/// <summary>
/// A packed weight format of one's own: bfloat16-rounded values held as float32 in a tensor no scope captures, multiplied
/// through the base class's expanded product.
/// </summary>
public sealed class RoundedWeight : PackedWeight
{
    /// <summary>The name it is registered under.</summary>
    public const string FormatName = "outside-rounded";

    private RoundedWeight(Tensor values) => Values = values;

    /// <summary>The rounded values, [rows, columns].</summary>
    public Tensor Values { get; private set; }

    /// <summary>Products computed through this format.</summary>
    public int Products { get; private set; }

    /// <inheritdoc />
    public override string Name => FormatName;

    /// <inheritdoc />
    public override int Rows => Values.Shape[0];

    /// <inheritdoc />
    public override int Columns => Values.Shape[1];

    /// <inheritdoc />
    public override long Bytes => 4L * Values.Size;

    /// <summary>Packs <paramref name="values"/> ([rows, columns], row-major).</summary>
    public static RoundedWeight Pack(ReadOnlySpan<float> values, int rows, int columns, Device device)
    {
        var rounded = new float[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            uint bits = BitConverter.SingleToUInt32Bits(values[i]);
            rounded[i] = BitConverter.UInt32BitsToSingle(((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16) << 16);
        }

        return new RoundedWeight(Tensor.Persistent(rounded, [rows, columns], device));
    }

    /// <inheritdoc />
    public override Tensor Dequantize() => Values * 1f;

    /// <inheritdoc />
    public override IEnumerable<Tensor> Buffers() => [Values];

    /// <inheritdoc />
    protected override void MoveTo(Device device, Func<Tensor, Device, Tensor> move) => Values = move(Values, device);

    /// <inheritdoc />
    public override Tensor MatMul(Tensor input)
    {
        Products++;
        return base.MatMul(input);
    }

    /// <inheritdoc />
    public override void Dispose() => Values.Dispose();
}

/// <summary>
/// A key/value cache format of one's own: each cached row divided by its mean magnitude, which is kept as the row's
/// scale; written with public operations and the protected <see cref="KeyValueLayout.WriteRows"/>, attended through the
/// default composed attention.
/// </summary>
public sealed class ScaledLayout : KeyValueLayout
{
    /// <summary>The name it is registered under.</summary>
    public const string FormatName = "outside-scaled";

    /// <summary>Writes into a cache through this format.</summary>
    public int Writes { get; private set; }

    /// <inheritdoc />
    public override string Name => FormatName;

    /// <inheritdoc />
    public override int RowWidth(int headDim) => headDim;

    /// <inheritdoc />
    public override bool HasScales => true;

    /// <inheritdoc />
    public override void Write(Tensor keys, Tensor values, KeyValueCache cache, Tensor position)
    {
        Writes++;
        WriteScaled(keys, cache.Keys, cache.KeyScales!, position);
        WriteScaled(values, cache.Values, cache.ValueScales!, position);
    }

    private static void WriteScaled(Tensor source, Tensor rows, Tensor scales, Tensor position)
    {
        int n = source.Shape[0], steps = source.Shape[1], dim = source.Shape[2];
        var scale = source.Abs().Mean(2, keepDim: true) + 1e-6f;                  // [n, steps, 1]
        var inverse = scale.Pow(-1f).Reshape(n * steps, 1);
        WriteRows(source * Widen(inverse, dim).Reshape(n, steps, dim), rows, position);
        WriteRows(scale, scales.Reshape(n, scales.Shape[1], 1), position);
    }

    // [r, 1] → [r, dim], each row's value repeated along it.
    private static Tensor Widen(Tensor column, int dim)
    {
        using var ones = Tensor.Ones([1, dim], column.Device);
        return column.MatMul(ones);
    }

    /// <inheritdoc />
    public override Tensor Expand(KeyValueCache cache, bool keys)
    {
        var (rows, scales) = keys ? (cache.Keys, cache.KeyScales!) : (cache.Values, cache.ValueScales!);
        int n = rows.Shape[0], capacity = rows.Shape[1], dim = rows.Shape[2];
        return rows * Widen(scales.Reshape(n * capacity, 1), dim).Reshape(n, capacity, dim);
    }
}
