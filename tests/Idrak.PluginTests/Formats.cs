// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Layers;

namespace Idrak.PluginTests;

/// <summary>
/// A packed weight format of one's own: values rounded to bfloat16, two to a four-byte word (low half first) in a tensor
/// no scope captures, multiplied through the base class's expanded product. The weights are expanded by the plug-in
/// operation <see cref="PluginKernels.Unpack"/>: its default kernel on the host, or a device kernel registered for it
/// (<see cref="PluginKernels.Install"/>).
/// </summary>
public sealed class RoundedWeight : PackedWeight
{
    /// <summary>The name it is registered under.</summary>
    public const string FormatName = "outside-rounded";

    private RoundedWeight(Tensor words, int rows, int columns) => (Words, Rows, Columns) = (words, rows, columns);

    /// <summary>The packed words, ⌈rows · columns / 2⌉ of them.</summary>
    public Tensor Words { get; private set; }

    /// <summary>Products computed through this format.</summary>
    public int Products { get; private set; }

    /// <inheritdoc />
    public override string Name => FormatName;

    /// <inheritdoc />
    public override int Rows { get; }

    /// <inheritdoc />
    public override int Columns { get; }

    /// <inheritdoc />
    public override long Bytes => 4L * Words.Size;

    /// <summary>Packs <paramref name="values"/> ([rows, columns], row-major).</summary>
    public static RoundedWeight Pack(ReadOnlySpan<float> values, int rows, int columns, Device device)
    {
        var words = new float[(values.Length + 1) / 2];
        for (int i = 0; i < values.Length; i++)
        {
            uint bits = BitConverter.SingleToUInt32Bits(values[i]);
            uint half = (bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16;              // to nearest, ties to even
            words[i >> 1] = BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(words[i >> 1]) | half << (16 * (i & 1)));
        }

        return new RoundedWeight(Tensor.Persistent(words, [words.Length], device), rows, columns);
    }

    /// <inheritdoc />
    public override Tensor Dequantize()
    {
        var values = Tensor.Empty([Rows, Columns], Words.Device);
        var backend = values.Backend;
        PluginKernels.Unpack.KernelFor(backend)(backend, Words.Storage, values.Storage, Rows * Columns);
        return values;
    }

    /// <inheritdoc />
    public override IEnumerable<Tensor> Buffers() => [Words];

    /// <inheritdoc />
    protected override void MoveTo(Device device, Func<Tensor, Device, Tensor> move) => Words = move(Words, device);

    /// <inheritdoc />
    public override Tensor MatMul(Tensor input)
    {
        Products++;
        return base.MatMul(input);
    }

    /// <inheritdoc />
    public override void Dispose() => Words.Dispose();
}

/// <summary>
/// A key/value cache format of one's own: each cached row divided by its mean magnitude, which is kept as the row's
/// scale; written with public operations and the protected <see cref="KeyValueLayout.WriteRows"/>, attended through the
/// default composed attention. Its rows are expanded by the plug-in operation <see cref="PluginKernels.Scale"/>: its
/// default kernel (composed of the library's operations), or a device kernel registered for it
/// (<see cref="PluginKernels.Install"/>).
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
        var expanded = Tensor.Empty(rows.Shape, rows.Device);
        var backend = rows.Backend;
        PluginKernels.Scale.KernelFor(backend)(backend, rows.Storage, scales.Storage, expanded.Storage, rows.Size, rows.Shape[2]);
        return expanded;
    }
}
