// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Collections.Concurrent;
using Idrak.Abstraction.Devices;

namespace Idrak.Abstraction.Generation;

/// <summary>
/// A packed format made from a registered block element type that stores and widens float32 values (its
/// <see cref="ElementType.FromFloat32"/> and <see cref="ElementType.ToFloat32"/>; ggml's q8_0 among them), under the type's
/// own name: float values are stored in the type, a row of k inputs per output as a file holds them, and kept as stored
/// (<see cref="StoredWeight"/>) where the device multiplies the type (<see cref="WeightStorage.Multiplies"/>). Elsewhere
/// they are expanded to float32 once, as a load of a file in the type expands them for that device, so a run there gives
/// the stored values' products. Nothing here names a type: a type registered with both conversions packs this way.
/// </summary>
internal sealed class StoredPackedFormat : IPackedFormat, IRowPackedFormat
{
    private static readonly ConcurrentDictionary<ElementType, StoredPackedFormat> Made = new();

    private readonly ElementType _type;

    private StoredPackedFormat(ElementType type) => _type = type;

    /// <summary>The format of the block element type named <paramref name="name"/>, or null when there is none that stores and widens.</summary>
    public static StoredPackedFormat? Of(ReadOnlySpan<char> name) =>
        ElementTypes.Find(name) is { FromFloat32: not null, ToFloat32: not null } type && (type.IsBlock || type.Float is not null)
            ? Made.GetOrAdd(type, static t => new StoredPackedFormat(t)) : null;

    public string Name => _type.Name;

    public string Summary => $"{_type.Describe()} stored from float32 values, kept as stored where the device multiplies it";

    public IReadOnlySet<string> Keys => OptionKeys.None;

    public override string ToString() => _type.Name;

    public unsafe IPackedWeight Create(PackRequest request)
    {
        int k = request.Rows, n = request.Columns;
        StoredWeight.CheckRows(_type, k);
        var device = request.Device;
        fixed (float* pinned = request.Values)
        {
            nint values = (nint)pinned;
            if (WeightStorage.Multiplies(device.Backend, _type))
            {
                return StoredWeight.FromFill(_type, k, n, null, device, stored =>
                {
                    StoreColumns((float*)values, k, n, stored);
                    return true;
                })!;
            }

            // As a load expands a file's weights for a device that has no product of the type: the stored values in float32.
            long rowBytes = _type.ByteCount(k);
            var bytes = new byte[checked(rowBytes * n)];
            StoreColumns((float*)values, k, n, bytes);
            var expanded = new float[(long)k * n];
            var row = new float[k];
            for (int j = 0; j < n; j++)
            {
                _type.Decode(bytes.AsSpan((int)(j * rowBytes), (int)rowBytes), row);
                for (int r = 0; r < k; r++)
                {
                    expanded[(long)r * n + j] = row[r];
                }
            }

            return new ExpandedWeight(_type.Name, Tensor.Persistent(expanded, [k, n], device));
        }
    }

    public IPackedWeight? CreateFromRows(WeightRows rows, int inputs, int outputs, Device device)
    {
        int k = inputs, n = outputs;
        StoredWeight.CheckRows(_type, k);
        if (!WeightStorage.Multiplies(device.Backend, _type))
        {
            return null;                                                         // expanded to float32: packed from the values
        }

        long rowBytes = _type.ByteCount(k);
        var type = _type;
        return StoredWeight.FromFill(type, k, n, null, device, stored => EncodeRows(rows, k, n, stored, rowBytes, (_, row, into) => type.Encode(row, into)));
    }

    /// <summary>
    /// The library's int8 or bfloat16 weights as the generic and measured routes keep them (ProductRoutes): stored in the
    /// file's layout, n rows of k values. int8: today's codes, clamp(round(w / s), −127, 127) with one scale s = max |w| /
    /// 127 an output (1 where an output is all zeros), now a scale per stored row; bfloat16: each value rounded to nearest,
    /// ties to even, as the packed bfloat16 weights round it.
    /// </summary>
    internal static unsafe IPackedWeight Routed(string format, ReadOnlySpan<float> values, int k, int n, Device device)
    {
        fixed (float* pinned = values)
        {
            nint at = (nint)pinned;
            return RoutedRows(format, (first, count, destination) =>
            {
                Columns((float*)at, k, n, first, count, destination);
                return true;
            }, k, n, device)!;
        }
    }

    /// <summary>
    /// <see cref="Routed"/> from the weights' output rows read a block at a time (a checkpoint's rows as stored): the same
    /// weights, with no float32 copy of the matrix and no transposition. Null when the rows cannot be read so.
    /// </summary>
    internal static IPackedWeight? RoutedRows(string format, WeightRows rows, int k, int n, Device device)
    {
        if (format != "int8")
        {
            var bf16 = ElementType.BFloat16;
            return StoredWeight.FromFill(bf16, k, n, null, device, stored => EncodeRows(rows, k, n, stored, bf16.ByteCount(k), (_, row, into) => bf16.Encode(row, into)));
        }

        // int8: each row's scale max |w| / 127 (1 for a row of zeros), its codes clamp(round(w / s), −127, 127).
        var scales = new float[n];
        return StoredWeight.FromFill(ElementType.Int8, k, n, new Formats.StoredScaling(scales, null, new StoredScales(1, k, 1)), device, stored => EncodeRows(rows, k, n, stored, k, (j, row, into) =>
        {
            float largest = 0f;
            foreach (float v in row)
            {
                largest = MathF.Max(largest, MathF.Abs(v));
            }

            float scale = largest > 0f ? largest / 127f : 1f;
            scales[j] = scale;
            for (int r = 0; r < row.Length; r++)
            {
                into[r] = (byte)(sbyte)Math.Clamp(MathF.Round(row[r] / scale), -127f, 127f);
            }
        }));
    }

    /// <summary>Encodes one output row's values into its stored bytes.</summary>
    internal delegate void RowEncoder(int row, ReadOnlySpan<float> values, Span<byte> stored);

    /// <summary>
    /// Each output row of <paramref name="rows"/> encoded into its <paramref name="rowBytes"/> of <paramref name="stored"/>:
    /// a block of rows read at a time into one pooled buffer (about 8 MB of float32), the block's rows encoded in parallel.
    /// False when the first block cannot be read (nothing written).
    /// </summary>
    internal static unsafe bool EncodeRows(WeightRows rows, int k, int n, Span<byte> stored, long rowBytes, RowEncoder encode)
    {
        int per = (int)Math.Clamp((1L << 21) / Math.Max(1, k), 1, Math.Max(1, n));
        float[] buffer = ArrayPool<float>.Shared.Rent(Math.Max(1, per * k));
        try
        {
            fixed (byte* bytes = stored)
            {
                nint into = (nint)bytes;
                for (int first = 0; first < n; first += per)
                {
                    int count = Math.Min(per, n - first);
                    if (!rows(first, count, buffer.AsSpan(0, count * k)))
                    {
                        return first == 0 ? false
                            : throw new InvalidDataException($"The weights' output rows from {first} could not be read, though the rows before them were.");
                    }

                    int block = first;
                    Parallel.For(0, count, ComputeResources.ParallelOptions, c =>
                        encode(block + c, buffer.AsSpan(c * k, k), new Span<byte>((byte*)into + (block + c) * rowBytes, (int)rowBytes)));
                }
            }

            return true;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
        }
    }

    // Columns [first, first + count) of [k, n] values as rows of k (the values read along their rows).
    private static unsafe void Columns(float* values, int k, int n, int first, int count, Span<float> destination)
    {
        for (int r = 0; r < k; r++)
        {
            long at = (long)r * n + first;
            for (int c = 0; c < count; c++)
            {
                destination[c * k + r] = values[at + c];
            }
        }
    }

    // The [k, n] values as n stored rows of k values each into `stored`, a block of 32 columns at a time so the values are
    // read in their order (rule 72), the blocks in parallel, each into a pooled buffer.
    private unsafe void StoreColumns(float* values, int k, int n, Span<byte> stored)
    {
        const int Width = 32;
        long rowBytes = _type.ByteCount(k);
        var type = _type;
        fixed (byte* bytes = stored)
        {
            nint into = (nint)bytes, from = (nint)values;
            Parallel.For(0, (n + Width - 1) / Width, ComputeResources.ParallelOptions, block =>
            {
                int j0 = block * Width, count = Math.Min(Width, n - j0);
                float[] rows = ArrayPool<float>.Shared.Rent(count * k);
                try
                {
                    Columns((float*)from, k, n, j0, count, rows);
                    for (int c = 0; c < count; c++)
                    {
                        type.Encode(rows.AsSpan(c * k, k), new Span<byte>((byte*)into + (j0 + c) * rowBytes, (int)rowBytes));
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(rows);
                }
            });
        }
    }
}

/// <summary>
/// Float32 weights a packed format expanded once for a device that has no product of its type (<see cref="StoredPackedFormat"/>):
/// the float32 product, named after the type they came from.
/// </summary>
internal sealed class ExpandedWeight(string name, Tensor weights) : PackedWeight, IRecordable
{
    private Tensor _weights = weights;

    public override string Name => name;

    public override int Rows => _weights.Shape[0];

    public override int Columns => _weights.Shape[1];

    public override long Bytes => _weights.Storage.ByteLength;

    public override Device Device => _weights.Device;

    public override string Description => $"{name} (expanded to float32: {Device} has no product of its own for it)";

    public override Tensor Dequantize() => _weights.Clone();

    public override IEnumerable<Tensor> Buffers()
    {
        yield return _weights;
    }

    protected override void MoveTo(Device device, Func<Tensor, Device, Tensor> move) => _weights = move(_weights, device);

    public override Tensor MatMul(Tensor input) => input.MatMul(_weights);

    public override void Dispose() => _weights.Dispose();
}
