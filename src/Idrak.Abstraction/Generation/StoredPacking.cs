// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Collections.Concurrent;
using Idrak.Abstraction.Devices;

namespace Idrak.Abstraction.Generation;

/// <summary>
/// A packed format made from a registered element type that stores and widens float32 values (its
/// <see cref="ElementType.FromFloat32"/> and <see cref="ElementType.ToFloat32"/>; ggml's q8_0 among them), under the type's
/// own name: float values are stored in the type, a row of k inputs per output as a file holds them, and kept as stored
/// (<see cref="StoredWeight"/>) where the device multiplies the type (<see cref="WeightStorage.Multiplies"/>). Elsewhere
/// they are expanded to float32 once, as a load of a file in the type expands them for that device, so a run there gives
/// the stored values' products. Nothing here names a type: a block type or a float type packs its values as they encode; a
/// signed integer type of whole bytes (int16, int32, int64, ...) packs codes of its own bits with one float32 scale an
/// output row (<see cref="EncodeScaledRow"/>), as the routes' int8 weights do.
/// </summary>
internal sealed class StoredPackedFormat : IPackedFormat, IRowPackedFormat
{
    private static readonly ConcurrentDictionary<ElementType, StoredPackedFormat> Made = new();

    private readonly ElementType _type;

    private StoredPackedFormat(ElementType type) => _type = type;

    /// <summary>The format of the element type named <paramref name="name"/>, or null when there is none that stores and widens.</summary>
    public static StoredPackedFormat? Of(ReadOnlySpan<char> name) =>
        ElementTypes.Find(name) is { FromFloat32: not null, ToFloat32: not null } type && (type.IsBlock || type.Float is not null || RowScaled(type))
            ? Made.GetOrAdd(type, static t => new StoredPackedFormat(t)) : null;

    /// <summary>
    /// Whether <paramref name="type"/>'s weights are codes of its own bits with one scale an output row: a signed integer
    /// type of whole bytes (its codes two's complement, little-endian).
    /// </summary>
    internal static bool RowScaled(ElementType type) => type is { IsInteger: true, IsSigned: true, BlockSize: 1 } && type.BlockBits % 8 == 0 && type.BlockBits is >= 8 and <= 64;

    public string Name => _type.Name;

    public string Summary => RowScaled(_type)
        ? $"{_type.Describe()} codes clamp(round(w / s)) with one float32 scale s = max |w| / {_type.MaxValue} an output row, kept as stored where the device multiplies it"
        : $"{_type.Describe()} stored from float32 values, kept as stored where the device multiplies it";

    public IReadOnlySet<string> Keys => OptionKeys.None;

    public override string ToString() => _type.Name;

    public unsafe IPackedWeight Create(PackRequest request)
    {
        int k = request.Rows, n = request.Columns;
        StoredWeight.CheckRows(_type, k);
        var device = request.Device;
        var type = _type;
        fixed (float* pinned = request.Values)
        {
            nint values = (nint)pinned;
            if (WeightStorage.Multiplies(device.Backend, _type))
            {
                if (RowScaled(type))
                {
                    var scales = new float[n];
                    return StoredWeight.FromFillAt(type, k, n, RowScaling(scales, k), device, (stored, _) => EncodeRows((first, count, destination) =>
                    {
                        Columns((float*)values, k, n, first, count, destination);
                        return true;
                    }, k, n, stored, type.ByteCount(k), (j, row, into) => scales[j] = EncodeScaledRow(type, row, into)))!;
                }

                return StoredWeight.FromFillAt(_type, k, n, null, device, (stored, _) =>
                {
                    StoreColumns((float*)values, k, n, stored);
                    return true;
                })!;
            }

            // As a load expands a file's weights for a device that has no product of the type: the stored values in float32,
            // a block of output rows encoded and expanded at a time (an 8-byte type's head, 151,936 × 2,048 values, takes
            // 2.5 GB stored: past what one array of bytes holds).
            long rowBytes = _type.ByteCount(k);
            int per = (int)Math.Clamp((32L << 20) / Math.Max(1, rowBytes), 1, Math.Max(1, n));
            byte[] chunk = ArrayPool<byte>.Shared.Rent((int)Math.Max(1, per * rowBytes));
            var rowScales = new float[per];
            var expanded = new float[(long)k * n];
            var row = new float[k];
            try
            {
                fixed (byte* into = chunk)
                {
                    for (int first = 0; first < n; first += per)
                    {
                        int count = Math.Min(per, n - first), at = first;
                        WeightRows block = (f, c, destination) =>
                        {
                            Columns((float*)values, k, n, at + f, c, destination);
                            return true;
                        };

                        if (RowScaled(type))
                        {
                            EncodeRows(block, k, count, into, rowBytes, (j, source, stored) => rowScales[j] = EncodeScaledRow(type, source, stored));
                        }
                        else
                        {
                            EncodeRows(block, k, count, into, rowBytes, (_, source, stored) => type.Encode(source, stored));
                            rowScales.AsSpan().Fill(1f);
                        }

                        for (int j = 0; j < count; j++)
                        {
                            _type.Decode(chunk.AsSpan((int)(j * rowBytes), (int)rowBytes), row);
                            for (int r = 0; r < k; r++)
                            {
                                expanded[(long)r * n + first + j] = row[r] * rowScales[j];
                            }
                        }
                    }
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(chunk);
            }

            return new ExpandedWeight(_type.Name, Tensor.Persistent(expanded, [k, n], device));
        }
    }

    public unsafe IPackedWeight? CreateFromRows(WeightRows rows, int inputs, int outputs, Device device)
    {
        int k = inputs, n = outputs;
        StoredWeight.CheckRows(_type, k);
        if (!WeightStorage.Multiplies(device.Backend, _type))
        {
            return null;                                                         // expanded to float32: packed from the values
        }

        long rowBytes = _type.ByteCount(k);
        var type = _type;
        if (RowScaled(type))
        {
            var scales = new float[n];
            return StoredWeight.FromFillAt(type, k, n, RowScaling(scales, k), device,
                (stored, _) => EncodeRows(rows, k, n, stored, rowBytes, (j, row, into) => scales[j] = EncodeScaledRow(type, row, into)));
        }

        return StoredWeight.FromFillAt(type, k, n, null, device, (stored, _) => EncodeRows(rows, k, n, stored, rowBytes, (_, row, into) => type.Encode(row, into)));
    }

    // One scale an output row of k values (filled as the rows are encoded).
    private static Formats.StoredScaling RowScaling(float[] scales, int k) => new(scales, null, new StoredScales(1, k, 1));

    /// <summary>
    /// One output row of a signed integer type of whole bytes (<see cref="RowScaled"/>) into <paramref name="into"/>: the
    /// row's scale s = max |w| / the type's largest value (1 for a row of zeros), as float32, and each code
    /// clamp(round(w / s)) to ± the largest value a float64 holds within the type's range (computed in float64, ties to
    /// even), little-endian. Returns s: each value is code · s.
    /// </summary>
    internal static float EncodeScaledRow(ElementType type, ReadOnlySpan<float> row, Span<byte> into)
    {
        float largest = 0f;
        foreach (float v in row)
        {
            largest = MathF.Max(largest, MathF.Abs(v));
        }

        double top = (double)type.MaxValue;
        if ((Int128)top > type.MaxValue)
        {
            top = Math.BitDecrement(top);                                  // int64: 2^63 - 1 rounds up to 2^63 in float64
        }

        float scale = largest > 0f ? (float)(largest / top) : 1f;
        if (scale == 0f)
        {
            scale = float.Epsilon;                                          // a row too small for float32's scale: its smallest
        }

        double inverse = 1.0 / scale;
        int bytes = type.BlockBits / 8;
        for (int i = 0; i < row.Length; i++)
        {
            double code = Math.Clamp(Math.Round(row[i] * inverse), -top, top);
            switch (bytes)
            {
                case 8:
                    System.Buffers.Binary.BinaryPrimitives.WriteInt64LittleEndian(into[(8 * i)..], (long)code);
                    break;
                case 4:
                    System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(into[(4 * i)..], (int)code);
                    break;
                case 2:
                    System.Buffers.Binary.BinaryPrimitives.WriteInt16LittleEndian(into[(2 * i)..], (short)code);
                    break;
                case 1:
                    into[i] = (byte)(sbyte)code;
                    break;
                default:
                    long whole = (long)code;
                    for (int b = 0; b < bytes; b++)
                    {
                        into[bytes * i + b] = (byte)(whole >> (8 * b));
                    }

                    break;
            }
        }

        return scale;
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
    internal static unsafe IPackedWeight? RoutedRows(string format, WeightRows rows, int k, int n, Device device)
    {
        if (format != "int8")
        {
            var bf16 = ElementType.BFloat16;
            return StoredWeight.FromFillAt(bf16, k, n, null, device, (stored, _) => EncodeRows(rows, k, n, stored, bf16.ByteCount(k), (_, row, into) => bf16.Encode(row, into)));
        }

        // int8: each row's scale max |w| / 127 (1 for a row of zeros), its codes clamp(round(w / s), −127, 127).
        var scales = new float[n];
        return StoredWeight.FromFillAt(ElementType.Int8, k, n, RowScaling(scales, k), device, (stored, _) => EncodeRows(rows, k, n, stored, k, (j, row, into) =>
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
    /// Each output row of <paramref name="rows"/> encoded into its <paramref name="rowBytes"/> from <paramref name="stored"/>
    /// (row j at j · rowBytes: a matrix past 2 GB too): a block of rows read at a time into one pooled buffer (about 8 MB of
    /// float32), the block's rows encoded in parallel. False when the first block cannot be read (nothing written).
    /// </summary>
    internal static unsafe bool EncodeRows(WeightRows rows, int k, int n, byte* stored, long rowBytes, RowEncoder encode)
    {
        int per = (int)Math.Clamp((1L << 21) / Math.Max(1, k), 1, Math.Max(1, n));
        float[] buffer = ArrayPool<float>.Shared.Rent(Math.Max(1, per * k));
        try
        {
            nint into = (nint)stored;
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
    private unsafe void StoreColumns(float* values, int k, int n, byte* stored)
    {
        const int Width = 32;
        long rowBytes = _type.ByteCount(k);
        var type = _type;
        nint into = (nint)stored, from = (nint)values;
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
