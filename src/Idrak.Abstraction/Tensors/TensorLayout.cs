// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction;

/// <summary>
/// Where a tensor's elements sit in its storage (plan 37, V0): its shape, the step in elements between neighbours along
/// each dimension (its strides, 0 along a broadcast dimension), and the element its first one is at (its offset). A tensor
/// made by an operation is contiguous (row-major from element 0); a view (<see cref="Tensor.Permute"/>,
/// <see cref="Tensor.Narrow"/>, <see cref="Tensor.Select"/>, <see cref="Tensor.Expand"/>) reads another tensor's storage
/// through its own layout. Built once per tensor; read as spans.
/// </summary>
public sealed class TensorLayout
{
    private readonly int[] _shape;
    private readonly long[] _strides;

    /// <summary>A layout of <paramref name="shape"/> with <paramref name="strides"/> (in elements) from element <paramref name="offset"/>.</summary>
    /// <exception cref="ArgumentException">The strides are not one per dimension, or a size or the offset is negative.</exception>
    public TensorLayout(ReadOnlySpan<int> shape, ReadOnlySpan<long> strides, long offset)
    {
        if (shape.Length != strides.Length || offset < 0)
        {
            throw new ArgumentException($"A layout takes one stride per dimension ({shape.Length}) and an offset of 0 or more; got {strides.Length} strides and offset {offset}.");
        }

        _shape = shape.ToArray();
        _strides = strides.ToArray();
        Offset = offset;
        long elements = 1;
        foreach (int d in _shape)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(d, nameof(shape));
            elements *= d;
        }

        Elements = elements;
        IsContiguous = Dense(_shape, _strides);
    }

    /// <summary>The row-major layout of <paramref name="shape"/> from element 0.</summary>
    public static TensorLayout Contiguous(ReadOnlySpan<int> shape) => new(shape, RowMajor(shape), 0);

    /// <summary>The row-major strides of <paramref name="shape"/> (the last dimension's 1).</summary>
    public static long[] RowMajor(ReadOnlySpan<int> shape)
    {
        var strides = new long[shape.Length];
        long step = 1;
        for (int d = shape.Length - 1; d >= 0; d--)
        {
            strides[d] = step;
            step *= Math.Max(1, shape[d]);
        }

        return strides;
    }

    /// <summary>The length of each dimension.</summary>
    public ReadOnlySpan<int> Shape => _shape;

    /// <summary>The step in elements between neighbours along each dimension (0 along a broadcast one).</summary>
    public ReadOnlySpan<long> Strides => _strides;

    /// <summary>The element of the storage the first element is at.</summary>
    public long Offset { get; }

    /// <summary>The number of elements.</summary>
    public long Elements { get; }

    /// <summary>Whether the elements sit row-major one after another (from <see cref="Offset"/>), with no gap or repeat.</summary>
    public bool IsContiguous { get; }

    /// <summary>Whether the layout is the row-major one from element 0: what every operation reads.</summary>
    public bool IsContiguousFromStart => IsContiguous && Offset == 0;

    /// <summary>The storage element of the element at <paramref name="index"/> (one index per dimension).</summary>
    public long ElementAt(ReadOnlySpan<int> index)
    {
        long at = Offset;
        for (int d = 0; d < index.Length; d++)
        {
            at += index[d] * _strides[d];
        }

        return at;
    }

    // Whether strides walk the shape row-major with no gap (size-1 dimensions may have any stride).
    private static bool Dense(int[] shape, long[] strides)
    {
        long expected = 1;
        for (int d = shape.Length - 1; d >= 0; d--)
        {
            if (shape[d] == 1)
            {
                continue;
            }

            if (strides[d] != expected)
            {
                return false;
            }

            expected *= shape[d];
        }

        return true;
    }

    /// <summary>The layout with dimension k taken from dimension <paramref name="dims"/>[k] (no element moves).</summary>
    public TensorLayout Permute(ReadOnlySpan<int> dims)
    {
        var shape = new int[dims.Length];
        var strides = new long[dims.Length];
        for (int k = 0; k < dims.Length; k++)
        {
            shape[k] = _shape[dims[k]];
            strides[k] = _strides[dims[k]];
        }

        return new TensorLayout(shape, strides, Offset);
    }

    /// <summary>The layout of positions [<paramref name="start"/>, start + <paramref name="length"/>) along <paramref name="dim"/>.</summary>
    public TensorLayout Narrow(int dim, int start, int length)
    {
        int[] shape = [.. _shape];
        shape[dim] = length;
        return new TensorLayout(shape, _strides, Offset + start * _strides[dim]);
    }

    /// <summary>The layout of position <paramref name="index"/> along <paramref name="dim"/>, that dimension dropped.</summary>
    public TensorLayout Select(int dim, int index)
    {
        int[] shape = [.. _shape[..dim], .. _shape[(dim + 1)..]];
        long[] strides = [.. _strides[..dim], .. _strides[(dim + 1)..]];
        return new TensorLayout(shape, strides, Offset + index * _strides[dim]);
    }

    /// <summary>
    /// The layout broadcast to <paramref name="shape"/> (as many or more dimensions, new ones leading): a dimension of size 1
    /// grows with stride 0, every element read again; others keep their size.
    /// </summary>
    public TensorLayout Expand(ReadOnlySpan<int> shape)
    {
        int extra = shape.Length - _shape.Length;
        var strides = new long[shape.Length];
        for (int d = 0; d < shape.Length; d++)
        {
            int own = d - extra;
            strides[d] = own < 0 || _shape[own] == 1 && shape[d] != 1 ? 0 : _strides[own];
        }

        return new TensorLayout(shape, strides, Offset);
    }

    /// <summary>
    /// The layout of the same elements in the same row-major order as <paramref name="shape"/> (as many elements), when
    /// strides express it: dimensions merge where they sit one after another and split anywhere. Null when they do not (a
    /// reshape merging a permuted pair, such as [n, h, t, d] permuted to [n, t, h, d] read as [n·t, h·d]): such a reshape of
    /// a view keeps this layout and reads it in its order (<see cref="Tensor.Reshape"/>).
    /// </summary>
    /// <exception cref="ArgumentException">The shape holds another number of elements.</exception>
    public TensorLayout? Reshape(ReadOnlySpan<int> shape)
    {
        long elements = 1;
        foreach (int d in shape)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(d, nameof(shape));
            elements *= d;
        }

        if (elements != Elements)
        {
            throw new ArgumentException($"A reshape keeps the {Elements} elements of {this}; [{string.Join(", ", shape.ToArray())}] holds {elements}.", nameof(shape));
        }

        if (Elements == 0 || _shape.Length == 0)
        {
            return new TensorLayout(shape, RowMajor(shape), Offset);
        }

        // From the last dimension: each run of old dimensions that sit one after another (a chunk) takes the new dimensions
        // that hold as many elements, their strides stepping from the chunk's innermost one.
        var strides = new long[shape.Length];
        int target = shape.Length - 1;
        long chunkStride = _strides[^1], chunkElements = 1, taken = 1;
        for (int d = _shape.Length - 1; d >= 0; d--)
        {
            chunkElements *= _shape[d];
            if (d == 0 || _shape[d - 1] != 1 && _strides[d - 1] != chunkElements * chunkStride)
            {
                while (target >= 0 && (taken < chunkElements || shape[target] == 1))
                {
                    strides[target] = taken * chunkStride;
                    taken *= shape[target];
                    target--;
                }

                if (taken != chunkElements)
                {
                    return null;
                }

                if (d > 0)
                {
                    chunkStride = _strides[d - 1];
                    chunkElements = 1;
                    taken = 1;
                }
            }
        }

        return target == -1 ? new TensorLayout(shape, strides, Offset) : null;
    }

    /// <summary>
    /// Whether every run of <paramref name="length"/> elements in row-major order sits one after another (the last dimension
    /// a multiple of it, stride 1): the rows a kernel reads in place through <see cref="RowOffsets"/>.
    /// </summary>
    internal bool HasRows(int length) =>
        length > 0 && (_shape.Length == 0 ? length == 1 : _shape[^1] % length == 0 && (_strides[^1] == 1 || length == 1));

    /// <summary>The storage element each run of <paramref name="length"/> elements (see <see cref="HasRows"/>) starts at, in order.</summary>
    internal void RowOffsets(int length, Span<long> offsets)
    {
        if (_shape.Length == 0)
        {
            if (offsets.Length > 0)
            {
                offsets[0] = Offset;
            }

            return;
        }

        int outerRank = _shape.Length - 1, perRow = _shape[^1] / length;
        long step = length * _strides[^1];
        Span<int> index = outerRank <= 16 ? stackalloc int[outerRank] : new int[outerRank];
        index.Clear();
        int row = 0;
        while (row < offsets.Length)
        {
            long at = Offset;
            for (int d = 0; d < outerRank; d++)
            {
                at += index[d] * _strides[d];
            }

            for (int j = 0; j < perRow && row < offsets.Length; j++)
            {
                offsets[row++] = at + j * step;
            }

            for (int d = outerRank - 1; d >= 0; d--)
            {
                if (++index[d] < _shape[d])
                {
                    break;
                }

                index[d] = 0;
            }
        }
    }

    /// <inheritdoc />
    public override string ToString() =>
        $"[{string.Join(", ", _shape)}] strides [{string.Join(", ", _strides)}] offset {Offset}{(IsContiguous ? " (contiguous)" : "")}";
}

/// <summary>
/// The views' setting and counts (plan 37, V1): whether tensors on a device that reads strided storage
/// (<see cref="Devices.BackendCapabilities.StridedInputs"/>: the CPU) make views instead of copies, and how many views
/// were made, read in place, or copied once to a contiguous block because what read them did not take strides.
/// </summary>
public static class TensorViews
{
    private static long s_made, s_materialized, s_materializedBytes, s_scratch, s_scratchBytes;

    /// <summary>The variable that turns views off when "0", "false", "off" or "no": "IDRAK_VIEWS".</summary>
    public const string EnvironmentVariable = "IDRAK_VIEWS";

    /// <summary>
    /// Whether permutes, transposes, narrows, selects and expands return views sharing their input's storage where the
    /// device reads strides: on unless <see cref="EnvironmentVariable"/> turns it off (today's copies, to check against).
    /// Results are the same either way. While recorded activations are kept as bfloat16
    /// (<see cref="ActivationMemory.CompressToBFloat16"/>) these operations copy: a copy is packed to half its size, where a
    /// view would keep its source's float values.
    /// </summary>
    public static bool Enabled { get; set; } =
        Environment.GetEnvironmentVariable(EnvironmentVariable)?.Trim().ToLowerInvariant() is not ("0" or "false" or "off" or "no");

    /// <summary>Views made so far.</summary>
    public static long Made => Interlocked.Read(ref s_made);

    /// <summary>Views copied to a contiguous block of their own, once, because something read their storage directly.</summary>
    public static long Materialized => Interlocked.Read(ref s_materialized);

    /// <summary>The bytes those copies took.</summary>
    public static long MaterializedBytes => Interlocked.Read(ref s_materializedBytes);

    /// <summary>Views an operation read through a short-lived contiguous copy (given back after the operation).</summary>
    public static long Scratch => Interlocked.Read(ref s_scratch);

    /// <summary>The bytes of those short-lived copies.</summary>
    public static long ScratchBytes => Interlocked.Read(ref s_scratchBytes);

    internal static void CountMade() => Interlocked.Increment(ref s_made);

    internal static void CountMaterialized(long bytes)
    {
        Interlocked.Increment(ref s_materialized);
        Interlocked.Add(ref s_materializedBytes, bytes);
    }

    internal static void CountScratch(long bytes)
    {
        Interlocked.Increment(ref s_scratch);
        Interlocked.Add(ref s_scratchBytes, bytes);
    }
}
