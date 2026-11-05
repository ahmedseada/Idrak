// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Runtime.InteropServices;
using Idrak.Abstraction.Generation;

namespace Idrak.Abstraction.Devices.Cpu;

// Weights kept in their file's format (plan 37, Q0) on the CPU. A tile of outputs at a time: the tile's stored rows widened
// by their element type (and scaled) into a pooled [columns, k] buffer, laid out as the [k, columns] panel the float32
// weights would be, and multiplied by CpuMatMul, whose every path computes each element as one FMA chain over k from zero:
// so each output is the float32 product's over the weights expanded at load, bit for bit, while only a tile is ever
// expanded. The tiles are independent (each its own columns of y), spread over the threads, each product on one thread;
// where the whole product splits over k (CpuMatMul.SplitsK on its m, n and k, as the float32 product decides) they run one
// after another, each tile's product split over k in the same chunks. The widening goes through a source of rows
// (StoredRows), the one place a tile's float values come from: stored bytes in the CPU's memory, or a checkpoint's file
// mapping read in place (plan 37, W3: MappedStorage), whose float32 rows are laid into the panel straight from the file's
// pages, with no copy of their own.
internal sealed partial class CpuBackend
{
    public override void StoredMatMulKernel(Storage x, Storage weight, Storage? scales, Storage? zeros, Storage y, int m, int n, int k, StoredScales layout)
    {
        var rows = StoredRows.Of(weight, scales, zeros, layout, n, k);
        float[] xv = D(x), yv = D(y);
        if (m == 0 || n == 0)
        {
            return;
        }

        if (k == 0)
        {
            yv.AsSpan(0, m * n).Clear();
            return;
        }

        int tile = StoredTile(n, k), tiles = (n + tile - 1) / tile;
        bool split = CpuMatMul.SplitsK(m, n, k);                            // decided on the whole product, never a tile's
        if (tiles == 1 || split)
        {
            var how = split ? TileProduct.SplitK : TileProduct.Together;
            for (int t = 0; t < tiles; t++)
            {
                StoredTileProduct(rows, xv, yv, m, n, k, t * tile, Math.Min(tile, n - t * tile), how);
            }

            return;
        }

        Parallel.For(0, tiles, ComputeResources.ParallelOptions, t => StoredTileProduct(rows, xv, yv, m, n, k, t * tile, Math.Min(tile, n - t * tile), TileProduct.OneThread));
    }

    // How a tile's product runs: on this thread alone (the tiles in parallel), on every thread as one product (a single
    // tile), or over chunks of k on every thread (the whole product splits).
    private enum TileProduct : byte
    {
        OneThread,
        Together,
        SplitK,
    }

    public override void ExpandStoredKernel(Storage weight, Storage? scales, Storage? zeros, Storage w, int n, int k, StoredScales layout)
    {
        var rows = StoredRows.Of(weight, scales, zeros, layout, n, k);
        float[] wv = D(w);
        if (n == 0 || k == 0)
        {
            return;
        }

        if (weight is MappedStorage)
        {
            MappedWeights.CountCopied((long)n * k * sizeof(float), "expanded to float32");   // a layout materialized (plan 37, W3)
        }

        int tile = StoredTile(n, k), tiles = (n + tile - 1) / tile;
        Parallel.For(0, tiles, ComputeResources.ParallelOptions, t =>
        {
            int j0 = t * tile, columns = Math.Min(tile, n - j0);
            float[]? decoded = null;
            try
            {
                var source = rows.Floats(j0, columns);
                if (source.IsEmpty)
                {
                    decoded = ArrayPool<float>.Shared.Rent(columns * k);
                    rows.Widen(j0, columns, decoded);
                    source = decoded.AsSpan(0, columns * k);
                }

                for (int j = 0; j < columns; j++)
                {
                    var row = source.Slice(j * k, k);
                    for (int p = 0; p < k; p++)
                    {
                        wv[p * n + j0 + j] = row[p];
                    }
                }
            }
            finally
            {
                if (decoded is not null)
                {
                    ArrayPool<float>.Shared.Return(decoded);
                }
            }
        });
    }

    // Columns [j0, j0 + columns) of y = x · W: the tile's rows widened, laid out as the [k, columns] panel, multiplied as
    // `how` says (TileProduct).
    private static void StoredTileProduct(StoredRows rows, float[] x, float[] y, int m, int n, int k, int j0, int columns, TileProduct how)
    {
        float[]? decoded = null;
        float[] panel = ArrayPool<float>.Shared.Rent(k * columns);
        float[] c = ArrayPool<float>.Shared.Rent(m * columns);
        try
        {
            // Float32 rows read in place (a mapped file's) go straight into the panel; any other are widened first.
            var source = rows.Floats(j0, columns);
            if (source.IsEmpty)
            {
                decoded = ArrayPool<float>.Shared.Rent(columns * k);
                rows.Widen(j0, columns, decoded);
                source = decoded.AsSpan(0, columns * k);
            }

            for (int j = 0; j < columns; j++)
            {
                var row = source.Slice(j * k, k);
                for (int p = 0; p < k; p++)
                {
                    panel[p * columns + j] = row[p];
                }
            }

            switch (how)
            {
                case TileProduct.SplitK:
                    CpuMatMul.SplitK(x, 0, panel, 0, c, 0, m, columns, k, false, false, 0f, ComputeResources.CpuProductKChunk);
                    break;
                case TileProduct.Together:
                    CpuMatMul.Unsplit(x, 0, panel, 0, c, 0, m, columns, k, false, false, 0f);
                    break;
                default:
                    CpuMatMul.Multiply(x, 0, panel, 0, c, 0, m, columns, k, false, false, 0f, long.MaxValue, OneThread);
                    break;
            }

            for (int i = 0; i < m; i++)
            {
                c.AsSpan(i * columns, columns).CopyTo(y.AsSpan(i * n + j0, columns));
            }
        }
        finally
        {
            if (decoded is not null)
            {
                ArrayPool<float>.Shared.Return(decoded);
            }

            ArrayPool<float>.Shared.Return(panel);
            ArrayPool<float>.Shared.Return(c);
        }
    }

    // The outputs a tile covers: its widened rows and panel within a core's L2 (measured, CpuInfo), at least a few vectors'
    // worth, and small enough that the tiles keep every thread busy. Only the scratch and the speed depend on it.
    private static int StoredTile(int n, int k)
    {
        long budget = Math.Max(1L << 18, CpuInfo.Current.L2);
        int fit = (int)Math.Clamp(budget / (8L * Math.Max(1, k)), 1, n);
        int threads = Math.Max(1, ComputeResources.ParallelOptions.MaxDegreeOfParallelism > 0 ? ComputeResources.ParallelOptions.MaxDegreeOfParallelism : Environment.ProcessorCount);
        int spread = (int)Math.Max(1, ((long)n + 2 * threads - 1) / (2 * threads));
        return Math.Min(n, Math.Max(Math.Min(fit, spread), Math.Min(n, 2 * System.Numerics.Vector<float>.Count)));
    }
}

/// <summary>
/// Where a stored weight's tiles get their float values (plan 37, Q0): rows [first, first + count) of the stored
/// [n, k] matrix widened to float32 into [count, k]. Stored bytes of a registered element type with their scales in the
/// CPU's memory are one source; a checkpoint's file mapping read in place (plan 37, W3) is another, plugged in here
/// without changing the tiled product.
/// </summary>
internal abstract class StoredRows(ElementType elementType, float[]? scaleValues, float[]? zeroValues, StoredScales scaleLayout, int rowValues)
{
    /// <summary>Rows <paramref name="first"/>.. of the stored matrix, <paramref name="count"/> of them, as [count, k] floats.</summary>
    public void Widen(int first, int count, Span<float> destination)
    {
        int k = rowValues;
        long rowBytes = elementType.ByteCount(k);
        var values = destination[..(count * k)];
        elementType.Decode(StoredBytes(first * rowBytes, checked((int)(count * rowBytes))), values);
        if (scaleValues is not { } scales)
        {
            return;
        }

        var zeros = zeroValues;
        for (int r = 0; r < count; r++)
        {
            int row = first + r;
            var span = values.Slice(r * k, k);
            for (int c = 0; c < span.Length; c++)
            {
                int i = scaleLayout.IndexOf(row, c);
                span[c] = zeros is null ? span[c] * scales[i] : (span[c] - zeros[i]) * scales[i];
            }
        }
    }

    /// <summary>
    /// Rows <paramref name="first"/>.. as [count, k] float32 values read where they are stored (float32 rows without
    /// scales in a file's mapping, in place), or empty when they must be widened (<see cref="Widen"/>): the same values.
    /// </summary>
    public ReadOnlySpan<float> Floats(int first, int count) =>
        InPlace && ReferenceEquals(elementType, ElementType.Float32) && scaleValues is null && BitConverter.IsLittleEndian
            ? MemoryMarshal.Cast<byte, float>(StoredBytes((long)first * rowValues * sizeof(float), checked(count * rowValues * sizeof(float))))
            : default;

    // `count` stored bytes from byte `start` of the matrix.
    private protected abstract ReadOnlySpan<byte> StoredBytes(long start, int count);

    // Whether the stored rows are read where they lie (a mapping) rather than from the CPU's own memory, which the widening
    // reads into the tile's buffer as before.
    private protected abstract bool InPlace { get; }

    // The rows of a storage of stored bytes: its element type (checked to take rows of k), its scales and zero points; a
    // mapped storage's read from the file's pages in place.
    public static StoredRows Of(Storage weight, Storage? scales, Storage? zeros, StoredScales layout, int n, int k)
    {
        var type = weight.ElementType;
        StoredWeight.CheckRows(type, k);
        if (weight.ElementCount < (long)n * k)
        {
            throw new ArgumentException($"A {type.Name} storage of {weight.ElementCount} values cannot hold [{n}, {k}].", nameof(weight));
        }

        float[]? s = scales is null ? null : CpuBackend.D(scales), z = zeros is null ? null : CpuBackend.D(zeros);
        return weight is MappedStorage mapped
            ? new Mapped(type, mapped.Mapping, s, z, layout, k)
            : new Bytes(type, CpuBackend.D(weight), weight.Length, s, z, layout, k);
    }

    // Stored bytes in the CPU's memory: each row widened by the type, then (value - zero) * scale where there are scales, as
    // the stores do.
    private sealed class Bytes(ElementType type, float[] words, int length, float[]? scales, float[]? zeros, StoredScales layout, int k)
        : StoredRows(type, scales, zeros, layout, k)
    {
        private protected override ReadOnlySpan<byte> StoredBytes(long start, int count) =>
            MemoryMarshal.AsBytes(words.AsSpan(0, length)).Slice(checked((int)start), count);

        private protected override bool InPlace => false;
    }

    // Stored bytes in a checkpoint's file mapping (plan 37, W3): read where the file's pages lie, widened as Bytes widens
    // them (the same values, so the same products bit for bit).
    private sealed class Mapped(ElementType type, FileMapping mapping, float[]? scales, float[]? zeros, StoredScales layout, int k)
        : StoredRows(type, scales, zeros, layout, k)
    {
        private protected override ReadOnlySpan<byte> StoredBytes(long start, int count) => mapping.Bytes(start, count);

        private protected override bool InPlace => true;
    }
}
