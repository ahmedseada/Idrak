// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Idrak.Abstraction.Devices.Cpu;

// The product routes on the CPU (ProductRoutes): the products of int8 and bfloat16 weights by their own
// kernels (ProductRoute.Name, today), by one generic product that widens any type a tile at a time (ProductRoute.Generic),
// or by the faster of the two, measured once per shape (ProductRoute.Measured). Weights kept as stored (q8_0 and the other
// block formats) already take the generic product here (StoredMatMul, CpuBackend.Stored.cs) under every route.
internal sealed partial class CpuBackend
{
    // Up to this many rows the tensor's int8 product takes the direct kernel; above it today's route expands the weights
    // whole and takes the float32 product (Tensor.MatMulInt8's Int8DirectRows).
    private const int Int8DirectRows = 8;

    public override void Int8MatMulKernel(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k)
    {
        int stride = (n + 3) / 4 * 4;                                      // bytes per weight row
        switch (ProductRoutes.Current)
        {
            case ProductRoute.Name:
                Int8Direct(x, q, scales, y, m, n, k);
                return;
            case ProductRoute.Generic:
                TypedInputsMajor(ElementType.Int8, x, q, scales, y, m, n, k, stride);
                return;
        }

        if (NamedLane("int8", m, n, k) ?? MeasureInt8(x, q, scales, y, m, n, k, stride))
        {
            Int8Named(x, q, scales, y, m, n, k);
        }
        else
        {
            TypedInputsMajor(ElementType.Int8, x, q, scales, y, m, n, k, stride);
        }
    }

    public override void BFloat16MatMulKernel(Storage x, Storage packed, Storage y, int m, int n, int k)
    {
        int stride = (n + 1) / 2 * 2;                                      // values per weight row
        switch (ProductRoutes.Current)
        {
            case ProductRoute.Name:
                BFloat16Direct(x, packed, y, m, n, k);
                return;
            case ProductRoute.Generic:
                TypedInputsMajor(ElementType.BFloat16, x, packed, null, y, m, n, k, stride);
                return;
        }

        // The measured route: only the kernels that read the bfloat16 values in registers (no float32 copy of the weights,
        // whole or a tile), at any rows: register-tiled or streaming, the faster at the shape.
        bool tiled = NamedLane("bfloat16 in registers", m, n, k) ?? MeasureBFloat16(x, packed, y, m, n, k);
        BFloat16InRegisters(!tiled, x, packed, y, m, n, k);
    }

    // Today's route of an int8 product for any row count, as the tensor's product runs it on this device: few rows by the
    // direct kernel, more by the weights expanded whole and the float32 product.
    private void Int8Named(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k)
    {
        if (m <= Int8DirectRows)
        {
            Int8Direct(x, q, scales, y, m, n, k);
            return;
        }

        var w = Allocate(k * n, zeroed: false);
        try
        {
            Int8Dequantize(q, scales, w, k, n);
            MatMul(x, w, y, m, n, k, false, false, 0f);
        }
        finally
        {
            w.Release();
        }
    }

    // ------------------------------------------------------------------ the generic product

    /// <summary>
    /// y [m, n] = x [m, k] · W for W stored inputs-major as <paramref name="type"/>: k rows of <paramref name="stride"/>
    /// values (n ≤ stride), value (r, c) at element r · stride + c, widened by the type (its <see cref="ElementType.ToFloat32"/>)
    /// and times <paramref name="columnScales"/>[c] when given. A tile of output columns at a time is widened into the
    /// [k, columns] panel the float32 weights would be and multiplied by CpuMatMul, whose every path computes each element as
    /// one FMA chain over k from zero: so each output is the float32 product's over the widened values, bit for bit, while
    /// only a tile is ever widened. Nothing here names a type.
    /// </summary>
    internal void TypedInputsMajor(ElementType type, Storage x, Storage w, Storage? columnScales, Storage y, int m, int n, int k, int stride)
    {
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

        var weights = new TypedRows(type, D(w), w.Length, columnScales is null ? null : D(columnScales), stride);
        int tile = TypedTile(type, n, k), tiles = (n + tile - 1) / tile;
        bool split = CpuMatMul.SplitsK(m, n, k);                            // decided on the whole product, never a tile's
        if (tiles == 1 || split)
        {
            var how = split ? TileProduct.SplitK : TileProduct.Together;
            for (int t = 0; t < tiles; t++)
            {
                TypedTileProduct(weights, xv, yv, m, n, k, t * tile, Math.Min(tile, n - t * tile), how);
            }

            return;
        }

        Parallel.For(0, tiles, ComputeResources.ParallelOptions, t => TypedTileProduct(weights, xv, yv, m, n, k, t * tile, Math.Min(tile, n - t * tile), TileProduct.OneThread));
    }

    // The stored words of an inputs-major weight, its type, column scales and row stride.
    private sealed record TypedRows(ElementType Type, float[] Words, int Length, float[]? Scales, int Stride);

    // The outputs a tile covers: StoredTile's (a core's L2 for the panel, the threads kept busy), rounded so a tile starts
    // on a byte (a value of fewer than 8 bits).
    private static int TypedTile(ElementType type, int n, int k)
    {
        int tile = StoredTile(n, k);
        int perByte = type.BlockBits >= 8 ? 1 : 8 / type.BlockBits;
        return Math.Min(n, Math.Max(perByte, tile / perByte * perByte));
    }

    // Columns [j0, j0 + columns) of y = x · W: each of the k rows' values in the tile widened into the [k, columns] panel and
    // times its column's scale, then multiplied as `how` says (TileProduct).
    private static void TypedTileProduct(TypedRows weights, float[] x, float[] y, int m, int n, int k, int j0, int columns, TileProduct how)
    {
        float[] panel = ArrayPool<float>.Shared.Rent(k * columns);
        float[] c = ArrayPool<float>.Shared.Rent(m * columns);
        try
        {
            var type = weights.Type;
            var bytes = MemoryMarshal.AsBytes(weights.Words.AsSpan(0, weights.Length));
            long first = type.ByteCount(j0), taken = type.ByteCount(columns), rowBytes = type.ByteCount(weights.Stride);
            for (int r = 0; r < k; r++)
            {
                var row = panel.AsSpan(r * columns, columns);
                type.Decode(bytes.Slice(checked((int)(r * rowBytes + first)), (int)taken), row);
                if (weights.Scales is { } scales)
                {
                    Scale(row, scales.AsSpan(j0, columns));
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
            ArrayPool<float>.Shared.Return(panel);
            ArrayPool<float>.Shared.Return(c);
        }
    }

    // row[c] *= scales[c]: one rounding each, as the scalar product (vectors, then a scalar tail).
    private static void Scale(Span<float> row, ReadOnlySpan<float> scales)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            var values = MemoryMarshal.Cast<float, Vector<float>>(row);
            var by = MemoryMarshal.Cast<float, Vector<float>>(scales);
            for (int v = 0; v < values.Length; v++)
            {
                values[v] *= by[v];
            }

            i = values.Length * Vector<float>.Count;
        }

        for (; i < row.Length; i++)
        {
            row[i] *= scales[i];
        }
    }

    // ------------------------------------------------------------------ the measured choice

    // The lane measured for a product's shape: true for the named kernel, false for the generic product. The rows by class
    // (exact up to the few-row limit, then powers of two), so prompts of every length measure a few classes.
    private static readonly ConcurrentDictionary<(string Type, int Rows, int N, int K), bool> s_lanes = new();

    private static (string Type, int Rows, int N, int K) LaneKey(string type, int m, int n, int k) =>
        (type, m <= Int8DirectRows ? m : (int)BitOperations.RoundUpToPowerOf2((uint)m), n, k);

    // The lane measured for the shape, or null when it has not been measured (the caller then measures, apart, so the
    // per-token path makes no closure).
    private static bool? NamedLane(string type, int m, int n, int k) => s_lanes.TryGetValue(LaneKey(type, m, n, k), out bool named) ? named : null;

    [MethodImpl(MethodImplOptions.NoInlining)]
    private bool MeasureInt8(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k, int stride) =>
        MeasureLane(LaneKey("int8", m, n, k), m, n, k,
            () => Int8Named(x, q, scales, y, m, n, k),
            () => TypedInputsMajor(ElementType.Int8, x, q, scales, y, m, n, k, stride));

    // The bfloat16 kernels reading the weights in registers: register-tiled (true) or streaming, measured.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private bool MeasureBFloat16(Storage x, Storage packed, Storage y, int m, int n, int k) =>
        MeasureLane(LaneKey("bfloat16 in registers", m, n, k), m, n, k,
            () => BFloat16InRegisters(false, x, packed, y, m, n, k),
            () => BFloat16InRegisters(true, x, packed, y, m, n, k), "tiled", "streaming");

    // Times both lanes on the real operands (each writes all of y, which the caller writes again): one run of each first
    // (compiling and warming them), then rounds of one run each in turn, each lane's median; the second (`generic`) is taken
    // only when clearly faster (3%), the first (`named`) otherwise and with IDRAK_AUTOTUNE=0. Kept for the process.
    private static bool MeasureLane((string Type, int Rows, int N, int K) key, int m, int n, int k, Action named, Action generic,
        string first = "named", string second = "generic")
    {
        bool keepNamed = true;
        if (CpuTuning.Autotune)
        {
            named();
            generic();
            const int Rounds = 7;
            Span<double> a = stackalloc double[Rounds], b = stackalloc double[Rounds];
            for (int r = 0; r < Rounds; r++)
            {
                long t0 = Stopwatch.GetTimestamp();
                named();
                long t1 = Stopwatch.GetTimestamp();
                generic();
                long t2 = Stopwatch.GetTimestamp();
                (a[r], b[r]) = (t1 - t0, t2 - t1);
            }

            a.Sort();
            b.Sort();
            keepNamed = !(b[Rounds / 2] < 0.97 * a[Rounds / 2]);
            ProductRoutes.Chose("cpu", string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{key.Type} {m}x{k}->{n}"),
                string.Create(System.Globalization.CultureInfo.InvariantCulture,
                    $"{(keepNamed ? first : second)} ({first} {a[Rounds / 2] * 1e3 / Stopwatch.Frequency:0.###} ms, {second} {b[Rounds / 2] * 1e3 / Stopwatch.Frequency:0.###} ms)"));
        }

        s_lanes[key] = keepNamed;
        return keepNamed;
    }
}
