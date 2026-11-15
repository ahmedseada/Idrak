// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace Idrak.Abstraction.Devices.Cpu;

// The experiment's block route on the CPU (ProductRoute.Adad), ported from Adad's Cpu BlockProducts and MatMulBlocks and
// driven by the weights' description: x is quantized into q8_0's blocks (Adad's encoder: the block's largest |x| / 127
// stored as float16 and read back, each value rounded to the nearest code, ties to even, within ±127), the weights' codes
// widened to int8 as their layout says (BlockLayout: 8-bit codes or 4-bit codes in ggml's order, signed or biased, a
// 16-bit float scale in the block, or none and a scale a row), then y = Σ over blocks of dW·dX·(Σ of 32 integer products),
// each 32-term dot exact in s32 by the s8 dot instruction (AVX-VNNI-INT8, ARM's dot product). Up to 8 rows (Adad's
// GemvBlocks): a weight row at a time on the cores, its blocks dotted with every row of x. More (Adad's MatMulBlocks): a
// panel of the weights' rows unpacked once into int8 rows and a scale per 4 bytes, tasks of 4 rows of x against it, a
// 4 × 2 register tile of float vectors. One of the lanes the adad route measures (CpuBackend.Lanes.cs), only where the CPU
// has the s8 dot lane.
internal sealed partial class CpuBackend
{
    /// <summary>Whether the CPU has Adad's s8 dot lane: four signed bytes a 32-bit lane summed into s32.</summary>
    internal static bool S8DotLane => AvxVnniInt8.IsSupported && Vector<byte>.Count is 16 or 32 || Dp.IsSupported && Vector<byte>.Count == 16;

    // A block route operand's layout, from a description (the same rule as CUDA's PtxKernels.BlockLayout): bytes a block of
    // 32 values takes, its 16-bit float scale's byte (−1: none, the row's scale applies) and whether it is float16 (else
    // bfloat16), its codes' byte and bits (8, or 4 in ggml's order), their sign and the bias added before the scale.
    private sealed record BlockLayout(int Bytes, int ScaleOffset, bool HalfScale, int CodeOffset, int CodeBits, bool Signed, int Bias, IReadOnlyList<int>? Table = null)
    {
        // A table's entries as int8 lanes (each entry plus the bias within int8: Of checks).
        public Vector128<sbyte> Grid { get; } = Table is null ? default : Vector128.Create([.. Table.Select(e => (sbyte)e)]);

        public bool HasScale => ScaleOffset >= 0;

        public bool Direct => CodeBits == 8 && Signed && Bias == 0;

        public static BlockLayout? Of(ElementType type, bool rowScales)
        {
            if (type.Scaled is { Values: 32 } b)
            {
                bool half = b.Scale is { Exponent: 5, Mantissa: 10 }, scale = half || b.Scale is { Exponent: 8, Mantissa: 7 };
                bool codes = b.CodeBits == 8 && b.CodeTable is null || b.CodeBits == 4 && b.NibblesSplit && b.CodeTable is null or { Count: 16 };
                bool fits = b.CodeTable is { } table ? table.All(e => e + b.CodeBias is >= -128 and <= 127) : WithinInt8(b.CodeBits, b.SignedCodes, b.CodeBias);
                return !rowScales && scale && codes && fits
                    ? new BlockLayout(b.Bytes, b.ScaleOffset, half, b.CodeOffset, b.CodeBits, b.SignedCodes, b.CodeBias, b.CodeTable)
                    : null;
            }

            return rowScales && !type.IsBlock && type.Float is null && type.IsInteger && type.BlockBits == 8 && WithinInt8(8, type.IsSigned, 0)
                ? new BlockLayout(32, -1, true, 0, 8, type.IsSigned, 0)
                : null;
        }

        private static bool WithinInt8(int bits, bool signed, int bias) =>
            (signed ? -(1 << (bits - 1)) : 0) + bias >= -128 && (signed ? (1 << (bits - 1)) - 1 : (1 << bits) - 1) + bias <= 127;
    }

    // Whether the block route can take this product: the adad route, a block layout from the weights' description, k whole
    // blocks, the CPU's s8 dot lane.
    private static bool BlockLane(ElementType type, float[]? rowScales, int k) =>
        ProductRoutes.Current == ProductRoute.Adad && k % 32 == 0 && S8DotLane && BlockLayout.Of(type, rowScales is not null) is not null;

    /// <summary>Tests: the block route run directly (false when the weights' description gives no block layout or the CPU has no s8 dot lane).</summary>
    internal bool StoredBlocks(ElementType type, Storage x, Storage weight, Storage? scales, Storage y, int m, int n, int k)
    {
        bool super = SuperFits(type, scales is null ? null : D(scales), k), grid = GridFits(type, scales is null ? null : D(scales), k);
        if (!super && !grid && BlockLayout.Of(type, scales is not null) is null || !S8DotLane || k % 32 != 0 || weight is MappedStorage)
        {
            return false;
        }

        var lane = new StoredLane(type, type.Scaled, D(weight), weight.Length, scales is null ? null : D(scales), k, type.ByteCount(k));
        if (grid)
        {
            RunGridBlocks(lane, D(x), D(y), m, n, k);
        }
        else if (super)
        {
            RunSuperBlocks(lane, D(x), D(y), m, n, k);
        }
        else
        {
            RunBlocks(lane, D(x), D(y), m, n, k);
        }

        return true;
    }

    // ------------------------------------------------------------------ the grid codebooks' block route

    // A grid-codebook description (ElementType.Grid: IQ2_XXS, IQ3_XXS) the block route takes: sub-blocks of 32 values (each
    // one of x's blocks) and codes within int8 (GridBlock.HasCodes).
    private static bool GridFits(ElementType type, float[]? rowScales, int k) =>
        rowScales is null && type.Grid is { SubValues: 32 } g && k % g.Values == 0 && g.HasCodes;

    private static bool GridBlockLane(ElementType type, float[]? rowScales, int k) =>
        ProductRoutes.Current == ProductRoute.Adad && S8DotLane && GridFits(type, rowScales, k);

    // x as q8_0 (Adad's encoder), each weight block's codes and sub-block scales from its description (GridBlock.DecodeCodes),
    // then per 32 values dx · scale_j · Σ(qx · code): few rows a gemv over the weight rows, more the block product's tasks.
    private static void RunGridBlocks(StoredLane lane, float[] x, float[] y, int m, int n, int k)
    {
        var g = lane.Type.Grid!;
        int blocks = k / 32;
        var xq = ArrayPool<sbyte>.Shared.Rent(m * k);
        var xs = ArrayPool<float>.Shared.Rent(m * blocks);
        try
        {
            if (m == 1)
            {
                QuantizeRow(x.AsSpan(0, k), xq.AsSpan(0, k), xs.AsSpan(0, blocks));
            }
            else
            {
                Parallel.For(0, m, ComputeResources.ParallelOptions, i => QuantizeRow(x.AsSpan(i * k, k), xq.AsSpan(i * k, k), xs.AsSpan(i * blocks, blocks)));
            }

            if (m <= Int8DirectRows)
            {
                int threads = Math.Max(1, ComputeResources.MaxCpuThreads), chunk = Math.Max(1, n / (threads * 4));
                Parallel.For(0, (n + chunk - 1) / chunk, ComputeResources.ParallelOptions,
                    c => GemvGrid(g, lane, xq, xs, y, m, n, k, c * chunk, Math.Min(chunk, n - c * chunk)));
            }
            else
            {
                MatMulUnpacked((row, q, s) => UnpackGridRow(g, lane, row, k, q, s), xq, xs, y, m, n, k);
            }
        }
        finally
        {
            ArrayPool<sbyte>.Shared.Return(xq);
            ArrayPool<float>.Shared.Return(xs);
        }
    }

    // The grid gemv over weight rows [j0, j0 + count): each block's codes once, each sub-block dotted with every row of x.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemvGrid(GridBlock g, StoredLane lane, sbyte[] xq, float[] xs, float[] y, int m, int n, int k, int j0, int count)
    {
        var bytes = MemoryMarshal.AsBytes(lane.Words.AsSpan(0, lane.Length));
        int per = k / g.Values, subs = g.Values / 32, blocks = k / 32;
        Span<float> acc = stackalloc float[Int8DirectRows];
        Span<sbyte> codes = stackalloc sbyte[g.Values];
        Span<float> scales = stackalloc float[subs];
        for (int r = j0; r < j0 + count; r++)
        {
            acc[..m].Clear();
            var row = bytes.Slice((int)(r * lane.RowBytes), (int)lane.RowBytes);
            for (int b = 0; b < per; b++)
            {
                g.DecodeCodes(row.Slice(b * g.Bytes, g.Bytes), codes, scales);
                for (int j = 0; j < subs; j++)
                {
                    int xb = b * subs + j;
                    var cj = codes.Slice(j * 32, 32);
                    for (int i = 0; i < m; i++)
                    {
                        int dot = Dot32(cj, xq.AsSpan(i * k + xb * 32, 32));
                        acc[i] += scales[j] * xs[i * blocks + xb] * dot;
                    }
                }
            }

            for (int i = 0; i < m; i++)
            {
                y[i * n + r] = acc[i];
            }
        }
    }

    // A grid weight row's codes as int8 and its sub-blocks' scales a 4-byte group (what BlocksTask reads).
    private static void UnpackGridRow(GridBlock g, StoredLane lane, int row, int k, Span<sbyte> q, Span<float> s)
    {
        var bytes = MemoryMarshal.AsBytes(lane.Words.AsSpan(0, lane.Length)).Slice((int)(row * lane.RowBytes), (int)lane.RowBytes);
        int subs = g.Values / 32;
        Span<float> scales = stackalloc float[subs];
        for (int b = 0; b < k / g.Values; b++)
        {
            g.DecodeCodes(bytes.Slice(b * g.Bytes, g.Bytes), q.Slice(b * g.Values, g.Values), scales);
            for (int j = 0; j < subs; j++)
            {
                s.Slice((b * subs + j) * 8, 8).Fill(scales[j]);
            }
        }
    }

    // ------------------------------------------------------------------ the k-quants' block route

    // Each super-block description: whether the block route takes it (sub-blocks of 16 or 32 values, so each lies in one of
    // x's blocks; codes plus their bias within int8).
    private static readonly ConcurrentDictionary<SuperBlock, bool> SuperRoutes = new();

    private static bool SuperFits(ElementType type, float[]? rowScales, int k) =>
        rowScales is null && type.Super is { } s && k % s.Values == 0 && SuperRoutes.GetOrAdd(s, static d =>
            d.SubValues is 16 or 32 && d.Values % 32 == 0 && d.CodeBias >= -128 && (1 << d.Planes.Sum(p => p.Bits)) - 1 + d.CodeBias <= 127);

    // Whether the block route can take this super-block product: the adad route and the CPU's s8 dot lane.
    private static bool SuperBlockLane(ElementType type, float[]? rowScales, int k) =>
        ProductRoutes.Current == ProductRoute.Adad && S8DotLane && SuperFits(type, rowScales, k);

    // y [m, n] = x [m, k] · Wᵀ for a super-block W by the block route: x in q8_0's blocks; per sub-block j of a weight row,
    // dx · (scale_j · Σ q_x · code − min_j · Σ q_x), each Σ an exact integer (the s8 dot lane), the rest in float32.
    private static void RunSuperBlocks(StoredLane lane, float[] x, float[] y, int m, int n, int k)
    {
        var s = lane.Type.Super!;
        int blocks = k / 32, subs = k / s.SubValues;
        var xq = ArrayPool<sbyte>.Shared.Rent(m * k);
        var xs = ArrayPool<float>.Shared.Rent(m * blocks);
        var xsum = ArrayPool<int>.Shared.Rent(m * subs);
        try
        {
            Parallel.For(0, m, ComputeResources.ParallelOptions, i =>
            {
                QuantizeRow(x.AsSpan(i * k, k), xq.AsSpan(i * k, k), xs.AsSpan(i * blocks, blocks));
                for (int j = 0; j < subs; j++)
                {
                    int sum = 0;
                    foreach (sbyte q in xq.AsSpan(i * k + j * s.SubValues, s.SubValues))
                    {
                        sum += q;
                    }

                    xsum[i * subs + j] = sum;
                }
            });

            if (m <= Int8DirectRows)
            {
                int threads = Math.Max(1, ComputeResources.MaxCpuThreads), chunk = Math.Max(1, n / (threads * 4));
                Parallel.For(0, (n + chunk - 1) / chunk, ComputeResources.ParallelOptions,
                    c => GemvSuper(s, lane, xq, xs, xsum, y, m, n, k, c * chunk, Math.Min(chunk, n - c * chunk)));
            }
            else
            {
                MatMulSuper(s, lane, xq, xs, xsum, y, m, n, k);
            }
        }
        finally
        {
            ArrayPool<sbyte>.Shared.Return(xq);
            ArrayPool<float>.Shared.Return(xs);
            ArrayPool<int>.Shared.Return(xsum);
        }
    }

    // A description's 32-value chunks: per plane, each chunk's first byte and shift (its 32 fields lie in 32 consecutive
    // bytes at one shift), the plane's mask and its place in the code; null where the planes do not lie so.
    private sealed record Chunks(int[][] Offsets, int[][] Shifts, byte[] Masks, int[] Low);

    private static readonly ConcurrentDictionary<SuperBlock, Chunks?> ChunkTables = new();

    private static Chunks? ChunksOf(SuperBlock s) => ChunkTables.GetOrAdd(s, static d =>
    {
        if (d.Values % 32 != 0 || d.Planes.Sum(p => p.Bits) > 8 || d.Planes.Any(p => p.Group * p.Bits / 8 is var per && (per < 32 || per % 32 != 0) || p.Group % 32 != 0))
        {
            return null;
        }

        int chunks = d.Values / 32;
        var offsets = d.Planes.Select(p => Enumerable.Range(0, chunks).Select(c => p.Place(32 * c).Byte).ToArray()).ToArray();
        var shifts = d.Planes.Select(p => Enumerable.Range(0, chunks).Select(c => p.Place(32 * c).Shift).ToArray()).ToArray();
        if (offsets.Any(o => o.Any(at => at + 32 > d.Bytes)))
        {
            return null;
        }

        var low = new int[d.Planes.Count];
        for (int p = 1; p < low.Length; p++)
        {
            low[p] = low[p - 1] + d.Planes[p - 1].Bits;
        }

        return new Chunks(offsets, shifts, [.. d.Planes.Select(p => (byte)((1 << p.Bits) - 1))], low);
    });

    // Chunk c's 32 codes plus the bias as int8 lanes, straight from the planes: a 256-bit load, shift and mask a plane.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<sbyte> Codes32(Chunks chunks, ReadOnlySpan<byte> block, int c, Vector256<byte> bias)
    {
        var codes = Vector256<byte>.Zero;
        for (int p = 0; p < chunks.Masks.Length; p++)
        {
            var bytes = Vector256.Create(block.Slice(chunks.Offsets[p][c], 32));
            var field = Vector256.ShiftRightLogical(bytes.AsUInt16(), chunks.Shifts[p][c]).AsByte() & Vector256.Create(chunks.Masks[p]);
            codes |= chunks.Low[p] == 0 ? field : Vector256.ShiftLeft(field.AsUInt16(), chunks.Low[p]).AsByte();
        }

        return (codes + bias).AsSByte();
    }

    // Whether the fused unpack-and-dot runs here: 256-bit vectors and the s8 dot instruction on them.
    private static bool Fused32 => Vector256.IsHardwareAccelerated && AvxVnniInt8.IsSupported && Vector<byte>.Count == 32;

    // Weight rows [j0, j0 + count) against up to 8 rows of x: each super-block's codes decoded once, a sub-block a dot.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemvSuper(SuperBlock s, StoredLane lane, sbyte[] xq, float[] xs, int[] xsum, float[] y, int m, int n, int k, int j0, int count)
    {
        if (Fused32 && ChunksOf(s) is { } chunks)
        {
            GemvSuperFused(s, chunks, lane, xq, xs, xsum, y, m, n, k, j0, count);
            return;
        }

        var bytes = MemoryMarshal.AsBytes(lane.Words.AsSpan(0, lane.Length));
        int per = s.Values / s.SubValues, subs = k / s.SubValues, blocks = k / 32;
        Span<float> acc = stackalloc float[Int8DirectRows];
        Span<sbyte> codes = stackalloc sbyte[s.Values];
        Span<float> scales = stackalloc float[per], mins = stackalloc float[per];
        for (int r = j0; r < j0 + count; r++)
        {
            acc[..m].Clear();
            var row = bytes.Slice((int)(r * lane.RowBytes), (int)lane.RowBytes);
            for (int b = 0; b < k / s.Values; b++)
            {
                s.DecodeCodes(row.Slice(b * s.Bytes, s.Bytes), codes, scales, mins);
                for (int jj = 0; jj < per; jj++)
                {
                    int at = b * s.Values + jj * s.SubValues, j = at / s.SubValues;
                    var w = codes.Slice(jj * s.SubValues, s.SubValues);
                    for (int i = 0; i < m; i++)
                    {
                        var q = xq.AsSpan(i * k + at, s.SubValues);
                        int dot = s.SubValues == 32 ? Dot32(w, q) : Dot16(w, q);
                        acc[i] += xs[i * blocks + at / 32] * (scales[jj] * dot - mins[jj] * xsum[i * subs + j]);
                    }
                }
            }

            for (int i = 0; i < m; i++)
            {
                y[i * n + r] = acc[i];
            }
        }
    }

    // The fused unpack-and-dot (Fused32): per super-block its sub-blocks' scales and minimums, then per 32-value chunk
    // its codes from the planes in registers (Codes32) and one 256-bit s8 dot with each row of x: the dot's eight lanes
    // are the chunk's four-value sums, so lanes 0-3 and 4-7 are the two 16-value sub-blocks (or all eight one of 32).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemvSuperFused(SuperBlock s, Chunks chunks, StoredLane lane, sbyte[] xq, float[] xs, int[] xsum, float[] y, int m, int n, int k, int j0, int count)
    {
        var bytes = MemoryMarshal.AsBytes(lane.Words.AsSpan(0, lane.Length));
        int per = s.Values / s.SubValues, subs = k / s.SubValues, blocks = k / 32;
        Span<float> acc = stackalloc float[Int8DirectRows];
        Span<float> scales = stackalloc float[per], mins = stackalloc float[per];
        var bias = Vector256.Create(unchecked((byte)s.CodeBias));
        for (int r = j0; r < j0 + count; r++)
        {
            acc[..m].Clear();
            var row = bytes.Slice((int)(r * lane.RowBytes), (int)lane.RowBytes);
            for (int b = 0; b < k / s.Values; b++)
            {
                var block = row.Slice(b * s.Bytes, s.Bytes);
                float d = s.FloatAt(block, s.ScaleOffset), dmin = s.MinOffset >= 0 ? s.FloatAt(block, s.MinOffset) : 0f;
                for (int jj = 0; jj < per; jj++)
                {
                    scales[jj] = d * (SuperBlock.Assemble(s.SubScales[jj], block, s.SubScalesSigned) + s.SubScaleBias);
                    mins[jj] = s.SubMins is null ? 0f : dmin * SuperBlock.Assemble(s.SubMins[jj], block, false);
                }

                for (int c = 0; c < s.Values / 32; c++)
                {
                    var codes = Codes32(chunks, block, c, bias);
                    int at = b * s.Values + 32 * c;
                    for (int i = 0; i < m; i++)
                    {
                        var dots = AvxVnniInt8.MultiplyWideningAndAdd(Vector256<int>.Zero, codes, Vector256.Create(xq.AsSpan(i * k + at, 32)));
                        float dx = xs[i * blocks + at / 32];
                        if (s.SubValues == 32)
                        {
                            int jj = c, j = at / 32;
                            acc[i] += dx * (scales[jj] * Vector256.Sum(dots) - mins[jj] * xsum[i * subs + j]);
                        }
                        else
                        {
                            int jj = 2 * c, j = at / 16;
                            int lowSum = Vector128.Sum(dots.GetLower()), highSum = Vector128.Sum(dots.GetUpper());
                            acc[i] += dx * (scales[jj] * lowSum - mins[jj] * xsum[i * subs + j]);
                            acc[i] += dx * (scales[jj + 1] * highSum - mins[jj + 1] * xsum[i * subs + j + 1]);
                        }
                    }
                }
            }

            for (int i = 0; i < m; i++)
            {
                y[i * n + r] = acc[i];
            }
        }
    }

    // The 16-term integer dot: the s8 dot lane over a 128-bit vector where the CPU has it, else plain integers.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Dot16(ReadOnlySpan<sbyte> p, ReadOnlySpan<sbyte> q)
    {
        if (AvxVnniInt8.IsSupported)
        {
            return Vector128.Sum(AvxVnniInt8.MultiplyWideningAndAdd(Vector128<int>.Zero, Vector128.Create(p), Vector128.Create(q)));
        }

        if (Dp.IsSupported)
        {
            return Vector128.Sum(Dp.DotProduct(Vector128<int>.Zero, Vector128.Create(p), Vector128.Create(q)));
        }

        int sum = 0;
        for (int i = 0; i < 16; i++)
        {
            sum += p[i] * q[i];
        }

        return sum;
    }

    // Adad's tile over super-block rows: each panel row's codes and its sub-blocks' scales a 4-byte group, then the tile;
    // the minimums' terms after, as one small product: y −= (dx · Σ q_x)[m, k / SubValues] · minsᵀ.
    private static void MatMulSuper(SuperBlock s, StoredLane lane, sbyte[] xq, float[] xs, int[] xsum, float[] y, int m, int n, int k)
    {
        int k4 = k / 4, subs = k / s.SubValues, blocks = k / 32;
        int panel = (int)Math.Max(1, Math.Min(n, Math.Max(2, BlockPanelBytes / (2L * k))));
        var qb = ArrayPool<sbyte>.Shared.Rent(panel * k);
        var sb = ArrayPool<float>.Shared.Rent(panel * k4);
        var sa = ArrayPool<float>.Shared.Rent(m * k4);
        float[]? mins = s.SubMins is null ? null : ArrayPool<float>.Shared.Rent(n * subs);
        try
        {
            for (int i = 0; i < m * blocks; i++)
            {
                sa.AsSpan(i * 8, 8).Fill(xs[i]);
            }

            int threads = Math.Max(1, ComputeResources.MaxCpuThreads);
            for (int j0 = 0; j0 < n; j0 += panel)
            {
                int cols = Math.Min(panel, n - j0), from = j0;
                int chunk = Math.Max(1, cols / (threads * 4));
                Parallel.For(0, (cols + chunk - 1) / chunk, ComputeResources.ParallelOptions, c =>
                {
                    Span<float> scales = stackalloc float[s.Values / s.SubValues], blockMins = stackalloc float[s.Values / s.SubValues];
                    var bytes = MemoryMarshal.AsBytes(lane.Words.AsSpan(0, lane.Length));
                    for (int r = c * chunk; r < Math.Min(cols, (c + 1) * chunk); r++)
                    {
                        var row = bytes.Slice((int)((from + r) * lane.RowBytes), (int)lane.RowBytes);
                        for (int b = 0; b < k / s.Values; b++)
                        {
                            if (Fused32 && ChunksOf(s) is { } chunks)
                            {
                                var block = row.Slice(b * s.Bytes, s.Bytes);
                                var bias = Vector256.Create(unchecked((byte)s.CodeBias));
                                for (int ch = 0; ch < s.Values / 32; ch++)
                                {
                                    Codes32(chunks, block, ch, bias).CopyTo(qb.AsSpan(r * k + b * s.Values + 32 * ch, 32));
                                }

                                float d = s.FloatAt(block, s.ScaleOffset), dmin = s.MinOffset >= 0 ? s.FloatAt(block, s.MinOffset) : 0f;
                                for (int jj = 0; jj < scales.Length; jj++)
                                {
                                    scales[jj] = d * (SuperBlock.Assemble(s.SubScales[jj], block, s.SubScalesSigned) + s.SubScaleBias);
                                    blockMins[jj] = s.SubMins is null ? 0f : dmin * SuperBlock.Assemble(s.SubMins[jj], block, false);
                                }
                            }
                            else
                            {
                                s.DecodeCodes(row.Slice(b * s.Bytes, s.Bytes), qb.AsSpan(r * k + b * s.Values, s.Values), scales, blockMins);
                            }
                            for (int jj = 0; jj < scales.Length; jj++)
                            {
                                sb.AsSpan(r * k4 + (b * s.Values + jj * s.SubValues) / 4, s.SubValues / 4).Fill(scales[jj]);
                                if (mins is not null)
                                {
                                    mins[(from + r) * subs + b * scales.Length + jj] = blockMins[jj];
                                }
                            }
                        }
                    }
                });
                Parallel.For(0, (m + 3) / 4, ComputeResources.ParallelOptions, t => BlocksTask(t, xq, sa, qb, sb, y, m, n, k, from, cols));
            }

            if (mins is not null)
            {
                var weighted = ArrayPool<float>.Shared.Rent(m * subs);
                var correction = ArrayPool<float>.Shared.Rent(m * n);
                try
                {
                    for (int i = 0; i < m; i++)
                    {
                        for (int j = 0; j < subs; j++)
                        {
                            weighted[i * subs + j] = xs[i * blocks + j * s.SubValues / 32] * xsum[i * subs + j];
                        }
                    }

                    CpuMatMul.Multiply(weighted, mins, correction, m, n, subs, false, true, 0f);
                    for (int i = 0; i < m * n; i++)
                    {
                        y[i] -= correction[i];
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(weighted);
                    ArrayPool<float>.Shared.Return(correction);
                }
            }
        }
        finally
        {
            ArrayPool<sbyte>.Shared.Return(qb);
            ArrayPool<float>.Shared.Return(sb);
            ArrayPool<float>.Shared.Return(sa);
            if (mins is not null)
            {
                ArrayPool<float>.Shared.Return(mins);
            }
        }
    }

    // y [m, n] = x [m, k] · Wᵀ by the block route.
    private static void RunBlocks(StoredLane lane, float[] x, float[] y, int m, int n, int k)
    {
        var layout = BlockLayout.Of(lane.Type, lane.RowScales is not null)!;
        int blocks = k / 32;
        var xq = ArrayPool<sbyte>.Shared.Rent(m * k);
        var xs = ArrayPool<float>.Shared.Rent(m * blocks);
        try
        {
            if (m == 1)
            {
                QuantizeRow(x.AsSpan(0, k), xq.AsSpan(0, k), xs.AsSpan(0, blocks));
            }
            else
            {
                Parallel.For(0, m, ComputeResources.ParallelOptions, i => QuantizeRow(x.AsSpan(i * k, k), xq.AsSpan(i * k, k), xs.AsSpan(i * blocks, blocks)));
            }

            if (m <= Int8DirectRows)
            {
                int threads = Math.Max(1, ComputeResources.MaxCpuThreads), chunk = Math.Max(1, n / (threads * 4));
                Parallel.For(0, (n + chunk - 1) / chunk, ComputeResources.ParallelOptions,
                    c => GemvBlocks(layout, lane, xq, xs, y, m, n, k, c * chunk, Math.Min(chunk, n - c * chunk)));
            }
            else
            {
                MatMulBlocks(layout, lane, xq, xs, y, m, n, k);
            }
        }
        finally
        {
            ArrayPool<sbyte>.Shared.Return(xq);
            ArrayPool<float>.Shared.Return(xs);
        }
    }

    // Adad's q8_0 encoder for one row of x: each block's scale (float16, read back) and its 32 codes.
    private static void QuantizeRow(ReadOnlySpan<float> x, Span<sbyte> q, Span<float> scales)
    {
        for (int b = 0; b < scales.Length; b++)
        {
            var v = x.Slice(b * 32, 32);
            float amax = 0;
            for (int i = 0; i < 32; i++)
            {
                float a = MathF.Abs(v[i]);
                if (a > amax)
                {
                    amax = a;                                                    // NaN dropped, as the device's max drops it
                }
            }

            float d = (float)(Half)(amax / 127f), id = d == 0 ? 0 : 1f / d;
            scales[b] = d;
            var codes = q.Slice(b * 32, 32);
            for (int i = 0; i < 32; i++)
            {
                codes[i] = (sbyte)Math.Clamp((int)MathF.Round(v[i] * id), -127, 127);
            }
        }
    }

    // A weight block's scale (its own, or the row's) and its 32 codes widened to int8 by the layout.
    private static float UnpackBlock(BlockLayout layout, ReadOnlySpan<byte> block, float rowScale, Span<sbyte> q)
    {
        var codes = block[layout.CodeOffset..];
        if (layout.CodeBits == 4)
        {
            // 16 bytes a pass: each half widened to int8 in byte arithmetic (code + bias fits int8, so the wrap is exact).
            var v = Vector128.Create(codes[..16]);
            var mask = Vector128.Create((byte)15);
            var lo = v & mask;
            var hi = Vector128.ShiftRightLogical(v.AsUInt16(), 4).AsByte() & mask;
            if (layout.Table is not null)
            {
                lo = Vector128.Shuffle(layout.Grid, lo.AsSByte()).AsByte();
                hi = Vector128.Shuffle(layout.Grid, hi.AsSByte()).AsByte();
            }
            else if (layout.Signed)
            {
                var eight = Vector128.Create((byte)8);
                lo = (lo ^ eight) - eight;
                hi = (hi ^ eight) - eight;
            }

            var bias = Vector128.Create(unchecked((byte)layout.Bias));
            (lo + bias).AsSByte().CopyTo(q[..16]);
            (hi + bias).AsSByte().CopyTo(q.Slice(16, 16));
        }
        else if (layout.Direct)
        {
            MemoryMarshal.Cast<byte, sbyte>(codes[..32]).CopyTo(q);
        }
        else
        {
            for (int i = 0; i < 32; i++)
            {
                q[i] = (sbyte)((layout.Signed ? (sbyte)codes[i] : codes[i]) + layout.Bias);
            }
        }

        return BlockScale(layout, block, rowScale);
    }

    private static float BlockScale(BlockLayout layout, ReadOnlySpan<byte> block, float rowScale)
    {
        if (!layout.HasScale)
        {
            return rowScale;
        }

        ushort bits = BinaryPrimitives.ReadUInt16LittleEndian(block.Slice(layout.ScaleOffset, 2));
        return layout.HalfScale ? (float)BitConverter.UInt16BitsToHalf(bits) : BitConverter.Int32BitsToSingle(bits << 16);
    }

    // Adad's GemvBlocks over weight rows [j0, j0 + count): each row's blocks once, dotted with every row of x.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemvBlocks(BlockLayout layout, StoredLane lane, sbyte[] xq, float[] xs, float[] y, int m, int n, int k, int j0, int count)
    {
        var bytes = MemoryMarshal.AsBytes(lane.Words.AsSpan(0, lane.Length));
        int blocks = k / 32;
        Span<float> acc = stackalloc float[Int8DirectRows];
        Span<sbyte> unpacked = stackalloc sbyte[32];
        for (int r = j0; r < j0 + count; r++)
        {
            acc[..m].Clear();
            float rowScale = lane.RowScales?[r] ?? 1f;
            var row = bytes.Slice((int)(r * lane.RowBytes), (int)lane.RowBytes);
            for (int b = 0; b < blocks; b++)
            {
                var block = row.Slice(b * layout.Bytes, layout.Bytes);
                scoped ReadOnlySpan<sbyte> codes;
                float d;
                if (layout.Direct)
                {
                    codes = MemoryMarshal.Cast<byte, sbyte>(block.Slice(layout.CodeOffset, 32));
                    d = BlockScale(layout, block, rowScale);
                }
                else
                {
                    d = UnpackBlock(layout, block, rowScale, unpacked);
                    codes = unpacked;
                }

                for (int i = 0; i < m; i++)
                {
                    int dot = Dot32(codes, xq.AsSpan(i * k + b * 32, 32));
                    acc[i] += d * xs[i * blocks + b] * dot;
                }
            }

            for (int i = 0; i < m; i++)
            {
                y[i * n + r] = acc[i];
            }
        }
    }

    // The 32-term integer dot: the s8 dot lane over 4-byte groups where vectors hold 32 bytes, else plain integers.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Dot32(ReadOnlySpan<sbyte> p, ReadOnlySpan<sbyte> q)
    {
        if (Vector<int>.Count == 8)
        {
            var pv = Vector.AsVectorInt32(new Vector<sbyte>(p));
            var qv = Vector.AsVectorInt32(new Vector<sbyte>(q));
            return Vector.Sum(S8Fma(pv, qv, Vector<int>.Zero));
        }

        int sum = 0;
        for (int i = 0; i < 32; i++)
        {
            sum += p[i] * q[i];
        }

        return sum;
    }

    // Adad's S8Dot.Fma: c + the sums of four signed byte products in each 32-bit lane.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<int> S8Fma(Vector<int> a, Vector<int> b, Vector<int> c)
    {
        if (Vector<byte>.Count == 32)
        {
            return AvxVnniInt8.MultiplyWideningAndAdd(c.AsVector256(), a.AsVector256().AsSByte(), b.AsVector256().AsSByte()).AsVector();
        }

        if (AvxVnniInt8.IsSupported)
        {
            return AvxVnniInt8.MultiplyWideningAndAdd(c.AsVector128(), a.AsVector128().AsSByte(), b.AsVector128().AsSByte()).AsVector();
        }

        return Dp.DotProduct(c.AsVector128(), a.AsVector128().AsSByte(), b.AsVector128().AsSByte()).AsVector();
    }

    // Four signed byte products of a and b summed into c (the dot lane's scalar form).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Dot8(int a, int b, int c) =>
        c + (sbyte)a * (sbyte)b + (sbyte)(a >> 8) * (sbyte)(b >> 8) + (sbyte)(a >> 16) * (sbyte)(b >> 16) + (sbyte)(a >> 24) * (sbyte)(b >> 24);

    // A panel of the weights' unpacked rows: about this many bytes (Adad's BlockPanelBytes).
    private const long BlockPanelBytes = 8L << 20;

    // Adad's MatMulBlocks: the weights' rows unpacked a panel at a time, then tasks of 4 rows of x against the panel.
    private static void MatMulBlocks(BlockLayout layout, StoredLane lane, sbyte[] xq, float[] xs, float[] y, int m, int n, int k) =>
        MatMulUnpacked((row, q, s) => UnpackRow(layout, lane, row, k, q, s), xq, xs, y, m, n, k);

    // A weight row's int8 codes (k of them) and its blocks' scales a 4-byte group (k / 4), from any block format.
    private delegate void RowUnpack(int row, Span<sbyte> q, Span<float> s);

    // The block product over rows unpacked by `unpack`: a panel of rows at a time, then tasks of 4 rows of x against it.
    private static void MatMulUnpacked(RowUnpack unpack, sbyte[] xq, float[] xs, float[] y, int m, int n, int k)
    {
        int k4 = k / 4;
        int panel = (int)Math.Max(1, Math.Min(n, Math.Max(2, BlockPanelBytes / (2L * k))));
        var qb = ArrayPool<sbyte>.Shared.Rent(panel * k);
        var sb = ArrayPool<float>.Shared.Rent(panel * k4);
        var sa = ArrayPool<float>.Shared.Rent(m * k4);
        try
        {
            // x's scales a 4-byte group (each block's scale eight times), so any vector width reads them as it reads the codes.
            int blocks = k / 32;
            for (int i = 0; i < m * blocks; i++)
            {
                sa.AsSpan(i * 8, 8).Fill(xs[i]);
            }

            int threads = Math.Max(1, ComputeResources.MaxCpuThreads);
            for (int j0 = 0; j0 < n; j0 += panel)
            {
                int cols = Math.Min(panel, n - j0), from = j0;
                int chunk = Math.Max(1, cols / (threads * 4));
                Parallel.For(0, (cols + chunk - 1) / chunk, ComputeResources.ParallelOptions, c =>
                {
                    for (int r = c * chunk; r < Math.Min(cols, (c + 1) * chunk); r++)
                    {
                        unpack(from + r, qb.AsSpan(r * k, k), sb.AsSpan(r * k4, k4));
                    }
                });
                Parallel.For(0, (m + 3) / 4, ComputeResources.ParallelOptions, t => BlocksTask(t, xq, sa, qb, sb, y, m, n, k, from, cols));
            }
        }
        finally
        {
            ArrayPool<sbyte>.Shared.Return(qb);
            ArrayPool<float>.Shared.Return(sb);
            ArrayPool<float>.Shared.Return(sa);
        }
    }

    // A weight row's codes as int8 and its blocks' scales a 4-byte group.
    private static void UnpackRow(BlockLayout layout, StoredLane lane, int row, int k, Span<sbyte> q, Span<float> s)
    {
        var bytes = MemoryMarshal.AsBytes(lane.Words.AsSpan(0, lane.Length)).Slice((int)(row * lane.RowBytes), (int)lane.RowBytes);
        float rowScale = lane.RowScales?[row] ?? 1f;
        for (int b = 0; b < k / 32; b++)
        {
            float d = UnpackBlock(layout, bytes.Slice(b * layout.Bytes, layout.Bytes), rowScale, q.Slice(b * 32, 32));
            s.Slice(b * 8, 8).Fill(d);
        }
    }

    // A task: rows 4t .. 4t + 3 of x (the last row repeated past m) against the panel's rows (y's columns j0 ..).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void BlocksTask(int t, sbyte[] xq, float[] sa, sbyte[] qb, float[] sb, float[] y, int m, int n, int k, int j0, int cols)
    {
        int w = Vector<int>.Count, k4 = k / 4, i0 = t * 4;
        var QA = MemoryMarshal.Cast<sbyte, int>(xq.AsSpan(0, m * k));
        var QB = MemoryMarshal.Cast<sbyte, int>(qb.AsSpan(0, cols * k));
        int r0 = Math.Min(i0, m - 1), r1 = Math.Min(i0 + 1, m - 1), r2 = Math.Min(i0 + 2, m - 1), r3 = Math.Min(i0 + 3, m - 1);
        Span<float> acc = stackalloc float[8];
        for (int jj = 0; jj < cols; jj += 2)
        {
            int c0 = jj, c1 = Math.Min(jj + 1, cols - 1);
            Vector<float> a00 = default, a01 = default, a10 = default, a11 = default, a20 = default, a21 = default, a30 = default, a31 = default;
            int v = 0;
            for (; v + w <= k4; v += w)
            {
                var x0 = new Vector<int>(QA.Slice(r0 * k4 + v, w)); var x1 = new Vector<int>(QA.Slice(r1 * k4 + v, w));
                var x2 = new Vector<int>(QA.Slice(r2 * k4 + v, w)); var x3 = new Vector<int>(QA.Slice(r3 * k4 + v, w));
                var s0 = new Vector<float>(sa.AsSpan(r0 * k4 + v, w)); var s1 = new Vector<float>(sa.AsSpan(r1 * k4 + v, w));
                var s2 = new Vector<float>(sa.AsSpan(r2 * k4 + v, w)); var s3 = new Vector<float>(sa.AsSpan(r3 * k4 + v, w));
                var y0 = new Vector<int>(QB.Slice(c0 * k4 + v, w)); var y1 = new Vector<int>(QB.Slice(c1 * k4 + v, w));
                var t0 = new Vector<float>(sb.AsSpan(c0 * k4 + v, w)); var t1 = new Vector<float>(sb.AsSpan(c1 * k4 + v, w));
                a00 += Vector.ConvertToSingle(S8Fma(x0, y0, Vector<int>.Zero)) * (s0 * t0);
                a01 += Vector.ConvertToSingle(S8Fma(x0, y1, Vector<int>.Zero)) * (s0 * t1);
                a10 += Vector.ConvertToSingle(S8Fma(x1, y0, Vector<int>.Zero)) * (s1 * t0);
                a11 += Vector.ConvertToSingle(S8Fma(x1, y1, Vector<int>.Zero)) * (s1 * t1);
                a20 += Vector.ConvertToSingle(S8Fma(x2, y0, Vector<int>.Zero)) * (s2 * t0);
                a21 += Vector.ConvertToSingle(S8Fma(x2, y1, Vector<int>.Zero)) * (s2 * t1);
                a30 += Vector.ConvertToSingle(S8Fma(x3, y0, Vector<int>.Zero)) * (s3 * t0);
                a31 += Vector.ConvertToSingle(S8Fma(x3, y1, Vector<int>.Zero)) * (s3 * t1);
            }

            acc[0] = Vector.Sum(a00); acc[1] = Vector.Sum(a01); acc[2] = Vector.Sum(a10); acc[3] = Vector.Sum(a11);
            acc[4] = Vector.Sum(a20); acc[5] = Vector.Sum(a21); acc[6] = Vector.Sum(a30); acc[7] = Vector.Sum(a31);
            for (; v < k4; v++)                                                 // a vector wider than what is left of k
            {
                ReadOnlySpan<int> rows = [r0, r1, r2, r3];
                ReadOnlySpan<int> both = [c0, c1];
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 2; j++)
                    {
                        int dot = Dot8(QA[rows[i] * k4 + v], QB[both[j] * k4 + v], 0);
                        acc[i * 2 + j] += dot * (sa[rows[i] * k4 + v] * sb[both[j] * k4 + v]);
                    }
                }
            }

            for (int i = 0; i < 4 && i0 + i < m; i++)
            {
                for (int j = 0; j < 2 && jj + j < cols; j++)
                {
                    y[(i0 + i) * n + j0 + jj + j] = acc[i * 2 + j];
                }
            }
        }
    }
}
