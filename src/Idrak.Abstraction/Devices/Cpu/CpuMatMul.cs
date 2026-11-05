// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Idrak.Abstraction.Devices.Cpu;

/// <summary>
/// Single-precision GEMM: C = op(A) * op(B) + beta * C, all row-major. With hardware FMA the product runs in tiles of a
/// register-blocked kernel chosen by the instruction sets the CPU reports (<see cref="CpuTuning.Kernel"/>: AVX-512, AVX2,
/// NEON); otherwise rows of C are processed in blocks of <see cref="Mr"/>, each keeping a <see cref="Mr"/> x (2 * SIMD
/// width) tile of C in registers across the whole k loop. Tile sizes, column blocks and the parallel cut-over come from
/// <see cref="CpuTuning"/> (the machine's caches, vector width and threads, or measured). Every path computes each element
/// as one FMA chain over k from zero, then + beta · C as one FMA, so the choice of path and tiling never changes a result;
/// the one exception (the few-row transposed-B path sums each dot product in the fixed lanes, <see cref="CpuMath.LaneCount"/>)
/// is selected by the CPU's numerical contract (<see cref="CpuTuning.TransposedInPlace"/>), the same on every machine.
/// </summary>
internal static class CpuMatMul
{
    // Rows of the portable kernel's register block: 4 rows × 2 vectors of C (8 registers) plus 2 B vectors and a
    // broadcast fit the 16 vector registers every SIMD instruction set has (SSE, AVX2; NEON has 32). A fact of the
    // instruction sets, not of one processor; the wider blocks of 32-register sets are the tiled kernels.
    private const int Mr = 4;

    public static void Multiply(float[] a, float[] b, float[] c, int m, int n, int k, bool transA, bool transB, float beta) =>
        Multiply(a, 0, b, 0, c, 0, m, n, k, transA, transB, beta);

    /// <summary>
    /// Multiplies the matrices starting at the given element offsets of each array; over chunks of k where
    /// <see cref="SplitsK(int, int, int)"/> says so (<see cref="SplitK"/>), else as one product (<see cref="Unsplit"/>).
    /// </summary>
    public static void Multiply(float[] a, int aOffset, float[] b, int bOffset, float[] c, int cOffset, int m, int n, int k, bool transA, bool transB, float beta)
    {
        if (SplitsK(m, n, k))
        {
            SplitK(a, aOffset, b, bOffset, c, cOffset, m, n, k, transA, transB, beta, ComputeResources.CpuProductKChunk);
            return;
        }

        Unsplit(a, aOffset, b, bOffset, c, cOffset, m, n, k, transA, transB, beta);
    }

    /// <summary>The product as one FMA chain over k per element (never split over k), on the threads the cut-over allows.</summary>
    internal static void Unsplit(float[] a, int aOffset, float[] b, int bOffset, float[] c, int cOffset, int m, int n, int k, bool transA, bool transB, float beta) =>
        Multiply(a, aOffset, b, bOffset, c, cOffset, m, n, k, transA, transB, beta, CpuTuning.ProductCutover((long)m * n * k), ComputeResources.ParallelOptions);

    /// <summary>
    /// Whether an m × n × k product runs over chunks of k: <see cref="ComputeResources.CpuProductKChunk"/> set, k past it,
    /// and the product's row and column blocks too few to occupy <see cref="ComputeResources.MaxCpuThreads"/> (from the
    /// shape and the thread count at run time). A product whose rows and columns already fill the cores (a prompt's) is
    /// never split: there the chunks only add their partial products' traffic.
    /// </summary>
    internal static bool SplitsK(int m, int n, int k) => SplitsK(m, n, k, ComputeResources.CpuProductKChunk, ComputeResources.MaxCpuThreads);

    /// <summary><see cref="SplitsK(int, int, int)"/> for a chunk length and a thread count (tests).</summary>
    internal static bool SplitsK(int m, int n, int k, int chunk, int threads)
    {
        if (chunk <= 0 || k <= chunk || m <= 0 || n <= 0 || threads <= 1)
        {
            return false;
        }

        var kernel = CpuTuning.Kernel;
        int rows = kernel != TiledKernel.None ? CpuTuning.KernelShape(kernel).Rows : Mr;   // a register block's rows
        long rowBlocks = (m + rows - 1) / rows, columnBlocks = (n + CpuTuning.ColumnBlock - 1) / CpuTuning.ColumnBlock;
        return rowBlocks * columnBlocks < threads;
    }

    // One thread a chunk's product: the chunks are what runs in parallel.
    private static readonly ParallelOptions OneThread = new() { MaxDegreeOfParallelism = 1 };

    /// <summary>
    /// The product over k in chunks of <paramref name="chunk"/> (the last one shorter): each chunk's product on its own
    /// (one FMA chain over its part of k from zero), the chunks in parallel, then for each element the partial sums added
    /// in chunk order and + beta · C. The order of every sum follows from k and the chunk length alone (rule 86), whatever
    /// the threads. A chunk's slice of A or B that is not contiguous (A stored [m, k], B stored [n, k]) is copied out
    /// first; A stored [k, m] and B stored [k, n] (a weight's gradient) are read in place. Whether a product runs here is
    /// <see cref="SplitsK(int, int, int)"/>'s (callers that cut a product into parts decide on the whole product).
    /// </summary>
    internal static void SplitK(float[] a, int aOffset, float[] b, int bOffset, float[] c, int cOffset, int m, int n, int k, bool transA, bool transB, float beta, int chunk)
    {
        int chunks = (k + chunk - 1) / chunk, mn = m * n;
        float[] partial = ArrayPool<float>.Shared.Rent(checked(chunks * mn));
        var options = ComputeResources.ParallelOptions;
        try
        {
            Parallel.For(0, chunks, options, index =>
            {
                int k0 = index * chunk, kc = Math.Min(chunk, k - k0);
                float[]? packedA = null, packedB = null;
                try
                {
                    float[] ac = a, bc = b;
                    int ao = aOffset + k0 * m, bo = bOffset + k0 * n;          // [k, m] and [k, n]: rows k0 on, in place
                    if (!transA)
                    {
                        packedA = ArrayPool<float>.Shared.Rent(m * kc);          // [m, k]: columns k0 on of each row
                        for (int r = 0; r < m; r++)
                        {
                            Array.Copy(a, aOffset + (long)r * k + k0, packedA, r * kc, kc);
                        }

                        (ac, ao) = (packedA, 0);
                    }

                    if (transB)
                    {
                        packedB = ArrayPool<float>.Shared.Rent(n * kc);          // [n, k]: columns k0 on of each row
                        for (int j = 0; j < n; j++)
                        {
                            Array.Copy(b, bOffset + (long)j * k + k0, packedB, j * kc, kc);
                        }

                        (bc, bo) = (packedB, 0);
                    }

                    Multiply(ac, ao, bc, bo, partial, index * mn, m, n, kc, transA, transB, 0f, long.MaxValue, OneThread);
                }
                finally
                {
                    if (packedA is not null)
                    {
                        ArrayPool<float>.Shared.Return(packedA);
                    }

                    if (packedB is not null)
                    {
                        ArrayPool<float>.Shared.Return(packedB);
                    }
                }
            });

            // Each element: its chunks' sums in chunk order, then + beta · C (rows in parallel; no element's order changes).
            Parallel.For(0, m, options, r =>
            {
                for (int j = r * n, end = j + n; j < end; j++)
                {
                    float sum = partial[j];
                    for (int index = 1; index < chunks; index++)
                    {
                        sum += partial[index * mn + j];
                    }

                    ref float dst = ref c[cOffset + j];
                    dst = beta == 0f ? sum : MathF.FusedMultiplyAdd(beta, dst, sum);
                }
            });
        }
        finally
        {
            ArrayPool<float>.Shared.Return(partial);
        }
    }

    /// <summary>
    /// The product with an explicit parallel cut-over in m·n·k (<paramref name="parallelWork"/>) and thread options: what
    /// <see cref="CpuTuning"/> times to measure the cut-over.
    /// </summary>
    internal static void Multiply(float[] a, int aOffset, float[] b, int bOffset, float[] c, int cOffset, int m, int n, int k, bool transA, bool transB, float beta,
        long parallelWork, ParallelOptions options)
    {
        bool parallel = options.MaxDegreeOfParallelism > 1 && (long)m * n * k >= parallelWork;
        if (transB && m <= CpuTuning.TransposedInPlace)
        {
            // Few rows against B stored [n, k] (decoding through a tied head, backward products of one token): every
            // output is a dot product of an A row with a contiguous B row, so B is read once in place, not transposed.
            FewRowsTransposedB(a, aOffset, b, bOffset, c, cOffset, m, n, k, transA, beta, parallel, options);
            return;
        }

        float[]? rented = null;
        try
        {
            if (transB)
            {
                // B is stored [n, k]; transpose it once into [k, n] so the kernel always streams contiguous B rows.
                rented = ArrayPool<float>.Shared.Rent(k * n);
                Transpose(b, bOffset, rented, n, k, CpuTuning.TransposeSide, options);
                b = rented;
                bOffset = 0;
            }

            Product(a, aOffset, b, bOffset, c, cOffset, m, n, k, transA, beta, parallel, options);
        }
        finally
        {
            if (rented is not null)
            {
                ArrayPool<float>.Shared.Return(rented);
            }
        }
    }

    // The product with B stored [k, n] (transposed already when it was not).
    private static void Product(float[] a, int aOffset, float[] b, int bOffset, float[] c, int cOffset, int m, int n, int k, bool transA, float beta,
        bool parallel, ParallelOptions options)
    {
        // Element (i, p) of op(A) lives at a[i * rowStride + p * colStride].
        nint rowStride = transA ? 1 : k;
        nint colStride = transA ? m : 1;
        int blocks = (m + Mr - 1) / Mr;

        int columnBlock = CpuTuning.ColumnBlock;
        var kernel = CpuTuning.Kernel;
        // Rows below one register block of the tiled kernel (or below Mr without one) would otherwise go to the row-block
        // path, whose one or two blocks leave most threads idle (5 to 7 rows ran several times slower than 4 or 8).
        int fewRows = kernel != TiledKernel.None ? Math.Max(Mr, CpuTuning.KernelShape(kernel).Rows - 1) : Mr;
        bool columnSplit = beta == 0f
            ? m <= fewRows && n >= 2 * columnBlock && parallel               // same results either way: tuned
            : m <= CpuTuning.ColumnSplitRows && n >= CpuTuning.ColumnSplitColumns && (long)m * n * k >= CpuTuning.ColumnSplitWork
              && options.MaxDegreeOfParallelism > 1;                         // every path fuses beta · C: the same results
        if (columnSplit)
        {
            // Few rows (token-by-token decoding): a single row block would leave most threads idle, so split the columns
            // instead and stream B row by row (contiguous reads; each weight is read once).
            FewRows(a, aOffset, b, bOffset, c, cOffset, m, n, k, rowStride, colStride, beta, columnBlock, options);
            return;
        }

        if (kernel != TiledKernel.None && m >= CpuTuning.KernelShape(kernel).Rows)
        {
            // At least one register block of the tiled kernel (fewer rows would leave the kernel unused).
            Tiled(a, aOffset, b, bOffset, c, cOffset, m, n, k, rowStride, colStride, beta, kernel, parallel, options);
        }
        else if (!parallel || blocks == 1)
        {
            for (int block = 0; block < blocks; block++)
            {
                RunBlock(a, aOffset, b, bOffset, c, cOffset, block, m, n, k, rowStride, colStride, beta);
            }
        }
        else
        {
            float[] bb = b;
            int bo = bOffset;
            Parallel.For(0, blocks, options, block => RunBlock(a, aOffset, bb, bo, c, cOffset, block, m, n, k, rowStride, colStride, beta));
        }
    }

    private static void FewRowsTransposedB(float[] a, int aOffset, float[] b, int bOffset, float[] c, int cOffset, int m, int n, int k, bool transA, float beta,
        bool parallel, ParallelOptions options)
    {
        // The A rows contiguous ([m, k]); a transposed A ([k, m]) is gathered once (m·k values, small here).
        float[]? rows = null;
        float[] ar = a;
        int ao = aOffset;
        try
        {
            if (transA)
            {
                // Read in A's own order (row p of [k, m] holds column p of every A row), written to the m rows.
                rows = ArrayPool<float>.Shared.Rent(m * k);
                for (int p = 0; p < k; p++)
                {
                    for (int i = 0; i < m; i++)
                    {
                        rows[i * k + p] = a[aOffset + p * m + i];
                    }
                }

                ar = rows;
                ao = 0;
            }

            int Chunk = CpuTuning.ColumnBlock;                             // columns per work item (each a dot product over k)
            int chunks = (n + Chunk - 1) / Chunk;
            if (parallel && chunks > 1)
            {
                Parallel.For(0, chunks, options, index => DotColumns(ar, ao, b, bOffset, c, cOffset, m, n, k, beta, index * Chunk, Math.Min(n, (index + 1) * Chunk)));
            }
            else
            {
                DotColumns(ar, ao, b, bOffset, c, cOffset, m, n, k, beta, 0, n);
            }
        }
        finally
        {
            if (rows is not null)
            {
                ArrayPool<float>.Shared.Return(rows);
            }
        }
    }

    // C[i, j] = A[i, :] · B[j, :] (+ beta · C[i, j]) for the columns j0..j1; each B row is read once for all m rows. Each
    // dot product in the fixed lanes (CpuMath.DotFused: one FMA per term onto lane p mod 16), the same bits on every machine.
    private static void DotColumns(float[] a, int aOffset, float[] b, int bOffset, float[] c, int cOffset, int m, int n, int k, float beta, int j0, int j1)
    {
        ref float ra = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(a), aOffset);
        ref float rb = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(b), bOffset);
        for (int j = j0; j < j1; j++)
        {
            ref float row = ref Unsafe.Add(ref rb, (nint)j * k);
            for (int i = 0; i < m; i++)
            {
                float sum = CpuMath.DotFused(ref Unsafe.Add(ref ra, (nint)i * k), ref row, k);
                ref float dst = ref c[cOffset + i * n + j];
                dst = beta == 0f ? sum : MathF.FusedMultiplyAdd(beta, dst, sum);
            }
        }
    }

    /// <summary>
    /// x · row over k as <see cref="DotColumns"/> sums each output (one FMA per term onto its fixed lane, the lanes added in
    /// their fixed tree), with the row held as bfloat16 halves widened in registers as they are read (exactly: the halves
    /// as a float's high bits). The same operations on the same values, so the bits of the few-row transposed-B product
    /// over the widened row; attention over a bfloat16 cache read in place needs them (the tests compare it with the
    /// product bit for bit).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float DotLanesBFloat16(ref float x, ref ushort row, int k) => CpuMath.DotFused(ref x, ref row, k);

    private static void FewRows(float[] a, int aOffset, float[] b, int bOffset, float[] c, int cOffset, int m, int n, int k, nint rowStride, nint colStride, float beta,
        int columnBlock, ParallelOptions options)
    {
        int w = Vector<float>.Count;
        int threads = Math.Max(1, options.MaxDegreeOfParallelism is > 0 and var max ? max : Environment.ProcessorCount);
        int chunk = Math.Max(columnBlock, (n / (2 * threads) + w - 1) / w * w);   // about two chunks per thread, whole vectors
        int chunks = (n + chunk - 1) / chunk;
        Parallel.For(0, chunks, options, index =>
        {
            int j0 = index * chunk, width = Math.Min(chunk, n - j0);
            float[] acc = ArrayPool<float>.Shared.Rent(m * width);
            try
            {
                Array.Clear(acc, 0, m * width);
                ref float rb = ref MemoryMarshal.GetArrayDataReference(b);
                ref float rs = ref MemoryMarshal.GetArrayDataReference(acc);

                // Each weight row is read once and feeds every input row (per element: the same FMA chain over k).
                for (int p = 0; p < k; p++)
                {
                    ref float row = ref Unsafe.Add(ref rb, bOffset + (nint)p * n + j0);
                    for (int i = 0; i < m; i++)
                    {
                        float x = a[aOffset + i * rowStride + p * colStride];
                        if (x == 0f)
                        {
                            continue;
                        }

                        var xv = new Vector<float>(x);
                        ref float sum = ref Unsafe.Add(ref rs, i * width);
                        int j = 0;
                        for (; j <= width - w; j += w)
                        {
                            Vector.FusedMultiplyAdd(xv, Vector.LoadUnsafe(ref row, (nuint)j), Vector.LoadUnsafe(ref sum, (nuint)j)).StoreUnsafe(ref sum, (nuint)j);
                        }

                        for (; j < width; j++)
                        {
                            Unsafe.Add(ref sum, j) = MathF.FusedMultiplyAdd(x, Unsafe.Add(ref row, j), Unsafe.Add(ref sum, j));
                        }
                    }
                }

                for (int i = 0; i < m; i++)
                {
                    var dst = c.AsSpan(cOffset + i * n + j0, width);
                    var sums = acc.AsSpan(i * width, width);
                    for (int j = 0; j < width; j++)
                    {
                        dst[j] = beta == 0f ? sums[j] : MathF.FusedMultiplyAdd(beta, dst[j], sums[j]);   // as Store's lanes
                    }
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(acc);
            }
        });
    }

    // Hardware FMA (AVX-512, AVX2 or NEON): C in tiles of mc rows × nc columns (spread over the threads). A tile packs its
    // A rows once ([k][mr] per row block) and each nr-column panel of B ([k][nr], contiguous: no cache-set conflicts from
    // power-of-two row strides), which every row block then reuses from L2. The grid (CpuTuning.TileGrid) from the L2 size
    // and the thread count. Every element is still one FMA chain over k from zero, then + beta · C, as in the other paths.
    private static void Tiled(float[] a, int aOffset, float[] b, int bOffset, float[] c, int cOffset, int m, int n, int k, nint rowStride, nint colStride, float beta,
        TiledKernel kernel, bool parallel, ParallelOptions options)
    {
        var (mr, nr) = CpuTuning.KernelShape(kernel);
        int threads = Math.Max(1, options.MaxDegreeOfParallelism is > 0 and var max ? max : Environment.ProcessorCount);
        var (mc, nc) = CpuTuning.TileGrid(CpuTuning.Machine, m, n, k, mr, nr, parallel ? threads : 1);
        int mTiles = (m + mc - 1) / mc;
        int nTiles = (n + nc - 1) / nc;
        void RunTile(int tile)
        {
            ref float ra = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(a), aOffset);
            ref float rb = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(b), bOffset);
            ref float rc = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(c), cOffset);
            int i0 = tile / nTiles * mc, j0 = tile % nTiles * nc;
            int i1 = Math.Min(m, i0 + mc), j1 = Math.Min(n, j0 + nc);
            int fullRows = (i1 - i0) / mr * mr;
            float[] packedA = ArrayPool<float>.Shared.Rent(Math.Max(1, fullRows * k));
            float[] packedB = ArrayPool<float>.Shared.Rent(k * nr);
            try
            {
                ref float pa = ref MemoryMarshal.GetArrayDataReference(packedA);
                ref float pb = ref MemoryMarshal.GetArrayDataReference(packedB);
                for (int block = 0; block < fullRows / mr; block++)
                {
                    ref float target = ref Unsafe.Add(ref pa, (nint)block * k * mr);
                    nint row0 = (i0 + block * mr) * rowStride;
                    for (int p = 0; p < k; p++)
                    {
                        nint at = row0 + p * colStride;
                        for (int r = 0; r < mr; r++, at += rowStride)
                        {
                            Unsafe.Add(ref target, p * mr + r) = Unsafe.Add(ref ra, at);
                        }
                    }
                }

                int j = j0;
                for (; j + nr <= j1; j += nr)
                {
                    for (int p = 0; p < k; p++)
                    {
                        Unsafe.CopyBlockUnaligned(ref Unsafe.As<float, byte>(ref Unsafe.Add(ref pb, p * nr)),
                            ref Unsafe.As<float, byte>(ref Unsafe.Add(ref rb, (nint)p * n + j)), (uint)(nr * sizeof(float)));
                    }

                    for (int block = 0; block < fullRows / mr; block++)
                    {
                        ref float cBlock = ref Unsafe.Add(ref rc, (nint)(i0 + block * mr) * n + j);
                        ref float aBlock = ref Unsafe.Add(ref pa, (nint)block * k * mr);
                        switch (kernel)
                        {
                            case TiledKernel.Avx512: Kernel8x32(ref aBlock, ref pb, ref cBlock, n, k, beta); break;
                            case TiledKernel.Avx2: Kernel6x16(ref aBlock, ref pb, ref cBlock, n, k, beta); break;
                            default: Kernel8x8(ref aBlock, ref pb, ref cBlock, n, k, beta); break;
                        }
                    }

                    for (int i = i0 + fullRows; i < i1; i++)
                    {
                        Row(ref ra, ref rb, ref rc, i, j, j + nr, k, n, rowStride, colStride, beta);
                    }
                }

                if (j < j1)
                {
                    for (int i = i0; i < i1; i++)
                    {
                        Row(ref ra, ref rb, ref rc, i, j, j1, k, n, rowStride, colStride, beta);
                    }
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(packedA);
                ArrayPool<float>.Shared.Return(packedB);
            }
        }

        int tiles = mTiles * nTiles;
        if (tiles == 1 || !parallel)
        {
            for (int tile = 0; tile < tiles; tile++)
            {
                RunTile(tile);
            }
        }
        else
        {
            Parallel.For(0, tiles, options, RunTile);
        }
    }

    // 8 rows × 32 columns of C in 16 registers across the whole k loop (each element: one FMA chain over k);
    // A packed as [k][8], B as [k][32].
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel8x32(ref float a, ref float b, ref float c, int n, int k, float beta)
    {
        Vector512<float> c00 = default, c01 = default; Vector512<float> c10 = default, c11 = default; Vector512<float> c20 = default, c21 = default; Vector512<float> c30 = default, c31 = default; Vector512<float> c40 = default, c41 = default; Vector512<float> c50 = default, c51 = default; Vector512<float> c60 = default, c61 = default; Vector512<float> c70 = default, c71 = default;
        for (int p = 0; p < k; p++)
        {
            var b0 = Vector512.LoadUnsafe(ref b);
            var b1 = Vector512.LoadUnsafe(ref b, 16);
            var x = Vector512.Create(Unsafe.Add(ref a, 0));
            c00 = Vector512.FusedMultiplyAdd(x, b0, c00);
            c01 = Vector512.FusedMultiplyAdd(x, b1, c01);
            x = Vector512.Create(Unsafe.Add(ref a, 1));
            c10 = Vector512.FusedMultiplyAdd(x, b0, c10);
            c11 = Vector512.FusedMultiplyAdd(x, b1, c11);
            x = Vector512.Create(Unsafe.Add(ref a, 2));
            c20 = Vector512.FusedMultiplyAdd(x, b0, c20);
            c21 = Vector512.FusedMultiplyAdd(x, b1, c21);
            x = Vector512.Create(Unsafe.Add(ref a, 3));
            c30 = Vector512.FusedMultiplyAdd(x, b0, c30);
            c31 = Vector512.FusedMultiplyAdd(x, b1, c31);
            x = Vector512.Create(Unsafe.Add(ref a, 4));
            c40 = Vector512.FusedMultiplyAdd(x, b0, c40);
            c41 = Vector512.FusedMultiplyAdd(x, b1, c41);
            x = Vector512.Create(Unsafe.Add(ref a, 5));
            c50 = Vector512.FusedMultiplyAdd(x, b0, c50);
            c51 = Vector512.FusedMultiplyAdd(x, b1, c51);
            x = Vector512.Create(Unsafe.Add(ref a, 6));
            c60 = Vector512.FusedMultiplyAdd(x, b0, c60);
            c61 = Vector512.FusedMultiplyAdd(x, b1, c61);
            x = Vector512.Create(Unsafe.Add(ref a, 7));
            c70 = Vector512.FusedMultiplyAdd(x, b0, c70);
            c71 = Vector512.FusedMultiplyAdd(x, b1, c71);
            a = ref Unsafe.Add(ref a, 8);
            b = ref Unsafe.Add(ref b, 32);
        }

        ref float row = ref c;
        Store(ref row, c00, beta);
        Store(ref Unsafe.Add(ref row, 16), c01, beta);
        row = ref Unsafe.Add(ref row, n);
        Store(ref row, c10, beta);
        Store(ref Unsafe.Add(ref row, 16), c11, beta);
        row = ref Unsafe.Add(ref row, n);
        Store(ref row, c20, beta);
        Store(ref Unsafe.Add(ref row, 16), c21, beta);
        row = ref Unsafe.Add(ref row, n);
        Store(ref row, c30, beta);
        Store(ref Unsafe.Add(ref row, 16), c31, beta);
        row = ref Unsafe.Add(ref row, n);
        Store(ref row, c40, beta);
        Store(ref Unsafe.Add(ref row, 16), c41, beta);
        row = ref Unsafe.Add(ref row, n);
        Store(ref row, c50, beta);
        Store(ref Unsafe.Add(ref row, 16), c51, beta);
        row = ref Unsafe.Add(ref row, n);
        Store(ref row, c60, beta);
        Store(ref Unsafe.Add(ref row, 16), c61, beta);
        row = ref Unsafe.Add(ref row, n);
        Store(ref row, c70, beta);
        Store(ref Unsafe.Add(ref row, 16), c71, beta);
    }

    // 6 rows × 16 columns of C in 12 registers across the whole k loop (each element: one FMA chain over k);
    // A packed as [k][6], B as [k][16].
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel6x16(ref float a, ref float b, ref float c, int n, int k, float beta)
    {
        Vector256<float> c00 = default, c01 = default; Vector256<float> c10 = default, c11 = default; Vector256<float> c20 = default, c21 = default; Vector256<float> c30 = default, c31 = default; Vector256<float> c40 = default, c41 = default; Vector256<float> c50 = default, c51 = default;
        for (int p = 0; p < k; p++)
        {
            var b0 = Vector256.LoadUnsafe(ref b);
            var b1 = Vector256.LoadUnsafe(ref b, 8);
            var x = Vector256.Create(Unsafe.Add(ref a, 0));
            c00 = Vector256.FusedMultiplyAdd(x, b0, c00);
            c01 = Vector256.FusedMultiplyAdd(x, b1, c01);
            x = Vector256.Create(Unsafe.Add(ref a, 1));
            c10 = Vector256.FusedMultiplyAdd(x, b0, c10);
            c11 = Vector256.FusedMultiplyAdd(x, b1, c11);
            x = Vector256.Create(Unsafe.Add(ref a, 2));
            c20 = Vector256.FusedMultiplyAdd(x, b0, c20);
            c21 = Vector256.FusedMultiplyAdd(x, b1, c21);
            x = Vector256.Create(Unsafe.Add(ref a, 3));
            c30 = Vector256.FusedMultiplyAdd(x, b0, c30);
            c31 = Vector256.FusedMultiplyAdd(x, b1, c31);
            x = Vector256.Create(Unsafe.Add(ref a, 4));
            c40 = Vector256.FusedMultiplyAdd(x, b0, c40);
            c41 = Vector256.FusedMultiplyAdd(x, b1, c41);
            x = Vector256.Create(Unsafe.Add(ref a, 5));
            c50 = Vector256.FusedMultiplyAdd(x, b0, c50);
            c51 = Vector256.FusedMultiplyAdd(x, b1, c51);
            a = ref Unsafe.Add(ref a, 6);
            b = ref Unsafe.Add(ref b, 16);
        }

        ref float row = ref c;
        Store(ref row, c00, beta);
        Store(ref Unsafe.Add(ref row, 8), c01, beta);
        row = ref Unsafe.Add(ref row, n);
        Store(ref row, c10, beta);
        Store(ref Unsafe.Add(ref row, 8), c11, beta);
        row = ref Unsafe.Add(ref row, n);
        Store(ref row, c20, beta);
        Store(ref Unsafe.Add(ref row, 8), c21, beta);
        row = ref Unsafe.Add(ref row, n);
        Store(ref row, c30, beta);
        Store(ref Unsafe.Add(ref row, 8), c31, beta);
        row = ref Unsafe.Add(ref row, n);
        Store(ref row, c40, beta);
        Store(ref Unsafe.Add(ref row, 8), c41, beta);
        row = ref Unsafe.Add(ref row, n);
        Store(ref row, c50, beta);
        Store(ref Unsafe.Add(ref row, 8), c51, beta);
    }

    // 8 rows × 8 columns of C in 16 of the 32 NEON registers across the whole k loop (each element: one FMA chain over k);
    // A packed as [k][8], B as [k][8].
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel8x8(ref float a, ref float b, ref float c, int n, int k, float beta)
    {
        Vector128<float> c00 = default, c01 = default; Vector128<float> c10 = default, c11 = default; Vector128<float> c20 = default, c21 = default; Vector128<float> c30 = default, c31 = default; Vector128<float> c40 = default, c41 = default; Vector128<float> c50 = default, c51 = default; Vector128<float> c60 = default, c61 = default; Vector128<float> c70 = default, c71 = default;
        for (int p = 0; p < k; p++)
        {
            var b0 = Vector128.LoadUnsafe(ref b);
            var b1 = Vector128.LoadUnsafe(ref b, 4);
            var x = Vector128.Create(Unsafe.Add(ref a, 0));
            c00 = Vector128.FusedMultiplyAdd(x, b0, c00);
            c01 = Vector128.FusedMultiplyAdd(x, b1, c01);
            x = Vector128.Create(Unsafe.Add(ref a, 1));
            c10 = Vector128.FusedMultiplyAdd(x, b0, c10);
            c11 = Vector128.FusedMultiplyAdd(x, b1, c11);
            x = Vector128.Create(Unsafe.Add(ref a, 2));
            c20 = Vector128.FusedMultiplyAdd(x, b0, c20);
            c21 = Vector128.FusedMultiplyAdd(x, b1, c21);
            x = Vector128.Create(Unsafe.Add(ref a, 3));
            c30 = Vector128.FusedMultiplyAdd(x, b0, c30);
            c31 = Vector128.FusedMultiplyAdd(x, b1, c31);
            x = Vector128.Create(Unsafe.Add(ref a, 4));
            c40 = Vector128.FusedMultiplyAdd(x, b0, c40);
            c41 = Vector128.FusedMultiplyAdd(x, b1, c41);
            x = Vector128.Create(Unsafe.Add(ref a, 5));
            c50 = Vector128.FusedMultiplyAdd(x, b0, c50);
            c51 = Vector128.FusedMultiplyAdd(x, b1, c51);
            x = Vector128.Create(Unsafe.Add(ref a, 6));
            c60 = Vector128.FusedMultiplyAdd(x, b0, c60);
            c61 = Vector128.FusedMultiplyAdd(x, b1, c61);
            x = Vector128.Create(Unsafe.Add(ref a, 7));
            c70 = Vector128.FusedMultiplyAdd(x, b0, c70);
            c71 = Vector128.FusedMultiplyAdd(x, b1, c71);
            a = ref Unsafe.Add(ref a, 8);
            b = ref Unsafe.Add(ref b, 8);
        }

        ref float row = ref c;
        Store(ref row, c00, beta);
        Store(ref Unsafe.Add(ref row, 4), c01, beta);
        row = ref Unsafe.Add(ref row, n);
        Store(ref row, c10, beta);
        Store(ref Unsafe.Add(ref row, 4), c11, beta);
        row = ref Unsafe.Add(ref row, n);
        Store(ref row, c20, beta);
        Store(ref Unsafe.Add(ref row, 4), c21, beta);
        row = ref Unsafe.Add(ref row, n);
        Store(ref row, c30, beta);
        Store(ref Unsafe.Add(ref row, 4), c31, beta);
        row = ref Unsafe.Add(ref row, n);
        Store(ref row, c40, beta);
        Store(ref Unsafe.Add(ref row, 4), c41, beta);
        row = ref Unsafe.Add(ref row, n);
        Store(ref row, c50, beta);
        Store(ref Unsafe.Add(ref row, 4), c51, beta);
        row = ref Unsafe.Add(ref row, n);
        Store(ref row, c60, beta);
        Store(ref Unsafe.Add(ref row, 4), c61, beta);
        row = ref Unsafe.Add(ref row, n);
        Store(ref row, c70, beta);
        Store(ref Unsafe.Add(ref row, 4), c71, beta);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Store(ref float dst, Vector128<float> value, float beta)
    {
        if (beta != 0f)
        {
            value = Vector128.FusedMultiplyAdd(Vector128.Create(beta), Vector128.LoadUnsafe(ref dst), value);
        }

        value.StoreUnsafe(ref dst);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Store(ref float dst, Vector512<float> value, float beta)
    {
        if (beta != 0f)
        {
            value = Vector512.FusedMultiplyAdd(Vector512.Create(beta), Vector512.LoadUnsafe(ref dst), value);
        }

        value.StoreUnsafe(ref dst);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Store(ref float dst, Vector256<float> value, float beta)
    {
        if (beta != 0f)
        {
            value = Vector256.FusedMultiplyAdd(Vector256.Create(beta), Vector256.LoadUnsafe(ref dst), value);
        }

        value.StoreUnsafe(ref dst);
    }

    // Square tiles of `tile` (CpuTuning.TransposeTile: source and destination tiles within half the L1).
    private static void Transpose(float[] src, int srcOffset, float[] dst, int rows, int cols, int tile, ParallelOptions? options = null)
    {
        int Tile = tile;
        // One band of source columns (destination rows) at a time; the bands write disjoint parts of dst, so large
        // transposes (B of a big product) spread over the threads instead of running on one.
        void Band(int c0)
        {
            int c1 = Math.Min(c0 + Tile, cols);
            for (int r0 = 0; r0 < rows; r0 += Tile)
            {
                int r1 = Math.Min(r0 + Tile, rows);
                for (int r = r0; r < r1; r++)
                {
                    for (int col = c0; col < c1; col++)
                    {
                        dst[col * rows + r] = src[srcOffset + r * cols + col];
                    }
                }
            }
        }

        int bands = (cols + Tile - 1) / Tile;
        if (options is { MaxDegreeOfParallelism: not 1 } && bands > 1 && (long)rows * cols >= 1 << 16)
        {
            Parallel.For(0, bands, options, band => Band(band * Tile));
        }
        else
        {
            for (int band = 0; band < bands; band++)
            {
                Band(band * Tile);
            }
        }
    }

    private static void RunBlock(float[] a, int aOffset, float[] b, int bOffset, float[] c, int cOffset, int block, int m, int n, int k, nint rowStride, nint colStride, float beta)
    {
        ref float ra = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(a), aOffset);
        ref float rb = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(b), bOffset);
        ref float rc = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(c), cOffset);
        int i0 = block * Mr;
        int rows = Math.Min(Mr, m - i0);
        int j = 0;
        if (rows == Mr)
        {
            j = Block4(ref ra, ref rb, ref rc, i0, n, k, rowStride, colStride, beta);
        }

        for (int r = 0; r < rows; r++)
        {
            Row(ref ra, ref rb, ref rc, i0 + r, j, n, k, rowStride, colStride, beta);
        }
    }

    /// <summary>Computes the 4-row block for every full 2-vector column tile; returns the first column not yet written.</summary>
    private static int Block4(ref float a, ref float b, ref float c, int i0, int n, int k, nint rowStride, nint colStride, float beta)
    {
        int w = Vector<float>.Count;
        nint a0 = i0 * rowStride, a1 = a0 + rowStride, a2 = a1 + rowStride, a3 = a2 + rowStride;
        int j = 0;
        for (; j + 2 * w <= n; j += 2 * w)
        {
            Vector<float> c00 = default, c01 = default, c10 = default, c11 = default;
            Vector<float> c20 = default, c21 = default, c30 = default, c31 = default;
            ref float bj = ref Unsafe.Add(ref b, j);
            nint ap = 0;
            for (int p = 0; p < k; p++, ap += colStride)
            {
                ref float brow = ref Unsafe.Add(ref bj, (nint)p * n);
                var b0 = Vector.LoadUnsafe(ref brow);
                var b1 = Vector.LoadUnsafe(ref brow, (nuint)w);

                var x = new Vector<float>(Unsafe.Add(ref a, a0 + ap));
                c00 = Vector.FusedMultiplyAdd(x, b0, c00);
                c01 = Vector.FusedMultiplyAdd(x, b1, c01);
                x = new Vector<float>(Unsafe.Add(ref a, a1 + ap));
                c10 = Vector.FusedMultiplyAdd(x, b0, c10);
                c11 = Vector.FusedMultiplyAdd(x, b1, c11);
                x = new Vector<float>(Unsafe.Add(ref a, a2 + ap));
                c20 = Vector.FusedMultiplyAdd(x, b0, c20);
                c21 = Vector.FusedMultiplyAdd(x, b1, c21);
                x = new Vector<float>(Unsafe.Add(ref a, a3 + ap));
                c30 = Vector.FusedMultiplyAdd(x, b0, c30);
                c31 = Vector.FusedMultiplyAdd(x, b1, c31);
            }

            ref float c0 = ref Unsafe.Add(ref c, (nint)i0 * n + j);
            ref float c1 = ref Unsafe.Add(ref c0, n);
            ref float c2 = ref Unsafe.Add(ref c1, n);
            ref float c3 = ref Unsafe.Add(ref c2, n);
            Store(ref c0, c00, beta);
            Store(ref Unsafe.Add(ref c0, w), c01, beta);
            Store(ref c1, c10, beta);
            Store(ref Unsafe.Add(ref c1, w), c11, beta);
            Store(ref c2, c20, beta);
            Store(ref Unsafe.Add(ref c2, w), c21, beta);
            Store(ref c3, c30, beta);
            Store(ref Unsafe.Add(ref c3, w), c31, beta);
        }

        return j;
    }

    /// <summary>Computes row i of C from column j to the end: one vector at a time, then scalars.</summary>
    private static void Row(ref float a, ref float b, ref float c, int i, int j, int n, int k, nint rowStride, nint colStride, float beta) =>
        Row(ref a, ref b, ref c, i, j, n, k, n, rowStride, colStride, beta);

    /// <summary>Computes row i of C for columns [j, end) (rows of B and C are <paramref name="n"/> long).</summary>
    private static void Row(ref float a, ref float b, ref float c, int i, int j, int end, int k, int n, nint rowStride, nint colStride, float beta)
    {
        int w = Vector<float>.Count;
        nint ai = i * rowStride;
        ref float ci = ref Unsafe.Add(ref c, (nint)i * n);
        for (; j + w <= end; j += w)
        {
            Vector<float> acc = default;
            nint ap = ai;
            for (int p = 0; p < k; p++, ap += colStride)
            {
                acc = Vector.FusedMultiplyAdd(new Vector<float>(Unsafe.Add(ref a, ap)), Vector.LoadUnsafe(ref b, (nuint)((nint)p * n + j)), acc);
            }

            Store(ref Unsafe.Add(ref ci, j), acc, beta);
        }

        for (; j < end; j++)
        {
            float acc = 0f;
            nint ap = ai;
            for (int p = 0; p < k; p++, ap += colStride)
            {
                acc = MathF.FusedMultiplyAdd(Unsafe.Add(ref a, ap), Unsafe.Add(ref b, (nint)p * n + j), acc);
            }

            ref float dst = ref Unsafe.Add(ref ci, j);
            dst = beta == 0f ? acc : MathF.FusedMultiplyAdd(beta, dst, acc);   // as Store's lanes
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Store(ref float dst, Vector<float> value, float beta)
    {
        if (beta != 0f)
        {
            value = Vector.FusedMultiplyAdd(new Vector<float>(beta), Vector.LoadUnsafe(ref dst), value);
        }

        value.StoreUnsafe(ref dst);
    }
}
