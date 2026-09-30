using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Idrak.Backends.Cpu;

/// <summary>
/// Single-precision GEMM: C = op(A) * op(B) + beta * C, all row-major.
/// Rows of C are processed in blocks of <see cref="Mr"/>; each block keeps a
/// <see cref="Mr"/> x (2 * SIMD width) tile of C in registers across the whole k loop,
/// so every B load feeds <see cref="Mr"/> fused multiply-adds.
/// </summary>
internal static class CpuMatMul
{
    private const int Mr = 4;

    /// <summary>m * n * k above which row blocks run in parallel.</summary>
    private const long ParallelWork = 1L << 17;

    public static void Multiply(float[] a, float[] b, float[] c, int m, int n, int k, bool transA, bool transB, float beta) =>
        Multiply(a, 0, b, 0, c, 0, m, n, k, transA, transB, beta);

    /// <summary>Multiplies the matrices starting at the given element offsets of each array.</summary>
    public static void Multiply(float[] a, int aOffset, float[] b, int bOffset, float[] c, int cOffset, int m, int n, int k, bool transA, bool transB, float beta)
    {
        if (transB && m <= NtRows)
        {
            // Few rows against B stored [n, k] (decoding through a tied head, backward products of one token): every
            // output is a dot product of an A row with a contiguous B row, so B is read once in place, not transposed.
            FewRowsTransposedB(a, aOffset, b, bOffset, c, cOffset, m, n, k, transA, beta);
            return;
        }

        float[]? rented = null;
        if (transB)
        {
            // B is stored [n, k]; transpose it once into [k, n] so the kernel always streams contiguous B rows.
            rented = ArrayPool<float>.Shared.Rent(k * n);
            Transpose(b, bOffset, rented, n, k);
            b = rented;
            bOffset = 0;
        }

        // Element (i, p) of op(A) lives at a[i * rowStride + p * colStride].
        nint rowStride = transA ? 1 : k;
        nint colStride = transA ? m : 1;
        int blocks = (m + Mr - 1) / Mr;

        if (m <= SmallRows && n >= 2 * SmallColumns && (long)m * n * k >= ParallelWork && ComputeResources.AllowParallel)
        {
            // Few rows (token-by-token decoding): row blocks would leave most threads idle, so split the columns instead
            // and stream B row by row (contiguous reads; each weight is read once).
            FewRows(a, aOffset, b, bOffset, c, cOffset, m, n, k, rowStride, colStride, beta);
            if (rented is not null)
            {
                ArrayPool<float>.Shared.Return(rented);
            }

            return;
        }

        if (Fma.IsSupported && (Vector512.IsHardwareAccelerated || Vector256.IsHardwareAccelerated) && m >= 6)
        {
            Tiled(a, aOffset, b, bOffset, c, cOffset, m, n, k, rowStride, colStride, beta);
        }
        else if ((long)m * n * k < ParallelWork || blocks == 1 || !ComputeResources.AllowParallel)
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
            Parallel.For(0, blocks, ComputeResources.ParallelOptions, block => RunBlock(a, aOffset, bb, bo, c, cOffset, block, m, n, k, rowStride, colStride, beta));
        }

        if (rented is not null)
        {
            ArrayPool<float>.Shared.Return(rented);
        }
    }

    private const int SmallRows = 4;
    private const int SmallColumns = 64;

    /// <summary>Rows of op(A) up to which a product with a transposed B reads B in place instead of transposing it.</summary>
    private const int NtRows = 16;

    private static void FewRowsTransposedB(float[] a, int aOffset, float[] b, int bOffset, float[] c, int cOffset, int m, int n, int k, bool transA, float beta)
    {
        // The A rows contiguous ([m, k]); a transposed A ([k, m]) is gathered once (m·k values, small here).
        float[]? rows = null;
        float[] ar = a;
        int ao = aOffset;
        if (transA)
        {
            rows = ArrayPool<float>.Shared.Rent(m * k);
            for (int i = 0; i < m; i++)
            {
                for (int p = 0; p < k; p++)
                {
                    rows[i * k + p] = a[aOffset + p * m + i];
                }
            }

            ar = rows;
            ao = 0;
        }

        const int Chunk = 64;
        int chunks = (n + Chunk - 1) / Chunk;
        if ((long)m * n * k >= ParallelWork && chunks > 1 && ComputeResources.AllowParallel)
        {
            Parallel.For(0, chunks, ComputeResources.ParallelOptions, index => DotColumns(ar, ao, b, bOffset, c, cOffset, m, n, k, beta, index * Chunk, Math.Min(n, (index + 1) * Chunk)));
        }
        else
        {
            DotColumns(ar, ao, b, bOffset, c, cOffset, m, n, k, beta, 0, n);
        }

        if (rows is not null)
        {
            ArrayPool<float>.Shared.Return(rows);
        }
    }

    // C[i, j] = A[i, :] · B[j, :] (+ beta · C[i, j]) for the columns j0..j1; each B row is read once for all m rows.
    private static void DotColumns(float[] a, int aOffset, float[] b, int bOffset, float[] c, int cOffset, int m, int n, int k, float beta, int j0, int j1)
    {
        int w = Vector<float>.Count;
        ref float ra = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(a), aOffset);
        ref float rb = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(b), bOffset);
        for (int j = j0; j < j1; j++)
        {
            ref float row = ref Unsafe.Add(ref rb, (nint)j * k);
            for (int i = 0; i < m; i++)
            {
                ref float x = ref Unsafe.Add(ref ra, (nint)i * k);
                var s0 = Vector<float>.Zero;
                var s1 = Vector<float>.Zero;
                int p = 0;
                for (; p <= k - 2 * w; p += 2 * w)
                {
                    s0 = Vector.FusedMultiplyAdd(Vector.LoadUnsafe(ref x, (nuint)p), Vector.LoadUnsafe(ref row, (nuint)p), s0);
                    s1 = Vector.FusedMultiplyAdd(Vector.LoadUnsafe(ref x, (nuint)(p + w)), Vector.LoadUnsafe(ref row, (nuint)(p + w)), s1);
                }

                for (; p <= k - w; p += w)
                {
                    s0 = Vector.FusedMultiplyAdd(Vector.LoadUnsafe(ref x, (nuint)p), Vector.LoadUnsafe(ref row, (nuint)p), s0);
                }

                float sum = Vector.Sum(s0 + s1);
                for (; p < k; p++)
                {
                    sum = MathF.FusedMultiplyAdd(Unsafe.Add(ref x, p), Unsafe.Add(ref row, p), sum);
                }

                ref float dst = ref c[cOffset + i * n + j];
                dst = beta == 0f ? sum : sum + beta * dst;
            }
        }
    }

    private static void FewRows(float[] a, int aOffset, float[] b, int bOffset, float[] c, int cOffset, int m, int n, int k, nint rowStride, nint colStride, float beta)
    {
        int w = Vector<float>.Count;
        int threads = Math.Max(1, ComputeResources.ParallelOptions.MaxDegreeOfParallelism is > 0 and var max ? max : Environment.ProcessorCount);
        int chunk = Math.Max(SmallColumns, (n / (2 * threads) + w - 1) / w * w);   // about two chunks per thread, whole vectors
        int chunks = (n + chunk - 1) / chunk;
        Parallel.For(0, chunks, ComputeResources.ParallelOptions, index =>
        {
            int j0 = index * chunk, width = Math.Min(chunk, n - j0);
            float[] acc = ArrayPool<float>.Shared.Rent(m * width);
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
                    dst[j] = beta == 0f ? sums[j] : sums[j] + beta * dst[j];
                }
            }

            ArrayPool<float>.Shared.Return(acc);
        });
    }

    // x86 with AVX-512 or AVX2: C in tiles of mc rows × nc columns (spread over the threads). A tile packs its A rows
    // once ([k][mr] per row block) and each nr-column panel of B ([k][nr], contiguous: no cache-set conflicts from
    // power-of-two row strides), which every row block then reuses from L2. Every element is still one FMA chain over k
    // from zero, then + beta · C, as in the other paths.
    private static void Tiled(float[] a, int aOffset, float[] b, int bOffset, float[] c, int cOffset, int m, int n, int k, nint rowStride, nint colStride, float beta)
    {
        bool wide = Vector512.IsHardwareAccelerated;
        int mr = wide ? 8 : 6, nr = wide ? 32 : 16;
        int mc = Math.Clamp((256 * 1024 / Math.Max(1, k * sizeof(float))) / mr, 1, 16) * mr;   // A tile about 256 KB
        int mTiles = (m + mc - 1) / mc;
        int threads = Math.Max(1, ComputeResources.ParallelOptions.MaxDegreeOfParallelism is > 0 and var max ? max : Environment.ProcessorCount);
        int nSplit = Math.Max(1, (2 * threads + mTiles - 1) / mTiles);
        int nc = Math.Max(2 * nr, ((n + nSplit - 1) / nSplit + nr - 1) / nr * nr);
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
                    if (wide)
                    {
                        Kernel8x32(ref aBlock, ref pb, ref cBlock, n, k, beta);
                    }
                    else
                    {
                        Kernel6x16(ref aBlock, ref pb, ref cBlock, n, k, beta);
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

            ArrayPool<float>.Shared.Return(packedA);
            ArrayPool<float>.Shared.Return(packedB);
        }

        int tiles = mTiles * nTiles;
        if (tiles == 1 || (long)m * n * k < ParallelWork || !ComputeResources.AllowParallel)
        {
            for (int tile = 0; tile < tiles; tile++)
            {
                RunTile(tile);
            }
        }
        else
        {
            Parallel.For(0, tiles, ComputeResources.ParallelOptions, RunTile);
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

    private static void Transpose(float[] src, int srcOffset, float[] dst, int rows, int cols)
    {
        const int Tile = 32;
        for (int r0 = 0; r0 < rows; r0 += Tile)
        {
            int r1 = Math.Min(r0 + Tile, rows);
            for (int c0 = 0; c0 < cols; c0 += Tile)
            {
                int c1 = Math.Min(c0 + Tile, cols);
                for (int r = r0; r < r1; r++)
                {
                    for (int col = c0; col < c1; col++)
                    {
                        dst[col * rows + r] = src[srcOffset + r * cols + col];
                    }
                }
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
            dst = beta == 0f ? acc : acc + beta * dst;
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
