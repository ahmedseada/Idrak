// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Idrak.Abstraction.Devices.Cpu;

// float32's own products: c = op(a) · op(b) + β · c, written once over the float lanes. Each output starts at β · c (0
// where β is 0, or the bias of MatMulBias) and takes a fused multiply-add of each term in k's order, so the tiles, the
// threads and the width change its speed alone: tiles of six rows by two vectors of columns over B packed a block of k at
// a time, the width measured per shape (every width that runs here timed alike).
internal sealed unsafe partial class CpuBackend
{
    // Rows a tile takes, and values of k a packed block of B holds.
    private const int TileRows = 6, BlockK = 256;

    public override void BatchedMatMulKernel(Storage a, Storage b, Storage c, int batch, int m, int n, int k, bool transA, bool transB, float beta)
    {
        if (batch <= 0 || m <= 0 || n <= 0)
        {
            return;
        }

        fixed (float* pa = V(a), pb = V(b), pc = V(c))
        {
            Product(pa, pb, null, pc, batch, m, n, k, transA, transB, beta);
        }
    }

    public override bool MatMulBiasKernel(Storage a, Storage b, Storage bias, Storage c, int m, int n, int k)
    {
        if (m <= 0 || n <= 0)
        {
            return true;
        }

        fixed (float* pa = V(a), pb = V(b), pbias = V(bias), pc = V(c))
        {
            Product(pa, pb, pbias, pc, 1, m, n, k, false, false, 0f);
        }

        return true;
    }

    // `lda`: the step between A's rows where it is not transposed (0: k, its rows packed).
    private static void Product(float* a, float* b, float* bias, float* c, int batch, int m, int n, int k, bool transA, bool transB, float beta, long lda = 0)
    {
        var run = new ProductRun(new ProductShape(a, b, bias, c, batch, m, n, k, transA, transB, beta, lda > 0 ? lda : k));
        run.Run(ChosenLanes("MatMul", SizeClass((long)batch * m), SizeClass(n) << 6 | SizeClass(k), ref run));
    }

    private readonly struct ProductShape(float* a, float* b, float* bias, float* c, int batch, int m, int n, int k, bool transA, bool transB, float beta, long lda)
    {
        public readonly float* A = a, B = b, Bias = bias, C = c;
        public readonly int Batch = batch, M = m, N = n, K = k;
        public readonly bool TransA = transA, TransB = transB;
        public readonly float Beta = beta;
        public readonly long Lda = lda;

        public ProductShape WithC(float* to) => new(A, B, Bias, to, Batch, M, N, K, TransA, TransB, 0f, Lda);
    }

    private struct ProductRun(ProductShape shape) : ILanesRun
    {
        private ProductShape _shape = shape;

        public readonly void Run(Lanes lanes)
        {
            switch (lanes)
            {
                case Lanes.Wide512:
                    Products<Vector512<float>, F512>(_shape);
                    break;
                case Lanes.Wide256:
                    Products<Vector256<float>, F256>(_shape);
                    break;
                case Lanes.Wide128:
                    Products<Vector128<float>, F128>(_shape);
                    break;
                default:
                    Products<float, F1>(_shape);
                    break;
            }
        }

        public void Redirect(ref Scratch scratch) => _shape = _shape.WithC(scratch.Take((long)_shape.Batch * _shape.M * _shape.N));
    }

    // The product's tiles over the threads: each item a matrix of the batch, a panel of two vectors of columns and a group
    // of rows, its B packed a block of k at a time.
    private static void Products<TV, TF>(in ProductShape s) where TV : unmanaged where TF : struct, IFloats<TV>
    {
        int panel = 2 * TF.Count;
        int panels = (s.N + panel - 1) / panel;
        long work = (long)s.Batch * s.M * s.N * Math.Max(1, s.K);
        int threads = ComputeResources.MaxCpuThreads;
        int rowGroups = 1;
        if (threads > 1 && work >= ParallelThreshold)
        {
            // Enough items for every thread (a few each), rows grouped by whole tiles.
            long items = (long)s.Batch * panels;
            int wanted = (int)Math.Clamp((threads * 4 + items - 1) / Math.Max(1, items), 1, (s.M + TileRows - 1) / TileRows);
            rowGroups = wanted;
        }

        int rowsPerGroup = ((s.M + rowGroups - 1) / rowGroups + TileRows - 1) / TileRows * TileRows;
        rowGroups = (s.M + rowsPerGroup - 1) / rowsPerGroup;
        var body = new ProductBody<TV, TF> { Shape = s, Panels = panels, RowGroups = rowGroups, RowsPerGroup = rowsPerGroup };
        int count = s.Batch * panels * rowGroups;
        if (threads <= 1 || work < ParallelThreshold || count == 1)
        {
            body.Run(0, count);
            return;
        }

        CpuRanges.Run(count, 1, count, threads, ref body);
    }

    private struct ProductBody<TV, TF> : IRangeBody where TV : unmanaged where TF : struct, IFloats<TV>
    {
        public ProductShape Shape;
        public int Panels, RowGroups, RowsPerGroup;

        public readonly void Run(int first, int last)
        {
            int panel = 2 * TF.Count;
            float* packed = stackalloc float[BlockK * 2 * 16];
            float* tile = stackalloc float[TileRows * 2 * 16];
            var s = Shape;
            for (int item = first; item < last; item++)
            {
                int group = item % RowGroups, rest = item / RowGroups, p = rest % Panels, matrix = rest / Panels;
                int j0 = p * panel, columns = Math.Min(panel, s.N - j0);
                int i0 = group * RowsPerGroup, i1 = Math.Min(s.M, i0 + RowsPerGroup);
                float* a = s.A + (long)matrix * s.M * s.K, b = s.B + (long)matrix * s.K * s.N, c = s.C + (long)matrix * s.M * s.N;
                Start(s, c, i0, i1, j0, columns);
                for (int k0 = 0; k0 < s.K; k0 += BlockK)
                {
                    int kc = Math.Min(BlockK, s.K - k0);
                    Pack(b, s.N, s.K, s.TransB, k0, kc, j0, columns, panel, packed);
                    for (int i = i0; i < i1; i += TileRows)
                    {
                        int rows = Math.Min(TileRows, i1 - i);
                        bool whole = rows == TileRows && columns == panel;
                        float* target = whole ? c + (long)i * s.N + j0 : tile;
                        nint ldc = whole ? s.N : panel;
                        if (!whole)
                        {
                            new Span<float>(tile, TileRows * panel).Clear();
                            for (int r = 0; r < rows; r++)
                            {
                                new ReadOnlySpan<float>(c + (long)(i + r) * s.N + j0, columns).CopyTo(new Span<float>(tile + r * panel, panel));
                            }
                        }

                        // A's six rows at k0 (rows past the tile's read its first, their sums dropped) and the step between k's.
                        nint step = s.TransA ? s.M : 1;
                        float* r0 = Row(a, s, i, k0), r1 = Row(a, s, rows > 1 ? i + 1 : i, k0), r2 = Row(a, s, rows > 2 ? i + 2 : i, k0);
                        float* r3 = Row(a, s, rows > 3 ? i + 3 : i, k0), r4 = Row(a, s, rows > 4 ? i + 4 : i, k0), r5 = Row(a, s, rows > 5 ? i + 5 : i, k0);
                        Tile<TV, TF>(r0, r1, r2, r3, r4, r5, step, packed, kc, target, ldc);
                        if (!whole)
                        {
                            for (int r = 0; r < rows; r++)
                            {
                                new ReadOnlySpan<float>(tile + r * panel, columns).CopyTo(new Span<float>(c + (long)(i + r) * s.N + j0, columns));
                            }
                        }
                    }
                }
            }
        }

        // Row i of op(a) at k0.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float* Row(float* a, in ProductShape s, int i, int k0) => s.TransA ? a + (long)k0 * s.M + i : a + i * s.Lda + k0;
    }

    // Each output's start: β · c (0 where β is 0), or the bias.
    private static void Start(in ProductShape s, float* c, int i0, int i1, int j0, int columns)
    {
        for (int i = i0; i < i1; i++)
        {
            var row = new Span<float>(c + (long)i * s.N + j0, columns);
            if (s.Bias != null)
            {
                new ReadOnlySpan<float>(s.Bias + j0, columns).CopyTo(row);
            }
            else if (s.Beta == 0f)
            {
                row.Clear();
            }
            else if (s.Beta != 1f)
            {
                for (int j = 0; j < columns; j++)
                {
                    row[j] *= s.Beta;
                }
            }
        }
    }

    // B's values [k0, k0 + kc) × [j0, j0 + columns) packed k by k, `panel` a k (zeros past the columns).
    private static void Pack(float* b, int n, int k, bool transB, int k0, int kc, int j0, int columns, int panel, float* packed)
    {
        for (int kk = 0; kk < kc; kk++)
        {
            float* to = packed + kk * panel;
            if (transB)
            {
                for (int j = 0; j < columns; j++)
                {
                    to[j] = b[(long)(j0 + j) * k + k0 + kk];
                }
            }
            else
            {
                new ReadOnlySpan<float>(b + (long)(k0 + kk) * n + j0, columns).CopyTo(new Span<float>(to, columns));
            }

            for (int j = columns; j < panel; j++)
            {
                to[j] = 0f;
            }
        }
    }

    // Six rows by two vectors of columns: c += a · b, each term fused in k's order.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Tile<TV, TF>(float* a0, float* a1, float* a2, float* a3, float* a4, float* a5, nint step, float* packed, int kc, float* c, nint ldc)
        where TV : unmanaged where TF : struct, IFloats<TV>
    {
        int w = TF.Count;
        TV c00 = TF.Load(c), c01 = TF.Load(c + w), c10 = TF.Load(c + ldc), c11 = TF.Load(c + ldc + w), c20 = TF.Load(c + 2 * ldc), c21 = TF.Load(c + 2 * ldc + w);
        TV c30 = TF.Load(c + 3 * ldc), c31 = TF.Load(c + 3 * ldc + w), c40 = TF.Load(c + 4 * ldc), c41 = TF.Load(c + 4 * ldc + w);
        TV c50 = TF.Load(c + 5 * ldc), c51 = TF.Load(c + 5 * ldc + w);
        nint at = 0;
        for (int kk = 0; kk < kc; kk++, at += step)
        {
            float* p = packed + kk * 2 * w;
            TV b0 = TF.Load(p), b1 = TF.Load(p + w);
            TV x = TF.All(a0[at]);
            c00 = TF.Fma(x, b0, c00);
            c01 = TF.Fma(x, b1, c01);
            x = TF.All(a1[at]);
            c10 = TF.Fma(x, b0, c10);
            c11 = TF.Fma(x, b1, c11);
            x = TF.All(a2[at]);
            c20 = TF.Fma(x, b0, c20);
            c21 = TF.Fma(x, b1, c21);
            x = TF.All(a3[at]);
            c30 = TF.Fma(x, b0, c30);
            c31 = TF.Fma(x, b1, c31);
            x = TF.All(a4[at]);
            c40 = TF.Fma(x, b0, c40);
            c41 = TF.Fma(x, b1, c41);
            x = TF.All(a5[at]);
            c50 = TF.Fma(x, b0, c50);
            c51 = TF.Fma(x, b1, c51);
        }

        TF.Store(c, c00);
        TF.Store(c + w, c01);
        TF.Store(c + ldc, c10);
        TF.Store(c + ldc + w, c11);
        TF.Store(c + 2 * ldc, c20);
        TF.Store(c + 2 * ldc + w, c21);
        TF.Store(c + 3 * ldc, c30);
        TF.Store(c + 3 * ldc + w, c31);
        TF.Store(c + 4 * ldc, c40);
        TF.Store(c + 4 * ldc + w, c41);
        TF.Store(c + 5 * ldc, c50);
        TF.Store(c + 5 * ldc + w, c51);
    }
}
