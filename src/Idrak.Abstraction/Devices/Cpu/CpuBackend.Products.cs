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

        if (Wide64(a, b, c))
        {
            Product64(a, b, c, batch, m, n, k, transA, transB, beta);
            return;
        }

        fixed (float* wa = D(a), wb = D(b), pc = V(c))
        {
            // A typed B is read through its type as it is packed (BType); a typed A a block of its rows at a time.
            var bType = NativeFloats(b) ? null : TypedB(b, "MatMul");
            if (NativeFloats(a))
            {
                Product(wa, wb, null, pc, batch, m, n, k, transA, transB, beta, bType: bType);
                return;
            }

            if (transA)
            {
                throw new NotSupportedException($"The CPU's MatMul reads a transposed A of {a.Type.Name} as float32's own only: read it through its type a block of rows at a time untransposed.");
            }

            int step = Math.Max(TileRows, ChunkValues / Math.Max(1, k) / TileRows * TileRows);
            var ta = new TypedInput(a, wa, 0, (long)Math.Min(step, m) * k, "MatMul");
            for (int matrix = 0; matrix < batch; matrix++)
            {
                for (int i = 0; i < m; i += step)
                {
                    int rows = Math.Min(step, m - i);
                    long at = ((long)matrix * m + i) * k;
                    float* rowsA = ta.Range(at, (long)rows * k) + at;
                    float* bAt = bType is null ? wb + (long)matrix * k * n : wb;
                    Product(rowsA, bAt, null, pc + ((long)matrix * m + i) * n, 1, rows, n, k, false, transB, beta, bType: bType, bBase: bType is null ? 0 : (long)matrix * k * n);
                }
            }
        }
    }

    // A typed B's type, refused where its values need float64 (never narrowed).
    private static IElementType TypedB(Storage b, string operation) => WideValues(b.Type)
        ? throw new NotSupportedException($"The CPU's {operation} reads float32 values; {b.Type.Name}'s values need float64 (native kind {b.Type.Native}), which it never narrows: its float64 lane is not built.")
        : b.Type;

    public override bool MatMulBiasKernel(Storage a, Storage b, Storage bias, Storage c, int m, int n, int k)
    {
        if (m <= 0 || n <= 0)
        {
            return true;
        }

        fixed (float* wa = D(a), wb = D(b), wbias = D(bias), pc = V(c))
        {
            var tbias = new TypedInput(bias, wbias, 0, n, "MatMulBias");
            float* pbias = tbias.Range(0, n);
            var bType = NativeFloats(b) ? null : TypedB(b, "MatMulBias");
            // A typed A read through its type a block of rows at a time (each output row its own sum, from the bias).
            int step = NativeFloats(a) ? m : Math.Max(TileRows, ChunkValues / Math.Max(1, k) / TileRows * TileRows);
            var ta = new TypedInput(a, wa, 1, (long)Math.Min(step, m) * k, "MatMulBias");
            for (int i = 0; i < m; i += step)
            {
                int rows = Math.Min(step, m - i);
                Product(ta.Range((long)i * k, (long)rows * k) + (long)i * k, wb, pbias, pc + (long)i * n, 1, rows, n, k, false, false, 0f, bType: bType);
            }
        }

        return true;
    }

    // `lda`: the step between A's rows where it is not transposed (0: k, its rows packed). `bType`: B's type where its words
    // hold another type's values (read through it as B is packed, from element `bBase` on), else null.
    // A view's planes (aStride, bStride: the step between matrices, 0 for one shared, negative where packed; lda, ldb:
    // between stored rows, 0 or less where packed) are read where they lie.
    private static void Product(float* a, float* b, float* bias, float* c, int batch, int m, int n, int k, bool transA, bool transB, float beta, long lda = 0,
        IElementType? bType = null, long bBase = 0, long aStride = -1, long bStride = -1, long ldb = 0)
    {
        var shape = new ProductShape(a, b, bias, c, batch, m, n, k, transA, transB, beta, lda > 0 ? lda : transA ? m : k, bType, bBase,
            aStride >= 0 ? aStride : (long)m * k, bStride >= 0 ? bStride : (long)k * n, ldb > 0 ? ldb : transB ? k : n);
        var run = new ProductRun(shape);
        run.Run(ChosenLanes(bType is null ? "MatMul" : "MatMul typed B", SizeClass((long)batch * m), SizeClass(n) << 6 | SizeClass(k), ref run));
    }

    private readonly struct ProductShape(float* a, float* b, float* bias, float* c, int batch, int m, int n, int k, bool transA, bool transB, float beta, long lda,
        IElementType? bType, long bBase, long aStride, long bStride, long ldb)
    {
        public readonly float* A = a, B = b, Bias = bias, C = c;
        public readonly int Batch = batch, M = m, N = n, K = k;
        public readonly bool TransA = transA, TransB = transB;
        public readonly float Beta = beta;
        public readonly long Lda = lda, BBase = bBase, AStride = aStride, BStride = bStride, Ldb = ldb;
        public readonly IElementType? BType = bType;

        public ProductShape WithC(float* to) => new(A, B, Bias, to, Batch, M, N, K, TransA, TransB, 0f, Lda, BType, BBase, AStride, BStride, Ldb);
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
    // of rows, its B packed a block of k at a time; a few rows by a B as it is stored ([k, n], float32) by StreamProducts.
    private static void Products<TV, TF>(in ProductShape s) where TV : unmanaged where TF : struct, IFloats<TV>
    {
        if (s.M <= CpuTuning.FewRows && !s.TransB && s.BType is null)
        {
            StreamProducts<TV, TF>(s);
            return;
        }

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

            // A typed B's values read this thread's own way: a run of a row (or a column's k values) through the type, from
            // the unit holding its first.
            float* temp = null;
            if (s.BType is { } bt)
            {
                int unit = Math.Max(1, bt.Description.UnitValues);
                float* room = stackalloc float[2 * (unit + BlockK + 32)];
                temp = room;
            }

            for (int item = first; item < last; item++)
            {
                int group = item % RowGroups, rest = item / RowGroups, p = rest % Panels, matrix = rest / Panels;
                int j0 = p * panel, columns = Math.Min(panel, s.N - j0);
                int i0 = group * RowsPerGroup, i1 = Math.Min(s.M, i0 + RowsPerGroup);
                float* a = s.A + matrix * s.AStride, b = s.B + matrix * s.BStride, c = s.C + (long)matrix * s.M * s.N;
                Start(s, c, i0, i1, j0, columns);
                for (int k0 = 0; k0 < s.K; k0 += BlockK)
                {
                    int kc = Math.Min(BlockK, s.K - k0);
                    if (s.BType is { } type)
                    {
                        PackTyped(type, s.B, s.BBase + matrix * s.BStride, s.Ldb, s.TransB, k0, kc, j0, columns, panel, packed, temp);
                    }
                    else
                    {
                        Pack(b, s.Ldb, s.TransB, k0, kc, j0, columns, panel, packed);
                    }
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
                        nint step = s.TransA ? (nint)s.Lda : 1;
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
        private static float* Row(float* a, in ProductShape s, int i, int k0) => s.TransA ? a + k0 * s.Lda + i : a + i * s.Lda + k0;
    }

    // A few rows of op(a) by a B stored [k, n] of float32, read where it lies (a GGUF float32 weight as it loads): each item
    // a matrix and a run of whole cache lines of columns, its outputs started (Start) and then B's rows read once each, in
    // k's order, four at a time, every row of a fused into its outputs (each term fused in k's order, as Tile's: the same
    // bits, without B packed a block at a time for each panel of 16 columns and six rows of tile computed for one). A run's
    // outputs stay in L1 across k: as many columns as a quarter of it holds for every row, fewer to give every thread one
    // (each B row's run then the longest the threads allow: measured, Qwen3-1.7B F32 tg32 10.0 tok/s at a run a thread,
    // 9.6 at two, 9.4 at four, 8.5 at eight).
    private static void StreamProducts<TV, TF>(in ProductShape s) where TV : unmanaged where TF : struct, IFloats<TV>
    {
        int threads = ComputeResources.MaxCpuThreads;
        long work = (long)s.Batch * s.M * s.N * Math.Max(1, s.K);
        int most = Math.Max(16, CpuTuning.Machine.L1 / 4 / (sizeof(float) * s.M) / 16 * 16);
        int width = (int)Math.Clamp(((s.N + threads - 1) / threads + 15) / 16 * 16, 16, most);
        int runs = (s.N + width - 1) / width;
        var body = new StreamBody<TV, TF> { Shape = s, Width = width, Runs = runs };
        int count = s.Batch * runs;
        if (threads <= 1 || work < ParallelThreshold || count == 1)
        {
            body.Run(0, count);
            return;
        }

        CpuRanges.Run(count, 1, count, threads, ref body);
    }

    private struct StreamBody<TV, TF> : IRangeBody where TV : unmanaged where TF : struct, IFloats<TV>
    {
        public ProductShape Shape;
        public int Width, Runs;

        [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
        public readonly void Run(int first, int last)
        {
            var s = Shape;
            int w = TF.Count;
            for (int item = first; item < last; item++)
            {
                int matrix = item / Runs, j0 = item % Runs * Width, columns = Math.Min(Width, s.N - j0), whole = columns / w * w;
                float* a = s.A + matrix * s.AStride, b = s.B + matrix * s.BStride + j0, c = s.C + (long)matrix * s.M * s.N + j0;
                Start(s, c - j0, 0, s.M, j0, columns);
                int p = 0;
                for (; p + 4 <= s.K; p += 4)
                {
                    float* b0 = b + p * s.Ldb, b1 = b0 + s.Ldb, b2 = b1 + s.Ldb, b3 = b2 + s.Ldb;
                    for (int i = 0; i < s.M; i++)
                    {
                        float x0 = A(a, s, i, p), x1 = A(a, s, i, p + 1), x2 = A(a, s, i, p + 2), x3 = A(a, s, i, p + 3);
                        TV v0 = TF.All(x0), v1 = TF.All(x1), v2 = TF.All(x2), v3 = TF.All(x3);
                        float* ci = c + (long)i * s.N;
                        int j = 0;
                        for (; j < whole; j += w)
                        {
                            var y = TF.Fma(v0, TF.Load(b0 + j), TF.Load(ci + j));
                            y = TF.Fma(v1, TF.Load(b1 + j), y);
                            y = TF.Fma(v2, TF.Load(b2 + j), y);
                            TF.Store(ci + j, TF.Fma(v3, TF.Load(b3 + j), y));
                        }

                        for (; j < columns; j++)
                        {
                            float y = F1.Fma(x0, b0[j], ci[j]);
                            y = F1.Fma(x1, b1[j], y);
                            y = F1.Fma(x2, b2[j], y);
                            ci[j] = F1.Fma(x3, b3[j], y);
                        }
                    }
                }

                for (; p < s.K; p++)
                {
                    float* bp = b + p * s.Ldb;
                    for (int i = 0; i < s.M; i++)
                    {
                        float x = A(a, s, i, p);
                        TV v = TF.All(x);
                        float* ci = c + (long)i * s.N;
                        int j = 0;
                        for (; j < whole; j += w)
                        {
                            TF.Store(ci + j, TF.Fma(v, TF.Load(bp + j), TF.Load(ci + j)));
                        }

                        for (; j < columns; j++)
                        {
                            ci[j] = F1.Fma(x, bp[j], ci[j]);
                        }
                    }
                }
            }
        }

        // op(a)'s value at row i, k p.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float A(float* a, in ProductShape s, int i, int p) => s.TransA ? a[(long)p * s.Lda + i] : a[(long)i * s.Lda + p];
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

    // B's values [k0, k0 + kc) × [j0, j0 + columns) packed k by k, `panel` a k (zeros past the columns); `ldb` the step
    // between B's stored rows (k's rows, or n's where transposed).
    private static void Pack(float* b, long ldb, bool transB, int k0, int kc, int j0, int columns, int panel, float* packed)
    {
        for (int kk = 0; kk < kc; kk++)
        {
            float* to = packed + kk * panel;
            if (transB)
            {
                for (int j = 0; j < columns; j++)
                {
                    to[j] = b[(j0 + j) * ldb + k0 + kk];
                }
            }
            else
            {
                new ReadOnlySpan<float>(b + (k0 + kk) * ldb + j0, columns).CopyTo(new Span<float>(to, columns));
            }

            for (int j = columns; j < panel; j++)
            {
                to[j] = 0f;
            }
        }
    }

    // Pack for a B of another type: each row's run of columns (or, transposed, each column's run of k) read through the
    // type into the packed block (zeros past the columns), the same layout as Pack's.
    private static void PackTyped(IElementType type, float* words, long matrix, long ldb, bool transB, int k0, int kc, int j0, int columns, int panel, float* packed,
        float* temp)
    {
        if (transB)
        {
            float* column = temp + Math.Max(1, type.Description.UnitValues) + BlockK + 32;
            for (int j = 0; j < columns; j++)
            {
                ReadRange(type, words, matrix + (j0 + j) * ldb + k0, kc, column, temp);
                for (int kk = 0; kk < kc; kk++)
                {
                    packed[kk * panel + j] = column[kk];
                }
            }
        }
        else
        {
            for (int kk = 0; kk < kc; kk++)
            {
                ReadRange(type, words, matrix + (k0 + kk) * ldb + j0, columns, packed + kk * panel, temp);
            }
        }

        for (int kk = 0; kk < kc; kk++)
        {
            for (int j = columns; j < panel; j++)
            {
                packed[kk * panel + j] = 0f;
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
