// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using Idrak.Gpu.Hip;
using Idrak.Tests.Gcn;

// Plan 35, training order: the training sums grouped as CUDA's kernels of the same names group them (CUDA's BlockReduce,
// RowWriter.ReduceAdd; CUDA's halving tree, RowWriter.Reduce; a warp a row; a thread a row), each checked bit for bit
// against a C# model of CUDA's order (CudaBlockSum, GcnRowOrderTests.cs; CudaTreeSum below) whose divisions and square
// roots are correctly rounded, as the kernels' are (LLVM's expansions).
internal abstract partial class Tests
{
    // CUDA's halving tree over a block's partials (PtxKernels.Advanced.cs's BlockReduce): [t] += [t + s] for s = 128, …, 1.
    private static T CudaTree<T>(T[] partials, Func<T, T, T> combine)
    {
        var values = (T[])partials.Clone();
        for (int stride = values.Length / 2; stride > 0; stride /= 2)
        {
            for (int t = 0; t < stride; t++)
            {
                values[t] = combine(values[t], values[t + stride]);
            }
        }

        return values[0];
    }

    private static float CudaTreeSum(float[] partials) => CudaTree(partials, (a, b) => a + b);

    // Each of a block's 256 threads folds the indices t, t + 256, … below `count` into its sum from 0, in order.
    private static float[] CudaThreadSums(int count, Func<float, int, float> step)
    {
        var partials = new float[GcnKernels.RowThreads];
        for (int t = 0; t < partials.Length; t++)
        {
            float acc = 0f;
            for (int i = t; i < count; i += GcnKernels.RowThreads)
            {
                acc = step(acc, i);
            }

            partials[t] = acc;
        }

        return partials;
    }

    // A row's sum by CUDA's sum_axis_rows_f32: lane l adds the columns l, l + 32, … from 0, then the 32 lanes' butterfly.
    private static float CudaWarpRowSum(ReadOnlySpan<float> row)
    {
        var lanes = new float[GcnKernels.SumAxisRowLanes];
        for (int l = 0; l < lanes.Length; l++)
        {
            for (int c = l; c < row.Length; c += lanes.Length)
            {
                lanes[l] += row[c];
            }
        }

        for (int offset = lanes.Length / 2; offset >= 1; offset /= 2)
        {
            for (int l = 0; l < offset; l++)
            {
                lanes[l] += lanes[l + offset];
            }
        }

        return lanes[0];
    }

    // sum_axis_rows_f32, sum_rows_f32, softmax_bwd_f32, rms_norm_backward_f32, weight_norm_f32, weight_norm_bwd_f32's dgain
    // and attn_bwd_d_f32 against C# models of their CUDA kernels' order.
    internal static void GcnTrainingRowsCudaOrder(Device device)
    {
        _ = device;
        var random = new Random(85);
        float log2E = BitConverter.UInt32BitsToSingle(0x3fb8aa3b);
        foreach (var isa in TrainingIsas)
        {
            // sum_axis_rows_f32: 13 rows (more than a block's 8: grid-stride on one block), on the full grid and on one block.
            const int Outer = 13;
            const float Scale = 0.25f;
            foreach (int dim in new[] { 32, 37, 200, 1000 })
            {
                float[] x = TrainingRandom(random, Outer * dim, -2, 2), y = TrainingRandom(random, Outer, -1, 1);
                foreach (bool accumulate in new[] { false, true })
                {
                    float[] expected = new float[Outer];
                    for (int o = 0; o < Outer; o++)
                    {
                        float s = CudaWarpRowSum(x.AsSpan(o * dim, dim)) * Scale;
                        expected[o] = accumulate ? s + y[o] : s;
                    }

                    int lanes = Outer * GcnKernels.SumAxisRowLanes;
                    foreach (int grid in new[] { (lanes + 255) / 256, 1 })
                    {
                        float[][] got = RowsOnGcn([x, y], (m, a) => TrainingDispatch("sum_axis_rows_f32", isa, m, (grid, 1), 1, a[0], a[1], (ulong)dim, Bits(Scale),
                            accumulate ? 1UL : 0UL, (ulong)lanes));
                        RowOrderSame(expected, got[1], $"sum_axis_rows_f32{(accumulate ? " (adding)" : "")} ({isa}), {Outer} x {dim}, {grid} blocks");
                    }
                }
            }

            // sum_rows_f32: each column's rows from 0 in order, then added to y.
            const int SumRows = 5;
            foreach (int cols in new[] { 1, 300, 1000 })
            {
                float[] x = TrainingRandom(random, SumRows * cols, -2, 2), y = TrainingRandom(random, cols, -1, 1);
                float[] expected = new float[cols];
                for (int j = 0; j < cols; j++)
                {
                    float s = 0f;
                    for (int r = 0; r < SumRows; r++)
                    {
                        s += x[r * cols + j];
                    }

                    expected[j] = y[j] + s;
                }

                float[][] got = RowsOnGcn([x, y], (m, a) => TrainingDispatch("sum_rows_f32", isa, m, (1, 1), 1, a[0], a[1], SumRows, (ulong)cols));
                RowOrderSame(expected, got[1], $"sum_rows_f32 ({isa}), {SumRows} x {cols}");
            }

            const int Rows = 3;
            foreach (int cols in RowsLengths)
            {
                int n = Rows * cols;
                float[] y = TrainingRandom(random, n, 0.01f, 0.99f), dy = TrainingRandom(random, n, -1, 1), dx = TrainingRandom(random, n, -1, 1);
                string how = $"({isa}), {Rows} x {cols}";

                // softmax_bwd_f32: Σ dy · y (each product rounded, then added; log: Σ dy) in CUDA's BlockReduce order.
                foreach (bool log in new[] { false, true })
                {
                    float[] expected = new float[n];
                    for (int r = 0; r < Rows; r++)
                    {
                        int at = r * cols;
                        float s = CudaBlockSum(CudaThreadSums(cols, (acc, c) => acc + (log ? dy[at + c] : dy[at + c] * y[at + c])));
                        for (int c = 0; c < cols; c++)
                        {
                            float t = log ? dy[at + c] - GcnFloat.Exp(log2E * y[at + c]) * s : (dy[at + c] - s) * y[at + c];
                            expected[at + c] = dx[at + c] + t;
                        }
                    }

                    float[][] got = RowsOnGcn([y, dy, dx], (m, a) => TrainingDispatch("softmax_bwd_f32", isa, m, (2, 1), 1, a[0], a[1], a[2], (ulong)cols, log ? 1UL : 0UL, Rows));
                    RowOrderSame(expected, got[2], $"softmax_bwd_f32{(log ? " (log)" : "")} {how}");
                }

                // rms_norm_backward_f32: Σ dy · y fused in CUDA's BlockReduce order, its quotient by cols, then the gradient.
                {
                    float[] inv = TrainingRandom(random, Rows, 0.5f, 2f), expected = new float[n];
                    for (int r = 0; r < Rows; r++)
                    {
                        int at = r * cols;
                        float mean = CudaBlockSum(CudaThreadSums(cols, (acc, c) => MathF.FusedMultiplyAdd(dy[at + c], y[at + c], acc))) / cols;
                        for (int c = 0; c < cols; c++)
                        {
                            expected[at + c] = dx[at + c] + inv[r] * (dy[at + c] - y[at + c] * mean);
                        }
                    }

                    float[][] got = RowsOnGcn([dy, y, inv, dx], (m, a) => TrainingDispatch("rms_norm_backward_f32", isa, m, (2, 1), 1, a[0], a[1], a[2], a[3], (ulong)cols, Rows));
                    RowOrderSame(expected, got[3], $"rms_norm_backward_f32 {how}");
                }

                // weight_norm_f32 and weight_norm_bwd_f32's dgain: Σ v² (and Σ dweight · v) in CUDA's BlockReduce order.
                {
                    float[] gain = TrainingRandom(random, Rows, 0.5f, 2f), v = TrainingRandom(random, n, -1, 1), dgain = TrainingRandom(random, Rows, -1, 1);
                    float[] weight = new float[n], dgainExpected = new float[Rows];
                    for (int r = 0; r < Rows; r++)
                    {
                        int at = r * cols;
                        float ss = CudaBlockSum(CudaThreadSums(cols, (acc, c) => MathF.FusedMultiplyAdd(v[at + c], v[at + c], acc)));
                        float dot = CudaBlockSum(CudaThreadSums(cols, (acc, c) => MathF.FusedMultiplyAdd(dy[at + c], v[at + c], acc)));
                        float root = MathF.Sqrt(ss), scale = gain[r] / root;
                        for (int c = 0; c < cols; c++)
                        {
                            weight[at + c] = v[at + c] * scale;
                        }

                        dgainExpected[r] = dgain[r] + dot * (1f / root);
                    }

                    float[][] got = RowsOnGcn([gain, v, new float[n]], (m, a) => TrainingDispatch("weight_norm_f32", isa, m, (2, 1), 1, a[0], a[1], a[2], (ulong)cols, Rows));
                    RowOrderSame(weight, got[2], $"weight_norm_f32 {how}");
                    float[][] back = RowsOnGcn([gain, v, dy, dgain], (m, a) => TrainingDispatch("weight_norm_bwd_f32", isa, m, (2, 1), 1, a[0], a[1], a[2], a[3], a[1], (ulong)cols, 1UL, Rows));
                    RowOrderSame(dgainExpected, back[3], $"weight_norm_bwd_f32 {how}, dgain");
                }

                // attn_bwd_d_f32: a thread a row, the dimensions fused in order from 0 (on 1 block and on 2).
                {
                    float[] expected = new float[Rows];
                    for (int r = 0; r < Rows; r++)
                    {
                        float acc = 0f;
                        for (int c = 0; c < cols; c++)
                        {
                            acc = MathF.FusedMultiplyAdd(y[r * cols + c], dy[r * cols + c], acc);
                        }

                        expected[r] = acc;
                    }

                    foreach (int grid in new[] { 1, 2 })
                    {
                        float[][] got = RowsOnGcn([y, dy, new float[Rows]], (m, a) => TrainingDispatch("attn_bwd_d_f32", isa, m, (grid, 1), 1, a[0], a[1], a[2], (ulong)cols, Rows));
                        RowOrderSame(expected, got[2], $"attn_bwd_d_f32 {how}, {grid} blocks");
                    }
                }
            }
        }
    }

    // (count, mean, M2) merged as CUDA's ChanMerge (and HIP's): the count's floats, δ, the ratio correctly rounded, fused.
    private static (uint N, float Mean, float M2) CudaChan((uint N, float Mean, float M2) a, (uint N, float Mean, float M2) b)
    {
        uint n = a.N + b.N;
        float d = b.Mean - a.Mean, r = (float)b.N / n;
        float mean = MathF.FusedMultiplyAdd(d, r, a.Mean);
        float q = d * d * a.N;
        q = a.M2 + MathF.FusedMultiplyAdd(q, r, b.M2);
        return n != 0 ? (n, mean, q) : (n, a.Mean, a.M2);
    }

    // norm_stats_f32, group_reduce_f32 and their split forms (part, then final) against C# models of CUDA's order: each
    // thread's elements (or chunks) in order, CUDA's halving tree (the chunks' statistics merged by Chan's formula in it).
    internal static void GcnTrainingGroupsCudaOrder(Device device)
    {
        _ = device;
        var random = new Random(86);
        const float Eps = 1e-5f;
        foreach (var (outer, groups, inner) in new[] { (3, 4, 50), (20, 3, 1), (2, 2, 9000) })
        {
            int n = outer * groups * inner, m = outer * inner;
            int chunks = (m + GcnKernels.GroupChunk - 1) / GcnKernels.GroupChunk;
            bool split = m >= 2 * GcnKernels.GroupChunk;
            float[] x = TrainingRandom(random, n, -1.5f, 2.5f), b = TrainingRandom(random, n, -1, 1);
            float[] sumA = TrainingRandom(random, groups, -1, 1), sumAB = TrainingRandom(random, groups, -1, 1);
            int Index(int g, int j) => (j / inner * groups + g) * inner + j % inner;
            var (mean, variance, invstd) = (new float[groups], new float[groups], new float[groups]);
            var (sumAOut, sumABOut) = ((float[])sumA.Clone(), (float[])sumAB.Clone());
            var part = new float[2 * chunks * groups];
            var partSums = new float[2 * chunks * groups];
            for (int g = 0; g < groups; g++)
            {
                if (!split)
                {
                    float mu = CudaTreeSum(CudaThreadSums(m, (acc, j) => acc + x[Index(g, j)])) / m;
                    float spread = CudaTreeSum(CudaThreadSums(m, (acc, j) => MathF.FusedMultiplyAdd(x[Index(g, j)] - mu, x[Index(g, j)] - mu, acc))) / m;
                    (mean[g], variance[g], invstd[g]) = (mu, spread, 1f / MathF.Sqrt(Eps + spread));
                    sumAOut[g] += CudaTreeSum(CudaThreadSums(m, (acc, j) => acc + x[Index(g, j)]));
                    sumABOut[g] += CudaTreeSum(CudaThreadSums(m, (acc, j) => MathF.FusedMultiplyAdd(x[Index(g, j)], b[Index(g, j)], acc)));
                    continue;
                }

                for (int c = 0; c < chunks; c++)
                {
                    int first = c * GcnKernels.GroupChunk, count = Math.Min(GcnKernels.GroupChunk, m - first), block = g * chunks + c;
                    float mu = CudaTreeSum(CudaThreadSums(count, (acc, j) => acc + x[Index(g, first + j)])) / count;
                    part[block] = mu;
                    part[chunks * groups + block] = CudaTreeSum(CudaThreadSums(count, (acc, j) => MathF.FusedMultiplyAdd(x[Index(g, first + j)] - mu, x[Index(g, first + j)] - mu, acc)));
                    partSums[block] = CudaTreeSum(CudaThreadSums(count, (acc, j) => acc + x[Index(g, first + j)]));
                    partSums[chunks * groups + block] = CudaTreeSum(CudaThreadSums(count, (acc, j) => MathF.FusedMultiplyAdd(x[Index(g, first + j)], b[Index(g, first + j)], acc)));
                }

                var merged = new (uint N, float Mean, float M2)[GcnKernels.RowThreads];
                for (int t = 0; t < merged.Length; t++)
                {
                    for (int c = t; c < chunks; c += GcnKernels.RowThreads)
                    {
                        uint count = (uint)Math.Min(GcnKernels.GroupChunk, m - c * GcnKernels.GroupChunk);
                        merged[t] = CudaChan(merged[t], (count, part[g * chunks + c], part[chunks * groups + g * chunks + c]));
                    }
                }

                var (_, mus, m2) = CudaTree(merged, CudaChan);
                float v = m2 / m;
                (mean[g], variance[g], invstd[g]) = (mus, v, 1f / MathF.Sqrt(Eps + v));
                sumAOut[g] += CudaTreeSum(CudaThreadSums(chunks, (acc, c) => acc + partSums[g * chunks + c]));
                sumABOut[g] += CudaTreeSum(CudaThreadSums(chunks, (acc, c) => acc + partSums[chunks * groups + g * chunks + c]));
            }

            foreach (var isa in TrainingIsas)
            {
                string shape = $"({isa}), {outer} x {groups} x {inner}{(split ? $", {chunks} chunks a group" : "")}";
                float[][] stats = RowsOnGcn([x, new float[groups], new float[groups], new float[groups], new float[2 * chunks * groups]], (mem, a) =>
                {
                    if (split)
                    {
                        TrainingDispatch("norm_stats_part_f32", isa, mem, (2, 1), 1, a[0], a[4], (ulong)outer, (ulong)groups, (ulong)inner, (ulong)chunks);
                        TrainingDispatch("norm_stats_final_f32", isa, mem, (1, 1), 1, a[4], a[1], a[2], a[3], (ulong)outer, (ulong)groups, (ulong)inner, (ulong)chunks, Bits(Eps));
                    }
                    else
                    {
                        TrainingDispatch("norm_stats_f32", isa, mem, (2, 1), 1, a[0], a[1], a[2], a[3], (ulong)outer, (ulong)groups, (ulong)inner, Bits(Eps));
                    }
                });
                string stat = split ? "norm_stats_part_f32 + norm_stats_final_f32" : "norm_stats_f32";
                RowOrderSame(mean, stats[1], $"{stat} {shape}, mean");
                RowOrderSame(variance, stats[2], $"{stat} {shape}, var");
                RowOrderSame(invstd, stats[3], $"{stat} {shape}, invstd");
                float[][] sums = RowsOnGcn([x, b, sumA, sumAB, new float[2 * chunks * groups]], (mem, a) =>
                {
                    if (split)
                    {
                        TrainingDispatch("group_reduce_part_f32", isa, mem, (2, 1), 1, a[0], a[1], a[4], (ulong)outer, (ulong)groups, (ulong)inner, (ulong)chunks, 1UL);
                        TrainingDispatch("group_reduce_final_f32", isa, mem, (1, 1), 1, a[4], a[2], a[3], (ulong)groups, (ulong)chunks, 1UL);
                    }
                    else
                    {
                        TrainingDispatch("group_reduce_f32", isa, mem, (2, 1), 1, a[0], a[1], a[2], a[3], (ulong)outer, (ulong)groups, (ulong)inner, 1UL);
                    }
                });
                string reduce = split ? "group_reduce_part_f32 + group_reduce_final_f32" : "group_reduce_f32";
                RowOrderSame(sumAOut, sums[2], $"{reduce} {shape}, suma");
                RowOrderSame(sumABOut, sums[3], $"{reduce} {shape}, sumab");
            }
        }
    }
    // sumsq_f32 over its blocks (each thread's elements block · 256 + t + k · blocks · 256 in order, each block's sum in
    // CUDA's order), then sum_blocks_f32 adding the partials (each thread's partials t, t + 256, … in order, the block's sum
    // in CUDA's order) to the total, as CudaBackend.SumSquares launches them.
    internal static void GcnTrainingSquaresCudaOrder(Device device)
    {
        _ = device;
        var random = new Random(84);
        foreach (int count in new[] { 700, 70_001, 300_000 })
        {
            int blocks = Math.Min(1024, (count + 255) / 256);
            float[] x = TrainingRandom(random, count, -2, 2);
            const float Before = 1.5f;
            var partials = new float[blocks];
            for (int b = 0; b < blocks; b++)
            {
                var threads = new float[GcnKernels.RowThreads];
                for (int t = 0; t < threads.Length; t++)
                {
                    float acc = 0f;
                    for (int i = b * 256 + t; i < count; i += blocks * 256)
                    {
                        acc = MathF.FusedMultiplyAdd(x[i], x[i], acc);
                    }

                    threads[t] = acc;
                }

                partials[b] = CudaBlockSum(threads);
            }

            var parts = new float[GcnKernels.RowThreads];
            for (int t = 0; t < parts.Length; t++)
            {
                float acc = 0f;
                for (int i = t; i < blocks; i += GcnKernels.RowThreads)
                {
                    acc += partials[i];
                }

                parts[t] = acc;
            }

            float total = Before + CudaBlockSum(parts) * 1f;
            foreach (var isa in TrainingIsas)
            {
                string what = $"sumsq_f32 + sum_blocks_f32 ({isa}), {count} elements in {blocks} blocks";
                float[][] got = RowsOnGcn([x, [Before], new float[blocks]], (m, a) =>
                {
                    TrainingDispatch("sumsq_f32", isa, m, (blocks, 1), 1, a[0], a[1], a[2], (ulong)count, 0, (ulong)blocks);
                    TrainingDispatch("sum_blocks_f32", isa, m, (1, 1), 1, a[2], a[1], (ulong)blocks, BitConverter.SingleToUInt32Bits(1f), 1, 1);
                });
                for (int b = 0; b < blocks; b++)
                {
                    Check(BitConverter.SingleToUInt32Bits(got[2][b]) == BitConverter.SingleToUInt32Bits(partials[b]),
                        $"{what}: block {b}'s sum is {got[2][b]:R}, CUDA's order gives {partials[b]:R}");
                }

                Check(BitConverter.SingleToUInt32Bits(got[1][0]) == BitConverter.SingleToUInt32Bits(total),
                    $"{what}: the total is {got[1][0]:R}, CUDA's order gives {total:R}");
            }
        }
    }
}
