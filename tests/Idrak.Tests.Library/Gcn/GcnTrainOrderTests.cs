// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using Idrak.Gpu.Hip;

// Plan 35, training order: the training sums grouped as CUDA's (RowWriter.ReduceAdd), each checked bit for bit against a
// C# model of CUDA's order (CudaBlockSum, GcnRowOrderTests.cs).
internal abstract partial class Tests
{
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
