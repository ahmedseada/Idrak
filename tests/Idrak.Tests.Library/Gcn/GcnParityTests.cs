// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using Idrak.Gpu.Hip;
using Idrak.Tests.Gcn;

// Plan 35, HIP parity, in the GCN interpreter: the kernels CUDA had and HIP lacked (GcnKernels.Parity.cs) against the CPU.
internal abstract partial class Tests
{
    internal static void GcnParityCrossEntropy(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(81);
        foreach (var (rows, vocabulary) in new[] { (2, 1), (3, 200), (4, 1037), (2, 2050), (2, 4500) })
        {
            float[] targets = [.. Enumerable.Range(0, rows).Select(_ => (float)random.Next(vocabulary))];
            float[] weights = [.. Enumerable.Range(0, rows).Select(r => r == 1 ? 0f : 0.5f + random.NextSingle())];
            const float Scale = 0.25f;
            float[] logits = TrainingRandom(random, rows * vocabulary, -4, 4);
            float[][] inputs = [logits, targets, weights, new float[rows]];
            var (modelLogits, modelLosses) = CrossEntropyInCudaOrder(logits, targets, weights, rows, vocabulary, Scale);
            foreach (var isa in TrainingIsas)
            {
                string what = $"softmax_ce_rows_f32 ({isa}), {rows} rows of {vocabulary}";
                TrainingCase(what, inputs, [0, 3], (b, s) => b.SoftmaxCrossEntropyRows(s[0], s[1], s[2], s[3], rows, vocabulary, Scale),
                    (m, a, grid, seed) => CrossEntropyDispatch(isa, m, grid, seed, a, vocabulary, Scale, rows), (1, rows), worst, "softmax_ce_rows_f32");

                // CUDA's grouping, bit for bit: a model of its order with the interpreter's exponential and logarithm.
                float[][] got = RowsOnGcn(inputs, (m, a) => CrossEntropyDispatch(isa, m, rows, 3, a, vocabulary, Scale, rows));
                for (int i = 0; i < modelLogits.Length; i++)
                {
                    Check(BitConverter.SingleToUInt32Bits(got[0][i]) == BitConverter.SingleToUInt32Bits(modelLogits[i]),
                        $"{what}: gradient {i} is {got[0][i]:R}, CUDA's order gives {modelLogits[i]:R}");
                }

                for (int r = 0; r < rows; r++)
                {
                    Check(BitConverter.SingleToUInt32Bits(got[3][r]) == BitConverter.SingleToUInt32Bits(modelLosses[r]),
                        $"{what}: loss {r} is {got[3][r]:R}, CUDA's order gives {modelLosses[r]:R}");
                }
            }
        }

        TrainingReport(worst);
    }

    // One dispatch of softmax_ce_rows_f32 over `grid` blocks of 1,024.
    private static void CrossEntropyDispatch(Idrak.Gpu.Abstractions.GcnIsaVersion isa, GcnMemory memory, int grid, ulong seed, ulong[] a, int vocabulary, float scale, int rows)
    {
        var (code, image) = TrainingKernel("softmax_ce_rows_f32", isa);
        var arguments = new byte[Idrak.Gpu.Hip.GcnCodeObject.KernargBytes(code)];
        GcnKernels.PackArguments(code.Arguments, arguments, [a[0], a[1], a[2], a[3], Cu(vocabulary), Cf(scale), Cu(rows)]);
        GcnInterpreter.Run(image, memory, new GcnDispatch
        {
            Workgroups = (grid, 1, 1), WorkgroupSize = (GcnKernels.CrossEntropyThreads, 1, 1), Arguments = arguments, Seed = seed,
        });
    }

    // CUDA's softmax_ce_rows_f32 in its order (PtxKernels.Decoder.cs: 1,024 threads, each over its columns in order; each
    // warp's butterfly over offsets 16 … 1; the warps' sums added in order to 0), with the exponential and logarithm the
    // interpreter gives v_exp_f32 and v_log_f32 (2^x, log2 x).
    private static (float[] Logits, float[] Losses) CrossEntropyInCudaOrder(float[] logits, float[] targets, float[] weights, int rows, int vocabulary, float scale)
    {
        const int Threads = 1024;
        float log2E = BitConverter.UInt32BitsToSingle(0x3fb8aa3b), ln2 = BitConverter.UInt32BitsToSingle(0x3f317218);
        float[] outLogits = [.. logits], losses = new float[rows];
        for (int r = 0; r < rows; r++)
        {
            var row = logits.AsSpan(r * vocabulary, vocabulary);
            float max = float.NegativeInfinity;
            foreach (float v in row)
            {
                max = MathF.Max(max, v);
            }

            var partial = new float[Threads];
            for (int t = 0; t < Threads; t++)
            {
                float s = 0f;
                for (int c = t; c < vocabulary; c += Threads)
                {
                    s += GcnFloat.Exp(log2E * (row[c] - max));
                }

                partial[t] = s;
            }

            float sum = 0f;
            for (int warp = 0; warp < Threads / 32; warp++)
            {
                var lanes = partial.AsSpan(warp * 32, 32);
                for (int offset = 16; offset >= 1; offset /= 2)
                {
                    for (int l = 0; l < offset; l++)
                    {
                        lanes[l] += lanes[l + offset];
                    }
                }

                sum += lanes[0];
            }

            float lse = MathF.FusedMultiplyAdd(GcnFloat.Log(sum), ln2, max);
            int target = (int)targets[r];
            losses[r] = weights[r] * (lse - row[target]);
            float gradientScale = weights[r] * scale;
            for (int c = 0; c < vocabulary; c++)
            {
                float e = GcnFloat.Exp(log2E * (row[c] - lse));
                if (c == target)
                {
                    e = -1f + e;
                }

                outLogits[r * vocabulary + c] = e * gradientScale;
            }
        }

        return (outLogits, losses);
    }
}
