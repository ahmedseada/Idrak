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

    internal static void GcnParityGatedPacked(Device device)
    {
        _ = device;
        var random = new Random(82);
        float worstFloat = 0f, worstPacked = 0f;
        foreach (int n in new[] { 1, 255, 1003 })
        {
            int words = (n + 1) / 2;
            float[] Packed() => [.. Enumerable.Range(0, words).Select(_ =>
                BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(random.NextSingle() * 6 - 3) >> 16 | BitConverter.SingleToUInt32Bits(random.NextSingle() * 6 - 3) & 0xffff0000u))];
            foreach (int kind in new[] { 0, 1, 2, 3 })
            {
                foreach (int flags in new[] { 4, 6, 12, 14, 5, 9, 13 })
                {
                    float[][] inputs = [TrainingRandom(random, n, -3, 3), TrainingRandom(random, n, -3, 3), Packed(), Packed(), TrainingRandom(random, n, -1, 1), new float[words]];
                    float[][] expected = RowsOnCpu(inputs, (b, s) => b.GatedActivationPacked(s[0], s[1], s[2], s[3], s[4], s[5], n, kind, flags));
                    foreach (var isa in TrainingIsas)
                    {
                        string what = $"gated_act_bf16_f32 ({isa}), {n} elements, kind {kind}, flags {flags}";
                        int blocks = Math.Max(1, (words + 255) / 256);
                        float[][] first = RowsOnGcn(inputs, (m, a) => TrainingDispatch("gated_act_bf16_f32", isa, m, (1, 1), 1, a[0], a[1], a[2], a[3], a[4], a[5], Cu(kind),
                            Cu(flags), Cu(n), Cu(words)));
                        float[][] second = RowsOnGcn(inputs, (m, a) => TrainingDispatch("gated_act_bf16_f32", isa, m, (blocks + 2, 1), 7, a[0], a[1], a[2], a[3], a[4], a[5],
                            Cu(kind), Cu(flags), Cu(n), Cu(words)));
                        RowsSameBits(first, second, what);
                        if ((flags & 4) != 0)
                        {
                            worstFloat = MathF.Max(worstFloat, RowsCompare(expected[4], first[4], what + ", y"));
                        }

                        // Packed words: each half within one bfloat16 rounding of the CPU's (the activation may differ in
                        // its last float bit, which can move a rounding); gate and up packed from the inputs exactly.
                        foreach (int o in new[] { 2, 3, 5 })
                        {
                            bool written = o == 5 ? (flags & 8) != 0 : (flags & 2) != 0 && (flags & 1) == 0;
                            for (int i = 0; i < words; i++)
                            {
                                uint e = BitConverter.SingleToUInt32Bits(expected[o][i]), g = BitConverter.SingleToUInt32Bits(first[o][i]);
                                if (!written || o != 5)
                                {
                                    Check(e == g, $"{what}, storage {o}: word {i} is 0x{g:X8}, the CPU's 0x{e:X8}");
                                    continue;
                                }

                                foreach (int shift in new[] { 0, 16 })
                                {
                                    float ev = BitConverter.UInt32BitsToSingle(e >> shift << 16), gv = BitConverter.UInt32BitsToSingle(g >> shift << 16);
                                    float error = MathF.Abs(ev - gv) / MathF.Max(1f, MathF.Abs(ev));
                                    Check(error <= 1f / 128, $"{what}, y packed: word {i}'s {(shift == 0 ? "low" : "high")} half is {gv:R}, the CPU's {ev:R}");
                                    worstPacked = MathF.Max(worstPacked, error);
                                }
                            }
                        }

                    }
                }
            }
        }

        Console.WriteLine($"    gated_act_bf16_f32: largest error against the CPU {worstFloat:G3} (y), {worstPacked:G3} (y packed)");
    }
}
