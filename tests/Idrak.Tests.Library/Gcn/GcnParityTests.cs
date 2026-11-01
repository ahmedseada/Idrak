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
        foreach (var (rows, vocabulary) in new[] { (2, 1), (3, 200), (4, 1037), (2, 2050) })
        {
            float[] targets = [.. Enumerable.Range(0, rows).Select(_ => (float)random.Next(vocabulary))];
            float[] weights = [.. Enumerable.Range(0, rows).Select(r => r == 1 ? 0f : 0.5f + random.NextSingle())];
            const float Scale = 0.25f;
            foreach (var isa in TrainingIsas)
            {
                TrainingCase($"softmax_ce_rows_f32 ({isa}), {rows} rows of {vocabulary}",
                    [TrainingRandom(random, rows * vocabulary, -4, 4), targets, weights, new float[rows]], [0, 3],
                    (b, s) => b.SoftmaxCrossEntropyRows(s[0], s[1], s[2], s[3], rows, vocabulary, Scale),
                    (m, a, grid, seed) => TrainingDispatch("softmax_ce_rows_f32", isa, m, (grid, 1), seed, a[0], a[1], a[2], a[3], Cu(vocabulary), Cf(Scale), Cu(rows)),
                    (1, rows), worst, "softmax_ce_rows_f32");
            }
        }

        TrainingReport(worst);
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
