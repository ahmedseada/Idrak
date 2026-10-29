// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using System.Text;
using Idrak.Abstraction.Devices;
using Idrak.Gpu.Abstractions;
using Idrak.Gpu.Hip;
using Idrak.Tests.Gcn;

// HIP's training kernels in GCN (GcnKernels.Training), each run by the GCN interpreter and proven against the CPU (plan 35,
// G5b). Every kernel is assembled for gfx90c and gfx906 and run over sizes that are multiples of nothing (element-wise:
// 1, 255, 1,000 and 4,099 elements, and 1,003 with every buffer 4 bytes off 16-byte alignment; rows of 1 to 2,333
// columns; groups split into chunks; key and query tiles past 256 for the attention gradients; head sizes of 37, 64 and
// 300), with the flags, kinds and variants the operations take; compared to the CPU backend within the device-against-CPU
// tolerance (OnBoth's, 1e-3 of max(1, |expected|)) or bit for bit where the arithmetic is the CPU's; run again on another
// grid and another wave order to give the same bits (every sum in a fixed order, no atomics); and the interpreter reports
// no missing wait, race or access out of range (a report fails the run). Each test prints its largest errors. With
// IDRAK_GCN_PRINTED naming a folder, each kernel's printed text is written there for the development check against llvm-mc.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] GcnTrainingGroup =
    [
        ("gcn kernels: training: every training kernel assembles for gfx90c and gfx906 with its block of 256 declared, within its register budget, its printed text reading back to its bytes; GcnKernels.All lists each once", GcnTrainingAssembled),
        ("gcn kernels: training: the 13 unary backward kernels (sigmoid, tanh, ReLU, square, abs, exp, log, both GELUs, sqrt, sin, cos, SiLU) equal the CPU's dx += dy · op'(x) over 1 to 4,099 elements and 1,003 misaligned, the same bits on two grids and wave orders", GcnTrainingUnary),
        ("gcn kernels: training: extremum_bwd_f32 (max and min, ties to a, either gradient alone), where_bwd_f32 (NaN and ±0 conditions), clamp_bwd_f32 and leaky_relu_bwd_f32 equal the CPU's bit for bit; pow_bwd_f32 (odd, even, fractional and negative powers of negative and positive bases) and gated_act_bwd_f32 (every kind, added and written) within tolerance; the same bits on two grids and wave orders", GcnTrainingSelections),
        ("gcn kernels: training: norm_apply_f32, norm_bwd_f32, group_scale_shift_f32 (every flag), broadcast_axis_f32, and sum_axis_f32 equal the CPU's over groups of 1 to 400 elements (aligned and not), the same bits on two grids and wave orders", GcnTrainingGroups),
        ("gcn kernels: training: sgd_momentum_f32, adam_f32, dropout_f32 (forward and backward, the CPU's mask bit for bit), add_dropout_f32 and clip_factor_f32 (0, negative, tiny, huge, ∞ and NaN sums) equal the CPU's, the same bits on two grids and wave orders", GcnTrainingOptimizers),
        ("gcn kernels: training: adam8_f32 equals the CPU's 8-bit Adam (the parameters within tolerance, every moment's code and the block scales bit for bit) over 700 elements (a partial last block), the same bits on two grids and wave orders", GcnTrainingAdam8),
        ("gcn kernels: training: multi_sumsq_f32, sum_blocks_f32, clip_factor_f32 and multi_adamw_f32 equal the CPU's fused AdamW over three tensors (5,000, 300 and 1 elements; with and without clipping; the gradients zeroed), the same bits on two grids and wave orders", GcnTrainingFusedAdamW),
        ("gcn kernels: training: softmax_bwd_f32 (and log-softmax), rms_norm_backward_f32, weight_norm_f32, weight_norm_bwd_f32 (each gradient alone and both) and attn_bwd_d_f32 equal the CPU's over rows of 1 to 2,333 columns, the same bits on two grids and wave orders", GcnTrainingRows),
        ("gcn kernels: training: sum_rows_f32 (bit for bit), sum_cols_strided_f32 with one chunk and three (folded by fold_chunks_f32), sum_f32 and sumsq_f32 over one block and several (added by sum_blocks_f32), and sum_axis_rows_f32 equal the CPU's sums, the same bits on two grids and wave orders", GcnTrainingSums),
        ("gcn kernels: training: scatter_chain_u32 and scatter_walk_f32 add 300 rows into 37 ids (repeats past one tile of 256) bit for bit as the CPU's scatter-add, without atomics, the same bits on two grids and wave orders", GcnTrainingScatter),
        ("gcn kernels: training: norm_stats_f32 and group_reduce_f32 (one block a group), and norm_stats_part_f32 + norm_stats_final_f32 and group_reduce_part_f32 + group_reduce_final_f32 (groups of three chunks merged in chunk order) equal the CPU's statistics, the same bits on two grids and wave orders", GcnTrainingGroupStats),
        ("gcn kernels: training: attn_bwd_d_f32, attn_bwd_kv_f32 and attn_bwd_q_f32 equal the CPU's gradients of tiled attention over a cache (heads of 37, 64 and 300; plain, a window, a soft-cap, both), dk and dv summed without atomics, the same bits on two grids and wave orders", GcnTrainingAttentionTiled),
        ("gcn kernels: training: attn_spans_bwd_kv_f32 and attn_spans_bwd_q_f32 equal the CPU's gradients of attention over key ranges (grouped heads, two tables, an empty range, a soft-cap), the same bits on two grids and wave orders", GcnTrainingAttentionSpans),
    ];

    private static readonly GcnIsaVersion[] TrainingIsas = [GcnIsaVersion.Parse("gfx90c"), GcnIsaVersion.Parse("gfx906")];

    private static readonly int[] TrainingCounts = [1, 255, 1000, 4099];

    private static readonly Dictionary<(string, GcnIsaVersion), (GcnKernelCode Code, GcnKernelImage Image)> trainingKernels = [];

    // The kernel (of the training set, or fold_chunks_f32 of the rows) assembled for isa, written into a code object and
    // read back (once a process).
    private static (GcnKernelCode Code, GcnKernelImage Image) TrainingKernel(string name, GcnIsaVersion isa)
    {
        lock (trainingKernels)
        {
            if (trainingKernels.TryGetValue((name, isa), out var known))
            {
                return known;
            }

            var source = GcnKernels.All.Single(k => k.Name == name);
            var target = GcnTargets.For(isa);
            using var asm = new GcnAssembler(target, isa);
            var code = asm.AssembleKernel(source.Text);
            string printed = asm.PrintedText;
            byte[] again = AssembleGcn(target, isa, printed);
            Check(again.AsSpan().SequenceEqual(code.Code.Span), $"{name} ({isa}): its printed text reads back to other bytes:\n{printed}");
            if (Environment.GetEnvironmentVariable("IDRAK_GCN_PRINTED") is { Length: > 0 } folder)
            {
                Directory.CreateDirectory(folder);
                File.WriteAllText(Path.Combine(folder, $"{name}.{isa}.s"), printed);
                File.WriteAllBytes(Path.Combine(folder, $"{name}.{isa}.bin"), code.Code.ToArray());
            }

            var image = GcnCodeObjectReader.Read(GcnCodeObject.Write(target, isa, 5, [code]), name);
            return trainingKernels[(name, isa)] = (code, image);
        }
    }

    // One dispatch: the arguments as 64-bit values (addresses, 32-bit values, float bits) in order.
    private static void TrainingDispatch(string name, GcnIsaVersion isa, GcnMemory memory, (int X, int Y) grid, ulong seed, params ulong[] values)
    {
        var (code, image) = TrainingKernel(name, isa);
        var arguments = new byte[GcnCodeObject.KernargBytes(code)];
        GcnKernels.PackArguments(code.Arguments, arguments, values);
        GcnInterpreter.Run(image, memory, new GcnDispatch
        {
            Workgroups = (grid.X, grid.Y, 1), WorkgroupSize = (GcnKernels.RowThreads, 1, 1), Arguments = arguments, Seed = seed,
        });
    }

    // Runs `gcn` twice (grid `First` with seed 1, then `Second` with seed 7: another stride and wave order), the same bits both
    // times; compares the outputs with the CPU's (bit for bit when exact, NaN payloads aside) and keeps the worst error. With
    // a pad, every buffer is mapped `pad` floats later than its storage starts (off 16-byte alignment), the pad unwritten.
    private static void TrainingCase(string what, float[][] inputs, int[] outputs, Action<Backend, Storage[]> cpu, Action<GcnMemory, ulong[], int, ulong> gcn,
        (int First, int Second) grids, Dictionary<string, float> worst, string kernel, bool exact = false, int pad = 0)
    {
        float[][] expected = RowsOnCpu(inputs, cpu);
        float[][] padded = pad == 0 ? inputs : [.. inputs.Select(d => (float[])[.. new float[pad], .. d])];
        ulong[] Shift(ulong[] a) => pad == 0 ? a : [.. a.Select(x => x + (ulong)(4 * pad))];
        float[][] first = RowsOnGcn(padded, (m, a) => gcn(m, Shift(a), grids.First, 1));
        float[][] second = RowsOnGcn(padded, (m, a) => gcn(m, Shift(a), grids.Second, 7));
        RowsSameBits(first, second, what);
        foreach (int o in outputs)
        {
            Check(first[o].Take(pad).All(v => v == 0f), $"{what}, storage {o}: the pad before the buffer was written");
            float[] got = first[o][pad..];
            if (exact)
            {
                for (int i = 0; i < got.Length; i++)
                {
                    float e = expected[o][i], g = got[i];
                    Check(float.IsNaN(e) ? float.IsNaN(g) : BitConverter.SingleToUInt32Bits(e) == BitConverter.SingleToUInt32Bits(g),
                        $"{what}, storage {o}: element {i} is {g:R} (0x{BitConverter.SingleToUInt32Bits(g):X8}), the CPU's {e:R} (0x{BitConverter.SingleToUInt32Bits(e):X8})");
                }

                worst[kernel] = worst.GetValueOrDefault(kernel);
                continue;
            }

            float error = RowsCompare(expected[o], got, $"{what}, storage {o}");
            worst[kernel] = MathF.Max(worst.GetValueOrDefault(kernel), error);
        }
    }

    // An element-wise kernel over 1, 255, 1,000 and 4,099 elements (one block, then enough with idle ones) and 1,003
    // misaligned, on both processors.
    private static void ElementwiseCases(string kernel, Func<int, float[][]> inputs, int[] outputs, Action<Backend, Storage[], int> cpu,
        Func<ulong[], int, ulong[]> args, Dictionary<string, float> worst, bool exact = false, string label = "")
    {
        foreach (var isa in TrainingIsas)
        {
            foreach (int n in TrainingCounts.Append(1003))
            {
                int pad = n == 1003 ? 1 : 0;
                int blocks = Math.Max(1, (n + 1023) / 1024);
                TrainingCase($"{kernel}{label} ({isa}), {n} elements{(pad > 0 ? ", misaligned" : "")}", inputs(n), outputs, (b, s) => cpu(b, s, n),
                    (m, a, grid, seed) => TrainingDispatch(kernel, isa, m, (grid, 1), seed, args(a, n)), (1, blocks + 2), worst, kernel + label, exact, pad);
            }
        }
    }

    private static void TrainingReport(Dictionary<string, float> worst)
    {
        foreach (var (kernel, error) in worst)
        {
            Console.WriteLine($"    {kernel}: largest error against the CPU {error:G3}");
        }
    }

    private static float[] TrainingRandom(Random random, int n, float low, float high) => RowsRandom(random, n, (high - low) / 2, (high + low) / 2);

    internal static void GcnTrainingAssembled(Device device)
    {
        _ = device;
        var names = GcnKernels.Training.Select(k => k.Name).ToArray();
        Check(names.Length == 56 && names.Distinct().Count() == names.Length, $"56 training kernels (G5c's gated_act_bwd_bf16_f32 among them), each once ({names.Length}): {string.Join(", ", names)}");
        foreach (string name in names)
        {
            Check(GcnKernels.All.Count(k => k.Name == name) == 1, $"GcnKernels.All lists {name} once");
            Check(Encoding.UTF8.GetString(GcnKernels.Training.Single(k => k.Name == name).Text).StartsWith($".kernel {name}\n", StringComparison.Ordinal), $"{name}'s text starts with its .kernel line");
            Check(HipBackend.Cost(name).Bytes > 0, $"{name} has its profile cost (GcnKernels.Training.cs's table, HipBackend.TrainingCost)");
            foreach (var isa in TrainingIsas)
            {
                var (code, _) = TrainingKernel(name, isa);
                Check(code is { RequiredWorkgroupSize: (GcnKernels.RowThreads, 1, 1), MaxFlatWorkgroupSize: GcnKernels.RowThreads, WaveSize: 64 },
                    $"{name} ({isa}): declares the block of {GcnKernels.RowThreads} (required {code.RequiredWorkgroupSize}, max flat {code.MaxFlatWorkgroupSize})");
                Check(code.LdsBytes <= GcnKernels.AttentionBackwardLdsBytes, $"{name} ({isa}): {code.LdsBytes} bytes of LDS");
                if (isa == TrainingIsas[0])
                {
                    Console.WriteLine($"    {name}: {code.Code.Length} bytes of code, {code.Vgprs} VGPRs, {code.Sgprs} SGPRs, {code.LdsBytes} bytes of LDS");
                }
            }
        }
    }

    internal static void GcnTrainingUnary(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(51);
        (UnaryOp Op, float Low, float High)[] ops =
        [
            (UnaryOp.Sigmoid, -8, 8), (UnaryOp.Tanh, -6, 6), (UnaryOp.Relu, -3, 3), (UnaryOp.Square, -3, 3), (UnaryOp.Abs, -3, 3), (UnaryOp.Exp, -5, 5),
            (UnaryOp.Log, 0.05f, 6), (UnaryOp.Gelu, -6, 6), (UnaryOp.GeluErf, -6, 6), (UnaryOp.Sqrt, 0.01f, 9), (UnaryOp.Sin, -50, 50), (UnaryOp.Cos, -50, 50),
            (UnaryOp.Silu, -8, 8),
        ];
        foreach (var (op, low, high) in ops)
        {
            string kernel = HipBackend.UnaryBackwardKernelName(op)!;
            ElementwiseCases(kernel, n =>
            {
                float[] x = TrainingRandom(random, n, low, high);
                if (op is UnaryOp.Relu or UnaryOp.Abs && n > 3)
                {
                    x[1] = 0f;
                    x[2] = -0f;
                }

                float[] y = RowsOnCpu([x, new float[n]], (b, s) => b.UnaryKernel(op, s[0], s[1], n))[1];
                return [x, y, TrainingRandom(random, n, -1, 1), TrainingRandom(random, n, -1, 1)];
            }, [3], (b, s, n) => b.UnaryBackwardKernel(op, s[0], s[1], s[2], s[3], n), (a, n) => [a[0], a[1], a[2], a[3], (ulong)n], worst);
        }

        TrainingReport(worst);
    }

    internal static void GcnTrainingSelections(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(52);
        foreach (bool minimum in (bool[])[false, true])
        {
            foreach (int given in (int[])[3, 1, 2])
            {
                var op = minimum ? BinaryOp.Minimum : BinaryOp.Maximum;
                int flags = given | (minimum ? 4 : 0);
                ElementwiseCases("extremum_bwd_f32", n =>
                {
                    float[] a = TrainingRandom(random, n, -2, 2), b = TrainingRandom(random, n, -2, 2);
                    for (int i = 0; i < n; i += 5)
                    {
                        b[i] = a[i];                                                     // ties go to a
                    }

                    if (n > 7)
                    {
                        a[7] = float.NaN;
                    }

                    return [a, b, TrainingRandom(random, n, -1, 1), TrainingRandom(random, n, -1, 1), TrainingRandom(random, n, -1, 1)];
                }, [3, 4], (k, s, n) => k.ExtremumBackwardKernel(op, s[0], s[1], s[2], (given & 1) != 0 ? s[3] : null, (given & 2) != 0 ? s[4] : null, n),
                    (a, n) => [a[0], a[1], a[2], (given & 1) != 0 ? a[3] : a[2], (given & 2) != 0 ? a[4] : a[2], (ulong)flags, (ulong)n], worst, exact: true,
                    label: $" ({(minimum ? "min" : "max")}, flags {flags})");
            }
        }

        foreach (int given in (int[])[3, 1, 2])
        {
            ElementwiseCases("where_bwd_f32", n =>
            {
                float[] c = [.. Enumerable.Range(0, n).Select(i => (i % 4) switch { 0 => 0f, 1 => -0f, 2 => float.NaN, _ => random.NextSingle() - 0.5f })];
                return [c, TrainingRandom(random, n, -1, 1), TrainingRandom(random, n, -1, 1), TrainingRandom(random, n, -1, 1)];
            }, [2, 3], (k, s, n) => k.WhereBackwardKernel(s[0], s[1], (given & 1) != 0 ? s[2] : null, (given & 2) != 0 ? s[3] : null, n),
                (a, n) => [a[0], a[1], (given & 1) != 0 ? a[2] : a[1], (given & 2) != 0 ? a[3] : a[1], (ulong)given, (ulong)n], worst, exact: true, label: $" (flags {given})");
        }

        foreach (var (lo, hi) in new[] { (-1f, 0.5f), (float.NegativeInfinity, 0.25f) })
        {
            ElementwiseCases("clamp_bwd_f32", n => [TrainingRandom(random, n, -2, 2), TrainingRandom(random, n, -1, 1), TrainingRandom(random, n, -1, 1)], [2],
                (k, s, n) => k.ClampBackwardKernel(s[0], s[1], s[2], n, lo, hi), (a, n) => [a[0], a[1], a[2], Bits(lo), Bits(hi), (ulong)n], worst, exact: true,
                label: $" [{lo}, {hi}]");
        }

        ElementwiseCases("leaky_relu_bwd_f32", n => [TrainingRandom(random, n, -2, 2), TrainingRandom(random, n, -1, 1), TrainingRandom(random, n, -1, 1)], [2],
            (k, s, n) => k.LeakyReluBackwardKernel(s[0], s[1], s[2], n, 0.1f), (a, n) => [a[0], a[1], a[2], Bits(0.1f), (ulong)n], worst, exact: true);
        foreach (float exponent in (float[])[2.5f, 3f, 2f, 1f, -1.5f, 0.5f, 4f])
        {
            float p = exponent - 1f;
            ElementwiseCases("pow_bwd_f32", n => [TrainingRandom(random, n, -3, 3), TrainingRandom(random, n, -1, 1), TrainingRandom(random, n, -1, 1)], [2],
                (k, s, n) => k.PowBackwardKernel(s[0], s[1], s[2], n, exponent),
                (a, n) => [a[0], a[1], a[2], Bits(exponent), Bits(p), GcnKernels.PowGradientFlags(p), (ulong)n], worst, label: $" (e = {exponent})");
        }

        foreach (int kind in (int[])[0, 1, 2, 3])
        {
            foreach (int flags in (int[])[3, 1, 2, 15])
            {
                ElementwiseCases("gated_act_bwd_f32",
                    n => [TrainingRandom(random, n, -4, 4), TrainingRandom(random, n, -3, 3), TrainingRandom(random, n, -1, 1), TrainingRandom(random, n, -1, 1), TrainingRandom(random, n, -1, 1)],
                    [3, 4], (k, s, n) => k.GatedActivationBackwardKernel(s[0], s[1], s[2], s[3], s[4], n, kind, flags),
                    (a, n) => [a[0], a[1], a[2], a[3], a[4], (ulong)kind, (ulong)flags, (ulong)n], worst, label: $" (kind {kind}, flags {flags})");
            }
        }

        TrainingReport(worst);
    }

    internal static void GcnTrainingGroups(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(53);
        foreach (var (outer, groups, inner) in new[] { (3, 5, 7), (50, 9, 1), (4, 3, 400) })
        {
            int n = outer * groups * inner, blocks = Math.Max(1, (n + 1023) / 1024);
            foreach (var isa in TrainingIsas)
            {
                foreach (int pad in (int[])[0, 1])
                {
                    string shape = $"({isa}), {outer} x {groups} x {inner}{(pad > 0 ? ", misaligned" : "")}";
                    float[] inv = TrainingRandom(random, groups, 0.5f, 2f);
                    TrainingCase($"norm_apply_f32 {shape}", [TrainingRandom(random, n, -3, 3), TrainingRandom(random, groups, -1, 1), inv, new float[n]], [3],
                        (b, s) => b.NormApplyKernel(s[0], s[1], s[2], s[3], outer, groups, inner),
                        (m, a, grid, seed) => TrainingDispatch("norm_apply_f32", isa, m, (grid, 1), seed, a[0], a[1], a[2], a[3], (ulong)groups, (ulong)inner, (ulong)n),
                        (1, blocks + 1), worst, "norm_apply_f32", pad: pad);
                    TrainingCase($"norm_bwd_f32 {shape}",
                        [TrainingRandom(random, n, -1, 1), TrainingRandom(random, n, -2, 2), TrainingRandom(random, groups, -5, 5), TrainingRandom(random, groups, -5, 5), inv,
                            TrainingRandom(random, n, -1, 1)], [5],
                        (b, s) => b.NormBackwardKernel(s[0], s[1], s[2], s[3], s[4], s[5], outer, groups, inner),
                        (m, a, grid, seed) => TrainingDispatch("norm_bwd_f32", isa, m, (grid, 1), seed, a[0], a[1], a[2], a[3], a[4], a[5], (ulong)groups, (ulong)inner,
                            Bits(outer * inner), (ulong)n), (1, blocks + 1), worst, "norm_bwd_f32", pad: pad);
                    foreach (int flags in (int[])[0, 1, 2, 3, 7, 4])
                    {
                        bool scale = (flags & 1) != 0, shift = (flags & 2) != 0, accumulate = (flags & 4) != 0;
                        TrainingCase($"group_scale_shift_f32 (flags {flags}) {shape}",
                            [TrainingRandom(random, n, -2, 2), TrainingRandom(random, groups, -2, 2), TrainingRandom(random, groups, -1, 1), TrainingRandom(random, n, -1, 1)], [3],
                            (b, s) => b.GroupScaleShiftKernel(s[0], scale ? s[1] : null, shift ? s[2] : null, s[3], n, groups, inner, accumulate),
                            (m, a, grid, seed) => TrainingDispatch("group_scale_shift_f32", isa, m, (grid, 1), seed, a[0], scale ? a[1] : a[0], shift ? a[2] : a[0], a[3],
                                (ulong)groups, (ulong)inner, (ulong)flags, (ulong)n), (1, blocks + 1), worst, "group_scale_shift_f32", pad: pad);
                    }

                    // broadcast_axis_f32 and sum_axis_f32 over [outer, groups as the axis, inner].
                    TrainingCase($"broadcast_axis_f32 {shape}", [TrainingRandom(random, outer * inner, -2, 2), TrainingRandom(random, n, -1, 1)], [1],
                        (b, s) => b.BroadcastAxisKernel(s[0], s[1], outer, groups, inner, 0.75f),
                        (m, a, grid, seed) => TrainingDispatch("broadcast_axis_f32", isa, m, (grid, 1), seed, a[0], a[1], (ulong)(groups * inner), (ulong)inner, Bits(0.75f), (ulong)n),
                        (1, blocks + 1), worst, "broadcast_axis_f32", exact: true, pad: pad);
                    foreach (bool accumulate in (bool[])[false, true])
                    {
                        int outputs = outer * inner;
                        TrainingCase($"sum_axis_f32{(accumulate ? " (adding)" : "")} {shape}", [TrainingRandom(random, n, -2, 2), TrainingRandom(random, outputs, -1, 1)], [1],
                            (b, s) => b.SumAxisKernel(s[0], s[1], outer, groups, inner, 0.5f, accumulate),
                            (m, a, grid, seed) => TrainingDispatch("sum_axis_f32", isa, m, (grid, 1), seed, a[0], a[1], (ulong)groups, (ulong)inner, Bits(0.5f),
                                accumulate ? 1UL : 0UL, (ulong)outputs), (1, Math.Max(2, (outputs + 255) / 256)), worst, "sum_axis_f32", pad: pad);
                    }
                }
            }
        }

        TrainingReport(worst);
    }

    internal static void GcnTrainingOptimizers(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(54);
        ElementwiseCases("sgd_momentum_f32", n => [TrainingRandom(random, n, -1, 1), TrainingRandom(random, n, -1, 1), TrainingRandom(random, n, -1, 1)], [0, 2],
            (b, s, n) => b.SgdStepKernel(s[0], s[1], s[2], n, 0.1f, 0.9f), (a, n) => [a[0], a[1], a[2], Bits(-0.1f), Bits(0.9f), (ulong)n], worst);
        const float Lr = 1e-3f, B1 = 0.9f, B2 = 0.999f, Eps = 1e-8f;
        ElementwiseCases("adam_f32",
            n => [TrainingRandom(random, n, -1, 1), TrainingRandom(random, n, -1, 1), TrainingRandom(random, n, -0.1f, 0.1f), TrainingRandom(random, n, 0f, 0.01f)], [0, 2, 3],
            (b, s, n) => b.AdamStepKernel(s[0], s[1], s[2], s[3], n, Lr, B1, B2, Eps),
            (a, n) => [a[0], a[1], a[2], a[3], Bits(Lr), Bits(B1), Bits(B2), Bits(1f - B1), Bits(1f - B2), Bits(Eps), (ulong)n], worst);
        const float P = 0.3f;
        const uint Seed = 12345;
        float scale = 1f / (1f - P);
        foreach (bool backward in (bool[])[false, true])
        {
            ElementwiseCases("dropout_f32", n => [TrainingRandom(random, n, -2, 2), TrainingRandom(random, n, -1, 1)], [1],
                (b, s, n) =>
                {
                    if (backward)
                    {
                        b.DropoutBackwardKernel(s[0], s[1], n, P, Seed);
                    }
                    else
                    {
                        b.DropoutKernel(s[0], s[1], n, P, Seed);
                    }
                }, (a, n) => [a[0], a[1], Bits(P), Bits(scale), Seed, backward ? 1UL : 0UL, (ulong)n], worst, exact: true, label: backward ? " (backward)" : "");
        }

        ElementwiseCases("add_dropout_f32", n => [TrainingRandom(random, n, -1, 1), TrainingRandom(random, n, -2, 2), new float[n]], [2],
            (b, s, n) => b.AddDropoutKernel(s[0], s[1], s[2], n, P, Seed), (a, n) => [a[0], a[1], a[2], Bits(P), Bits(scale), Seed, (ulong)n], worst, exact: true);
        foreach (float sum in (float[])[0f, -1f, 4f, 1e-10f, 1e6f, float.PositiveInfinity, float.NaN, 0.25f])
        {
            foreach (var isa in TrainingIsas)
            {
                TrainingCase($"clip_factor_f32 ({isa}), Σ = {sum}", [[sum], [7f]], [1], (b, s) => b.ClipFactorKernel(s[0], s[1], 1.5f),
                    (m, a, grid, seed) => TrainingDispatch("clip_factor_f32", isa, m, (grid, 1), seed, a[0], a[1], Bits(1.5f), 1UL), (1, 2), worst, "clip_factor_f32");
            }
        }

        TrainingReport(worst);
    }

    // A sorted map of 256 codes: signed in [-1, 1], unsigned in [0, 1] (denser near 0, as 8-bit moments' maps are).
    private static float[] Adam8Map()
    {
        var map = new float[512];
        for (int i = 0; i < 256; i++)
        {
            float t = (i - 127.5f) / 127.5f;
            map[i] = t * t * t;
            float u = i / 255f;
            map[256 + i] = u * u;
        }

        return map;
    }

    internal static void GcnTrainingAdam8(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(55);
        const int N = 700, Blocks = (N + 255) / 256;
        const float Lr = 1e-3f, B1 = 0.9f, B2 = 0.999f, Eps = 1e-8f, Scale = 0.5f, Decay = 0.999f;
        var codes = new byte[2 * ((N + 3) / 4 * 4)];
        random.NextBytes(codes);
        float[] m = MemoryMarshal.Cast<byte, float>(codes.AsSpan(0, (N + 3) / 4 * 4)).ToArray(), v = MemoryMarshal.Cast<byte, float>(codes.AsSpan((N + 3) / 4 * 4)).ToArray();
        float[] absmax = TrainingRandom(random, 2 * Blocks, 0.01f, 0.1f);
        foreach (var isa in TrainingIsas)
        {
            float[][] inputs = [TrainingRandom(random, N, -1, 1), TrainingRandom(random, N, -1, 1), m, v, absmax, Adam8Map()];
            Action<Backend, Storage[]> cpu = (b, s) => b.AdamStep8BitKernel(s[0], s[1], s[2], s[3], s[4], s[5], N, Lr, B1, B2, Eps, Scale, Decay);
            Action<GcnMemory, ulong[], int, ulong> gcn = (mem, a, grid, seed) => TrainingDispatch("adam8_f32", isa, mem, (grid, 1), seed, a[0], a[1], a[2], a[3], a[4], a[5],
                Bits(Lr), Bits(B1), Bits(B2), Bits(1f - B1), Bits(1f - B2), Bits(Eps), N, Bits(Scale), Bits(Decay), Blocks);
            TrainingCase($"adam8_f32 ({isa}), {N} elements: parameters and scales", inputs, [0, 4], cpu, gcn, (1, Blocks), worst, "adam8_f32");
            TrainingCase($"adam8_f32 ({isa}), {N} elements: the moments' codes", inputs, [2, 3], cpu, gcn, (2, Blocks), worst, "adam8_f32 (codes)", exact: true);
        }

        TrainingReport(worst);
    }

    internal static void GcnTrainingFusedAdamW(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(56);
        int[] sizes = [5000, 300, 1];
        const float Lr = 1e-3f, Decay = 0.99f, B1 = 0.9f, B2 = 0.999f, Eps = 1e-8f;
        var chunks = new List<uint>();
        for (int t = 0; t < sizes.Length; t++)
        {
            for (int start = 0; start < sizes[t]; start += GcnKernels.MultiAdamChunk)
            {
                chunks.Add((uint)t);
                chunks.Add((uint)start);
            }
        }

        int count = chunks.Count / 2;
        foreach (float maxNorm in (float[])[1f, 0f])
        {
            foreach (var isa in TrainingIsas)
            {
                var inputs = new List<float[]>();
                foreach (int size in sizes)
                {
                    inputs.AddRange([TrainingRandom(random, size, -1, 1), TrainingRandom(random, size, -0.2f, 0.2f), TrainingRandom(random, size, -0.1f, 0.1f),
                        TrainingRandom(random, size, 0f, 0.01f)]);
                }

                inputs.Add([0f]);                                                       // the factor
                inputs.Add(new float[count]);                                           // the chunks' sums of squares
                int factor = 4 * sizes.Length, partials = factor + 1;
                int[] outputs = [.. Enumerable.Range(0, 4 * sizes.Length)];
                TrainingCase($"multi_adamw_f32 ({isa}), maxnorm {maxNorm}", [.. inputs], outputs,
                    (b, s) =>
                    {
                        var tensors = new (Storage P, Storage G, Storage M, Storage V, int N)[sizes.Length];
                        for (int t = 0; t < sizes.Length; t++)
                        {
                            tensors[t] = (s[4 * t], s[4 * t + 1], s[4 * t + 2], s[4 * t + 3], sizes[t]);
                        }

                        IDisposable? cache = null;
                        b.FusedAdamWKernel(tensors, ref cache, maxNorm, Lr, Decay, B1, B2, Eps, zeroGradients: true);
                        cache?.Dispose();
                    },
                    (mem, a, grid, seed) =>
                    {
                        ulong chunkAt = mem.Map<uint>("chunks", CollectionsMarshal.AsSpan(chunks));
                        var table = new ulong[5 * sizes.Length];
                        for (int t = 0; t < sizes.Length; t++)
                        {
                            (table[5 * t], table[5 * t + 1], table[5 * t + 2], table[5 * t + 3], table[5 * t + 4]) = (a[4 * t], a[4 * t + 1], a[4 * t + 2], a[4 * t + 3], (ulong)sizes[t]);
                        }

                        ulong tableAt = mem.Map<ulong>("table", table);
                        if (maxNorm > 0f)
                        {
                            TrainingDispatch("multi_sumsq_f32", isa, mem, (grid, 1), seed, chunkAt, tableAt, a[partials], (ulong)count);
                            TrainingDispatch("sum_blocks_f32", isa, mem, (1, 1), seed, a[partials], a[factor], (ulong)count, Bits(1f), 0UL, 1UL);
                            TrainingDispatch("clip_factor_f32", isa, mem, (1, 1), seed, a[factor], a[factor], Bits(maxNorm), 1UL);
                        }
                        else
                        {
                            mem.Bytes(a[factor]).AsSpan(0, 4).Clear();
                            BitConverter.TryWriteBytes(mem.Bytes(a[factor]).AsSpan(0, 4), 1f);
                        }

                        TrainingDispatch("multi_adamw_f32", isa, mem, (grid, 1), seed, chunkAt, tableAt, a[factor], Bits(Lr), Bits(Decay), Bits(B1), Bits(B2), Bits(1f - B1),
                            Bits(1f - B2), Bits(Eps), 1UL, (ulong)count);
                    }, (1, count), worst, "multi_sumsq_f32 + multi_adamw_f32");
            }
        }

        TrainingReport(worst);
    }

    internal static void GcnTrainingRows(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(57);
        const int Rows = 3;
        foreach (int cols in RowsLengths)
        {
            int n = Rows * cols;
            float[] logits = TrainingRandom(random, n, -4, 4);
            foreach (var isa in TrainingIsas)
            {
                foreach (bool log in (bool[])[false, true])
                {
                    float[] y = RowsOnCpu([logits, new float[n]], (b, s) => b.SoftmaxKernel(s[0], s[1], Rows, cols, log))[1];
                    TrainingCase($"softmax_bwd_f32{(log ? " (log)" : "")} ({isa}), {Rows} x {cols}", [y, TrainingRandom(random, n, -1, 1), TrainingRandom(random, n, -1, 1)], [2],
                        (b, s) => b.SoftmaxBackwardKernel(s[0], s[1], s[2], Rows, cols, log),
                        (m, a, grid, seed) => TrainingDispatch("softmax_bwd_f32", isa, m, (grid, 1), seed, a[0], a[1], a[2], (ulong)cols, log ? 1UL : 0UL, Rows),
                        (2, Rows), worst, log ? "softmax_bwd_f32 (log)" : "softmax_bwd_f32");
                }

                float[][] normed = RowsOnCpu([logits, new float[n], new float[Rows]], (b, s) => b.RmsNormKernel(s[0], s[1], s[2], Rows, cols, 1e-6f));
                TrainingCase($"rms_norm_backward_f32 ({isa}), {Rows} x {cols}", [TrainingRandom(random, n, -1, 1), normed[1], normed[2], TrainingRandom(random, n, -1, 1)], [3],
                    (b, s) => b.RmsNormBackwardKernel(s[0], s[1], s[2], s[3], Rows, cols),
                    (m, a, grid, seed) => TrainingDispatch("rms_norm_backward_f32", isa, m, (grid, 1), seed, a[0], a[1], a[2], a[3], (ulong)cols, Rows), (2, Rows), worst,
                    "rms_norm_backward_f32");
                float[] gain = TrainingRandom(random, Rows, 0.5f, 2f), direction = TrainingRandom(random, n, -1, 1);
                TrainingCase($"weight_norm_f32 ({isa}), {Rows} x {cols}", [gain, direction, new float[n]], [2], (b, s) => b.WeightNormKernel(s[0], s[1], s[2], Rows, cols),
                    (m, a, grid, seed) => TrainingDispatch("weight_norm_f32", isa, m, (grid, 1), seed, a[0], a[1], a[2], (ulong)cols, Rows), (2, Rows), worst, "weight_norm_f32");
                foreach (int flags in (int[])[3, 1, 2])
                {
                    TrainingCase($"weight_norm_bwd_f32 (flags {flags}) ({isa}), {Rows} x {cols}",
                        [gain, direction, TrainingRandom(random, n, -1, 1), TrainingRandom(random, Rows, -1, 1), TrainingRandom(random, n, -1, 1)], [3, 4],
                        (b, s) => b.WeightNormBackwardKernel(s[0], s[1], s[2], (flags & 1) != 0 ? s[3] : null, (flags & 2) != 0 ? s[4] : null, Rows, cols),
                        (m, a, grid, seed) => TrainingDispatch("weight_norm_bwd_f32", isa, m, (grid, 1), seed, a[0], a[1], a[2], (flags & 1) != 0 ? a[3] : a[0],
                            (flags & 2) != 0 ? a[4] : a[1], (ulong)cols, (ulong)flags, Rows), (2, Rows), worst, "weight_norm_bwd_f32");
                }

                // Δ = dout · output per row, as the CPU's attention gradients take it (its Dot): against the rows' dot products.
                float[] o = TrainingRandom(random, n, -1, 1), g = TrainingRandom(random, n, -1, 1);
                float[] delta = [.. Enumerable.Range(0, Rows).Select(r => Enumerable.Range(0, cols).Aggregate(0f, (acc, j) => MathF.FusedMultiplyAdd(g[r * cols + j], o[r * cols + j], acc)))];
                float[] got = RowsOnGcn([o, g, new float[Rows]], (m, a) => TrainingDispatch("attn_bwd_d_f32", isa, m, (2, 1), 3, a[0], a[1], a[2], (ulong)cols, Rows))[2];
                worst["attn_bwd_d_f32"] = MathF.Max(worst.GetValueOrDefault("attn_bwd_d_f32"), RowsCompare(delta, got, $"attn_bwd_d_f32 ({isa}), {Rows} x {cols}"));
            }
        }

        TrainingReport(worst);
    }

    internal static void GcnTrainingSums(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(58);
        foreach (var isa in TrainingIsas)
        {
            foreach (int cols in (int[])[1, 300, 1000])
            {
                const int Rows = 5;
                int blocks = (cols + 255) / 256;
                TrainingCase($"sum_rows_f32 ({isa}), {Rows} x {cols}", [TrainingRandom(random, Rows * cols, -2, 2), TrainingRandom(random, cols, -1, 1)], [1],
                    (b, s) => b.SumRowsKernel(s[0], s[1], Rows, cols),
                    (m, a, grid, seed) => TrainingDispatch("sum_rows_f32", isa, m, (grid, 1), seed, a[0], a[1], Rows, (ulong)cols), (1, blocks + 1), worst, "sum_rows_f32", exact: true);
            }

            foreach (int rows in (int[])[3, 150])
            {
                const int Cols = 300, Ld = 310, Offset = 7, Chunk = 64;
                int chunks = (rows + Chunk - 1) / Chunk, blocks = (Cols + 255) / 256;
                TrainingCase($"sum_cols_strided_f32{(chunks > 1 ? " + fold_chunks_f32" : "")} ({isa}), {rows} x {Cols} (ld {Ld}), {chunks} chunks",
                    [TrainingRandom(random, Offset + rows * Ld, -2, 2), TrainingRandom(random, Cols, -1, 1), new float[chunks * Cols]], [1],
                    (b, s) => b.SumColumnsKernel(s[0], Offset, Ld, s[1], rows, Cols),
                    (m, a, grid, seed) =>
                    {
                        TrainingDispatch("sum_cols_strided_f32", isa, m, (grid, chunks), seed, a[0] + 4 * Offset, a[1], chunks > 1 ? a[2] : a[1], (ulong)rows, Cols, Ld, Chunk);
                        if (chunks > 1)
                        {
                            TrainingDispatch("fold_chunks_f32", isa, m, (grid, 1), seed, a[2], a[1], Cols, (ulong)chunks, Cols);
                        }
                    }, (1, blocks), worst, "sum_cols_strided_f32");
            }

            foreach (int n in (int[])[100, 70_000])
            {
                int blocks = Math.Min((n + 255) / 256, 1024);
                TrainingCase($"sum_f32{(blocks > 1 ? " + sum_blocks_f32" : "")} ({isa}), {n} elements, {blocks} blocks", [TrainingRandom(random, n, -1, 1), [3f], new float[blocks]], [1],
                    (b, s) => b.SumKernel(s[0], s[1], n, 0.5f),
                    (m, a, grid, seed) =>
                    {
                        TrainingDispatch("sum_f32", isa, m, (blocks, 1), seed, a[0], a[1], blocks > 1 ? a[2] : a[1], (ulong)n, Bits(0.5f));
                        if (blocks > 1)
                        {
                            TrainingDispatch("sum_blocks_f32", isa, m, (grid, 1), seed, a[2], a[1], (ulong)blocks, Bits(0.5f), 0UL, 1UL);
                        }
                    }, (1, 2), worst, "sum_f32");
                TrainingCase($"sumsq_f32{(blocks > 1 ? " + sum_blocks_f32" : "")} ({isa}), {n} elements, {blocks} blocks", [TrainingRandom(random, n, -1, 1), [3f], new float[blocks]], [1],
                    (b, s) => b.SumSquaresKernel(s[0], s[1], n),
                    (m, a, grid, seed) =>
                    {
                        TrainingDispatch("sumsq_f32", isa, m, (blocks, 1), seed, a[0], a[1], blocks > 1 ? a[2] : a[1], (ulong)n, blocks > 1 ? 0UL : 1UL, (ulong)blocks);
                        if (blocks > 1)
                        {
                            TrainingDispatch("sum_blocks_f32", isa, m, (grid, 1), seed, a[2], a[1], (ulong)blocks, Bits(1f), 1UL, 1UL);
                        }
                    }, (1, 2), worst, "sumsq_f32");
            }

            foreach (int dim in (int[])[64, 1000])
            {
                const int Outer = 4;
                foreach (bool accumulate in (bool[])[false, true])
                {
                    TrainingCase($"sum_axis_rows_f32{(accumulate ? " (adding)" : "")} ({isa}), {Outer} x {dim}", [TrainingRandom(random, Outer * dim, -2, 2), TrainingRandom(random, Outer, -1, 1)], [1],
                        (b, s) => b.SumAxisKernel(s[0], s[1], Outer, dim, 1, 0.25f, accumulate),
                        (m, a, grid, seed) => TrainingDispatch("sum_axis_rows_f32", isa, m, (grid, 1), seed, a[0], a[1], (ulong)dim, Bits(0.25f), accumulate ? 1UL : 0UL, Outer),
                        (2, Outer), worst, "sum_axis_rows_f32");
                }
            }
        }

        TrainingReport(worst);
    }

    internal static void GcnTrainingScatter(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(59);
        const int Count = 300, Dim = 13, Vocabulary = 37;
        float[] ids = [.. Enumerable.Range(0, Count).Select(_ => (float)random.Next(Vocabulary))];
        ids[5] = Vocabulary - 1;
        foreach (var isa in TrainingIsas)
        {
            TrainingCase($"scatter_chain_u32 + scatter_walk_f32 ({isa}), {Count} rows of {Dim} into {Vocabulary} ids",
                [TrainingRandom(random, Count * Dim, -1, 1), ids, TrainingRandom(random, Vocabulary * Dim, -1, 1), new float[Count]], [2],
                (b, s) => b.ScatterAddKernel(s[0], s[1], s[2], Count, Dim, Vocabulary),
                (m, a, grid, seed) =>
                {
                    int blocks = (Count + 255) / 256;
                    TrainingDispatch("scatter_chain_u32", isa, m, (Math.Min(grid, blocks), 1), seed, a[1], a[3], Count, Vocabulary - 1, (ulong)blocks);
                    TrainingDispatch("scatter_walk_f32", isa, m, (grid, 1), seed, a[0], a[3], a[1], a[2], Dim, Vocabulary - 1, Count, Count * Dim);
                }, (1, 4), worst, "scatter_chain_u32 + scatter_walk_f32", exact: true);
        }

        TrainingReport(worst);
    }

    internal static void GcnTrainingGroupStats(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(60);
        const float Eps = 1e-5f;
        foreach (var isa in TrainingIsas)
        {
            foreach (var (outer, groups, inner) in new[] { (3, 4, 50), (20, 3, 1), (2, 2, 9000) })
            {
                int n = outer * groups * inner, m = outer * inner;
                int chunks = (m + GcnKernels.GroupChunk - 1) / GcnKernels.GroupChunk;
                bool split = m >= 2 * GcnKernels.GroupChunk;
                string shape = $"({isa}), {outer} x {groups} x {inner}{(split ? $", {chunks} chunks a group" : "")}";
                float[] x = TrainingRandom(random, n, -1.5f, 2.5f), b2 = TrainingRandom(random, n, -1, 1);
                TrainingCase($"norm_stats{(split ? "_part_f32 + norm_stats_final_f32" : "_f32")} {shape}", [x, new float[groups], new float[groups], new float[groups], new float[2 * chunks * groups]],
                    [1, 2, 3], (b, s) => b.NormStatsKernel(s[0], s[1], s[2], s[3], outer, groups, inner, Eps),
                    (mem, a, grid, seed) =>
                    {
                        if (split)
                        {
                            TrainingDispatch("norm_stats_part_f32", isa, mem, (grid, 1), seed, a[0], a[4], (ulong)outer, (ulong)groups, (ulong)inner, (ulong)chunks);
                            TrainingDispatch("norm_stats_final_f32", isa, mem, (Math.Min(grid, groups), 1), seed, a[4], a[1], a[2], a[3], (ulong)outer, (ulong)groups, (ulong)inner,
                                (ulong)chunks, Bits(Eps));
                        }
                        else
                        {
                            TrainingDispatch("norm_stats_f32", isa, mem, (Math.Min(grid, groups), 1), seed, a[0], a[1], a[2], a[3], (ulong)outer, (ulong)groups, (ulong)inner, Bits(Eps));
                        }
                    }, (1, split ? chunks * groups : groups), worst, split ? "norm_stats_part_f32 + norm_stats_final_f32" : "norm_stats_f32");
                foreach (bool hasB in (bool[])[true, false])
                {
                    TrainingCase($"group_reduce{(split ? "_part_f32 + group_reduce_final_f32" : "_f32")}{(hasB ? "" : " (a alone)")} {shape}",
                        [x, b2, TrainingRandom(random, groups, -1, 1), TrainingRandom(random, groups, -1, 1), new float[2 * chunks * groups]], [2, 3],
                        (b, s) => b.GroupReduceKernel(s[0], hasB ? s[1] : null, s[2], hasB ? s[3] : null, outer, groups, inner),
                        (mem, a, grid, seed) =>
                        {
                            ulong sab = hasB ? a[3] : a[2], bb = hasB ? a[1] : a[0];
                            if (split)
                            {
                                TrainingDispatch("group_reduce_part_f32", isa, mem, (grid, 1), seed, a[0], bb, a[4], (ulong)outer, (ulong)groups, (ulong)inner, (ulong)chunks, hasB ? 1UL : 0UL);
                                TrainingDispatch("group_reduce_final_f32", isa, mem, (Math.Min(grid, groups), 1), seed, a[4], a[2], sab, (ulong)groups, (ulong)chunks, hasB ? 1UL : 0UL);
                            }
                            else
                            {
                                TrainingDispatch("group_reduce_f32", isa, mem, (Math.Min(grid, groups), 1), seed, a[0], bb, a[2], sab, (ulong)outer, (ulong)groups, (ulong)inner, hasB ? 1UL : 0UL);
                            }
                        }, (1, split ? chunks * groups : groups), worst, split ? "group_reduce_part_f32 + group_reduce_final_f32" : "group_reduce_f32");
                }
            }
        }

        TrainingReport(worst);
    }

    internal static void GcnTrainingAttentionTiled(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(61);
        const int Heads = 2, Rph = 12, Steps = 6, Cap = 9;
        const float Scale = 0.15f;
        int rows = Heads * Rph;
        foreach (int dim in (int[])[37, 64, 300])
        {
            foreach (var variant in new[] { new AttentionVariant(0, 0f), new AttentionVariant(3, 0f), new AttentionVariant(0, 4f), new AttentionVariant(2, 3f) })
            {
                if (dim == 300 && !variant.IsPlain)
                {
                    continue;
                }

                float[] q = TrainingRandom(random, rows * dim, -1, 1), k = TrainingRandom(random, Heads * Cap * dim, -1, 1), v = TrainingRandom(random, Heads * Cap * dim, -1, 1);
                float[][] forward = RowsOnCpu([q, k, v, [0f], new float[rows * dim], new float[rows]],
                    (b, s) => b.AttentionTiledKernel(s[0], s[1], s[2], s[3], s[4], s[5], Heads, Rph, Steps, Cap, dim, Scale, variant));
                float[] dout = TrainingRandom(random, rows * dim, -1, 1);
                foreach (var isa in TrainingIsas)
                {
                    TrainingCase($"attn_bwd_kv_f32 + attn_bwd_q_f32 ({isa}), head {dim}, window {variant.Window}, soft-cap {variant.Softcap}",
                        [q, k, v, forward[4], forward[5], dout, TrainingRandom(random, rows * dim, -1, 1), TrainingRandom(random, Heads * Cap * dim, -1, 1),
                            TrainingRandom(random, Heads * Cap * dim, -1, 1), new float[rows]], [6, 7, 8],
                        (b, s) => b.AttentionTiledBackwardKernel(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], Heads, Rph, Steps, Cap, dim, Scale, variant),
                        (m, a, grid, seed) =>
                        {
                            TrainingDispatch("attn_bwd_d_f32", isa, m, (Math.Min(grid, rows), 1), seed, a[3], a[5], a[9], (ulong)dim, (ulong)rows);
                            ulong[] Args(int n) => [a[0], a[1], a[2], a[5], a[4], a[9], a[6], a[7], a[8], Rph, Steps, Cap, (ulong)dim, Bits(Scale), (ulong)variant.Window,
                                Bits(variant.Softcap), (ulong)n];
                            TrainingDispatch("attn_bwd_kv_f32", isa, m, (Math.Min(grid, Heads * Cap), 1), seed, Args(Heads * Cap));
                            TrainingDispatch("attn_bwd_q_f32", isa, m, (Math.Min(grid, rows), 1), seed, Args(rows));
                        }, (2, rows), worst, "attn_bwd_d_f32 + attn_bwd_kv_f32 + attn_bwd_q_f32");
                }
            }
        }

        TrainingReport(worst);
    }

    internal static void GcnTrainingAttentionSpans(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(62);
        const int Heads = 4, KvHeads = 2, Hpt = 2, Rows = 9, KeyRows = 11;
        const float Scale = 0.2f;
        int tables = Heads / Hpt;
        float[] starts = [.. Enumerable.Range(0, tables * Rows).Select(i => (float)(i % 3))], ends = [.. Enumerable.Range(0, tables * Rows).Select(i => (float)Math.Min(KeyRows, 3 + i % Rows))];
        ends[4] = 1f;                                                                    // row 4 of the first table: an empty range (starts at 1)
        foreach (int dim in (int[])[37, 64])
        {
            foreach (float softcap in (float[])[0f, 5f])
            {
                var variant = new AttentionVariant(0, softcap);
                float[] q = TrainingRandom(random, Heads * Rows * dim, -1, 1), k = TrainingRandom(random, KvHeads * KeyRows * dim, -1, 1), v = TrainingRandom(random, KvHeads * KeyRows * dim, -1, 1);
                float[][] forward = RowsOnCpu([q, k, v, starts, ends, new float[Heads * Rows * dim], new float[Heads * Rows]],
                    (b, s) => b.AttentionSpansKernel(s[0], s[1], s[2], s[3], s[4], s[5], s[6], Heads, KvHeads, Hpt, Rows, KeyRows, dim, Scale, variant));
                float[] dout = TrainingRandom(random, Heads * Rows * dim, -1, 1);
                foreach (var isa in TrainingIsas)
                {
                    TrainingCase($"attn_spans_bwd_kv_f32 + attn_spans_bwd_q_f32 ({isa}), head {dim}, soft-cap {softcap}",
                        [q, k, v, starts, ends, forward[5], forward[6], dout, TrainingRandom(random, Heads * Rows * dim, -1, 1), TrainingRandom(random, KvHeads * KeyRows * dim, -1, 1),
                            TrainingRandom(random, KvHeads * KeyRows * dim, -1, 1), new float[Heads * Rows]], [8, 9, 10],
                        (b, s) => b.AttentionSpansBackwardKernel(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10], Heads, KvHeads, Hpt, Rows, KeyRows, dim, Scale, variant),
                        (m, a, grid, seed) =>
                        {
                            int all = Heads * Rows;
                            TrainingDispatch("attn_bwd_d_f32", isa, m, (Math.Min(grid, all), 1), seed, a[5], a[7], a[11], (ulong)dim, (ulong)all);
                            ulong[] Args(int n) => [a[0], a[1], a[2], a[3], a[4], a[7], a[6], a[11], a[8], a[9], a[10], Rows, KeyRows, (ulong)dim, Bits(Scale), Heads / KvHeads, Hpt,
                                Bits(softcap), Heads, (ulong)n];
                            TrainingDispatch("attn_spans_bwd_kv_f32", isa, m, (Math.Min(grid, KvHeads * KeyRows), 1), seed, Args(KvHeads * KeyRows));
                            TrainingDispatch("attn_spans_bwd_q_f32", isa, m, (Math.Min(grid, all), 1), seed, Args(all));
                        }, (2, Heads * Rows), worst, "attn_spans_bwd_kv_f32 + attn_spans_bwd_q_f32");
                }
            }
        }

        TrainingReport(worst);
    }
}
