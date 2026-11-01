// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using Idrak.Gpu.Abstractions;
using Idrak.Gpu.Hip;
using Idrak.Tests.Gcn;

// Plan 35, G6c, in the GCN interpreter: CUDA's tiled implicit convolution products (and the split reduce) against the
// CPU's convolution and its gradients; the count-only maximum-pool gradient (chained and walked, no atomics) bit for bit
// against the CPU's; the recurrent sequence kernels against their step kernels run a step at a time (the same bits: each
// item is the step kernel's block), with the grid barrier's words back at zero arrivals after the launch.
internal abstract partial class Tests
{
    // One dispatch over x × y × z workgroups of 256 (the implicit products' grid).
    private static GcnRunResult G6cDispatch(string name, GcnIsaVersion isa, GcnMemory memory, (int X, int Y, int Z) grid, ulong seed, params ulong[] values)
    {
        var (code, image) = TrainingKernel(name, isa);
        var arguments = new byte[GcnCodeObject.KernargBytes(code)];
        GcnKernels.PackArguments(code.Arguments, arguments, values);
        return GcnInterpreter.Run(image, memory, new GcnDispatch
        {
            Workgroups = grid, WorkgroupSize = (GcnKernels.RowThreads, 1, 1), Arguments = arguments, Seed = seed,
        });
    }

    // The 18 geometry values of the implicit products (CUDA's order).
    private static ulong[] ImplicitArgs(ulong[] pointers, in ConvGeometry g, int filters, int groups, int flags, int splits) =>
        [.. pointers, Cu(g.N), Cu(g.C), Cu(g.H), Cu(g.W), Cu(g.KH), Cu(g.KW), Cu(g.SH), Cu(g.SW), Cu(g.PH), Cu(g.PW), Cu(g.OH), Cu(g.OW), Cu(g.DH), Cu(g.DW),
            Cu(filters), Cu(groups), Cu(flags), Cu(splits)];

    private static int Tiles(int count) => (count + GcnKernels.ConvTileEdge - 1) / GcnKernels.ConvTileEdge;

    internal static void GcnConvolutionImplicit(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(71);
        foreach (var (geometry, filters, groups) in new[]
        {
            (new ConvGeometry(2, 4, 7, 6, 3, 3, 1, 1, 1, 1), 6, 2),
            (new ConvGeometry(1, 3, 9, 8, 3, 2, 2, 1, 1, 0) { DH = 2 }, 20, 1),
            (new ConvGeometry(2, 2, 5, 11, 2, 3, 2, 2, 0, 1) { DW = 2 }, 17, 1),
        })
        {
            var g = geometry;
            int input = g.N * g.C * g.H * g.W, outputs = g.N * filters * g.OH * g.OW, patch = g.PatchSize / groups, weights = filters * patch;
            int fg = filters / groups, cg = g.C / groups;
            string shape = $"{Describe(g)}, {filters} filters, {groups} groups";
            float[] x = TrainingRandom(random, input, -1, 1), w = TrainingRandom(random, weights, -1, 1), bias = TrainingRandom(random, filters, -0.5f, 0.5f);
            float[] dy = TrainingRandom(random, outputs, -1, 1);
            foreach (var isa in TrainingIsas)
            {
                // Forward: tiles of (columns: output positions, rows: the group's filters), entries (n, q) along z; the first
                // run on one z (the kernel loops over the entries), the second on all of them.
                foreach (var activation in new[] { ConvActivation.None, ConvActivation.Relu, ConvActivation.Gelu })
                {
                    bool hasBias = activation != ConvActivation.Relu;
                    int flags = (hasBias ? 1 : 0) | (int)activation << 1;
                    TrainingCase($"conv_fwd_tile_f32 ({isa}), {shape}, {activation}{(hasBias ? ", a bias" : "")}", [x, w, bias, new float[outputs]], [3],
                        (b, s) => b.Convolution(s[0], s[1], hasBias ? s[2] : null, s[3], g, filters, groups, activation),
                        (m, a, z, seed) => G6cDispatch("conv_fwd_tile_f32", isa, m, (Tiles(g.OH * g.OW), Tiles(fg), z), seed,
                            ImplicitArgs([a[0], a[1], hasBias ? a[2] : a[1], a[3]], g, filters, groups, flags, 1)),
                        (1, g.N * groups), worst, "conv_fwd_tile_f32");
                }

                TrainingCase($"conv_bwd_input_tile_f32 ({isa}), {shape}", [dy, w, TrainingRandom(random, input, -1, 1)], [2],
                    (b, s) => b.ConvolutionBackwardInput(s[0], s[1], s[2], g, filters, groups),
                    (m, a, z, seed) => G6cDispatch("conv_bwd_input_tile_f32", isa, m, (Tiles(g.H * g.W), Tiles(cg), z), seed,
                        ImplicitArgs([a[0], a[1], a[2]], g, filters, groups, 0, 1)),
                    (1, g.N * groups), worst, "conv_bwd_input_tile_f32");

                // Weight: one split added to dw; four splits written to part, then added to dw in split order.
                TrainingCase($"conv_bwd_weight_tile_f32 ({isa}), {shape}, one split", [x, dy, TrainingRandom(random, weights, -1, 1), new float[1]], [2],
                    (b, s) => b.ConvolutionBackwardWeight(s[0], s[1], s[2], g, filters, groups),
                    (m, a, z, seed) => G6cDispatch("conv_bwd_weight_tile_f32", isa, m, (Tiles(patch), Tiles(fg), z), seed,
                        ImplicitArgs([a[0], a[1], a[2], a[2]], g, filters, groups, 0, 1)),
                    (1, groups), worst, "conv_bwd_weight_tile_f32");
                const int Splits = 4;
                TrainingCase($"conv_bwd_weight_tile_f32 + conv_split_reduce_f32 ({isa}), {shape}, {Splits} splits",
                    [x, dy, TrainingRandom(random, weights, -1, 1), new float[Splits * weights]], [2],
                    (b, s) => b.ConvolutionBackwardWeight(s[0], s[1], s[2], g, filters, groups),
                    (m, a, z, seed) =>
                    {
                        G6cDispatch("conv_bwd_weight_tile_f32", isa, m, (Tiles(patch), Tiles(fg), z), seed, ImplicitArgs([a[0], a[1], a[2], a[3]], g, filters, groups, 0, Splits));
                        TrainingDispatch("conv_split_reduce_f32", isa, m, (Math.Max(1, (weights + 255) / 256), 1), seed, a[3], a[2], Cu(weights), Cu(Splits), Cu(weights));
                    },
                    (1, groups * Splits), worst, "conv_split_reduce_f32");
            }
        }

        TrainingReport(worst);
    }

    internal static void GcnConvolutionMaxPoolChain(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(72);
        foreach (var (count, length) in new[] { (1, 1), (37, 20), (300, 64), (700, 999) })
        {
            // argmax as the CPU's maximum pool writes it: raw int bits, positions repeated (overlapping windows).
            float[] argmax = [.. Enumerable.Range(0, count).Select(_ => BitConverter.Int32BitsToSingle(random.Next(length)))];
            foreach (var isa in TrainingIsas)
            {
                int blocks = (count + GcnKernels.RowThreads - 1) / GcnKernels.RowThreads;
                TrainingCase($"maxpool_bwd_chain_u32 + maxpool_bwd_walk_f32 ({isa}), {count} gradients into {length}",
                    [TrainingRandom(random, count, -1, 1), argmax, TrainingRandom(random, length, -1, 1), new float[count]], [2],
                    (b, s) => b.MaxPoolBackward(s[0], s[1], s[2], count),
                    (m, a, grid, seed) =>
                    {
                        TrainingDispatch("maxpool_bwd_chain_u32", isa, m, (Math.Min(grid, blocks), 1), seed, a[1], a[3], Cu(count), Cu(length - 1), Cu(blocks));
                        TrainingDispatch("maxpool_bwd_walk_f32", isa, m, (grid, 1), seed, a[0], a[3], a[1], a[2], 1, Cu(length - 1), Cu(count), Cu(count));
                    },
                    (1, blocks + 2), worst, "maxpool_bwd_walk_f32", exact: true);
            }
        }

        TrainingReport(worst);
    }

    internal static void GcnConvolutionSequences(Device device)
    {
        _ = device;
        var random = new Random(73);
        const int Batch = 2, Steps = 3, H = 70;
        int blocks = Batch * ((H + GcnKernels.StepUnits - 1) / GcnKernels.StepUnits);
        float[] Fresh(int count) => TrainingRandom(random, count, -1, 1);
        float[] Small(int count) => TrainingRandom(random, count, -0.2f, 0.2f);
        float[] Gate(int count) => TrainingRandom(random, count, 0.05f, 0.95f);
        foreach (bool reverse in new[] { false, true })
        {
            int reversed = reverse ? GcnKernels.SequenceReverse : 0;
            foreach (var isa in TrainingIsas)
            {
                // The step taken k-th, forward: t = k (reverse: T-1-k), the previous step when k > 0.
                (int Step, int Previous) Forward(int k)
                {
                    int t = reverse ? Steps - 1 - k : k;
                    return (t, k == 0 ? -1 : reverse ? t + 1 : t - 1);
                }

                // Backward, the steps from the last taken: t = T-1-k (reverse: k), the previous step unless it is the first
                // taken, the later step unless it is the last taken.
                (int Step, int Previous, int Next) Backward(int k)
                {
                    int t = reverse ? k : Steps - 1 - k;
                    return (t, k == Steps - 1 ? -1 : reverse ? t + 1 : t - 1, k == 0 ? -1 : reverse ? t - 1 : t + 1);
                }

                string how = $"({isa}), {Batch} x {H}, {Steps} steps{(reverse ? ", reversed" : "")}";
                SequenceCase($"lstm_seq_f32 {how}",
                    [Fresh(Batch * Steps * 4 * H), Small(H * 4 * H), Fresh(Batch * H), Fresh(Batch * Steps * H), Fresh(Batch * Steps * 4 * H), Fresh(Batch * Steps * H)], [2, 3, 4, 5],
                    (m, a, seed) =>
                    {
                        for (int k = 0; k < Steps; k++)
                        {
                            var (t, previous) = Forward(k);
                            TrainingDispatch("lstm_step_f32", isa, m, (blocks, 1), seed, a[0], a[1], a[2], a[3], a[4], a[5], Cu(t), Cu(Math.Max(previous, 0)), Steps, H,
                                Cu(3 | (previous >= 0 ? 4 : 0)));
                        }
                    },
                    (m, a, barrier, seed) => G6cDispatch("lstm_seq_f32", isa, m, (1, 1, 1), seed, a[0], a[1], a[2], a[3], a[4], a[5], barrier, Steps, H, Cu(3 | reversed), Batch));
                SequenceCase($"gru_seq_f32 {how}",
                    [Fresh(Batch * Steps * 3 * H), Small(H * 3 * H), Fresh(H), Fresh(Batch * Steps * H), Fresh(Batch * Steps * 4 * H)], [3, 4],
                    (m, a, seed) =>
                    {
                        for (int k = 0; k < Steps; k++)
                        {
                            var (t, previous) = Forward(k);
                            TrainingDispatch("gru_step_f32", isa, m, (blocks, 1), seed, a[0], a[1], a[2], a[3], a[4], Cu(t), Cu(Math.Max(previous, 0)), Steps, H,
                                Cu(3 | (previous >= 0 ? 4 : 0)));
                        }
                    },
                    (m, a, barrier, seed) => G6cDispatch("gru_seq_f32", isa, m, (1, 1, 1), seed, a[0], a[1], a[2], a[3], a[4], barrier, Steps, H, Cu(3 | reversed), Batch));
                SequenceCase($"lstm_seq_bwd_f32 {how}",
                    [Gate(Batch * Steps * 4 * H), Fresh(Batch * Steps * H), Fresh(Batch * Steps * H), Small(4 * H * H), Fresh(Batch * H), Fresh(Batch * Steps * 4 * H)], [4, 5],
                    (m, a, seed) =>
                    {
                        for (int k = 0; k < Steps; k++)
                        {
                            var (t, previous, next) = Backward(k);
                            TrainingDispatch("lstm_step_bwd_f32", isa, m, (blocks, 1), seed, a[0], a[1], a[2], a[3], a[4], a[5], Cu(t), Cu(Math.Max(next, 0)),
                                Cu(Math.Max(previous, 0)), Steps, H, Cu(1 | (previous >= 0 ? 2 : 0) | (next >= 0 ? 4 : 0)));
                        }
                    },
                    (m, a, barrier, seed) => G6cDispatch("lstm_seq_bwd_f32", isa, m, (1, 1, 1), seed, a[0], a[1], a[2], a[3], a[4], a[5], barrier, Steps, H, Cu(1 | reversed), Batch));
                SequenceCase($"gru_seq_bwd_f32 {how}",
                    [Gate(Batch * Steps * 4 * H), Fresh(Batch * Steps * H), Fresh(Batch * Steps * H), Small(3 * H * H), Fresh(Batch * H), Fresh(Batch * Steps * 3 * H),
                        Fresh(Batch * Steps * 3 * H)], [4, 5, 6],
                    (m, a, seed) =>
                    {
                        for (int k = 0; k < Steps; k++)
                        {
                            var (t, previous, next) = Backward(k);
                            TrainingDispatch("gru_step_bwd_f32", isa, m, (blocks, 1), seed, a[0], a[1], a[2], a[3], a[4], a[5], a[6], Cu(t), Cu(Math.Max(next, 0)),
                                Cu(Math.Max(previous, 0)), Steps, H, Cu(1 | (previous >= 0 ? 2 : 0) | (next >= 0 ? 4 : 0)));
                        }
                    },
                    (m, a, barrier, seed) => G6cDispatch("gru_seq_bwd_f32", isa, m, (1, 1, 1), seed, a[0], a[1], a[2], a[3], a[4], a[5], a[6], barrier, Steps, H, Cu(1 | reversed),
                        Batch));
            }
        }
    }

    // A sequence kernel against its step kernel run a step at a time: the outputs the same bits, on two wave orders; the
    // grid barrier (a storage of two words after the inputs) back at zero arrivals, its generation moved once a step gap.
    private static void SequenceCase(string what, float[][] inputs, int[] outputs, Action<GcnMemory, ulong[], ulong> steps,
        Action<GcnMemory, ulong[], ulong, ulong> sequence)
    {
        float[][] expected = RowsOnGcn(inputs, (m, a) => steps(m, a, 1));
        float[][] withBarrier = [.. inputs, new float[2]];
        foreach (ulong seed in new ulong[] { 1, 7 })
        {
            float[][] got = RowsOnGcn(withBarrier, (m, a) => sequence(m, a[..^1], a[^1], seed));
            foreach (int o in outputs)
            {
                for (int i = 0; i < expected[o].Length; i++)
                {
                    uint e = BitConverter.SingleToUInt32Bits(expected[o][i]), g = BitConverter.SingleToUInt32Bits(got[o][i]);
                    Check(e == g, $"{what}, seed {seed}, storage {o}: element {i} is {got[o][i]:R} (0x{g:X8}), the step kernels' {expected[o][i]:R} (0x{e:X8})");
                }
            }

            uint arrivals = BitConverter.SingleToUInt32Bits(got[^1][0]), generation = BitConverter.SingleToUInt32Bits(got[^1][1]);
            Check(arrivals == 0 && generation == (uint)(SequenceSteps(what) - 1),
                $"{what}, seed {seed}: the grid barrier left {arrivals} arrivals (0: it resets itself) and generation {generation} (one a step gap)");
        }
    }

    // The steps a sequence case's name gives ("…, 3 steps…").
    private static int SequenceSteps(string what)
    {
        int at = what.IndexOf(" steps", StringComparison.Ordinal);
        int start = what.LastIndexOf(' ', at - 1) + 1;
        return int.Parse(what.AsSpan(start, at - start), System.Globalization.CultureInfo.InvariantCulture);
    }
}
