// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using Idrak.Gpu.Abstractions;
using Idrak.Gpu.Hip;
using Idrak.Tests.Gcn;

// HIP's mixture-of-experts kernels in GCN (GcnKernels.Routing.cs, plan 37 R0b), each run by the GCN interpreter and proven
// against the CPU backend's RouteTopK, ExpertGatedPair and ExpertMatMul. Every kernel is assembled for gfx90c, gfx906,
// gfx1030 and gfx1100; the routing (int32 choices, offsets, tokens and rows, as T1b's index buffers) and the experts'
// products equal the CPU's bit for bit, as does the ReLU gated pair (its activation exact); the other gated activations
// equal it within the device-against-CPU tolerance (their exponential is the device's, as gated_*_f32's). Each runs
// again on another grid and wave order to give the same bits, and the interpreter reports no missing wait, race or
// access out of range (a report fails the run).
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] GcnRoutingGroup =
    [
        ("gcn kernels: routing: every routing and expert kernel assembles for gfx90c, gfx906, gfx1030 and gfx1100 with its block of 256 declared, within its register budget, its printed text reading back to its bytes, with its profile cost; GcnKernels.All lists each once", GcnRoutingAssembled),
        ("gcn kernels: routing: route_choose_f32, route_counts_f32, route_offsets_f32 and route_place_f32 equal the CPU's RouteTopK bit for bit (weights normalized or not, ties to the lowest index, k of 1 and k of every expert, a NaN and -inf, tokens past two blocks, choices left as garbage before), the same bits on two grids and wave orders", GcnRoutingRoute),
        ("gcn kernels: routing: expert_matmul_f32 and expert_gated_{silu,gelu,relu,tanh}_f32 equal the CPU's ExpertMatMul and ExpertGatedPair over rows grouped by expert (an empty expert, biases and none, odd widths, outputs past four blocks): the products and the ReLU pair bit for bit, the other activations within tolerance, the same bits on two grids and wave orders", GcnRoutingExperts),
    ];

    private static readonly Dictionary<(string, GcnIsaVersion), (GcnKernelCode Code, GcnKernelImage Image)> routingKernels = [];

    // The kernel assembled for isa, written into a code object and read back (once a process).
    private static (GcnKernelCode Code, GcnKernelImage Image) RoutingKernel(string name, GcnIsaVersion isa)
    {
        lock (routingKernels)
        {
            if (routingKernels.TryGetValue((name, isa), out var known))
            {
                return known;
            }

            var source = GcnKernels.Routing.Single(k => k.Name == name);
            var target = KernelTarget(isa);
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
            return routingKernels[(name, isa)] = (code, image);
        }
    }

    // One dispatch of a routing kernel: its arguments as 64-bit values (addresses, 32-bit values) in order.
    private static void RoutingDispatch(string name, GcnIsaVersion isa, GcnMemory memory, int blocks, ulong seed, params ulong[] values)
    {
        var (code, image) = RoutingKernel(name, isa);
        var arguments = new byte[GcnCodeObject.KernargBytes(code)];
        GcnKernels.PackArguments(code.Arguments, arguments, values);
        GcnInterpreter.Run(image, memory, new GcnDispatch
        {
            Workgroups = (blocks, 1, 1), WorkgroupSize = (GcnKernels.ShapeBlock, 1, 1), Arguments = arguments, Seed = seed, ShuffleGroups = seed != 1,
        });
    }

    internal static void GcnRoutingAssembled(Device device)
    {
        _ = device;
        var names = GcnKernels.Routing.Select(k => k.Name).ToArray();
        Check(names.Length == 9 && names.Distinct().Count() == names.Length, $"9 routing and expert kernels, each once: {string.Join(", ", names)}");
        foreach (string name in names)
        {
            Check(GcnKernels.All.Count(k => k.Name == name) == 1, $"GcnKernels.All lists {name} once");
            Check(Encoding.UTF8.GetString(GcnKernels.Routing.Single(k => k.Name == name).Text).StartsWith($".kernel {name}\n", StringComparison.Ordinal),
                $"{name}'s text starts with its .kernel line");
            Check(HipBackend.Cost(name).Bytes > 0, $"{name} has its profile cost (HipBackend.RoutingCost)");
            foreach (var isa in AttentionIsas)
            {
                var (code, _) = RoutingKernel(name, isa);
                Check(code is { RequiredWorkgroupSize: (256, 1, 1), MaxFlatWorkgroupSize: 256, WaveSize: 64, LdsBytes: 0 },
                    $"{name} ({isa}): declares the block of 256 and no LDS (required {code.RequiredWorkgroupSize}, max flat {code.MaxFlatWorkgroupSize}, wave {code.WaveSize}, LDS {code.LdsBytes})");
                Console.WriteLine($"    {name} ({isa}): {code.Code.Length} bytes of code, {code.Vgprs} VGPRs, {code.Sgprs} SGPRs");
            }
        }
    }

    internal static void GcnRoutingRoute(Device device)
    {
        _ = device;
        var random = new Random(91);
        // Probabilities on a grid of eighths, so ties are common; one token with a NaN and -inf.
        (int Tokens, int Experts, int K, bool Normalize)[] cases = [(600, 7, 3, true), (5, 4, 4, false), (3, 9, 1, true), (40, 6, 2, false)];
        foreach (var (tokens, experts, k, normalize) in cases)
        {
            float[] probabilities = [.. Enumerable.Range(0, tokens * experts).Select(_ => random.Next(0, 9) / 8f)];
            if (tokens >= 3)
            {
                probabilities[experts + 1] = float.NaN;
                probabilities[2 * experts] = float.NegativeInfinity;
            }

            int slots = tokens * k;
            int[] garbage = [.. Enumerable.Range(0, slots).Select(i => i % 3 == 0 ? -7 : random.Next(0, 1000))];
            int blocks = ShapeGrid(tokens) + 1;
            foreach (var isa in AttentionIsas)
            {
                // probabilities, weights, offsets, tokenOfRow, slotRow (the CPU's), then route_choose's and route_counts' scratch.
                float[][] inputs = [probabilities, new float[slots], Int32Words(new int[experts + 1]), Int32Words(new int[slots]), Int32Words(new int[slots]),
                    Int32Words(garbage), Int32Words(new int[experts])];
                ShapesCase($"route_*_f32 ({isa}), {tokens} tokens, {experts} experts, k {k}, normalize {normalize}", inputs, blocks, [1, 2, 3, 4],
                    (b, s) => b.RouteTopKKernel(s[0], s[1], s[2], s[3], s[4], tokens, experts, k, normalize),
                    (m, a, grid, seed) =>
                    {
                        RoutingDispatch("route_choose_f32", isa, m, grid.X, seed, a[0], a[1], a[5], (ulong)experts, (ulong)k, normalize ? 1UL : 0UL, (ulong)tokens);
                        RoutingDispatch("route_counts_f32", isa, m, grid.X, seed, a[5], a[6], (ulong)slots, (ulong)experts);
                        RoutingDispatch("route_offsets_f32", isa, m, 1, seed, a[6], a[2], (ulong)slots, (ulong)experts, 1UL);
                        RoutingDispatch("route_place_f32", isa, m, grid.X, seed, a[5], a[2], a[3], a[4], (ulong)slots, (ulong)k, (ulong)experts);
                    });
            }
        }
    }

    internal static void GcnRoutingExperts(Device device)
    {
        _ = device;
        var random = new Random(92);
        const int Experts = 4, Rows = 37, K = 11, N = 29;
        int[] offsets = [0, 5, 5, 30, Rows];                                    // expert 1 has no rows
        float[] x = RowsRandom(random, Rows * K, 2f);
        int blocks = ShapeGrid(Rows * N);
        var worst = new Dictionary<string, float>();

        // The gated pairs: x, offsets, then each expert's gate, gate bias, up and up bias (the gate's absent on odd experts,
        // the up's on expert 0), then hidden.
        float[][] gated = [x, Int32Words(offsets), .. Enumerable.Range(0, Experts).SelectMany(_ => new[]
        {
            RowsRandom(random, K * N, 1f), RowsRandom(random, N, 1f), RowsRandom(random, K * N, 1f), RowsRandom(random, N, 1f),
        }), new float[Rows * N]];
        bool GateBiased(int e) => e % 2 == 0;
        bool UpBiased(int e) => e != 0;
        foreach (var isa in AttentionIsas)
        {
            for (int kind = 0; kind < GcnKernels.ExpertGatedKinds.Length; kind++)
            {
                int activation = kind;
                string name = GcnKernels.ExpertGatedKinds[kind];
                Action<Backend, Storage[]> cpu = (b, s) =>
                {
                    var pairs = new (Storage Gate, Storage? GateBias, Storage Up, Storage? UpBias)[Experts];
                    for (int e = 0; e < Experts; e++)
                    {
                        int at = 2 + 4 * e;
                        pairs[e] = (s[at], GateBiased(e) ? s[at + 1] : null, s[at + 2], UpBiased(e) ? s[at + 3] : null);
                    }

                    b.ExpertGatedPairKernel(s[0], s[1], pairs, s[^1], Rows, K, N, activation);
                };
                Action<GcnMemory, ulong[], (int X, int Y), ulong> gcn = (m, a, grid, seed) =>
                {
                    for (int e = 0; e < Experts; e++)
                    {
                        int at = 2 + 4 * e;
                        RoutingDispatch(name, isa, m, grid.X, seed, a[0], a[1], a[at], GateBiased(e) ? a[at + 1] : 0UL, a[at + 2], UpBiased(e) ? a[at + 3] : 0UL, a[^1],
                            (ulong)e, K, N);
                    }
                };
                string what = $"{name} ({isa}), {Rows} rows of {K} into {N} over {Experts} experts";
                if (kind == 2)
                {
                    ShapesCase(what, gated, blocks, [gated.Length - 1], cpu, gcn);
                }
                else
                {
                    RowsCase(what, gated, blocks, [gated.Length - 1], cpu, gcn, worst, name);
                }
            }

            // The products: x, offsets, then each expert's weight and bias (absent on expert 3), then y.
            float[][] products = [x, Int32Words(offsets), .. Enumerable.Range(0, Experts).SelectMany(_ => new[] { RowsRandom(random, K * N, 1f), RowsRandom(random, N, 1f) }),
                new float[Rows * N]];
            bool Biased(int e) => e != 3;
            ShapesCase($"expert_matmul_f32 ({isa}), {Rows} rows of {K} into {N} over {Experts} experts", products, blocks, [products.Length - 1],
                (b, s) =>
                {
                    var weights = new (Storage Weight, Storage? Bias)[Experts];
                    for (int e = 0; e < Experts; e++)
                    {
                        weights[e] = (s[2 + 2 * e], Biased(e) ? s[3 + 2 * e] : null);
                    }

                    b.ExpertMatMulKernel(s[0], s[1], weights, s[^1], Rows, K, N);
                },
                (m, a, grid, seed) =>
                {
                    for (int e = 0; e < Experts; e++)
                    {
                        RoutingDispatch("expert_matmul_f32", isa, m, grid.X, seed, a[0], a[1], a[2 + 2 * e], Biased(e) ? a[3 + 2 * e] : 0UL, a[^1], (ulong)e, K, N);
                    }
                });
        }

        RowsReport(worst);
    }
}
