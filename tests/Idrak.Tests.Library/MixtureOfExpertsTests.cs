// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Abstraction.Operations;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Text;
using Idrak.Text.Abstractions;
using Idrak.Text.Plugins.Mistral;
using Idrak.Text.Plugins.Qwen;

// Mixture of experts (Mixtral, Qwen2-MoE, Qwen3-MoE): the routing, decoders against the loop reference (prompt, training,
// cached decoding, batches, packed experts), gradients, fine-tuning, and checkpoints in each family's names (safetensors
// and GGUF).
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] MixtureOfExpertsGroup =
    [
        ("mixture of experts: routing takes the k most probable experts (ties to the lowest index, as torch.topk), renormalized when configured; a known routing on the device", ExpertRouting),
        ("mixture of experts: routing where the tensors are (RouteTopK, ExpertGatedPair, ExpertMatMul) gives the host routing's values bit for bit (biases, odd widths, experts with many rows and few, every activation, a shared expert); RouteTopK matches the layer's host routing; a device without the kernels routes on the host, counted as a host fallback of RouteTopK; IDRAK_MOE_ROUTING=host turns it off", ExpertRoutingOnDevice),
        ("mixture of experts: CUDA's routing and expert kernels (PtxKernels.Routing.cs), read on the host: each declares the parameters its launch passes and only the registers it declares, every branch reaches a label of its own kernel, indices are read as int32, the weights divided by div.rn, each product one fma.rn chain from zero with its bias added by add.rn, the gated pair's activation is gated_act_f32's, and an expert launch walks its experts' slots grid-stride", ExpertPtx),
        ("mixture of experts: gradients of router, experts, shared expert and input match finite differences; load-balancing loss; LoRA on attention fine-tunes a model with experts", ExpertTraining),
    ];

    // SmallSpec with experts, shaped like each family.
    internal static (string Name, DecoderSpec Spec)[] ExpertSpecs =>
    [
        ("Mixtral-shaped: 4 experts, 2 per token, renormalized", SmallSpec with { Experts = 4, ExpertsPerToken = 2 }),
        ("Qwen2-MoE-shaped: 6 experts, 3 per token, not renormalized, gated shared expert, qkv biases", SmallSpec with
        {
            Experts = 6, ExpertsPerToken = 3, ExpertFfDim = 10, SharedExpertFfDim = 20, NormalizeTopK = false, QkvBias = true,
        }),
        ("Qwen3-MoE-shaped: 5 experts, 2 per token, q/k norm, a dense first layer", SmallSpec with
        {
            Experts = 5, ExpertsPerToken = 2, ExpertFfDim = 12, QkNorm = true, ExpertLayers = [false, true],
        }),
    ];

    internal static void ExpertRouting(Device device)
    {
        // Ties go to the lowest index; the chosen experts are listed in increasing order.
        float[] probabilities =
        [
            0.1f, 0.3f, 0.3f, 0.3f,
            0.4f, 0.1f, 0.4f, 0.1f,
            0.25f, 0.25f, 0.25f, 0.25f,
            0.05f, 0.15f, 0.2f, 0.6f,
        ];
        var (choices, weights) = MixtureOfExperts.Route(probabilities, 4, 4, 2, normalize: true);
        Check(choices.SequenceEqual([1, 2, 0, 2, 0, 1, 2, 3]), $"choices {string.Join(",", choices)}");
        AssertClose([0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.25f, 0.75f], weights, 1e-6f, "renormalized weights");
        (_, weights) = MixtureOfExperts.Route(probabilities, 4, 4, 2, normalize: false);
        AssertClose([0.3f, 0.3f, 0.4f, 0.4f, 0.25f, 0.25f, 0.2f, 0.6f], weights, 0f, "weights as the probabilities");
        Check(MixtureOfExperts.Route(probabilities, 4, 4, 1, normalize: true).Choices.SequenceEqual([1, 0, 0, 3]), "one per token");
        Check(MixtureOfExperts.Route(probabilities, 4, 4, 4, normalize: true).Choices.SequenceEqual([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]), "every expert");

        // Random rows against a stable sort by decreasing probability.
        var random = new Random(3);
        var rows = Enumerable.Range(0, 50 * 7).Select(_ => (float)Math.Round(random.NextDouble(), 1)).ToArray();     // many ties
        var (picked, _) = MixtureOfExperts.Route(rows, 50, 7, 3, normalize: true);
        for (int i = 0; i < 50; i++)
        {
            var expected = Enumerable.Range(0, 7).OrderByDescending(e => rows[i * 7 + e]).Take(3).Order();
            Check(picked.AsSpan(i * 3, 3).ToArray().SequenceEqual(expected), $"row {i}: {string.Join(",", picked.AsSpan(i * 3, 3).ToArray())}");
        }

        foreach (var (k, what) in new[] { (0, "no expert"), (5, "more experts than there are") })
        {
            try
            {
                using var bad = new MixtureOfExperts(8, 6, 4, k, device: device);
                Check(false, $"{what} should be refused");
            }
            catch (ArgumentOutOfRangeException)
            {
            }
        }

        // A known routing on the device: the router sends token i first to expert i mod 4 (a large score), then, the other
        // scores being equal, to the lowest other index.
        const int Dim = 8, E = 4;
        using var moe = new MixtureOfExperts(Dim, 6, E, 2, normalizeTopK: true, device: device, random: new Random(5));
        var router = new float[Dim * E];
        for (int e = 0; e < E; e++)
        {
            router[e * E + e] = 3f;                                         // input feature e scores expert e
        }

        moe.Router.Weight.CopyFrom(router);
        const int T = 6;
        var x = new float[T * Dim];
        for (int t = 0; t < T; t++)
        {
            x[t * Dim + t % E] = 1f;
            x[t * Dim + 4 + t % 4] = 0.5f;                                  // features the router does not read
        }

        float[] expected2 = new float[T * Dim];
        using (Autograd.NoGrad())
        {
            double big = Math.Exp(3), share = big / (big + 1);            // renormalized: e^3 / (e^3 + e^0)
            for (int t = 0; t < T; t++)
            {
                int first = t % E, second = first == 0 ? 1 : 0;
                using var row = Tensor.From(x.AsSpan(t * Dim, Dim), [1, Dim], device);
                var a = moe.Experts[first].Forward(row).ToArray();
                var b = moe.Experts[second].Forward(row).ToArray();
                for (int i = 0; i < Dim; i++)
                {
                    expected2[t * Dim + i] = (float)(share * a[i] + (1 - share) * b[i]);
                }
            }

            using var input = Tensor.From(x, [1, T, Dim], device);
            AssertClose(expected2, moe.Forward(input).ToArray(), 1e-5f, "the known routing");
        }
    }

    internal static void ExpertRoutingOnDevice(Device device)
    {
        var backend = device.Backend;
        var random = new Random(11);

        // The operation against the layer's host routing (MixtureOfExperts.Route), ties included, bit for bit.
        foreach (var (tokens, experts, k, normalize) in new[] { (50, 7, 3, true), (33, 5, 1, false), (8, 4, 4, true), (1, 3, 2, true) })
        {
            float[] rows = [.. Enumerable.Range(0, tokens * experts).Select(_ => (float)Math.Round(random.NextDouble(), 1) + 0.05f)];
            var (choices, weights) = MixtureOfExperts.Route(rows, tokens, experts, k, normalize);
            int slots = tokens * k;
            using var probabilities = Tensor.From(rows, [tokens, experts], device);
            using var routed = Tensor.Empty([slots], device, track: false);
            using var offsets = Tensor.Empty([experts + 1], device, track: false);
            using var tokenOfRow = Tensor.Empty([slots], device, track: false);
            using var slotRow = Tensor.Empty([slots], device, track: false);
            backend.RouteTopK(probabilities.Storage, routed.Storage, offsets.Storage, tokenOfRow.Storage, slotRow.Storage, tokens, experts, k, normalize);
            var expectedOffsets = new float[experts + 1];
            var expectedTokens = new float[slots];
            var expectedRows = new float[slots];
            int next = 0;
            for (int e = 0; e < experts; e++)
            {
                expectedOffsets[e] = next;
                for (int slot = 0; slot < slots; slot++)
                {
                    if (choices[slot] == e)
                    {
                        expectedTokens[next] = slot / k;
                        expectedRows[slot] = next++;
                    }
                }
            }

            expectedOffsets[experts] = slots;
            string what = $"{tokens} tokens, {experts} experts, k {k}";
            Check(Bits(routed.ToArray()).SequenceEqual(Bits(weights)), $"{what}: the weights of Route, bit for bit");
            Check(offsets.ToArray().SequenceEqual(expectedOffsets) && tokenOfRow.ToArray().SequenceEqual(expectedTokens) && slotRow.ToArray().SequenceEqual(expectedRows),
                $"{what}: offsets {string.Join(",", offsets.ToArray())}, expected {string.Join(",", expectedOffsets)}");
        }

        // Whole layers: the device routing against the host routing (RoutesOnDevice off), bit for bit. 40 tokens over 4 to 6
        // experts give each many rows; 3 tokens give each a few (the few-row products).
        var cases = new (string Name, int Dim, int Hidden, int Experts, int K, bool Normalize, int Shared, FeedForwardActivation Activation, bool Bias, int Tokens)[]
        {
            ("Mixtral-like, SiLU", 16, 24, 4, 2, true, 0, FeedForwardActivation.Silu, false, 40),
            ("odd widths, biases, GELU, a shared expert", 9, 13, 6, 3, false, 7, FeedForwardActivation.Gelu, true, 40),
            ("ReLU, few rows an expert", 8, 6, 5, 2, true, 0, FeedForwardActivation.Relu, true, 3),
            ("tanh · sigmoid, every expert a token", 7, 5, 3, 3, true, 0, FeedForwardActivation.TanhSigmoid, false, 11),
        };
        bool own = backend.RoutesExperts;
        bool expected = device.Type == DeviceType.Cpu || device.Kind is "vulkan" or "hip" or "cuda";
        Check(own == expected, $"{backend.Name}: RoutesExperts {own}, expected {expected} (the CPU, Vulkan, HIP and CUDA have the kernels, plan 37 R0a, R0b and R0c)");
        bool before = MixtureOfExperts.RoutesOnDevice;
        try
        {
            foreach (var c in cases)
            {
                using var moe = new MixtureOfExperts(c.Dim, c.Hidden, c.Experts, c.K, c.Normalize, c.Shared, c.Activation, c.Bias, device, new Random(c.Dim * 31 + c.Hidden));
                if (c.Bias)
                {
                    foreach (var layer in moe.Experts.SelectMany(e => new[] { e.Gate!, e.Up, e.Down }))
                    {
                        layer.Bias!.CopyFrom(Uniform(random, layer.OutFeatures));
                    }
                }

                using var input = Tensor.From(Uniform(random, 2 * c.Tokens * c.Dim), [2, c.Tokens, c.Dim], device);
                float[] Run(bool onDevice, out long routeCalls, out long hostFallbacks)
                {
                    MixtureOfExperts.RoutesOnDevice = onDevice;
                    using var trace = Kernels.Trace(backend);
                    using var scope = new TensorScope();
                    using (Autograd.NoGrad())
                    {
                        var output = moe.Forward(input).ToArray();
                        routeCalls = trace.Calls(Ops.RouteTopK);
                        hostFallbacks = trace.HostCallsByOperation.GetValueOrDefault(nameof(Ops.RouteTopK));
                        return output;
                    }
                }

                var onDevice = Run(true, out long deviceCalls, out long deviceFallbacks);
                var onHost = Run(false, out long hostCalls, out long hostFallbacks);
                // Where the device has the kernels, the host routing runs its experts through the same ExpertGatedPair and
                // ExpertMatMul (plan 37, R0, option B): on a GPU the outputs move in the last bits against the old per-expert
                // Linear products, within the tolerance of "mixture of experts: Mixtral-, Qwen2-MoE- and Qwen3-MoE-shaped
                // decoders match the loop reference" and "pretrained: Mixtral, Qwen2-MoE and Qwen3-MoE configs and checkpoints".
                Check(Bits(onDevice).SequenceEqual(Bits(onHost)), $"{c.Name}: the device routing gives the host routing's values bit for bit (largest difference {onDevice.Zip(onHost, (a, b) => MathF.Abs(a - b)).Max()}; "
                    + "both run the experts through ExpertGatedPair and ExpertMatMul where the device has them, so GPU outputs moved in the last bits against the per-expert Linear products, within the decoder and checkpoint tests' tolerance)");
                Check(own ? deviceCalls == 1 && deviceFallbacks == 0 && hostCalls == 0 && hostFallbacks == 0 : deviceCalls == 0 && deviceFallbacks == 1 && hostFallbacks == 1,
                    $"{c.Name} on {backend.Name}: RouteTopK called {deviceCalls} times on the device path and {hostCalls} with it off; host fallbacks of RouteTopK {deviceFallbacks} and {hostFallbacks}");
            }

            // Training keeps the host routing (its gradients and load-balancing loss read the host's choices).
            MixtureOfExperts.RoutesOnDevice = true;
            using var trained = new MixtureOfExperts(8, 6, 4, 2, device: device, random: new Random(2));
            trained.Train();
            using var x = Tensor.From(Uniform(random, 5 * 8), [5, 8], device);
            using (var trace = Kernels.Trace(backend))
            using (new TensorScope())
            {
                trained.Forward(x).Sum().Backward();
                Check(trace.Calls(Ops.RouteTopK) == 0 && trained.LoadBalancingLoss is not null, "training routes on the host, with its load-balancing loss");
            }
        }
        finally
        {
            MixtureOfExperts.RoutesOnDevice = before;
        }

        static int[] Bits(float[] values) => [.. values.Select(BitConverter.SingleToInt32Bits)];
        static float[] Uniform(Random random, int count) => [.. Enumerable.Range(0, count).Select(_ => random.NextSingle() * 2 - 1)];
    }

    internal static void ExpertPtx(Device device)
    {
        _ = device;
        string source = Idrak.Gpu.Cuda.PtxKernels.Source;
        var entries = Idrak.Gpu.Cuda.PtxKernels.Entries(source);
        var counts = Idrak.Gpu.Cuda.PtxKernels.ParameterCounts;
        // The launches' arguments (CudaBackend.Routing.cs): the routing's pointers, scalars and thread count; an expert
        // launch's x, offsets, output, scalars, then a slot of addresses for each expert it walks (R0d).
        int slots = Idrak.Gpu.Cuda.PtxKernels.ExpertsPerLaunch;
        var expected = new Dictionary<string, int>
        {
            ["route_choose_f32"] = 7, ["route_counts_f32"] = 4, ["route_offsets_f32"] = 5, ["route_place_f32"] = 7, ["expert_gated_f32"] = 8 + 4 * slots,
            ["expert_matmul_f32"] = 7 + 2 * slots,
        };
        Check(Idrak.Gpu.Cuda.PtxKernels.RoutingNames.Order().SequenceEqual(expected.Keys.Order()), $"the routing kernels: {string.Join(", ", Idrak.Gpu.Cuda.PtxKernels.RoutingNames)}");
        foreach (var (name, parameters) in expected)
        {
            int at = entries.FindIndex(e => e.Name == name);
            Check(at >= 0 && entries.Count(e => e.Name == name) == 1, $"{name}: declared once in the main module");
            Check(counts[name] == parameters, $"{name}: {counts[name]} parameters, its launch passes {parameters}");
            string text = source[entries[at].Start..(at + 1 < entries.Count ? entries[at + 1].Start : source.Length)];

            // Registers within what the kernel declares (.reg .x %name<N>), and every branch to a label of this kernel.
            var declared = System.Text.RegularExpressions.Regex.Matches(text, @"\.reg \.\w+ %(\w+?)<(\d+)>")
                .ToDictionary(m => m.Groups[1].Value, m => int.Parse(m.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture));
            foreach (System.Text.RegularExpressions.Match used in System.Text.RegularExpressions.Regex.Matches(text, @"%(rd|r|f|p)(\d+)\b"))
            {
                string file = used.Groups[1].Value;
                int index = int.Parse(used.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture);
                Check(declared.TryGetValue(file, out int size) && index < size, $"{name}: {used.Value} is past the declared %{file}<{(declared.TryGetValue(file, out int s) ? s : 0)}>");
            }

            var labels = System.Text.RegularExpressions.Regex.Matches(text, @"^\s*(\w+):\s*$", System.Text.RegularExpressions.RegexOptions.Multiline).Select(m => m.Groups[1].Value).ToList();
            Check(labels.Count == labels.Distinct().Count(), $"{name}: a label defined twice ({string.Join(", ", labels)})");
            foreach (System.Text.RegularExpressions.Match branch in System.Text.RegularExpressions.Regex.Matches(text, @"\bbra(?:\.uni)?\s+(\w+);"))
            {
                Check(labels.Contains(branch.Groups[1].Value), $"{name}: branches to {branch.Groups[1].Value}, not a label of its own");
            }

            Check(!text.Contains("ld.global.f32 %r", StringComparison.Ordinal) && !text.Contains(".s64 %f", StringComparison.Ordinal), $"{name}: indices read as int32 words, values as floats");
        }

        string Kernel(string name)
        {
            int at = entries.FindIndex(e => e.Name == name);
            return source[entries[at].Start..(at + 1 < entries.Count ? entries[at + 1].Start : source.Length)];
        }

        string choose = Kernel("route_choose_f32"), gated = Kernel("expert_gated_f32"), product = Kernel("expert_matmul_f32");
        Check(choose.Contains("div.rn.f32 %f4, %f2, %f3;", StringComparison.Ordinal) && choose.Contains("add.rn.f32 %f3, %f3, %f2;", StringComparison.Ordinal)
              && choose.Contains("setp.gt.f32 %p3, %f1, %f2;", StringComparison.Ordinal) && choose.Contains("ld.global.u32", StringComparison.Ordinal),
            "route_choose_f32: the sum by add.rn in expert order, the weights by div.rn, a later expert inserted only where strictly more probable (ties to the lowest index), choices as int32");
        Check(product.Contains("fma.rn.f32 %f20, %f22, %f23, %f20;", StringComparison.Ordinal) && product.Contains("add.rn.f32 %f20, %f20, %f22;", StringComparison.Ordinal)
              && product.Contains("mov.f32 %f20, 0f00000000;", StringComparison.Ordinal) && !product.Contains("mul.f32", StringComparison.Ordinal),
            "expert_matmul_f32: one fma.rn chain from zero, the bias added by add.rn (nothing to contract)");
        Check(gated.Contains("fma.rn.f32 %f20, %f22, %f23, %f20;", StringComparison.Ordinal) && gated.Contains("fma.rn.f32 %f21, %f22, %f24, %f21;", StringComparison.Ordinal)
              && gated.Contains("add.rn.f32 %f21, %f21, %f22;", StringComparison.Ordinal),
            "expert_gated_f32: the gate and up chains by fma.rn from zero, their biases by add.rn");
        for (int j = 0; j < slots; j++)
        {
            Check(gated.Contains($"@%p1 ld.param.u64 %rd13, [p_upbias{j}];", StringComparison.Ordinal) && product.Contains($"@%p1 ld.param.u64 %rd11, [p_bias{j}];", StringComparison.Ordinal)
                  && product.Contains($"setp.eq.u32 %p1, %r27, {j};", StringComparison.Ordinal),
                $"expert slot {j}: its addresses read where the walk reaches it");
        }

        Check(product.Contains("add.u32 %r25, %r25, %r29;", StringComparison.Ordinal) && product.Contains("mul.lo.u32 %r29, %r1, %r2;", StringComparison.Ordinal)
              && product.Contains("setp.ge.u32 %p0, %r27, %s_experts;", StringComparison.Ordinal),
            "the expert launches walk their experts, each one's outputs grid-stride over the whole grid");
        string act = Kernel("gated_act_f32");
        int from = act.IndexOf("mov.f32 %f17", StringComparison.Ordinal), to = act.IndexOf("ACT_DONE:", StringComparison.Ordinal);
        Check(from > 0 && to > from && gated.Contains(act[from..to], StringComparison.Ordinal), "expert_gated_f32: the activation of gated_act_f32, line for line");
    }

    internal static void ExpertTraining(Device device)
    {
        // Gradients on the device against central differences of the loss on the CPU.
        const int Dim = 8, T = 5;
        var random = new Random(11);
        var x = Enumerable.Range(0, T * Dim).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        var r = Enumerable.Range(0, T * Dim).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        MixtureOfExperts Make(Device d) => new(Dim, 6, 4, 2, normalizeTopK: true, sharedHidden: 5, device: d, random: new Random(12));
        float Loss(MixtureOfExperts m, Device d)
        {
            using var scope = new TensorScope();
            using (Autograd.NoGrad())
            {
                return (m.Forward(Tensor.From(x, [1, T, Dim], d)) * Tensor.From(r, [1, T, Dim], d)).Sum().Item();
            }
        }

        using var cpu = Make(Device.Cpu);
        using var onDevice = Make(device);
        using var scope = new TensorScope();
        var input = Tensor.From(x, [1, T, Dim], device, requiresGrad: true);
        (onDevice.Forward(input) * Tensor.From(r, [1, T, Dim], device)).Sum().Backward();
        foreach (var (what, select) in new (string, Func<MixtureOfExperts, Tensor>)[]
        {
            ("router", m => m.Router.Weight),
            ("expert 1, up", m => m.Experts[1].Up.Weight),
            ("expert 2, down", m => m.Experts[2].Down.Weight),
            ("shared expert, gate", m => m.SharedExpert!.Gate!.Weight),
            ("shared expert's gate", m => m.SharedExpertGate!.Weight),
        })
        {
            var analytic = select(onDevice).Grad!.ToArray();
            var parameter = select(cpu);
            var values = parameter.ToArray();
            var numeric = new float[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                const float Eps = 1e-2f;
                float original = values[i];
                values[i] = original + Eps;
                parameter.CopyFrom(values);
                float plus = Loss(cpu, Device.Cpu);
                values[i] = original - Eps;
                parameter.CopyFrom(values);
                float minus = Loss(cpu, Device.Cpu);
                values[i] = original;
                parameter.CopyFrom(values);
                numeric[i] = (plus - minus) / (2 * Eps);
            }

            AssertClose(numeric, analytic, 2e-3f, $"{what} gradient");
            Check(analytic.Any(g => MathF.Abs(g) > 1e-3f), $"{what}: a gradient");
        }

        var inputNumeric = new float[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            float original = x[i];
            x[i] = original + 1e-2f;
            float plus = Loss(cpu, Device.Cpu);
            x[i] = original - 1e-2f;
            float minus = Loss(cpu, Device.Cpu);
            x[i] = original;
            inputNumeric[i] = (plus - minus) / 2e-2f;
        }

        AssertClose(inputNumeric, input.Grad!.ToArray(), 2e-3f, "input gradient");

        // The load-balancing loss: experts · Σ share · mean probability, from the routing, with a gradient to the router.
        onDevice.Router.Weight.ZeroGrad();
        using (var pass = new TensorScope())
        {
            onDevice.Train();
            onDevice.Forward(Tensor.From(x, [1, T, Dim], device));
            var balance = onDevice.LoadBalancingLoss!;
            using var probabilities = Tensor.From(x, [T, Dim], Device.Cpu).MatMul(cpu.Router.Weight).Softmax();
            var p = probabilities.ToArray();
            var (choices, _) = MixtureOfExperts.Route(p, T, 4, 2, normalize: true);
            double expected = 0;
            for (int e = 0; e < 4; e++)
            {
                double share = choices.Count(c => c == e) / (double)T, mean = Enumerable.Range(0, T).Average(t => p[t * 4 + e]);
                expected += 4 * share * mean;
            }

            AssertClose([(float)expected], [balance.Item()], 1e-5f, "load-balancing loss");
            balance.Backward();
            Check(onDevice.Router.Weight.Grad!.ToArray().Any(g => g != 0f), "the load-balancing loss reaches the router");
            onDevice.Eval();
            using (Autograd.NoGrad())
            {
                onDevice.Forward(Tensor.From(x, [1, T, Dim], device));
            }

            Check(onDevice.LoadBalancingLoss is null, "no load-balancing loss at inference");
        }

        // LoRA on the attention of a model with experts: the loss falls, with and without the load-balancing loss.
        var spec = ExpertSpecs[1].Spec with { Vocabulary = 23 };
        string folder = TempFolder();
        try
        {
            WriteExpertCheckpoint(folder, "Qwen2MoeForCausalLM", QwenArchitectures.Qwen2Moe(), spec, new RandomWeights(94));
            var random2 = new Random(4);
            var sequences = Enumerable.Range(0, 4).Select(_ =>
            {
                int[] tokens = [.. Enumerable.Range(0, 12).Select(i => (i * 3 + random2.Next(2)) % 23)];
                return new TrainingSequence(tokens, [.. tokens.Select(_ => true)]);
            }).ToList();
            foreach (float balance in new[] { 0f, 0.01f })
            {
                using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
                var options = new FineTuningOptions
                {
                    Rank = 4, Alpha = 8, LearningRate = 2e-2f, Epochs = 15, BatchTokens = 256, WarmupFraction = 0f, Seed = 5,
                    Targets = ["q", "k", "v", "o"], LoadBalancingWeight = balance,
                };
                float before = FineTuner.Evaluate(model, sequences);
                FineTuner.Train(model, sequences, null, options);
                float after = FineTuner.Evaluate(model, sequences);
                Check(after < before * 0.8f, $"load balancing {balance}: loss {before:F3} → {after:F3}");
                Check(model.Network.Descendants().OfType<Linear>().Count(l => l.Adapter is not null) == 4 * spec.Layers
                      && model.Network.Descendants().OfType<MixtureOfExperts>().All(m => m.Router.Adapter is null), "adapters on attention only");
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    // A config.json of the family with experts, in the Hugging Face keys, for spec's sizes.
    internal static JsonObject ExpertConfig(string architecture, DecoderSpec spec)
    {
        var config = FamilyConfig(architecture);
        config[architecture == "MixtralForCausalLM" ? "num_local_experts" : "num_experts"] = spec.Experts;
        config["num_experts_per_tok"] = spec.ExpertsPerToken;
        if (architecture != "MixtralForCausalLM")
        {
            config["moe_intermediate_size"] = spec.ExpertFfDim;
            config["norm_topk_prob"] = spec.NormalizeTopK;
            if (spec.SharedExpertFfDim > 0)
            {
                config["shared_expert_intermediate_size"] = spec.SharedExpertFfDim;
            }

            if (spec.ExpertLayers is { } layers)
            {
                config["mlp_only_layers"] = new JsonArray([.. Enumerable.Range(0, layers.Count).Where(i => !layers[i]).Select(i => (JsonNode)i)]);
            }
        }

        return Parsed(config);
    }

    // The weights of spec (drawn from weights) as the checkpoint of family (the architecture its config names), with its config.
    internal static void WriteExpertCheckpoint(string folder, string architecture, IPretrainedArchitecture family, DecoderSpec spec, RandomWeights weights)
    {
        using (spec.Build(weights, new DecoderBuildOptions { Device = Device.Cpu }))
        {
        }

        File.WriteAllText(Path.Combine(folder, "config.json"), ExpertConfig(architecture, spec).ToJsonString());
        var tensors = new List<(string, int[], float[])>();
        foreach (var (name, values) in weights.Values)
        {
            string stored = family.TensorName(name) ?? throw new InvalidOperationException($"no checkpoint name for {name}");
            int[] shape = ShapeOf(spec, name);
            tensors.Add(family.Transposed(name) ? (stored, [shape[1], shape[0]], Transpose2D(values, shape[0], shape[1])) : (stored, shape, values));
        }

        SafeTensorsWriter.Write(Path.Combine(folder, "model.safetensors"), tensors);
    }
}
