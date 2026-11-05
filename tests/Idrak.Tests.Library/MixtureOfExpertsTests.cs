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
        ("mixture of experts: a device that can offload (IOffloadingBackend, as CUDA always is) routes its experts on the device while nothing is offloaded and offloading is off; with a storage offloaded or ComputeResources.OffloadToHostMemory on, it reads the routing back (a recorded graph could not follow a moved weight)", ExpertsWithOffloadSupport),
        ("mixture of experts: Mixtral-, Qwen2-MoE- and Qwen3-MoE-shaped decoders match the loop reference (prompt, training, cached decoding, batches); packed experts; recording falls back; JSON and parameter counts", ExpertDecoders),
        ("mixture of experts: CUDA's routing and expert kernels (PtxKernels.Routing.cs), read on the host: each declares the parameters its launch passes and only the registers it declares, every branch reaches a label of its own kernel, indices are read as int32, the weights divided by div.rn, each product one fma.rn chain from zero with its bias added by add.rn, the gated pair's activation is gated_act_f32's, and an expert launch walks its experts' slots grid-stride", ExpertPtx),
        ("mixture of experts: gradients of router, experts, shared expert and input match finite differences; load-balancing loss; LoRA on attention fine-tunes a model with experts", ExpertTraining),
        ("pretrained: Mixtral, Qwen2-MoE and Qwen3-MoE configs and checkpoints (their tensor names) load and match the reference; GGUF files with experts (qwen2moe, qwen3moe, llama as Mixtral) too", ExpertCheckpoints),
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

    // A device that can offload (IOffloadingBackend) with the routing operations as kernels of its own: plain arrays for its
    // memory, every operation through the host fallback (the routing's too, so its results are the CPU's), and the number
    // of storages it counts as offloaded set by the test (none at first). CUDA is such a device: it always implements
    // IOffloadingBackend, whether or not anything is offloaded. The offloading tests use it too (FusedPassesWithOffloadSupport).
    internal sealed class OffloadCapableBackend : Backend, IOffloadingBackend, IMemoryOffload
    {
        public static readonly OffloadCapableBackend Instance = new();

        public int Offloaded;

        public override BackendCapabilities Capabilities { get; } = Device.Cpu.Backend.Capabilities;

        public override string Kind => "moe-offload-test";

        public override string Name => "mixture-of-experts test (can offload, routing kernels of its own)";

        public override Storage Allocate(int length, bool zeroed) => new MinimalStorage(this, new float[length], length);

        public override void Return(Storage storage)
        {
        }

        protected override void Detach(Storage storage) => ((MinimalStorage)storage).Data = null!;

        protected override void Attach(Storage storage, Storage fresh) => ((MinimalStorage)storage).Data = ((MinimalStorage)fresh).Data;

        public override MemoryUsage GetMemoryUsage() => new(0, 0, null);

        public override void ReleaseCachedMemory()
        {
        }

        public override void Upload(ReadOnlySpan<float> source, Storage destination) => source.CopyTo(((MinimalStorage)destination).Data);

        public override void Download(Storage source, Span<float> destination) => ((MinimalStorage)source).Data.AsSpan(0, destination.Length).CopyTo(destination);

        public override void DownloadRange(Storage source, int offset, Span<float> destination) =>
            ((MinimalStorage)source).Data.AsSpan(offset, destination.Length).CopyTo(destination);

        public override void Synchronize()
        {
        }

        // The routing operations as its own kernels (RoutesExperts): the host fallback's work, declared here.
        public override void RouteTopKKernel(Storage probabilities, Storage weights, Storage offsets, Storage tokenOfRow, Storage slotRow, int tokens, int experts, int k,
            bool normalize) => base.RouteTopKKernel(probabilities, weights, offsets, tokenOfRow, slotRow, tokens, experts, k, normalize);

        public override void ExpertGatedPairKernel(Storage x, Storage offsets, ReadOnlySpan<(Storage Gate, Storage? GateBias, Storage Up, Storage? UpBias)> experts,
            Storage hidden, int rows, int k, int n, int activation) => base.ExpertGatedPairKernel(x, offsets, experts, hidden, rows, k, n, activation);

        public override void ExpertMatMulKernel(Storage x, Storage offsets, ReadOnlySpan<(Storage Weight, Storage? Bias)> experts, Storage y, int rows, int k, int n) =>
            base.ExpertMatMulKernel(x, offsets, experts, y, rows, k, n);

        IMemoryOffload IOffloadingBackend.Offload => this;

        public int OffloadedCount => Offloaded;

        public bool IsOffloaded(Storage storage) => false;

        public void SetPriority(Storage storage, OffloadPriority priority)
        {
        }

        public bool MoveToHost(Storage storage, bool keep) => false;

        public bool MoveToDevice(Storage storage) => false;

        public int Rebalance(bool makeRoom = true) => 0;

        public void Prefetch(IReadOnlyList<Storage> storages)
        {
        }

        public IDisposable? Stage(IReadOnlyList<Storage> storages) => null;
    }

    internal sealed class OffloadCapableProvider : IDeviceProvider
    {
        public string Name => "moe-offload-test";

        public string Summary => "a test device that can offload, with the mixture-of-experts routing as kernels of its own";

        public DeviceType Type => DeviceType.Other;

        public int Count => 1;

        public Backend Create(int ordinal) => OffloadCapableBackend.Instance;

        public bool IsStarted(int ordinal) => false;

        public bool Listed(int ordinal) => false;

        public string? Note(int ordinal) => "test backend: by name only";
    }

    internal static void ExpertsWithOffloadSupport(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(ExpertsWithOffloadSupport)))
        {
            return;
        }

        if (DeviceProviders.Find("moe-offload-test") is null)
        {
            DeviceProviders.Register(new OffloadCapableProvider());
        }

        var capable = Device.Get("moe-offload-test");
        var backend = OffloadCapableBackend.Instance;
        bool routesBefore = MixtureOfExperts.RoutesOnDevice, offloadBefore = ComputeResources.OffloadToHostMemory;
        try
        {
            MixtureOfExperts.RoutesOnDevice = true;
            ComputeResources.OffloadToHostMemory = false;
            backend.Offloaded = 0;
            var random = new Random(23);
            using var moe = new MixtureOfExperts(8, 6, 4, 2, normalizeTopK: true, device: capable, random: new Random(24));
            using var input = Tensor.From([.. Enumerable.Range(0, 5 * 8).Select(_ => random.NextSingle() * 2 - 1)], [5, 8], capable);
            long RouteCalls()
            {
                using var trace = Kernels.Trace(backend);
                using var scope = new TensorScope();
                using (Autograd.NoGrad())
                {
                    moe.Forward(input).ToArray();
                    return trace.Calls(Ops.RouteTopK);
                }
            }

            Check(backend.Offload is not null && backend.RoutesExperts, $"{backend.Name}: can offload ({backend.Offload is not null}) and routes experts ({backend.RoutesExperts})");
            long calls = RouteCalls();
            Check(calls == 1 && !moe.RoundTrips(backend),
                $"nothing offloaded and offloading off: the experts route on the device (RouteTopK called {calls} times, round trips {moe.RoundTrips(backend)}), so a decoding step through them is recorded");

            backend.Offloaded = 1;
            calls = RouteCalls();
            Check(calls == 0 && moe.RoundTrips(backend), $"a storage offloaded: the routing is read back (RouteTopK called {calls} times, round trips {moe.RoundTrips(backend)})");

            backend.Offloaded = 0;
            ComputeResources.OffloadToHostMemory = true;
            calls = RouteCalls();
            Check(calls == 0 && moe.RoundTrips(backend),
                $"offloading on (ComputeResources.OffloadToHostMemory): the routing is read back (RouteTopK called {calls} times, round trips {moe.RoundTrips(backend)})");
        }
        finally
        {
            MixtureOfExperts.RoutesOnDevice = routesBefore;
            ComputeResources.OffloadToHostMemory = offloadBefore;
            backend.Offloaded = 0;
        }
    }

    internal static void ExpertDecoders(Device device)
    {
        int[] ids = [1, 4, 9, 16, 2, 7, 11, 3, 8, 20];
        int T = ids.Length, V = SmallSpec.Vocabulary;
        using var sequence = Tensor.From([.. ids.Select(i => (float)i)], [1, T], device);
        foreach (var (name, spec) in ExpertSpecs)
        {
            var weights = new RandomWeights(91);
            using var model = spec.Build(weights, new DecoderBuildOptions { Device = device });
            var reference = ReferenceDecoder(spec, weights, ids);
            var full = model.Predict(sequence).ToArray();
            AssertClose(reference, full, 1e-3f, name);
            Check(model.Descendants().OfType<MixtureOfExperts>().Count() == Enumerable.Range(0, spec.Layers).Count(spec.IsExpertLayer)
                  && model.Descendants().OfType<DecoderBlock>().All(b => b.FeedForward is MixtureOfExperts == spec.IsExpertLayer(int.Parse(b.Name!.Split('.')[1]))),
                $"{name}: experts in the expert layers");

            // Parameters: the spec's count (biases aside), and the active count.
            long biases = model.Parameters().Where(p => p.Rank == 1).Sum(p => (long)p.Size) - spec.Layers * spec.Dim * 2 - spec.Dim - (spec.QkNorm ? 2 * spec.Layers * spec.HeadDim : 0);
            Check(model.Parameters().Sum(p => (long)p.Size) - biases == spec.ParameterCount && spec.ActiveParameterCount < spec.ParameterCount,
                $"{name}: parameters {model.Parameters().Sum(p => (long)p.Size) - biases} = {spec.ParameterCount}, active {spec.ActiveParameterCount}");

            var round = DecoderSpec.FromJson(spec.ToJson());
            Check(round.ToJson().ToJsonString() == spec.ToJson().ToJsonString() && round.Experts == spec.Experts && round.ExpertsPerToken == spec.ExpertsPerToken
                  && round.ExpertFfDim == spec.ExpertFfDim && round.SharedExpertFfDim == spec.SharedExpertFfDim && round.NormalizeTopK == spec.NormalizeTopK
                  && (round.ExpertLayers?.SequenceEqual(spec.ExpertLayers!) ?? spec.ExpertLayers is null), $"{name}: JSON round trip");

            // The training path: the same values, and gradients reach the router and the experts.
            model.Train();
            using (var scope = new TensorScope())
            {
                var output = model.Forward(sequence);
                AssertClose(reference, output.ToArray(), 1e-3f, $"{name} (training path)");
                output.Sum().Backward();
                var moe = model.Descendants().OfType<MixtureOfExperts>().First();
                var routerGradient = moe.Router.Weight.Grad!.ToArray();
                Check(routerGradient.All(float.IsFinite) && routerGradient.Any(g => g != 0f), $"{name}: gradients reach the router");
                Check(moe.Experts.Count(e => e.Up.Weight.Grad?.ToArray().Any(g => g != 0f) == true) >= spec.ExpertsPerToken, $"{name}: gradients reach the experts used");
                Check(moe.LoadBalancingLoss is { } balance && balance.Item() is > 0f and < 6f, $"{name}: load-balancing loss {moe.LoadBalancingLoss?.Item()}");
            }

            model.Eval();

            // Cached decoding: a prefill, then one token at a time.
            foreach (int prompt in new[] { 1, 4 })
            {
                using var context = new DecodingContext(device, 1, 16);
                using (Autograd.NoGrad())
                {
                    using var first = Tensor.From([.. ids.Take(prompt).Select(i => (float)i)], [1, prompt], device);
                    AssertClose(full[..(prompt * V)], model.ForwardCached(first, context).ToArray(), 1e-3f, $"{name}: prefill of {prompt}");
                    for (int t = prompt; t < T; t++)
                    {
                        using var next = Tensor.From([(float)ids[t]], [1, 1], device);
                        AssertClose(full[(t * V)..((t + 1) * V)], model.ForwardCached(next, context).ToArray(), 1e-3f, $"{name}: step {t} after {prompt}");
                    }
                }
            }

            // Two rows at once: each as alone.
            int[] other = [5, 5, 12, 0, 19, 3, 3, 14, 6, 1];
            using (var both = Tensor.From([.. ids.Concat(other).Select(i => (float)i)], [2, T], device))
            {
                var batch = model.Predict(both).ToArray();
                AssertClose(full, batch[..(T * V)], 1e-3f, $"{name}: first row of a batch");
                AssertClose(ReferenceDecoder(spec, weights, other), batch[(T * V)..], 1e-3f, $"{name}: second row of a batch");
            }
        }

        // Packed experts (int8, int4, bfloat16): the same on the device as on the CPU (packed alike on the host), int8 and
        // bfloat16 near the float32 model, and cached decoding matching their own full pass.
        var packedSpec = SmallSpec with { Dim = 32, HeadDim = 8, FfDim = 64, Experts = 4, ExpertsPerToken = 2, SharedExpertFfDim = 32 };
        using var floatModel = packedSpec.Build(new RandomWeights(92), new DecoderBuildOptions { Device = device });
        var floatLogits = floatModel.Predict(sequence).ToArray();
        float range = floatLogits.Max(MathF.Abs);
        foreach (var (format, tolerance) in new[] { ("int8", 0.05f), ("int4", float.NaN), ("bfloat16", 0.03f) })
        {
            using var packed = packedSpec.Build(new RandomWeights(92), new DecoderBuildOptions { Device = device, PackedFormatName = format });
            var moe = packed.Descendants().OfType<MixtureOfExperts>().First();
            Check(moe.Experts.All(e => e.Up.PackedWeight is not null && e.Down.PackedWeight is not null) && moe.SharedExpert!.Gate!.PackedWeight is not null
                  && moe.Router.PackedWeight is null && moe.SharedExpertGate!.PackedWeight is null, $"{format}: experts packed, router and gate float32");
            var logits = packed.Predict(sequence).ToArray();
            using (var onCpu = packedSpec.Build(new RandomWeights(92), new DecoderBuildOptions { Device = Device.Cpu, PackedFormatName = format }))
            {
                AssertClose(onCpu.Predict(Tensor.From([.. ids.Select(i => (float)i)], [1, T], Device.Cpu)).ToArray(), logits, 2e-3f * range, $"{format}: as on the CPU");
            }

            if (!float.IsNaN(tolerance))
            {
                AssertClose(floatLogits, logits, tolerance * range, $"{format}: near the float32 model");
            }

            using var context = new DecodingContext(device, 1, 16);
            using (Autograd.NoGrad())
            {
                using var first = Tensor.From([.. ids.Take(3).Select(i => (float)i)], [1, 3], device);
                AssertClose(logits[..(3 * V)], packed.ForwardCached(first, context).ToArray(), 2e-3f * range, $"{format}: prefill");
                for (int t = 3; t < T; t++)
                {
                    using var next = Tensor.From([(float)ids[t]], [1, 1], device);
                    AssertClose(logits[(t * V)..((t + 1) * V)], packed.ForwardCached(next, context).ToArray(), 2e-3f * range, $"{format}: step {t}");
                }
            }
        }

        // Recording a decoding step through the experts: where the device routes them itself (plan 37, R0) the step is
        // recorded; elsewhere it fails cleanly (the routing is read back). Either way replays sample what direct steps do.
        using var recorded = ExpertSpecs[0].Spec.Build(new RandomWeights(93), new DecoderBuildOptions { Device = device });
        int[][] Run(bool useGraph, out ComputeGraph? kept)
        {
            kept = null;
            using var context = new DecodingContext(device, 1, 12);
            using var sampler = new TokenSampler(device, 1, V, 9) { Temperature = 0.9f, Seed = 7 };
            using (Autograd.NoGrad())
            {
                using (var scope = new TensorScope())
                {
                    sampler.Sample(recorded.ForwardCached(Tensor.From([1f, 2f], [1, 2], device), context));
                }

                void Step() => sampler.Sample(recorded.ForwardCached(sampler.Ids.Reshape(1, 1), context));
                var graph = useGraph ? context.CaptureStep(Step) : null;
                for (int s = 0; s < 8; s++)
                {
                    if (graph is not null)
                    {
                        context.ReplayStep(graph);
                    }
                    else
                    {
                        using var scope = new TensorScope();
                        Step();
                    }
                }

                kept = graph;
            }

            return [.. sampler.Read(0, 9).Select(step => step.Select(t => t.Id).ToArray())];
        }

        var direct = Run(useGraph: false, out _);
        var replayed = Run(useGraph: true, out var capture);
        using (capture)
        {
            bool records = device.Backend.SupportsGraphs && device.Backend.RoutesExperts && MixtureOfExperts.RoutesOnDevice;
            Check(records ? capture!.IsRecorded : !capture!.IsRecorded && (!device.Backend.SupportsGraphs || capture.FailureReason?.Contains("experts", StringComparison.Ordinal) == true),
                records ? $"the step through the experts is recorded where the device routes them ({capture.FailureReason})" : $"the recording falls back ({capture.FailureReason})");
        }

        Check(direct.Zip(replayed).All(p => p.First.SequenceEqual(p.Second)),
            $"replayed steps sample what direct steps do: {string.Join(" ", direct.Select(d => d[0]))} vs {string.Join(" ", replayed.Select(d => d[0]))}");
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
            WriteExpertCheckpoint(folder, "Qwen2MoeForCausalLM", spec, new RandomWeights(94));
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

    // The weights of spec (drawn from weights) as the family's checkpoint, with its config.
    internal static void WriteExpertCheckpoint(string folder, string architecture, DecoderSpec spec, RandomWeights weights)
    {
        var family = PretrainedArchitectures.Get(architecture);
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

        SafeTensorsWriter.Write(Path.Combine(folder, "model.safetensors"), tensors, SafeTensorType.F32);
    }

    internal static void ExpertCheckpoints(Device device)
    {
        DecoderSpec Read(JsonObject config) => PretrainedArchitectures.Get((string)config["architectures"]![0]!).Spec(config, []);

        // Configs: each family's keys and defaults.
        var mixtral = Read(ExpertConfig("MixtralForCausalLM", SmallSpec with { Experts = 8, ExpertsPerToken = 2 }));
        Check(mixtral is { Experts: 8, ExpertsPerToken: 2, ExpertFfDim: 0, SharedExpertFfDim: 0, NormalizeTopK: true, ExpertLayers: null, QkvBias: false },
            $"Mixtral: {mixtral.ToJson().ToJsonString()}");
        var qwen2Config = ExpertConfig("Qwen2MoeForCausalLM", SmallSpec with { Experts = 60, ExpertsPerToken = 4, ExpertFfDim = 14, SharedExpertFfDim = 56 });
        qwen2Config.Remove("norm_topk_prob");
        qwen2Config["decoder_sparse_step"] = 2;
        var qwen2 = Read(qwen2Config);
        Check(qwen2 is { Experts: 60, ExpertsPerToken: 4, ExpertFfDim: 14, SharedExpertFfDim: 56, NormalizeTopK: false, QkvBias: true }
              && !qwen2.IsExpertLayer(0) && qwen2.IsExpertLayer(1), $"Qwen2-MoE: {qwen2.ToJson().ToJsonString()}");
        var qwen3Config = ExpertConfig("Qwen3MoeForCausalLM", SmallSpec with { Experts = 128, ExpertsPerToken = 8, ExpertFfDim = 7, NormalizeTopK = true });
        qwen3Config["mlp_only_layers"] = new JsonArray(0);
        var qwen3 = Read(qwen3Config);
        Check(qwen3 is { Experts: 128, ExpertsPerToken: 8, ExpertFfDim: 7, NormalizeTopK: true, QkNorm: true } && !qwen3.IsExpertLayer(0) && qwen3.IsExpertLayer(1),
            $"Qwen3-MoE: {qwen3.ToJson().ToJsonString()}");
        try
        {
            var dense = ExpertConfig("MixtralForCausalLM", SmallSpec with { Experts = 8, ExpertsPerToken = 2 });
            dense["architectures"] = new JsonArray("LlamaForCausalLM");
            Read(dense);
            Check(false, "a dense family should refuse experts");
        }
        catch (NotSupportedException ex)
        {
            Check(ex.Message.Contains("MixtralForCausalLM", StringComparison.Ordinal), ex.Message);
        }

        // Tensor names.
        Check(MistralArchitectures.MixtralTensorName("layers.3.mlp.router.weight") == "model.layers.3.block_sparse_moe.gate.weight"
              && MistralArchitectures.MixtralTensorName("layers.3.mlp.experts.5.gate.weight") == "model.layers.3.block_sparse_moe.experts.5.w1.weight"
              && MistralArchitectures.MixtralTensorName("layers.3.mlp.experts.5.up.weight") == "model.layers.3.block_sparse_moe.experts.5.w3.weight"
              && MistralArchitectures.MixtralTensorName("layers.3.mlp.experts.5.down.weight") == "model.layers.3.block_sparse_moe.experts.5.w2.weight"
              && MistralArchitectures.MixtralTensorName("layers.3.attn.q.weight") == "model.layers.3.self_attn.q_proj.weight", "Mixtral names");
        Check(QwenArchitectures.MoeTensorName("layers.0.mlp.router.weight") == "model.layers.0.mlp.gate.weight"
              && QwenArchitectures.MoeTensorName("layers.0.mlp.experts.12.down.weight") == "model.layers.0.mlp.experts.12.down_proj.weight"
              && QwenArchitectures.MoeTensorName("layers.0.mlp.shared.up.weight") == "model.layers.0.mlp.shared_expert.up_proj.weight"
              && QwenArchitectures.MoeTensorName("layers.0.mlp.shared_gate.weight") == "model.layers.0.mlp.shared_expert_gate.weight"
              && QwenArchitectures.MoeTensorName("layers.1.mlp.gate.weight") == "model.layers.1.mlp.gate_proj.weight", "Qwen MoE names");

        // Checkpoints in each family's names load and match the reference.
        int[] ids = [2, 9, 14, 3, 7, 1, 20];
        using var input = Tensor.From([.. ids.Select(i => (float)i)], [1, ids.Length], device);
        string[] architectures = ["MixtralForCausalLM", "Qwen2MoeForCausalLM", "Qwen3MoeForCausalLM"];
        string cache = Path.Combine(Path.GetTempPath(), "ns-gguf-" + Guid.NewGuid().ToString("N"));
        for (int f = 0; f < 3; f++)
        {
            var spec = Read(ExpertConfig(architectures[f], ExpertSpecs[f].Spec));
            var weights = new RandomWeights(95 + f);
            string folder = TempFolder();
            try
            {
                WriteExpertCheckpoint(folder, architectures[f], spec, weights);
                var expected = ReferenceDecoder(spec, weights, ids);
                using (var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
                {
                    Check(model.Notes.Count == 0 && model.Spec.Experts == spec.Experts, $"{architectures[f]}: {string.Join("; ", model.Notes)}");
                    AssertClose(expected, model.Network.Predict(input).ToArray(), 1e-3f, $"{architectures[f]} checkpoint");
                }

                // The same model as a GGUF file (float32 tensors in llama.cpp's names and layouts).
                string gguf = Path.Combine(folder, "model.gguf");
                WriteExpertGguf(gguf, f switch { 0 => "llama", 1 => "qwen2moe", _ => "qwen3moe" }, spec, weights);
                using (var fromGguf = PretrainedModel.Load(GgufModel.Prepare(gguf, cache), new PretrainedOptions { Device = device }))
                {
                    var read = fromGguf.Spec;
                    Check((string?)fromGguf.Config["architectures"]?[0] == architectures[f] && read.Experts == spec.Experts && read.ExpertsPerToken == spec.ExpertsPerToken
                          && read.ExpertFfDim == spec.ExpertFfDim && read.SharedExpertFfDim == spec.SharedExpertFfDim && read.NormalizeTopK == spec.NormalizeTopK
                          && read.QkvBias == spec.QkvBias && read.QkNorm == spec.QkNorm && Enumerable.Range(0, spec.Layers).All(l => read.IsExpertLayer(l) == spec.IsExpertLayer(l)),
                        $"{architectures[f]} from GGUF: {read.ToJson().ToJsonString()}\n  vs {spec.ToJson().ToJsonString()}");
                    AssertClose(expected, fromGguf.Network.Predict(input).ToArray(), 1e-3f, $"{architectures[f]} from GGUF");
                }
            }
            finally
            {
                Directory.Delete(folder, recursive: true);
            }
        }

        if (Directory.Exists(cache))
        {
            Directory.Delete(cache, true);
        }
    }

    // spec's weights as a GGUF file of the llama.cpp family arch: float32 tensors (llama's query and key rows interleaved
    // per head, as llama.cpp stores them), the experts of each layer stacked in one tensor per projection.
    internal static void WriteExpertGguf(string path, string arch, DecoderSpec spec, RandomWeights weights)
    {
        var tensors = new List<(string Name, int Type, long[] Dimensions, byte[] Data)>();
        void Add(string name, float[] values, params int[] shape) =>
            tensors.Add((name, 0, [.. shape.Reverse().Select(d => (long)d)], MemoryMarshal.AsBytes(values.AsSpan()).ToArray()));
        float[] W(string name) => weights.Values[name];
        // A [in, out] Idrak weight as the [out, in] rows llama.cpp stores.
        float[] Rows(string name, int inputs, int outputs) => Transpose2D(W(name), inputs, outputs);
        float[] Interleave(float[] rows, int heads, int columns)
        {
            if (arch != "llama")
            {
                return rows;
            }

            int headDim = rows.Length / columns / heads, half = headDim / 2;
            var result = new float[rows.Length];
            for (int h = 0; h < heads; h++)
            {
                for (int i = 0; i < half; i++)
                {
                    for (int a = 0; a < 2; a++)
                    {
                        rows.AsSpan((h * headDim + a * half + i) * columns, columns).CopyTo(result.AsSpan((h * headDim + 2 * i + a) * columns));
                    }
                }
            }

            return result;
        }

        int D = spec.Dim, hd = spec.HeadDim, ff = spec.ExpertFfDim > 0 ? spec.ExpertFfDim : spec.FfDim;
        Add("token_embd.weight", W("embed.weight"), spec.Vocabulary, D);
        Add("output_norm.weight", W("norm.weight"), D);
        Add("output.weight", Rows("head.weight", D, spec.Vocabulary), spec.Vocabulary, D);
        for (int l = 0; l < spec.Layers; l++)
        {
            string p = $"layers.{l}", b = $"blk.{l}";
            Add($"{b}.attn_norm.weight", W($"{p}.attn_norm.weight"), D);
            Add($"{b}.ffn_norm.weight", W($"{p}.mlp_norm.weight"), D);
            Add($"{b}.attn_q.weight", Interleave(Rows($"{p}.attn.q.weight", D, spec.Heads * hd), spec.Heads, D), spec.Heads * hd, D);
            Add($"{b}.attn_k.weight", Interleave(Rows($"{p}.attn.k.weight", D, spec.KvHeads * hd), spec.KvHeads, D), spec.KvHeads * hd, D);
            Add($"{b}.attn_v.weight", Rows($"{p}.attn.v.weight", D, spec.KvHeads * hd), spec.KvHeads * hd, D);
            Add($"{b}.attn_output.weight", Rows($"{p}.attn.o.weight", spec.Heads * hd, D), D, spec.Heads * hd);
            if (spec.QkvBias)
            {
                foreach (var (part, size) in new[] { ("q", spec.Heads * hd), ("k", spec.KvHeads * hd), ("v", spec.KvHeads * hd) })
                {
                    Add($"{b}.attn_{part}.bias", W($"{p}.attn.{part}.bias"), size);
                }
            }

            if (spec.QkNorm)
            {
                Add($"{b}.attn_q_norm.weight", W($"{p}.attn.q_norm.weight"), hd);
                Add($"{b}.attn_k_norm.weight", W($"{p}.attn.k_norm.weight"), hd);
            }

            if (!spec.IsExpertLayer(l))
            {
                Add($"{b}.ffn_gate.weight", Rows($"{p}.mlp.gate.weight", D, spec.FfDim), spec.FfDim, D);
                Add($"{b}.ffn_up.weight", Rows($"{p}.mlp.up.weight", D, spec.FfDim), spec.FfDim, D);
                Add($"{b}.ffn_down.weight", Rows($"{p}.mlp.down.weight", spec.FfDim, D), D, spec.FfDim);
                continue;
            }

            Add($"{b}.ffn_gate_inp.weight", Rows($"{p}.mlp.router.weight", D, spec.Experts), spec.Experts, D);
            foreach (var (part, inputs, outputs) in new[] { ("gate", D, ff), ("up", D, ff), ("down", ff, D) })
            {
                var stacked = Enumerable.Range(0, spec.Experts).SelectMany(j => Rows($"{p}.mlp.experts.{j}.{part}.weight", inputs, outputs)).ToArray();
                Add($"{b}.ffn_{part}_exps.weight", stacked, spec.Experts, outputs, inputs);
            }

            if (spec.SharedExpertFfDim > 0)
            {
                int sff = spec.SharedExpertFfDim;
                Add($"{b}.ffn_gate_shexp.weight", Rows($"{p}.mlp.shared.gate.weight", D, sff), sff, D);
                Add($"{b}.ffn_up_shexp.weight", Rows($"{p}.mlp.shared.up.weight", D, sff), sff, D);
                Add($"{b}.ffn_down_shexp.weight", Rows($"{p}.mlp.shared.down.weight", sff, D), D, sff);
                Add($"{b}.ffn_gate_inp_shexp.weight", W($"{p}.mlp.shared_gate.weight"), D);
            }
        }

        var metadata = new List<KeyValuePair<string, object>>
        {
            new("general.architecture", arch),
            new($"{arch}.embedding_length", (long)D), new($"{arch}.block_count", (long)spec.Layers), new($"{arch}.feed_forward_length", (long)spec.FfDim),
            new($"{arch}.attention.head_count", (long)spec.Heads), new($"{arch}.attention.head_count_kv", (long)spec.KvHeads),
            new($"{arch}.attention.key_length", (long)hd), new($"{arch}.context_length", (long)spec.MaxPositions),
            new($"{arch}.attention.layer_norm_rms_epsilon", (double)spec.NormEpsilon), new($"{arch}.rope.freq_base", (double)spec.Rope!.Theta),
            new($"{arch}.expert_count", (long)spec.Experts), new($"{arch}.expert_used_count", (long)spec.ExpertsPerToken),
            new("tokenizer.ggml.model", "gpt2"),
            new("tokenizer.ggml.tokens", Enumerable.Range(0, spec.Vocabulary).Select(i => ((char)('a' + i)).ToString()).ToArray()),
            new("tokenizer.ggml.merges", Array.Empty<string>()),
        };
        if (spec.ExpertFfDim > 0)
        {
            metadata.Add(new($"{arch}.expert_feed_forward_length", (long)spec.ExpertFfDim));
        }

        if (spec.SharedExpertFfDim > 0)
        {
            metadata.Add(new($"{arch}.expert_shared_feed_forward_length", (long)spec.SharedExpertFfDim));
        }

        WriteGguf(path, metadata, tensors);
    }
}
