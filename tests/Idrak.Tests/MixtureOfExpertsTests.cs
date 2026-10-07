// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Nlp;
using Idrak.Nlp.Abstractions;

// Mixture of experts (Mixtral, Qwen2-MoE, Qwen3-MoE): the routing, decoders against the loop reference (prompt, training,
// cached decoding, batches, packed experts), gradients, fine-tuning, and checkpoints in each family's names (safetensors
// and GGUF).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] MixtureOfExpertsGroup =
    [
        ("mixture of experts: routing takes the k most probable experts (ties to the lowest index, as torch.topk), renormalized when configured; a known routing on the device", ExpertRouting),
        ("mixture of experts: Mixtral-, Qwen2-MoE- and Qwen3-MoE-shaped decoders match the loop reference (prompt, training, cached decoding, batches); packed experts; recording falls back; JSON and parameter counts", ExpertDecoders),
        ("mixture of experts: gradients of router, experts, shared expert and input match finite differences; load-balancing loss; LoRA on attention fine-tunes a model with experts", ExpertTraining),
        ("pretrained: Mixtral, Qwen2-MoE and Qwen3-MoE configs and checkpoints (their tensor names) load and match the reference; GGUF files with experts (qwen2moe, qwen3moe, llama as Mixtral) too", ExpertCheckpoints),
    ];

    // SmallSpec with experts, shaped like each family.
    private static (string Name, DecoderSpec Spec)[] ExpertSpecs =>
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

    private static void ExpertRouting(Device device)
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

    private static void ExpertDecoders(Device device)
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

        // Recording a decoding step through the experts fails cleanly (the routing is read back): replays run the step and
        // sample what direct steps do.
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
            Check(!capture!.IsRecorded && (!device.Backend.SupportsGraphs || capture.FailureReason?.Contains("experts", StringComparison.Ordinal) == true),
                $"the recording falls back ({capture.FailureReason})");
        }

        Check(direct.Zip(replayed).All(p => p.First.SequenceEqual(p.Second)),
            $"replayed steps sample what direct steps do: {string.Join(" ", direct.Select(d => d[0]))} vs {string.Join(" ", replayed.Select(d => d[0]))}");
    }

    private static void ExpertTraining(Device device)
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
    private static JsonObject ExpertConfig(string architecture, DecoderSpec spec)
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
    private static void WriteExpertCheckpoint(string folder, string architecture, DecoderSpec spec, RandomWeights weights)
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

    private static void ExpertCheckpoints(Device device)
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
        Check(PretrainedFamilies.MixtralTensorName("layers.3.mlp.router.weight") == "model.layers.3.block_sparse_moe.gate.weight"
              && PretrainedFamilies.MixtralTensorName("layers.3.mlp.experts.5.gate.weight") == "model.layers.3.block_sparse_moe.experts.5.w1.weight"
              && PretrainedFamilies.MixtralTensorName("layers.3.mlp.experts.5.up.weight") == "model.layers.3.block_sparse_moe.experts.5.w3.weight"
              && PretrainedFamilies.MixtralTensorName("layers.3.mlp.experts.5.down.weight") == "model.layers.3.block_sparse_moe.experts.5.w2.weight"
              && PretrainedFamilies.MixtralTensorName("layers.3.attn.q.weight") == "model.layers.3.self_attn.q_proj.weight", "Mixtral names");
        Check(PretrainedFamilies.QwenMoeTensorName("layers.0.mlp.router.weight") == "model.layers.0.mlp.gate.weight"
              && PretrainedFamilies.QwenMoeTensorName("layers.0.mlp.experts.12.down.weight") == "model.layers.0.mlp.experts.12.down_proj.weight"
              && PretrainedFamilies.QwenMoeTensorName("layers.0.mlp.shared.up.weight") == "model.layers.0.mlp.shared_expert.up_proj.weight"
              && PretrainedFamilies.QwenMoeTensorName("layers.0.mlp.shared_gate.weight") == "model.layers.0.mlp.shared_expert_gate.weight"
              && PretrainedFamilies.QwenMoeTensorName("layers.1.mlp.gate.weight") == "model.layers.1.mlp.gate_proj.weight", "Qwen MoE names");

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
    private static void WriteExpertGguf(string path, string arch, DecoderSpec spec, RandomWeights weights)
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
