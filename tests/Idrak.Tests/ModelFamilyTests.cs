// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;

// Model families beyond the plain decoder: RoPE scaling methods, sliding-window attention, soft-capping, families read
// from config.json (Mistral, Qwen2, Gemma 2 and 3) and families that build their own network.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] ModelFamilyGroup =
    [
        ("rope scaling: linear and llama3 unchanged; YaRN and dynamic NTK match the reference formulas; registered methods; unknown types list the registered ones", RopeScalingMethods),
        ("rope scaling: a decoder uses the registered method (frequencies, attention factor, per-position frequencies) as the reference computes it", RopeScalingDecoders),
        ("decoder: sliding windows, soft-capping, score scale and local rotary base match the reference (prompt, training, cached decoding with every cache format); windows past the sequence change nothing", SlidingWindowDecoders),
        ("pretrained: Mistral, Qwen2, Gemma 2 and Gemma 3 configs read windows, soft-capping and scales; a Gemma 2 checkpoint loads; a family with its own Build", ModelFamilyConfigs),
        ("gguf: YaRN scaling keys are read and match the Hugging Face config with the same rope_scaling", GgufYarn),
        ("pretrained: Gemma3ForConditionalGeneration loads in its three tensor layouts and both config formats; a text prompt gives transformers' logits; the vision part is read as data", Gemma3VisionLanguageLoads),
    ];

    private static void RopeScalingMethods(Device device)
    {
        _ = device;
        // linear and llama3 give what the formulas before the registry gave (copied here), bit for bit.
        const double theta = 500000;
        const int dim = 64;
        double[] Base() => [.. Enumerable.Range(0, dim / 2).Select(i => 1.0 / Math.Pow(theta, 2.0 * i / dim))];
        var linear = Base().Select(f => f / 2.0).ToArray();
        Check(new RopeSettings((float)theta, Scaling: RopeScaling.Linear(2)).Frequencies(dim).SequenceEqual(linear), "linear unchanged");
        var llama3 = Base();
        {
            double factor = 8, low = 1, high = 4, original = 8192, lowWavelength = original / low, highWavelength = original / high;
            for (int i = 0; i < llama3.Length; i++)
            {
                double wavelength = 2 * Math.PI / llama3[i];
                if (wavelength > lowWavelength)
                {
                    llama3[i] /= factor;
                }
                else if (wavelength >= highWavelength)
                {
                    double smooth = (original / wavelength - low) / (high - low);
                    llama3[i] = (1 - smooth) * llama3[i] / factor + smooth * llama3[i];
                }
            }
        }

        var llamaScaled = new RopeSettings((float)theta, Scaling: RopeScaling.Llama3(8)).Scaled(dim);
        Check(llamaScaled.Frequencies.SequenceEqual(llama3) && llamaScaled.AttentionFactor == 1 && llamaScaled.FrequenciesAt is null, "llama3 unchanged");

        // YaRN, from the formulas of transformers' _compute_yarn_parameters written out independently.
        (double[] Frequencies, double Attention) Yarn(double b, int d, double factor, double original, double? attentionFactor, double? mscale,
            double? mscaleAllDim, double betaFast, double betaSlow, bool truncate)
        {
            double CorrectionDim(double rotations) => d * Math.Log(original / (rotations * 2 * Math.PI)) / (2 * Math.Log(b));
            double low = CorrectionDim(betaFast), high = CorrectionDim(betaSlow);
            if (truncate)
            {
                low = Math.Floor(low);
                high = Math.Ceiling(high);
            }

            low = Math.Max(low, 0);
            high = Math.Min(high, d - 1);
            if (low == high)
            {
                high += 0.001;
            }

            var frequencies = new double[d / 2];
            for (int i = 0; i < frequencies.Length; i++)
            {
                double posFreq = Math.Pow(b, 2.0 * i / d);
                double extrapolationFactor = 1 - Math.Clamp((i - low) / (high - low), 0, 1);
                frequencies[i] = 1 / (factor * posFreq) * (1 - extrapolationFactor) + 1 / posFreq * extrapolationFactor;
            }

            double GetMscale(double scale, double m = 1) => scale <= 1 ? 1 : 0.1 * m * Math.Log(scale) + 1;
            double attention = attentionFactor ?? (mscale is { } m1 && mscaleAllDim is { } m2 && m1 != 0 && m2 != 0
                ? GetMscale(factor, m1) / GetMscale(factor, m2) : GetMscale(factor));
            return (frequencies, attention);
        }

        void CheckYarn(string what, RopeScaling scaling, (double[] Frequencies, double Attention) expected)
        {
            var actual = new RopeSettings(10000f, Scaling: scaling).Scaled(dim);
            Check(actual.Frequencies.Length == expected.Frequencies.Length
                  && actual.Frequencies.Zip(expected.Frequencies).All(p => Math.Abs(p.First - p.Second) <= 1e-12 * p.Second)
                  && Math.Abs(actual.AttentionFactor - expected.Attention) < 1e-12,
                $"yarn {what}: [{string.Join(", ", actual.Frequencies.Take(4))} …] × {actual.AttentionFactor}, expected [{string.Join(", ", expected.Frequencies.Take(4))} …] × {expected.Attention}");
        }

        CheckYarn("defaults", RopeScaling.Yarn(4, 2048), Yarn(10000, dim, 4, 2048, null, null, null, 32, 1, true));
        CheckYarn("betas, attention factor", RopeScaling.Yarn(8, 4096, attentionFactor: 1.2, betaFast: 16, betaSlow: 2), Yarn(10000, dim, 8, 4096, 1.2, null, null, 16, 2, true));
        CheckYarn("mscale, no truncation, max_position_embeddings", new RopeScaling("yarn", new JsonObject
        {
            ["factor"] = 40, ["max_position_embeddings"] = 4096, ["mscale"] = 1.0, ["mscale_all_dim"] = 0.707, ["truncate"] = false,
        }), Yarn(10000, dim, 40, 4096, null, 1.0, 0.707, 32, 1, false));
        var yarnFrequencies = new RopeSettings(10000f, Scaling: RopeScaling.Yarn(4, 2048)).Frequencies(dim);
        var unscaled = new RopeSettings(10000f).Frequencies(dim);
        Check(yarnFrequencies[0] == unscaled[0] && Math.Abs(yarnFrequencies[^1] - unscaled[^1] / 4) < 1e-15, "yarn keeps the fastest pair and stretches the slowest");

        // Dynamic NTK: unchanged within max_position_embeddings; beyond, b' = b · (factor · s / M - (factor - 1))^(d / (d - 2)).
        var dynamic = new RopeSettings(10000f, Scaling: RopeScaling.Dynamic(2, 16)).Scaled(8);
        var plain = new RopeSettings(10000f).Frequencies(8);
        Check(dynamic.FrequenciesAt is not null && dynamic.Frequencies.SequenceEqual(plain) && dynamic.FrequenciesAt(15).SequenceEqual(plain), "dynamic: unchanged within the context");
        foreach (int position in new[] { 16, 40 })
        {
            double length = position + 1, grownBase = 10000 * Math.Pow(2 * length / 16 - 1, 8.0 / 6);
            var expected = Enumerable.Range(0, 4).Select(i => 1 / Math.Pow(grownBase, 2.0 * i / 8)).ToArray();
            var actual = dynamic.FrequenciesAt!(position);
            Check(actual.Zip(expected).All(p => Math.Abs(p.First - p.Second) <= 1e-12 * p.Second), $"dynamic at {position}: [{string.Join(", ", actual)}], expected [{string.Join(", ", expected)}]");
        }

        // Registered, replaced, removed; names ignore case; unknown types list the registered ones.
        RopeScalings.Register("Test-Halve", input => new RopeScalingResult([.. input.Frequencies.Select(f => f / 2)], 1.5));
        try
        {
            var halved = new RopeSettings(10000f, Scaling: new RopeScaling("test-halve")).Scaled(8);
            Check(halved.Frequencies.SequenceEqual(plain.Select(f => f / 2)) && halved.AttentionFactor == 1.5 && RopeScalings.Names.Contains("Test-Halve"), "a registered method");
        }
        finally
        {
            Check(RopeScalings.Unregister("test-halve") && !RopeScalings.Contains("Test-Halve"), "unregistered");
        }

        try
        {
            new RopeSettings(10000f, Scaling: new RopeScaling("no-such-scaling")).Frequencies(8);
            Check(false, "an unknown scaling is rejected");
        }
        catch (NotSupportedException ex)
        {
            Check(ex.Message.Contains("no-such-scaling") && ex.Message.Contains("yarn") && ex.Message.Contains("dynamic") && ex.Message.Contains("RopeScalings.Register"), ex.Message);
        }

        // config.json: the parameters are forwarded with the model's context length added; unknown types list the names.
        var config = FamilyConfig("LlamaForCausalLM");
        config["rope_scaling"] = new JsonObject { ["rope_type"] = "yarn", ["factor"] = 4.0, ["original_max_position_embeddings"] = 8 };
        var spec = PretrainedFamilies.CommonSpec(Parsed(config), []);
        var parameters = spec.Rope!.Scaling!.Parameters;
        Check(spec.Rope.Scaling.Type == "yarn" && (double)parameters["factor"]! == 4 && (int)parameters["original_max_position_embeddings"]! == 8
              && (int)parameters["max_position_embeddings"]! == 32 && parameters["rope_type"] is null, $"yarn from config.json: {spec.Rope.Scaling}");
        config["rope_scaling"] = new JsonObject { ["type"] = "dynamic", ["factor"] = 2.0 };
        Check(PretrainedFamilies.CommonSpec(Parsed(config), []).Rope!.Scaling == RopeScaling.Dynamic(2, 32), "dynamic from config.json (the older 'type' key)");
        config["rope_scaling"] = new JsonObject { ["rope_type"] = "longrope", ["factor"] = 2.0 };
        try
        {
            PretrainedFamilies.CommonSpec(Parsed(config), []);
            Check(false, "an unregistered rope_type is rejected");
        }
        catch (NotSupportedException ex)
        {
            Check(ex.Message.Contains("longrope") && ex.Message.Contains("llama3") && ex.Message.Contains("yarn"), ex.Message);
        }

        // Specs keep the scaling through JSON; specs written before the registry still read.
        var withYarn = SmallSpec with { Rope = new RopeSettings(500f, Scaling: RopeScaling.Yarn(4, 8)) };
        Check(DecoderSpec.FromJson(withYarn.ToJson()) == withYarn, "a YaRN spec round-trips through JSON");
        var old = SmallSpec.ToJson();
        old["rope"]!["scaling"] = new JsonObject { ["type"] = "llama3", ["factor"] = 8.0, ["lowFrequencyFactor"] = 1.0, ["highFrequencyFactor"] = 4.0, ["originalMaxPositions"] = 8 };
        Check(DecoderSpec.FromJson(old).Rope!.Scaling == RopeScaling.Llama3(8, 1, 4, 8), "the earlier JSON form of llama3 reads");
    }

    private static void RopeScalingDecoders(Device device)
    {
        int[] ids = [3, 17, 5, 5, 22, 0, 9, 13, 2, 8, 19, 1];
        using var input = Tensor.From([.. ids.Select(i => (float)i)], [1, ids.Length], device);
        RopeScalings.Register("test-halve", input => new RopeScalingResult([.. input.Frequencies.Select(f => f / 2)]));
        RopeScalings.Register("test-loud", input => new RopeScalingResult(input.Frequencies, 1.25));
        try
        {
            // A registered method makes the same tables as the built-in it imitates.
            using var halved = (SmallSpec with { Rope = new RopeSettings(500f, Scaling: new RopeScaling("test-halve")) }).Build(new RandomWeights(81), new DecoderBuildOptions { Device = device });
            using var linear = (SmallSpec with { Rope = new RopeSettings(500f, Scaling: RopeScaling.Linear(2)) }).Build(new RandomWeights(81), new DecoderBuildOptions { Device = device });
            Check(halved.Predict(input).ToArray().SequenceEqual(linear.Predict(input).ToArray()), "a registered scaling is used by the decoder");

            foreach (var (name, rope) in new[]
            {
                ("attention factor of a registered method", new RopeSettings(500f, Scaling: new RopeScaling("test-loud"))),
                ("yarn", new RopeSettings(500f, Scaling: RopeScaling.Yarn(4, 8))),
                ("dynamic NTK beyond the context", new RopeSettings(500f, Scaling: RopeScaling.Dynamic(2, 6))),
            })
            {
                var spec = SmallSpec with { Rope = rope };
                var weights = new RandomWeights(82);
                using var model = spec.Build(weights, new DecoderBuildOptions { Device = device });
                AssertClose(ReferenceDecoder(spec, weights, ids), model.Predict(input).ToArray(), 1e-3f, name);
            }
        }
        finally
        {
            RopeScalings.Unregister("test-halve");
            RopeScalings.Unregister("test-loud");
        }
    }

    private static void SlidingWindowDecoders(Device device)
    {
        int[] ids = [1, 4, 9, 16, 2, 7, 11, 3, 8, 20];
        int T = ids.Length, V = SmallSpec.Vocabulary;
        using var sequence = Tensor.From([.. ids.Select(i => (float)i)], [1, T], device);
        var variants = new (string Name, DecoderSpec Spec)[]
        {
            ("window 3 in every layer", SmallSpec with { SlidingWindow = 3 }),
            ("Gemma 2 style: alternating window, soft-capped scores and logits, score scale, post-norms", SmallSpec with
            {
                SlidingWindow = 3, SlidingWindowLayers = [true, false], AttentionSoftcap = 1.5f, LogitSoftcap = 2f, AttentionScale = 0.3f,
                PostNorms = true, NormOffset = 1f, Activation = FeedForwardActivation.Gelu, TieEmbeddings = true, EmbeddingScale = 4f,
            }),
            ("Gemma 3 style: local rotary base in the windowed layer, q/k norm, scaled global rope", SmallSpec with
            {
                SlidingWindow = 4, SlidingWindowLayers = [true, false], SlidingWindowRope = new RopeSettings(50f), QkNorm = true,
                Rope = new RopeSettings(5000f, Scaling: RopeScaling.Linear(2)), AttentionScale = 0.5f, PostNorms = true,
            }),
            ("soft-capped scores only", SmallSpec with { AttentionSoftcap = 1f }),
        };
        foreach (var (name, spec) in variants)
        {
            var weights = new RandomWeights(83);
            using var model = spec.Build(weights, new DecoderBuildOptions { Device = device });
            var reference = ReferenceDecoder(spec, weights, ids);
            var full = model.Predict(sequence).ToArray();
            AssertClose(reference, full, 1e-3f, name);
            model.Train();
            using (var scope = new TensorScope())
            {
                var output = model.Forward(sequence);
                AssertClose(reference, output.ToArray(), 1e-3f, $"{name} (training path)");
                output.Sum().Backward();
                var gradient = model.Parameters().First(p => p.Grad is not null).Grad!.ToArray();
                Check(gradient.All(float.IsFinite) && gradient.Any(g => g != 0f), $"{name}: gradients reach the weights");
            }

            model.Eval();
            float range = full.Max(MathF.Abs);
            foreach (var format in new[] { KeyValueFormat.Float32, KeyValueFormat.Int8, KeyValueFormat.BFloat16 })
            {
                // Quantized caches only roughly match (soft-capped logits have a small range, so the bounds are looser than
                // the plain decoder's); the float32 cache checks the attention exactly.
                float tolerance = format switch { KeyValueFormat.Float32 => 1e-4f, KeyValueFormat.BFloat16 => 0.03f * range, _ => 0.1f * range };
                foreach (int prompt in new[] { 2, 5 })
                {
                    using var context = new DecodingContext(device, 1, 16, format);
                    using (Autograd.NoGrad())
                    {
                        using var first = Tensor.From([.. ids.Take(prompt).Select(i => (float)i)], [1, prompt], device);
                        AssertClose(full[..(prompt * V)], model.ForwardCached(first, context).ToArray(), tolerance, $"{name}, {format}: prefill of {prompt}");
                        for (int t = prompt; t < T; t++)
                        {
                            using var next = Tensor.From([(float)ids[t]], [1, 1], device);
                            AssertClose(full[(t * V)..((t + 1) * V)], model.ForwardCached(next, context).ToArray(), tolerance, $"{name}, {format}: step {t} after {prompt}");
                        }
                    }
                }
            }
        }

        // A recorded decoding step stays right when replayed past the window (captured while the window masks nothing yet).
        using (var windowed = variants[0].Spec.Build(new RandomWeights(83), new DecoderBuildOptions { Device = device }))
        {
            int[][] Run(bool useGraph)
            {
                using var context = new DecodingContext(device, 1, 12);
                using var sampler = new TokenSampler(device, 1, V, 9) { Temperature = 0.9f, Seed = 7 };
                using (Autograd.NoGrad())
                {
                    using (var scope = new TensorScope())
                    {
                        sampler.Sample(windowed.ForwardCached(Tensor.From([1f, 2f], [1, 2], device), context));
                    }

                    void Step() => sampler.Sample(windowed.ForwardCached(sampler.Ids.Reshape(1, 1), context));
                    using var graph = useGraph ? context.CaptureStep(Step) : null;
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
                }

                return [.. sampler.Read(0, 9).Select(step => step.Select(t => t.Id).ToArray())];
            }

            var direct = Run(useGraph: false);
            var replayed = Run(useGraph: true);
            Check(direct.Zip(replayed).All(p => p.First.SequenceEqual(p.Second)),
                $"replayed steps sample what direct steps do: {string.Join(" ", direct.Select(d => d[0]))} vs {string.Join(" ", replayed.Select(d => d[0]))}");
        }

        // A window at least as long as the sequence changes nothing; a shorter one does.
        using var unwindowed = SmallSpec.Build(new RandomWeights(84), new DecoderBuildOptions { Device = device });
        using var wide = (SmallSpec with { SlidingWindow = T }).Build(new RandomWeights(84), new DecoderBuildOptions { Device = device });
        using var narrow = (SmallSpec with { SlidingWindow = 2 }).Build(new RandomWeights(84), new DecoderBuildOptions { Device = device });
        var expected = unwindowed.Predict(sequence).ToArray();
        Check(wide.Predict(sequence).ToArray().SequenceEqual(expected), "a window as long as the sequence equals full attention");
        Check(narrow.Predict(sequence).ToArray().Zip(expected).Max(p => MathF.Abs(p.First - p.Second)) > 1e-3f, "a shorter window changes the outputs");
        Check(PackedSequences.Supports(narrow) == narrow.Descendants().OfType<CausalSelfAttention>().All(a => device.Backend.SupportsSegmentedAttention(a.HeadDim, a.Variant)),
            "packing is offered for windows that can mask where the device's packed attention takes a window");
        var round = DecoderSpec.FromJson(variants[1].Spec.ToJson());
        Check(round.ToJson().ToJsonString() == variants[1].Spec.ToJson().ToJsonString() && round.SlidingWindowLayers!.SequenceEqual([true, false]) && round.LogitSoftcap == 2f,
            "windows and soft-capping round-trip through JSON");
    }

    // The configuration as read from a file (numbers parsed from text, as config.json gives them).
    private static JsonObject Parsed(JsonObject config) => JsonNode.Parse(config.ToJsonString())!.AsObject();

    // A small config.json in the Hugging Face layout (SmallSpec's sizes).
    private static JsonObject FamilyConfig(string architecture) => new()
    {
        ["architectures"] = new JsonArray(architecture),
        ["vocab_size"] = SmallSpec.Vocabulary, ["hidden_size"] = SmallSpec.Dim, ["num_hidden_layers"] = SmallSpec.Layers,
        ["num_attention_heads"] = SmallSpec.Heads, ["num_key_value_heads"] = SmallSpec.KvHeads, ["head_dim"] = SmallSpec.HeadDim,
        ["intermediate_size"] = SmallSpec.FfDim, ["max_position_embeddings"] = SmallSpec.MaxPositions, ["rms_norm_eps"] = SmallSpec.NormEpsilon,
        ["rope_theta"] = 500.0, ["hidden_act"] = "silu",
    };

    private static void ModelFamilyConfigs(Device device)
    {
        DecoderSpec Read(JsonObject config) => PretrainedArchitectures.Get((string)config["architectures"]![0]!).Spec(Parsed(config), []);

        // Mistral: the window in every layer (none when null).
        var mistral = FamilyConfig("MistralForCausalLM");
        mistral["sliding_window"] = 3;
        var spec = Read(mistral);
        Check(spec.SlidingWindow == 3 && spec.SlidingWindowLayers is null && spec.IsWindowed(0) && spec.IsWindowed(1), "Mistral: window in every layer");
        mistral["sliding_window"] = null;
        Check(Read(mistral).SlidingWindow is null, "Mistral: no window");

        // Qwen2: only with use_sliding_window, from max_window_layers on; layer_types when given.
        var qwen = FamilyConfig("Qwen2ForCausalLM");
        qwen["sliding_window"] = 4;
        qwen["max_window_layers"] = 1;
        qwen["use_sliding_window"] = false;
        Check(Read(qwen).SlidingWindow is null, "Qwen2: use_sliding_window false");
        qwen["use_sliding_window"] = true;
        spec = Read(qwen);
        Check(spec.SlidingWindow == 4 && !spec.IsWindowed(0) && spec.IsWindowed(1), "Qwen2: windows from max_window_layers on");
        qwen["layer_types"] = new JsonArray("sliding_attention", "full_attention");
        spec = Read(qwen);
        Check(spec.IsWindowed(0) && !spec.IsWindowed(1), "layer_types decide");

        // Gemma 2: transformers' defaults; Gemma 3: q/k norm, local base, five windowed layers in six.
        var gemma2 = FamilyConfig("Gemma2ForCausalLM");
        gemma2["hidden_activation"] = "gelu_pytorch_tanh";
        gemma2.Remove("hidden_act");
        gemma2["query_pre_attn_scalar"] = 16;
        gemma2["sliding_window"] = 3;
        spec = Read(gemma2);
        Check(spec is { AttentionSoftcap: 50f, LogitSoftcap: 30f, AttentionScale: 0.25f, SlidingWindow: 3, PostNorms: true, NormOffset: 1f, TieEmbeddings: true, QkNorm: false }
              && spec.Activation == FeedForwardActivation.Gelu && spec.IsWindowed(0) && !spec.IsWindowed(1), $"Gemma 2 spec: {spec.ToJson().ToJsonString()}");
        var gemma3 = FamilyConfig("Gemma3ForCausalLM");
        gemma3["num_hidden_layers"] = 7;
        gemma3["rope_local_base_freq"] = 10.0;
        gemma3["attn_logit_softcapping"] = null;
        spec = Read(gemma3);
        Check(spec is { QkNorm: true, AttentionSoftcap: null, LogitSoftcap: null, SlidingWindow: 4096 } && spec.SlidingWindowRope!.Theta == 10f
              && Enumerable.Range(0, 7).Count(spec.IsWindowed) == 6 && !spec.IsWindowed(5), $"Gemma 3 spec: {spec.ToJson().ToJsonString()}");

        // A Gemma 2 checkpoint (four norms per layer under their own names) loads and matches its spec.
        gemma2["sliding_window"] = 3;
        gemma2["attn_logit_softcapping"] = 2.0;
        gemma2["final_logit_softcapping"] = 3.0;
        var gemmaSpec = Read(gemma2);
        var weights = new RandomWeights(85);
        using var reference = gemmaSpec.Build(weights, new DecoderBuildOptions { Device = device });
        int[] ids = [2, 9, 14, 3, 7, 1, 20];
        using var input = Tensor.From([.. ids.Select(i => (float)i)], [1, ids.Length], device);
        var expected = reference.Predict(input).ToArray();
        AssertClose(ReferenceDecoder(gemmaSpec, weights, ids), expected, 1e-3f, "Gemma 2 spec against the reference");
        string folder = TempFolder();
        try
        {
            File.WriteAllText(Path.Combine(folder, "config.json"), gemma2.ToJsonString());
            var tensors = new List<(string, int[], float[])>();
            foreach (var (name, values) in weights.Values)
            {
                string stored = PretrainedFamilies.GemmaTensorName(name)!;
                bool linear = (name.Contains(".attn.") && !name.Contains("_norm") || name.Contains(".mlp.")) && name.EndsWith(".weight");
                int[] shape = ShapeOf(gemmaSpec, name);
                tensors.Add(linear ? (stored, [shape[1], shape[0]], Transpose2D(values, shape[0], shape[1])) : (stored, shape, values));
            }

            Check(tensors.Any(t => t.Item1 == "model.layers.1.pre_feedforward_layernorm.weight") && tensors.Any(t => t.Item1 == "model.layers.0.post_feedforward_layernorm.weight"),
                "Gemma 2 norm names");
            SafeTensorsWriter.Write(Path.Combine(folder, "model.safetensors"), tensors, SafeTensorType.F32);
            using (var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
            {
                Check(model.Notes.Count == 0, string.Join("; ", model.Notes));
                AssertClose(expected, model.Network.Predict(input).ToArray(), 1e-4f, "Gemma 2 checkpoint");
            }

            // A family building its own network: here the spec's network with logits capped at 1, so the result shows it was used.
            PretrainedBuildContext? seen = null;
            PretrainedArchitectures.Register("TestOwnBuildForCausalLM", new PretrainedArchitecture
            {
                Spec = (config, notes) => GemmaSpecOf(config, notes),
                TensorName = PretrainedFamilies.GemmaTensorName,
                Build = context =>
                {
                    seen = context;
                    context.Notes.Add("built by the test family");
                    return (context.Spec with { LogitSoftcap = 1f }).Build(context.Weights, context.Options);
                },
            });
            try
            {
                using var own = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, Architecture = "TestOwnBuildForCausalLM", MaxPositions = 16 });
                var logits = own.Network.Predict(input).ToArray();
                Check(seen is { Options.MaxPositions: 16 } && seen.Checkpoint.Contains("model.embed_tokens.weight") && own.Notes.Contains("built by the test family"),
                    "the build delegate receives the config, weights, checkpoint and options");
                Check(logits.All(v => MathF.Abs(v) < 1f) && expected.Any(v => MathF.Abs(v) > 1f), "the network built by the family is the one loaded");
            }
            finally
            {
                Check(PretrainedArchitectures.Unregister("TestOwnBuildForCausalLM") && !PretrainedArchitectures.Names.Contains("TestOwnBuildForCausalLM"), "unregistered");
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }

        static DecoderSpec GemmaSpecOf(JsonObject config, List<string> notes) => PretrainedArchitectures.Get("Gemma2ForCausalLM").Spec(config, notes);
    }

    private static void GgufYarn(Device device)
    {
        string cache = Path.Combine(Path.GetTempPath(), "ns-gguf-" + Guid.NewGuid().ToString("N"));
        string hf = TempFolder();
        try
        {
            // The tiny Qwen3 file with llama.cpp's YaRN keys added (inserted before the others, padded to the file's
            // 32-byte alignment so the tensor data stays aligned).
            var original = File.ReadAllBytes(TestData("gguf/tiny-qwen3-q8.gguf"));
            var inserted = new MemoryStream();
            var writer = new BinaryWriter(inserted);
            void Key(string key, int type)
            {
                writer.Write((ulong)Encoding.UTF8.GetByteCount(key));
                writer.Write(Encoding.UTF8.GetBytes(key));
                writer.Write(type);
            }

            Key("qwen3.rope.scaling.type", 8);
            writer.Write((ulong)4);
            writer.Write("yarn"u8);
            Key("qwen3.rope.scaling.factor", 6);
            writer.Write(4f);
            Key("qwen3.rope.scaling.original_context_length", 4);
            writer.Write(64u);
            Key("qwen3.rope.scaling.yarn_beta_fast", 6);
            writer.Write(16f);
            Key("test.padding", 8);
            long length = inserted.Length + 8;
            int pad = (int)((32 - length % 32) % 32);
            writer.Write((ulong)pad);
            writer.Write(new byte[pad]);
            writer.Flush();
            Check(inserted.Length % 32 == 0, "inserted bytes keep the alignment");
            const int countOffset = 4 + 4 + 8;                               // magic, version, tensor count, then the key count
            var patched = new byte[original.Length + inserted.Length];
            original.AsSpan(0, countOffset + 8).CopyTo(patched);
            BitConverter.TryWriteBytes(patched.AsSpan(countOffset, 8), BitConverter.ToUInt64(original, countOffset) + 5);
            inserted.ToArray().CopyTo(patched, countOffset + 8);
            original.AsSpan(countOffset + 8).CopyTo(patched.AsSpan(countOffset + 8 + (int)inserted.Length));
            Directory.CreateDirectory(cache);
            string file = Path.Combine(cache, "tiny-qwen3-yarn.gguf");
            File.WriteAllBytes(file, patched);

            // The Hugging Face copy with the same rope_scaling.
            foreach (var path in Directory.GetFiles(TestData("gguf/tiny-qwen3-q8-hf")))
            {
                File.Copy(path, Path.Combine(hf, Path.GetFileName(path)));
            }

            var config = JsonNode.Parse(File.ReadAllText(Path.Combine(hf, "config.json")))!.AsObject();
            config["rope_scaling"] = new JsonObject { ["rope_type"] = "yarn", ["factor"] = 4.0, ["original_max_position_embeddings"] = 64, ["beta_fast"] = 16.0 };
            File.WriteAllText(Path.Combine(hf, "config.json"), config.ToJsonString());

            using var fromGguf = PretrainedModel.Load(GgufModel.Prepare(file, cache), new PretrainedOptions { Device = device });
            using var fromHf = PretrainedModel.Load(hf, new PretrainedOptions { Device = device });
            var scaling = fromGguf.Spec.Rope!.Scaling;
            Check(scaling is { Type: "yarn" } && scaling == fromHf.Spec.Rope!.Scaling && fromGguf.Spec.MaxPositions == fromHf.Spec.MaxPositions,
                $"yarn read from GGUF: {scaling} vs {fromHf.Spec.Rope!.Scaling}");
            int[] ids = [.. fromHf.Tokenizer!.Encode("the answer is in the thin rain, and then another answer")];
            using var input = Tensor.From([.. ids.Select(i => (float)i)], [1, ids.Length], device);
            AssertClose(fromHf.Network.Predict(input).ToArray(), fromGguf.Network.Predict(input).ToArray(), 1e-3f, "logits from GGUF and from the Hugging Face copy");
        }
        finally
        {
            Directory.Delete(hf, recursive: true);
            if (Directory.Exists(cache))
            {
                Directory.Delete(cache, recursive: true);
            }
        }
    }
}
