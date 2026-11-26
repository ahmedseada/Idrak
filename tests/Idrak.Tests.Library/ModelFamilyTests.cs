// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Text.Plugins.Gemma;
using Idrak.Text.Plugins.Llama;
using Idrak.Text.Plugins.Mistral;
using Idrak.Text.Plugins.Qwen;

// Model families beyond the plain decoder: RoPE scaling methods, sliding-window attention, soft-capping, families read
// from config.json (Mistral, Qwen2, Gemma 2 and 3) and families that build their own network.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] ModelFamilyGroup =
    [
        ("rope scaling: linear and llama3 unchanged; YaRN and dynamic NTK match the reference formulas", RopeScalingMethods),
        ("pretrained: Mistral, Qwen2, Gemma 2 and Gemma 3 configs read windows, soft-capping and scales; a Gemma 2 checkpoint loads", ModelFamilyConfigs),
        ("gguf: YaRN scaling keys are read and match the Hugging Face config with the same rope_scaling", GgufYarn),
        ("pretrained: Gemma3ForConditionalGeneration loads in its three tensor layouts and both config formats; a text prompt gives transformers' logits; the vision part is read as data", Gemma3VisionLanguageLoads),
    ];

    internal static void RopeScalingMethods(Device device)
    {
        _ = device;
        // linear and llama3 give what the formulas before the registry gave (copied here), bit for bit.
        const double theta = 500000;
        const int dim = 64;
        double[] Base() => [.. Enumerable.Range(0, dim / 2).Select(i => 1.0 / PortableMath.Pow(theta, 2.0 * i / dim))];   // the library's own pow (rule 70)
        var linear = Base().Select(f => f / 2.0).ToArray();
        Check(RopeScalings.Linear(new RopeSettings((float)theta).Frequencies(dim), 2).Frequencies.SequenceEqual(linear), "linear unchanged");
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

        var llamaScaled = LlamaRopeScaling.Scale(new RopeSettings((float)theta).Frequencies(dim), 8);
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

        void CheckYarn(string what, RopeScalingResult actual, (double[] Frequencies, double Attention) expected)
        {
            Check(actual.Frequencies.Length == expected.Frequencies.Length
                  && actual.Frequencies.Zip(expected.Frequencies).All(p => Math.Abs(p.First - p.Second) <= 1e-12 * p.Second)
                  && Math.Abs(actual.AttentionFactor - expected.Attention) < 1e-12,
                $"yarn {what}: [{string.Join(", ", actual.Frequencies.Take(4))} …] × {actual.AttentionFactor}, expected [{string.Join(", ", expected.Frequencies.Take(4))} …] × {expected.Attention}");
        }

        double[] Unscaled() => new RopeSettings(10000f).Frequencies(dim);
        CheckYarn("defaults", RopeScalings.Yarn(Unscaled(), 10000, dim, 4, 2048), Yarn(10000, dim, 4, 2048, null, null, null, 32, 1, true));
        CheckYarn("betas, attention factor", RopeScalings.Yarn(Unscaled(), 10000, dim, 8, 4096, betaFast: 16, betaSlow: 2, attentionFactor: 1.2),
            Yarn(10000, dim, 8, 4096, 1.2, null, null, 16, 2, true));
        CheckYarn("mscale, no truncation, max_position_embeddings", RopeScalings.Yarn(Unscaled(), 10000, dim, 40, 4096, truncate: false, mscale: 1.0, mscaleAllDim: 0.707),
            Yarn(10000, dim, 40, 4096, null, 1.0, 0.707, 32, 1, false));
        var yarnFrequencies = RopeScalings.Yarn(Unscaled(), 10000, dim, 4, 2048).Frequencies;
        var unscaled = Unscaled();
        Check(yarnFrequencies[0] == unscaled[0] && Math.Abs(yarnFrequencies[^1] - unscaled[^1] / 4) < 1e-15, "yarn keeps the fastest pair and stretches the slowest");

        // Dynamic NTK: unchanged within max_position_embeddings; beyond, b' = b · (factor · s / M - (factor - 1))^(d / (d - 2)).
        var dynamic = RopeScalings.Dynamic(new RopeSettings(10000f).Frequencies(8), 8, 2, 16);
        var plain = new RopeSettings(10000f).Frequencies(8);
        Check(dynamic.FrequenciesAt is not null && dynamic.Frequencies.SequenceEqual(plain) && dynamic.FrequenciesAt.At(15).SequenceEqual(plain), "dynamic: unchanged within the context");
        foreach (int position in new[] { 16, 40 })
        {
            double length = position + 1, grownBase = 10000 * Math.Pow(2 * length / 16 - 1, 8.0 / 6);
            var expected = Enumerable.Range(0, 4).Select(i => 1 / Math.Pow(grownBase, 2.0 * i / 8)).ToArray();
            var actual = dynamic.FrequenciesAt!.At(position);
            Check(actual.Zip(expected).All(p => Math.Abs(p.First - p.Second) <= 1e-12 * p.Second), $"dynamic at {position}: [{string.Join(", ", actual)}], expected [{string.Join(", ", expected)}]");
        }
    }

    // The configuration as read from a file (numbers parsed from text, as config.json gives them).
    internal static JsonObject Parsed(JsonObject config) => JsonNode.Parse(config.ToJsonString())!.AsObject();

    // A small config.json in the Hugging Face layout (SmallSpec's sizes).
    internal static JsonObject FamilyConfig(string architecture) => new()
    {
        ["architectures"] = new JsonArray(architecture),
        ["vocab_size"] = SmallSpec.Vocabulary, ["hidden_size"] = SmallSpec.Dim, ["num_hidden_layers"] = SmallSpec.Layers,
        ["num_attention_heads"] = SmallSpec.Heads, ["num_key_value_heads"] = SmallSpec.KvHeads, ["head_dim"] = SmallSpec.HeadDim,
        ["intermediate_size"] = SmallSpec.FfDim, ["max_position_embeddings"] = SmallSpec.MaxPositions, ["rms_norm_eps"] = SmallSpec.NormEpsilon,
        ["rope_theta"] = 500.0, ["hidden_act"] = "silu",
    };

    internal static void ModelFamilyConfigs(Device device)
    {
        static DecoderSpec Read(IPretrainedArchitecture architecture, JsonObject config) => architecture.Spec(Parsed(config), []);

        // Mistral: the window in every layer (none when null).
        var mistral = FamilyConfig("MistralForCausalLM");
        mistral["sliding_window"] = 3;
        var spec = Read(MistralArchitectures.Mistral(), mistral);
        Check(spec.SlidingWindow == 3 && spec.SlidingWindowLayers is null && spec.IsWindowed(0) && spec.IsWindowed(1), "Mistral: window in every layer");
        mistral["sliding_window"] = null;
        Check(Read(MistralArchitectures.Mistral(), mistral).SlidingWindow is null, "Mistral: no window");

        // Qwen2: only with use_sliding_window, from max_window_layers on; layer_types when given.
        var qwen = FamilyConfig("Qwen2ForCausalLM");
        qwen["sliding_window"] = 4;
        qwen["max_window_layers"] = 1;
        qwen["use_sliding_window"] = false;
        Check(Read(QwenArchitectures.Qwen2(), qwen).SlidingWindow is null, "Qwen2: use_sliding_window false");
        qwen["use_sliding_window"] = true;
        spec = Read(QwenArchitectures.Qwen2(), qwen);
        Check(spec.SlidingWindow == 4 && !spec.IsWindowed(0) && spec.IsWindowed(1), "Qwen2: windows from max_window_layers on");
        qwen["layer_types"] = new JsonArray("sliding_attention", "full_attention");
        spec = Read(QwenArchitectures.Qwen2(), qwen);
        Check(spec.IsWindowed(0) && !spec.IsWindowed(1), "layer_types decide");

        // Gemma 2: transformers' defaults; Gemma 3: q/k norm, local base, five windowed layers in six.
        var gemma2 = FamilyConfig("Gemma2ForCausalLM");
        gemma2["hidden_activation"] = "gelu_pytorch_tanh";
        gemma2.Remove("hidden_act");
        gemma2["query_pre_attn_scalar"] = 16;
        gemma2["sliding_window"] = 3;
        spec = Read(GemmaArchitectures.Gemma2(), gemma2);
        Check(spec is { AttentionSoftcap: 50f, LogitSoftcap: 30f, AttentionScale: 0.25f, SlidingWindow: 3, PostNorms: true, NormOffset: 1f, TieEmbeddings: true, QkNorm: false }
              && spec.Activation == FeedForwardActivation.Gelu && spec.IsWindowed(0) && !spec.IsWindowed(1), $"Gemma 2 spec: {spec.ToJson().ToJsonString()}");
        var gemma3 = FamilyConfig("Gemma3ForCausalLM");
        gemma3["num_hidden_layers"] = 7;
        gemma3["rope_local_base_freq"] = 10.0;
        gemma3["attn_logit_softcapping"] = null;
        spec = Read(GemmaArchitectures.Gemma3(), gemma3);
        Check(spec is { QkNorm: true, AttentionSoftcap: null, LogitSoftcap: null, SlidingWindow: 4096 } && spec.SlidingWindowRope!.Theta == 10f
              && Enumerable.Range(0, 7).Count(spec.IsWindowed) == 6 && !spec.IsWindowed(5), $"Gemma 3 spec: {spec.ToJson().ToJsonString()}");

        // A Gemma 2 checkpoint (four norms per layer under their own names) loads and matches its spec.
        gemma2["sliding_window"] = 3;
        gemma2["attn_logit_softcapping"] = 2.0;
        gemma2["final_logit_softcapping"] = 3.0;
        var gemmaSpec = Read(GemmaArchitectures.Gemma2(), gemma2);
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
                string stored = GemmaArchitectures.TensorName(name)!;
                bool linear = (name.Contains(".attn.") && !name.Contains("_norm") || name.Contains(".mlp.")) && name.EndsWith(".weight");
                int[] shape = ShapeOf(gemmaSpec, name);
                tensors.Add(linear ? (stored, [shape[1], shape[0]], Transpose2D(values, shape[0], shape[1])) : (stored, shape, values));
            }

            Check(tensors.Any(t => t.Item1 == "model.layers.1.pre_feedforward_layernorm.weight") && tensors.Any(t => t.Item1 == "model.layers.0.post_feedforward_layernorm.weight"),
                "Gemma 2 norm names");
            SafeTensorsWriter.Write(Path.Combine(folder, "model.safetensors"), tensors);
            using (var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
            {
                Check(model.Notes.Count == 0, string.Join("; ", model.Notes));
                AssertClose(expected, model.Network.Predict(input).ToArray(), 1e-4f, "Gemma 2 checkpoint");
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    internal static void GgufYarn(Device device)
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
            Check(fromGguf.Spec.MaxPositions == fromHf.Spec.MaxPositions, $"the context from GGUF: {fromGguf.Spec.MaxPositions} vs {fromHf.Spec.MaxPositions}");
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
