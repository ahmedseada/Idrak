// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;

// Gemma 3's vision-language model (plan 11, phase 4): the tiny reference of tests/Idrak.Tests/data/vlm loads in its three
// tensor layouts and gives transformers' logits for a text-only prompt; its vision part is read as data.
internal static partial class Tests
{
    private static void Gemma3VisionLanguageLoads(Device device)
    {
        long[] ids = ReadNpyInt64(TestData("vlm/reference/prompt-text-input_ids.npy"));
        float[] expected = ReadNpyFloat32(TestData("vlm/reference/prompt-text-logits.npy"));
        using var input = Tensor.From([.. ids.Select(i => (float)i)], [1, ids.Length], device);
        string? firstSpec = null;
        float[]? firstProjection = null;
        foreach (var (folder, layout) in new[] { ("tiny-gemma3-pre452", "transformers 4.x"), ("tiny-gemma3", "4.52 to 4.57"), ("tiny-gemma3-v5", "transformers 5") })
        {
            using var model = PretrainedModel.Load(TestData($"vlm/{folder}"), new PretrainedOptions { Device = device });
            var spec = model.Spec;
            Check(model.Notes.Count == 0, $"{folder}: {string.Join("; ", model.Notes)}");
            Check(spec is { Vocabulary: 366, Dim: 24, Layers: 2, Heads: 4, KvHeads: 2, HeadDim: 8, FfDim: 48, LogitSoftcap: null, AttentionSoftcap: null, QkNorm: true, PostNorms: true, TieEmbeddings: true, SlidingWindow: 8 }
                  && spec.IsWindowed(0) && !spec.IsWindowed(1) && spec.EmbeddingScale == MathF.Sqrt(24f) && spec.AttentionScale == 1f / MathF.Sqrt(12f)
                  && spec.Rope!.Theta == 1e6f && spec.Rope.Scaling is { Type: "linear" } && spec.SlidingWindowRope is { Theta: 1e4f, Scaling: null },
                $"{folder}: spec {spec.ToJson().ToJsonString()}");
            firstSpec ??= spec.ToJson().ToJsonString();
            Check(spec.ToJson().ToJsonString() == firstSpec, $"{folder}: the same spec from every config format");

            // The vision part, as data for the encoder and the image tokens.
            var vision = model.Vision ?? throw new InvalidOperationException($"{folder}: no vision part");
            Check(vision.Layout.Contains(layout, StringComparison.Ordinal), $"{folder}: layout {vision.Layout}");
            Check(vision.ImageTokens == new ImageTokenIds(7, 8, 365, 4) && vision.TextDim == 24 && vision.PoolSize == 2 && vision.ProjectorNormEpsilon == 1e-6f,
                $"{folder}: image tokens {vision.ImageTokens}, pooling {vision.PoolSize}");
            Check(vision.Encoder is { Dim: 8, FfDim: 16, Layers: 2, Heads: 2, HeadDim: 4, ImageSize: 56, PatchSize: 14, PatchesPerSide: 4, Patches: 16, Channels: 3, Activation: "gelu_pytorch_tanh", UseHead: false },
                $"{folder}: encoder {vision.Encoder}");
            Check(vision.Tensors.Count == 3 + 2 * 16 + 2 + 2 && vision.Tensors["projector.mm_input_projection_weight"].Shape.SequenceEqual([8, 24])
                  && vision.Tensors["vision.embeddings.patch_embedding.weight"].Shape.SequenceEqual([8, 3, 14, 14]),
                $"{folder}: {vision.Tensors.Count} vision tensors");
            using (var tensors = vision.OpenTensors())
            {
                var projection = tensors.Read("projector.mm_input_projection_weight");
                firstProjection ??= projection;
                Check(tensors.Names.Count() == vision.Tensors.Count && projection.SequenceEqual(firstProjection), $"{folder}: the projector reads the same under every layout");
            }

            // A text-only prompt gives transformers' logits.
            var logits = model.Network.Predict(input).ToArray();
            float worst = expected.Zip(logits, (a, b) => MathF.Abs(a - b)).Max();
            Console.WriteLine($"    {folder} on {device}: largest logit difference {worst:G3}");
            AssertClose(expected, logits, 2e-5f, $"{folder}: text prompt logits");
        }

        // The original gemma-3-4b-it's config.json names few keys: transformers' defaults fill the rest; the embedding scale is
        // held in bfloat16 and the logits are never soft-capped (whatever final_logit_softcapping says).
        var original = JsonNode.Parse("""
            {"architectures": ["Gemma3ForConditionalGeneration"], "boi_token_index": 255999, "eoi_token_index": 256000, "image_token_index": 262144,
             "mm_tokens_per_image": 256, "model_type": "gemma3", "torch_dtype": "bfloat16",
             "text_config": {"hidden_size": 2560, "intermediate_size": 10240, "model_type": "gemma3_text", "num_hidden_layers": 34,
                             "rope_scaling": {"factor": 8.0, "rope_type": "linear"}, "sliding_window": 1024, "final_logit_softcapping": 30.0},
             "vision_config": {"hidden_size": 1152, "image_size": 896, "intermediate_size": 4304, "model_type": "siglip_vision_model",
                               "num_attention_heads": 16, "num_hidden_layers": 27, "patch_size": 14, "vision_use_head": false}}
            """)!.AsObject();
        var big = PretrainedArchitectures.Get("Gemma3ForConditionalGeneration").Spec(original, []);
        Check(big is { Vocabulary: 262_208, Dim: 2560, Layers: 34, Heads: 8, KvHeads: 4, HeadDim: 256, FfDim: 10240, SlidingWindow: 1024, LogitSoftcap: null, EmbeddingScale: 50.5f, TieEmbeddings: true }
              && Enumerable.Range(0, 34).Count(big.IsWindowed) == 29 && !big.IsWindowed(5) && big.Rope!.Theta == 1e6f && big.Rope.Scaling is { Type: "linear" }
              && big.SlidingWindowRope!.Theta == 1e4f && big.AttentionScale == 1f / 16f,
            $"gemma-3-4b-it's config: {big.ToJson().ToJsonString()}");
        var encoder = VisionEncoderConfig.FromJson(original["vision_config"]!.AsObject());
        Check(encoder is { Patches: 4096, HeadDim: 72, LayerNormEpsilon: 1e-6f }, $"gemma-3-4b-it's vision: {encoder}");

        // Gemma3ForCausalLM reads transformers 5's rope_parameters as the older keys.
        var older = JsonNode.Parse(File.ReadAllText(TestData("vlm/tiny-gemma3/config.json")))!["text_config"]!.AsObject();
        var newer = JsonNode.Parse(File.ReadAllText(TestData("vlm/tiny-gemma3-v5/config.json")))!["text_config"]!.AsObject();
        var causal = PretrainedArchitectures.Get("Gemma3ForCausalLM");
        Check(causal.Spec(older, []).ToJson().ToJsonString() == causal.Spec(newer, []).ToJson().ToJsonString(), "Gemma3ForCausalLM: rope_parameters read as rope_theta, rope_scaling and rope_local_base_freq");

        // A vision tensor whose shape vision_config does not give is refused, naming it.
        string broken = TempFolder();
        try
        {
            string source = TestData("vlm/tiny-gemma3-v5");
            File.Copy(Path.Combine(source, "model.safetensors"), Path.Combine(broken, "model.safetensors"));
            var config = JsonNode.Parse(File.ReadAllText(Path.Combine(source, "config.json")))!.AsObject();
            config["vision_config"]!["intermediate_size"] = 32;
            File.WriteAllText(Path.Combine(broken, "config.json"), config.ToJsonString());
            try
            {
                using var _ = PretrainedModel.Load(broken, new PretrainedOptions { Device = device });
                Check(false, "a vision tensor of the wrong shape is refused");
            }
            catch (InvalidDataException ex)
            {
                Check(ex.Message.Contains("vision_tower.encoder.layers.0.mlp.fc1.weight", StringComparison.Ordinal), ex.Message);
            }
        }
        finally
        {
            Directory.Delete(broken, recursive: true);
        }
    }

    // A NumPy .npy array of little-endian float32 or int64 values, in C order.
    private static float[] ReadNpyFloat32(string path)
    {
        var (bytes, start) = ReadNpy(path, "<f4");
        var values = new float[(bytes.Length - start) / 4];
        Buffer.BlockCopy(bytes, start, values, 0, values.Length * 4);
        return values;
    }

    private static long[] ReadNpyInt64(string path)
    {
        var (bytes, start) = ReadNpy(path, "<i8");
        var values = new long[(bytes.Length - start) / 8];
        Buffer.BlockCopy(bytes, start, values, 0, values.Length * 8);
        return values;
    }

    private static (byte[] Bytes, int Start) ReadNpy(string path, string type)
    {
        var bytes = File.ReadAllBytes(path);
        int major = bytes[6];
        int length = major == 1 ? BitConverter.ToUInt16(bytes, 8) : BitConverter.ToInt32(bytes, 8);
        int start = (major == 1 ? 10 : 12) + length;
        string header = Encoding.ASCII.GetString(bytes, major == 1 ? 10 : 12, length);
        Check(header.Contains($"'descr': '{type}'", StringComparison.Ordinal) && header.Contains("'fortran_order': False", StringComparison.Ordinal), $"{path}: {header}");
        return (bytes, start);
    }
}
