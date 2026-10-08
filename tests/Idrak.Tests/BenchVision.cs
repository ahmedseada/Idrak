// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;

// Gemma 3 4B's image side at its real sizes with random weights (SigLIP: 896 x 896, 4,096 patches, 27 layers of 1,152,
// 16 heads, MLP 4,304; the projector to 256 tokens of 2,560), on each device IDRAK_DEVICES names (default the CPU):
// the time of one image through the encoder and the projector, float32.
internal static partial class Tests
{
    internal static int BenchVision()
    {
        var config = VisionEncoderConfig.FromJson(JsonNode.Parse("""
            {"hidden_size": 1152, "image_size": 896, "intermediate_size": 4304, "num_attention_heads": 16, "num_hidden_layers": 27, "patch_size": 14}
            """)!.AsObject());
        List<Device> devices = Environment.GetEnvironmentVariable("IDRAK_DEVICES") is { Length: > 0 } chosen
            ? [.. chosen.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(Device.Parse)]
            : [Device.Cpu];
        foreach (var device in devices)
        {
            var build = Stopwatch.StartNew();
            using var store = new RandomTensorStore(config, textDim: 2560);
            using var encoder = SiglipVisionEncoder.FromTensors(config, store, device: device);
            using var projector = ImageProjector.FromTensors(store, config.Dim, 2560, poolSize: 4, normEpsilon: 1e-6f, device: device);
            Console.WriteLine($"{device}: {encoder}, {encoder.ParameterCount / 1e6:F0} M parameters, built in {build.Elapsed.TotalSeconds:F1} s");
            using var pixels = Tensor.From(RandomArray(new Random(5), 3 * 896 * 896), [1, 3, 896, 896], device);
            for (int run = 0; run < 2; run++)
            {
                ComputeResources.ResetPeakMemoryUsage(device);
                var watch = Stopwatch.StartNew();
                using (Autograd.NoGrad())
                using (new TensorScope())
                {
                    var features = projector.Forward(encoder.Forward(pixels));
                    _ = features.ToArray();
                    if (run == 1)
                    {
                        Console.WriteLine($"{device}: one 896 x 896 image to {Tensor.FormatShape(features.Shape)} in {watch.Elapsed.TotalSeconds:F2} s "
                            + $"(run {run + 1}), peak device memory {ComputeResources.GetMemoryUsage(device).Peak >> 20} MB, process peak {PeakMegabytes()} MB");
                    }
                    else
                    {
                        Console.WriteLine($"{device}: first run {watch.Elapsed.TotalSeconds:F2} s");
                    }
                }
            }
        }

        return 0;
    }

    // SigLIP's and the projector's tensors under PretrainedVision's names, random (norm gains near 1, weights near 0).
    private sealed class RandomTensorStore : ITensorStore
    {
        private readonly Dictionary<string, int[]> _shapes = new(StringComparer.Ordinal);

        public RandomTensorStore(VisionEncoderConfig c, int textDim)
        {
            int d = c.Dim;
            _shapes["vision.embeddings.patch_embedding.weight"] = [d, c.Channels, c.PatchSize, c.PatchSize];
            _shapes["vision.embeddings.patch_embedding.bias"] = [d];
            _shapes["vision.embeddings.position_embedding.weight"] = [c.Patches, d];
            for (int i = 0; i < c.Layers; i++)
            {
                string l = $"vision.encoder.layers.{i}";
                foreach (string p in new[] { "q_proj", "k_proj", "v_proj", "out_proj" })
                {
                    _shapes[$"{l}.self_attn.{p}.weight"] = [d, d];
                    _shapes[$"{l}.self_attn.{p}.bias"] = [d];
                }

                _shapes[$"{l}.layer_norm1.weight"] = _shapes[$"{l}.layer_norm1.bias"] = _shapes[$"{l}.layer_norm2.weight"] = _shapes[$"{l}.layer_norm2.bias"] = [d];
                _shapes[$"{l}.mlp.fc1.weight"] = [c.FfDim, d];
                _shapes[$"{l}.mlp.fc1.bias"] = [c.FfDim];
                _shapes[$"{l}.mlp.fc2.weight"] = [d, c.FfDim];
                _shapes[$"{l}.mlp.fc2.bias"] = [d];
            }

            _shapes["vision.post_layernorm.weight"] = _shapes["vision.post_layernorm.bias"] = [d];
            _shapes["projector.mm_soft_emb_norm.weight"] = [d];
            _shapes["projector.mm_input_projection_weight"] = [d, textDim];
        }

        public IEnumerable<string> Names => _shapes.Keys;

        public bool Contains(string name) => _shapes.ContainsKey(name);

        public int[] ShapeOf(string name) => _shapes[name];

        public float[] Read(string name)
        {
            var shape = _shapes[name];
            var random = new Random(name.Length * 7919 + shape.Sum());
            var values = new float[shape.Aggregate(1, (a, b) => a * b)];
            bool gain = name.Contains("norm", StringComparison.Ordinal) && name.EndsWith(".weight", StringComparison.Ordinal);
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = (gain ? 1f : 0f) + (float)(random.NextDouble() - 0.5) * 0.04f;
            }

            return values;
        }

        public void Dispose()
        {
        }
    }
}
