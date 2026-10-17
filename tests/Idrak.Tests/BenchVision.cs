// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Abstraction.Generation;
using Idrak.Gemma3Vision;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Vision;

// Gemma 3 4B's image side at its real sizes with random weights (SigLIP: 896 x 896, 4,096 patches, 27 layers of 1,152,
// 16 heads, MLP 4,304; the projector to 256 tokens of 2,560), on each device IDRAK_DEVICES names (default the CPU):
// the time of one image through the encoder and the projector, float32. With "bf16" the projections are packed
// bfloat16 weights, as the real checkpoint stores them and a read builds them (EncoderWeights.BFloat16), timed under
// both product precisions (float32, and bfloat16 tensor cores as ImageReaderOptions.EncoderPrecision sets them) for
// batches of 1 and 5 images, cold and warm, with the peak device memory and how far the bfloat16 features are from the
// float32 ones.
internal static partial class Tests
{
    internal static int BenchVision(string? weights = null)
    {
        var config = SiglipVisionConfig.FromJson(JsonNode.Parse("""
            {"hidden_size": 1152, "image_size": 896, "intermediate_size": 4304, "num_attention_heads": 16, "num_hidden_layers": 27, "patch_size": 14}
            """)!.AsObject());
        List<Device> devices = Environment.GetEnvironmentVariable("IDRAK_DEVICES") is { Length: > 0 } chosen
            ? [.. chosen.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(Device.Parse)]
            : [Device.Cpu];
        if (weights is not null)
        {
            if (weights != "bf16")
            {
                Console.WriteLine($"--bench-vision takes bf16 or nothing, not '{weights}'.");
                return 2;
            }

            foreach (var device in devices)
            {
                BenchVisionPacked(config, device);
            }

            return 0;
        }

        foreach (var device in devices)
        {
            var build = Stopwatch.StartNew();
            using var store = new RandomTensorStore(config, textDim: 2560);
            using var encoder = VisionTransformerEncoder.FromTensors(config.ToTransformer(), store, "vision.", device);
            using var projector = Gemma3Projector.FromTensors(store, config.Dim, 2560, poolSize: 4, normEpsilon: 1e-6f, device: device);
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

    // The encoder and projector over packed bfloat16 weights: each precision and batch, cold then the best of three warm
    // passes (each pass ends with the features read back), ms a pass and an image, the peak device memory; then the
    // bfloat16 features' error against the float32 ones (one image).
    private static void BenchVisionPacked(SiglipVisionConfig config, Device device)
    {
        var build = Stopwatch.StartNew();
        using var store = new RandomTensorStore(config, textDim: 2560);
        using var encoder = VisionTransformerEncoder.FromTensors(config.ToTransformer(), store, "vision.", device, EncoderWeights.BFloat16);
        using var projector = Gemma3Projector.FromTensors(store, config.Dim, 2560, poolSize: 4, normEpsilon: 1e-6f, device: device, weights: EncoderWeights.BFloat16);
        Console.WriteLine($"{device}: {encoder} with bfloat16 projections, built in {build.Elapsed.TotalSeconds:F1} s; tensor cores: "
            + $"{MixedPrecision.TensorCoresUnavailable(device) ?? "yes"}");
        var features = new Dictionary<MatMulPrecision, float[]>();
        foreach (var precision in new[] { MatMulPrecision.Float32, MatMulPrecision.BFloat16 })
        {
            foreach (int batch in new[] { 1, 5 })
            {
                using var pixels = Tensor.From(RandomArray(new Random(5), batch * 3 * 896 * 896), [batch, 3, 896, 896], device);
                double cold = 0, warm = double.MaxValue;
                long peak = 0;
                for (int run = 0; run < 4; run++)
                {
                    ComputeResources.ResetPeakMemoryUsage(device);
                    var watch = Stopwatch.StartNew();
                    using (MixedPrecision.Use(precision))
                    using (Autograd.NoGrad())
                    using (new TensorScope())
                    {
                        var output = projector.Forward(encoder.Forward(pixels));
                        var values = output.ToArray();
                        if (batch == 1 && run == 0)
                        {
                            features[precision] = values;
                        }
                    }

                    double ms = watch.Elapsed.TotalMilliseconds;
                    (cold, warm) = run == 0 ? (ms, warm) : (cold, Math.Min(warm, ms));
                    peak = Math.Max(peak, ComputeResources.GetMemoryUsage(device).Peak);
                }

                Console.WriteLine($"  {precision} products, {batch} image{(batch == 1 ? "" : "s")}: cold {cold:F0} ms, warm {warm:F1} ms a pass, {warm / batch:F1} ms an image; "
                    + $"peak device memory {peak >> 20} MB");
            }
        }

        var (f32, bf16) = (features[MatMulPrecision.Float32], features[MatMulPrecision.BFloat16]);
        double dot = 0, a = 0, b = 0, error = 0, norm = 0;
        for (int i = 0; i < f32.Length; i++)
        {
            (dot, a, b) = (dot + (double)f32[i] * bf16[i], a + (double)f32[i] * f32[i], b + (double)bf16[i] * bf16[i]);
            (error, norm) = (error + ((double)f32[i] - bf16[i]) * (f32[i] - bf16[i]), norm + (double)f32[i] * f32[i]);
        }

        Console.WriteLine($"  bfloat16 features against float32: cosine {dot / Math.Sqrt(a * b):F5}, relative error {Math.Sqrt(error / Math.Max(norm, 1e-30)):G3}");
    }

    // SigLIP's and the projector's tensors under Gemma3Vision's names, random (norm gains near 1, weights near 0).
    private sealed class RandomTensorStore : ITensorStore
    {
        private readonly Dictionary<string, int[]> _shapes = new(StringComparer.Ordinal);

        public RandomTensorStore(SiglipVisionConfig c, int textDim)
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
