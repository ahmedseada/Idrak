// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Generation;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Optimizers;

// Decoder-only language models described by DecoderSpec, checked against a plain-loop reference implementation.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] Decoder =
    [
        ("decoder: RMSNorm and rotary embedding gradients (finite differences)", DecoderGradients),
        ("decoder: tiled and decoding attention match a direct computation (offsets, groups, odd head sizes)", AttentionKernels),
        ("decoder: tiled attention gradients (finite differences through queries, keys and values)", AttentionGradients),
        ("decoder: DecoderSpec variants match a plain reference implementation (GQA, q/k norm, biases, rope, tied, post-norms, parallel)", DecoderMatchesReference),
        ("decoder: cached decoding (float32 and int8 KV) and int8 weights match the full pass; generation", DecoderCachedAndInt8),
        ("decoder: cached decoding through int8 / int4 / bfloat16 weights (fused head layout, cache writes, split attention, gate/up + activation, projection + residual + norm) matches the full pass and the CPU", DecoderCachedPacked),
        ("decoder: LoRA by layer name trains; JSON and package round trip", DecoderLoraAndPackage),
        ("decoder: tied head shares the table; last-position prefill; bf16 embedding tables save and load", DecoderMemory),
    ];

    private static void DecoderMemory(Device device)
    {
        var spec = SmallSpec with { TieEmbeddings = true };
        int[] ids = [1, 4, 9, 16, 2, 7];
        int T = ids.Length, V = spec.Vocabulary;
        using var sequence = Tensor.From([.. ids.Select(i => (float)i)], [1, T], device);

        // Tied: one table (the head owns no weight), and the head's gradient reaches the embedding.
        using var model = spec.Build(new RandomWeights(64), new DecoderBuildOptions { Device = device });
        var head = model.Descendants().OfType<Linear>().Last();
        var embedding = model.Descendants().OfType<Embedding>().Single();
        Check(ReferenceEquals(head.TiedTo, embedding) && !head.Parameters().Any(), "the head reads the embedding table");
        Check(model.Parameters().Sum(p => (long)p.Size) == spec.ParameterCount, $"parameters {model.Parameters().Sum(p => (long)p.Size)} = {spec.ParameterCount}");
        var full = model.Predict(sequence).ToArray();
        using (var scope = new TensorScope())
        {
            using var hidden = Tensor.From(Enumerable.Range(0, spec.Dim).Select(i => MathF.Sin(i)).ToArray(), [1, spec.Dim], device);
            head.Forward(hidden).Sum().Backward();
            Check(embedding.Weight.Grad is { } g && g.ToArray().Any(v => v != 0f), "tied head gradient reaches the embedding");
        }

        // Last position only: the same last-row logits from a [1, 1, V] output.
        using (Autograd.NoGrad())
        {
            model.Eval();
            using var context = new DecodingContext(device, 1, 16) { LastPositionOnly = true };
            var last = model.ForwardCached(sequence, context);
            Check(last.Shape.SequenceEqual([1, 1, V]), $"last-position logits {Tensor.FormatShape(last.Shape)}");
            AssertClose(full[((T - 1) * V)..], last.ToArray(), 1e-4f, "last-position logits");
        }

        // Frozen builds keep the table as bfloat16; a float model loads the file with a bf16 table as bf16.
        using var half = spec.Build(new RandomWeights(64), new DecoderBuildOptions { Device = device, BFloat16 = true });
        Check(half.Descendants().OfType<Embedding>().Single().BFloat16 is not null, "bf16 embedding table");
        var predicted = half.Predict(sequence).ToArray();
        AssertClose(full, predicted, 0.05f * full.Max(MathF.Abs), "bf16 build near float32");
        string path = Path.GetTempFileName();
        try
        {
            half.Save(path);
            using var reloaded = (SmallSpec with { TieEmbeddings = true }).Build(new RandomWeights(1), new DecoderBuildOptions { Device = device, BFloat16 = true });
            reloaded.Descendants().OfType<Embedding>().Single().ToFloat32(trainable: false);
            reloaded.Load(path);
            Check(reloaded.Descendants().OfType<Embedding>().Single().BFloat16 is not null, "the table loads as bf16");
            Check(reloaded.Predict(sequence).ToArray().SequenceEqual(predicted), "reloaded predictions are identical");
        }
        finally
        {
            File.Delete(path);
        }

        Check(half.ToFloat32() > 0 && half.Descendants().OfType<Embedding>().Single().BFloat16 is null, "ToFloat32 restores the table");
    }

    private static void DecoderGradients(Device device)
    {
        GradCheck(device, [3, 6], x => (x.RmsNormalize(1e-6f) * x).Sum(), tolerance: 3e-2f);
        var r = new Random(60);
        var freq = Enumerable.Range(0, 3).Select(i => 1.0 / Math.Pow(100, 2.0 * i / 6)).ToArray();
        var cos = new float[8 * 3];
        var sin = new float[8 * 3];
        for (int p = 0; p < 8; p++)
        {
            for (int i = 0; i < 3; i++)
            {
                cos[p * 3 + i] = (float)Math.Cos(p * freq[i]);
                sin[p * 3 + i] = (float)Math.Sin(p * freq[i]);
            }
        }

        using var tc = Tensor.From(cos, [8, 3], device);
        using var ts = Tensor.From(sin, [8, 3], device);
        using var positions = Tensor.From([2f, 5f, 7f], [3], device);
        using var weights = Tensor.From([.. Enumerable.Range(0, 2 * 3 * 2 * 8).Select(_ => (float)r.NextDouble())], [2, 3, 2, 8], device);
        foreach (bool interleaved in new[] { false, true })
        {
            GradCheck(device, [2, 3, 2, 8], x => (x.Rope(tc, ts, positions, 3, interleaved) * weights).Sum());
        }

        // Every dimension rotated (no pass-through copy).
        var cos4 = new float[8 * 4];
        var sin4 = new float[8 * 4];
        for (int p = 0; p < 8; p++)
        {
            for (int i = 0; i < 4; i++)
            {
                cos4[p * 4 + i] = (float)Math.Cos(p * (i + 1) * 0.3);
                sin4[p * 4 + i] = (float)Math.Sin(p * (i + 1) * 0.3);
            }
        }

        using var tc4 = Tensor.From(cos4, [8, 4], device);
        using var ts4 = Tensor.From(sin4, [8, 4], device);
        GradCheck(device, [2, 3, 2, 8], x => (x.Rope(tc4, ts4, positions, 4, false) * weights).Sum());

        // Fused act(gate) · up: gradients through both inputs, and the same values as the separate operations.
        using var other = Tensor.From([.. Enumerable.Range(0, 15).Select(_ => (float)(r.NextDouble() * 2 - 1))], [3, 5], device);
        using var mix = Tensor.From([.. Enumerable.Range(0, 15).Select(_ => (float)r.NextDouble())], [3, 5], device);
        for (int kind = 0; kind < 3; kind++)
        {
            int k = kind;
            GradCheck(device, [3, 5], x => (Tensor.GatedActivation(x, other, k) * mix).Sum(), avoidZero: k == 2);
            GradCheck(device, [3, 5], x => (Tensor.GatedActivation(other, x, k) * mix).Sum());
            using var g = Tensor.From([.. Enumerable.Range(0, 15).Select(i => (i - 7) * 0.4f)], [3, 5], device);
            var separate = (k switch { 0 => g * g.Sigmoid(), 1 => g.Gelu(), _ => g.Relu() }) * other;
            AssertClose(separate.ToArray(), Tensor.GatedActivation(g, other, k).ToArray(), 1e-5f, $"gated activation {k}");
        }

        // Fused RMS normalization with a gain (and an offset, as Gemma stores it).
        using var gain = Tensor.From([.. Enumerable.Range(0, 6).Select(i => 0.5f + i * 0.1f)], [6], device);
        using var rows = Tensor.From([.. Enumerable.Range(0, 18).Select(i => MathF.Sin(i * 0.7f))], [3, 6], device);
        foreach (float offset in new[] { 0f, 1f })
        {
            var normalized = rows.RmsNormalize(1e-6f).ToArray();
            var g = gain.ToArray();
            var expected = normalized.Select((v, i) => v * (g[i % 6] + offset)).ToArray();
            AssertClose(expected, rows.RmsNormAffine(gain, 1e-6f, offset).ToArray(), 1e-5f, $"fused RMS norm, offset {offset}");

            // Residual addition and normalization in one pass.
            using var other2 = Tensor.From([.. Enumerable.Range(0, 18).Select(i => MathF.Cos(i * 0.3f))], [3, 6], device);
            var (sum, normalizedSum) = Tensor.AddRmsNormAffine(rows, other2, gain, 1e-6f, offset);
            using var plainSum = rows + other2;
            AssertClose(plainSum.ToArray(), sum.ToArray(), 1e-6f, "fused residual sum");
            AssertClose(plainSum.RmsNormAffine(gain, 1e-6f, offset).ToArray(), normalizedSum.ToArray(), 1e-5f, $"fused add + RMS norm, offset {offset}");
        }

        // Per-head normalization and rotation in one pass, partial and full rotation, interleaved or not.
        using var headGain = Tensor.From([.. Enumerable.Range(0, 8).Select(i => 0.7f + i * 0.05f)], [8], device);
        using var heads = Tensor.From([.. Enumerable.Range(0, 2 * 3 * 2 * 8).Select(i => MathF.Sin(i * 0.37f))], [2, 3, 2, 8], device);
        foreach (var (half, tcos, tsin) in new[] { (3, tc, ts), (4, tc4, ts4) })
        {
            foreach (bool interleaved in new[] { false, true })
            {
                var separate = heads.RmsNormAffine(headGain, 1e-6f, 0f).Rope(tcos, tsin, positions, half, interleaved).ToArray();
                var fused = heads.RmsNormRope(headGain, 1e-6f, 0f, tcos, tsin, positions, half, interleaved).ToArray();
                AssertClose(separate, fused, 1e-5f, $"fused norm + rope, half {half}, interleaved {interleaved}");
            }
        }
    }

    private static void AttentionKernels(Device device)
    {
        var r = new Random(61);
        foreach (var (heads, rowsPerHead, steps, capacity, dim, offset) in new[] { (3, 80, 40, 96, 100, 7), (2, 64, 64, 64, 128, 0), (4, 6, 3, 50, 32, 20), (2, 33, 11, 60, 6, 5),
            (2, 2, 1, 700, 128, 650), (4, 1, 1, 1000, 64, 3), (1, 4, 2, 300, 40, 150) })   // large caches: split over positions (some chunks empty)
        {
            float scale = 1f / MathF.Sqrt(dim);
            var q = Enumerable.Range(0, heads * rowsPerHead * dim).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray();
            var k = Enumerable.Range(0, heads * capacity * dim).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray();
            var v = Enumerable.Range(0, heads * capacity * dim).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray();
            var expected = new float[heads * rowsPerHead * dim];
            for (int h = 0; h < heads; h++)
            {
                for (int i = 0; i < rowsPerHead; i++)
                {
                    int count = Math.Min(offset + i % steps, capacity - 1) + 1;
                    var scores = new double[count];
                    for (int c = 0; c < count; c++)
                    {
                        double dot = 0;
                        for (int d = 0; d < dim; d++)
                        {
                            dot += q[(h * rowsPerHead + i) * dim + d] * k[(h * capacity + c) * dim + d];
                        }

                        scores[c] = dot * scale;
                    }

                    double max = scores.Max(), sum = scores.Sum(x => Math.Exp(x - max));
                    for (int d = 0; d < dim; d++)
                    {
                        double value = 0;
                        for (int c = 0; c < count; c++)
                        {
                            value += Math.Exp(scores[c] - max) / sum * v[(h * capacity + c) * dim + d];
                        }

                        expected[(h * rowsPerHead + i) * dim + d] = (float)value;
                    }
                }
            }

            using var tq = Tensor.From(q, [heads, rowsPerHead, dim], device);
            using var tk = Tensor.From(k, [heads, capacity, dim], device);
            using var tv = Tensor.From(v, [heads, capacity, dim], device);
            using var position = Tensor.From([(float)offset], [1], device);
            string name = $"{heads}×{rowsPerHead} rows, steps {steps}, capacity {capacity}, dim {dim}, offset {offset}";
            AssertClose(expected, Tensor.AttentionTiled(tq, tk, tv, position, steps, scale).ToArray(), 2e-4f, "tiled: " + name);
            using var cache = new KeyValueCache(heads, capacity, dim, device, KeyValueFormat.Float32);
            cache.Keys.Load(k);
            cache.Values.Load(v);
            AssertClose(expected, Tensor.AttentionDecode(tq, cache, position, steps, scale).ToArray(), 2e-4f, "decoding: " + name);

            // Int8 cache: the new kernels read only filled positions; the previous path scores every slot and masks.
            using var cache8 = new KeyValueCache(heads, capacity, dim, device, KeyValueFormat.Int8);
            using var zero = Tensor.From([0f], [1], device);
            Tensor.WriteKeyValuesInt8(tk, cache8.Keys, cache8.KeyScales!, zero);
            Tensor.WriteKeyValuesInt8(tv, cache8.Values, cache8.ValueScales!, zero);
            using var mask = Tensor.DecoderMask(position, steps, capacity);
            var weights = Tensor.AttentionScoresInt8(tq, cache8).ScaleMaskSoftmax(scale, mask);
            var previous = Tensor.AttentionContextInt8(weights, cache8).ToArray();
            AssertClose(previous, Tensor.AttentionInt8(tq, cache8, position, steps, scale, tiled: false).ToArray(), 2e-4f, "int8 decoding: " + name);
            AssertClose(previous, Tensor.AttentionInt8(tq, cache8, position, steps, scale, tiled: true).ToArray(), 2e-4f, "int8 tiled: " + name);

            // bfloat16 cache: the same as float attention over the rounded keys and values.
            float Bf(float x) => BitConverter.Int32BitsToSingle(BFloat16Weight.Round(x) << 16);
            using var rk = Tensor.From([.. k.Select(Bf)], [heads, capacity, dim], device);
            using var rv = Tensor.From([.. v.Select(Bf)], [heads, capacity, dim], device);
            var rounded = Tensor.AttentionTiled(tq, rk, rv, position, steps, scale).ToArray();
            using var cache16 = new KeyValueCache(heads, capacity, dim, device, KeyValueFormat.BFloat16);
            Tensor.WriteKeyValuesBFloat16(tk, cache16.Keys, zero, dim);
            Tensor.WriteKeyValuesBFloat16(tv, cache16.Values, zero, dim);
            AssertClose(rounded, Tensor.AttentionBFloat16(tq, cache16, position, steps, scale, tiled: false).ToArray(), 2e-4f, "bf16 decoding: " + name);
            AssertClose(rounded, Tensor.AttentionBFloat16(tq, cache16, position, steps, scale, tiled: true).ToArray(), 2e-4f, "bf16 tiled: " + name);
        }
    }

    private static void AttentionGradients(Device device)
    {
        // Two heads, two query rows per key/value head and step (grouped-query layout: rows g·t + step), head size 6.
        const int Heads = 2, Steps = 5, Rows = 2 * Steps, Dim = 6;
        var r = new Random(62);
        float[] Random(int n) => [.. Enumerable.Range(0, n).Select(_ => (float)(r.NextDouble() * 2 - 1))];
        using var q = Tensor.From(Random(Heads * Rows * Dim), [Heads, Rows, Dim], device);
        using var k = Tensor.From(Random(Heads * Steps * Dim), [Heads, Steps, Dim], device);
        using var v = Tensor.From(Random(Heads * Steps * Dim), [Heads, Steps, Dim], device);
        using var weights = Tensor.From(Random(Heads * Rows * Dim), [Heads, Rows, Dim], device);
        using var zero = Tensor.From([0f], [1], device);
        const float Scale = 0.4f;
        GradCheck(device, [Heads, Rows, Dim], x => (Tensor.CausalAttention(x, k, v, zero, Steps, Scale) * weights).Sum());
        GradCheck(device, [Heads, Steps, Dim], x => (Tensor.CausalAttention(q, x, v, zero, Steps, Scale) * weights).Sum());
        GradCheck(device, [Heads, Steps, Dim], x => (Tensor.CausalAttention(q, k, x, zero, Steps, Scale) * weights).Sum());

        // Several query and key tiles, a head size that is not a multiple of 32: the device gradients equal the CPU's.
        if (device.Type != DeviceType.Cpu)
        {
            const int H = 2, T = 70, R = 2 * T, E = 100;
            float[] qs = Random(H * R * E), ks = Random(H * T * E), vs = Random(H * T * E), ws = Random(H * R * E);
            (float[] Q, float[] K, float[] V) Gradients(Device on)
            {
                using var tq = Tensor.From(qs, [H, R, E], on, requiresGrad: true);
                using var tk = Tensor.From(ks, [H, T, E], on, requiresGrad: true);
                using var tv = Tensor.From(vs, [H, T, E], on, requiresGrad: true);
                using var tw = Tensor.From(ws, [H, R, E], on);
                using var tz = Tensor.From([0f], [1], on);
                (Tensor.CausalAttention(tq, tk, tv, tz, T, 0.1f) * tw).Sum().Backward();
                return (tq.Grad!.ToArray(), tk.Grad!.ToArray(), tv.Grad!.ToArray());
            }

            var (dq, dk, dv) = Gradients(device);
            var (cq, ck, cv) = Gradients(Device.Cpu);
            AssertClose(cq, dq, 2e-4f, "query gradient, several tiles");
            AssertClose(ck, dk, 2e-4f, "key gradient, several tiles");
            AssertClose(cv, dv, 2e-4f, "value gradient, several tiles");
        }
    }

    // Random named weights in Idrak's layout, shared by DecoderSpec.Build and the reference implementation.
    private sealed class RandomWeights(int seed) : IWeightSource
    {
        private readonly Random _random = new(seed);

        public Dictionary<string, float[]> Values { get; } = [];

        public float[] Read(string name, IReadOnlyList<int> shape)
        {
            if (!Values.TryGetValue(name, out var values))
            {
                int count = shape.Aggregate(1, (a, b) => a * b);
                bool gain = name.EndsWith("norm.weight", StringComparison.Ordinal);
                Values[name] = values = [.. Enumerable.Range(0, count).Select(_ => gain ? 0.8f + 0.4f * _random.NextSingle() : (_random.NextSingle() * 2 - 1) * 0.5f)];
            }

            return values;
        }
    }

    // The model written out with loops, following the published definitions of these architectures.
    private static float[] ReferenceDecoder(DecoderSpec s, RandomWeights w, int[] ids)
    {
        int T = ids.Length, D = s.Dim, hd = s.HeadDim, H = s.Heads, KV = s.KvHeads, G = H / KV;
        float[] W(string name) => w.Values[name];
        float[] MatVec(float[] x, float[] m, int inputs, int outputs, float[]? bias)
        {
            var y = new float[outputs];
            for (int j = 0; j < outputs; j++)
            {
                double sum = bias?[j] ?? 0;
                for (int i = 0; i < inputs; i++)
                {
                    sum += x[i] * m[i * outputs + j];
                }

                y[j] = (float)sum;
            }

            return y;
        }

        float[] Norm(float[] x, string name, int offset = 0, int length = -1)
        {
            length = length < 0 ? x.Length : length;
            var y = (float[])x.Clone();
            var g = W($"{name}.weight");
            if (s.Norm == DecoderNorm.Rms)
            {
                double ms = 0;
                for (int i = 0; i < length; i++)
                {
                    ms += (double)x[offset + i] * x[offset + i];
                }

                double inv = 1 / Math.Sqrt(ms / length + s.NormEpsilon);
                for (int i = 0; i < length; i++)
                {
                    y[offset + i] = (float)(x[offset + i] * inv * (g[i] + s.NormOffset));
                }
            }
            else
            {
                var b = W($"{name}.bias");
                double mean = x.Skip(offset).Take(length).Average(v => (double)v);
                double variance = x.Skip(offset).Take(length).Average(v => (v - mean) * (v - mean));
                for (int i = 0; i < length; i++)
                {
                    y[offset + i] = (float)((x[offset + i] - mean) / Math.Sqrt(variance + s.NormEpsilon) * g[i] + b[i]);
                }
            }

            return y;
        }

        float Act(float v) => s.Activation switch
        {
            FeedForwardActivation.Silu => v / (1 + MathF.Exp(-v)),
            FeedForwardActivation.Gelu => 0.5f * v * (1 + MathF.Tanh(0.7978845608f * (v + 0.044715f * v * v * v))),
            _ => MathF.Max(0, v),
        };

        // The rotation of one head at a position: the scaled frequencies (per position for a scaling that depends on it), the
        // cos and sin multiplied by the scaling's attention factor, as transformers applies them.
        void Rotate(float[] x, int offset, int position, RopeSettings? rope)
        {
            if (rope is null)
            {
                return;
            }

            var scaled = rope.Scaled(hd);
            var freq = scaled.FrequenciesAt?.Invoke(position) ?? scaled.Frequencies;
            int half = freq.Length;
            var original = x.AsSpan(offset, hd).ToArray();
            for (int i = 0; i < half; i++)
            {
                double c = Math.Cos(position * freq[i]) * scaled.AttentionFactor, sn = Math.Sin(position * freq[i]) * scaled.AttentionFactor;
                int a = rope.Interleaved ? 2 * i : i, b = rope.Interleaved ? 2 * i + 1 : i + half;
                x[offset + a] = (float)(original[a] * c - original[b] * sn);
                x[offset + b] = (float)(original[b] * c + original[a] * sn);
            }
        }

        var embed = W("embed.weight");
        var h = Enumerable.Range(0, T).Select(t => embed.AsSpan(ids[t] * D, D).ToArray().Select(v => v * (s.EmbeddingScale ?? 1f)).ToArray()).ToList();
        for (int layer = 0; layer < s.Layers; layer++)
        {
            string p = $"layers.{layer}";
            var layerRope = s.IsWindowed(layer) ? s.SlidingWindowRope ?? s.Rope : s.Rope;
            int WindowStart(int t) => s.IsWindowed(layer) ? Math.Max(0, t - s.SlidingWindow!.Value + 1) : 0;   // the window: the last positions, itself included
            var normed = h.Select(x => Norm(x, $"{p}.attn_norm")).ToList();
            float[]? Bias(string name) => w.Values.GetValueOrDefault(name);
            var q = normed.Select(x => MatVec(x, W($"{p}.attn.q.weight"), D, H * hd, Bias($"{p}.attn.q.bias"))).ToList();
            var k = normed.Select(x => MatVec(x, W($"{p}.attn.k.weight"), D, KV * hd, Bias($"{p}.attn.k.bias"))).ToList();
            var v = normed.Select(x => MatVec(x, W($"{p}.attn.v.weight"), D, KV * hd, Bias($"{p}.attn.v.bias"))).ToList();
            for (int t = 0; t < T; t++)
            {
                for (int head = 0; head < H; head++)
                {
                    if (s.QkNorm)
                    {
                        q[t] = Norm(q[t], $"{p}.attn.q_norm", head * hd, hd);
                    }

                    Rotate(q[t], head * hd, t, layerRope);
                }

                for (int head = 0; head < KV; head++)
                {
                    if (s.QkNorm)
                    {
                        k[t] = Norm(k[t], $"{p}.attn.k_norm", head * hd, hd);
                    }

                    Rotate(k[t], head * hd, t, layerRope);
                }
            }

            var attended = new List<float[]>();
            for (int t = 0; t < T; t++)
            {
                var context = new float[H * hd];
                for (int head = 0; head < H; head++)
                {
                    int kvHead = head / G;
                    var scores = Enumerable.Repeat(double.NegativeInfinity, t + 1).ToArray();
                    for (int c = WindowStart(t); c <= t; c++)
                    {
                        double dot = 0;
                        for (int d = 0; d < hd; d++)
                        {
                            dot += q[t][head * hd + d] * k[c][kvHead * hd + d];
                        }

                        scores[c] = dot * (s.AttentionScale ?? 1 / Math.Sqrt(hd));
                        if (s.AttentionSoftcap is { } cap)
                        {
                            scores[c] = cap * Math.Tanh(scores[c] / cap);
                        }
                    }

                    double max = scores.Max(), total = scores.Sum(v => Math.Exp(v - max));
                    for (int c = WindowStart(t); c <= t; c++)
                    {
                        double weight = Math.Exp(scores[c] - max) / total;
                        for (int d = 0; d < hd; d++)
                        {
                            context[head * hd + d] += (float)(weight * v[c][kvHead * hd + d]);
                        }
                    }
                }

                var o = MatVec(context, W($"{p}.attn.o.weight"), H * hd, D, Bias($"{p}.attn.o.bias"));
                attended.Add(s.PostNorms ? Norm(o, $"{p}.post_attn_norm") : o);
            }

            float[] Dense(float[] x, string m, int hidden)
            {
                var up = MatVec(x, W($"{m}.up.weight"), D, hidden, Bias($"{m}.up.bias"));
                if (s.Gated)
                {
                    var gate = MatVec(x, W($"{m}.gate.weight"), D, hidden, Bias($"{m}.gate.bias"));
                    up = [.. up.Select((u, i) => Act(gate[i]) * u)];
                }
                else
                {
                    up = [.. up.Select(Act)];
                }

                return MatVec(up, W($"{m}.down.weight"), hidden, D, Bias($"{m}.down.bias"));
            }

            // Mixtral / Qwen-MoE: softmax over the router's scores, the k most probable experts (a stable sort, so ties go to
            // the lowest index, as torch.topk), their probabilities renormalized when configured, the weighted sum of
            // their outputs; plus a shared expert scaled by sigmoid of its gate.
            float[] Experts(float[] x)
            {
                var scores = MatVec(x, W($"{p}.mlp.router.weight"), D, s.Experts, null);
                double max = scores.Max();
                var probabilities = scores.Select(v => Math.Exp(v - max)).ToArray();
                double total = probabilities.Sum();
                var chosen = Enumerable.Range(0, s.Experts).OrderByDescending(e => (float)(probabilities[e] / total)).Take(s.ExpertsPerToken).ToList();
                double norm = s.NormalizeTopK ? chosen.Sum(e => probabilities[e] / total) : 1;
                var output = new double[D];
                foreach (int e in chosen)
                {
                    var y = Dense(x, $"{p}.mlp.experts.{e}", s.ExpertFfDim > 0 ? s.ExpertFfDim : s.FfDim);
                    double weight = probabilities[e] / total / norm;
                    for (int i = 0; i < D; i++)
                    {
                        output[i] += weight * y[i];
                    }
                }

                if (s.SharedExpertFfDim > 0)
                {
                    var y = Dense(x, $"{p}.mlp.shared", s.SharedExpertFfDim);
                    double gate = 1 / (1 + Math.Exp(-MatVec(x, W($"{p}.mlp.shared_gate.weight"), D, 1, null)[0]));
                    for (int i = 0; i < D; i++)
                    {
                        output[i] += gate * y[i];
                    }
                }

                return [.. output.Select(v => (float)v)];
            }

            float[] FeedForward(float[] x)
            {
                var down = s.IsExpertLayer(layer) ? Experts(x) : Dense(x, $"{p}.mlp", s.FfDim);
                return s.PostNorms ? Norm(down, $"{p}.post_mlp_norm") : down;
            }

            for (int t = 0; t < T; t++)
            {
                if (s.ParallelBlocks)
                {
                    var f = FeedForward(normed[t]);
                    h[t] = [.. h[t].Select((x, i) => x + attended[t][i] + f[i])];
                }
                else
                {
                    var x = h[t].Select((value, i) => value + attended[t][i]).ToArray();
                    var f = FeedForward(Norm(x, $"{p}.mlp_norm"));
                    h[t] = [.. x.Select((value, i) => value + f[i])];
                }
            }
        }

        var headWeights = s.TieEmbeddings ? null : W("head.weight");
        var logits = new List<float>();
        foreach (var x in h)
        {
            var normed = Norm(x, "norm");
            for (int token = 0; token < s.Vocabulary; token++)
            {
                double sum = s.HeadBias ? w.Values["head.bias"][token] : 0;
                for (int i = 0; i < D; i++)
                {
                    sum += normed[i] * (headWeights is null ? embed[token * D + i] : headWeights[i * s.Vocabulary + token]);
                }

                logits.Add((float)(s.LogitSoftcap is { } cap ? cap * Math.Tanh(sum / cap) : sum));
            }
        }

        return [.. logits];
    }

    private static DecoderSpec SmallSpec => new()
    {
        Vocabulary = 23, Dim = 16, Layers = 2, Heads = 4, KvHeads = 2, HeadDim = 6, FfDim = 24, MaxPositions = 32,
        Rope = new RopeSettings(500f), NormEpsilon = 1e-5f,
    };

    private static void DecoderMatchesReference(Device device)
    {
        var variants = new (string Name, DecoderSpec Spec)[]
        {
            ("grouped-query, head size 6", SmallSpec),
            ("q/k norm, qkv biases, untied", SmallSpec with { QkNorm = true, QkvBias = true }),
            ("tied, partial interleaved rope, scale, offset", SmallSpec with
            {
                TieEmbeddings = true, EmbeddingScale = 4f, NormOffset = 1f, Rope = new RopeSettings(100f, RotaryDim: 4, Interleaved: true),
            }),
            ("post-norms, GELU, llama3 rope scaling", SmallSpec with
            {
                PostNorms = true, Activation = FeedForwardActivation.Gelu,
                Rope = new RopeSettings(500f, Scaling: RopeScaling.Llama3(8, 1, 4, 8)),
            }),
            ("parallel, layer norm, plain feed-forward, all biases", SmallSpec with
            {
                ParallelBlocks = true, Norm = DecoderNorm.Layer, Gated = false, QkvBias = true, OutputBias = true, FeedForwardBias = true,
                HeadBias = true, KvHeads = 4,
            }),
        };
        int[] ids = [3, 17, 5, 5, 22, 0, 9];
        foreach (var (name, spec) in variants)
        {
            var weights = new RandomWeights(61);
            using var model = spec.Build(weights, new DecoderBuildOptions { Device = device });
            var reference = ReferenceDecoder(spec, weights, ids);
            using var input = Tensor.From([.. ids.Select(i => (float)i)], [1, ids.Length], device);
            using var logits = model.Predict(input);
            AssertClose(reference, logits.ToArray(), 1e-3f, name);
            model.Train();
            using (var scope = new TensorScope())
            {
                AssertClose(reference, model.Forward(input).ToArray(), 1e-3f, $"{name} (training path)");
            }
        }
    }

    private static void DecoderCachedAndInt8(Device device)
    {
        var spec = SmallSpec with { QkNorm = true };
        using var model = spec.Build(new RandomWeights(62), new DecoderBuildOptions { Device = device });
        model.Eval();
        int[] ids = [1, 4, 9, 16, 2, 7, 11, 3, 8, 20];
        int T = ids.Length, V = spec.Vocabulary;
        using var sequence = Tensor.From([.. ids.Select(i => (float)i)], [1, T], device);
        var full = model.Predict(sequence).ToArray();
        float range = full.Max(MathF.Abs);
        foreach (var format in new[] { KeyValueFormat.Float32, KeyValueFormat.Int8, KeyValueFormat.BFloat16 })
        {
            float tolerance = format switch { KeyValueFormat.Float32 => 1e-4f, KeyValueFormat.BFloat16 => 0.01f * range, _ => 0.03f * range };
            using var context = new DecodingContext(device, 1, 16, format);
            using (Autograd.NoGrad())
            {
                using var prompt = Tensor.From([.. ids.Take(4).Select(i => (float)i)], [1, 4], device);
                AssertClose(full[..(4 * V)], model.ForwardCached(prompt, context).ToArray(), tolerance, $"{format} prefill");
                for (int t = 4; t < T; t++)
                {
                    using var next = Tensor.From([(float)ids[t]], [1, 1], device);
                    AssertClose(full[(t * V)..((t + 1) * V)], model.ForwardCached(next, context).ToArray(), tolerance, $"{format} step {t}");
                }
            }
        }

        using var quantized = spec.Build(new RandomWeights(62), new DecoderBuildOptions { Device = device, Int8 = true });
        Check(quantized.Descendants().OfType<Linear>().All(l => l.Int8 is not null), "every projection is int8");
        var int8 = quantized.Predict(sequence).ToArray();
        float change = int8.Zip(full, (a, b) => MathF.Abs(a - b)).Max();
        Check(change < 0.05f * range, $"int8 logit change {change} (range {range})");

        var generator = new TextGenerator(model, new CharTokenizer("abcdefghijklmnopqrstuvw"), 16);
        var (text, _, stats) = generator.Generate("abc", new GenerationOptions { Temperature = 0f, TopK = 1, NumPredict = 8 });
        Check(stats.GeneratedTokens == 8 && text.Length == 8, $"generated '{text}'");
    }

    // Packed weights and a long cache: on CUDA the decoding steps run the fused kernels (q/k normalization, rotation,
    // head layout and cache writes in one pass; attention split over the cache and merged by its last block; output and
    // down projections with the residual addition and next normalization), the full pass the composed ones. Quantized
    // caches only roughly match the full pass, so each device's decoding is also compared with the CPU's (same rounding).
    private static void DecoderCachedPacked(Device device)
    {
        var specs = new (string Name, DecoderSpec Spec)[]
        {
            ("q/k norm", SmallSpec with { Dim = 64, HeadDim = 16, FfDim = 128, MaxPositions = 256, QkNorm = true }),
            ("partial interleaved rope, norm offset, GELU", SmallSpec with
            {
                Dim = 64, HeadDim = 16, FfDim = 128, MaxPositions = 256, NormOffset = 1f, Rope = new RopeSettings(100f, RotaryDim: 8, Interleaved: true),
                Activation = FeedForwardActivation.Gelu,
            }),
        };
        int[] ids = [1, 4, 9, 16, 2, 7, 11, 3, 8, 20];
        int T = ids.Length;
        foreach (var (name, spec) in specs)
        {
            foreach (var weights in new[] { "int8", "int4", "bfloat16" })
            {
                var options = new DecoderBuildOptions { Int8 = weights == "int8", Int4 = weights == "int4", BFloat16 = weights == "bfloat16" };
                using var model = spec.Build(new RandomWeights(63), options with { Device = device });
                using var cpuModel = device.Type == DeviceType.Cpu ? null : spec.Build(new RandomWeights(63), options with { Device = Device.Cpu });
                int V = spec.Vocabulary;
                using var sequence = Tensor.From([.. ids.Select(i => (float)i)], [1, T], device);
                var full = model.Predict(sequence).ToArray();
                float range = full.Max(MathF.Abs);

                // Logits of a 4-token prefill, then one token at a time (the cache is long enough to be split).
                float[] Decode(Sequential m, Device d, KeyValueFormat format)
                {
                    m.Eval();
                    var logits = new List<float>();
                    using var context = new DecodingContext(d, 1, 256, format);
                    using (Autograd.NoGrad())
                    {
                        using var prompt = Tensor.From([.. ids.Take(4).Select(i => (float)i)], [1, 4], d);
                        logits.AddRange(m.ForwardCached(prompt, context).ToArray());
                        for (int t = 4; t < T; t++)
                        {
                            using var next = Tensor.From([(float)ids[t]], [1, 1], d);
                            logits.AddRange(m.ForwardCached(next, context).ToArray());
                        }
                    }

                    return [.. logits];
                }

                foreach (var format in new[] { KeyValueFormat.Float32, KeyValueFormat.BFloat16, KeyValueFormat.Int8 })
                {
                    string label = $"{name}, {weights} weights, {format} cache";
                    var cached = Decode(model, device, format);
                    float tolerance = format switch { KeyValueFormat.Float32 => 1e-3f * range, KeyValueFormat.BFloat16 => 0.04f * range, _ => 0.1f * range };
                    AssertClose(full, cached, tolerance, $"{label}: against the full pass");
                    if (cpuModel is not null)
                    {
                        AssertClose(Decode(cpuModel, Device.Cpu, format), cached, (format == KeyValueFormat.Int8 ? 0.01f : 2e-3f) * range, $"{label}: against the CPU");
                    }
                }
            }
        }
    }

    private static void DecoderLoraAndPackage(Device device)
    {
        var spec = SmallSpec;
        using var model = spec.Build(options: new DecoderBuildOptions { Device = device, Seed = 63 });
        int adapters = model.AddLora(rank: 2, alpha: 4, targets: l => l.Name is "q" or "v", freezeBase: true, random: new Random(64));
        Check(adapters == 2 * spec.Layers, $"{adapters} adapters on the q and v projections");
        var random = new Random(65);
        var ids = Enumerable.Range(0, 4 * 9).Select(_ => (float)random.Next(spec.Vocabulary)).ToArray();
        using var input = Tensor.From(ids, [4, 9], device);
        using var targets = Tensor.OneHot(Tensor.From([.. ids.Select((_, i) => ids[(i / 9) * 9 + (i % 9 + 1) % 9])], [4, 9], device), spec.Vocabulary);
        using var optimizer = new Adam(model.TrainableParameters(), 0.05f);
        model.Train();
        float first = 0, last = 0;
        for (int step = 0; step < 30; step++)
        {
            using var scope = new TensorScope();
            var loss = Losses.CrossEntropy(model.Forward(input), targets);
            optimizer.ZeroGrad();
            loss.Backward();
            optimizer.Step();
            (first, last) = step == 0 ? (loss.Item(), loss.Item()) : (first, loss.Item());
        }

        Check(last < first * 0.8f, $"LoRA loss {first:F3} → {last:F3}");
        model.MergeLora();
        model.Eval();

        Check(DecoderSpec.FromJson(spec.ToJson()) == spec with { }, "JSON round trip");
        var withRope = spec with { Rope = new RopeSettings(1e6f, 4, true, RopeScaling.Linear(2)) };
        Check(DecoderSpec.FromJson(withRope.ToJson()).Rope == withRope.Rope, "rope settings round trip");

        string path = Path.Combine(Path.GetTempPath(), $"ns-{Guid.NewGuid():N}.ikm");
        try
        {
            ModelPackage.Create(path).Architecture(ModelPackage.DefaultModelName, spec.ToJson()).Weights(model).Save();
            using var package = ModelPackage.Open(path);
            using var reloaded = package.BuildModel(device: device);
            using var probe = Tensor.From([1f, 2f, 3f], [1, 3], device);
            AssertClose(model.Predict(probe).ToArray(), reloaded.Predict(probe).ToArray(), 1e-5f, "package round trip");
        }
        finally
        {
            File.Delete(path);
        }
    }
}
