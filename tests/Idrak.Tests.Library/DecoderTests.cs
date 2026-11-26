// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Abstraction.Operations;
using Idrak.Datasets.Abstractions;
using Idrak.Generation;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Models.Abstractions;
using Idrak.Optimizers;
using Idrak.Text.Plugins.Llama;

// Decoder-only language models described by DecoderSpec, checked against a plain-loop reference implementation.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] Decoder =
    [
        ("decoder: RMSNorm and rotary embedding gradients (finite differences)", DecoderGradients),
        ("decoder: tiled attention gradients (finite differences through queries, keys and values)", AttentionGradients),
        ("decoder: DecoderSpec variants match a plain reference implementation (GQA, q/k norm, biases, rope, tied, post-norms, parallel)", DecoderMatchesReference),
        ("decoder: LoRA by layer name trains; JSON and package round trip", DecoderLoraAndPackage),
    ];

    internal static void DecoderGradients(Device device)
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
            GradCheck(device, [3, 5], x => (Tensor.GatedActivation(x, other, (FeedForwardActivation)k) * mix).Sum(), avoidZero: k == 2);
            GradCheck(device, [3, 5], x => (Tensor.GatedActivation(other, x, (FeedForwardActivation)k) * mix).Sum());
            using var g = Tensor.From([.. Enumerable.Range(0, 15).Select(i => (i - 7) * 0.4f)], [3, 5], device);
            var separate = (k switch { 0 => g * g.Sigmoid(), 1 => g.Gelu(), _ => g.Relu() }) * other;
            AssertClose(separate.ToArray(), Tensor.GatedActivation(g, other, (FeedForwardActivation)k).ToArray(), 1e-5f, $"gated activation {k}");
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

    // Fails at the first element whose bits differ.
    internal static void ExpectBits(float[] expected, float[] actual, string what)
    {
        if (SameBits(expected, actual, what) is { } problem)
        {
            Check(false, problem + " (bit for bit)");
        }
    }

    // The int8 decoding kernel as it ran one query row at a time, for each variant: each score the bytes' dot product times
    // the key's scale and the scale, the softmax terms, each value's bytes added with its weight times its scale.
    internal static float[][] Int8Reference(float[] q, KeyValueCache cache, int heads, int rowsPerHead, int steps, int position, int capacity, int dim, float scale,
        AttentionVariant[] variants)
    {
        int stride = (dim + 3) / 4 * 4;
        var kb = System.Runtime.InteropServices.MemoryMarshal.Cast<float, sbyte>(cache.Keys.ToArray()).ToArray();
        var vb = System.Runtime.InteropServices.MemoryMarshal.Cast<float, sbyte>(cache.Values.ToArray()).ToArray();
        float[] ks = cache.KeyScales!.ToArray(), vs = cache.ValueScales!.ToArray();
        var results = new float[variants.Length][];
        for (int v = 0; v < variants.Length; v++)
        {
            var variant = variants[v];
            var y = results[v] = new float[heads * rowsPerHead * dim];
            var scores = new float[capacity];
            for (int row = 0; row < heads * rowsPerHead; row++)
            {
                int h = row / rowsPerHead, end = Math.Min(position + row % rowsPerHead % steps, capacity - 1) + 1, begin = variant.Start(end), count = end - begin;
                var query = q.AsSpan(row * dim, dim);
                float max = float.NegativeInfinity;
                for (int c = 0; c < count; c++)
                {
                    scores[c] = variant.Cap(DotBytes(query, kb.AsSpan((h * capacity + begin + c) * stride, dim)) * ks[h * capacity + begin + c] * scale);
                    max = MathF.Max(max, scores[c]);
                }

                float sum = CpuMath.ExpShifted(scores.AsSpan(0, count), max);
                var output = y.AsSpan(row * dim, dim);
                for (int c = 0; c < count; c++)
                {
                    AddBytes(output, vb.AsSpan((h * capacity + begin + c) * stride, dim), scores[c] / sum * vs[h * capacity + begin + c]);
                }
            }
        }

        return results;

        // Σ a[j] · q[j] in the CPU's fixed lanes, plainly (plan 37, P0): product j rounded and added onto lane j mod 16 in
        // order, then lane i += lane i + 8, + 4, + 2, + 1.
        static float DotBytes(ReadOnlySpan<float> a, ReadOnlySpan<sbyte> q)
        {
            Span<float> lanes = stackalloc float[16];
            lanes.Clear();
            for (int j = 0; j < q.Length; j++)
            {
                lanes[j % 16] += a[j] * q[j];
            }

            for (int width = 8; width > 0; width /= 2)
            {
                for (int i = 0; i < width; i++)
                {
                    lanes[i] += lanes[i + width];
                }
            }

            return lanes[0];
        }

        // acc[j] += scale · q[j] (one product, one sum per element).
        static void AddBytes(Span<float> acc, ReadOnlySpan<sbyte> q, float scale)
        {
            for (int j = 0; j < q.Length; j++)
            {
                acc[j] += scale * q[j];
            }
        }
    }

    internal static void AttentionGradients(Device device)
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
    internal sealed class RandomWeights(int seed) : IWeightSource
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
    internal static float[] ReferenceDecoder(DecoderSpec s, RandomWeights w, int[] ids)
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
            var freq = scaled.FrequenciesAt?.At(position) ?? scaled.Frequencies;
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

    internal static DecoderSpec SmallSpec => new()
    {
        Vocabulary = 23, Dim = 16, Layers = 2, Heads = 4, KvHeads = 2, HeadDim = 6, FfDim = 24, MaxPositions = 32,
        Rope = new RopeSettings(500f), NormEpsilon = 1e-5f,
    };

    internal static void DecoderMatchesReference(Device device)
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
                Rope = new RopeSettings(500f, Scaling: LlamaRopeScaling.Create(8, 1, 4, 8)),
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

    // Runs `action`, which must throw T with `fragment` in its message.
    internal static void Expect<T>(Action action, string fragment, string what) where T : Exception
    {
        try
        {
            action();
        }
        catch (T ex)
        {
            Check(ex.Message.Contains(fragment, StringComparison.Ordinal), $"{what}: {ex.Message}");
            return;
        }

        Check(false, $"{what} must be refused");
    }

    // One decoding step of a sandwich block (norms after attention and the feed-forward): where the device has the fused
    // kernels, each block's output and down projections run with the norm after them, the residual addition and the next
    // norm as one operation (two a block); on CUDA one launch each, so the step runs no norm kernel of its own but the
    // first block's (four a block before: the norm after each projection, then the addition with the next norm).
    internal static void SandwichStep(Sequential model, Device device, DecoderSpec spec, int[] ids, string label)
    {
        model.Eval();
        using var context = new DecodingContext(device, 1, 256);
        using (Autograd.NoGrad())
        {
            using (var prompt = Tensor.From([.. ids.Take(4).Select(i => (float)i)], [1, 4], device))
            using (var first = Tensor.From([(float)ids[4]], [1, 1], device))
            {
                model.ForwardCached(prompt, context).Dispose();
                model.ForwardCached(first, context).Dispose();               // the products' split counts measured
            }

            using var next = Tensor.From([(float)ids[5]], [1, 1], device);
            long fused, norms, adds;
            using (var trace = Kernels.Trace(device.Backend))
            {
                model.ForwardCached(next, context).Dispose();
                device.Synchronize();
                (fused, norms, adds) = (trace.Calls(Ops.PackedMatMulNormAddRmsNorm), trace.Calls(Ops.RmsNormAffine), trace.Calls(Ops.AddRmsNormAffine));
            }

            // A down projection that reads the activation itself (int4) keeps its own product, its norm and the addition.
            bool downFuses = model.Descendants().OfType<FeedForward>().First().DownFusable;
            int expected = device.Backend.FusedKernels ? (downFuses ? 2 : 1) * spec.Layers : 0;
            Check(fused == expected, $"{label}: {fused} projections with their norm, addition and next norm in a step (expected {expected})");
            if (device.Type == DeviceType.Cuda)
            {
                var (norm, add) = downFuses ? (1, 0) : (1 + spec.Layers, spec.Layers);
                Check(norms == norm && adds == add, $"{label}: {norms} RmsNormAffine and {adds} AddRmsNormAffine launches in a step (expected {norm} and {add}: the norms after the output and down projections and the additions with the next norms run inside those projections)");
            }
        }
    }

    internal static void DecoderLoraAndPackage(Device device)
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
