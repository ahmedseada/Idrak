// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.PluginTests;
using Idrak.Text.Retrieval;

// Plan 26, phase P8: the engine for encoders and encoder-decoders. Attention options (score scale, key padding,
// position bias), the encoder and seq2seq contracts with their suites, and toy families made in
// tests/Idrak.PluginTests through the public API.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] EngineGroup =
    [
        ("engine: attention options: with nothing set, MultiHeadAttention and TransformerEncoderLayer give the bits of the same layers with the default score scale set explicitly and of a padding scope that pads nothing (inference, fused, and with the gradient); a score scale of 2/√d gives the bits of the default layer with its query projection doubled", AttentionOptionsOff),
        ("engine: attention options: MultiHeadAttention with a score scale, heads narrower than the model, a position bias (the library's \"relative\"), causal or not, with key padding or none, equals a host reference within 1e-4 at every real position, in inference and with the gradient recorded; each sequence of a padded batch equals itself alone; the bias table gets a gradient", AttentionOptionsReference),
        ("engine: attention options: cached decoding with a position bias (a causal MultiHeadAttention, and a CausalSelfAttention in a DecoderBlock with and without its fused step) equals the whole sequence at once, step by step and after a prefill; a bias shared by two layers is computed once per shape", AttentionOptionsCached),
        ("engine: attention options: invalid settings are refused naming the setting (ScoreScale 0, -1 and NaN on MultiHeadAttention, CausalSelfAttention and CrossAttention; a bias for other heads; max_distance of \"relative\"); the position bias suite passes \"relative\" both ways (its values the table's buckets) and fails a bias that ignores the query's position", AttentionOptionsRefused),
        ("engine: encoders: the poolers first, mean and max pass the pooler suite against plain loops; a pooler that reads the padding fails it", EncoderPoolers),
        ("engine: encoders: the library's SequenceEncoder built from layers (token types, learned positions, blocks sharing a position bias) passes the encoder suite; TextEncoder over it gives each text's unit vector alone and in a batch with the mean, first and max poolers; TextEncoder over a plain model gives the bits of its average as it always computed it", EncoderLibrary),
    ];

    // A projection [in, out] with the given values (its own copies), on the device.
    internal static Linear Projection(float[] weight, float[] bias, int inputs, int outputs, Device device) =>
        Linear.FromWeights(Tensor.Persistent(weight, [inputs, outputs], device, requiresGrad: true), Tensor.Persistent(bias, [outputs], device, requiresGrad: true));

    internal static void BitsEqual(float[] expected, float[] actual, string what)
    {
        Check(expected.Length == actual.Length, $"{what}: {actual.Length} values, {expected.Length} expected");
        for (int i = 0; i < expected.Length; i++)
        {
            Check(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]), $"{what}: element {i} is {actual[i]}, {expected[i]} expected (bit for bit)");
        }
    }

    // ---------------------------------------------------------------- attention options

    internal static void AttentionOptionsOff(Device device)
    {
        var random = new Random(11);
        const int n = 2, t = 7, d = 16, heads = 4, dh = d / heads;
        float[] wqkv = RandomArray(random, d * 3 * d, 0.4f), bqkv = RandomArray(random, 3 * d, 0.1f), wo = RandomArray(random, d * d, 0.4f), bo = RandomArray(random, d, 0.1f);
        float[] x = RandomArray(random, n * t * d);
        MultiHeadAttention Layer(float[] w, float[] b, float? scale, bool causal = false) => scale is { } s
            ? new MultiHeadAttention(Projection(w, b, d, 3 * d, device), Projection(wo, bo, d, d, device), heads, causal) { ScoreScale = s }
            : new MultiHeadAttention(Projection(w, b, d, 3 * d, device), Projection(wo, bo, d, d, device), heads, causal);

        foreach (bool causal in new[] { false, true })
        {
            using var plain = Layer(wqkv, bqkv, null, causal);
            using var explicitScale = Layer(wqkv, bqkv, 1f / MathF.Sqrt(dh), causal);
            float[] Run(MultiHeadAttention layer, bool grad, PaddedSequences? padding = null)
            {
                using var scope = new TensorScope();
                using var noGrad = grad ? null : (IDisposable)Autograd.NoGrad();
                var input = Tensor.From(x, [n, t, d], device);
                return layer.Forward(input, padding).ToArray();
            }

            foreach (bool grad in new[] { false, true })
            {
                float[] reference = Run(plain, grad);
                BitsEqual(reference, Run(explicitScale, grad), $"causal {causal}, gradient {grad}: the default score scale set explicitly");
                using var full = new PaddedSequences([t, t], t, device);
                BitsEqual(reference, Run(plain, grad, full), $"causal {causal}, gradient {grad}: a padding scope that pads nothing");
            }

            // 2/√d on the scores equals the default layer whose query columns (and bias) are doubled: power-of-two scalings are exact.
            float[] doubled = [.. wqkv], doubledBias = [.. bqkv];
            for (int row = 0; row < d; row++)
            {
                for (int c = 0; c < d; c++)
                {
                    doubled[row * 3 * d + c] *= 2f;
                }
            }

            for (int c = 0; c < d; c++)
            {
                doubledBias[c] *= 2f;
            }

            using var scaled = Layer(wqkv, bqkv, 2f / MathF.Sqrt(dh), causal);
            using var queriesDoubled = Layer(doubled, doubledBias, null, causal);
            BitsEqual(Run(queriesDoubled, false), Run(scaled, false), $"causal {causal}: a score scale of 2/√d");
        }

        // A TransformerEncoderLayer (its fused inference pass reads the attention through its epilogues) around both layers.
        var ln = () => new LayerNorm(d, device: device);
        float[] f1 = RandomArray(random, d * 2 * d, 0.3f), f1b = RandomArray(random, 2 * d, 0.1f), f2 = RandomArray(random, 2 * d * d, 0.3f), f2b = RandomArray(random, d, 0.1f);
        using var blockPlain = TransformerEncoderLayer.FromLayers(ln(), Layer(wqkv, bqkv, null), ln(), Projection(f1, f1b, d, 2 * d, device), Projection(f2, f2b, 2 * d, d, device));
        using var blockExplicit = TransformerEncoderLayer.FromLayers(ln(), Layer(wqkv, bqkv, 1f / MathF.Sqrt(dh)), ln(), Projection(f1, f1b, d, 2 * d, device), Projection(f2, f2b, 2 * d, d, device));
        foreach (bool grad in new[] { false, true })
        {
            using var scope = new TensorScope();
            using var noGrad = grad ? null : (IDisposable)Autograd.NoGrad();
            BitsEqual(blockPlain.Forward(Tensor.From(x, [n, t, d], device)).ToArray(), blockExplicit.Forward(Tensor.From(x, [n, t, d], device)).ToArray(),
                $"TransformerEncoderLayer, gradient {grad}: the default score scale set explicitly");
        }
    }

    // The layer's output on the host in doubles: q, k and v from the packed projection, scores · scale + bias (+ the causal
    // mask) over each sequence's first lengths[b] keys (a causal layer: the keys up to the query), softmax, · v, the output
    // projection.
    internal static float[] HostAttention(MultiHeadAttention layer, float[] x, int n, int t, int[]? lengths, float[]? bias)
    {
        var parts = layer.Children().OfType<Linear>().ToList();
        float[] w = parts[0].Weight.ToArray(), wb = parts[0].Bias?.ToArray() ?? new float[3 * layer.Inner];
        float[] o = parts[1].Weight.ToArray(), ob = parts[1].Bias?.ToArray() ?? new float[layer.Dim];
        int d = layer.Dim, inner = layer.Inner, h = layer.Heads, dh = layer.HeadDim, width = 3 * inner;
        var projected = new double[n * t * width];
        for (int r = 0; r < n * t; r++)
        {
            for (int c = 0; c < width; c++)
            {
                double sum = wb[c];
                for (int k = 0; k < d; k++)
                {
                    sum += x[r * d + k] * (double)w[k * width + c];
                }

                projected[r * width + c] = sum;
            }
        }

        var context = new double[n * t * inner];
        var scores = new double[t];
        for (int b = 0; b < n; b++)
        {
            int length = lengths?[b] ?? t;
            for (int head = 0; head < h; head++)
            {
                for (int i = 0; i < t; i++)
                {
                    int end = layer.Causal ? i + 1 : length;
                    double max = double.NegativeInfinity;
                    for (int j = 0; j < end; j++)
                    {
                        double s = 0;
                        for (int e = 0; e < dh; e++)
                        {
                            s += projected[(b * t + i) * width + head * dh + e] * projected[(b * t + j) * width + inner + head * dh + e];
                        }

                        scores[j] = s * layer.ScoreScale + (bias is null ? 0 : bias[(head * t + i) * t + j]);
                        max = Math.Max(max, scores[j]);
                    }

                    double total = 0;
                    for (int j = 0; j < end; j++)
                    {
                        total += scores[j] = Math.Exp(scores[j] - max);
                    }

                    for (int e = 0; e < dh; e++)
                    {
                        double sum = 0;
                        for (int j = 0; j < end; j++)
                        {
                            sum += scores[j] / total * projected[(b * t + j) * width + 2 * inner + head * dh + e];
                        }

                        context[(b * t + i) * inner + head * dh + e] = sum;
                    }
                }
            }
        }

        var output = new float[n * t * d];
        for (int r = 0; r < n * t; r++)
        {
            for (int c = 0; c < d; c++)
            {
                double sum = ob[c];
                for (int e = 0; e < inner; e++)
                {
                    sum += context[r * inner + e] * o[e * d + c];
                }

                output[r * d + c] = (float)sum;
            }
        }

        return output;
    }

    // The values of the real positions only (sequence after sequence).
    internal static float[] RealPositions(float[] values, int[] lengths, int t, int width) =>
        [.. lengths.SelectMany((length, b) => values.Skip(b * t * width).Take(length * width))];

    internal static void AttentionOptionsReference(Device device)
    {
        var random = new Random(23);
        const int n = 3, t = 6, d = 16;
        int[] lengths = [6, 2, 4];
        float[] x = RandomArray(random, n * t * d);
        var cases = new (string Name, int Heads, int Inner, bool Causal, string? Bias, float? Scale, bool Padded)[]
        {
            ("plain, padded", 4, 16, false, null, null, true),
            ("scale 1, padded", 4, 16, false, null, 1f, true),
            ("narrow heads, padded", 2, 8, false, null, null, true),
            ("relative bias", 4, 16, false, RelativePositionBias.KindName, null, false),
            ("relative bias, padded, scale 1", 4, 16, false, RelativePositionBias.KindName, 1f, true),
            ("causal, relative one-way bias", 4, 16, true, RelativePositionBias.KindName, null, false),
            ("causal, padded (the padding after the real positions changes nothing)", 4, 16, true, null, null, true),
        };
        foreach (var c in cases)
        {
            var bias = c.Bias is null ? null : new RelativePositionBias(c.Heads, 3, !c.Causal, device, new Random(4));
            var qkv = new Linear(d, 3 * c.Inner, device: device, random: random);
            var output = new Linear(c.Inner, d, device: device, random: random);
            using var layer = c.Scale is { } scale
                ? new MultiHeadAttention(qkv, output, c.Heads, c.Causal) { ScoreScale = scale, PositionBias = bias }
                : new MultiHeadAttention(qkv, output, c.Heads, c.Causal) { PositionBias = bias };
            int[]? own = c.Padded ? lengths : null;
            int[] real = own ?? [t, t, t];
            float[]? biasValues;
            using (Autograd.NoGrad())
            {
                biasValues = bias?.For(t, t, 0, device).ToArray();
            }

            float[] expected = RealPositions(HostAttention(layer, x, n, t, own, biasValues), real, t, d);
            foreach (bool grad in new[] { false, true })
            {
                using var scope = new TensorScope();
                using var noGrad = grad ? null : (IDisposable)Autograd.NoGrad();
                using var padding = own is null ? null : new PaddedSequences(own, t, device);
                var result = layer.Forward(Tensor.From(x, [n, t, d], device), padding);
                AssertClose(expected, RealPositions(result.ToArray(), real, t, d), 1e-4f, $"{c.Name}, gradient {grad}: the host reference");
                if (grad && bias is RelativePositionBias relative)
                {
                    relative.Table.ZeroGrad();
                    result.Sum().Backward();
                    Check(relative.Table.Grad?.ToArray().Any(v => v != 0f) == true, $"{c.Name}: the bias table gets a gradient");
                }
            }

            // Each sequence alone (its own length, nothing padded) equals its rows of the padded batch.
            if (own is not null)
            {
                using var scope = new TensorScope();
                using var noGrad = Autograd.NoGrad();
                using var padding = new PaddedSequences(own, t, device);
                float[] batch = layer.Forward(Tensor.From(x, [n, t, d], device), padding).ToArray();
                for (int b = 0; b < n; b++)
                {
                    float[] alone = layer.Forward(Tensor.From(x.AsSpan(b * t * d, own[b] * d), [1, own[b], d], device)).ToArray();
                    AssertClose(alone, batch.AsSpan(b * t * d, own[b] * d).ToArray(), 1e-5f, $"{c.Name}: sequence {b} alone and in the padded batch");
                }
            }
        }
    }

    // A bias counting its computations, shared by layers.
    internal sealed class CountingBias(int heads) : AttentionPositionBias(heads)
    {
        public int Computed { get; private set; }

        protected override Tensor Compute(int queries, int keys, int queryStart, Device device)
        {
            Computed++;
            var values = new float[Heads * queries * keys];
            for (int h = 0; h < Heads; h++)
            {
                for (int i = 0; i < queries; i++)
                {
                    for (int j = 0; j < keys; j++)
                    {
                        values[(h * queries + i) * keys + j] = 0.1f * (h + 1) * MathF.Sin(j - queryStart - i);
                    }
                }
            }

            return Tensor.From(values, [Heads, queries, keys], device);
        }
    }

    internal static void AttentionOptionsCached(Device device)
    {
        var random = new Random(31);
        const int t = 9, d = 16, heads = 4;
        float[] x = RandomArray(random, t * d);

        // Logits of a cached module (an attention layer, or a sequential of blocks) step by step from `prefill` at once.
        float[] Cached(Func<Tensor, DecodingContext, Tensor> step, int prefill)
        {
            using var context = new DecodingContext(device, 1, t);
            var result = new List<float>();
            for (int at = 0; at < t;)
            {
                int count = at == 0 ? prefill : 1;
                using var scope = new TensorScope();
                using var noGrad = Autograd.NoGrad();
                context.BeginStep(count);
                var y = step(Tensor.From(x.AsSpan(at * d, count * d), [1, count, d], device), context);
                context.EndStep(count);
                result.AddRange(y.ToArray());
                at += count;
            }

            return [.. result];
        }

        // A causal MultiHeadAttention with a one-way relative bias.
        var relative = new RelativePositionBias(heads, 4, bidirectional: false, device, new Random(2));
        using (var layer = new MultiHeadAttention(new Linear(d, 3 * d, device: device, random: random), new Linear(d, d, device: device, random: random), heads, causal: true) { PositionBias = relative })
        {
            float[] whole;
            using (new TensorScope())
            using (Autograd.NoGrad())
            {
                whole = layer.Forward(Tensor.From(x, [1, t, d], device)).ToArray();
            }

            AssertClose(whole, Cached(layer.ForwardCached, 1), 1e-4f, "MultiHeadAttention with a bias: cached steps");
            AssertClose(whole, Cached(layer.ForwardCached, 4), 1e-4f, "MultiHeadAttention with a bias: a prefill of 4, then steps");
        }

        // CausalSelfAttention in a DecoderBlock: RMS norms (the block's fused step reads the heads before the output projection)
        // and layer norms (the plain step).
        foreach (bool rms in new[] { true, false })
        {
            var bias = new RelativePositionBias(heads, 3, bidirectional: false, device, new Random(5));
            var attention = new CausalSelfAttention(d, heads, heads, d / heads, null, t, device: device, random: random) { PositionBias = bias };
            Module Norm() => rms ? new RMSNorm(d, device: device) : new LayerNorm(d, device: device);
            using var block = new Sequential(new DecoderBlock(Norm(), attention, Norm(), new FeedForward(d, 2 * d, false, FeedForwardActivation.Gelu, device: device, random: random)));
            float[] whole;
            using (new TensorScope())
            using (Autograd.NoGrad())
            {
                whole = block.Forward(Tensor.From(x, [1, t, d], device)).ToArray();
            }

            float[] Steps(int prefill)
            {
                using var context = new DecodingContext(device, 1, t);
                var result = new List<float>();
                for (int at = 0; at < t;)
                {
                    int count = at == 0 ? prefill : 1;
                    using var scope = new TensorScope();
                    using var noGrad = Autograd.NoGrad();
                    result.AddRange(block.ForwardCached(Tensor.From(x.AsSpan(at * d, count * d), [1, count, d], device), context).ToArray());
                    at += count;
                }

                return [.. result];
            }

            AssertClose(whole, Steps(1), 1e-4f, $"a DecoderBlock ({(rms ? "RMS" : "layer")} norms) with a biased CausalSelfAttention: cached steps");
            AssertClose(whole, Steps(5), 1e-4f, $"a DecoderBlock ({(rms ? "RMS" : "layer")} norms) with a biased CausalSelfAttention: a prefill of 5, then steps");
        }

        // A bias shared by two layers is computed once per shape.
        var counting = new CountingBias(heads);
        using var first = new MultiHeadAttention(new Linear(d, 3 * d, device: device, random: random), new Linear(d, d, device: device, random: random), heads) { PositionBias = counting };
        using var second = new MultiHeadAttention(new Linear(d, 3 * d, device: device, random: random), new Linear(d, d, device: device, random: random), heads)
        {
            PositionBias = counting,
            SharesPositionBias = true,
        };
        Check(first.Children().Contains(counting) && !second.Children().Contains(counting), "the first layer owns the shared bias, the second does not");
        for (int pass = 0; pass < 2; pass++)
        {
            using var scope = new TensorScope();
            using var noGrad = Autograd.NoGrad();
            second.Forward(first.Forward(Tensor.From(x, [1, t, d], device)));
        }

        Check(counting.Computed == 1, $"a shared bias computed once per shape (two layers, two passes): computed {counting.Computed} times");
    }

    // A bias that ignores the query's position (a step's rows are the first rows, wrong for cached decoding).
    internal sealed class StartlessBias(int heads) : AttentionPositionBias(heads)
    {
        protected override Tensor Compute(int queries, int keys, int queryStart, Device device)
        {
            var values = new float[Heads * queries * keys];
            for (int h = 0; h < Heads; h++)
            {
                for (int i = 0; i < queries; i++)
                {
                    for (int j = 0; j < keys; j++)
                    {
                        values[(h * queries + i) * keys + j] = -0.25f * Math.Abs(j - i);
                    }
                }
            }

            return Tensor.From(values, [Heads, queries, keys], device);
        }
    }

    internal static void AttentionOptionsRefused(Device device)
    {
        foreach (float bad in new[] { 0f, -1f, float.NaN, float.PositiveInfinity })
        {
            ThrowsNaming<ArgumentOutOfRangeException>(() => _ = new MultiHeadAttention(16, 4, device: device) { ScoreScale = bad }, $"MultiHeadAttention ScoreScale {bad}", "ScoreScale");
            ThrowsNaming<ArgumentOutOfRangeException>(() => _ = new CausalSelfAttention(16, 4, 4, 4, null, 8, device: device) { ScoreScale = bad }, $"CausalSelfAttention ScoreScale {bad}", "ScoreScale");
            ThrowsNaming<ArgumentOutOfRangeException>(() => _ = new CrossAttention(16, 4, device: device) { ScoreScale = bad }, $"CrossAttention ScoreScale {bad}", "ScoreScale");
        }

        var threeHeads = new RelativePositionBias(3, 4, true, device);
        ThrowsNaming<ArgumentException>(() => _ = new MultiHeadAttention(16, 4, device: device) { PositionBias = threeHeads }, "a bias of 3 heads on 4", "PositionBias");
        ThrowsNaming<ArgumentException>(() => _ = new CausalSelfAttention(16, 4, 4, 4, null, 8, device: device) { PositionBias = threeHeads }, "a bias of 3 heads on 4", "PositionBias");
        ThrowsNaming<ArgumentException>(() => { using var _ = new RelativePositionBias(4, 0, true, device); }, "max_distance 0", "max_distance");

        // "relative": its values are the table's rows by the clipped distance, and it passes the suite both ways.
        foreach (bool bidirectional in new[] { true, false })
        {
            var relative = new RelativePositionBias(3, 5, bidirectional, device, new Random(9));
            float[] table = relative.Table.ToArray(), values;
            using (Autograd.NoGrad())
            {
                values = relative.For(4, 12, 6, device).ToArray();
            }

            for (int h = 0; h < 3; h++)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 12; j++)
                    {
                        float expected = table[relative.Bucket(j - (6 + i)) * 3 + h];
                        Check(values[(h * 4 + i) * 12 + j] == expected, $"relative (bidirectional {bidirectional}): head {h}, query {6 + i}, key {j} is {values[(h * 4 + i) * 12 + j]}, the table's {expected}");
                    }
                }
            }

            Conformance.Check(relative, new PositionBiasSuite(device: device)).ThrowIfFailed();
        }

        var startless = Conformance.Check(new StartlessBias(2), new PositionBiasSuite(device: device));
        Check(!startless.Passed && startless.ToString().Contains("whole sequence's rows", StringComparison.Ordinal), $"a bias that ignores the query's position fails the suite: {startless}");
    }

    // ---------------------------------------------------------------- encoders

    // The library's poolers as plain loops on the host.
    internal sealed class HostPooler(string name) : ISequencePooler
    {
        public string Name => name;

        public Tensor Pool(Tensor hidden, IReadOnlyList<int>? lengths)
        {
            int n = hidden.Shape[0], t = hidden.Shape[1], w = hidden.Shape[2];
            float[] states = hidden.ToArray();
            var result = new float[n * w];
            for (int b = 0; b < n; b++)
            {
                int length = Math.Max(lengths?[b] ?? t, 1);
                for (int c = 0; c < w; c++)
                {
                    double value = name == SequencePoolers.Max ? double.NegativeInfinity : 0;
                    int positions = name == SequencePoolers.First ? 1 : length;
                    for (int p = 0; p < positions; p++)
                    {
                        float s = states[(b * t + p) * w + c];
                        value = name == SequencePoolers.Max ? Math.Max(value, s) : value + s;
                    }

                    result[b * w + c] = (float)(name == SequencePoolers.Mean ? value / length : value);
                }
            }

            return Tensor.From(result, [n, w], hidden.Device);
        }
    }

    // A mean over every position, padding included.
    internal sealed class PaddingPooler : ISequencePooler
    {
        public string Name => "padding-mean";

        public Tensor Pool(Tensor hidden, IReadOnlyList<int>? lengths) => hidden.Mean(1);
    }

    internal static void EncoderPoolers(Device device)
    {
        foreach (var pooler in new ISequencePooler[] { new FirstPooler(), new MeanPooler(), new MaxPooler() })
        {
            Conformance.Check(pooler, new SequencePoolerSuite(new HostPooler(pooler.Name), device)).ThrowIfFailed();
        }

        var report = Conformance.Check<ISequencePooler>(new PaddingPooler(), new SequencePoolerSuite(device: device));
        Check(!report.Passed && report.ToString().Contains("the padding does not matter", StringComparison.Ordinal), $"a pooler that reads the padding fails the suite: {report}");
    }

    // An encoder of layers: token types, learned positions, an embedding norm, two blocks sharing a relative bias.
    internal static SequenceEncoder LibraryEncoder(Device device, int vocabulary, int width, int maxPositions, int seed)
    {
        var random = new Random(seed);
        const int heads = 2;
        var bias = new RelativePositionBias(heads, 4, bidirectional: true, device, random);
        var blocks = Enumerable.Range(0, 2).Select(i => (Module)TransformerEncoderLayer.FromLayers(new LayerNorm(width, device: device),
            new MultiHeadAttention(new Linear(width, 3 * width, device: device, random: random), new Linear(width, width, device: device, random: random), heads)
            {
                PositionBias = bias,
                SharesPositionBias = i > 0,
            },
            new LayerNorm(width, device: device), new Linear(width, 2 * width, device: device, random: random), new Linear(2 * width, width, device: device, random: random),
            new ExactGELU())).ToList();
        return new SequenceEncoder(new Embedding(vocabulary, width, device, random), blocks,
            new PositionEmbedding(Tensor.Persistent(RandomArray(random, maxPositions * width, 0.3f), [maxPositions, width], device, requiresGrad: true)),
            new Embedding(2, width, device, random), new LayerNorm(width, device: device), new LayerNorm(width, device: device));
    }

    internal static void EncoderLibrary(Device device)
    {
        const string alphabet = "abcdefghijklmnopqrstuvwxyz .";
        using var encoder = LibraryEncoder(device, alphabet.Length + 1, 16, 24, seed: 3);
        Check(encoder.TypeVocabulary == 2 && encoder.MaxPositions == 24 && encoder.Width == 16, $"the encoder's sizes: {encoder}");
        Conformance.Check<ISequenceEncoder>(encoder, new SequenceEncoderSuite(device)).ThrowIfFailed();

        var tokenizer = new CharTokenizer(alphabet);
        string[] texts = ["a cat sat.", "dogs", "", "the quick brown fox jumps", "z"];
        foreach (string pooler in new[] { SequencePoolers.Mean, SequencePoolers.First, SequencePoolers.Max })
        {
            var textEncoder = new TextEncoder((ISequenceEncoder)encoder, tokenizer, maxLength: 20, padId: alphabet.Length) { Pooler = pooler };
            Check(textEncoder.Encoder == encoder && textEncoder.Pooler == pooler, "the text encoder over the sequence encoder");
            var batched = textEncoder.Encode(texts, batchSize: 5);
            for (int i = 0; i < texts.Length; i++)
            {
                AssertClose(textEncoder.Encode(texts[i]), batched[i], 1e-5f, $"pooler {pooler}: text {i} alone and in a batch");
                double norm = Math.Sqrt(batched[i].Sum(v => (double)v * v));
                Check(Math.Abs(norm - 1) < 1e-4, $"pooler {pooler}: text {i}'s vector has length {norm}");
            }
        }

        ThrowsNaming<ArgumentOutOfRangeException>(() => _ = new TextEncoder((ISequenceEncoder)encoder, tokenizer, 25, alphabet.Length), "a length past the encoder's", "maxLength");

        // A plain model (ids alone): the mean pooler gives the bits of the average the encoder always computed.
        var random = new Random(8);
        using var model = new Sequential(new Embedding(alphabet.Length + 1, 12, device, random), new Linear(12, 12, device: device, random: random));
        var plain = new TextEncoder(model, tokenizer, maxLength: 8, padId: alphabet.Length);
        float[][] vectors = plain.Encode(texts, batchSize: 5);
        model.Eval();
        using (new TensorScope())
        using (Autograd.NoGrad())
        {
            int n = texts.Length, t = 8;
            var ids = new float[n * t];
            var weights = new float[n * t];
            for (int i = 0; i < n; i++)
            {
                var tokens = tokenizer.Encode(texts[i]);
                int length = Math.Min(tokens.Count, t);
                for (int j = 0; j < t; j++)
                {
                    ids[i * t + j] = j < length ? tokens[j] : alphabet.Length;
                    weights[i * t + j] = j < length ? 1f / Math.Max(length, 1) : 0f;
                }

                if (length == 0)
                {
                    weights[i * t] = 1f;
                }
            }

            var input = Tensor.From(ids, [n, t], device);
            var mean = Tensor.From(weights, [n, 1, t], device);
            var hidden = model.Forward(input);
            int dim = hidden.Shape[^1];
            var pooled = mean.MatMul(hidden).Reshape(n, dim);
            var inverseNorm = ((pooled.Square().Sum(1) + 1e-12f).Log() * -0.5f).Exp();
            float[] expected = inverseNorm.Reshape(n, 1, 1).MatMul(pooled.Reshape(n, 1, dim)).Reshape(n, dim).ToArray();
            BitsEqual(expected, [.. vectors.SelectMany(v => v)], "TextEncoder over a plain model: the average as it always was");
        }
    }

    // ---------------------------------------------------------------- seq2seq
}
