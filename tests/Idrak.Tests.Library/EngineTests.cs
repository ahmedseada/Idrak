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
// position bias), the encoder and seq2seq contracts with their suites, and toy families registered from
// tests/Idrak.PluginTests through the public API.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] EngineGroup =
    [
        ("engine: attention options: with nothing set, MultiHeadAttention and TransformerEncoderLayer give the bits of the same layers with the default score scale set explicitly and of a padding scope that pads nothing (inference, fused, and with the gradient); a score scale of 2/√d gives the bits of the default layer with its query projection doubled", AttentionOptionsOff),
        ("engine: attention options: MultiHeadAttention with a score scale, heads narrower than the model, a position bias (the library's \"relative\" and an outside kind), causal or not, with key padding or none, equals a host reference within 1e-4 at every real position, in inference and with the gradient recorded; each sequence of a padded batch equals itself alone; the bias table gets a gradient", AttentionOptionsReference),
        ("engine: attention options: cached decoding with a position bias (a causal MultiHeadAttention, and a CausalSelfAttention in a DecoderBlock with and without its fused step) equals the whole sequence at once, step by step and after a prefill; a bias shared by two layers is computed once per shape", AttentionOptionsCached),
        ("engine: attention options: invalid settings are refused naming the setting (ScoreScale 0, -1 and NaN on MultiHeadAttention, CausalSelfAttention and CrossAttention; a bias for other heads; max_distance and bidirectional of \"relative\"); an unknown bias kind is not registered; the position bias suite passes \"relative\" both ways (its values the table's buckets) and the outside kind, and fails a bias that ignores the query's position", AttentionOptionsRefused),
        ("engine: encoders: the poolers first, mean and max pass the pooler suite against plain loops; a pooler that reads the padding fails it; an unknown pooler is not registered", EncoderPoolers),
        ("engine: encoders: the library's SequenceEncoder built from layers (token types, learned positions, blocks sharing a position bias) passes the encoder suite; TextEncoder over it gives each text's unit vector alone and in a batch with the mean, first and max poolers; TextEncoder over a plain model gives the bits of its average as it always computed it", EncoderLibrary),
        ("engine: encoders: an encoder family registered from Idrak.PluginTests (public API only) loads through PretrainedEncoderModel.Load and passes the encoder suite with no bias, \"relative\" and an outside kind; its classifier and masked-language-model heads give a padded batch's logits one by one; unregistered, its checkpoint is refused as not registered naming PretrainedEncoders; head settings refused naming the setting; an unknown head kind is not registered", EncoderPlugin),
        ("engine: seq2seq: an encoder-decoder family registered from Idrak.PluginTests loads through PretrainedSeq2SeqModel.Load and passes the seq2seq suite (cached equals uncached, padded batch equals one by one, one memory for two rows, greedy) with no bias, \"relative\" and an outside kind; unregistered, it is refused as not registered naming PretrainedSeq2Seq", Seq2SeqPlugin),
        ("engine: seq2seq: Seq2SeqGenerator generates through the library's sampler: greedy ids equal greedy over the whole target recomputed each step and a one-beam search; Generate, Stream and StreamAsync give the same text; NumPredict ends it with \"length\" unless the end token came first; a seed repeats a sampled answer; a padded batch of memories decodes through the cache as each alone; memory lengths out of range refused", Seq2SeqGeneration),
    ];

    internal static string EngineFolder(string what)
    {
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-engine-{what}-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        return folder;
    }

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
        ToyEncoderPlugin.Register();
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
            ("outside bias, padded, wide heads", 2, 24, false, ToyEncoderPlugin.DistanceBias, null, true),
            ("causal, relative one-way bias", 4, 16, true, RelativePositionBias.KindName, null, false),
            ("causal, padded (the padding after the real positions changes nothing)", 4, 16, true, null, null, true),
        };
        foreach (var c in cases)
        {
            var bias = c.Bias is null ? null : AttentionPositionBiases.Create(c.Bias, new PositionBiasRequest(c.Heads,
                new JsonObject { ["max_distance"] = 3, ["bidirectional"] = !c.Causal }, device, Seed: 4));
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
        ToyEncoderPlugin.Register();
        foreach (float bad in new[] { 0f, -1f, float.NaN, float.PositiveInfinity })
        {
            ThrowsNaming<ArgumentOutOfRangeException>(() => _ = new MultiHeadAttention(16, 4, device: device) { ScoreScale = bad }, $"MultiHeadAttention ScoreScale {bad}", "ScoreScale");
            ThrowsNaming<ArgumentOutOfRangeException>(() => _ = new CausalSelfAttention(16, 4, 4, 4, null, 8, device: device) { ScoreScale = bad }, $"CausalSelfAttention ScoreScale {bad}", "ScoreScale");
            ThrowsNaming<ArgumentOutOfRangeException>(() => _ = new CrossAttention(16, 4, device: device) { ScoreScale = bad }, $"CrossAttention ScoreScale {bad}", "ScoreScale");
        }

        var threeHeads = new RelativePositionBias(3, 4, true, device);
        ThrowsNaming<ArgumentException>(() => _ = new MultiHeadAttention(16, 4, device: device) { PositionBias = threeHeads }, "a bias of 3 heads on 4", "PositionBias");
        ThrowsNaming<ArgumentException>(() => _ = new CausalSelfAttention(16, 4, 4, 4, null, 8, device: device) { PositionBias = threeHeads }, "a bias of 3 heads on 4", "PositionBias");
        ThrowsNaming<ArgumentException>(() => AttentionPositionBiases.Create(RelativePositionBias.KindName, new PositionBiasRequest(4, new JsonObject { ["max_distance"] = 0 }, device)),
            "max_distance 0", "max_distance");
        ThrowsNaming<ArgumentException>(() => AttentionPositionBiases.Create(RelativePositionBias.KindName, new PositionBiasRequest(4, new JsonObject { ["max_distance"] = "far" }, device)),
            "max_distance a word", "max_distance");
        ThrowsNaming<ArgumentException>(() => AttentionPositionBiases.Create(RelativePositionBias.KindName, new PositionBiasRequest(4, new JsonObject { ["bidirectional"] = 1 }, device)),
            "bidirectional a number", "bidirectional");
        ThrowsNaming<NotSupportedException>(() => AttentionPositionBiases.Create("no-such-bias", new PositionBiasRequest(4, [], device)), "an unknown bias kind",
            "not registered", "AttentionPositionBiases.Register");
        Check(AttentionPositionBiases.Origin(RelativePositionBias.KindName) == Overrides.Library, "\"relative\" is the library's");

        // "relative": its values are the table's rows by the clipped distance, and it passes the suite both ways.
        foreach (bool bidirectional in new[] { true, false })
        {
            var relative = (RelativePositionBias)AttentionPositionBiases.Create(RelativePositionBias.KindName,
                new PositionBiasRequest(3, new JsonObject { ["max_distance"] = 5, ["bidirectional"] = bidirectional }, device, Seed: 9));
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

        Conformance.Check(AttentionPositionBiases.Create(ToyEncoderPlugin.DistanceBias, new PositionBiasRequest(2, [], device)), new PositionBiasSuite(device: device)).ThrowIfFailed();
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
        foreach (string name in new[] { SequencePoolers.First, SequencePoolers.Mean, SequencePoolers.Max })
        {
            Check(SequencePoolers.Origin(name) == Overrides.Library, $"the pooler {name} is the library's");
            Conformance.Check(SequencePoolers.Get(name), new SequencePoolerSuite(new HostPooler(name), device)).ThrowIfFailed();
        }

        var report = Conformance.Check<ISequencePooler>(new PaddingPooler(), new SequencePoolerSuite(device: device));
        Check(!report.Passed && report.ToString().Contains("the padding does not matter", StringComparison.Ordinal), $"a pooler that reads the padding fails the suite: {report}");
        ThrowsNaming<NotSupportedException>(() => SequencePoolers.Get("no-such-pooler"), "an unknown pooler", "not registered", "SequencePoolers.Register");
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

        ThrowsNaming<NotSupportedException>(() => _ = new TextEncoder((ISequenceEncoder)encoder, tokenizer, 20, alphabet.Length) { Pooler = "nope" }, "an unknown pooler", "not registered");
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

    internal static void EncoderPlugin(Device device)
    {
        ToyEncoderPlugin.Register();
        string root = EngineFolder("encoder");
        try
        {
            foreach (string bias in new[] { "none", RelativePositionBias.KindName, ToyEncoderPlugin.DistanceBias })
            {
                var sizes = new ToySizes(PositionBias: bias, Seed: bias.Length);
                string folder = ToyEncoderPlugin.WriteEncoder(Path.Combine(root, bias), sizes);
                using var model = PretrainedEncoderModel.Load(folder, new PretrainedEncoderOptions { Device = device });
                Check(model.Architecture == ToyEncoderPlugin.Architecture && model.Pooler == SequencePoolers.First && model.Heads.Count == 2,
                    $"bias {bias}: the toy encoder with its two heads and the family's pooler ({model.Pooler}, {string.Join(", ", model.Heads.Keys)})");
                Check(PretrainedEncoders.Origin(ToyEncoderPlugin.Architecture) == typeof(ToyEncoderPlugin).Assembly.GetName().Name, "the family is the plug-in's");
                Conformance.Check(model.Encoder, new SequenceEncoderSuite(device)).ThrowIfFailed();

                // The heads over a padded batch, one by one.
                var random = new Random(4);
                int[] lengths = [7, 3, 5];
                int n = lengths.Length, t = 7;
                float[] ids = [.. Enumerable.Range(0, n * t).Select(_ => (float)random.Next(sizes.Vocabulary))];
                float[] types = [.. Enumerable.Range(0, n * t).Select(i => (float)(i % t >= 2 ? 1 : 0))];
                using var scope = new TensorScope();
                using var noGrad = Autograd.NoGrad();
                var classes = model.Logits(EncoderHeads.SequenceClassification, Tensor.From(ids, [n, t], device), lengths, Tensor.From(types, [n, t], device));
                var words = model.Logits(EncoderHeads.MaskedLanguageModel, Tensor.From(ids, [n, t], device), lengths, Tensor.From(types, [n, t], device));
                Check(classes.Shape.SequenceEqual([n, sizes.Labels]) && words.Shape.SequenceEqual([n, t, sizes.Vocabulary]),
                    $"bias {bias}: the heads' logits are [batch, labels] and [batch, time, vocabulary]: {Tensor.FormatShape(classes.Shape)}, {Tensor.FormatShape(words.Shape)}");
                float[] classValues = classes.ToArray(), wordValues = words.ToArray();
                for (int b = 0; b < n; b++)
                {
                    int length = lengths[b];
                    var ownIds = Tensor.From(ids.AsSpan(b * t, length), [1, length], device);
                    var ownTypes = Tensor.From(types.AsSpan(b * t, length), [1, length], device);
                    AssertClose(model.Logits(EncoderHeads.SequenceClassification, ownIds, null, ownTypes).ToArray(), classValues.AsSpan(b * sizes.Labels, sizes.Labels).ToArray(), 1e-4f,
                        $"bias {bias}: sequence {b}'s class logits alone and in the batch");
                    AssertClose(model.Logits(EncoderHeads.MaskedLanguageModel, ownIds, null, ownTypes).ToArray(),
                        wordValues.AsSpan(b * t * sizes.Vocabulary, length * sizes.Vocabulary).ToArray(), 1e-4f, $"bias {bias}: sequence {b}'s word logits alone and in the batch");
                }

                ThrowsNaming<ArgumentException>(() => model.Logits("token-classification", Tensor.From(ids, [n, t], device)), "a head the model does not hold",
                    "token-classification", EncoderHeads.MaskedLanguageModel);
            }

            ThrowsNaming<ArgumentException>(() => EncoderHeads.Create(EncoderHeads.MaskedLanguageModel, new EncoderHeadRequest(16, 0, [], device)), "outputs 0", "outputs");
            ThrowsNaming<ArgumentException>(() => EncoderHeads.Create(EncoderHeads.SequenceClassification, new EncoderHeadRequest(16, 3,
                new JsonObject { ["dense"] = true, ["activation"] = "swish" }, device)), "an unknown activation", "activation");
            ThrowsNaming<ArgumentException>(() => EncoderHeads.Create(EncoderHeads.MaskedLanguageModel, new EncoderHeadRequest(16, 3,
                new JsonObject { ["transform"] = "yes" }, device)), "transform a word", "transform");
            ThrowsNaming<NotSupportedException>(() => EncoderHeads.Get("no-such-head"), "an unknown head kind", "not registered", "EncoderHeads.Register");
            ThrowsNaming<NotSupportedException>(() => PretrainedEncoderModel.Load(Path.Combine(root, "none"), new PretrainedEncoderOptions { Device = device, Pooler = "nope" }),
                "an unknown pooler asked for", "not registered");

            ToyEncoderPlugin.Unregister();
            ThrowsNaming<NotSupportedException>(() => PretrainedEncoderModel.Load(Path.Combine(root, "none"), new PretrainedEncoderOptions { Device = device }),
                "an unregistered encoder family", "not registered", "PretrainedEncoders.Register", ToyEncoderPlugin.Architecture);
        }
        finally
        {
            ToyEncoderPlugin.Register();
            Directory.Delete(root, recursive: true);
        }
    }

    // ---------------------------------------------------------------- seq2seq

    internal static void Seq2SeqPlugin(Device device)
    {
        ToyEncoderPlugin.Register();
        string root = EngineFolder("seq2seq");
        try
        {
            foreach (string bias in new[] { "none", RelativePositionBias.KindName, ToyEncoderPlugin.DistanceBias })
            {
                string folder = ToyEncoderPlugin.WriteSeq2Seq(Path.Combine(root, bias), new ToySizes(PositionBias: bias, Seed: 7 + bias.Length));
                using var model = PretrainedSeq2SeqModel.Load(folder, new PretrainedEncoderOptions { Device = device });
                Check(model.Architecture == ToyEncoderPlugin.Seq2SeqArchitecture && model.Model.MaxTargetPositions == 16 && model.Model.Vocabulary == 24,
                    $"bias {bias}: the toy encoder-decoder ({model.Model})");
                Conformance.Check(model.Model, new Seq2SeqModelSuite(device)).ThrowIfFailed();
            }

            ToyEncoderPlugin.Unregister();
            ThrowsNaming<NotSupportedException>(() => PretrainedSeq2SeqModel.Load(Path.Combine(root, "none"), new PretrainedEncoderOptions { Device = device }),
                "an unregistered encoder-decoder family", "not registered", "PretrainedSeq2Seq.Register", ToyEncoderPlugin.Seq2SeqArchitecture);
        }
        finally
        {
            ToyEncoderPlugin.Register();
            Directory.Delete(root, recursive: true);
        }
    }

    internal static void Seq2SeqGeneration(Device device)
    {
        ToyEncoderPlugin.Register();
        string root = EngineFolder("generation");
        try
        {
            string folder = ToyEncoderPlugin.WriteSeq2Seq(Path.Combine(root, "model"), new ToySizes(PositionBias: RelativePositionBias.KindName, Seed: 12));
            using var loaded = PretrainedSeq2SeqModel.Load(folder, new PretrainedEncoderOptions { Device = device });
            var model = loaded.Model;
            const string alphabet = "abcdefghijklmnopqrstuvwx";                // 24 characters: the decoder's vocabulary
            var tokenizer = new CharTokenizer(alphabet);
            var generator = loaded.CreateGenerator(tokenizer);
            const string prompt = "hello world";
            var source = tokenizer.Encode(prompt.Replace(" ", "", StringComparison.Ordinal));
            var greedy = new GenerationOptions { TopK = 1, RepeatPenalty = 1f, NumPredict = 10, ChunkSize = 2 };

            // Greedy over the whole target, recomputed at each step (no cache).
            var expected = new List<int>();
            bool ended = false;
            using (new TensorScope())
            using (Autograd.NoGrad())
            {
                var memory = model.Encoder.Encode(Tensor.From([.. source.Select(i => (float)i)], [1, source.Count], device));
                var target = new List<float> { model.DecoderStartToken };
                for (int i = 0; i < 10 && !ended; i++)
                {
                    float[] logits = model.Decode(Tensor.From([.. target], [1, target.Count], device), memory).ToArray();
                    var row = logits.AsSpan((target.Count - 1) * model.Vocabulary, model.Vocabulary);
                    int best = 0;
                    for (int c = 1; c < row.Length; c++)
                    {
                        best = row[c] > row[best] ? c : best;
                    }

                    if (best == model.EndToken)
                    {
                        ended = true;
                    }
                    else
                    {
                        expected.Add(best);
                        target.Add(best);
                    }
                }
            }

            var source2 = source.Select(i => i).ToList();
            var ids = generator.GenerateIds(source2, greedy);
            Check(ids.SequenceEqual(expected), $"greedy through the cache [{string.Join(", ", ids)}] equals the whole target's [{string.Join(", ", expected)}]");

            using (var context = new DecodingContext(device, 1, model.MaxTargetPositions) { LastPositionOnly = true })
            {
                using (new TensorScope())
                using (Autograd.NoGrad())
                {
                    model.SetMemory(model.Encoder.Encode(Tensor.From([.. source.Select(i => (float)i)], [1, source.Count], device)), context);
                }

                var beam = BeamSearch.Decode((input, ctx) => model.DecodeCached(input, ctx), context, [model.DecoderStartToken],
                    new BeamOptions { Beams = 1, MaxTokens = 10, EndToken = model.EndToken })[0];
                Check(beam.Tokens.SequenceEqual(expected), $"a one-beam search [{string.Join(", ", beam.Tokens)}] equals greedy [{string.Join(", ", expected)}]");
            }

            // The text through every entry point.
            string sourceText = tokenizer.Decode(source);
            var (text, reason, stats) = generator.Generate(sourceText, greedy);
            Check(text == tokenizer.Decode(expected), $"Generate's text '{text}' is the greedy tokens' '{tokenizer.Decode(expected)}'");
            Check(reason == (ended ? "stop" : "length") && stats.GeneratedTokens == expected.Count && stats.PromptTokens == source.Count,
                $"Generate ends with {reason} after {stats.GeneratedTokens} tokens of a {stats.PromptTokens}-token prompt (ended {ended}, {expected.Count} tokens)");
            var chunks = generator.Stream(sourceText, greedy).ToList();
            Check(string.Concat(chunks.Select(c => c.Text)) == text && chunks[^1].Done && chunks.Take(chunks.Count - 1).All(c => !c.Done), "Stream gives Generate's text, the last chunk done");
            var streamed = new System.Text.StringBuilder();
            var enumerator = generator.StreamAsync(sourceText, greedy).GetAsyncEnumerator();
            while (enumerator.MoveNextAsync().AsTask().GetAwaiter().GetResult())
            {
                streamed.Append(enumerator.Current.Text);
            }

            Check(streamed.ToString() == text, $"StreamAsync (the ITextModel contract) gives Generate's text: '{streamed}'");
            var (shortText, shortReason, _) = generator.Generate(sourceText, greedy with { NumPredict = 2 });
            bool endedEarly = ended && expected.Count < 2;
            Check(shortText == tokenizer.Decode(expected.Take(2).ToList()) && shortReason == (endedEarly ? "stop" : "length"),
                $"NumPredict 2: '{shortText}' ({shortReason})");

            var sampled = new GenerationOptions { Temperature = 1.2f, TopK = 0, TopP = 1f, RepeatPenalty = 1f, Seed = 5, NumPredict = 8 };
            Check(generator.Generate(sourceText, sampled).Text == generator.Generate(sourceText, sampled).Text, "a seed repeats a sampled answer");

            // A padded batch of memories through the cache: each row as its source alone.
            int[] lengths = [6, 3];
            var random = new Random(3);
            float[] sources = [.. Enumerable.Range(0, 12).Select(_ => (float)random.Next(24))];
            float[] steps = [0f, 5f, 9f, 2f];
            using (new TensorScope())
            using (Autograd.NoGrad())
            {
                var memory = model.Encoder.Encode(Tensor.From(sources, [2, 6], device), lengths);
                using var batch = new DecodingContext(device, 2, 4);
                model.SetMemory(memory, batch, lengths);
                var rows = new List<float[]>();
                foreach (float token in steps)
                {
                    rows.Add(model.DecodeCached(Tensor.From([token, token], [2, 1], device), batch).ToArray());
                }

                for (int b = 0; b < 2; b++)
                {
                    var own = model.Encoder.Encode(Tensor.From(sources.AsSpan(b * 6, lengths[b]), [1, lengths[b]], device));
                    using var alone = new DecodingContext(device, 1, 4);
                    model.SetMemory(own, alone);
                    for (int s = 0; s < steps.Length; s++)
                    {
                        float[] single = model.DecodeCached(Tensor.From([steps[s]], [1, 1], device), alone).ToArray();
                        AssertClose(single, rows[s].AsSpan(b * model.Vocabulary, model.Vocabulary).ToArray(), 1e-4f, $"row {b}, step {s}: a padded batch of memories and the row alone");
                    }
                }

                using var refused = new DecodingContext(device, 2, 4);
                ThrowsNaming<ArgumentException>(() => model.SetMemory(memory, refused, [0, 3]), "a memory length of 0", "outside 1");
                ThrowsNaming<ArgumentException>(() => model.SetMemory(memory, refused, [7, 3]), "a memory length past the memory", "outside 1");
                ThrowsNaming<ArgumentException>(() => model.SetMemory(memory, refused, [3]), "one length for two sequences", "lengths");
            }
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }
}
