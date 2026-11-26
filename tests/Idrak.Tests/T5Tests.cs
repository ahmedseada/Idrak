// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Generation;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Layers.Abstractions;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.PluginTests;
using Idrak.Text.Inference;
using Idrak.Text.Plugins.T5;

// Plan 26, phase P10: the generic pieces T5 needed (the "bucketed" position bias, RMS norms and a feed-forward block in
// TransformerEncoderLayer, the beam setting of encoder-decoder generation, run and the engine's "seq2seq" kind) and the T5
// plug-in (Idrak.Text.Plugins.T5) against transformers' outputs for two tiny random T5 models (tools/t5/make_tiny_t5.py,
// tests/Idrak.Tests/data/plugins/t5): v1.0 (ReLU, tied head, the output scaled) and v1.1 (gated GELU, its own head).
//
// The tolerance against transformers is 1e-4, absolute up to 1 and relative above (AssertClose): both sides compute in
// float32 and differ only by summation order (blocked products, the softmax, the RMS norms' mean of squares) carried
// through three encoder and two decoder layers. The mistakes the test is for move the outputs far more: a score scaled
// by 1/sqrt(d), a bias bucket off by one, an RMS norm that takes the mean away, an epsilon of 1e-5, the tanh GELU swapped
// for erf, a missing d_model^-0.5 or one applied to an untied head, a transposed projection each move them by 1e-3 or more.
// Generations are compared token for token, so a near tie between two tokens could flip; the fixtures' logits are far
// enough apart that none does.
internal abstract partial class DesktopTests : Tests
{

    // The owner's real models: six or eight layers of width 512 (t5-small) and larger activations, so more rounding.
    private const float T5RealTolerance = 2e-3f;

    private static readonly (string Name, Action<Device> Run)[] T5Group =
    [
        ("engine: bucketed position bias: BucketedPositionBias's buckets equal transformers' _relative_position_bucket over -700..700 for 32 buckets with max distances 20 and 128, both ways; its bias is the table's rows by bucket, passes the position bias suite both ways; num_buckets and max_distance that do not fit are refused naming the setting", T5BucketedBias),
        ("engine: rms-norm block: TransformerEncoderLayer from RMS norms and a feed-forward block (gated GELU and plain ReLU, pre- and post-norm) equals its sublayers composed, in inference and with the gradient recorded (every parameter gets a gradient); cached causal steps equal the whole sequence; Normalization = RmsNorm on a new block and the network builder carry it both ways; LayerNorm stays the default; mixed, mis-sized or foreign norms and an undefined Normalization are refused naming the setting", T5RmsNormBlock),
        ("engine: position bias steps: a decoding step's bias of the \"bucketed\" and \"relative\" kinds (one way and both ways) equals the row of the whole sequence's bias bit for bit, past the max distance and across the widening of the values by distance", PositionBiasSteps),
        ("t5 real: (the owner's: make_tiny_t5.py --real first) google-t5/t5-small and google/mt5-small from the cache give transformers' ids, encoder states and logits within 2e-3 and its greedy generation token for token", T5Real),
    ];

    private static string T5Data(string relative) => TestData(Path.Combine("plugins", "t5", relative));

    private static JsonObject T5Record() => JsonNode.Parse(File.ReadAllText(T5Data("reference.json")))!.AsObject();

    private static PretrainedSeq2SeqModel LoadT5(string folder, Device device, string? architecture = null) =>
        PretrainedSeq2SeqModel.Load(folder, new PretrainedEncoderOptions { Device = device, Architecture = architecture });

    // A case of reference.safetensors: its padded source ids, lengths, the target and the shapes.
    private sealed record T5Case(string Name, float[] Ids, int[] Lengths, int Time, float[] Target, int Steps);

    // The encoder's states and the decoder's logits for a case's padded sources and its target.
    private static (float[] States, float[] Logits) T5Outputs(ISeq2SeqModel model, T5Case c, Device device)
    {
        int n = c.Lengths.Length;
        var memory = model.Encoder.Encode(Tensor.From(c.Ids, [n, c.Time], device), c.Lengths);
        var logits = model.Decode(Tensor.From(c.Target, [n, c.Steps], device), memory, c.Lengths);
        return (memory.ToArray(), logits.ToArray());
    }

    private static GenerationOptions T5Greedy(int tokens) => new() { TopK = 1, RepeatPenalty = 1f, NumPredict = tokens };

    // ---------------------------------------------------------------- the bucketed position bias (library)

    private static void T5BucketedBias(Device device)
    {
        var buckets = T5Record()["buckets"]!.AsObject();
        int from = (int)buckets["from"]!;
        foreach (var (key, values) in buckets["tables"]!.AsObject())
        {
            string[] parts = key.Split('/');
            int count = int.Parse(parts[0], System.Globalization.CultureInfo.InvariantCulture), distance = int.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture);
            bool both = parts[2] == "bidirectional";
            using var bias = new BucketedPositionBias(2, count, distance, both, device, new Random(7));
            int[] expected = Ints(values);
            var wrong = Enumerable.Range(0, expected.Length).Where(i => bias.Bucket(from + i) != expected[i]).Select(i => from + i).ToList();
            Check(wrong.Count == 0, $"{key}: the buckets of {wrong.Count} distances differ from transformers' (first {string.Join(", ", wrong.Take(5))})");

            // The bias is the table's rows by bucket, [heads, queries, keys], for any first query position.
            float[] table = bias.Table.ToArray();
            foreach (var (q, k, start) in new[] { (5, 5, 0), (1, 40, 39), (3, 64, 20) })
            {
                using var scope = new TensorScope();
                float[] actual = bias.For(q, k, start, device).ToArray();
                var wanted = new float[2 * q * k];
                for (int h = 0; h < 2; h++)
                {
                    for (int i = 0; i < q; i++)
                    {
                        for (int j = 0; j < k; j++)
                        {
                            wanted[(h * q + i) * k + j] = table[bias.Bucket(j - (start + i)) * 2 + h];
                        }
                    }
                }

                BitsEqual(wanted, actual, $"{key}: the bias of {q} queries from {start} over {k} keys");
            }

            Conformance.Check<IAttentionPositionBias>(bias, new PositionBiasSuite(device: device)).ThrowIfFailed();
        }

        // Counts that do not fit are refused naming the setting.
        ThrowsNaming<ArgumentException>(() => { using var _ = new BucketedPositionBias(2, 3, 128, true, device); }, "an odd number of buckets both ways", "num_buckets");
        ThrowsNaming<ArgumentException>(() => { using var _ = new BucketedPositionBias(2, 0, 128, true, device); }, "no buckets", "num_buckets");
        ThrowsNaming<ArgumentException>(() => { using var _ = new BucketedPositionBias(2, 32, 8, true, device); }, "a max distance the exact buckets reach", "max_distance");
    }

    // ---------------------------------------------------------------- the RMS-norm block (library)

    private static void T5RmsNormBlock(Device device)
    {
        var random = new Random(2610);
        const int n = 2, t = 6, d = 16, heads = 4, ff = 24;
        float[] g1 = [.. RandomArray(random, d, 0.1f).Select(v => 1f + v)], g2 = [.. RandomArray(random, d, 0.1f).Select(v => 1f + v)],
            wqkv = RandomArray(random, d * 3 * d, 0.4f), bqkv = RandomArray(random, 3 * d, 0.1f), wo = RandomArray(random, d * d, 0.4f), bo = RandomArray(random, d, 0.1f),
            wg = RandomArray(random, d * ff, 0.4f), bg = RandomArray(random, ff, 0.1f), w1 = RandomArray(random, d * ff, 0.4f), c1 = RandomArray(random, ff, 0.1f),
            w2 = RandomArray(random, ff * d, 0.4f), c2 = RandomArray(random, d, 0.1f), x = RandomArray(random, n * t * d);
        RMSNorm Rms(float[] gain) => RMSNorm.FromWeights(Tensor.Persistent(gain, [d], device, requiresGrad: true), 1e-6f);
        MultiHeadAttention Attention(bool causal) => new(Projection(wqkv, bqkv, d, 3 * d, device), Projection(wo, bo, d, d, device), heads, causal);
        FeedForward Fed(bool gated) => new(gated ? Projection(wg, bg, d, ff, device) : null, Projection(w1, c1, d, ff, device), Projection(w2, c2, ff, d, device),
            gated ? FeedForwardActivation.Gelu : FeedForwardActivation.Relu);
        TransformerEncoderLayer Block(bool gated, NormPlacement placement, bool causal = false) =>
            TransformerEncoderLayer.FromLayers(Rms(g1), Attention(causal), Rms(g2), Fed(gated), placement);

        foreach (bool gated in new[] { true, false })
        {
            foreach (var placement in new[] { NormPlacement.Before, NormPlacement.After })
            {
                string name = $"{(gated ? "gated GELU" : "ReLU")}, {placement}";
                using var block = Block(gated, placement);
                Check(block.Normalization == BlockNormalization.RmsNorm && block.FeedForwardBlock is not null && block.ToString().Contains("rms norm", StringComparison.Ordinal),
                    $"{name}: the block says it has RMS norms and a feed-forward block: {block}");
                var parts = block.Children().ToList();
                Check(parts.Count == 4, $"{name}: norm, attention, norm, feed-forward: {string.Join(", ", parts)}");
                var (norm1, attention, norm2, fed) = ((RMSNorm)parts[0], (MultiHeadAttention)parts[1], (RMSNorm)parts[2], (FeedForward)parts[3]);
                float[] expected, inference, recorded;
                using (new TensorScope())
                using (Autograd.NoGrad())
                {
                    var input = Tensor.From(x, [n, t, d], device);
                    if (placement == NormPlacement.Before)
                    {
                        var sum = input + attention.Forward(norm1.Forward(input));
                        expected = (sum + fed.Forward(norm2.Forward(sum))).ToArray();
                    }
                    else
                    {
                        var y = norm1.Forward(input + attention.Forward(input));
                        expected = norm2.Forward(y + fed.Forward(y)).ToArray();
                    }

                    inference = block.Forward(input).ToArray();
                }

                AssertClose(expected, inference, 1e-5f, $"{name}: the block in inference against its sublayers composed");
                using (new TensorScope())
                {
                    var input = Tensor.From(x, [n, t, d], device);
                    var output = block.Forward(input);
                    recorded = output.ToArray();
                    output.Square().Sum().Backward();
                    Check(block.Parameters().All(p => p.Grad?.ToArray().Any(v => v != 0f) == true), $"{name}: every parameter gets a gradient");
                }

                AssertClose(expected, recorded, 1e-5f, $"{name}: the block with the gradient recorded");
            }
        }

        // Cached steps of a causal block equal the whole sequence.
        using (var causal = Block(gated: true, NormPlacement.Before, causal: true))
        {
            float[] whole;
            using (new TensorScope())
            using (Autograd.NoGrad())
            {
                whole = causal.Forward(Tensor.From(x.AsSpan(0, t * d), [1, t, d], device)).ToArray();
            }

            var steps = new List<float>();
            using var context = new DecodingContext(device, 1, t);
            for (int at = 0; at < t;)
            {
                int count = at == 0 ? 2 : 1;
                using var scope = new TensorScope();
                using var noGrad = Autograd.NoGrad();
                context.BeginStep(count);
                steps.AddRange(causal.ForwardCached(Tensor.From(x.AsSpan(at * d, count * d), [1, count, d], device), context).ToArray());
                context.EndStep(count);
                at += count;
            }

            AssertClose(whole, [.. steps], 1e-4f, "a causal RMS-norm block: cached steps after a prefill of 2 against the whole sequence");
        }

        // A new block takes the setting; the builder carries it; LayerNorm stays the default.
        using (var plain = new TransformerEncoderLayer(d, heads, ff, 0f, device: device))
        {
            Check(plain.Normalization == BlockNormalization.LayerNorm && plain.Children().First() is LayerNorm, "LayerNorm is the default");
        }

        using (var described = new TransformerEncoderLayer(d, heads, ff, 0f, device: device) { Normalization = BlockNormalization.RmsNorm, Placement = NormPlacement.After })
        {
            var children = described.Children().ToList();
            Check(children[0] is RMSNorm { Features: d } && children[2] is RMSNorm { Features: d } && described.Normalization == BlockNormalization.RmsNorm,
                $"Normalization = RmsNorm on a new block gives it RMS norms: {string.Join(", ", children)}");
        }

        var builder = Network.Tokens(5).Seed(45).Embedding(11, 8).TransformerEncoderLayer(2, ffDim: 12, dropout: 0f, normalization: BlockNormalization.RmsNorm).MeanOverTime();
        var json = builder.ToJson();
        Check(json.ToJsonString().Contains("\"norm\":\"rms\"", StringComparison.Ordinal), $"the builder writes the normalization: {json.ToJsonString()}");
        var replayed = Network.FromJson(json);
        Check(JsonNode.DeepEquals(replayed.ToJson(), json), "the builder's JSON replays");
        using (var built = builder.OnDevice(device).Build())
        using (var again = replayed.OnDevice(device).Build())
        {
            Check(built.Descendants().OfType<TransformerEncoderLayer>().Single().Normalization == BlockNormalization.RmsNorm
                && again.Descendants().OfType<TransformerEncoderLayer>().Single().Normalization == BlockNormalization.RmsNorm, "the builder's layer has RMS norms, replayed too");
            built.Eval();
            again.Eval();
            using var ids = Tensor.From([1f, 4f, 2f, 9f, 3f, 0f, 7f, 5f, 6f, 10f], [2, 5], device);
            BitsEqual(built.Predict(ids).ToArray(), again.Predict(ids).ToArray(), "the replayed RMS-norm network");
        }

        // Refused naming the setting.
        LayerNorm Layer() => new(d, device: device);
        ThrowsNaming<ArgumentException>(() => TransformerEncoderLayer.FromLayers(Layer(), Attention(false), Rms(g2), Fed(true)).Dispose(), "a LayerNorm and an RMS norm", "Normalization");
        ThrowsNaming<ArgumentException>(() => TransformerEncoderLayer.FromLayers(RMSNorm.FromWeights(Tensor.Persistent(new float[d + 1], [d + 1], device), 1e-6f), Attention(false),
            Rms(g2), Fed(true)).Dispose(), "an RMS norm of another width", "Normalization");
        ThrowsNaming<ArgumentException>(() => TransformerEncoderLayer.FromLayers(new Tanh(), Attention(false), new Tanh(), Fed(true)).Dispose(), "norms that are not norms", "Normalization");
        ThrowsNaming<ArgumentOutOfRangeException>(() => new TransformerEncoderLayer(d, heads, ff, 0f, device: device) { Normalization = (BlockNormalization)5 }.Dispose(),
            "an undefined normalization", "Normalization");
        ThrowsNaming<ArgumentOutOfRangeException>(() => TransformerEncoderLayer.FromLayers(Rms(g1), Attention(false), Rms(g2), Fed(true), (NormPlacement)9).Dispose(),
            "an undefined placement", "Placement");
    }

    // ---------------------------------------------------------------- the beam setting (library)

    // ---------------------------------------------------------------- T5 against transformers

    // ---------------------------------------------------------------- leftover round L1: steps, beams on the device, the tied head

    // A step's bias (one query, nothing training) is a slice of the values by distance on the device; it equals the row
    // of the whole sequence's bias (looked up through an index) bit for bit, for starts past the max distance and across
    // the widenings of the values (16, 32, 64 distances back).
    private static void PositionBiasSteps(Device device)
    {
        const int Length = 80, Heads = 4;
        foreach (bool bidirectional in new[] { false, true })
        {
            foreach (var bias in new DistancePositionBias[]
                     {
                         new BucketedPositionBias(Heads, 32, 20, bidirectional, device, new Random(1)), new RelativePositionBias(Heads, 6, bidirectional, device, new Random(2)),
                     })
            {
                using var owner = bias;
                using var noGrad = Autograd.NoGrad();
                float[] whole = bias.For(Length, Length, 0, device).ToArray();
                for (int start = 0; start < 70; start++)
                {
                    int keys = bidirectional ? Length : start + 1;
                    float[] step = bias.For(1, keys, start, device).ToArray();
                    var expected = new float[Heads * keys];
                    for (int h = 0; h < Heads; h++)
                    {
                        Array.Copy(whole, (h * Length + start) * Length, expected, h * keys, keys);
                    }

                    BitsEqual(expected, step, $"{bias} at {start} over {keys} keys");
                }
            }
        }
    }

    // ---------------------------------------------------------------- idrak run with an encoder-decoder

    // ---------------------------------------------------------------- the owner's real models

    // Each real/<name>.json written by make_tiny_t5.py --real: the model from the cache (IDRAK_CACHE, else ~/.cache/idrak)
    // against transformers' references. 2e-3 (absolute up to 1, relative above): six or eight layers of width 512 with
    // feed-forwards of 1,024 to 2,048 terms sum in float32 in other orders than PyTorch's, and T5's activations grow large;
    // a wrong layer still moves the states by far more.
    private static void T5Real(Device device)
    {
        string real = Path.Combine(Path.GetDirectoryName(T5Data("manifest.json"))!, "real");
        string[] records = Directory.Exists(real) ? Directory.GetFiles(real, "*.json") : [];
        if (records.Length == 0)
        {
            Skip($"no real references in {real}: run tools/t5/make_tiny_t5.py --real first (the owner's run; it downloads the models)");
        }

        string cache = Environment.GetEnvironmentVariable("IDRAK_CACHE") is { Length: > 0 } root ? root
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "idrak");
        foreach (string file in records.Order(StringComparer.Ordinal))
        {
            var record = JsonNode.Parse(File.ReadAllText(file))!.AsObject();
            string repo = (string)record["repo"]!, folder = Path.Combine(cache, (string)record["folder"]!);
            Check(Directory.Exists(folder), $"{repo}: no model in {folder}: run tools/t5/make_tiny_t5.py --real (with the same IDRAK_CACHE)");
            using var references = SafeTensorsReader.Open(Path.ChangeExtension(file, ".safetensors"));
            using var model = LoadT5(folder, device, (string)record["architecture"]!);
            var tokenizer = model.Tokenizer ?? throw new InvalidOperationException($"{repo}: the folder's tokenizer was not read");
            var generator = model.CreateGenerator();
            var cases = record["cases"]!.AsArray();
            int tokens = (int)record["max_new_tokens"]!;
            float worst = 0f;
            for (int i = 0; i < cases.Count; i++)
            {
                var c = cases[i]!.AsObject();
                int[] ids = Ints(c["ids"]), target = Ints(c["target"]), greedy = Ints(c["greedy"]);
                string text = (string)c["text"]!;
                var encoding = tokenizer.Encode(text, null, new TokenizerEncodeOptions());
                var source = encoding.Ids.Length > 0 && encoding.Ids[^1] == model.Model.EndToken ? encoding.Ids : [.. encoding.Ids, model.Model.EndToken];
                Check(source.SequenceEqual(ids), $"{repo}: text {i}: the source ids {string.Join(" ", source)}, transformers {string.Join(" ", ids)}");
                using (new TensorScope())
                using (Autograd.NoGrad())
                {
                    var memory = model.Model.Encoder.Encode(Tensor.From([.. ids.Select(v => (float)v)], [1, ids.Length], device));
                    worst = MathF.Max(worst, Compared(references.Read($"text{i}.encoder"), memory.ToArray(), T5RealTolerance, $"{repo}: text {i}: encoder states"));
                    Compared(references.Read($"text{i}.logits"), model.Model.Decode(Tensor.From([.. target.Select(v => (float)v)], [1, target.Length], device), memory).ToArray(),
                        T5RealTolerance, $"{repo}: text {i}: decoder logits");
                }

                var mine = generator.GenerateIds(ids, T5Greedy(tokens));
                int[] expected = [.. greedy.TakeWhile(t => t != model.Model.EndToken)];
                Check(mine.SequenceEqual(expected), $"{repo}: text {i}: greedy [{string.Join(", ", mine)}], transformers [{string.Join(", ", expected)}]");
                string spoken = generator.Generate(text, T5Greedy(tokens)).Text, theirs = (string)c["greedy_text"]!;
                Check(spoken == theirs, $"{repo}: text {i}: the greedy text '{spoken}', transformers '{theirs}'");
            }

            Console.WriteLine($"    {repo}: largest difference of the encoder states from transformers {worst:E1}");
        }
    }
}
