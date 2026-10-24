// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Plugins;
using Idrak.Abstraction.Testing;
using Idrak.Generation;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Layers.Abstractions;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.PluginTests;
using Idrak.Text.Inference;
using Idrak.Text.Plugins;
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
internal static partial class Tests
{
    private const float T5Tolerance = 1e-4f;

    // The owner's real models: six or eight layers of width 512 (t5-small) and larger activations, so more rounding.
    private const float T5RealTolerance = 2e-3f;

    private static readonly (string Name, Action<Device> Run)[] T5Group =
    [
        ("engine: bucketed position bias: BucketedPositionBias's buckets equal transformers' _relative_position_bucket over -700..700 for 32 buckets with max distances 20 and 128, both ways; its bias is the table's rows by bucket, passes the position bias suite both ways, is the library's \"bucketed\" kind made from settings and a table; num_buckets, max_distance and bidirectional that do not fit are refused naming the setting", T5BucketedBias),
        ("engine: rms-norm block: TransformerEncoderLayer from RMS norms and a feed-forward block (gated GELU and plain ReLU, pre- and post-norm) equals its sublayers composed, in inference and with the gradient recorded (every parameter gets a gradient); cached causal steps equal the whole sequence; Normalization = RmsNorm on a new block, the layer description (\"norm\": \"rms\") and the network builder carry it both ways; LayerNorm stays the default; mixed, mis-sized or foreign norms and an undefined Normalization are refused naming the setting", T5RmsNormBlock),
        ("engine: beam setting: GenerationOptions.NumBeams (1 by default), LengthPenalty and EarlyStopping; NumBeams below 1 and a non-finite LengthPenalty are refused naming the setting; Seq2SeqGenerator with NumBeams 3 answers with BeamSearch.Decode's best hypothesis (GenerateIds, GenerateBeams, Generate's text, \"stop\" or \"length\"), NumBeams 1 is the sampler's greedy answer unchanged", T5BeamSetting),
        ("plugins: t5: the tiny T5 v1.0 (ReLU, tied head scaled by d_model^-0.5) and v1.1 (gated GELU, its own head) read as T5ForConditionalGeneration give transformers' encoder states and decoder logits within 1e-4 for three texts and a padded batch, in inference and with the gradient recorded; the layers are the library's (RMS-norm blocks, one bucketed bias per stack shared, the shared embedding read in place); the folder's Unigram tokenizer gives transformers' ids", T5MatchesTransformers),
        ("plugins: t5: greedy generations (some ending at </s>) and beam searches of 3 (early_stopping true and false, ended and running hypotheses) equal transformers' token for token, in its order, with its scores within 1e-4, and their texts; KV-cached decoding equals the whole target; each sequence of a padded batch equals itself alone (states and logits); the seq2seq and encoder suites pass", T5Generation),
        ("plugins: t5: T5EncoderModel (and the encoder of a T5ForConditionalGeneration read as one) through PretrainedEncoderModel gives the encoder-decoder's states bit for bit with the mean pooler; MT5ForConditionalGeneration reads the same checkpoint; a folder with spiece.model alone (no post-processor) generates as tokenizer.json does (</s> appended); the inference engine's \"seq2seq\" kind answers through ITextModel as the generator does", T5EncoderAndEngine),
        ("plugins: t5: T5Plugin is the plug-in \"t5\" of TextPlugins.All; Register twice registers its two encoder-decoder and two encoder architectures once, from Idrak.Text.Plugins; Unregister leaves PretrainedSeq2Seq and PretrainedEncoders as found; without it a checkpoint is refused as not registered, naming the architecture and Idrak.Text.Plugins", T5Registration),
        ("plugins: t5: refused naming the cause: feed_forward_proj other than relu, gated-gelu and gated-silu, relative_attention_num_buckets odd or below 4, relative_attention_max_distance not above the exact buckets, an untied checkpoint without lm_head.weight, a tensor of another shape than the configuration's, a missing decoder tensor", T5Refused),
        ("engine: position bias steps: a decoding step's bias of the \"bucketed\" and \"relative\" kinds (one way and both ways) equals the row of the whole sequence's bias bit for bit, past the max distance and across the widening of the values by distance", PositionBiasSteps),
        ("engine: beam search on the device: BeamSearch.Decode choosing candidates where the logits are (SelectOnDevice) gives the host's hypotheses bit for bit on the CPU (tokens, scores, log-probabilities), with suppressed tokens, begin suppressions, a least length, the n-gram rule, a length penalty and both early-stopping rules; Seq2SeqGenerator.SelectBeamsOnDevice false gives the same", BeamsOnDevice),
        ("pretrained: SafeTensorsReader.SameValues compares two tensors chunk by chunk (float32 and bfloat16 files of a sharded checkpoint, a difference in the last chunk, another shape, 0 and -0, NaN) as the whole arrays compare, without holding either, also through a RenamedTensorStore", SafeTensorsSameValues),
        ("plugins: t5: a tied checkpoint that also holds lm_head.weight: equal to the shared embedding, the head reads the one table (the parameters and logits of the checkpoint without it); one value apart, the head is its own", T5TiedHead),
        ("cli run: encoder-decoder: idrak run on the tiny T5 folder answers with transformers' greedy text (--temperature 0, \"stop\" at </s>, --json with the architecture, the tokens and the beams) and with the best of --beams 3; --schema, --system and --beams 0 refused naming the option; --beams on a decoder refused", CliRunSeq2Seq),
        ("t5 real: (the owner's: make_tiny_t5.py --real first) google-t5/t5-small and google/mt5-small from the cache give transformers' ids, encoder states and logits within 2e-3 and its greedy generation token for token", T5Real),
    ];

    private static readonly string[] T5Models = ["tiny-t5", "tiny-t5-gated"];

    private static readonly string[] T5Cases = ["single0", "single1", "single2", "batch"];

    private static string T5Data(string relative) => TestData(Path.Combine("plugins", "t5", relative));

    private static JsonObject T5Record() => JsonNode.Parse(File.ReadAllText(T5Data("reference.json")))!.AsObject();

    // A tiny model in a folder of its own, with the committed Unigram tokenizer beside it (tokenizer.json, or the
    // SentencePiece model as spiece.model alone); `edit` changes its config.json.
    private static string T5Folder(string model, Action<JsonObject>? edit = null, bool sentencePiece = false)
    {
        string folder = EngineFolder("t5");
        File.Copy(T5Data($"{model}/model.safetensors"), Path.Combine(folder, "model.safetensors"));
        var config = JsonNode.Parse(File.ReadAllText(T5Data($"{model}/config.json")))!.AsObject();
        edit?.Invoke(config);
        File.WriteAllText(Path.Combine(folder, "config.json"), config.ToJsonString());
        File.Copy(TestData("tokenizers/unigram/tokenizer_config.json"), Path.Combine(folder, "tokenizer_config.json"));
        if (sentencePiece)
        {
            File.Copy(TestData("tokenizers/unigram/tokenizer.model"), Path.Combine(folder, "spiece.model"));
        }
        else
        {
            File.Copy(TestData("tokenizers/unigram/tokenizer.json"), Path.Combine(folder, "tokenizer.json"));
        }

        return folder;
    }

    private static PretrainedSeq2SeqModel LoadT5(string folder, Device device, string? architecture = null) =>
        PretrainedSeq2SeqModel.Load(folder, new PretrainedEncoderOptions { Device = device, Architecture = architecture });

    // A case of reference.safetensors: its padded source ids, lengths, the target and the shapes.
    private sealed record T5Case(string Name, float[] Ids, int[] Lengths, int Time, float[] Target, int Steps);

    private static T5Case ReadT5Case(SafeTensorsReader references, string model, string name)
    {
        int[] source = references.Tensors[$"{model}.{name}.ids"].Shape, target = references.Tensors[$"{model}.{name}.target"].Shape;
        return new T5Case(name, references.Read($"{model}.{name}.ids"), [.. references.Read($"{model}.{name}.lengths").Select(l => (int)l)], source[1],
            references.Read($"{model}.{name}.target"), target[1]);
    }

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

        // The library's kind, made from settings and a table.
        Check(AttentionPositionBiases.Names.Contains(BucketedPositionBias.KindName) && AttentionPositionBiases.Origin(BucketedPositionBias.KindName) == Overrides.Library,
            "\"bucketed\" is the library's");
        float[] values2 = [.. Enumerable.Range(0, 8 * 3).Select(i => i * 0.25f - 2f)];
        var made = AttentionPositionBiases.Create(BucketedPositionBias.KindName, new PositionBiasRequest(3,
            new JsonObject { ["num_buckets"] = 8, ["max_distance"] = 9, ["bidirectional"] = false }, device, 0, (name, _) => name == "table" ? values2 : null));
        Check(made is BucketedPositionBias { Buckets: 8, MaxDistance: 9, Bidirectional: false } kind && kind.Table.ToArray().SequenceEqual(values2),
            $"the kind made from settings and a table: {made}");
        ((IDisposable)made).Dispose();
        using (var defaults = (BucketedPositionBias)AttentionPositionBiases.Create(BucketedPositionBias.KindName, new PositionBiasRequest(4, [], device)))
        {
            Check(defaults is { Buckets: 32, MaxDistance: 128, Bidirectional: true }, $"the settings' defaults: {defaults}");
        }

        PositionBiasRequest Request(JsonObject settings) => new(2, settings, device);
        ThrowsNaming<ArgumentException>(() => AttentionPositionBiases.Create(BucketedPositionBias.KindName, Request(new JsonObject { ["num_buckets"] = 3 })),
            "an odd number of buckets both ways", "num_buckets");
        ThrowsNaming<ArgumentException>(() => AttentionPositionBiases.Create(BucketedPositionBias.KindName, Request(new JsonObject { ["num_buckets"] = 0 })),
            "no buckets", "num_buckets");
        ThrowsNaming<ArgumentException>(() => AttentionPositionBiases.Create(BucketedPositionBias.KindName, Request(new JsonObject { ["num_buckets"] = 32, ["max_distance"] = 8 })),
            "a max distance the exact buckets reach", "max_distance");
        ThrowsNaming<ArgumentException>(() => AttentionPositionBiases.Create(BucketedPositionBias.KindName, Request(new JsonObject { ["bidirectional"] = "yes" })),
            "bidirectional not a truth value", "bidirectional");
        ThrowsNaming<ArgumentException>(() => AttentionPositionBiases.Create(BucketedPositionBias.KindName, Request(new JsonObject { ["max_distance"] = 2.5 })),
            "a max distance that is not whole", "max_distance");
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

        // A new block takes the setting; the description and the builder carry it; LayerNorm stays the default.
        using (var plain = new TransformerEncoderLayer(d, heads, ff, 0f, device: device))
        {
            Check(plain.Normalization == BlockNormalization.LayerNorm && plain.Children().First() is LayerNorm && LayerTypes.Describe(plain)["norm"] is null,
                "LayerNorm is the default, described as before");
        }

        using (var described = new TransformerEncoderLayer(d, heads, ff, 0f, device: device) { Normalization = BlockNormalization.RmsNorm, Placement = NormPlacement.After })
        {
            var children = described.Children().ToList();
            Check(children[0] is RMSNorm { Features: d } && children[2] is RMSNorm { Features: d } && described.Normalization == BlockNormalization.RmsNorm,
                $"Normalization = RmsNorm on a new block gives it RMS norms: {string.Join(", ", children)}");
            var description = LayerTypes.Describe(described);
            Check((string?)description["norm"] == "rms" && (string?)description["norms"] == "after", $"the description says so: {description.ToJsonString()}");
            using var rebuilt = (TransformerEncoderLayer)LayerTypes.Create(description, device);
            Check(rebuilt.Normalization == BlockNormalization.RmsNorm && rebuilt.Placement == NormPlacement.After, "the description rebuilds an RMS-norm, post-norm layer");
            description["norm"] = "batch";
            ThrowsNaming<InvalidDataException>(() => LayerTypes.Create(description, device), "an unknown norm in a description", "norm", "batch");
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

    private static void T5BeamSetting(Device device)
    {
        var defaults = new GenerationOptions();
        Check(defaults is { NumBeams: 1, LengthPenalty: 1f, EarlyStopping: true }, "NumBeams 1, LengthPenalty 1 and EarlyStopping true by default");
        ThrowsNaming<ArgumentOutOfRangeException>(() => _ = new GenerationOptions { NumBeams = 0 }, "no beams", "NumBeams");
        ThrowsNaming<ArgumentOutOfRangeException>(() => _ = defaults with { NumBeams = -2 }, "negative beams", "NumBeams");
        ThrowsNaming<ArgumentOutOfRangeException>(() => _ = new GenerationOptions { LengthPenalty = float.NaN }, "a NaN length penalty", "LengthPenalty");
        ThrowsNaming<ArgumentOutOfRangeException>(() => _ = new GenerationOptions { LengthPenalty = float.PositiveInfinity }, "an infinite length penalty", "LengthPenalty");

        ToyEncoderPlugin.Register();
        string root = EngineFolder("beams");
        try
        {
            string folder = ToyEncoderPlugin.WriteSeq2Seq(Path.Combine(root, "model"), new ToySizes(PositionBias: RelativePositionBias.KindName, Seed: 31));
            using var loaded = PretrainedSeq2SeqModel.Load(folder, new PretrainedEncoderOptions { Device = device });
            var model = loaded.Model;
            const string alphabet = "abcdefghijklmnopqrstuvwx";
            var tokenizer = new CharTokenizer(alphabet);
            var generator = loaded.CreateGenerator(tokenizer);
            var source = tokenizer.Encode("beamsearch");
            foreach (var (beams, penalty, early) in new[] { (3, 1f, true), (3, 0.5f, false), (2, 1.5f, true) })
            {
                IReadOnlyList<BeamHypothesis> direct;
                using (var context = new DecodingContext(device, beams, 9) { LastPositionOnly = true })
                {
                    using (new TensorScope())
                    using (Autograd.NoGrad())
                    {
                        model.SetMemory(model.Encoder.Encode(Tensor.From([.. source.Select(i => (float)i)], [1, source.Count], device)), context);
                    }

                    direct = BeamSearch.Decode((ids, ctx) => model.DecodeCached(ids, ctx), context, [model.DecoderStartToken],
                        new BeamOptions { Beams = beams, MaxTokens = 9, EndToken = model.EndToken, LengthPenalty = penalty, EarlyStopping = early });
                }

                var options = T5Greedy(9) with { NumBeams = beams, LengthPenalty = penalty, EarlyStopping = early };
                var found = generator.GenerateBeams(source, options);
                Check(found.Count == direct.Count && found.Zip(direct).All(p => p.First.Tokens.SequenceEqual(p.Second.Tokens) && p.First.Score == p.Second.Score),
                    $"{beams} beams, penalty {penalty}, early {early}: GenerateBeams is BeamSearch.Decode's list");
                var best = direct[0].Tokens;
                var ids = generator.GenerateIds(source, options);
                Check(ids.SequenceEqual(best), $"{beams} beams: GenerateIds [{string.Join(", ", ids)}] is the best hypothesis [{string.Join(", ", best)}]");
                var (text, reason, stats) = generator.Generate(tokenizer.Decode(source), options);
                Check(text == tokenizer.Decode(best) && reason == (best.Count < 9 ? "stop" : "length") && stats.GeneratedTokens == best.Count,
                    $"{beams} beams: Generate's '{text}' ({reason}, {stats.GeneratedTokens} tokens) is the best hypothesis's");
            }

            // One beam is the sampler's greedy answer, as before the setting.
            var greedy = generator.GenerateIds(source, T5Greedy(9));
            Check(generator.GenerateIds(source, T5Greedy(9) with { NumBeams = 1 }).SequenceEqual(greedy), "NumBeams 1 is the greedy answer");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    // ---------------------------------------------------------------- T5 against transformers

    private static void T5MatchesTransformers(Device device)
    {
        T5Plugin.Register();
        using var references = SafeTensorsReader.Open(T5Data("reference.safetensors"));
        var record = T5Record();
        foreach (string name in T5Models)
        {
            string folder = T5Folder(name);
            try
            {
                using var loaded = LoadT5(folder, device);
                var model = loaded.Model;
                bool tied = name == "tiny-t5";
                Check(loaded.Architecture == T5Plugin.T5ForConditionalGeneration && loaded.EndsSource, $"{name}: read as T5ForConditionalGeneration, its sources ending with </s>");
                var encoder = model.Encoder as SequenceEncoder ?? throw new InvalidOperationException($"{name}: not the library's encoder: {model.Encoder}");
                var network = model as Seq2SeqNetwork ?? throw new InvalidOperationException($"{name}: not the library's encoder-decoder: {model}");
                Check(encoder.Blocks.Count == 3 && encoder.Blocks.All(b => b is TransformerEncoderLayer { Normalization: BlockNormalization.RmsNorm, Placement: NormPlacement.Before })
                    && encoder.FinalNorm is RMSNorm && encoder.Positions is null && encoder.TypeVocabulary == 0, $"{name}: three pre-norm RMS blocks and a final RMS norm: {encoder}");
                var encoderAttention = encoder.Descendants().OfType<MultiHeadAttention>().ToList();
                Check(encoderAttention.All(a => a.ScoreScale == 1f && ReferenceEquals(a.PositionBias, encoderAttention[0].PositionBias))
                    && encoderAttention[0].PositionBias is BucketedPositionBias { Bidirectional: true, Buckets: 32, MaxDistance: 20 }
                    && encoderAttention.Skip(1).All(a => a.SharesPositionBias) && encoderAttention[0].Inner == 24,
                    $"{name}: unscaled encoder attention of inner width 24, the first block's bidirectional bucketed bias shared by the others");
                var selfAttention = network.Decoder.Descendants().OfType<CausalSelfAttention>().ToList();
                Check(selfAttention.Count == 2 && selfAttention.All(a => a.ScoreScale == 1f && ReferenceEquals(a.PositionBias, selfAttention[0].PositionBias))
                    && selfAttention[0].PositionBias is BucketedPositionBias { Bidirectional: false } && network.Decoder.Descendants().OfType<CrossAttention>().All(c => c.ScoreScale == 1f),
                    $"{name}: unscaled decoder attention, the first block's one-way bias shared");
                var head = (Linear)network.Decoder.Last();
                Check(tied ? head.TiedTo == encoder.Words && network.Decoder.OfType<Scale>().Single().Factor == 1f / MathF.Sqrt(32)
                        : head.TiedTo is null && !network.Decoder.OfType<Scale>().Any(),
                    $"{name}: the head {(tied ? "reads the shared embedding, the output scaled by 32^-0.5" : "has its own weights, nothing scaled")}");
                Check(model.Network.Parameters().Count(p => ReferenceEquals(p, encoder.Words.Weight)) == 1, $"{name}: the shared embedding is one parameter");
                Check(model.DecoderStartToken == 0 && model.EndToken == 1 && model.Vocabulary == 1002 && model.MaxTargetPositions == 512, $"{name}: tokens and limits from config.json");

                var tokenizer = loaded.Tokenizer ?? throw new InvalidOperationException($"{name}: the folder's tokenizer was not read");
                var cases = record["models"]![name]!["cases"]!.AsObject();
                float worstStates = 0f, worstLogits = 0f;
                foreach (string caseName in T5Cases)
                {
                    var c = ReadT5Case(references, name, caseName);
                    string[] texts = cases[caseName]!["texts"] is JsonArray batch ? [.. batch.Select(v => (string)v!)] : [(string)cases[caseName]!["text"]!];
                    for (int b = 0; b < texts.Length; b++)
                    {
                        var encoding = tokenizer.Encode(texts[b], null, new TokenizerEncodeOptions());
                        Check(encoding.Ids.SequenceEqual(c.Ids.AsSpan(b * c.Time, c.Lengths[b]).ToArray().Select(v => (int)v)),
                            $"{name} {caseName}[{b}]: the tokenizer's ids {string.Join(" ", encoding.Ids)}");
                    }

                    float[] states = RealPositions(references.Read($"{name}.{caseName}.encoder"), c.Lengths, c.Time, 32);
                    float[] logits = references.Read($"{name}.{caseName}.logits");
                    using (new TensorScope())
                    using (Autograd.NoGrad())
                    {
                        model.Network.Eval();
                        var (mine, myLogits) = T5Outputs(model, c, device);
                        worstStates = MathF.Max(worstStates, Compared(states, RealPositions(mine, c.Lengths, c.Time, 32), T5Tolerance, $"{name} {caseName}: encoder states"));
                        worstLogits = MathF.Max(worstLogits, Compared(logits, myLogits, T5Tolerance, $"{name} {caseName}: decoder logits"));
                    }

                    using (new TensorScope())
                    {
                        var (mine, myLogits) = T5Outputs(model, c, device);
                        Compared(states, RealPositions(mine, c.Lengths, c.Time, 32), T5Tolerance, $"{name} {caseName}: encoder states, gradient recorded");
                        Compared(logits, myLogits, T5Tolerance, $"{name} {caseName}: decoder logits, gradient recorded");
                    }
                }

                Console.WriteLine($"    {name}: largest difference from transformers: encoder states {worstStates:E1}, logits {worstLogits:E1}");
            }
            finally
            {
                Directory.Delete(folder, recursive: true);
            }
        }
    }

    private static void T5Generation(Device device)
    {
        T5Plugin.Register();
        using var references = SafeTensorsReader.Open(T5Data("reference.safetensors"));
        var record = T5Record();
        foreach (string name in T5Models)
        {
            string folder = T5Folder(name);
            try
            {
                using var loaded = LoadT5(folder, device);
                var model = loaded.Model;
                var tokenizer = loaded.Tokenizer!;
                var generator = loaded.CreateGenerator();
                var cases = record["models"]![name]!["cases"]!.AsObject();
                bool anyEnded = false;
                foreach (string caseName in T5Cases.Take(3))
                {
                    var c = cases[caseName]!.AsObject();
                    int[] source = Ints(c["ids"]![0]);
                    string text = (string)c["text"]!;

                    // Greedy, by ids and by text (the source encoded with its </s>, the answer decoded without special tokens).
                    int[] greedy = Ints(c["greedy"]);
                    bool ended = (bool)c["greedy_ended"]!;
                    anyEnded |= ended;
                    var ids = generator.GenerateIds(source, T5Greedy(16));
                    Check(ids.SequenceEqual(greedy), $"{name} {caseName}: greedy [{string.Join(", ", ids)}], transformers [{string.Join(", ", greedy)}]");
                    var (answer, reason, stats) = generator.Generate(text, T5Greedy(16));
                    Check(answer == (string)c["greedy_text"]! && reason == (ended ? "stop" : "length") && stats.PromptTokens == source.Length,
                        $"{name} {caseName}: greedy text '{answer}' ({reason}, {stats.PromptTokens} source tokens), transformers '{(string)c["greedy_text"]!}'");

                    // Beam search, both stopping rules: transformers' hypotheses in its order, with its scores.
                    foreach (bool early in new[] { true, false })
                    {
                        var expected = c[early ? "beam_early" : "beam_late"]!.AsArray();
                        var found = generator.GenerateBeams(source, T5Greedy(10) with { NumBeams = 3, EarlyStopping = early });
                        Check(found.Count == expected.Count, $"{name} {caseName}: {found.Count} hypotheses, transformers {expected.Count} (early {early})");
                        for (int h = 0; h < expected.Count; h++)
                        {
                            int[] tokens = Ints(expected[h]!["ids"]);
                            Check(found[h].Tokens.SequenceEqual(tokens), $"{name} {caseName}: hypothesis {h} [{string.Join(", ", found[h].Tokens)}], transformers [{string.Join(", ", tokens)}] (early {early})");
                            AssertClose([(float)(double)expected[h]!["score"]!], [found[h].Score], T5Tolerance, $"{name} {caseName}: hypothesis {h}'s score (early {early})");
                        }

                        var best = found[0].Tokens.ToList();
                        var (beamText, beamReason, _) = generator.Generate(text, T5Greedy(10) with { NumBeams = 3, EarlyStopping = early });
                        Check(beamText == tokenizer.Decode(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(best), skipSpecialTokens: true)
                              && beamReason == ((bool)expected[0]!["ended"]! ? "stop" : "length"), $"{name} {caseName}: the beam search's text '{beamText}' ({beamReason})");
                    }
                }

                Check(name != "tiny-t5-gated" || anyEnded, $"{name}: a greedy generation ends at </s>");

                // KV-cached decoding equals the whole target: the greedy target of the first text, after a prefill of 3.
                using (new TensorScope())
                using (Autograd.NoGrad())
                {
                    var c = cases["single1"]!.AsObject();
                    int[] source = Ints(c["ids"]![0]);
                    float[] target = [model.DecoderStartToken, .. Ints(c["greedy"]).Take(9).Select(v => (float)v)];
                    var memory = model.Encoder.Encode(Tensor.From([.. source.Select(v => (float)v)], [1, source.Length], device));
                    float[] whole = model.Decode(Tensor.From(target, [1, target.Length], device), memory).ToArray();
                    using var context = new DecodingContext(device, 1, target.Length);
                    model.SetMemory(memory, context);
                    var steps = new List<float>();
                    for (int at = 0; at < target.Length;)
                    {
                        int count = at == 0 ? 3 : 1;
                        steps.AddRange(model.DecodeCached(Tensor.From(target.AsSpan(at, count), [1, count], device), context).ToArray());
                        at += count;
                    }

                    AssertClose(whole, [.. steps], T5Tolerance, $"{name}: KV-cached decoding against the whole target");
                }

                // Each sequence of the padded batch equals itself alone.
                var batch = ReadT5Case(references, name, "batch");
                using (new TensorScope())
                using (Autograd.NoGrad())
                {
                    var (states, logits) = T5Outputs(model, batch, device);
                    int steps = batch.Steps, vocabulary = model.Vocabulary;
                    for (int b = 0; b < batch.Lengths.Length; b++)
                    {
                        int length = batch.Lengths[b];
                        var alone = new T5Case($"batch[{b}]", batch.Ids.AsSpan(b * batch.Time, length).ToArray(), [length], length,
                            batch.Target.AsSpan(b * steps, steps).ToArray(), steps);
                        var (ownStates, ownLogits) = T5Outputs(model, alone, device);
                        AssertClose(states.AsSpan(b * batch.Time * 32, length * 32).ToArray(), ownStates, T5Tolerance, $"{name}: sequence {b}'s states alone and in the batch");
                        AssertClose(logits.AsSpan(b * steps * vocabulary, steps * vocabulary).ToArray(), ownLogits, T5Tolerance, $"{name}: sequence {b}'s logits alone and in the batch");
                    }
                }

                Conformance.Check(model, new Seq2SeqModelSuite(device)).ThrowIfFailed();
                Conformance.Check(model.Encoder, new SequenceEncoderSuite(device)).ThrowIfFailed();
            }
            finally
            {
                Directory.Delete(folder, recursive: true);
            }
        }
    }

    private static void T5EncoderAndEngine(Device device)
    {
        T5Plugin.Register();
        var record = T5Record();
        string folder = T5Folder("tiny-t5-gated");
        string encoderOnly = T5Folder("tiny-t5-gated", config => config["architectures"] = new JsonArray(T5Plugin.T5EncoderModel));
        string spiece = T5Folder("tiny-t5-gated", sentencePiece: true);
        try
        {
            using var references = SafeTensorsReader.Open(T5Data("reference.safetensors"));
            var c = ReadT5Case(references, "tiny-t5-gated", "batch");
            int n = c.Lengths.Length;
            float[] Seq2SeqStates(PretrainedSeq2SeqModel model)
            {
                using var scope = new TensorScope();
                using var noGrad = Autograd.NoGrad();
                return model.Model.Encoder.Encode(Tensor.From(c.Ids, [n, c.Time], device), c.Lengths).ToArray();
            }

            float[] expected;
            using (var full = LoadT5(folder, device))
            {
                expected = Seq2SeqStates(full);
            }

            // The encoder alone: a T5EncoderModel checkpoint (only its tensors), and the full checkpoint read as one.
            var tensors = new List<(string Name, int[] Shape, float[] Values)>();
            using (var stored = SafeTensorsReader.Open(Path.Combine(encoderOnly, "model.safetensors")))
            {
                tensors.AddRange(stored.Tensors.Keys.Where(k => !k.StartsWith("decoder.", StringComparison.Ordinal) && k != "lm_head.weight").Order(StringComparer.Ordinal)
                    .Select(k => (k, stored.Tensors[k].Shape, stored.Read(k))));
            }

            SafeTensorsWriter.Write(Path.Combine(encoderOnly, "model.safetensors"), tensors, SafeTensorType.F32);
            foreach (var (path, architecture) in new[] { (encoderOnly, (string?)null), (folder, T5Plugin.T5EncoderModel), (folder, T5Plugin.MT5EncoderModel) })
            {
                using var encoder = PretrainedEncoderModel.Load(path, new PretrainedEncoderOptions { Device = device, Architecture = architecture });
                Check(encoder.Pooler == SequencePoolers.Mean && encoder.Heads.Count == 0, $"{encoder.Architecture}: the mean pooler and no heads");
                using var scope = new TensorScope();
                using var noGrad = Autograd.NoGrad();
                BitsEqual(expected, encoder.Hidden(Tensor.From(c.Ids, [n, c.Time], device), c.Lengths).ToArray(), $"{encoder.Architecture} from {Path.GetFileName(path)}: the states");
                Check(encoder.Pooled(Tensor.From(c.Ids, [n, c.Time], device), c.Lengths).Shape.SequenceEqual([n, 32]), $"{encoder.Architecture}: one vector per sequence");
            }

            var cases = record["models"]!["tiny-t5-gated"]!["cases"]!.AsObject();
            string text = (string)cases["single0"]!["text"]!, greedyText = (string)cases["single0"]!["greedy_text"]!;
            using (var mt5 = LoadT5(folder, device, T5Plugin.MT5ForConditionalGeneration))
            {
                BitsEqual(expected, Seq2SeqStates(mt5), "MT5ForConditionalGeneration: the states");
            }

            // spiece.model alone: no post-processor, so the generator appends </s> (the family's sources end with it).
            using (var sentencePiece = LoadT5(spiece, device))
            {
                var tokenizer = sentencePiece.Tokenizer!;
                var bare = tokenizer.Encode(text, null, new TokenizerEncodeOptions());
                Check(bare.Ids.Length > 0 && bare.Ids[^1] != 1, "the SentencePiece model adds no </s>");
                var (answer, reason, stats) = sentencePiece.CreateGenerator().Generate(text, T5Greedy(16));
                Check(answer == greedyText && stats.PromptTokens == bare.Ids.Length + 1, $"spiece.model: '{answer}' ({reason}) from {stats.PromptTokens} source tokens, transformers '{greedyText}'");
            }

            // The inference engine's encoder-decoder kind.
            var engine = InferenceEngine.Create().Seq2SeqModel("t5", folder, b => b.Device(device)).BuildAsync().GetAwaiter().GetResult();
            try
            {
                Check(engine.Models.Single().Kind == "seq2seq" && engine.Models.Single().Loaded, $"the engine hosts a loaded \"seq2seq\" model: {engine.Models.Single()}");
                var streamed = new System.Text.StringBuilder();
                GenerationChunk? last = null;
                var chunks = engine.Model<ITextModel>("t5").StreamAsync(text, T5Greedy(16)).GetAsyncEnumerator();
                while (chunks.MoveNextAsync().AsTask().GetAwaiter().GetResult())
                {
                    streamed.Append(chunks.Current.Text);
                    last = chunks.Current;
                }

                Check(streamed.ToString() == greedyText && last is { Done: true, DoneReason: "stop" }, $"the engine's answer '{streamed}', transformers '{greedyText}'");
                ThrowsNaming<InvalidOperationException>(() => InferenceEngine.Create().Seq2SeqModel("x", folder, b => b.Template(new Idrak.Text.JinjaChatTemplate("{{ messages }}", []))),
                    "a chat template on an encoder-decoder", "encoder-decoder", "Template");
            }
            finally
            {
                engine.DisposeAsync().AsTask().GetAwaiter().GetResult();
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
            Directory.Delete(encoderOnly, recursive: true);
            Directory.Delete(spiece, recursive: true);
        }
    }

    private static void T5Registration(Device device)
    {
        bool wasRegistered = T5Plugin.IsRegistered;
        string folder = T5Folder("tiny-t5");
        string[] all = [.. T5Plugin.Seq2SeqArchitectures, .. T5Plugin.EncoderArchitectures];
        try
        {
            Check(TextPlugins.All.Contains(T5Plugin.Plugin) && T5Plugin.Plugin.Name == "t5" && T5Plugin.Plugin.GetType().Namespace == "Idrak.Text.Plugins.T5",
                "T5Plugin.Plugin is \"t5\", of TextPlugins.All, in Idrak.Text.Plugins.T5");
            T5Plugin.Unregister();
            Check(!T5Plugin.IsRegistered && IdrakPlugins.Find("t5") is null && T5Plugin.Seq2SeqArchitectures.All(a => PretrainedSeq2Seq.Find(a) is null)
                && T5Plugin.EncoderArchitectures.All(a => PretrainedEncoders.Find(a) is null), "unregistered: no T5 architecture is registered");
            var seq2seq = PretrainedSeq2Seq.Names.Order(StringComparer.Ordinal).ToList();
            var encoders = PretrainedEncoders.Names.Order(StringComparer.Ordinal).ToList();
            ThrowsNaming<NotSupportedException>(() => LoadT5(folder, device).Dispose(), "a T5 checkpoint without the plug-in", "not registered",
                T5Plugin.T5ForConditionalGeneration, TextPlugins.Package);
            ThrowsNaming<NotSupportedException>(() => PretrainedEncoderModel.Load(folder, new PretrainedEncoderOptions { Device = device, Architecture = T5Plugin.T5EncoderModel }).Dispose(),
                "a T5 encoder without the plug-in", "not registered", T5Plugin.T5EncoderModel, TextPlugins.Package);

            T5Plugin.Register();
            T5Plugin.Register();
            Check(T5Plugin.IsRegistered && ReferenceEquals(IdrakPlugins.Find("t5"), T5Plugin.Plugin), "registered and recorded as \"t5\"");
            Check(PretrainedSeq2Seq.Names.Order(StringComparer.Ordinal).SequenceEqual(seq2seq.Concat(T5Plugin.Seq2SeqArchitectures).Order(StringComparer.Ordinal))
                && PretrainedEncoders.Names.Order(StringComparer.Ordinal).SequenceEqual(encoders.Concat(T5Plugin.EncoderArchitectures).Order(StringComparer.Ordinal)),
                $"Register twice registers the four architectures once: {string.Join(", ", PretrainedSeq2Seq.Names)}; {string.Join(", ", PretrainedEncoders.Names)}");
            Check(T5Plugin.Seq2SeqArchitectures.All(a => PretrainedSeq2Seq.Origin(a) == TextPlugins.Package) && T5Plugin.EncoderArchitectures.All(a => PretrainedEncoders.Origin(a) == TextPlugins.Package),
                "the architectures are Idrak.Text.Plugins'");
            using (var model = LoadT5(folder, device))
            {
                Check(model.Architecture == T5Plugin.T5ForConditionalGeneration, "registered, the checkpoint loads");
            }

            T5Plugin.Unregister();
            Check(PretrainedSeq2Seq.Names.Order(StringComparer.Ordinal).SequenceEqual(seq2seq) && PretrainedEncoders.Names.Order(StringComparer.Ordinal).SequenceEqual(encoders),
                "Unregister leaves PretrainedSeq2Seq and PretrainedEncoders as found");
            Check(all.All(a => PretrainedSeq2Seq.Find(a) is null && PretrainedEncoders.Find(a) is null), "nothing of T5 is left");
            TextPlugins.RegisterAll();
            Check(T5Plugin.IsRegistered, "TextPlugins.RegisterAll registers T5");
        }
        finally
        {
            if (wasRegistered)
            {
                T5Plugin.Register();
            }
            else
            {
                T5Plugin.Unregister();
            }

            Directory.Delete(folder, recursive: true);
        }
    }

    private static void T5Refused(Device device)
    {
        T5Plugin.Register();
        var folders = new List<string>();
        string Edited(string model, Action<JsonObject> edit)
        {
            string path = T5Folder(model, edit);
            folders.Add(path);
            return path;
        }

        try
        {
            ThrowsNaming<NotSupportedException>(() => LoadT5(Edited("tiny-t5", j => j["feed_forward_proj"] = "gated-mish"), device).Dispose(), "an unknown feed-forward",
                "feed_forward_proj", "gated-mish");
            ThrowsNaming<NotSupportedException>(() => LoadT5(Edited("tiny-t5", j => j["relative_attention_num_buckets"] = 7), device).Dispose(), "an odd number of buckets",
                "relative_attention_num_buckets");
            ThrowsNaming<NotSupportedException>(() => LoadT5(Edited("tiny-t5", j => j["relative_attention_max_distance"] = 8), device).Dispose(), "a max distance the exact buckets cover",
                "relative_attention_max_distance");
            ThrowsNaming<InvalidDataException>(() => LoadT5(Edited("tiny-t5", j => j["tie_word_embeddings"] = false), device).Dispose(), "an untied model without its head",
                "lm_head.weight");
            ThrowsNaming<InvalidDataException>(() => LoadT5(Edited("tiny-t5", j => j["d_ff"] = 64), device).Dispose(), "a tensor of another shape",
                "encoder.block.0.layer.1.DenseReluDense.wo.weight", "configuration");
            ThrowsNaming<InvalidDataException>(() => LoadT5(Edited("tiny-t5", j => j["num_decoder_layers"] = 3), device).Dispose(), "a decoder block the checkpoint lacks",
                "decoder.block.2.");
            ThrowsNaming<InvalidDataException>(() => LoadT5(Edited("tiny-t5", j => j.Remove("d_kv")), device).Dispose(), "a missing size", "d_kv");
        }
        finally
        {
            foreach (string path in folders)
            {
                Directory.Delete(path, recursive: true);
            }
        }
    }

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

    // BeamSearch.Decode with the candidates chosen on the device and on the host: the same hypotheses, bit for bit on the CPU.
    private static void BeamsOnDevice(Device device)
    {
        string folder = T5Folder("tiny-t5-gated");
        try
        {
            using var loaded = LoadT5(folder, device);
            var model = loaded.Model;
            var c = T5Record()["models"]!["tiny-t5-gated"]!["cases"]!["single0"]!.AsObject();
            float[] source = [.. Ints(c["ids"]![0]).Select(i => (float)i)];
            IReadOnlyList<BeamHypothesis> Search(BeamOptions options)
            {
                using var context = new DecodingContext(device, options.Beams, 32) { LastPositionOnly = true };
                model.Network.Eval();
                using (Autograd.NoGrad())
                using (new TensorScope())
                {
                    model.SetMemory(model.Encoder.Encode(Tensor.From(source, [1, source.Length], device)), context);
                }

                return BeamSearch.Decode((ids, ctx) => model.DecodeCached(ids, ctx), context, [model.DecoderStartToken], options);
            }

            foreach (var options in new[]
                     {
                         new BeamOptions { Beams = 3, MaxTokens = 10, EndToken = model.EndToken },
                         new BeamOptions { Beams = 4, MaxTokens = 12, EndToken = model.EndToken, EarlyStopping = false, LengthPenalty = 0.6f, MinTokens = 3, NoRepeatNgram = 2,
                             SuppressTokens = [5, 9, 9, 5000], BeginSuppressTokens = [2, 3] },
                         new BeamOptions { Beams = 2, MaxTokens = 6, EndToken = -1 },
                     })
            {
                var host = Search(options with { SelectOnDevice = false });
                var onDevice = Search(options);
                bool same = host.Count == onDevice.Count && host.Zip(onDevice).All(p => p.First.Tokens.SequenceEqual(p.Second.Tokens)
                    && (device == Device.Cpu ? p.First.Score.Equals(p.Second.Score) && p.First.LogProb.Equals(p.Second.LogProb) : MathF.Abs(p.First.Score - p.Second.Score) < 1e-4f));
                Check(same, $"beams {options.Beams}: on the device {string.Join(" | ", onDevice.Select(h => $"{string.Join(",", h.Tokens)} {h.Score:R}"))}; "
                            + $"on the host {string.Join(" | ", host.Select(h => $"{string.Join(",", h.Tokens)} {h.Score:R}"))}");
            }

            var tokenizer = loaded.Tokenizer!;
            var beams = new GenerationOptions { NumBeams = 3, NumPredict = 10 };
            var ids = Ints(c["ids"]![0]);
            var fromDevice = new Seq2SeqGenerator(model, tokenizer).GenerateBeams(ids, beams);
            var fromHost = new Seq2SeqGenerator(model, tokenizer) { SelectBeamsOnDevice = false }.GenerateBeams(ids, beams);
            Check(fromDevice.Zip(fromHost).All(p => p.First.Tokens.SequenceEqual(p.Second.Tokens)) && fromDevice[0].Tokens.SequenceEqual(Ints(c["beam_early"]![0]!["ids"])),
                "Seq2SeqGenerator: the same beams with SelectBeamsOnDevice off, transformers' best");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    // SameValues reads both tensors a chunk at a time: the answers of the whole arrays' comparison, and (warm) next to no
    // allocation for two tensors of a million values and more.
    private static void SafeTensorsSameValues(Device device)
    {
        _ = device;
        string folder = EngineFolder("same-values");
        try
        {
            int n = (1 << 20) + 37;
            float[] d = [.. Enumerable.Range(0, n).Select(i => (float)(i % 256 - 128))];       // exact in bfloat16
            float[] c = (float[])d.Clone(), a = (float[])d.Clone(), b = (float[])d.Clone(), h = (float[])d.Clone();
            c[n - 1] += 1f;                                                                     // apart from d in the last chunk only
            (a[0], b[0]) = (0f, -0f);                                                           // equal as floats
            a[5] = b[5] = float.NaN;
            h[0] = 0f;                                                                          // a with 0 where a holds NaN
            SafeTensorsWriter.Write(Path.Combine(folder, "one.safetensors"), [("a", [n], a), ("b", [n], b), ("c", [n], c), ("d", [n], d), ("s", [1, n], d)], SafeTensorType.F32);
            SafeTensorsWriter.Write(Path.Combine(folder, "two.safetensors"), [("g", [n], d), ("h", [n], h)], SafeTensorType.BF16);
            var map = new JsonObject();
            foreach (string name in new[] { "a", "b", "c", "d", "s" })
            {
                map[name] = "one.safetensors";
            }

            map["g"] = "two.safetensors";
            map["h"] = "two.safetensors";
            File.WriteAllText(Path.Combine(folder, "model.safetensors.index.json"), new JsonObject { ["weight_map"] = map }.ToJsonString());
            using var reader = SafeTensorsReader.Open(folder);
            var whole = new TensorDictionary();
            foreach (string name in reader.Tensors.Keys)
            {
                whole.Add(name, reader.Tensors[name].Shape, reader.Read(name));
            }

            using var renamed = new RenamedTensorStore(reader, name => name[1..], ownsInner: false);
            foreach (var (x, y, expected) in new[]
                     {
                         ("a", "b", true), ("d", "g", true), ("g", "g", true), ("c", "d", false), ("c", "g", false), ("s", "d", false), ("a", "h", false), ("a", "d", false),
                     })
            {
                bool chunked = reader.SameValues(x, y), arrays = ((ITensorStore)whole).SameValues(x, y);
                Check(chunked == expected && arrays == expected && renamed.SameValues("x" + x, "x" + y) == expected,
                    $"SameValues({x}, {y}): chunked {chunked}, whole arrays {arrays}, expected {expected}");
            }

            long before = GC.GetAllocatedBytesForCurrentThread();
            Check(reader.SameValues("a", "b"), "a equals b");
            long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            Check(allocated < 64 * 1024, $"SameValues of two tensors of {n} values allocated {allocated} bytes (warm): no whole copy");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    // A tied checkpoint holding lm_head.weight: compared with the shared embedding at load (piece by piece); equal, the
    // head reads the one table; one value apart, the head is its own.
    private static void T5TiedHead(Device device)
    {
        string plain = T5Folder("tiny-t5"), same = T5Folder("tiny-t5"), apart = T5Folder("tiny-t5");
        try
        {
            void AddHead(string folder, bool change)
            {
                string path = Path.Combine(folder, "model.safetensors");
                List<(string Name, int[] Shape, float[] Values)> tensors;
                using (var stored = SafeTensorsReader.Open(path))
                {
                    tensors = [.. stored.Tensors.Keys.Order(StringComparer.Ordinal).Select(k => (k, stored.Tensors[k].Shape, stored.Read(k)))];
                }

                var shared = tensors.Single(t => t.Name == "shared.weight");
                var head = (float[])shared.Values.Clone();
                if (change)
                {
                    head[^1] += 0.25f;
                }

                tensors.Add(("lm_head.weight", shared.Shape, head));
                SafeTensorsWriter.Write(path, tensors, SafeTensorType.F32);
            }

            AddHead(same, change: false);
            AddHead(apart, change: true);
            var c = T5Record()["models"]!["tiny-t5"]!["cases"]!["single0"]!.AsObject();
            float[] source = [.. Ints(c["ids"]![0]).Select(i => (float)i)];
            (long Parameters, float[] Logits) Run(string folder)
            {
                using var loaded = LoadT5(folder, device);
                var model = loaded.Model;
                model.Network.Eval();
                using var noGrad = Autograd.NoGrad();
                using var scope = new TensorScope();
                var memory = model.Encoder.Encode(Tensor.From(source, [1, source.Length], device));
                return (model.Network.ParameterCount, model.Decode(Tensor.From([0f, 5f, 7f], [1, 3], device), memory).ToArray());
            }

            var (plainCount, plainLogits) = Run(plain);
            var (sameCount, sameLogits) = Run(same);
            var (apartCount, apartLogits) = Run(apart);
            Check(sameCount == plainCount && apartCount > plainCount, $"parameters: without lm_head {plainCount}, equal {sameCount} (one table), apart {apartCount} (a head of its own)");
            BitsEqual(plainLogits, sameLogits, "the head equal to the shared embedding: the tied model's logits");
            Check(!apartLogits.SequenceEqual(plainLogits), "a head one value apart is read as its own");
        }
        finally
        {
            Directory.Delete(plain, recursive: true);
            Directory.Delete(same, recursive: true);
            Directory.Delete(apart, recursive: true);
        }
    }

    // ---------------------------------------------------------------- idrak run with an encoder-decoder

    private static void CliRunSeq2Seq(Device device)
    {
        string folder = T5Folder("tiny-t5-gated");
        try
        {
            var c = T5Record()["models"]!["tiny-t5-gated"]!["cases"]!["single0"]!.AsObject();
            string text = (string)c["text"]!, greedyText = (string)c["greedy_text"]!;
            var (code, output, error) = RunIdrakOn(device, null, "run", folder, text, "--temperature", "0", "--max-tokens", "16", "--json");
            Check(code == 0, $"run on the T5 folder exits 0: {error}");
            var json = JsonOf(output, "run --json");
            Check((string?)json["text"] == greedyText && (string?)json["done_reason"] == "stop" && (string?)json["architecture"] == T5Plugin.T5ForConditionalGeneration
                  && (int?)json["generated_tokens"] == Ints(c["greedy"]).Length && (int?)json["prompt_tokens"] == Ints(c["ids"]![0]).Length && (int?)json["beams"] == 1,
                $"run --json: {output}");
            (code, output, error) = RunIdrakOn(device, null, "run", folder, text, "--temperature", "0", "--max-tokens", "16");
            Check(code == 0 && output.Trim() == greedyText.Trim(), $"run prints transformers' greedy text: '{output}' ({error})");

            var tokenizer = (IPretrainedTokenizer)PretrainedTokenizers.Load(folder);
            var best = Ints(c["beam_early"]![0]!["ids"]);
            (code, output, error) = RunIdrakOn(device, null, "run", folder, text, "--beams", "3", "--max-tokens", "10", "--json");
            var beamJson = JsonOf(output, "run --beams 3 --json");
            Check(code == 0 && (string?)beamJson["text"] == tokenizer.Decode(best, skipSpecialTokens: true) && (int?)beamJson["beams"] == 3,
                $"run --beams 3 answers with the best hypothesis: {output} ({error})");

            foreach (var (args, option) in new[]
                     {
                         (new[] { "--schema", "person.json" }, "--schema"), (new[] { "--system", "be brief" }, "--system"), (new[] { "--beams", "0" }, "--beams"),
                         (new[] { "--length-penalty", "nan" }, "--length-penalty"),
                     })
            {
                (code, _, error) = RunIdrakOn(device, null, ["run", folder, text, .. args]);
                Check(code == Idrak.Cli.ExitCodes.Usage && error.Contains(option, StringComparison.Ordinal), $"run {string.Join(" ", args)} is refused naming {option}: {error}");
            }

            (code, _, error) = RunIdrakOn(device, null, "run", CliModel, "hello", "--beams", "2");
            Check(code == Idrak.Cli.ExitCodes.Usage && error.Contains("--beams", StringComparison.Ordinal) && error.Contains("decoder", StringComparison.Ordinal),
                $"run --beams 2 on a decoder is refused: {error}");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    // ---------------------------------------------------------------- the owner's real models

    // Each real/<name>.json written by make_tiny_t5.py --real: the model from the cache (IDRAK_CACHE, else ~/.cache/idrak)
    // against transformers' references. 2e-3 (absolute up to 1, relative above): six or eight layers of width 512 with
    // feed-forwards of 1,024 to 2,048 terms sum in float32 in other orders than PyTorch's, and T5's activations grow large;
    // a wrong layer still moves the states by far more.
    private static void T5Real(Device device)
    {
        T5Plugin.Register();
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
