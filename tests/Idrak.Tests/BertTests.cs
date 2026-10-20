// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Plugins;
using Idrak.Abstraction.Testing;
using Idrak.Layers;
using Idrak.Layers.Abstractions;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Text.Plugins;
using Idrak.Text.Plugins.Bert;
using Idrak.Text.Retrieval;

// Plan 26, phase P9: the post-norm encoder block (generic, in the library) and the BERT plug-in (Idrak.Text.Plugins.Bert)
// against transformers' outputs for a tiny random BERT (tools/bert/make_tiny_bert.py, tests/Idrak.Tests/data/plugins/bert).
//
// The tolerance against transformers is 1e-4, absolute up to 1 and relative above (AssertClose): both sides compute in
// float32, so they differ only by summation order (blocked products, the softmax, LayerNorm's mean and variance) and by
// the erf of exact GELU (the library's within 1.5e-7), carried through two layers and three LayerNorms whose epsilon
// (1e-12) divides by the spread itself. That is some hundred float32 steps at values near 1, while the mistakes the test
// is for are far larger: a LayerNorm epsilon of 1e-5 instead of 1e-12, tanh GELU instead of erf, a pre-norm block, a
// transposed projection or a missing bias each move the outputs by 1e-3 or more.
internal static partial class Tests
{
    private const float BertTolerance = 1e-4f;

    // The owner's real models: more layers and width, so more rounding (see BertReal).
    private const float BertRealTolerance = 2e-3f;

    private static readonly (string Name, Action<Device> Run)[] BertGroup =
    [
        ("engine: post-norm block: TransformerEncoderLayer with NormPlacement.After equals its own sublayers composed as LayerNorm(x + Attention(x)), then LayerNorm(x + FeedForward(x)) (tanh GELU fused, and an exact GELU), in inference and with the gradient recorded (every parameter gets a gradient); cached steps equal the whole causal sequence; Before stays the default and the earlier overloads' bits; the layer description and the network builder carry \"norms\": \"after\" both ways; an undefined placement and unknown norms are refused naming the setting", PostNormBlock),
        ("plugins: bert: the tiny checkpoint read as BertForMaskedLM, BertModel and BertForSequenceClassification gives transformers' hidden states, pooled output, masked-LM logits and classification logits within 1e-4 for three single texts and a padded batch of pairs with token types, in inference and with the gradient recorded; the folder's WordPiece tokenizer gives transformers' ids and type ids", BertMatchesTransformers),
        ("plugins: bert: each sequence of the padded batch equals itself alone (states, pooled output, masked-LM and class logits); the BERT encoder passes the sequence encoder suite; the masked-LM decoder reads the word embeddings in place (no copy), and an untied checkpoint (tie_word_embeddings false, its own decoder weight) gives the same logits", BertBatchAndHeads),
        ("plugins: bert: TextEncoder over the BERT encoder with SpecialTokens gives transformers' mean and first-token unit vectors within 1e-4, a text alone as in the batch; SpecialTokens is off by default and refused for a tokenizer without special tokens", BertTextEncoder),
        ("plugins: bert: BertPlugin is the plug-in \"bert\" of TextPlugins.All; Register twice registers BertModel, BertForMaskedLM and BertForSequenceClassification once, from Idrak.Text.Plugins; Unregister leaves PretrainedEncoders as found; without it the checkpoint is refused as not registered, naming the architecture and Idrak.Text.Plugins", BertRegistration),
        ("plugins: bert: BertModel's layout (no \"bert.\" prefix, LayerNorm gamma and beta, an int64 position_ids the safetensors reader leaves out) gives the tiny model's bits; refused naming the cause: position_embedding_type relative_key, an unknown hidden_act, a tensor of another shape than the configuration's, a BertForSequenceClassification or BertForMaskedLM without its head, an untied model without its decoder; the pooler head kind and a tied head of another shape are refused naming the setting", BertLayouts),
        ("bert real: (the owner's: make_tiny_bert.py --real first) google-bert/bert-base-uncased and aubmindlab/bert-base-arabertv02 from the cache give transformers' ids, hidden states and pooled output within 2e-3 and the masked-LM logits of each [MASK] at transformers' 32 best tokens, the best the same", BertReal),
    ];

    private static readonly string[] BertCases = ["single0", "single1", "single2", "batch"];

    private static string BertData(string relative) => TestData(Path.Combine("plugins", "bert", relative));

    // The tiny model in a folder of its own, with the committed WordPiece tokenizer beside it; `edit` changes its config.json.
    private static string BertFolder(Action<JsonObject>? edit = null)
    {
        string folder = EngineFolder("bert");
        File.Copy(BertData("tiny-bert/model.safetensors"), Path.Combine(folder, "model.safetensors"));
        var config = JsonNode.Parse(File.ReadAllText(BertData("tiny-bert/config.json")))!.AsObject();
        edit?.Invoke(config);
        File.WriteAllText(Path.Combine(folder, "config.json"), config.ToJsonString());
        foreach (string name in new[] { "tokenizer.json", "tokenizer_config.json", "vocab.txt" })
        {
            File.Copy(TestData($"tokenizers/wordpiece-uncased/{name}"), Path.Combine(folder, name));
        }

        return folder;
    }

    private static PretrainedEncoderModel LoadBert(string folder, Device device, string? architecture = null) =>
        PretrainedEncoderModel.Load(folder, new PretrainedEncoderOptions { Device = device, Architecture = architecture });

    // A case of reference.safetensors: its padded ids, type ids, lengths and shape.
    private sealed record BertCase(string Name, float[] Ids, float[] Types, int[] Lengths, int Time);

    private static BertCase ReadBertCase(SafeTensorsReader references, string name)
    {
        int[] shape = references.Tensors[$"{name}.ids"].Shape;
        return new BertCase(name, references.Read($"{name}.ids"), references.Read($"{name}.types"),
            [.. references.Read($"{name}.lengths").Select(l => (int)l)], shape[1]);
    }

    // The largest difference, after AssertClose checked every element.
    private static float Compared(float[] expected, float[] actual, float tolerance, string what)
    {
        AssertClose(expected, actual, tolerance, what);
        float worst = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            worst = MathF.Max(worst, MathF.Abs(expected[i] - actual[i]));
        }

        return worst;
    }

    // The outputs of the model for a case: states, the pooler head's, the masked-LM logits at the case's positions and the class logits.
    private static (float[] Hidden, float[] Pooled, float[] Words, float[] Classes) BertOutputs(PretrainedEncoderModel model, BertCase c, float[] positions, Device device)
    {
        int n = c.Lengths.Length;
        Tensor Ids() => Tensor.From(c.Ids, [n, c.Time], device);
        Tensor Types() => Tensor.From(c.Types, [n, c.Time], device);
        float[] hidden = model.Hidden(Ids(), c.Lengths, Types()).ToArray();
        float[] pooled = model.Logits(EncoderHeads.Pooler, Ids(), c.Lengths, Types()).ToArray();
        float[] all = model.Logits(EncoderHeads.MaskedLanguageModel, Ids(), c.Lengths, Types()).ToArray();
        int vocabulary = model.Encoder.Vocabulary, per = positions.Length / n;
        var words = new float[positions.Length * vocabulary];
        for (int b = 0; b < n; b++)
        {
            for (int k = 0; k < per; k++)
            {
                int at = (int)positions[b * per + k];
                all.AsSpan((b * c.Time + at) * vocabulary, vocabulary).CopyTo(words.AsSpan((b * per + k) * vocabulary));
            }
        }

        float[] classes = model.Logits(EncoderHeads.SequenceClassification, Ids(), c.Lengths, Types()).ToArray();
        return (hidden, pooled, words, classes);
    }

    // ---------------------------------------------------------------- the post-norm block (library)

    private static void PostNormBlock(Device device)
    {
        var random = new Random(26);
        const int n = 2, t = 6, d = 16, heads = 4, ff = 24;
        float[] g1 = [.. RandomArray(random, d, 0.1f).Select(v => 1f + v)], b1 = RandomArray(random, d, 0.1f), g2 = [.. RandomArray(random, d, 0.1f).Select(v => 1f + v)],
            b2 = RandomArray(random, d, 0.1f), wqkv = RandomArray(random, d * 3 * d, 0.4f), bqkv = RandomArray(random, 3 * d, 0.1f), wo = RandomArray(random, d * d, 0.4f),
            bo = RandomArray(random, d, 0.1f), w1 = RandomArray(random, d * ff, 0.4f), c1 = RandomArray(random, ff, 0.1f), w2 = RandomArray(random, ff * d, 0.4f),
            c2 = RandomArray(random, d, 0.1f), x = RandomArray(random, n * t * d);
        LayerNorm Norm(float[] gain, float[] shift) =>
            LayerNorm.FromWeights(Tensor.Persistent(gain, [d], device, requiresGrad: true), Tensor.Persistent(shift, [d], device, requiresGrad: true), 1e-12f);
        MultiHeadAttention Attention(bool causal) => new(Projection(wqkv, bqkv, d, 3 * d, device), Projection(wo, bo, d, d, device), heads, causal);
        TransformerEncoderLayer Block(NormPlacement? placement, Module? activation, bool causal = false) => placement is { } p
            ? TransformerEncoderLayer.FromLayers(Norm(g1, b1), Attention(causal), Norm(g2, b2), Projection(w1, c1, d, ff, device), Projection(w2, c2, ff, d, device), activation, p)
            : activation is null
                ? TransformerEncoderLayer.FromLayers(Norm(g1, b1), Attention(causal), Norm(g2, b2), Projection(w1, c1, d, ff, device), Projection(w2, c2, ff, d, device))
                : TransformerEncoderLayer.FromLayers(Norm(g1, b1), Attention(causal), Norm(g2, b2), Projection(w1, c1, d, ff, device), Projection(w2, c2, ff, d, device), activation);

        foreach (Module? activation in new Module?[] { null, new ExactGELU() })
        {
            string name = activation is null ? "tanh GELU" : "exact GELU";
            using var block = Block(NormPlacement.After, activation);
            Check(block.Placement == NormPlacement.After && block.ToString().Contains("post-norm", StringComparison.Ordinal), $"{name}: the block says it is post-norm: {block}");
            var parts = block.Children().ToList();
            var (norm1, attention, norm2, ff1, ff2) = ((LayerNorm)parts[0], (MultiHeadAttention)parts[1], (LayerNorm)parts[2], (Linear)parts[3], (Linear)parts[^1]);
            float[] expected, inference, recorded;
            using (new TensorScope())
            using (Autograd.NoGrad())
            {
                var input = Tensor.From(x, [n, t, d], device);
                var y = norm1.Forward(input + attention.Forward(input));
                var hidden = activation is null ? ff1.Forward(y).Gelu() : activation.Forward(ff1.Forward(y));
                expected = norm2.Forward(y + ff2.Forward(hidden)).ToArray();
                inference = block.Forward(input).ToArray();
            }

            AssertClose(expected, inference, 1e-6f, $"{name}: the post-norm block in inference against its sublayers composed");
            using (new TensorScope())
            {
                var input = Tensor.From(x, [n, t, d], device);
                var output = block.Forward(input);
                recorded = output.ToArray();
                output.Square().Sum().Backward();
                Check(block.Parameters().All(p => p.Grad?.ToArray().Any(v => v != 0f) == true), $"{name}: every parameter of the post-norm block gets a gradient");
            }

            AssertClose(expected, recorded, 1e-5f, $"{name}: the post-norm block with the gradient recorded");

            // Pre-norm stays the default, and the explicit placement gives the earlier overloads' bits.
            using var before = Block(NormPlacement.Before, activation);
            using var earlier = Block(null, activation);
            using (new TensorScope())
            using (Autograd.NoGrad())
            {
                var input = Tensor.From(x, [n, t, d], device);
                Check(earlier.Placement == NormPlacement.Before && new TransformerEncoderLayer(d, heads, ff, 0f, device: device).Placement == NormPlacement.Before,
                    "pre-norm is the default");
                BitsEqual(earlier.Forward(input).ToArray(), before.Forward(input).ToArray(), $"{name}: NormPlacement.Before against the earlier overload");
                Check(!expected.SequenceEqual(before.Forward(input).ToArray()), $"{name}: pre-norm and post-norm differ");
            }
        }

        // Cached steps of a causal post-norm block equal the whole sequence.
        using (var causal = Block(NormPlacement.After, new ExactGELU(), causal: true))
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

            AssertClose(whole, [.. steps], 1e-4f, "a causal post-norm block: cached steps after a prefill of 2 against the whole sequence");
        }

        // The layer description and the network builder carry the placement.
        using (var described = new TransformerEncoderLayer(d, heads, ff, 0f, device: device) { Placement = NormPlacement.After })
        {
            var description = LayerTypes.Describe(described);
            Check((string?)description["norms"] == "after", $"a post-norm layer's description says so: {description.ToJsonString()}");
            using var rebuilt = (TransformerEncoderLayer)LayerTypes.Create(description, device);
            Check(rebuilt.Placement == NormPlacement.After, "the description rebuilds a post-norm layer");
            using var pre = new TransformerEncoderLayer(d, heads, ff, 0f, device: device);
            Check(LayerTypes.Describe(pre)["norms"] is null, "a pre-norm layer is described as before");
            description["norms"] = "sideways";
            ThrowsNaming<InvalidDataException>(() => LayerTypes.Create(description, device), "unknown norms in a description", "norms", "sideways");
        }

        var builder = Network.Tokens(5).Seed(45).Embedding(11, 8).TransformerEncoderLayer(2, ffDim: 12, dropout: 0f, placement: NormPlacement.After).MeanOverTime();
        var json = builder.ToJson();
        Check(json.ToJsonString().Contains("\"norms\":\"after\"", StringComparison.Ordinal), $"the builder writes the placement: {json.ToJsonString()}");
        var replayed = Network.FromJson(json);
        Check(JsonNode.DeepEquals(replayed.ToJson(), json), "the builder's JSON replays");
        using (var built = builder.OnDevice(device).Build())
        using (var again = replayed.OnDevice(device).Build())
        {
            Check(built.Descendants().OfType<TransformerEncoderLayer>().Single().Placement == NormPlacement.After
                && again.Descendants().OfType<TransformerEncoderLayer>().Single().Placement == NormPlacement.After, "the builder's layer is post-norm, replayed too");
            built.Eval();
            again.Eval();
            using var ids = Tensor.From([1f, 4f, 2f, 9f, 3f, 0f, 7f, 5f, 6f, 10f], [2, 5], device);
            BitsEqual(built.Predict(ids).ToArray(), again.Predict(ids).ToArray(), "the replayed post-norm network");
        }

        ThrowsNaming<ArgumentOutOfRangeException>(() => Block((NormPlacement)7, null).Dispose(), "an undefined placement", "Placement");
        ThrowsNaming<ArgumentOutOfRangeException>(() => new TransformerEncoderLayer(d, heads, ff, 0f, device: device) { Placement = (NormPlacement)7 }.Dispose(),
            "an undefined placement set on a new layer", "Placement");
    }

    // ---------------------------------------------------------------- BERT against transformers

    private static void BertMatchesTransformers(Device device)
    {
        BertPlugin.Register();
        string folder = BertFolder();
        try
        {
            using var references = SafeTensorsReader.Open(BertData("reference.safetensors"));
            var cases = JsonNode.Parse(File.ReadAllText(BertData("reference.json")))!["cases"]!.AsObject();
            using var model = LoadBert(folder, device);
            Check(model.Architecture == BertPlugin.BertForMaskedLM && model.Pooler == SequencePoolers.First
                && model.Heads.Keys.Order(StringComparer.Ordinal).SequenceEqual([EncoderHeads.MaskedLanguageModel, EncoderHeads.Pooler, EncoderHeads.SequenceClassification]),
                $"BertForMaskedLM with the pooler, the masked-LM head and the classifier the checkpoint holds ({model.Pooler}; {string.Join(", ", model.Heads.Keys)})");
            Check(model.Encoder is SequenceEncoder { Blocks.Count: 2, TypeVocabulary: 2, MaxPositions: 64, Width: 32, Vocabulary: 1861 } encoder
                && encoder.Blocks.All(b => b is TransformerEncoderLayer { Placement: NormPlacement.After }), $"two post-norm blocks: {model.Encoder}");
            var tokenizer = model.Tokenizer ?? throw new InvalidOperationException("the folder's tokenizer was not read");
            int width = model.Encoder.Width, vocabulary = model.Encoder.Vocabulary;
            float worstHidden = 0f, worstPooled = 0f, worstWords = 0f, worstClasses = 0f;
            foreach (string name in BertCases)
            {
                var c = ReadBertCase(references, name);
                int n = c.Lengths.Length;

                // The tokenizer gives transformers' ids and type ids (special tokens and pairs included).
                var record = cases[name]!.AsObject();
                for (int b = 0; b < n; b++)
                {
                    var encoding = record["pairs"] is JsonArray pairs
                        ? tokenizer.Encode((string)pairs[b]![0]!, (string)pairs[b]![1]!, new TokenizerEncodeOptions())
                        : tokenizer.Encode((string)record["text"]!, null, new TokenizerEncodeOptions());
                    int length = c.Lengths[b];
                    Check(encoding.Ids.SequenceEqual(c.Ids.AsSpan(b * c.Time, length).ToArray().Select(v => (int)v))
                        && encoding.TypeIds.SequenceEqual(c.Types.AsSpan(b * c.Time, length).ToArray().Select(v => (int)v)),
                        $"{name}[{b}]: the tokenizer's ids {string.Join(" ", encoding.Ids)} and types {string.Join(" ", encoding.TypeIds)}");
                }

                float[] positions = references.Read($"{name}.mlm_positions");
                float[] hidden = RealPositions(references.Read($"{name}.hidden"), c.Lengths, c.Time, width);
                (float[] Hidden, float[] Pooled, float[] Words, float[] Classes) inference;
                using (new TensorScope())
                using (Autograd.NoGrad())
                {
                    inference = BertOutputs(model, c, positions, device);
                }

                worstHidden = MathF.Max(worstHidden, Compared(hidden, RealPositions(inference.Hidden, c.Lengths, c.Time, width), BertTolerance, $"{name}: hidden states"));
                worstPooled = MathF.Max(worstPooled, Compared(references.Read($"{name}.pooled"), inference.Pooled, BertTolerance, $"{name}: pooled output"));
                worstWords = MathF.Max(worstWords, Compared(references.Read($"{name}.mlm"), inference.Words, BertTolerance, $"{name}: masked-LM logits"));
                worstClasses = MathF.Max(worstClasses, Compared(references.Read($"{name}.classes"), inference.Classes, BertTolerance, $"{name}: classification logits"));
                Check(inference.Words.Length == positions.Length * vocabulary, $"{name}: logits over the whole vocabulary");

                // With the gradient recorded (the blocks' unfused path).
                using (new TensorScope())
                {
                    var recorded = BertOutputs(model, c, positions, device);
                    worstHidden = MathF.Max(worstHidden, Compared(hidden, RealPositions(recorded.Hidden, c.Lengths, c.Time, width), BertTolerance, $"{name}: hidden states, gradient recorded"));
                    Compared(references.Read($"{name}.mlm"), recorded.Words, BertTolerance, $"{name}: masked-LM logits, gradient recorded");
                }
            }

            Console.WriteLine($"    largest difference from transformers: hidden {worstHidden:E1}, pooled {worstPooled:E1}, masked-LM {worstWords:E1}, classes {worstClasses:E1}");

            // The same checkpoint read as the other two architectures: the same states and heads.
            var batch = ReadBertCase(references, "batch");
            float[] batchPositions = references.Read("batch.mlm_positions");
            foreach (string architecture in new[] { BertPlugin.BertModel, BertPlugin.BertForSequenceClassification })
            {
                using var other = LoadBert(folder, device, architecture);
                Check(other.Architecture == architecture && other.Heads.Count == 3, $"{architecture}: read with the heads the checkpoint holds ({string.Join(", ", other.Heads.Keys)})");
                using var scope = new TensorScope();
                using var noGrad = Autograd.NoGrad();
                var mine = BertOutputs(model, batch, batchPositions, device);
                var theirs = BertOutputs(other, batch, batchPositions, device);
                BitsEqual(mine.Hidden, theirs.Hidden, $"{architecture}: the states");
                BitsEqual(mine.Classes, theirs.Classes, $"{architecture}: the class logits");
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void BertBatchAndHeads(Device device)
    {
        BertPlugin.Register();
        string folder = BertFolder();
        string untied = BertFolder(config => config["tie_word_embeddings"] = false);
        try
        {
            using var references = SafeTensorsReader.Open(BertData("reference.safetensors"));
            using var model = LoadBert(folder, device);
            Conformance.Check(model.Encoder, new SequenceEncoderSuite(device)).ThrowIfFailed();
            var c = ReadBertCase(references, "batch");
            float[] positions = references.Read("batch.mlm_positions");
            int n = c.Lengths.Length, width = model.Encoder.Width, vocabulary = model.Encoder.Vocabulary, labels = model.Heads[EncoderHeads.SequenceClassification].Outputs;
            using (new TensorScope())
            using (Autograd.NoGrad())
            {
                var batch = BertOutputs(model, c, positions, device);
                for (int b = 0; b < n; b++)
                {
                    int length = c.Lengths[b];
                    var alone = new BertCase($"batch[{b}]", c.Ids.AsSpan(b * c.Time, length).ToArray(), c.Types.AsSpan(b * c.Time, length).ToArray(), [length], length);
                    var own = BertOutputs(model, alone, positions.AsSpan(b * 2, 2).ToArray(), device);
                    AssertClose(batch.Hidden.AsSpan(b * c.Time * width, length * width).ToArray(), own.Hidden, 1e-4f, $"sequence {b}: its states alone and in the batch");
                    AssertClose(batch.Pooled.AsSpan(b * width, width).ToArray(), own.Pooled, 1e-4f, $"sequence {b}: its pooled output alone and in the batch");
                    AssertClose(batch.Words.AsSpan(b * 2 * vocabulary, 2 * vocabulary).ToArray(), own.Words, 1e-4f, $"sequence {b}: its masked-LM logits alone and in the batch");
                    AssertClose(batch.Classes.AsSpan(b * labels, labels).ToArray(), own.Classes, 1e-4f, $"sequence {b}: its class logits alone and in the batch");
                }
            }

            // The decoder is the word embedding, read in place: no second table among the head's parameters.
            var words = ((SequenceEncoder)model.Encoder).Words;
            var head = model.Heads[EncoderHeads.MaskedLanguageModel];
            Check(head.Network.Descendants().OfType<Linear>().Last().TiedTo == words, "the masked-LM decoder is tied to the word embeddings");
            Check(head.Parameters().All(p => p.Size < words.Weight.Size), "the masked-LM head holds no copy of the word embeddings");

            // An untied checkpoint (its own decoder weight, here the embeddings' values) gives the same logits.
            var tensors = new List<(string Name, int[] Shape, float[] Values)>();
            using (var stored = SafeTensorsReader.Open(Path.Combine(untied, "model.safetensors")))
            {
                tensors.AddRange(stored.Tensors.Keys.Order(StringComparer.Ordinal).Select(name => (name, stored.Tensors[name].Shape, stored.Read(name))));
            }

            tensors.Add(("cls.predictions.decoder.weight", [vocabulary, width], tensors.Single(t => t.Name == "bert.embeddings.word_embeddings.weight").Values));
            SafeTensorsWriter.Write(Path.Combine(untied, "model.safetensors"), tensors, SafeTensorType.F32);
            using var separate = LoadBert(untied, device);
            Check(separate.Heads[EncoderHeads.MaskedLanguageModel].Network.Descendants().OfType<Linear>().Last().TiedTo is null, "an untied checkpoint's decoder has weights of its own");
            using (new TensorScope())
            using (Autograd.NoGrad())
            {
                AssertClose(BertOutputs(model, c, positions, device).Words, BertOutputs(separate, c, positions, device).Words, 1e-5f, "the untied decoder against the tied one");
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
            Directory.Delete(untied, recursive: true);
        }
    }

    private static void BertTextEncoder(Device device)
    {
        BertPlugin.Register();
        string folder = BertFolder();
        try
        {
            using var references = SafeTensorsReader.Open(BertData("reference.safetensors"));
            var cases = JsonNode.Parse(File.ReadAllText(BertData("reference.json")))!["cases"]!.AsObject();
            string[] texts = [.. Enumerable.Range(0, 3).Select(i => (string)cases[$"single{i}"]!["text"]!)];
            using var model = LoadBert(folder, device);
            var tokenizer = model.Tokenizer!;
            int pad = tokenizer.IdOf("[PAD]") ?? throw new InvalidOperationException("no [PAD]");
            Check(!new TextEncoder(model.Encoder, tokenizer, 32, pad).SpecialTokens, "SpecialTokens is off by default");
            foreach (string pooler in new[] { SequencePoolers.Mean, SequencePoolers.First })
            {
                var encoder = new TextEncoder(model.Encoder, tokenizer, 32, pad) { Pooler = pooler, SpecialTokens = true };
                var vectors = encoder.Encode(texts);
                for (int i = 0; i < texts.Length; i++)
                {
                    string reference = pooler == SequencePoolers.Mean ? "mean" : "first";
                    AssertClose(references.Read($"single{i}.{reference}"), vectors[i], BertTolerance, $"{pooler}: text {i}'s vector against transformers'");
                    AssertClose(vectors[i], encoder.Encode(texts[i]), 1e-5f, $"{pooler}: text {i} alone and in the batch");
                }
            }

            ThrowsNaming<ArgumentException>(() => _ = new TextEncoder(model.Encoder, new CharTokenizer("abc"), 8, 0) { SpecialTokens = true },
                "SpecialTokens with a tokenizer that has none", "SpecialTokens", "IPretrainedTokenizer");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void BertRegistration(Device device)
    {
        bool wasRegistered = BertPlugin.IsRegistered;
        string folder = BertFolder();
        try
        {
            Check(TextPlugins.All.Contains(BertPlugin.Plugin) && BertPlugin.Plugin.Name == "bert" && BertPlugin.Plugin.GetType().Namespace == "Idrak.Text.Plugins.Bert",
                "BertPlugin.Plugin is \"bert\", of TextPlugins.All, in Idrak.Text.Plugins.Bert");
            BertPlugin.Unregister();
            Check(!BertPlugin.IsRegistered && IdrakPlugins.Find("bert") is null && BertPlugin.Architectures.All(a => PretrainedEncoders.Find(a) is null),
                "unregistered: no BERT architecture is registered");
            var found = PretrainedEncoders.Names.Order(StringComparer.Ordinal).ToList();
            ThrowsNaming<NotSupportedException>(() => LoadBert(folder, device).Dispose(), "a BERT checkpoint without the plug-in", "not registered",
                BertPlugin.BertForMaskedLM, TextPlugins.Package);

            BertPlugin.Register();
            BertPlugin.Register();
            Check(BertPlugin.IsRegistered && ReferenceEquals(IdrakPlugins.Find("bert"), BertPlugin.Plugin), "registered and recorded as \"bert\"");
            Check(PretrainedEncoders.Names.Order(StringComparer.Ordinal).SequenceEqual(found.Concat(BertPlugin.Architectures).Order(StringComparer.Ordinal)),
                $"Register twice registers the three architectures once: {string.Join(", ", PretrainedEncoders.Names)}");
            Check(BertPlugin.Architectures.All(a => PretrainedEncoders.Origin(a) == TextPlugins.Package), "the architectures are Idrak.Text.Plugins'");
            using (var model = LoadBert(folder, device))
            {
                Check(model.Architecture == BertPlugin.BertForMaskedLM, "registered, the checkpoint loads");
            }

            BertPlugin.Unregister();
            Check(PretrainedEncoders.Names.Order(StringComparer.Ordinal).SequenceEqual(found), "Unregister leaves PretrainedEncoders as found");
            TextPlugins.RegisterAll();
            Check(BertPlugin.IsRegistered, "TextPlugins.RegisterAll registers BERT");
        }
        finally
        {
            if (wasRegistered)
            {
                BertPlugin.Register();
            }
            else
            {
                BertPlugin.Unregister();
            }

            Directory.Delete(folder, recursive: true);
        }
    }

    private static void BertLayouts(Device device)
    {
        BertPlugin.Register();
        string folder = BertFolder();
        string old = BertFolder(config => config["architectures"] = new JsonArray(BertPlugin.BertModel));
        var edited = new List<string>();
        try
        {
            // BertModel's layout as older checkpoints store it: no prefix, gamma and beta, no heads but the pooler, and an
            // int64 position_ids buffer beside the weights.
            var tensors = new List<(string Name, int[] Shape, float[] Values)>();
            using (var stored = SafeTensorsReader.Open(Path.Combine(folder, "model.safetensors")))
            {
                foreach (string name in stored.Tensors.Keys.Where(k => k.StartsWith("bert.", StringComparison.Ordinal)).Order(StringComparer.Ordinal))
                {
                    string renamed = name["bert.".Length..].Replace("LayerNorm.weight", "LayerNorm.gamma", StringComparison.Ordinal)
                        .Replace("LayerNorm.bias", "LayerNorm.beta", StringComparison.Ordinal);
                    tensors.Add((renamed, stored.Tensors[name].Shape, stored.Read(name)));
                }
            }

            string oldFile = Path.Combine(old, "model.safetensors");
            WriteSafeTensorsWithInt64(oldFile, tensors, ("embeddings.position_ids", [1, 64], [.. Enumerable.Range(0, 64).Select(i => (long)i)]));
            using (var reader = SafeTensorsReader.Open(oldFile))
            {
                Check(reader.LeftOut.TryGetValue("embeddings.position_ids", out string? type) && type == "I64" && !reader.Contains("embeddings.position_ids"),
                    "the reader leaves the int64 buffer out");
                ThrowsNaming<NotSupportedException>(() => reader.Read("embeddings.position_ids"), "reading the int64 buffer", "I64", "embeddings.position_ids");
            }

            using var references = SafeTensorsReader.Open(BertData("reference.safetensors"));
            var c = ReadBertCase(references, "batch");
            int n = c.Lengths.Length;
            using (var model = LoadBert(folder, device))
            using (var older = LoadBert(old, device))
            using (new TensorScope())
            using (Autograd.NoGrad())
            {
                Check(older.Architecture == BertPlugin.BertModel && older.Heads.Keys.SequenceEqual([EncoderHeads.Pooler]), $"BertModel's layout: the pooler alone ({string.Join(", ", older.Heads.Keys)})");
                BitsEqual(model.Hidden(Tensor.From(c.Ids, [n, c.Time], device), c.Lengths, Tensor.From(c.Types, [n, c.Time], device)).ToArray(),
                    older.Hidden(Tensor.From(c.Ids, [n, c.Time], device), c.Lengths, Tensor.From(c.Types, [n, c.Time], device)).ToArray(), "BertModel's layout: the states");
                BitsEqual(model.Logits(EncoderHeads.Pooler, Tensor.From(c.Ids, [n, c.Time], device), c.Lengths, Tensor.From(c.Types, [n, c.Time], device)).ToArray(),
                    older.Logits(EncoderHeads.Pooler, Tensor.From(c.Ids, [n, c.Time], device), c.Lengths, Tensor.From(c.Types, [n, c.Time], device)).ToArray(), "BertModel's layout: the pooled output");
            }

            ThrowsNaming<InvalidDataException>(() => LoadBert(old, device, BertPlugin.BertForSequenceClassification).Dispose(), "a classifier checkpoint without its classifier",
                "classifier.weight");
            ThrowsNaming<InvalidDataException>(() => LoadBert(old, device, BertPlugin.BertForMaskedLM).Dispose(), "a masked-LM checkpoint without its head",
                "cls.predictions.transform.dense.weight");

            // config.json the family does not read, each refused naming its cause.
            string Edited(Action<JsonObject> edit)
            {
                string path = BertFolder(edit);
                edited.Add(path);
                return path;
            }

            ThrowsNaming<NotSupportedException>(() => LoadBert(Edited(j => j["position_embedding_type"] = "relative_key"), device).Dispose(), "relative positions",
                "position_embedding_type", "relative_key");
            ThrowsNaming<NotSupportedException>(() => LoadBert(Edited(j => j["hidden_act"] = "mish"), device).Dispose(), "an unknown activation", "hidden_act", "mish");
            ThrowsNaming<InvalidDataException>(() => LoadBert(Edited(j => j["intermediate_size"] = 48), device).Dispose(), "a tensor of another shape",
                "encoder.layer.0.intermediate.dense.weight", "configuration");
            ThrowsNaming<InvalidDataException>(() => LoadBert(Edited(j => j["tie_word_embeddings"] = false), device).Dispose(), "an untied model without its decoder",
                "cls.predictions.decoder.weight");

            // The library's head kinds refuse what does not fit.
            ThrowsNaming<ArgumentException>(() => EncoderHeads.Create(EncoderHeads.Pooler, new EncoderHeadRequest(16, 8, [], device)), "a pooler of other outputs than its width", "outputs");
            using var table = new Embedding(10, 8, device);
            ThrowsNaming<ArgumentException>(() => EncoderHeads.Create(EncoderHeads.MaskedLanguageModel, new EncoderHeadRequest(16, 10, [], device) { TiedTo = table }),
                "a tied output of another width", "TiedTo");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
            Directory.Delete(old, recursive: true);
            foreach (string path in edited)
            {
                Directory.Delete(path, recursive: true);
            }
        }
    }

    // A safetensors file of float32 tensors and one int64 tensor (SafeTensorsWriter writes floats only), little-endian.
    private static void WriteSafeTensorsWithInt64(string path, IReadOnlyList<(string Name, int[] Shape, float[] Values)> tensors, (string Name, int[] Shape, long[] Values) extra)
    {
        var header = new JsonObject();
        long offset = 0;
        foreach (var (name, shape, values) in tensors)
        {
            header[name] = new JsonObject { ["dtype"] = "F32", ["shape"] = new JsonArray([.. shape.Select(s => (JsonNode)s)]), ["data_offsets"] = new JsonArray(offset, offset + 4L * values.Length) };
            offset += 4L * values.Length;
        }

        header[extra.Name] = new JsonObject { ["dtype"] = "I64", ["shape"] = new JsonArray([.. extra.Shape.Select(s => (JsonNode)s)]), ["data_offsets"] = new JsonArray(offset, offset + 8L * extra.Values.Length) };
        var text = new StringBuilder(header.ToJsonString());
        while (Encoding.UTF8.GetByteCount(text.ToString()) % 8 != 0)
        {
            text.Append(' ');
        }

        byte[] json = Encoding.UTF8.GetBytes(text.ToString());
        using var stream = File.Create(path);
        var size = new byte[8];
        BinaryPrimitives.WriteInt64LittleEndian(size, json.Length);
        stream.Write(size);
        stream.Write(json);
        foreach (var (_, _, values) in tensors)
        {
            var bytes = new byte[4 * values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(4 * i), values[i]);
            }

            stream.Write(bytes);
        }

        var longs = new byte[8 * extra.Values.Length];
        for (int i = 0; i < extra.Values.Length; i++)
        {
            BinaryPrimitives.WriteInt64LittleEndian(longs.AsSpan(8 * i), extra.Values[i]);
        }

        stream.Write(longs);
    }

    // ---------------------------------------------------------------- the owner's real models

    // Each real/<name>.json written by make_tiny_bert.py --real: the model from the cache (IDRAK_CACHE, else ~/.cache/idrak)
    // against transformers' references. 2e-3 (absolute up to 1, relative above): twelve layers of width 768 and products
    // of 3,072 terms sum in float32 in other orders than PyTorch's; a wrong layer still moves the states by far more.
    private static void BertReal(Device device)
    {
        BertPlugin.Register();
        string real = Path.Combine(Path.GetDirectoryName(BertData("manifest.json"))!, "real");
        string[] records = Directory.Exists(real) ? Directory.GetFiles(real, "*.json") : [];
        Check(records.Length > 0, $"no real references in {real}: run tools/bert/make_tiny_bert.py --real first (the owner's run; it downloads the models)");
        string cache = Environment.GetEnvironmentVariable("IDRAK_CACHE") is { Length: > 0 } root ? root
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "idrak");
        foreach (string file in records.Order(StringComparer.Ordinal))
        {
            var record = JsonNode.Parse(File.ReadAllText(file))!.AsObject();
            string repo = (string)record["repo"]!, folder = Path.Combine(cache, (string)record["folder"]!);
            Check(Directory.Exists(folder), $"{repo}: no model in {folder}: run tools/bert/make_tiny_bert.py --real (with the same IDRAK_CACHE)");
            using var references = SafeTensorsReader.Open(Path.ChangeExtension(file, ".safetensors"));
            using var model = LoadBert(folder, device, (string)record["architecture"]!);
            var tokenizer = model.Tokenizer ?? throw new InvalidOperationException($"{repo}: the folder's tokenizer was not read");
            bool pooler = (bool)record["has_pooler"]!;
            Check(!pooler || model.Heads.ContainsKey(EncoderHeads.Pooler), $"{repo}: the checkpoint's pooler is read");
            var cases = record["cases"]!.AsArray();
            float worst = 0f;
            for (int i = 0; i < cases.Count; i++)
            {
                var c = cases[i]!.AsObject();
                int[] ids = [.. c["ids"]!.AsArray().Select(v => (int)v!)];
                var encoding = tokenizer.Encode((string)c["text"]!, null, new TokenizerEncodeOptions());
                Check(encoding.Ids.SequenceEqual(ids), $"{repo}: text {i}: the tokenizer gives {string.Join(" ", encoding.Ids)}, transformers {string.Join(" ", ids)}");
                using var scope = new TensorScope();
                using var noGrad = Autograd.NoGrad();
                var input = Tensor.From([.. ids.Select(v => (float)v)], [1, ids.Length], device);
                worst = MathF.Max(worst, Compared(references.Read($"text{i}.hidden"), model.Hidden(input).ToArray(), BertRealTolerance, $"{repo}: text {i}: hidden states"));
                if (pooler)
                {
                    Compared(references.Read($"text{i}.pooled"), model.Logits(EncoderHeads.Pooler, input).ToArray(), BertRealTolerance, $"{repo}: text {i}: pooled output");
                }

                float[] logits = model.Logits(EncoderHeads.MaskedLanguageModel, input).ToArray();
                int vocabulary = model.Encoder.Vocabulary;
                float[] values = references.Read($"text{i}.mlm_top_values"), best = references.Read($"text{i}.mlm_top_ids");
                int[] masks = [.. c["mask_positions"]!.AsArray().Select(v => (int)v!)];
                int top = values.Length / masks.Length;
                for (int m = 0; m < masks.Length; m++)
                {
                    int rowStart = masks[m] * vocabulary;
                    float[] mine = [.. Enumerable.Range(0, top).Select(k => logits[rowStart + (int)best[m * top + k]])];
                    var row = logits.AsSpan(rowStart, vocabulary);
                    Compared(values.AsSpan(m * top, top).ToArray(), mine, BertRealTolerance, $"{repo}: text {i}: the masked-LM logits of [MASK] {m} at transformers' best tokens");
                    int argmax = 0;
                    for (int v = 1; v < vocabulary; v++)
                    {
                        argmax = row[v] > row[argmax] ? v : argmax;
                    }

                    Check(argmax == (int)best[m * top], $"{repo}: text {i}: [MASK] {m}'s best token is {argmax}, transformers' {(int)best[m * top]}");
                }
            }

            Console.WriteLine($"    {repo}: largest difference of the hidden states from transformers {worst:E1}");
        }
    }
}
