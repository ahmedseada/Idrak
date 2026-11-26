// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Layers;
using Idrak.Layers.Abstractions;
using Idrak.Models;
using Idrak.Models.Abstractions;
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
internal abstract partial class Tests
{

    // The owner's real models: more layers and width, so more rounding (see BertReal).
    internal const float BertRealTolerance = 2e-3f;

    internal static readonly (string Name, Action<Device> Run)[] BertGroup =
    [
        ("engine: post-norm block: TransformerEncoderLayer with NormPlacement.After equals its own sublayers composed as LayerNorm(x + Attention(x)), then LayerNorm(x + FeedForward(x)) (tanh GELU fused, and an exact GELU), in inference and with the gradient recorded (every parameter gets a gradient); cached steps equal the whole causal sequence; Before stays the default and the earlier overloads' bits; the layer description and the network builder carry \"norms\": \"after\" both ways; an undefined placement and unknown norms are refused naming the setting", PostNormBlock),
        ("bert real: (the owner's: make_tiny_bert.py --real first) google-bert/bert-base-uncased and aubmindlab/bert-base-arabertv02 from the cache give transformers' ids, hidden states and pooled output within 2e-3 and the masked-LM logits of each [MASK] at transformers' 32 best tokens, the best the same", BertReal),
    ];

    internal static string BertData(string relative) => TestData(Path.Combine("plugins", "bert", relative));

    internal static PretrainedEncoderModel LoadBert(string folder, Device device, string? architecture = null) =>
        PretrainedEncoderModel.Load(folder, new PretrainedEncoderOptions { Device = device, Architecture = architecture });

    // A case of reference.safetensors: its padded ids, type ids, lengths and shape.
    internal sealed record BertCase(string Name, float[] Ids, float[] Types, int[] Lengths, int Time);

    // The largest difference, after AssertClose checked every element.
    internal static float Compared(float[] expected, float[] actual, float tolerance, string what)
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
    internal static (float[] Hidden, float[] Pooled, float[] Words, float[] Classes) BertOutputs(PretrainedEncoderModel model, BertCase c, float[] positions, Device device)
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

    internal static void PostNormBlock(Device device)
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

        // The network builder carries the placement.
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

    // ---------------------------------------------------------------- the owner's real models

    // Each real/<name>.json written by make_tiny_bert.py --real: the model from the cache (IDRAK_CACHE, else ~/.cache/idrak)
    // against transformers' references. 2e-3 (absolute up to 1, relative above): twelve layers of width 768 and products
    // of 3,072 terms sum in float32 in other orders than PyTorch's; a wrong layer still moves the states by far more.
    internal static void BertReal(Device device)
    {
        string real = Path.Combine(Path.GetDirectoryName(BertData("manifest.json"))!, "real");
        string[] records = Directory.Exists(real) ? Directory.GetFiles(real, "*.json") : [];
        if (records.Length == 0)
        {
            Skip($"no real references in {real}: run tools/bert/make_tiny_bert.py --real first (the owner's run; it downloads the models)");
        }

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
