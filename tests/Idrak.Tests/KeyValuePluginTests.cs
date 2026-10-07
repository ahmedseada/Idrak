// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models.Abstractions;

// Key/value cache formats defined outside the library: registered by name, written with public operations, and attended
// through the composed fallback (expanded to float32, masked products) by decoder and multi-head attention layers.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] KeyValuePluginGroup =
    [
        ("kv plugins: the built-in formats are registered and expand what they wrote to float32; an unknown name lists them; a format of one's own registers, sizes its cache (with scales) and reports the Custom format", KeyValuePluginRegistry),
        ("kv plugins: a decoder and a multi-head attention model decode token by token through a format of one's own like the float32 cache; a prompt at once too", KeyValuePluginDecoding),
        ("kv plugins: TextGenerator with a format of one's own (CacheLayout) generates the float32 cache's greedy text, with the kept cache and in a batch", KeyValuePluginGeneration),
    ];

    private const string KeyValuePluginName = "test-kv-scaled";

    // Each cached row divided by its mean magnitude, the magnitude kept as the row's scale: device work only, from
    // public operations; expanded back by one product.
    private sealed class ScaledLayout : KeyValueLayout
    {
        public int Writes { get; private set; }

        public int Expansions { get; private set; }

        public override string Name => KeyValuePluginName;

        public override int RowWidth(int headDim) => headDim;

        public override bool HasScales => true;

        public override void Write(Tensor keys, Tensor values, KeyValueCache cache, Tensor position)
        {
            Writes++;
            WriteScaled(keys, cache.Keys, cache.KeyScales!, position);
            WriteScaled(values, cache.Values, cache.ValueScales!, position);
        }

        private static void WriteScaled(Tensor source, Tensor rows, Tensor scales, Tensor position)
        {
            int n = source.Shape[0], steps = source.Shape[1], dim = source.Shape[2];
            var scale = source.Abs().Mean(2, keepDim: true) + 1e-6f;              // [n, steps, 1]
            var inverse = (scale.Log() * -1f).Exp().Reshape(n * steps, 1);
            WriteRows(source * Widen(inverse, dim).Reshape(n, steps, dim), rows, position);
            WriteRows(scale, scales.Reshape(n, scales.Shape[1], 1), position);
        }

        // [r, 1] → [r, dim], each row's value repeated along it (a product with a row of ones).
        private static Tensor Widen(Tensor column, int dim)
        {
            using var ones = Tensor.Ones([1, dim], column.Device);
            return column.MatMul(ones);
        }

        public override Tensor Expand(KeyValueCache cache, bool keys)
        {
            Expansions++;
            var (rows, scales) = keys ? (cache.Keys, cache.KeyScales!) : (cache.Values, cache.ValueScales!);
            int n = rows.Shape[0], capacity = rows.Shape[1], dim = rows.Shape[2];
            return rows * Widen(scales.Reshape(n * capacity, 1), dim).Reshape(n, capacity, dim);
        }
    }

    private static void KeyValuePluginRegistry(Device device)
    {
        Check(new[] { "float32", "int8", "bfloat16" }.All(KeyValueLayouts.Names.Contains), $"built-in formats registered: {string.Join(", ", KeyValueLayouts.Names)}");
        try
        {
            KeyValueLayouts.Get("no-such-format");
            Check(false, "an unknown format is refused");
        }
        catch (NotSupportedException e)
        {
            Check(e.Message.Contains("int8", StringComparison.Ordinal) && e.Message.Contains("KeyValueLayouts.Register", StringComparison.Ordinal), $"the error lists the formats: {e.Message}");
        }

        Check(KeyValueLayouts.Get("BFloat16").Format == KeyValueFormat.BFloat16, "names ignore case");

        // Every built-in format expands what it wrote back to float32 (the composed fallback's input).
        var r = new Random(7);
        var keys = Enumerable.Range(0, 3 * 2 * 5).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray();
        foreach (var format in new[] { KeyValueFormat.Float32, KeyValueFormat.Int8, KeyValueFormat.BFloat16 })
        {
            var builtIn = KeyValueLayouts.For(format);
            using var cache = new KeyValueCache(3, 4, 5, device, builtIn);
            using var position = Tensor.Persistent([1f], [1], device, requiresGrad: false);
            using var scope = new TensorScope();
            var source = Tensor.From(keys, [3, 2, 5], device);
            builtIn.Write(source, source * 2f, cache, position);
            var expanded = builtIn.Expand(cache, keys: true).ToArray();
            var expandedValues = builtIn.Expand(cache, keys: false).ToArray();
            Check(expanded.Length == 3 * 4 * 5, $"{format}: expands to [rows, capacity, headDim]");
            for (int row = 0; row < 3; row++)
            {
                AssertClose(keys[(row * 10)..((row + 1) * 10)], expanded[(row * 20 + 5)..(row * 20 + 15)], 0.02f, $"{format}: expanded keys, row {row}");
                AssertClose([.. keys[(row * 10)..((row + 1) * 10)].Select(k => 2 * k)], expandedValues[(row * 20 + 5)..(row * 20 + 15)], 0.04f, $"{format}: expanded values, row {row}");
            }
        }
        var layout = new ScaledLayout();
        KeyValueLayouts.Register(KeyValuePluginName, layout);
        try
        {
            Check(KeyValueLayouts.Names.Contains(KeyValuePluginName) && KeyValueLayouts.Get(KeyValuePluginName) == layout, "a format of one's own registered");
            using var cache = new KeyValueCache(3, 7, 5, device, layout);
            Check(cache.Layout == layout && cache.Format == KeyValueFormat.Custom && cache.Keys.Shape[2] == 5 && cache.KeyScales is { } scales
                  && scales.Shape[0] == 3 && scales.Shape[1] == 7 && cache.Bytes == 4L * (2 * 3 * 7 * 5 + 2 * 3 * 7), "its cache: sized by the layout, with scales, no built-in format");
            using var context = new DecodingContext(device, 1, 8, layout);
            Check(context.Layout == layout && context.Format == KeyValueFormat.Custom, "a context with the layout");
            try
            {
                using var custom = new DecodingContext(device, 1, 8, KeyValueFormat.Custom);
                Check(false, "Custom is not a format to choose");
            }
            catch (ArgumentException)
            {
            }
        }
        finally
        {
            KeyValueLayouts.Unregister(KeyValuePluginName);
        }

        Check(!KeyValueLayouts.Names.Contains(KeyValuePluginName), "the registry restored");
    }

    private static void KeyValuePluginDecoding(Device device)
    {
        var layout = new ScaledLayout();
        float[][] Decode(Sequential model, KeyValueLayout cacheLayout, int[] ids, int vocabulary, bool prompt)
        {
            using var context = new DecodingContext(device, 1, 16, cacheLayout);
            var steps = new List<float[]>();
            using (Autograd.NoGrad())
            {
                if (prompt)
                {
                    using var scope = new TensorScope();
                    var all = model.ForwardCached(Tensor.From([.. ids.Select(i => (float)i)], [1, ids.Length], device), context).ToArray();
                    for (int i = 0; i < ids.Length; i++)
                    {
                        steps.Add(all[(i * vocabulary)..((i + 1) * vocabulary)]);
                    }
                }
                else
                {
                    foreach (int id in ids)
                    {
                        using var scope = new TensorScope();
                        steps.Add(model.ForwardCached(Tensor.From([id], [1, 1], device), context).ToArray()[^vocabulary..]);
                    }
                }
            }

            return [.. steps];
        }

        var float32 = KeyValueLayouts.For(KeyValueFormat.Float32);
        int[] ids = [1, 5, 3, 7, 2, 9, 4, 8, 6, 10];
        using var decoder = SmallSpec.Build(new RandomWeights(41), new DecoderBuildOptions { Device = device });
        var (attentionModel, tokenizer) = TinyLanguageModel(device);
        foreach (var (name, model, vocabulary) in new[] { ("decoder", decoder, SmallSpec.Vocabulary), ("multi-head attention", attentionModel, tokenizer.VocabularySize) })
        {
            foreach (bool prompt in new[] { false, true })
            {
                int writes = layout.Writes, expansions = layout.Expansions;
                var reference = Decode(model, float32, ids, vocabulary, prompt);
                var actual = Decode(model, layout, ids, vocabulary, prompt);
                Check(layout.Writes > writes && layout.Expansions > expansions, $"{name}: written and attended through the layout");
                for (int i = 0; i < ids.Length; i++)
                {
                    AssertClose(reference[i], actual[i], 1e-3f, $"{name}, {(prompt ? "prompt" : "token by token")}, position {i}");
                }
            }
        }
    }

    private static void KeyValuePluginGeneration(Device device)
    {
        var (model, tokenizer) = TinyLanguageModel(device);
        var layout = new ScaledLayout();
        var options = new GenerationOptions { Temperature = 0f, TopK = 1, RepeatPenalty = 1f, NumPredict = 12 };
        string expected = new TextGenerator(model, tokenizer, 32) { KeepCache = false }.Generate("abc", options).Text;
        var generator = new TextGenerator(model, tokenizer, 32) { CacheFormat = KeyValueFormat.Int8, CacheLayout = layout };
        string first = generator.Generate("abc", options).Text;
        Check(first == expected && layout.Writes > 0, $"with the layout '{first}', float32 cache '{expected}'");
        string again = generator.Generate("abc", options).Text;                // the kept cache, its prefix reused
        Check(again == expected, $"again with the kept cache '{again}'");
        generator.ReleaseCache();

        var batch = generator.GenerateBatch(["abc", "hello"], options);
        var single = new TextGenerator(model, tokenizer, 32) { KeepCache = false }.Generate("hello", options).Text;
        Check(batch[0].Text == expected && batch[1].Text == single, $"a batch: '{batch[0].Text}', '{batch[1].Text}' (alone: '{expected}', '{single}')");
    }
}
