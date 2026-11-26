// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models.Abstractions;

// Key/value cache formats defined outside the library: written with public operations, and attended
// through the composed fallback (expanded to float32, masked products) by decoder and multi-head attention layers.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] KeyValuePluginGroup =
    [
        ("kv plugins: a decoder and a multi-head attention model decode token by token through a format of one's own like the float32 cache; a prompt at once too", KeyValuePluginDecoding),
    ];

    internal const string KeyValuePluginName = "test-kv-scaled";

    // Each cached row divided by its mean magnitude, the magnitude kept as the row's scale: device work only, from
    // public operations; expanded back by one product.
    internal sealed class ScaledLayout : KeyValueLayout, IScaledLayout
    {
        public int Writes { get; private set; }

        public int Expansions { get; private set; }

        public override string Name => KeyValuePluginName;

        public override int RowWidth(int headDim) => headDim;

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

    internal static void KeyValuePluginDecoding(Device device)
    {
        var layout = new ScaledLayout();
        float[][] Decode(Sequential model, IKeyValueLayout cacheLayout, int[] ids, int vocabulary, bool prompt)
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

}
