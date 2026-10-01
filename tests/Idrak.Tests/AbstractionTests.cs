// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Generation;
using Idrak.Layers;

// The abstractions core code goes through instead of branching on a device or a format: each is checked against more
// than one implementation.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] AbstractionGroup =
    [
        ("abstractions: every KeyValueFormat has a layout that sizes its cache, and decoding through each layout matches the float32 cache (decoder and multi-head attention layers)", KeyValueLayoutsMatch),
        ("abstractions: a custom ITokenSampler (greedy, on the host, not recordable) plugged into TextGenerator gives the built-in sampler's top-1 text, with and without the KV cache", CustomSampler),
    ];

    // Greedy sampling on the host: downloads the logits each step (so a decoding step using it cannot be recorded).
    private sealed class HostGreedySampler(Device device, int rows, int vocabulary) : ITokenSampler
    {
        private readonly List<int[]> _steps = [];

        public int Rows => rows;

        public int Vocabulary => vocabulary;

        public Tensor Ids { get; } = Tensor.Persistent(new float[rows], [rows], device, requiresGrad: false);

        public bool Recordable => false;

        public int Samples { get; private set; }

        public void Sample(Tensor logits)
        {
            var values = logits.ToArray();
            int stride = values.Length / rows;
            var ids = new int[rows];
            for (int r = 0; r < rows; r++)
            {
                var last = values.AsSpan(r * stride + stride - vocabulary, vocabulary);
                for (int i = 1; i < last.Length; i++)
                {
                    if (last[i] > last[ids[r]])
                    {
                        ids[r] = i;
                    }
                }
            }

            Ids.Load([.. ids.Select(i => (float)i)]);
            _steps.Add(ids);
            Samples++;
        }

        public void SetHistory(IReadOnlyList<int> tokens)
        {
        }

        public void Reset() => _steps.Clear();

        public SampledToken[][] Read(int fromStep, int toStep) =>
            [.. _steps.Skip(fromStep).Take(toStep - fromStep).Select(ids => ids.Select(id => new SampledToken(id, 1f, 0f, [(id, 1f), (-1, 0f), (-1, 0f), (-1, 0f), (-1, 0f)])).ToArray())];

        public void Dispose() => Ids.Dispose();
    }

    private static void CustomSampler(Device device)
    {
        var (model, tokenizer) = TinyLanguageModel(device);
        foreach (bool cache in new[] { true, false })
        {
            var options = new GenerationOptions { TopK = 1, Temperature = 1f, RepeatPenalty = 1f, NumPredict = 12, UseCache = cache, Seed = 1 };
            var builtIn = new TextGenerator(model, tokenizer, 32) { KeepCache = false };
            string expected = builtIn.Generate("abc", options).Text;

            HostGreedySampler? custom = null;
            var plugged = new TextGenerator(model, tokenizer, 32)
            {
                KeepCache = false,
                CreateSampler = request => custom = new HostGreedySampler(request.Device, request.Rows, request.Vocabulary),
            };
            string actual = plugged.Generate("abc", options).Text;
            Check(custom is { Samples: > 0 }, "the custom sampler chose the tokens");
            Check(actual == expected, $"cache {cache}: custom sampler '{actual}', built-in '{expected}'");
        }
    }

    private static void KeyValueLayoutsMatch(Device device)
    {
        foreach (var format in Enum.GetValues<KeyValueFormat>())
        {
            var layout = KeyValueLayouts.For(format);
            Check(layout.Format == format, $"{format}: its own layout");
            using var cache = new KeyValueCache(2, 5, 6, device, format);
            Check(cache.Keys.Shape[2] == layout.RowWidth(6) && (cache.KeyScales is not null) == layout.HasScales, $"{format}: cache sized by its layout");
        }

        // A decoder (rotary, grouped heads) and a multi-head attention model, each decoded token by token through every
        // format they support, against the float32 cache.
        var spec = SmallSpec;
        var (attentionModel, tokenizer) = TinyLanguageModel(device);
        float[][] Decode(Sequential model, KeyValueFormat format, int[] ids, int vocabulary)
        {
            using var context = new DecodingContext(device, 1, 16, format);
            var steps = new List<float[]>();
            using (Autograd.NoGrad())
            {
                foreach (int id in ids)
                {
                    using var scope = new TensorScope();
                    steps.Add(model.ForwardCached(Tensor.From([id], [1, 1], device), context).ToArray()[^vocabulary..]);
                }
            }

            return [.. steps];
        }

        using var decoder = spec.Build(new RandomWeights(41), new DecoderBuildOptions { Device = device });
        int[] ids = [1, 5, 3, 7, 2, 9, 4];
        var reference = Decode(decoder, KeyValueFormat.Float32, ids, spec.Vocabulary);
        foreach (var format in new[] { KeyValueFormat.Int8, KeyValueFormat.BFloat16 })
        {
            var actual = Decode(decoder, format, ids, spec.Vocabulary);
            for (int i = 0; i < ids.Length; i++)
            {
                AssertClose(reference[i], actual[i], 0.05f, $"decoder, {format} cache, step {i}");
            }
        }

        var mhaReference = Decode(attentionModel, KeyValueFormat.Float32, ids, tokenizer.VocabularySize);
        var mhaInt8 = Decode(attentionModel, KeyValueFormat.Int8, ids, tokenizer.VocabularySize);
        for (int i = 0; i < ids.Length; i++)
        {
            AssertClose(mhaReference[i], mhaInt8[i], 0.05f, $"multi-head attention, int8 cache, step {i}");
        }

        try
        {
            Decode(attentionModel, KeyValueFormat.BFloat16, ids, tokenizer.VocabularySize);
            Check(false, "multi-head attention refuses a bfloat16 cache");
        }
        catch (NotSupportedException)
        {
        }
    }
}
