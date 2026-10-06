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
        ("abstractions: every KeyValueFormat has a layout that sizes its cache, and decoding through each layout matches the float32 cache (decoder and multi-head attention layers, bfloat16 expanded for the latter)", KeyValueLayoutsMatch),
        ("abstractions: every PackedFormat behind PackedWeight multiplies, expands, lists its buffers and builds layers like its own class (int8, 4-bit, bfloat16), one row and many", PackedWeightsMatch),
        ("abstractions: a custom ITokenSampler (greedy, on the host, not recordable) plugged into TextGenerator gives the built-in sampler's top-1 text, with and without the KV cache", CustomSampler),
        ("abstractions: Idrak.Abstraction names no GPU device, and Idrak's (cuda, vulkan, hip) are registered first, in that order", LibraryDevicesFirst),
    ];

    private static void LibraryDevicesFirst(Device device)
    {
        _ = device;
        string[] kinds = [.. DeviceProviders.All.Select(p => p.Kind)];
        Check(kinds.Length >= 3 && kinds[..3].SequenceEqual(["cuda", "vulkan", "hip"]), $"providers: {string.Join(", ", kinds)}");
        Check(DeviceProviders.All.Take(3).All(p => p.GetType().Assembly == typeof(Tensor).Assembly), "the GPU providers come from Idrak");
        var references = typeof(Device).Assembly.GetReferencedAssemblies().Select(a => a.Name).ToArray();
        Check(!references.Contains("Idrak"), $"Idrak.Abstraction references: {string.Join(", ", references)}");
    }

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
        foreach (var format in Enum.GetValues<KeyValueFormat>().Where(f => f != KeyValueFormat.Custom))
        {
            var layout = KeyValueLayouts.For(format);
            Check(layout.Format == format && KeyValueLayouts.Get(format.ToString()) == layout, $"{format}: its own layout, registered by name");
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
        using (var context = new DecodingContext(device, 1, 16, KeyValueFormat.Int8))
        using (Autograd.NoGrad())
        using (var scope = new TensorScope())
        {
            var attention = decoder.Descendants().OfType<CausalSelfAttention>().First();
            Check(context[attention] is null, "no cache before the first step");
            decoder.ForwardCached(Tensor.From([1f], [1, 1], device), context);
            Check(context[attention] is { Format: KeyValueFormat.Int8 } layerCache && layerCache.Bytes > 0, "the layer's cache through the indexer");
        }

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
        foreach (var format in new[] { KeyValueFormat.Int8, KeyValueFormat.BFloat16 })
        {
            var actual = Decode(attentionModel, format, ids, tokenizer.VocabularySize);
            for (int i = 0; i < ids.Length; i++)
            {
                AssertClose(mhaReference[i], actual[i], 0.05f, $"multi-head attention, {format} cache, step {i}");
            }
        }
    }

    private static void PackedWeightsMatch(Device device)
    {
        var r = new Random(51);
        const int Rows = 64, Columns = 96;
        var values = Enumerable.Range(0, Rows * Columns).Select(_ => (float)(r.NextDouble() * 2 - 1) * 0.1f).ToArray();
        var inputs = Enumerable.Range(0, 70 * Rows).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray();
        foreach (var format in Enum.GetValues<PackedFormat>())
        {
            using var linear = Linear.FromPacked(PackedWeight.FromValues(format, values, Rows, Columns, device));
            var weight = linear.PackedWeight!;
            Check(weight.Format == format && weight.Rows == Rows && weight.Columns == Columns, $"{format}: format and shape");
            Check((format == PackedFormat.Int8) == (linear.Int8 is not null) && (format == PackedFormat.Int4) == (linear.Int4 is not null)
                  && (format == PackedFormat.BFloat16) == (linear.BFloat16 is not null), $"{format}: the typed view matches");
            Check(linear.Buffers().Count() == (format == PackedFormat.BFloat16 ? 1 : 2), $"{format}: packed words (and scales) as buffers");
            using var expanded = weight.Dequantize();
            foreach (int rows in new[] { 1, 70 })
            {
                using var scope = new TensorScope();
                var x = Tensor.From(inputs.AsSpan(0, rows * Rows), [rows, Rows], device);
                AssertClose(x.MatMul(expanded).ToArray(), linear.Forward(x).ToArray(), 2e-2f, $"{format}, {rows} rows: the packed product equals the expanded one");
            }
        }
    }
}
