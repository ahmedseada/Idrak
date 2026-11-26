// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models.Abstractions;

// The abstractions core code goes through instead of branching on a device or a format: each is checked against more
// than one implementation.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] AbstractionGroup =
    [
        ("abstractions: a custom ITokenSampler (greedy, on the host, not recordable) plugged into TextGenerator gives the built-in sampler's top-1 text, with and without the KV cache, and its request carries the generator's tokenizer", CustomSampler),
        ("abstractions: Idrak.Abstraction and Idrak name no GPU device, and Idrak.Gpu references Idrak.Abstraction alone", LibraryDevicesFirst),
    ];

    internal static void LibraryDevicesFirst(Device device)
    {
        _ = device;
        var gpu = System.Reflection.Assembly.Load("Idrak.Gpu");
        var references = typeof(Device).Assembly.GetReferencedAssemblies().Select(a => a.Name).ToArray();
        Check(!references.Any(r => r!.StartsWith("Idrak", StringComparison.Ordinal)), $"Idrak.Abstraction references: {string.Join(", ", references)}");
        references = typeof(Sequential).Assembly.GetReferencedAssemblies().Select(a => a.Name).ToArray();
        Check(!references.Contains("Idrak.Gpu"), $"Idrak references: {string.Join(", ", references)}");
        references = gpu.GetReferencedAssemblies().Select(a => a.Name).Where(r => r!.StartsWith("Idrak", StringComparison.Ordinal)).ToArray();
        Check(references.SequenceEqual(["Idrak.Abstraction"]), $"Idrak.Gpu references: {string.Join(", ", references)}");
    }

    // Greedy sampling on the host: downloads the logits each step (so a decoding step using it cannot be recorded).
    internal sealed class HostGreedySampler(Device device, int rows, int vocabulary) : ITokenSampler
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

    internal static void CustomSampler(Device device)
    {
        var (model, tokenizer) = TinyLanguageModel(device);
        foreach (bool cache in new[] { true, false })
        {
            var options = new GenerationOptions { TopK = 1, Temperature = 1f, RepeatPenalty = 1f, NumPredict = 12, UseCache = cache, Seed = 1 };
            var builtIn = new TextGenerator(model, tokenizer, 32) { KeepCache = false };
            string expected = builtIn.Generate("abc", options).Text;

            HostGreedySampler? custom = null;
            ITokenizer? given = null;
            var plugged = new TextGenerator(model, tokenizer, 32)
            {
                KeepCache = false,
                CreateSampler = request => { given = request.Tokenizer; return custom = new HostGreedySampler(request.Device, request.Rows, request.Vocabulary); },
            };
            string actual = plugged.Generate("abc", options).Text;
            Check(custom is { Samples: > 0 }, "the custom sampler chose the tokens");
            Check(ReferenceEquals(given, tokenizer), "the sampler's request carries the generator's tokenizer (SamplerRequest.Tokenizer), for a sampler that reads the tokens' text");
            Check(actual == expected, $"cache {cache}: custom sampler '{actual}', built-in '{expected}'");
        }
    }

}
