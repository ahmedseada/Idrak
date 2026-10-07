// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// An app that overrides one contract of the library: its own token sampler (SanitizingSampler), which fixes the one
// case the app hit (logits that are not finite) and leaves the rest to the library's sampler. The override goes where
// generation asks for a sampler (TextGenerator.CreateSampler); the app's own tests
// (samples/Idrak.Samples.Override.Tests) check it with the testing kit, Idrak.Abstraction.Testing.
//
//   dotnet run -c Release --project samples/Idrak.Samples.Override
//   dotnet run -c Release --project samples/Idrak.Samples.Override.Tests      the kit, run from the app's tests

using Idrak.Generation;
using Idrak.Generation.Abstractions;
using Idrak.Layers;
using Idrak.Samples.Override;

var device = Device.Cpu;

// One step whose logits went wrong: token 3 is the best, but token 1 overflowed to NaN.
float[] logits = [0.5f, float.NaN, 1f, 4f, 2f];
var request = new SamplerRequest(device, Rows: 1, Vocabulary: logits.Length, MaxSteps: 1, HistoryCapacity: 1, new GenerationOptions { TopK = 1, Seed = 1 });
foreach (var (name, create) in new (string, Func<SamplerRequest, ITokenSampler>)[] { ("library", TokenSampler.Create), ("app", SanitizingSampler.Create) })
{
    using var sampler = create(request);
    using var step = Tensor.From(logits, [1, logits.Length], device);
    try
    {
        sampler.Sample(step);
        Console.WriteLine($"{name,-8} sampler, greedy on [{string.Join(", ", logits)}]: token {sampler.Ids.ToArray()[0]}");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"{name,-8} sampler, greedy on [{string.Join(", ", logits)}]: {ex.GetType().Name} ({ex.Message})");
    }
}

// The override in a generation: a small untrained character model, its text sampled by the app's sampler.
var tokenizer = new CharTokenizer("abcdefghijklmnopqrstuvwxyz .");
var random = new Random(4);
using var model = new Sequential
{
    new Embedding(tokenizer.VocabularySize, 16, device, random),
    new PositionalEncoding(32, 16, device),
    new TransformerEncoderLayer(16, 2, dropout: 0f, causal: true, device: device, random: random),
    new LayerNorm(16, device: device),
    new Linear(16, tokenizer.VocabularySize, device: device, random: random),
};
var generator = new TextGenerator(model, tokenizer, 32) { CreateSampler = SanitizingSampler.Create };
var text = generator.Generate("the ", new GenerationOptions { TopK = 1, NumPredict = 12, Seed = 1 }).Text;
Console.WriteLine($"generated with the app's sampler: \"the {text}\"");
