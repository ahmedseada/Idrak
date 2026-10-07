// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// An app at the end of the override loop (plan 10, "Promotion"). It once shipped its own token sampler, because the
// library's failed on logits that were not finite (a model that overflows now and then). The app's sampler proved the
// fix with the testing kit; the library then made the fix general (one rule for NaN and ±∞ on every device) and shipped
// it as version 2 of its default sampler. The app deleted its override: it runs on the library's default, and its tests
// (samples/Idrak.Samples.Override.Tests) check that default with the kit. Version 1 stays reachable for one more release,
// a one-line way back if the new default ever misbehaves for this app:
//
//   TokenSamplers.Register(TokenSamplers.DefaultName, TokenSamplers.Default(TokenSamplers.DefaultName, version: 1)!);
//
//   dotnet run -c Release --project samples/Idrak.Samples.Override
//   dotnet run -c Release --project samples/Idrak.Samples.Override.Tests      the kit, run from the app's tests

using Idrak.Generation;
using Idrak.Generation.Abstractions;
using Idrak.Layers;

var device = Device.Cpu;
Console.WriteLine($"token sampler: the library's default, version {TokenSamplers.DefaultVersion(TokenSamplers.DefaultName)}; "
                  + $"overrides: {(Overrides.Report().Count == 0 ? "none" : string.Join(", ", Overrides.Report()))}");

// The step that once made the app write its own sampler: token 3 is the best, but token 1 overflowed to NaN.
float[] logits = [0.5f, float.NaN, 1f, 4f, 2f];
var request = new SamplerRequest(device, Rows: 1, Vocabulary: logits.Length, MaxSteps: 1, HistoryCapacity: 1, new GenerationOptions { TopK = 1, Seed = 1 });
using (var sampler = TokenSamplers.Create(request))
using (var step = Tensor.From(logits, [1, logits.Length], device))
{
    sampler.Sample(step);
    Console.WriteLine($"greedy on [{string.Join(", ", logits)}]: token {sampler.Ids.ToArray()[0]}");
}

// A generation: a small untrained character model, its text sampled by the library's default.
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
var generator = new TextGenerator(model, tokenizer, 32);
var text = generator.Generate("the ", new GenerationOptions { TopK = 1, NumPredict = 12, Seed = 1 }).Text;
Console.WriteLine($"generated: \"the {text}\"");
