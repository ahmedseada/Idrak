// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Quantization: what smaller weights cost in accuracy and gain in size and speed.
//   1. A word-level summarizer (the Summarizer sample's model and data, trained here) is compared as float32, int8
//      (QuantizeInt8: one byte per weight, one scale per output column), and loaded from Float16 / BFloat16 files:
//      exact summaries, next-token agreement and the largest logit change against float32, and file sizes.
//   2. A GPT with about 100 million parameters (random weights: speed does not depend on what was learned) generates
//      token by token as float32 and as int8. Each new token reads every weight once, so decoding is limited by memory
//      bandwidth and int8 weights (a quarter of the bytes) are faster where bandwidth is the bottleneck.
//
//   dotnet run -c Release --project samples/Idrak.Samples.Quantization            (add --cpu / --cuda)
//   dotnet run -c Release --project samples/Idrak.Samples.Quantization -- --size small     (a 25M-parameter GPT for part 2)
//   dotnet run -c Release --project samples/Idrak.Samples.Quantization -- --part speed     (only part 2)

using System.Diagnostics;
using Idrak.Data;
using Idrak.Diagnostics;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models.Abstractions;
using Idrak.Optimizers;
using Idrak.Samples.Summarizer;
using Idrak.Samples;
using Idrak.Training;
using Idrak;

if (SampleOptions.Parse(args, ("size", "GPT for the speed test: base (≈100M parameters, default) or small (≈25M)"),
    ("part", "accuracy, speed or all (default)")) is not { } options)
{
    return 0;
}

var device = options.Device;
Console.WriteLine($"Device: {device} - {device.Name}\n");
using var logger = Telemetry.Subscribe(new ConsoleLogger(options.LogLevels, epochInterval: 2));
string folder = Path.Combine(Path.GetTempPath(), $"idrak-quantization-{Environment.ProcessId}");
Directory.CreateDirectory(folder);

const int Context = 96, Dim = 96;
var train = Reports.Generate(5000, new Random(1));
var test = Reports.Generate(300, new Random(2));
var tokenizer = WordTokenizer.FromTexts(train.SelectMany(r => new[] { r.Document, r.Summary }), ["<pad>", "<sum>", "<end>"]);
var greedy = new GenerationOptions { Temperature = 0f, TopK = 1, RepeatPenalty = 1f, NumPredict = 30, Stop = ["<end>"], NumCtx = Context };
string part = options.Get("part", "all")!;
if (part is "all" or "accuracy")
{
    Accuracy();
}

if (part is "all" or "speed")
{
    Speed();
}

Directory.Delete(folder, recursive: true);
return 0;

// ---------------------------------------------------------------- 1. accuracy: a trained summarizer
void Accuracy()
{
var builder = Architectures.Gpt(tokenizer.VocabularySize, Context, Dim, heads: 4, layers: 3, ffDim: 4 * Dim, dropout: 0.1f).OnDevice(device).Seed(2);
using var model = builder.Build();
int epochs = options.Epochs ?? 7;
Console.WriteLine($"Summarizer: {model.Parameters().Sum(p => p.Size):N0} parameters, {train.Count} reports, {epochs} epochs");
var clock = Stopwatch.StartNew();
var run = new TrainingRun
{
    Model = model,
    Loss = (logits, next) => SummaryLoss(logits, next),
    Optimizer = p => new AdamW(p, learningRate: 0.002f, weightDecay: 0.01f),
    Scheduler = o => new CosineAnnealing(o, epochs, minLearningRate: 1e-4f, warmupEpochs: 1),
    Train = new DataLoader(Sequences(train), options.BatchSize ?? 32, shuffle: true, device: device, seed: 5),
    Epochs = epochs,
    MaxGradientNorm = 1f,
};
run.Fit();
model.Eval();
Console.WriteLine($"Trained in {clock.Elapsed.TotalSeconds:F0} s\n");

string floatPath = Path.Combine(folder, "summarizer.f32.ikw");
model.Save(floatPath);
var reference = Summaries(model, KeyValueFormat.Float32);
var referenceLogits = Logits(model);

var rows = new List<(string Name, long Bytes, double Exact, double SameSummary, double SameNextToken, float LogitChange)>();
void Measure(string name, Sequential candidate, string path, KeyValueFormat cache = KeyValueFormat.Float32)
{
    var summaries = Summaries(candidate, cache);
    var logits = Logits(candidate);
    double exact = test.Select((r, i) => WordTokenizer.Split(summaries[i]).SequenceEqual(WordTokenizer.Split(r.Summary)) ? 1.0 : 0.0).Average();
    double same = summaries.Select((s, i) => s == reference[i] ? 1.0 : 0.0).Average();
    int vocabulary = tokenizer.VocabularySize, positions = referenceLogits.Length / vocabulary, agree = 0;
    float change = 0;
    for (int p = 0; p < positions; p++)
    {
        var a = referenceLogits.AsSpan(p * vocabulary, vocabulary);
        var b = logits.AsSpan(p * vocabulary, vocabulary);
        agree += a.IndexOf(Max(a)) == b.IndexOf(Max(b)) ? 1 : 0;
        for (int v = 0; v < vocabulary; v++)
        {
            change = MathF.Max(change, MathF.Abs(a[v] - b[v]));
        }
    }

    rows.Add((name, new FileInfo(path).Length, exact, same, (double)agree / positions, change));
}

Measure("float32", model, floatPath);
foreach (var format in new[] { WeightFormat.Float16, WeightFormat.BFloat16 })
{
    string path = Path.Combine(folder, $"summarizer.{format}.ikw");
    model.Save(path, format);
    using var loaded = builder.Build();
    loaded.Load(path);
    loaded.Eval();
    Measure($"{format} file", loaded, path);
}

using (var quantized = builder.Build())
{
    quantized.Load(floatPath);
    quantized.Eval();
    quantized.QuantizeInt8();
    string path = Path.Combine(folder, "summarizer.int8.ikw");
    quantized.Save(path);
    Measure("int8 (QuantizeInt8)", quantized, path);
    Measure("int8 + int8 KV cache", quantized, path, KeyValueFormat.Int8);
}

Console.WriteLine($"Accuracy on {test.Count} new reports      file        exact   same summary   same next token   largest logit change");
foreach (var r in rows)
{
    Console.WriteLine($"  {r.Name,-24} {r.Bytes / 1024.0,8:N0} KB   {r.Exact,6:P1}   {r.SameSummary,12:P1}   {r.SameNextToken,15:P2}   {r.LogitChange,20:F4}");
}

Console.WriteLine("  (same summary / next token: agreement with the float32 model; logits span about " +
    $"{referenceLogits.Min():F0} to {referenceLogits.Max():F0})\n");

}

// ---------------------------------------------------------------- 2. speed and memory: a larger GPT
void Speed()
{
bool small = options.Get("size", "base") == "small";
int dim = small ? 512 : 768, layers = small ? 8 : 12, vocabularySize = 8192, context = 256;
var words = new WordTokenizer(Enumerable.Range(0, vocabularySize).Select(i => $"w{i}"));
string prompt = string.Join(' ', Enumerable.Range(1, 16).Select(i => $"w{i * 7}"));
var decode = new GenerationOptions { Temperature = 0f, TopK = 1, RepeatPenalty = 1f, NumPredict = 64, NumCtx = context };
Console.WriteLine($"Decoding speed: GPT with dim {dim}, {layers} layers, vocabulary {vocabularySize} (random weights), " +
    $"prompt of 16 tokens, 64 new tokens, greedy, KV cache, {device}");
using var gpt = Architectures.Gpt(words.VocabularySize, context, dim, heads: dim / 64, layers, ffDim: 4 * dim, dropout: 0f).OnDevice(device).Seed(3).Build();
gpt.Eval();
long parameters = gpt.Parameters().Sum(p => (long)p.Size);
foreach (var (name, int8, cache) in new[] { ("float32", false, KeyValueFormat.Float32), ("int8", true, KeyValueFormat.Float32), ("int8 + KV", true, KeyValueFormat.Int8) })
{
    if (int8)
    {
        gpt.QuantizeInt8();
    }

    long bytes = gpt.Parameters().Concat(gpt.Buffers()).Sum(p => 4L * p.Size);
    long cacheBytes;
    using (var probe = new DecodingContext(device, 1, context, cache))
    using (Autograd.NoGrad())
    using (var scope = new TensorScope())
    {
        gpt.ForwardCached(Tensor.From([1f], [1, 1], device), probe);            // creates every layer's cache
        cacheBytes = probe.CacheBytes;
    }

    var generator = new TextGenerator(gpt, words, context) { CacheFormat = cache };
    generator.Generate(prompt, decode with { NumPredict = 8 });                        // warm-up (kernel loading, caches)
    var results = Enumerable.Range(0, 3).Select(_ => generator.Generate(prompt, decode)).ToList();
    var best = results.MaxBy(r => r.Stats.TokensPerSecond)!;
    Console.WriteLine($"  {name,-10} weights {bytes / 1048576.0,5:N0} MB, KV cache {cacheBytes / 1048576.0,5:F1} MB ({context} positions): " +
        $"{best.Stats.TokensPerSecond,6:F1} tokens/s ({1000 / best.Stats.TokensPerSecond,5:F1} ms per token), prompt {best.Stats.PromptDuration.TotalMilliseconds,5:F0} ms");
}

Console.WriteLine($"  ({parameters / 1e6:F1}M parameters; the int8 KV cache stores each head's {dim / (dim / 64)} values as bytes plus one scale)");
}

// ---------------------------------------------------------------- helpers
List<string> Summaries(Sequential m, KeyValueFormat cache)
{
    var generator = new TextGenerator(m, tokenizer, Context) { CacheFormat = cache };
    return [.. test.Select(r => generator.Generate(r.Document + " <sum>", greedy).Text.Trim())];
}

// Next-token logits at every position of every test report followed by its reference summary.
float[] Logits(Sequential m)
{
    var rowsOfIds = test.Select(r => tokenizer.Encode(r.Document + " <sum> " + r.Summary).Take(Context).ToArray()).ToList();
    var ids = new float[test.Count * Context];
    for (int i = 0; i < test.Count; i++)
    {
        for (int t = 0; t < Context; t++)
        {
            ids[i * Context + t] = t < rowsOfIds[i].Length ? rowsOfIds[i][t] : tokenizer["<pad>"];
        }
    }

    using var input = Tensor.From(ids, [test.Count, Context], device);
    using var output = m.Predict(input);
    var all = output.ToArray();
    int v = tokenizer.VocabularySize;
    return [.. rowsOfIds.SelectMany((r, i) => all.AsSpan(i * Context * v, r.Length * v).ToArray())];
}

static float Max(ReadOnlySpan<float> values)
{
    float max = float.NegativeInfinity;
    foreach (float v in values)
    {
        max = MathF.Max(max, v);
    }

    return max;
}

// "<report> <sum> <summary> <end>": next-token targets on the summary only (as in the Summarizer sample).
Dataset Sequences(List<Report> reports)
{
    int ignore = tokenizer.VocabularySize;
    var features = new float[reports.Count * Context];
    var targets = new float[reports.Count * Context];
    Array.Fill(targets, ignore);
    for (int i = 0; i < reports.Count; i++)
    {
        var promptIds = tokenizer.Encode(reports[i].Document + " <sum>");
        var ids = promptIds.Concat(tokenizer.Encode(reports[i].Summary + " <end>")).Take(Context + 1).ToList();
        for (int t = 0; t < Context; t++)
        {
            features[i * Context + t] = t < ids.Count ? ids[t] : tokenizer["<pad>"];
            if (t + 1 < ids.Count && t + 1 >= promptIds.Count)
            {
                targets[i * Context + t] = ids[t + 1];
            }
        }
    }

    string[] positions = [.. Enumerable.Range(0, Context).Select(t => $"t{t}")];
    return Dataset.FromFlat(features, targets, reports.Count, positions, positions);
}

static Tensor SummaryLoss(Tensor logits, Tensor next)
{
    int vocabulary = logits.Shape[^1];
    var targets = Tensor.OneHot(next, vocabulary + 1).Narrow(2, 0, vocabulary);
    return (targets * logits.LogSoftmax()).Sum() * (-1f / next.Shape[0] / 12f);
}
