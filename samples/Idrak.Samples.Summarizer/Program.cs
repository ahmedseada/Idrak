// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Text summarization: short reports (matches, weather, company results, fires) → one-sentence summaries.
// Two extractive baselines pick a sentence from the report; the abstractive model, a word-level decoder-only
// transformer, writes the summary token by token through the Generation layer (WordTokenizer + TextGenerator).
// Training teaches it only the summary part: "<report> <sum> <summary> <end>", with the loss masked on the report.
//
//   dotnet run -c Release --project samples/Idrak.Samples.Summarizer            (add --cpu / --cuda)
//   dotnet run -c Release --project samples/Idrak.Samples.Summarizer -- --predict --input "the lions played the owls in kelso on friday . the owls scored 2 goals . the lions scored 4 goals ."

using System.Diagnostics;
using Idrak;
using Idrak.Data;
using Idrak.Diagnostics;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Samples;
using Idrak.Samples.Summarizer;
using Idrak.Training;

if (SampleOptions.Parse(args) is not { } options)
{
    return 0;
}

var device = options.Device;
Console.WriteLine($"Device: {device} - {device.Name}\n");
using var logger = Telemetry.Subscribe(new ConsoleLogger(options.LogLevels, epochInterval: 2));

const int Context = 96, Dim = 96, Heads = 4, Layers = 3;
string modelPath = options.ModelPath("summarizer.weights");
string vocabularyPath = Path.ChangeExtension(modelPath, ".vocab.txt");
var greedy = new GenerationOptions
{
    Temperature = 0f, TopK = 1, RepeatPenalty = 1f, NumPredict = 30, Stop = ["<end>"], NumCtx = Context,
};

if (options.PredictOnly)
{
    // Inference mode: load the vocabulary and weights, summarize --input (or a fresh generated report).
    if (!options.RequireModel(modelPath))
    {
        return 1;
    }

    var loadedTokenizer = new WordTokenizer(File.ReadAllLines(vocabularyPath));
    using var loaded = Build(loadedTokenizer.VocabularySize, new Random(2));
    loaded.Load(modelPath);
    loaded.Eval();
    string report = options.Input ?? Reports.Generate(1, new Random(Environment.TickCount))[0].Document;
    var (summary, reason, stats) = new TextGenerator(loaded, loadedTokenizer, Context).Generate(report + " <sum>", greedy);
    Console.WriteLine($"Report:  {report}\nSummary: {summary.Trim()}");
    Console.WriteLine($"         ({stats.GeneratedTokens} tokens, {reason}, {stats.TotalDuration.TotalMilliseconds:F1} ms)");
    return 0;
}

// ---------------------------------------------------------------- data
var train = Reports.Generate(8000, new Random(1));
var test = Reports.Generate(400, new Random(2));
var tokenizer = WordTokenizer.FromTexts(train.SelectMany(r => new[] { r.Document, r.Summary }), ["<pad>", "<sum>", "<end>"]);
Console.WriteLine($"{train.Count} training / {test.Count} test reports, vocabulary {tokenizer.VocabularySize} words");
Console.WriteLine($"Example ({test[0].Kind}):\n  report:  {test[0].Document}\n  summary: {test[0].Summary}\n");

// ---------------------------------------------------------------- extractive baselines
Console.WriteLine("Extractive baselines (pick one sentence of the report):");
var baselines = new (string Name, Func<string, string> Summarize)[]
{
    ("First sentence", doc => Sentences(doc)[0]),
    ("Most central sentence", Central),
};
var rows = new List<(string Name, Scores Scores)>();
foreach (var (name, summarize) in baselines)
{
    var scores = Evaluate(test, summarize);
    rows.Add((name, scores));
    Console.WriteLine($"  {name,-24} {scores}");
}

// ---------------------------------------------------------------- abstractive model
var dataset = Sequences(train, tokenizer, out double summaryTokens);
var (trainSet, validation) = dataset.Split(0.95, seed: 4);
using var model = Build(tokenizer.VocabularySize, new Random(2));
int epochs = options.Epochs ?? 12;
Console.WriteLine($"\nAbstractive transformer: {Layers} layers, dim {Dim}, {model.Parameters().Sum(p => p.Size):N0} parameters, {epochs} epochs");
using var optimizer = new AdamW(model.Parameters(), learningRate: 0.002f, weightDecay: 0.01f);
var trainer = new Trainer(model, optimizer, (logits, next) => SummaryLoss(logits, next, summaryTokens))
{
    Scheduler = new CosineAnnealing(optimizer, epochs, warmupEpochs: 1, minLearningRate: 1e-4f),
    MaxGradientNorm = 1f,
};
var clock = Stopwatch.StartNew();
var history = trainer.Fit(new DataLoader(trainSet, options.BatchSize ?? 32, shuffle: true, device: device, seed: 5), epochs,
    validation: new DataLoader(validation, 200, device: device));
Console.WriteLine($"Trained in {clock.Elapsed.TotalSeconds:F0} s; validation loss per summary token {history.Epochs[^1].ValidationLoss:F4}");
model.Save(modelPath);
File.WriteAllLines(vocabularyPath, tokenizer.Vocabulary);
Console.WriteLine($"Saved {modelPath} and {Path.GetFileName(vocabularyPath)}\n");

model.Eval();
var generator = new TextGenerator(model, tokenizer, Context);
double generatedTokens = 0;
clock.Restart();
var abstractive = Evaluate(test, doc =>
{
    var (text, _, stats) = generator.Generate(doc + " <sum>", greedy);
    generatedTokens += stats.GeneratedTokens;
    return text.Trim();
});
double msPerSummary = clock.Elapsed.TotalMilliseconds / test.Count;
rows.Add(("Abstractive transformer", abstractive));

Console.WriteLine("Summarizer                    ROUGE-1  ROUGE-2  ROUGE-L  exact   numbers right");
foreach (var (name, scores) in rows)
{
    Console.WriteLine($"  {name,-26} {scores.Rouge1,7:F3}  {scores.Rouge2,7:F3}  {scores.RougeL,7:F3}  {scores.Exact,5:P0}  {scores.Numbers,8:P0}");
}

Console.WriteLine($"\nGeneration: {msPerSummary:F1} ms per summary, {generatedTokens / test.Count:F1} tokens each (greedy, KV cache, {device})");
Console.WriteLine("\nPer kind (abstractive, exact match):");
foreach (var kind in test.Select(r => r.Kind).Distinct())
{
    var subset = test.Where(r => r.Kind == kind).ToList();
    Console.WriteLine($"  {kind,-8} {Evaluate(subset, doc => generator.Generate(doc + " <sum>", greedy).Text.Trim()).Exact,5:P0}");
}

Console.WriteLine("\nNew reports:");
foreach (var report in Reports.Generate(4, new Random(7)))
{
    Console.WriteLine($"  report:    {report.Document}\n  reference: {report.Summary}\n  generated: {generator.Generate(report.Document + " <sum>", greedy).Text.Trim()}\n");
}

return abstractive.Rouge1 > 0.9 ? 0 : 1;

// The decoder-only language model (same shape as the GPT samples, but over words).
Sequential Build(int vocabulary, Random random)
{
    var net = new Sequential { new Embedding(vocabulary, Dim, device, random), new PositionalEncoding(Context, Dim, device) };
    for (int layer = 0; layer < Layers; layer++)
    {
        net.Add(new TransformerEncoderLayer(Dim, Heads, ffDim: 4 * Dim, dropout: 0.1f, causal: true, device: device, random: random));
    }

    net.Add(new LayerNorm(Dim, device: device));
    net.Add(new Linear(Dim, vocabulary, device: device, random: random));
    net.Name = "summarizer";
    return net;
}

// Inputs are "<report> <sum> <summary> <end> <pad>..."; the target of each position is the next token, but only
// where that next token belongs to the summary. Every other position gets the id `vocabulary`, which one-hot
// encodes (with vocabulary + 1 classes) into a column that is then cut off: a zero row that adds nothing to the loss.
Dataset Sequences(List<Report> reports, WordTokenizer words, out double averageSummaryTokens)
{
    int ignore = words.VocabularySize;
    var features = new float[reports.Count * Context];
    var targets = new float[reports.Count * Context];
    Array.Fill(targets, ignore);
    long summaryTotal = 0;
    for (int i = 0; i < reports.Count; i++)
    {
        var prompt = words.Encode(reports[i].Document + " <sum>");
        var ids = prompt.Concat(words.Encode(reports[i].Summary + " <end>")).Take(Context + 1).ToList();
        for (int t = 0; t < Context; t++)
        {
            features[i * Context + t] = t < ids.Count ? ids[t] : words["<pad>"];
            if (t + 1 < ids.Count && t + 1 >= prompt.Count)
            {
                targets[i * Context + t] = ids[t + 1];
                summaryTotal++;
            }
        }
    }

    averageSummaryTokens = (double)summaryTotal / reports.Count;
    string[] positions = [.. Enumerable.Range(0, Context).Select(t => $"t{t}")];
    return Dataset.FromFlat(features, targets, reports.Count, positions, positions);
}

// Cross-entropy over summary tokens only, averaged per summary token.
static Tensor SummaryLoss(Tensor logits, Tensor next, double summaryTokensPerRow)
{
    int vocabulary = logits.Shape[^1];
    var targets = Tensor.OneHot(next, vocabulary + 1).Narrow(2, 0, vocabulary);
    return (targets * logits.LogSoftmax()).Sum() * (float)(-1.0 / (next.Shape[0] * summaryTokensPerRow));
}

static List<string> Sentences(string document) =>
    [.. document.Split(" . ", StringSplitOptions.RemoveEmptyEntries).Select(s => s.TrimEnd(' ', '.') + " .")];

// The sentence whose words occur most often in the whole report (a classic frequency-based extractive summary).
static string Central(string document)
{
    var counts = WordTokenizer.Split(document).GroupBy(w => w).ToDictionary(g => g.Key, g => g.Count());
    return Sentences(document).MaxBy(s => WordTokenizer.Split(s).Average(w => (double)counts.GetValueOrDefault(w)))!;
}

static Scores Evaluate(List<Report> reports, Func<string, string> summarize)
{
    double r1 = 0, r2 = 0, rl = 0, exact = 0, numbers = 0;
    foreach (var report in reports)
    {
        var candidate = WordTokenizer.Split(summarize(report.Document)).ToList();
        var reference = WordTokenizer.Split(report.Summary).ToList();
        r1 += Rouge.N(candidate, reference, 1);
        r2 += Rouge.N(candidate, reference, 2);
        rl += Rouge.L(candidate, reference);
        exact += candidate.SequenceEqual(reference) ? 1 : 0;
        var needed = reference.Where(w => char.IsDigit(w[0])).ToList();
        numbers += needed.Count == 0 || needed.SequenceEqual(candidate.Where(w => char.IsDigit(w[0]))) ? 1 : 0;
    }

    int n = reports.Count;
    return new Scores(r1 / n, r2 / n, rl / n, exact / n, numbers / n);
}

/// <summary>Mean F1 of ROUGE-1, ROUGE-2 and ROUGE-L, the share of exact summaries, and the share with every number right and in order.</summary>
internal sealed record Scores(double Rouge1, double Rouge2, double RougeL, double Exact, double Numbers)
{
    public override string ToString() => $"ROUGE-1 {Rouge1:F3}  ROUGE-2 {Rouge2:F3}  ROUGE-L {RougeL:F3}";
}

/// <summary>ROUGE F1 scores: overlap of n-grams (N) and the longest common subsequence (L) between candidate and reference.</summary>
internal static class Rouge
{
    public static double N(IReadOnlyList<string> candidate, IReadOnlyList<string> reference, int n)
    {
        var c = Grams(candidate, n);
        var r = Grams(reference, n);
        int overlap = c.Sum(p => Math.Min(p.Value, r.GetValueOrDefault(p.Key)));
        return F1(overlap, c.Values.Sum(), r.Values.Sum());
    }

    public static double L(IReadOnlyList<string> candidate, IReadOnlyList<string> reference)
    {
        var table = new int[candidate.Count + 1, reference.Count + 1];
        for (int i = 1; i <= candidate.Count; i++)
        {
            for (int j = 1; j <= reference.Count; j++)
            {
                table[i, j] = candidate[i - 1] == reference[j - 1] ? table[i - 1, j - 1] + 1 : Math.Max(table[i - 1, j], table[i, j - 1]);
            }
        }

        return F1(table[candidate.Count, reference.Count], candidate.Count, reference.Count);
    }

    private static Dictionary<string, int> Grams(IReadOnlyList<string> words, int n) =>
        Enumerable.Range(0, Math.Max(0, words.Count - n + 1)).Select(i => string.Join(' ', words.Skip(i).Take(n)))
            .GroupBy(g => g).ToDictionary(g => g.Key, g => g.Count());

    private static double F1(int overlap, int candidateCount, int referenceCount)
    {
        if (overlap == 0)
        {
            return 0;
        }

        double precision = (double)overlap / candidateCount, recall = (double)overlap / referenceCount;
        return 2 * precision * recall / (precision + recall);
    }
}
