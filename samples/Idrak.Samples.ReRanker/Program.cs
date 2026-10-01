// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Search re-ranking: a fast word-matching first stage (BM25) finds 20 candidate passages for a question, then a
// cross-encoder transformer reads the question and each candidate together and re-orders them by relevance.
// The model is trained listwise: one answer and 7 hard negatives per question, softmax cross-entropy over the 8 scores.
// Questions about 40 towns the model never saw in training measure whether it learned matching rather than memorizing;
// words too rare to learn (town names) become placeholders <w0>, <w1>, ... so matching them transfers to new names.
//
//   dotnet run -c Release --project samples/Idrak.Samples.ReRanker            (add --cpu / --cuda)
//   dotnet run -c Release --project samples/Idrak.Samples.ReRanker -- --predict --input "how many people live in armor"
//   dotnet run -c Release --project samples/Idrak.Samples.ReRanker -- --predict --input "is it cold?|winters are icy and long .|the shop opens at nine ."
//        (with '|': re-rank your own passages for the question before the first '|')

using System.Diagnostics;
using Idrak;
using Idrak.Data;
using Idrak.Diagnostics;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Samples;
using Idrak.Samples.ReRanker;
using Idrak.Training;

if (SampleOptions.Parse(args) is not { } options)
{
    return 0;
}

var device = options.Device;
Console.WriteLine($"Device: {device} - {device.Name}\n");
using var logger = Telemetry.Subscribe(new ConsoleLogger(options.LogLevels, epochInterval: 2));

const int Length = 28, Dim = 64, Placeholders = 8, Group = 8, Candidates = 20, TrainTowns = 120, Towns = 160;
string modelPath = options.ModelPath("reranker.weights");
string vocabularyPath = Path.ChangeExtension(modelPath, ".vocab.txt");
var collection = new Collection(Towns, seed: 1);
var bm25 = new Bm25(collection.Passages.Select(p => p.Text));

if (options.PredictOnly)
{
    // Inference mode: re-rank the collection's BM25 candidates for --input, or your own passages ("question|p1|p2...").
    if (!options.RequireModel(modelPath))
    {
        return 1;
    }

    var loadedTokenizer = new WordTokenizer(File.ReadAllLines(vocabularyPath));
    using var loaded = Scorer(loadedTokenizer.VocabularySize, new Random(2));
    loaded.Load(modelPath);
    loaded.Eval();
    var parts = (options.Input ?? "how many people live in " + collection.Towns[^1]).Split('|', StringSplitOptions.TrimEntries);
    string question = parts[0];
    string[] passages = parts.Length > 1 ? parts[1..] : [.. bm25.Search(question, Candidates).Select(id => collection.Passages[id].Text)];
    var scores = Score(loaded, loadedTokenizer, question, passages);
    Console.WriteLine($"Question: {question}\n\n  first stage (BM25 order)                          re-ranked (score)");
    var order = Enumerable.Range(0, passages.Length).OrderByDescending(i => scores[i]).ToArray();
    for (int i = 0; i < Math.Min(5, passages.Length); i++)
    {
        Console.WriteLine($"  {i + 1}. {Clip(passages[i]),-46}  {Clip(passages[order[i]]),-46} ({scores[order[i]]:F2})");
    }

    return 0;
}

// ---------------------------------------------------------------- data
var trainQueries = collection.QueriesFor(Enumerable.Range(0, TrainTowns));
var testQueries = collection.QueriesFor(Enumerable.Range(TrainTowns, Towns - TrainTowns));
// Vocabulary: words used for at least two training towns. Words specific to one town (its name) cannot be learned,
// so they are encoded as placeholders <w0>, <w1>, ... numbered by first appearance in each (question, passage) pair.
var townsPerWord = collection.Passages.Where(p => p.Entity < TrainTowns).Select(p => (p.Entity, p.Text))
    .Concat(trainQueries.Select(q => (q.Entity, q.Text)))
    .SelectMany(p => WordTokenizer.Split(p.Text).Select(w => (w, p.Entity))).Distinct()
    .GroupBy(p => p.w).ToDictionary(g => g.Key, g => g.Count());
var tokenizer = new WordTokenizer([.. Specials(), .. townsPerWord.Where(p => p.Value >= 2).OrderByDescending(p => p.Value).ThenBy(p => p.Key).Select(p => p.Key)]);
Console.WriteLine($"{collection.Passages.Count} passages about {Towns} towns; {trainQueries.Count} training questions ({TrainTowns} towns), " +
    $"{testQueries.Count} test questions ({Towns - TrainTowns} unseen towns); vocabulary {tokenizer.VocabularySize} (+ placeholders for town names)");
var sample = testQueries[0];
Console.WriteLine($"Example: \"{sample.Text}\" -> \"{collection.Passages[sample.Answer].Text}\"\n");

// ---------------------------------------------------------------- first stage alone
var firstStage = testQueries.Select(q => bm25.Search(q.Text, Candidates)).ToList();
var bm25Scores = Measure(testQueries, firstStage);
Console.WriteLine($"BM25 alone:            {bm25Scores}");

// ---------------------------------------------------------------- listwise training
var random = new Random(3);
var groups = Groups(trainQueries, random);
var (trainSet, validation) = groups.Split(0.95, seed: 4);
var scorer = Scorer(tokenizer.VocabularySize, new Random(2));                  // disposed with groupModel
// The group model scores all 8 pairs of a row at once: [B, 8·L] → [B·8, L] → scorer → [B·8, 1] → [B, 8] logits.
using var groupModel = new Sequential
{
    new Lambda(x => x.Reshape(-1, Length), "PairsOfGroup"),
    scorer,
    new Lambda(x => x.Reshape(-1, Group), "ScoresOfGroup"),
};
int epochs = options.Epochs ?? 12;
Console.WriteLine($"\nCross-encoder: 2 layers, dim {Dim}, {scorer.Parameters().Sum(p => p.Size):N0} parameters; groups of {Group} (1 answer + {Group - 1} hard negatives), {epochs} epochs");
using var optimizer = new AdamW(groupModel.Parameters(), learningRate: 0.002f, weightDecay: 0.01f);
var trainer = new Trainer(groupModel, optimizer, (logits, targets) => Losses.CrossEntropy(logits, targets))
{
    Metrics = { Metric.Accuracy },
    Scheduler = new CosineAnnealing(optimizer, epochs, minLearningRate: 1e-4f, warmupEpochs: 1),
    MaxGradientNorm = 1f,
};
var clock = Stopwatch.StartNew();
trainer.Fit(new DataLoader(trainSet, options.BatchSize ?? 32, shuffle: true, device: device, seed: 5), epochs,
    validation: new DataLoader(validation, 200, device: device));
Console.WriteLine($"Trained in {clock.Elapsed.TotalSeconds:F0} s");
scorer.Save(modelPath);
File.WriteAllLines(vocabularyPath, tokenizer.Vocabulary);
Console.WriteLine($"Saved {modelPath} and {Path.GetFileName(vocabularyPath)}\n");

// ---------------------------------------------------------------- re-rank the first stage's candidates
scorer.Eval();
clock.Restart();
var reranked = testQueries.Select((q, i) =>
{
    var scores = Score(scorer, tokenizer, q.Text, [.. firstStage[i].Select(id => collection.Passages[id].Text)]);
    return firstStage[i].Select((id, j) => (id, s: scores[j])).OrderByDescending(p => p.s).Select(p => p.id).ToArray();
}).ToList();
double msPerQuery = clock.Elapsed.TotalMilliseconds / testQueries.Count;
var rerankScores = Measure(testQueries, reranked);
Console.WriteLine($"BM25 + cross-encoder:  {rerankScores}");
Console.WriteLine($"The re-ranker can only order what the first stage found: of the {rerankScores.RecallAt20:P1} of questions whose answer " +
    $"is among the {Candidates} candidates, it puts {rerankScores.HitAt1 / rerankScores.RecallAt20:P1} first.");
Console.WriteLine($"Re-ranking cost: {msPerQuery:F2} ms per question ({Candidates} pairs in one batch, {device})\n");

Console.WriteLine("Hit@1 per aspect (unseen towns)      BM25   re-ranked");
foreach (var aspect in Collection.Questions.Keys)
{
    var rows = testQueries.Select((q, i) => (q, i)).Where(p => p.q.Aspect == aspect).ToList();
    double Hit(List<int[]> ranking) => rows.Average(p => ranking[p.i][0] == p.q.Answer ? 1.0 : 0.0);
    Console.WriteLine($"  {aspect,-32} {Hit(firstStage),6:P0}  {Hit(reranked),8:P0}");
}

Console.WriteLine();
foreach (int i in new[] { 0, 9, 22 })
{
    var q = testQueries[i];
    Console.WriteLine($"\"{q.Text}\"\n  BM25 top:      {collection.Passages[firstStage[i][0]].Text}\n  re-ranked top: {collection.Passages[reranked[i][0]].Text}");
}

return rerankScores.HitAt1 > bm25Scores.HitAt1 ? 0 : 1;

// The pair scorer: "<cls> question <sep> passage" → one relevance score, read from the <cls> position.
Sequential Scorer(int vocabulary, Random r) => new()
{
    new Embedding(vocabulary, Dim, device, r),
    new PositionalEncoding(Length, Dim, device),
    new TransformerEncoderLayer(Dim, heads: 4, ffDim: 2 * Dim, dropout: 0.1f, device: device, random: r),
    new TransformerEncoderLayer(Dim, heads: 4, ffDim: 2 * Dim, dropout: 0.1f, device: device, random: r),
    new LayerNorm(Dim, device: device),
    new Lambda(x => x.Narrow(1, 0, 1).Reshape(-1, Dim), "ClsToken"),
    new Linear(Dim, 1, device: device, random: r),
};

static string[] Specials() => ["<pad>", "<cls>", "<sep>", "<unk>", .. Enumerable.Range(0, Placeholders).Select(i => $"<w{i}>")];

void Encode(WordTokenizer words, string question, string passage, float[] into, int offset)
{
    var placeholders = new Dictionary<string, int>();
    int Id(string word) => words[word] != words["<unk>"] ? words[word]
        : placeholders.Count < Placeholders || placeholders.ContainsKey(word)
            ? words[$"<w{(placeholders.TryGetValue(word, out int k) ? k : placeholders[word] = placeholders.Count)}>"]
            : words["<unk>"];
    var ids = new List<int> { words["<cls>"] };
    ids.AddRange(WordTokenizer.Split(question).Select(Id));
    ids.Add(words["<sep>"]);
    ids.AddRange(WordTokenizer.Split(passage).Select(Id));
    for (int t = 0; t < Length; t++)
    {
        into[offset + t] = t < ids.Count ? ids[t] : words["<pad>"];
    }
}

// Scores every (question, passage) pair in one batch.
float[] Score(Sequential model, WordTokenizer words, string question, string[] passages)
{
    var flat = new float[passages.Length * Length];
    for (int p = 0; p < passages.Length; p++)
    {
        Encode(words, question, passages[p], flat, p * Length);
    }

    using var input = Tensor.From(flat, [passages.Length, Length], device);
    using var scores = model.Predict(input);
    return scores.ToArray();
}

// One row per question: the answer at a random position among the 7 best-scoring BM25 passages that are not the answer.
Dataset Groups(List<Query> queries, Random r)
{
    var features = new float[queries.Count, Group * Length];
    var labels = new int[queries.Count];
    var row = new float[Group * Length];
    for (int i = 0; i < queries.Count; i++)
    {
        var q = queries[i];
        var negatives = bm25.Search(q.Text, Group + 4).Where(id => id != q.Answer).Take(Group - 1).ToList();
        labels[i] = r.Next(Group);
        negatives.Insert(labels[i], q.Answer);
        for (int g = 0; g < Group; g++)
        {
            Encode(tokenizer, q.Text, collection.Passages[negatives[g]].Text, row, g * Length);
        }

        for (int c = 0; c < row.Length; c++)
        {
            features[i, c] = row[c];
        }
    }

    return Dataset.FromClassLabels(features, labels, Group);
}

static RankingScores Measure(List<Query> queries, List<int[]> rankings)
{
    double hit = 0, mrr = 0, recall = 0;
    for (int i = 0; i < queries.Count; i++)
    {
        int rank = Array.IndexOf(rankings[i], queries[i].Answer);
        hit += rank == 0 ? 1 : 0;
        mrr += rank is >= 0 and < 10 ? 1.0 / (rank + 1) : 0;
        recall += rank >= 0 ? 1 : 0;
    }

    return new RankingScores(hit / queries.Count, mrr / queries.Count, recall / queries.Count);
}

static string Clip(string text) => text.Length <= 44 ? text : text[..43] + "…";

/// <summary>Hit@1 (answer ranked first), MRR@10 (mean of 1/rank within the top 10) and Recall@20 (answer among the candidates).</summary>
internal sealed record RankingScores(double HitAt1, double MrrAt10, double RecallAt20)
{
    public override string ToString() => $"Hit@1 {HitAt1:P1}   MRR@10 {MrrAt10:F3}   Recall@20 {RecallAt20:P1}";
}
