// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Retrieval-augmented generation over a collection about invented towns: questions are answered by a small chat model
// that reads the passages a search finds and cites them. Everything is trained here, from scratch:
//   1. a bi-encoder (TextEncoder) for vector search, trained contrastively on (question, answer passage) pairs;
//   2. a cross-encoder re-ranker, trained listwise on hard negatives (as in the ReRanker sample);
//   3. a word-level chat model (ChatML) that answers from numbered passages with a citation, or says it does not know.
// Search combines BM25 keywords and vectors with reciprocal rank fusion, re-ranks the 10 best and shows the model 3.
// All numbers are measured on 40 towns none of the models saw in training.
//
//   dotnet run -c Release --project samples/Idrak.Samples.Rag            (add --cpu / --cuda)
//   dotnet run -c Release --project samples/Idrak.Samples.Rag -- --predict --input "who is the mayor of <town>"

using System.Diagnostics;
using Idrak;
using Idrak.Data;
using Idrak.Diagnostics;
using Idrak.Generation;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Retrieval;
using Idrak.Retrieval.Abstractions;
using Idrak.Samples;
using Idrak.Samples.Rag;
using Idrak.Samples.ReRanker;
using Idrak.Training;

if (SampleOptions.Parse(args) is not { } options)
{
    return 0;
}

var device = options.Device;
Console.WriteLine($"Device: {device} - {device.Name}\n");
using var logger = Telemetry.Subscribe(new ConsoleLogger(options.LogLevels, epochInterval: 4));

const int Towns = 160, TrainTowns = 120, Placeholders = 8, Group = 8;
const int EncoderLength = 20, PairLength = 36, Context = 128, Candidates = 10, Shown = 3;
string packagePath = options.ModelPath("rag.ikm");
string indexPath = Path.ChangeExtension(packagePath, ".index");
var collection = new Collection(Towns, seed: 1);
var documents = collection.Towns.Select((name, t) =>
    new Document(name, string.Join(" ", collection.Passages.Where(p => p.Entity == t).Select(p => p.Text)))).ToList();
var template = new ChatMLTemplate();
var greedy = new GenerationOptions { Temperature = 0f, TopK = 1, RepeatPenalty = 1f, NumPredict = 24, NumCtx = Context };

if (options.PredictOnly)
{
    // Inference mode: load the three models and the index, answer --input (a question about any of the 160 towns).
    if (!options.RequireModel(packagePath) || !options.RequireModel(indexPath))
    {
        return 1;
    }

    using var package = ModelPackage.Open(packagePath);
    var loadedTokenizer = new TownTokenizer(package.WordTokenizer("words"), Placeholders, hashBuckets: 0);
    var loadedHashing = new TownTokenizer(loadedTokenizer.Words, placeholders: 0, TownTokenizer.Buckets);
    var (encoderNet, scorerNet, generatorNet) = (Encoder(loadedTokenizer.VocabularySize), Scorer(loadedTokenizer.VocabularySize), Generator(loadedTokenizer.VocabularySize));
    using var _ = encoderNet;
    using var __ = scorerNet;
    using var ___ = generatorNet;
    package.LoadWeights(encoderNet, "encoder");
    package.LoadWeights(scorerNet, "reranker");
    package.LoadWeights(generatorNet, "generator");
    var loadedIndex = RetrievalIndex.Load(indexPath, new TextEncoder(encoderNet, loadedHashing, EncoderLength, loadedTokenizer.Words["<pad>"]));
    var pipeline = Pipeline(new ChatGenerator(new TextGenerator(generatorNet, loadedTokenizer, Context), template), loadedIndex,
        new CrossEncoder(scorerNet, (q, p) => EncodePair(loadedTokenizer, q, p), [PairLength]));
    string question = options.Input ?? $"who is the mayor of {collection.Towns[^1]}";
    Console.WriteLine($"Question: {question}\n\nPassages shown to the model:");
    foreach (var p in pipeline.Retrieve(question))
    {
        Console.WriteLine($"  [{p.Number}] ({p.Label}, score {p.Score:F2}) {p.Chunk.Text}");
    }

    Console.Write("\nAnswer:");
    await foreach (var delta in pipeline.StreamAsync(question))
    {
        Console.Write(delta.Content);
    }

    Console.WriteLine($"\nCited: {string.Join(", ", pipeline.LastAnswer!.Cited.Select(c => $"[{c.Number}] {c.Chunk.Text}"))}");
    return 0;
}

// ---------------------------------------------------------------- data
var trainQueries = collection.QueriesFor(Enumerable.Range(0, TrainTowns));
var testQueries = collection.QueriesFor(Enumerable.Range(TrainTowns, Towns - TrainTowns));
var trainPassages = collection.Passages.Where(p => p.Entity < TrainTowns).ToList();
// Vocabulary: words used for at least two training towns (so not town names; those become placeholders), the chat
// template's markers and the words of the answers. Numbers are spelled with digits, so they need no entries.
var townsPerWord = trainPassages.Select(p => (p.Entity, p.Text)).Concat(trainQueries.Select(q => (q.Entity, q.Text)))
    .SelectMany(p => TownTokenizer.WordsOf(p.Text).Select(w => (w, p.Entity))).Distinct().GroupBy(p => p.w).ToDictionary(g => g.Key, g => g.Count());
string scaffolding = template.Render([new ChatMessage("user", Prompt("?", []))], [], think: false) + Answers.Unknown + " [ ] . <|im_end|>";
var answerWords = trainPassages.Where(p => p.Aspect is not null).SelectMany(p => TownTokenizer.WordsOf(Answers.Of(p)));
string[] specials = [.. TownTokenizer.Specials, .. TownTokenizer.Placeholders(Placeholders)];
var tokenizer = new TownTokenizer(new WordTokenizer([.. specials, .. townsPerWord.Where(p => p.Value >= 2).Select(p => p.Key)
    .Concat(TownTokenizer.WordsOf(scaffolding)).Concat(answerWords).Distinct().Except(specials).Order(StringComparer.Ordinal)]), Placeholders, hashBuckets: 0);
// The bi-encoder reads questions and passages separately, so it gets the same words with town names hashed instead.
var hashing = new TownTokenizer(tokenizer.Words, placeholders: 0, TownTokenizer.Buckets);
int vocabulary = tokenizer.VocabularySize;
Console.WriteLine($"{collection.Passages.Count} passages about {Towns} towns ({documents.Count} documents); {trainQueries.Count} training questions " +
    $"({TrainTowns} towns), {testQueries.Count} test questions ({Towns - TrainTowns} unseen towns); vocabulary {vocabulary}");

// ---------------------------------------------------------------- 1. vector search: a bi-encoder trained on question → passage pairs
var clock = Stopwatch.StartNew();
using var encoderModel = Encoder(vocabulary);
var encoder = new TextEncoder(encoderModel, hashing, EncoderLength, tokenizer.Words["<pad>"]);
var pairs = trainQueries.Select(q => (q.Text, collection.Passages[q.Answer].Text)).ToList();
var encoderLosses = encoder.Train(pairs, epochs: options.Epochs ?? 20, batchSize: 64,
    p => new AdamW(p, learningRate: 0.002f, weightDecay: 0.01f), temperature: 0.05f, seed: 3);
Console.WriteLine($"\nBi-encoder: {encoderModel.Parameters().Sum(p => p.Size):N0} parameters, contrastive loss {encoderLosses[0]:F3} → {encoderLosses[^1]:F3} " +
    $"({encoderLosses.Count} epochs, {clock.Elapsed.TotalSeconds:F0} s)");

clock.Restart();
var index = RetrievalIndex.Create().Documents(documents, ChunkUnit.Sentences, size: 1, overlap: 0)
    .Bm25(k1: 1.2, b: 0.75).Embeddings(encoder).Fusion(k: 60, depth: 20).Build();
var keywordsOnly = RetrievalIndex.Create().Add(index.Chunks).Bm25(k1: 1.2, b: 0.75).Build();
var vectorsOnly = RetrievalIndex.Create().Add(index.Chunks).Embeddings(encoder).Build();
Check(index.Chunks.Select(c => c.Text).SequenceEqual(collection.Passages.Select(p => p.Text)), "one chunk per passage, in order");
Console.WriteLine($"Index: {index.Chunks.Count} chunks (one sentence each), BM25 + vectors + reciprocal rank fusion (k 60, depth 20), built in {clock.Elapsed.TotalSeconds:F1} s");

// ---------------------------------------------------------------- 2. re-ranking: a cross-encoder trained listwise
clock.Restart();
using var scorer = Scorer(vocabulary);
using var groupModel = new Sequential
{
    new Lambda(x => x.Reshape(-1, PairLength), "PairsOfGroup"),
    scorer,
    new Lambda(x => x.Reshape(-1, Group), "ScoresOfGroup"),
};
var (groupTrain, groupValidation) = Groups(trainQueries, new Random(6)).Split(0.95, seed: 7);
int rerankEpochs = options.Epochs ?? 6;
var rerankRun = new TrainingRun
{
    Model = groupModel,
    Loss = (logits, targets) => Losses.CrossEntropy(logits, targets),
    Optimizer = p => new AdamW(p, learningRate: 0.002f, weightDecay: 0.01f),
    Scheduler = o => new CosineAnnealing(o, rerankEpochs, minLearningRate: 1e-4f, warmupEpochs: 1),
    Train = new DataLoader(groupTrain, options.BatchSize ?? 32, shuffle: true, device: device, seed: 8),
    Validation = new DataLoader(groupValidation, 200, device: device),
    Metrics = [Metric.Accuracy],
    Epochs = rerankEpochs,
    MaxGradientNorm = 1f,
};
var rerankHistory = rerankRun.Fit();
var reranker = new CrossEncoder(scorer, (q, p) => EncodePair(tokenizer, q, p), [PairLength]);
Console.WriteLine($"Cross-encoder: {scorer.Parameters().Sum(p => p.Size):N0} parameters, picks the answer among {Group} in " +
    $"{rerankHistory.Epochs[^1].ValidationMetrics!.GetValueOrDefault(Metric.Accuracy.Name):P1} of validation groups ({clock.Elapsed.TotalSeconds:F0} s)");

// ---------------------------------------------------------------- retrieval quality on unseen towns
clock.Restart();
var rankings = new (string Name, List<int[]> Ranking)[]
{
    ("BM25 keywords", [.. testQueries.Select(q => Ids(keywordsOnly.Search(q.Text, Candidates)))]),
    ("Vectors (bi-encoder)", [.. testQueries.Select(q => Ids(vectorsOnly.Search(q.Text, Candidates)))]),
    ("Hybrid (fusion)", [.. testQueries.Select(q => Ids(index.Search(q.Text, Candidates)))]),
    ("Hybrid + re-ranker", [.. testQueries.Select(q => Ids(reranker.Rerank(q.Text, index.Search(q.Text, Candidates), Candidates)))]),
};
Console.WriteLine($"\nRetrieval on unseen towns ({testQueries.Count} questions)   Hit@1   Recall@{Shown}   Recall@{Candidates}");
foreach (var (name, ranking) in rankings)
{
    double Recall(int k) => testQueries.Select((q, i) => ranking[i].Take(k).Contains(q.Answer) ? 1.0 : 0.0).Average();
    Console.WriteLine($"  {name,-44} {Recall(1),6:P1}   {Recall(Shown),7:P1}   {Recall(Candidates),8:P1}");
}

Console.WriteLine($"  ({clock.Elapsed.TotalMilliseconds / testQueries.Count / rankings.Length:F1} ms per question and method)");

// ---------------------------------------------------------------- 3. the chat model: answer from numbered passages, with a citation
clock.Restart();
var transcripts = Transcripts(trainQueries, new Random(9));
var sequences = Sequences(transcripts, out double replyTokens);
var (chatTrain, chatValidation) = sequences.Split(0.95, seed: 10);
using var generator = Generator(vocabulary);
int chatEpochs = options.Epochs ?? 8;
var chatRun = new TrainingRun
{
    Model = generator,
    Loss = (logits, next) => ReplyLoss(logits, next, replyTokens),
    Optimizer = p => new AdamW(p, learningRate: 0.002f, weightDecay: 0.01f),
    Scheduler = o => new CosineAnnealing(o, chatEpochs, minLearningRate: 1e-4f, warmupEpochs: 1),
    Train = new DataLoader(chatTrain, options.BatchSize ?? 32, shuffle: true, device: device, seed: 11),
    Validation = new DataLoader(chatValidation, 200, device: device),
    Epochs = chatEpochs,
    MaxGradientNorm = 1f,
};
var chatHistory = chatRun.Fit();
Console.WriteLine($"\nChat model: {generator.Parameters().Sum(p => p.Size):N0} parameters, {transcripts.Count} transcripts, " +
    $"validation loss per reply token {chatHistory.Epochs[^1].ValidationLoss:F4} ({clock.Elapsed.TotalSeconds:F0} s)");
Console.WriteLine($"Example transcript:\n  {transcripts[0].Replace("\n", "\n  ")}");

generator.Eval();
var chat = new ChatGenerator(new TextGenerator(generator, tokenizer, Context), template);
var rag = Pipeline(chat, index, reranker);

ModelPackage.Create(packagePath).Weights("encoder", encoderModel).Weights("reranker", scorer).Weights("generator", generator)
    .Tokenizer("words", tokenizer.Words).Save();
index.Save(indexPath);
Console.WriteLine($"\nSaved {packagePath} (three models and the vocabulary) and {Path.GetFileName(indexPath)} (chunks, settings, vectors)");

// ---------------------------------------------------------------- end to end on unseen towns
clock.Restart();
var closedBook = new List<Outcome>();
var retrieved = new List<Outcome>();
var oracle = new List<Outcome>();
var oracleRandom = new Random(12);
foreach (var q in testQueries)
{
    string expected = Answers.Of(collection.Passages[q.Answer]);
    closedBook.Add(await Ask(q, expected, []));
    var answer = await rag.AskAsync(q.Text);
    retrieved.Add(Score(q, expected, answer.Text, answer.Cited, answer.Passages));

    // Oracle: the answer passage plus the two best other retrieved passages, the answer at a random position.
    var others = rag.Retrieve(q.Text).Where(p => p.Chunk.Id != q.Answer).Take(Shown - 1).Select(p => p.Chunk).ToList();
    others.Insert(oracleRandom.Next(Shown), index.Chunks[q.Answer]);
    oracle.Add(await Ask(q, expected, [.. others.Select((c, i) => new Citation(i + 1, c.DocumentId, c, 0))]));
}

double msPerAnswer = clock.Elapsed.TotalMilliseconds / testQueries.Count / 3;
Console.WriteLine($"\nAnswers on unseen towns ({testQueries.Count} questions)             correct   cites the answer passage   \"i do not know\"");
foreach (var (name, outcomes) in new[] { ("Closed book (no passages)", closedBook), ("RAG: hybrid + re-ranker, 3 passages", retrieved), ("Oracle (answer passage always shown)", oracle) })
{
    Console.WriteLine($"  {name,-52} {outcomes.Average(o => o.Correct ? 1.0 : 0.0),7:P1}   {outcomes.Average(o => o.CitesAnswer ? 1.0 : 0.0),24:P1}   {outcomes.Average(o => o.Abstained ? 1.0 : 0.0),15:P1}");
}

var shown = retrieved.Where(o => o.AnswerShown).ToList();
var missing = retrieved.Where(o => !o.AnswerShown).ToList();
Console.WriteLine($"  RAG, answer passage among the 3 shown ({shown.Count}): {shown.Average(o => o.Correct ? 1.0 : 0.0):P1} correct; " +
    $"not shown ({missing.Count}): {(missing.Count == 0 ? 0 : missing.Average(o => o.Abstained ? 1.0 : 0.0)):P1} said \"i do not know\"");
Console.WriteLine($"  Generation: {msPerAnswer:F1} ms per answer (greedy, KV cache, {device})");

Console.WriteLine("\nPer aspect (RAG, correct):");
foreach (var aspect in Collection.Questions.Keys)
{
    var rows = testQueries.Select((q, i) => (q, i)).Where(p => p.q.Aspect == aspect).ToList();
    Console.WriteLine($"  {aspect,-11} {rows.Average(p => retrieved[p.i].Correct ? 1.0 : 0.0),6:P0}");
}

Console.WriteLine();
foreach (int i in new[] { 0, 11, 29 })
{
    var q = testQueries[i];
    var answer = await rag.AskAsync(q.Text);
    Console.WriteLine($"Q: {q.Text}\n{string.Join("\n", answer.Passages.Select(p => $"   [{p.Number}] {p.Chunk.Text}"))}\nA: {answer.Text.Trim()}\n");
}

return retrieved.Average(o => o.Correct ? 1.0 : 0.0) > closedBook.Average(o => o.Correct ? 1.0 : 0.0) + 0.5 ? 0 : 1;

// ---------------------------------------------------------------- models
Sequential Encoder(int words) => Network.Tokens(EncoderLength).OnDevice(device).Seed(2)
    .Embedding(words, 64).PositionalEncoding().TransformerEncoderLayer(heads: 4, ffDim: 128, dropout: 0.1f).LayerNorm().Build();

// "<cls> question <sep> passage" → one relevance score, read from the <cls> position.
Sequential Scorer(int words) => Network.Tokens(PairLength).OnDevice(device).Seed(4)
    .Embedding(words, 64).PositionalEncoding().Repeat(2, b => b.TransformerEncoderLayer(heads: 4, ffDim: 128, dropout: 0.1f))
    .LayerNorm().FirstStep().Linear(1).Build();

Sequential Generator(int words) => Architectures.Gpt(words, Context, dim: 96, heads: 4, layers: 3, ffDim: 384, dropout: 0.1f)
    .OnDevice(device).Seed(5).Build();

RagPipeline Pipeline(IChatModel model, RetrievalIndex searchIndex, CrossEncoder crossEncoder) =>
    Rag.For(model).Retrieve(searchIndex, Candidates).Rerank(crossEncoder, Shown).Prompt(Prompt).Think(false).Options(greedy).Build();

// The question first, so its town is always the first placeholder <w0>; then the numbered passages.
static string Prompt(string question, IReadOnlyList<Citation> passages) =>
    $"question : {question} passages :" + string.Concat(passages.Select(p => $" [{p.Number}] {p.Chunk.Text}"));

static float[] EncodePair(TownTokenizer words, string question, string passage)
{
    var ids = words.Encode($"<cls> {question} <sep> {passage}");
    return [.. Enumerable.Range(0, PairLength).Select(t => (float)(t < ids.Count ? ids[t] : words.Words["<pad>"]))];
}

// ---------------------------------------------------------------- training data
// One row per question: the answer passage at a random position among the 7 best other passages of the hybrid search.
Dataset Groups(List<Query> queries, Random random)
{
    var features = new float[queries.Count, Group * PairLength];
    var labels = new int[queries.Count];
    for (int i = 0; i < queries.Count; i++)
    {
        var q = queries[i];
        var group = Ids(index.Search(q.Text, Group + 4)).Where(id => id != q.Answer).Take(Group - 1).ToList();
        labels[i] = random.Next(Group);
        group.Insert(labels[i], q.Answer);
        for (int g = 0; g < Group; g++)
        {
            var pair = EncodePair(tokenizer, q.Text, collection.Passages[group[g]].Text);
            for (int t = 0; t < PairLength; t++)
            {
                features[i, g * PairLength + t] = pair[t];
            }
        }
    }

    return Dataset.FromClassLabels(features, labels, Group);
}

// Chat transcripts like the ones the pipeline produces: 3 passages from the hybrid search. Per question: two with the
// answer passage at a random position (reply: the answer and its citation), one without it and, for some questions,
// one with no passages at all (reply: "i do not know").
List<string> Transcripts(List<Query> queries, Random random)
{
    var result = new List<string>();
    foreach (var q in queries)
    {
        var others = Ids(index.Search(q.Text, 8)).Where(id => id != q.Answer).Take(5).ToList();
        string answer = Answers.Of(collection.Passages[q.Answer]);
        for (int variant = 0; variant < 4; variant++)
        {
            if (variant == 3 && random.NextDouble() > 0.3)
            {
                continue;
            }

            var chosen = variant == 3 ? [] : others.OrderBy(_ => random.Next()).Take(variant == 2 ? Shown : Shown - 1).ToList();
            int position = -1;
            if (variant < 2)
            {
                position = random.Next(Shown);
                chosen.Insert(position, q.Answer);
            }

            var passages = chosen.Select((id, n) => new Citation(n + 1, index.Chunks[id].DocumentId, index.Chunks[id], 0)).ToList();
            string prompt = template.Render([new ChatMessage("user", Prompt(q.Text, passages))], [], think: false);
            result.Add(prompt + Answers.Reply(position < 0 ? null : answer, position + 1) + " <|im_end|>");
        }
    }

    return result;
}

// Inputs are "<prompt> <reply> <|im_end|> <pad>..."; each position's target is the next token where it belongs to the
// reply, and the id `vocabulary` elsewhere (a zero one-hot row after the cut in ReplyLoss, so it adds nothing).
Dataset Sequences(List<string> texts, out double averageReplyTokens)
{
    var features = new float[texts.Count * Context];
    var targets = new float[texts.Count * Context];
    Array.Fill(targets, vocabulary);
    long replyTotal = 0;
    for (int i = 0; i < texts.Count; i++)
    {
        int split = texts[i].LastIndexOf("</think>", StringComparison.Ordinal) + "</think>".Length;
        var prompt = tokenizer.Encode(texts[i][..split]);
        var ids = prompt.Concat(tokenizer.Encode(texts[i][split..])).ToList();
        Check(ids.Count <= Context + 1, $"transcript of {ids.Count} tokens exceeds the context of {Context}");
        for (int t = 0; t < Context; t++)
        {
            features[i * Context + t] = t < ids.Count ? ids[t] : tokenizer.Words["<pad>"];
            if (t + 1 < ids.Count && t + 1 >= prompt.Count)
            {
                targets[i * Context + t] = ids[t + 1];
                replyTotal++;
            }
        }
    }

    averageReplyTokens = (double)replyTotal / texts.Count;
    string[] positions = [.. Enumerable.Range(0, Context).Select(t => $"t{t}")];
    return Dataset.FromFlat(features, targets, texts.Count, positions, positions);
}

// Cross-entropy over reply tokens only, averaged per reply token.
static Tensor ReplyLoss(Tensor logits, Tensor next, double replyTokensPerRow)
{
    int words = logits.Shape[^1];
    var targets = Tensor.OneHot(next, words + 1).Narrow(2, 0, words);
    return (targets * logits.LogSoftmax()).Sum() * (float)(-1.0 / (next.Shape[0] * replyTokensPerRow));
}

// ---------------------------------------------------------------- evaluation
async Task<Outcome> Ask(Query q, string expected, IReadOnlyList<Citation> passages)
{
    var reply = await chat.ChatAsync(new ChatRequest(rag.Messages(q.Text, passages), Think: false, Options: greedy));
    string text = reply.Message!.Content;
    return Score(q, expected, text, RagPipeline.CitedIn(text, passages), passages);
}

static Outcome Score(Query q, string expected, string text, IReadOnlyList<Citation> cited, IReadOnlyList<Citation> passages) => new(
    Correct: Answers.Strip(text) == expected,
    CitesAnswer: cited.Count == 1 && cited[0].Chunk.Id == q.Answer,
    Abstained: Answers.Strip(text) == Answers.Unknown,
    AnswerShown: passages.Any(p => p.Chunk.Id == q.Answer));

static int[] Ids(IEnumerable<RetrievedChunk> results) => [.. results.Select(r => r.Chunk.Id)];

static void Check(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

internal sealed record Outcome(bool Correct, bool CitesAnswer, bool Abstained, bool AnswerShown);
