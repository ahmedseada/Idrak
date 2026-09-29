// Sequence classification: sentiment of short sentences where "not" flips the next word
// ("not good" is negative, "not bad" positive). A bag-of-words model cannot see word order;
// LSTM, GRU and a Transformer can. Embedding, LSTM, GRU, PositionalEncoding, TransformerEncoderLayer.
//
//   dotnet run -c Release --project samples/Idrak.Samples.Sequences            (add --cpu / --cuda)
//   dotnet run -c Release --project samples/Idrak.Samples.Sequences -- --predict --input "the movie was not good;not bad at all"
//        classify sentences with every saved model (separate sentences with ';')

using Idrak;
using Idrak.Data;
using Idrak.Diagnostics;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Samples;
using Idrak.Training;

if (SampleOptions.Parse(args) is not { } options)
{
    return 0;
}

var device = options.Device;
Console.WriteLine($"Device: {device} - {device.Name}\n");
using var logger = Telemetry.Subscribe(new ConsoleLogger(options.LogLevels, epochInterval: 5));

// ---------------------------------------------------------------- vocabulary and synthetic sentences
string[] positive = ["good", "great", "excellent", "love", "wonderful", "fun"];
string[] negative = ["bad", "awful", "terrible", "hate", "boring", "dull"];
string[] neutral = ["the", "movie", "was", "a", "plot", "acting", "really", "very", "it", "this", "i", "at", "all", "and"];
string[] vocabulary = ["<pad>", "<unk>", "not", .. positive, .. negative, .. neutral];
var ids = vocabulary.Select((word, id) => (word, id)).ToDictionary(p => p.word, p => p.id);
const int MaxLength = 12;

// ---------------------------------------------------------------- models
const int Dim = 32;
// Transformers learn word order from positional encodings rather than from recurrence, so they need more steps.
var models = new (string Name, int Epochs, Func<Random, Module> Create)[]
{
    ("Bag of words (order-blind baseline)", 15, r => new Sequential
    {
        new Embedding(vocabulary.Length, Dim, device, r),
        new Lambda(x => x.Mean(1), "MeanOverWords"),
        new Linear(Dim, 2, device: device, random: r),
    }),
    ("LSTM", 15, r => new Sequential
    {
        new Embedding(vocabulary.Length, Dim, device, r),
        new LSTM(Dim, 64, device: device, random: r),
        new Linear(64, 2, device: device, random: r),
    }),
    ("GRU", 15, r => new Sequential
    {
        new Embedding(vocabulary.Length, Dim, device, r),
        new GRU(Dim, 64, device: device, random: r),
        new Linear(64, 2, device: device, random: r),
    }),
    ("Transformer", 40, r => new Sequential
    {
        new Embedding(vocabulary.Length, Dim, device, r),
        new PositionalEncoding(MaxLength, Dim, device),
        new TransformerEncoderLayer(Dim, heads: 4, ffDim: 64, dropout: 0f, device: device, random: r),
        new TransformerEncoderLayer(Dim, heads: 4, ffDim: 64, dropout: 0f, device: device, random: r),
        new LayerNorm(Dim, device: device),
        new Lambda(x => x.Mean(1), "MeanOverWords"),
        new Linear(Dim, 2, device: device, random: r),
    }),
};

string ModelFile(string name) => options.ModelPath($"sentiment-{new string([.. name.ToLowerInvariant().TakeWhile(char.IsLetter)])}.weights");
string[] sentences = options.Input is { } input
    ? input.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
    : ["i really love this movie", "the movie was not good", "not bad at all", "the plot was boring and the acting was awful", "it was not boring it was wonderful", "i hate it"];

if (options.PredictOnly)
{
    // Inference mode: load every saved model and compare their verdicts on the sentences.
    var loaded = new List<(string Name, Module Model)>();
    foreach (var (name, _, create) in models)
    {
        string path = ModelFile(name);
        if (File.Exists(path))
        {
            var model = create(new Random(2));
            model.Load(path);
            loaded.Add((name, model));
        }
    }

    if (loaded.Count == 0)
    {
        return options.RequireModel(ModelFile(models[1].Name)) ? 0 : 1;
    }

    Console.WriteLine($"Loaded {loaded.Count} models from {Path.GetDirectoryName(ModelFile("x"))}\n");
    var verdicts = loaded.Select(m => PositiveProbabilities(m.Model, sentences)).ToList();
    for (int s = 0; s < sentences.Length; s++)
    {
        Console.WriteLine($"\"{sentences[s]}\"");
        for (int m = 0; m < loaded.Count; m++)
        {
            float positiveProbability = verdicts[m][s];
            Console.WriteLine($"    {loaded[m].Name,-36} {(positiveProbability >= 0.5f ? "positive" : "negative"),-8} ({Math.Max(positiveProbability, 1 - positiveProbability):P0})");
        }
    }

    loaded.ForEach(m => m.Model.Dispose());
    return 0;
}

var random = new Random(1);
var (train, trainNegated) = Generate(6000);
var (test, testNegated) = Generate(1500);
Console.WriteLine($"{train.Count} training / {test.Count} test sentences, vocabulary {vocabulary.Length}, length {MaxLength}");
Console.WriteLine($"Example: \"{Decode(test.GetFeatures(0))}\" -> {(test.GetTargets(0)[1] == 1 ? "positive" : "negative")}\n");

var results = new List<(string Name, double Accuracy, double Negated, Module Model)>();
foreach (var (name, defaultEpochs, create) in models)
{
    int epochs = options.Epochs ?? defaultEpochs;
    Console.WriteLine($"=== {name}");
    var model = create(new Random(2));
    var optimizer = new AdamW(model.Parameters(), learningRate: 0.003f, weightDecay: 1e-4f);
    var trainer = new Trainer(model, optimizer, (logits, targets) => Losses.CrossEntropy(logits, targets))
    {
        Metrics = { Metric.Accuracy },
        Scheduler = new CosineAnnealing(optimizer, epochs, warmupEpochs: 1),
        MaxGradientNorm = 1f,
    };
    trainer.Fit(new DataLoader(train, options.BatchSize ?? 64, shuffle: true, device: device, seed: 3), epochs);
    double accuracy = trainer.Evaluate(new DataLoader(test, 500, device: device)).Metrics["accuracy"];
    double negated = trainer.Evaluate(new DataLoader(testNegated, 500, device: device)).Metrics["accuracy"];
    results.Add((name, accuracy, negated, model));
    model.Save(ModelFile(name));
    Console.WriteLine($"Saved to {ModelFile(name)}\n");
}

Console.WriteLine("Model                                   test accuracy   sentences with \"not\"");
foreach (var (name, accuracy, negated, _) in results)
{
    Console.WriteLine($"  {name,-36}   {accuracy,10:P1}   {negated,18:P1}");
}

// ---------------------------------------------------------------- try new sentences with the best model
var best = results.MaxBy(r => r.Accuracy);
Console.WriteLine($"\nClassifying new sentences with the {best.Name}:");
var bestVerdicts = PositiveProbabilities(best.Model, sentences);
for (int s = 0; s < sentences.Length; s++)
{
    float positiveProbability = bestVerdicts[s];
    Console.WriteLine($"  {(positiveProbability >= 0.5f ? "positive" : "negative"),-8} ({Math.Max(positiveProbability, 1 - positiveProbability):P0})  \"{sentences[s]}\"");
}

Console.WriteLine("\nCompare all saved models on your own sentences with: --predict --input \"sentence one;sentence two\"");

foreach (var r in results)
{
    r.Model.Dispose();
}

return best.Accuracy > 0.9 ? 0 : 1;

// Tokenizes sentences (unknown words map to <unk>) and returns each one's probability of being positive.
float[] PositiveProbabilities(Module model, string[] texts)
{
    var encoded = new float[texts.Length, MaxLength];
    for (int s = 0; s < texts.Length; s++)
    {
        var words = texts[s].ToLowerInvariant().Split(' ', StringSplitOptions.RemoveEmptyEntries);
        for (int t = 0; t < Math.Min(words.Length, MaxLength); t++)
        {
            encoded[s, t] = ids.GetValueOrDefault(words[t], ids["<unk>"]);
        }
    }

    using var input = Tensor.From(encoded, device);
    using var logits = model.Predict(input);
    using var probabilities = logits.Softmax();
    var p = probabilities.ToArray();
    return [.. Enumerable.Range(0, texts.Length).Select(s => p[s * 2 + 1])];
}

// Builds `count` labelled sentences, plus the subset that contains a negation.
(Dataset All, Dataset Negated) Generate(int count)
{
    var features = new float[count, MaxLength];
    var labels = new int[count];
    var negatedRows = new List<int>();
    for (int s = 0; s < count; s++)
    {
        int score;
        bool hasNot;
        do
        {
            Array.Clear(labels, s, 1);
            for (int t = 0; t < MaxLength; t++)
            {
                features[s, t] = 0;
            }

            int length = random.Next(5, MaxLength + 1);
            score = 0;
            hasNot = false;
            for (int t = 0; t < length; t++)
            {
                double roll = random.NextDouble();
                string word;
                if (roll < 0.15 && t < length - 1)
                {
                    word = "not";
                    string next = random.Next(2) == 0 ? positive[random.Next(positive.Length)] : negative[random.Next(negative.Length)];
                    features[s, t] = ids[word];
                    features[s, t + 1] = ids[next];
                    score -= positive.Contains(next) ? 1 : -1;
                    hasNot = true;
                    t++;
                    continue;
                }

                word = roll < 0.35 ? positive[random.Next(positive.Length)]
                    : roll < 0.55 ? negative[random.Next(negative.Length)]
                    : neutral[random.Next(neutral.Length)];
                features[s, t] = ids[word];
                score += positive.Contains(word) ? 1 : negative.Contains(word) ? -1 : 0;
            }
        }
        while (score == 0);

        labels[s] = score > 0 ? 1 : 0;
        if (hasNot)
        {
            negatedRows.Add(s);
        }
    }

    var all = Dataset.FromClassLabels(features, labels, 2, ["negative", "positive"]);
    return (all, all.Subset([.. negatedRows]));
}

string Decode(ReadOnlySpan<float> tokens)
{
    var words = new List<string>();
    foreach (float t in tokens)
    {
        if (t != 0)
        {
            words.Add(vocabulary[(int)t]);
        }
    }

    return string.Join(' ', words);
}
