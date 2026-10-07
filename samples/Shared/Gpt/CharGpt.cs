// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json;
using System.Text;
using Idrak.Generation;
using Idrak.Layers;

namespace Idrak.Samples.Gpt;

/// <summary>Architecture and training record of a character-level GPT, saved next to its weights as JSON.</summary>
public sealed record GptConfig(string Vocabulary, int Context = 64, int Dim = 96, int Heads = 4, int Layers = 3)
{
    /// <summary>Epochs trained, final validation loss/accuracy and when (filled in after training).</summary>
    public int TrainedEpochs { get; init; }

    public double? ValidationLoss { get; init; }

    public double? ValidationAccuracy { get; init; }

    public DateTimeOffset? TrainedAt { get; init; }

    public string? Corpus { get; init; }

    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true };

    public static string ConfigPath(string weightsPath) => Path.ChangeExtension(weightsPath, ".json");

    public void Save(string weightsPath) => File.WriteAllText(ConfigPath(weightsPath), JsonSerializer.Serialize(this, Json));

    public static GptConfig Load(string weightsPath) =>
        JsonSerializer.Deserialize<GptConfig>(File.ReadAllText(ConfigPath(weightsPath)), Json)
        ?? throw new InvalidDataException($"Invalid model config {ConfigPath(weightsPath)}.");
}

/// <summary>Sampling and execution settings for <see cref="CharGpt.Generate"/>.</summary>
/// <param name="Length">Characters to generate per sample.</param>
/// <param name="Temperature">Softmax temperature: below 1 is more conservative, above 1 more random.</param>
/// <param name="TopK">Sample only among the k most likely characters (0 = all).</param>
/// <param name="Seed">Random seed for reproducible output (null = random).</param>
/// <param name="Samples">Independent continuations generated together as one batch (1-64).</param>
/// <param name="UseCache">Incremental decoding with a KV cache (false = recompute the whole window every step).</param>
/// <param name="UseGraph">Record the decoding step once and replay it (CUDA Graphs on GPU); needs the cache.</param>
/// <param name="ChunkSize">Tokens generated between host synchronizations when streaming (larger = faster, less live).</param>
public sealed record GenerationSettings(
    int Length = 200, float Temperature = 0.7f, int TopK = 0, int? Seed = null, int Samples = 1,
    bool UseCache = true, bool UseGraph = true, int ChunkSize = 16);

/// <summary>A likely alternative for one position.</summary>
public sealed record Alternative(string Token, float Probability);

/// <summary>One generated character of one sample, with its probability, the distribution's entropy (bits), top alternatives and latency.</summary>
public sealed record GeneratedToken(int Sample, int Index, string Token, float Probability, float Entropy, IReadOnlyList<Alternative> Alternatives, double LatencyMs);

/// <summary>One generated continuation.</summary>
public sealed record GeneratedSequence(int Sample, string Text, IReadOnlyList<GeneratedToken> Tokens);

/// <summary>Aggregate inference metrics (throughput counts every sample's characters).</summary>
public sealed record GenerationMetrics(
    string Device,
    string Mode,
    bool GraphRecorded,
    string? GraphNote,
    int Samples,
    int PromptTokens,
    int GeneratedTokens,
    int TotalTokens,
    double TotalMs,
    double FirstTokenMs,
    double MsPerToken,
    double MsPerStep,
    double TokensPerSecond,
    double AverageProbability,
    double Perplexity,
    double AverageEntropyBits,
    int ContextResets,
    long MemoryInUseBytes,
    long MemoryCachedBytes);

/// <summary>The output of a generation call.</summary>
public sealed record GenerationResult(string Prompt, IReadOnlyList<GeneratedSequence> Samples, GenerationMetrics Metrics)
{
    /// <summary>The first sample's text.</summary>
    public string Text => Samples[0].Text;
}

/// <summary>
/// A decoder-only character-level transformer: embeddings + sinusoidal positions → causal transformer blocks →
/// per-position next-character logits. Shared by the Transformer console sample and the GPT Web API.
/// </summary>
public sealed class CharGpt : IDisposable
{
    private readonly Dictionary<char, int> _index;

    private CharGpt(GptConfig config, Sequential model, Device device)
    {
        Config = config;
        Model = model;
        Device = device;
        _index = config.Vocabulary.Select((c, i) => (c, i)).ToDictionary(p => p.c, p => p.i);
    }

    public GptConfig Config { get; private set; }

    public Sequential Model { get; }

    public Device Device { get; private set; }

    /// <summary>Builds an untrained model.</summary>
    public static CharGpt Create(GptConfig config, Device device, Random? random = null)
    {
        random ??= new Random(2);
        var model = new Sequential
        {
            new Embedding(config.Vocabulary.Length, config.Dim, device, random),
            new PositionalEncoding(config.Context, config.Dim, device),
        };
        for (int layer = 0; layer < config.Layers; layer++)
        {
            model.Add(new TransformerEncoderLayer(config.Dim, config.Heads, ffDim: 4 * config.Dim, dropout: 0.1f, causal: true, device: device, random: random));
        }

        model.Add(new LayerNorm(config.Dim, device: device));
        model.Add(new Linear(config.Dim, config.Vocabulary.Length, device: device, random: random));
        model.Name = "char-gpt";
        return new CharGpt(config, model, device);
    }

    /// <summary>Loads weights and their JSON config.</summary>
    public static CharGpt Load(string weightsPath, Device device)
    {
        var gpt = Create(GptConfig.Load(weightsPath), device);
        gpt.Model.Load(weightsPath);
        return gpt;
    }

    public void Save(string weightsPath, GptConfig? config = null)
    {
        Config = config ?? Config;
        Model.Save(weightsPath);
        Config.Save(weightsPath);
    }

    /// <summary>Moves the weights to another device.</summary>
    public void MoveTo(Device device)
    {
        if (device != Device)
        {
            Model.To(device);
            Device = device;
        }
    }

    /// <summary>Encodes text as token ids, dropping characters outside the vocabulary.</summary>
    public List<int> Encode(string text) => [.. text.Where(_index.ContainsKey).Select(c => _index[c])];

    /// <summary>
    /// Generates <see cref="GenerationSettings.Samples"/> continuations of <paramref name="prompt"/>. With the cache,
    /// the prompt is processed once (prefill) and each step then processes only the newest character per sample;
    /// sampling runs on the model's device, so on a GPU the loop runs ahead without waiting for the host except
    /// every <see cref="GenerationSettings.ChunkSize"/> tokens (to report progress) and when the context window is
    /// full (it then re-reads the last half of the window). <paramref name="onToken"/> receives tokens chunk by chunk.
    /// </summary>
    public GenerationResult Generate(string prompt, GenerationSettings settings, Action<GeneratedToken>? onToken = null, CancellationToken cancellationToken = default)
    {
        int samples = Math.Clamp(settings.Samples, 1, 64), length = Math.Max(settings.Length, 1);
        uint seed = (uint)(settings.Seed ?? Random.Shared.Next());
        var promptIds = Encode(prompt);
        int promptTokens = promptIds.Count;
        if (promptIds.Count == 0)
        {
            promptIds.Add(_index.GetValueOrDefault(' '));
        }

        var history = Enumerable.Range(0, samples).Select(_ => new List<int>(promptIds)).ToArray();
        var tokens = Enumerable.Range(0, samples).Select(_ => new List<GeneratedToken>(length)).ToArray();
        using var sampler = new TokenSampler(Device, samples, Config.Vocabulary.Length, length)
        {
            Temperature = settings.Temperature,
            TopK = settings.TopK,
            Seed = seed,
        };

        var clock = Stopwatch.StartNew();
        double firstTokenMs = 0, lastChunkMs = 0;
        int emitted = 0, resets = 0;
        bool graphRecorded = false;
        string? graphNote = null;

        // Downloads the statistics of steps [emitted, produced) (one synchronization) and reports them.
        void Flush(int produced)
        {
            if (produced <= emitted)
            {
                return;
            }

            var chunk = sampler.Read(emitted, produced);
            double now = clock.Elapsed.TotalMilliseconds;
            double perStep = (now - lastChunkMs) / chunk.Length;
            lastChunkMs = now;
            for (int s = 0; s < chunk.Length; s++)
            {
                for (int r = 0; r < samples; r++)
                {
                    var t = chunk[s][r];
                    history[r].Add(t.Id);
                    var token = new GeneratedToken(r, emitted + s, Config.Vocabulary[t.Id].ToString(), t.Probability, t.Entropy,
                        [.. t.Alternatives.Where(a => a.Id >= 0).Select(a => new Alternative(Config.Vocabulary[a.Id].ToString(), a.Probability))],
                        emitted + s == 0 ? firstTokenMs : perStep);
                    tokens[r].Add(token);
                    onToken?.Invoke(token);
                }
            }

            emitted = produced;
        }

        Tensor Window(int keep)
        {
            var window = new float[samples * keep];
            for (int r = 0; r < samples; r++)
            {
                var h = history[r];
                for (int t = 0; t < keep; t++)
                {
                    window[r * keep + t] = h[h.Count - keep + t];
                }
            }

            return Tensor.From(window, [samples, keep], Device);
        }

        using (Autograd.NoGrad())
        {
            if (!settings.UseCache)
            {
                // Reference path: recompute the whole window every step (O(steps²) attention, one sync per step).
                for (int step = 0; step < length && !cancellationToken.IsCancellationRequested; step++)
                {
                    using var scope = new TensorScope();
                    var logits = Model.Predict(Window(Math.Min(history[0].Count, Config.Context)));
                    sampler.Sample(logits);
                    if (step == 0)
                    {
                        sampler.Read(0, 1);
                        firstTokenMs = clock.Elapsed.TotalMilliseconds;
                    }

                    Flush(step + 1);
                }
            }
            else
            {
                Model.Eval();
                using var context = new DecodingContext(Device, samples, Config.Context) { LastPositionOnly = true };
                ComputeGraph? graph = null;
                void Step()
                {
                    var logits = Model.ForwardCached(sampler.Ids.Reshape(samples, 1), context);
                    sampler.Sample(logits);
                }

                void Prefill(int keep)
                {
                    using var scope = new TensorScope();
                    context.Reset();
                    sampler.Sample(Model.ForwardCached(Window(keep), context));
                }

                try
                {
                    Prefill(Math.Min(history[0].Count, Config.Context - 1));
                    sampler.Read(0, 1);   // synchronize once: time to first token
                    firstTokenMs = clock.Elapsed.TotalMilliseconds;
                    int produced = 1;
                    for (; produced < length && !cancellationToken.IsCancellationRequested; produced++)
                    {
                        if (context.Length >= Config.Context)
                        {
                            // Window full: fetch pending tokens, then re-read the last half of each sequence.
                            Flush(produced);
                            Prefill(Config.Context / 2);
                            resets++;
                            continue;
                        }

                        if (graph is null && settings.UseGraph)
                        {
                            graph = context.CaptureStep(Step);
                            graphRecorded = graph.IsRecorded;
                            graphNote = graph.IsRecorded ? "decode step recorded as a CUDA graph"
                                : Device.Type == DeviceType.Cpu ? "graphs are GPU-only; CPU runs the step directly"
                                : $"graph recording failed, using normal launches: {graph.FailureReason}";
                        }

                        if (graph is not null)
                        {
                            context.ReplayStep(graph);
                        }
                        else
                        {
                            using var scope = new TensorScope();
                            Step();
                        }

                        if (onToken is not null && (produced + 1) % Math.Max(settings.ChunkSize, 1) == 0)
                        {
                            Flush(produced + 1);
                        }
                    }

                    Flush(produced);
                }
                finally
                {
                    graph?.Dispose();
                }
            }

            Flush(Math.Max(emitted, tokens[0].Count));
        }

        Device.Synchronize();
        clock.Stop();
        var all = tokens.SelectMany(t => t).ToList();
        int generated = tokens[0].Count, n = Math.Max(all.Count, 1);
        double totalMs = clock.Elapsed.TotalMilliseconds;
        var memory = ComputeResources.GetMemoryUsage(Device);
        string mode = !settings.UseCache ? "full recompute (no cache)" : graphRecorded ? "KV cache + CUDA graph" : "KV cache";
        var metrics = new GenerationMetrics(
            $"{Device} ({Device.Name})", mode, graphRecorded, graphNote, samples, promptTokens, generated, all.Count,
            totalMs, firstTokenMs, totalMs / Math.Max(generated, 1) / samples, (totalMs - firstTokenMs) / Math.Max(generated - 1, 1),
            all.Count / Math.Max(totalMs / 1000, 1e-9),
            all.Average(t => (double)t.Probability), Math.Exp(-all.Sum(t => Math.Log(Math.Max(t.Probability, 1e-12f))) / n),
            all.Average(t => (double)t.Entropy), resets, memory.InUse, memory.Cached);
        var sequences = tokens.Select((t, r) => new GeneratedSequence(r, string.Concat(t.Select(x => x.Token)), t)).ToList();
        return new GenerationResult(prompt, sequences, metrics);
    }

    public void Dispose() => Model.Dispose();
}

/// <summary>A tiny English-like grammar, so generated text can be checked word by word.</summary>
public static class Grammar
{
    private static readonly string[] Subjects = ["the little robot", "a curious cat", "the old wizard", "my neighbor", "the young pilot", "a quiet student", "the tired farmer", "our teacher"];
    private static readonly string[] Verbs = ["found", "built", "painted", "carried", "watched", "repaired", "opened", "forgot", "sold", "borrowed"];
    private static readonly string[] Objects = ["a shiny key", "the broken clock", "an old map", "a tiny garden", "the red door", "a wooden boat", "the heavy box", "a strange letter"];
    private static readonly string[] Places = ["near the river", "in the tower", "behind the market", "under the bridge", "at the station", "on the hill"];
    private static readonly string[] Times = ["every morning", "last night", "before dawn", "after lunch", "on sunday", "in the spring"];

    /// <summary>Every word the grammar can produce.</summary>
    public static readonly HashSet<string> Words =
        [.. new[] { Subjects, Verbs, Objects, Places, Times }.SelectMany(p => p).SelectMany(p => p.Split(' ')).Append("and").Append("then")];

    /// <summary>Random sentences, five per line.</summary>
    public static string Corpus(int sentences, int seed)
    {
        var random = new Random(seed);
        string Pick(string[] options) => options[random.Next(options.Length)];
        var text = new StringBuilder();
        for (int i = 0; i < sentences; i++)
        {
            text.Append(random.Next(4) switch
            {
                0 => $"{Pick(Times)}, {Pick(Subjects)} {Pick(Verbs)} {Pick(Objects)}.",
                1 => $"{Pick(Subjects)} {Pick(Verbs)} {Pick(Objects)} and then {Pick(Verbs)} {Pick(Objects)}.",
                _ => $"{Pick(Subjects)} {Pick(Verbs)} {Pick(Objects)} {Pick(Places)}.",
            });
            text.Append(i % 5 == 4 ? '\n' : ' ');
        }

        return text.ToString();
    }

    /// <summary>Fraction of complete generated words that are grammar words.</summary>
    public static (int Valid, int Total) Score(string generated)
    {
        var words = generated.Split([' ', '.', ',', '\n'], StringSplitOptions.RemoveEmptyEntries).SkipLast(1).ToList();
        return (words.Count(Words.Contains), words.Count);
    }
}

/// <summary>Builds training windows (inputs and next-character targets) and trains a <see cref="CharGpt"/>.</summary>
public static class GptTraining
{
    public static Data.Dataset Windows(string corpus, GptConfig config, int maxWindows, int seed)
    {
        var index = config.Vocabulary.Select((c, i) => (c, i)).ToDictionary(p => p.c, p => p.i);
        var random = new Random(seed);
        int context = config.Context;
        int windows = Math.Min(maxWindows, Math.Max(1, corpus.Length / 4));
        var features = new float[windows * context];
        var targets = new float[windows * context];
        for (int w = 0; w < windows; w++)
        {
            int start = random.Next(corpus.Length - context - 1);
            for (int t = 0; t < context; t++)
            {
                features[w * context + t] = index[corpus[start + t]];
                targets[w * context + t] = index[corpus[start + t + 1]];
            }
        }

        string[] positions = [.. Enumerable.Range(0, context).Select(t => $"t{t}")];
        return Data.Dataset.FromFlat(features, targets, windows, positions, positions);
    }

    /// <summary>Trains with AdamW, cosine schedule and gradient clipping; returns the config with training results.</summary>
    public static GptConfig Train(CharGpt gpt, string corpus, int epochs, int batchSize, string corpusName, CancellationToken cancellationToken = default)
    {
        var data = Windows(corpus, gpt.Config, maxWindows: 20_000, seed: 3);
        var (train, validation) = data.Split(0.95, seed: 4);
        using var optimizer = new Abstraction.Training.AdamW(gpt.Model.Parameters(), learningRate: 0.002f, weightDecay: 0.01f);
        var trainer = new Training.Trainer(gpt.Model, optimizer, (logits, next) => Losses.SparseCrossEntropy(logits, next))
        {
            Metrics = { Training.Metric.SparseAccuracy },
            Scheduler = new Abstraction.Training.CosineAnnealing(optimizer, epochs, minLearningRate: 2e-4f),
            MaxGradientNorm = 1f,
        };
        var history = trainer.Fit(new Data.DataLoader(train, batchSize, shuffle: true, device: gpt.Device, seed: 5), epochs,
            validation: new Data.DataLoader(validation, 256, device: gpt.Device), cancellationToken);
        var last = history.Epochs[^1];
        return gpt.Config with
        {
            TrainedEpochs = history.Epochs.Count,
            ValidationLoss = last.ValidationLoss,
            ValidationAccuracy = last.ValidationMetrics?["accuracy"],
            TrainedAt = DateTimeOffset.UtcNow,
            Corpus = corpusName,
        };
    }
}
