// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Data;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Training;

namespace Idrak.Cli.Shared;

/// <summary>One candidate of <c>--search</c>: its sizes, its score on the held-out data, and how long it took.</summary>
internal sealed record SearchResult(int Index, Variant Variant, long Parameters, double Score, string Metric, double ValidationLoss, double Seconds, string? Error);

/// <summary>
/// The optional measuring step of <c>idrak suggest --search N</c>: trains each candidate (<see cref="DesignRules.Candidates"/>)
/// briefly on the chosen device with the same prepared data, split and seed, and scores it on the held-out part:
/// accuracy for classes, root mean squared error (of the scaled target) for numbers, loss for language models. The
/// best score wins; a tie keeps the earlier candidate, so the first guess wins unless another one is better.
/// </summary>
internal static class DesignSearch
{
    /// <summary>Whether a higher score is better for <paramref name="metric"/>.</summary>
    public static bool HigherIsBetter(string metric) => metric == "accuracy";

    /// <summary>Trains and scores <paramref name="candidates"/>; <paramref name="progress"/> hears about each one.</summary>
    public static List<SearchResult> Run(DesignPlan design, DataPreparation.Prepared data, IReadOnlyList<Variant> candidates, Device device, Action<SearchResult>? progress = null)
    {
        var train = design.Train!;
        int batch = (int)train["batchSize"]!;
        float decay = (float)train["weightDecay"]!;
        string loss = (string)train["loss"]!;
        string metric = (string)train["metric"]!;

        int epochs = Epochs(design, data);
        var results = new List<SearchResult>();
        for (int i = 0; i < candidates.Count; i++)
        {
            var variant = candidates[i];
            var watch = Stopwatch.StartNew();
            SearchResult result;
            try
            {
                using var model = design.Make!(variant).OnDevice(device).Build();
                using var optimizer = new AdamW(model.Parameters(), learningRate: variant.LearningRate, weightDecay: decay);
                using var trainer = new Trainer(model, optimizer, Loss(loss))
                {
                    Scheduler = new CosineAnnealing(optimizer, epochs),
                    MaxGradientNorm = 1f,
                };
                trainer.Fit(new DataLoader(data.Train, batch, shuffle: true, device: device, seed: (int?)train["seed"] ?? 1), epochs);
                double validationLoss = trainer.Evaluate(new DataLoader(data.Validation, 256, device: device)).Loss;
                double score = metric switch
                {
                    "accuracy" => Accuracy(trainer.Predict(data.Validation, 256), data.Validation),
                    "rmse" => Math.Sqrt(validationLoss),
                    _ => validationLoss,
                };
                result = new SearchResult(i + 1, variant, model.ParameterCount, score, metric, validationLoss, watch.Elapsed.TotalSeconds, null);
            }
            catch (Exception e) when (e is InvalidOperationException or ResourceLimitExceededException or ArgumentException)
            {
                result = new SearchResult(i + 1, variant, 0, double.NaN, metric, double.NaN, watch.Elapsed.TotalSeconds, e.Message);
            }

            results.Add(result);
            progress?.Invoke(result);
        }

        return results;
    }

    /// <summary>The best result: the highest accuracy or lowest error, the lower validation loss on a tie, then the earlier one.</summary>
    public static SearchResult? Best(IReadOnlyList<SearchResult> results)
    {
        SearchResult? best = null;
        foreach (var r in results.Where(r => r.Error is null && double.IsFinite(r.Score)))
        {
            if (best is null || Better(r, best))
            {
                best = r;
            }
        }

        return best;
    }

    /// <summary>The epochs each candidate trains for: about 200 steps, between 3 and 20 epochs, never more than the full setup's.</summary>
    public static int Epochs(DesignPlan design, DataPreparation.Prepared data) =>
        Math.Clamp((int)Math.Ceiling(200.0 * (int)design.Train!["batchSize"]! / Math.Max(1, data.Train.Count)), 3, Math.Min(20, (int)design.Train["epochs"]!));

    /// <summary>The loss function a train.json names.</summary>
    public static Func<Tensor, Tensor, Tensor> Loss(string name) => name switch
    {
        "mse" => Losses.MeanSquaredError,
        "cross-entropy" => (logits, targets) => Losses.CrossEntropy(logits, targets),
        "token-cross-entropy" => (logits, targets) => Losses.SparseCrossEntropy(logits, targets),
        _ => throw new InvalidDataException($"Unknown loss '{name}' (mse, cross-entropy, token-cross-entropy)."),
    };

    private static bool Better(SearchResult a, SearchResult b)
    {
        bool higher = HigherIsBetter(a.Metric);
        if (a.Score != b.Score)
        {
            return higher ? a.Score > b.Score : a.Score < b.Score;
        }

        return a.ValidationLoss < b.ValidationLoss;
    }

    // The share of rows whose highest output is the target's class.
    private static double Accuracy(float[,] predictions, Idrak.Data.Dataset data)
    {
        int right = 0, classes = data.TargetCount;
        for (int i = 0; i < data.Count; i++)
        {
            int guess = 0;
            for (int c = 1; c < classes; c++)
            {
                guess = predictions[i, c] > predictions[i, guess] ? c : guess;
            }

            right += data.GetTargets(i)[guess] == 1f ? 1 : 0;
        }

        return data.Count == 0 ? 0 : right / (double)data.Count;
    }
}
