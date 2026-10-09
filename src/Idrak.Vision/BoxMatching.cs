// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Vision.Abstractions;

namespace Idrak.Vision;

/// <summary>
/// Matching predictions to true objects (the library's <see cref="BoxMatchers"/>) and the quality matrices they read:
/// IoU or GIoU between two lists of boxes, and the cost a set predictor's outputs are matched by.
/// </summary>
public static class BoxMatching
{
    /// <summary>The IoU of every prediction with every truth, [predictions, truths] row-major.</summary>
    public static float[] IouMatrix(IReadOnlyList<BoundingBox> predictions, IReadOnlyList<BoundingBox> truths) => Matrix(predictions, truths, generalized: false);

    /// <summary>The generalized IoU (in [-1, 1]) of every prediction with every truth, [predictions, truths] row-major.</summary>
    public static float[] GeneralizedIouMatrix(IReadOnlyList<BoundingBox> predictions, IReadOnlyList<BoundingBox> truths) => Matrix(predictions, truths, generalized: true);

    private static float[] Matrix(IReadOnlyList<BoundingBox> predictions, IReadOnlyList<BoundingBox> truths, bool generalized)
    {
        ArgumentNullException.ThrowIfNull(predictions);
        ArgumentNullException.ThrowIfNull(truths);
        var result = new float[predictions.Count * truths.Count];
        for (int p = 0; p < predictions.Count; p++)
        {
            for (int g = 0; g < truths.Count; g++)
            {
                result[p * truths.Count + g] = generalized ? GeneralizedIou(predictions[p], truths[g]) : predictions[p].IntersectionOverUnion(truths[g]);
            }
        }

        return result;
    }

    /// <summary>The generalized IoU of two boxes: the IoU less the share of their enclosing box neither covers.</summary>
    public static float GeneralizedIou(BoundingBox a, BoundingBox b)
    {
        float intersection = a.IntersectionArea(b), union = a.Area + b.Area - intersection;
        float enclosing = (Math.Max(a.Right, b.Right) - Math.Min(a.X, b.X)) * (Math.Max(a.Bottom, b.Bottom) - Math.Min(a.Y, b.Y));
        float iou = union > 0 ? intersection / union : 0;
        return enclosing > 0 ? iou - (enclosing - union) / enclosing : iou;
    }

    /// <summary>
    /// The quality a set predictor's outputs are matched to the true objects by (higher is better), [predictions,
    /// truths]: the negated cost <c>classWeight · -p(truth's class) + l1Weight · L1(corners / scale) + giouWeight · -GIoU</c>,
    /// where <paramref name="classProbabilities"/> is [predictions, classes] (a softmax or sigmoid of the scores) and
    /// <paramref name="scale"/> divides the corners (the image's size, so the L1 term is in image fractions).
    /// </summary>
    public static float[] SetPredictionQuality(ReadOnlySpan<float> classProbabilities, int classes, IReadOnlyList<BoundingBox> predicted,
        IReadOnlyList<int> truthClasses, IReadOnlyList<BoundingBox> truthBoxes, float classWeight = 1f, float l1Weight = 5f, float giouWeight = 2f, float scale = 1f)
    {
        ArgumentNullException.ThrowIfNull(predicted);
        ArgumentNullException.ThrowIfNull(truthClasses);
        ArgumentNullException.ThrowIfNull(truthBoxes);
        int p = predicted.Count, g = truthBoxes.Count;
        if (classProbabilities.Length != p * classes || truthClasses.Count != g || truthClasses.Any(c => (uint)c >= (uint)classes) || !(scale > 0))
        {
            throw new ArgumentException($"Set matching takes [{p}, {classes}] class probabilities, a class in [0, {classes}) for each of the {g} truths and a positive scale.");
        }

        var quality = new float[p * g];
        for (int i = 0; i < p; i++)
        {
            var a = predicted[i];
            for (int j = 0; j < g; j++)
            {
                var b = truthBoxes[j];
                float l1 = (Math.Abs(a.X - b.X) + Math.Abs(a.Y - b.Y) + Math.Abs(a.Right - b.Right) + Math.Abs(a.Bottom - b.Bottom)) / scale;
                float cost = -classWeight * classProbabilities[i * classes + truthClasses[j]] + l1Weight * l1 - giouWeight * GeneralizedIou(a, b);
                quality[i * g + j] = -cost;
            }
        }

        return quality;
    }

    /// <summary>
    /// "iou-threshold" (torchvision's <c>Matcher</c>): each prediction to its best truth (the first of equals); below
    /// <see cref="BoxMatchOptions.NegativeThreshold"/> a negative, below <see cref="BoxMatchOptions.PositiveThreshold"/>
    /// ignored; with <see cref="BoxMatchOptions.AllowLowQuality"/> the predictions of each truth's best quality keep their
    /// best truth whatever the thresholds.
    /// </summary>
    public static int[] MatchByThreshold(ReadOnlyMemory<float> quality, int predictions, int truths, BoxMatchOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (!(options.NegativeThreshold <= options.PositiveThreshold))
        {
            throw new ArgumentException($"The negative threshold ({options.NegativeThreshold}) is at most the positive one ({options.PositiveThreshold}).", nameof(options));
        }

        var q = quality.Span;
        var matches = new int[predictions];
        var best = new int[predictions];
        for (int p = 0; p < predictions; p++)
        {
            int argmax = -1;
            float max = float.NegativeInfinity;
            for (int g = 0; g < truths; g++)
            {
                if (q[p * truths + g] > max || argmax < 0 && !float.IsNaN(q[p * truths + g]))
                {
                    (max, argmax) = (q[p * truths + g], g);
                }
            }

            best[p] = argmax;
            matches[p] = argmax < 0 || float.IsNegativeInfinity(max) || max < options.NegativeThreshold ? BoxMatchers.Negative
                : max < options.PositiveThreshold ? BoxMatchers.Ignored : argmax;
        }

        if (options.AllowLowQuality)
        {
            for (int g = 0; g < truths; g++)
            {
                float max = float.NegativeInfinity;
                for (int p = 0; p < predictions; p++)
                {
                    max = Math.Max(max, q[p * truths + g]);
                }

                for (int p = 0; p < predictions && !float.IsNegativeInfinity(max); p++)
                {
                    if (q[p * truths + g] == max)
                    {
                        matches[p] = best[p];
                    }
                }
            }
        }

        return matches;
    }

    /// <summary>
    /// "hungarian": the one-to-one assignment of the largest total quality (min(predictions, truths) pairs; the other
    /// predictions are negatives), by the Hungarian algorithm with potentials (Kuhn 1955, Munkres 1957) in O(n²·m), n the
    /// smaller side. Pairs of quality -∞ are never matched.
    /// </summary>
    public static int[] MatchOneToOne(ReadOnlyMemory<float> quality, int predictions, int truths, BoxMatchOptions options)
    {
        var matches = new int[predictions];
        Array.Fill(matches, BoxMatchers.Negative);
        if (predictions == 0 || truths == 0)
        {
            return matches;
        }

        var q = quality.Span;
        bool transposed = predictions > truths;
        int rows = transposed ? truths : predictions, columns = transposed ? predictions : truths;
        double finiteMax = 0;
        foreach (float v in q)
        {
            if (float.IsNaN(v) || float.IsPositiveInfinity(v))
            {
                throw new ArgumentException("Match qualities are finite numbers or -∞ (a forbidden pair).", nameof(quality));
            }

            if (!float.IsNegativeInfinity(v))
            {
                finiteMax = Math.Max(finiteMax, Math.Abs(v));
            }
        }

        // Costs (rows x columns, 1-based for the algorithm): the negated quality; a forbidden pair costs more than any
        // assignment of allowed pairs could.
        double forbidden = (finiteMax * 2 + 1) * (rows + 1);
        var cost = new double[(rows + 1) * (columns + 1)];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < columns; c++)
            {
                float v = transposed ? q[c * truths + r] : q[r * truths + c];
                cost[(r + 1) * (columns + 1) + c + 1] = float.IsNegativeInfinity(v) ? forbidden : -v;
            }
        }

        var assigned = Assign(cost, rows, columns);
        for (int c = 1; c <= columns; c++)
        {
            int r = assigned[c];
            if (r == 0)
            {
                continue;
            }

            int prediction = transposed ? c - 1 : r - 1, truth = transposed ? r - 1 : c - 1;
            if (!float.IsNegativeInfinity(q[prediction * truths + truth]))
            {
                matches[prediction] = truth;
            }
        }

        return matches;
    }

    // The Hungarian algorithm for rows <= columns on 1-based costs; returns, per column, its row (0: none).
    private static int[] Assign(double[] cost, int rows, int columns)
    {
        int stride = columns + 1;
        var u = new double[rows + 1];
        var v = new double[columns + 1];
        var row = new int[columns + 1];
        var way = new int[columns + 1];
        var minimum = new double[columns + 1];
        var used = new bool[columns + 1];
        for (int i = 1; i <= rows; i++)
        {
            row[0] = i;
            int j0 = 0;
            Array.Fill(minimum, double.PositiveInfinity);
            Array.Clear(used);
            do
            {
                used[j0] = true;
                int i0 = row[j0], j1 = 0;
                double delta = double.PositiveInfinity;
                for (int j = 1; j <= columns; j++)
                {
                    if (used[j])
                    {
                        continue;
                    }

                    double reduced = cost[i0 * stride + j] - u[i0] - v[j];
                    if (reduced < minimum[j])
                    {
                        (minimum[j], way[j]) = (reduced, j0);
                    }

                    if (minimum[j] < delta)
                    {
                        (delta, j1) = (minimum[j], j);
                    }
                }

                for (int j = 0; j <= columns; j++)
                {
                    if (used[j])
                    {
                        u[row[j]] += delta;
                        v[j] -= delta;
                    }
                    else
                    {
                        minimum[j] -= delta;
                    }
                }

                j0 = j1;
            }
            while (row[j0] != 0);

            do
            {
                int j1 = way[j0];
                row[j0] = row[j1];
                j0 = j1;
            }
            while (j0 != 0);
        }

        return row;
    }
}
