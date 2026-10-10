// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Vision.Abstractions;

namespace Idrak.Vision;

/// <summary>How <see cref="DetectionObjective"/> matches and scores a detector's candidates; every name is a registry's.</summary>
public sealed record DetectionObjectiveOptions
{
    /// <summary>
    /// The <see cref="BoxMatchers"/> entry that assigns candidates to true objects (default "iou-threshold", which reads
    /// the IoU; "hungarian" reads the set-prediction quality, <see cref="BoxMatching.SetPredictionQuality"/>).
    /// </summary>
    public string Matcher { get; init; } = BoxMatchers.Threshold;

    /// <summary>The matcher's settings (default: thresholds 0.5 and 0.4, every truth keeping its best candidates).</summary>
    public BoxMatchOptions Match { get; init; } = new() { AllowLowQuality = true };

    /// <summary>The <see cref="VisionLosses"/> entry for the matched candidates' boxes against their truths' (default "giou").</summary>
    public string BoxLoss { get; init; } = "giou";

    /// <summary>The <see cref="VisionLosses"/> entry for the class logits against one-hot targets (default "focal").</summary>
    public string ClassLoss { get; init; } = "focal";

    /// <summary>The losses' settings (their reduction is ignored: each candidate's loss is weighted and summed here).</summary>
    public VisionLossOptions Losses { get; init; } = new();

    /// <summary>The weight of the box term (default 1).</summary>
    public float BoxWeight { get; init; } = 1f;

    /// <summary>The weight of the class term (default 1).</summary>
    public float ClassWeight { get; init; } = 1f;

    /// <summary>The weight of the objectness term, when the head has one (default 1).</summary>
    public float ObjectnessWeight { get; init; } = 1f;

    /// <summary>Set matching: what the L1 term of the quality divides the corners by (the network input's size; default 1).</summary>
    public float Scale { get; init; } = 1f;
}

/// <summary>
/// The training loss of a detector: its candidates (<see cref="DetectionCandidates"/>, from the <see cref="DetectionHeads"/>
/// entry of its decoder) matched to each image's true boxes by a registered <see cref="BoxMatchers">matcher</see>, then the
/// matched candidates' boxes scored by a registered <see cref="VisionLosses">box loss</see> and the class logits by a
/// registered class loss (one-hot targets: the matched truth's class, nothing for background), each summed over the
/// candidates and divided by the matched count (at least 1). With an objectness logit the class loss reads the matched
/// candidates only, and a binary cross-entropy teaches the objectness (matched 1, background 0, averaged over both);
/// without one, every candidate not ignored learns its class scores. Candidates the matcher ignores count nowhere.
/// </summary>
public static class DetectionObjective
{
    /// <summary>
    /// The loss (a scalar with gradients to the candidates) of <paramref name="candidates"/> against each image's
    /// <paramref name="boxes"/> (in the network input's pixels) and <paramref name="labels"/>.
    /// </summary>
    /// <exception cref="ArgumentException">The images disagree in number, or a label is not one of the candidates' classes.</exception>
    public static Tensor Loss(DetectionCandidates candidates, IReadOnlyList<IReadOnlyList<BoundingBox>> boxes, IReadOnlyList<IReadOnlyList<int>> labels,
        DetectionObjectiveOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(candidates);
        ArgumentNullException.ThrowIfNull(boxes);
        ArgumentNullException.ThrowIfNull(labels);
        options ??= new DetectionObjectiveOptions();
        int n = candidates.Images, p = candidates.Candidates, c = candidates.Classes, rows = n * p;
        if (candidates.Boxes.Rank != 3 || candidates.Boxes.Shape[2] != 4 || candidates.ClassLogits.Rank != 3 || candidates.ClassLogits.Shape[0] != n
            || candidates.ClassLogits.Shape[1] != p || candidates.Objectness is { } o && (o.Rank != 2 || o.Shape[0] != n || o.Shape[1] != p))
        {
            throw new ArgumentException("Detection candidates are boxes [N, P, 4], class logits [N, P, classes] and an objectness [N, P] or none.", nameof(candidates));
        }

        if (boxes.Count != n || labels.Count != n)
        {
            throw new ArgumentException($"{n} images of candidates, {boxes.Count} of boxes and {labels.Count} of labels.");
        }

        var matcher = BoxMatchers.Get(options.Matcher);
        bool set = string.Equals(options.Matcher, BoxMatchers.Hungarian, StringComparison.OrdinalIgnoreCase);
        float[] predicted = candidates.Boxes.ToArray();
        float[]? probabilities = set ? [.. candidates.ClassLogits.ToArray().Select(v => 1f / (1f + MathF.Exp(-v)))] : null;
        bool objectness = candidates.Objectness is not null;

        var boxTargets = new float[rows * 4];
        var boxWeights = new float[rows];
        var classTargets = new float[rows * c];
        var classWeights = new float[rows * c];
        var objectTargets = new float[rows];
        var objectWeights = new float[rows];
        int matched = 0;
        for (int i = 0; i < n; i++)
        {
            var truths = boxes[i];
            var classes = labels[i];
            if (truths.Count != classes.Count || classes.Any(k => (uint)k >= (uint)c))
            {
                throw new ArgumentException($"Image {i}: {truths.Count} boxes and {classes.Count} labels, each a class below {c}.");
            }

            var mine = new BoundingBox[p];
            for (int k = 0; k < p; k++)
            {
                int at = (i * p + k) * 4;
                mine[k] = BoundingBox.FromCorners(predicted[at], predicted[at + 1], predicted[at + 2], predicted[at + 3]);
            }

            int[] match;
            if (truths.Count == 0)
            {
                match = new int[p];
                Array.Fill(match, BoxMatchers.Negative);
            }
            else
            {
                var quality = set
                    ? BoxMatching.SetPredictionQuality(probabilities.AsSpan(i * p * c, p * c), c, mine, classes, truths, scale: options.Scale)
                    : Overlaps(mine, truths);
                match = matcher(quality, p, truths.Count, options.Match);
            }

            for (int k = 0; k < p; k++)
            {
                int row = i * p + k, m = match[k];
                if (m < 0)
                {
                    predicted.AsSpan(4 * row, 4).CopyTo(boxTargets.AsSpan(4 * row));   // its own box: not scored (weight 0), and finite
                }

                if (m >= 0)
                {
                    var truth = truths[m];
                    (boxTargets[4 * row], boxTargets[4 * row + 1], boxTargets[4 * row + 2], boxTargets[4 * row + 3]) = (truth.X, truth.Y, truth.Right, truth.Bottom);
                    matched++;
                    boxWeights[row] = 1f;
                    objectTargets[row] = 1f;
                    objectWeights[row] = 1f;
                    classTargets[row * c + classes[m]] = 1f;
                    classWeights.AsSpan(row * c, c).Fill(1f);
                }
                else if (m == BoxMatchers.Negative)
                {
                    objectWeights[row] = 1f;
                    if (!objectness)
                    {
                        classWeights.AsSpan(row * c, c).Fill(1f);
                    }
                }
            }
        }

        // The tensors below join the step's graph (as a composed loss's do): the caller's tensor scope releases them after
        // the backward pass.
        var device = candidates.Boxes.Device;
        float normalizer = Math.Max(1, matched);
        var lossOptions = options.Losses with { Reduction = LossReduction.None };
        var boxLoss = VisionLosses.Get(options.BoxLoss)(candidates.Boxes.Reshape(rows, 4), Tensor.From(boxTargets, [rows, 4], device), lossOptions);
        var classLoss = VisionLosses.Get(options.ClassLoss)(candidates.ClassLogits.Reshape(rows, c), Tensor.From(classTargets, [rows, c], device), lossOptions);
        var total = Weighted(boxLoss, boxWeights, rows, device) * (options.BoxWeight / normalizer)
                    + Weighted(classLoss, classWeights, rows, device) * (options.ClassWeight / normalizer);
        if (candidates.Objectness is not { } logits)
        {
            return total;
        }

        var objectLoss = DetectionLosses.Focal(logits, Tensor.From(objectTargets, [n, p], device), alpha: -1f, gamma: 0f, LossReduction.None);   // binary cross-entropy
        return total + Weighted(objectLoss, objectWeights, rows, device) * (options.ObjectnessWeight / Math.Max(1f, objectWeights.Sum()));
    }

    // Σ loss · weight: the weights given per value of the loss, or per candidate row and spread over that row's values.
    private static Tensor Weighted(Tensor loss, float[] weights, int rows, Device device)
    {
        int per = loss.Size / Math.Max(1, rows);
        if (per * rows != loss.Size || weights.Length != loss.Size && weights.Length != rows)
        {
            throw new InvalidOperationException($"A loss of {loss.Size} values for {rows} candidates.");
        }

        float[] spread = weights.Length == loss.Size ? weights : [.. Enumerable.Range(0, loss.Size).Select(j => weights[j / per])];
        return (loss * Tensor.From(spread, loss.Shape, device)).Sum();
    }

    // The IoU of every candidate with every truth; a truth that no candidate overlaps is scored by its generalized IoU less
    // 1 (below 0, so only a matcher's low-quality rule can match it, to its nearest candidate).
    private static float[] Overlaps(IReadOnlyList<BoundingBox> candidates, IReadOnlyList<BoundingBox> truths)
    {
        var quality = BoxMatching.IouMatrix(candidates, truths);
        int g = truths.Count;
        for (int j = 0; j < g; j++)
        {
            bool overlapped = false;
            for (int k = 0; k < candidates.Count && !overlapped; k++)
            {
                overlapped = quality[k * g + j] > 0;
            }

            if (!overlapped)
            {
                for (int k = 0; k < candidates.Count; k++)
                {
                    quality[k * g + j] = BoxMatching.GeneralizedIou(candidates[k], truths[j]) - 1f;
                }
            }
        }

        return quality;
    }
}
