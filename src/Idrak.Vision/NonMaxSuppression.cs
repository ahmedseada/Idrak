// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Vision.Abstractions;

namespace Idrak.Vision;

/// <summary>
/// Non-maximum suppression: of detections that overlap by more than an intersection-over-union threshold, only the
/// highest-scoring is kept. Detection networks predict many overlapping boxes per object; this keeps one each.
/// </summary>
public static class NonMaxSuppression
{
    /// <summary>
    /// The detections kept, best score first. Below <paramref name="minScore"/> a detection is dropped first;
    /// <paramref name="perClass"/> (the default) only suppresses boxes of the same class; at most
    /// <paramref name="maxDetections"/> are kept.
    /// </summary>
    public static List<Detection> Apply(IReadOnlyList<Detection> detections, float iouThreshold = 0.5f, float minScore = 0f,
        bool perClass = true, int maxDetections = int.MaxValue)
    {
        ArgumentNullException.ThrowIfNull(detections);
        if (iouThreshold is < 0 or > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(iouThreshold), "The threshold is an intersection over union, in [0, 1].");
        }

        var candidates = detections.Where(d => d.Score >= minScore).OrderByDescending(d => d.Score).ToList();
        var kept = new List<Detection>();
        var suppressed = new bool[candidates.Count];
        for (int i = 0; i < candidates.Count && kept.Count < maxDetections; i++)
        {
            if (suppressed[i])
            {
                continue;
            }

            var best = candidates[i];
            kept.Add(best);
            for (int j = i + 1; j < candidates.Count; j++)
            {
                if (!suppressed[j] && (!perClass || candidates[j].Class == best.Class)
                    && best.Box.IntersectionOverUnion(candidates[j].Box) > iouThreshold)
                {
                    suppressed[j] = true;
                }
            }
        }

        return kept;
    }
}
