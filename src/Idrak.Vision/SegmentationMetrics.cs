// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Vision.Abstractions;

namespace Idrak.Vision;

/// <summary>How well predicted masks match the true ones: pixel accuracy and each class's intersection over union.</summary>
/// <param name="PixelAccuracy">The share of pixels given their true class.</param>
/// <param name="ClassIoU">Each class's intersection over union; NaN for a class in neither mask.</param>
/// <param name="MeanIoU">The mean of the classes' IoU that are not NaN.</param>
public sealed record SegmentationScore(double PixelAccuracy, IReadOnlyList<double> ClassIoU, double MeanIoU);

/// <summary>Scores segmentation masks; add any number of (predicted, true) pairs, then read <see cref="Score"/>.</summary>
public sealed class SegmentationMetrics(int classCount)
{
    private readonly long[] _intersection = new long[classCount];
    private readonly long[] _predicted = new long[classCount];
    private readonly long[] _expected = new long[classCount];
    private long _correct, _pixels;

    /// <summary>Adds a predicted mask and the true one (same size and classes).</summary>
    public SegmentationMetrics Add(SegmentationMask predicted, SegmentationMask expected)
    {
        ArgumentNullException.ThrowIfNull(predicted);
        ArgumentNullException.ThrowIfNull(expected);
        if (predicted.Width != expected.Width || predicted.Height != expected.Height)
        {
            throw new ArgumentException($"A {predicted.Width} x {predicted.Height} mask against a {expected.Width} x {expected.Height} one.");
        }

        var p = predicted.Labels;
        var e = expected.Labels;
        for (int i = 0; i < p.Length; i++)
        {
            _predicted[p[i]]++;
            _expected[e[i]]++;
            if (p[i] == e[i])
            {
                _intersection[p[i]]++;
                _correct++;
            }
        }

        _pixels += p.Length;
        return this;
    }

    /// <summary>The score of every pair added so far.</summary>
    public SegmentationScore Score()
    {
        var iou = new double[classCount];
        for (int c = 0; c < classCount; c++)
        {
            long union = _predicted[c] + _expected[c] - _intersection[c];
            iou[c] = union > 0 ? (double)_intersection[c] / union : double.NaN;
        }

        var present = iou.Where(v => !double.IsNaN(v)).ToArray();
        return new SegmentationScore(_pixels > 0 ? (double)_correct / _pixels : 0, iou, present.Length > 0 ? present.Average() : double.NaN);
    }

    /// <summary>The score of one pair.</summary>
    public static SegmentationScore Compute(SegmentationMask predicted, SegmentationMask expected) =>
        new SegmentationMetrics(Math.Max(predicted.ClassCount, expected.ClassCount)).Add(predicted, expected).Score();
}
