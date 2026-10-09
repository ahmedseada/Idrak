// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Vision.Abstractions;

namespace Idrak.Vision;

/// <summary>How well predicted masks match the true ones: pixel accuracy and each class's intersection over union.</summary>
/// <param name="PixelAccuracy">The share of pixels given their true class.</param>
/// <param name="ClassIoU">Each class's intersection over union; NaN for a class in neither mask.</param>
/// <param name="MeanIoU">The mean of the classes' IoU that are not NaN.</param>
public sealed record SegmentationScore(double PixelAccuracy, IReadOnlyList<double> ClassIoU, double MeanIoU)
{
    /// <summary>Each class's accuracy (the share of its true pixels predicted as it); NaN for a class with no true pixel.</summary>
    public IReadOnlyList<double> ClassAccuracy { get; init; } = [];

    /// <summary>The mean of the classes' accuracy that are not NaN.</summary>
    public double MeanAccuracy { get; init; } = double.NaN;

    /// <summary>The classes' IoU weighted by how many true pixels each has.</summary>
    public double FrequencyWeightedIoU { get; init; } = double.NaN;
}

/// <summary>
/// Scores segmentation masks ("miou" in <see cref="VisionMetrics"/>); add any number of (predicted, true) pairs, then read
/// <see cref="Score"/> (or <see cref="Compute()"/>: <c>miou</c>, <c>pixel_accuracy</c>, <c>mean_accuracy</c>, <c>fwiou</c>,
/// then <c>iou/CLASS</c> per class). Pixels whose true class is <paramref name="ignoreClass"/> are left out.
/// </summary>
/// <param name="classCount">The number of classes.</param>
/// <param name="ignoreClass">A true class whose pixels are left out (a "void" label), or null.</param>
/// <param name="classNames">The class names for <see cref="Compute()"/>'s keys; the indices when null.</param>
public sealed class SegmentationMetrics(int classCount, int? ignoreClass = null, IReadOnlyList<string>? classNames = null) : ISegmentationMetric
{
    private readonly long[] _intersection = new long[classCount];
    private readonly long[] _predicted = new long[classCount];
    private readonly long[] _expected = new long[classCount];
    private long _correct, _pixels;

    /// <inheritdoc />
    public string Name => "miou";

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
            if (e[i] == ignoreClass)
            {
                continue;
            }

            if ((uint)p[i] >= (uint)classCount || (uint)e[i] >= (uint)classCount)
            {
                throw new ArgumentException($"Pixel {i}: class {p[i]} predicted, {e[i]} true; the metric has {classCount} classes.");
            }

            _predicted[p[i]]++;
            _expected[e[i]]++;
            if (p[i] == e[i])
            {
                _intersection[p[i]]++;
                _correct++;
            }

            _pixels++;
        }

        return this;
    }

    void ISegmentationMetric.Add(SegmentationMask predicted, SegmentationMask expected) => Add(predicted, expected);

    /// <inheritdoc />
    public void Reset()
    {
        Array.Clear(_intersection);
        Array.Clear(_predicted);
        Array.Clear(_expected);
        (_correct, _pixels) = (0, 0);
    }

    /// <summary>The score of every pair added so far.</summary>
    public SegmentationScore Score()
    {
        var iou = new double[classCount];
        var accuracy = new double[classCount];
        double weighted = 0;
        for (int c = 0; c < classCount; c++)
        {
            long union = _predicted[c] + _expected[c] - _intersection[c];
            iou[c] = union > 0 ? (double)_intersection[c] / union : double.NaN;
            accuracy[c] = _expected[c] > 0 ? (double)_intersection[c] / _expected[c] : double.NaN;
            weighted += _expected[c] > 0 ? (double)_expected[c] / _pixels * iou[c] : 0;
        }

        return new SegmentationScore(_pixels > 0 ? (double)_correct / _pixels : 0, iou, Mean(iou))
        {
            ClassAccuracy = accuracy,
            MeanAccuracy = Mean(accuracy),
            FrequencyWeightedIoU = _pixels > 0 ? weighted : double.NaN,
        };

        static double Mean(double[] values)
        {
            var present = values.Where(v => !double.IsNaN(v)).ToArray();
            return present.Length > 0 ? present.Average() : double.NaN;
        }
    }

    /// <inheritdoc />
    public IReadOnlyDictionary<string, double> Compute()
    {
        var score = Score();
        var result = new Dictionary<string, double>
        {
            ["miou"] = score.MeanIoU,
            ["pixel_accuracy"] = _pixels > 0 ? score.PixelAccuracy : double.NaN,
            ["mean_accuracy"] = score.MeanAccuracy,
            ["fwiou"] = score.FrequencyWeightedIoU,
        };
        for (int c = 0; c < classCount; c++)
        {
            result["iou/" + CocoAveragePrecision.ClassName(classNames, c)] = score.ClassIoU[c];
        }

        return result;
    }

    /// <summary>The score of one pair.</summary>
    public static SegmentationScore Compute(SegmentationMask predicted, SegmentationMask expected) =>
        new SegmentationMetrics(Math.Max(predicted.ClassCount, expected.ClassCount)).Add(predicted, expected).Score();
}
