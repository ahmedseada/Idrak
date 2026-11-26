// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Vision.Abstractions;

/// <summary>
/// A measure of a detector or a segmenter over a test set: add its answers image by image, then <see cref="Compute"/>.
/// <see cref="IDetectionMetric"/> takes detections, <see cref="ISegmentationMetric"/> masks.
/// </summary>
public interface IVisionMetric
{
    /// <summary>The name it is registered under (<c>coco</c>).</summary>
    string Name { get; }

    /// <summary>Forgets every image added.</summary>
    void Reset();

    /// <summary>
    /// The values over every image added so far, by name, the metric's headline value first (<c>map</c> for detection,
    /// <c>miou</c> for segmentation). A value that has no images to stand on is NaN.
    /// </summary>
    IReadOnlyDictionary<string, double> Compute();
}

/// <summary>A detection metric: the detections of each image against its true objects.</summary>
public interface IDetectionMetric : IVisionMetric
{
    /// <summary>
    /// Adds one image: the detector's <paramref name="predictions"/> and the <paramref name="truths"/> (crowd and difficult
    /// objects count neither way, as COCO and Pascal VOC have it).
    /// </summary>
    void Add(IReadOnlyList<Detection> predictions, IReadOnlyList<ObjectAnnotation> truths);
}

/// <summary>A segmentation metric: each image's predicted mask against its true one.</summary>
public interface ISegmentationMetric : IVisionMetric
{
    /// <summary>Adds one image's predicted and true masks (the same size).</summary>
    void Add(SegmentationMask predicted, SegmentationMask expected);
}

/// <summary>What a metric is created with (<c>IVisionMetricKind</c>, <c>VisionMetrics.Create</c>); each reads what it needs.</summary>
public sealed record VisionMetricOptions
{
    /// <summary>The number of classes (segmentation: required; detection: 0 to take the classes from the images).</summary>
    public int Classes { get; init; }

    /// <summary>Segmentation: pixels of this true class are left out (as a "void" label), or null for none.</summary>
    public int? IgnoreClass { get; init; }

    /// <summary>Detection: the most detections of an image that count, the best first (COCO's 100).</summary>
    public int MaxDetections { get; init; } = 100;

    /// <summary>The class names, for the per-class values' keys (<c>ap/cat</c>); the indices when null.</summary>
    public IReadOnlyList<string>? ClassNames { get; init; }
}

