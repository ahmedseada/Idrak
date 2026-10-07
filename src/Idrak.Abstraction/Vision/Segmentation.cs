// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.


namespace Idrak.Abstraction.Vision;

/// <summary>A class for every pixel of an image (semantic segmentation): row by row, 0 to <see cref="ClassCount"/> - 1.</summary>
public sealed class SegmentationMask
{
    private readonly int[] _labels;

    /// <summary>Wraps per-pixel class indices (row by row).</summary>
    public SegmentationMask(int[] labels, int width, int height, int classCount)
    {
        ArgumentNullException.ThrowIfNull(labels);
        if (labels.Length != width * height)
        {
            throw new ArgumentException($"{labels.Length} labels for a {width} x {height} mask.", nameof(labels));
        }

        _labels = labels;
        Width = width;
        Height = height;
        ClassCount = classCount;
    }

    /// <summary>The mask's width.</summary>
    public int Width { get; }

    /// <summary>The mask's height.</summary>
    public int Height { get; }

    /// <summary>The number of classes.</summary>
    public int ClassCount { get; }

    /// <summary>The pixels' classes, row by row.</summary>
    public ReadOnlySpan<int> Labels => _labels;

    /// <summary>The class of the pixel at (<paramref name="x"/>, <paramref name="y"/>).</summary>
    public int LabelAt(int x, int y) => _labels[y * Width + x];

    /// <summary>The number of pixels of each class.</summary>
    public int[] Counts()
    {
        var counts = new int[ClassCount];
        foreach (int label in _labels)
        {
            counts[label]++;
        }

        return counts;
    }

    /// <summary>The mask at another size, each pixel taking the class of the pixel under its centre (nearest neighbour).</summary>
    public SegmentationMask Resize(int width, int height)
    {
        var labels = new int[width * height];
        for (int y = 0; y < height; y++)
        {
            int sy = Math.Min(Height - 1, (int)((y + 0.5) * Height / height));
            for (int x = 0; x < width; x++)
            {
                labels[y * width + x] = _labels[sy * Width + Math.Min(Width - 1, (int)((x + 0.5) * Width / width))];
            }
        }

        return new SegmentationMask(labels, width, height, ClassCount);
    }

    /// <summary>One image's mask from its logits [classes, height, width]: each pixel's likeliest class.</summary>
    public static SegmentationMask FromLogits(ReadOnlySpan<float> logits, int classes, int height, int width)
    {
        int plane = height * width;
        if (logits.Length != classes * plane)
        {
            throw new ArgumentException($"{logits.Length} logits for {classes} x {height} x {width}.", nameof(logits));
        }

        var labels = new int[plane];
        for (int i = 0; i < plane; i++)
        {
            int best = 0;
            float max = logits[i];
            for (int c = 1; c < classes; c++)
            {
                float v = logits[c * plane + i];
                if (v > max)
                {
                    (max, best) = (v, c);
                }
            }

            labels[i] = best;
        }

        return new SegmentationMask(labels, width, height, classes);
    }

    /// <summary>The masks of a batch of logits [N, classes, height, width].</summary>
    public static SegmentationMask[] FromLogits(Tensor logits)
    {
        ArgumentNullException.ThrowIfNull(logits);
        if (logits.Rank != 4)
        {
            throw new ArgumentException($"Segmentation logits are [N, classes, height, width], not {Tensor.FormatShape(logits.Shape)}.", nameof(logits));
        }

        int n = logits.Shape[0], classes = logits.Shape[1], h = logits.Shape[2], w = logits.Shape[3], per = classes * h * w;
        var values = logits.ToArray();
        return [.. Enumerable.Range(0, n).Select(i => FromLogits(values.AsSpan(i * per, per), classes, h, w))];
    }
}

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

/// <summary>Gives every pixel of an image a class. Implement it over any segmentation network; <see cref="ModelSegmenter"/> is one.</summary>
public interface ISegmenter
{
    /// <summary>The mask of <paramref name="image"/>, at its size.</summary>
    SegmentationMask Segment(ImageData image);
}

/// <summary>
/// An <see cref="ISegmenter"/> over any network that maps images [N, channels, height, width] (values in [0, 1]; put
/// normalization in the network) to logits [N, classes, h, w]: each image is resized to the network's input, the
/// masks come from the logits, and are resized back to the image (nearest neighbour).
/// </summary>
public sealed class ModelSegmenter(Module model, int channels, int height, int width, Device? device = null) : ISegmenter
{
    private readonly Module _model = model ?? throw new ArgumentNullException(nameof(model));
    private readonly Device _device = device ?? model.WeightsDevice ?? Device.Default;

    /// <inheritdoc />
    public SegmentationMask Segment(ImageData image) => Segment([image])[0];

    /// <summary>The masks of several images, run through the network as one batch.</summary>
    public IReadOnlyList<SegmentationMask> Segment(IReadOnlyList<ImageData> images)
    {
        ArgumentNullException.ThrowIfNull(images);
        if (images.Count == 0)
        {
            return [];
        }

        int input = channels * height * width;
        var batch = new float[images.Count * input];
        for (int i = 0; i < images.Count; i++)
        {
            images[i].Resize(channels, height, width, batch.AsSpan(i * input, input));
        }

        using var x = Tensor.From(batch, [images.Count, channels, height, width], _device);
        using var logits = _model.Predict(x);
        var masks = SegmentationMask.FromLogits(logits);
        return [.. masks.Select((m, i) => m.Width == images[i].Width && m.Height == images[i].Height ? m : m.Resize(images[i].Width, images[i].Height))];
    }
}
