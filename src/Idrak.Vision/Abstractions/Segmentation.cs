// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Numerics;

namespace Idrak.Vision.Abstractions;

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
        var columns = new int[width];                               // the source column of each output column, found once
        for (int x = 0; x < width; x++)
        {
            columns[x] = Math.Min(Width - 1, (int)((x + 0.5) * Width / width));
        }

        for (int y = 0; y < height; y++)
        {
            int sy = Math.Min(Height - 1, (int)((y + 0.5) * Height / height));
            var source = _labels.AsSpan(sy * Width, Width);
            var row = labels.AsSpan(y * width, width);
            for (int x = 0; x < row.Length; x++)
            {
                row[x] = source[columns[x]];
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

        // Class plane by class plane, in memory order: each pixel keeps its largest logit so far and its class (the first
        // of equals, as a strict comparison per pixel in class order gives), vectors where the hardware has them.
        var labels = new int[plane];
        var best = ArrayPool<float>.Shared.Rent(plane);
        try
        {
            var max = best.AsSpan(0, plane);
            logits[..plane].CopyTo(max);
            for (int c = 1; c < classes; c++)
            {
                var values = logits.Slice(c * plane, plane);
                int i = 0;
                if (Vector.IsHardwareAccelerated && plane >= Vector<float>.Count)
                {
                    var label = new Vector<int>(c);
                    var classOf = labels.AsSpan();
                    for (; i <= plane - Vector<float>.Count; i += Vector<float>.Count)
                    {
                        var v = new Vector<float>(values[i..]);
                        var m = new Vector<float>(max[i..]);
                        var greater = Vector.GreaterThan(v, m);
                        Vector.ConditionalSelect(greater, v, m).CopyTo(max[i..]);
                        Vector.ConditionalSelect(greater, label, new Vector<int>(classOf[i..])).CopyTo(classOf[i..]);
                    }
                }

                for (; i < plane; i++)
                {
                    if (values[i] > max[i])
                    {
                        (max[i], labels[i]) = (values[i], c);
                    }
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(best);
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

/// <summary>Gives every pixel of an image a class. Implement it over any segmentation network; Idrak.Vision's <c>ModelSegmenter</c> is one.</summary>
public interface ISegmenter
{
    /// <summary>The mask of <paramref name="image"/>, at its size.</summary>
    SegmentationMask Segment(ImageData image);
}
