// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Data;

namespace Idrak.Vision;

/// <summary>
/// Per-channel mean and standard deviation of images, for <see cref="Layers.NetworkBuilder.Normalize"/>: the usual
/// values (ImageNet's), or a data set's own.
/// </summary>
public sealed record ChannelStatistics(IReadOnlyList<float> Mean, IReadOnlyList<float> Std)
{
    /// <summary>ImageNet's RGB statistics (for [0, 1] images), which most pretrained image networks expect.</summary>
    public static ChannelStatistics ImageNet { get; } = new([0.485f, 0.456f, 0.406f], [0.229f, 0.224f, 0.225f]);

    /// <summary>
    /// The statistics of a source's images ([C, H, W] features; [H, W] counts as one channel), read one sample at a time
    /// in double precision. A channel that never varies gets std 1, so normalizing leaves it centred.
    /// </summary>
    public static ChannelStatistics Compute(ISampleSource source)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (source.Count == 0)
        {
            throw new ArgumentException("The source has no samples.", nameof(source));
        }

        int features = source.FeatureShape.Aggregate(1, (a, b) => a * b);
        int targets = Math.Max(1, source.TargetShape.Aggregate(1, (a, b) => a * b));
        var (channels, h, w) = ImageShape.Of(source.FeatureShape, features);
        int plane = h * w;
        var sum = new double[channels];
        var squares = new double[channels];
        var sample = new float[features];
        var target = new float[targets];
        for (int i = 0; i < source.Count; i++)
        {
            source.Read(i, sample, target);
            for (int c = 0; c < channels; c++)
            {
                foreach (float v in sample.AsSpan(c * plane, plane))
                {
                    sum[c] += v;
                    squares[c] += (double)v * v;
                }
            }
        }

        double n = (double)source.Count * plane;
        var mean = new float[channels];
        var std = new float[channels];
        for (int c = 0; c < channels; c++)
        {
            double m = sum[c] / n, variance = Math.Max(squares[c] / n - m * m, 0);
            mean[c] = (float)m;
            std[c] = variance > 1e-12 ? (float)Math.Sqrt(variance) : 1f;
        }

        return new ChannelStatistics(mean, std);
    }
}
