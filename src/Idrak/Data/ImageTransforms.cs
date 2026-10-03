// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;

namespace Idrak.Data;

/// <summary>
/// Flips images left to right (and, when asked, upside down), each with probability <see cref="Probability"/>. The
/// features are [..., height, width]: every leading dimension (the channels) is flipped alike.
/// </summary>
/// <param name="horizontal">Flip left to right.</param>
/// <param name="vertical">Flip upside down (not for digits or text).</param>
/// <param name="probability">The chance of each flip.</param>
public sealed class RandomFlip(bool horizontal = true, bool vertical = false, double probability = 0.5) : ISampleTransform
{
    /// <summary>Whether images are flipped left to right.</summary>
    public bool Horizontal { get; } = horizontal;

    /// <summary>Whether images are flipped upside down.</summary>
    public bool Vertical { get; } = vertical;

    /// <summary>The chance of each flip.</summary>
    public double Probability { get; } = probability is >= 0 and <= 1 ? probability : throw new ArgumentOutOfRangeException(nameof(probability));

    /// <inheritdoc />
    public void Apply(Span<float> features, Span<float> targets, IReadOnlyList<int> featureShape, Random random)
    {
        var (planes, h, w) = ImageShape.Of(featureShape, features.Length);
        bool flipX = Horizontal && random.NextDouble() < Probability, flipY = Vertical && random.NextDouble() < Probability;
        for (int p = 0; p < planes; p++)
        {
            var plane = features.Slice(p * h * w, h * w);
            for (int y = 0; flipX && y < h; y++)
            {
                plane.Slice(y * w, w).Reverse();
            }

            for (int y = 0; flipY && y < h / 2; y++)
            {
                var top = plane.Slice(y * w, w);
                var bottom = plane.Slice((h - 1 - y) * w, w);
                for (int x = 0; x < w; x++)
                {
                    (top[x], bottom[x]) = (bottom[x], top[x]);
                }
            }
        }
    }
}

/// <summary>
/// Moves images by a whole number of pixels, up to <see cref="MaxPixels"/> in each direction (chosen uniformly for x
/// and y); the uncovered border takes <see cref="Fill"/>.
/// </summary>
/// <param name="maxPixels">The largest move, in pixels.</param>
/// <param name="fill">The value of the uncovered pixels (0 is black).</param>
public sealed class RandomShift(int maxPixels, float fill = 0f) : ISampleTransform
{
    /// <summary>The largest move, in pixels.</summary>
    public int MaxPixels { get; } = maxPixels >= 0 ? maxPixels : throw new ArgumentOutOfRangeException(nameof(maxPixels));

    /// <summary>The value of the uncovered pixels.</summary>
    public float Fill { get; } = fill;

    /// <inheritdoc />
    public void Apply(Span<float> features, Span<float> targets, IReadOnlyList<int> featureShape, Random random)
    {
        var (planes, h, w) = ImageShape.Of(featureShape, features.Length);
        int dx = random.Next(-MaxPixels, MaxPixels + 1), dy = random.Next(-MaxPixels, MaxPixels + 1);
        if (dx == 0 && dy == 0)
        {
            return;
        }

        var scratch = ArrayPool<float>.Shared.Rent(h * w);
        try
        {
            for (int p = 0; p < planes; p++)
            {
                var plane = features.Slice(p * h * w, h * w);
                plane.CopyTo(scratch);
                for (int y = 0; y < h; y++)
                {
                    int sy = y - dy;
                    for (int x = 0; x < w; x++)
                    {
                        int sx = x - dx;
                        plane[y * w + x] = (uint)sy < (uint)h && (uint)sx < (uint)w ? scratch[sy * w + sx] : Fill;
                    }
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(scratch);
        }
    }
}

/// <summary>
/// Rotates images about their centre by an angle chosen uniformly in [-<see cref="MaxDegrees"/>, <see cref="MaxDegrees"/>],
/// sampling bilinearly; pixels from outside the image take <see cref="Fill"/>.
/// </summary>
/// <param name="maxDegrees">The largest rotation, in degrees, either way.</param>
/// <param name="fill">The value of pixels rotated in from outside (0 is black).</param>
public sealed class RandomRotation(float maxDegrees, float fill = 0f) : ISampleTransform
{
    /// <summary>The largest rotation, in degrees, either way.</summary>
    public float MaxDegrees { get; } = maxDegrees >= 0 ? maxDegrees : throw new ArgumentOutOfRangeException(nameof(maxDegrees));

    /// <summary>The value of pixels rotated in from outside.</summary>
    public float Fill { get; } = fill;

    /// <inheritdoc />
    public void Apply(Span<float> features, Span<float> targets, IReadOnlyList<int> featureShape, Random random)
    {
        var (planes, h, w) = ImageShape.Of(featureShape, features.Length);
        double angle = (random.NextDouble() * 2 - 1) * MaxDegrees * Math.PI / 180;
        if (angle == 0)
        {
            return;
        }

        float cos = (float)Math.Cos(angle), sin = (float)Math.Sin(angle);
        float cx = (w - 1) / 2f, cy = (h - 1) / 2f;
        var scratch = ArrayPool<float>.Shared.Rent(h * w);
        try
        {
            for (int p = 0; p < planes; p++)
            {
                var plane = features.Slice(p * h * w, h * w);
                plane.CopyTo(scratch);
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        // The source of each output pixel: the output position rotated back by the angle.
                        float ox = x - cx, oy = y - cy;
                        float sx = cos * ox + sin * oy + cx, sy = -sin * ox + cos * oy + cy;
                        int x0 = (int)MathF.Floor(sx), y0 = (int)MathF.Floor(sy);
                        float fx = sx - x0, fy = sy - y0;
                        float At(int yy, int xx) => (uint)yy < (uint)h && (uint)xx < (uint)w ? scratch[yy * w + xx] : Fill;
                        float top = At(y0, x0) * (1 - fx) + At(y0, x0 + 1) * fx, bottom = At(y0 + 1, x0) * (1 - fx) + At(y0 + 1, x0 + 1) * fx;
                        plane[y * w + x] = top * (1 - fy) + bottom * fy;
                    }
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(scratch);
        }
    }
}

/// <summary>Adds normal noise with standard deviation <see cref="StandardDeviation"/> to every feature (images or not).</summary>
/// <param name="standardDeviation">The noise's standard deviation, in feature units.</param>
/// <param name="clamp">Keep values in [0, 1] (for images).</param>
public sealed class GaussianNoise(float standardDeviation, bool clamp = false) : ISampleTransform
{
    /// <summary>The noise's standard deviation.</summary>
    public float StandardDeviation { get; } = standardDeviation >= 0 ? standardDeviation : throw new ArgumentOutOfRangeException(nameof(standardDeviation));

    /// <summary>Whether values are kept in [0, 1].</summary>
    public bool Clamp { get; } = clamp;

    /// <inheritdoc />
    public void Apply(Span<float> features, Span<float> targets, IReadOnlyList<int> featureShape, Random random)
    {
        for (int i = 0; i < features.Length; i += 2)
        {
            // Box-Muller: two independent normal values from two uniform ones.
            double u = 1 - random.NextDouble(), v = random.NextDouble();
            double r = Math.Sqrt(-2 * Math.Log(u)) * StandardDeviation;
            features[i] += (float)(r * Math.Cos(2 * Math.PI * v));
            if (i + 1 < features.Length)
            {
                features[i + 1] += (float)(r * Math.Sin(2 * Math.PI * v));
            }
        }

        if (Clamp)
        {
            foreach (ref float value in features)
            {
                value = Math.Clamp(value, 0f, 1f);
            }
        }
    }
}

internal static class ImageShape
{
    /// <summary>A feature shape [..., height, width] as (planes, height, width): the leading dimensions multiplied.</summary>
    public static (int Planes, int Height, int Width) Of(IReadOnlyList<int> shape, int size)
    {
        if (shape.Count < 2)
        {
            throw new ArgumentException($"Image transforms need features shaped [channels, height, width] or [height, width], not [{string.Join(", ", shape)}].");
        }

        int h = shape[^2], w = shape[^1];
        return (size / (h * w), h, w);
    }
}
