// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Vision.Abstractions;

/// <summary>
/// An image's foreground as one value per pixel in [0, 1], high on the foreground (dark foreground is inverted), with
/// the threshold that separates it from the background. <see cref="Foreground.Extract"/> makes one;
/// <see cref="ConnectedComponents"/>, <c>ContentFrame</c> and <c>RegionClassifier</c> read it.
/// </summary>
public sealed class ForegroundImage
{
    private readonly float[] _values;

    /// <summary>Wraps values (row by row, foreground high) and their threshold.</summary>
    public ForegroundImage(float[] values, int width, int height, float threshold, bool inverted = false)
    {
        ArgumentNullException.ThrowIfNull(values);
        if (values.Length != width * height)
        {
            throw new ArgumentException($"{values.Length} values for a {width} x {height} image.", nameof(values));
        }

        _values = values;
        Width = width;
        Height = height;
        Threshold = threshold;
        Inverted = inverted;
    }

    /// <summary>The image's width in pixels.</summary>
    public int Width { get; }

    /// <summary>The image's height in pixels.</summary>
    public int Height { get; }

    /// <summary>The value above which a pixel is foreground.</summary>
    public float Threshold { get; }

    /// <summary>Whether the image had dark foreground on a light background, so its values were inverted.</summary>
    public bool Inverted { get; }

    /// <summary>The values, row by row: high on the foreground.</summary>
    public ReadOnlySpan<float> Values => _values;

    /// <summary>Whether the pixel at (<paramref name="x"/>, <paramref name="y"/>) is foreground.</summary>
    public bool IsForeground(int x, int y) => _values[y * Width + x] > Threshold;

    internal float[] Data => _values;
}
