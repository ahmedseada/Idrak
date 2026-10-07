// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.


namespace Idrak.Abstraction.Vision;

/// <summary>Which way round an image's foreground is.</summary>
public enum Polarity
{
    /// <summary>Decided from the image: a mostly light image has dark foreground (ink on paper), and the other way round.</summary>
    Auto,

    /// <summary>Dark foreground on a light background (ink on paper, a dark object on a white table).</summary>
    DarkOnLight,

    /// <summary>Light foreground on a dark background (MNIST-style digits, fluorescence images).</summary>
    LightOnDark,
}

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

/// <summary>Separates an image's foreground from its background.</summary>
public static class Foreground
{
    /// <summary>
    /// The foreground of <paramref name="image"/> (any channels; colour is averaged to grey). The polarity is decided
    /// from the image unless given, and the threshold is Otsu's unless given. One value per pixel is the only buffer made.
    /// </summary>
    public static ForegroundImage Extract(ImageData image, Polarity polarity = Polarity.Auto, float? threshold = null)
    {
        ArgumentNullException.ThrowIfNull(image);
        var values = Grey(image);
        bool invert = polarity switch
        {
            Polarity.DarkOnLight => true,
            Polarity.LightOnDark => false,
            _ => Mean(values) > 0.5,
        };

        if (invert)
        {
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = 1f - values[i];
            }
        }

        return new ForegroundImage(values, image.Width, image.Height, threshold ?? Otsu(values), invert);
    }

    /// <summary>The image in grey: the channels' mean, row by row.</summary>
    public static float[] Grey(ImageData image)
    {
        ArgumentNullException.ThrowIfNull(image);
        int pixels = image.Width * image.Height;
        if (image.Channels == 1)
        {
            return image.Pixels.AsSpan(0, pixels).ToArray();
        }

        var grey = new float[pixels];
        float weight = 1f / image.Channels;
        for (int c = 0; c < image.Channels; c++)
        {
            var plane = image.Pixels.AsSpan(c * pixels, pixels);
            for (int i = 0; i < pixels; i++)
            {
                grey[i] += plane[i] * weight;
            }
        }

        return grey;
    }

    /// <summary>
    /// Otsu's threshold of values in [0, 1]: the level that best separates their histogram (256 bins) into two groups,
    /// at least 0.1 so a blank image does not turn noise into foreground.
    /// </summary>
    public static float Otsu(ReadOnlySpan<float> values)
    {
        Span<int> histogram = stackalloc int[256];
        histogram.Clear();
        foreach (float v in values)
        {
            histogram[Math.Clamp((int)(v * 255f), 0, 255)]++;
        }

        double total = values.Length, sum = 0;
        for (int i = 0; i < 256; i++)
        {
            sum += i * (double)histogram[i];
        }

        double backgroundSum = 0, best = -1;
        long backgroundCount = 0;
        int level = 128;
        for (int i = 0; i < 256; i++)
        {
            backgroundCount += histogram[i];
            if (backgroundCount == 0 || backgroundCount == total)
            {
                continue;
            }

            backgroundSum += i * (double)histogram[i];
            double meanBack = backgroundSum / backgroundCount, meanFore = (sum - backgroundSum) / (total - backgroundCount);
            double between = backgroundCount * (total - backgroundCount) * (meanBack - meanFore) * (meanBack - meanFore);
            if (between > best)
            {
                (best, level) = (between, i);
            }
        }

        return Math.Max((level + 1) / 255f, 0.1f);   // bins up to `level` are background: values below (level + 1) / 255
    }

    private static double Mean(float[] values)
    {
        double sum = 0;
        foreach (float v in values)
        {
            sum += v;
        }

        return sum / Math.Max(values.Length, 1);
    }
}
