// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.


using Idrak.Vision.Abstractions;

namespace Idrak.Vision;

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
