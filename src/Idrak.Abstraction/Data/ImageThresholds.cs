// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Data;

/// <summary>
/// Levels measured from an image's values that tell its foreground (ink) from its background (paper), in the lowest
/// place every package can use them: Idrak.Vision's <c>Foreground</c> separates a page with them, Idrak.Imaging's
/// "contour" outline tracer cuts a glyph's coverage with them.
/// </summary>
public static class ImageThresholds
{
    /// <summary>The least level <see cref="Otsu"/> gives, so that a blank image does not turn its noise into foreground.</summary>
    public const float OtsuLeast = 0.1f;

    /// <summary>
    /// Otsu's threshold of values in [0, 1]: the level that best separates their histogram (256 bins) into two groups,
    /// at least <see cref="OtsuLeast"/> so a blank image does not turn noise into foreground. Values from the level up are
    /// the high group (the foreground of ink made high).
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

        return Math.Max((level + 1) / 255f, OtsuLeast);   // bins up to `level` are background: values below (level + 1) / 255
    }
}
