// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Vision.Abstractions;

namespace Idrak.Vision;

/// <summary>
/// "projection" (<see cref="TextLineSegmenters.Projection"/>): page → lines by horizontal projection. The page is binarized
/// (<see cref="Foreground"/>: Otsu's threshold, the polarity from the page); its skew is measured by which small shear makes
/// the row profile of the ink sharpest, and undone; rows with ink are grouped into bands; bands of dots and marks (shorter
/// than <see cref="LineSegmentationOptions.MarkShare"/> of a typical line, as Arabic's dots and harakat are when a line's
/// body has none of their rows) join the nearer neighbour; specks are left out; each band is cut out with padding, top to
/// bottom. Lines that touch (no empty row between them) stay one band. Every size is relative to the page or the line
/// (<see cref="LineSegmentationOptions"/>).
/// </summary>
public sealed class ProjectionLineSegmenter : ITextLineSegmenter
{
    /// <inheritdoc />
    public string Name => TextLineSegmenters.Projection;

    /// <inheritdoc />
    public PageLines Find(ImageData page, LineSegmentationOptions options)
    {
        ArgumentNullException.ThrowIfNull(page);
        ArgumentNullException.ThrowIfNull(options);
        Validate(options);
        int width = page.Width, height = page.Height;
        var grey = Foreground.Grey(page);                                             // grey once: the foreground is made from it
        var foreground = Foreground.Extract(page.Channels == 1 ? page : new ImageData(grey, 1, height, width));
        ReadOnlySpan<float> ink = foreground.Values;
        float lo = float.PositiveInfinity, hi = float.NegativeInfinity;
        foreach (float v in ink)
        {
            lo = Math.Min(lo, v);
            hi = Math.Max(hi, v);
        }

        if (hi - lo < 0.1f)
        {
            return new PageLines([], 0);                                              // a blank page: no contrast, no ink
        }

        float threshold = foreground.Threshold;
        double skew = options.MaxSkew > 0 ? MeasureSkew(ink, width, height, threshold, options) : 0;
        float paper = foreground.Inverted ? 1f : 0f;
        if (skew != 0)
        {
            grey = Shear(grey, width, height, skew, paper);
            ink = Shear(ink, width, height, skew, 0f);
        }

        // Rows with ink: at least RowInk of the width (one pixel in a thousand by default), row by row.
        int minRow = Math.Max(1, (int)(width * options.RowInk));
        var rowInk = new int[height];
        for (int y = 0; y < height; y++)
        {
            int count = 0;
            var row = ink.Slice(y * width, width);
            for (int x = 0; x < row.Length; x++)
            {
                count += row[x] > threshold ? 1 : 0;
            }

            rowInk[y] = count;
        }

        var bands = new List<(int Top, int Bottom)>();                                // rows [Top, Bottom)
        for (int y = 0; y < height;)
        {
            if (rowInk[y] < minRow)
            {
                y++;
                continue;
            }

            int top = y;
            while (y < height && rowInk[y] >= minRow)
            {
                y++;
            }

            bands.Add((top, y));
        }

        if (bands.Count == 0)
        {
            return new PageLines([], skew);
        }

        bands = JoinMarks(bands, options.MarkShare);
        var inks = new long[bands.Count];
        for (int b = 0; b < bands.Count; b++)
        {
            for (int y = bands[b].Top; y < bands[b].Bottom; y++)
            {
                inks[b] += rowInk[y];
            }
        }

        long typicalInk = inks.Order().ElementAt(inks.Length / 2);
        var lines = new List<PageLine>();
        for (int b = 0; b < bands.Count; b++)
        {
            var band = bands[b];
            if (inks[b] < typicalInk * options.SpeckShare)
            {
                continue;                                                               // specks: a twentieth of a typical line's ink by default
            }

            int left = width, right = -1;
            for (int y = band.Top; y < band.Bottom; y++)
            {
                var row = ink.Slice(y * width, width);
                for (int x = 0; x < row.Length; x++)
                {
                    if (row[x] > threshold)
                    {
                        left = Math.Min(left, x);
                        right = Math.Max(right, x);
                    }
                }
            }

            int pad = Math.Max(options.MinPadding, (int)Math.Round(options.Padding * (band.Bottom - band.Top)));
            int x0 = Math.Max(0, left - pad), x1 = Math.Min(width, right + 1 + pad);
            int y0 = Math.Max(0, band.Top - pad), y1 = Math.Min(height, band.Bottom + pad);
            var crop = new float[(x1 - x0) * (y1 - y0)];
            for (int y = y0; y < y1; y++)
            {
                Array.Copy(grey, y * width + x0, crop, (y - y0) * (x1 - x0), x1 - x0);
            }

            lines.Add(new PageLine(lines.Count + 1, x0, y0, x1 - x0, y1 - y0, new ImageData(crop, 1, y1 - y0, x1 - x0)));
        }

        return new PageLines(lines, skew);
    }

    /// <inheritdoc />
    public override string ToString() => "horizontal projection of the binarized, deskewed page; marks joined to their line";

    private static void Validate(LineSegmentationOptions options)
    {
        string? problem = !(options.MaxSkew >= 0 && options.MaxSkew < 45) ? $"MaxSkew {options.MaxSkew} (0 to under 45 degrees)"
            : !(options.SkewStep > 0) ? $"SkewStep {options.SkewStep} (above 0)"
            : !(options.MinSkewGain >= 1) ? $"MinSkewGain {options.MinSkewGain} (at least 1)"
            : options.MaxSkewSamples <= 0 ? $"MaxSkewSamples {options.MaxSkewSamples} (above 0)"
            : !(options.Padding >= 0) ? $"Padding {options.Padding} (at least 0)"
            : options.MinPadding < 0 ? $"MinPadding {options.MinPadding} (at least 0)"
            : !(options.RowInk is >= 0 and <= 1) ? $"RowInk {options.RowInk} (0 to 1)"
            : !(options.SpeckShare is >= 0 and <= 1) ? $"SpeckShare {options.SpeckShare} (0 to 1)"
            : !(options.MarkShare is >= 0 and <= 1) ? $"MarkShare {options.MarkShare} (0 to 1)"
            : null;
        if (problem is not null)
        {
            throw new ArgumentOutOfRangeException(nameof(options), $"Line segmentation: {problem}.");
        }
    }

    // Bands shorter than markShare of a typical line (dots, marks, a broken stroke) join the nearer neighbour band when the
    // gap is under a typical line's height; the typical height is the median of the bands at least 40% as tall as the tallest.
    private static List<(int Top, int Bottom)> JoinMarks(List<(int Top, int Bottom)> bands, double markShare)
    {
        int tallest = bands.Max(b => b.Bottom - b.Top);
        var tall = bands.Select(b => b.Bottom - b.Top).Where(h => h * 10 >= tallest * 4).Order().ToArray();
        int typical = tall[tall.Length / 2];
        double mark = typical * markShare;
        bool joined = true;
        while (joined && bands.Count > 1)
        {
            joined = false;
            for (int i = 0; i < bands.Count; i++)
            {
                var band = bands[i];
                if (band.Bottom - band.Top >= mark)
                {
                    continue;
                }

                int above = i > 0 ? band.Top - bands[i - 1].Bottom : int.MaxValue;
                int below = i + 1 < bands.Count ? bands[i + 1].Top - band.Bottom : int.MaxValue;
                int gap = Math.Min(above, below);
                if (gap >= typical)
                {
                    continue;
                }

                int other = above <= below ? i - 1 : i + 1;
                var merged = (Math.Min(band.Top, bands[other].Top), Math.Max(band.Bottom, bands[other].Bottom));
                bands[Math.Min(i, other)] = merged;
                bands.RemoveAt(Math.Max(i, other));
                joined = true;
                break;
            }
        }

        return bands;
    }

    // The shear (in degrees) whose row profile of the ink is sharpest (the largest sum of squared row counts), in steps of
    // SkewStep; 0 unless another angle is MinSkewGain sharper. Ink pixels only, at most about MaxSkewSamples of them.
    private static double MeasureSkew(ReadOnlySpan<float> ink, int width, int height, float threshold, LineSegmentationOptions options)
    {
        int total = 0;
        foreach (float v in ink)
        {
            total += v > threshold ? 1 : 0;
        }

        if (total == 0)
        {
            return 0;
        }

        // Every step-th ink pixel, row by row: ceil(total / step) of them.
        int step = Math.Max(1, total / options.MaxSkewSamples);
        var xs = new int[(total + step - 1) / step];
        var ys = new int[xs.Length];
        for (int y = 0, seen = 0, n = 0; y < height; y++)
        {
            var row = ink.Slice(y * width, width);
            for (int x = 0; x < width; x++)
            {
                if (row[x] > threshold && seen++ % step == 0)
                {
                    (xs[n], ys[n]) = (x, y);
                    n++;
                }
            }
        }

        var rows = new int[height];                                                   // one buffer for every angle, grown as the margin needs
        double Score(double degrees)
        {
            double slope = Math.Tan(degrees * Math.PI / 180);
            int margin = (int)Math.Ceiling(Math.Abs(slope) * width) + 1;
            int length = height + 2 * margin;
            if (rows.Length < length)
            {
                rows = new int[length];
            }

            var counts = rows.AsSpan(0, length);
            counts.Clear();
            for (int i = 0; i < xs.Length; i++)
            {
                int row = (int)Math.Round(ys[i] - (xs[i] - width / 2.0) * slope) + margin;
                counts[Math.Clamp(row, 0, length - 1)]++;
            }

            double sum = 0;
            foreach (int c in counts)
            {
                sum += (double)c * c;
            }

            return sum;
        }

        double maxDegrees = options.MaxSkew, stepDegrees = options.SkewStep;
        int decimals = Decimals(stepDegrees);
        double flat = Score(0), best = flat, bestAngle = 0;
        for (double angle = -maxDegrees; angle <= maxDegrees + 1e-9; angle += stepDegrees)
        {
            double score = Score(angle);
            if (score > best)
            {
                (best, bestAngle) = (score, Math.Round(angle, decimals));
            }
        }

        return best > flat * options.MinSkewGain ? bestAngle : 0;
    }

    // The decimals an angle is rounded to: those of the step, at least one (0.2 → 1, 0.05 → 2).
    private static int Decimals(double step)
    {
        int decimals = 1;
        while (decimals < 6 && Math.Abs(step * Math.Pow(10, decimals) - Math.Round(step * Math.Pow(10, decimals))) > 1e-9)
        {
            decimals++;
        }

        return decimals;
    }

    // dest(y, x) = src(y + (x - width / 2) · tan(angle), x), linearly between rows; outside the page: fill. Row by row (the
    // arrays' order): each column's shift is worked out once.
    private static float[] Shear(ReadOnlySpan<float> source, int width, int height, double degrees, float fill)
    {
        double slope = Math.Tan(degrees * Math.PI / 180);
        var shifts = new double[width];
        for (int x = 0; x < width; x++)
        {
            shifts[x] = (x - width / 2.0) * slope;
        }

        var result = new float[source.Length];
        for (int y = 0; y < height; y++)
        {
            var row = result.AsSpan(y * width, width);
            for (int x = 0; x < width; x++)
            {
                double sy = y + shifts[x];
                int y0 = (int)Math.Floor(sy);
                double t = sy - y0;
                float a = y0 >= 0 && y0 < height ? source[y0 * width + x] : fill;
                float b = y0 + 1 >= 0 && y0 + 1 < height ? source[(y0 + 1) * width + x] : fill;
                row[x] = (float)(a * (1 - t) + b * t);
            }
        }

        return result;
    }
}
