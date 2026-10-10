// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Vision;

namespace Idrak.Samples.ArabicOcr;

/// <summary>A text line found on a page: its place (in the deskewed page's pixels) and its pixels (grey, as on the page).</summary>
/// <param name="Index">1 for the top line.</param>
internal sealed record TextLine(int Index, int X, int Y, int Width, int Height, ImageData Image);

/// <summary>How lines are found (all relative to the page, nothing in absolute pixels but the floor of the padding).</summary>
internal sealed record SegmentationSettings
{
    /// <summary>The largest skew corrected, in degrees either way (0: no deskew).</summary>
    public double MaxSkew { get; init; } = 3;

    /// <summary>Padding around a line, as a share of its height.</summary>
    public double Padding { get; init; } = 0.2;
}

/// <summary>
/// Page → lines by horizontal projection: the page binarized (<see cref="Foreground"/>: Otsu's threshold, the polarity
/// from the page), its skew measured by which small shear makes the row profile sharpest and undone, rows with ink
/// grouped into bands, bands of dots and marks (shorter than half a typical line, as Arabic's dots and harakat are when a
/// line's body has none of their rows) joined to the nearer neighbour, then each band cut out with padding, top to bottom.
/// Lines that touch (no empty row between them) stay one band.
/// </summary>
internal static class LineSegmenter
{
    public static readonly double[] Zz = new double[4];
    /// <summary>The lines of <paramref name="page"/>, top to bottom, and the skew undone (degrees).</summary>
    public static (IReadOnlyList<TextLine> Lines, double Skew) Find(ImageData page, SegmentationSettings? settings = null)
    {
        settings ??= new SegmentationSettings();
        long zt = System.Diagnostics.Stopwatch.GetTimestamp();
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
            return ([], 0);                                                       // a blank page: no contrast, no ink
        }

        float threshold = foreground.Threshold;
        Zz[0] += System.Diagnostics.Stopwatch.GetElapsedTime(zt).TotalMilliseconds; zt = System.Diagnostics.Stopwatch.GetTimestamp();
        double skew = settings.MaxSkew > 0 ? MeasureSkew(ink, width, height, threshold, settings.MaxSkew) : 0;
        float paper = foreground.Inverted ? 1f : 0f;
        Zz[1] += System.Diagnostics.Stopwatch.GetElapsedTime(zt).TotalMilliseconds; zt = System.Diagnostics.Stopwatch.GetTimestamp();
        if (skew != 0)
        {
            grey = Shear(grey, width, height, skew, paper);
            ink = Shear(ink, width, height, skew, 0f);
        }

        Zz[2] += System.Diagnostics.Stopwatch.GetElapsedTime(zt).TotalMilliseconds; zt = System.Diagnostics.Stopwatch.GetTimestamp();
        // Rows with ink: at least one pixel in a thousand of the width.
        int minRow = Math.Max(1, width / 1000);
        var rowInk = new int[height];
        for (int y = 0; y < height; y++)
        {
            int count = 0;
            for (int x = 0, row = y * width; x < width; x++)
            {
                count += ink[row + x] > threshold ? 1 : 0;
            }

            rowInk[y] = count;
        }

        var bands = new List<(int Top, int Bottom)>();                            // rows [Top, Bottom)
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
            return ([], skew);
        }

        bands = JoinMarks(bands);
        long Ink((int Top, int Bottom) band) => Enumerable.Range(band.Top, band.Bottom - band.Top).Sum(y => (long)rowInk[y]);
        var inks = bands.Select(Ink).ToArray();
        long typicalInk = inks.Order().ElementAt(inks.Length / 2);

        var lines = new List<TextLine>();
        foreach (var (band, inkCount) in bands.Zip(inks))
        {
            if (inkCount * 20 < typicalInk)
            {
                continue;                                                           // specks: a twentieth of a typical line's ink
            }

            int left = width, right = -1;
            for (int y = band.Top; y < band.Bottom; y++)
            {
                for (int x = 0, row = y * width; x < width; x++)
                {
                    if (ink[row + x] > threshold)
                    {
                        left = Math.Min(left, x);
                        right = Math.Max(right, x);
                    }
                }
            }

            int pad = Math.Max(2, (int)Math.Round(settings.Padding * (band.Bottom - band.Top)));
            int x0 = Math.Max(0, left - pad), x1 = Math.Min(width, right + 1 + pad);
            int y0 = Math.Max(0, band.Top - pad), y1 = Math.Min(height, band.Bottom + pad);
            var crop = new float[(x1 - x0) * (y1 - y0)];
            for (int y = y0; y < y1; y++)
            {
                Array.Copy(grey, y * width + x0, crop, (y - y0) * (x1 - x0), x1 - x0);
            }

            lines.Add(new TextLine(lines.Count + 1, x0, y0, x1 - x0, y1 - y0, new ImageData(crop, 1, y1 - y0, x1 - x0)));
        }

        Zz[3] += System.Diagnostics.Stopwatch.GetElapsedTime(zt).TotalMilliseconds;
        return (lines, skew);
    }

    // Bands shorter than half a typical line (dots, marks, a broken stroke) join the nearer neighbour band when the gap is
    // under a typical line's height; the typical height is the median of the bands at least 40% as tall as the tallest.
    private static List<(int Top, int Bottom)> JoinMarks(List<(int Top, int Bottom)> bands)
    {
        int tallest = bands.Max(b => b.Bottom - b.Top);
        var tall = bands.Select(b => b.Bottom - b.Top).Where(h => h * 10 >= tallest * 4).Order().ToArray();
        int typical = tall[tall.Length / 2];
        bool joined = true;
        while (joined && bands.Count > 1)
        {
            joined = false;
            for (int i = 0; i < bands.Count; i++)
            {
                var band = bands[i];
                if ((band.Bottom - band.Top) * 2 >= typical)
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
    // a fifth of a degree; 0 unless another angle is clearly sharper. Ink pixels only, at most about 200,000 of them.
    private static double MeasureSkew(ReadOnlySpan<float> ink, int width, int height, float threshold, double maxDegrees)
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
        int step = Math.Max(1, total / 200_000);
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

        double flat = Score(0), best = flat, bestAngle = 0;
        for (double angle = -maxDegrees; angle <= maxDegrees + 1e-9; angle += 0.2)
        {
            double score = Score(angle);
            if (score > best)
            {
                (best, bestAngle) = (score, Math.Round(angle, 1));
            }
        }

        return best > flat * 1.02 ? bestAngle : 0;
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
