// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Data;

namespace Idrak.Vision;

/// <summary>How <see cref="PageSegmenter"/> splits a page. The fractions are of the line's height.</summary>
public sealed record SegmentationOptions
{
    /// <summary>The ink threshold in [0, 1]; null (the default) takes Otsu's threshold of the page.</summary>
    public float? Threshold { get; init; }

    /// <summary>Rows of ink separated by fewer blank rows than this belong to one line (default 2).</summary>
    public int MinLineGap { get; init; } = 2;

    /// <summary>
    /// A band of rows thinner than this share of the median line joins the nearer neighbouring line (default 0.45): the
    /// dots above and below letters (Arabic ب ت ث, Latin i j) and stray marks stay with their line.
    /// </summary>
    public double ThinLineFraction { get; init; } = 0.45;

    /// <summary>A gap between characters wider than this share of the line's height is a space (default 0.4).</summary>
    public double SpaceFraction { get; init; } = 0.4;

    /// <summary>A character wider than this share of its line's height is split into touching characters (default 1.3).</summary>
    public double SplitFraction { get; init; } = 1.3;

    /// <summary>The expected width of one of those touching characters, as a share of the line's height (default 0.8).</summary>
    public double PieceFraction { get; init; } = 0.8;

    /// <summary>Marks with fewer pixels in their box than this are dropped as specks (default 6).</summary>
    public int MinGlyphArea { get; init; } = 6;
}

/// <summary>
/// Finds the text lines and characters of a page by the gaps between them: ink is separated from paper (Otsu's
/// threshold, either polarity, any colour), lines are runs of rows with ink, characters are runs of columns with ink
/// within a line, and wide gaps are spaces. One pass over the pixels makes the ink and the row counts; each line then
/// counts its columns once. The page's ink (one float per pixel) is the only page-sized buffer.
/// </summary>
/// <remarks>
/// It reads clean pages of separate characters: printed-style handwriting, forms, printed text with spaced letters,
/// isolated Arabic letters with their dots. Joined (cursive) writing and tilted lines need a word-level model.
/// </remarks>
public static class PageSegmenter
{
    /// <summary>The lines and characters of a page (any number of channels; colour is read as grey).</summary>
    public static PageLayout Segment(ImageData page, SegmentationOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(page);
        options ??= new SegmentationOptions();
        int width = page.Width, height = page.Height, pixels = width * height;

        // Grey (the channels' mean), and its mean to tell dark ink on light paper from the opposite.
        var ink = new float[pixels];
        var source = page.Pixels;
        double sum = 0;
        for (int c = 0; c < page.Channels; c++)
        {
            var plane = source.AsSpan(c * pixels, pixels);
            for (int i = 0; i < pixels; i++)
            {
                ink[i] += plane[i] / page.Channels;
            }
        }

        for (int i = 0; i < pixels; i++)
        {
            sum += ink[i];
        }

        bool darkOnLight = sum / Math.Max(pixels, 1) > 0.5;
        if (darkOnLight)
        {
            for (int i = 0; i < pixels; i++)
            {
                ink[i] = 1f - ink[i];
            }
        }

        // Ink against the threshold is tested where it is needed: no mask is kept (a page's worth of memory saved).
        float threshold = options.Threshold ?? Otsu(ink);
        var rows = new int[height];
        for (int y = 0; y < height; y++)
        {
            int count = 0;
            for (int x = 0, i = y * width; x < width; x++, i++)
            {
                if (ink[i] > threshold)
                {
                    count++;
                }
            }

            rows[y] = count;
        }

        var lines = new List<TextLineRegion>();
        var columns = new int[width];
        foreach (var (top, bottom) in MergeThin(Runs(rows, options.MinLineGap, 1), options.ThinLineFraction))
        {
            int lineHeight = bottom - top;
            Array.Clear(columns);
            for (int y = top; y < bottom; y++)
            {
                for (int x = 0, i = y * width; x < width; x++, i++)
                {
                    if (ink[i] > threshold)
                    {
                        columns[x]++;
                    }
                }
            }

            var glyphs = new List<GlyphRegion>();
            int previousRight = -1, lineLeft = width, lineRight = 0, lineTop = bottom, lineBottom = top;
            foreach (var (left, right) in Split(Runs(columns, 1, 1), columns, lineHeight, options))
            {
                int glyphTop = bottom, glyphBottom = top;
                for (int y = top; y < bottom; y++)
                {
                    for (int x = left, i = y * width + left; x < right; x++, i++)
                    {
                        if (ink[i] > threshold)
                        {
                            glyphTop = Math.Min(glyphTop, y);
                            glyphBottom = y + 1;
                            break;
                        }
                    }
                }

                if (glyphBottom <= glyphTop || (right - left) * (glyphBottom - glyphTop) < options.MinGlyphArea)
                {
                    continue;
                }

                bool space = previousRight >= 0 && left - previousRight > options.SpaceFraction * lineHeight;
                previousRight = right;
                glyphs.Add(new GlyphRegion(new PixelBox(left, glyphTop, right - left, glyphBottom - glyphTop), space));
                (lineLeft, lineRight) = (Math.Min(lineLeft, left), Math.Max(lineRight, right));
                (lineTop, lineBottom) = (Math.Min(lineTop, glyphTop), Math.Max(lineBottom, glyphBottom));
            }

            if (glyphs.Count > 0)
            {
                lines.Add(new TextLineRegion(new PixelBox(lineLeft, lineTop, lineRight - lineLeft, lineBottom - lineTop), glyphs));
            }
        }

        return new PageLayout(width, height, ink, threshold, darkOnLight, lines);
    }

    /// <summary>Otsu's threshold of values in [0, 1]: the level that best separates their histogram into two groups.</summary>
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
            double meanBack = backgroundSum / backgroundCount, meanInk = (sum - backgroundSum) / (total - backgroundCount);
            double between = backgroundCount * (total - backgroundCount) * (meanBack - meanInk) * (meanBack - meanInk);
            if (between > best)
            {
                (best, level) = (between, i);
            }
        }

        return Math.Max((level + 1) / 255f, 0.1f);   // bins up to `level` are paper: values below (level + 1) / 255
    }

    // Runs of non-zero counts; runs separated by fewer than minGap zeros merge, runs shorter than minLength drop.
    private static List<(int Start, int End)> Runs(int[] counts, int minGap, int minLength)
    {
        var runs = new List<(int Start, int End)>();
        int start = -1;
        for (int i = 0; i <= counts.Length; i++)
        {
            bool on = i < counts.Length && counts[i] > 0;
            if (on && start < 0)
            {
                start = i;
            }
            else if (!on && start >= 0)
            {
                if (runs.Count > 0 && start - runs[^1].End < minGap)
                {
                    runs[^1] = (runs[^1].Start, i);
                }
                else
                {
                    runs.Add((start, i));
                }

                start = -1;
            }
        }

        runs.RemoveAll(r => r.End - r.Start < minLength);
        return runs;
    }

    // Bands much thinner than the median line join the nearer neighbour when it is close (dots above and below letters);
    // thin bands far from any line are dropped as marks. Thinness is judged on each band's own rows (Core), not on the
    // rows it gained by merging, so a merged mark cannot make a real line look thin.
    private static List<(int Start, int End)> MergeThin(List<(int Start, int End)> runs, double fraction)
    {
        var bands = runs.Select(r => (r.Start, r.End, Core: r.End - r.Start)).ToList();
        while (bands.Count > 1)
        {
            var cores = bands.Select(b => b.Core).Order().ToArray();
            double median = cores[(cores.Length - 1) / 2];
            int thin = bands.FindIndex(b => b.Core < fraction * median);
            if (thin < 0)
            {
                break;
            }

            int gapAbove = thin > 0 ? bands[thin].Start - bands[thin - 1].End : int.MaxValue;
            int gapBelow = thin < bands.Count - 1 ? bands[thin + 1].Start - bands[thin].End : int.MaxValue;
            if (Math.Min(gapAbove, gapBelow) > 0.6 * median)
            {
                bands.RemoveAt(thin);   // a mark on its own, not part of a line
                continue;
            }

            int other = gapAbove <= gapBelow ? thin - 1 : thin + 1;
            var (first, second) = other < thin ? (other, thin) : (thin, other);
            bands[first] = (bands[first].Start, bands[second].End, Math.Max(bands[first].Core, bands[second].Core));
            bands.RemoveAt(second);
        }

        return [.. bands.Where(b => b.End - b.Start >= 3).Select(b => (b.Start, b.End))];
    }

    // Runs much wider than a character (touching characters) split at their thinnest columns near even divisions.
    private static List<(int Start, int End)> Split(List<(int Start, int End)> runs, int[] columns, int lineHeight, SegmentationOptions options)
    {
        var result = new List<(int Start, int End)>(runs.Count);
        foreach (var (start, end) in runs)
        {
            int pieces = (int)Math.Round((end - start) / (options.PieceFraction * lineHeight));
            if (pieces < 2 || end - start < options.SplitFraction * lineHeight)
            {
                result.Add((start, end));
                continue;
            }

            int from = start;
            for (int p = 1; p < pieces; p++)
            {
                int expected = start + (end - start) * p / pieces, window = Math.Max(1, (end - start) / pieces / 3);
                int cut = expected, low = int.MaxValue;
                for (int x = Math.Max(from + 1, expected - window); x <= expected + window && x < end - 1; x++)
                {
                    if (columns[x] < low)
                    {
                        (low, cut) = (columns[x], x);
                    }
                }

                result.Add((from, cut));
                from = cut;
            }

            result.Add((from, end));
        }

        return result;
    }
}
