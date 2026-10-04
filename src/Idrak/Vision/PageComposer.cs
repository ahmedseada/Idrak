// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Data;

namespace Idrak.Vision;

/// <summary>A character image cropped to its ink: width x height values, ink high.</summary>
public sealed record GlyphImage(float[] Pixels, int Width, int Height)
{
    /// <summary>A data set's character image (rows x columns, ink high) cropped to the pixels above <paramref name="threshold"/>.</summary>
    public static GlyphImage Crop(ReadOnlySpan<float> image, int rows, int columns, float threshold = 0.2f)
    {
        int left = columns, right = -1, top = rows, bottom = -1;
        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < columns; x++)
            {
                if (image[y * columns + x] > threshold)
                {
                    (left, right, top, bottom) = (Math.Min(left, x), Math.Max(right, x), Math.Min(top, y), Math.Max(bottom, y));
                }
            }
        }

        if (right < 0)
        {
            return new GlyphImage([0f], 1, 1);
        }

        int width = right - left + 1, height = bottom - top + 1;
        var pixels = new float[width * height];
        for (int y = 0; y < height; y++)
        {
            image.Slice((top + y) * columns + left, width).CopyTo(pixels.AsSpan(y * width, width));
        }

        return new GlyphImage(pixels, width, height);
    }
}

/// <summary>How <see cref="PageComposer"/> lays out a page; sizes are in glyph pixels before <see cref="Scale"/>.</summary>
public sealed record PageCompositionOptions
{
    /// <summary>Page pixels per glyph pixel (default 2, so a reader has to shrink the characters).</summary>
    public int Scale { get; init; } = 2;

    /// <summary>The height of a line (default 36).</summary>
    public int LineHeight { get; init; } = 36;

    /// <summary>Blank rows between lines (default 8).</summary>
    public int LineGap { get; init; } = 8;

    /// <summary>The page's margin (default 16).</summary>
    public int Margin { get; init; } = 16;

    /// <summary>The width of a space (default 16).</summary>
    public int WordGap { get; init; } = 16;

    /// <summary>The smallest and largest gap between letters, chosen at random for each (default 3 and 6).</summary>
    public (int Min, int Max) LetterGap { get; init; } = (3, 6);

    /// <summary>The seed of the gaps and of the glyph choices (default 1).</summary>
    public int Seed { get; init; } = 1;
}

/// <summary>
/// Writes a page of handwriting for a text out of character images (test images a model never trained on, say): dark
/// ink on white paper, one line per text line. A line in a right-to-left script is laid out as it is written (words
/// right to left, numbers left to right). For tests, demos and synthetic training pages.
/// </summary>
public static class PageComposer
{
    /// <summary>
    /// The page for <paramref name="text"/> (lines separated by '\n') as a one-channel image in [0, 1].
    /// <paramref name="glyph"/> gives an image of a character, or null when there is none (an error names it).
    /// </summary>
    public static ImageData Compose(string text, Func<char, Random, GlyphImage?> glyph, PageCompositionOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(glyph);
        options ??= new PageCompositionOptions();
        var random = new Random(options.Seed);
        var placed = new List<(int X, int Y, GlyphImage Glyph)>();
        int pageWidth = 0, y = options.Margin;
        foreach (var line in text.Replace("\r", "", StringComparison.Ordinal).Split('\n'))
        {
            int x = options.Margin;
            var words = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            // Page order is the reverse of reading order for a right-to-left line (the mapping is its own inverse).
            var visual = TextOrder.Reorder(words, WritingScript.Of(line).RightToLeft).Split(' ', StringSplitOptions.RemoveEmptyEntries);
            for (int w = 0; w < visual.Length; w++)
            {
                if (w > 0)
                {
                    x += options.WordGap;
                }

                foreach (char ch in visual[w])
                {
                    var image = glyph(ch, random) ?? throw new ArgumentException($"There is no image of '{ch}'.", nameof(glyph));
                    placed.Add((x, y + (options.LineHeight - image.Height) / 2, image));
                    x += image.Width + random.Next(options.LetterGap.Min, options.LetterGap.Max + 1);
                }
            }

            pageWidth = Math.Max(pageWidth, x + options.Margin);
            y += options.LineHeight + options.LineGap;
        }

        int scale = options.Scale, width = pageWidth * scale, height = (y + options.Margin) * scale;
        var page = new float[width * height];
        Array.Fill(page, 1f);
        foreach (var (gx, gy, image) in placed)
        {
            for (int py = 0; py < image.Height * scale; py++)
            {
                int row = gy * scale + py;
                if (row < 0 || row >= height)
                {
                    continue;
                }

                for (int px = 0; px < image.Width * scale; px++)
                {
                    int i = row * width + gx * scale + px;
                    page[i] = Math.Min(page[i], 1f - image.Pixels[py / scale * image.Width + px / scale]);
                }
            }
        }

        return new ImageData(page, 1, height, width);
    }
}
