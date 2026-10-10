// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Data.Abstractions;

namespace Idrak.Samples.ArabicOcr.Tests;

/// <summary>
/// Generated text images, no fonts: each of eight Arabic letters is a few strokes (rectangles) in a 18-pixel-high cell,
/// with its dots above or below as Arabic has them (so lines have marks apart from their body), drawn right to left as
/// Arabic is written, with jitter. Ink is dark on light paper, as a scan.
/// </summary>
internal static class Synthetic
{
    public const int LineHeight = 18;

    // Each letter: its width and its strokes (x, y, width, height) in the cell, x from the cell's left.
    private static readonly Dictionary<char, (int Width, (int X, int Y, int W, int H)[] Strokes)> Glyphs = new()
    {
        ['ا'] = (4, [(1, 3, 2, 10)]),                                                                  // alef: a bar
        ['ب'] = (10, [(0, 10, 10, 2), (0, 7, 2, 4), (8, 7, 2, 4), (4, 14, 2, 2)]),                     // beh: a bowl, a dot below
        ['ت'] = (10, [(0, 10, 10, 2), (0, 7, 2, 4), (8, 7, 2, 4), (2, 4, 2, 2), (6, 4, 2, 2)]),        // teh: two dots above
        ['ث'] = (10, [(0, 10, 10, 2), (0, 7, 2, 4), (8, 7, 2, 4), (2, 4, 2, 2), (6, 4, 2, 2), (4, 1, 2, 2)]),   // theh: three dots
        ['د'] = (8, [(4, 5, 2, 4), (6, 8, 2, 3), (0, 10, 8, 2)]),                                      // dal: an angle
        ['ر'] = (7, [(4, 8, 2, 3), (3, 10, 2, 3), (1, 12, 2, 3)]),                                     // reh: down to the left
        ['س'] = (12, [(0, 10, 12, 2), (0, 7, 2, 4), (5, 7, 2, 4), (10, 7, 2, 4)]),                     // seen: three teeth
        ['م'] = (7, [(1, 7, 5, 2), (1, 7, 2, 5), (4, 7, 2, 5), (1, 10, 5, 2), (0, 11, 2, 5)]),         // meem: a loop and a tail
    };

    /// <summary>The letters drawn.</summary>
    public static readonly char[] Letters = [.. Glyphs.Keys];

    /// <summary>Random text: <paramref name="min"/> to <paramref name="max"/> letters, a space now and then (never first, last or twice).</summary>
    public static string Text(Random random, int min, int max)
    {
        int count = random.Next(min, max + 1);
        var text = new System.Text.StringBuilder();
        for (int i = 0; i < count; i++)
        {
            if (i > 0 && i < count - 1 && text[^1] != ' ' && random.Next(5) == 0)
            {
                text.Append(' ');
            }

            text.Append(Letters[random.Next(Letters.Length)]);
        }

        return text.ToString();
    }

    /// <summary>The ink of a line of text ([height, width], 1 on the strokes), the first letter at the right.</summary>
    public static (float[] Ink, int Width) Line(string text, Random random, int margin = 3)
    {
        int width = margin * 2 + text.Sum(c => c == ' ' ? 6 : Glyphs[c].Width + 3);
        var ink = new float[LineHeight * width];
        int cursor = width - margin;
        foreach (char c in text)
        {
            if (c == ' ')
            {
                cursor -= 6;
                continue;
            }

            var (glyphWidth, strokes) = Glyphs[c];
            int left = cursor - glyphWidth, dy = random.Next(-1, 2);
            foreach (var (x, y, w, h) in strokes)
            {
                for (int yy = y + dy; yy < y + dy + h; yy++)
                {
                    for (int xx = left + x; xx < left + x + w; xx++)
                    {
                        if (yy >= 0 && yy < LineHeight && xx >= 0 && xx < width)
                        {
                            ink[yy * width + xx] = 1f;
                        }
                    }
                }
            }

            cursor = left - 2 - random.Next(0, 2);
        }

        return (ink, width);
    }

    /// <summary>Ink as a scan's grey: dark ink on light paper with a little noise.</summary>
    public static ImageData Scan(float[] ink, int width, int height, Random random)
    {
        var grey = new float[ink.Length];
        for (int i = 0; i < ink.Length; i++)
        {
            grey[i] = Math.Clamp(0.95f - 0.8f * ink[i] + (float)(random.NextDouble() - 0.5) * 0.06f, 0f, 1f);
        }

        return new ImageData(grey, 1, height, width);
    }

    /// <summary>A line image as a scan.</summary>
    public static ImageData LineImage(string text, Random random)
    {
        var (ink, width) = Line(text, random);
        return Scan(ink, width, LineHeight, random);
    }

    /// <summary>
    /// A page of <paramref name="lines"/> (right-aligned, 12 rows apart), turned by <paramref name="skewDegrees"/> (each ink
    /// pixel moved down by its distance from the centre times the slope). Also each line's rows on the page (unskewed).
    /// </summary>
    public static (ImageData Page, (int Top, int Bottom)[] Rows) Page(IReadOnlyList<string> lines, Random random, double skewDegrees = 0)
    {
        var rendered = lines.Select(l => Line(l, random)).ToArray();
        int width = rendered.Max(r => r.Width) + 40, pitch = LineHeight + 12;
        int height = 20 + pitch * lines.Count + 10;
        var ink = new float[width * height];
        double slope = Math.Tan(skewDegrees * Math.PI / 180);
        var rows = new (int, int)[lines.Count];
        for (int n = 0; n < rendered.Length; n++)
        {
            var (lineInk, lineWidth) = rendered[n];
            int top = 20 + n * pitch, left = width - 20 - lineWidth;
            rows[n] = (top, top + LineHeight);
            for (int y = 0; y < LineHeight; y++)
            {
                for (int x = 0; x < lineWidth; x++)
                {
                    if (lineInk[y * lineWidth + x] > 0)
                    {
                        int px = left + x, py = top + y + (int)Math.Round((px - width / 2.0) * slope);
                        if (py >= 0 && py < height)
                        {
                            ink[py * width + px] = 1f;
                        }
                    }
                }
            }
        }

        return (Scan(ink, width, height, random), rows);
    }

    /// <summary>Writes <paramref name="image"/> as a PNG (the library's encoders).</summary>
    public static void Save(string path, ImageData image) => ImageEncoders.Save(path, image);
}
