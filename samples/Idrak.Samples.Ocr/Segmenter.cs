// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Samples.Ocr;

/// <summary>One character cut out of a page, normalized to the classifier's input size.</summary>
/// <param name="Pixels">Cell×Cell image, ink = 1.</param>
/// <param name="SpaceBefore">Whether a word gap precedes this character.</param>
/// <param name="NewLineBefore">Whether this character starts a new text line (after the first).</param>
internal readonly record struct Glyph(float[] Pixels, bool SpaceBefore, bool NewLineBefore);

/// <summary>
/// Splits an image into text lines (rows with ink), lines into characters (columns with ink), and
/// detects word gaps. Works for printed text with separated characters, like the rendered samples.
/// </summary>
internal static class Segmenter
{
    private const float InkThreshold = 0.45f;

    public static List<Glyph> Segment(float[] pixels, int width, int height)
    {
        var glyphs = new List<Glyph>();
        bool firstLine = true;
        foreach (var (top, bottom) in Runs(height, r => RowHasInk(pixels, width, r), minGap: 2))
        {
            int lineHeight = bottom - top;
            var columns = Runs(width, c => ColumnHasInk(pixels, width, top, bottom, c), minGap: 1);
            int previousEnd = -1;
            bool firstInLine = true;
            foreach (var (left, right) in columns)
            {
                if (Ink(pixels, width, left, right, top, bottom) < 1.5f)
                {
                    continue; // a speck of noise
                }

                bool space = previousEnd >= 0 && left - previousEnd > Math.Max(5, lineHeight * 0.4f);
                glyphs.Add(new Glyph(Normalize(pixels, width, left, right, top, bottom), space, firstInLine && !firstLine));
                previousEnd = right;
                firstInLine = false;
            }

            firstLine = false;
        }

        return glyphs;
    }

    /// <summary>Contiguous [start, end) ranges where <paramref name="inked"/> holds, merging gaps shorter than minGap.</summary>
    private static List<(int Start, int End)> Runs(int count, Func<int, bool> inked, int minGap)
    {
        var runs = new List<(int, int)>();
        int start = -1, lastInk = -1;
        for (int i = 0; i < count; i++)
        {
            if (!inked(i))
            {
                continue;
            }

            if (start < 0)
            {
                start = i;
            }
            else if (i - lastInk > minGap)
            {
                runs.Add((start, lastInk + 1));
                start = i;
            }

            lastInk = i;
        }

        if (start >= 0)
        {
            runs.Add((start, lastInk + 1));
        }

        return runs;
    }

    private static bool RowHasInk(float[] p, int width, int row)
    {
        int count = 0;
        for (int c = 0; c < width; c++)
        {
            count += p[row * width + c] > InkThreshold ? 1 : 0;
        }

        return count >= 2;
    }

    private static bool ColumnHasInk(float[] p, int width, int top, int bottom, int column)
    {
        for (int r = top; r < bottom; r++)
        {
            if (p[r * width + column] > InkThreshold)
            {
                return true;
            }
        }

        return false;
    }

    private static float Ink(float[] p, int width, int left, int right, int top, int bottom)
    {
        float sum = 0;
        for (int r = top; r < bottom; r++)
        {
            for (int c = left; c < right; c++)
            {
                sum += p[r * width + c] > InkThreshold ? 1 : 0;
            }
        }

        return sum;
    }

    /// <summary>Crops the character's ink box and scales it (bilinear) to the training glyph height, centered in a Cell×Cell image.</summary>
    private static float[] Normalize(float[] p, int width, int left, int right, int top, int bottom)
    {
        while (top < bottom && !RowRangeHasInk(p, width, left, right, top))
        {
            top++;
        }

        while (bottom > top && !RowRangeHasInk(p, width, left, right, bottom - 1))
        {
            bottom--;
        }

        int w = right - left, h = Math.Max(bottom - top, 1);
        float scale = Math.Min(Renderer.NormalHeight / h, (Renderer.Cell - 2f) / w);
        float outW = w * scale, outH = h * scale;
        float x0 = (Renderer.Cell - outW) / 2, y0 = (Renderer.Cell - outH) / 2;
        var cell = new float[Renderer.Cell * Renderer.Cell];
        for (int y = 0; y < Renderer.Cell; y++)
        {
            for (int x = 0; x < Renderer.Cell; x++)
            {
                float sx = (x + 0.5f - x0) / scale - 0.5f + left;
                float sy = (y + 0.5f - y0) / scale - 0.5f + top;
                cell[y * Renderer.Cell + x] = Sample(p, width, sx, sy, left, right, top, bottom);
            }
        }

        return cell;
    }

    private static bool RowRangeHasInk(float[] p, int width, int left, int right, int row)
    {
        for (int c = left; c < right; c++)
        {
            if (p[row * width + c] > InkThreshold)
            {
                return true;
            }
        }

        return false;
    }

    private static float Sample(float[] p, int width, float x, float y, int left, int right, int top, int bottom)
    {
        int x0 = (int)MathF.Floor(x), y0 = (int)MathF.Floor(y);
        float fx = x - x0, fy = y - y0;
        float At(int cx, int cy) => cx >= left && cx < right && cy >= top && cy < bottom ? p[cy * width + cx] : 0f;
        return (At(x0, y0) * (1 - fx) + At(x0 + 1, y0) * fx) * (1 - fy) + (At(x0, y0 + 1) * (1 - fx) + At(x0 + 1, y0 + 1) * fx) * fy;
    }
}
