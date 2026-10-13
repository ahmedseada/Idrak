// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;

namespace Idrak.Imaging;

/// <summary>
/// "coverage": fills an outline with each pixel's exact covered area (signed area accumulated along every edge, then one
/// running sum per row: the nonzero rule for contours that do not overlap the other way, which glyph outlines do not).
/// Curves are flattened to lines within 1/32 of a pixel (as FreeType). No hinting: an outline is drawn as designed, at any size and
/// any fractional position.
/// </summary>
public sealed class CoverageRasterizer : IOutlineRasterizer
{
    // The most a flattened curve strays from the true one, in pixels (FreeType splits to about this).
    private const float Tolerance = 1f / 32;

    /// <inheritdoc/>
    public string Name => OutlineRasterizers.Coverage;

    /// <inheritdoc/>
    public void Fill(GlyphOutline outline, float scale, float x, float y, Span<float> coverage, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(outline);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        if (coverage.Length < (long)width * height)
        {
            throw new ArgumentException($"{coverage.Length} values hold no {width} x {height} image.", nameof(coverage));
        }

        if (outline.IsEmpty || !(scale > 0f))
        {
            return;
        }

        // The outline's box in pixels (y down), with a pixel of room on the right for the accumulation's carry.
        var (minX, minY, maxX, maxY) = outline.Bounds();
        int left = (int)MathF.Floor(x + minX * scale), right = (int)MathF.Ceiling(x + maxX * scale);
        int top = (int)MathF.Floor(y - maxY * scale), bottom = (int)MathF.Ceiling(y - minY * scale);
        if (right <= 0 || bottom <= 0 || left >= width || top >= height)
        {
            return;                                                             // nothing of it on the image
        }

        int boxWidth = right - left + 2, boxHeight = bottom - top;
        if (boxHeight <= 0 || right - left <= 0)
        {
            return;
        }

        int size = boxWidth * boxHeight;
        float[] area = ArrayPool<float>.Shared.Rent(size);
        try
        {
            var accumulation = area.AsSpan(0, size);
            accumulation.Clear();
            Trace(outline, scale, x - left, y - top, accumulation, boxWidth, boxHeight);
            Composite(accumulation, boxWidth, boxHeight, left, top, coverage, width, height);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(area);
        }
    }

    // Every edge of the outline, flattened, into the accumulation buffer (origin at (ox, oy) in the box).
    private static void Trace(GlyphOutline outline, float scale, float ox, float oy, Span<float> a, int w, int h)
    {
        var verbs = outline.Verbs.Span;
        var p = outline.Points.Span;
        int at = 0;
        float sx = 0, sy = 0, cx = 0, cy = 0;                                    // the contour's start and the pen
        foreach (var verb in verbs)
        {
            switch (verb)
            {
                case OutlineVerb.Move:
                    if (cx != sx || cy != sy)
                    {
                        Line(cx, cy, sx, sy, a, w, h);                          // an unclosed contour closes
                    }

                    sx = cx = ox + p[at] * scale;
                    sy = cy = oy - p[at + 1] * scale;
                    at += 2;
                    break;
                case OutlineVerb.Line:
                {
                    float nx = ox + p[at] * scale, ny = oy - p[at + 1] * scale;
                    Line(cx, cy, nx, ny, a, w, h);
                    (cx, cy) = (nx, ny);
                    at += 2;
                    break;
                }

                case OutlineVerb.Quadratic:
                {
                    float qx = ox + p[at] * scale, qy = oy - p[at + 1] * scale, nx = ox + p[at + 2] * scale, ny = oy - p[at + 3] * scale;
                    // Uniform steps: a parabola's chords stray at most |p0 - 2 p1 + p2| / (8 n²).
                    float dd = MathF.Sqrt(Square(cx - 2 * qx + nx) + Square(cy - 2 * qy + ny));
                    int n = Math.Max(1, (int)MathF.Ceiling(MathF.Sqrt(dd / (8 * Tolerance))));
                    float px = cx, py = cy;
                    for (int i = 1; i <= n; i++)
                    {
                        float t = (float)i / n, u = 1 - t;
                        float qxi = u * u * cx + 2 * u * t * qx + t * t * nx, qyi = u * u * cy + 2 * u * t * qy + t * t * ny;
                        Line(px, py, qxi, qyi, a, w, h);
                        (px, py) = (qxi, qyi);
                    }

                    (cx, cy) = (nx, ny);
                    at += 4;
                    break;
                }

                case OutlineVerb.Cubic:
                {
                    float c1x = ox + p[at] * scale, c1y = oy - p[at + 1] * scale, c2x = ox + p[at + 2] * scale, c2y = oy - p[at + 3] * scale;
                    float nx = ox + p[at + 4] * scale, ny = oy - p[at + 5] * scale;
                    // Uniform steps: a cubic's chords stray at most 3/4 max|second difference| / n².
                    float dd = MathF.Max(MathF.Sqrt(Square(cx - 2 * c1x + c2x) + Square(cy - 2 * c1y + c2y)),
                        MathF.Sqrt(Square(c1x - 2 * c2x + nx) + Square(c1y - 2 * c2y + ny)));
                    int n = Math.Max(1, (int)MathF.Ceiling(MathF.Sqrt(3 * dd / (4 * Tolerance))));
                    float px = cx, py = cy;
                    for (int i = 1; i <= n; i++)
                    {
                        float t = (float)i / n, u = 1 - t;
                        float b0 = u * u * u, b1 = 3 * u * u * t, b2 = 3 * u * t * t, b3 = t * t * t;
                        float qx = b0 * cx + b1 * c1x + b2 * c2x + b3 * nx, qy = b0 * cy + b1 * c1y + b2 * c2y + b3 * ny;
                        Line(px, py, qx, qy, a, w, h);
                        (px, py) = (qx, qy);
                    }

                    (cx, cy) = (nx, ny);
                    at += 6;
                    break;
                }

                case OutlineVerb.Close:
                    if (cx != sx || cy != sy)
                    {
                        Line(cx, cy, sx, sy, a, w, h);
                    }

                    (cx, cy) = (sx, sy);
                    break;
            }
        }

        if (cx != sx || cy != sy)
        {
            Line(cx, cy, sx, sy, a, w, h);
        }
    }

    private static float Square(float v) => v * v;

    // One edge's signed area into the rows it crosses: each row gets the area left of the edge in the cells it passes,
    // and the rest of its height in the cell after (the running sum carries it to the right).
    private static void Line(float x0, float y0, float x1, float y1, Span<float> a, int w, int h)
    {
        if (y0 == y1)
        {
            return;
        }

        float direction = 1f;
        if (y0 > y1)
        {
            (x0, y0, x1, y1) = (x1, y1, x0, y0);
            direction = -1f;
        }

        float dxdy = (x1 - x0) / (y1 - y0);
        float x = x0;
        if (y0 < 0)
        {
            x -= y0 * dxdy;
        }

        int rowEnd = Math.Min(h, (int)MathF.Ceiling(y1));
        for (int row = Math.Max(0, (int)y0); row < rowEnd; row++)
        {
            int start = row * w;
            float dy = MathF.Min(row + 1, y1) - MathF.Max(row, y0);
            float xNext = x + dxdy * dy;
            float d = dy * direction;
            float lo = MathF.Min(x, xNext), hi = MathF.Max(x, xNext);
            lo = MathF.Max(lo, 0f);
            hi = MathF.Min(hi, w - 1.000001f);
            float loFloor = MathF.Floor(lo);
            int loCell = (int)loFloor, hiCell = (int)MathF.Ceiling(hi);
            if (hiCell <= loCell + 1)
            {
                float mid = 0.5f * (x + xNext) - loFloor;
                a[start + loCell] += d - d * mid;
                a[start + loCell + 1] += d * mid;
            }
            else
            {
                float s = 1f / (hi - lo);
                float loFraction = lo - loFloor;
                float first = 0.5f * s * (1f - loFraction) * (1f - loFraction);
                float hiFraction = hi - hiCell + 1f;
                float last = 0.5f * s * hiFraction * hiFraction;
                a[start + loCell] += d * first;
                if (hiCell == loCell + 2)
                {
                    a[start + loCell + 1] += d * (1f - first - last);
                }
                else
                {
                    float second = s * (1.5f - loFraction);
                    a[start + loCell + 1] += d * (second - first);
                    for (int cell = loCell + 2; cell < hiCell - 1; cell++)
                    {
                        a[start + cell] += d * s;
                    }

                    float before = second + (hiCell - loCell - 3) * s;
                    a[start + hiCell - 1] += d * (1f - before - last);
                }

                a[start + hiCell] += d * last;
            }

            x = xNext;
        }
    }

    // The running sum of each box row is the pixel's coverage; laid onto the image as ink over ink, row by row.
    private static void Composite(ReadOnlySpan<float> a, int boxWidth, int boxHeight, int left, int top, Span<float> coverage, int width, int height)
    {
        int rowFrom = Math.Max(0, -top), rowTo = Math.Min(boxHeight, height - top);
        for (int row = rowFrom; row < rowTo; row++)
        {
            var source = a.Slice(row * boxWidth, boxWidth);
            var target = coverage.Slice((top + row) * width, width);
            float sum = 0f;
            for (int col = 0; col < boxWidth; col++)
            {
                sum += source[col];
                int x = left + col;
                if ((uint)x >= (uint)width)
                {
                    continue;
                }

                float c = MathF.Min(1f, MathF.Abs(sum));
                if (c > 0f)
                {
                    float t = target[x];
                    target[x] = t + c - t * c;                                   // 1 - (1 - t)(1 - c)
                }
            }
        }
    }
}
