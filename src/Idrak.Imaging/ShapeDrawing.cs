// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using Idrak.Imaging.Shaping;

namespace Idrak.Imaging;

/// <summary>
/// What a page gives its shapes' texts (<see cref="ShapeText"/>): the style they are set in (its size each text's own),
/// the page's own text for a text with no words of its own (null: none; such a text is left out), and the page's
/// direction (a text's words read that way are said first).
/// </summary>
internal sealed record ShapeLettering(TextStyle Style, string? PageText, bool RightToLeft);

/// <summary>
/// Shape plans drawn on a page, the one way every layout draws them: each plan's areas, then its paths through a pen,
/// then its texts along their paths (upright to the path's left, centred, smaller to fit); plans turned and moved; a
/// stamp's coverage laid on with uneven ink.
/// </summary>
internal static class ShapeDrawing
{
    /// <summary>
    /// Draws <paramref name="drawn"/> on <paramref name="layer"/>: each plan's areas, its paths through
    /// <paramref name="pen"/> (each stroke's seed mixed from <paramref name="penSeed"/> in order), then its texts set by
    /// <paramref name="lettering"/> (none drawn without it); a "shape" block for each plan that drew something when
    /// <paramref name="blocks"/> is given. Returns the box around everything drawn (its right below its left when nothing was).
    /// </summary>
    public static (float X0, float Y0, float X1, float Y1) Draw(PageCanvas canvas, Span<float> layer, IReadOnlyList<(ShapeStep Step, ShapePlan Plan)> drawn, IPen pen,
        PenSettings settings, int penSeed, float strokeWidth, List<PageBlock>? blocks, ShapeLettering? lettering)
    {
        float allX0 = float.MaxValue, allY0 = float.MaxValue, allX1 = float.MinValue, allY1 = float.MinValue;
        using var stroke = new PenStroke(256);
        using var outline = new OutlineBuilder();
        var path = ArrayPool<PenPoint>.Shared.Rent(64);
        try
        {
            int strokes = 0;
            foreach (var (_, plan) in drawn)
            {
                float x0 = float.MaxValue, y0 = float.MaxValue, x1 = float.MinValue, y1 = float.MinValue;
                foreach (var area in plan.Areas)
                {
                    outline.Clear();
                    outline.Polygon(area.Points.Span);
                    canvas.Fill(layer, outline.ToOutline(), area.Tone);
                    var (ax0, ay0, ax1, ay1) = area.Bounds();
                    (x0, y0, x1, y1) = (Math.Min(x0, ax0), Math.Min(y0, ay0), Math.Max(x1, ax1), Math.Max(y1, ay1));
                }

                foreach (var shapePath in plan.Paths)
                {
                    int count = shapePath.Count;
                    if (path.Length < count)
                    {
                        ArrayPool<PenPoint>.Shared.Return(path);
                        path = ArrayPool<PenPoint>.Shared.Rent(count);
                    }

                    float w = strokeWidth * shapePath.Width;
                    var points = shapePath.Points.Span;
                    for (int i = 0; i < count; i++)
                    {
                        path[i] = new PenPoint(points[2 * i], points[2 * i + 1], w);
                    }

                    pen.Stroke(path.AsSpan(0, count), shapePath.Closed, settings, PageDrawing.Mix(penSeed, strokes++), stroke);
                    outline.Clear();
                    StrokeOutline.Add(outline, stroke.Points, stroke.Closed, shapePath.Dash.Span, w);
                    canvas.Fill(layer, outline.ToOutline(), stroke.MeanInk() * shapePath.Tone);
                    var (px0, py0, px1, py1) = shapePath.Bounds();
                    float reach = w / 2 + pen.Stray(settings, w);
                    (x0, y0, x1, y1) = (Math.Min(x0, px0 - reach), Math.Min(y0, py0 - reach), Math.Max(x1, px1 + reach), Math.Max(y1, py1 + reach));
                }

                if (lettering is not null)
                {
                    foreach (var text in plan.Texts)
                    {
                        if (Text(canvas, layer, text, lettering) is { } box)
                        {
                            (x0, y0, x1, y1) = (Math.Min(x0, box.X0), Math.Min(y0, box.Y0), Math.Max(x1, box.X1), Math.Max(y1, box.Y1));
                        }
                    }
                }

                if (x1 > x0 && y1 > y0)
                {
                    blocks?.Add(new PageBlock("shape", "", Math.Max(0, x0), Math.Max(0, y0), Math.Min(canvas.Width, x1) - Math.Max(0, x0), Math.Min(canvas.Height, y1) - Math.Max(0, y0)));
                    (allX0, allY0, allX1, allY1) = (Math.Min(allX0, x0), Math.Min(allY0, y0), Math.Max(allX1, x1), Math.Max(allY1, y1));
                }
            }
        }
        finally
        {
            ArrayPool<PenPoint>.Shared.Return(path);
        }

        return (allX0, allY0, allX1, allY1);
    }

    /// <summary>
    /// Sets <paramref name="text"/> along its path on <paramref name="layer"/>: what it says (see <see cref="ShapeText"/>),
    /// laid out on one line in the lettering's style at the text's size, made smaller to fit the path's length and to keep
    /// its ink within the size of the path, centred along it, each glyph turned to the path where its middle falls. Returns
    /// the box around its outline, or null when nothing is drawn (no word the fonts draw, a path of no length).
    /// </summary>
    public static (float X0, float Y0, float X1, float Y1)? Text(PageCanvas canvas, Span<float> layer, ShapeText text, ShapeLettering lettering)
    {
        if (Place(text, lettering, blanks: false) is not { } placed)
        {
            return null;
        }

        // Every glyph's outline turned onto the path at its middle, all in one outline (joined glyphs fill once).
        var glyphs = placed.Glyphs;
        var verbs = new List<OutlineVerb>(glyphs.Count * 16);
        var points = new List<float>(glyphs.Count * 64);
        float x0 = float.MaxValue, y0 = float.MaxValue, x1 = float.MinValue, y1 = float.MinValue;
        foreach (var on in glyphs)
        {
            var g = on.Glyph;
            var glyph = on.Outline;
            float middle = on.Middle, lift = on.Lift, k = on.K, px = on.Px, py = on.Py, tx = on.Tx, ty = on.Ty;
            var source = glyph.Points.Span;
            for (int i = 0; i < source.Length; i += 2)
            {
                // Along the path u, up from it v: at P + u t + v n, n = (ty, -tx) the path's left.
                float u = (g.X + source[i] * g.Scale - middle) * k, v = (source[i + 1] * g.Scale + lift) * k;
                float x = px + u * tx + v * ty, y = py + u * ty - v * tx;
                points.Add(x);
                points.Add(-y);                                    // y up, as outlines are: the canvas flips it back
                (x0, y0, x1, y1) = (Math.Min(x0, x), Math.Min(y0, y), Math.Max(x1, x), Math.Max(y1, y));
            }

            verbs.AddRange(glyph.Verbs.Span);
        }

        if (verbs.Count == 0)
        {
            return null;
        }

        canvas.Fill(layer, new GlyphOutline([.. verbs], [.. points]), text.Tone);
        return (x0, y0, x1, y1);
    }

    /// <summary>
    /// A glyph of a shape's text as set along its path: the glyph and its outline, its middle and its lift above the
    /// line's baseline (pixels of the line as laid out), the text's scale <see cref="K"/> (1 or less: smaller to fit), and
    /// the point of the path its middle stands on with the path's direction there (<see cref="Tx"/>, <see cref="Ty"/>).
    /// A point (gx, gy) of the glyph's outline (font units, y up) lands at P + u t + v n, with u = (X + gx scale - middle) k,
    /// v = (gy scale + lift) k and n = (ty, -tx) the path's left.
    /// </summary>
    internal readonly record struct PathGlyph(PlacedGlyph Glyph, GlyphOutline Outline, float Middle, float Lift, float K, float Px, float Py, float Tx, float Ty);

    /// <summary>
    /// What <paramref name="text"/> says, its line as laid out, and its glyphs as set along its path (see
    /// <see cref="Text"/>): each glyph with ink at its middle, in the line's drawing order, and with
    /// <paramref name="blanks"/> each glyph without ink too (a space) at the middle of its advance; null when nothing is
    /// drawn (no word the fonts draw, a path of no length, no ink).
    /// </summary>
    internal static (string Said, TextLine Line, List<PathGlyph> Glyphs)? Place(ShapeText text, ShapeLettering lettering, bool blanks)
    {
        var p = text.Path.Span;
        int n = p.Length / 2;
        if (n < 2 || Say(text, lettering) is not { } said)
        {
            return null;
        }

        var block = TextLayout.Layout(said, lettering.Style with { Size = text.Size, LineSpacing = 1 });
        if (block.Lines.Count == 0)
        {
            return null;
        }

        // The ink's extent along the line, and above and below its baseline (a mark's raise counted).
        var line = block.Lines[0];
        float left = float.MaxValue, right = float.MinValue, above = 0, below = 0;
        foreach (var g in line.Glyphs)
        {
            var glyph = g.Font.Outline(g.Glyph);
            if (glyph.IsEmpty)
            {
                continue;
            }

            var (minX, minY, maxX, maxY) = glyph.Bounds();
            float lift = line.Baseline - g.Y;
            (left, right) = (Math.Min(left, g.X + minX * g.Scale), Math.Max(right, g.X + maxX * g.Scale));
            (above, below) = (Math.Max(above, maxY * g.Scale + lift), Math.Max(below, -(minY * g.Scale + lift)));
        }

        if (!(right > left))
        {
            return null;
        }

        float[] along = ArrayPool<float>.Shared.Rent(n);
        try
        {
            along[0] = 0;
            for (int i = 1; i < n; i++)
            {
                float dx = p[2 * i] - p[2 * i - 2], dy = p[2 * i + 1] - p[2 * i - 1];
                along[i] = along[i - 1] + MathF.Sqrt(dx * dx + dy * dy);
            }

            float length = along[n - 1];
            if (!(length > 0))
            {
                return null;
            }

            float k = Math.Min(1, length / (right - left));
            k = above > 0 ? Math.Min(k, text.Size / above) : k;
            k = below > 0 ? Math.Min(k, text.Size / below) : k;
            float start = (length - (right - left) * k) / 2;
            var cumulative = along.AsSpan(0, n);
            var placed = new List<PathGlyph>(line.Glyphs.Count);
            foreach (var g in line.Glyphs)
            {
                var glyph = g.Font.Outline(g.Glyph);
                float middle;
                if (!glyph.IsEmpty)
                {
                    var (minX, _, maxX, _) = glyph.Bounds();
                    middle = g.X + (minX + maxX) / 2 * g.Scale;
                }
                else if (blanks)
                {
                    middle = g.X + g.Font.Advance(g.Glyph) / 2f * g.Scale;
                }
                else
                {
                    continue;
                }

                float lift = line.Baseline - g.Y;
                var (px, py, tx, ty) = At(p, cumulative, start + (middle - left) * k);
                placed.Add(new PathGlyph(g, glyph, middle, lift, k, px, py, tx, ty));
            }

            return (said, line, placed);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(along);
        }
    }

    /// <summary>
    /// What <paramref name="text"/> says on a page: of its words (the page's own text when it has none), as initials when
    /// it asks, those the lettering's fonts draw with no missing glyph, those read in the page's direction first; its
    /// pick of them counted round. Null when none is drawn.
    /// </summary>
    public static string? Say(ShapeText text, ShapeLettering lettering)
    {
        IReadOnlyList<string> words = text.Words;
        if (words.Count == 0)
        {
            words = lettering.PageText is { Length: > 0 } own ? [own] : [];
        }

        List<string>? drawable = null, inDirection = null;
        foreach (string word in words)
        {
            string said = text.Initials ? Initials(word) : word.Trim();
            if (said.Length == 0 || !Drawable(said, lettering.Style))
            {
                continue;
            }

            (drawable ??= []).Add(said);
            if (Bidi.Resolve(said).RightToLeft == lettering.RightToLeft)
            {
                (inDirection ??= []).Add(said);
            }
        }

        var from = inDirection ?? drawable;
        return from is null ? null : from[text.Pick % from.Count];
    }

    /// <summary>
    /// The first letters of <paramref name="text"/>'s first three words: together and in capitals when they read left to
    /// right, a space between them when they read right to left (so they stand apart, unjoined); "" when it has none.
    /// </summary>
    public static string Initials(string text)
    {
        var letters = new List<string>(3);
        foreach (string word in text.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            for (int i = 0; i < word.Length; i++)
            {
                if (char.IsLetter(word, i))
                {
                    letters.Add(char.IsSurrogatePair(word, i) ? word.Substring(i, 2) : word[i].ToString());
                    break;
                }
            }

            if (letters.Count == 3)
            {
                break;
            }
        }

        if (letters.Count == 0)
        {
            return "";
        }

        string joined = string.Concat(letters);
        return Bidi.Resolve(joined).RightToLeft ? string.Join(" ", letters) : joined.ToUpperInvariant();
    }

    // Whether every glyph of `text` set in `style` (its fallbacks included) is one the fonts have.
    private static bool Drawable(string text, TextStyle style)
    {
        foreach (var line in TextLayout.Layout(text, style).Lines)
        {
            foreach (var g in line.Glyphs)
            {
                if (g.Glyph == 0)
                {
                    return false;
                }
            }
        }

        return true;
    }

    // The point `s` along the polyline `p` (whose lengths so far are `along`) and the unit direction there, clamped to its ends.
    private static (float X, float Y, float Tx, float Ty) At(ReadOnlySpan<float> p, ReadOnlySpan<float> along, float s)
    {
        int n = along.Length;
        s = Math.Clamp(s, 0, along[n - 1]);
        int i = along.BinarySearch(s);
        int segment = i >= 0 ? Math.Min(i, n - 2) : Math.Clamp(~i - 1, 0, n - 2);
        // A segment of no length takes the direction of the next one that has some.
        int from = segment, to = segment + 1;
        while (to < n - 1 && along[to] - along[from] <= 0)
        {
            to++;
        }

        float length = along[to] - along[from];
        float dx = p[2 * to] - p[2 * from], dy = p[2 * to + 1] - p[2 * from + 1];
        float t = length > 0 ? (s - along[from]) / length : 0;
        float norm = MathF.Sqrt(dx * dx + dy * dy);
        return norm > 0 ? (p[2 * from] + dx * t, p[2 * from + 1] + dy * t, dx / norm, dy / norm) : (p[2 * from], p[2 * from + 1], 1, 0);
    }

    /// <summary>
    /// The plans turned by <paramref name="angle"/> (radians, clockwise as seen) about (<paramref name="cx"/>,
    /// <paramref name="cy"/>) and moved by (<paramref name="dx"/>, <paramref name="dy"/>): their paths, areas and texts
    /// (a text's glyphs turn with its path); regions are left out.
    /// </summary>
    public static List<(ShapeStep Step, ShapePlan Plan)> Turned(IReadOnlyList<(ShapeStep Step, ShapePlan Plan)> drawn, float cx, float cy, float angle, float dx, float dy)
    {
        float cos = (float)PortableMath.Cos(angle), sin = (float)PortableMath.Sin(angle);
        float[] Move(ReadOnlySpan<float> source)
        {
            var moved = new float[source.Length];
            for (int i = 0; i < source.Length; i += 2)
            {
                float x = source[i] - cx, y = source[i + 1] - cy;
                moved[i] = cx + x * cos - y * sin + dx;
                moved[i + 1] = cy + x * sin + y * cos + dy;
            }

            return moved;
        }

        var turned = new List<(ShapeStep, ShapePlan)>(drawn.Count);
        foreach (var (step, plan) in drawn)
        {
            var paths = plan.Paths.Select(s => new ShapePath(Move(s.Points.Span), s.Closed, s.Width, s.Dash.ToArray(), s.Tone)).ToList();
            var areas = plan.Areas.Select(a => new ShapeArea(Move(a.Points.Span), a.Tone)).ToList();
            var texts = plan.Texts.Select(t => new ShapeText(Move(t.Path.Span), t.Size, t.Words, t.Pick, t.Initials, t.Tone)).ToList();
            turned.Add((step, new ShapePlan(paths, areas) { Texts = texts }));
        }

        return turned;
    }

    /// <summary>
    /// Lays <paramref name="coverage"/> (<paramref name="h"/> rows of <paramref name="w"/>, its top-left at
    /// (<paramref name="left"/>, <paramref name="top"/>) of the page) on <paramref name="layer"/>, ink over ink, each
    /// pixel's ink thinned by <paramref name="uneven"/> (a stamp's patchy ink); pixels off the page are left out.
    /// </summary>
    public static void LayUneven(Span<float> layer, int width, int height, ReadOnlySpan<float> coverage, int left, int top, int w, int h, UnevenInk uneven)
    {
        int fromRow = Math.Max(0, -top), toRow = Math.Min(h, height - top), fromX = Math.Max(0, -left), toX = Math.Min(w, width - left);
        for (int row = fromRow; row < toRow; row++)
        {
            var source = coverage.Slice(row * w, w);
            var target = layer.Slice((top + row) * width + left, w);
            for (int x = fromX; x < toX; x++)
            {
                float c = source[x];
                if (c <= 0)
                {
                    continue;
                }

                c *= uneven.At(x, row);
                float t = target[x];
                target[x] = t + c - t * c;
            }
        }
    }
}

/// <summary>
/// A stamp's patchy ink: smooth noise of two scales (blotches and grain) from a seed, read per pixel as the share of ink
/// left (1: all of it); its amount (0 to 1) is how uneven it is.
/// </summary>
internal sealed class UnevenInk
{
    private readonly float[] _coarse, _fine;
    private readonly int _coarseColumns, _fineColumns;
    private readonly float _coarseCell, _fineCell, _amount;

    public UnevenInk(int seed, float amount, int width, int height, float coarseCell, float fineCell)
    {
        _amount = Math.Clamp(amount, 0, 1);
        _coarseCell = Math.Max(2, coarseCell);
        _fineCell = Math.Max(1, fineCell);
        var random = new Random(seed);
        (_coarse, _coarseColumns) = Grid(random, width, height, _coarseCell);
        (_fine, _fineColumns) = Grid(random, width, height, _fineCell);
    }

    /// <summary>The share of ink left at (<paramref name="x"/>, <paramref name="y"/>) of the stamp's box.</summary>
    public float At(int x, int y)
    {
        if (_amount <= 0)
        {
            return 1;
        }

        float noise = 0.65f * Sample(_coarse, _coarseColumns, _coarseCell, x, y) + 0.35f * Sample(_fine, _fineColumns, _fineCell, x, y);
        return Math.Clamp(1 - _amount * (1.5f * noise - 0.25f), 0, 1);
    }

    // A grid of values from 0 to 1, a cell apart, over the box (one more each way).
    private static (float[] Values, int Columns) Grid(Random random, int width, int height, float cell)
    {
        int columns = (int)MathF.Ceiling(width / cell) + 2, rows = (int)MathF.Ceiling(height / cell) + 2;
        var values = new float[columns * rows];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = (float)random.NextDouble();
        }

        return (values, columns);
    }

    // The grid's value at a pixel: the four values around it, mixed by where it falls between them (smoothly).
    private static float Sample(float[] values, int columns, float cell, int x, int y)
    {
        float gx = x / cell, gy = y / cell;
        int ix = (int)gx, iy = (int)gy;
        float fx = gx - ix, fy = gy - iy;
        fx = fx * fx * (3 - 2 * fx);
        fy = fy * fy * (3 - 2 * fy);
        int at = iy * columns + ix;
        float top = values[at] + (values[at + 1] - values[at]) * fx;
        float bottom = values[at + columns] + (values[at + columns + 1] - values[at + columns]) * fx;
        return top + (bottom - top) * fy;
    }
}
