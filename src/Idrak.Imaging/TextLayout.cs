// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Imaging.Shaping;

namespace Idrak.Imaging;

/// <summary>How the lines of a block sit between its edges.</summary>
public enum TextAlignment
{
    /// <summary>At the paragraph's start: right for right-to-left text, left for left-to-right.</summary>
    Start,

    /// <summary>At the paragraph's end.</summary>
    End,

    /// <summary>Against the left edge.</summary>
    Left,

    /// <summary>Against the right edge.</summary>
    Right,

    /// <summary>Centred.</summary>
    Center,

    /// <summary>Both edges, the spaces widened (a paragraph's last line at its start).</summary>
    Justify,
}

/// <summary>How text is set: its font and size, the fonts for characters the font lacks, the line spacing, the language and features.</summary>
/// <param name="Font">The font.</param>
/// <param name="Size">The size in pixels per em.</param>
public sealed record TextStyle(IFontFace Font, float Size)
{
    /// <summary>Fonts for a run whose characters the font does not all have, tried in order (Latin or digits in an Arabic font, say).</summary>
    public IReadOnlyList<IFontFace> Fallbacks { get; init; } = [];

    /// <summary>The distance between baselines as a multiple of the font's line height (1: as the font says).</summary>
    public float LineSpacing { get; init; } = 1f;

    /// <summary>The text's language, a BCP 47 tag ("ar"), or null.</summary>
    public string? Language { get; init; }

    /// <summary>OpenType features turned on ("+smcp") or off ("-liga") over the shapers' defaults; null for the defaults.</summary>
    public IReadOnlyCollection<string>? Features { get; init; }

    /// <summary>Pixels per font unit of <paramref name="font"/> at this size.</summary>
    public float Scale(IFontFace font) => Size / font.Metrics.UnitsPerEm;
}

/// <summary>A glyph placed on a line: its font, glyph, pen position (the baseline at y, pixels) and scale (pixels per font unit).</summary>
public readonly record struct PlacedGlyph(IFontFace Font, int Glyph, float X, float Y, float Scale);

/// <summary>One line of a laid-out block: its text (in reading order), where it sits and its glyphs in drawing order.</summary>
public sealed class TextLine
{
    /// <summary>The line's text in reading (logical) order, trailing spaces left out.</summary>
    public required string Text { get; init; }

    /// <summary>Its first index in the block's text, and its length (with its trailing spaces).</summary>
    public required int Start { get; init; }

    /// <inheritdoc cref="Start"/>
    public required int Length { get; init; }

    /// <summary>The width of its ink advance, in pixels (trailing spaces left out).</summary>
    public required float Width { get; init; }

    /// <summary>Its baseline, in pixels from the block's top.</summary>
    public required float Baseline { get; init; }

    /// <summary>Whether its paragraph reads right to left.</summary>
    public required bool RightToLeft { get; init; }

    /// <summary>The glyphs, left to right as drawn, positions relative to the block's top-left corner.</summary>
    public required IReadOnlyList<PlacedGlyph> Glyphs { get; init; }

    /// <summary>The left edge of its ink advance, in pixels from the block's left.</summary>
    public required float Left { get; init; }
}

/// <summary>Text laid out in lines within a width.</summary>
public sealed class TextBlock
{
    /// <summary>The lines, top to bottom.</summary>
    public required IReadOnlyList<TextLine> Lines { get; init; }

    /// <summary>The width it was laid out in (pixels).</summary>
    public required float Width { get; init; }

    /// <summary>Its height: the last line's bottom (pixels).</summary>
    public required float Height { get; init; }

    /// <summary>The style it was set in.</summary>
    public required TextStyle Style { get; init; }
}

/// <summary>
/// Lays out text in any script: each paragraph (lines of the text) through the Unicode bidirectional algorithm, cut into
/// runs of one script and direction, each shaped by its script's shaper (<see cref="TextShapers"/>) with the style's font
/// or the first fallback that has its characters; lines broken at spaces (between CJK characters, and inside a word too
/// long for a line), each line's runs in visual order, aligned or justified.
/// </summary>
public static class TextLayout
{
    /// <summary>Lays <paramref name="text"/> out in lines no wider than <paramref name="maxWidth"/> pixels.</summary>
    /// <param name="text">The text; each line of it ('\n') is a paragraph.</param>
    /// <param name="style">The font, size and the rest.</param>
    /// <param name="maxWidth">The widest a line may be (pixels); <see cref="float.PositiveInfinity"/> for one line per paragraph.</param>
    /// <param name="alignment">Where lines sit.</param>
    /// <param name="rightToLeft">The paragraphs' direction, or null for each one's first strong character's.</param>
    public static TextBlock Layout(string text, TextStyle style, float maxWidth = float.PositiveInfinity, TextAlignment alignment = TextAlignment.Start,
        bool? rightToLeft = null)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(style);
        if (!(maxWidth > 0))
        {
            throw new ArgumentOutOfRangeException(nameof(maxWidth), maxWidth, "A line's width must be above 0.");
        }

        var metrics = style.Font.Metrics;
        float scale = style.Scale(style.Font);
        float lineHeight = metrics.LineHeight * scale * style.LineSpacing;
        float ascent = metrics.Ascender * scale;
        var lines = new List<TextLine>();
        float baseline = ascent;
        int paragraphStart = 0;
        foreach (string paragraph in text.Split('\n'))
        {
            string p = paragraph.TrimEnd('\r');
            if (p.Length == 0)
            {
                baseline += lineHeight;
                paragraphStart += paragraph.Length + 1;
                continue;
            }

            var (levels, classes, rtl) = Bidi.Resolve(p, rightToLeft);
            var breaks = BreakLines(p, levels, style, maxWidth);
            for (int b = 0; b < breaks.Count; b++)
            {
                var (start, end) = breaks[b];
                bool last = b == breaks.Count - 1;
                lines.Add(SetLine(p, paragraphStart, start, end, levels, classes, rtl, style, maxWidth, alignment, last, baseline));
                baseline += lineHeight;
            }

            paragraphStart += paragraph.Length + 1;
        }

        float height = lines.Count == 0 ? 0 : baseline - lineHeight - metrics.Descender * scale;
        return new TextBlock { Lines = lines, Width = float.IsFinite(maxWidth) ? maxWidth : lines.Count == 0 ? 0 : lines.Max(l => l.Width), Height = height, Style = style };
    }

    /// <summary>The width of <paramref name="text"/> set on one line (pixels).</summary>
    public static float Measure(string text, TextStyle style) => Layout(text, style).Lines.Sum(l => l.Width);

    // The shaped runs of [start, end) of a paragraph with its levels, in text order, each with the font that shapes it.
    private static List<(TextRun Run, byte Level, IFontFace Font, ShapedGlyph[] Glyphs)> Shape(string p, byte[] levels, int start, int end, TextStyle style)
    {
        var shaped = new List<(TextRun, byte, IFontFace, ShapedGlyph[])>();
        foreach (var (run, level) in Itemizer.Runs(p, levels, start, end, style.Language))
        {
            var font = FontFor(p, run, style);
            shaped.Add((run, level, font, TextShapers.Shape(font, p, run, style.Features)));
        }

        return shaped;
    }

    // The style's font when it has every visible character of the run, else the first fallback that has them all.
    private static IFontFace FontFor(string p, TextRun run, TextStyle style)
    {
        if (Covers(style.Font, p, run) || style.Fallbacks.Count == 0)
        {
            return style.Font;
        }

        foreach (var fallback in style.Fallbacks)
        {
            if (Covers(fallback, p, run))
            {
                return fallback;
            }
        }

        return style.Font;
    }

    private static bool Covers(IFontFace font, string p, TextRun run)
    {
        for (int i = run.Start; i < run.Start + run.Length;)
        {
            int cp = UnicodeText.CodePointAt(p, i, out int w);
            i += w;
            if (cp is ' ' or '\t' or 0xA0 || UnicodeText.IsDefaultIgnorable(cp) || UnicodeText.IsMark(cp))
            {
                continue;
            }

            if (font.Glyph(cp) == 0)
            {
                return false;
            }
        }

        return true;
    }

    // Greedy breaks: the widths of each character's clusters from one shaping of the whole paragraph, a line ending after
    // the last space (or between CJK characters) that keeps it within the width, inside a word when no such place fits.
    private static List<(int Start, int End)> BreakLines(string p, byte[] levels, TextStyle style, float maxWidth)
    {
        var widths = new float[p.Length];
        foreach (var (_, _, font, glyphs) in Shape(p, levels, 0, p.Length, style))
        {
            float s = style.Scale(font);
            foreach (var g in glyphs)
            {
                widths[g.Cluster] += g.XAdvance * s;
            }
        }

        var lines = new List<(int, int)>();
        int lineStart = 0;
        while (lineStart < p.Length)
        {
            float width = 0;
            int lastBreak = -1, i = lineStart;
            for (; i < p.Length; i++)
            {
                bool space = p[i] is ' ' or '\t';
                if (!space && width + widths[i] > maxWidth && i > lineStart)
                {
                    break;
                }

                width += widths[i];
                if (space || IsBreakAfter(p, i))
                {
                    lastBreak = i + 1;
                }
            }

            if (i >= p.Length)
            {
                lines.Add((lineStart, p.Length));
                break;
            }

            int end = lastBreak > lineStart ? lastBreak : Math.Max(i, lineStart + 1);
            if (char.IsLowSurrogate(p[Math.Min(end, p.Length - 1)]) && end < p.Length)
            {
                end++;
            }

            lines.Add((lineStart, end));
            lineStart = end;
        }

        return lines;
    }

    // A line may end after a CJK character, or after a hyphen or slash between letters.
    private static bool IsBreakAfter(string p, int i)
    {
        char c = p[i];
        if (c is '-' or '/' && i + 1 < p.Length && char.IsLetterOrDigit(p[i + 1]))
        {
            return true;
        }

        string script = UnicodeText.Script(c);
        return script is "Hani" or "Hira" or "Kana" or "Hang" or "Bopo" || c is '、' or '。';
    }

    private static TextLine SetLine(string p, int paragraphStart, int start, int end, byte[] paragraphLevels, BidiClass[] classes, bool rtl, TextStyle style,
        float maxWidth, TextAlignment alignment, bool lastOfParagraph, float baseline)
    {
        // Trailing spaces stay in the line but are not drawn or counted.
        int inkEnd = end;
        while (inkEnd > start && p[inkEnd - 1] is ' ' or '\t')
        {
            inkEnd--;
        }

        var (_, lineLevels) = Bidi.Reorder(paragraphLevels, classes, start, inkEnd, rtl);
        var levels = (byte[])paragraphLevels.Clone();
        Array.Copy(lineLevels, 0, levels, start, inkEnd - start);
        var runs = Shape(p, levels, start, inkEnd, style);

        // L2 over runs: from the highest level down to the lowest odd one, each sequence at or above it reversed.
        var order = Enumerable.Range(0, runs.Count).ToArray();
        if (runs.Count > 1)
        {
            int highest = runs.Max(r => r.Level), lowestOdd = runs.Where(r => r.Level % 2 == 1).Select(r => (int)r.Level).DefaultIfEmpty(int.MaxValue).Min();
            var runLevels = runs.Select(r => (int)r.Level).ToArray();
            for (int level = highest; level >= lowestOdd && level > 0; level--)
            {
                for (int i = 0; i < order.Length; i++)
                {
                    if (runLevels[i] < level)
                    {
                        continue;
                    }

                    int j = i;
                    while (j < order.Length && runLevels[j] >= level)
                    {
                        j++;
                    }

                    Array.Reverse(order, i, j - i);
                    Array.Reverse(runLevels, i, j - i);
                    i = j;
                }
            }
        }

        // Pen positions left to right; spaces counted for justification.
        var placed = new List<(PlacedGlyph Glyph, bool Space, float Advance)>();
        float x = 0;
        foreach (int r in order)
        {
            var (_, _, font, glyphs) = runs[r];
            float s = style.Scale(font);
            foreach (var g in glyphs)
            {
                bool space = g.Cluster < p.Length && p[g.Cluster] is ' ' or '\t' or ' ';
                placed.Add((new PlacedGlyph(font, g.Glyph, x + g.XOffset * s, baseline - g.YOffset * s, s), space, g.XAdvance * s));
                x += g.XAdvance * s;
            }
        }

        float width = x;
        var align = alignment switch
        {
            TextAlignment.Start => rtl ? TextAlignment.Right : TextAlignment.Left,
            TextAlignment.End => rtl ? TextAlignment.Left : TextAlignment.Right,
            TextAlignment.Justify when lastOfParagraph || !float.IsFinite(maxWidth) => rtl ? TextAlignment.Right : TextAlignment.Left,
            _ => alignment,
        };

        float shift = 0, extra = 0;
        if (float.IsFinite(maxWidth))
        {
            switch (align)
            {
                case TextAlignment.Right:
                    shift = maxWidth - width;
                    break;
                case TextAlignment.Center:
                    shift = (maxWidth - width) / 2;
                    break;
                case TextAlignment.Justify:
                    int spaces = placed.Count(g => g.Space);
                    extra = spaces > 0 && width < maxWidth ? (maxWidth - width) / spaces : 0;
                    break;
            }
        }

        var glyphsOut = new List<PlacedGlyph>(placed.Count);
        float added = 0;
        foreach (var (g, space, _) in placed)
        {
            glyphsOut.Add(g with { X = g.X + shift + added });
            if (space)
            {
                added += extra;
            }
        }

        return new TextLine
        {
            Text = p[start..inkEnd], Start = paragraphStart + start, Length = end - start, Width = width + added, Baseline = baseline, RightToLeft = rtl,
            Glyphs = glyphsOut, Left = shift,
        };
    }
}

/// <summary>Draws laid-out text as ink coverage (0 none, 1 full) with an outline rasterizer.</summary>
public static class TextRenderer
{
    /// <summary>
    /// Adds the glyphs of <paramref name="block"/>, its top-left corner at (<paramref name="x"/>, <paramref name="y"/>), to
    /// <paramref name="ink"/> (<paramref name="height"/> rows of <paramref name="width"/>).
    /// </summary>
    public static void Draw(TextBlock block, float x, float y, Span<float> ink, int width, int height, string rasterizer = OutlineRasterizers.Coverage)
    {
        ArgumentNullException.ThrowIfNull(block);
        var fill = OutlineRasterizers.Get(rasterizer);
        foreach (var line in block.Lines)
        {
            foreach (var g in line.Glyphs)
            {
                fill.Fill(g.Font.Outline(g.Glyph), g.Scale, x + g.X, y + g.Y, ink, width, height);
            }
        }
    }
}
