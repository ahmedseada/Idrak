// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Imaging;

/// <summary>
/// How <see cref="DocumentShapes"/> writes a page the library draws as vector content: the scale from the drawing's pixels
/// to page points, the stroke width the shapes' widths multiply, the ink and paper a tone mixes, and the stroke's ends
/// and corners.
/// </summary>
public sealed record DocumentShapeStyle
{
    private readonly float _scale = 0.75f, _strokeWidth = 2f;

    /// <summary>Page points per pixel of the drawing (0.75: a drawing at 96 pixels an inch).</summary>
    /// <exception cref="ArgumentOutOfRangeException">Not finite and above 0.</exception>
    public float Scale
    {
        get => _scale;
        init => _scale = value > 0 && float.IsFinite(value) ? value : throw new ArgumentOutOfRangeException(nameof(Scale), value, "A scale is finite and above 0.");
    }

    /// <summary>The stroke width, in pixels, a path's <see cref="ShapePath.Width"/> multiplies (2).</summary>
    /// <exception cref="ArgumentOutOfRangeException">Not finite and above 0.</exception>
    public float StrokeWidth
    {
        get => _strokeWidth;
        init => _strokeWidth = value > 0 && float.IsFinite(value) ? value : throw new ArgumentOutOfRangeException(nameof(StrokeWidth), value, "A stroke width is finite and above 0.");
    }

    /// <summary>The ink: a tone of 1 (black).</summary>
    public DocumentColour Ink { get; init; } = DocumentColour.Black;

    /// <summary>The paper a tone of 0 leaves (white).</summary>
    public DocumentColour Paper { get; init; } = DocumentColour.White;

    /// <summary>The strokes' ends (round, as a pen leaves them).</summary>
    public DocumentLineCap Cap { get; init; } = DocumentLineCap.Round;

    /// <summary>The strokes' corners (round).</summary>
    public DocumentLineJoin Join { get; init; } = DocumentLineJoin.Round;

    /// <summary>The colour a tone gives: the paper mixed towards the ink by <paramref name="tone"/> (grey when both are grey, else RGB).</summary>
    public DocumentColour Tone(float tone)
    {
        float t = Math.Clamp(tone, 0f, 1f);
        if (Ink.Space == DocumentColourSpace.Grey && Paper.Space == DocumentColourSpace.Grey)
        {
            return DocumentColour.Grey(Mix(Paper[0], Ink[0], t));
        }

        return DocumentColour.Rgb(Mix(Paper.Red, Ink.Red, t), Mix(Paper.Green, Ink.Green, t), Mix(Paper.Blue, Ink.Blue, t));
    }

    private static float Mix(float paper, float ink, float t) => Math.Clamp(paper + (ink - paper) * t, 0f, 1f);
}

/// <summary>
/// A page the library draws (the shape pipeline's <see cref="ShapePlan"/>s and text laid out by <see cref="TextLayout"/>)
/// written through any document writer's <see cref="IDocumentPageBuilder"/> as vector content, not as an image: each
/// filled area a closed path, each pen path an ideal stroke (its width, dashes and tone), each line of text its glyphs in
/// their fonts with the text they stand for, so the page's text is searchable and its drawing sharp at any size.
/// </summary>
public static class DocumentShapes
{
    /// <summary>The plan's areas (filled, in their tones) then its paths (stroked), as <see cref="Paths"/> gives them, written to <paramref name="page"/>.</summary>
    public static void Write(IDocumentPageBuilder page, ShapePlan plan, DocumentShapeStyle style)
    {
        ArgumentNullException.ThrowIfNull(page);
        foreach (var path in Paths(plan, style))
        {
            page.Path(path);
        }
    }

    /// <summary>
    /// The plan as document paths in page points: each area a closed polygon filled (nonzero) in its tone, then each path a
    /// polyline (closed when it is) stroked at the style's stroke width times its width, its dashes in those widths (an
    /// "on" of 0 a dot: a round end draws it), in its tone; a path of one point a dot.
    /// </summary>
    public static IReadOnlyList<DocumentPath> Paths(ShapePlan plan, DocumentShapeStyle style)
    {
        ArgumentNullException.ThrowIfNull(plan);
        ArgumentNullException.ThrowIfNull(style);
        var paths = new List<DocumentPath>(plan.Areas.Count + plan.Paths.Count);
        foreach (var area in plan.Areas)
        {
            var p = area.Points.Span;
            var verbs = new OutlineVerb[p.Length / 2 + 1];
            Array.Fill(verbs, OutlineVerb.Line);
            (verbs[0], verbs[^1]) = (OutlineVerb.Move, OutlineVerb.Close);
            paths.Add(new DocumentPath(verbs, Scaled(p, style.Scale)) { Fill = DocumentFill.NonZero, FillColour = style.Tone(area.Tone) });
        }

        foreach (var shape in plan.Paths)
        {
            var p = shape.Points.Span;
            float[] points = Scaled(p, style.Scale);
            if (shape.Count == 1)
            {
                points = [points[0], points[1], points[0], points[1]];             // a dot: a line of no length, its ends round
            }

            int steps = points.Length / 2;
            var verbs = new OutlineVerb[steps + (shape.Closed && shape.Count > 2 ? 1 : 0)];
            Array.Fill(verbs, OutlineVerb.Line);
            verbs[0] = OutlineVerb.Move;
            if (verbs.Length > steps)
            {
                verbs[^1] = OutlineVerb.Close;
            }

            float width = style.StrokeWidth * shape.Width * style.Scale;
            var dash = shape.Dash.Span;
            float[] lengths = new float[dash.Length];
            for (int i = 0; i < dash.Length; i++)
            {
                lengths[i] = dash[i] * width;
            }

            paths.Add(new DocumentPath(verbs, points)
            {
                Fill = DocumentFill.None, Stroke = width, Cap = style.Cap, Join = style.Join, Dash = lengths, StrokeColour = style.Tone(shape.Tone),
            });
        }

        return paths;
    }

    /// <summary>
    /// Writes <paramref name="block"/> (laid out from <paramref name="text"/> by <see cref="TextLayout.Layout"/>) with its
    /// top-left corner at pixel (<paramref name="x"/>, <paramref name="y"/>) of the drawing: each line's glyphs, a run for
    /// each stretch of one font, standing for the characters its glyphs were shaped from, in the ink (or as
    /// <paramref name="mode"/> says: <see cref="DocumentTextMode.Hidden"/> for a text layer over an image of the page).
    /// </summary>
    /// <exception cref="ArgumentException"><paramref name="text"/> is not the text the block was laid out from (its lines do not fit in it).</exception>
    public static void Write(IDocumentPageBuilder page, TextBlock block, string text, float x, float y, DocumentShapeStyle style, DocumentTextMode mode = DocumentTextMode.Fill)
    {
        ArgumentNullException.ThrowIfNull(page);
        ArgumentNullException.ThrowIfNull(block);
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(style);

        // Where each paragraph (a line of the text) starts: clusters count from their paragraph's start.
        var paragraphs = new List<int> { 0 };
        for (int i = text.IndexOf('\n'); i >= 0; i = text.IndexOf('\n', i + 1))
        {
            paragraphs.Add(i + 1);
        }

        var colour = style.Tone(1f);
        foreach (var line in block.Lines)
        {
            if (line.Glyphs.Count == 0)
            {
                continue;
            }

            if (line.Start < 0 || line.Start + line.Length > text.Length + 1)
            {
                throw new ArgumentException($"The text is not the block's: a line starts at {line.Start} with {line.Length} characters, past its {text.Length}.", nameof(text));
            }

            int found = paragraphs.BinarySearch(line.Start);
            int paragraph = paragraphs[found >= 0 ? found : Math.Max(0, ~found - 1)];
            int end = Math.Min(text.Length, line.Start + line.Length);
            while (end > line.Start && char.IsWhiteSpace(text[end - 1]))
            {
                end--;                                                         // trailing spaces draw nothing
            }

            // The line's clusters in text order, each up to the next (the last to the line's end).
            var clusters = line.Glyphs.Select(g => paragraph + g.Cluster).Distinct().Order().ToList();
            int i = 0;
            while (i < line.Glyphs.Count)
            {
                var first = line.Glyphs[i];
                int j = i;
                while (j + 1 < line.Glyphs.Count && ReferenceEquals(line.Glyphs[j + 1].Font, first.Font) && line.Glyphs[j + 1].Scale == first.Scale)
                {
                    j++;
                }

                WriteRun(page, line, i, j, paragraph, end, clusters, text, x, y, style, colour, mode);
                i = j + 1;
            }
        }
    }

    // Glyphs [first, last] of a line, one font at one scale, as a run standing for the characters of their clusters.
    private static void WriteRun(IDocumentPageBuilder page, TextLine line, int first, int last, int paragraph, int end, List<int> clusters, string text,
        float x, float y, DocumentShapeStyle style, DocumentColour colour, DocumentTextMode mode)
    {
        var font = line.Glyphs[first].Font;
        float pixelsPerUnit = line.Glyphs[first].Scale, toPoints = pixelsPerUnit * style.Scale;
        var glyphs = new DocumentGlyph[last - first + 1];
        var own = new SortedSet<int>();
        for (int k = first; k <= last; k++)
        {
            var g = line.Glyphs[k];
            glyphs[k - first] = new DocumentGlyph(g.Glyph, (x + g.X) * style.Scale, (y + g.Y) * style.Scale, font.Advance(g.Glyph) * toPoints);
            own.Add(paragraph + g.Cluster);
        }

        var characters = new System.Text.StringBuilder();
        foreach (int start in own)
        {
            int next = clusters.BinarySearch(start) + 1;
            int stop = next < clusters.Count ? clusters[next] : end;
            if (stop > start && start < text.Length)
            {
                characters.Append(text, start, Math.Min(stop, text.Length) - start);
            }
        }

        page.Glyphs(font, font.Metrics.UnitsPerEm * toPoints, glyphs, characters.ToString(), mode, colour);
    }

    private static float[] Scaled(ReadOnlySpan<float> points, float scale)
    {
        var scaled = new float[points.Length];
        for (int i = 0; i < points.Length; i++)
        {
            scaled[i] = points[i] * scale;
        }

        return scaled;
    }
}
