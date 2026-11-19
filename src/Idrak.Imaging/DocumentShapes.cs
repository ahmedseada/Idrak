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
    private readonly float _scale = 0.75f, _strokeWidth = 2f, _left, _top;

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

    /// <summary>The drawing's pixel column that lands on the page's left edge (0: the drawing's own; a page cut from a larger drawing: the cut's left).</summary>
    /// <exception cref="ArgumentOutOfRangeException">Not finite.</exception>
    public float Left
    {
        get => _left;
        init => _left = float.IsFinite(value) ? value : throw new ArgumentOutOfRangeException(nameof(Left), value, "A page's left in the drawing is finite.");
    }

    /// <summary>The drawing's pixel row that lands on the page's top edge (0: the drawing's own; a page cut from a larger drawing: the cut's top).</summary>
    /// <exception cref="ArgumentOutOfRangeException">Not finite.</exception>
    public float Top
    {
        get => _top;
        init => _top = float.IsFinite(value) ? value : throw new ArgumentOutOfRangeException(nameof(Top), value, "A page's top in the drawing is finite.");
    }

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
/// their fonts with the text they stand for, each text a shape sets along a path its glyphs turned onto the path as drawn,
/// so the page's text is searchable and its drawing sharp at any size.
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
    /// The plan's areas and paths as <see cref="Write(IDocumentPageBuilder, ShapePlan, DocumentShapeStyle)"/> writes them,
    /// then its texts (<see cref="ShapePlan.Texts"/>: an emblem's name round its ring, a stamp's word) as the page draws
    /// them: what each says (one of its words <paramref name="lettering"/>'s fonts draw, those read in the page's direction
    /// first, or <paramref name="pageText"/> when it has none) set on one line in <paramref name="lettering"/> at the text's
    /// size, smaller to fit its path, centred along it, each glyph turned onto the path where its middle falls, in its tone;
    /// written glyph by glyph in reading order with the characters each stands for, so a reader finds the words.
    /// </summary>
    /// <param name="page">The page written to.</param>
    /// <param name="plan">The plan.</param>
    /// <param name="style">The scale, stroke width, colours and the page's corner in the drawing.</param>
    /// <param name="lettering">The style the texts are set in (its size each text's own).</param>
    /// <param name="pageText">The text a text with no words of its own says (a letter's name line); null: such a text is left out.</param>
    /// <param name="rightToLeft">Whether the page reads right to left (its texts' words read that way are said first).</param>
    public static void Write(IDocumentPageBuilder page, ShapePlan plan, DocumentShapeStyle style, TextStyle lettering, string? pageText = null, bool rightToLeft = false)
    {
        ArgumentNullException.ThrowIfNull(lettering);
        Write(page, plan, style);
        WriteTexts(page, plan, style, new ShapeLettering(lettering, pageText, rightToLeft));
    }

    // The plan's texts, each as the page draws it (ShapeDrawing.Text), glyph by glyph.
    internal static void WriteTexts(IDocumentPageBuilder page, ShapePlan plan, DocumentShapeStyle style, ShapeLettering lettering)
    {
        foreach (var text in plan.Texts)
        {
            WriteText(page, text, style, lettering);
        }
    }

    // A text along its path: its glyphs (spaces too, so a reader keeps its words apart) in reading order, a call for each
    // stretch of one face and scale, each glyph through the matrix that turns it onto the path at its place.
    private static void WriteText(IDocumentPageBuilder page, ShapeText text, DocumentShapeStyle style, ShapeLettering lettering)
    {
        if (ShapeDrawing.Place(text, lettering, blanks: true) is not { } placed || placed.Glyphs.Count == 0)
        {
            return;
        }

        var (said, line, glyphs) = placed;
        var order = new int[glyphs.Count];
        for (int i = 0; i < order.Length; i++)
        {
            order[i] = i;
        }

        // Reading order: by cluster, the glyphs of one cluster as drawn (a stable sort).
        var clusterOf = glyphs.Select(g => g.Glyph.Cluster).ToArray();
        Array.Sort(order, (a, b) => clusterOf[a] != clusterOf[b] ? clusterOf[a].CompareTo(clusterOf[b]) : a.CompareTo(b));
        var clusters = line.Glyphs.Select(g => g.Cluster).Distinct().Order().ToList();
        int end = said.Length;
        while (end > 0 && char.IsWhiteSpace(said[end - 1]))
        {
            end--;
        }

        var colour = style.Tone(text.Tone);
        float scale = style.Scale;
        int first = 0;
        while (first < order.Length)
        {
            var lead = glyphs[order[first]].Glyph;
            int last = first;
            while (last + 1 < order.Length && ReferenceEquals(glyphs[order[last + 1]].Glyph.Font, lead.Font) && glyphs[order[last + 1]].Glyph.Scale == lead.Scale)
            {
                last++;
            }

            var face = lead.Font;
            var run = new DocumentGlyph[last - first + 1];
            var matrices = new DocumentMatrix[run.Length];
            var own = new SortedSet<int>();
            for (int k = first; k <= last; k++)
            {
                var on = glyphs[order[k]];
                var g = on.Glyph;
                // The glyph's origin on the path, and its em turned onto it: along the path t, up from it n = (ty, -tx).
                float u = (g.X - on.Middle) * on.K, v = on.Lift * on.K;
                float x = on.Px + u * on.Tx + v * on.Ty, y = on.Py + u * on.Ty - v * on.Tx;
                float em = face.Metrics.UnitsPerEm * g.Scale * on.K * scale;
                run[k - first] = new DocumentGlyph(g.Glyph, (x - style.Left) * scale, (y - style.Top) * scale, face.Advance(g.Glyph) * g.Scale * on.K * scale);
                matrices[k - first] = new DocumentMatrix(em * on.Tx, em * on.Ty, em * on.Ty, -em * on.Tx, 0f, 0f);
                own.Add(g.Cluster);
            }

            page.Glyphs(face, run, matrices, Characters(own, clusters, said, 0, end), DocumentTextMode.Fill, colour);
            first = last + 1;
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
            paths.Add(new DocumentPath(verbs, Scaled(p, style)) { Fill = DocumentFill.NonZero, FillColour = style.Tone(area.Tone) });
        }

        foreach (var shape in plan.Paths)
        {
            var p = shape.Points.Span;
            float[] points = Scaled(p, style);
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
    public static void Write(IDocumentPageBuilder page, TextBlock block, string text, float x, float y, DocumentShapeStyle style, DocumentTextMode mode = DocumentTextMode.Fill) =>
        Write(page, block, text, x, y, style, mode, default);

    /// <summary>
    /// <see cref="Write(IDocumentPageBuilder, TextBlock, string, float, float, DocumentShapeStyle, DocumentTextMode)"/> with
    /// each glyph posed as a page drew it (<paramref name="poses"/>: one a glyph, line after line in drawing order; none:
    /// every glyph as set): moved, scaled, slanted and turned through its own matrix, in its ink's share of the colour.
    /// </summary>
    internal static void Write(IDocumentPageBuilder page, TextBlock block, string text, float x, float y, DocumentShapeStyle style, DocumentTextMode mode,
        ReadOnlySpan<GlyphPose> poses)
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

        int at = 0;                                                            // the line's first glyph among the poses
        foreach (var line in block.Lines)
        {
            int count = line.Glyphs.Count;
            var linePoses = poses.IsEmpty ? default : poses.Slice(at, count);
            at += count;
            if (count == 0)
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
            while (i < count)
            {
                var first = line.Glyphs[i];
                int j = i;
                while (j + 1 < count && ReferenceEquals(line.Glyphs[j + 1].Font, first.Font) && line.Glyphs[j + 1].Scale == first.Scale
                       && (linePoses.IsEmpty || linePoses[j + 1].Ink == linePoses[i].Ink))
                {
                    j++;
                }

                WriteRun(page, line, i, j, paragraph, end, clusters, text, x, y, style, mode, linePoses.IsEmpty ? default : linePoses.Slice(i, j - i + 1));
                i = j + 1;
            }
        }
    }

    // Glyphs [first, last] of a line, one font at one scale (and one ink), as a run standing for the characters of their
    // clusters: upright at the run's size, or each through its pose's matrix when one is posed.
    private static void WriteRun(IDocumentPageBuilder page, TextLine line, int first, int last, int paragraph, int end, List<int> clusters, string text,
        float x, float y, DocumentShapeStyle style, DocumentTextMode mode, ReadOnlySpan<GlyphPose> poses)
    {
        var font = line.Glyphs[first].Font;
        float pixelsPerUnit = line.Glyphs[first].Scale, toPoints = pixelsPerUnit * style.Scale;
        bool posed = false;
        foreach (var pose in poses)
        {
            posed |= !pose.IsIdentity;
        }

        var glyphs = new DocumentGlyph[last - first + 1];
        var matrices = posed ? new DocumentMatrix[glyphs.Length] : null;
        var own = new SortedSet<int>();
        float em = font.Metrics.UnitsPerEm * toPoints;
        for (int k = first; k <= last; k++)
        {
            var g = line.Glyphs[k];
            if (matrices is null)
            {
                glyphs[k - first] = new DocumentGlyph(g.Glyph, (x - style.Left + g.X) * style.Scale, (y - style.Top + g.Y) * style.Scale, font.Advance(g.Glyph) * toPoints);
            }
            else
            {
                // A posed glyph (GlyphPainter.Transform): its height scaled, slanted, turned about its origin, moved.
                var pose = poses[k - first];
                float cos = (float)PortableMath.Cos(pose.Rotation), sin = (float)PortableMath.Sin(pose.Rotation), tall = em * pose.Scale;
                glyphs[k - first] = new DocumentGlyph(g.Glyph, (x - style.Left + g.X + pose.Dx) * style.Scale, (y - style.Top + g.Y + pose.Dy) * style.Scale,
                    font.Advance(g.Glyph) * toPoints);
                matrices[k - first] = new DocumentMatrix(em * cos, -em * sin, tall * (pose.Shear * cos - sin), -tall * (pose.Shear * sin + cos), 0f, 0f);
            }

            own.Add(paragraph + g.Cluster);
        }

        string characters = Characters(own, clusters, text, 0, end);
        var colour = style.Tone(poses.IsEmpty ? 1f : poses[0].Ink);
        if (matrices is null)
        {
            page.Glyphs(font, em, glyphs, characters, mode, colour);
        }
        else
        {
            page.Glyphs(font, glyphs, matrices, characters, mode, colour);
        }
    }

    // Pixels of the drawing (x, y pairs) as page points: less the page's corner, times the scale.
    private static float[] Scaled(ReadOnlySpan<float> points, DocumentShapeStyle style)
    {
        var scaled = new float[points.Length];
        float scale = style.Scale, left = style.Left, top = style.Top;
        for (int i = 0; i < points.Length; i += 2)
        {
            scaled[i] = (points[i] - left) * scale;
            scaled[i + 1] = (points[i + 1] - top) * scale;
        }

        return scaled;
    }

    // The characters of a stretch's clusters (`own`), each from its start to the next cluster of the line (the last to `end`).
    private static string Characters(SortedSet<int> own, List<int> clusters, string text, int paragraph, int end)
    {
        var characters = new System.Text.StringBuilder();
        foreach (int start in own)
        {
            int at = paragraph + start;
            int next = clusters.BinarySearch(start) + 1;
            int stop = next < clusters.Count ? paragraph + clusters[next] : end;
            if (stop > at && at < text.Length)
            {
                characters.Append(text, at, Math.Min(stop, text.Length) - at);
            }
        }

        return characters.ToString();
    }
}
