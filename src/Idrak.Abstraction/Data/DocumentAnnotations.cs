// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Data;

/// <summary>How an annotation behaves (the document's flags).</summary>
[Flags]
public enum DocumentAnnotationFlags
{
    /// <summary>Nothing set.</summary>
    None = 0,

    /// <summary>Not shown when its kind is not one the reader knows.</summary>
    Invisible = 1,

    /// <summary>Neither shown nor printed.</summary>
    Hidden = 2,

    /// <summary>Printed with the page.</summary>
    Print = 4,

    /// <summary>Its size does not follow the zoom.</summary>
    NoZoom = 8,

    /// <summary>It does not turn with the page.</summary>
    NoRotate = 16,

    /// <summary>Not shown on screen (it may be printed).</summary>
    NoView = 32,

    /// <summary>It does not take input.</summary>
    ReadOnly = 64,

    /// <summary>It cannot be moved, resized or removed.</summary>
    Locked = 128,

    /// <summary><see cref="NoView"/> turns over when it is pointed at.</summary>
    ToggleNoView = 256,

    /// <summary>Its contents cannot be changed.</summary>
    LockedContents = 512,
}

/// <summary>How a <see cref="DocumentDestination"/> shows its page.</summary>
public enum DocumentDestinationFit : byte
{
    /// <summary>At a position (its left and top edges) and a zoom, each kept from the viewer when not given.</summary>
    Position,

    /// <summary>The whole page in the window.</summary>
    Page,

    /// <summary>The page's width in the window, its top at <see cref="DocumentDestination.Y"/>.</summary>
    Width,

    /// <summary>The page's height in the window, its left edge at <see cref="DocumentDestination.X"/>.</summary>
    Height,

    /// <summary>A box of the page in the window (<see cref="DocumentDestination.X"/>, <see cref="DocumentDestination.Y"/>, <see cref="DocumentDestination.Right"/>, <see cref="DocumentDestination.Bottom"/>).</summary>
    Box,

    /// <summary>The bounds of the page's content in the window.</summary>
    Bounds,

    /// <summary>The width of the page's content in the window, its top at <see cref="DocumentDestination.Y"/>.</summary>
    BoundsWidth,

    /// <summary>The height of the page's content in the window, its left edge at <see cref="DocumentDestination.X"/>.</summary>
    BoundsHeight,
}

/// <summary>
/// Where a run stands in a tagged document's structure (the reading order its author gave): the element holding it, by
/// the type the document names ("P", "H1", "Table", or its own), its place among the structure's elements, and the place
/// of its marked content in the structure's order. An artifact (a page number, a running header the author marked as not
/// part of the content) has <see cref="Element"/> -1.
/// </summary>
/// <param name="Type">The element's type as the document names it ("Artifact" for an artifact).</param>
/// <param name="Element">The element's place among the structure's elements, in the structure's order (-1: an artifact).</param>
/// <param name="Order">The run's marked content's place in the structure's order (-1: an artifact).</param>
public readonly record struct DocumentTag(string Type, int Element, int Order)
{
    /// <summary>An artifact: not part of the tagged content.</summary>
    public static DocumentTag Artifact { get; } = new("Artifact", -1, -1);

    /// <summary>Whether the run is an artifact, outside the tagged content.</summary>
    public bool IsArtifact => Element < 0;
}

/// <summary>
/// An area to redact on a page: a box in page points from the page's top-left corner. Applying it removes what lies under
/// it for real (text, the parts of paths and the pixels of images), then fills it with <see cref="Fill"/>.
/// </summary>
/// <param name="Box">The area.</param>
public sealed record DocumentRedaction(DocumentBox Box)
{
    /// <summary>The colour the area is filled with once applied, and the colour image pixels under it take (black); null leaves it unfilled (pixels under it then black too).</summary>
    public DocumentColour? Fill { get; init; } = DocumentColour.Black;
}

/// <summary>What applying a page's redactions removed: counted per kind of content.</summary>
/// <param name="Page">The page's index.</param>
public sealed record DocumentRedactionResult(int Page)
{
    /// <summary>The areas applied.</summary>
    public int Areas { get; init; }

    /// <summary>The glyphs removed from the page's text (and the hidden text over a scan).</summary>
    public int Glyphs { get; init; }

    /// <summary>The characters those glyphs stood for.</summary>
    public int Characters { get; init; }

    /// <summary>The paths removed whole (inside an area).</summary>
    public int PathsRemoved { get; init; }

    /// <summary>The paths and shadings clipped (crossing an area: only their parts outside it drawn).</summary>
    public int PathsClipped { get; init; }

    /// <summary>The images whose pixels under an area were overwritten.</summary>
    public int Images { get; init; }

    /// <summary>The pixels overwritten.</summary>
    public long Pixels { get; init; }

    /// <summary>The annotations removed (those over an area, and the redaction marks applied).</summary>
    public int Annotations { get; init; }

    /// <summary>Whether anything was removed.</summary>
    public bool Removed => Glyphs > 0 || PathsRemoved > 0 || PathsClipped > 0 || Pixels > 0 || Annotations > 0;
}

/// <summary>
/// Areas to redact found on a page, format-neutral: every place a text stands (<see cref="Find"/>), or the page's own
/// redaction marks (<see cref="Marked"/>). An editor applies them (<see cref="IDocumentEditor.Redact"/>).
/// </summary>
public static class DocumentRedactions
{
    /// <summary>
    /// An area over every place <paramref name="text"/> stands on <paramref name="page"/> (read with its text, and its
    /// glyphs for exact places): within each run, matched as <paramref name="comparison"/> says on the run's text; the area
    /// spans the matched glyphs (their origins and advances) and the run's height. A run whose glyphs are not one a
    /// character (a ligature) or were not read gets the share of its box its characters take.
    /// </summary>
    public static IReadOnlyList<DocumentRedaction> Find(DocumentPage page, string text, StringComparison comparison = StringComparison.Ordinal, DocumentColour? fill = null)
    {
        ArgumentNullException.ThrowIfNull(page);
        ArgumentException.ThrowIfNullOrEmpty(text);
        var colour = fill ?? DocumentColour.Black;
        var areas = new List<DocumentRedaction>();
        foreach (var run in page.Runs)
        {
            if (run.Box.IsEmpty)
            {
                continue;
            }

            string own = run.Text;
            for (int at = own.IndexOf(text, comparison); at >= 0; at = own.IndexOf(text, at + Math.Max(1, text.Length), comparison))
            {
                areas.Add(new DocumentRedaction(Span(run, at, text.Length)) { Fill = colour });
                if (at + text.Length >= own.Length)
                {
                    break;
                }
            }
        }

        return areas;
    }

    /// <summary>
    /// The page's own redaction marks (<see cref="DocumentAnnotationKinds.Redaction"/>, read with the annotations) as areas:
    /// each of a mark's quads, or its box when it has none, filled with its interior colour (unfilled when it has none).
    /// </summary>
    public static IReadOnlyList<DocumentRedaction> Marked(DocumentPage page)
    {
        ArgumentNullException.ThrowIfNull(page);
        var areas = new List<DocumentRedaction>();
        foreach (var annotation in page.Annotations)
        {
            if (!string.Equals(annotation.Kind, DocumentAnnotationKinds.Redaction, StringComparison.Ordinal))
            {
                continue;
            }

            var points = annotation.Points.Span;
            if (points.Length >= 8)
            {
                for (int q = 0; q + 8 <= points.Length; q += 8)
                {
                    float x0 = float.MaxValue, y0 = float.MaxValue, x1 = float.MinValue, y1 = float.MinValue;
                    for (int k = q; k < q + 8; k += 2)
                    {
                        (x0, x1, y0, y1) = (Math.Min(x0, points[k]), Math.Max(x1, points[k]), Math.Min(y0, points[k + 1]), Math.Max(y1, points[k + 1]));
                    }

                    areas.Add(new DocumentRedaction(new DocumentBox(x0, y0, x1 - x0, y1 - y0)) { Fill = annotation.InteriorColour });
                }
            }
            else if (!annotation.Box.IsEmpty)
            {
                areas.Add(new DocumentRedaction(annotation.Box) { Fill = annotation.InteriorColour });
            }
        }

        return areas;
    }

    // The box of characters [at, at + length) of a run: from its glyphs when there is one a character, else its share of the box.
    private static DocumentBox Span(DocumentTextRun run, int at, int length)
    {
        var glyphs = run.Glyphs.Span;
        var box = run.Box;
        bool upright = run.GlyphMatrix is not { } m || (m.A > 0 && Math.Abs(m.B) < 1e-4f * m.A);
        if (glyphs.Length == run.Text.Length && !run.Logical && upright && run.Baseline is not null)
        {
            float x0 = float.MaxValue, x1 = float.MinValue;
            for (int g = at; g < at + length; g++)
            {
                (x0, x1) = (Math.Min(x0, glyphs[g].X), Math.Max(x1, glyphs[g].X + glyphs[g].Advance));
            }

            return new DocumentBox(x0, box.Y, x1 - x0, box.Height);
        }

        float share = box.Width / Math.Max(1, run.Text.Length);
        return box.Width >= box.Height
            ? new DocumentBox(box.X + share * at, box.Y, share * length, box.Height)
            : new DocumentBox(box.X, box.Y + box.Height / Math.Max(1, run.Text.Length) * at, box.Width, box.Height / Math.Max(1, run.Text.Length) * length);
    }
}

/// <summary>
/// How an annotation looks when its document gives no appearance: drawn from its properties (its kind, box, points, ink,
/// colours, border width and opacity) as paths on its page. The library's drawer draws an annotation without an appearance
/// this way, and its writer makes a created annotation's appearance from it, so the two agree.
/// </summary>
public static class DocumentAnnotationLooks
{
    private const float Kappa = 0.5523f;   // a quarter circle's cubic control distance, as a share of the radius

    /// <summary>
    /// <paramref name="annotation"/>'s look on a page of <paramref name="width"/> x <paramref name="height"/> points: a
    /// rectangle, an ellipse, a line with its ends, a polygon or polyline, ink strokes, the marks over text (a highlight
    /// multiplied, an underline, a strike-out, a wavy line), a note's icon, a caret, a stamp's or file attachment's frame,
    /// a link's border (none when it has no colour or width) or a redaction mark's; null for what is drawn from text alone (free text: its font is the document's),
    /// a popup, a form field's widget and a kind not known.
    /// </summary>
    public static DocumentPage? Look(DocumentAnnotation annotation, float width, float height)
    {
        ArgumentNullException.ThrowIfNull(annotation);
        var paths = new List<DocumentPath>();
        float alpha = Math.Clamp(annotation.Opacity, 0f, 1f);
        var box = annotation.Box;
        float w = annotation.BorderWidth ?? 1f;
        var stroke = annotation.Colour?.WithAlpha(alpha);
        var fill = annotation.InteriorColour?.WithAlpha(alpha);
        var points = annotation.Points.Span;
        switch (annotation.Kind)
        {
            case DocumentAnnotationKinds.Square or DocumentAnnotationKinds.Circle or DocumentAnnotationKinds.Redaction or DocumentAnnotationKinds.Stamp
                or DocumentAnnotationKinds.FileAttachment:
            {
                float inset = Math.Min(w / 2f, Math.Min(box.Width, box.Height) / 2f);
                var inner = new DocumentBox(box.X + inset, box.Y + inset, Math.Max(0f, box.Width - 2 * inset), Math.Max(0f, box.Height - 2 * inset));
                var shape = annotation.Kind == DocumentAnnotationKinds.Circle ? Ellipse(inner) : DocumentPath.Rectangle(inner);
                bool filled = fill is not null && annotation.Kind != DocumentAnnotationKinds.Redaction;
                if (annotation.Kind == DocumentAnnotationKinds.FileAttachment)
                {
                    (filled, fill) = (true, stroke ?? DocumentColour.Grey(0.5f).WithAlpha(alpha));
                }

                Add(paths, shape, filled ? fill : null, stroke, w, DocumentLineCap.Butt, DocumentLineJoin.Miter);
                break;
            }

            case DocumentAnnotationKinds.Line when points.Length >= 4:
                Add(paths, Polyline(points[..4], closed: false), null, stroke, w, DocumentLineCap.Butt, DocumentLineJoin.Miter);
                for (int end = 0; end < Math.Min(2, annotation.LineEndings.Count); end++)
                {
                    var (tipX, tipY, fromX, fromY) = end == 0 ? (points[0], points[1], points[2], points[3]) : (points[2], points[3], points[0], points[1]);
                    Ending(paths, annotation.LineEndings[end], tipX, tipY, fromX, fromY, w, stroke, fill);
                }

                break;
            case DocumentAnnotationKinds.Polygon or DocumentAnnotationKinds.Polyline when points.Length >= 4:
                bool closed = annotation.Kind == DocumentAnnotationKinds.Polygon;
                Add(paths, Polyline(points, closed), closed ? fill : null, stroke, w, DocumentLineCap.Butt, DocumentLineJoin.Miter);
                break;
            case DocumentAnnotationKinds.Ink:
                foreach (var ink in annotation.Ink)
                {
                    if (!ink.IsEmpty)
                    {
                        Add(paths, ink, null, stroke ?? DocumentColour.Black.WithAlpha(alpha), w, DocumentLineCap.Round, DocumentLineJoin.Round);
                    }
                }

                break;
            case DocumentAnnotationKinds.Highlight or DocumentAnnotationKinds.Underline or DocumentAnnotationKinds.StrikeOut or DocumentAnnotationKinds.Squiggly:
                Marks(paths, annotation.Kind, points.Length >= 8 ? points : Corners(box), stroke ?? DocumentColour.Rgb(1f, 1f, 0f).WithAlpha(alpha));
                break;
            case DocumentAnnotationKinds.Note:
            {
                // The icon: a square of the colour with a frame and two lines, 20 points from the box's top-left corner.
                var icon = new DocumentBox(box.X, box.Y, Math.Min(20f, Math.Max(box.Width, 1f)), Math.Min(20f, Math.Max(box.Height, 1f)));
                Add(paths, DocumentPath.Rectangle(icon), stroke ?? DocumentColour.Rgb(1f, 0.85f, 0f).WithAlpha(alpha), DocumentColour.Black.WithAlpha(alpha), 1f,
                    DocumentLineCap.Butt, DocumentLineJoin.Miter);
                float x0 = icon.X + icon.Width * 0.25f, x1 = icon.X + icon.Width * 0.75f;
                foreach (float share in (ReadOnlySpan<float>)[0.35f, 0.55f])
                {
                    float y = icon.Y + icon.Height * share;
                    Add(paths, new DocumentPath([OutlineVerb.Move, OutlineVerb.Line], [x0, y, x1, y]), null, DocumentColour.Black.WithAlpha(alpha), 1f, DocumentLineCap.Butt, DocumentLineJoin.Miter);
                }

                break;
            }

            case DocumentAnnotationKinds.Caret:
                Add(paths, new DocumentPath([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close],
                    [box.X, box.Bottom, box.X + box.Width / 2f, box.Y, box.Right, box.Bottom]), stroke ?? DocumentColour.Black.WithAlpha(alpha), null, w,
                    DocumentLineCap.Butt, DocumentLineJoin.Miter);
                break;
            case DocumentAnnotationKinds.Link:
                if (stroke is not null && w > 0f)
                {
                    Add(paths, DocumentPath.Rectangle(new DocumentBox(box.X + w / 2f, box.Y + w / 2f, Math.Max(0f, box.Width - w), Math.Max(0f, box.Height - w))), null, stroke, w,
                        DocumentLineCap.Butt, DocumentLineJoin.Miter);
                }

                break;
            default:
                return null;
        }

        return new DocumentPage(Math.Max(0, annotation.Page), width, height) { Paths = paths };
    }

    private static void Add(List<DocumentPath> paths, DocumentPath shape, DocumentColour? fill, DocumentColour? stroke, float width, DocumentLineCap cap, DocumentLineJoin join)
    {
        if (fill is null && (stroke is null || width <= 0f))
        {
            return;
        }

        paths.Add(new DocumentPath(shape.Verbs.ToArray(), shape.Points.ToArray())
        {
            Fill = fill is null ? DocumentFill.None : DocumentFill.NonZero, FillColour = fill ?? default, Stroke = stroke is not null && width > 0f ? width : null,
            StrokeColour = stroke ?? default, Cap = cap, Join = join, Order = paths.Count,
        });
    }

    // An ellipse inside a box, four cubic quarters from its right point, clockwise on the page.
    private static DocumentPath Ellipse(DocumentBox box)
    {
        float cx = box.X + box.Width / 2f, cy = box.Y + box.Height / 2f, rx = box.Width / 2f, ry = box.Height / 2f, kx = Kappa * rx, ky = Kappa * ry;
        return new DocumentPath([OutlineVerb.Move, OutlineVerb.Cubic, OutlineVerb.Cubic, OutlineVerb.Cubic, OutlineVerb.Cubic, OutlineVerb.Close],
        [
            cx + rx, cy,
            cx + rx, cy + ky, cx + kx, cy + ry, cx, cy + ry,
            cx - kx, cy + ry, cx - rx, cy + ky, cx - rx, cy,
            cx - rx, cy - ky, cx - kx, cy - ry, cx, cy - ry,
            cx + kx, cy - ry, cx + rx, cy - ky, cx + rx, cy,
        ]);
    }

    private static DocumentPath Polyline(ReadOnlySpan<float> points, bool closed)
    {
        int n = points.Length / 2;
        var verbs = new OutlineVerb[n + (closed ? 1 : 0)];
        verbs[0] = OutlineVerb.Move;
        for (int i = 1; i < n; i++)
        {
            verbs[i] = OutlineVerb.Line;
        }

        if (closed)
        {
            verbs[^1] = OutlineVerb.Close;
        }

        return new DocumentPath(verbs, points[..(2 * n)].ToArray());
    }

    // A box's corners as one quad (upper left, upper right, lower left, lower right).
    private static float[] Corners(DocumentBox box) => [box.X, box.Y, box.Right, box.Y, box.X, box.Bottom, box.Right, box.Bottom];

    // A line's end drawn as its name says: an arrow (open or closed), a circle, a square, a diamond, a bar across it.
    private static void Ending(List<DocumentPath> paths, string name, float tipX, float tipY, float fromX, float fromY, float width, DocumentColour? stroke, DocumentColour? fill)
    {
        float dx = tipX - fromX, dy = tipY - fromY, length = MathF.Sqrt(dx * dx + dy * dy);
        if (length <= 0f || stroke is null)
        {
            return;
        }

        (dx, dy) = (dx / length, dy / length);
        float size = Math.Max(6f, 4.5f * width), half = size / 2f;
        float cos = MathF.Cos(MathF.PI / 6f), sin = MathF.Sin(MathF.PI / 6f);
        (float X, float Y) Back(float turn) => (tipX - size * (dx * cos - dy * sin * turn), tipY - size * (dy * cos + dx * sin * turn));
        switch (name)
        {
            case "OpenArrow" or "ROpenArrow" or "ClosedArrow" or "RClosedArrow":
            {
                var (ax, ay) = Back(1f);
                var (bx, by) = Back(-1f);
                bool closed = name.EndsWith("ClosedArrow", StringComparison.Ordinal);
                Add(paths, new DocumentPath(closed ? [OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close] : [OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line],
                    [ax, ay, tipX, tipY, bx, by]), closed ? fill : null, stroke, width, DocumentLineCap.Butt, DocumentLineJoin.Miter);
                break;
            }

            case "Circle":
                Add(paths, Ellipse(new DocumentBox(tipX - half, tipY - half, size, size)), fill, stroke, width, DocumentLineCap.Butt, DocumentLineJoin.Miter);
                break;
            case "Square":
                Add(paths, DocumentPath.Rectangle(new DocumentBox(tipX - half, tipY - half, size, size)), fill, stroke, width, DocumentLineCap.Butt, DocumentLineJoin.Miter);
                break;
            case "Diamond":
                Add(paths, new DocumentPath([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close],
                    [tipX, tipY - half, tipX + half, tipY, tipX, tipY + half, tipX - half, tipY]), fill, stroke, width, DocumentLineCap.Butt, DocumentLineJoin.Miter);
                break;
            case "Butt" or "Slash":
                Add(paths, new DocumentPath([OutlineVerb.Move, OutlineVerb.Line], [tipX - dy * half, tipY + dx * half, tipX + dy * half, tipY - dx * half]), null, stroke, width,
                    DocumentLineCap.Butt, DocumentLineJoin.Miter);
                break;
        }
    }

    // Marks over text, one a quad (four corners: upper left, upper right, lower left, lower right): a highlight filled
    // and multiplied, an underline near the foot, a strike-out through the middle, a wavy line along the foot.
    private static void Marks(List<DocumentPath> paths, string kind, ReadOnlySpan<float> quads, DocumentColour colour)
    {
        for (int q = 0; q + 8 <= quads.Length; q += 8)
        {
            float ulx = quads[q], uly = quads[q + 1], urx = quads[q + 2], ury = quads[q + 3], llx = quads[q + 4], lly = quads[q + 5], lrx = quads[q + 6], lry = quads[q + 7];
            float tall = MathF.Sqrt((llx - ulx) * (llx - ulx) + (lly - uly) * (lly - uly));
            float line = Math.Max(0.5f, tall / 14f);
            (float X, float Y) At(float down, bool left) =>
                left ? (ulx + (llx - ulx) * down, uly + (lly - uly) * down) : (urx + (lrx - urx) * down, ury + (lry - ury) * down);
            switch (kind)
            {
                case DocumentAnnotationKinds.Highlight:
                    paths.Add(new DocumentPath([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close], [ulx, uly, urx, ury, lrx, lry, llx, lly])
                    {
                        Fill = DocumentFill.NonZero, FillColour = colour, Blend = DocumentBlend.Multiply, Order = paths.Count,
                    });
                    break;
                case DocumentAnnotationKinds.Underline or DocumentAnnotationKinds.StrikeOut:
                {
                    float share = kind == DocumentAnnotationKinds.StrikeOut ? 0.55f : 1f - 1.5f * line / Math.Max(tall, 1e-3f);
                    var (ax, ay) = At(share, left: true);
                    var (bx, by) = At(share, left: false);
                    Add(paths, new DocumentPath([OutlineVerb.Move, OutlineVerb.Line], [ax, ay, bx, by]), null, colour, line, DocumentLineCap.Butt, DocumentLineJoin.Miter);
                    break;
                }

                default:
                {
                    // A wave of steps two lines long, between the foot and a line above it.
                    var (ax, ay) = At(1f - line / Math.Max(tall, 1e-3f), left: true);
                    var (bx, by) = At(1f - line / Math.Max(tall, 1e-3f), left: false);
                    float dx = bx - ax, dy = by - ay, length = MathF.Sqrt(dx * dx + dy * dy);
                    int steps = Math.Clamp((int)(length / Math.Max(2f * line, 0.5f)), 1, 4096);
                    var verbs = new OutlineVerb[steps + 1];
                    var points = new float[2 * (steps + 1)];
                    float nx = length > 0f ? -dy / length : 0f, ny = length > 0f ? dx / length : 0f;
                    for (int k = 0; k <= steps; k++)
                    {
                        float t = (float)k / steps, up = k % 2 == 0 ? 0f : -line;
                        verbs[k] = k == 0 ? OutlineVerb.Move : OutlineVerb.Line;
                        (points[2 * k], points[2 * k + 1]) = (ax + dx * t + nx * up, ay + dy * t + ny * up);
                    }

                    Add(paths, new DocumentPath(verbs, points), null, colour, Math.Max(0.5f, line * 0.8f), DocumentLineCap.Round, DocumentLineJoin.Round);
                    break;
                }
            }
        }
    }
}
