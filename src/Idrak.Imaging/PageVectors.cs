// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Imaging;

/// <summary>
/// A page the built-in layouts drew, kept as what it was drawn from (<see cref="IPageVectors"/>, when
/// <see cref="PageStyle.Vectors"/> asks): its text blocks with the glyph painter that posed them, its shapes with their
/// lettering, its rules, each in its layer (the ink, or the second ink of signatures and stamps), and the colours the page
/// was composed in. Nothing is copied: the blocks and plans the page was drawn from are kept, and written through
/// <see cref="DocumentShapes"/> when the page is written, in the order they were drawn, the ink's before the second ink's
/// (the page lays its second ink over its ink).
/// </summary>
internal sealed class PageRecord : IPageVectors
{
    private readonly List<Part> _parts = [];
    private int _texts;
    private DocumentColour _paper = DocumentColour.White, _ink = DocumentColour.Black, _blue = DocumentColour.Black;

    public int Width { get; private set; }

    public int Height { get; private set; }

    /// <summary>
    /// A block of the page's own text drawn with its top-left at (<paramref name="x"/>, <paramref name="y"/>) by
    /// <paramref name="painter"/> from line number <paramref name="firstLine"/> (its <see cref="GlyphPainter.Line"/> before
    /// it drew the block), in reading order after the texts recorded before it, or at <paramref name="order"/> when given (a
    /// layout whose reading order is not its drawing order: group, index, part).
    /// </summary>
    public void Text(TextBlock block, float x, float y, bool blue, GlyphPainter painter, int firstLine, (int Group, int Index, int Part)? order = null) =>
        _parts.Add(new Part(PartKind.Text, blue) { Block = block, X = x, Y = y, Painter = painter, FirstLine = firstLine, Order = order ?? (0, _texts, 0), Sequence = _texts++ });

    /// <summary>Shape plans drawn with strokes <paramref name="strokeWidth"/> wide and their texts set by <paramref name="lettering"/> (none without it).</summary>
    public void Shapes(IReadOnlyList<(ShapeStep Step, ShapePlan Plan)> drawn, float strokeWidth, ShapeLettering? lettering, bool blue) =>
        _parts.Add(new Part(PartKind.Shapes, blue) { Drawn = drawn, StrokeWidth = strokeWidth, Lettering = lettering });

    /// <summary>A filled rectangle of full ink (a rule).</summary>
    public void Rule(float x, float y, float width, float height) =>
        _parts.Add(new Part(PartKind.Rule, false) { X = x, Y = y, RuleWidth = width, RuleHeight = height });

    /// <summary>
    /// The page's size and colours, as <see cref="PageCanvas.Compose"/> composes it: the paper, the ink at full coverage
    /// (the paper darkened by <paramref name="darkness"/>) and the second ink; each grey on a page that is not in colour.
    /// </summary>
    public PageRecord Finish(int width, int height, (float R, float G, float B) paper, float darkness, (float R, float G, float B) blue, bool color)
    {
        (Width, Height) = (width, height);
        if (color)
        {
            _paper = DocumentColour.Rgb(paper.R, paper.G, paper.B);
            _ink = DocumentColour.Rgb(paper.R * (1 - darkness), paper.G * (1 - darkness), paper.B * (1 - darkness));
            _blue = DocumentColour.Rgb(blue.R, blue.G, blue.B);
        }
        else
        {
            float grey = (paper.R + paper.G + paper.B) / 3;
            _paper = DocumentColour.Grey(grey);
            _ink = DocumentColour.Grey(grey * (1 - darkness));
            _blue = DocumentColour.Grey((blue.R + blue.G + blue.B) / 3);
        }

        return this;
    }

    public void Write(IDocumentPageBuilder page, float scale, float x, float y)
    {
        ArgumentNullException.ThrowIfNull(page);
        var ink = Style(scale, x, y, _ink);
        var blue = Style(scale, x, y, _blue);

        // The paper: the page filled in its colour (white paper is the page's own).
        if (!(_paper.Space == DocumentColourSpace.Grey && _paper[0] >= 1f) && !(_paper.Space == DocumentColourSpace.Rgb && _paper.Red >= 1f && _paper.Green >= 1f && _paper.Blue >= 1f))
        {
            page.Path(Filled(new DocumentBox(0, 0, page.Width, page.Height), _paper));
        }

        foreach (bool second in (ReadOnlySpan<bool>)[false, true])
        {
            foreach (var part in _parts)
            {
                if (part.Blue == second)
                {
                    WritePart(page, part, second ? blue : ink, DocumentTextMode.Fill);
                }
            }
        }
    }

    public void WriteText(IDocumentPageBuilder page, float scale, float x, float y, DocumentTextMode mode)
    {
        ArgumentNullException.ThrowIfNull(page);
        var ink = Style(scale, x, y, _ink);
        var blue = Style(scale, x, y, _blue);
        var texts = _parts.Where(p => p.Kind == PartKind.Text).OrderBy(p => p.Order).ThenBy(p => p.Sequence).ToList();
        foreach (var part in texts)
        {
            WritePart(page, part, part.Blue ? blue : ink, mode);
        }
    }

    private void WritePart(IDocumentPageBuilder page, Part part, DocumentShapeStyle style, DocumentTextMode mode)
    {
        switch (part.Kind)
        {
            case PartKind.Text:
                var block = part.Block!;
                var poses = part.Painter!.IsIdentity ? null : part.Painter.Posed(block, part.FirstLine).Select(p => p.Pose).ToArray();
                DocumentShapes.Write(page, block, block.Text, part.X, part.Y, style, mode, poses);
                break;
            case PartKind.Shapes:
                var stroked = style with { StrokeWidth = part.StrokeWidth };
                foreach (var (_, plan) in part.Drawn!)
                {
                    DocumentShapes.Write(page, plan, stroked);
                    if (part.Lettering is { } lettering)
                    {
                        DocumentShapes.WriteTexts(page, plan, stroked, lettering);
                    }
                }

                break;
            case PartKind.Rule:
                float s = style.Scale;
                var box = new DocumentBox((part.X - style.Left) * s, (part.Y - style.Top) * s, part.RuleWidth * s, part.RuleHeight * s);
                page.Path(Filled(box, style.Tone(1f)));
                break;
        }
    }

    // A box filled in a colour.
    private static DocumentPath Filled(DocumentBox box, DocumentColour colour) =>
        new([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close], [box.X, box.Y, box.Right, box.Y, box.Right, box.Bottom, box.X, box.Bottom])
        {
            Fill = DocumentFill.NonZero, FillColour = colour,
        };

    // The style a layer is written in: the page's scale and corner, its ink over the paper.
    private DocumentShapeStyle Style(float scale, float x, float y, DocumentColour ink)
    {
        if (!(scale > 0f) || !float.IsFinite(scale))
        {
            throw new ArgumentOutOfRangeException(nameof(scale), scale, "A page's scale is a finite number of points a pixel above 0.");
        }

        return new DocumentShapeStyle { Scale = scale, Left = x, Top = y, Ink = ink, Paper = _paper };
    }

    private enum PartKind : byte
    {
        Text,
        Shapes,
        Rule,
    }

    // What was drawn: a text block, shape plans or a rule, in a layer.
    private sealed class Part(PartKind kind, bool blue)
    {
        public PartKind Kind { get; } = kind;

        public bool Blue { get; } = blue;

        public TextBlock? Block { get; init; }

        public float X { get; init; }

        public float Y { get; init; }

        public GlyphPainter? Painter { get; init; }

        public int FirstLine { get; init; }

        public (int Group, int Index, int Part) Order { get; init; }

        public int Sequence { get; init; }

        public IReadOnlyList<(ShapeStep Step, ShapePlan Plan)>? Drawn { get; init; }

        public float StrokeWidth { get; init; }

        public ShapeLettering? Lettering { get; init; }

        public float RuleWidth { get; init; }

        public float RuleHeight { get; init; }
    }
}
