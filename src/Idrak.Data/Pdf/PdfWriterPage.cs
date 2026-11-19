// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Data.Pdf;

/// <summary>
/// One page being written: its content stream built in a pooled buffer, in the page's own frame (points from the top-left
/// corner, y down: the content starts by turning the user space over, so every point is written as given and reads back
/// exactly), the graphics state kept to write only what changes; images written to the file as they are given, the
/// content stream and the page object when the document moves on. Content over an existing page (an editor's) starts by
/// the page's own frame instead (its display map turned back), and names its resources after a prefix that the page's
/// own do not start with.
/// </summary>
internal sealed partial class PdfPageBuilder : IDocumentPageBuilder
{
    // Text runs closer than this to one baseline share one showing operator.
    private const float SameBaseline = 1e-4f;

    // A glyph this close to its place (points) is not moved there: the rounding of advances to /W's decimals and of the
    // caller's float sums, not a kern. Each glyph is measured against its own place, so the error never adds up.
    private const double Placed = 2.5e-4;

    private readonly Func<IFontFace, PdfWriterFont> _font;
    private readonly Func<ImageData, DocumentImageEncoding, int> _image;
    private readonly List<GraphicsState> _saved = [];
    private readonly List<int> _images = [];
    private readonly List<PdfWriterFont> _fonts = [];
    private readonly List<(float Fill, float Stroke)> _alphas = [];
    private readonly List<(PdfValue Source, Func<int>? New)> _forms = [];
    private readonly PdfMatrix? _frame;
    private readonly byte[] _names;
    private PdfSyntax? _content;
    private GraphicsState _state = GraphicsState.Initial;
    private bool _written;

    /// <param name="font">The face's font in the document the page belongs to (a builder's or an editor's), made on its first use.</param>
    /// <param name="image">Takes an image as an image XObject of that document: the number the page's resources name it by.</param>
    /// <param name="index">The page's index.</param>
    /// <param name="width">Its width in points.</param>
    /// <param name="height">Its height in points.</param>
    /// <param name="frame">The map from page points (top-left origin, y down) to the page's user space; null for a new page's (its user space turned over).</param>
    /// <param name="names">What its resources' names start with ("" for a new page: /Im1, /F1, /GS1).</param>
    public PdfPageBuilder(Func<IFontFace, PdfWriterFont> font, Func<ImageData, DocumentImageEncoding, int> image, int index, float width, float height,
        PdfMatrix? frame = null, string names = "") =>
        (_font, _image, Index, Width, Height, _frame, _names) = (font, image, index, width, height, frame, System.Text.Encoding.ASCII.GetBytes(names));

    public int Index { get; }

    public float Width { get; }

    public float Height { get; }

    public void Image(ImageData image, DocumentBox box, DocumentImageEncoding encoding)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (!(box.Width > 0 && box.Height > 0) || !float.IsFinite(box.X) || !float.IsFinite(box.Y) || !float.IsFinite(box.Width) || !float.IsFinite(box.Height))
        {
            throw new ArgumentOutOfRangeException(nameof(box), box, "An image's box has a finite place and a width and height above 0.");
        }

        var content = Content();
        int number = _image(image, encoding);
        int resource = _images.IndexOf(number);
        if (resource < 0)
        {
            _images.Add(number);
            resource = _images.Count - 1;
        }

        // The unit square onto the box, its first row at the top (the page's frame is y down).
        content.Raw("q "u8).Number(box.Width).Raw(" 0 0 "u8).Number(-box.Height).Space().Number(box.X).Space().Number(box.Y + box.Height)
            .Raw(" cm /"u8).Raw(_names).Raw("Im"u8).Integer(resource + 1).Raw(" Do Q\n"u8);
    }

    public void Path(DocumentPath path)
    {
        ArgumentNullException.ThrowIfNull(path);
        Check(path);
        bool fill = path.Fill != DocumentFill.None, stroke = path.Stroke is not null;
        if (path.IsEmpty || (!fill && !stroke))
        {
            ThrowIfWritten();
            return;                                                            // nothing to paint
        }

        var content = Content();
        Alpha(content, fill ? path.FillColour.Alpha : _state.FillAlpha, stroke ? path.StrokeColour.Alpha : _state.StrokeAlpha);
        if (fill)
        {
            FillColour(content, path.FillColour);
        }

        if (stroke)
        {
            StrokeColour(content, path.StrokeColour);
            LineState(content, path);
        }

        Construct(content, path);
        bool evenOdd = path.Fill == DocumentFill.EvenOdd;
        content.Raw(fill && stroke ? evenOdd ? "B*\n"u8 : "B\n"u8 : fill ? evenOdd ? "f*\n"u8 : "f\n"u8 : "S\n"u8);
    }

    public void Glyphs(IFontFace face, float size, ReadOnlySpan<DocumentGlyph> glyphs, string text, DocumentTextMode mode, DocumentColour colour)
    {
        ArgumentNullException.ThrowIfNull(face);
        ArgumentNullException.ThrowIfNull(text);
        if (!(size > 0) || !float.IsFinite(size))
        {
            throw new ArgumentOutOfRangeException(nameof(size), size, "A run's size is a finite number of points above 0.");
        }

        if (mode is not (DocumentTextMode.Fill or DocumentTextMode.Stroke or DocumentTextMode.FillStroke or DocumentTextMode.Hidden))
        {
            throw new ArgumentOutOfRangeException(nameof(mode), mode, "Not a text mode.");
        }

        ThrowIfWritten();
        if (glyphs.IsEmpty)
        {
            return;                                                            // nothing to show
        }

        int limit = Math.Min(face.GlyphCount, 65536);
        foreach (var g in glyphs)
        {
            if ((uint)g.Glyph >= (uint)limit)
            {
                throw new ArgumentOutOfRangeException(nameof(glyphs), g.Glyph, $"Glyph {g.Glyph} is not one of the face's {face.GlyphCount} (a document shows ids up to 65535).");
            }

            if (!float.IsFinite(g.X) || !float.IsFinite(g.Y))
            {
                throw new ArgumentException($"Glyph {g.Glyph} is placed at ({g.X}, {g.Y}): places are finite.", nameof(glyphs));
            }
        }

        var font = _font(face);
        var content = Content();
        if (!_fonts.Contains(font))
        {
            _fonts.Add(font);
        }

        bool plain = font.Map(glyphs, text);
        bool filled = mode is DocumentTextMode.Fill or DocumentTextMode.FillStroke, stroked = mode is DocumentTextMode.Stroke or DocumentTextMode.FillStroke;
        if (mode != DocumentTextMode.Hidden)
        {
            Alpha(content, filled ? colour.Alpha : _state.FillAlpha, stroked ? colour.Alpha : _state.StrokeAlpha);
            if (filled)
            {
                FillColour(content, colour);
            }

            if (stroked)
            {
                StrokeColour(content, colour);
            }
        }

        if (!plain)
        {
            // The text the glyphs stand for, as written (a reader takes it over the glyphs' own).
            content.Raw("/Span<</ActualText "u8).Text(text).Raw(">>BDC\n"u8);
        }

        content.Raw("BT\n"u8);
        if (!ReferenceEquals(_state.Font, font) || _state.Size != size)
        {
            content.Raw("/"u8).Raw(_names).Raw("F"u8).Integer(font.Index).Space().Number(size).Raw(" Tf\n"u8);
            (_state.Font, _state.Size) = (font, size);
        }

        int render = mode switch { DocumentTextMode.Stroke => 1, DocumentTextMode.FillStroke => 2, DocumentTextMode.Hidden => 3, _ => 0 };
        if (_state.Render != render)
        {
            content.Integer(render).Raw(" Tr\n"u8);
            _state.Render = render;
        }

        // Each baseline's glyphs in one TJ: the pen moved to each glyph's place by the offsets a reader adds to the widths.
        double written = PdfSyntax.Written(size);
        double pen = 0;
        bool open = false;
        float baseline = float.NaN;
        foreach (var g in glyphs)
        {
            if (!open || MathF.Abs(g.Y - baseline) > SameBaseline)
            {
                if (open)
                {
                    content.Raw(">]TJ\n"u8);
                }

                baseline = g.Y;
                content.Raw("1 0 0 -1 "u8).Number(g.X).Space().Number(g.Y).Raw(" Tm\n[<"u8);
                pen = PdfSyntax.Written(g.X);
                open = true;
            }
            else
            {
                double target = PdfSyntax.Written(g.X);
                if (Math.Abs(pen - target) > Placed)
                {
                    double adjust = PdfSyntax.Written((pen - target) * 1000 / written, 3);
                    if (adjust != 0)
                    {
                        content.Byte((byte)'>').Number(adjust, 3).Byte((byte)'<');
                        pen += -adjust / 1000 * written;                        // as a reader moves the pen
                    }
                }
            }

            content.HexPair(g.Glyph);
            pen += font.Width(g.Glyph) / 1000 * written;                       // a reader's width: /W over 1000, times the size
        }

        content.Raw(">]TJ\nET\n"u8);
        if (!plain)
        {
            content.Raw("EMC\n"u8);
        }
    }

    public void Save()
    {
        Content().Raw("q\n"u8);
        _saved.Add(_state);
    }

    public void Restore()
    {
        ThrowIfWritten();
        if (_saved.Count == 0)
        {
            throw new InvalidOperationException($"Page {Index + 1}: nothing is saved; Restore follows a Save on the same page.");
        }

        Content().Raw("Q\n"u8);
        _state = _saved[^1];
        _saved.RemoveAt(_saved.Count - 1);
    }

    public void Clip(DocumentPath path)
    {
        ArgumentNullException.ThrowIfNull(path);
        Check(path);
        var content = Content();
        if (path.IsEmpty)
        {
            content.Raw("0 0 0 0 re W n\n"u8);                                 // inside nothing: nothing shows
            return;
        }

        Construct(content, path);
        content.Raw(path.Fill == DocumentFill.EvenOdd ? "W* n\n"u8 : "W n\n"u8);
    }

    /// <summary>The numbers of the images its resources name (<c>/Im1</c>, <c>/Im2</c>, ... after the prefix), in order.</summary>
    internal IReadOnlyList<int> ImageNumbers => _images;

    /// <summary>The fonts its resources name (<c>/F</c> and each one's index, after the prefix).</summary>
    internal IReadOnlyList<PdfWriterFont> Fonts => _fonts;

    /// <summary>The opacities its graphics states set (<c>/GS1</c>, <c>/GS2</c>, ... after the prefix), in order.</summary>
    internal IReadOnlyList<(float Fill, float Stroke)> Alphas => _alphas;

    /// <summary>What its resources' names start with.</summary>
    internal ReadOnlySpan<byte> Names => _names;

    /// <summary>
    /// The form XObjects its resources name (<c>/Fw1</c>, <c>/Fw2</c>, ... after the prefix), in order: a source document's
    /// object, or a new one's number given when the document is written.
    /// </summary>
    internal IReadOnlyList<(PdfValue Source, Func<int>? New)> Forms => _forms;

    /// <summary>The widget annotations the page object lists in its <c>/Annots</c> (a builder's form fields on it), or null.</summary>
    internal List<int>? Annotations { get; set; }

    /// <summary>The map from page points (top-left origin, y down) to the page's user space: its frame, or a new page's user space turned over.</summary>
    internal PdfMatrix ToUserSpace => _frame ?? new PdfMatrix(1, 0, 0, -1, 0, Height);

    /// <summary>
    /// Draws a form XObject (a flattened field's appearance), placed in the page's user space by <paramref name="placement"/>:
    /// <paramref name="source"/>, an object of the document the page is of, or (when null) the new object
    /// <paramref name="number"/> gives the number of when the page is written.
    /// </summary>
    internal void Form(PdfValue source, Func<int>? number, PdfMatrix placement)
    {
        var content = Content();
        int index = source.IsReference ? _forms.FindIndex(f => f.New is null && f.Source.IsReference && f.Source.AsReference == source.AsReference) : -1;
        if (index < 0)
        {
            _forms.Add((source, number));
            index = _forms.Count - 1;
        }

        // The form's space to user space, then (the content being in the page's frame) back to page points.
        var m = placement.Then(PdfEditPages.Inverse(ToUserSpace));
        content.Raw("q "u8).Number(m.A, 6).Space().Number(m.B, 6).Space().Number(m.C, 6).Space().Number(m.D, 6).Space().Number(m.E, 6).Space().Number(m.F, 6)
            .Raw(" cm /"u8).Raw(_names).Raw("Fw"u8).Integer(index + 1).Raw(" Do Q\n"u8);
    }

    /// <summary>
    /// Ends the content (closing what it left open: each Save never restored, then the frame) and gives it, or null when
    /// nothing was drawn; the buffer stays the page's until <see cref="Release"/>. Nothing more can be drawn after.
    /// </summary>
    internal PdfSyntax? Close()
    {
        if (_written)
        {
            return _content;                                                   // closed before (an editor writes a page's object twice when it encrypts)
        }

        _written = true;
        if (_content is not { } content)
        {
            return null;
        }

        for (int i = 0; i < _saved.Count; i++)
        {
            content.Raw("Q\n"u8);                                              // a Save never restored ends with the page
        }

        content.Raw("Q\n"u8);                                                  // the page's frame
        return content;
    }

    /// <summary>Writes the page's content stream and the page object (closing what the content left open); the page object's number.</summary>
    public int Write(PdfFileWriter file, int parent)
    {
        try
        {
            int contents = 0;
            if (Close() is { } content)
            {
                contents = file.Allocate();
                file.BeginStream(contents);
                file.EndStream(content.Span, deflate: true);
            }

            int number = file.Allocate();
            var s = file.BeginSmall(number);
            s.Raw("<</Type/Page/Parent "u8).Reference(parent).Raw("/MediaBox[0 0 "u8).Number(Width).Space().Number(Height).Raw("]/Resources<<"u8);
            if (_images.Count > 0)
            {
                s.Raw("/XObject<<"u8);
                for (int i = 0; i < _images.Count; i++)
                {
                    s.Raw("/"u8).Raw(_names).Raw("Im"u8).Integer(i + 1).Space().Reference(_images[i]);
                }

                s.Raw(">>"u8);
            }

            if (_fonts.Count > 0)
            {
                s.Raw("/Font<<"u8);
                foreach (var font in _fonts)
                {
                    s.Raw("/"u8).Raw(_names).Raw("F"u8).Integer(font.Index).Space().Reference(font.Number);
                }

                s.Raw(">>"u8);
            }

            if (_alphas.Count > 0)
            {
                s.Raw("/ExtGState<<"u8);
                for (int i = 0; i < _alphas.Count; i++)
                {
                    s.Raw("/"u8).Raw(_names).Raw("GS"u8).Integer(i + 1).Raw("<</ca "u8).Number(_alphas[i].Fill).Raw("/CA "u8).Number(_alphas[i].Stroke).Raw(">>"u8);
                }

                s.Raw(">>"u8);
            }

            s.Raw(">>"u8);
            if (contents > 0)
            {
                s.Raw("/Contents "u8).Reference(contents);
            }

            var annotations = AnnotationEntry(s, file, Annotations);          // the form's widgets, then the annotations added (D10)
            s.Raw(">>"u8);
            file.EndSmall();
            if (annotations is not null)
            {
                AnnotationsWritten?.Invoke(number, annotations);               // written at the document's end, every page numbered then
            }

            return number;
        }
        finally
        {
            Release();
        }
    }

    /// <summary>Returns the page's buffers (a page left unwritten when the document is dropped).</summary>
    public void Release()
    {
        _written = true;
        _content?.Dispose();
        _content = null;
    }

    // The content buffer, started with the page's frame: user space turned over so points are page points from the top-left
    // (or, over an existing page, the page's own display map turned back).
    private PdfSyntax Content()
    {
        ThrowIfWritten();
        if (_content is null)
        {
            _content = new PdfSyntax(4096);
            if (_frame is { } f)
            {
                _content.Raw("q "u8).Number(f.A, 6).Space().Number(f.B, 6).Space().Number(f.C, 6).Space().Number(f.D, 6).Space()
                    .Number(f.E, 6).Space().Number(f.F, 6).Raw(" cm\n"u8);
            }
            else
            {
                _content.Raw("q 1 0 0 -1 0 "u8).Number(Height).Raw(" cm\n"u8);
            }
        }

        return _content;
    }

    private void ThrowIfWritten()
    {
        if (_written)
        {
            throw new InvalidOperationException($"Page {Index + 1} was written when the document moved on (a later page or Finish); a page is built before the next one starts.");
        }
    }

    // A path the content can hold: it starts with a move, its stroke is a width of 0 or more, its dashes are lengths not all 0.
    private static void Check(DocumentPath path)
    {
        var verbs = path.Verbs.Span;
        if (verbs.Length > 0 && verbs[0] != OutlineVerb.Move)
        {
            throw new ArgumentException($"A path starts with a move, not {verbs[0]}.", nameof(path));
        }

        for (int i = 1; i < verbs.Length; i++)
        {
            if (verbs[i] != OutlineVerb.Move && verbs[i - 1] == OutlineVerb.Close)
            {
                throw new ArgumentException($"A contour closed by step {i} is followed by {verbs[i]}: a new contour starts with a move.", nameof(path));
            }
        }

        if (path.Stroke is { } width && !(width >= 0 && float.IsFinite(width)))
        {
            throw new ArgumentException($"A stroke's width is a finite number of points from 0, not {width}.", nameof(path));
        }

        if (path.Stroke is not null && (!(path.MiterLimit >= 1) || !float.IsFinite(path.MiterLimit) || !float.IsFinite(path.DashPhase)))
        {
            throw new ArgumentException($"A stroke's miter limit is at least 1 ({path.MiterLimit}) and its dash phase finite ({path.DashPhase}).", nameof(path));
        }

        var dash = path.Dash.Span;
        if (path.Stroke is not null && dash.Length > 0)
        {
            bool any = false;
            foreach (float d in dash)
            {
                if (!(d >= 0) || !float.IsFinite(d))
                {
                    throw new ArgumentException($"A dash's lengths are finite and not negative ({string.Join(", ", dash.ToArray())}).", nameof(path));
                }

                any |= d > 0;
            }

            if (!any)
            {
                throw new ArgumentException("A dash's lengths are not all 0.", nameof(path));
            }
        }
    }

    // The path's contours as construction operators: a rectangle as re, curves as c (v or y when a control point is an end).
    private static void Construct(PdfSyntax content, DocumentPath path)
    {
        var verbs = path.Verbs.Span;
        var p = path.Points.Span;
        if (verbs.SequenceEqual([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close])
            && p[1] == p[3] && p[2] == p[4] && p[5] == p[7] && p[6] == p[0] && p[2] != p[0] && p[5] != p[1])
        {
            content.Number(p[0]).Space().Number(p[1]).Space().Number(p[2] - p[0]).Space().Number(p[5] - p[1]).Raw(" re\n"u8);
            return;
        }

        int at = 0;
        float x = 0, y = 0, startX = 0, startY = 0;
        foreach (var verb in verbs)
        {
            switch (verb)
            {
                case OutlineVerb.Move:
                    (x, y) = (startX, startY) = (p[at], p[at + 1]);
                    content.Number(x).Space().Number(y).Raw(" m\n"u8);
                    at += 2;
                    break;
                case OutlineVerb.Line:
                    (x, y) = (p[at], p[at + 1]);
                    content.Number(x).Space().Number(y).Raw(" l\n"u8);
                    at += 2;
                    break;
                case OutlineVerb.Quadratic:
                    // The same curve as a cubic: its controls two thirds of the way from each end to the quadratic's.
                    double qx = p[at], qy = p[at + 1];
                    float ex = p[at + 2], ey = p[at + 3];
                    content.Number(x + 2 * (qx - x) / 3, 6).Space().Number(y + 2 * (qy - y) / 3, 6).Space()
                        .Number(ex + 2 * (qx - ex) / 3, 6).Space().Number(ey + 2 * (qy - ey) / 3, 6).Space()
                        .Number(ex).Space().Number(ey).Raw(" c\n"u8);
                    (x, y) = (ex, ey);
                    at += 4;
                    break;
                case OutlineVerb.Cubic:
                    float x1 = p[at], y1 = p[at + 1], x2 = p[at + 2], y2 = p[at + 3], x3 = p[at + 4], y3 = p[at + 5];
                    if (x1 == x && y1 == y)
                    {
                        content.Number(x2).Space().Number(y2).Space().Number(x3).Space().Number(y3).Raw(" v\n"u8);
                    }
                    else if (x2 == x3 && y2 == y3)
                    {
                        content.Number(x1).Space().Number(y1).Space().Number(x3).Space().Number(y3).Raw(" y\n"u8);
                    }
                    else
                    {
                        content.Number(x1).Space().Number(y1).Space().Number(x2).Space().Number(y2).Space().Number(x3).Space().Number(y3).Raw(" c\n"u8);
                    }

                    (x, y) = (x3, y3);
                    at += 6;
                    break;
                default:
                    content.Raw("h\n"u8);
                    (x, y) = (startX, startY);
                    break;
            }
        }
    }

    private void FillColour(PdfSyntax content, DocumentColour colour)
    {
        var opaque = colour.WithAlpha(1f);
        if (opaque == _state.Fill)
        {
            return;
        }

        Components(content, opaque).Raw(opaque.Space switch { DocumentColourSpace.Grey => " g\n"u8, DocumentColourSpace.Rgb => " rg\n"u8, _ => " k\n"u8 });
        _state.Fill = opaque;
    }

    private void StrokeColour(PdfSyntax content, DocumentColour colour)
    {
        var opaque = colour.WithAlpha(1f);
        if (opaque == _state.Stroke)
        {
            return;
        }

        Components(content, opaque).Raw(opaque.Space switch { DocumentColourSpace.Grey => " G\n"u8, DocumentColourSpace.Rgb => " RG\n"u8, _ => " K\n"u8 });
        _state.Stroke = opaque;
    }

    private static PdfSyntax Components(PdfSyntax content, DocumentColour colour)
    {
        for (int c = 0; c < colour.Count; c++)
        {
            if (c > 0)
            {
                content.Space();
            }

            content.Number(colour[c]);
        }

        return content;
    }

    // The fill and stroke opacity, through an ExtGState of the page's when they change.
    private void Alpha(PdfSyntax content, float fill, float stroke)
    {
        if (fill == _state.FillAlpha && stroke == _state.StrokeAlpha)
        {
            return;
        }

        int index = _alphas.IndexOf((fill, stroke));
        if (index < 0)
        {
            _alphas.Add((fill, stroke));
            index = _alphas.Count - 1;
        }

        content.Raw("/"u8).Raw(_names).Raw("GS"u8).Integer(index + 1).Raw(" gs\n"u8);
        (_state.FillAlpha, _state.StrokeAlpha) = (fill, stroke);
    }

    // The stroke's width, ends, corners, miter limit and dashes, where they change.
    private void LineState(PdfSyntax content, DocumentPath path)
    {
        float width = path.Stroke!.Value;
        if (width != _state.LineWidth)
        {
            content.Number(width).Raw(" w\n"u8);
            _state.LineWidth = width;
        }

        if (path.Cap != _state.Cap)
        {
            content.Integer((int)path.Cap).Raw(" J\n"u8);
            _state.Cap = path.Cap;
        }

        if (path.Join != _state.Join)
        {
            content.Integer((int)path.Join).Raw(" j\n"u8);
            _state.Join = path.Join;
        }

        if (path.MiterLimit != _state.MiterLimit)
        {
            content.Number(path.MiterLimit).Raw(" M\n"u8);
            _state.MiterLimit = path.MiterLimit;
        }

        var dash = path.Dash.Span;
        if (!dash.SequenceEqual(_state.Dash.Span) || path.DashPhase != _state.DashPhase)
        {
            content.Byte((byte)'[');
            for (int i = 0; i < dash.Length; i++)
            {
                if (i > 0)
                {
                    content.Space();
                }

                content.Number(dash[i]);
            }

            content.Raw("] "u8).Number(path.DashPhase).Raw(" d\n"u8);
            (_state.Dash, _state.DashPhase) = (dash.ToArray(), path.DashPhase);
        }
    }

    // What the content has set, so only changes are written (PDF's defaults to start; saved and restored with q and Q).
    private struct GraphicsState
    {
        public DocumentColour Fill;
        public DocumentColour Stroke;
        public float FillAlpha;
        public float StrokeAlpha;
        public float LineWidth;
        public DocumentLineCap Cap;
        public DocumentLineJoin Join;
        public float MiterLimit;
        public ReadOnlyMemory<float> Dash;
        public float DashPhase;
        public PdfWriterFont? Font;
        public float Size;
        public int Render;

        public static GraphicsState Initial => new()
        {
            Fill = DocumentColour.Black, Stroke = DocumentColour.Black, FillAlpha = 1f, StrokeAlpha = 1f, LineWidth = 1f, MiterLimit = 10f,
        };
    }
}
