// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Data.Pdf;

/// <summary>
/// A content stream with redaction areas applied (<see cref="PdfRedactor"/>): its new bytes, the images and forms it
/// draws in place of those it changed (new names in its resources), the names it no longer draws, and what was removed.
/// </summary>
internal sealed class PdfRedactedContent
{
    /// <summary>The content's bytes.</summary>
    public byte[] Content { get; set; } = [];

    /// <summary>Images whose pixels under an area were overwritten, under their new names.</summary>
    public List<(string Name, ImageData Image)> Images { get; } = [];

    /// <summary>Forms whose content was redacted, under their new names: the form as it was (its dictionary kept) and its content redacted.</summary>
    public List<(string Name, PdfStream Original, PdfDictionary? Resources, PdfRedactedContent Content)> Forms { get; } = [];

    /// <summary>XObject names the content no longer draws (each drawn only where it was replaced): left out of its resources.</summary>
    public HashSet<string> Dropped { get; } = new(StringComparer.Ordinal);

    /// <summary>Property lists named by spans whose text was cut: left out of its resources (their /ActualText could hold removed text).</summary>
    public HashSet<string> DroppedProperties { get; } = new(StringComparer.Ordinal);

    public int Glyphs { get; set; }

    public int Characters { get; set; }

    public int PathsRemoved { get; set; }

    public int PathsClipped { get; set; }

    public int ImagesChanged { get; set; }

    public long Pixels { get; set; }

    /// <summary>Whether anything was changed in it.</summary>
    public bool Changed => Glyphs > 0 || PathsRemoved > 0 || PathsClipped > 0 || Pixels > 0 || Forms.Count > 0;

    /// <summary>The counts with an inner content's added.</summary>
    public void Add(PdfRedactedContent inner)
    {
        Glyphs += inner.Glyphs;
        Characters += inner.Characters;
        PathsRemoved += inner.PathsRemoved;
        PathsClipped += inner.PathsClipped;
        ImagesChanged += inner.ImagesChanged;
        Pixels += inner.Pixels;
    }
}

/// <summary>A page with redaction areas applied: its content redacted, the annotations removed (by their place in its /Annots), the areas and what was removed.</summary>
internal sealed class PdfRedactedPage(PdfRedactedContent content, HashSet<int> annotations, IReadOnlyList<DocumentRedaction> areas, DocumentRedactionResult result)
{
    public PdfRedactedContent Content { get; } = content;

    public HashSet<int> Annotations { get; } = annotations;

    public IReadOnlyList<DocumentRedaction> Areas { get; } = areas;

    public DocumentRedactionResult Result { get; } = result;
}

/// <summary>
/// Redaction applied for real (contract C23's <see cref="IDocumentEditor.Redact"/>): a page's content interpreted as it is
/// read and written again operator by operator. A text-showing operator keeps the glyphs whose boxes do not overlap an
/// area (at least a hundredth of the glyph each way) and moves past the removed ones by their advances (<c>TJ</c>
/// offsets), so what stays stands where it stood; a span's <c>/ActualText</c> over removed glyphs is dropped. A path inside
/// an area is not painted (a clip it makes is kept); one crossing an area is painted under a clip that leaves out each area
/// it crosses (even-odd: the page around the area); a shading too. An image (an XObject or inline) under an area is drawn
/// from a copy whose pixels with their centre inside an area take the area's fill (black when it has none); a form under
/// an area is drawn from a copy of its content redacted the same way. The annotations over an area are removed (a popup
/// with its annotation), and each area with a fill is filled after the page's content. Hidden text (a scan's text layer)
/// is text like any other.
/// </summary>
internal static class PdfRedactor
{
    /// <summary>Page <paramref name="index"/> of <paramref name="document"/> with <paramref name="areas"/> (page points as shown) applied.</summary>
    public static PdfRedactedPage Apply(PdfDocument document, int index, IReadOnlyList<DocumentRedaction> areas)
    {
        var file = document.File;
        var (node, _, _, _, inherited) = document.PageObject(index);
        var page = file.Dictionary(node) ?? PdfDictionary.Empty;
        var resources = file.Dictionary(page.TryGet("Resources"u8, out var own) ? own : inherited);
        var (display, width, height, _) = document.Frame(index);
        var boxes = areas.Select(a => a.Box).ToArray();
        var issues = new PdfIssues();
        var content = new PdfRedactedContent();
        using (var reader = new PdfContentReader(file, index + 1, issues, includeHidden: true))
        {
            reader.ReadImages();
            reader.ReadVector();
            using var rewriter = new PdfRewriter(boxes, Fills(areas), width, height, resources, content, 0);
            reader.Rewriting(rewriter);
            reader.Read(page["Contents"u8], resources, display);
            content.Content = rewriter.Finish(Marks(areas, display));
        }

        // The annotations over an area (a popup leaves with its annotation, not on its own).
        var removed = new HashSet<int>();
        var annots = file.Resolve(page["Annots"u8]).AsArray;
        var popups = new List<(int Index, int Parent)>();
        for (int k = 0; k < (annots?.Count ?? 0); k++)
        {
            var a = file.Dictionary(annots![k]);
            if (a is null)
            {
                continue;
            }

            if (file.Resolve(a["Subtype"u8]).IsName("Popup"u8))
            {
                popups.Add((k, a["Parent"u8].IsReference ? a["Parent"u8].AsReference.Number : -1));
                continue;
            }

            if (PdfAnnotations.Numbers(file, a["Rect"u8]) is { Length: >= 4 } r)
            {
                var box = PdfAnnotations.Bounds(display, r[0], r[1], r[2], r[3]);
                if (boxes.Any(area => Overlap(area, box.X, box.Y, box.Right, box.Bottom)))
                {
                    removed.Add(k);
                }
            }
        }

        foreach (var (k, parent) in popups)
        {
            if (parent >= 0 && removed.Any(r => annots![r].IsReference && annots[r].AsReference.Number == parent))
            {
                removed.Add(k);
            }
        }

        var result = new DocumentRedactionResult(index)
        {
            Areas = areas.Count, Glyphs = content.Glyphs, Characters = content.Characters, PathsRemoved = content.PathsRemoved, PathsClipped = content.PathsClipped,
            Images = content.ImagesChanged, Pixels = content.Pixels, Annotations = removed.Count,
        };
        return new PdfRedactedPage(content, removed, areas, result);
    }

    /// <summary>Whether a box (two corners, page points) and an area share some of the plane.</summary>
    public static bool Overlap(DocumentBox area, double x0, double y0, double x1, double y1) =>
        Math.Min(x1, area.Right) > Math.Max(x0, area.X) && Math.Min(y1, area.Bottom) > Math.Max(y0, area.Y);

    /// <summary>
    /// Writes a redacted content's images and forms into the result (not the content itself); the names it adds and the
    /// object numbers they are written under (<paramref name="images"/> writes an image, <paramref name="link"/> makes a
    /// value of the result's reference that the copier writes as it is).
    /// </summary>
    public static List<(string Name, int Number)> Write(PdfRedactedContent content, PdfFileWriter file, PdfEditCopier copier, PdfEditCopier.Source source,
        Func<ImageData, int> images, Func<int, PdfValue> link)
    {
        var added = new List<(string, int)>();
        foreach (var (name, image) in content.Images)
        {
            added.Add((name, images(image)));
        }

        foreach (var (name, original, resources, inner) in content.Forms)
        {
            var innerAdded = Write(inner, file, copier, source, images, link);
            int number = file.Allocate();
            var dictionary = original.Dictionary;
            var redactedResources = Resources(source.File, resources, inner, innerAdded, link);
            using var scratch = new PdfSyntax(1024);
            void Body(PdfSyntax s)
            {
                for (int i = 0; i < dictionary.Count; i++)
                {
                    var key = dictionary.KeyAt(i);
                    if (!(key.Is("Length"u8) || key.Is("Filter"u8) || key.Is("DecodeParms"u8) || key.Is("Resources"u8) || key.Is("Length1"u8)))
                    {
                        copier.Entry(s, source, key.Bytes, dictionary.ValueAt(i));
                    }
                }

                if (redactedResources is not null)
                {
                    copier.Entry(s, source, "Resources"u8, PdfValue.Of(redactedResources));
                }
            }

            Body(scratch);
            var target = file.BeginStream(number);
            if (target.Encrypts)
            {
                Body(target);
            }
            else
            {
                target.Raw(scratch.Span);
            }

            file.EndStream(inner.Content, deflate: true);
            added.Add((name, number));
        }

        return added;
    }

    /// <summary>
    /// <paramref name="resources"/> with a redacted content's changes: the names it dropped left out of <c>/XObject</c> and
    /// <c>/Properties</c>, the names it added (<paramref name="added"/>) put in; null when there were none.
    /// </summary>
    public static PdfDictionary? Resources(PdfFile file, PdfDictionary? resources, PdfRedactedContent content, List<(string Name, int Number)> added, Func<int, PdfValue> link)
    {
        resources ??= PdfDictionary.Empty;
        var keys = new List<PdfName>();
        var values = new List<PdfValue>();
        bool xobjects = false;
        for (int i = 0; i < resources.Count; i++)
        {
            var key = resources.KeyAt(i);
            var value = resources.ValueAt(i);
            if (key.Is("XObject"u8))
            {
                value = PdfValue.Of(Category(file.Dictionary(value), content.Dropped, added, link));
                xobjects = true;
            }
            else if (key.Is("Properties"u8) && content.DroppedProperties.Count > 0)
            {
                value = PdfValue.Of(Category(file.Dictionary(value), content.DroppedProperties, [], link));
            }

            keys.Add(key);
            values.Add(value);
        }

        if (!xobjects && added.Count > 0)
        {
            keys.Add(PdfName.From("XObject"u8));
            values.Add(PdfValue.Of(Category(null, content.Dropped, added, link)));
        }

        return new PdfDictionary([.. keys], [.. values]);
    }

    // One category of resources (as written: a dictionary, or a reference resolved by the caller) less the dropped names, plus the added ones.
    private static PdfDictionary Category(PdfDictionary? category, HashSet<string> dropped, List<(string Name, int Number)> added, Func<int, PdfValue> link)
    {
        var keys = new List<PdfName>();
        var values = new List<PdfValue>();
        for (int i = 0; i < (category?.Count ?? 0); i++)
        {
            if (!dropped.Contains(category!.KeyAt(i).ToString()))
            {
                keys.Add(category.KeyAt(i));
                values.Add(category.ValueAt(i));
            }
        }

        foreach (var (name, number) in added)
        {
            keys.Add(PdfName.From(System.Text.Encoding.ASCII.GetBytes(name)));
            values.Add(link(number));
        }

        return new PdfDictionary([.. keys], [.. values]);
    }

    // The colour each area's pixels take: its fill, else black.
    private static DocumentColour[] Fills(IReadOnlyList<DocumentRedaction> areas) => [.. areas.Select(a => a.Fill ?? DocumentColour.Black)];

    // Each area with a fill, filled in the page's user space after its content.
    private static byte[] Marks(IReadOnlyList<DocumentRedaction> areas, PdfMatrix display)
    {
        var inverse = PdfEditPages.Inverse(display);
        using var s = new PdfSyntax(256);
        foreach (var area in areas)
        {
            if (area.Fill is not { } fill)
            {
                continue;
            }

            var opaque = fill.WithAlpha(1f);
            s.Raw("q "u8);
            for (int c = 0; c < opaque.Count; c++)
            {
                s.Number(opaque[c]).Space();
            }

            s.Raw(opaque.Space switch { DocumentColourSpace.Grey => "g\n"u8, DocumentColourSpace.Rgb => "rg\n"u8, _ => "k\n"u8 });
            PdfRewriter.Polygon(s, inverse, area.Box);
            s.Raw("f Q\n"u8);
        }

        return s.Span.ToArray();
    }
}

/// <summary>
/// What redacting one content keeps as it goes: the areas, the output (a span with <c>/ActualText</c> held apart until its
/// end, so the text it carries is dropped when a glyph under it is removed), the path being built (written when it is
/// painted, as the areas decide) and the names it changes.
/// </summary>
internal sealed class PdfRewriter(DocumentBox[] areas, DocumentColour[] fills, float width, float height, PdfDictionary? resources, PdfRedactedContent content, int depth)
    : IDisposable
{
    private readonly PdfSyntax _out = new(4096);
    private readonly PdfSyntax _path = new(256);
    private readonly HashSet<string> _kept = new(StringComparer.Ordinal), _replaced = new(StringComparer.Ordinal);
    private PdfSyntax? _span;
    private byte[]? _spanStart;
    private string? _spanProperties;
    private bool _spanCut;
    private int _names;

    public DocumentBox[] Areas => areas;

    public DocumentColour[] AreaFills => fills;

    public float Width => width;

    public float Height => height;

    public PdfDictionary? Resources => resources;

    public PdfRedactedContent Result => content;

    public int Depth => depth;

    private PdfSyntax Out => _span ?? _out;

    /// <summary>An operator written as it was.</summary>
    public void Copy(ReadOnlySpan<byte> bytes) => Out.Raw(bytes).Line();

    /// <summary>Text written in place of an operator.</summary>
    public PdfSyntax Write() => Out;

    /// <summary>A path construction or clip operator, held until the path is painted.</summary>
    public void Hold(ReadOnlySpan<byte> bytes) => _path.Raw(bytes).Line();

    /// <summary>The held path, as it was, then <paramref name="paint"/>.</summary>
    public void Release(ReadOnlySpan<byte> paint)
    {
        Out.Raw(_path.Span).Raw(paint).Line();
        _path.Clear();
    }

    /// <summary>The held path dropped (painted nothing; its clip, when it makes one, kept by <c>n</c>).</summary>
    public void Drop(bool keepClip)
    {
        if (keepClip)
        {
            Out.Raw(_path.Span).Raw("n"u8).Line();
        }

        _path.Clear();
    }

    /// <summary>
    /// The held path painted by <paramref name="paint"/> outside the areas <paramref name="crossed"/> lists, under
    /// <paramref name="ctm"/> (the clips written in its user space), its clip (when it makes one) kept after.
    /// </summary>
    public void Clipped(ReadOnlySpan<byte> paint, PdfMatrix ctm, IEnumerable<int> crossed, bool keepClip)
    {
        var s = Out;
        s.Raw("q\n"u8);
        Exclude(s, ctm, crossed);
        s.Raw(_path.Span).Raw(paint).Line().Raw("Q\n"u8);
        if (keepClip)
        {
            s.Raw(_path.Span).Raw("n"u8).Line();
        }

        _path.Clear();
    }

    /// <summary>An operator (a shading) drawn outside every area.</summary>
    public void Outside(ReadOnlySpan<byte> bytes, PdfMatrix ctm)
    {
        var s = Out;
        s.Raw("q\n"u8);
        Exclude(s, ctm, Enumerable.Range(0, areas.Length));
        s.Raw(bytes).Line().Raw("Q\n"u8);
    }

    // A clip for each area: the page far around it less the area (even-odd), in the user space of `ctm`.
    private void Exclude(PdfSyntax s, PdfMatrix ctm, IEnumerable<int> crossed)
    {
        var inverse = PdfEditPages.Inverse(ctm);
        float margin = 10 * Math.Max(Math.Max(width, height), 1000f);
        var outside = new DocumentBox(-margin, -margin, width + 2 * margin, height + 2 * margin);
        foreach (int a in crossed)
        {
            Polygon(s, inverse, outside);
            Polygon(s, inverse, areas[a]);
            s.Raw("W* n\n"u8);
        }
    }

    /// <summary>A box of the page as shown, as a closed contour of user space (<paramref name="inverse"/>: page points to user space).</summary>
    public static void Polygon(PdfSyntax s, PdfMatrix inverse, DocumentBox box)
    {
        Span<double> corners = [box.X, box.Y, box.Right, box.Y, box.Right, box.Bottom, box.X, box.Bottom];
        for (int k = 0; k < 8; k += 2)
        {
            double x = inverse.A * corners[k] + inverse.C * corners[k + 1] + inverse.E, y = inverse.B * corners[k] + inverse.D * corners[k + 1] + inverse.F;
            s.Number(x, 4).Space().Number(y, 4).Raw(k == 0 ? " m\n"u8 : " l\n"u8);
        }

        s.Raw("h\n"u8);
    }

    /// <summary>A name not yet in the resources' category (or given here): RdIm1, RdFm2, ...</summary>
    public string Name(string stem)
    {
        var category = resources?["XObject"u8].AsDictionary;
        while (true)
        {
            string name = stem + (++_names).ToString(CultureInfo.InvariantCulture);
            if (category is null || !category.Contains(System.Text.Encoding.ASCII.GetBytes(name)))
            {
                return name;
            }
        }
    }

    /// <summary>An XObject drawn as it was (its name stays in the resources).</summary>
    public void Kept(string name) => _kept.Add(name);

    /// <summary>An XObject drawn from a redacted copy here.</summary>
    public void Replaced(string name) => _replaced.Add(name);

    /// <summary>A span with /ActualText starts: what follows is held apart until it ends.</summary>
    public void SpanStart(ReadOnlySpan<byte> bdc, string? properties)
    {
        _span ??= new PdfSyntax(1024);
        _span.Clear();
        (_spanStart, _spanProperties, _spanCut) = (bdc.ToArray(), properties, false);
    }

    /// <summary>A glyph removed: a span open now loses its /ActualText.</summary>
    public void Cut() => _spanCut |= _span is not null && _spanStart is not null;

    /// <summary>The span ends (<paramref name="emc"/>): written with its /ActualText, or, when a glyph in it was removed, as a plain span.</summary>
    public void SpanEnd(ReadOnlySpan<byte> emc)
    {
        if (_spanStart is null || _span is null)
        {
            Copy(emc);
            return;
        }

        _out.Raw(_spanCut ? "/Span BMC"u8 : _spanStart).Line().Raw(_span.Span).Raw(emc).Line();
        if (_spanCut && _spanProperties is { } named)
        {
            content.DroppedProperties.Add(named);
        }

        (_spanStart, _spanProperties, _spanCut) = (null, null, false);
        _span.Clear();
        _span = null;
    }

    /// <summary>The content's bytes: in a saved graphics state, then <paramref name="after"/> (the areas' fills).</summary>
    public byte[] Finish(ReadOnlySpan<byte> after)
    {
        if (_spanStart is not null)
        {
            SpanEnd("EMC"u8);                                                  // a span the content never closes ends with it
        }

        if (_path.Length > 0)
        {
            Drop(keepClip: false);                                             // a path never painted: nothing
        }

        foreach (string name in _replaced)
        {
            if (!_kept.Contains(name))
            {
                content.Dropped.Add(name);
            }
        }

        var bytes = new byte[_out.Length + after.Length + 4];
        "q\n"u8.CopyTo(bytes);
        _out.Span.CopyTo(bytes.AsSpan(2));
        "Q\n"u8.CopyTo(bytes.AsSpan(2 + _out.Length));
        after.CopyTo(bytes.AsSpan(4 + _out.Length));
        return bytes;
    }

    public void Dispose()
    {
        _out.Dispose();
        _path.Dispose();
        _span?.Dispose();
    }
}

/// <summary>The content interpreter's redacting mode: each operator at the content's top level run as it is read, then written again by <see cref="PdfRewriter"/>.</summary>
internal sealed partial class PdfContentReader
{
    private PdfRewriter? _rewrite;

    /// <summary>Redacts the content as it is read (the areas and the output in <paramref name="rewriter"/>).</summary>
    public void Rewriting(PdfRewriter rewriter) => _rewrite = rewriter;

    // One operator of the content's top level while redacting.
    private void Rewrite(ref PdfContentLexer lexer, ReadOnlySpan<byte> word, ReadOnlySpan<byte> data, PdfDictionary? resources, int start)
    {
        var w = _rewrite!;
        int from = _operandCount > 0 ? Math.Min(_operands[0].Start - (_operands[0].Kind is PdfToken.Name or PdfToken.LiteralString or PdfToken.HexString ? 1 : 0), start) : start;
        int end = start + word.Length;
        var bytes = data[from..end];
        if (word.SequenceEqual("BI"u8))
        {
            int before = Images.Count;
            InlineImage(ref lexer, data, resources);
            int after = Math.Max(lexer.Position, start);
            if (Images.Count > before && RedactImage(w, Images[^1], null))
            {
                return;
            }

            w.Copy(data[start..Math.Min(after, data.Length)]);
            return;
        }

        int op = Key(word);
        switch (op)
        {
            case OpMoveTo or OpLine or OpCurve or OpCurveV or OpCurveY or OpClosePath or OpRectangle or OpClip or OpClipEvenOdd:
                Execute(op, data, resources, 0);
                w.Hold(bytes);
                return;
            case OpStroke or OpCloseStroke or OpFill or OpFillOld or OpFillEvenOdd or OpFillStroke or OpFillStrokeEvenOdd or OpCloseFillStroke or OpCloseFillStrokeEvenOdd
                or OpEndPath:
                PaintPath(w, op, data, resources, bytes);
                return;
            case OpShade:
                Execute(op, data, resources, 0);
                w.Outside(bytes, _state.Ctm);
                w.Result.PathsClipped++;
                return;
            case OpShow or OpShowArray or OpQuote or OpDoubleQuote:
            {
                int first = _glyphCount;
                Execute(op, data, resources, 0);
                ShowRedacted(w, op, data, bytes, first);
                return;
            }

            case OpDraw:
            {
                int before = Images.Count;
                var ctm = _state.Ctm;
                Execute(op, data, resources, 0);
                string? name = _operandCount >= 1 && _operands[_operandCount - 1].Kind == PdfToken.Name ? NameText(data, _operands[_operandCount - 1]) : null;
                if (Images.Count > before)
                {
                    if (!RedactImage(w, Images[^1], name))
                    {
                        w.Copy(bytes);
                        if (name is not null)
                        {
                            w.Kept(name);
                        }
                    }

                    return;
                }

                if (name is not null && RedactForm(w, name, resources, ctm))
                {
                    return;
                }

                w.Copy(bytes);
                if (name is not null)
                {
                    w.Kept(name);
                }

                return;
            }

            case OpMarkedProperties:
            {
                bool open = _span is not null;
                Execute(op, data, resources, 0);
                if (!open && _span is not null)
                {
                    var properties = _operands[_operandCount - 1];
                    w.SpanStart(bytes, properties.Kind == PdfToken.Name ? NameText(data, properties) : null);
                }
                else
                {
                    w.Copy(bytes);
                }

                return;
            }

            case OpEndMarked:
            {
                bool open = _span is not null;
                Execute(op, data, resources, 0);
                if (open && _span is null)
                {
                    w.SpanEnd(bytes);
                }
                else
                {
                    w.Copy(bytes);
                }

                return;
            }

            default:
                Execute(op, data, resources, 0);
                w.Copy(bytes);
                return;
        }
    }

    // A painting operator: the path held so far painted as the areas decide (as it was, not at all, or clipped outside the areas it crosses).
    private void PaintPath(PdfRewriter w, int op, ReadOnlySpan<byte> data, PdfDictionary? resources, ReadOnlySpan<byte> bytes)
    {
        bool clip = _clipRule != 0;
        bool paints = op != OpEndPath;
        double minX = double.MaxValue, minY = double.MaxValue, maxX = double.MinValue, maxY = double.MinValue;
        if (!_path.IsEmpty)
        {
            var (_, points) = _path.ToArrays();
            for (int i = 0; i + 1 < points.Length; i += 2)
            {
                (minX, maxX, minY, maxY) = (Math.Min(minX, points[i]), Math.Max(maxX, points[i]), Math.Min(minY, points[i + 1]), Math.Max(maxY, points[i + 1]));
            }

            if (op is OpStroke or OpCloseStroke or OpFillStroke or OpFillStrokeEvenOdd or OpCloseFillStroke or OpCloseFillStrokeEvenOdd)
            {
                double half = _state.LineWidth * Math.Sqrt(Math.Abs(_state.Ctm.A * _state.Ctm.D - _state.Ctm.B * _state.Ctm.C)) / 2;
                (minX, minY, maxX, maxY) = (minX - half, minY - half, maxX + half, maxY + half);
            }
        }

        var ctm = _state.Ctm;
        Execute(op, data, resources, 0);
        if (!paints || minX > maxX)
        {
            w.Release(bytes);
            return;
        }

        const double Slack = 1e-3;
        var crossed = new List<int>();
        bool inside = false;
        for (int a = 0; a < w.Areas.Length; a++)
        {
            var area = w.Areas[a];
            if (minX >= area.X - Slack && maxX <= area.Right + Slack && minY >= area.Y - Slack && maxY <= area.Bottom + Slack)
            {
                inside = true;
                break;
            }

            if (PdfRedactor.Overlap(area, minX, minY, maxX, maxY))
            {
                crossed.Add(a);
            }
        }

        if (inside)
        {
            w.Drop(clip);
            w.Result.PathsRemoved++;
        }
        else if (crossed.Count > 0)
        {
            w.Clipped(bytes, ctm, crossed, clip);
            w.Result.PathsClipped++;
        }
        else
        {
            w.Release(bytes);
        }
    }

    // A text-showing operator: written as it was when none of its glyphs lies under an area, else as a TJ of the glyphs
    // kept with the removed ones' advances as offsets (', " written as the moves and spacing they set, then the TJ).
    private void ShowRedacted(PdfRewriter w, int op, ReadOnlySpan<byte> data, ReadOnlySpan<byte> bytes, int first)
    {
        int count = _glyphCount - first;
        Span<double> xy = stackalloc double[8];
        bool[]? removed = null;
        for (int g = 0; g < count; g++)
        {
            Corners(_glyphs[first + g], xy);
            double x0 = Math.Min(Math.Min(xy[0], xy[2]), Math.Min(xy[4], xy[6])), x1 = Math.Max(Math.Max(xy[0], xy[2]), Math.Max(xy[4], xy[6]));
            double y0 = Math.Min(Math.Min(xy[1], xy[3]), Math.Min(xy[5], xy[7])), y1 = Math.Max(Math.Max(xy[1], xy[3]), Math.Max(xy[5], xy[7]));
            foreach (var area in w.Areas)
            {
                if (Covers(area, x0, x1, area.X, area.Right) && Covers(area, y0, y1, area.Y, area.Bottom))
                {
                    (removed ??= new bool[count])[g] = true;
                    break;
                }
            }
        }

        if (removed is null)
        {
            w.Copy(bytes);
            return;
        }

        var s = w.Write();
        if (op == OpQuote)
        {
            s.Raw("T*\n"u8);
        }
        else if (op == OpDoubleQuote)
        {
            s.Number(_state.WordSpacing, 6).Raw(" Tw "u8).Number(_state.CharSpacing, 6).Raw(" Tc T*\n"u8);
        }

        // The strings shown, in order: the TJ's items, or the one string.
        int itemsFrom = _operandCount - 1, itemsTo = _operandCount - 1;
        if (op == OpShowArray)
        {
            itemsTo = _operandCount - 1;
            while (itemsTo >= 0 && _operands[itemsTo].Kind != PdfToken.ArrayEnd)
            {
                itemsTo--;
            }

            itemsFrom = itemsTo - 1;
            while (itemsFrom >= 0 && _operands[itemsFrom].Kind != PdfToken.ArrayStart)
            {
                itemsFrom--;
            }

            (itemsFrom, itemsTo) = (itemsFrom + 1, itemsTo - 1);
        }

        var font = _runStyle.Font;
        double size = _state.Size;
        s.Byte((byte)'[');
        int glyph = 0;
        bool hexOpen = false;
        for (int i = itemsFrom; i <= itemsTo && i >= 0; i++)
        {
            var item = _operands[i];
            if (item.Kind == PdfToken.Number)
            {
                if (hexOpen)
                {
                    s.Byte((byte)'>');
                    hexOpen = false;
                }

                s.Space().Raw(data.Slice(item.Start, item.Length)).Space();
                continue;
            }

            if (!IsString(item))
            {
                continue;
            }

            var codes = StringBytes(data, item).ToArray();
            for (int at = 0; at < codes.Length && glyph < count;)
            {
                int length = Math.Max(1, font.NextCode(codes, at, out uint code));
                if (removed[glyph])
                {
                    if (hexOpen)
                    {
                        s.Byte((byte)'>');
                        hexOpen = false;
                    }

                    var g = _glyphs[first + glyph];
                    double spacing = _state.CharSpacing + (font.IsWordSpace(code, length) ? _state.WordSpacing : 0);
                    double advance = (font.Vertical ? g.Height : g.Width) + (size != 0 ? spacing / size : 0);
                    s.Space().Number(-advance * 1000, 4).Space();
                    w.Result.Glyphs++;
                    w.Result.Characters += g.Unmapped ? 1 : g.TextLength;
                    w.Cut();
                }
                else
                {
                    if (!hexOpen)
                    {
                        s.Byte((byte)'<');
                        hexOpen = true;
                    }

                    foreach (byte b in codes.AsSpan(at, Math.Min(length, codes.Length - at)))
                    {
                        s.Byte("0123456789ABCDEF"u8[b >> 4]).Byte("0123456789ABCDEF"u8[b & 15]);
                    }
                }

                at += length;
                glyph++;
            }
        }

        if (hexOpen)
        {
            s.Byte((byte)'>');
        }

        s.Raw("]TJ\n"u8);
    }

    // Whether a glyph's span [lo, hi] along one axis overlaps an area's [areaLo, areaHi] by at least a hundredth of the
    // glyph (a glyph of no extent: when it lies strictly inside), so a neighbour that only touches an area stays.
    private static bool Covers(DocumentBox area, double lo, double hi, double areaLo, double areaHi)
    {
        _ = area;
        double extent = hi - lo;
        if (extent <= 1e-9)
        {
            return lo > areaLo && lo < areaHi;
        }

        return Math.Min(hi, areaHi) - Math.Max(lo, areaLo) > Math.Max(1e-6, extent * 0.01);
    }

    // An image just drawn (`name`, or inline when null): drawn from a copy whose pixels with their centre inside an area
    // take its fill; false when no pixel lies inside an area (or the image cannot be decoded: drawn as it was).
    private bool RedactImage(PdfRewriter w, DocumentImage image, string? name)
    {
        var m = image.Placement;
        int width = image.PixelWidth, height = image.PixelHeight;
        double det = (double)m.A * m.D - (double)m.B * m.C;
        if (width <= 0 || height <= 0 || det == 0 || !double.IsFinite(det))
        {
            return false;
        }

        // Each area's pixel range found through the inverse map, then each pixel's centre tested exactly.
        var hits = new List<(int Area, int I0, int I1, int J0, int J1)>();
        for (int a = 0; a < w.Areas.Length; a++)
        {
            var area = w.Areas[a];
            double u0 = double.MaxValue, u1 = double.MinValue, v0 = double.MaxValue, v1 = double.MinValue;
            foreach (var (x, y) in new (double, double)[] { (area.X, area.Y), (area.Right, area.Y), (area.Right, area.Bottom), (area.X, area.Bottom) })
            {
                double dx = x - m.E, dy = y - m.F;
                double u = (m.D * dx - m.C * dy) / det, v = (-m.B * dx + m.A * dy) / det;
                (u0, u1, v0, v1) = (Math.Min(u0, u), Math.Max(u1, u), Math.Min(v0, v), Math.Max(v1, v));
            }

            int i0 = Math.Max(0, (int)Math.Floor(u0 * width) - 1), i1 = Math.Min(width - 1, (int)Math.Ceiling(u1 * width) + 1);
            int j0 = Math.Max(0, (int)Math.Floor(v0 * height) - 1), j1 = Math.Min(height - 1, (int)Math.Ceiling(v1 * height) + 1);
            if (i0 <= i1 && j0 <= j1)
            {
                hits.Add((a, i0, i1, j0, j1));
            }
        }

        if (hits.Count == 0)
        {
            return false;
        }

        float[]? pixels = null;
        int channels = 0;
        long changed = 0;
        foreach (var (a, i0, i1, j0, j1) in hits)
        {
            var area = w.Areas[a];
            var fill = w.AreaFills[a];
            for (int j = j0; j <= j1; j++)
            {
                for (int i = i0; i <= i1; i++)
                {
                    double u = (i + 0.5) / width, v = (j + 0.5) / height;
                    double x = m.A * u + m.C * v + m.E, y = m.B * u + m.D * v + m.F;
                    if (!(x > area.X && x < area.Right && y > area.Y && y < area.Bottom))
                    {
                        continue;
                    }

                    if (pixels is null)
                    {
                        ImageData decoded;
                        try
                        {
                            decoded = image.Decode();
                        }
                        catch (Exception e) when (e is NotSupportedException or InvalidDataException)
                        {
                            _issues.Add(DocumentIssue.NotSupported, $"page {_page}: an image under a redaction area cannot be decoded, so its pixels are not overwritten: {e.Message}");
                            return false;
                        }

                        (pixels, channels) = (decoded.Pixels, decoded.Channels);
                    }

                    int at = j * width + i, plane = width * height;
                    if (channels == 1)
                    {
                        pixels[at] = fill.Luma;
                    }
                    else
                    {
                        (pixels[at], pixels[plane + at], pixels[2 * plane + at]) = (fill.Red, fill.Green, fill.Blue);
                    }

                    changed++;
                }
            }
        }

        if (pixels is null)
        {
            return false;
        }

        string own = w.Name("RdIm");
        w.Result.Images.Add((own, new ImageData(pixels, channels, height, width)));
        w.Result.ImagesChanged++;
        w.Result.Pixels += changed;
        if (name is not null)
        {
            w.Replaced(name);
        }

        w.Write().Byte((byte)'/').Raw(System.Text.Encoding.ASCII.GetBytes(own)).Raw(" Do\n"u8);
        return true;
    }

    // A form just drawn (`name`, under `ctm`) whose box lies over an area: drawn from a copy of its content redacted the
    // same way; false when it lies clear of the areas or nothing in it was under one.
    private bool RedactForm(PdfRewriter w, string name, PdfDictionary? resources, PdfMatrix ctm)
    {
        PdfStream? stream;
        try
        {
            stream = _file.Resolve(Lookup(resources, "XObject"u8, System.Text.Encoding.UTF8.GetBytes(name))).AsStream;
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            return false;
        }

        if (stream is null || !_file.Resolve(stream.Dictionary["Subtype"u8]).IsName("Form"u8))
        {
            return false;
        }

        var matrix = PdfShadings.Matrix(_file, stream.Dictionary["Matrix"u8]);
        var box = PdfFunction.Numbers(_file, stream.Dictionary["BBox"u8]);
        var placed = matrix.Then(ctm);
        if (box is { Length: 4 })
        {
            var bounds = PdfAnnotations.Bounds(placed, box[0], box[1], box[2], box[3]);
            if (!w.Areas.Any(a => PdfRedactor.Overlap(a, bounds.X, bounds.Y, bounds.Right, bounds.Bottom)))
            {
                return false;
            }
        }

        if (w.Depth + 1 > _file.MaxDepth)
        {
            throw PdfLimit.Error(string.Create(CultureInfo.InvariantCulture,
                $"forms redacted within forms {w.Depth + 1} levels deep, deeper than the limit {nameof(DocumentOptions.MaxDepth)} (DocumentOptions.MaxDepth, {_file.MaxDepth})"));
        }

        var formResources = _file.Dictionary(stream.Dictionary["Resources"u8]) ?? resources;
        var inner = new PdfRedactedContent();
        using (var reader = new PdfContentReader(_file, _page, _issues, _includeHidden))
        using (var rewriter = new PdfRewriter(w.Areas, w.AreaFills, w.Width, w.Height, formResources, inner, w.Depth + 1))
        {
            reader.ReadImages();
            reader.ReadVector();
            reader.Rewriting(rewriter);
            using var decoded = _file.Decode(stream);
            reader.ReadContent(decoded.Span, formResources, placed, placed, box, 0);
            inner.Content = rewriter.Finish([]);
        }

        if (!inner.Changed)
        {
            return false;
        }

        string own = w.Name("RdFm");
        w.Result.Forms.Add((own, stream, formResources, inner));
        w.Result.Add(inner);
        w.Replaced(name);
        w.Write().Byte((byte)'/').Raw(System.Text.Encoding.ASCII.GetBytes(own)).Raw(" Do\n"u8);
        return true;
    }

    // A name operand as text (its #xx escapes decoded).
    private static string NameText(ReadOnlySpan<byte> data, Operand operand)
    {
        Span<byte> buffer = stackalloc byte[128];
        return PdfText.Utf8OrLatin1(NameBytes(data.Slice(operand.Start, operand.Length), buffer));
    }
}
