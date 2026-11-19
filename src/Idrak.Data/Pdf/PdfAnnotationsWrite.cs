// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Data.Pdf;

/// <summary>A page of the file being written, as a destination or an annotation's page: its object and the map from its points (as shown) to its user space.</summary>
/// <param name="Number">The page object's number.</param>
/// <param name="Generation">Its generation.</param>
/// <param name="ToUser">Page points (top-left origin, y down) to the page's user space.</param>
internal readonly record struct PdfPageTarget(int Number, int Generation, PdfMatrix ToUser);

/// <summary>
/// An annotation to write (ISO 32000-2, 12.5): its look drawn when it is added (its own appearance's parts, else
/// <see cref="DocumentAnnotationLooks.Look"/> of its properties) into a form of the page's user space, and its dictionary
/// written with the page: its kind, rectangle, text, author, dates, colours, opacity, flags, border, points (quads, a
/// line's ends, vertices, ink strokes), line endings, icon, thread (<c>/IRT</c>, <c>/RT</c>), state, popup, a link's web
/// address or destination, a file attachment's file, and <c>/AP /N</c>.
/// </summary>
internal sealed class PdfAnnotationDraft
{
    private readonly PdfPageBuilder? _look;
    private readonly bool _multiply;

    private PdfAnnotationDraft(DocumentAnnotation annotation, PdfMatrix toUser, PdfPageBuilder? look, bool multiply) =>
        (Annotation, ToUser, _look, _multiply) = (annotation, toUser, look, multiply);

    /// <summary>The annotation as given (its page the one it is written on).</summary>
    public DocumentAnnotation Annotation { get; }

    /// <summary>Its page's points to the page's user space.</summary>
    public PdfMatrix ToUser { get; }

    /// <summary>
    /// <paramref name="annotation"/> for page <paramref name="index"/> of <paramref name="width"/> x
    /// <paramref name="height"/> points (<paramref name="frame"/> from its points to its user space; null: a new page's),
    /// its look drawn now through the document's fonts and images.
    /// </summary>
    /// <exception cref="ArgumentException">No kind, or a box that is not finite.</exception>
    public static PdfAnnotationDraft Make(DocumentAnnotation annotation, int index, float width, float height, PdfMatrix? frame, Func<IFontFace, PdfWriterFont> font,
        Func<ImageData, DocumentImageEncoding, int> image)
    {
        ArgumentNullException.ThrowIfNull(annotation);
        ArgumentException.ThrowIfNullOrWhiteSpace(annotation.Kind, nameof(annotation));
        var box = annotation.Box;
        if (!float.IsFinite(box.X) || !float.IsFinite(box.Y) || !(box.Width >= 0) || !(box.Height >= 0) || !float.IsFinite(box.Width) || !float.IsFinite(box.Height))
        {
            throw new ArgumentException($"An annotation's box has a finite place, width and height ({box}).", nameof(annotation));
        }

        annotation = annotation with { Page = index };
        var look = annotation.Kind == DocumentAnnotationKinds.Popup ? null : annotation.Appearance ?? DocumentAnnotationLooks.Look(annotation, width, height);
        PdfPageBuilder? builder = null;
        bool multiply = false;
        if (look is not null)
        {
            builder = new PdfPageBuilder(font, image, index, width, height, frame);
            foreach (var path in look.Paths)
            {
                if (!path.IsEmpty && (path.Fill != DocumentFill.None || path.Stroke is not null))
                {
                    builder.Path(path);
                    multiply |= path.Blend == DocumentBlend.Multiply;
                }
            }

            foreach (var picture in look.Images)
            {
                if (picture.Box is { Width: > 0, Height: > 0 } placed)
                {
                    builder.Image(picture.Decode(), placed, DocumentImageEncoding.Flate);
                }
            }

            foreach (var run in look.Runs)
            {
                if (run.Face is { } face && !run.Glyphs.IsEmpty && run.FontSize > 0)
                {
                    builder.Glyphs(face, run.FontSize, run.Glyphs.Span, run.Text, run.Mode, run.Colour);
                }
            }
        }

        return new PdfAnnotationDraft(annotation, frame ?? new PdfMatrix(1, 0, 0, -1, 0, height), builder, multiply);
    }

    /// <summary>Lets go of the look's buffers (an annotation never written).</summary>
    public void Release() => _look?.Release();

    /// <summary>
    /// Writes the annotation as object <paramref name="number"/> of page <paramref name="page"/>: its look first (a form),
    /// a file attachment's file, then its dictionary. <paramref name="image"/> gives the number an image the look names
    /// is written under, <paramref name="pages"/> a destination's page, <paramref name="annotations"/> the object another
    /// annotation's <see cref="DocumentAnnotation.Id"/> names (its thread, its popup).
    /// </summary>
    public void Write(PdfFileWriter file, int number, PdfPageTarget page, Func<int, int> image, Func<int, PdfPageTarget?> pages,
        Func<string, (int Number, int Generation)?> annotations)
    {
        var a = Annotation;
        var rect = Rect(a.Box);
        int appearance = WriteLook(file, rect, image);
        int attachment = a.Attachment is { } file2 && a.Kind == DocumentAnnotationKinds.FileAttachment ? WriteFile(file, file2) : 0;
        // The annotations it names, found now: finding one of the source's may copy it into the file, which writes an object.
        var irt = a.InReplyTo is { } replyTo ? annotations(replyTo) : null;
        var popup = a.Popup is { } popupId ? annotations(popupId) : null;
        var parent = a.Parent is { } parentId ? annotations(parentId) : null;
        var s = file.BeginSmall(number);
        s.Raw("<</Type/Annot/Subtype"u8).Name(PdfAnnotations.Subtype(a.Kind)).Raw("/Rect["u8);
        Numbers(s, rect);
        s.Raw("]/P "u8).Integer(page.Number).Space().Integer(page.Generation).Raw(" R"u8);
        if (a.Id is { Length: > 0 } id && !id.StartsWith('#'))
        {
            s.Raw("/NM"u8).Text(id);
        }

        var flags = a.Flags == DocumentAnnotationFlags.None && a.Kind != DocumentAnnotationKinds.Popup ? DocumentAnnotationFlags.Print : a.Flags;
        if (flags != DocumentAnnotationFlags.None)
        {
            s.Raw("/F "u8).Integer((int)flags);
        }

        Text(s, "Contents"u8, a.Contents);
        Text(s, "T"u8, a.Author);
        Text(s, "State"u8, a.State);
        Text(s, "StateModel"u8, a.StateModel);
        Text(s, "OverlayText"u8, a.OverlayText);
        if (a.Created is { } created)
        {
            s.Raw("/CreationDate"u8).Text(PdfDocumentBuilder.Date(created));
        }

        if (a.Modified is { } modified)
        {
            s.Raw("/M"u8).Text(PdfDocumentBuilder.Date(modified));
        }

        Colour(s, "C"u8, a.Colour);
        Colour(s, "IC"u8, a.InteriorColour);
        if (a.Opacity < 1f)
        {
            s.Raw("/CA "u8).Number(Math.Clamp(a.Opacity, 0f, 1f));
        }

        if (a.BorderWidth is { } border)
        {
            s.Raw(a.Kind == DocumentAnnotationKinds.Link ? "/Border[0 0 "u8 : "/BS<</W "u8).Number(Math.Max(0f, border)).Raw(a.Kind == DocumentAnnotationKinds.Link ? "]"u8 : ">>"u8);
        }
        else if (a.Kind == DocumentAnnotationKinds.Link)
        {
            s.Raw("/Border[0 0 0]"u8);                                         // a link draws no border unless given one
        }

        Points(s, a);
        if (a.LineEndings.Count > 0)
        {
            s.Raw("/LE["u8);
            foreach (string ending in a.LineEndings.Take(2))
            {
                s.Name(ending);
            }

            s.Byte((byte)']');
        }

        if (a.Icon is { Length: > 0 } icon)
        {
            s.Raw("/Name"u8).Name(icon);
        }

        if (a.Open)
        {
            s.Raw("/Open true"u8);
        }

        Reference(s, "IRT"u8, irt);
        if (irt is not null)
        {
            s.Raw(a.ReplyType == "group" ? "/RT/Group"u8 : "/RT/R"u8);
        }

        Reference(s, "Popup"u8, popup);
        Reference(s, "Parent"u8, parent);
        if (a.Uri is { Length: > 0 } uri)
        {
            s.Raw("/A<</S/URI/URI"u8).Literal(System.Text.Encoding.ASCII.GetBytes(uri)).Raw(">>"u8);
        }
        else if (a.Destination is { } destination)
        {
            s.Raw("/Dest"u8);
            PdfNavigationWriter.Destination(s, destination, pages);
        }

        if (attachment > 0)
        {
            s.Raw("/FS "u8).Reference(attachment);
        }

        if (appearance > 0)
        {
            s.Raw("/AP<</N "u8).Reference(appearance).Raw(">>"u8);
        }

        s.Raw(">>"u8);
        file.EndSmall();
    }

    // The look as a form of the page's user space: its box the annotation's rectangle, its matrix the identity; 0 when it has none.
    private int WriteLook(PdfFileWriter file, double[] rect, Func<int, int> image)
    {
        if (_look?.Close() is not { } content)
        {
            return 0;
        }

        // Its images' numbers found first: an editor writes an image when it is first named.
        var images = _look.ImageNumbers.Select(image).ToList();
        int number = file.Allocate();
        var s = file.BeginStream(number);
        s.Raw("/Type/XObject/Subtype/Form/BBox["u8);
        Numbers(s, rect);
        s.Raw("]/Resources<<"u8);
        if (_look.ImageNumbers.Count > 0)
        {
            s.Raw("/XObject<<"u8);
            for (int i = 0; i < _look.ImageNumbers.Count; i++)
            {
                s.Raw("/"u8).Raw(_look.Names).Raw("Im"u8).Integer(i + 1).Space().Reference(images[i]);
            }

            s.Raw(">>"u8);
        }

        if (_look.Fonts.Count > 0)
        {
            s.Raw("/Font<<"u8);
            foreach (var font in _look.Fonts)
            {
                s.Raw("/"u8).Raw(_look.Names).Raw("F"u8).Integer(font.Index).Space().Reference(font.Number);
            }

            s.Raw(">>"u8);
        }

        if (_look.Alphas.Count > 0 || _multiply)
        {
            s.Raw("/ExtGState<<"u8);
            for (int i = 0; i < _look.Alphas.Count; i++)
            {
                s.Raw("/"u8).Raw(_look.Names).Raw("GS"u8).Integer(i + 1).Raw("<</ca "u8).Number(_look.Alphas[i].Fill).Raw("/CA "u8).Number(_look.Alphas[i].Stroke).Raw(">>"u8);
            }

            if (_multiply)
            {
                s.Raw("/LkM<</BM/Multiply>>"u8);                              // a highlight: multiplied over the text
            }

            s.Raw(">>"u8);
        }

        s.Raw(">>"u8);
        try
        {
            if (_multiply)
            {
                using var joined = new PdfSyntax(content.Length + 16);
                joined.Raw("/LkM gs\n"u8).Raw(content.Span);
                file.EndStream(joined.Span, deflate: true);
            }
            else
            {
                file.EndStream(content.Span, deflate: true);
            }
        }
        finally
        {
            _look.Release();
        }

        return number;
    }

    // A file attachment's file: its bytes deflated into an embedded file, and a file specification naming it.
    private static int WriteFile(PdfFileWriter file, DocumentAttachment attachment)
    {
        int stream = file.Allocate(), spec = file.Allocate();
        using (var content = attachment.Read())
        using (var data = new PdfSyntax(64 * 1024))
        {
            long size = PdfDeflate.Compress(content, file.Level, data.Buffer);
            var s = file.BeginStream(stream).Raw("/Type/EmbeddedFile"u8);
            if (!string.IsNullOrEmpty(attachment.MediaType))
            {
                s.Raw("/Subtype"u8).Name(attachment.MediaType);
            }

            s.Raw("/Params<</Size "u8).Integer(size).Raw(">>"u8);
            if (file.Level > 0)
            {
                s.Raw("/Filter/FlateDecode"u8);
            }

            file.EndStream(data.Span, deflate: false, PdfCryptUse.EmbeddedFile);
        }

        var f = file.BeginSmall(spec);
        f.Raw("<</Type/Filespec/F "u8).Text(attachment.Name).Raw("/UF "u8).Text(attachment.Name).Raw("/EF<</F "u8).Reference(stream).Raw(">>"u8);
        if (attachment.Description is { } description)
        {
            f.Raw("/Desc"u8).Text(description);
        }

        f.Raw(">>"u8);
        file.EndSmall();
        return spec;
    }

    // The quads, a line's ends, the vertices and the ink strokes, in the page's user space.
    private void Points(PdfSyntax s, DocumentAnnotation a)
    {
        var points = a.Points.Span;
        ReadOnlySpan<byte> key = a.Kind switch
        {
            DocumentAnnotationKinds.Line => "/L["u8,
            DocumentAnnotationKinds.Polygon or DocumentAnnotationKinds.Polyline => "/Vertices["u8,
            DocumentAnnotationKinds.Highlight or DocumentAnnotationKinds.Underline or DocumentAnnotationKinds.Squiggly or DocumentAnnotationKinds.StrikeOut
                or DocumentAnnotationKinds.Redaction => "/QuadPoints["u8,
            _ => default,
        };
        int whole = a.Kind == DocumentAnnotationKinds.Line ? 4 : key.SequenceEqual("/QuadPoints["u8) ? 8 : 2;
        if (!key.IsEmpty && points.Length >= whole)
        {
            s.Raw(key);
            UserPoints(s, points[..(points.Length - points.Length % whole)]);
            s.Byte((byte)']');
        }

        if (a.Ink.Count > 0)
        {
            s.Raw("/InkList["u8);
            foreach (var stroke in a.Ink)
            {
                // A stroke's points in order: its moves and lines' ends, a curve's end (its controls dropped).
                var verbs = stroke.Verbs.Span;
                var p = stroke.Points.Span;
                var ends = new List<float>(p.Length);
                int at = 0;
                foreach (var verb in verbs)
                {
                    int take = verb switch { OutlineVerb.Move or OutlineVerb.Line => 2, OutlineVerb.Quadratic => 4, OutlineVerb.Cubic => 6, _ => 0 };
                    if (take > 0)
                    {
                        ends.Add(p[at + take - 2]);
                        ends.Add(p[at + take - 1]);
                    }

                    at += take;
                }

                s.Byte((byte)'[');
                UserPoints(s, [.. ends]);
                s.Byte((byte)']');
            }

            s.Byte((byte)']');
        }
    }

    private void UserPoints(PdfSyntax s, ReadOnlySpan<float> points)
    {
        var m = ToUser;
        for (int i = 0; i + 1 < points.Length; i += 2)
        {
            if (i > 0)
            {
                s.Space();
            }

            double x = points[i], y = points[i + 1];
            s.Number(m.A * x + m.C * y + m.E, 4).Space().Number(m.B * x + m.D * y + m.F, 4);
        }
    }

    // The annotation's box as a rectangle of the page's user space (llx lly urx ury).
    private double[] Rect(DocumentBox box)
    {
        var m = ToUser;
        double minX = double.MaxValue, minY = double.MaxValue, maxX = double.MinValue, maxY = double.MinValue;
        foreach (var (x, y) in new (double, double)[] { (box.X, box.Y), (box.Right, box.Y), (box.Right, box.Bottom), (box.X, box.Bottom) })
        {
            double ux = m.A * x + m.C * y + m.E, uy = m.B * x + m.D * y + m.F;
            (minX, minY, maxX, maxY) = (Math.Min(minX, ux), Math.Min(minY, uy), Math.Max(maxX, ux), Math.Max(maxY, uy));
        }

        return [minX, minY, maxX, maxY];
    }

    private static void Numbers(PdfSyntax s, ReadOnlySpan<double> numbers)
    {
        for (int i = 0; i < numbers.Length; i++)
        {
            if (i > 0)
            {
                s.Space();
            }

            s.Number(numbers[i], 4);
        }
    }

    private static void Text(PdfSyntax s, ReadOnlySpan<byte> key, string? text)
    {
        if (text is not null)
        {
            s.Byte((byte)'/').Raw(key).Text(text);
        }
    }

    private static void Colour(PdfSyntax s, ReadOnlySpan<byte> key, DocumentColour? colour)
    {
        if (colour is not { } c)
        {
            return;
        }

        s.Byte((byte)'/').Raw(key).Byte((byte)'[');
        for (int i = 0; i < c.Count; i++)
        {
            if (i > 0)
            {
                s.Space();
            }

            s.Number(c[i]);
        }

        s.Byte((byte)']');
    }

    private static void Reference(PdfSyntax s, ReadOnlySpan<byte> key, (int Number, int Generation)? target)
    {
        if (target is { } t)
        {
            s.Byte((byte)'/').Raw(key).Space().Integer(t.Number).Space().Integer(t.Generation).Raw(" R"u8);
        }
    }
}

/// <summary>Destinations, bookmarks and page labels written (ISO 32000-2, 12.3).</summary>
internal static class PdfNavigationWriter
{
    /// <summary>
    /// Writes <paramref name="destination"/>: an explicit one (its page through <paramref name="pages"/>, the fit and its
    /// numbers in that page's user space), or its name when it names one and gives no page; null when its page is not
    /// in the file.
    /// </summary>
    public static void Destination(PdfSyntax s, DocumentDestination destination, Func<int, PdfPageTarget?> pages)
    {
        if (destination.Page < 0 || pages(destination.Page) is not { } page)
        {
            if (destination.Name is { Length: > 0 } name)
            {
                s.Text(name);
            }
            else
            {
                s.Raw("null"u8);
            }

            return;
        }

        var m = page.ToUser;
        double UserX(float x) => m.A * x + m.E;                               // an edge's coordinate (pages turned by 90 are rare targets: the map's own axis)
        double UserY(float y) => m.D * y + m.F;
        s.Byte((byte)'[').Integer(page.Number).Space().Integer(page.Generation).Raw(" R"u8);
        void Value(double? value)
        {
            s.Space();
            if (value is { } v && double.IsFinite(v))
            {
                s.Number(v, 4);
            }
            else
            {
                s.Raw("null"u8);
            }
        }

        switch (destination.Fit)
        {
            case DocumentDestinationFit.Page:
                s.Raw("/Fit"u8);
                break;
            case DocumentDestinationFit.Bounds:
                s.Raw("/FitB"u8);
                break;
            case DocumentDestinationFit.Width or DocumentDestinationFit.BoundsWidth:
                s.Raw(destination.Fit == DocumentDestinationFit.Width ? "/FitH"u8 : "/FitBH"u8);
                Value(destination.Y is { } y ? UserY(y) : null);
                break;
            case DocumentDestinationFit.Height or DocumentDestinationFit.BoundsHeight:
                s.Raw(destination.Fit == DocumentDestinationFit.Height ? "/FitV"u8 : "/FitBV"u8);
                Value(destination.X is { } x ? UserX(x) : null);
                break;
            case DocumentDestinationFit.Box:
                s.Raw("/FitR"u8);
                double l = UserX(destination.X ?? 0), t = UserY(destination.Y ?? 0), r = UserX(destination.Right ?? destination.X ?? 0), b = UserY(destination.Bottom ?? destination.Y ?? 0);
                Value(Math.Min(l, r));
                Value(Math.Min(t, b));
                Value(Math.Max(l, r));
                Value(Math.Max(t, b));
                break;
            default:
                s.Raw("/XYZ"u8);
                Value(destination.X is { } left ? UserX(left) : null);
                Value(destination.Y is { } top ? UserY(top) : null);
                Value(destination.Zoom);
                break;
        }

        s.Byte((byte)']');
    }

    /// <summary>Writes the bookmarks <paramref name="items"/> (with their children) as an outline tree; its root's number, or 0 when there are none.</summary>
    public static int Outline(PdfFileWriter file, IReadOnlyList<DocumentOutline> items, Func<int, PdfPageTarget?> pages)
    {
        if (items.Count == 0)
        {
            return 0;
        }

        int root = file.Allocate();
        var numbers = Number(file, items);
        Items(file, items, numbers, root, pages);
        var s = file.BeginSmall(root);
        s.Raw("<</Type/Outlines/First "u8).Reference(numbers[0].Number).Raw("/Last "u8).Reference(numbers[^1].Number).Raw("/Count "u8).Integer(Shown(items)).Raw(">>"u8);
        file.EndSmall();
        return root;
    }

    /// <summary>Writes the page labels as a catalog entry (<c>/PageLabels</c>, one number tree node), the ranges by their first page.</summary>
    public static void Labels(PdfSyntax s, IReadOnlyList<DocumentPageLabel> labels)
    {
        s.Raw("/PageLabels<</Nums["u8);
        bool first = true;
        foreach (var label in labels.OrderBy(l => l.StartPage))
        {
            if (!first)
            {
                s.Space();
            }

            first = false;
            s.Integer(label.StartPage).Raw("<<"u8);
            s.Raw(label.Numbering switch
            {
                DocumentPageNumbering.Decimal => "/S/D"u8,
                DocumentPageNumbering.UpperRoman => "/S/R"u8,
                DocumentPageNumbering.LowerRoman => "/S/r"u8,
                DocumentPageNumbering.UpperLetters => "/S/A"u8,
                DocumentPageNumbering.LowerLetters => "/S/a"u8,
                _ => default,
            });
            if (label.Prefix is { Length: > 0 } prefix)
            {
                s.Raw("/P"u8).Text(prefix);
            }

            if (label.FirstNumber != 1)
            {
                s.Raw("/St "u8).Integer(label.FirstNumber);
            }

            s.Raw(">>"u8);
        }

        s.Raw("]>>"u8);
    }

    // Each item's number and its children's, depth first.
    private static List<Numbered> Number(PdfFileWriter file, IReadOnlyList<DocumentOutline> items) =>
        [.. items.Select(item => new Numbered(file.Allocate(), item.Children.Count > 0 ? Number(file, item.Children) : []))];

    private static void Items(PdfFileWriter file, IReadOnlyList<DocumentOutline> items, List<Numbered> numbers, int parent, Func<int, PdfPageTarget?> pages)
    {
        for (int i = 0; i < items.Count; i++)
        {
            var item = items[i];
            var children = numbers[i].Children;
            if (children.Count > 0)
            {
                Items(file, item.Children, children, numbers[i].Number, pages);
            }

            var s = file.BeginSmall(numbers[i].Number);
            s.Raw("<</Title"u8).Text(item.Title).Raw("/Parent "u8).Reference(parent);
            if (i > 0)
            {
                s.Raw("/Prev "u8).Reference(numbers[i - 1].Number);
            }

            if (i + 1 < items.Count)
            {
                s.Raw("/Next "u8).Reference(numbers[i + 1].Number);
            }

            if (children.Count > 0)
            {
                int below = item.Children.Sum(c => 1 + Shown(c));
                s.Raw("/First "u8).Reference(children[0].Number).Raw("/Last "u8).Reference(children[^1].Number).Raw("/Count "u8).Integer(item.Open ? below : -below);
            }

            if (item.Uri is { Length: > 0 } uri)
            {
                s.Raw("/A<</S/URI/URI"u8).Literal(System.Text.Encoding.ASCII.GetBytes(uri)).Raw(">>"u8);
            }
            else if (item.Destination is { } destination)
            {
                s.Raw("/Dest"u8);
                Destination(s, destination, pages);
            }

            if (item.Colour is { } colour)
            {
                s.Raw("/C["u8).Number(colour.Red).Space().Number(colour.Green).Space().Number(colour.Blue).Byte((byte)']');
            }

            if (item.Style != DocumentOutlineStyle.None)
            {
                s.Raw("/F "u8).Integer((int)item.Style);
            }

            s.Raw(">>"u8);
            file.EndSmall();
        }
    }

    // The items shown under an open item (or at the top level): each, and those shown under each that is open.
    private static int Shown(IReadOnlyList<DocumentOutline> items) => items.Sum(i => 1 + Shown(i));

    private static int Shown(DocumentOutline item) => item.Open ? Shown(item.Children) : 0;

    // An item's number and its children's.
    private sealed record Numbered(int Number, List<Numbered> Children);
}

/// <summary>A page's annotations and links (contract C24): each drawn when it is added, written with the page.</summary>
internal sealed partial class PdfPageBuilder
{
    private List<PdfAnnotationDraft>? _drafts;

    /// <summary>What takes the page's annotations as they are written with it: the page object's number, each annotation's number and draft (written by the document at its end, when every page has its number).</summary>
    internal Action<int, IReadOnlyList<(int Number, PdfAnnotationDraft Draft)>>? AnnotationsWritten { get; init; }

    /// <summary>The annotations added, in order.</summary>
    internal IReadOnlyList<PdfAnnotationDraft> Drafts => _drafts ?? (IReadOnlyList<PdfAnnotationDraft>)[];

    public void AddAnnotation(DocumentAnnotation annotation)
    {
        ArgumentNullException.ThrowIfNull(annotation);
        ThrowIfWritten();
        (_drafts ??= []).Add(PdfAnnotationDraft.Make(annotation, Index, Width, Height, _frame, _font, _image));
    }

    public void AddLink(DocumentLink link)
    {
        ArgumentNullException.ThrowIfNull(link);
        if (link.Uri is null && link.Destination is null)
        {
            throw new ArgumentException("A link leads to a web address or a place in the document; this one gives neither.", nameof(link));
        }

        AddAnnotation(new DocumentAnnotation(DocumentAnnotationKinds.Link, Index, link.Box) { Uri = link.Uri, Destination = link.Destination });
    }

    // The page object's /Annots entry: the form's widgets (D9's, written already) and a number for each annotation added,
    // kept for the annotations to be written after the page; null when none was added.
    private List<(int Number, PdfAnnotationDraft Draft)>? AnnotationEntry(PdfSyntax s, PdfFileWriter file, List<int>? widgets)
    {
        var numbered = _drafts is { Count: > 0 } drafts ? drafts.Select(d => (file.Allocate(), d)).ToList() : null;
        if (widgets is not { Count: > 0 } && numbered is null)
        {
            return null;
        }

        s.Raw("/Annots["u8);
        bool first = true;
        foreach (int number in (widgets ?? []).Concat(numbered?.Select(n => n.Item1) ?? []))
        {
            if (!first)
            {
                s.Space();
            }

            first = false;
            s.Reference(number);
        }

        s.Byte((byte)']');
        return numbered;
    }
}

/// <summary>A document's bookmarks, page labels and the annotations of its pages, written at its end (contract C24).</summary>
internal sealed partial class PdfDocumentBuilder
{
    private readonly List<PdfPageTarget> _targets = [];
    private readonly List<(int Page, int Number, PdfAnnotationDraft Draft)> _annotations = [];
    private readonly List<DocumentOutline> _outline = [];
    private IReadOnlyList<DocumentPageLabel>? _labels;
    private int _outlineNumber;

    public void AddOutline(DocumentOutline outline)
    {
        ArgumentNullException.ThrowIfNull(outline);
        ThrowIfDone();
        _outline.Add(outline);
    }

    public void SetPageLabels(IReadOnlyList<DocumentPageLabel> labels)
    {
        ArgumentNullException.ThrowIfNull(labels);
        ThrowIfDone();
        foreach (var label in labels)
        {
            ArgumentNullException.ThrowIfNull(label, nameof(labels));
            ArgumentOutOfRangeException.ThrowIfNegative(label.StartPage, nameof(labels));
            ArgumentOutOfRangeException.ThrowIfLessThan(label.FirstNumber, 1, nameof(labels));
        }

        if (labels.Select(l => l.StartPage).Distinct().Count() != labels.Count)
        {
            throw new ArgumentException("Two page label ranges start on the same page.", nameof(labels));
        }

        _labels = labels.Count == 0 ? null : [.. labels];
    }

    // A page added: its annotations' drafts kept as it is written; its place as a destination recorded when written.
    private void Annotated(int page, IReadOnlyList<(int Number, PdfAnnotationDraft Draft)> annotations)
    {
        foreach (var (number, draft) in annotations)
        {
            _annotations.Add((page, number, draft));
        }
    }

    // Page `index` as a destination: once it is written (a copied page's annotations, written with it, reach earlier pages only).
    private PdfPageTarget? PageTarget(int index) => (uint)index < (uint)_targets.Count ? _targets[index] : null;

    // The page just written: its object and frame, for destinations.
    private void Target(int number, PdfMatrix toUser) => _targets.Add(new PdfPageTarget(number, 0, toUser));

    // At the end: the pages' annotations (every page numbered now) and the bookmarks.
    private void WriteNavigation()
    {
        var Page = PageTarget;
        var byId = new Dictionary<string, (int, int)>(StringComparer.Ordinal);
        foreach (var (_, number, draft) in _annotations)
        {
            if (draft.Annotation.Id is { } id)
            {
                byId.TryAdd(id, (number, 0));
            }
        }

        foreach (var (page, number, draft) in _annotations)
        {
            var target = _targets.Find(t => t.Number == page);
            draft.Write(_file, number, target.Number == page ? target : new PdfPageTarget(page, 0, draft.ToUser), n => n, Page,
                id => byId.TryGetValue(id, out var known) ? known : null);
        }

        _annotations.Clear();
        _outlineNumber = PdfNavigationWriter.Outline(_file, _outline, Page);
    }

    // The catalog's navigation entries: the bookmarks' root and the page labels.
    private void CatalogNavigation(PdfSyntax s)
    {
        if (_outlineNumber > 0)
        {
            s.Raw("/Outlines "u8).Reference(_outlineNumber);
        }

        if (_labels is { Count: > 0 } labels)
        {
            PdfNavigationWriter.Labels(s, labels);
        }
    }
}
