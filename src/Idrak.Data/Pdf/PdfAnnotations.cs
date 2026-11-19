// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Data.Pdf;

/// <summary>
/// A page's annotations (ISO 32000-2, 12.5), in the order of its <c>/Annots</c>: every kind by the library's names
/// (<see cref="DocumentAnnotationKinds"/>; another kind by its own subtype), its box and points on the page as shown (quad
/// points, a line's ends, a polygon's vertices), ink strokes as paths, text, author, dates, colours, opacity, flags, border
/// width and line endings, a link's web address or destination (named ones resolved), a note's icon and state, a file
/// attachment's file, its popup and the thread it replies in (each annotation's replies under it too), and its appearance
/// stream (<c>/AP /N</c>, the state <c>/AS</c> picks) read as parts placed on the page through its <c>/Rect</c>, its form's
/// <c>/BBox</c> and <c>/Matrix</c> (12.5.5). An annotation without <c>/NM</c> is known by its object (<c>#12</c>), or by its
/// place in the page's list when it is not an object of its own (<c>#p1.3</c>). A damaged annotation is passed over and
/// counted; a link's action other than going to a place or a web address is not taken.
/// </summary>
internal static class PdfAnnotations
{
    /// <summary>Page <paramref name="index"/>'s annotations (<paramref name="page"/>, shown through <paramref name="display"/>), their appearances holding the parts <paramref name="parts"/> asks for.</summary>
    public static IReadOnlyList<DocumentAnnotation> Read(PdfDocument document, int index, PdfDictionary page, PdfMatrix display, float width, float height, DocumentParts parts,
        PdfIssues issues)
    {
        var file = document.File;
        PdfArray? annots;
        try
        {
            annots = file.Resolve(page["Annots"u8]).AsArray;
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            issues.Add(DocumentIssue.DamagedStream, $"page {index + 1}: its annotations: {e.Message}");
            return [];
        }

        if (annots is null || annots.Count == 0)
        {
            return [];
        }

        var read = new List<DocumentAnnotation>(annots.Count);
        for (int k = 0; k < annots.Count; k++)
        {
            try
            {
                if (file.Dictionary(annots[k]) is { } annotation)
                {
                    read.Add(One(document, index, annots[k], annotation, k, display, width, height, parts, issues));
                }
            }
            catch (InvalidDataException e) when (!PdfLimit.Is(e))
            {
                issues.Add(DocumentIssue.DamagedStream, $"page {index + 1}: annotation {k + 1}: {e.Message}");
            }
        }

        return Threads(read);
    }

    /// <summary>The library's kind for a subtype (another subtype keeps its own name).</summary>
    public static string Kind(PdfName? subtype) => subtype?.ToString() switch
    {
        "Text" => DocumentAnnotationKinds.Note,
        "FreeText" => DocumentAnnotationKinds.FreeText,
        "Line" => DocumentAnnotationKinds.Line,
        "Square" => DocumentAnnotationKinds.Square,
        "Circle" => DocumentAnnotationKinds.Circle,
        "Polygon" => DocumentAnnotationKinds.Polygon,
        "PolyLine" => DocumentAnnotationKinds.Polyline,
        "Highlight" => DocumentAnnotationKinds.Highlight,
        "Underline" => DocumentAnnotationKinds.Underline,
        "Squiggly" => DocumentAnnotationKinds.Squiggly,
        "StrikeOut" => DocumentAnnotationKinds.StrikeOut,
        "Ink" => DocumentAnnotationKinds.Ink,
        "Stamp" => DocumentAnnotationKinds.Stamp,
        "Caret" => DocumentAnnotationKinds.Caret,
        "Link" => DocumentAnnotationKinds.Link,
        "FileAttachment" => DocumentAnnotationKinds.FileAttachment,
        "Popup" => DocumentAnnotationKinds.Popup,
        "Redact" => DocumentAnnotationKinds.Redaction,
        "Widget" => DocumentAnnotationKinds.Widget,
        null => "unknown",
        var other => other,
    };

    /// <summary>The subtype a kind is written as (a kind the library does not name is written as its own name).</summary>
    public static string Subtype(string kind) => kind switch
    {
        DocumentAnnotationKinds.Note => "Text",
        DocumentAnnotationKinds.FreeText => "FreeText",
        DocumentAnnotationKinds.Line => "Line",
        DocumentAnnotationKinds.Square => "Square",
        DocumentAnnotationKinds.Circle => "Circle",
        DocumentAnnotationKinds.Polygon => "Polygon",
        DocumentAnnotationKinds.Polyline => "PolyLine",
        DocumentAnnotationKinds.Highlight => "Highlight",
        DocumentAnnotationKinds.Underline => "Underline",
        DocumentAnnotationKinds.Squiggly => "Squiggly",
        DocumentAnnotationKinds.StrikeOut => "StrikeOut",
        DocumentAnnotationKinds.Ink => "Ink",
        DocumentAnnotationKinds.Stamp => "Stamp",
        DocumentAnnotationKinds.Caret => "Caret",
        DocumentAnnotationKinds.Link => "Link",
        DocumentAnnotationKinds.FileAttachment => "FileAttachment",
        DocumentAnnotationKinds.Popup => "Popup",
        DocumentAnnotationKinds.Redaction => "Redact",
        DocumentAnnotationKinds.Widget => "Widget",
        _ => kind,
    };

    /// <summary>The key an annotation is known by: its <c>/NM</c>, else its object, else its place in the page's list.</summary>
    public static string Id(PdfFile file, PdfValue entry, PdfDictionary annotation, int page, int place)
    {
        if (file.Resolve(annotation["NM"u8]).AsString is { } name && name.Bytes.Length > 0)
        {
            return PdfText.Decode(name.Bytes);
        }

        return entry.IsReference
            ? string.Create(CultureInfo.InvariantCulture, $"#{entry.AsReference.Number}")
            : string.Create(CultureInfo.InvariantCulture, $"#p{page + 1}.{place + 1}");
    }

    // The key of an annotation another one refers to (/IRT, /Popup, /Parent): its /NM, else its object.
    private static string? IdOf(PdfFile file, PdfValue reference)
    {
        if (reference.IsNull)
        {
            return null;
        }

        var dictionary = file.Dictionary(reference);
        if (dictionary is not null && file.Resolve(dictionary["NM"u8]).AsString is { } name && name.Bytes.Length > 0)
        {
            return PdfText.Decode(name.Bytes);
        }

        return reference.IsReference ? string.Create(CultureInfo.InvariantCulture, $"#{reference.AsReference.Number}") : null;
    }

    private static DocumentAnnotation One(PdfDocument document, int index, PdfValue entry, PdfDictionary a, int place, PdfMatrix display, float width, float height,
        DocumentParts parts, PdfIssues issues)
    {
        var file = document.File;
        string kind = Kind(file.Resolve(a["Subtype"u8]).AsName);
        var rect = Numbers(file, a["Rect"u8]);
        var box = rect is { Length: >= 4 } r ? Bounds(display, r[0], r[1], r[2], r[3]) : DocumentBox.Empty;
        float opacity = file.Resolve(a["CA"u8]).TryNumber(out double ca) && double.IsFinite(ca) ? (float)Math.Clamp(ca, 0, 1) : 1f;
        long flags = file.Resolve(a["F"u8]).TryInteger(out long f) ? f : 0;
        var bs = file.Dictionary(a["BS"u8]);
        float? border = bs is not null && file.Resolve(bs["W"u8]).TryNumber(out double bw) && double.IsFinite(bw) ? (float)bw
            : Numbers(file, a["Border"u8]) is { Length: >= 3 } borders && double.IsFinite(borders[2]) ? (float)borders[2] : null;

        // The points: quads, a line's ends, vertices; ink strokes as paths.
        float[]? points = kind is DocumentAnnotationKinds.Line ? Points(display, Numbers(file, a["L"u8]), 4)
            : kind is DocumentAnnotationKinds.Polygon or DocumentAnnotationKinds.Polyline ? Points(display, Numbers(file, a["Vertices"u8]), 2)
            : Points(display, Numbers(file, a["QuadPoints"u8]), 8);
        var ink = new List<DocumentPath>();
        if (file.Resolve(a["InkList"u8]).AsArray is { } strokes)
        {
            foreach (var stroke in strokes.Items)
            {
                if (Points(display, Numbers(file, stroke), 2) is { Length: >= 2 } p)
                {
                    var verbs = new OutlineVerb[p.Length / 2];
                    Array.Fill(verbs, OutlineVerb.Line);
                    verbs[0] = OutlineVerb.Move;
                    ink.Add(new DocumentPath(verbs, p) { Fill = DocumentFill.None, Stroke = border ?? 1f, StrokeColour = document.Colour(a["C"u8]) ?? DocumentColour.Black, Order = ink.Count });
                }
            }
        }

        // A link's target: its /Dest, else its action (a web address or a place; any other action is not taken).
        string? uri = null;
        DocumentDestination? destination = null;
        if (a.Contains("Dest"u8))
        {
            destination = document.Destination(a["Dest"u8], 0);
        }
        else if (a.Contains("A"u8))
        {
            (uri, destination) = document.Action(a["A"u8]);
        }

        var irt = a["IRT"u8];
        string? replyType = null;
        if (!irt.IsNull)
        {
            replyType = file.Resolve(a["RT"u8]).AsName is { } rt && rt.Is("Group"u8) ? "group" : "reply";
        }

        var endings = file.Resolve(a["LE"u8]).AsArray is { } le ? le.Items.Select(e => file.Resolve(e).AsName?.ToString()).OfType<string>().ToArray() : [];
        string id = Id(file, entry, a, index, place);
        return new DocumentAnnotation(kind, index, box)
        {
            Id = id,
            Contents = Text(file, a, "Contents"u8),
            Author = Text(file, a, "T"u8),
            Created = file.Resolve(a["CreationDate"u8]).AsString is { } created ? PdfText.Date(created.Bytes) : null,
            Modified = file.Resolve(a["M"u8]).AsString is { } modified ? PdfText.Date(modified.Bytes) : null,
            Colour = document.Colour(a["C"u8]),
            InteriorColour = document.Colour(a["IC"u8]),
            Opacity = opacity,
            Points = points ?? ReadOnlyMemory<float>.Empty,
            Ink = ink,
            Uri = uri,
            Destination = destination,
            Icon = kind is DocumentAnnotationKinds.Note or DocumentAnnotationKinds.Stamp or DocumentAnnotationKinds.FileAttachment ? file.Resolve(a["Name"u8]).AsName?.ToString() : null,
            Attachment = kind == DocumentAnnotationKinds.FileAttachment ? PdfAttachments.Of(file, a["FS"u8], a) : null,
            Appearance = Appearance(document, index, a, rect, display, width, height, parts, opacity, issues),
            BorderWidth = border,
            LineEndings = endings,
            Flags = (DocumentAnnotationFlags)(flags & 1023),
            InReplyTo = IdOf(file, irt),
            ReplyType = replyType,
            State = Text(file, a, "State"u8),
            StateModel = Text(file, a, "StateModel"u8),
            Popup = IdOf(file, a["Popup"u8]),
            Parent = kind == DocumentAnnotationKinds.Popup ? IdOf(file, a["Parent"u8]) : null,
            Open = file.Resolve(a["Open"u8]).AsBoolean,
            OverlayText = Text(file, a, "OverlayText"u8),
        };
    }

    // The normal appearance (/AP /N, or the state /AS names of a dictionary of states; none for a hidden annotation) read
    // through its placement as content of the page: the path a form's widgets are read by (PdfForms, D9), its text, paths
    // and images always (an annotation's look is whole), with its opacity.
    private static DocumentPage? Appearance(PdfDocument document, int index, PdfDictionary a, double[]? rect, PdfMatrix display, float width, float height, DocumentParts parts,
        float opacity, PdfIssues issues)
    {
        var file = document.File;
        if (rect is not { Length: >= 4 } || PdfForms.Hides(file, a) || PdfForms.AppearanceStream(file, a, PdfForms.State(file, a)) is not { } stream
            || PdfForms.Placement(file, stream, (Math.Min(rect[0], rect[2]), Math.Min(rect[1], rect[3]), Math.Max(rect[0], rect[2]), Math.Max(rect[1], rect[3]))) is not { } placement)
        {
            return null;
        }

        var own = new PdfIssues();
        var contentParts = (parts & ~DocumentParts.Annotations) | DocumentParts.Text | DocumentParts.Paths | DocumentParts.Images;
        try
        {
            var content = document.Content(index, width, height, contentParts, own, reader => reader.ReadForm(stream, placement.Then(display), opacity));
            return new DocumentPage(index, width, height)
            {
                Runs = content.Runs, Lines = content.Lines, Text = content.Text, Images = content.Images, Paths = content.Paths, Clips = content.Clips,
                Shadings = content.Shadings, Groups = content.Groups, SoftMasks = content.SoftMasks, Issues = own.ToList(),
            };
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            issues.Add(DocumentIssue.DamagedStream, $"page {index + 1}: an annotation's appearance (object {stream.Number}): {e.Message}");
            return null;
        }
    }

    // Each annotation with the replies to it (a thread: those whose /IRT names it and that are replies, not grouped),
    // in the page's order; a reply that comes back to itself ends there.
    private static IReadOnlyList<DocumentAnnotation> Threads(List<DocumentAnnotation> read)
    {
        var replies = new Dictionary<string, List<int>>(StringComparer.Ordinal);
        for (int i = 0; i < read.Count; i++)
        {
            if (read[i].InReplyTo is { } parent && read[i].ReplyType == "reply")
            {
                (replies.TryGetValue(parent, out var list) ? list : replies[parent] = []).Add(i);
            }
        }

        if (replies.Count == 0)
        {
            return read;
        }

        var made = new DocumentAnnotation?[read.Count];
        var open = new bool[read.Count];
        DocumentAnnotation Make(int i)
        {
            if (made[i] is { } done)
            {
                return done;
            }

            if (open[i] || read[i].Id is not { } id || !replies.TryGetValue(id, out var children))
            {
                return read[i];                                                // no replies, or a thread that loops
            }

            open[i] = true;
            var thread = new List<DocumentAnnotation>(children.Count);
            foreach (int c in children)
            {
                if (c != i)
                {
                    thread.Add(Make(c));
                }
            }

            open[i] = false;
            return (made[i] = read[i] with { Replies = thread })!;
        }

        var result = new DocumentAnnotation[read.Count];
        for (int i = 0; i < read.Count; i++)
        {
            result[i] = Make(i);
        }

        return result;
    }

    private static string? Text(PdfFile file, PdfDictionary dictionary, ReadOnlySpan<byte> key) =>
        file.Resolve(dictionary[key]).AsString is { } text ? PdfText.Decode(text.Bytes) : null;

    /// <summary>An array of numbers (each finite), or null when it is not one.</summary>
    public static double[]? Numbers(PdfFile file, PdfValue value)
    {
        if (file.Resolve(value).AsArray is not { } array)
        {
            return null;
        }

        var numbers = new double[array.Count];
        for (int i = 0; i < numbers.Length; i++)
        {
            if (!file.Resolve(array[i]).TryNumber(out numbers[i]) || !double.IsFinite(numbers[i]))
            {
                return null;
            }
        }

        return numbers;
    }

    // Numbers taken as points (x, y pairs), a whole number of `group` numbers, through the display map; null when none.
    private static float[]? Points(PdfMatrix display, double[]? numbers, int group)
    {
        if (numbers is null || numbers.Length < group)
        {
            return null;
        }

        int count = numbers.Length - numbers.Length % group;
        var points = new float[count];
        for (int i = 0; i < count; i += 2)
        {
            double x = numbers[i], y = numbers[i + 1];
            points[i] = (float)(display.A * x + display.C * y + display.E);
            points[i + 1] = (float)(display.B * x + display.D * y + display.F);
        }

        return points;
    }

    /// <summary>The box on the page as shown of a rectangle of user space (two corners).</summary>
    public static DocumentBox Bounds(PdfMatrix display, double x0, double y0, double x1, double y1)
    {
        double minX = double.MaxValue, minY = double.MaxValue, maxX = double.MinValue, maxY = double.MinValue;
        foreach (var (x, y) in new[] { (x0, y0), (x1, y0), (x1, y1), (x0, y1) })
        {
            double tx = display.A * x + display.C * y + display.E, ty = display.B * x + display.D * y + display.F;
            (minX, minY, maxX, maxY) = (Math.Min(minX, tx), Math.Min(minY, ty), Math.Max(maxX, tx), Math.Max(maxY, ty));
        }

        return new DocumentBox((float)minX, (float)minY, (float)(maxX - minX), (float)(maxY - minY));
    }
}
