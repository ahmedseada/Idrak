// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Data.Pdf;

/// <summary>How a page is written again: its number, its parent, its rotation, and what it states itself.</summary>
/// <param name="Number">The page object's number in the result.</param>
/// <param name="Parent">Its parent page-tree node.</param>
/// <param name="Rotation">Its rotation (0, 90, 180 or 270).</param>
/// <param name="Materialize">Whether it states what it inherits (its media box, crop box and resources): a page under a new parent.</param>
/// <param name="ParentCrops">Whether its parent passes a crop box (a page with none of its own then states its media box as its crop box).</param>
/// <param name="ParentRotates">Whether its parent passes a rotation (the page then states its own, 0 too).</param>
/// <param name="ParentGeneration">Its parent's generation.</param>
internal readonly record struct PdfPageWrite(int Number, int Parent, int Rotation, bool Materialize, bool ParentCrops, bool ParentRotates, int ParentGeneration = 0);

/// <summary>A page's annotations changed by a form edit: widgets removed (the source's objects, by number) and added (new objects, numbered when written).</summary>
internal sealed class PdfAnnotsEdit
{
    /// <summary>The source's annotation objects left out of the page's <c>/Annots</c>.</summary>
    public HashSet<int> Removed { get; } = [];

    /// <summary>New annotations added to it (each one's number, given when the document is written).</summary>
    public List<Func<int>> Added { get; } = [];
}

/// <summary>
/// Pages written again by an editor or a builder copying them: the page's own entries (its references made the result's
/// through a <see cref="PdfEditCopier"/>), what it inherits stated when it moves under another parent, its rotation, and
/// content drawn over it: a stream after the page's own (which a stream holding <c>q</c> opens before them, so what they
/// leave of the graphics state does not reach the new content), its resources added to the page's own under names they do
/// not start with. Pages an editor adds are written here too.
/// </summary>
internal static class PdfEditPages
{
    /// <summary>The catalog entry that marks AES-256 (revision 6) in a file before PDF 2.0: Adobe's extension level 8.</summary>
    public static ReadOnlySpan<byte> Aes256Extension => "/Extensions<</ADBE<</BaseVersion/1.7/ExtensionLevel 8>>>>"u8;

    /// <summary>The map back from page points as shown to the page's user space (a display map turned back).</summary>
    public static PdfMatrix Inverse(PdfMatrix m)
    {
        double det = m.A * m.D - m.B * m.C;
        if (det == 0 || !double.IsFinite(det))
        {
            return PdfMatrix.Identity;
        }

        return new PdfMatrix(m.D / det, -m.B / det, -m.C / det, m.A / det, (m.C * m.F - m.D * m.E) / det, (m.B * m.E - m.A * m.F) / det);
    }

    /// <summary>Page <paramref name="index"/>'s resources: its own, else those it inherits (null when it has none).</summary>
    public static PdfValue Resources(PdfDocument document, int index)
    {
        var (node, _, _, _, inherited) = document.PageObject(index);
        try
        {
            return document.File.Dictionary(node) is { } page && page.TryGet("Resources"u8, out var own) ? own : inherited;
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            return inherited;
        }
    }

    /// <summary>
    /// What the names of content drawn over page <paramref name="index"/> start with: "" when none of the page's own
    /// images, forms, fonts and graphics states is named as new content names them (<c>/Im1</c>, <c>/Fw1</c>, <c>/F1</c>,
    /// <c>/GS1</c>), else the first of "Ov1", "Ov2", ... that none of them starts with.
    /// </summary>
    public static string Prefix(PdfDocument document, int index)
    {
        var file = document.File;
        var names = new List<byte[]>();
        bool clash = false;
        try
        {
            if (file.Dictionary(Resources(document, index)) is { } resources)
            {
                foreach (var (category, own) in new[] { ("XObject", "Im"), ("XObject", "Fw"), ("Font", "F"), ("ExtGState", "GS") })
                {
                    if (file.Dictionary(resources[System.Text.Encoding.ASCII.GetBytes(category)]) is not { } dictionary)
                    {
                        continue;
                    }

                    for (int i = 0; i < dictionary.Count; i++)
                    {
                        var name = dictionary.KeyAt(i).Bytes;
                        names.Add(name.ToArray());
                        clash |= Generated(name, own);
                    }
                }
            }
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            clash = true;                                                     // damaged resources: named apart, to be safe
        }

        if (!clash)
        {
            return "";
        }

        for (int k = 1; ; k++)
        {
            byte[] prefix = System.Text.Encoding.ASCII.GetBytes("Ov" + k.ToString(System.Globalization.CultureInfo.InvariantCulture));
            if (!names.Exists(n => n.AsSpan().StartsWith(prefix)))
            {
                return System.Text.Encoding.ASCII.GetString(prefix);
            }
        }
    }

    /// <summary>Writes the stream that opens a page's own content before content drawn over it ("q"); its number.</summary>
    public static int WriteSave(PdfFileWriter file)
    {
        int number = file.Allocate();
        file.BeginStream(number);
        file.EndStream("q\n"u8, deflate: false);
        return number;
    }

    /// <summary>
    /// Writes page <paramref name="index"/> of <paramref name="source"/>'s document again as <paramref name="how"/> says,
    /// with <paramref name="content"/> drawn over it (its stream written now, its images numbered by
    /// <paramref name="image"/>; <paramref name="save"/> numbers the stream that opens the page's own content).
    /// </summary>
    public static void Write(PdfEditCopier copier, PdfFileWriter file, PdfEditCopier.Source source, int index, PdfPageWrite how, PdfPageBuilder? content,
        Func<int, int> image, Func<int> save, PdfAnnotsEdit? annots = null, PdfPageEdits? edits = null)
    {
        var pdf = source.File;
        var (node, media, crop, rotate, resources) = source.Document.PageObject(index);
        PdfDictionary page;
        try
        {
            page = pdf.Dictionary(node) ?? PdfDictionary.Empty;
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            page = PdfDictionary.Empty;                                       // a damaged page: written as an empty one of its size
        }

        if (edits is { Any: true })
        {
            page = edits.Apply(page, resources, copier, source, file);        // annotations added or removed, content redacted (PdfAnnotationsEdit)
        }

        bool ownMedia = page.TryGet("MediaBox"u8, out var value);
        media = ownMedia ? value : media;
        bool ownCrop = page.TryGet("CropBox"u8, out value);
        crop = ownCrop ? value : crop;
        bool ownRotate = page.Contains("Rotate"u8);
        bool ownResources = page.TryGet("Resources"u8, out value);
        resources = ownResources ? value : resources;
        var contents = page["Contents"u8];

        // The content drawn over the page: its stream now, after a Q that closes the q opening the page's own.
        var drawn = content?.Close();
        int overlay = 0;
        if (drawn is not null)
        {
            overlay = file.Allocate();
            file.BeginStream(overlay);
            if (contents.IsNull)
            {
                file.EndStream(drawn.Span, deflate: true);
            }
            else
            {
                using var joined = new PdfSyntax(drawn.Length + 2);
                joined.Raw("Q\n"u8).Raw(drawn.Span);
                file.EndStream(joined.Span, deflate: true);
            }
        }

        bool adds = content is not null && (content.ImageNumbers.Count > 0 || content.Fonts.Count > 0 || content.Alphas.Count > 0 || content.Forms.Count > 0);
        copier.WriteObject(how.Number, s =>
        {
            s.Raw("<</Type/Page"u8);
            if (how.Parent > 0)
            {
                s.Raw("/Parent "u8).Integer(how.Parent).Space().Integer(how.ParentGeneration).Raw(" R"u8);
            }

            if (how.Materialize || ownMedia)
            {
                if (!media.IsNull)
                {
                    copier.Entry(s, source, "MediaBox"u8, media);
                }
                else
                {
                    s.Raw("/MediaBox[0 0 612 792]"u8);                       // none anywhere: the size a reader takes
                }
            }

            if (how.Materialize || ownCrop)
            {
                if (!crop.IsNull)
                {
                    copier.Entry(s, source, "CropBox"u8, crop);
                }
                else if (how.ParentCrops)
                {
                    if (!media.IsNull)
                    {
                        copier.Entry(s, source, "CropBox"u8, media);
                    }
                    else
                    {
                        s.Raw("/CropBox[0 0 612 792]"u8);
                    }
                }
            }

            if (how.Rotation != 0 || ownRotate || how.ParentRotates || (!how.Materialize && !rotate.IsNull))
            {
                s.Raw("/Rotate "u8).Integer(how.Rotation);
            }

            if (adds)
            {
                MergedResources(s, copier, source, resources, content!, image);
            }
            else if (how.Materialize || ownResources)
            {
                if (!resources.IsNull)
                {
                    copier.Entry(s, source, "Resources"u8, resources);
                }
                else
                {
                    s.Raw("/Resources<<>>"u8);
                }
            }

            if (overlay > 0 && contents.IsNull)
            {
                s.Raw("/Contents "u8).Reference(overlay);
            }
            else if (overlay > 0)
            {
                s.Raw("/Contents["u8).Reference(save()).Space();
                PdfArray? array = null;
                try
                {
                    array = pdf.Resolve(contents).AsArray;
                }
                catch (InvalidDataException e) when (!PdfLimit.Is(e))
                {
                    // Contents that cannot be read: referred to as they are.
                }

                foreach (var item in array?.Items ?? new[] { contents })
                {
                    copier.Value(s, source, item);
                    s.Space();
                }

                s.Reference(overlay).Byte((byte)']');
            }
            else if (!contents.IsNull)
            {
                copier.Entry(s, source, "Contents"u8, contents);
            }

            bool annotated = false;
            for (int i = 0; i < page.Count; i++)
            {
                var key = page.KeyAt(i);
                if (annots is not null && key.Is("Annots"u8))
                {
                    annotated = true;
                    Annotations(s, copier, source, page.ValueAt(i), annots);
                }
                else if (!(key.Is("Type"u8) || key.Is("Parent"u8) || key.Is("MediaBox"u8) || key.Is("CropBox"u8) || key.Is("Rotate"u8) || key.Is("Resources"u8)
                      || key.Is("Contents"u8)))
                {
                    copier.Entry(s, source, key.Bytes, page.ValueAt(i));
                }
            }

            if (annots is not null && !annotated)
            {
                Annotations(s, copier, source, PdfValue.Null, annots);
            }

            s.Raw(">>"u8);
        });
    }

    // A page's /Annots with a form edit's widgets left out and added (none left: no entry).
    private static void Annotations(PdfSyntax s, PdfEditCopier copier, PdfEditCopier.Source source, PdfValue value, PdfAnnotsEdit annots)
    {
        PdfArray? array = null;
        try
        {
            array = source.File.Resolve(value).AsArray;
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            // Annotations that cannot be read: the added ones alone.
        }

        var kept = (array?.Items ?? []).Where(a => !(a.IsReference && annots.Removed.Contains(a.AsReference.Number))).ToList();
        if (kept.Count == 0 && annots.Added.Count == 0)
        {
            return;
        }

        s.Raw("/Annots["u8);
        foreach (var item in kept)
        {
            copier.Value(s, source, item);
            s.Space();
        }

        foreach (var added in annots.Added)
        {
            s.Reference(added()).Space();
        }

        s.Byte((byte)']');
    }

    /// <summary>
    /// Writes a page an editor added as <paramref name="number"/>: its size, its content (the stream written now) and the
    /// resources it names (images numbered by <paramref name="image"/>).
    /// </summary>
    public static void WriteNew(PdfEditCopier copier, PdfFileWriter file, int number, int parent, PdfPageBuilder content, int rotation, bool parentCrops,
        bool parentRotates, Func<int, int> image, PdfAnnotsEdit? annots = null, PdfPageEdits? edits = null)
    {
        var annotations = edits is { Any: true } ? edits.WriteNew(file) : null;
        var drawn = content.Close();
        int stream = 0;
        if (drawn is not null)
        {
            stream = file.Allocate();
            file.BeginStream(stream);
            file.EndStream(drawn.Span, deflate: true);
        }

        copier.WriteObject(number, s =>
        {
            s.Raw("<</Type/Page/Parent "u8).Reference(parent).Raw("/MediaBox[0 0 "u8).Number(content.Width).Space().Number(content.Height).Byte((byte)']');
            if (parentCrops)
            {
                s.Raw("/CropBox[0 0 "u8).Number(content.Width).Space().Number(content.Height).Byte((byte)']');
            }

            if (rotation != 0 || parentRotates)
            {
                s.Raw("/Rotate "u8).Integer(rotation);
            }

            s.Raw("/Resources<<"u8);
            Added(s, content, "XObject", image);
            Added(s, content, "Font", image);
            Added(s, content, "ExtGState", image);
            s.Raw(">>"u8);
            if (stream > 0)
            {
                s.Raw("/Contents "u8).Reference(stream);
            }

            if (annots is { Added.Count: > 0 } || annotations is { Count: > 0 })
            {
                s.Raw("/Annots["u8);
                foreach (var added in annots?.Added ?? [])
                {
                    s.Reference(added()).Space();
                }

                foreach (int annotation in annotations ?? [])
                {
                    s.Reference(annotation).Space();
                }

                s.Byte((byte)']');
            }

            s.Raw(">>"u8);
        });
    }

    // The page's resources with the new content's added: each category it adds to holds the page's own and the new.
    private static void MergedResources(PdfSyntax s, PdfEditCopier copier, PdfEditCopier.Source source, PdfValue resources, PdfPageBuilder content, Func<int, int> image)
    {
        PdfDictionary dictionary;
        try
        {
            dictionary = source.File.Dictionary(resources) ?? PdfDictionary.Empty;
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            dictionary = PdfDictionary.Empty;
        }

        s.Raw("/Resources<<"u8);
        var left = new List<string>(3);
        if (content.ImageNumbers.Count > 0 || content.Forms.Count > 0)
        {
            left.Add("XObject");
        }

        if (content.Fonts.Count > 0)
        {
            left.Add("Font");
        }

        if (content.Alphas.Count > 0)
        {
            left.Add("ExtGState");
        }

        for (int i = 0; i < dictionary.Count; i++)
        {
            var key = dictionary.KeyAt(i);
            string? category = left.Find(c => key.Is(System.Text.Encoding.ASCII.GetBytes(c)));
            if (category is null)
            {
                copier.Entry(s, source, key.Bytes, dictionary.ValueAt(i));
                continue;
            }

            left.Remove(category);
            s.Name(key.Bytes).Raw("<<"u8);
            PdfDictionary? own = null;
            try
            {
                own = source.File.Dictionary(dictionary.ValueAt(i));
            }
            catch (InvalidDataException e) when (!PdfLimit.Is(e))
            {
                // A category that cannot be read: the new content's alone.
            }

            for (int k = 0; k < (own?.Count ?? 0); k++)
            {
                copier.Entry(s, source, own!.KeyAt(k).Bytes, own.ValueAt(k));
            }

            AddedEntries(s, content, category, image, copier, source);
            s.Raw(">>"u8);
        }

        foreach (string category in left)
        {
            Added(s, content, category, image, copier, source);
        }

        s.Raw(">>"u8);
    }

    // A category of the new content's resources, whole ("/Font<<...>>"), when it names any.
    private static void Added(PdfSyntax s, PdfPageBuilder content, string category, Func<int, int> image, PdfEditCopier? copier = null, PdfEditCopier.Source? source = null)
    {
        int count = category switch { "XObject" => content.ImageNumbers.Count + content.Forms.Count, "Font" => content.Fonts.Count, _ => content.Alphas.Count };
        if (count == 0)
        {
            return;
        }

        s.Name(category).Raw("<<"u8);
        AddedEntries(s, content, category, image, copier, source);
        s.Raw(">>"u8);
    }

    // The new content's entries of a category, named as its content names them (a form of the page's document through
    // the copier, a new one by its number).
    private static void AddedEntries(PdfSyntax s, PdfPageBuilder content, string category, Func<int, int> image, PdfEditCopier? copier, PdfEditCopier.Source? source)
    {
        switch (category)
        {
            case "XObject":
                for (int i = 0; i < content.ImageNumbers.Count; i++)
                {
                    s.Raw("/"u8).Raw(content.Names).Raw("Im"u8).Integer(i + 1).Space().Reference(image(content.ImageNumbers[i]));
                }

                for (int i = 0; i < content.Forms.Count; i++)
                {
                    var (form, made) = content.Forms[i];
                    s.Raw("/"u8).Raw(content.Names).Raw("Fw"u8).Integer(i + 1).Space();
                    if (made is not null)
                    {
                        s.Reference(made());
                    }
                    else if (copier is not null && source is not null)
                    {
                        copier.Value(s, source, form);
                    }
                    else
                    {
                        s.Raw("null"u8);
                    }
                }

                break;
            case "Font":
                foreach (var font in content.Fonts)
                {
                    s.Raw("/"u8).Raw(content.Names).Raw("F"u8).Integer(font.Index).Space().Reference(font.Number);
                }

                break;
            default:
                for (int i = 0; i < content.Alphas.Count; i++)
                {
                    var (fill, stroke) = content.Alphas[i];
                    s.Raw("/"u8).Raw(content.Names).Raw("GS"u8).Integer(i + 1).Raw("<</ca "u8).Number(fill).Raw("/CA "u8).Number(stroke).Raw(">>"u8);
                }

                break;
        }
    }

    // Whether a resource name is one new content gives in its category ("Im", "F" or "GS" and digits).
    private static bool Generated(ReadOnlySpan<byte> name, string own)
    {
        if (name.Length <= own.Length)
        {
            return false;
        }

        for (int i = 0; i < own.Length; i++)
        {
            if (name[i] != own[i])
            {
                return false;
            }
        }

        foreach (byte b in name[own.Length..])
        {
            if (b is < (byte)'0' or > (byte)'9')
            {
                return false;
            }
        }

        return true;
    }
}
