// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Data.Pdf;

/// <summary>
/// An image a redaction rewrote: its pixels (one under an area takes the area's colour), its mask as grey with the pixels
/// under an area made opaque (null when it has none), and whether it is an image mask (painted in the fill colour where it
/// is dark; a pixel under an area left unpainted).
/// </summary>
internal sealed record PdfRedactedImage(ImageData Pixels, ImageData? Mask, bool Stencil);

/// <summary>
/// A content stream with redaction areas applied (<see cref="PdfRedactor"/>): its new bytes, the images, forms and graphics
/// states it draws in place of those it changed (new names in its resources), the patterns and Type 3 fonts its resources
/// name anew (the same names drawn from redacted copies), the names it no longer draws, and what was removed.
/// </summary>
internal sealed class PdfRedactedContent
{
    /// <summary>The content's bytes.</summary>
    public byte[] Content { get; set; } = [];

    /// <summary>Images whose pixels under an area were overwritten, under their new names.</summary>
    public List<(string Name, PdfRedactedImage Image)> Images { get; } = [];

    /// <summary>Forms whose content was redacted, under their new names: the form as it was (its dictionary kept) and its content redacted.</summary>
    public List<(string Name, PdfStream Original, PdfDictionary? Resources, PdfRedactedContent Content)> Forms { get; } = [];

    /// <summary>
    /// Tiling patterns whose cell was redacted where a copy of it lies under an area: the content's own name for each now
    /// names the copy (the pattern as it was, its resources, its cell redacted).
    /// </summary>
    public List<(string Name, PdfStream Original, PdfDictionary? Resources, PdfRedactedContent Content)> Patterns { get; } = [];

    /// <summary>
    /// Graphics states whose soft mask's group was redacted, under their new names: the state and its mask as they were,
    /// the group as it was, its resources and its content redacted.
    /// </summary>
    public List<(string Name, PdfDictionary State, PdfDictionary Mask, PdfStream Group, PdfDictionary? Resources, PdfRedactedContent Content)> States { get; } = [];

    /// <summary>Type 3 fonts the content shows with glyphs removed or drawn from redacted copies (<see cref="PdfType3Redaction"/>).</summary>
    public List<PdfType3Redaction> Fonts { get; } = [];

    /// <summary>XObject names the content no longer draws (each drawn only where it was replaced): left out of its resources.</summary>
    public HashSet<string> Dropped { get; } = new(StringComparer.Ordinal);

    /// <summary>Graphics state names the content no longer sets (each set only where it was replaced): left out of its resources.</summary>
    public HashSet<string> DroppedStates { get; } = new(StringComparer.Ordinal);

    /// <summary>Property lists named by spans whose text was cut: left out of its resources (their /ActualText could hold removed text).</summary>
    public HashSet<string> DroppedProperties { get; } = new(StringComparer.Ordinal);

    public int Glyphs { get; set; }

    public int Characters { get; set; }

    public int PathsRemoved { get; set; }

    public int PathsClipped { get; set; }

    public int ImagesChanged { get; set; }

    public long Pixels { get; set; }

    /// <summary>Whether anything was changed in it.</summary>
    public bool Changed => Glyphs > 0 || PathsRemoved > 0 || PathsClipped > 0 || Pixels > 0 || Forms.Count > 0 || Patterns.Count > 0 || States.Count > 0 || Fonts.Count > 0;

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

/// <summary>
/// A page with redaction areas applied: its content redacted, the annotations removed (by their place in its /Annots, and
/// by their object numbers, for what else names them), the areas and what was removed.
/// </summary>
internal sealed class PdfRedactedPage(PdfRedactedContent content, HashSet<int> annotations, HashSet<int> objects, IReadOnlyList<DocumentRedaction> areas, DocumentRedactionResult result)
{
    public PdfRedactedContent Content { get; } = content;

    public HashSet<int> Annotations { get; } = annotations;

    /// <summary>The removed annotations' object numbers (with the popups they name): a rewrite leaves out every mention of them (<see cref="PdfRedactor.Reach"/>).</summary>
    public HashSet<int> Objects { get; } = objects;

    public IReadOnlyList<DocumentRedaction> Areas { get; } = areas;

    public DocumentRedactionResult Result { get; } = result;
}

/// <summary>What writing a redacted content's objects made, by the resource category each goes in: its name and object number.</summary>
internal sealed class PdfRedactedObjects
{
    /// <summary>Images and forms, under new names.</summary>
    public List<(string Name, int Number)> XObjects { get; } = [];

    /// <summary>Tiling patterns' copies, each under the name of the pattern it stands for.</summary>
    public List<(string Name, int Number)> Patterns { get; } = [];

    /// <summary>Graphics states, under new names.</summary>
    public List<(string Name, int Number)> States { get; } = [];

    /// <summary>Type 3 fonts: a copy under its font's name, or the redacted glyphs' own font under a new name.</summary>
    public List<(string Name, int Number)> Fonts { get; } = [];
}

/// <summary>
/// New names for a content's resources (RdIm1, RdFm2, RdGs3, ...): one count for every category, each name one its category
/// (in <paramref name="resources"/>, as written) does not hold yet. Shared by contents that end up in one dictionary.
/// </summary>
internal sealed class PdfNames(PdfDictionary? resources)
{
    private int _count;

    /// <summary>A name of <paramref name="stem"/> and the next count that <paramref name="category"/> does not hold.</summary>
    public string Next(string stem, ReadOnlySpan<byte> category)
    {
        var existing = resources?[category].AsDictionary;
        while (true)
        {
            string name = stem + (++_count).ToString(CultureInfo.InvariantCulture);
            if (existing is null || !existing.Contains(System.Text.Encoding.ASCII.GetBytes(name)))
            {
                return name;
            }
        }
    }
}

/// <summary>
/// Redaction applied for real (contract C23's <see cref="IDocumentEditor.Redact"/>): a page's content interpreted as it is
/// read and written again operator by operator. A text-showing operator keeps the glyphs whose boxes do not overlap an
/// area (at least a hundredth of the glyph each way) and moves past the removed ones by their advances (<c>TJ</c>
/// offsets), so what stays stands where it stood; a span's <c>/ActualText</c> over removed glyphs is dropped. A path inside
/// an area is not painted (a clip it makes is kept); one crossing an area is painted under a clip that leaves out each area
/// it crosses (even-odd: the page around the area); a shading too. An image (an XObject or inline) under an area is drawn
/// from a copy whose pixels with their centre inside an area take the area's fill (black when it has none), its mask kept
/// with those pixels made opaque, an image mask left unpainted there; a form under an area is drawn from a copy of its
/// content redacted the same way, and so is a soft mask's group, a tiling pattern's cell where a copy of it lies under an
/// area (<see cref="PdfContentReader.FinishRedaction"/>), and a Type 3 glyph whose drawing reaches an area. The
/// annotations over an area are removed with their popups and their replies, every mention of them left out of the
/// rewrite (<see cref="Reach"/>), and each area with a fill is filled after the page's content. Hidden text (a scan's text
/// layer) is text like any other.
/// </summary>
internal static class PdfRedactor
{
    private static readonly HashSet<string> None = new(StringComparer.Ordinal);

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
            reader.FinishRedaction(rewriter, resources);
            content.Content = rewriter.Finish(Marks(areas, display));
        }

        // The annotations over an area (a popup leaves with its annotation, not on its own).
        var removed = new HashSet<int>();
        var annots = file.Resolve(page["Annots"u8]).AsArray;
        int count = annots?.Count ?? 0;
        for (int k = 0; k < count; k++)
        {
            var a = file.Dictionary(annots![k]);
            if (a is null || file.Resolve(a["Subtype"u8]).IsName("Popup"u8))
            {
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

        // What leaves with them on this page: their popups (each one's /Popup, a popup's /Parent) and their replies (an /IRT
        // naming one, at any depth); what names them elsewhere in the document leaves when it is saved (Reach).
        var objects = new HashSet<int>();
        void Named(int k)
        {
            if (annots![k].IsReference)
            {
                objects.Add(annots[k].AsReference.Number);
            }

            if (file.Dictionary(annots[k])?["Popup"u8] is { IsReference: true } popup)
            {
                objects.Add(popup.AsReference.Number);
            }
        }

        foreach (int k in removed)
        {
            Named(k);
        }

        for (bool grew = objects.Count > 0; grew;)
        {
            grew = false;
            for (int k = 0; k < count; k++)
            {
                if (removed.Contains(k) || file.Dictionary(annots![k]) is not { } a)
                {
                    continue;
                }

                var named = file.Resolve(a["Subtype"u8]).IsName("Popup"u8) ? a["Parent"u8] : a["IRT"u8];
                if ((annots[k].IsReference && objects.Contains(annots[k].AsReference.Number)) || (named.IsReference && objects.Contains(named.AsReference.Number)))
                {
                    removed.Add(k);
                    Named(k);
                    grew = true;
                }
            }
        }

        var result = new DocumentRedactionResult(index)
        {
            Areas = areas.Count, Glyphs = content.Glyphs, Characters = content.Characters, PathsRemoved = content.PathsRemoved, PathsClipped = content.PathsClipped,
            Images = content.ImagesChanged, Pixels = content.Pixels, Annotations = removed.Count,
        };
        return new PdfRedactedPage(content, removed, objects, areas, result);
    }

    /// <summary>
    /// What a rewrite of <paramref name="document"/> leaves out with the annotations redaction removed (<paramref name="seeds"/>,
    /// by object number): every reply (an /IRT naming one, at any depth) and popup (a /Popup of one, a popup whose /Parent is
    /// one) on any page; a form field left with no kid, and so up its parents; the structure tree's object references to
    /// them. Each is removed with every mention of it (<see cref="PdfEditCopier.Source.Removed"/>). Also the structure
    /// elements left with no kid (each held only such references): written again without their alternate, actual and
    /// expanded texts and their title (<see cref="Emptied"/>), which could describe what was removed.
    /// </summary>
    public static (HashSet<int> Removed, List<int> Emptied) Reach(PdfDocument document, IEnumerable<int> seeds)
    {
        var file = document.File;
        var removed = new HashSet<int>(seeds);
        if (removed.Count == 0)
        {
            return (removed, []);
        }

        // Every annotation of every page, by what follows it out: its replies (by their /IRT), its popups (by their
        // /Parent, and its own /Popup).
        var followers = new Dictionary<int, List<int>>();
        void Follow(int leader, int follower) => (followers.TryGetValue(leader, out var list) ? list : followers[leader] = []).Add(follower);
        for (int i = 0; i < document.PageCount; i++)
        {
            PdfArray? annots;
            try
            {
                annots = file.Resolve(file.Dictionary(document.PageObject(i).Node)?["Annots"u8] ?? PdfValue.Null).AsArray;
            }
            catch (InvalidDataException e) when (!PdfLimit.Is(e))
            {
                continue;                                                      // a damaged list: read as none, as a reader takes it
            }

            foreach (var entry in annots?.Items ?? [])
            {
                PdfDictionary? a;
                try
                {
                    a = entry.IsReference ? file.Dictionary(entry) : null;
                }
                catch (InvalidDataException e) when (!PdfLimit.Is(e))
                {
                    continue;
                }

                if (a is null)
                {
                    continue;
                }

                int number = entry.AsReference.Number;
                if (a["IRT"u8] is { IsReference: true } irt)
                {
                    Follow(irt.AsReference.Number, number);
                }

                if (a["Popup"u8] is { IsReference: true } popup)
                {
                    Follow(number, popup.AsReference.Number);
                }

                if (a["Parent"u8] is { IsReference: true } parent && file.Resolve(a["Subtype"u8]).IsName("Popup"u8))
                {
                    Follow(parent.AsReference.Number, number);
                }
            }
        }

        var queue = new Queue<int>(removed);
        while (queue.TryDequeue(out int leader))
        {
            foreach (int follower in followers.TryGetValue(leader, out var list) ? list : [])
            {
                if (removed.Add(follower))
                {
                    queue.Enqueue(follower);
                }
            }
        }

        // A form field whose every kid is gone (its widgets redacted) goes too, its value with it; and so up its parents.
        queue = new Queue<int>(removed);
        while (queue.TryDequeue(out int number))
        {
            var parent = Object(file, number)?["Parent"u8] ?? PdfValue.Null;
            if (!parent.IsReference || removed.Contains(parent.AsReference.Number) || Object(file, number)?["Subtype"u8] is { } subtype && file.Resolve(subtype).IsName("Popup"u8))
            {
                continue;
            }

            int field = parent.AsReference.Number;
            if (file.Resolve(Object(file, field)?["Kids"u8] ?? PdfValue.Null).AsArray is { Count: > 0 } kids && kids.Items.All(k => k.IsReference && removed.Contains(k.AsReference.Number)))
            {
                removed.Add(field);
                queue.Enqueue(field);
            }
        }

        // The structure tree: an object reference (/OBJR, its own object) to one is removed; an element whose every kid is
        // gone is emptied.
        var emptied = new List<int>();
        if (file.Dictionary(file.Trailer["Root"u8]) is { } catalog && file.Dictionary(catalog["StructTreeRoot"u8]) is { } tree)
        {
            var elements = new List<(int Number, PdfValue Kids)>();
            var seen = new HashSet<int>();
            var stack = new Stack<PdfValue>();
            stack.Push(tree["K"u8]);
            while (stack.TryPop(out var value))
            {
                int number = value.IsReference ? value.AsReference.Number : -1;
                if (number >= 0)
                {
                    if (!seen.Add(number))
                    {
                        continue;                                              // met before (a cycle ends here)
                    }

                    try
                    {
                        value = file.Get(number);
                    }
                    catch (InvalidDataException e) when (!PdfLimit.Is(e))
                    {
                        continue;
                    }
                }

                if (value.Kind == PdfKind.Array)
                {
                    foreach (var item in value.AsArray!.Items)
                    {
                        stack.Push(item);
                    }

                    continue;
                }

                var element = value.AsDictionary;
                if (element is null)
                {
                    continue;
                }

                if (element["Obj"u8] is { IsReference: true } target)
                {
                    if (number >= 0 && removed.Contains(target.AsReference.Number))
                    {
                        removed.Add(number);                                   // an object reference of its own: gone with what it names
                    }

                    continue;
                }

                if (number >= 0 && !element["K"u8].IsNull)
                {
                    elements.Add((number, element["K"u8]));
                }

                stack.Push(element["K"u8]);
            }

            foreach (var (number, kids) in elements)
            {
                var items = kids.Kind == PdfKind.Array ? kids.AsArray!.Items : [kids];
                if (items.Length > 0 && items.All(k => Gone(file, removed, k)))
                {
                    emptied.Add(number);
                }
            }
        }

        return (removed, emptied);
    }

    // An object by number, when it is a dictionary (or a stream's); null when it is not, or cannot be read.
    private static PdfDictionary? Object(PdfFile file, int number)
    {
        try
        {
            return file.Get(number).AsDictionary;
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            return null;
        }
    }

    // Whether a structure element's kid is gone: a removed object, or an object reference (its own or written in) to one.
    private static bool Gone(PdfFile file, HashSet<int> removed, PdfValue kid) =>
        kid.IsReference ? removed.Contains(kid.AsReference.Number) : kid.AsDictionary?["Obj"u8] is { IsReference: true } target && removed.Contains(target.AsReference.Number);

    /// <summary>A structure element <see cref="Reach"/> emptied, written again: its kids that are left, without its alternate, actual and expanded texts and its title.</summary>
    public static void Emptied(PdfSyntax s, PdfEditCopier copier, PdfEditCopier.Source source, int number)
    {
        var element = Object(source.File, number) ?? PdfDictionary.Empty;
        s.Raw("<<"u8);
        for (int i = 0; i < element.Count; i++)
        {
            var key = element.KeyAt(i);
            if (!(key.Is("Alt"u8) || key.Is("ActualText"u8) || key.Is("E"u8) || key.Is("T"u8)))
            {
                copier.Entry(s, source, key.Bytes, element.ValueAt(i));
            }
        }

        s.Raw(">>"u8);
    }

    /// <summary>Whether a box (two corners, page points) and an area share some of the plane.</summary>
    public static bool Overlap(DocumentBox area, double x0, double y0, double x1, double y1) =>
        Math.Min(x1, area.Right) > Math.Max(x0, area.X) && Math.Min(y1, area.Bottom) > Math.Max(y0, area.Y);

    /// <summary>
    /// Writes a redacted content's images, forms, patterns, graphics states and Type 3 fonts into the result (not the
    /// content itself); the names they go under in its resources and the object numbers they are written under
    /// (<paramref name="images"/> writes an image, <paramref name="link"/> makes a value of the result's reference that the
    /// copier writes as it is).
    /// </summary>
    public static PdfRedactedObjects Write(PdfRedactedContent content, PdfFileWriter file, PdfEditCopier copier, PdfEditCopier.Source source,
        Func<PdfRedactedImage, int> images, Func<int, PdfValue> link)
    {
        var written = new PdfRedactedObjects();
        foreach (var (name, image) in content.Images)
        {
            written.XObjects.Add((name, images(image)));
        }

        foreach (var (name, original, resources, inner) in content.Forms)
        {
            written.XObjects.Add((name, Stream(original.Dictionary, resources, inner, file, copier, source, images, link)));
        }

        foreach (var (name, original, resources, inner) in content.Patterns)
        {
            written.Patterns.Add((name, Stream(original.Dictionary, resources, inner, file, copier, source, images, link)));
        }

        foreach (var (name, state, mask, group, resources, inner) in content.States)
        {
            int redacted = Stream(group.Dictionary, resources, inner, file, copier, source, images, link);
            int number = file.Allocate();
            copier.WriteObject(number, s =>
            {
                // The state as it was, its soft mask's group the redacted copy.
                s.Raw("<<"u8);
                for (int i = 0; i < state.Count; i++)
                {
                    if (!state.KeyAt(i).Is("SMask"u8))
                    {
                        copier.Entry(s, source, state.KeyAt(i).Bytes, state.ValueAt(i));
                    }
                }

                s.Raw("/SMask<<"u8);
                for (int i = 0; i < mask.Count; i++)
                {
                    if (!mask.KeyAt(i).Is("G"u8))
                    {
                        copier.Entry(s, source, mask.KeyAt(i).Bytes, mask.ValueAt(i));
                    }
                }

                copier.Entry(s, source, "G"u8, link(redacted));
                s.Raw(">>>>"u8);
            });
            written.States.Add((name, number));
        }

        foreach (var font in content.Fonts)
        {
            font.Write(file, copier, source, images, link, written);
        }

        return written;
    }

    /// <summary>
    /// A content stream's redacted copy written as a new object: its dictionary as it was (but its length, filters and
    /// resources), <paramref name="resources"/> with the content's changes, the redacted content; its number.
    /// </summary>
    public static int Stream(PdfDictionary dictionary, PdfDictionary? resources, PdfRedactedContent inner, PdfFileWriter file, PdfEditCopier copier, PdfEditCopier.Source source,
        Func<PdfRedactedImage, int> images, Func<int, PdfValue> link)
    {
        var innerWritten = Write(inner, file, copier, source, images, link);
        int number = file.Allocate();
        var redactedResources = Resources(source.File, resources, inner, innerWritten, link);
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

            copier.Entry(s, source, "Resources"u8, PdfValue.Of(redactedResources));
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
        return number;
    }

    /// <summary>
    /// <paramref name="resources"/> with a redacted content's changes: the names it dropped left out of <c>/XObject</c>,
    /// <c>/ExtGState</c> and <c>/Properties</c>, the names it added (<paramref name="written"/>) put in, the patterns and
    /// fonts it names anew drawn from their copies.
    /// </summary>
    public static PdfDictionary Resources(PdfFile file, PdfDictionary? resources, PdfRedactedContent content, PdfRedactedObjects written, Func<int, PdfValue> link)
    {
        resources ??= PdfDictionary.Empty;
        var keys = new List<PdfName>();
        var values = new List<PdfValue>();
        bool xobjects = false, states = false, fonts = false;
        for (int i = 0; i < resources.Count; i++)
        {
            var key = resources.KeyAt(i);
            var value = resources.ValueAt(i);
            if (key.Is("XObject"u8))
            {
                value = PdfValue.Of(Category(file.Dictionary(value), content.Dropped, written.XObjects, link));
                xobjects = true;
            }
            else if (key.Is("Properties"u8) && content.DroppedProperties.Count > 0)
            {
                value = PdfValue.Of(Category(file.Dictionary(value), content.DroppedProperties, [], link));
            }
            else if (key.Is("Pattern"u8) && written.Patterns.Count > 0)
            {
                value = PdfValue.Of(Category(file.Dictionary(value), None, written.Patterns, link));
            }
            else if (key.Is("ExtGState"u8) && (content.DroppedStates.Count > 0 || written.States.Count > 0))
            {
                value = PdfValue.Of(Category(file.Dictionary(value), content.DroppedStates, written.States, link));
                states = true;
            }
            else if (key.Is("Font"u8) && written.Fonts.Count > 0)
            {
                value = PdfValue.Of(Category(file.Dictionary(value), None, written.Fonts, link));
                fonts = true;
            }

            keys.Add(key);
            values.Add(value);
        }

        if (!xobjects && written.XObjects.Count > 0)
        {
            keys.Add(PdfName.From("XObject"u8));
            values.Add(PdfValue.Of(Category(null, content.Dropped, written.XObjects, link)));
        }

        if (!states && written.States.Count > 0)
        {
            keys.Add(PdfName.From("ExtGState"u8));
            values.Add(PdfValue.Of(Category(null, None, written.States, link)));
        }

        if (!fonts && written.Fonts.Count > 0)
        {
            keys.Add(PdfName.From("Font"u8));
            values.Add(PdfValue.Of(Category(null, None, written.Fonts, link)));
        }

        return new PdfDictionary([.. keys], [.. values]);
    }

    // One category of resources (as written: a dictionary, or a reference resolved by the caller) less the dropped names,
    // a name the written ones hold drawn from its copy, then the written names it does not hold.
    private static PdfDictionary Category(PdfDictionary? category, HashSet<string> dropped, List<(string Name, int Number)> written, Func<int, PdfValue> link)
    {
        var keys = new List<PdfName>();
        var values = new List<PdfValue>();
        var held = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < (category?.Count ?? 0); i++)
        {
            string key = category!.KeyAt(i).ToString();
            if (dropped.Contains(key))
            {
                continue;
            }

            int at = written.FindIndex(w => string.Equals(w.Name, key, StringComparison.Ordinal));
            keys.Add(category.KeyAt(i));
            values.Add(at >= 0 ? link(written[at].Number) : category.ValueAt(i));
            held.Add(key);
        }

        foreach (var (name, number) in written)
        {
            if (!held.Contains(name))
            {
                keys.Add(PdfName.From(System.Text.Encoding.ASCII.GetBytes(name)));
                values.Add(link(number));
            }
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
/// painted, as the areas decide), the names it changes (new ones from <paramref name="names"/> when contents share a
/// dictionary), the tiling patterns it paints with and the Type 3 fonts it shows.
/// </summary>
internal sealed class PdfRewriter(DocumentBox[] areas, DocumentColour[] fills, float width, float height, PdfDictionary? resources, PdfRedactedContent content, int depth,
    PdfNames? names = null) : IDisposable
{
    private readonly PdfSyntax _out = new(4096);
    private readonly PdfSyntax _path = new(256);
    private readonly HashSet<string> _kept = new(StringComparer.Ordinal), _replaced = new(StringComparer.Ordinal);
    private readonly HashSet<string> _keptStates = new(StringComparer.Ordinal), _replacedStates = new(StringComparer.Ordinal);
    private PdfNames? _names;
    private PdfSyntax? _span;
    private byte[]? _spanStart;
    private string? _spanProperties;
    private bool _spanCut;

    /// <summary>Each tiling pattern the content paints with, by its name: where, and whether a paint reached an area.</summary>
    public Dictionary<string, PdfPatternUses> Patterns { get; } = new(StringComparer.Ordinal);

    /// <summary>Each font the content shows text in, by its name: how a Type 3 font's glyphs are shown (null: not a Type 3 font).</summary>
    public Dictionary<string, PdfType3Redaction?> Type3 { get; } = new(StringComparer.Ordinal);

    /// <summary>Whether the content draws a form that has no resources of its own (so takes the content's: a Type 3 font's glyphs stay whole for it).</summary>
    public bool Inherits { get; set; }

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

    /// <summary>A name not yet in the resources' XObjects (or given here): RdIm1, RdFm2, ...</summary>
    public string Name(string stem) => Names.Next(stem, "XObject"u8);

    /// <summary>A name not yet in the resources' <paramref name="category"/> (or given here): RdGs1, RdFt2, ...</summary>
    public string Name(string stem, ReadOnlySpan<byte> category) => Names.Next(stem, category);

    // The names it makes: shared when given, else its own over its resources.
    private PdfNames Names => names ?? (_names ??= new PdfNames(resources));

    /// <summary>An XObject drawn as it was (its name stays in the resources).</summary>
    public void Kept(string name) => _kept.Add(name);

    /// <summary>An XObject drawn from a redacted copy here.</summary>
    public void Replaced(string name) => _replaced.Add(name);

    /// <summary>A graphics state set as it was (its name stays in the resources).</summary>
    public void KeptState(string name) => _keptStates.Add(name);

    /// <summary>A graphics state set from a redacted copy here.</summary>
    public void ReplacedState(string name) => _replacedStates.Add(name);

    /// <summary>
    /// The tiling pattern <paramref name="name"/> painted within a box of the page (two corners, page points) in the
    /// pattern space of <paramref name="space"/>; <paramref name="touched"/> when the paint reached an area (removed or
    /// clipped), <paramref name="covers"/> when it covers its box (a path; text's glyphs cover only part of theirs).
    /// </summary>
    public void UsePattern(string name, PdfMatrix space, double x0, double y0, double x1, double y1, bool touched, bool covers)
    {
        if (!Patterns.TryGetValue(name, out var uses))
        {
            Patterns[name] = uses = new PdfPatternUses(space);
        }

        uses.Add(x0, y0, x1, y1, touched, covers);
    }

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

    /// <summary>
    /// The content's bytes: in a saved graphics state, then <paramref name="after"/> (the areas' fills); as they are when not
    /// <paramref name="wrap"/> (a Type 3 glyph's, whose first operator is its d0 or d1).
    /// </summary>
    public byte[] Finish(ReadOnlySpan<byte> after, bool wrap = true)
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

        foreach (string name in _replacedStates)
        {
            if (!_keptStates.Contains(name))
            {
                content.DroppedStates.Add(name);
            }
        }

        if (!wrap)
        {
            return [.. _out.Span, .. after];
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
                ShowRedacted(w, op, data, bytes, first, resources);
                return;
            }

            case OpState when _operandCount >= 1 && _operands[_operandCount - 1].Kind == PdfToken.Name:
            {
                var ctm = _state.Ctm;
                var operand = _operands[_operandCount - 1];
                Execute(op, data, resources, 0);
                string name = NameText(data, operand);
                if (RedactSoftMask(w, data.Slice(operand.Start, operand.Length), name, resources, ctm))
                {
                    return;
                }

                w.Copy(bytes);
                w.KeptState(name);
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

        bool fills = op is not (OpStroke or OpCloseStroke), strokes = op is OpStroke or OpCloseStroke or OpFillStroke or OpFillStrokeEvenOdd or OpCloseFillStroke or OpCloseFillStrokeEvenOdd;
        UsePatterns(w, fills, strokes, minX, minY, maxX, maxY, inside || crossed.Count > 0, covers: true);
    }

    // A text-showing operator: written as it was when none of its glyphs lies under an area, else as a TJ of the glyphs
    // kept with the removed ones' advances as offsets (', " written as the moves and spacing they set, then the TJ). A
    // Type 3 glyph kept whose drawing reaches an area is shown from its redacted copy in a font of its own (Type3Glyph):
    // the TJ ends, that font shows it, and the font shown before takes up again.
    private void ShowRedacted(PdfRewriter w, int op, ReadOnlySpan<byte> data, ReadOnlySpan<byte> bytes, int first, PdfDictionary? resources)
    {
        int count = _glyphCount - first;
        Span<double> xy = stackalloc double[8];
        bool[]? removed = null;
        double bx0 = double.MaxValue, by0 = double.MaxValue, bx1 = double.MinValue, by1 = double.MinValue;
        for (int g = 0; g < count; g++)
        {
            Corners(_glyphs[first + g], xy);
            double x0 = Math.Min(Math.Min(xy[0], xy[2]), Math.Min(xy[4], xy[6])), x1 = Math.Max(Math.Max(xy[0], xy[2]), Math.Max(xy[4], xy[6]));
            double y0 = Math.Min(Math.Min(xy[1], xy[3]), Math.Min(xy[5], xy[7])), y1 = Math.Max(Math.Max(xy[1], xy[3]), Math.Max(xy[5], xy[7]));
            (bx0, by0, bx1, by1) = (Math.Min(bx0, x0), Math.Min(by0, y0), Math.Max(bx1, x1), Math.Max(by1, y1));
            foreach (var area in w.Areas)
            {
                if (Covers(area, x0, x1, area.X, area.Right) && Covers(area, y0, y1, area.Y, area.Bottom))
                {
                    (removed ??= new bool[count])[g] = true;
                    break;
                }
            }
        }

        if (count > 0 && (_state.FillPattern is not null || _state.StrokePattern is not null))
        {
            int mode = _state.Mode & 3;
            UsePatterns(w, mode is 0 or 2, mode is 1 or 2, bx0, by0, bx1, by1, removed is not null || w.Areas.Any(a => PdfRedactor.Overlap(a, bx0, by0, bx1, by1)), covers: false);
        }

        // A Type 3 font's glyphs: each kept one drawn from a redacted copy where its drawing reaches an area.
        int[]? swapped = null;
        var type3 = count > 0 ? Type3(w, resources) : null;
        if (type3 is not null)
        {
            for (int g = 0; g < count; g++)
            {
                ref readonly var shown = ref _glyphs[first + g];
                int code = (int)(shown.Code & 0xFF);
                if (removed?[g] == true)
                {
                    type3.Gone[code]++;
                }
                else if (Type3Glyph(w, type3, shown) is >= 0 and var copy)
                {
                    (swapped ??= Enumerable.Repeat(-1, count).ToArray())[g] = copy;
                    type3.Gone[code]++;
                }
                else if (type3.Redacted.Count >= 256 && Type3Reaches(w, type3, shown))
                {
                    (removed ??= new bool[count])[g] = true;                  // no code left for a copy: the glyph leaves whole
                    type3.Gone[code]++;
                }
                else
                {
                    type3.Kept[code]++;
                }
            }
        }

        if (removed is null && swapped is null)
        {
            w.Copy(bytes);
            return;
        }

        removed ??= new bool[count];

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
                else if (swapped?[glyph] is >= 0 and var copy)
                {
                    // The glyph's redacted copy, shown by its own font; then the font before it again.
                    if (hexOpen)
                    {
                        s.Byte((byte)'>');
                        hexOpen = false;
                    }

                    s.Raw("]TJ\n"u8).Name(type3!.Own!).Space().Number(size, 6).Raw(" Tf <"u8);
                    s.Byte("0123456789ABCDEF"u8[copy >> 4]).Byte("0123456789ABCDEF"u8[copy & 15]).Raw(">Tj\n"u8);
                    s.Name(type3.Key.Bytes).Space().Number(size, 6).Raw(" Tf\n["u8);
                    if (font.IsWordSpace(code, length) && _state.WordSpacing != 0 && size != 0)
                    {
                        s.Space().Number(-_state.WordSpacing * 1000 / size, 4).Space();   // the word spacing the copy's code did not take
                    }
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
    // take its fill (an image mask's are left unpainted), its mask kept with its own pixels under an area made opaque;
    // false when no pixel of it or its mask lies inside an area (or it cannot be decoded: drawn as it was).
    private bool RedactImage(PdfRewriter w, DocumentImage image, string? name)
    {
        var m = image.Placement;
        int width = image.PixelWidth, height = image.PixelHeight;
        double det = (double)m.A * m.D - (double)m.B * m.C;
        if (width <= 0 || height <= 0 || det == 0 || !double.IsFinite(det))
        {
            return false;
        }

        var hits = Hits(w, m, det, width, height);
        if (hits.Count == 0)
        {
            return false;
        }

        bool stencil = image.Stencil is not null;
        float[]? pixels = null;
        int channels = 0;
        long changed = 0;
        System.Collections.BitArray? counted = null;                           // under several areas: each pixel counted once
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
                    if (stencil)
                    {
                        pixels[at] = 1f;                                       // an image mask: not painted here
                    }
                    else if (channels == 1)
                    {
                        pixels[at] = fill.Luma;
                    }
                    else
                    {
                        (pixels[at], pixels[plane + at], pixels[2 * plane + at]) = (fill.Red, fill.Green, fill.Blue);
                    }

                    if (hits.Count > 1)
                    {
                        counted ??= new System.Collections.BitArray(plane);
                        if (counted[at])
                        {
                            continue;
                        }

                        counted[at] = true;
                    }

                    changed++;
                }
            }
        }

        // Its mask (a soft mask, an explicit mask or a colour key, as grey): kept, its pixels under an area opaque.
        ImageData? mask = null;
        long masked = 0;
        if (!stencil)
        {
            try
            {
                mask = image.Mask();
            }
            catch (Exception e) when (e is NotSupportedException or InvalidDataException)
            {
                _issues.Add(DocumentIssue.NotSupported, $"page {_page}: the mask of an image under a redaction area cannot be decoded, so the image is drawn without it: {e.Message}");
            }

            if (mask is not null)
            {
                (mask, masked) = RedactMask(w, mask, m, det);
            }
        }

        if (pixels is null && masked == 0)
        {
            return false;
        }

        if (pixels is null)
        {
            try
            {
                var decoded = image.Decode();
                (pixels, channels) = (decoded.Pixels, decoded.Channels);
            }
            catch (Exception e) when (e is NotSupportedException or InvalidDataException)
            {
                _issues.Add(DocumentIssue.NotSupported, $"page {_page}: an image under a redaction area cannot be decoded, so its mask is not overwritten: {e.Message}");
                return false;
            }
        }

        string own = w.Name("RdIm");
        w.Result.Images.Add((own, new PdfRedactedImage(new ImageData(pixels, channels, height, width), mask, stencil)));
        w.Result.ImagesChanged++;
        w.Result.Pixels += changed;
        if (name is not null)
        {
            w.Replaced(name);
        }

        w.Write().Byte((byte)'/').Raw(System.Text.Encoding.ASCII.GetBytes(own)).Raw(" Do\n"u8);
        return true;
    }

    // Each area's range of pixels of an image of `width` x `height` placed by `m` (determinant `det`), found through the
    // inverse map with a pixel to spare each way; a pixel's centre is then tested exactly.
    private static List<(int Area, int I0, int I1, int J0, int J1)> Hits(PdfRewriter w, DocumentMatrix m, double det, int width, int height)
    {
        var hits = new List<(int Area, int I0, int I1, int J0, int J1)>();
        Span<double> corners = stackalloc double[8];
        for (int a = 0; a < w.Areas.Length; a++)
        {
            var area = w.Areas[a];
            (corners[0], corners[1], corners[2], corners[3], corners[4], corners[5], corners[6], corners[7]) = (area.X, area.Y, area.Right, area.Y, area.Right, area.Bottom, area.X, area.Bottom);
            double u0 = double.MaxValue, u1 = double.MinValue, v0 = double.MaxValue, v1 = double.MinValue;
            for (int k = 0; k < 8; k += 2)
            {
                double dx = corners[k] - m.E, dy = corners[k + 1] - m.F;
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

        return hits;
    }

    // An image's mask (grey, at its own size, placed as the image is by `m`) with each pixel whose centre is inside an area
    // made opaque, so a redacted pixel shows its fill whole; the copy and the count changed (the mask itself when none).
    private static (ImageData Mask, long Changed) RedactMask(PdfRewriter w, ImageData mask, DocumentMatrix m, double det)
    {
        int width = mask.Width, height = mask.Height;
        float[]? values = null;
        long changed = 0;
        foreach (var (a, i0, i1, j0, j1) in Hits(w, m, det, width, height))
        {
            var area = w.Areas[a];
            for (int j = j0; j <= j1; j++)
            {
                for (int i = i0; i <= i1; i++)
                {
                    double u = (i + 0.5) / width, v = (j + 0.5) / height;
                    double x = m.A * u + m.C * v + m.E, y = m.B * u + m.D * v + m.F;
                    if (x > area.X && x < area.Right && y > area.Y && y < area.Bottom)
                    {
                        values ??= mask.Channels == 1 ? (float[])mask.Pixels.Clone() : mask.Pixels[..(width * height)];
                        values[j * width + i] = 1f;
                        changed++;
                    }
                }
            }
        }

        return values is null ? (mask, 0) : (new ImageData(values, 1, height, width), changed);
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

        w.Inherits |= stream.Dictionary["Resources"u8].IsNull;                // it takes the content's resources
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

        var formResources = _file.Dictionary(stream.Dictionary["Resources"u8]) ?? resources;
        var inner = RedactNested(w, stream, formResources, placed, box, w.Areas, w.AreaFills, bare: false);
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
