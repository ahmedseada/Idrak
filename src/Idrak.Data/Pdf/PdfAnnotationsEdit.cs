// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Data.Pdf;

/// <summary>
/// The changes to one page an editor (or a builder copying it) writes with it: annotations added (their objects written
/// first, the page's <c>/Annots</c> naming them), annotations removed (by their place in its <c>/Annots</c>), and its
/// content redacted (a new content stream, its resources with the redacted images and forms in place of those they
/// replace). The page's dictionary is given back changed; the copier writes it, the result's new objects named through
/// references it writes as they are.
/// </summary>
internal sealed class PdfPageEdits
{
    // Numbers no source object has, each standing for one of the result's objects (a source copied, not referred to as it is).
    private static int s_stand = int.MaxValue;

    /// <summary>The annotations added.</summary>
    public List<PdfAnnotationDraft> Added { get; } = [];

    /// <summary>The annotations removed, by their place in the page's /Annots.</summary>
    public HashSet<int> Removed { get; } = [];

    /// <summary>A page builder whose annotations are added too (content drawn over the page, or a page the editor adds).</summary>
    public PdfPageBuilder? Content { get; set; }

    /// <summary>The page's content redacted, or null.</summary>
    public PdfRedactedPage? Redacted { get; set; }

    /// <summary>The page itself in the result (what its new annotations name as their page).</summary>
    public PdfPageTarget Self { get; set; }

    /// <summary>The result's pages, for destinations (none: a destination is written null).</summary>
    public Func<int, PdfPageTarget?>? Pages { get; set; }

    /// <summary>The result's object for an annotation another names by its id (its thread, its popup), besides those added here.</summary>
    public Func<string, (int Number, int Generation)?>? Annotations { get; set; }

    /// <summary>Writes an image the redaction made; its number.</summary>
    public Func<ImageData, int>? Images { get; set; }

    /// <summary>The number an image a new annotation's look names is written under (the editor's own numbering, resolved at save).</summary>
    public Func<int, int> ImageNumber { get; set; } = n => n;

    /// <summary>Whether anything changes.</summary>
    public bool Any => Added.Count > 0 || Removed.Count > 0 || Redacted is not null || Content?.Drafts.Count > 0;

    private IEnumerable<PdfAnnotationDraft> Drafts => Added.Concat(Content?.Drafts ?? []);

    /// <summary>
    /// <paramref name="page"/> (a source page dictionary; <paramref name="inherited"/> its inherited resources) with the
    /// edits: the new annotations and the redacted content's objects written now into <paramref name="file"/>, the
    /// dictionary given back holding references to them.
    /// </summary>
    public PdfDictionary Apply(PdfDictionary page, PdfValue inherited, PdfEditCopier copier, PdfEditCopier.Source source, PdfFileWriter file)
    {
        var keys = new List<PdfName>();
        var values = new List<PdfValue>();
        for (int i = 0; i < page.Count; i++)
        {
            keys.Add(page.KeyAt(i));
            values.Add(page.ValueAt(i));
        }

        void Set(ReadOnlySpan<byte> key, PdfValue value)
        {
            int at = -1;
            for (int k = 0; k < keys.Count && at < 0; k++)
            {
                at = keys[k].Is(key) ? k : -1;
            }

            if (at >= 0)
            {
                values[at] = value;
            }
            else
            {
                keys.Add(PdfName.From(key));
                values.Add(value);
            }
        }

        PdfValue Link(int number) => Stand(source, number);
        var removed = new HashSet<int>(Removed);
        if (Redacted is { } redacted)
        {
            removed.UnionWith(redacted.Annotations);
            var added = PdfRedactor.Write(redacted.Content, file, copier, source, Images ?? throw new InvalidOperationException("A redacted page is written with its images."), Link);
            int content = file.Allocate();
            file.BeginStream(content);
            file.EndStream(redacted.Content.Content, deflate: true);
            Set("Contents"u8, Link(content));
            var own = page.TryGet("Resources"u8, out var resources) ? resources : inherited;
            var dictionary = PdfRedactor.Resources(source.File, source.File.Dictionary(own), redacted.Content, added, Link);
            Set("Resources"u8, PdfValue.Of(dictionary!));
        }

        var drafts = Drafts.ToList();
        if (removed.Count > 0 || drafts.Count > 0)
        {
            var items = new List<PdfValue>();
            if (source.File.Resolve(page["Annots"u8]).AsArray is { } annots)
            {
                for (int k = 0; k < annots.Count; k++)
                {
                    if (!removed.Contains(k))
                    {
                        items.Add(annots[k]);
                    }
                }
            }

            foreach (int number in Write(drafts, file, Self))
            {
                items.Add(Link(number));
            }

            Set("Annots"u8, PdfValue.Of(new PdfArray([.. items])));
        }

        return new PdfDictionary([.. keys], [.. values]);
    }

    /// <summary>The new annotations of a page the editor adds, written now; their numbers, for its /Annots.</summary>
    public List<int> WriteNew(PdfFileWriter file) => Write([.. Drafts], file, Self);

    private List<int> Write(List<PdfAnnotationDraft> drafts, PdfFileWriter file, PdfPageTarget self)
    {
        var numbers = drafts.Select(_ => file.Allocate()).ToList();
        var byId = new Dictionary<string, (int, int)>(StringComparer.Ordinal);
        for (int i = 0; i < drafts.Count; i++)
        {
            if (drafts[i].Annotation.Id is { } id)
            {
                byId.TryAdd(id, (numbers[i], 0));
            }
        }

        (int Number, int Generation)? Find(string id) => byId.TryGetValue(id, out var known) ? known : Annotations?.Invoke(id);
        var pages = Pages ?? (_ => null);
        for (int i = 0; i < drafts.Count; i++)
        {
            drafts[i].Write(file, numbers[i], self, ImageNumber, pages, Find);
        }

        return numbers;
    }

    /// <summary>
    /// A value the copier writes as a reference to the result's object <paramref name="number"/>: the number itself when the
    /// source is referred to as it is (an update), else a number no source object has, mapped to it.
    /// </summary>
    public static PdfValue Stand(PdfEditCopier.Source source, int number)
    {
        if (source.Identity)
        {
            return PdfValue.Reference(number, 0);
        }

        int stand = Interlocked.Decrement(ref s_stand);
        source.Numbers[stand] = number;
        return PdfValue.Reference(stand, 0);
    }
}

/// <summary>The editor's annotations, bookmarks and redaction (contract C23).</summary>
internal sealed partial class PdfEditor
{
    private List<DocumentOutline>? _outlines;

    public void AddAnnotation(DocumentAnnotation annotation)
    {
        ArgumentNullException.ThrowIfNull(annotation);
        ThrowIfDone();
        CheckIndex(annotation.Page, nameof(annotation));
        var page = _pages[annotation.Page];
        var (frame, width, height) = PageFrame(page);
        (page.Edits ??= new PdfPageEdits()).Added.Add(PdfAnnotationDraft.Make(annotation, annotation.Page, width, height, frame, Font, WriteImage));
    }

    public void RemoveAnnotation(DocumentAnnotation annotation)
    {
        ArgumentNullException.ThrowIfNull(annotation);
        ThrowIfDone();
        CheckIndex(annotation.Page, nameof(annotation));
        var page = _pages[annotation.Page];
        if (annotation.Id is not { } id || page.Document is null)
        {
            throw new ArgumentException($"Annotation '{annotation.Id}' is not one page {annotation.Page + 1} holds: an annotation is removed by the Id it was read with.", nameof(annotation));
        }

        var file = page.Document.File;
        var node = page.Document.PageObject(page.Index).Node;
        var annots = file.Resolve(file.Dictionary(node)?["Annots"u8] ?? PdfValue.Null).AsArray;
        var edits = page.Edits ??= new PdfPageEdits();
        for (int k = 0; k < (annots?.Count ?? 0); k++)
        {
            if (file.Dictionary(annots![k]) is { } a && PdfAnnotations.Id(file, annots[k], a, page.Index, k) == id)
            {
                edits.Removed.Add(k);
                if (a["Popup"u8] is { IsReference: true } popup)
                {
                    for (int p = 0; p < annots.Count; p++)
                    {
                        if (annots[p].IsReference && annots[p].AsReference.Number == popup.AsReference.Number)
                        {
                            edits.Removed.Add(p);                             // its popup leaves with it
                        }
                    }
                }

                return;
            }
        }

        throw new ArgumentException($"Page {annotation.Page + 1} holds no annotation '{id}' (an annotation is removed by the Id it was read with).", nameof(annotation));
    }

    public void AddOutline(DocumentOutline outline)
    {
        ArgumentNullException.ThrowIfNull(outline);
        ThrowIfDone();
        (_outlines ??= []).Add(outline);
    }

    /// <summary>
    /// Redacts page <paramref name="index"/> now (<see cref="PdfRedactor"/>): its content read and written again without
    /// what lies under the areas, kept until the document is saved (as a rewrite only). A page redacted again is redacted
    /// from its source with every area given so far.
    /// </summary>
    public DocumentRedactionResult Redact(int index, IReadOnlyList<DocumentRedaction> areas)
    {
        ArgumentNullException.ThrowIfNull(areas);
        ThrowIfDone();
        CheckIndex(index, nameof(index));
        var page = _pages[index];
        if (page.Document is null)
        {
            throw new ArgumentException($"Page {index + 1} is one the editor adds: draw it without what is to be removed instead of redacting it.", nameof(index));
        }

        foreach (var area in areas)
        {
            ArgumentNullException.ThrowIfNull(area, nameof(areas));
            if (!float.IsFinite(area.Box.X) || !float.IsFinite(area.Box.Y) || !(area.Box.Width > 0) || !(area.Box.Height > 0) || !float.IsFinite(area.Box.Width)
                || !float.IsFinite(area.Box.Height))
            {
                throw new ArgumentException($"A redaction area has a finite place and a width and height above 0 ({area.Box}).", nameof(areas));
            }
        }

        var edits = page.Edits ??= new PdfPageEdits();
        IReadOnlyList<DocumentRedaction> all = edits.Redacted is { } earlier ? [.. earlier.Areas, .. areas] : [.. areas];
        var turned = page.Rotation is { } rotation && rotation != page.Document.Frame(page.Index).Rotation;
        if (turned)
        {
            throw new InvalidOperationException($"Page {index + 1} is turned in this edit: redact it before turning it (areas are read on the page as it is shown).");
        }

        edits.Redacted = PdfRedactor.Apply(page.Document, page.Index, all);
        return edits.Redacted.Result with { Page = index };
    }

    // Whether a page is redacted (a save then is a rewrite).
    private bool Redacting => _pages.Exists(p => p.Edits?.Redacted is not null);

    // A page's frame for content and annotations over it: its points to its user space, and its size as shown.
    private static (PdfMatrix? Frame, float Width, float Height) PageFrame(PdfEditPage page)
    {
        if (page.Document is null)
        {
            return (null, page.Content!.Width, page.Content.Height);
        }

        var (display, width, height, _) = page.Document.Frame(page.Index, page.Rotation);
        return (PdfEditPages.Inverse(display), width, height);
    }

    // Before the pages are written: each page's edits told the result's pages (destinations), its annotations' objects
    // (threads and popups naming one the source holds) and how images are written.
    private void Targets(int[] numbers, int[] generations, PdfEditCopier copier, PdfFileWriter writer)
    {
        var targets = new PdfPageTarget[_pages.Count];
        for (int i = 0; i < _pages.Count; i++)
        {
            var (frame, _, height) = PageFrame(_pages[i]);
            targets[i] = new PdfPageTarget(numbers[i], generations[i], frame ?? new PdfMatrix(1, 0, 0, -1, 0, height));
        }

        for (int t = 0; t < _pages.Count; t++)
        {
            var page = _pages[t];
            if (page.Edits is null && page.Content?.Drafts.Count is not > 0)
            {
                continue;
            }

            var edits = page.Edits ??= new PdfPageEdits();
            edits.Content = page.Content;
            edits.Self = targets[t];
            edits.Pages = i => (uint)i < (uint)targets.Length ? targets[i] : null;
            edits.ImageNumber = placeholder => ImageNumber(writer, placeholder);
            edits.Images = image =>
            {
                int number = writer.Allocate();
                PdfWriterImages.Write(writer, number, image, DocumentImageEncoding.Flate, _options);
                return number;
            };
            if (page.Document is { } document)
            {
                var source = copier.For(document);
                var file = document.File;
                edits.Annotations = id =>
                {
                    // An annotation of the source page by its id: its object, as the result names it.
                    var annots = file.Resolve(file.Dictionary(document.PageObject(page.Index).Node)?["Annots"u8] ?? PdfValue.Null).AsArray;
                    for (int k = 0; k < (annots?.Count ?? 0); k++)
                    {
                        if (annots![k].IsReference && file.Dictionary(annots[k]) is { } a && PdfAnnotations.Id(file, annots[k], a, page.Index, k) == id)
                        {
                            return copier.Reference(source, annots[k].AsReference);
                        }
                    }

                    return null;
                };
            }
        }
    }

    // The bookmarks: the source's, their pages followed to where they stand now (a page removed: no destination), and those
    // added; written as a new outline tree; its root's number, or 0 when none was added.
    private int WriteOutlines(PdfFileWriter writer, int[] numbers, int[] generations)
    {
        if (_outlines is null)
        {
            return 0;
        }

        var placed = new Dictionary<int, int>();
        for (int i = _pages.Count - 1; i >= 0; i--)
        {
            if (_pages[i].Document == _source)
            {
                placed[_pages[i].Index] = i;
            }
        }

        DocumentOutline Moved(DocumentOutline item) => item with
        {
            Destination = item.Destination is { Page: >= 0 } d ? placed.TryGetValue(d.Page, out int now) ? d with { Page = now } : null : item.Destination,
            Children = [.. item.Children.Select(Moved)],
        };

        var items = _source.Outline.Select(Moved).Concat(_outlines).ToList();
        var targets = new PdfPageTarget[_pages.Count];
        for (int i = 0; i < _pages.Count; i++)
        {
            var (frame, _, height) = PageFrame(_pages[i]);
            targets[i] = new PdfPageTarget(numbers[i], generations[i], frame ?? new PdfMatrix(1, 0, 0, -1, 0, height));
        }

        return PdfNavigationWriter.Outline(writer, items, i => (uint)i < (uint)targets.Length ? targets[i] : null);
    }
}
