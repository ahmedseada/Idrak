// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Data.Pdf;

/// <summary>
/// An open PDF document: its information and its pages (each page's size and rotation; the content is read and checked,
/// its text, images and paths come with the later phases). Opening reads the cross-reference and the page tree only; a
/// page's objects are read when it is. Safe to read from several threads.
/// </summary>
internal sealed class PdfDocument : IDocument
{
    private readonly PdfFile _file;
    private readonly PageNode[] _pages;
    private readonly string _version;
    private DocumentInfo? _info;
    private volatile bool _disposed;

    private PdfDocument(PdfFile file, PageNode[] pages, string version) => (_file, _pages, _version) = (file, pages, version);

    /// <summary>The object layer (the tests read objects through it).</summary>
    internal PdfFile File => _file;

    public string Format => PdfFormat.FormatName;

    public int PageCount => _pages.Length;

    public DocumentInfo Info
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            var info = _info ??= ReadInfo();
            return info.Repaired == _file.Repaired ? info : _info = info with { Repaired = _file.Repaired };
        }
    }

    /// <summary>The files the document carries: read from phase D4 of the documents plan (none listed before it).</summary>
    public IReadOnlyList<DocumentAttachment> Attachments => [];

    public IReadOnlyList<DocumentField> Fields => throw NotYet("form fields", "D9");

    public IReadOnlyList<DocumentOutline> Outline => throw NotYet("bookmarks", "D10");

    public IReadOnlyList<DocumentSignature> Signatures => throw NotYet("signatures", "D12");

    public string? PageLabel(int index) => throw NotYet("page labels", "D10");

    public DocumentDrawing DrawPage(int index, DocumentDrawOptions options) => DocumentPageDrawers.DrawPage(this, index, options);

    /// <summary>Opens the document <paramref name="stream"/> holds (an encrypted one with the empty password or <see cref="DocumentOptions.Password"/>).</summary>
    /// <exception cref="InvalidDataException">
    /// Not a PDF file, damaged beyond repair, a cycle, or a limit passed (each named); or encrypted, and neither the empty
    /// password nor <see cref="DocumentOptions.Password"/> opens it (the message names the option).
    /// </exception>
    /// <exception cref="NotSupportedException">The file is encrypted by a security handler that is not read (a public-key one: named).</exception>
    public static PdfDocument Open(Stream stream, string name, DocumentOptions options)
    {
        PdfFile? file = null;
        try
        {
            file = PdfFile.Open(stream, name, options);
            PdfSecurity.Attach(file, options.Password);
            var catalog = Catalog(file);
            var pages = Pages(file, catalog);
            if (pages.Length == 0 && !file.Repaired && file.RepairNow(file.Xref))
            {
                catalog = Catalog(file);
                pages = Pages(file, catalog);
            }

            var document = new PdfDocument(file, pages, Version(file, catalog));
            file = null;
            return document;
        }
        catch (InvalidDataException e) when (!e.Message.StartsWith(name, StringComparison.Ordinal))
        {
            throw new InvalidDataException($"{name}: {e.Message}", e);
        }
        catch (Exception e) when (Unexpected(e))
        {
            throw new InvalidDataException($"{name}: a damaged PDF document ({e.GetType().Name}: {e.Message}).", e);
        }
        finally
        {
            file?.Dispose();
        }
    }

    public DocumentPage ReadPage(int index, DocumentParts parts)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(index, _pages.Length);
        ObjectDisposedException.ThrowIf(_disposed, this);
        string name = _file.Name;
        try
        {
            return Page(index, parts);
        }
        catch (InvalidDataException e) when (!e.Message.StartsWith(name, StringComparison.Ordinal))
        {
            throw new InvalidDataException($"{name}: page {index + 1}: {e.Message}", e);
        }
        catch (Exception e) when (Unexpected(e))
        {
            throw new InvalidDataException($"{name}: page {index + 1}: damaged ({e.GetType().Name}: {e.Message}).", e);
        }
    }

    public void Dispose()
    {
        _disposed = true;
        _file.Dispose();
    }

    // What a damaged file may throw that is not already an InvalidDataException (a cut number, an index past an end).
    private static bool Unexpected(Exception e) => e is not (InvalidDataException or NotSupportedException or OperationCanceledException or OutOfMemoryException
        or ObjectDisposedException or InsufficientExecutionStackException);

    private DocumentPage Page(int index, DocumentParts parts)
    {
        var (node, inherited) = _pages[index];
        var page = _file.Dictionary(node) ?? throw new InvalidDataException($"{_file.Name}: page {index + 1} is not a dictionary ({node}).");
        var media = Box(page.TryGet("MediaBox"u8, out var own) ? own : inherited.MediaBox) ?? (0, 0, 612, 792);
        var crop = Box(page.TryGet("CropBox"u8, out own) ? own : inherited.CropBox) is { } cropped ? Intersect(cropped, media) : media;
        int rotation = Rotation(page.TryGet("Rotate"u8, out own) ? own : inherited.Rotate);
        double unit = _file.Resolve(page["UserUnit"u8]).TryNumber(out double u) && u > 0 && double.IsFinite(u) ? u : 1;
        double width = (crop.Right - crop.Left) * unit, height = (crop.Top - crop.Bottom) * unit;
        if (rotation is 90 or 270)
        {
            (width, height) = (height, width);
        }

        if (!(width >= 0 && height >= 0 && width < float.MaxValue && height < float.MaxValue))
        {
            throw new InvalidDataException($"{_file.Name}: page {index + 1} is {width} x {height} points.");
        }

        var issues = new List<DocumentIssue>();
        if ((parts & (DocumentParts.Text | DocumentParts.Images | DocumentParts.Paths | DocumentParts.Glyphs)) != 0)
        {
            ReadContents(index, page, issues);
        }

        return new DocumentPage(index, (float)width, (float)height) { Rotation = rotation, Issues = issues };
    }

    // The page's content streams, decoded and checked (the content interpreter reads them from phase D3): a damaged one
    // is counted in the page's issues; a limit passed or a cycle fails the page.
    private void ReadContents(int index, PdfDictionary page, List<DocumentIssue> issues)
    {
        int damaged = 0;
        var detail = new List<string>();
        PdfValue contents;
        try
        {
            contents = _file.Resolve(page["Contents"u8]);
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            issues.Add(new DocumentIssue(DocumentIssue.DamagedStream, $"page {index + 1}: its contents: {e.Message}"));
            return;
        }

        void One(PdfValue part)
        {
            try
            {
                if (_file.Resolve(part).AsStream is not { } stream)
                {
                    return;
                }

                using var decoded = _file.Decode(stream);
                if (decoded.Damaged)
                {
                    damaged++;
                    detail.Add($"object {stream.Number} is damaged part of the way");
                }
            }
            catch (InvalidDataException e) when (!PdfLimit.Is(e))
            {
                damaged++;
                detail.Add(e.Message);
            }
        }

        if (contents.AsArray is { } array)
        {
            foreach (var part in array.Items)
            {
                One(part);
            }
        }
        else if (contents.Kind == PdfKind.Stream)
        {
            One(contents);
        }

        if (damaged > 0)
        {
            issues.Add(new DocumentIssue(DocumentIssue.DamagedStream, $"page {index + 1}: {string.Join("; ", detail)}", damaged));
        }
    }

    // A rectangle [llx lly urx ury] (its corners in any order), or null when it is not one.
    private (double Left, double Bottom, double Right, double Top)? Box(PdfValue value)
    {
        if (_file.Resolve(value).AsArray is not { Count: >= 4 } array)
        {
            return null;
        }

        Span<double> n = stackalloc double[4];
        for (int i = 0; i < 4; i++)
        {
            if (!_file.Resolve(array[i]).TryNumber(out n[i]) || !double.IsFinite(n[i]))
            {
                return null;
            }
        }

        return (Math.Min(n[0], n[2]), Math.Min(n[1], n[3]), Math.Max(n[0], n[2]), Math.Max(n[1], n[3]));
    }

    private static (double Left, double Bottom, double Right, double Top) Intersect((double Left, double Bottom, double Right, double Top) a, (double Left, double Bottom, double Right, double Top) b)
    {
        var box = (Math.Max(a.Left, b.Left), Math.Max(a.Bottom, b.Bottom), Math.Min(a.Right, b.Right), Math.Min(a.Top, b.Top));
        return box.Item3 > box.Item1 && box.Item4 > box.Item2 ? box : b;   // a crop box outside the media box shows the media box
    }

    // /Rotate: a multiple of 90, clockwise; any other value is 0, as readers take it.
    private int Rotation(PdfValue value)
    {
        if (!_file.Resolve(value).TryInteger(out long degrees) || degrees % 90 != 0)
        {
            return 0;
        }

        return (int)((degrees % 360 + 360) % 360);
    }

    private static NotSupportedException NotYet(string what, string phase) =>
        new($"PDF {what} are not read yet: they come with phase {phase} of the documents plan.");

    private static PdfDictionary Catalog(PdfFile file)
    {
        var root = file.Trailer["Root"u8];
        var catalog = file.Dictionary(root);
        if (catalog is null && !file.Repaired && file.RepairNow(file.Xref))
        {
            catalog = file.Dictionary(file.Trailer["Root"u8]);
        }

        return catalog ?? throw new InvalidDataException($"{file.Name}: no document catalog (the trailer's /Root is {root}).");
    }

    // The header's version, or the catalog's /Version when that is later (an update can raise it).
    private static string Version(PdfFile file, PdfDictionary catalog)
    {
        string header = file.HeaderVersion ?? "";
        if (file.Resolve(catalog["Version"u8]).AsName?.ToString() is { } stated && double.TryParse(stated, NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture, out double later)
            && (!double.TryParse(header, NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture, out double first) || later > first))
        {
            return stated;
        }

        return header;
    }

    // The page tree walked once, in order, without recursion: each leaf with what it inherits (MediaBox, CropBox, Rotate,
    // Resources). /Count is not trusted: the kids are counted. A node met again is a cycle; a kid that cannot be read is
    // a page of its own, which fails alone when it is read.
    private static PageNode[] Pages(PdfFile file, PdfDictionary catalog)
    {
        var pages = new List<PageNode>();
        var visited = new HashSet<int>();
        var stack = new Stack<(PdfValue Node, Inherited Inherited, int Depth)>();
        stack.Push((catalog["Pages"u8], Inherited.None, 1));
        while (stack.Count > 0)
        {
            var (node, inherited, depth) = stack.Pop();
            PdfDictionary? dictionary;
            try
            {
                dictionary = file.Dictionary(node);
            }
            catch (InvalidDataException e) when (!PdfLimit.Is(e))
            {
                pages.Add(new PageNode(node, inherited));
                continue;
            }

            if (dictionary is null)
            {
                continue;   // not a dictionary (a reference to nothing): no page
            }

            var type = dictionary["Type"u8];
            var kids = type.IsName("Page"u8) ? null : file.Resolve(dictionary["Kids"u8]).AsArray;
            if (kids is null && type.IsName("Pages"u8))
            {
                continue;   // a node of the tree with no kids: no page
            }

            if (kids is null)
            {
                pages.Add(new PageNode(node, inherited));
                if (pages.Count > file.MaxObjects)
                {
                    throw file.ObjectsLimit(pages.Count, "the page tree");
                }

                continue;
            }

            if (node.IsReference && !visited.Add(node.AsReference.Number))
            {
                throw PdfLimit.Error($"{file.Name}: a cycle in the page tree: the node {node} is its own descendant.");
            }

            if (depth > file.MaxDepth)
            {
                throw PdfLimit.Error($"{file.Name}: a page tree nested deeper than {file.MaxDepth} levels, the limit {nameof(DocumentOptions.MaxDepth)} (DocumentOptions.MaxDepth).");
            }

            var passed = inherited.With(dictionary);
            for (int i = kids.Count - 1; i >= 0; i--)
            {
                stack.Push((kids[i], passed, depth + 1));
            }
        }

        return [.. pages];
    }

    private DocumentInfo ReadInfo()
    {
        PdfDictionary? info = null;
        try
        {
            info = _file.Dictionary(_file.Trailer["Info"u8]);
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            // Damaged information: none.
        }

        var entries = new Dictionary<string, string>(StringComparer.Ordinal);
        string? title = null, author = null, subject = null, keywords = null, creator = null, producer = null;
        DateTimeOffset? created = null, modified = null;
        for (int i = 0; i < (info?.Count ?? 0); i++)
        {
            var key = info!.KeyAt(i);
            PdfValue value;
            try
            {
                value = _file.Resolve(info.ValueAt(i));
            }
            catch (InvalidDataException e) when (!PdfLimit.Is(e))
            {
                continue;
            }

            if (key.Is("CreationDate"u8) || key.Is("ModDate"u8))
            {
                var date = value.AsString is { } written ? PdfText.Date(written.Bytes) : null;
                if (key.Is("CreationDate"u8))
                {
                    created = date;
                }
                else
                {
                    modified = date;
                }

                if (date is null && Text(value) is { } odd)
                {
                    entries[key.ToString()] = odd;   // a date that is not one: kept as written
                }

                continue;
            }

            string? text = Text(value);
            if (key.Is("Title"u8))
            {
                title = text;
            }
            else if (key.Is("Author"u8))
            {
                author = text;
            }
            else if (key.Is("Subject"u8))
            {
                subject = text;
            }
            else if (key.Is("Keywords"u8))
            {
                keywords = text;
            }
            else if (key.Is("Creator"u8))
            {
                creator = text;
            }
            else if (key.Is("Producer"u8))
            {
                producer = text;
            }
            else if (text is not null)
            {
                entries[key.ToString()] = text;
            }
        }

        return new DocumentInfo
        {
            Title = title, Author = author, Subject = subject, Keywords = keywords, Creator = creator, Producer = producer, Created = created, Modified = modified,
            Version = _version, Encrypted = _file.Security is not null, Repaired = _file.Repaired, Entries = entries.AsReadOnly(),
        };
    }

    // An entry as text: a string decoded, a name without its slash, a number or a boolean as written, anything else as PDF.
    private static string? Text(PdfValue value) => value.Kind switch
    {
        PdfKind.Null => null,
        PdfKind.String => PdfText.Decode(value.AsString!.Bytes),
        PdfKind.Name => value.AsName!.ToString(),
        _ => value.ToString(),
    };

    // A leaf of the page tree: the page (a reference, or the dictionary itself) and what it inherits.
    private readonly record struct PageNode(PdfValue Node, Inherited Inherited);

    // What a page inherits from the nodes above it; shared by every page under a node that sets none of it.
    private sealed class Inherited(PdfValue mediaBox, PdfValue cropBox, PdfValue rotate, PdfValue resources)
    {
        public static Inherited None { get; } = new(PdfValue.Null, PdfValue.Null, PdfValue.Null, PdfValue.Null);

        public PdfValue MediaBox => mediaBox;

        public PdfValue CropBox => cropBox;

        public PdfValue Rotate => rotate;

        public PdfValue Resources => resources;

        public Inherited With(PdfDictionary node)
        {
            bool m = node.TryGet("MediaBox"u8, out var media), c = node.TryGet("CropBox"u8, out var crop), r = node.TryGet("Rotate"u8, out var rotation),
                s = node.TryGet("Resources"u8, out var own);
            return m || c || r || s ? new Inherited(m ? media : mediaBox, c ? crop : cropBox, r ? rotation : rotate, s ? own : resources) : this;
        }
    }
}
