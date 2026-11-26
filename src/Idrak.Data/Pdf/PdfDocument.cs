// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Data.Pdf;

/// <summary>
/// An open PDF document: its information, its pages (each page's size and rotation, its text: runs with their boxes, the
/// lines and text the layout makes of them; its images, placed, their pixels decoded when asked; its vector paths, clips,
/// shadings, patterns, transparency groups and soft masks; its form's widgets with their appearances; drawn to pixels
/// through <see cref="DocumentPageDrawers"/>), the files it carries and its form's fields. Opening reads the cross-reference and the page tree only; a page's objects are read
/// when it is, and the fonts its pages share are kept for the others. Safe to read from several threads.
/// </summary>
internal sealed partial class PdfDocument : IDocument
{
    private readonly PdfFile _file;
    private readonly PageNode[] _pages;
    private readonly string _version;
    private readonly DocumentOptions _options;
    private DocumentInfo? _info;
    private IReadOnlyList<DocumentAttachment>? _attachments;
    private IReadOnlyList<DocumentSignature>? _signatures;
    private PdfForm? _form;
    private object? _template;                                                 // the form's template, or NoTemplate, once read
    private int _xfa = -1;
    private PdfContentReader? _reader;                                         // the last page's reader, for the next page
    private volatile bool _disposed;

    private PdfDocument(PdfFile file, PageNode[] pages, string version, DocumentOptions options) => (_file, _pages, _version, _options) = (file, pages, version, options);

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

    /// <summary>
    /// The files the document carries: its <c>EmbeddedFiles</c> name tree's, then its pages' file-attachment annotations'
    /// (a file listed once), found on the first call; each file's bytes are read when it is opened.
    /// </summary>
    public IReadOnlyList<DocumentAttachment> Attachments
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (Volatile.Read(ref _attachments) is { } known)
            {
                return known;
            }

            string name = _file.Name;
            IReadOnlyList<DocumentAttachment> found;
            try
            {
                found = PdfAttachments.Read(_file, [.. _pages.Select(p => p.Node)]);
            }
            catch (InvalidDataException e) when (!e.Message.StartsWith(name, StringComparison.Ordinal))
            {
                throw new InvalidDataException($"{name}: its embedded files: {e.Message}", e);
            }
            catch (Exception e) when (Unexpected(e))
            {
                throw new InvalidDataException($"{name}: its embedded files: damaged ({e.GetType().Name}: {e.Message}).", e);
            }

            return Interlocked.CompareExchange(ref _attachments, found, null) ?? found;
        }
    }

    /// <summary>
    /// The form's fields: the AcroForm's terminal fields in tree order (values, flags and appearance taken from their
    /// ancestors where they inherit them), each with its widgets' pages, boxes and states; then an XFA form's data values
    /// no AcroForm field holds, as text fields named by their path. Read on the first call.
    /// </summary>
    public IReadOnlyList<DocumentField> Fields
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return Form.Public;
        }
    }

    /// <summary>The form as the editor changes it (its fields' objects, widgets and resources), read on the first call; empty for a document without one.</summary>
    internal PdfForm Form
    {
        get
        {
            if (Volatile.Read(ref _form) is { } known)
            {
                return known;
            }

            string name = _file.Name;
            PdfForm read;
            try
            {
                read = PdfForms.Read(this);
            }
            catch (InvalidDataException e) when (!e.Message.StartsWith(name, StringComparison.Ordinal))
            {
                throw new InvalidDataException($"{name}: its form: {e.Message}", e);
            }
            catch (Exception e) when (Unexpected(e))
            {
                throw new InvalidDataException($"{name}: its form: damaged ({e.GetType().Name}: {e.Message}).", e);
            }

            return Interlocked.CompareExchange(ref _form, read, null) ?? read;
        }
    }

    /// <summary>
    /// The form's design as its XFA template gives it (<see cref="PdfFormTemplates"/>: sections, fields with their captions,
    /// kinds, choices and bindings, linked to the form's data and to <see cref="Fields"/>), or null for a document without
    /// one. Read on the first call; its scripts are listed as issues, never run.
    /// </summary>
    public DocumentFormTemplate? FormTemplate
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (Volatile.Read(ref _template) is { } known)
            {
                return known as DocumentFormTemplate;
            }

            string name = _file.Name;
            DocumentFormTemplate? read;
            try
            {
                read = PdfFormTemplates.Read(this, Form);
            }
            catch (InvalidDataException e) when (!e.Message.StartsWith(name, StringComparison.Ordinal))
            {
                throw new InvalidDataException($"{name}: its form's template: {e.Message}", e);
            }
            catch (Exception e) when (Unexpected(e))
            {
                throw new InvalidDataException($"{name}: its form's template: damaged ({e.GetType().Name}: {e.Message}).", e);
            }

            return Interlocked.CompareExchange(ref _template, read ?? NoTemplate, null) as DocumentFormTemplate ?? read;
        }
    }

    // What the template slot holds once a document is found to have none.
    private static readonly object NoTemplate = new();

    /// <summary>Whether the document holds an XFA form (its catalog's AcroForm has /XFA), found once.</summary>
    internal bool HasXfa
    {
        get
        {
            int known = Volatile.Read(ref _xfa);
            if (known < 0)
            {
                known = PdfForms.HasXfa(_file) ? 1 : 0;
                Volatile.Write(ref _xfa, known);
            }

            return known == 1;
        }
    }

    /// <summary>Page <paramref name="index"/>'s object number (0 for a page held inside another object): what a widget's /P names.</summary>
    internal int PageNumber(int index) => _pages[index].Node.IsReference ? _pages[index].Node.AsReference.Number : 0;

    /// <summary>
    /// The signatures (D12): every signature field's signature, in the order they were made, each with its byte range read
    /// against the file, what changed after it, and its check (<see cref="PdfSignatureCheck"/>); found on the first call.
    /// </summary>
    public IReadOnlyList<DocumentSignature> Signatures
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (Volatile.Read(ref _signatures) is { } known)
            {
                return known;
            }

            string name = _file.Name;
            IReadOnlyList<DocumentSignature> found;
            try
            {
                found = PdfSignatures.Read(this);
            }
            catch (InvalidDataException e) when (!e.Message.StartsWith(name, StringComparison.Ordinal))
            {
                throw new InvalidDataException($"{name}: its signatures: {e.Message}", e);
            }
            catch (Exception e) when (Unexpected(e))
            {
                throw new InvalidDataException($"{name}: its signatures: damaged ({e.GetType().Name}: {e.Message}).", e);
            }

            return Interlocked.CompareExchange(ref _signatures, found, null) ?? found;
        }
    }

    // The page drawers were registered (DocumentPageDrawers) and picked by the options' name: not on the new design yet.
    // The default, "coverage" (Idrak.Imaging's CoverageDocumentDrawer), is reached directly; another name stops.
    private static readonly Idrak.Imaging.CoverageDocumentDrawer Coverage = new();

    public DocumentDrawing DrawPage(int index, DocumentDrawOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return options.Drawer.Equals(DocumentPageDrawers.Coverage, StringComparison.OrdinalIgnoreCase)
            ? Coverage.Draw(ReadPage(index, DocumentPageDrawers.PartsFor(options)), options)
            : throw new NotSupportedException($"Page {index}: the page drawer '{options.Drawer}' by name (DocumentPageDrawers) is not on the new design yet.");
    }

    /// <summary>
    /// Opens the document <paramref name="stream"/> holds (an encrypted one with <see cref="DocumentOptions.Password"/>, as
    /// its owner's then its user's password, or the empty password; a damaged one repaired, its key found before its
    /// object streams are listed).
    /// </summary>
    /// <exception cref="InvalidDataException">
    /// Not a PDF file, damaged beyond repair, a cycle, or a limit passed (each named); or encrypted, and neither
    /// <see cref="DocumentOptions.Password"/> nor the empty password opens it (the message names the option).
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

            var document = new PdfDocument(file, pages, Version(file, catalog), options);
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

        var issues = new PdfIssues();
        var content = ContentParts.None;
        var display = Display(crop, rotation, unit);
        if ((parts & (DocumentParts.Text | DocumentParts.Images | DocumentParts.Paths | DocumentParts.Glyphs)) != 0)
        {
            PdfDictionary? resources = null;
            try
            {
                resources = _file.Dictionary(page.TryGet("Resources"u8, out own) ? own : inherited.Resources);
            }
            catch (InvalidDataException e) when (!PdfLimit.Is(e))
            {
                issues.Add(DocumentIssue.DamagedStream, $"page {index + 1}: its resources: {e.Message}");
            }

            content = Content(index, (float)width, (float)height, parts, issues, (Document: this, Parts: parts, Node: node, Page: page, Resources: resources, Display: display),
                static (reader, at) =>
                {
                    if ((at.Parts & (DocumentParts.Text | DocumentParts.Glyphs)) != 0)
                    {
                        reader.Marks = at.Document.Marks(at.Node);             // a tagged file: each run's place in its structure (D10)
                    }

                    reader.Read(at.Page["Contents"u8], at.Resources, at.Display);
                });
        }

        // The page's annotations (D10: every kind, a form's widgets among them, each with its appearance read as content of
        // the page); an XFA form's layout is not drawn, which every page read says.
        IReadOnlyList<DocumentAnnotation> annotations = [];
        if ((parts & DocumentParts.Annotations) != 0)
        {
            annotations = PdfAnnotations.Read(this, index, page, display, (float)width, (float)height, parts, issues);
        }

        if (parts != DocumentParts.None && HasXfa)
        {
            issues.Add(DocumentIssue.NotSupported, $"page {index + 1}: an XFA form's layout is not drawn: its data is read as fields (IDocument.Fields), and its AcroForm widgets are the form shown");
        }

        if (parts != DocumentParts.None && _file.Security is { PasswordRefused: true })
        {
            issues.Add(DocumentIssue.WrongPassword, $"page {index + 1}: the password given in DocumentOptions.Password is neither the user nor the owner password: the empty user password opened the file, as its user");
        }

        return new DocumentPage(index, (float)width, (float)height)
        {
            Rotation = rotation, Runs = content.Runs, Lines = content.Lines, Text = content.Text, Images = content.Images, Paths = content.Paths, Clips = content.Clips,
            Shadings = content.Shadings, Groups = content.Groups, SoftMasks = content.SoftMasks, Annotations = annotations, Issues = issues.ToList(),
        };
    }

    /// <summary>
    /// The parts <paramref name="parts"/> asks of content <paramref name="interpret"/> runs through a reader (a page's
    /// content, or a form XObject drawn on its own: an annotation's appearance) on page <paramref name="index"/> of
    /// <paramref name="width"/> x <paramref name="height"/> points, what could not be read counted in <paramref name="issues"/>.
    /// </summary>
    internal ContentParts Content(int index, float width, float height, DocumentParts parts, PdfIssues issues, Action<PdfContentReader> interpret) =>
        Content(index, width, height, parts, issues, interpret, static (reader, act) => act(reader));

    /// <summary>
    /// <see cref="Content(int, float, float, DocumentParts, PdfIssues, Action{PdfContentReader})"/> with what
    /// <paramref name="interpret"/> needs passed as <paramref name="state"/> (a static lambda: nothing made to hold it).
    /// The document keeps a reader from one page to the next (taken by one thread at a time: a page read beside another
    /// makes its own), its working buffers rented for a page and given back at its end.
    /// </summary>
    internal ContentParts Content<TState>(int index, float width, float height, DocumentParts parts, PdfIssues issues, TState state, Action<PdfContentReader, TState> interpret)
    {
        var reader = Interlocked.Exchange(ref _reader, null) ?? new PdfContentReader(_file);
        try
        {
            reader.Begin(index + 1, issues, _options.IncludeHidden);
            if ((parts & DocumentParts.Images) != 0)
            {
                reader.ReadImages();
            }

            // Colours, clips and transparency come with what is drawn (paths, glyphs or images); text alone stays as cheap as before.
            if ((parts & (DocumentParts.Paths | DocumentParts.Glyphs | DocumentParts.Images)) != 0)
            {
                reader.ReadVector();
            }

            interpret(reader, state);
            var content = ContentParts.None with { Images = reader.Images, Clips = reader.Clips, Groups = reader.Groups, SoftMasks = reader.SoftMasks };
            if ((parts & DocumentParts.Paths) != 0)
            {
                content = content with { Paths = reader.Paths, Shadings = reader.Shadings };
            }

            if ((parts & (DocumentParts.Text | DocumentParts.Glyphs)) != 0)
            {
                var read = reader.TextRuns((parts & DocumentParts.Glyphs) != 0, out int unmapped, out string? fonts);
                if (unmapped > 0)
                {
                    issues.Add(DocumentIssue.UnmappedGlyphs, $"page {index + 1}: glyphs of {fonts} have no code point (no ToUnicode, glyph name or cmap gives one): their text is U+FFFD", unmapped);
                }

                content = content with { Runs = read };
                if ((parts & DocumentParts.Text) != 0)
                {
                    var arranged = reader.Arrange(read, width, height, _options);
                    content = content with { Lines = arranged.Lines, Text = arranged.Text };
                }
            }

            return content;
        }
        finally
        {
            reader.End();
            Volatile.Write(ref _reader, reader);
        }
    }

    /// <summary>What a content's reading gives a page (each part empty unless asked).</summary>
    internal readonly record struct ContentParts(IReadOnlyList<DocumentTextRun> Runs, IReadOnlyList<DocumentLine> Lines, string Text, IReadOnlyList<DocumentImage> Images,
        IReadOnlyList<DocumentPath> Paths, IReadOnlyList<DocumentPath> Clips, IReadOnlyList<DocumentShading> Shadings, IReadOnlyList<DocumentGroup> Groups,
        IReadOnlyList<DocumentSoftMask> SoftMasks)
    {
        public static ContentParts None => new([], [], "", [], [], [], [], [], []);
    }

    // The map from the page's user space onto the page as shown: the crop box's lower left to the origin, y turned down,
    // the page's rotation (clockwise), and its unit.
    private static PdfMatrix Display((double Left, double Bottom, double Right, double Top) crop, int rotation, double unit)
    {
        double w = crop.Right - crop.Left, h = crop.Top - crop.Bottom, x0 = crop.Left, y0 = crop.Bottom;
        var map = rotation switch
        {
            90 => new PdfMatrix(0, 1, 1, 0, -y0, -x0),
            180 => new PdfMatrix(-1, 0, 0, 1, w + x0, -y0),
            270 => new PdfMatrix(0, -1, -1, 0, h + y0, w + x0),
            _ => new PdfMatrix(1, 0, 0, -1, -x0, h + y0),
        };
        return unit == 1 ? map : map.Then(new PdfMatrix(unit, 0, 0, unit, 0, 0));
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
            Version = _version, Encrypted = _file.Security is not null, Security = _file.Security?.Describe(), Repaired = _file.Repaired, Entries = entries.AsReadOnly(),
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

    /// <summary>The format version the document states (the header's, or the catalog's when later).</summary>
    internal string StatedVersion => _version;

    /// <summary>The options it was opened with (a revision of it is opened alike: the limits, the password).</summary>
    internal DocumentOptions Options => _options;

    /// <summary>Whether it was disposed (a signature's check reads the file).</summary>
    internal bool Disposed => _disposed;

    /// <summary>
    /// Page <paramref name="index"/>'s object (a reference, or the dictionary itself) and what the page tree above it sets
    /// of what a page inherits (null where no node does): what an editor writes a page again from.
    /// </summary>
    internal (PdfValue Node, PdfValue MediaBox, PdfValue CropBox, PdfValue Rotate, PdfValue Resources) PageObject(int index)
    {
        var (node, inherited) = _pages[index];
        return (node, inherited.MediaBox, inherited.CropBox, inherited.Rotate, inherited.Resources);
    }

    /// <summary>
    /// Page <paramref name="index"/> as shown when turned by <paramref name="rotation"/> (its own when null): the map from
    /// its user space onto the page (top-left origin, y down; the same map <see cref="ReadPage"/> places runs by), its size
    /// and its rotation. An editor writes content over the page through the map turned back.
    /// </summary>
    internal (PdfMatrix Display, float Width, float Height, int Rotation) Frame(int index, int? rotation = null)
    {
        var (node, inherited) = _pages[index];
        var page = _file.Dictionary(node) ?? PdfDictionary.Empty;
        var media = Box(page.TryGet("MediaBox"u8, out var own) ? own : inherited.MediaBox) ?? (0, 0, 612, 792);
        var crop = Box(page.TryGet("CropBox"u8, out own) ? own : inherited.CropBox) is { } cropped ? Intersect(cropped, media) : media;
        int turned = rotation ?? Rotation(page.TryGet("Rotate"u8, out own) ? own : inherited.Rotate);
        double unit = _file.Resolve(page["UserUnit"u8]).TryNumber(out double u) && u > 0 && double.IsFinite(u) ? u : 1;
        double width = (crop.Right - crop.Left) * unit, height = (crop.Top - crop.Bottom) * unit;
        if (turned is 90 or 270)
        {
            (width, height) = (height, width);
        }

        return (Display(crop, turned, unit), (float)width, (float)height, turned);
    }

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
