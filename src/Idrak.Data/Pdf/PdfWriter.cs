// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Security.Cryptography;
using System.Text;

namespace Idrak.Data.Pdf;

/// <summary>
/// The library's PDF writer as "pdf": pages of images, vector paths and text
/// written as they finish, the reader's object model and filters shared. The file: the header, objects written on spans
/// (numbers invariant, strings escaped, names with <c>#xx</c>), streams Flate-compressed at
/// <see cref="DocumentWriteOptions.Compression"/> (0 stores them), small objects in object streams and a cross-reference
/// stream (a classic table for <see cref="DocumentWriteOptions.Version"/> before 1.5, 1.7 unless set), no date or
/// identifier unless given (<see cref="DocumentWriteOptions.Deterministic"/>): the same input writes the same bytes. Each
/// page's images go to the file as they are given, its content stream and page object when the next page starts; the page
/// tree, the fonts (one composite font a face, its program subset through the TrueType writer, a ToUnicode map from
/// the runs' text) and the cross-reference at <see cref="IDocumentBuilder.Finish"/>. A builder disposed without it writes
/// the page in hand, the page tree, the catalog and a trailer naming it, and no cross-reference: the reader rebuilds it
/// and says the file was repaired (fonts, written at the end, are missing). Pages are copied from other PDF documents
/// (merging and splitting, <see cref="IDocumentBuilder.CopyPage"/>: what each uses copied once, duplicates written once)
/// and a document is saved encrypted (<see cref="IDocumentBuilder.Encrypt"/>: AES-256, revision 6); form fields are added
/// with their appearances (<see cref="IDocumentBuilder.AddField"/>, <see cref="PdfFormBuilder"/>); a PDF document is edited
/// through <see cref="Edit"/> (<see cref="PdfEditor"/>) and signed (<see cref="IDocumentBuilder.Sign"/>, D12: the
/// finished file signed as incremental updates, <see cref="PdfSigning"/>). Annotations, links, bookmarks and page labels
/// come with the later phase of the documents plan (D10), each a <see cref="NotSupportedException"/> naming its phase
/// until it lands.
/// </summary>
public sealed class PdfWriter : IDocumentWriter
{
    /// <summary>"pdf".</summary>
    public string Name => PdfFormat.FormatName;

    /// <summary>".pdf".</summary>
    public IReadOnlyCollection<string> Extensions { get; } = [".pdf"];

    /// <summary>"application/pdf".</summary>
    public IReadOnlyCollection<string> MediaTypes { get; } = ["application/pdf"];

    /// <inheritdoc/>
    public IDocumentBuilder Create(Stream destination, DocumentWriteOptions options)
    {
        ArgumentNullException.ThrowIfNull(destination);
        ArgumentNullException.ThrowIfNull(options);
        if (!destination.CanWrite)
        {
            throw new ArgumentException("A PDF document is written to a stream that writes.", nameof(destination));
        }

        return new PdfDocumentBuilder(destination, options);
    }

    /// <inheritdoc/>
    public IDocumentEditor Edit(IDocument source, Stream destination, DocumentWriteOptions options)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(destination);
        ArgumentNullException.ThrowIfNull(options);
        if (!destination.CanWrite)
        {
            throw new ArgumentException("A PDF document is written to a stream that writes.", nameof(destination));
        }

        return new PdfEditor(source, Own(source, "edits"), destination, options);
    }

    /// <summary>
    /// The PDF document <paramref name="document"/> is: the writer reads its objects to edit it or copy its pages.
    /// </summary>
    /// <exception cref="NotSupportedException">It is not a document the library's pdf format opened (the message names its format).</exception>
    internal static PdfDocument Own(IDocument document, string what) =>
        document as PdfDocument
        ?? throw new NotSupportedException(
            $"The PDF writer {what} PDF documents the library's pdf format opened, whose objects it reads; this one is a '{document.Format}' document another reader opened: open the file with the library's pdf format.");
}

/// <summary>A PDF document being written (see <see cref="PdfWriter"/>). Not safe from several threads.</summary>
internal sealed partial class PdfDocumentBuilder : IDocumentBuilder
{
    private const int CatalogNumber = 1, PagesNumber = 2;

    private readonly Stream _destination;
    private readonly PdfSpillStream _output;
    private readonly List<PdfSignRequest> _signatures = [];
    private readonly DocumentWriteOptions _options;
    private readonly string _version;
    private readonly PdfFileWriter _file;
    private DocumentEncryption? _encryption;
    private readonly bool _classic;
    private readonly List<int> _pages = [];
    private readonly Dictionary<IFontFace, PdfWriterFont> _fonts = new(ReferenceEqualityComparer.Instance);
    private readonly List<PdfWriterFont> _fontOrder = [];
    private readonly List<(string Name, byte[] Key, int Spec)> _attachments = [];
    private PdfPageBuilder? _page;
    private (PdfDocument Document, int Index)? _copied;                       // the page in hand is a copy of this one
    private PdfEditCopier? _copier;
    private PdfFormBuilder? _form;
    private PdfSecurity? _security;
    private int _saveStream;
    private DocumentInfo _info = DocumentInfo.Empty;
    private bool _finished, _disposed, _closing, _started;

    public PdfDocumentBuilder(Stream destination, DocumentWriteOptions options)
    {
        string version = options.Version ?? "1.7";
        if (!IsVersion(version))
        {
            throw new ArgumentException($"A PDF version is 1.0 to 1.7 or 2.0, not '{version}' (DocumentWriteOptions.Version).", nameof(options));
        }

        (_destination, _options, _version) = (destination, options, version);
        _classic = string.CompareOrdinal(version, "1.5") < 0;                  // cross-reference and object streams came with 1.5
        _output = new PdfSpillStream(destination);                             // the destination, or a temporary file once signing is asked
        _file = new PdfFileWriter(_output, version, options.Compression, packing: !_classic && options.ObjectStreams);
        _file.Allocate();                                                      // 1: the catalog
        _file.Allocate();                                                      // 2: the page tree's root
    }

    public DocumentInfo Info
    {
        get => _info;
        set
        {
            ArgumentNullException.ThrowIfNull(value);
            ThrowIfDone();
            _info = value;
        }
    }

    public int PageCount => _pages.Count + (_page is null ? 0 : 1);

    public IDocumentPageBuilder AddPage(float width, float height)
    {
        ThrowIfDone();
        if (!(width > 0 && height > 0) || !float.IsFinite(width) || !float.IsFinite(height))
        {
            throw new ArgumentOutOfRangeException(nameof(width), $"A page is {width} x {height} points: sizes are finite and above 0.");
        }

        FinishPage();
        _started = true;
        _page = new PdfPageBuilder(Font, WriteImage, _pages.Count, width, height) { AnnotationsWritten = Annotated };
        return _page;
    }

    /// <summary>
    /// Adds page <paramref name="index"/> of <paramref name="from"/> (a PDF document; it stays open until the next page
    /// starts or the document is finished): written when the next page starts, with what it uses copied (once for every
    /// page copied from that document; images and fonts the same as ones written already are not written again), what
    /// it inherits from its page tree written on it, and a reference to a page of <paramref name="from"/> that is never
    /// copied null. The builder given back draws over the page in its frame as shown (points from its top-left corner),
    /// its resources named apart from the page's own.
    /// </summary>
    public IDocumentPageBuilder CopyPage(IDocument from, int index)
    {
        ArgumentNullException.ThrowIfNull(from);
        ThrowIfDone();
        var document = PdfWriter.Own(from, "copies pages of");
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(index, document.PageCount);
        FinishPage();
        _started = true;
        var (display, width, height, _) = document.Frame(index);
        _page = new PdfPageBuilder(Font, WriteImage, _pages.Count, width, height, PdfEditPages.Inverse(display), PdfEditPages.Prefix(document, index)) { AnnotationsWritten = Annotated };
        _copied = (document, index);
        return _page;
    }

    public void Attach(string name, string mediaType, Stream content)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(content);
        ThrowIfDone();
        _started = true;
        int file = _file.Allocate(), spec = _file.Allocate();
        long size;
        using (var data = new PdfSyntax(64 * 1024))
        {
            size = PdfDeflate.Compress(content, _file.Level, data.Buffer);
            var s = _file.BeginStream(file).Raw("/Type/EmbeddedFile"u8);
            if (!string.IsNullOrEmpty(mediaType))
            {
                s.Raw("/Subtype"u8).Name(mediaType);
            }

            s.Raw("/Params<</Size "u8).Integer(size).Raw(">>"u8);
            if (_file.Level > 0)
            {
                s.Raw("/Filter/FlateDecode"u8);
            }

            _file.EndStream(data.Span, deflate: false, PdfCryptUse.EmbeddedFile);  // deflated as it was read
        }

        var f = _file.BeginSmall(spec);
        f.Raw("<</Type/Filespec/F "u8).Text(name).Raw("/UF "u8).Text(name).Raw("/EF<</F "u8).Reference(file).Raw(">>>>"u8);
        _file.EndSmall();
        _attachments.Add((name, StringBytes(name), spec));
    }

    // A text string's bytes as Text writes them (ASCII, or UTF-16BE after its mark): what a name tree is sorted by.
    private static byte[] StringBytes(string text) =>
        text.All(c => c is >= ' ' and <= '~') ? Encoding.ASCII.GetBytes(text) : [0xFE, 0xFF, .. Encoding.BigEndianUnicode.GetBytes(text)];

    /// <summary>
    /// Adds a form field: its objects numbered and its widgets' appearances written now (each made to show its value), each
    /// widget written with its page (the page being built or one added later), the field and the form at the end.
    /// </summary>
    /// <exception cref="ArgumentException">A widget on a page written already, a name another field has, or a field that cannot be written as given.</exception>
    public void AddField(DocumentField field)
    {
        ArgumentNullException.ThrowIfNull(field);
        ThrowIfDone();
        _started = true;                                                       // its appearances are written now: encryption comes before
        (_form ??= new PdfFormBuilder(_file, _options, Font)).Add(field, _pages.Count, PageCount, index => index == _pages.Count ? _page : null);
    }

    /// <summary>
    /// Encrypts the document as it is written: AES-256 (revision 6) with the user and owner passwords and permissions of
    /// <paramref name="encryption"/>, every string and stream after this call; called before the first page or file.
    /// </summary>
    /// <exception cref="InvalidOperationException">A page or a file was added already (it would be written plain), or the document is encrypted already.</exception>
    public void Encrypt(DocumentEncryption encryption)
    {
        ArgumentNullException.ThrowIfNull(encryption);
        ThrowIfDone();
        if (_started || _security is not null)
        {
            throw new InvalidOperationException(_security is not null
                ? "The PDF document is encrypted already: Encrypt is called once."
                : "Encrypt is called before the first page or attached file: what is written before it would be written plain.");
        }

        _security = PdfSecurity.Create(encryption);
        _encryption = encryption;                                              // its passwords open the file again when it is signed
        int number = _file.Allocate();
        (_file.Security, _file.PlainNumber, _file.EncryptNumber) = (_security, number, number);
    }

    /// <summary>
    /// Signs the document in the signature field <paramref name="field"/> when it is finished: the file is written whole
    /// (to a temporary file, from this call on), then each signature appended to it as an incremental update of its own on
    /// the destination (<see cref="PdfSigning"/>); called before the first page or file, like <see cref="Encrypt"/>.
    /// </summary>
    /// <exception cref="InvalidOperationException">A page or a file was added already (written to the destination as it is).</exception>
    public void Sign(string field, SigningOptions options)
    {
        ThrowIfDone();
        PdfSigning.Validate(field, options);
        if (_started)
        {
            throw new InvalidOperationException("Sign is called before the first page or attached file: what was written before it went to the destination already, outside the file the signature is added to.");
        }

        _output.Spill();
        _signatures.Add(new PdfSignRequest(field, options));
    }

    public void Finish()
    {
        ThrowIfDone();
        if (_form?.Unplaced(PageCount) is { Count: > 0 } unplaced)
        {
            throw new InvalidOperationException($"Form fields name pages the document does not have ({PageCount} pages): {string.Join(", ", unplaced)}; add those pages, then finish.");
        }

        _closing = true;
        try
        {
            FinishPage();
            _form?.Finish();
            WriteNavigation();                                                 // the pages' annotations and the bookmarks, every page numbered now
            foreach (var font in _fontOrder)
            {
                font.Write(_file, _options.EmbedFonts);
            }

            _copier?.Finish();
            WriteStructure(packed: true, form: true);
            int info = WriteInfo();
            WriteEncryption();
            _file.Finish(CatalogNumber, info, Identifier(), _classic);
            if (_output.Spilled is { } spill)
            {
                // The finished file, opened from the temporary file, signed onto the destination.
                spill.Flush();
                using var written = PdfDocument.Open(spill, "the PDF document being signed", new DocumentOptions { Password = _encryption?.OwnerPassword ?? _encryption?.UserPassword });
                PdfSigning.WriteSigned(written, _signatures, _destination, _options);
            }

            _destination.Flush();
        }
        finally
        {
            _finished = true;
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        try
        {
            if (!_finished && !_closing)
            {
                // Unfinished: the page in hand, the page tree and the catalog (standing on their own, so a scan finds them),
                // a trailer naming the catalog, and no cross-reference: a reader rebuilds it and says the file was repaired.
                _closing = true;
                FinishPage();
                WriteStructure(packed: false);
                WriteEncryption();
                _file.Unfinished(CatalogNumber);
                if (_output.Spilled is { } spill)
                {
                    spill.Flush();
                    spill.Position = 0;
                    spill.CopyTo(_destination);                               // unsigned, as far as it got
                }

                _destination.Flush();
            }
        }
        catch (Exception e) when (e is IOException or ObjectDisposedException or NotSupportedException or InvalidOperationException or InvalidDataException or ArgumentException)
        {
            // A destination that fails, or a page that cannot be written: the file stays as far as it got.
        }
        finally
        {
            _page?.Release();
            _page = null;
            _copier?.Dispose();
            _security?.Dispose();                                              // the file key cleared
            _file.Dispose();
            _output.Dispose();                                                 // the temporary file deleted
        }
    }

    /// <summary>The face's font in this document, made on its first use (its Type0 font's number taken now, its objects written at the end).</summary>
    internal PdfWriterFont Font(IFontFace face)
    {
        if (!_fonts.TryGetValue(face, out var font))
        {
            font = new PdfWriterFont(face, _file.Allocate(), _fontOrder.Count + 1);
            _fonts.Add(face, font);
            _fontOrder.Add(font);
        }

        return font;
    }

    /// <summary>Writes <paramref name="image"/> to the file now as an image XObject; its object number.</summary>
    internal int WriteImage(ImageData image, DocumentImageEncoding encoding)
    {
        int number = _file.Allocate();
        PdfWriterImages.Write(_file, number, image, encoding, _options);
        return number;
    }

    internal static NotSupportedException NotYet(string what, string phase) =>
        new($"The PDF writer does not write {what} yet: they come with phase {phase} of the documents plan.");

    /// <summary>Whether <paramref name="version"/> is a PDF version the writer writes (1.0 to 1.7, or 2.0).</summary>
    internal static bool IsVersion(string version) =>
        version is "2.0" || (version.Length == 3 && version[0] == '1' && version[1] == '.' && version[2] is >= '0' and <= '7');

    private void ThrowIfDone()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_finished || _closing)
        {
            throw new InvalidOperationException("The PDF document is finished: nothing more is added to it.");
        }
    }

    private void FinishPage()
    {
        if (_page is { } page)
        {
            _page = null;
            int index = _pages.Count;
            var widgets = _form?.Annotations(index, page);
            if (_copied is { } copy)
            {
                _copied = null;
                var toUser = PdfEditPages.Inverse(copy.Document.Frame(copy.Index).Display);
                _pages.Add(WriteCopy(page, copy.Document, copy.Index, widgets));
                Target(_pages[^1], toUser);
            }
            else
            {
                page.Annotations = widgets;
                _pages.Add(page.Write(_file, PagesNumber));
                Target(_pages[^1], new PdfMatrix(1, 0, 0, -1, 0, page.Height));
            }

            _form?.PageWritten(index, _pages[index]);                         // its widgets, which name it
            _file.Flush();                                                     // the page goes to the destination as it finishes
        }
    }

    // A copied page: what it uses copied (once a source document), what it inherits written on it, the content drawn over
    // it after its own; its object's number (one a page copied earlier referred to it by, else a new one).
    private int WriteCopy(PdfPageBuilder content, PdfDocument document, int index, List<int>? widgets)
    {
        try
        {
            var copier = _copier ??= new PdfEditCopier(_file, _options.Compression, recompress: false) { Root = PagesNumber };
            var source = copier.For(document, reserve: true);
            var node = document.PageObject(index).Node;
            int number = node.IsReference ? copier.PageNumber(source, node.AsReference.Number) : _file.Allocate();
            var how = new PdfPageWrite(number, PagesNumber, document.Frame(index).Rotation, Materialize: true, ParentCrops: false, ParentRotates: false);
            PdfAnnotsEdit? annots = null;
            if (widgets is { Count: > 0 })
            {
                annots = new PdfAnnotsEdit();
                foreach (int widget in widgets)
                {
                    annots.Added.Add(() => widget);
                }
            }

            PdfEditPages.Write(copier, _file, source, index, how, content, image => image, () => _saveStream != 0 ? _saveStream : _saveStream = PdfEditPages.WriteSave(_file),
                annots, content.Drafts.Count > 0 ? new PdfPageEdits { Content = content, Pages = PageTarget, Self = new PdfPageTarget(number, 0, PdfEditPages.Inverse(document.Frame(index).Display)) } : null);
            return number;
        }
        finally
        {
            content.Release();
        }
    }

    // The encryption dictionary, when the document is encrypted (never itself encrypted).
    private void WriteEncryption()
    {
        if (_security is null)
        {
            return;
        }

        var s = _file.BeginObject(_file.EncryptNumber);
        _security.WriteDictionary(s);
        _file.EndObject();
    }

    // The page tree (one node over every page) and the catalog, with the attached files' names and (when `form`) the form;
    // in object streams when `packed` (and the file packs objects), else each standing on its own.
    private void WriteStructure(bool packed, bool form = false)
    {
        PdfSyntax Begin(int number) => packed ? _file.BeginSmall(number) : _file.BeginObject(number);
        void End()
        {
            if (packed)
            {
                _file.EndSmall();
            }
            else
            {
                _file.EndObject();
            }
        }

        var s = Begin(PagesNumber);
        s.Raw("<</Type/Pages/Count "u8).Integer(_pages.Count).Raw("/Kids["u8);
        for (int i = 0; i < _pages.Count; i++)
        {
            if (i > 0)
            {
                s.Space();
            }

            s.Reference(_pages[i]);
        }

        s.Raw("]>>"u8);
        End();

        s = Begin(CatalogNumber);
        s.Raw("<</Type/Catalog/Pages "u8).Reference(PagesNumber);
        if (_security is not null && _version != "2.0")
        {
            s.Raw(PdfEditPages.Aes256Extension);                               // AES-256 in a file before 2.0: Adobe's extension level 8
        }

        if (_attachments.Count > 0)
        {
            // The name tree's one node: the names in the order of their bytes, as the format asks.
            s.Raw("/Names<</EmbeddedFiles<</Names["u8);
            var sorted = _attachments.Select((a, i) => (a.Name, a.Key, a.Spec, i)).OrderBy(a => a.Key, ByteOrder.Instance).ThenBy(a => a.i).ToList();
            foreach (var (name, _, spec, _) in sorted)
            {
                s.Text(name).Space().Reference(spec).Space();
            }

            s.Raw("]>>>>"u8);
        }

        if (form && _form is { Any: true } fields)
        {
            fields.WriteAcroForm(s);
        }

        CatalogNavigation(s);
        s.Raw(">>"u8);
        End();
    }

    // The information dictionary, with only what was given (and, when not deterministic, the dates of now); 0 when empty.
    private int WriteInfo()
    {
        var info = _info;
        if (!_options.Deterministic)
        {
            var now = DateTimeOffset.Now;
            info = info with { Created = info.Created ?? now, Modified = info.Modified ?? now };
        }

        var entries = InfoEntries(info);
        if (entries.Count == 0)
        {
            return 0;
        }

        int number = _file.Allocate();
        WriteInfo(_file.BeginSmall(number), entries);
        _file.EndSmall();
        return number;
    }

    /// <summary>The entries of an information dictionary saying <paramref name="info"/>: only what is given, the standard keys first, then the others by key.</summary>
    internal static List<(string Key, string? Text, DateTimeOffset? Date)> InfoEntries(DocumentInfo info)
    {
        var entries = new List<(string Key, string? Text, DateTimeOffset? Date)>
        {
            ("Title", info.Title, null), ("Author", info.Author, null), ("Subject", info.Subject, null), ("Keywords", info.Keywords, null),
            ("Creator", info.Creator, null), ("Producer", info.Producer, null), ("CreationDate", null, info.Created), ("ModDate", null, info.Modified),
        };
        foreach (var (key, value) in info.Entries.OrderBy(e => e.Key, StringComparer.Ordinal))
        {
            if (!entries.Any(e => e.Key == key))
            {
                entries.Add((key, value, null));
            }
        }

        entries.RemoveAll(e => e.Text is null && e.Date is null);
        return entries;
    }

    /// <summary>Writes an information dictionary of <paramref name="entries"/> (<see cref="InfoEntries"/>).</summary>
    internal static void WriteInfo(PdfSyntax s, List<(string Key, string? Text, DateTimeOffset? Date)> entries)
    {
        s.Raw("<<"u8);
        foreach (var (key, text, date) in entries)
        {
            if (text is not null)
            {
                s.Name(key).Text(text);
            }
            else if (date is { } when)
            {
                s.Name(key).Text(Date(when));
            }
        }

        s.Raw(">>"u8);
    }

    // A date as PDF writes it: D:YYYYMMDDHHmmSS and the offset from UT (Z, or +HH'mm').
    internal static string Date(DateTimeOffset when)
    {
        var offset = when.Offset;
        string zone = offset == TimeSpan.Zero ? "Z"
            : string.Create(CultureInfo.InvariantCulture, $"{(offset < TimeSpan.Zero ? '-' : '+')}{Math.Abs(offset.Hours):D2}'{Math.Abs(offset.Minutes):D2}'");
        return "D:" + when.ToString("yyyyMMddHHmmss", CultureInfo.InvariantCulture) + zone;
    }

    // The file's identifier: the one given, else (not deterministic, or encrypted) one made now, else none.
    private ReadOnlySpan<byte> Identifier()
    {
        if (!_options.Id.IsEmpty)
        {
            return _options.Id.Span;
        }

        return _options.Deterministic && _security is null ? [] : RandomNumberGenerator.GetBytes(16);
    }

    // Byte strings in the order of their bytes.
    private sealed class ByteOrder : IComparer<byte[]>
    {
        public static ByteOrder Instance { get; } = new();

        public int Compare(byte[]? x, byte[]? y) => x.AsSpan().SequenceCompareTo(y);
    }
}
