// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Security.Cryptography;
using System.Text;

namespace Idrak.Data.Pdf;

/// <summary>
/// The library's PDF writer, registered in <see cref="DocumentWriters"/> as "pdf": pages of images, vector paths and text
/// written as they finish, the reader's object model and filters shared. The file: the header, objects written on spans
/// (numbers invariant, strings escaped, names with <c>#xx</c>), streams Flate-compressed at
/// <see cref="DocumentWriteOptions.Compression"/> (0 stores them), small objects in object streams and a cross-reference
/// stream (a classic table for <see cref="DocumentWriteOptions.Version"/> before 1.5, 1.7 unless set), no date or
/// identifier unless given (<see cref="DocumentWriteOptions.Deterministic"/>): the same input writes the same bytes. Each
/// page's images go to the file as they are given, its content stream and page object when the next page starts; the page
/// tree, the fonts (one composite font a face, its program subset through <see cref="FontWriters"/>, a ToUnicode map from
/// the runs' text) and the cross-reference at <see cref="IDocumentBuilder.Finish"/>. A builder disposed without it writes
/// the page in hand, the page tree, the catalog and a trailer naming it, and no cross-reference: the reader rebuilds it
/// and says the file was repaired (fonts, written at the end, are missing). Forms, annotations, links, bookmarks, page labels, copied pages, encryption, signing and editing come with
/// the later phases of the documents plan (D9-D12), each a <see cref="NotSupportedException"/> naming its phase until it lands.
/// </summary>
internal sealed class PdfWriter : IDocumentWriter
{
    public string Name => PdfFormat.FormatName;

    public IReadOnlyCollection<string> Extensions { get; } = [".pdf"];

    public IReadOnlyCollection<string> MediaTypes { get; } = ["application/pdf"];

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

    public IDocumentEditor Edit(IDocument source, Stream destination, DocumentWriteOptions options) => throw PdfDocumentBuilder.NotYet("editing documents", "D11");
}

/// <summary>A PDF document being written (see <see cref="PdfWriter"/>). Not safe from several threads.</summary>
internal sealed class PdfDocumentBuilder : IDocumentBuilder
{
    private const int CatalogNumber = 1, PagesNumber = 2;

    private readonly Stream _destination;
    private readonly DocumentWriteOptions _options;
    private readonly PdfFileWriter _file;
    private readonly bool _classic;
    private readonly List<int> _pages = [];
    private readonly Dictionary<IFontFace, PdfWriterFont> _fonts = new(ReferenceEqualityComparer.Instance);
    private readonly List<PdfWriterFont> _fontOrder = [];
    private readonly List<(string Name, byte[] Key, int Spec)> _attachments = [];
    private PdfPageBuilder? _page;
    private DocumentInfo _info = DocumentInfo.Empty;
    private bool _finished, _disposed, _closing;

    public PdfDocumentBuilder(Stream destination, DocumentWriteOptions options)
    {
        string version = options.Version ?? "1.7";
        if (!(version is "2.0" || (version.Length == 3 && version[0] == '1' && version[1] == '.' && version[2] is >= '0' and <= '7')))
        {
            throw new ArgumentException($"A PDF version is 1.0 to 1.7 or 2.0, not '{version}' (DocumentWriteOptions.Version).", nameof(options));
        }

        (_destination, _options) = (destination, options);
        _classic = string.CompareOrdinal(version, "1.5") < 0;                  // cross-reference and object streams came with 1.5
        _file = new PdfFileWriter(destination, version, options.Compression, packing: !_classic && options.ObjectStreams);
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
        _page = new PdfPageBuilder(this, _pages.Count, width, height);
        return _page;
    }

    public IDocumentPageBuilder CopyPage(IDocument from, int index) => throw NotYet("copying pages from another document", "D11");

    public void Attach(string name, string mediaType, Stream content)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(content);
        ThrowIfDone();
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

            _file.EndStream(data.Span, deflate: false);                        // deflated as it was read
        }

        var f = _file.BeginSmall(spec);
        f.Raw("<</Type/Filespec/F "u8).Text(name).Raw("/UF "u8).Text(name).Raw("/EF<</F "u8).Reference(file).Raw(">>>>"u8);
        _file.EndSmall();
        _attachments.Add((name, StringBytes(name), spec));
    }

    // A text string's bytes as Text writes them (ASCII, or UTF-16BE after its mark): what a name tree is sorted by.
    private static byte[] StringBytes(string text) =>
        text.All(c => c is >= ' ' and <= '~') ? Encoding.ASCII.GetBytes(text) : [0xFE, 0xFF, .. Encoding.BigEndianUnicode.GetBytes(text)];

    public void AddField(DocumentField field) => throw NotYet("form fields", "D9");

    public void AddOutline(DocumentOutline outline) => throw NotYet("bookmarks", "D10");

    public void SetPageLabels(IReadOnlyList<DocumentPageLabel> labels) => throw NotYet("page labels", "D10");

    public void Encrypt(DocumentEncryption encryption) => throw NotYet("encryption", "D11");

    public void Sign(string field, SigningOptions options) => throw NotYet("signatures", "D12");

    public void Finish()
    {
        ThrowIfDone();
        _closing = true;
        try
        {
            FinishPage();
            foreach (var font in _fontOrder)
            {
                font.Write(_file, _options.EmbedFonts);
            }

            WriteStructure(packed: true);
            int info = WriteInfo();
            _file.Finish(CatalogNumber, info, Identifier(), _classic);
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
                _file.Unfinished(CatalogNumber);
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
            _file.Dispose();
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
            _pages.Add(page.Write(_file, PagesNumber));
            _file.Flush();                                                     // the page goes to the destination as it finishes
        }
    }

    // The page tree (one node over every page) and the catalog, with the attached files' names; in object streams when
    // `packed` (and the file packs objects), else each standing on its own.
    private void WriteStructure(bool packed)
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

        if (entries.All(e => e.Text is null && e.Date is null))
        {
            return 0;
        }

        int number = _file.Allocate();
        var s = _file.BeginSmall(number);
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
        _file.EndSmall();
        return number;
    }

    // A date as PDF writes it: D:YYYYMMDDHHmmSS and the offset from UT (Z, or +HH'mm').
    private static string Date(DateTimeOffset when)
    {
        var offset = when.Offset;
        string zone = offset == TimeSpan.Zero ? "Z"
            : string.Create(CultureInfo.InvariantCulture, $"{(offset < TimeSpan.Zero ? '-' : '+')}{Math.Abs(offset.Hours):D2}'{Math.Abs(offset.Minutes):D2}'");
        return "D:" + when.ToString("yyyyMMddHHmmss", CultureInfo.InvariantCulture) + zone;
    }

    // The file's identifier: the one given, else (not deterministic) one made now, else none.
    private ReadOnlySpan<byte> Identifier()
    {
        if (!_options.Id.IsEmpty)
        {
            return _options.Id.Span;
        }

        return _options.Deterministic ? [] : RandomNumberGenerator.GetBytes(16);
    }

    // Byte strings in the order of their bytes.
    private sealed class ByteOrder : IComparer<byte[]>
    {
        public static ByteOrder Instance { get; } = new();

        public int Compare(byte[]? x, byte[]? y) => x.AsSpan().SequenceCompareTo(y);
    }
}
