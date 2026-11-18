// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace Idrak.Abstraction.Data;

/// <summary>What <see cref="IDocument.ReadPage"/> reads of a page; reading text alone stays as cheap as the format makes it.</summary>
[Flags]
public enum DocumentParts
{
    /// <summary>The page's size only.</summary>
    None = 0,

    /// <summary>Text runs, and the lines and text the layout makes of them.</summary>
    Text = 1,

    /// <summary>Images (their pixels are read when decoded).</summary>
    Images = 2,

    /// <summary>Vector paths and clips.</summary>
    Paths = 4,

    /// <summary>Each run's glyphs and font face (what drawing the text needs).</summary>
    Glyphs = 8,

    /// <summary>Annotations.</summary>
    Annotations = 16,

    /// <summary>Everything.</summary>
    All = Text | Images | Paths | Glyphs | Annotations,
}

/// <summary>
/// What a document says about itself. Every field is optional; a writer writes only those given (no date or name is made
/// up), and a reader gives the document's other entries by key.
/// </summary>
public sealed record DocumentInfo
{
    /// <summary>Nothing given.</summary>
    public static DocumentInfo Empty { get; } = new();

    /// <summary>The title.</summary>
    public string? Title { get; init; }

    /// <summary>The author.</summary>
    public string? Author { get; init; }

    /// <summary>The subject.</summary>
    public string? Subject { get; init; }

    /// <summary>The keywords.</summary>
    public string? Keywords { get; init; }

    /// <summary>The program the document was made in.</summary>
    public string? Creator { get; init; }

    /// <summary>The program that wrote the file.</summary>
    public string? Producer { get; init; }

    /// <summary>When the document was made.</summary>
    public DateTimeOffset? Created { get; init; }

    /// <summary>When it was last changed.</summary>
    public DateTimeOffset? Modified { get; init; }

    /// <summary>The file's format version as it states it ("1.7"), or null.</summary>
    public string? Version { get; init; }

    /// <summary>Whether the file is encrypted (opened with <see cref="DocumentOptions.Password"/> or the empty password).</summary>
    public bool Encrypted { get; init; }

    /// <summary>Whether the file was damaged and rebuilt to be read (its index found by scanning it).</summary>
    public bool Repaired { get; init; }

    /// <summary>The document's other entries, by key, as text.</summary>
    public IReadOnlyDictionary<string, string> Entries { get; init; } = System.Collections.ObjectModel.ReadOnlyDictionary<string, string>.Empty;
}

/// <summary>
/// How a document is opened and its text arranged. Every limit guards against a hostile file: passing one is an
/// <see cref="InvalidDataException"/> naming it, never a hang or an out-of-memory. The size limits default to a share of the
/// memory the machine has free when a document is opened (within <see cref="ComputeResources.CpuMemoryLimit"/>), never a
/// fixed size; the environment variables named below set them, the layout and the reading order.
/// </summary>
public sealed record DocumentOptions
{
    /// <summary>The variable that names the layout (<see cref="Layout"/>): "IDRAK_DOCUMENT_LAYOUT".</summary>
    public const string LayoutVariable = "IDRAK_DOCUMENT_LAYOUT";

    /// <summary>The variable that names the reading order (<see cref="ReadingOrder"/>): "IDRAK_DOCUMENT_READING_ORDER".</summary>
    public const string ReadingOrderVariable = "IDRAK_DOCUMENT_READING_ORDER";

    /// <summary>The variable that sets <see cref="MaxStreamBytes"/> (bytes, or with k, m or g): "IDRAK_DOCUMENT_MAX_STREAM_BYTES".</summary>
    public const string MaxStreamBytesVariable = "IDRAK_DOCUMENT_MAX_STREAM_BYTES";

    /// <summary>The variable that sets <see cref="CacheBytes"/> (bytes, or with k, m or g): "IDRAK_DOCUMENT_CACHE_BYTES".</summary>
    public const string CacheBytesVariable = "IDRAK_DOCUMENT_CACHE_BYTES";

    private readonly long _maxStreamBytes = DocumentSettings.Bytes(MaxStreamBytesVariable) ?? 0;
    private readonly long _cacheBytes = DocumentSettings.Bytes(CacheBytesVariable) ?? 0;
    private readonly long _maxObjects;
    private readonly float _wordGap = 0.2f, _lineTolerance = 0.5f;
    private readonly int _maxDepth = 64;

    /// <summary>
    /// The password to open an encrypted file with (as either of its passwords), tried after the empty one; null tries the
    /// empty one only. When neither opens the file, opening it is an <see cref="InvalidDataException"/> naming this option.
    /// </summary>
    public string? Password { get; init; }

    /// <summary>The layout that arranges a page's runs into lines and text: a <see cref="DocumentTextLayouts"/> name ("lines" unless <see cref="LayoutVariable"/> names another).</summary>
    public string Layout { get; init; } = DocumentSettings.Text(LayoutVariable) ?? DocumentTextLayouts.Lines;

    /// <summary>The order that puts a line's text in logical order: a <see cref="ReadingOrders"/> name ("bidi" unless <see cref="ReadingOrderVariable"/> names another).</summary>
    public string ReadingOrder { get; init; } = DocumentSettings.Text(ReadingOrderVariable) ?? ReadingOrders.Bidi;

    /// <summary>Whether invisible text (a scan's text layer) is read, marked <see cref="DocumentTextRun.Hidden"/> (true); false leaves it out.</summary>
    public bool IncludeHidden { get; init; } = true;

    /// <summary>
    /// The direction the layout gives lines: <see cref="DocumentTextDirection.Auto"/> (each line's own, from its first
    /// strong character by Unicode's rule; a line with none takes the page's), or one for every line.
    /// </summary>
    public DocumentTextDirection Direction { get; init; }

    /// <summary>
    /// The layout's word gap: two runs on a line are separate words where the gap between them is more than this share of
    /// their font size (0.2).
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Negative or not finite.</exception>
    public float WordGap
    {
        get => _wordGap;
        init => _wordGap = value >= 0f && float.IsFinite(value) ? value : throw new ArgumentOutOfRangeException(nameof(WordGap), value, "The word gap is a share of the font size, 0 or more.");
    }

    /// <summary>
    /// The layout's line tolerance: two runs are on one line when their baselines are within this share of the larger
    /// font size (0.5: a superscript stays on its line, the next line of text does not join it).
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Negative or not finite.</exception>
    public float LineTolerance
    {
        get => _lineTolerance;
        init => _lineTolerance = value >= 0f && float.IsFinite(value) ? value : throw new ArgumentOutOfRangeException(nameof(LineTolerance), value, "The line tolerance is a share of the font size, 0 or more.");
    }

    /// <summary>
    /// The most bytes one decoded stream of the file may take (a page's content, an image, a font): a quarter of the memory
    /// free when the document is opened unless set (<see cref="MaxStreamBytesVariable"/>), at most the largest array .NET
    /// makes. A format reads it once, when it opens a document.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Not positive.</exception>
    public long MaxStreamBytes
    {
        get => _maxStreamBytes > 0 ? _maxStreamBytes : Math.Min(Array.MaxLength, DocumentSettings.FreeMemory() / 4);
        init => _maxStreamBytes = value > 0 ? value : throw new ArgumentOutOfRangeException(nameof(MaxStreamBytes), value, $"A stream limit is a number of bytes, more than 0 ({MaxStreamBytesVariable}).");
    }

    /// <summary>
    /// The most objects a document may hold: one per kilobyte of the memory free when it is opened unless set. A format
    /// reads it once, when it opens a document.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Not positive.</exception>
    public long MaxObjects
    {
        get => _maxObjects > 0 ? _maxObjects : Math.Max(1, DocumentSettings.FreeMemory() / 1024);
        init => _maxObjects = value > 0 ? value : throw new ArgumentOutOfRangeException(nameof(MaxObjects), value, "An object limit is more than 0.");
    }

    /// <summary>The deepest nesting a document may hold (containers within containers, content drawn within content): 64 unless set.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Not positive.</exception>
    public int MaxDepth
    {
        get => _maxDepth;
        init => _maxDepth = value > 0 ? value : throw new ArgumentOutOfRangeException(nameof(MaxDepth), value, "A nesting limit is more than 0.");
    }

    /// <summary>
    /// The most bytes a document keeps decoded for reuse (fonts and other parts its pages share, each decoded once): an
    /// eighth of the memory free when it is opened unless set (<see cref="CacheBytesVariable"/>).
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Negative.</exception>
    public long CacheBytes
    {
        get => _cacheBytes > 0 ? _cacheBytes : Math.Max(0, DocumentSettings.FreeMemory() / 8);
        init => _cacheBytes = value >= 0 ? value : throw new ArgumentOutOfRangeException(nameof(CacheBytes), value, $"A cache size is a number of bytes, 0 or more ({CacheBytesVariable}).");
    }

    // The password is never printed.
    private bool PrintMembers(StringBuilder builder)
    {
        builder.Append("Password = ").Append(Password is null ? "none" : "(set)").Append(", Layout = ").Append(Layout).Append(", ReadingOrder = ").Append(ReadingOrder)
            .Append(", IncludeHidden = ").Append(IncludeHidden).Append(", Direction = ").Append(Direction).Append(", WordGap = ").Append(WordGap.ToString(System.Globalization.CultureInfo.InvariantCulture))
            .Append(", LineTolerance = ").Append(LineTolerance.ToString(System.Globalization.CultureInfo.InvariantCulture))
            .Append(", MaxDepth = ").Append(MaxDepth).Append(", MaxStreamBytes = ").Append(_maxStreamBytes > 0 ? _maxStreamBytes.ToString(System.Globalization.CultureInfo.InvariantCulture) : "measured")
            .Append(", MaxObjects = ").Append(_maxObjects > 0 ? _maxObjects.ToString(System.Globalization.CultureInfo.InvariantCulture) : "measured")
            .Append(", CacheBytes = ").Append(_cacheBytes > 0 ? _cacheBytes.ToString(System.Globalization.CultureInfo.InvariantCulture) : "measured");
        return true;
    }
}

/// <summary>
/// An open document: its information, its pages read one at a time and what it carries (files, form fields, bookmarks,
/// signatures). Safe to read from several threads at once, so pages can be read in parallel. Dispose it when done; the
/// stream it was opened from stays the caller's.
/// </summary>
public interface IDocument : IDisposable
{
    /// <summary>The name of the <see cref="IDocumentFormat"/> that opened it.</summary>
    string Format { get; }

    /// <summary>What the document says about itself.</summary>
    DocumentInfo Info { get; }

    /// <summary>The number of pages (one for a format without pages, or one per section).</summary>
    int PageCount { get; }

    /// <summary>
    /// Page <paramref name="index"/> (from 0) with <paramref name="parts"/> read. One damaged page fails alone (an
    /// <see cref="InvalidDataException"/>); a damaged part of a page is skipped and counted in its issues.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">No such page.</exception>
    /// <exception cref="InvalidDataException">The page cannot be read, or a limit of the options was passed (the message names it).</exception>
    DocumentPage ReadPage(int index, DocumentParts parts);

    /// <summary>The files the document carries, each read when asked.</summary>
    IReadOnlyList<DocumentAttachment> Attachments { get; }

    /// <summary>
    /// Page <paramref name="index"/> drawn to pixels: read with the parts drawing needs and drawn by
    /// <see cref="DocumentPageDrawers"/> (<see cref="DocumentPageDrawers.DrawPage"/> does it for a format).
    /// </summary>
    /// <exception cref="NotSupportedException">The format's pages cannot be drawn (the message says why), or no drawer is registered.</exception>
    DocumentDrawing DrawPage(int index, DocumentDrawOptions options);

    /// <summary>The form's fields (empty for a document without a form, or a format that reads none).</summary>
    IReadOnlyList<DocumentField> Fields { get; }

    /// <summary>The bookmarks, as a tree (empty when there are none).</summary>
    IReadOnlyList<DocumentOutline> Outline { get; }

    /// <summary>The label page <paramref name="index"/> shows ("iv", "A-3"), or null when the document gives none.</summary>
    string? PageLabel(int index);

    /// <summary>The signatures, in the order they were made (empty when there are none).</summary>
    IReadOnlyList<DocumentSignature> Signatures { get; }
}

/// <summary>
/// Reads one document format; register it with <see cref="DocumentFormats.Register"/>. A format decides from a file's first
/// bytes whether the file is in its format, so the extension only decides which files a folder picks up.
/// </summary>
public interface IDocumentFormat
{
    /// <summary>The format's name (names ignore case).</summary>
    string Name { get; }

    /// <summary>The file extensions of the format, with the dot.</summary>
    IReadOnlyCollection<string> Extensions { get; }

    /// <summary>The media types of the format.</summary>
    IReadOnlyCollection<string> MediaTypes { get; }

    /// <summary>Whether a file starting with <paramref name="header"/> (its first <see cref="DocumentFormats.HeaderBytes"/>, or all of a smaller file) is in this format.</summary>
    bool Accepts(ReadOnlySpan<byte> header);

    /// <summary>
    /// Opens the document <paramref name="stream"/> holds, from its start: what is read when (the index, pages, parts) is
    /// the format's to choose, within <paramref name="options"/>' limits. The stream stays the caller's, to dispose after
    /// the document.
    /// </summary>
    /// <param name="stream">A seekable stream of the file.</param>
    /// <param name="name">A name for messages (the file's).</param>
    /// <param name="options">How to open it.</param>
    /// <exception cref="InvalidDataException">
    /// The file is damaged beyond repair, a limit was passed, or it is encrypted and <see cref="DocumentOptions.Password"/>
    /// does not open it (the message names the limit or the option).
    /// </exception>
    /// <exception cref="NotSupportedException">A valid file in a variant the format does not read (the message names it).</exception>
    IDocument Open(Stream stream, string name, DocumentOptions options);
}

/// <summary>
/// The document formats, by name (ignoring case), the most recently registered asked first. A format is chosen by a
/// file's first bytes. None is built in yet; a plug-in registers its own (an Office plug-in is to come) with
/// <see cref="Register"/>, and one registered under a library name takes its place, which stays behind it as its fallback
/// or its comparison (<see cref="SetPolicy"/>).
/// </summary>
public static class DocumentFormats
{
    /// <summary>The bytes of a file's start that <see cref="IDocumentFormat.Accepts"/> receives.</summary>
    public const int HeaderBytes = 4096;

    private static readonly SlotTable<string, IDocumentFormat> Registry =
        new(nameof(DocumentFormats), (slot, app, library) => new GuardedFormat(slot, app, library), StringComparer.OrdinalIgnoreCase, newestFirst: true);

    static DocumentFormats() => LibraryDefaults.Ensure(typeof(DocumentFormats));

    /// <summary>
    /// Registers <paramref name="format"/>: one of a registered name takes its place (the library's stays behind it, see
    /// <see cref="SetPolicy"/>), a new one is asked first.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IDocumentFormat format)
    {
        ArgumentNullException.ThrowIfNull(format);
        ArgumentException.ThrowIfNullOrWhiteSpace(format.Name);
        Registry.Register(format.Name, format, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's format <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered format names, the first asked first.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The extensions of the registered formats (with the dot, lower case).</summary>
    public static IReadOnlyCollection<string> Extensions => [.. Registry.Values.SelectMany(f => f.Extensions).Select(e => e.ToLowerInvariant()).Distinct()];

    /// <summary>The media types of the registered formats.</summary>
    public static IReadOnlyCollection<string> MediaTypes => [.. Registry.Values.SelectMany(f => f.MediaTypes).Distinct(StringComparer.OrdinalIgnoreCase)];

    /// <summary>The format registered as <paramref name="name"/>, or null.</summary>
    public static IDocumentFormat? Find(ReadOnlySpan<char> name) => Registry.Find(name);

    /// <summary>The format registered as <paramref name="name"/> (any case).</summary>
    /// <exception cref="NotRegisteredException">None is: the message names the registered formats and what to do.</exception>
    public static IDocumentFormat Get(string name) => Registry.Find(name)
        ?? throw new NotRegisteredException(nameof(DocumentFormats), name, Registry.Keys,
            $"No document format '{name}' is registered (registered: {Listed()}); add it with DocumentFormats.Register, or reference a package that registers it (Office documents: a plug-in to come).");

    /// <summary>The library's format <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IDocumentFormat? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the format <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>
    /// What happens when the app's format <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set;
    /// <see cref="SlotPolicy.FallBack"/>: the library's format opens the file and answers that call), or whether it only
    /// runs beside the library's (<see cref="SlotPolicy.Shadow"/>: the library's document answers, each page's runs, text
    /// and images compared on a sample of calls).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(name, policy, shadowRate);

    /// <summary>Whether a registered format lists the extension of <paramref name="path"/>.</summary>
    public static bool CanOpen(string path)
    {
        string extension = Path.GetExtension(path);
        return extension.Length > 0 && Registry.Values.Any(f => f.Extensions.Contains(extension, StringComparer.OrdinalIgnoreCase));
    }

    /// <summary>The first format that accepts a file starting with <paramref name="header"/>, or null when none does.</summary>
    public static IDocumentFormat? Detect(ReadOnlySpan<byte> header)
    {
        header = header[..Math.Min(header.Length, HeaderBytes)];
        foreach (var format in Registry.Values)
        {
            if (format.Accepts(header))
            {
                return format;
            }
        }

        return null;
    }

    /// <summary>Opens the document file <paramref name="path"/> with the format its first bytes name; the file is closed with the document.</summary>
    /// <exception cref="InvalidDataException">No registered format reads the file, or it is damaged.</exception>
    public static IDocument Open(string path, DocumentOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 4096, FileOptions.RandomAccess);
        try
        {
            return new OwningDocument(Open(stream, path, options), stream);
        }
        catch
        {
            stream.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Opens the document <paramref name="stream"/> holds (seekable, the file from its start) with the format its first
    /// bytes name. The stream stays the caller's, to dispose after the document.
    /// </summary>
    /// <param name="stream">The file.</param>
    /// <param name="name">A name for messages (the file's).</param>
    /// <param name="options">How to open it; null for the defaults.</param>
    /// <exception cref="ArgumentException">The stream cannot seek (copy it into memory or a file first).</exception>
    /// <exception cref="InvalidDataException">No registered format reads the file, or it is damaged.</exception>
    public static IDocument Open(Stream stream, string name, DocumentOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(name);
        if (!stream.CanSeek || !stream.CanRead)
        {
            throw new ArgumentException($"{name}: a document is opened from a stream that reads and seeks; copy it into memory or a file first.", nameof(stream));
        }

        options ??= new DocumentOptions();
        byte[] head = ArrayPool<byte>.Shared.Rent(HeaderBytes);
        IDocumentFormat? format;
        try
        {
            stream.Position = 0;
            int length = stream.ReadAtLeast(head.AsSpan(0, HeaderBytes), HeaderBytes, throwOnEndOfStream: false);
            format = Detect(head.AsSpan(0, length));
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(head);
        }

        if (format is null)
        {
            throw new InvalidDataException($"{name}: not a document of the registered formats ({Listed()}); register one with DocumentFormats.Register.");
        }

        stream.Position = 0;
        return OpenWith(format, stream, name, options);
    }

    /// <summary>Opens a document held in memory (kept by the document, not copied when it is an array's) with the format its first bytes name.</summary>
    /// <exception cref="InvalidDataException">No registered format reads the bytes, or they are damaged.</exception>
    public static IDocument Open(ReadOnlyMemory<byte> file, string name, DocumentOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(name);
        var stream = MemoryMarshal.TryGetArray(file, out var segment)
            ? new MemoryStream(segment.Array!, segment.Offset, segment.Count, writable: false, publiclyVisible: true)
            : new MemoryStream(file.ToArray(), writable: false);
        return Open(stream, name, options);
    }

    // Opens with one format, its errors worded with the file's name.
    private static IDocument OpenWith(IDocumentFormat format, Stream stream, string name, DocumentOptions options)
    {
        try
        {
            return format.Open(stream, name, options);
        }
        catch (Exception e) when (e is IndexOutOfRangeException or ArgumentOutOfRangeException or EndOfStreamException
                                      || (e is ArgumentException && e is not ArgumentNullException))
        {
            throw new InvalidDataException($"{name}: a damaged {format.Name} document ({e.Message}).", e);
        }
        catch (InvalidDataException e) when (!e.Message.StartsWith(name, StringComparison.Ordinal))
        {
            throw new InvalidDataException($"{name}: {e.Message}", e);
        }
    }

    private static string Listed() => Registry.Keys.Count == 0 ? "none" : string.Join(", ", Registry.Keys);

    // A document the registry opened from a file it owns: the file is closed after the document.
    private sealed class OwningDocument(IDocument inner, Stream stream) : IDocument
    {
        public string Format => inner.Format;

        public DocumentInfo Info => inner.Info;

        public int PageCount => inner.PageCount;

        public IReadOnlyList<DocumentAttachment> Attachments => inner.Attachments;

        public IReadOnlyList<DocumentField> Fields => inner.Fields;

        public IReadOnlyList<DocumentOutline> Outline => inner.Outline;

        public IReadOnlyList<DocumentSignature> Signatures => inner.Signatures;

        public DocumentPage ReadPage(int index, DocumentParts parts) => inner.ReadPage(index, parts);

        public DocumentDrawing DrawPage(int index, DocumentDrawOptions options) => inner.DrawPage(index, options);

        public string? PageLabel(int index) => inner.PageLabel(index);

        public void Dispose()
        {
            try
            {
                inner.Dispose();
            }
            finally
            {
                stream.Dispose();
            }
        }
    }

    // An app's format over a library one. Each document the guard opens reads the caller's stream through a view of its
    // own (its own position, reads under one lock), so the app's document and the library's, opened when a call falls back
    // or is compared, never move each other's place in it.
    private sealed class GuardedFormat(Slot slot, IDocumentFormat app, IDocumentFormat library) : IDocumentFormat
    {
        public string Name => app.Name;

        public IReadOnlyCollection<string> Extensions => app.Extensions;

        public IReadOnlyCollection<string> MediaTypes => app.MediaTypes;

        public bool Accepts(ReadOnlySpan<byte> header)
        {
            if (slot.Policy == SlotPolicy.Shadow)
            {
                var run = slot.Shadow();
                bool answer = library.Accepts(header);
                if (run is not null)
                {
                    run.Answered();
                    try
                    {
                        run.Done(Comparisons.Exact(answer, app.Accepts(header)));
                    }
                    catch (Exception e) when (e is not OperationCanceledException)
                    {
                        run.Failed(e);
                    }
                }

                return answer;
            }

            try
            {
                return app.Accepts(header);
            }
            catch (Exception e) when (slot.Failed(e))
            {
                return library.Accepts(header);
            }
        }

        public IDocument Open(Stream stream, string name, DocumentOptions options)
        {
            var gate = new Lock();
            IDocument OpenApp() => app.Open(new StreamView(stream, gate), name, options);
            IDocument OpenLibrary() => library.Open(new StreamView(stream, gate), name, options);
            if (slot.Policy == SlotPolicy.Shadow)
            {
                return new GuardedDocument(slot, null, OpenApp, OpenLibrary(), OpenLibrary);
            }

            try
            {
                return new GuardedDocument(slot, OpenApp(), OpenApp, null, OpenLibrary);
            }
            catch (Exception e) when (slot.Failed(e))
            {
                return new GuardedDocument(slot, null, OpenApp, OpenLibrary(), OpenLibrary);
            }
        }
    }

    // A document opened by the guard: every call goes to the slot (Slot.Call), the app's document and the library's each
    // opened on first need and disposed with this one.
    private sealed class GuardedDocument(Slot slot, IDocument? app, Func<IDocument> openApp, IDocument? library, Func<IDocument> openLibrary) : IDocument
    {
        private readonly Lock _gate = new();
        private IDocument? _app = app, _library = library;
        private bool _disposed;

        private IDocument App => Opened(ref _app, openApp);

        private IDocument Library => Opened(ref _library, openLibrary);

        public string Format => slot.Call(() => App.Format, () => Library.Format, (a, b) => Comparisons.Exact(a, b));

        public DocumentInfo Info => slot.Call(() => App.Info, () => Library.Info, (a, b) => Comparisons.Exact(a.Title, b.Title) ?? Comparisons.Exact(a.Version, b.Version));

        public int PageCount => slot.Call(() => App.PageCount, () => Library.PageCount, (a, b) => Comparisons.Exact(a, b));

        public IReadOnlyList<DocumentAttachment> Attachments =>
            slot.Call(() => App.Attachments, () => Library.Attachments, (a, b) => Comparisons.Sequences(a.Select(x => x.Name).ToArray(), b.Select(x => x.Name).ToArray()));

        public IReadOnlyList<DocumentField> Fields =>
            slot.Call(() => App.Fields, () => Library.Fields, (a, b) => Comparisons.Sequences(a.Select(x => $"{x.Name}={x.Value}").ToArray(), b.Select(x => $"{x.Name}={x.Value}").ToArray()));

        public IReadOnlyList<DocumentOutline> Outline =>
            slot.Call(() => App.Outline, () => Library.Outline, (a, b) => Comparisons.Sequences(a.Select(x => x.Title).ToArray(), b.Select(x => x.Title).ToArray()));

        public IReadOnlyList<DocumentSignature> Signatures =>
            slot.Call(() => App.Signatures, () => Library.Signatures, (a, b) => Comparisons.Sequences(a.Select(x => x.Field).ToArray(), b.Select(x => x.Field).ToArray()));

        public DocumentPage ReadPage(int index, DocumentParts parts) => slot.Call(() => App.ReadPage(index, parts), () => Library.ReadPage(index, parts), (a, b) => ComparePages(a, b));

        public DocumentDrawing DrawPage(int index, DocumentDrawOptions options) =>
            slot.Call(() => App.DrawPage(index, options), () => Library.DrawPage(index, options), (a, b) => DocumentPageDrawers.CompareDrawings(a, b));

        public string? PageLabel(int index) => slot.Call(() => App.PageLabel(index), () => Library.PageLabel(index), (a, b) => Comparisons.Exact(a, b));

        public void Dispose()
        {
            IDocument? a, l;
            lock (_gate)
            {
                (a, l, _app, _library, _disposed) = (_app, _library, null, null, true);
            }

            try
            {
                a?.Dispose();
            }
            finally
            {
                l?.Dispose();
            }
        }

        private IDocument Opened(ref IDocument? document, Func<IDocument> open)
        {
            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                return document ??= open();
            }
        }

        // The library's page against the app's: the runs' text and boxes, the text, the images.
        private static string? ComparePages(DocumentPage expected, DocumentPage actual) =>
            Comparisons.Exact((expected.Index, expected.Width, expected.Height), (actual.Index, actual.Width, actual.Height))
            ?? Comparisons.Sequences(expected.Runs.Select(r => (r.Text, r.Box)).ToArray(), actual.Runs.Select(r => (r.Text, r.Box)).ToArray())
            ?? Comparisons.Difference(expected.Text, actual.Text)
            ?? Comparisons.Sequences(expected.Images.Select(i => (i.Encoding, i.PixelWidth, i.PixelHeight, i.Box)).ToArray(),
                actual.Images.Select(i => (i.Encoding, i.PixelWidth, i.PixelHeight, i.Box)).ToArray());
    }

    // A view of a seekable stream with its own position: reads go through one lock, each moving the shared stream to the
    // view's place first.
    private sealed class StreamView(Stream inner, Lock gate) : Stream
    {
        private long _position;

        public override bool CanRead => true;

        public override bool CanSeek => true;

        public override bool CanWrite => false;

        public override long Length
        {
            get
            {
                lock (gate)
                {
                    return inner.Length;
                }
            }
        }

        public override long Position
        {
            get => _position;
            set => _position = value >= 0 ? value : throw new ArgumentOutOfRangeException(nameof(value));
        }

        public override int Read(byte[] buffer, int offset, int count) => Read(buffer.AsSpan(offset, count));

        public override int Read(Span<byte> buffer)
        {
            lock (gate)
            {
                inner.Position = _position;
                int read = inner.Read(buffer);
                _position += read;
                return read;
            }
        }

        public override long Seek(long offset, SeekOrigin origin) => Position = origin switch
        {
            SeekOrigin.Begin => offset,
            SeekOrigin.Current => _position + offset,
            _ => Length + offset,
        };

        public override void Flush()
        {
        }

        public override void SetLength(long value) => throw new NotSupportedException("A document's stream is read, not written.");

        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException("A document's stream is read, not written.");
    }
}

// The document settings' environment variables, read the same way everywhere: text trimmed (empty is unset), numbers
// invariant, a value that cannot be read refused naming the variable.
internal static class DocumentSettings
{
    public static string? Text(string variable) => Environment.GetEnvironmentVariable(variable)?.Trim() is { Length: > 0 } value ? value : null;

    // Bytes: a whole number, or one followed by k, m or g (KiB, MiB, GiB; any case, an optional "b" or "ib" after it).
    public static long? Bytes(string variable)
    {
        if (Text(variable) is not { } text)
        {
            return null;
        }

        var span = text.AsSpan();
        long unit = 1;
        if (span.EndsWith("ib", StringComparison.OrdinalIgnoreCase))
        {
            span = span[..^2];
        }
        else if (span.EndsWith("b", StringComparison.OrdinalIgnoreCase))
        {
            span = span[..^1];
        }

        if (span.Length > 0 && char.ToLowerInvariant(span[^1]) is 'k' or 'm' or 'g')
        {
            unit = char.ToLowerInvariant(span[^1]) switch { 'k' => 1L << 10, 'm' => 1L << 20, _ => 1L << 30 };
            span = span[..^1];
        }

        return long.TryParse(span.Trim(), System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out long value) && value > 0
               && value <= long.MaxValue / unit
            ? value * unit
            : throw new ArgumentException($"{variable}={text}: a number of bytes more than 0 (such as 268435456, 256m or 1g).", variable);
    }

    public static int? Int(string variable, int min, int max)
    {
        if (Text(variable) is not { } text)
        {
            return null;
        }

        return int.TryParse(text, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out int value) && value >= min && value <= max
            ? value
            : throw new ArgumentException($"{variable}={text}: a whole number from {min} to {max}.", variable);
    }

    public static float? Float(string variable, float min, float max)
    {
        if (Text(variable) is not { } text)
        {
            return null;
        }

        return float.TryParse(text, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float value) && value >= min && value <= max
            ? value
            : throw new ArgumentException($"{variable}={text}: a number from {min.ToString(System.Globalization.CultureInfo.InvariantCulture)} to {max.ToString(System.Globalization.CultureInfo.InvariantCulture)}.", variable);
    }

    // One of an enum's names (any case); "0", "off", "false" or "no" mean false and "1", "on", "true" or "yes" true for a switch.
    public static TEnum? Choice<TEnum>(string variable) where TEnum : struct, Enum
    {
        if (Text(variable) is not { } text)
        {
            return null;
        }

        return Enum.TryParse<TEnum>(text, ignoreCase: true, out var value) && Enum.IsDefined(value) && !char.IsAsciiDigit(text[0])
            ? value
            : throw new ArgumentException($"{variable}={text}: one of {string.Join(", ", Enum.GetNames<TEnum>().Select(n => n.ToLowerInvariant()))}.", variable);
    }

    public static bool? Switch(string variable)
    {
        if (Text(variable) is not { } text)
        {
            return null;
        }

        if (ComputeResources.SwitchedOff(text))
        {
            return false;
        }

        return text.Equals("1", StringComparison.Ordinal) || text.Equals("on", StringComparison.OrdinalIgnoreCase) || text.Equals("true", StringComparison.OrdinalIgnoreCase)
               || text.Equals("yes", StringComparison.OrdinalIgnoreCase)
            ? true
            : throw new ArgumentException($"{variable}={text}: on or off (1, on, true, yes; 0, off, false, no).", variable);
    }

    // The memory free now, as the CPU device reports it (within ComputeResources.CpuMemoryLimit).
    public static long FreeMemory()
    {
        long free = Device.Cpu.Backend.AvailableMemory() ?? GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
        if (ComputeResources.CpuMemoryLimit is { } limit)
        {
            free = Math.Min(free, limit);
        }

        return Math.Max(1L << 20, free);
    }
}
