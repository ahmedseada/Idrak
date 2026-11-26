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

    /// <summary>Whether the file is encrypted (opened with <see cref="DocumentOptions.Password"/> or the empty password; <see cref="Security"/> says how, and by which).</summary>
    public bool Encrypted { get; init; }

    /// <summary>
    /// How the file is encrypted (its handler, revision, key length, ciphers and permissions; never a key, salt or
    /// initialization vector), or null when it is not or the format does not say. Read only: a writer encrypts through
    /// <see cref="IDocumentBuilder.Encrypt"/> or <see cref="IDocumentEditor.Encrypt"/>, never from this.
    /// </summary>
    public DocumentSecurity? Security { get; init; }

    /// <summary>Whether the file was damaged and rebuilt to be read (its index found by scanning it).</summary>
    public bool Repaired { get; init; }

    /// <summary>The document's other entries, by key, as text.</summary>
    public IReadOnlyDictionary<string, string> Entries { get; init; } = System.Collections.ObjectModel.ReadOnlyDictionary<string, string>.Empty;
}

/// <summary>
/// How a document is opened and its text arranged. Every limit guards against a hostile file: passing one is an
/// <see cref="InvalidDataException"/> naming it, never a hang or an out-of-memory. The size limits (a stream, the objects,
/// the cache) default to a share of the memory the machine has free when a document is opened (within
/// <see cref="ComputeResources.CpuMemoryLimit"/>), never a fixed size; the nesting limit is a count of levels, and each
/// level is also checked against the stack the reading thread has left. Each is set by its option or its environment
/// variable (named below, as are the layout's and the reading order's), the option winning.
/// </summary>
public sealed record DocumentOptions
{
    /// <summary>The variable that names the layout (<see cref="Layout"/>): "IDRAK_DOCUMENT_LAYOUT".</summary>
    public const string LayoutVariable = "IDRAK_DOCUMENT_LAYOUT";

    /// <summary>The variable that sets <see cref="MaxStreamBytes"/> (bytes, or with k, m or g): "IDRAK_DOCUMENT_MAX_STREAM_BYTES".</summary>
    public const string MaxStreamBytesVariable = "IDRAK_DOCUMENT_MAX_STREAM_BYTES";

    /// <summary>The variable that sets <see cref="CacheBytes"/> (bytes, or with k, m or g): "IDRAK_DOCUMENT_CACHE_BYTES".</summary>
    public const string CacheBytesVariable = "IDRAK_DOCUMENT_CACHE_BYTES";

    /// <summary>The variable that sets <see cref="MaxObjects"/> (a whole number): "IDRAK_DOCUMENT_MAX_OBJECTS".</summary>
    public const string MaxObjectsVariable = "IDRAK_DOCUMENT_MAX_OBJECTS";

    /// <summary>The variable that sets <see cref="MaxDepth"/> (a whole number of levels): "IDRAK_DOCUMENT_MAX_DEPTH".</summary>
    public const string MaxDepthVariable = "IDRAK_DOCUMENT_MAX_DEPTH";

    /// <summary>The nesting limit when neither the option nor <see cref="MaxDepthVariable"/> sets one: 64 levels.</summary>
    public const int DefaultMaxDepth = 64;

    private readonly long _maxStreamBytes = DocumentSettings.Bytes(MaxStreamBytesVariable) ?? 0;
    private readonly long _cacheBytes = DocumentSettings.Bytes(CacheBytesVariable, zero: true) ?? -1;   // -1: measured
    private readonly long _maxObjects = DocumentSettings.Count(MaxObjectsVariable) ?? 0;
    private readonly float _wordGap = 0.2f, _lineTolerance = 0.5f;
    private readonly int _maxDepth = DocumentSettings.Int(MaxDepthVariable, 1, MostNesting) ?? DefaultMaxDepth;

    /// <summary>
    /// The password to open an encrypted file with: tried as its owner password, then as its user password (so the owner
    /// password opens as the owner, also where the user password is empty, and <see cref="DocumentSecurity.OwnerAccess"/>
    /// says so). When it is neither, the empty user password is tried: the file then opens as its user, and every page
    /// read says so (<see cref="DocumentIssue.WrongPassword"/>). Null (or empty) tries the empty password only. When
    /// nothing opens the file, opening it is an <see cref="InvalidDataException"/> naming this option.
    /// </summary>
    public string? Password { get; init; }

    /// <summary>The layout that arranges a page's runs into lines and text: a <see cref="DocumentTextLayouts"/> name ("lines" unless <see cref="LayoutVariable"/> names another).</summary>
    public string Layout { get; init; } = DocumentSettings.Text(LayoutVariable) ?? DocumentTextLayouts.Lines;

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
    /// The most objects a document may hold: one per kilobyte of the memory free when it is opened unless set
    /// (<see cref="MaxObjectsVariable"/>). A format reads it once, when it opens a document.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Not positive.</exception>
    public long MaxObjects
    {
        get => _maxObjects > 0 ? _maxObjects : Math.Max(1, DocumentSettings.FreeMemory() / 1024);
        init => _maxObjects = value > 0 ? value : throw new ArgumentOutOfRangeException(nameof(MaxObjects), value, "An object limit is more than 0.");
    }

    /// <summary>
    /// The deepest nesting a document may hold (containers within containers, content drawn within content):
    /// <see cref="DefaultMaxDepth"/> unless set (<see cref="MaxDepthVariable"/>). A count of levels, not a size: a reader also
    /// checks each level against the stack its thread has left, and fails naming this limit when a level would not fit.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Not positive, or more than <see cref="MostNesting"/>.</exception>
    public int MaxDepth
    {
        get => _maxDepth;
        init => _maxDepth = value is > 0 and <= MostNesting ? value
            : throw new ArgumentOutOfRangeException(nameof(MaxDepth), value, $"A nesting limit is more than 0 and at most {MostNesting} ({MaxDepthVariable}).");
    }

    /// <summary>The highest <see cref="MaxDepth"/> takes: 1,048,576 levels, far deeper than a thread's stack holds (a reader's counts of nested parts stay within a whole number).</summary>
    public const int MostNesting = 1 << 20;

    /// <summary>
    /// The most bytes a document keeps decoded for reuse (fonts and other parts its pages share, each decoded once): an
    /// eighth of the memory free when it is opened unless set (<see cref="CacheBytesVariable"/>); 0 keeps nothing, every
    /// part read again when asked.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Negative.</exception>
    public long CacheBytes
    {
        get => _cacheBytes >= 0 ? _cacheBytes : Math.Max(0, DocumentSettings.FreeMemory() / 8);
        init => _cacheBytes = value >= 0 ? value : throw new ArgumentOutOfRangeException(nameof(CacheBytes), value, $"A cache size is a number of bytes, 0 or more ({CacheBytesVariable}).");
    }

    // The password is never printed.
    private bool PrintMembers(StringBuilder builder)
    {
        builder.Append("Password = ").Append(Password is null ? "none" : "(set)").Append(", Layout = ").Append(Layout)
            .Append(", IncludeHidden = ").Append(IncludeHidden).Append(", Direction = ").Append(Direction).Append(", WordGap = ").Append(WordGap.ToString(System.Globalization.CultureInfo.InvariantCulture))
            .Append(", LineTolerance = ").Append(LineTolerance.ToString(System.Globalization.CultureInfo.InvariantCulture))
            .Append(", MaxDepth = ").Append(MaxDepth).Append(", MaxStreamBytes = ").Append(_maxStreamBytes > 0 ? _maxStreamBytes.ToString(System.Globalization.CultureInfo.InvariantCulture) : "measured")
            .Append(", MaxObjects = ").Append(_maxObjects > 0 ? _maxObjects.ToString(System.Globalization.CultureInfo.InvariantCulture) : "measured")
            .Append(", CacheBytes = ").Append(_cacheBytes >= 0 ? _cacheBytes.ToString(System.Globalization.CultureInfo.InvariantCulture) : "measured");
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
    /// <see cref="DocumentPageDrawers"/> (<c>DrawPage</c> does it for a format).
    /// </summary>
    /// <exception cref="NotSupportedException">The format's pages cannot be drawn (the message says why), or the options
    /// name a drawer other than the library's (by its name: not on the new design yet).</exception>
    DocumentDrawing DrawPage(int index, DocumentDrawOptions options);

    /// <summary>The form's fields (empty for a document without a form, or a format that reads none).</summary>
    IReadOnlyList<DocumentField> Fields { get; }

    /// <summary>The bookmarks, as a tree (empty when there are none).</summary>
    IReadOnlyList<DocumentOutline> Outline { get; }

    /// <summary>The label page <paramref name="index"/> shows ("iv", "A-3"), or null when the document gives none.</summary>
    string? PageLabel(int index);

    /// <summary>The signatures, in the order they were made (empty when there are none).</summary>
    IReadOnlyList<DocumentSignature> Signatures { get; }

    /// <summary>The named destinations (each with its <see cref="DocumentDestination.Name"/>), in the order of their names; empty when there are none or the format names none.</summary>
    IReadOnlyList<DocumentDestination> Destinations => [];

    /// <summary>
    /// The form's design as its template gives it (sections, fields with their captions, kinds, choices and bindings to the
    /// form's data), or null when the document has none or the format reads none. Read on the first call; its scripts are
    /// never run.
    /// </summary>
    DocumentFormTemplate? FormTemplate => null;
}

/// <summary>
/// Reads one document format. A format decides from a file's
/// first bytes whether the file is in its format, so the extension only decides which files a folder picks up.
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
/// What the document formats share. A format decides from a file's first bytes; picking one for a file is not on the
/// new design yet.
/// </summary>
public static class DocumentFormats
{
    /// <summary>The bytes of a file's start that <see cref="IDocumentFormat.Accepts"/> receives.</summary>
    public const int HeaderBytes = 4096;
}

// The document settings' environment variables, read the same way everywhere: text trimmed (empty is unset), numbers
// invariant, a value that cannot be read refused naming the variable.
internal static class DocumentSettings
{
    public static string? Text(string variable) => Environment.GetEnvironmentVariable(variable)?.Trim() is { Length: > 0 } value ? value : null;

    // Bytes: a whole number, or one followed by k, m or g (KiB, MiB, GiB; any case, an optional "b" or "ib" after it);
    // more than 0, or 0 too where `zero` says it means something (a cache that keeps nothing).
    public static long? Bytes(string variable, bool zero = false)
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

        return long.TryParse(span.Trim(), System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out long value) && (value > 0 || zero && value == 0)
               && value <= long.MaxValue / unit
            ? value * unit
            : throw new ArgumentException($"{variable}={text}: a number of bytes {(zero ? "0 or more" : "more than 0")} (such as 268435456, 256m or 1g).", variable);
    }

    // A count: a whole number more than 0 (no unit).
    public static long? Count(string variable)
    {
        if (Text(variable) is not { } text)
        {
            return null;
        }

        return long.TryParse(text, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out long value) && value > 0
            ? value
            : throw new ArgumentException($"{variable}={text}: a whole number more than 0.", variable);
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
