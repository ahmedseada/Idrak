// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Data.Pdf;

/// <summary>
/// The library's PDF reader as "pdf": a file whose <c>%PDF-</c> header is
/// within its first KiB. A document is opened, not loaded: the cross-reference (classic tables, cross-reference streams,
/// object streams, hybrid files, incremental updates; rebuilt by scanning when it is damaged, which
/// <see cref="DocumentInfo.Repaired"/> says) and the page tree are read at open, every other object when it is needed,
/// from the stream at any offset (a file's handle without a shared position, or memory in place), each parsed once into
/// a cache within <see cref="DocumentOptions.CacheBytes"/>. Streams decode through the standard filters and predictors,
/// each output within <see cref="DocumentOptions.MaxStreamBytes"/>; objects nest no deeper than
/// <see cref="DocumentOptions.MaxDepth"/>, a file lists no more than <see cref="DocumentOptions.MaxObjects"/>, and a
/// cycle of references, of page-tree nodes or of cross-reference sections is reported: each an
/// <see cref="InvalidDataException"/> naming the limit and its option, or the cycle. The information (PDFDocEncoding,
/// UTF-16 and UTF-8 strings, dates), the version, each page's size and rotation, and its text are read: the content
/// interpreted (text state, matrices, form XObjects within <see cref="DocumentOptions.MaxDepth"/>, inline images,
/// <c>/ActualText</c>), fonts to text by ToUnicode, encodings and glyph names, or the embedded program's own cmap (read
/// as its font descriptor's key says), glyph widths placing each run's box on the page as shown, and the lines and text
/// made by <see cref="DocumentOptions.Layout"/>; a glyph with no text is U+FFFD and counted in the page's issues. An
/// encrypted file is read through the standard security handler (revisions 2-6: RC4, AES-128, AES-256) with
/// <see cref="DocumentOptions.Password"/> (as the owner's, then the user's password) or the empty password, a damaged one
/// repaired with its key found before its object streams are listed; another handler is a
/// <see cref="NotSupportedException"/> naming it. Images, vector paths, forms, annotations and signatures come with the
/// later phases of the documents plan. Safe to read from several threads.
/// </summary>
public sealed class PdfFormat : IDocumentFormat
{
    /// <summary>The format's name: "pdf".</summary>
    public const string FormatName = "pdf";

    /// <summary>How far into a file its header may start (bytes before it are allowed, as mail and download tools add them).</summary>
    public const int HeaderWithin = 1024;

    /// <inheritdoc />
    public string Name => FormatName;

    /// <inheritdoc />
    public IReadOnlyCollection<string> Extensions { get; } = [".pdf"];

    /// <inheritdoc />
    public IReadOnlyCollection<string> MediaTypes { get; } = ["application/pdf"];

    /// <inheritdoc />
    public bool Accepts(ReadOnlySpan<byte> header)
    {
        int at = header[..Math.Min(header.Length, HeaderWithin + 4)].IndexOf("%PDF-"u8);
        return at >= 0 && at < HeaderWithin;
    }

    /// <inheritdoc />
    public IDocument Open(Stream stream, string name, DocumentOptions options)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(name);
        ArgumentNullException.ThrowIfNull(options);
        if (!stream.CanSeek || !stream.CanRead)
        {
            throw new ArgumentException($"{name}: a PDF document is opened from a stream that reads and seeks; copy it into memory or a file first.", nameof(stream));
        }

        return PdfDocument.Open(stream, name, options);
    }
}
