// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.IO.Compression;

namespace Idrak.Text;

/// <summary>
/// Synthetic pages written as documents (<see cref="SyntheticPageOptions.Output"/>) through the options' document writer
/// (<see cref="DocumentWriters"/>, any one): a page alone as a whole document (<see cref="One"/>, made where the
/// page is made, beside the others), or several in the order they are given, <see cref="SyntheticPageOptions.PagesPerDocument"/>
/// pages a file: each page added to the open document as it comes (the writer writes a page when the next starts) and the
/// file finished once it holds its pages, so one page's buffers are held at a time and memory does not grow with the
/// pages. A scanned page is its image (its look and all) with its text hidden over the lines as drawn; a vector page is
/// the page as drawn, its text and shapes as vectors. A page of W x H pixels is W x H x 72 / <see cref="SyntheticPageOptions.Dpi"/> points
/// (the variable's resolution when it is 0).
/// </summary>
internal sealed class SyntheticDocuments : IDisposable
{
    private readonly SyntheticPageOptions _options;
    private readonly IDocumentWriter _writer;
    private readonly string _folder;
    private readonly ZipArchive? _zip;
    private readonly float _scale;
    private Stream? _stream;
    private IDocumentBuilder? _builder;
    private string _name = "";
    private int _pages;

    // Picking the document writers by name is not on the new design yet. The default, "pdf" (Idrak.Data's PdfWriter), is
    // reached directly; another name stops.
    private static readonly Idrak.Data.Pdf.PdfWriter Pdf = new();

    internal static IDocumentWriter WriterNamed(string name) =>
        name.Equals(Pdf.Name, StringComparison.OrdinalIgnoreCase) ? Pdf : throw new NotSupportedException($"The document writer '{name}' by name is not on the new design yet.");

    /// <param name="options">The pages' options: the output, the writer and its options, the pages a document, the resolution.</param>
    /// <param name="folder">The folder the documents' names are relative to (the records').</param>
    /// <param name="zip">The zip the documents go into instead, or null.</param>
    /// <exception cref="NotSupportedException">The options' writer is not "pdf": a document writer by name is not on the new design yet.</exception>
    public SyntheticDocuments(SyntheticPageOptions options, string folder, ZipArchive? zip)
    {
        ArgumentNullException.ThrowIfNull(options);
        _options = options;
        _writer = WriterNamed(options.Writer);
        _folder = folder;
        _zip = zip;
        _scale = 72f / options.DocumentDpi;
    }

    /// <summary>A writer's file extension (its first, lower case, with its dot; "" when it lists none).</summary>
    public static string ExtensionOf(IDocumentWriter writer) => writer.Extensions.FirstOrDefault()?.ToLowerInvariant() ?? "";

    /// <summary><paramref name="page"/> alone as a whole document, written through the options' writer: its bytes.</summary>
    /// <exception cref="ArgumentException">The page was made without its vectors (not for documents).</exception>
    public static byte[] One(SyntheticPage page, SyntheticPageOptions options)
    {
        ArgumentNullException.ThrowIfNull(page);
        ArgumentNullException.ThrowIfNull(options);
        using var stream = new MemoryStream();
        using (var builder = WriterNamed(options.Writer).Create(stream, options.WriteOptions))
        {
            Write(builder, page, options.Output, 72f / options.DocumentDpi);
            builder.Finish();
        }

        return stream.ToArray();
    }

    /// <summary>
    /// Adds <paramref name="page"/> as the open document's next page, opening one named <paramref name="name"/> (with its
    /// extension, <see cref="ExtensionOf"/>) when none is open (a document is named after its first page); returns the
    /// document's name and the page's index in it, from 0. The document is finished when it holds its pages.
    /// </summary>
    /// <exception cref="ArgumentException">The page was made without its vectors (not for documents).</exception>
    public (string Name, int Index) Add(SyntheticPage page, string name)
    {
        ArgumentNullException.ThrowIfNull(page);
        if (_builder is null)
        {
            _name = name;
            _stream = _zip is not null
                ? _zip.CreateEntry(_name, CompressionLevel.NoCompression).Open()
                : new FileStream(Path.Combine(_folder, _name), FileMode.Create, FileAccess.Write, FileShare.None, 1 << 16);
            _builder = _writer.Create(_stream, _options.WriteOptions);
        }

        int index = _pages++;
        Write(_builder, page, _options.Output, _scale);
        if (_pages == _options.PagesPerDocument)
        {
            Finish();
        }

        return (_name, index);
    }

    // A page added to a document at `scale` points a pixel: a vector page as drawn, or a scanned page's image with its text
    // on the lines as drawn (the drawn page's pixels onto the image's: a look that changes the size scales them with it).
    private static void Write(IDocumentBuilder builder, SyntheticPage page, SyntheticPageOutput output, float scale)
    {
        var vectors = page.Vectors ?? throw new ArgumentException("A page written to a document is made for documents (its vectors kept).", nameof(page));
        var image = page.Image;
        float width = image.Width * scale, height = image.Height * scale;
        var target = builder.AddPage(width, height);
        if (output == SyntheticPageOutput.VectorDocuments)
        {
            vectors.Write(target, scale, page.Drawn.X, page.Drawn.Y);
            return;
        }

        target.Image(image, new DocumentBox(0, 0, width, height), DocumentImageEncoding.Auto);
        vectors.WriteText(target, scale * image.Width / Math.Max(1, page.Drawn.Width), page.Drawn.X, page.Drawn.Y, DocumentTextMode.Hidden);
    }

    /// <summary>Finishes the open document (the last one, holding what was left).</summary>
    public void Dispose() => Finish();

    private void Finish()
    {
        if (_builder is null)
        {
            return;
        }

        try
        {
            _builder.Finish();
        }
        finally
        {
            _builder.Dispose();
            _stream?.Dispose();
            (_builder, _stream, _pages) = (null, null, 0);
        }
    }
}
