// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Data.Pdf;

// Plan 28, phase D6: what a written PDF reads back as. Text (runs, glyph ids, places, boxes), images (D4) and vector paths
// and clips (D7) come from the library's reader; attached files the reader gives none of are read here through D1's
// objects (the reader's own are taken when it gives some).
internal abstract partial class Tests
{
    /// <summary>The PDF format, its pages as the library reads them (see the file's comment).</summary>
    internal sealed class PdfWrittenReader : IDocumentFormat
    {
        private readonly PdfFormat _pdf = new();

        public string Name => _pdf.Name;

        public IReadOnlyCollection<string> Extensions => _pdf.Extensions;

        public IReadOnlyCollection<string> MediaTypes => _pdf.MediaTypes;

        public bool Accepts(ReadOnlySpan<byte> header) => _pdf.Accepts(header);

        public IDocument Open(Stream stream, string name, DocumentOptions options) => new PdfWrittenDocument(PdfDocument.Open(stream, name, options));
    }

    internal sealed class PdfWrittenDocument(PdfDocument inner) : IDocument
    {
        /// <summary>The file's object layer.</summary>
        public PdfFile Objects => inner.File;

        public string Format => inner.Format;

        public DocumentInfo Info => inner.Info;

        public int PageCount => inner.PageCount;

        public IReadOnlyList<DocumentField> Fields => inner.Fields;

        public IReadOnlyList<DocumentOutline> Outline => inner.Outline;

        public IReadOnlyList<DocumentSignature> Signatures => inner.Signatures;

        public string? PageLabel(int index) => inner.PageLabel(index);

        public DocumentDrawing DrawPage(int index, DocumentDrawOptions options) => inner.DrawPage(index, options);

        public IReadOnlyList<DocumentAttachment> Attachments => inner.Attachments.Count > 0 ? inner.Attachments : PdfReadAttachments(inner.File);

        public DocumentPage ReadPage(int index, DocumentParts parts) => inner.ReadPage(index, parts);

        public void Dispose() => inner.Dispose();
    }

    // The pages' dictionaries in order (the page tree walked).
    internal static List<PdfDictionary> PdfPageDictionaries(PdfFile file)
    {
        var pages = new List<PdfDictionary>();
        var catalog = file.Dictionary(file.Trailer["Root"u8])!;
        void Walk(PdfValue node, int depth)
        {
            var dictionary = file.Dictionary(node);
            if (dictionary is null || depth > 64)
            {
                return;
            }

            if (file.Resolve(dictionary["Kids"u8]).AsArray is { } kids && !dictionary["Type"u8].IsName("Page"u8))
            {
                foreach (var kid in kids.Items)
                {
                    Walk(kid, depth + 1);
                }

                return;
            }

            pages.Add(dictionary);
        }

        Walk(catalog["Pages"u8], 0);
        return pages;
    }

    // A page's content streams decoded and joined.
    internal static byte[] PdfPageContent(PdfFile file, PdfDictionary page)
    {
        var contents = file.Resolve(page["Contents"u8]);
        var parts = contents.AsArray?.Items ?? (contents.IsNull ? [] : [contents]);
        using var joined = new MemoryStream();
        foreach (var part in parts)
        {
            if (file.Resolve(part).AsStream is { } stream)
            {
                using var decoded = file.Decode(stream);
                joined.Write(decoded.Span);
                joined.WriteByte((byte)'\n');
            }
        }

        return joined.ToArray();
    }

    // The files a document carries: its EmbeddedFiles name tree, each file's name, media type, size and bytes.
    internal static List<DocumentAttachment> PdfReadAttachments(PdfFile file)
    {
        var found = new List<DocumentAttachment>();
        var catalog = file.Dictionary(file.Trailer["Root"u8]);
        var names = catalog is null ? null : file.Dictionary(catalog["Names"u8]);
        var tree = names is null ? null : file.Dictionary(names["EmbeddedFiles"u8]);
        void Walk(PdfDictionary? node, int depth)
        {
            if (node is null || depth > 32)
            {
                return;
            }

            if (file.Resolve(node["Names"u8]).AsArray is { } pairs)
            {
                for (int i = 0; i + 1 < pairs.Count; i += 2)
                {
                    var spec = file.Dictionary(pairs[i + 1]);
                    if (spec is null || file.Dictionary(spec["EF"u8]) is not { } ef || file.Resolve(ef["F"u8]).AsStream is not { } stream)
                    {
                        continue;
                    }

                    var label = file.Resolve(spec["UF"u8]).AsString ?? file.Resolve(spec["F"u8]).AsString ?? file.Resolve(pairs[i]).AsString;
                    string name = label is null ? "" : PdfText.Decode(label.Bytes);
                    byte[] bytes;
                    using (var decoded = file.Decode(stream))
                    {
                        bytes = decoded.Span.ToArray();
                    }

                    found.Add(new DocumentAttachment(name, () => new MemoryStream(bytes, writable: false))
                    {
                        MediaType = file.Resolve(stream.Dictionary["Subtype"u8]).AsName?.ToString(), Size = bytes.Length,
                    });
                }
            }

            if (file.Resolve(node["Kids"u8]).AsArray is { } kids)
            {
                foreach (var kid in kids.Items)
                {
                    Walk(file.Dictionary(kid), depth + 1);
                }
            }
        }

        Walk(tree, 0);
        return found;
    }
}
