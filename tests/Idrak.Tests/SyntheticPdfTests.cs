// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Idrak.Abstraction.Data;
using Idrak.Data.Pdf;
using Idrak.Imaging;
using Idrak.Text;

// Plan 28's follow-up: idrak data synth writes PDFs. --format pdf: each drawn page (its look and all) an image page with
// the label's text hidden on the lines as drawn; --format pdf-vector: each page as drawn before any look, its text and
// shapes as vectors (the emblems' and stamps' words as text too). Every record names its document and the page's index in
// it, and every file is read back with the library's reader and drawn by its drawer.
internal abstract partial class DesktopTests
{
    private static readonly (string Name, Action<Device> Run)[] SyntheticPdfGroup =
    [
    ];

    // Each record of a written data set: its record and variant (from its manifest line), its document and page.
    private static List<(int Record, int Variant, string File, int Page, JsonNode Line, JsonNode About)> PdfSynthRecords(SyntheticPagesResult result)
    {
        var lines = File.ReadAllLines(result.Data).Select(l => JsonNode.Parse(l)!).ToList();
        var about = File.ReadAllLines(result.Manifest).Select(l => JsonNode.Parse(l)!).ToList();
        Check(lines.Count == about.Count && lines.Count == result.Pages, $"a record and a manifest line a page: {lines.Count}, {about.Count}, {result.Pages}");
        string folder = Path.GetDirectoryName(result.Data)!;
        return [.. lines.Select((l, i) => ((int)about[i]["record"]!, (int)about[i]["variant"]!, Path.Combine(folder, (string)l["images"]![0]!), (int)l["pages"]![0]!, l, about[i]))];
    }

    // A page builder that keeps the text runs it is given (their faces, sizes, glyphs, matrices and text) and counts the rest.
    private sealed class PdfSynthCapture : IDocumentPageBuilder
    {
        public List<(IFontFace Face, float Size, DocumentGlyph[] Glyphs, DocumentMatrix[]? Matrices, string Text, DocumentTextMode Mode)> Runs { get; } = [];

        public int Paths { get; private set; }

        public int Index => 0;

        public float Width => 1000;

        public float Height => 1000;

        public void Image(ImageData image, DocumentBox box, DocumentImageEncoding encoding)
        {
        }

        public void Path(DocumentPath path) => Paths++;

        public void Glyphs(IFontFace face, float size, ReadOnlySpan<DocumentGlyph> glyphs, string text, DocumentTextMode mode, DocumentColour colour) =>
            Runs.Add((face, size, glyphs.ToArray(), null, text, mode));

        public void Glyphs(IFontFace face, ReadOnlySpan<DocumentGlyph> glyphs, ReadOnlySpan<DocumentMatrix> matrices, string text, DocumentTextMode mode, DocumentColour colour) =>
            Runs.Add((face, MathF.Sqrt(matrices[0].A * matrices[0].A + matrices[0].B * matrices[0].B), glyphs.ToArray(), matrices.ToArray(), text, mode));

        public void Save()
        {
        }

        public void Restore()
        {
        }

        public void Clip(DocumentPath path)
        {
        }

        public void AddAnnotation(DocumentAnnotation annotation)
        {
        }

        public void AddLink(DocumentLink link)
        {
        }
    }
}
