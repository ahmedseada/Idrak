// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Data.Pdf;

/// <summary>A Type 3 glyph's procedure read as a page of parts, for a glyph that is more than filled shapes (ISO 32000-2, 9.6.4).</summary>
internal sealed partial class PdfContentReader
{
    /// <summary>
    /// <paramref name="content"/>, a Type 3 glyph's procedure, read as a page of parts (paths and their strokes, images,
    /// shadings, runs, clips, groups, soft masks) through <paramref name="units"/> (glyph space through the font matrix to
    /// the face's units, y up), with the font's <paramref name="resources"/>. A glyph that is not coloured (<c>d1</c>) has
    /// every part painted black, its colour operators ignored: its shape is what a drawer paints in the run's colour.
    /// What it meets is not reported (a glyph drawn at a page's drawing has no page to report to; a damaged glyph draws
    /// what it reads). A limit passed throws, naming it.
    /// </summary>
    internal static DocumentPage GlyphPage(PdfFile file, ReadOnlySpan<byte> content, PdfDictionary? resources, PdfMatrix units, bool coloured)
    {
        using var reader = new PdfContentReader(file, 1, new PdfIssues(), includeHidden: false) { _nesting = 1, _fixed = coloured ? null : DocumentColour.Black };
        reader.ReadVector();
        reader.ReadImages();
        reader.ReadContent(content, resources, units, units, null, 1);
        var runs = reader.TextRuns(glyphs: true, out _, out _);
        return new DocumentPage(0, 0f, 0f)
        {
            Runs = runs, Images = reader.Images, Paths = reader.Paths, Clips = reader.Clips, Shadings = reader.Shadings, Groups = reader.Groups, SoftMasks = reader.SoftMasks,
        };
    }
}
