// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Data;
using Idrak.Imaging;

// Plan 18, phase 1: the library's OpenType reader and outline rasterizer, checked against fontTools' reading of a DejaVu
// Sans subset and FreeType's unhinted renders of its glyphs (tools/fonts/make_font_references.py, tests/Idrak.Tests/data/fonts).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] FontGroup =
    [
        ("fonts: the OpenType reader gives fontTools' names, metrics, character map, every advance and every glyph's bounds (composites resolved) for a DejaVu Sans subset; a collection gives its two fonts", FontReaderMatchesFontTools),
        ("fonts: glyphs drawn by the coverage rasterizer match FreeType's unhinted renders (Latin, a composite, Arabic, digits; 24 and 48 pixels per em)", FontRasterMatchesFreeType),
        ("fonts: exact coverage of a square at fractional positions, ink over ink where outlines overlap, clipping at the image's edges", FontRasterCoverage),
        ("fonts: a CFF font, a damaged file and bytes that are no font are refused with the reason; the registries list opentype and coverage", FontRefusals),
    ];

    private static string FontData(string name) => TestData($"fonts/{name}");

    private static IFontFace SubsetFace() => FontFormats.Read(FontData("dejavu-sans-subset.ttf")).Single();

    private static void FontReaderMatchesFontTools(Device device)
    {
        var face = SubsetFace();
        var expected = JsonNode.Parse(File.ReadAllText(FontData("face.json")))!;
        Check(face.Family == (string)expected["family"]! && face.Style == (string)expected["style"]! && face.Format == "opentype",
            $"names {face.Family} / {face.Style} / {face.Format}");
        Check(face.Metrics == new FontMetrics((int)expected["unitsPerEm"]!, (int)expected["ascender"]!, (int)expected["descender"]!, (int)expected["lineGap"]!),
            $"metrics {face.Metrics}");
        Check(face.GlyphCount == (int)expected["glyphCount"]!, $"{face.GlyphCount} glyphs");

        var cmap = expected["cmap"]!.AsArray();
        foreach (var pair in cmap)
        {
            int codePoint = (int)pair![0]!, glyph = (int)pair[1]!;
            Check(face.Glyph(codePoint) == glyph, $"U+{codePoint:X4}: glyph {face.Glyph(codePoint)}, fontTools {glyph}");
        }

        var mapped = cmap.Select(p => (int)p![0]!).ToHashSet();
        foreach (int codePoint in (int[])[0x0100, 0x0700, 0x4E00, 0x1F600, 0x10FFFF])
        {
            Check(mapped.Contains(codePoint) || face.Glyph(codePoint) == 0, $"U+{codePoint:X4} unmapped gives glyph 0, not {face.Glyph(codePoint)}");
        }

        var advances = expected["advances"]!.AsArray();
        var bounds = expected["bounds"]!.AsArray();
        for (int g = 0; g < face.GlyphCount; g++)
        {
            Check(face.Advance(g) == (int)advances[g]!, $"glyph {g}: advance {face.Advance(g)}, fontTools {advances[g]}");
            var outline = face.Outline(g);
            if (bounds[g] is not JsonArray box)
            {
                Check(outline.IsEmpty, $"glyph {g} has no contour in fontTools, an outline here");
                continue;
            }

            var (minX, minY, maxX, maxY) = outline.Bounds();
            Check(MathF.Abs(minX - (int)box[0]!) <= 0.5f && MathF.Abs(minY - (int)box[1]!) <= 0.5f && MathF.Abs(maxX - (int)box[2]!) <= 0.5f
                  && MathF.Abs(maxY - (int)box[3]!) <= 0.5f, $"glyph {g}: bounds ({minX}, {minY}, {maxX}, {maxY}), fontTools {box.ToJsonString()}");
        }

        Check(ReferenceEquals(face.Outline(5), face.Outline(5)), "an outline is read once");
        Check(face.Table("GSUB").Length > 0 && face.Table("GPOS").Length > 0 && face.Table("GDEF").Length > 0 && face.Table("CFF ").IsEmpty,
            "the layout tables are there, a missing table is empty");

        var pair2 = FontFormats.Read(FontData("dejavu-sans-pair.ttc"));
        Check(pair2.Count == 2 && pair2[0].Family == "DejaVu Sans" && pair2[1].Family == "DejaVu Sans" && pair2[0].Style != pair2[1].Style
              && pair2[1].Style.Contains("Bold", StringComparison.Ordinal) && pair2[0].Glyph('ب') > 0 && pair2[1].Glyph('ب') > 0,
            $"collection: {string.Join(", ", pair2.Select(f => $"{f.Family} {f.Style}"))}");
        Check(pair2[0].Advance(pair2[0].Glyph('A')) == face.Advance(face.Glyph('A')), "the collection's first font is the subset");
    }

    private static void FontRasterMatchesFreeType(Device device)
    {
        var face = SubsetFace();
        var rasterizer = OutlineRasterizers.Get();
        var renders = JsonNode.Parse(File.ReadAllText(FontData("freetype.json")))!.AsArray();
        Check(renders.Count >= 12, $"{renders.Count} renders");
        double worstMean = 0, worstMax = 0;
        foreach (var render in renders)
        {
            string text = (string)render!["text"]!;
            int glyph = (int)render["glyph"]!, ppem = (int)render["ppem"]!, left = (int)render["left"]!, top = (int)render["top"]!;
            int width = (int)render["width"]!, rows = (int)render["rows"]!;
            Check(face.Glyph(char.ConvertToUtf32(text, 0)) == glyph, $"{text}: glyph {face.Glyph(char.ConvertToUtf32(text, 0))}, FreeType {glyph}");
            byte[] reference = File.ReadAllBytes(FontData((string)render["file"]!));
            Check(reference.Length == width * rows, $"{text}: {reference.Length} bytes for {width} x {rows}");

            // The bitmap's top-left pixel is (left, top) from the pen with y up: the origin sits at (-left, top) in it.
            var coverage = new float[width * rows];
            rasterizer.Fill(face.Outline(glyph), ppem / (float)face.Metrics.UnitsPerEm, -left, top, coverage, width, rows);
            double sum = 0, max = 0, ink = 0;
            for (int i = 0; i < coverage.Length; i++)
            {
                double d = Math.Abs(coverage[i] - reference[i] / 255.0);
                sum += d;
                max = Math.Max(max, d);
                ink += reference[i] / 255.0;
            }

            double mean = sum / Math.Max(1, coverage.Length);
            double inkHere = coverage.Sum(v => (double)v);
            worstMean = Math.Max(worstMean, mean);
            worstMax = Math.Max(worstMax, max);
            Check(mean < 0.006 && max < 0.08 && Math.Abs(inkHere - ink) < 0.01 * ink + 0.5,
                $"{text} at {ppem}: mean difference {mean:F4}, largest {max:F3}, ink {inkHere:F1} against FreeType's {ink:F1}");
        }

        Console.WriteLine($"    against FreeType: mean difference at most {worstMean:F4}, largest pixel difference {worstMax:F3}");
    }

    private static void FontRasterCoverage(Device device)
    {
        var rasterizer = OutlineRasterizers.Get(OutlineRasterizers.Coverage);
        // A 2 x 2 unit square (y up, from (0, 0) to (2, 2)) at scale 1 with its origin at (1.5, 3.5): it covers x 1.5..3.5,
        // y 1.5..3.5 (y down), so the corner pixels hold a quarter, the edge pixels a half, the middle one all.
        var square = new GlyphOutline([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close], [0, 0, 2, 0, 2, 2, 0, 2]);
        var coverage = new float[5 * 5];
        rasterizer.Fill(square, 1f, 1.5f, 3.5f, coverage, 5, 5);
        float[] expected =
        [
            0, 0, 0, 0, 0,
            0, 0.25f, 0.5f, 0.25f, 0,
            0, 0.5f, 1f, 0.5f, 0,
            0, 0.25f, 0.5f, 0.25f, 0,
            0, 0, 0, 0, 0,
        ];
        Check(coverage.Zip(expected).All(p => MathF.Abs(p.First - p.Second) < 1e-5f), $"square: {string.Join(" ", coverage.Select(v => v.ToString("F2")))}");

        // The same square drawn again over itself: ink over ink, 1 - (1 - a)².
        rasterizer.Fill(square, 1f, 1.5f, 3.5f, coverage, 5, 5);
        Check(MathF.Abs(coverage[6] - (1 - 0.75f * 0.75f)) < 1e-5f && MathF.Abs(coverage[12] - 1f) < 1e-6f, $"ink over ink: {coverage[6]}, {coverage[12]}");

        // Wound the other way (clockwise): the same coverage.
        var reversed = new GlyphOutline([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close], [0, 0, 0, 2, 2, 2, 2, 0]);
        var other = new float[25];
        rasterizer.Fill(reversed, 1f, 1.5f, 3.5f, other, 5, 5);
        Check(other.Zip(expected).All(p => MathF.Abs(p.First - p.Second) < 1e-5f), "either winding");

        // Half off the image on every side: what lies inside is drawn, nothing outside is written.
        var big = new GlyphOutline([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close], [0, 0, 10, 0, 10, 10, 0, 10]);
        var clipped = new float[4 * 3];
        rasterizer.Fill(big, 1f, -3f, 7f, clipped, 4, 3);
        Check(clipped.All(v => MathF.Abs(v - 1f) < 1e-5f), $"clipped: {string.Join(" ", clipped)}");
        var none = new float[9];
        rasterizer.Fill(big, 1f, 50f, 50f, none, 3, 3);
        Check(none.All(v => v == 0f), "off the image: nothing");

        // A glyph of quadratic curves (the letter o) stays within [0, 1] and covers a plausible area.
        var font = SubsetFace();
        var o = font.Outline(font.Glyph('o'));
        var page = new float[64 * 64];
        rasterizer.Fill(o, 48f / font.Metrics.UnitsPerEm, 8f, 48f, page, 64, 64);
        float area = page.Sum();
        Check(area > 100 && area < 600 && page.All(v => v is >= 0f and <= 1f), $"the letter o covers {area:F1} pixels, all in [0, 1]");
    }

    private static void FontRefusals(Device device)
    {
        Check(FontFormats.Names.Contains("opentype", StringComparer.OrdinalIgnoreCase) && FontFormats.Extensions.Contains(".ttc")
              && OutlineRasterizers.Names.Contains("coverage", StringComparer.OrdinalIgnoreCase),
            $"formats {string.Join(", ", FontFormats.Names)}; rasterizers {string.Join(", ", OutlineRasterizers.Names)}");

        byte[] cff = File.ReadAllBytes(FontData("dejavu-sans-subset.ttf"));
        "OTTO"u8.CopyTo(cff);
        try
        {
            FontFormats.Read(cff, "cff");
            Check(false, "a CFF font must be refused");
        }
        catch (NotSupportedException ex)
        {
            Check(ex.Message.Contains("CFF", StringComparison.Ordinal) && ex.Message.Contains("TrueType outlines", StringComparison.Ordinal), ex.Message);
        }

        try
        {
            FontFormats.Read(new byte[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }, "noise");
            Check(false, "bytes that are no font must be refused");
        }
        catch (InvalidDataException ex)
        {
            Check(ex.Message.Contains("not a font of the registered formats", StringComparison.Ordinal), ex.Message);
        }

        byte[] cut = File.ReadAllBytes(FontData("dejavu-sans-subset.ttf"))[..2000];
        try
        {
            FontFormats.Read(cut, "cut");
            Check(false, "a damaged font must be refused");
        }
        catch (InvalidDataException ex)
        {
            Check(ex.Message.StartsWith("cut:", StringComparison.Ordinal), ex.Message);
        }
    }
}
