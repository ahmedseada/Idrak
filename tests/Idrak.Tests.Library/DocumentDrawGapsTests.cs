// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;

// Plan 28: the drawing gaps D7 left, closed (2026-11-19): groups that are not isolated, a stroke's pen under a matrix
// that is not uniform, Type 3 glyphs that are drawings, a tiling cell clipped to its box (documents drawers); the pen
// read and written (documents pdf: vector); vertical position vectors (documents pdf: text). (TrueType glyphs by their
// standard Macintosh names and a bare CFF's own encoding read the program the file embeds, through the font formats tried
// by its header, which is not on the new design yet.) The samples are
// tools/documents/docfixtures/samples_d7_gaps.py's, drawn against pdfium by "documents drawers: the library's coverage
// drawer ...".
internal abstract partial class Tests
{
    // ---- documents drawers -----------------------------------------------------------------------------------------------

    // Over an opaque yellow backdrop, a group at 0.8 holding a Normal square at 0.5 and a Multiply square at 0.7 that
    // overlaps it: as a group that is not isolated, one that is, and one that is not isolated and knocks out. Each region's
    // colour is the transparency rules' (ISO 32000-2, 11.4.7-11.4.8), worked here in doubles.
    private static void PdfDrawGroupsNotIsolated(Device device)
    {
        CpuOnly(device);
        (double R, double G, double B) yellow = (0.95, 0.85, 0.15), blue = (0.15, 0.3, 0.85), orange = (0.95, 0.55, 0.15);
        static DocumentPath Square(float left, float across, (double R, double G, double B) paint, float opacity, int order, int group, DocumentBlend mode)
        {
            var box = DocumentPath.Rectangle(new DocumentBox(left, 0f, across, 20f));
            return new DocumentPath(box.Verbs.ToArray(), box.Points.ToArray())
            {
                FillColour = DocumentColour.Rgb((float)paint.R, (float)paint.G, (float)paint.B).WithAlpha(opacity), Order = order, Group = group, Blend = mode,
            };
        }

        DocumentPage Page(bool iso, bool ko) => new(0, 60f, 20f)
        {
            Paths = [Square(0f, 60f, yellow, 1f, 0, -1, DocumentBlend.Normal), Square(10f, 20f, blue, 0.5f, 1, 0, DocumentBlend.Normal), Square(20f, 20f, orange, 0.7f, 2, 0, DocumentBlend.Multiply)],
            Groups = [new DocumentGroup { Isolated = iso, Knockout = ko, Alpha = 0.8f }],
        };

        static (double, double, double) Mix((double R, double G, double B) under, (double R, double G, double B) over, double t) =>
            (under.R + (over.R - under.R) * t, under.G + (over.G - under.G) * t, under.B + (over.B - under.B) * t);
        static (double, double, double) Times((double R, double G, double B) one, (double R, double G, double B) other) => (one.R * other.R, one.G * other.G, one.B * other.B);

        // Not isolated, not knockout: the parts blend with the backdrop; the backdrop is taken back out at the close.
        var first = Mix(yellow, blue, 0.5);                                    // the blue square at 0.5 over the backdrop
        var overlap = Mix(first, Times(first, orange), 0.7);                   // the Multiply square at 0.7 over that
        double groupAlpha = 0.5 + 0.7 - 0.5 * 0.7;
        var overlapGroup = Mix(yellow, overlap, 1 / groupAlpha);               // C = Cn + (Cn - C0)(1 / ag - 1)
        var expected = new Dictionary<string, (double, double, double)[]>
        {
            ["not isolated"] = [Mix(yellow, blue, 0.4), Mix(yellow, overlapGroup, groupAlpha * 0.8), Mix(yellow, Times(yellow, orange), 0.56)],
            ["not isolated, knockout"] = [Mix(yellow, blue, 0.4), Mix(yellow, Times(yellow, orange), 0.56), Mix(yellow, Times(yellow, orange), 0.56)],
        };

        // Isolated: the parts blend with nothing under them (the Multiply square alone is its own colour), as always: the
        // Multiply square over the blue at 0.5 in the group's own layer (premultiplied), its alpha 0.85, composed at 0.8.
        (double, double, double) mixed = ((1 - 0.7) * 0.5 * blue.R + 0.7 * ((1 - 0.5) * orange.R + 0.5 * blue.R * orange.R),
            (1 - 0.7) * 0.5 * blue.G + 0.7 * ((1 - 0.5) * orange.G + 0.5 * blue.G * orange.G), (1 - 0.7) * 0.5 * blue.B + 0.7 * ((1 - 0.5) * orange.B + 0.5 * blue.B * orange.B));
        var isolatedOverlap = Mix(yellow, (mixed.Item1 / groupAlpha, mixed.Item2 / groupAlpha, mixed.Item3 / groupAlpha), groupAlpha * 0.8);
        expected["isolated"] = [Mix(yellow, blue, 0.4), isolatedOverlap, Mix(yellow, orange, 0.56)];

        var options = new DocumentDrawOptions { PixelsPerPoint = 1f };
        var failures = new List<string>();
        foreach (var (name, isolated, knockout) in new[] { ("not isolated", false, false), ("not isolated, knockout", false, true), ("isolated", true, false) })
        {
            var drawing = CoverageDrawn(Page(isolated, knockout), options);
            for (int region = 0; region < 3; region++)
            {
                int x = 15 + 10 * region;
                var want = expected[name][region];
                int[] got = [.. Enumerable.Range(0, 3).Select(c => PdfPixel(drawing.Image, c, x, 10))];
                int[] rgb = [Level(want.Item1), Level(want.Item2), Level(want.Item3)];
                if (rgb.Zip(got).Any(p => Math.Abs(p.First - p.Second) > 2))
                {
                    failures.Add($"{name}, region {region} (x {x}): drawn {string.Join(" ", got)}, the rules give {string.Join(" ", rgb)}");
                }
            }
        }

        // The Multiply square alone, not isolated, differs from the same isolated (it blends with the yellow under it).
        Check(Level(expected["not isolated"][2].Item1) != Level(expected["isolated"][2].Item1) || Level(expected["not isolated"][2].Item3) != Level(expected["isolated"][2].Item3),
            "the test's own expectation: a Multiply part blends differently in a group that is not isolated");
        Check(failures.Count == 0, string.Join("\n      ", failures));
    }

    // Lines along x and along y under a pen stretched 4 times up (a CTM of 1 by 4, normalised: 0.5 by 2): the first as
    // thick as the line width times 4, the second as the width alone; with no pen, both the width times the root of 4.
    private static void PdfDrawStrokePen(Device device)
    {
        CpuOnly(device);
        var pen = new DocumentMatrix(0.5f, 0f, 0f, 2f, 0f, 0f);
        DocumentPage Page(DocumentMatrix? matrix) => new(0, 100f, 100f)
        {
            Paths =
            [
                new DocumentPath([OutlineVerb.Move, OutlineVerb.Line], [10f, 20f, 90f, 20f]) { Fill = DocumentFill.None, Stroke = 2f, StrokeMatrix = matrix, Order = 0 },
                new DocumentPath([OutlineVerb.Move, OutlineVerb.Line], [50f, 40f, 50f, 90f]) { Fill = DocumentFill.None, Stroke = 2f, StrokeMatrix = matrix, Order = 1 },
            ],
        };

        var options = new DocumentDrawOptions { PixelsPerPoint = 4f, Colour = DocumentDrawColour.Grey };
        int Thickness(ImageData image, bool down, int at)
        {
            int n = 0;
            for (int k = 0; k < (down ? image.Height : image.Width); k++)
            {
                n += (down ? PdfPixel(image, 0, at, k) : PdfPixel(image, 0, k, at)) < 128 ? 1 : 0;
            }

            return n;
        }

        var stretched = CoverageDrawn(Page(pen), options).Image;
        var round = CoverageDrawn(Page(null), options).Image;
        int across = Thickness(stretched, down: true, 30 * 4), along = Thickness(stretched, down: false, 70 * 4);
        int roundAcross = Thickness(round, down: true, 30 * 4), roundAlong = Thickness(round, down: false, 70 * 4);
        Check(Math.Abs(across - 16) <= 1 && Math.Abs(along - 4) <= 1, $"under the pen: the line along x {across} pixels thick (4 points: 16 expected), along y {along} (1 point: 4 expected)");
        Check(Math.Abs(roundAcross - 8) <= 1 && Math.Abs(roundAlong - 8) <= 1, $"a round pen: {roundAcross} and {roundAlong} pixels thick (2 points: 8 expected)");
    }

    // A Type 3 font whose glyphs are a stroked frame (d1), an image mask (d1), a red square in its own colour (d0) and a
    // filled square (d1, an outline alone), shown in green: the frame and the mask in green, the d0 square red, the filled
    // square green as it always was; the face gives a drawing for the first three only.
    private static void PdfDrawType3Drawings(Device device)
    {
        CpuOnly(device);
        byte[] mask = [0x40, 0x00];                                            // 2 x 2: the top right pixel unpainted
        string[] procs =
        [
            "1000 0 0 0 1000 1000 d1 100 w 200 200 m 800 200 l 800 800 l 200 800 l h S",
            "1000 0 0 0 1000 1000 d1 q 800 0 0 800 100 100 cm BI /W 2 /H 2 /IM true /BPC 1 ID " + Encoding.Latin1.GetString(mask) + " EI Q",
            "1000 0 d0 1 0 0 rg 0 0 1000 1000 re f",
            "1000 0 0 0 1000 1000 d1 0 0 1000 1000 re f",
        ];
        byte[] file = PdfHandMadeFile([
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 60] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            PdfHandMadeStream("", Encoding.ASCII.GetBytes("BT /F1 40 Tf 0 0.5 0 rg 10 10 Td (abcd) Tj ET")),
            "<< /Type /Font /Subtype /Type3 /FontBBox [0 0 1000 1000] /FontMatrix [0.001 0 0 0.001 0 0] /CharProcs << /a 6 0 R /b 7 0 R /c 8 0 R /d 9 0 R >> "
            + "/Encoding << /Type /Encoding /Differences [97 /a /b /c /d] >> /FirstChar 97 /LastChar 100 /Widths [1000 1000 1000 1000] /Resources << >> >>",
            PdfHandMadeStream("", Encoding.Latin1.GetBytes(procs[0])),
            PdfHandMadeStream("", Encoding.Latin1.GetBytes(procs[1])),
            PdfHandMadeStream("", Encoding.Latin1.GetBytes(procs[2])),
            PdfHandMadeStream("", Encoding.Latin1.GetBytes(procs[3])),
        ]);
        using var document = PdfOpened(file.AsMemory(), "type3-drawings.pdf");
        var run = document.ReadPage(0, DocumentParts.All).Runs.Single();
        Check(run.Face is IDocumentGlyphDrawings, $"the Type 3 face draws glyphs by their parts ({run.Face?.GetType().Name})");
        var faceDrawings = (IDocumentGlyphDrawings)run.Face!;
        var kinds = run.Glyphs.ToArray().Select(g => faceDrawings.Drawing(g.Glyph) is { } d ? (d.Coloured ? "coloured" : "shape") : "outline").ToList();
        Check(kinds.SequenceEqual(["shape", "shape", "coloured", "outline"]), $"the glyphs' kinds: {string.Join(", ", kinds)} (a stroked frame and a mask are shapes, the d0 square coloured, a filled square an outline)");

        var drawing = document.DrawPage(0, new DocumentDrawOptions { PixelsPerPoint = 2f, Fonts = null });
        var failures = new List<string>();
        foreach (var (x, y, rgb, what) in new (float X, float Y, int[] Rgb, string What)[]
        {
            (18f, 30f, new[] { 0, 128, 0 }, "the frame's stroke (d1: the text's green)"),
            (30f, 30f, new[] { 255, 255, 255 }, "inside the frame (stroked, not filled)"),
            (62f, 22f, new[] { 0, 128, 0 }, "the mask's painted top left (d1: green)"),
            (78f, 22f, new[] { 255, 255, 255 }, "the mask's unpainted top right"),
            (78f, 38f, new[] { 0, 128, 0 }, "the mask's painted bottom right"),
            (110f, 30f, new[] { 255, 0, 0 }, "the d0 square (its own red)"),
            (150f, 30f, new[] { 0, 128, 0 }, "the filled square (an outline in green)"),
        })
        {
            int[] got = [.. Enumerable.Range(0, 3).Select(c => PdfPixel(drawing.Image, c, (int)(x * 2), (int)(y * 2)))];
            if (rgb.Zip(got).Any(p => Math.Abs(p.First - p.Second) > 3))
            {
                failures.Add($"{what} at ({x}, {y}): {string.Join(" ", got)}, expected {string.Join(" ", rgb)}");
            }
        }

        Check(failures.Count == 0 && drawing.Issues.Count == 0, string.Join("\n      ", failures.Concat(drawing.Issues.Select(i => $"{i.Kind}: {i.Detail}"))));
    }

    // A tiling whose cell (a 14 x 14 red square) is larger than its 10 x 10 box, every 16 points: the box clips the cell
    // (ISO 32000-2, 8.7.3.1), upright and turned; a cell whose box is its steps is drawn as it always was.
    private static void PdfDrawTilingBox(Device device)
    {
        CpuOnly(device);
        var square = DocumentPath.Rectangle(new DocumentBox(0f, 0f, 14f, 14f));
        var cell = new DocumentPage(0, 0f, 0f) { Paths = [new DocumentPath(square.Verbs.ToArray(), square.Points.ToArray()) { FillColour = DocumentColour.Rgb(0.9f, 0.1f, 0.1f) }] };
        float c = MathF.Cos(MathF.PI / 6), s = MathF.Sin(MathF.PI / 6);
        var failures = new List<string>();
        foreach (var (name, matrix) in new[] { ("upright", new DocumentMatrix(1f, 0f, 0f, -1f, 48f, 80f)), ("turned", new DocumentMatrix(c, -s, -s, -c, 48f, 80f)) })
        {
            var pattern = new DocumentPattern(cell, new DocumentBox(0f, 0f, 10f, 10f), 16f, 16f, matrix);
            var fill = DocumentPath.Rectangle(new DocumentBox(0f, 0f, 120f, 120f));
            var page = new DocumentPage(0, 120f, 120f) { Paths = [new DocumentPath(fill.Verbs.ToArray(), fill.Points.ToArray()) { FillPattern = pattern }] };
            var image = CoverageDrawn(page, new DocumentDrawOptions { PixelsPerPoint = 2f }).Image;
            foreach (var (u, v, red, what) in new (float U, float V, bool Red, string What)[] { (5f, 5f, true, "inside the box"), (12.5f, 5f, false, "the square past the box across"), (5f, 12.5f, false, "past it up") })
            {
                var (x, y) = matrix.Transform(u, v);
                int r = PdfPixel(image, 0, (int)(x * 2), (int)(y * 2)), g = PdfPixel(image, 1, (int)(x * 2), (int)(y * 2));
                if (red != (r > 200 && g < 60) || (!red && (r < 240 || g < 240)))
                {
                    failures.Add($"{name}: {what} (pattern {u}, {v}): {r} {g}, expected {(red ? "red" : "white")}");
                }
            }
        }

        Check(failures.Count == 0, string.Join("\n      ", failures));
    }

    // A channel of a drawn image at a pixel, 0 to 255.
    private static int PdfPixel(ImageData image, int channel, int x, int y) =>
        (int)MathF.Round(image.Pixels[Math.Min(channel, image.Channels - 1) * image.Width * image.Height + y * image.Width + x] * 255);

    private static int Level(double v) => (int)Math.Round(Math.Clamp(v, 0, 1) * 255);

    // ---- documents pdf: vector -------------------------------------------------------------------------------------------

    // A stroke's pen: read from the CTM (a stretch of 4 up: the width times 2 and a pen of 0.5 by 2, y turned down by the
    // page; a uniform scale and a turn written to three places: round), written by the pdf writer under its matrix and
    // read back the same (points, width, pen).
    private static void PdfVectorStrokePen(Device device)
    {
        CpuOnly(device);
        string content = "q 1 0 0 4 0 0 cm 1 w 10 5 m 90 5 l S Q q 2 0 0 2 0 0 cm 1 w 10 30 m 40 30 l S Q q 0.866 0.5 -0.5 0.866 50 50 cm 2 w 0 0 m 20 0 l S Q";
        byte[] file = PdfHandMadeFile([
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents 4 0 R >>",
            PdfHandMadeStream("", Encoding.ASCII.GetBytes(content)),
        ]);
        DocumentPath stretched;
        using (var document = PdfOpened(file.AsMemory(), "pen.pdf"))
        {
            var paths = document.ReadPage(0, DocumentParts.Paths).Paths;
            Check(paths.Count == 3, $"{paths.Count} paths");
            stretched = paths[0];
            Check(stretched.Stroke is { } w && MathF.Abs(w - 2f) < 1e-4f && stretched.StrokeMatrix is { } m && MathF.Abs(m.A - 0.5f) < 1e-5f && MathF.Abs(m.B) < 1e-6f && MathF.Abs(m.C) < 1e-6f && MathF.Abs(m.D + 2f) < 1e-5f,
                $"under 1 0 0 4 cm: width {stretched.Stroke}, pen {stretched.StrokeMatrix} (expected 2 and (0.5, 0, 0, -2))");
            Check(paths[1].StrokeMatrix is null && paths[1].Stroke == 2f, $"under a uniform scale: pen {paths[1].StrokeMatrix}, width {paths[1].Stroke}");
            Check(paths[2].StrokeMatrix is null, $"under a turn written to three places: pen {paths[2].StrokeMatrix}");
        }

        byte[] written = PdfWriteDocument(b => b.AddPage(200f, 100f).Path(stretched));
        using var back = PdfOpened(written.AsMemory(), "pen-written.pdf");
        var read = back.ReadPage(0, DocumentParts.Paths).Paths.Single();
        var p = read.Points.Span;
        var q = stretched.Points.Span;
        Check(read.StrokeMatrix is { } r && MathF.Abs(r.A - 0.5f) < 1e-4f && MathF.Abs(r.D + 2f) < 1e-4f && MathF.Abs(r.B) < 1e-4f && MathF.Abs(r.C) < 1e-4f
              && read.Stroke is { } rw && MathF.Abs(rw - 2f) < 1e-3f && p.Length == q.Length && Enumerable.Range(0, q.Length).All(i => MathF.Abs(read.Points.Span[i] - stretched.Points.Span[i]) < 1e-3f),
            $"written and read back: width {read.Stroke}, pen {read.StrokeMatrix}, points {string.Join(", ", read.Points.ToArray())} (expected {string.Join(", ", stretched.Points.ToArray())})");
    }

    // ---- documents pdf: text ---------------------------------------------------------------------------------------------

    private static PdfSampleEntry PdfGapSample(string name) =>
        PdfSamples().FirstOrDefault(s => s.Name == name)
        ?? throw new InvalidOperationException($"The sample '{name}' is not in the samples: make them again (python -I tools/documents/make_fixtures.py -o test-output/documents/samples).");

    // A page's runs' glyph ids against the sample's `glyphIds` (one list a run, in order).
    private static void PdfCheckGlyphIds(string what, DocumentPage page, JsonArray expected)
    {
        Check(page.Runs.Count == expected.Count, $"{what}: {page.Runs.Count} runs, {expected.Count} expected");
        for (int k = 0; k < expected.Count; k++)
        {
            int[] ids = [.. page.Runs[k].Glyphs.ToArray().Select(g => g.Glyph)];
            int[] want = [.. expected[k]!["glyphs"]!.AsArray().Select(v => (int)v!)];
            Check(ids.SequenceEqual(want), $"{what}: run {k} \"{page.Runs[k].Text}\": glyphs {string.Join(" ", ids)}, expected {string.Join(" ", want)}");
        }
    }

    // Vertical text whose /W2 gives some glyphs a position vector of their own: each glyph's box across the column from
    // -vx to its width less vx (its run's box as the sample's), and its origin moved by its vector (vx, vy).
    private static void PdfTextVerticalW2(Device device)
    {
        CpuOnly(device);
        var sample = PdfGapSample("vector-vertical-w2");
        using var document = PdfOpened(sample.Path);
        PdfCheckTextPage("vector-vertical-w2", document.ReadPage(0, DocumentParts.Text), sample.Expected["pages"]![0]!.AsObject());

        // "Report" at 22 points from (280, 270) up the page (the right column, read first): R's vector (200, 900)
        // thousandths, against /DW2's (half its width, 880) for the glyphs /W2 does not list.
        var page = document.ReadPage(0, DocumentParts.Text | DocumentParts.Glyphs);
        var run = page.Runs[0];
        var r = run.Glyphs.Span[0];
        float top = page.Height - 270f;
        Check(MathF.Abs(r.X - (280f - 0.2f * 22f)) < 1e-3f && MathF.Abs(r.Y - (top + 0.9f * 22f)) < 1e-3f,
            $"R's origin ({r.X.ToString(CultureInfo.InvariantCulture)}, {r.Y.ToString(CultureInfo.InvariantCulture)}), expected ({280f - 4.4f}, {top + 19.8f}): the vertical origin moved by /W2's vector");
    }
}
