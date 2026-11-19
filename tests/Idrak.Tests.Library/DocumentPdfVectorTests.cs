// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Data.Pdf;
using Idrak.Imaging;

// Plan 28, phase D7: PDF vector content read (paths, clips, colours in every space, shadings, patterns, groups, soft
// masks, the fonts drawn with) and any page drawn to pixels by the library's "coverage" drawer, checked on the samples
// tools/documents/make_fixtures.py writes into the ignored test-output/documents/samples (their JSON and pdfium's renders),
// on pages the writer writes, and on hand-made bytes. Drawn pages are saved under test-output/documents/drawn.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] DocumentPdfVectorGroup =
    [
        ("documents pdf: vector: every D7 sample page's paths (verbs, points within 0.002 point, fill rule, stroke width, cap, join, miter limit, dash, colours as RGB within their tolerance and device components as given, opacity, blend mode, pattern, clip, group, soft mask and order), clips (rule, parent, points, a text clip naming its run), shadings (type, order, clip, opacity), groups (isolated, knockout) and the runs' and images' colours, clips and opacity, as their JSON says", PdfVectorSamples),
        ("documents pdf: vector: every font of vector-fonts read as a face to draw with (Type 1 through eexec, Type 3 from its glyphs' streams, TrueType, a bare CFF, OpenType CFF), the Type 1 glyphs' outlines and widths as fontTools reads the program, each run's glyphs through its face", PdfVectorFonts),
        ("documents pdf: vector: a quadratic written by the pdf writer reads back as that quadratic, a cubic that is not one stays a cubic; PDF functions of types 0, 2, 3 and 4 on known values (the calculator's operators and bounds); hand-made pages: every path operator, a dash, nested clips, colour spaces by tint transform, a Separation image; text alone reads no vectors", PdfVectorHandMade),
    ];

    internal static readonly (string Name, Action<Device> Run)[] DocumentDrawerGroup =
    [
        ("documents drawers: the library's coverage drawer is registered and passes the drawer suite (with a font); every D3, D4 and D7 sample page drawn at its reference's resolution matches pdfium's render (after a 3 x 3 mean of both, the share of pixels off by more than 32 levels within the page's tolerance; every score printed); the knockout group's and every shading's probes within their tolerance", PdfDrawSamples),
        ("documents drawers: pages the pdf writer writes draw back as the library draws the same parts in memory (paths, clips, dashes, a glyph run, the page of shapes) and an image page as its image; a scan page draws its image, and its hidden text only when asked; a run whose font is neither embedded nor found is left out and counted, never thrown", PdfDrawWritten),
        ("documents drawers: the fill rules (nonzero over overlaps either way, even-odd) on hand-made outlines, the glyph rule as it was; a large page at a high resolution drawn band by band with bounded memory; pages drawn in parallel give the same pixels as one by one (timed); drawing timed cold and warm apart (time and managed allocation a page at 100 and 300 dpi for a vector, a text and a scan page)", PdfDrawRulesAndSpeed),
        ("documents drawers: the drawer's settings: the defaults are the numbers it always had (a band's 128 Ki pixels and 8 to 1,024 rows, 1,024 shading steps, a largest tile of 1 Mi pixels, a substitute face's 0.75 to 4/3 stretch, 0.21 slant and 3% stroke, flatness 1/32 and 0.1 pixel), each variable read (IDRAK_DOCUMENT_BAND_PIXELS, _SHADING_STEPS, _MAX_TILE_PIXELS, _FILL_FLATNESS, _STROKE_FLATNESS), a value out of range refused naming its setting; fewer shading steps, a smaller tile, coarser flatness and a substitute's slant, stroke and stretch each change a sample's drawing, finer ones stay within pdfium's tolerance; the rasterizer takes the flatness, through an app's guard too", PdfDrawSettings),
        ("documents drawers: bands never change the drawing: the nonzero and even-odd fills of outlines into bands of a target (one, seven and 64 rows, y moved by whole rows with and without a fraction) are the whole target's fill row for row; every D3, D4 and D7 sample page drawn with one-row bands, seven-row bands and the whole page as one band is the default bands' drawing, byte for byte, with the same issues", PdfDrawBands),
        ("documents drawers: clips' memory: pages of 10, 100 and 1,000 clips as the pdf writer writes them (siblings under one page clip, each part under an upright rectangle, a turned one, or three nested) render every band within one fixed allocation past the shared pools whatever the count, and each clip adds no more than its outline made ready to a draw; the large set's deep-forms page (one form drawn 1,000 times, each under its /BBox) drawn within a fixed bound beyond its pixels, when the set is there (timed cold and warm)", PdfDrawClipMemory),
        ("documents drawers: clips draw as they did: a clip's coverage worked over its own box by the library's rasterizer is the band-wide coverage value for value (upright rectangles at sub-pixel and whole-pixel edges, smaller than a pixel, of no width or height, off and past the surface, closed by a fifth line, by either rule; turned ones, a circle, an even-odd frame; nested in each other, six deep, missing their parent; bands of 1, 7, 64 rows and one); a crafted page of every kind of clip under parts that reach past them (a shading, an image, a stroke, a tiling wholly outside its clip, a text clip whose runs are not there, a group, a soft mask) draws the same bytes and issues as through an app's rasterizer (the band-wide clips), at 150 dpi, 2 pixels a point in grey, one-row bands and one band", PdfDrawClipsExact),
    ];

    // ---- documents pdf: vector -------------------------------------------------------------------------------------------

    private static List<PdfSampleEntry> PdfVectorSampleList()
    {
        var samples = PdfSamples().Where(s => s.Readable && s.Phases.Contains("D7")).ToList();
        Check(samples.Count >= 11, $"{samples.Count} D7 samples; the generator writes 11 (make them again: tools/documents/make_fixtures.py)");
        return samples;
    }

    private static void PdfVectorSamples(Device device)
    {
        CpuOnly(device);
        int paths = 0, clips = 0, shadings = 0, runs = 0, images = 0;
        foreach (var sample in PdfVectorSampleList())
        {
            using var document = DocumentFormats.Open(sample.Path);
            foreach (var expectedPage in sample.Expected["pages"]!.AsArray())
            {
                int index = (int)expectedPage!["index"]!;
                string what = $"{sample.Name}: page {index}";
                var page = document.ReadPage(index, DocumentParts.All);
                var damaged = page.Issues.Where(i => i.Kind is DocumentIssue.DamagedStream or DocumentIssue.NotSupported).ToList();
                Check(damaged.Count == 0, $"{what}: issues {string.Join("; ", damaged.Select(i => $"{i.Kind}: {i.Detail}"))}");

                var expectedPaths = (expectedPage["paths"] as JsonArray ?? []).Select(p => p!.AsObject()).OrderBy(p => (int)p["order"]!).ToList();
                Check(page.Paths.Count == expectedPaths.Count, $"{what}: {page.Paths.Count} paths, expected {expectedPaths.Count}");
                for (int k = 0; k < expectedPaths.Count; k++)
                {
                    PdfVectorPath($"{what}: path {k} (order {expectedPaths[k]["order"]})", page.Paths[k], expectedPaths[k]);
                    paths++;
                }

                var expectedClips = (expectedPage["clips"] as JsonArray ?? []).Select(c => c!.AsObject()).ToList();
                Check(page.Clips.Count == expectedClips.Count, $"{what}: {page.Clips.Count} clips, expected {expectedClips.Count}");
                for (int k = 0; k < expectedClips.Count; k++)
                {
                    var e = expectedClips[k];
                    var clip = page.Clips[k];
                    string c = $"{what}: clip {k}";
                    Check(clip.Clip == ((int?)e["parent"] ?? -1), $"{c}: made under clip {clip.Clip}, expected {e["parent"]}");
                    if (e["textRun"] is { } run)
                    {
                        Check(clip.ClipRun == (int)run && clip.ClipRunCount >= 1 && clip.IsEmpty, $"{c}: a clip of text from run {clip.ClipRun} ({clip.ClipRunCount}), expected run {run}");
                    }
                    else
                    {
                        Check(clip.Fill == ((string?)e["rule"] == "evenodd" ? DocumentFill.EvenOdd : DocumentFill.NonZero), $"{c}: rule {clip.Fill}, expected {e["rule"]}");
                        PdfVectorPoints(c, clip, e);
                    }

                    clips++;
                }

                var expectedShadings = (expectedPage["shadings"] as JsonArray ?? []).Select(s => s!.AsObject()).OrderBy(s => (int)s["order"]!).ToList();
                Check(page.Shadings.Count == expectedShadings.Count, $"{what}: {page.Shadings.Count} shadings, expected {expectedShadings.Count}");
                for (int k = 0; k < Math.Min(page.Shadings.Count, expectedShadings.Count); k++)
                {
                    var e = expectedShadings[k];
                    var s = page.Shadings[k];
                    var kind = (int)e["shadingType"]! switch { 1 => DocumentShadingKind.Function, 2 => DocumentShadingKind.Axial, 3 => DocumentShadingKind.Radial, _ => DocumentShadingKind.Mesh };
                    Check(s.Kind == kind && s.Order == (int)e["order"]! && s.Clip == ((int?)e["clip"] ?? -1) && MathF.Abs(s.Alpha - (float)e["alpha"]!["fill"]!) < 1e-4f,
                        $"{what}: shading {k}: {s.Kind} at {s.Order} under clip {s.Clip}, alpha {s.Alpha}; expected {kind} at {e["order"]} under {e["clip"]}");
                    Check(s.Kind != DocumentShadingKind.Mesh || s.Triangles.Length >= 15, $"{what}: shading {k}: a mesh of {s.Triangles.Length / 15} triangles");
                    shadings++;
                }

                foreach (var e in expectedPage["runs"]!.AsArray().Select(r => r!.AsObject()).Where(r => r["fillColour"] is not null))
                {
                    var run = page.Runs.FirstOrDefault(r => r.Order == (int)e["order"]!);
                    if (run is null)
                    {
                        Check((bool)e["hidden"]!, $"{what}: no run at {e["order"]}");
                        continue;
                    }

                    string r = $"{what}: run {run.Order} \"{run.Text}\"";
                    PdfVectorColour($"{r}: colour", run.Colour, e["fillColour"]!.AsObject(), (float)e["alpha"]!["fill"]!);
                    Check(run.Clip == ((int?)e["clip"] ?? -1) && run.Group == ((int?)e["group"] ?? -1), $"{r}: clip {run.Clip}, group {run.Group}; expected {e["clip"]}, {e["group"]}");
                    Check(run.Mode is DocumentTextMode.Fill or DocumentTextMode.Hidden || run.StrokeWidth > 0, $"{r}: a stroked run of no width");
                    runs++;
                }

                foreach (var e in (expectedPage["images"] as JsonArray ?? []).Select(i => i!.AsObject()).Where(i => i["clip"] is not null || i["alpha"] is not null))
                {
                    var image = page.Images.FirstOrDefault(i => i.Order == (int)e["order"]!);
                    Check(image is not null, $"{what}: no image at {e["order"]}");
                    Check(image!.Clip == ((int?)e["clip"] ?? -1) && MathF.Abs(image.Alpha - ((float?)e["alpha"]?["fill"] ?? 1f)) < 1e-4f && (image.Stencil is not null) == (bool)e["imageMask"]!,
                        $"{what}: image {image.Order}: clip {image.Clip}, alpha {image.Alpha}, stencil {image.Stencil}; expected {e["clip"]}, {e["alpha"]}, {e["imageMask"]}");
                    images++;
                }

                if (expectedPage["groups"] is JsonArray groups)
                {
                    Check(page.Groups.Count == groups.Count && groups.Select((g, i) => page.Groups[i].Isolated == (bool)g!["isolated"]! && page.Groups[i].Knockout == (bool)g["knockout"]!).All(b => b),
                        $"{what}: groups {string.Join(", ", page.Groups.Select(g => $"{g.Isolated}/{g.Knockout}"))}");
                }
            }
        }

        Console.WriteLine($"    {paths} paths, {clips} clips, {shadings} shadings, {runs} runs' and {images} images' paint read as their JSON says");
        Check(paths >= 100 && clips >= 10 && shadings >= 7, $"{paths} paths, {clips} clips, {shadings} shadings: fewer than the samples hold");
    }

    // A path against its JSON (see tools/documents/README.md, "Vector content").
    private static void PdfVectorPath(string what, DocumentPath path, JsonObject e)
    {
        Check(path.Order == (int)e["order"]!, $"{what}: at {path.Order}");
        PdfVectorPoints(what, path, e);
        var fill = (string?)e["fill"] switch { "nonzero" => DocumentFill.NonZero, "evenodd" => DocumentFill.EvenOdd, _ => DocumentFill.None };
        Check(path.Fill == fill, $"{what}: fill {path.Fill}, expected {fill}");
        if (e["stroke"] is JsonObject stroke)
        {
            Check(path.Stroke is { } w && MathF.Abs(w - (float)stroke["width"]!) <= 2e-3f, $"{what}: stroke {path.Stroke}, expected {stroke["width"]}");
            var cap = (string)stroke["cap"]! switch { "round" => DocumentLineCap.Round, "square" => DocumentLineCap.Square, _ => DocumentLineCap.Butt };
            var join = (string)stroke["join"]! switch { "round" => DocumentLineJoin.Round, "bevel" => DocumentLineJoin.Bevel, _ => DocumentLineJoin.Miter };
            Check(path.Cap == cap && path.Join == join && MathF.Abs(path.MiterLimit - (float)stroke["miterLimit"]!) < 1e-3f,
                $"{what}: {path.Cap} {path.Join} {path.MiterLimit}, expected {cap} {join} {stroke["miterLimit"]}");
            float[] dash = [.. stroke["dash"]!["pattern"]!.AsArray().Select(d => (float)d!)];
            bool solid = dash.Length == 0 || dash.All(d => d == 0);
            Check(solid ? path.Dash.IsEmpty : path.Dash.Length == dash.Length && dash.Select((d, i) => MathF.Abs(d - path.Dash.Span[i]) < 2e-3f).All(b => b)
                    && MathF.Abs(path.DashPhase - (float)stroke["dash"]!["phase"]!) < 2e-3f,
                $"{what}: dash [{string.Join(" ", path.Dash.ToArray())}] from {path.DashPhase}, expected [{string.Join(" ", dash)}] from {stroke["dash"]!["phase"]}");
            PdfVectorPaint($"{what}: stroke", path.StrokeColour, path.StrokePattern, e["strokeColour"]!.AsObject(), (float)e["alpha"]!["stroke"]!);
        }
        else
        {
            Check(path.Stroke is null, $"{what}: stroked ({path.Stroke}), expected no stroke");
        }

        if (fill != DocumentFill.None)
        {
            PdfVectorPaint($"{what}: fill", path.FillColour, path.FillPattern, e["fillColour"]!.AsObject(), (float)e["alpha"]!["fill"]!);
        }

        Check(path.Clip == ((int?)e["clip"] ?? -1) && path.Group == ((int?)e["group"] ?? -1) && (path.SoftMask >= 0) == e["softMask"] is not null,
            $"{what}: clip {path.Clip}, group {path.Group}, soft mask {path.SoftMask}; expected {e["clip"]}, {e["group"]}, {e["softMask"]}");
        string blend = (string?)e["blend"] ?? "Normal";
        Check(path.Blend.ToString() == blend, $"{what}: blend {path.Blend}, expected {blend}");
    }

    // A path's verbs and its points (page points, top-left origin) against the JSON's, within float rounding.
    private static void PdfVectorPoints(string what, DocumentPath path, JsonObject e)
    {
        var verbs = e["verbs"]!.AsArray().Select(v => (string)v! switch
        {
            "move" => OutlineVerb.Move, "line" => OutlineVerb.Line, "cubic" => OutlineVerb.Cubic, "quadratic" => OutlineVerb.Quadratic, _ => OutlineVerb.Close,
        }).ToArray();
        Check(path.Verbs.Span.SequenceEqual(verbs), $"{what}: verbs {string.Join(" ", path.Verbs.ToArray())}, expected {string.Join(" ", verbs)}");
        float[] points = [.. e["points"]!.AsArray().SelectMany(p => p!.AsArray().Select(v => (float)v!))];
        var read = path.Points.Span;
        int wrong = Enumerable.Range(0, Math.Min(points.Length, read.Length)).FirstOrDefault(i => MathF.Abs(points[i] - path.Points.Span[i]) > 2e-3f, -1);
        Check(read.Length == points.Length && wrong < 0, wrong < 0 ? $"{what}: {read.Length / 2} points, expected {points.Length / 2}" : $"{what}: point {wrong / 2} is {read[wrong & ~1]}, {read[wrong | 1]}, expected {points[wrong & ~1]}, {points[wrong | 1]}");
    }

    // A fill's or a stroke's paint: a pattern by its type (and an uncoloured tiling's colour), else the colour.
    private static void PdfVectorPaint(string what, DocumentColour colour, DocumentPattern? pattern, JsonObject e, float alpha)
    {
        if ((string?)e["space"] == "Pattern")
        {
            int type = (int)e["patternType"]!;
            Check(pattern is not null && (type == 2 ? pattern.Shading is not null : pattern.Cell is not null), $"{what}: pattern {(pattern?.Shading is not null ? "shading" : pattern?.Cell is not null ? "tiling" : "none")}, expected type {type}");
            if (e["rgb"] is not null && pattern?.Cell is { } cell)
            {
                // An uncoloured tiling: every part of its cell in the colour given with the pattern.
                foreach (var part in cell.Paths)
                {
                    PdfVectorColour($"{what}: its cell's part", part.Fill != DocumentFill.None ? part.FillColour : part.StrokeColour, e, 1f, ignoreAlpha: true);
                }
            }

            return;
        }

        Check(pattern is null, $"{what}: a pattern where a colour is expected");
        PdfVectorColour(what, colour, e, alpha);
    }

    private static void PdfVectorColour(string what, DocumentColour colour, JsonObject e, float alpha, bool ignoreAlpha = false)
    {
        int[] rgb = [.. e["rgb"]!.AsArray().Select(v => (int)v!)];
        int tolerance = (int?)e["tolerance"] ?? 0;
        if (colour.Space == DocumentColourSpace.Cmyk || (string?)e["space"] is "Indexed" or "Separation" or "DeviceN")
        {
            tolerance += 1;   // the drawn RGB of ink: CMYK through floats here, through 8 bits in the JSON
        }

        int[] read = [(int)MathF.Round(colour.Red * 255), (int)MathF.Round(colour.Green * 255), (int)MathF.Round(colour.Blue * 255)];
        Check(rgb.Zip(read).All(p => Math.Abs(p.First - p.Second) <= tolerance), $"{what}: RGB {string.Join(" ", read)}, expected {string.Join(" ", rgb)} within {tolerance} ({e["space"]} {e["components"]?.ToJsonString()})");
        if ((string?)e["space"] is "DeviceGray" or "DeviceRGB" or "DeviceCMYK")
        {
            float[] given = [.. e["components"]!.AsArray().Select(v => (float)v!)];
            Check(colour.Count == given.Length && given.Select((v, i) => MathF.Abs(v - colour[i]) < 1e-4f).All(b => b), $"{what}: {colour}, given {e["components"]!.ToJsonString()}");
        }

        Check(ignoreAlpha || MathF.Abs(colour.Alpha - alpha) < 1e-4f, $"{what}: alpha {colour.Alpha}, expected {alpha}");
    }

    private static void PdfVectorFonts(Device device)
    {
        CpuOnly(device);
        var sample = PdfVectorSampleList().Single(s => s.Name == "vector-fonts");
        using var document = DocumentFormats.Open(sample.Path);
        var page = document.ReadPage(0, DocumentParts.All);
        var faces = page.Runs.Where(r => r.Face is not null).Select(r => r.Face!.Format).Distinct().ToList();
        Check(page.Runs.Skip(1).All(r => r.Face is not null) && faces.Contains("type1") && faces.Contains("type3") && faces.Contains("cff") && faces.Contains("opentype"),
            $"faces: {string.Join(", ", page.Runs.Select(r => $"\"{r.Text}\" {r.Font}: {r.Face?.Format ?? "none"}"))}");

        // The Type 1 program's glyphs as fontTools reads them (1,000 units an em).
        var type1 = page.Runs.First(r => r.Face?.Format == "type1").Face!;
        var names = PdfPostNames(type1);
        var program = sample.Expected["fontPrograms"]!["type1"]!["glyphs"]!.AsObject();
        int checkedGlyphs = 0;
        foreach (var (name, glyph) in program)
        {
            Check(names.TryGetValue(name, out int id), $"type1: no glyph {name} (it has {names.Count})");
            Check(type1.Advance(id) == (int)glyph!["width"]!, $"type1: {name}'s width {type1.Advance(id)}, expected {glyph["width"]}");
            var expected = new List<(OutlineVerb Verb, float[] Points)>();
            foreach (var step in glyph["outline"]!.AsArray())
            {
                var parts = step!.AsArray();
                var verb = (string)parts[0]! switch { "moveTo" => OutlineVerb.Move, "lineTo" => OutlineVerb.Line, "curveTo" => OutlineVerb.Cubic, _ => OutlineVerb.Close };
                expected.Add((verb, [.. parts.Skip(1).SelectMany(p => p!.AsArray().Select(v => (float)v!))]));
            }

            var outline = type1.Outline(id);
            float[] points = [.. expected.SelectMany(s => s.Points)];
            Check(outline.Verbs.Span.SequenceEqual(expected.Select(s => s.Verb).ToArray()) && outline.Points.Length == points.Length
                  && points.Select((v, i) => MathF.Abs(v - outline.Points.Span[i]) < 1e-3f).All(b => b),
                $"type1: {name}: {string.Join(" ", outline.Verbs.ToArray())} / {string.Join(",", outline.Points.ToArray().Take(8))}, expected {string.Join(" ", expected.Select(s => s.Verb))} / {string.Join(",", points.Take(8))}");
            checkedGlyphs++;
        }

        // Each run's glyphs reach outlines through its face (the space excepted).
        foreach (var run in page.Runs.Where(r => r.Face is not null))
        {
            var face = run.Face!;
            var glyphs = run.Glyphs.Span;
            int drawn = 0;
            foreach (var g in glyphs)
            {
                drawn += g.Glyph > 0 && g.Glyph < face.GlyphCount && !face.Outline(g.Glyph).IsEmpty ? 1 : 0;
            }

            int ink = run.Text.Count(c => !char.IsWhiteSpace(c));
            Check(run.GlyphMatrix is not null && drawn >= ink * 9 / 10, $"\"{run.Text}\" ({face.Format}): {drawn} glyphs with outlines of {glyphs.Length}, {ink} not spaces");
        }

        Console.WriteLine($"    vector-fonts: {faces.Count} kinds of face ({string.Join(", ", faces)}); {checkedGlyphs} Type 1 glyphs as fontTools reads them");
    }

    // A face's glyphs by their names (its "post" table of version 2).
    private static Dictionary<string, int> PdfPostNames(IFontFace face)
    {
        var post = face.Table("post").Span;
        var names = new Dictionary<string, int>(StringComparer.Ordinal);
        if (post.Length < 34)
        {
            return names;
        }

        int count = post[32] << 8 | post[33];
        var strings = new List<string>();
        for (int at = 34 + 2 * count; at < post.Length;)
        {
            int length = post[at];
            strings.Add(Encoding.Latin1.GetString(post.Slice(at + 1, length)));
            at += 1 + length;
        }

        for (int g = 0; g < count; g++)
        {
            int index = (post[34 + 2 * g] << 8 | post[35 + 2 * g]) - 258;
            if ((uint)index < (uint)strings.Count)
            {
                names.TryAdd(strings[index], g);
            }
        }

        return names;
    }

    private static void PdfVectorHandMade(Device device)
    {
        CpuOnly(device);

        // A quadratic and a cubic written, read back: the quadratic as itself, the cubic as a cubic.
        float[] points = [10f, 20f, 60f, 5f, 110f, 20f, 130f, 60f, 150f, 90f, 120f, 100f];
        var path = new DocumentPath([OutlineVerb.Move, OutlineVerb.Quadratic, OutlineVerb.Cubic], points) { Fill = DocumentFill.None, Stroke = 1.5f };
        byte[] written;
        using (var output = new MemoryStream())
        {
            using (var builder = DocumentWriters.Get("pdf").Create(output, new DocumentWriteOptions()))
            {
                builder.AddPage(200, 120).Path(path);
                builder.Finish();
            }

            written = output.ToArray();
        }

        using (var document = DocumentFormats.Open(written.AsMemory(), "quadratic.pdf"))
        {
            var read = document.ReadPage(0, DocumentParts.Paths).Paths.Single();
            Check(read.Verbs.Span.SequenceEqual([OutlineVerb.Move, OutlineVerb.Quadratic, OutlineVerb.Cubic]) && read.Points.Length == points.Length
                  && points.Select((v, i) => MathF.Abs(v - read.Points.Span[i]) < 1e-3f).All(b => b) && read.Stroke == 1.5f,
                $"the quadratic read back: {string.Join(" ", read.Verbs.ToArray())} / {string.Join(", ", read.Points.ToArray())}");
        }

        PdfVectorFunctions();

        // Every path operator, a dash, nested clips and colour spaces through tint transforms, on one page.
        string content = string.Join("\n",
            "q 2 w 1 J 1 j 0.5 0 0 RG [6 3] 1 d 10 10 m 50 10 l 50 50 50 60 70 60 c 80 70 90 70 v 100 70 110 80 y h S Q",
            "q 20 100 60 40 re W n 0.2 g 0 0 200 200 re f q 30 110 20 20 re W* n /CS0 cs 0.5 sc 0 0 200 200 re f Q Q",
            "/CS1 CS 0.25 0.75 SC 4 w 120 20 m 180 20 l S",
            "/GS0 gs 0 0 1 rg 120 100 40 40 re f",
            "/Im0 Do");
        byte[] file = PdfHandMadeFile([
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R /Resources << /ColorSpace << /CS0 [/Separation /Spot /DeviceRGB << /FunctionType 2 /Domain [0 1] /C0 [1 1 1] /C1 [0 0.5 1] /N 1 >>] /CS1 [/DeviceN [/A /B] /DeviceGray 5 0 R] >> /ExtGState << /GS0 << /ca 0.25 /BM /Multiply >> >> /XObject << /Im0 6 0 R >> >> >>",
            PdfHandMadeStream("", Encoding.ASCII.GetBytes(content)),
            PdfHandMadeStream("/FunctionType 4 /Domain [0 1 0 1] /Range [0 1]", "{ add 2 div }"u8.ToArray()),
            PdfHandMadeStream("/Type /XObject /Subtype /Image /Width 2 /Height 1 /BitsPerComponent 8 /ColorSpace [/Separation /Spot /DeviceRGB << /FunctionType 2 /Domain [0 1] /C0 [1 1 1] /C1 [0 0.5 1] /N 1 >>]", [0, 255]),
        ]);
        using (var document = DocumentFormats.Open(file.AsMemory(), "vector-hand-made.pdf"))
        {
            var page = document.ReadPage(0, DocumentParts.All);
            Check(page.Paths.Count == 5 && page.Clips.Count == 2 && page.Images.Count == 1, $"{page.Paths.Count} paths, {page.Clips.Count} clips, {page.Images.Count} images");
            var first = page.Paths[0];
            Check(first.Verbs.Span.SequenceEqual([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Cubic, OutlineVerb.Cubic, OutlineVerb.Cubic, OutlineVerb.Close])
                  && first.Points.Span[4..6].SequenceEqual([50f, 150f]) && first.Points.Span[10..12].SequenceEqual([70f, 140f])
                  && first.Points.Span[18..20].SequenceEqual([110f, 120f]) && first.Points.Span[20..22].SequenceEqual([110f, 120f]),
                $"m l c v y h: {string.Join(" ", first.Verbs.ToArray())} / {string.Join(", ", first.Points.ToArray())}");
            Check(first.Stroke == 2f && first.Cap == DocumentLineCap.Round && first.Join == DocumentLineJoin.Round && first.Dash.Span.SequenceEqual([6f, 3f]) && first.DashPhase == 1f
                  && first.Fill == DocumentFill.None && first.StrokeColour.Space == DocumentColourSpace.Rgb && first.StrokeColour[0] == 0.5f, $"the line state: {first.Stroke} {first.Cap} {first.Join}");
            Check(page.Clips[0].Clip == -1 && page.Clips[1].Clip == 0 && page.Clips[1].Fill == DocumentFill.EvenOdd && page.Paths[1].Clip == 0 && page.Paths[2].Clip == 1 && page.Paths[3].Clip == -1,
                $"nested clips: {string.Join(", ", page.Paths.Select(p => p.Clip))}");
            var spot = page.Paths[2].FillColour;
            Check(MathF.Abs(spot.Red - 0.5f) < 0.003f && MathF.Abs(spot.Green - 0.75f) < 0.003f && MathF.Abs(spot.Blue - 1f) < 0.003f, $"a Separation's colour through its type 2 transform: {spot}");
            var devn = page.Paths[3].StrokeColour;
            Check(MathF.Abs(devn.Red - 0.5f) < 0.003f && devn.Space == DocumentColourSpace.Rgb, $"a DeviceN's colour through its calculator ((0.25 + 0.75) / 2 grey): {devn}");
            var blended = page.Paths[4];
            Check(MathF.Abs(blended.FillColour.Alpha - 0.25f) < 1e-4f && blended.Blend == DocumentBlend.Multiply, $"gs: ca {blended.FillColour.Alpha}, blend {blended.Blend}");
            var pixels = PdfInterleaved(page.Images[0].Decode());
            Check(pixels.AsSpan().SequenceEqual((byte[])[255, 255, 255, 0, 128, 255]), $"a Separation image: {string.Join(" ", pixels)}");

            var text = document.ReadPage(0, DocumentParts.Text);
            Check(text.Paths.Count == 0 && text.Clips.Count == 0, "text alone reads no vectors");
        }
    }

    // Functions of each type on values worked out by hand.
    private static void PdfVectorFunctions()
    {
        byte[] file = PdfHandMadeFile([
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [8 0 R] /Count 1 >>",
            "<< /FunctionType 2 /Domain [0 1] /C0 [0 1] /C1 [1 0] /N 2 >>",
            PdfHandMadeStream("/FunctionType 0 /Domain [0 1] /Range [0 1 0 1] /Size [3] /BitsPerSample 8", [0, 255, 255, 0, 0, 255]),
            "<< /FunctionType 3 /Domain [0 1] /Functions [3 0 R 3 0 R] /Bounds [0.5] /Encode [0 1 1 0] >>",
            PdfHandMadeStream("/FunctionType 4 /Domain [0 10 0 10] /Range [-100 100 -100 100 0 1]",
                "{ 2 copy mul 3 1 roll exch sub dup 0 lt { neg } if 1 index 10 gt { 1 } { 0 } ifelse }"u8.ToArray()),
            PdfHandMadeStream("/FunctionType 4 /Domain [0 1] /Range [0 1]", "{ dup dup dup dup dup dup dup dup dup }"u8.ToArray()),
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 10 10] >>",
        ]);
        using var document = (PdfDocument)DocumentFormats.Default("pdf")!.Open(new MemoryStream(file), "functions.pdf", new DocumentOptions());
        var objects = document.File;
        double[] Evaluate(int number, params double[] input)
        {
            var function = PdfFunction.Read(objects, PdfValue.Reference(number, 0));
            var output = new double[function.Outputs];
            function.Evaluate(input, output);
            return output;
        }

        bool Near(double[] a, params double[] b) => a.Length == b.Length && a.Zip(b).All(p => Math.Abs(p.First - p.Second) < 1e-9);
        Check(Near(Evaluate(3, 0.5), 0.25, 0.75), $"type 2 at 0.5: {string.Join(" ", Evaluate(3, 0.5))}");
        Check(Near(Evaluate(4, 0.25), 0.5, 0.5) && Near(Evaluate(4, 1), 0, 1), $"type 0 at 0.25 and 1: {string.Join(" ", Evaluate(4, 0.25))}; {string.Join(" ", Evaluate(4, 1))}");
        Check(Near(Evaluate(5, 0.25), 0.25, 0.75) && Near(Evaluate(5, 0.75), 0.25, 0.75), $"type 3 (the second encoded backwards): {string.Join(" ", Evaluate(5, 0.75))}");
        Check(Near(Evaluate(6, 3, 5), 15, 2, 1) && Near(Evaluate(6, 0, 7), 0, 7, 0), $"type 4 (copy, roll, if, ifelse, index): {string.Join(" ", Evaluate(6, 3, 5))}; {string.Join(" ", Evaluate(6, 0, 7))}");
        Check(Evaluate(7, 0.5).Length == 1, "type 4 leaving extra values: the last one given");
    }

    // ---- documents drawers -----------------------------------------------------------------------------------------------

    // Where drawn pages are saved, and a folder of fonts for runs whose font is not embedded (DejaVu Sans of the test data).
    private static string PdfDrawnFolder() => Directory.CreateDirectory(RepositoryPath("test-output", "documents", "drawn")).FullName;

    private static string PdfDrawFonts()
    {
        string folder = Directory.CreateDirectory(RepositoryPath("test-output", "documents", "draw-fonts")).FullName;
        string target = Path.Combine(folder, "dejavu-sans-subset.ttf");
        if (!File.Exists(target))
        {
            File.Copy(FontData("dejavu-sans-subset.ttf"), target);
        }

        return folder;
    }

    // The tolerance a sample page is held to: the share of pixels off by more than 32 levels after a 3 x 3 mean of both
    // drawings. Pages of text in a standard font not embedded draw it with another font (DejaVu Sans here, every style of
    // it; pdfium its own substitutes, one a style): their glyphs differ in shape, so they are held to the share their text
    // covers.
    private static double PdfDrawTolerance(PdfSampleEntry sample, int page) =>
        sample.Expected["pages"]![page]!["runs"]!.AsArray().Any(r => (string?)r!["fontKind"] == "standard14") ? 0.08 : 0.02;

    // Parts pdfium draws by rules of its own where the samples' JSON and the library follow the documents plan's (D4's
    // colour rules): left out of the comparison by their boxes, a path's colour checked against its JSON instead.
    private static readonly Dictionary<string, (string Kind, int[] Orders, string Why)> PdfDrawGaps = new()
    {
        ["vector-colour-spaces"] = ("paths", [3, 7, 11, 14, 15, 17],
            "pdfium turns CMYK ink (and Separation and DeviceN into CMYK) into light by a rule of its own, not D4's that the JSON states, and paints a DeviceN whose tint transform is a one-input stitching function white"),
        ["colour-spaces"] = ("images", [3, 5, 6, 11], "pdfium's own CMYK and Lab conversions, and an index past the highest painted otherwise than clamped (D4's rule)"),
        ["predictors"] = ("images", [7], "pdfium misreads the TIFF predictor on 4-bit samples (the pixels are checked exactly by the D1 and D4 groups)"),
    };

    private static void PdfDrawSamples(Device device)
    {
        CpuOnly(device);
        var drawer = DocumentPageDrawers.Get(DocumentPageDrawers.Coverage);
        Check(DocumentPageDrawers.Origin(DocumentPageDrawers.Coverage) == Overrides.Library && DocumentPageDrawers.Default(DocumentPageDrawers.Coverage) is not null,
            $"the coverage drawer registered by the library: {DocumentPageDrawers.Describe()}");
        Conformance.CheckDocumentPageDrawer(drawer, SubsetFace()).ThrowIfFailed();

        string fonts = PdfDrawFonts(), drawn = PdfDrawnFolder();
        var png = ImageCodecs.Get("png");
        var scores = new List<string>();
        var failures = new List<string>();
        int pages = 0;
        foreach (var sample in PdfSamples().Where(s => s.Readable && (s.Phases.Contains("D3") || s.Phases.Contains("D4") || s.Phases.Contains("D7"))))
        {
            string folder = Path.GetDirectoryName(sample.Path)!;
            double dpi = (double?)sample.Expected["reference"]?["dpi"] ?? 100;
            using var document = DocumentFormats.Open(sample.Path);
            for (int index = 0; index < document.PageCount; index++)
            {
                string reference = Path.Combine(folder, $"{sample.Name}.p{index + 1}.png");
                if (!File.Exists(reference))
                {
                    continue;
                }

                var expected = png.Decode(File.ReadAllBytes(reference));
                var drawing = document.DrawPage(index, new DocumentDrawOptions { Dpi = (float)dpi, Fonts = fonts });
                File.WriteAllBytes(Path.Combine(drawn, $"{sample.Name}.p{index + 1}.png"), ImageEncoders.Get("png").Encode(drawing.Image));
                string what = $"{sample.Name}: page {index + 1}";
                // pdfium's render is the page's size times the scale rounded up (its last column and row a part pixel),
                // the drawer's rounded: the two are compared over the region they share.
                if (Math.Abs(drawing.Image.Width - expected.Width) > 1 || Math.Abs(drawing.Image.Height - expected.Height) > 1)
                {
                    failures.Add($"{what}: {drawing.Image.Width} x {drawing.Image.Height} pixels, pdfium's {expected.Width} x {expected.Height}");
                    continue;
                }

                bool[]? skip = null;
                if (PdfDrawGaps.TryGetValue(sample.Name, out var gap))
                {
                    skip = PdfDrawSkip(drawing.Image, sample.Expected["pages"]![index]!.AsObject(), gap.Kind, gap.Orders, dpi, what, failures);
                }

                var (share, mean) = PdfDrawDifference(drawing.Image, expected, 32, skip);
                double tolerance = PdfDrawTolerance(sample, index);
                scores.Add(string.Create(CultureInfo.InvariantCulture,
                    $"{what}: {share * 100:F2}% off by more than 32 levels (mean {mean:F2}), within {tolerance * 100:F0}%{(skip is null ? "" : $" ({gap.Kind} {string.Join(", ", gap.Orders)} left out: {gap.Why})")}{(share > tolerance ? ": PAST IT" : "")}"));
                if (share > tolerance)
                {
                    failures.Add(scores[^1]);
                }

                var badIssues = drawing.Issues.Where(i => i.Kind is DocumentIssue.NotDrawn or DocumentIssue.MissingFont).ToList();
                if (badIssues.Count > 0)
                {
                    failures.Add($"{what}: {string.Join("; ", badIssues.Select(i => $"{i.Kind}: {i.Detail}"))}");
                }

                PdfDrawProbes(what, drawing.Image, sample.Expected["pages"]![index]!.AsObject(), dpi, failures);
                pages++;
            }
        }

        foreach (string score in scores)
        {
            Console.WriteLine($"    {score}");
        }

        Check(failures.Count == 0, $"{failures.Count} pages past their tolerance:\n      {string.Join("\n      ", failures)}");
        Check(pages >= 50, $"{pages} sample pages drawn; the D3, D4 and D7 samples hold more (make them again: tools/documents/make_fixtures.py)");
    }

    // The colours a page's JSON pins down: a group's probes (what a knockout group must give) and each shading's samples.
    private static void PdfDrawProbes(string what, ImageData image, JsonObject page, double dpi, List<string> failures)
    {
        var probes = new List<(float X, float Y, int[] Rgb, int Tolerance, string Note)>();
        foreach (var group in page["groups"] as JsonArray ?? [])
        {
            foreach (var probe in group!["probes"]!.AsArray())
            {
                probes.Add(((float)probe!["point"]![0]!, (float)probe["point"]![1]!, [.. probe["rgb"]!.AsArray().Select(v => (int)v!)], (int)probe["tolerance"]!, $"group {group["index"]}"));
            }
        }

        foreach (var shading in page["shadings"] as JsonArray ?? [])
        {
            foreach (var sample in shading!["samples"]!.AsArray())
            {
                probes.Add(((float)sample!["point"]![0]!, (float)sample["point"]![1]!, [.. sample["rgb"]!.AsArray().Select(v => (int)v!)], (int)sample["tolerance"]! + 2, $"shading {shading["resource"]}"));
            }
        }

        float scale = (float)(dpi / 72);
        foreach (var (x, y, rgb, tolerance, note) in probes)
        {
            int px = (int)(x * scale), py = (int)(y * scale);
            if ((uint)px >= (uint)image.Width || (uint)py >= (uint)image.Height)
            {
                continue;
            }

            int[] got = [.. Enumerable.Range(0, 3).Select(c => (int)MathF.Round(image.Pixels[Math.Min(c, image.Channels - 1) * image.Width * image.Height + py * image.Width + px] * 255))];
            if (rgb.Zip(got).Any(p => Math.Abs(p.First - p.Second) > tolerance))
            {
                failures.Add($"{what}: {note} at ({x}, {y}): {string.Join(" ", got)}, expected {string.Join(" ", rgb)} within {tolerance}");
            }
        }
    }

    // Both drawings as 8-bit RGB over the region they share from the top-left corner, each a 3 x 3 mean (the edges' half
    // pixels and anti-aliasing forgiven), then the share of pixels whose largest channel difference passes `levels`, and
    // the mean of the differences.
    internal static (double Share, double Mean) PdfDrawDifference(ImageData a, ImageData b, int levels, bool[]? skip = null)
    {
        int w = Math.Min(a.Width, b.Width), h = Math.Min(a.Height, b.Height);
        float[] ba = PdfBlur(a, w, h), bb = PdfBlur(b, w, h);
        long off = 0, counted = 0;
        double sum = 0;
        for (int i = 0; i < w * h; i++)
        {
            if (skip is not null && skip[(i / w) * a.Width + i % w])
            {
                continue;
            }

            counted++;
            float worst = 0;
            for (int c = 0; c < 3; c++)
            {
                worst = MathF.Max(worst, MathF.Abs(ba[3 * i + c] - bb[3 * i + c]));
            }

            off += worst > levels ? 1 : 0;
            sum += worst;
        }

        return ((double)off / Math.Max(1, counted), sum / Math.Max(1, counted));
    }

    // The pixels of the parts left out (their boxes and 2 pixels round), and each left-out path's colour in the drawing
    // at its box's middle against its JSON's.
    private static bool[] PdfDrawSkip(ImageData image, JsonObject page, string kind, int[] orders, double dpi, string what, List<string> failures)
    {
        var skip = new bool[image.Width * image.Height];
        float scale = (float)(dpi / 72);
        foreach (var part in (page[kind] as JsonArray ?? []).Select(p => p!.AsObject()).Where(p => orders.Contains((int)p["order"]!)))
        {
            float[] box = [.. (part["bbox"] ?? part["box"])!.AsArray().Select(v => (float)v!)];
            int x0 = Math.Max(0, (int)(box[0] * scale) - 2), y0 = Math.Max(0, (int)(box[1] * scale) - 2);
            int x1 = Math.Min(image.Width, (int)((box[0] + box[2]) * scale) + 3), y1 = Math.Min(image.Height, (int)((box[1] + box[3]) * scale) + 3);
            for (int y = y0; y < y1; y++)
            {
                skip.AsSpan(y * image.Width + x0, Math.Max(0, x1 - x0)).Fill(true);
            }

            if (kind == "paths" && (part["fillColour"] ?? part["strokeColour"]) is JsonObject colour)
            {
                int px = (int)((box[0] + box[2] / 2) * scale), py = (int)((box[1] + box[3] / 2) * scale);
                int[] rgb = [.. colour["rgb"]!.AsArray().Select(v => (int)v!)];
                int[] got = [.. Enumerable.Range(0, 3).Select(c => (int)MathF.Round(image.Pixels[Math.Min(c, image.Channels - 1) * image.Width * image.Height + py * image.Width + px] * 255))];
                if (rgb.Zip(got).Any(p => Math.Abs(p.First - p.Second) > 3))
                {
                    failures.Add($"{what}: the path at {part["order"]} drawn {string.Join(" ", got)} at its middle, its JSON's colour {string.Join(" ", rgb)}");
                }
            }
        }

        return skip;
    }

    private static float[] PdfBlur(ImageData image, int w, int h)
    {
        int stride = image.Width, size = image.Width * image.Height;
        var pixels = image.Pixels;
        var output = new float[3 * w * h];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                for (int c = 0; c < 3; c++)
                {
                    int plane = Math.Min(c, image.Channels - 1) * size;
                    float s = 0;
                    int n = 0;
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            int xx = x + dx, yy = y + dy;
                            if ((uint)xx < (uint)w && (uint)yy < (uint)h)
                            {
                                s += pixels[plane + yy * stride + xx];
                                n++;
                            }
                        }
                    }

                    output[3 * (y * w + x) + c] = s / n * 255f;
                }
            }
        }

        return output;
    }

    private static void PdfDrawWritten(Device device)
    {
        CpuOnly(device);
        var face = SubsetFace();
        var options = new DocumentDrawOptions { PixelsPerPoint = 2f };

        // A page of paths, clips, dashes and a glyph run, drawn as parts in memory and as the pdf writer writes them.
        float[] star = [.. Enumerable.Range(0, 5).SelectMany(k => new[] { 80 + 50 * MathF.Sin(k * 4 * MathF.PI / 5), 70 - 50 * MathF.Cos(k * 4 * MathF.PI / 5) })];
        var ring = new DocumentPath([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close], star) { Fill = DocumentFill.EvenOdd, FillColour = DocumentColour.Rgb(0.2f, 0.4f, 0.9f), Order = 0 };
        var dashed = new DocumentPath([OutlineVerb.Move, OutlineVerb.Cubic], [150f, 20f, 200f, 120f, 260f, 0f, 290f, 110f])
        {
            Fill = DocumentFill.None, Stroke = 4f, Cap = DocumentLineCap.Round, Join = DocumentLineJoin.Round, Dash = new float[] { 9f, 5f }, DashPhase = 2f, StrokeColour = DocumentColour.Rgb(0.8f, 0.1f, 0.1f), Order = 1,
        };
        var clip = DocumentPath.Rectangle(new DocumentBox(20f, 130f, 140f, 50f));
        var clipped = new DocumentPath(DocumentPath.Rectangle(new DocumentBox(0f, 120f, 300f, 80f)).Verbs.ToArray(), DocumentPath.Rectangle(new DocumentBox(0f, 120f, 300f, 80f)).Points.ToArray())
        {
            FillColour = DocumentColour.Cmyk(0.1f, 0.5f, 0.9f, 0f).WithAlpha(0.6f), Clip = 0, Order = 2,
        };
        var glyphs = PdfShapedGlyphs(face, "Quarterly report", 170f, 180f, 18f);
        var run = new DocumentTextRun("Quarterly report", new DocumentBox(170f, 162f, 140f, 22f), 18f)
        {
            Glyphs = glyphs, Face = face, Baseline = 180f, Order = 3, Colour = DocumentColour.Rgb(0.1f, 0.5f, 0.2f),
        };
        var inMemory = new DocumentPage(0, 320f, 200f) { Paths = [ring, dashed, clipped], Clips = [clip], Runs = [run] };
        byte[] written = PdfWriteDocument(b =>
        {
            var page = b.AddPage(320f, 200f);
            page.Path(ring);
            page.Path(dashed);
            page.Save();
            page.Clip(clip);
            page.Path(clipped);
            page.Restore();
            page.Glyphs(face, 18f, glyphs, "Quarterly report", DocumentTextMode.Fill, run.Colour);
        });
        using (var document = DocumentFormats.Open(written.AsMemory(), "written-vector.pdf"))
        {
            var expected = DocumentPageDrawers.Draw(inMemory, options);
            var back = document.DrawPage(0, options);
            File.WriteAllBytes(Path.Combine(PdfDrawnFolder(), "written-vector.png"), ImageEncoders.Get("png").Encode(back.Image));
            var (share, mean) = PdfDrawDifference(back.Image, expected.Image, 8);
            Check(share == 0 && mean < 0.05, string.Create(CultureInfo.InvariantCulture, $"the written vector page drawn back: {share * 100:F3}% off by more than 8 levels, mean {mean:F4}"));
        }

        // The page of shapes: its paths as DocumentShapes gives them, drawn in memory and as written.
        var plan = new ShapePlan(
            [new ShapePath([40, 40, 560, 40, 560, 300, 40, 300], closed: true, width: 1.5f), new ShapePath([40, 120, 560, 120], width: 0.5f, dash: [4, 2], tone: 0.6f), new ShapePath([300, 200])],
            [new ShapeArea([40, 40, 560, 40, 560, 80, 40, 80], 0.15f)]);
        var style = new DocumentShapeStyle { Scale = 0.75f, StrokeWidth = 2f };
        byte[] shapes = PdfWriteDocument(b => DocumentShapes.Write(b.AddPage(450, 255), plan, style));
        using (var document = DocumentFormats.Open(shapes.AsMemory(), "written-shapes.pdf"))
        {
            var paths = DocumentShapes.Paths(plan, style).Select((p, i) => new DocumentPath(p.Verbs.ToArray(), p.Points.ToArray())
            {
                Fill = p.Fill, Stroke = p.Stroke, Cap = p.Cap, Join = p.Join, MiterLimit = p.MiterLimit, Dash = p.Dash, DashPhase = p.DashPhase,
                FillColour = p.FillColour, StrokeColour = p.StrokeColour, Order = i,
            }).ToList();
            var expected = DocumentPageDrawers.Draw(new DocumentPage(0, 450, 255) { Paths = paths }, options);
            var (share, mean) = PdfDrawDifference(document.DrawPage(0, options).Image, expected.Image, 8);
            Check(share == 0 && mean < 0.05, string.Create(CultureInfo.InvariantCulture, $"the written page of shapes drawn back: {share * 100:F3}% off by more than 8 levels, mean {mean:F4}"));
        }

        // An image page: its image, pixel for pixel, at one pixel a point.
        var lineArt = new ImageData([.. Enumerable.Range(0, 3 * 40 * 64).Select(i => (i * 37 % 251) / 255f)], 3, 40, 64);
        byte[] imagePage = PdfWriteDocument(b => b.AddPage(64, 40).Image(lineArt, new DocumentBox(0, 0, 64, 40), DocumentImageEncoding.Flate));
        using (var document = DocumentFormats.Open(imagePage.AsMemory(), "written-image.pdf"))
        {
            var back = document.DrawPage(0, new DocumentDrawOptions { PixelsPerPoint = 1f });
            var expected = document.ReadPage(0, DocumentParts.Images).Images[0].Decode();
            Check(back.Image.Width == 64 && back.Image.Height == 40 && PdfInterleaved(back.Image).AsSpan().SequenceEqual(PdfInterleaved(expected)),
                "the image page drawn back is its image, pixel for pixel");
        }

        // A scan page: its image; its hidden text only when asked.
        var scanSample = PdfSamples().Single(s => s.Name == "scan-ocr");
        using (var document = DocumentFormats.Open(scanSample.Path))
        {
            var read = document.ReadPage(0, DocumentParts.All);
            var image = read.Images.Single();
            float dpi = image.PixelWidth / read.Width * 72;
            var plain = document.DrawPage(0, new DocumentDrawOptions { Dpi = dpi });
            var imageOnly = document.DrawPage(0, new DocumentDrawOptions { Dpi = dpi, Parts = DocumentParts.Images });
            var textOnly = document.DrawPage(0, new DocumentDrawOptions { Dpi = dpi, Parts = DocumentParts.Text, Hidden = true, Fonts = PdfDrawFonts() });
            var decoded = image.Decode();
            Check(PdfInterleaved(plain.Image).AsSpan().SequenceEqual(PdfInterleaved(imageOnly.Image)), "the scan page drawn is its image alone (its hidden text is not drawn unless asked)");
            var (share, _) = PdfDrawDifference(imageOnly.Image, new ImageData(decoded.Resize(3, imageOnly.Image.Height, imageOnly.Image.Width), 3, imageOnly.Image.Height, imageOnly.Image.Width), 8);
            Check(share < 0.001, string.Create(CultureInfo.InvariantCulture, $"the scan's image drawn over its page: {share * 100:F3}% off its decoded pixels"));
            int inked = PdfInterleaved(textOnly.Image).Count(v => v < 128);
            Check(inked > 100 && textOnly.Issues.All(i => i.Kind != DocumentIssue.MissingFont), $"its hidden text drawn when asked: {inked} dark values; issues {string.Join("; ", textOnly.Issues.Select(i => i.Detail))}");
        }

        // A standard font with no fonts folder: its runs left out and counted, nothing thrown.
        var standard = PdfSamples().Single(s => s.Name == "standard14");
        using (var document = DocumentFormats.Open(standard.Path))
        {
            int runs = document.ReadPage(0, DocumentParts.Text).Runs.Count;
            var drawing = document.DrawPage(0, new DocumentDrawOptions { Dpi = 72, Fonts = null });
            int missing = drawing.Issues.Where(i => i.Kind == DocumentIssue.MissingFont).Sum(i => i.Count);
            Check(missing == runs && PdfInterleaved(drawing.Image).All(v => v == 255), $"standard14 without fonts: {missing} runs counted missing of {runs}, the page {(PdfInterleaved(drawing.Image).All(v => v == 255) ? "blank" : "inked")}");
            var empty = document.DrawPage(0, new DocumentDrawOptions { Dpi = 72, Fonts = Directory.CreateDirectory(RepositoryPath("test-output", "documents", "no-fonts")).FullName });
            Check(empty.Issues.Where(i => i.Kind == DocumentIssue.MissingFont).Sum(i => i.Count) == runs, "an empty fonts folder: every run counted");
        }
    }

    private static DocumentGlyph[] PdfShapedGlyphs(IFontFace face, string text, float x, float baseline, float size)
    {
        var glyphs = new List<DocumentGlyph>();
        float scale = size / face.Metrics.UnitsPerEm, at = x;
        foreach (var rune in text.EnumerateRunes())
        {
            int glyph = face.Glyph(rune.Value);
            float advance = face.Advance(glyph) * scale;
            glyphs.Add(new DocumentGlyph(glyph, at, baseline, advance));
            at += advance;
        }

        return [.. glyphs];
    }

    private static byte[] PdfWriteDocument(Action<IDocumentBuilder> build)
    {
        using var output = new MemoryStream();
        using (var builder = DocumentWriters.Get("pdf").Create(output, new DocumentWriteOptions { ImageEncoding = DocumentImageEncoding.Flate }))
        {
            build(builder);
            builder.Finish();
        }

        return output.ToArray();
    }

    private static void PdfDrawRulesAndSpeed(Device device)
    {
        CpuOnly(device);
        var rasterizer = OutlineRasterizers.Get();

        // Two squares wound the same way and a third against them: nonzero fills the first two's overlap (wound twice) and
        // leaves the third's overlap with the second (wound once each way, so not at all), even-odd leaves both; the glyph
        // rule as it always was.
        float[] Square(float x, float y, float s, bool reverse) => reverse ? [x, -y, x, -(y + s), x + s, -(y + s), x + s, -y] : [x, -y, x + s, -y, x + s, -(y + s), x, -(y + s)];
        var outline = new GlyphOutline(
            [OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close, OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close,
                OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close],
            [.. Square(10, 10, 40, false), .. Square(30, 30, 40, false), .. Square(20, 50, 40, true)]);
        float At(OutlineFill rule, int x, int y)
        {
            var coverage = new float[100 * 100];
            rasterizer.Fill(outline, 1f, 0f, 0f, coverage, 100, 100, rule);
            return coverage[y * 100 + x];
        }

        Check(At(OutlineFill.NonZero, 40, 40) == 1f && At(OutlineFill.EvenOdd, 40, 40) == 0f && At(OutlineFill.NonZero, 15, 15) == 1f && At(OutlineFill.EvenOdd, 15, 15) == 1f
              && At(OutlineFill.NonZero, 25, 65) == 1f && At(OutlineFill.EvenOdd, 25, 65) == 1f && At(OutlineFill.NonZero, 35, 55) == 0f && At(OutlineFill.EvenOdd, 35, 55) == 0f
              && At(OutlineFill.NonZero, 90, 90) == 0f,
            $"the rules: nonzero {At(OutlineFill.NonZero, 40, 40)} {At(OutlineFill.NonZero, 35, 55)}, even-odd {At(OutlineFill.EvenOdd, 40, 40)}");
        var glyphCoverage = new float[100 * 100];
        var defaultCoverage = new float[100 * 100];
        rasterizer.Fill(outline, 1f, 0f, 0f, glyphCoverage, 100, 100, OutlineFill.Glyph);
        rasterizer.Fill(outline, 1f, 0f, 0f, defaultCoverage, 100, 100);
        Check(glyphCoverage.AsSpan().SequenceEqual(defaultCoverage), "the glyph rule is the fill as it was");
        // An outline far past a band: only the band's rows and columns are worked, as the whole target would give them.
        var big = new GlyphOutline([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close], [-1e5f, 1e5f, 1e5f, 1e5f, 1e5f, -1e5f, -1e5f, -1e5f]);
        var band = new float[64 * 8];
        rasterizer.Fill(big, 1f, 0f, -20f, band, 64, 8, OutlineFill.NonZero);
        Check(band.All(v => v == 1f), "a shape far larger than the band covers it whole");

        var samples = PdfSamples();
        var vector = samples.Single(s => s.Name == "vector-transparency");
        var text = samples.Single(s => s.Name == "truetype-simple");
        var scan = samples.Single(s => s.Name == "scan-ocr");
        string fonts = PdfDrawFonts();

        // Pages in parallel and one by one: the same pixels.
        var all = PdfVectorSampleList();
        var documents = all.Select(s => DocumentFormats.Open(File.ReadAllBytes(s.Path).AsMemory(), Path.GetFileName(s.Path))).ToList();
        try
        {
            var options = new DocumentDrawOptions { Dpi = 150, Fonts = fonts };
            long start = Stopwatch.GetTimestamp();
            var one = documents.Select(d => DocumentPageDrawers.DrawPages(d, [.. Enumerable.Range(0, d.PageCount)], options, parallel: false)).ToList();
            double sequential = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            start = Stopwatch.GetTimestamp();
            var many = documents.Select(d => DocumentPageDrawers.DrawPages(d, [.. Enumerable.Range(0, d.PageCount)], options)).ToList();
            double parallel = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            Check(one.Zip(many).All(p => p.First.Zip(p.Second).All(q => PdfInterleaved(q.First.Image).AsSpan().SequenceEqual(PdfInterleaved(q.Second.Image)))),
                "pages drawn in parallel are the pages drawn one by one");
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"    the {all.Count} D7 samples' pages at 150 dpi (warm): one page after another {sequential:F0} ms, pages in parallel {parallel:F0} ms ({Environment.ProcessorCount} cores)"));
        }
        finally
        {
            documents.ForEach(d => d.Dispose());
        }

        // Cold and warm apart: time and managed allocation a page.
        foreach (var (sample, kind) in new[] { (vector, "vector"), (text, "text"), (scan, "scan") })
        {
            foreach (float dpi in (float[])[100f, 300f])
            {
                using var document = DocumentFormats.Open(File.ReadAllBytes(sample.Path).AsMemory(), Path.GetFileName(sample.Path));
                var options = new DocumentDrawOptions { Dpi = dpi, Fonts = fonts };
                long allocated = GC.GetTotalAllocatedBytes(precise: true);
                long start = Stopwatch.GetTimestamp();
                var first = document.DrawPage(0, options);
                double cold = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                long coldBytes = GC.GetTotalAllocatedBytes(precise: true) - allocated;
                var warm = new List<double>();
                long warmBytes = 0;
                for (int pass = 0; pass < 5; pass++)
                {
                    allocated = GC.GetTotalAllocatedBytes(precise: true);
                    start = Stopwatch.GetTimestamp();
                    var again = document.DrawPage(0, options);
                    warm.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds);
                    warmBytes = GC.GetTotalAllocatedBytes(precise: true) - allocated;
                    Check(PdfInterleaved(again.Image).AsSpan().SequenceEqual(PdfInterleaved(first.Image)), $"{sample.Name} at {dpi} dpi drawn again: other pixels");
                }

                warm.Sort();
                long imageBytes = (long)first.Image.Width * first.Image.Height * first.Image.Channels;
                Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                    $"    a {kind} page ({sample.Name}, {first.Image.Width} x {first.Image.Height} at {dpi} dpi): cold {cold:F1} ms and {coldBytes / 1048576.0:F2} MiB, warm median {warm[2]:F1} ms and {warmBytes / 1048576.0:F2} MiB ({imageBytes / 1048576.0:F2} MiB of it the pixels)"));
            }
        }

        // A large page at a high resolution: the pixels themselves, and a band's buffers besides.
        using (var document = DocumentFormats.Open(File.ReadAllBytes(vector.Path).AsMemory(), "vector-transparency.pdf"))
        {
            var options = new DocumentDrawOptions { Dpi = 1200, Fonts = fonts };
            GC.Collect();
            long before = GC.GetTotalAllocatedBytes(precise: true);
            long heapBefore = GC.GetTotalMemory(forceFullCollection: true);
            long start = Stopwatch.GetTimestamp();
            var drawing = document.DrawPage(0, options);
            double ms = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            long allocated = GC.GetTotalAllocatedBytes(precise: true) - before;
            long pixels = (long)drawing.Image.Width * drawing.Image.Height * drawing.Image.Channels;
            var info = GC.GetGCMemoryInfo();
            long beyond = allocated - pixels;
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"    a large page (vector-transparency at 1200 dpi, {drawing.Image.Width} x {drawing.Image.Height}): {ms:F0} ms, {allocated / 1048576.0:F1} MiB allocated of which {pixels / 1048576.0:F1} MiB its pixels, {beyond / 1048576.0:F1} MiB besides; heap before {heapBefore / 1048576.0:F1} MiB, the GC's peak heap {info.HeapSizeBytes / 1048576.0:F1} MiB"));
            Check(beyond < 96L << 20, string.Create(CultureInfo.InvariantCulture, $"a large page allocated {beyond / 1048576.0:F1} MiB besides its pixels: not bounded by a band"));
            GC.KeepAlive(drawing);
        }
    }
}
