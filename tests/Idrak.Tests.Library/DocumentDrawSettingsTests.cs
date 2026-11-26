// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Imaging;

// Plan 28, phase D7's drawer settings: the page drawer's numbers (its bands, a shading's steps, the largest tile, a
// substitute face's stretch, slant and stroke, the curves' flatness) as DocumentDrawOptions settings with the values the
// drawer always had as defaults, their variables, their ranges, and each one's effect on the D7 samples' drawings
// (tools/documents/make_fixtures.py writes them into the ignored test-output/documents/samples).
internal abstract partial class Tests
{
    // A page drawn by the library's drawer, reached directly (DocumentPageDrawers drew with the drawer the options named).
    internal static DocumentDrawing CoverageDrawn(DocumentPage page, DocumentDrawOptions? options = null) => new CoverageDocumentDrawer().Draw(page, options ?? new DocumentDrawOptions());

    private static void PdfDrawSettings(Device device)
    {
        CpuOnly(device);
        string[] variables =
        [
            DocumentDrawOptions.BandPixelsVariable, DocumentDrawOptions.ShadingStepsVariable, DocumentDrawOptions.MaxTilePixelsVariable,
            DocumentDrawOptions.FillFlatnessVariable, DocumentDrawOptions.StrokeFlatnessVariable,
        ];
        Check(variables.All(v => v.StartsWith("IDRAK_DOCUMENT_", StringComparison.Ordinal)) && variables.Distinct().Count() == 5, "the five variables");

        // Unset: the numbers the drawer always had, and the band rows it always took.
        WithEnvironment(variables.ToDictionary(v => v, _ => (string?)null), () =>
        {
            var d = new DocumentDrawOptions();
            Check(d.BandPixels == 1 << 17 && d.MinBandRows == 8 && d.MaxBandRows == 1024 && d.ShadingSteps == 1024 && d.MaxTilePixels == 1 << 20
                  && d.MinSubstituteStretch == 0.75f && d.MaxSubstituteStretch == 4f / 3f && d.SubstituteSlant == 0.21f && d.SubstituteStroke == 0.03f
                  && d.FillFlatness == 1f / 32 && d.StrokeFlatness == 0.1f, $"the defaults: {d}");
            for (int width = 1; width <= 40_000; width += width < 300 ? 1 : 37)
            {
                int rows = CoverageDocumentDrawer.BandRows(width, d.BandPixels, d.MinBandRows, d.MaxBandRows), always = Math.Clamp((1 << 17) / width, 8, 1024);
                Check(rows == always, $"a page {width} pixels wide: {rows} rows a band, where the drawer always took {always}");
            }
        });
        Check(CoverageDocumentDrawer.BandRows(1000, 4096, 1, 1024) == 4 && CoverageDocumentDrawer.BandRows(1000, 1, 1, 1024) == 1 && CoverageDocumentDrawer.BandRows(10, 1 << 26, 8, 4096) == 4096,
            "a band's rows follow its pixels within its bounds");

        // Set: each variable read; a value that cannot be read refused, naming its variable and the value.
        WithEnvironment(new Dictionary<string, string?>
        {
            [DocumentDrawOptions.BandPixelsVariable] = "4096", [DocumentDrawOptions.ShadingStepsVariable] = "256", [DocumentDrawOptions.MaxTilePixelsVariable] = "65536",
            [DocumentDrawOptions.FillFlatnessVariable] = "0.25", [DocumentDrawOptions.StrokeFlatnessVariable] = "0.5",
        }, () =>
        {
            var d = new DocumentDrawOptions();
            Check(d.BandPixels == 4096 && d.ShadingSteps == 256 && d.MaxTilePixels == 65536 && d.FillFlatness == 0.25f && d.StrokeFlatness == 0.5f, $"read: {d}");
        });
        foreach (var (variable, value) in new[]
                 {
                     (DocumentDrawOptions.BandPixelsVariable, "0"), (DocumentDrawOptions.BandPixelsVariable, "128k"), (DocumentDrawOptions.ShadingStepsVariable, "1"),
                     (DocumentDrawOptions.MaxTilePixelsVariable, "-4"), (DocumentDrawOptions.FillFlatnessVariable, "0"), (DocumentDrawOptions.StrokeFlatnessVariable, "fine"),
                 })
        {
            WithEnvironment(new Dictionary<string, string?> { [variable] = value }, () =>
            {
                try
                {
                    _ = new DocumentDrawOptions();
                    Check(false, $"{variable}={value} was taken");
                }
                catch (ArgumentException e)
                {
                    Check(e.Message.Contains(variable, StringComparison.Ordinal) && e.Message.Contains(value, StringComparison.Ordinal), e.Message);
                }
            });
        }

        // Out of range in code: refused, naming the setting and its range; the rows' bounds crossed: drawing refuses.
        foreach (var (setting, make) in new (string, Func<DocumentDrawOptions>)[]
                 {
                     (nameof(DocumentDrawOptions.BandPixels), () => new DocumentDrawOptions { BandPixels = 0 }),
                     (nameof(DocumentDrawOptions.BandPixels), () => new DocumentDrawOptions { BandPixels = (1 << 26) + 1 }),
                     (nameof(DocumentDrawOptions.MinBandRows), () => new DocumentDrawOptions { MinBandRows = 0 }),
                     (nameof(DocumentDrawOptions.MaxBandRows), () => new DocumentDrawOptions { MaxBandRows = (1 << 16) + 1 }),
                     (nameof(DocumentDrawOptions.ShadingSteps), () => new DocumentDrawOptions { ShadingSteps = 1 }),
                     (nameof(DocumentDrawOptions.MaxTilePixels), () => new DocumentDrawOptions { MaxTilePixels = 0 }),
                     (nameof(DocumentDrawOptions.MinSubstituteStretch), () => new DocumentDrawOptions { MinSubstituteStretch = 0f }),
                     (nameof(DocumentDrawOptions.MinSubstituteStretch), () => new DocumentDrawOptions { MinSubstituteStretch = 1.5f }),
                     (nameof(DocumentDrawOptions.MaxSubstituteStretch), () => new DocumentDrawOptions { MaxSubstituteStretch = 0.9f }),
                     (nameof(DocumentDrawOptions.MaxSubstituteStretch), () => new DocumentDrawOptions { MaxSubstituteStretch = float.NaN }),
                     (nameof(DocumentDrawOptions.SubstituteSlant), () => new DocumentDrawOptions { SubstituteSlant = -0.1f }),
                     (nameof(DocumentDrawOptions.SubstituteStroke), () => new DocumentDrawOptions { SubstituteStroke = 0.3f }),
                     (nameof(DocumentDrawOptions.FillFlatness), () => new DocumentDrawOptions { FillFlatness = 0f }),
                     (nameof(DocumentDrawOptions.FillFlatness), () => new DocumentDrawOptions { FillFlatness = float.PositiveInfinity }),
                     (nameof(DocumentDrawOptions.StrokeFlatness), () => new DocumentDrawOptions { StrokeFlatness = 1e-4f }),
                     (nameof(DocumentDrawOptions.StrokeFlatness), () => new DocumentDrawOptions { StrokeFlatness = float.NaN }),
                 })
        {
            try
            {
                make();
                Check(false, $"a {setting} out of range was taken");
            }
            catch (ArgumentOutOfRangeException e)
            {
                Check(e.ParamName == setting && e.Message.Contains($"DocumentDrawOptions.{setting} is ", StringComparison.Ordinal)
                      && (e.Message.Contains(" to ", StringComparison.Ordinal) || e.Message.Contains("at most", StringComparison.Ordinal)), $"{setting}: {e.Message}");
            }
        }

        var small = new DocumentPage(0, 40f, 30f) { Paths = [DocumentPath.Rectangle(new DocumentBox(5f, 5f, 20f, 10f))] };
        try
        {
            CoverageDrawn(small, new DocumentDrawOptions { MinBandRows = 64, MaxBandRows = 16 });
            Check(false, "a band of at least 64 and at most 16 rows was drawn");
        }
        catch (ArgumentOutOfRangeException e)
        {
            Check(e.Message.Contains("MinBandRows (64)", StringComparison.Ordinal) && e.Message.Contains("MaxBandRows (16)", StringComparison.Ordinal), e.Message);
        }

        PdfDrawSettingsEffects();
        PdfDrawSettingsRasterizer();
    }

    // Each setting's coarser value changes the drawing of a sample that has what it governs; a finer value stays within the
    // page's tolerance against pdfium's render (the 3 x 3 mean, the share of pixels off by more than 32 levels) and its probes.
    private static void PdfDrawSettingsEffects()
    {
        string fonts = PdfDrawFonts();
        var png = new PngCodec();
        var failures = new List<string>();
        var cases = new (string Sample, string Change, Func<DocumentDrawOptions, DocumentDrawOptions> Apply, bool Within)[]
        {
            ("vector-shadings", "2 shading steps", o => o with { ShadingSteps = 2 }, false),
            ("vector-shadings", "256 shading steps", o => o with { ShadingSteps = 256 }, true),
            ("vector-patterns", "a largest tile of 1 pixel", o => o with { MaxTilePixels = 1 }, false),
            ("vector-patterns", "a largest tile of 400 pixels", o => o with { MaxTilePixels = 400 }, false),
            ("vector-paths", "fills flattened within 4 pixels", o => o with { FillFlatness = 4f }, false),
            ("vector-paths", "fills flattened within 1/128 of a pixel", o => o with { FillFlatness = 1f / 128 }, true),
            ("vector-strokes", "strokes flattened within 4 pixels", o => o with { StrokeFlatness = 4f }, false),
            ("vector-strokes", "strokes flattened within 0.02 pixels", o => o with { StrokeFlatness = 0.02f }, true),
            ("standard14", "substitutes not slanted", o => o with { SubstituteSlant = 0f }, false),
            ("standard14", "substitutes not stroked", o => o with { SubstituteStroke = 0f }, false),
            ("standard14", "substitutes not stretched", o => o with { MinSubstituteStretch = 1f, MaxSubstituteStretch = 1f }, false),
        };
        var samples = PdfSamples();
        foreach (var group in cases.GroupBy(c => c.Sample))
        {
            var sample = samples.Single(s => s.Name == group.Key);
            double dpi = (double?)sample.Expected["reference"]?["dpi"] ?? 100;
            var expected = png.Decode(File.ReadAllBytes(Path.Combine(Path.GetDirectoryName(sample.Path)!, $"{sample.Name}.p1.png")));
            using var document = PdfOpened(sample.Path);
            var options = new DocumentDrawOptions { Dpi = (float)dpi, Fonts = fonts };
            byte[] standard = PdfInterleaved(document.DrawPage(0, options).Image);
            double tolerance = PdfDrawTolerance(sample, 0);
            foreach (var (_, change, apply, within) in group)
            {
                var drawing = document.DrawPage(0, apply(options));
                byte[] pixels = PdfInterleaved(drawing.Image);
                int changed = pixels.Length == standard.Length ? pixels.Zip(standard).Count(p => p.First != p.Second) : pixels.Length;
                var (share, mean) = PdfDrawDifference(drawing.Image, expected, 32);
                Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                    $"    {sample.Name}, {change}: {changed} values of {pixels.Length} changed; against pdfium {share * 100:F2}% off by more than 32 levels (mean {mean:F2}), the page held to {tolerance * 100:F0}%"));
                // A coarser value changes the drawing; a finer one may draw the same bytes (256 steps read a smooth
                // shading as closely as 1,024 at 8 bits a channel), and stays within the page's tolerance.
                if (changed == 0 && !within)
                {
                    failures.Add($"{sample.Name}, {change}: the drawing did not change");
                }

                if (within)
                {
                    if (share > tolerance)
                    {
                        failures.Add(string.Create(CultureInfo.InvariantCulture, $"{sample.Name}, {change}: {share * 100:F2}% off pdfium's render, past {tolerance * 100:F0}%"));
                    }

                    PdfDrawProbes($"{sample.Name}, {change}", drawing.Image, sample.Expected["pages"]![0]!.AsObject(), dpi, failures);
                }
            }
        }

        Check(failures.Count == 0, $"{failures.Count} settings without their effect:\n      {string.Join("\n      ", failures)}");
    }

    // The rasterizer's flatness: the default is the fill as it was, a coarser one flattens a circle into fewer lines, a
    // flatness that is not more than 0 is refused.
    private static void PdfDrawSettingsRasterizer()
    {
        var rasterizer = new CoverageRasterizer();
        var circle = DrawSettingsCircle(50f, 50f, 30f);
        float[] Fill(OutlineFill rule, float? flatness)
        {
            var coverage = new float[100 * 100];
            if (flatness is { } f)
            {
                rasterizer.Fill(circle, 1f, 0f, 0f, coverage, 100, 100, rule, f);
            }
            else
            {
                rasterizer.Fill(circle, 1f, 0f, 0f, coverage, 100, 100, rule);
            }

            return coverage;
        }

        foreach (var rule in (OutlineFill[])[OutlineFill.Glyph, OutlineFill.NonZero, OutlineFill.EvenOdd])
        {
            Check(Fill(rule, 1f / 32).AsSpan().SequenceEqual(Fill(rule, null)), $"{rule}: the default flatness is the fill as it was");
            double fine = Fill(rule, null).Sum(), coarse = Fill(rule, 4f).Sum();
            Check(coarse < fine - 1, string.Create(CultureInfo.InvariantCulture, $"{rule}: a circle flattened within 4 pixels covers less ({coarse:F1}) than within 1/32 ({fine:F1})"));
        }

        foreach (float bad in (float[])[0f, -1f, float.NaN, float.PositiveInfinity])
        {
            try
            {
                rasterizer.Fill(circle, 1f, 0f, 0f, new float[100 * 100], 100, 100, OutlineFill.NonZero, bad);
                Check(false, $"a flatness of {bad} was taken");
            }
            catch (ArgumentOutOfRangeException e)
            {
                Check(e.ParamName == "flatness", e.Message);
            }
        }

    }

    // Bands never change the drawing: the rule fill of an outline into bands of a target (y moved by whole rows, with a
    // fraction too, one a float holds exactly at every row) is the whole target's fill row for row; every D3, D4 and D7 sample page drawn at its reference's dpi
    // with one-row bands, seven-row bands and the whole page as one band is the default bands' drawing, byte for byte.
    private static void PdfDrawBands(Device device)
    {
        CpuOnly(device);
        var rasterizer = new CoverageRasterizer();
        float[] square = [10f, -10f, 70f, -10f, 70f, -70f, 10f, -70f];
        var outlines = new (string Name, GlyphOutline Outline)[]
        {
            ("a circle", DrawSettingsCircle(50f, 50f, 30f)),
            ("a circle reaching past the target", DrawSettingsCircle(60f, 30f, 75f)),
            ("a sliver from far above", new GlyphOutline([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close], [3f, 5000f, 97f, -90f, 40f, -95f])),
            ("squares wound both ways", new GlyphOutline(
                [OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close, OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close],
                [.. square, 30f, -30f, 30f, -90f, 90f, -90f, 90f, -30f])),
        };
        foreach (var (name, outline) in outlines)
        {
            foreach (var rule in (OutlineFill[])[OutlineFill.NonZero, OutlineFill.EvenOdd])
            {
                foreach (float fraction in (float[])[0f, 0.375f])
                {
                    var whole = new float[100 * 100];
                    rasterizer.Fill(outline, 1f, 0f, fraction, whole, 100, 100, rule, 1f / 32);
                    foreach (int rows in (int[])[1, 7, 64])
                    {
                        var band = new float[100 * rows];
                        for (int top = 0; top < 100; top += rows)
                        {
                            int count = Math.Min(rows, 100 - top);
                            Array.Clear(band);
                            rasterizer.Fill(outline, 1f, 0f, fraction - top, band.AsSpan(0, 100 * count), 100, count, rule, 1f / 32);
                            Check(band.AsSpan(0, 100 * count).SequenceEqual(whole.AsSpan(top * 100, 100 * count)),
                                string.Create(CultureInfo.InvariantCulture, $"{name}, {rule}, y {fraction}: the band of {rows} rows at row {top} is not those rows of the whole fill"));
                        }
                    }
                }
            }
        }

        string fonts = PdfDrawFonts();
        int pages = 0;
        foreach (var sample in PdfSamples().Where(s => s.Readable && (s.Phases.Contains("D3") || s.Phases.Contains("D4") || s.Phases.Contains("D7"))))
        {
            double dpi = (double?)sample.Expected["reference"]?["dpi"] ?? 100;
            using var document = PdfOpened(sample.Path);
            for (int index = 0; index < document.PageCount; index++)
            {
                var options = new DocumentDrawOptions { Dpi = (float)dpi, Fonts = fonts };
                var standard = document.DrawPage(index, options);
                byte[] expected = PdfInterleaved(standard.Image);
                string issues = string.Join("; ", standard.Issues.Select(i => $"{i.Kind} {i.Count}"));
                foreach (var (bands, banded) in new (string, DocumentDrawOptions)[]
                         {
                             ("one-row bands", options with { BandPixels = 1, MinBandRows = 1 }),
                             ("seven-row bands", options with { BandPixels = 1, MinBandRows = 7, MaxBandRows = 7 }),
                             ("one band", options with { BandPixels = 1 << 26, MaxBandRows = 1 << 16 }),
                         })
                {
                    var drawing = document.DrawPage(index, banded);
                    byte[] pixels = PdfInterleaved(drawing.Image);
                    int differ = pixels.Length == expected.Length ? pixels.Zip(expected).Count(p => p.First != p.Second) : -1;
                    Check(differ == 0 && string.Join("; ", drawing.Issues.Select(i => $"{i.Kind} {i.Count}")) == issues,
                        $"{sample.Name}: page {index + 1} drawn with {bands}: {differ} values differ from the default bands' drawing (issues '{issues}')");
                }

                pages++;
            }
        }

        Check(pages >= 50, $"{pages} sample pages drawn; the D3, D4 and D7 samples hold more (make them again: tools/documents/make_fixtures.py)");
    }

    // A circle of four cubics about (cx, cy) in y-up units, as glyph outlines are held.
    private static GlyphOutline DrawSettingsCircle(float cx, float cy, float r)
    {
        const float k = 0.5522848f;
        float c = k * r;
        return new GlyphOutline(
            [OutlineVerb.Move, OutlineVerb.Cubic, OutlineVerb.Cubic, OutlineVerb.Cubic, OutlineVerb.Cubic, OutlineVerb.Close],
            [cx + r, -cy, cx + r, -(cy + c), cx + c, -(cy + r), cx, -(cy + r), cx - c, -(cy + r), cx - r, -(cy + c), cx - r, -cy,
                cx - r, -(cy - c), cx - c, -(cy - r), cx, -(cy - r), cx + c, -(cy - r), cx + r, -(cy - c), cx + r, -cy]);
    }
}
