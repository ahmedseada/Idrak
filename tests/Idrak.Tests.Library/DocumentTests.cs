// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using Idrak.Abstraction.Testing;
using Idrak.PluginTests;

// Plan 28, phase D0: the document contract (formats, layouts, writers, page drawers, font writers), its registries and
// guards, the options and their variables, and the conformance kit's document suites, checked on the outside toy format
// and writer (tests/Idrak.PluginTests) and on a drawer and a font writer of the test's own.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] DocumentContractGroup =
    [
        ("documents contracts: a name nobody registered is refused in every registry (formats, writers, layouts, page drawers, font writers), naming the registry, the registered ones and what to do; the library's layouts lines and content are registered", DocumentsRefuseUnknownNames),
        ("documents contracts: the options: the size limits measured from the free memory unless set, every variable read (IDRAK_DOCUMENT_*, IDRAK_FONT_CURVE_TOLERANCE), a bad value refused naming its variable, passwords never printed", DocumentsOptions),
        ("documents contracts: colours (grey, RGB, CMYK and their RGB, opacity), matrices, boxes, paths (points checked against their steps), page labels (roman, letters, prefixes), a check's validity", DocumentsValues),
        ("documents contracts: the format and writer suites pass on the outside toy and fail a format that puts a run outside its page, one that reads a page differently each time, one that fails a damaged file with another exception, and a writer that writes other bytes each time", DocumentsSuitesCatch),
        ("documents contracts: the page drawer and font writer suites pass on a drawer and a writer of the test's own and fail a drawer that throws for a missing font and a writer that drops outlines", DocumentsDrawerAndFontSuites),
        ("documents contracts: an app's page drawer and font writer over library ones: the failure reaches the caller by default, falls back per call under FallBack (the font writer's destination holding the library's bytes alone), and is compared under Shadow", DocumentsDrawerAndFontGuards),
    ];

    internal static readonly (string Name, Action<Device> Run)[] DocumentLayoutGroup =
    [
        ("document layouts: lines: runs grouped by baseline and size into lines from the top, ordered by place whatever their content order, a space where a gap passes the word gap (an option), none inside a word, a superscript kept on its line", LayoutsLinesLeftToRight),
        ("document layouts: lines: a right-to-left line of runs in visual order comes out in logical order, a number and a left-to-right word inside it and a right-to-left word in a left-to-right line through bidi; runs in logical order give the same text", LayoutsLinesBothDirections),
        ("document layouts: lines: each line's direction from its first strong character, a line with none taking the page's; the direction option sets every line; runs without a place are lines of their own, after the placed ones", LayoutsLinesDirection),
        ("document layouts: content: runs in content order, joined; the layout and the reading order named by the options and their variables; an unregistered layout or reading order refused naming its option", LayoutsContentAndNames),
        ("document layouts: an app's layout over lines answers, its failure reaches the caller, falls back per call under FallBack, is compared under Shadow; unregistered, the library's lines is back", LayoutsOverride),
    ];

    private static void DocumentsRefuseUnknownNames(Device device)
    {
        CpuOnly(device);
        const string unknown = "a-name-nobody-registered";
        Check(DocumentTextLayouts.Names.Contains(DocumentTextLayouts.Lines) && DocumentTextLayouts.Names.Contains(DocumentTextLayouts.Content)
              && DocumentTextLayouts.Origin(DocumentTextLayouts.Lines) == Overrides.Library && DocumentTextLayouts.Origin(DocumentTextLayouts.Content) == Overrides.Library
              && DocumentTextLayouts.Describe().Contains("lines: ", StringComparison.Ordinal), $"the layouts: {DocumentTextLayouts.Describe()}");
        (string Registry, Action Get, IReadOnlyCollection<string> Names, string Remedy)[] registries =
        [
            (nameof(DocumentFormats), () => DocumentFormats.Get(unknown), DocumentFormats.Names, "DocumentFormats.Register"),
            (nameof(DocumentWriters), () => DocumentWriters.Get(unknown), DocumentWriters.Names, "DocumentWriters.Register"),
            (nameof(DocumentTextLayouts), () => DocumentTextLayouts.Get(unknown), DocumentTextLayouts.Names, "DocumentTextLayouts.Register"),
            (nameof(DocumentPageDrawers), () => DocumentPageDrawers.Get(unknown), DocumentPageDrawers.Names, "DocumentPageDrawers.Register"),
            (nameof(FontWriters), () => FontWriters.Get(unknown), FontWriters.Names, "FontWriters.Register"),
        ];
        foreach (var (registry, get, names, remedy) in registries)
        {
            try
            {
                get();
                Check(false, $"{registry}: '{unknown}' was found");
            }
            catch (NotRegisteredException e)
            {
                Check(e.Registry == registry && e.Name == unknown && e.Registered.SequenceEqual(names) && e.Message.Contains(remedy, StringComparison.Ordinal)
                      && e.Message.Contains(unknown, StringComparison.Ordinal) && names.All(n => e.Message.Contains(n, StringComparison.Ordinal)),
                    $"{registry}: {e.Message} (registered: {string.Join(", ", names)})");
            }
        }

        try
        {
            DocumentFormats.Get(unknown);
        }
        catch (NotRegisteredException e)
        {
            Check(e.Message.Contains("Office", StringComparison.Ordinal), $"the formats' refusal names the plug-in to come: {e.Message}");
        }

        try
        {
            DocumentWriters.For("report.unknown-extension");
            Check(false, "a writer was found for an extension nobody writes");
        }
        catch (NotSupportedException e)
        {
            Check(e.Message.Contains(".unknown-extension", StringComparison.Ordinal), e.Message);
        }

        Check(!DocumentFormats.CanOpen("letter.unknown-extension") && DocumentFormats.Detect("not a document"u8) is null, "nothing opens what no format reads");
    }

    private static void DocumentsOptions(Device device)
    {
        CpuOnly(device);
        string[] variables =
        [
            DocumentOptions.LayoutVariable, DocumentOptions.ReadingOrderVariable, DocumentOptions.MaxStreamBytesVariable, DocumentOptions.CacheBytesVariable,
            DocumentWriteOptions.ImageEncodingVariable, DocumentWriteOptions.JpegQualityVariable, DocumentWriteOptions.EmbedFontsVariable, DocumentWriteOptions.SaveModeVariable,
            DocumentDrawOptions.DpiVariable, DocumentDrawOptions.FontsVariable, SignatureCheckOptions.OnlineVariable, FontWriteOptions.CurveToleranceVariable,
        ];
        Check(variables.All(v => v.StartsWith("IDRAK_DOCUMENT_", StringComparison.Ordinal) || v == "IDRAK_FONT_CURVE_TOLERANCE") && variables.Distinct().Count() == 12, "the twelve variables");

        // Unset: the defaults, the size limits measured from the free memory.
        WithEnvironment(variables.ToDictionary(v => v, _ => (string?)null), () =>
        {
            var options = new DocumentOptions();
            long total = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
            Check(options.Layout == DocumentTextLayouts.Lines && options.ReadingOrder == ReadingOrders.Bidi && options.IncludeHidden && options.Direction == DocumentTextDirection.Auto
                  && options.Password is null && options.MaxDepth == 64 && options.WordGap == 0.2f && options.LineTolerance == 0.5f, $"the defaults: {options}");
            Check(options.MaxStreamBytes > 0 && options.MaxStreamBytes <= Array.MaxLength && options.MaxStreamBytes <= Math.Max(1L << 20, total)
                  && options.CacheBytes > 0 && options.CacheBytes <= Math.Max(1L << 20, total) && options.MaxObjects > 0,
                $"measured limits: streams {options.MaxStreamBytes:N0}, cache {options.CacheBytes:N0}, objects {options.MaxObjects:N0} of {total:N0} bytes");
            Check(options.ToString().Contains("MaxStreamBytes = measured", StringComparison.Ordinal), options.ToString());
            var write = new DocumentWriteOptions();
            Check(write.ImageEncoding == DocumentImageEncoding.Auto && write.JpegQuality == 90 && write.EmbedFonts == DocumentFontEmbedding.Subset && write.SaveMode == DocumentSaveMode.Incremental
                  && write.Deterministic && write.ObjectStreams && write.Compression == 6 && write.Id.IsEmpty, $"the writer's defaults: {write}");
            var draw = new DocumentDrawOptions();
            Check(MathF.Abs(draw.Dpi - 150f) < 1e-3f && MathF.Abs(draw.PixelsPerPoint - 150f / 72f) < 1e-6f && draw.Fonts is null && draw.Colour == DocumentDrawColour.Rgb
                  && draw.Parts == (DocumentParts.Text | DocumentParts.Images | DocumentParts.Paths) && !draw.Hidden && draw.Drawer == DocumentPageDrawers.Coverage, $"the drawing defaults: {draw}");
            Check(!new SignatureCheckOptions().Online && new FontWriteOptions().CurveTolerance == 1f && new FontWriteOptions().KeepLayout, "a check is offline, curves within one unit");
        });

        // Set: each variable read.
        string fonts = Path.Combine(Path.GetTempPath(), "fonts for documents");
        WithEnvironment(new Dictionary<string, string?>
        {
            [DocumentOptions.LayoutVariable] = "content", [DocumentOptions.ReadingOrderVariable] = "runs", [DocumentOptions.MaxStreamBytesVariable] = "256m",
            [DocumentOptions.CacheBytesVariable] = "1GiB", [DocumentWriteOptions.ImageEncodingVariable] = "Jpeg", [DocumentWriteOptions.JpegQualityVariable] = "75",
            [DocumentWriteOptions.EmbedFontsVariable] = "whole", [DocumentWriteOptions.SaveModeVariable] = "rewrite", [DocumentDrawOptions.DpiVariable] = "300",
            [DocumentDrawOptions.FontsVariable] = fonts, [SignatureCheckOptions.OnlineVariable] = "on", [FontWriteOptions.CurveToleranceVariable] = "0.25",
        }, () =>
        {
            var options = new DocumentOptions();
            Check(options.Layout == "content" && options.ReadingOrder == "runs" && options.MaxStreamBytes == 256L << 20 && options.CacheBytes == 1L << 30, $"read: {options}");
            var write = new DocumentWriteOptions();
            Check(write.ImageEncoding == DocumentImageEncoding.Jpeg && write.JpegQuality == 75 && write.EmbedFonts == DocumentFontEmbedding.Whole && write.SaveMode == DocumentSaveMode.Rewrite,
                $"read: {write}");
            var draw = new DocumentDrawOptions();
            Check(MathF.Abs(draw.Dpi - 300f) < 1e-3f && draw.Fonts == fonts && new SignatureCheckOptions().Online && new FontWriteOptions().CurveTolerance == 0.25f, $"read: {draw}");
        });

        // A value that cannot be read is refused, naming its variable and what it takes.
        foreach (var (variable, value, make) in new (string, string, Func<object>)[]
                 {
                     (DocumentOptions.MaxStreamBytesVariable, "lots", () => new DocumentOptions()),
                     (DocumentOptions.CacheBytesVariable, "-5", () => new DocumentOptions()),
                     (DocumentWriteOptions.JpegQualityVariable, "200", () => new DocumentWriteOptions()),
                     (DocumentWriteOptions.ImageEncodingVariable, "gif", () => new DocumentWriteOptions()),
                     (DocumentWriteOptions.SaveModeVariable, "1", () => new DocumentWriteOptions()),
                     (DocumentDrawOptions.DpiVariable, "0", () => new DocumentDrawOptions()),
                     (SignatureCheckOptions.OnlineVariable, "maybe", () => new SignatureCheckOptions()),
                     (FontWriteOptions.CurveToleranceVariable, "none", () => new FontWriteOptions()),
                 })
        {
            WithEnvironment(new Dictionary<string, string?> { [variable] = value }, () =>
            {
                try
                {
                    make();
                    Check(false, $"{variable}={value} was taken");
                }
                catch (ArgumentException e)
                {
                    Check(e.Message.Contains(variable, StringComparison.Ordinal) && e.Message.Contains(value, StringComparison.Ordinal), e.Message);
                }
            });
        }

        // Set in code; refused when out of range; passwords never printed.
        var set = new DocumentOptions { MaxStreamBytes = 1024, MaxObjects = 10, MaxDepth = 3, CacheBytes = 0, Password = "a secret word" };
        Check(set.MaxStreamBytes == 1024 && set.MaxObjects == 10 && set.MaxDepth == 3 && !set.ToString().Contains("a secret word", StringComparison.Ordinal)
              && set.ToString().Contains("Password = (set)", StringComparison.Ordinal), set.ToString());
        Check(!new DocumentEncryption { UserPassword = "a secret word", OwnerPassword = "another" }.ToString().Contains("secret", StringComparison.Ordinal), "the encryption's passwords");
        foreach (var make in new Func<object>[]
                 {
                     () => new DocumentOptions { WordGap = -1f }, () => new DocumentOptions { MaxStreamBytes = 0 }, () => new DocumentOptions { MaxDepth = 0 },
                     () => new DocumentWriteOptions { Compression = 10 }, () => new DocumentWriteOptions { JpegQuality = 0 }, () => new DocumentDrawOptions { Dpi = 0f },
                     () => new FontWriteOptions { CurveTolerance = 0f },
                 })
        {
            try
            {
                make();
                Check(false, "a value out of range was taken");
            }
            catch (ArgumentOutOfRangeException)
            {
                // Refused, as it should be.
            }
        }

        Check(new DocumentDrawOptions { Dpi = 72f }.PixelsPerPoint == 1f && new DocumentDrawOptions { PixelsPerPoint = 2f }.Dpi == 144f, "dots per inch and pixels per point are one setting");
    }

    private static void DocumentsValues(Device device)
    {
        CpuOnly(device);
        var cyan = DocumentColour.Cmyk(1f, 0f, 0f, 0f);
        Check(cyan.Red == 0f && cyan.Green == 1f && cyan.Blue == 1f && cyan.Count == 4 && cyan[0] == 1f && DocumentColour.Cmyk(0f, 0f, 0f, 1f).Luma == 0f,
            $"CMYK to RGB: {cyan} is {cyan.Red} {cyan.Green} {cyan.Blue}");
        Check(default(DocumentColour) == DocumentColour.Black && DocumentColour.Black.Alpha == 1f && DocumentColour.White.Luma == 1f && DocumentColour.Grey(0.5f).Luma == 0.5f
              && DocumentColour.Rgb(1f, 0f, 0f).WithAlpha(0.5f).Alpha == 0.5f && DocumentColour.Rgb(1f, 0f, 0f).WithAlpha(0.5f).ToString() == "rgb 1 0 0, alpha 0.5",
            "black by default, opaque, grey, alpha");
        foreach (var bad in new Action[] { () => DocumentColour.Rgb(2f, 0f, 0f), () => DocumentColour.Grey(float.NaN), () => _ = DocumentColour.Rgb(0f, 0f, 0f)[3] })
        {
            try
            {
                bad();
                Check(false, "a colour out of range was taken");
            }
            catch (ArgumentOutOfRangeException)
            {
                // Refused.
            }
        }

        var box = new DocumentBox(10f, 20f, 30f, 40f);
        Check(DocumentMatrix.FromBox(box).Transform(1f, 1f) == (40f, 60f) && DocumentMatrix.FromBox(box).Bounds() == box && DocumentMatrix.Identity.Transform(3f, 4f) == (3f, 4f),
            "a box's matrix maps the unit square onto it");
        var turned = new DocumentMatrix(0f, 1f, -1f, 0f, 0f, 0f).Then(new DocumentMatrix(1f, 0f, 0f, 1f, 5f, 7f));
        Check(turned.Transform(1f, 0f) == (5f, 8f) && turned.Transform(0f, 1f) == (4f, 7f), $"turned, then moved: {turned}");
        Check(default(DocumentBox).IsEmpty && DocumentBox.Empty.IsEmpty && !new DocumentBox(0f, 0f, 0f, 5f).IsEmpty && box.Union(DocumentBox.Empty) == box
              && box.Union(new DocumentBox(0f, 0f, 1f, 1f)) == new DocumentBox(0f, 0f, 40f, 60f) && box.Within(new DocumentBox(0f, 0f, 40f, 60f)) && !box.Within(new DocumentBox(0f, 0f, 39f, 60f))
              && box.Within(new DocumentBox(0f, 0f, 39f, 60f), tolerance: 1f), "boxes");

        Check(DocumentPath.Rectangle(box).Bounds() == box && DocumentPath.Rectangle(box).Verbs.Length == 5 && DocumentPath.Rectangle(box).Fill == DocumentFill.NonZero
              && DocumentPath.Rectangle(box).Clip == -1 && DocumentPath.Rectangle(box).Stroke is null, "a rectangle");
        foreach (var bad in new Action[] { () => _ = new DocumentPath([OutlineVerb.Move, OutlineVerb.Line], [1f, 2f]), () => _ = new DocumentPath([OutlineVerb.Move], [1f, float.NaN]) })
        {
            try
            {
                bad();
                Check(false, "a path whose points do not fit was taken");
            }
            catch (ArgumentException)
            {
                // Refused.
            }
        }

        DocumentPageLabel[] labels =
        [
            new(0, DocumentPageNumbering.LowerRoman),
            new(4, DocumentPageNumbering.Decimal) { Prefix = "A-" },
            new(10, DocumentPageNumbering.UpperLetters) { FirstNumber = 27 },
            new(12, DocumentPageNumbering.None) { Prefix = "Cover" },
        ];
        string[] expected = ["i", "ii", "iii", "iv", "A-1", "A-2", "A-3", "A-4", "A-5", "A-6", "AA", "BB", "Cover"];
        for (int i = 0; i < expected.Length; i++)
        {
            Check(DocumentPageLabel.Find(labels, i) == expected[i], $"page {i}: {DocumentPageLabel.Find(labels, i)}, expected {expected[i]}");
        }

        Check(DocumentPageLabel.Find([new DocumentPageLabel(3, DocumentPageNumbering.Decimal)], 1) is null && new DocumentPageLabel(0, DocumentPageNumbering.UpperRoman).Label(1993) == "MCMXCIV",
            "no range, and a long roman number");
        Check(!new SignatureCheck([]).Valid && new SignatureCheck([new(SignatureCheck.Digest, true, ""), new(SignatureCheck.Chain, true, "")]).Valid
              && !new SignatureCheck([new(SignatureCheck.Digest, true, ""), new(SignatureCheck.Chain, false, "not trusted")]).Valid, "a check is valid when every step passed");
        var page = new DocumentPage(2, 612f, 792f);
        Check(page.Box == new DocumentBox(0f, 0f, 612f, 792f) && page.Runs.Count == 0 && page.Text.Length == 0 && page.Issues.Count == 0, "an empty page");
        var run = new DocumentTextRun("text", box, 12f) { Mode = DocumentTextMode.Hidden };
        Check(run.Hidden && !run.Logical && run.Clip == -1 && run.Glyphs.IsEmpty && run.Face is null, "a hidden run");
    }

    private static void DocumentsSuitesCatch(Device device)
    {
        CpuOnly(device);
        string folder = Path.Combine(Path.GetTempPath(), "idrak-documents-contracts", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(folder);
        try
        {
            var face = SubsetFace();
            var samples = ToySamples(folder, face);
            var format = new ToyDocumentFormat();
            var writer = new ToyDocumentWriter();
            Conformance.CheckDocumentFormat(format, samples, options: new ContractCheckOptions { RandomCases = 20 }).ThrowIfFailed();
            Conformance.CheckDocumentWriter(writer, format, face).ThrowIfFailed();

            // Formats that break the contract are caught by the property they break.
            void Fails(IDocumentFormat broken, string property)
            {
                var report = Conformance.CheckDocumentFormat(broken, samples, options: new ContractCheckOptions { RandomCases = 20 });
                Check(!report.Passed && report.Failures.Any(f => f.Check == property), $"{property} is not caught:\n{report}");
            }

            Fails(new TweakedFormat(format, page => Moved(page)), "every run's box is inside its page");
            Fails(new TweakedFormat(format, Counted()), "pages read in parallel give the same");
            Fails(new TweakedFormat(format, null, damagedThrows: true), "a damaged file reads or fails as damaged");

            var nondeterministic = Conformance.CheckDocumentWriter(new CountingWriter(writer), format, face);
            Check(!nondeterministic.Passed && nondeterministic.Failures.Any(f => f.Check == "the same input writes the same bytes"), $"a writer that writes other bytes each time:\n{nondeterministic}");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void DocumentsDrawerAndFontSuites(Device device)
    {
        CpuOnly(device);
        var face = SubsetFace();
        Conformance.CheckDocumentPageDrawer(new TestPageDrawer(), face).ThrowIfFailed();
        var throwing = Conformance.CheckDocumentPageDrawer(new TestPageDrawer(throwOnMissingFont: true), face);
        Check(!throwing.Passed && throwing.Failures.Any(f => f.Check == "a missing font is counted, not thrown"), $"a drawer that throws for a missing font:\n{throwing}");

        Conformance.CheckFontWriter(new TestFontWriter(), [face], read: TestFontWriter.Read).ThrowIfFailed();
        var dropping = Conformance.CheckFontWriter(new TestFontWriter(dropOutlines: true), [face], read: TestFontWriter.Read);
        Check(!dropping.Passed && dropping.Failures.Any(f => f.Check.Contains("outline", StringComparison.Ordinal)), $"a writer that drops outlines:\n{dropping}");
    }

    private static readonly Lock DocumentGuardGate = new();
    private static bool s_documentGuardDefaults;

    private static void DocumentsDrawerAndFontGuards(Device device)
    {
        CpuOnly(device);
        const string name = "test-guard";
        lock (DocumentGuardGate)
        {
            if (!s_documentGuardDefaults)
            {
                // Stand-ins for the library's drawer and font writer, defaults for the rest of the process.
                Overrides.AsLibraryDefaults(() =>
                {
                    DocumentPageDrawers.Register(new TestPageDrawer(name));
                    FontWriters.Register(new TestFontWriter(name));
                });
                s_documentGuardDefaults = true;
            }
        }

        var face = SubsetFace();
        var page = new DocumentPage(0, 60f, 40f) { Paths = [DocumentPath.Rectangle(new DocumentBox(10f, 10f, 20f, 20f))] };
        var options = new DocumentDrawOptions { PixelsPerPoint = 1f, Drawer = name };
        var expected = DocumentPageDrawers.Draw(page, options).Image.Pixels;
        byte[] library = FontBytes(FontWriters.Get(name), face);
        try
        {
            DocumentPageDrawers.Register(new TestPageDrawer(name, failing: true));
            FontWriters.Register(new TestFontWriter(name, failing: true));
            Check(DocumentPageDrawers.Origin(name) == typeof(Tests).Assembly.GetName().Name && FontWriters.Default(name) is not null, "the app's over the library's");
            foreach (var call in new Action[] { () => DocumentPageDrawers.Draw(page, options), () => FontBytes(FontWriters.Get(name), face) })
            {
                try
                {
                    call();
                    Check(false, "the app's failure did not reach the caller");
                }
                catch (InvalidOperationException)
                {
                    // Throw, the default policy.
                }
            }

            DocumentPageDrawers.SetPolicy(name, SlotPolicy.FallBack);
            FontWriters.SetPolicy(name, SlotPolicy.FallBack);
            Check(DocumentPageDrawers.Draw(page, options).Image.Pixels.AsSpan().SequenceEqual(expected), "under FallBack the library's drawer answers");
            Check(FontBytes(FontWriters.Get(name), face).AsSpan().SequenceEqual(library), "under FallBack the destination holds the library's font alone");

            DocumentPageDrawers.Register(new TestPageDrawer(name, inverted: true));
            FontWriters.Register(new TestFontWriter(name, dropOutlines: true));
            DocumentPageDrawers.SetPolicy(name, SlotPolicy.Shadow, shadowRate: 1);
            FontWriters.SetPolicy(name, SlotPolicy.Shadow, shadowRate: 1);
            Check(DocumentPageDrawers.Draw(page, options).Image.Pixels.AsSpan().SequenceEqual(expected) && FontBytes(FontWriters.Get(name), face).AsSpan().SequenceEqual(library),
                "under Shadow the library's answer is kept");
            foreach (string registry in (string[])[nameof(DocumentPageDrawers), nameof(FontWriters)])
            {
                var row = Overrides.Report().Single(o => o.Registry == registry && o.Name == name);
                Check(row.Compared >= 1 && row.Differed >= 1, $"{registry}: compared {row.Compared}, differed {row.Differed}");
            }
        }
        finally
        {
            DocumentPageDrawers.Unregister(name);
            FontWriters.Unregister(name);
            DocumentPageDrawers.SetPolicy(name, SlotPolicy.Throw);
            FontWriters.SetPolicy(name, SlotPolicy.Throw);
        }

        Check(DocumentPageDrawers.Origin(name) == Overrides.Library && FontWriters.Origin(name) == Overrides.Library, "unregistered, the library's are back");
    }

    private static byte[] FontBytes(IFontWriter writer, IFontFace face)
    {
        using var stream = new MemoryStream();
        writer.Write(face, stream, new FontWriteOptions());
        return stream.ToArray();
    }

    // Two toy files the writer makes: a page of text in the font, a hidden line and a path, and a page of an image.
    private static List<DocumentSample> ToySamples(string folder, IFontFace face)
    {
        string text = Path.Combine(folder, "report.toydoc"), image = Path.Combine(folder, "scan.toydoc");
        using (var stream = File.Create(text))
        using (var builder = new ToyDocumentWriter().Create(stream, new DocumentWriteOptions()))
        {
            var page = builder.AddPage(612f, 792f);
            WriteRun(page, face, "The quarterly report.", 72f, 100f, 14f, DocumentTextMode.Fill);
            WriteRun(page, face, "Figures are final.", 72f, 130f, 11f, DocumentTextMode.Hidden);
            page.Path(DocumentPath.Rectangle(new DocumentBox(72f, 150f, 300f, 2f)));
            builder.Info = new DocumentInfo { Title = "A technical report" };
            builder.Attach("notes.txt", "text/plain", new MemoryStream("Notes."u8.ToArray()));
            builder.Finish();
        }

        using (var stream = File.Create(image))
        using (var builder = new ToyDocumentWriter().Create(stream, new DocumentWriteOptions()))
        {
            var pixels = new float[3 * 8 * 6];
            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = i % 256 / 255f;
            }

            builder.AddPage(300f, 400f).Image(new ImageData(pixels, 3, 6, 8), new DocumentBox(10f, 10f, 280f, 210f), DocumentImageEncoding.Flate);
            builder.AddPage(300f, 400f);
            builder.Finish();
        }

        return [new(text, Pages: 1, Texts: ["The quarterly report.\nFigures are final."]), new(image, Pages: 2, Texts: ["", ""])];
    }

    private static void WriteRun(IDocumentPageBuilder page, IFontFace face, string text, float x, float baseline, float size, DocumentTextMode mode)
    {
        var glyphs = new List<DocumentGlyph>();
        float scale = size / face.Metrics.UnitsPerEm;
        foreach (var rune in text.EnumerateRunes())
        {
            int glyph = face.Glyph(rune.Value);
            glyphs.Add(new DocumentGlyph(glyph, x, baseline, face.Advance(glyph) * scale));
            x += face.Advance(glyph) * scale;
        }

        page.Glyphs(face, size, [.. glyphs], text, mode, DocumentColour.Black);
    }

    // A page whose runs stand past its right edge.
    private static DocumentPage Moved(DocumentPage page) => new(page.Index, page.Width, page.Height)
    {
        Runs = [.. page.Runs.Select(r => r with { Box = r.Box with { X = page.Width + 10f } })], Lines = page.Lines, Text = page.Text, Images = page.Images, Paths = page.Paths,
        Clips = page.Clips,
    };

    // Pages whose text changes with every read.
    private static Func<DocumentPage, DocumentPage> Counted()
    {
        int reads = 0;
        return page => new DocumentPage(page.Index, page.Width, page.Height)
        {
            Runs = page.Runs, Lines = page.Lines, Text = page.Text + Interlocked.Increment(ref reads).ToString(CultureInfo.InvariantCulture), Images = page.Images, Paths = page.Paths,
        };
    }

    private static DocumentTextRun Placed(string text, float x, float baseline, float size, int order, float? width = null, bool logical = false) =>
        new(text, new DocumentBox(x, baseline - size * 0.8f, width ?? text.Length * size * 0.5f, size), size) { Baseline = baseline, Order = order, Logical = logical };

    private static void LayoutsLinesLeftToRight(Device device)
    {
        CpuOnly(device);
        var options = new DocumentOptions { Layout = DocumentTextLayouts.Lines, ReadingOrder = ReadingOrders.Bidi };
        DocumentTextRun[] runs =
        [
            Placed("world", 100f, 50f, 10f, 0, 25f),
            Placed("Hello", 50f, 50f, 10f, 1, 25f),
            Placed("Sec", 50f, 70f, 10f, 2, 15f),
            Placed("ond", 65.5f, 70f, 10f, 3, 15f),
            Placed("line", 100f, 70f, 10f, 4, 20f),
            Placed("2", 125.5f, 46.5f, 6f, 5, 3f),
        ];
        var text = DocumentTextLayouts.Arrange(runs, 612f, 792f, options);
        Check(text.Text == "Hello world2\nSecond line", $"the page's text: \"{text.Text}\"");
        Check(text.Lines.Count == 2 && text.Lines[0].Runs.SequenceEqual([1, 0, 5]) && text.Lines[1].Runs.SequenceEqual([2, 3, 4]) && !text.Lines[0].RightToLeft
              && text.Lines[0].Box.X == 50f && MathF.Abs(text.Lines[0].Box.Right - 128.5f) < 1e-4f, $"the lines: {string.Join(" | ", text.Lines)}");
        var wide = DocumentTextLayouts.Arrange(runs, 612f, 792f, options with { WordGap = 5f });
        Check(wide.Text == "Helloworld2\nSecondline", $"with a wider word gap: \"{wide.Text}\"");
        var tight = DocumentTextLayouts.Arrange(runs, 612f, 792f, options with { LineTolerance = 0.1f });
        Check(tight.Lines.Count == 3 && tight.Text.StartsWith("2\n", StringComparison.Ordinal), $"a tighter line tolerance puts the superscript on its own line: \"{tight.Text}\"");
        Check(DocumentTextLayouts.Arrange([], 612f, 792f, options) == DocumentText.Empty && DocumentTextLayouts.Arrange([Placed("", 1f, 1f, 1f, 0)], 612f, 792f, options).Lines.Count == 0,
            "no runs, no text");
    }

    private static void LayoutsLinesBothDirections(Device device)
    {
        CpuOnly(device);
        var options = new DocumentOptions { Layout = DocumentTextLayouts.Lines, ReadingOrder = ReadingOrders.Bidi };

        // "قرأت 3 كتب" (I read 3 books) as a page shows it: the words' glyphs left to right, so each word's text reversed.
        DocumentTextRun[] visual =
        [
            Placed(LinesLayout.Reversed("قرأت"), 200f, 50f, 10f, 0, 40f),
            Placed("3", 180f, 50f, 10f, 1, 10f),
            Placed(LinesLayout.Reversed("كتب"), 140f, 50f, 10f, 2, 30f),
        ];
        var line = DocumentTextLayouts.Arrange(visual, 612f, 792f, options).Lines.Single();
        Check(line.Text == "قرأت 3 كتب" && line.RightToLeft && line.Runs.SequenceEqual([0, 1, 2]), $"a right-to-left line: \"{line.Text}\", right to left {line.RightToLeft}");

        DocumentTextRun[] logical =
        [
            Placed("قرأت", 200f, 50f, 10f, 0, 40f, logical: true),
            Placed("3", 180f, 50f, 10f, 1, 10f, logical: true),
            Placed("كتب", 140f, 50f, 10f, 2, 30f, logical: true),
        ];
        Check(DocumentTextLayouts.Arrange(logical, 612f, 792f, options).Text == "قرأت 3 كتب", "runs in logical order give the same text");

        // "افتح Idrak الآن" (open Idrak now): a left-to-right word in a right-to-left line.
        DocumentTextRun[] mixed =
        [
            Placed(LinesLayout.Reversed("افتح"), 300f, 50f, 10f, 0, 40f),
            Placed("Idrak", 240f, 50f, 10f, 1, 50f),
            Placed(LinesLayout.Reversed("الآن"), 190f, 50f, 10f, 2, 40f),
        ];
        string mixedText = DocumentTextLayouts.Arrange(mixed, 612f, 792f, options).Text;
        Check(mixedText == "افتح Idrak الآن", $"a left-to-right word in a right-to-left line: \"{mixedText}\"");

        // A right-to-left word in a left-to-right line: "Send the كتب today".
        DocumentTextRun[] english =
        [
            Placed("Send", 50f, 50f, 10f, 0, 20f),
            Placed("the", 75f, 50f, 10f, 1, 15f),
            Placed(LinesLayout.Reversed("كتب"), 95f, 50f, 10f, 2, 15f),
            Placed("today", 115f, 50f, 10f, 3, 25f),
        ];
        var left = DocumentTextLayouts.Arrange(english, 612f, 792f, options).Lines.Single();
        Check(left.Text == "Send the كتب today" && !left.RightToLeft, $"a right-to-left word in a left-to-right line: \"{left.Text}\"");
    }

    private static void LayoutsLinesDirection(Device device)
    {
        CpuOnly(device);
        var options = new DocumentOptions { Layout = DocumentTextLayouts.Lines, ReadingOrder = ReadingOrders.Bidi };

        // A line of digits alone takes the page's direction: right to left under an Arabic line, left to right under a Latin one.
        DocumentTextRun[] arabic = [Placed(LinesLayout.Reversed("كتب"), 100f, 50f, 10f, 0, 30f), Placed("12", 160f, 70f, 10f, 1, 10f), Placed("34", 140f, 70f, 10f, 2, 10f)];
        var lines = DocumentTextLayouts.Arrange(arabic, 612f, 792f, options).Lines;
        Check(lines.Count == 2 && lines[0].RightToLeft && lines[1].RightToLeft && lines[1].Text == "12 34", $"digits under Arabic: {string.Join(" | ", lines)}");
        DocumentTextRun[] latin = [Placed("Total", 100f, 50f, 10f, 0, 25f), Placed("12", 140f, 70f, 10f, 1, 10f), Placed("34", 160f, 70f, 10f, 2, 10f)];
        lines = DocumentTextLayouts.Arrange(latin, 612f, 792f, options).Lines;
        Check(lines.Count == 2 && !lines[1].RightToLeft && lines[1].Text == "12 34", $"digits under Latin: {string.Join(" | ", lines)}");

        // The option sets every line.
        DocumentTextRun[] words = [Placed("Hello", 50f, 50f, 10f, 0, 25f), Placed("world", 100f, 50f, 10f, 1, 25f)];
        var forced = DocumentTextLayouts.Arrange(words, 612f, 792f, options with { Direction = DocumentTextDirection.RightToLeft }).Lines.Single();
        Check(forced.RightToLeft && forced.Text == "Hello world" && forced.Runs.SequenceEqual([1, 0]), $"right to left by the option: {forced}");

        // Runs without a place: lines of their own, in content order, after the placed ones.
        DocumentTextRun[] notes = [new("note two", DocumentBox.Empty, 10f) { Order = 5 }, Placed("Title", 50f, 50f, 10f, 0), new("note one", DocumentBox.Empty, 10f) { Order = 3 }];
        string text = DocumentTextLayouts.Arrange(notes, 612f, 792f, options).Text;
        Check(text == "Title\nnote one\nnote two", $"runs without a place: \"{text}\"");
    }

    private static void LayoutsContentAndNames(Device device)
    {
        CpuOnly(device);
        DocumentTextRun[] runs = [Placed("c", 10f, 50f, 10f, 2), Placed("a", 30f, 90f, 10f, 0), Placed("b", 50f, 10f, 10f, 1)];
        var content = DocumentTextLayouts.Arrange(runs, 612f, 792f, new DocumentOptions { Layout = DocumentTextLayouts.Content });
        Check(content.Text == "abc" && content.Lines.Count == 3 && content.Lines[0].Runs.Single() == 1 && content.Lines.All(l => !l.RightToLeft), $"content order: \"{content.Text}\"");
        Check(DocumentTextLayouts.Arrange(runs, 612f, 792f, new DocumentOptions { ReadingOrder = ReadingOrders.Runs }).Text == "b\nc\na", "the runs reading order, by name");

        try
        {
            DocumentTextLayouts.Arrange(runs, 612f, 792f, new DocumentOptions { Layout = "a-layout-nobody-registered" });
            Check(false, "an unregistered layout was used");
        }
        catch (NotRegisteredException e)
        {
            Check(e.Registry == nameof(DocumentTextLayouts) && e.Message.Contains(DocumentOptions.LayoutVariable, StringComparison.Ordinal) && e.Message.Contains("lines", StringComparison.Ordinal), e.Message);
        }

        try
        {
            DocumentTextLayouts.Arrange(runs, 612f, 792f, new DocumentOptions { ReadingOrder = "an-order-nobody-registered" });
            Check(false, "an unregistered reading order was used");
        }
        catch (NotRegisteredException e)
        {
            Check(e.Registry == nameof(ReadingOrders) && e.Message.Contains(DocumentOptions.ReadingOrderVariable, StringComparison.Ordinal)
                  && e.Message.Contains("Idrak.Imaging", StringComparison.Ordinal) && e.Message.Contains(ReadingOrders.Bidi, StringComparison.Ordinal), e.Message);
        }

        WithEnvironment(new Dictionary<string, string?> { [DocumentOptions.LayoutVariable] = DocumentTextLayouts.Content, [DocumentOptions.ReadingOrderVariable] = ReadingOrders.Runs }, () =>
        {
            var options = new DocumentOptions();
            Check(options.Layout == DocumentTextLayouts.Content && options.ReadingOrder == ReadingOrders.Runs && DocumentTextLayouts.Arrange(runs, 612f, 792f, options).Text == "abc",
                "the layout and the reading order from their variables");
        });

        Check(ReadingOrders.Get(ReadingOrders.Bidi).RightToLeft("123 كتب abc") == true && ReadingOrders.Get(ReadingOrders.Bidi).RightToLeft("(12) abc كتب") == false
              && ReadingOrders.Get(ReadingOrders.Bidi).RightToLeft("12, 34") is null && ReadingOrders.Get(ReadingOrders.Runs).RightToLeft("שלום world") == true
              && ReadingOrders.Get(ReadingOrders.Bidi).RightToLeft("⁧abc⁩ كتب") == true, "the first strong character, an isolate's skipped");
    }

    private static void LayoutsOverride(Device device)
    {
        CpuOnly(device);
        DocumentTextRun[] runs = [Placed("Hello", 50f, 50f, 10f, 0, 25f), Placed("world", 100f, 50f, 10f, 1, 25f)];
        var options = new DocumentOptions { Layout = DocumentTextLayouts.Lines };
        string expected = DocumentTextLayouts.Arrange(runs, 612f, 792f, options).Text;
        Check(expected == "Hello world", expected);
        try
        {
            DocumentTextLayouts.Register(new TestLayout(failing: false));
            Check(DocumentTextLayouts.Origin(DocumentTextLayouts.Lines) == typeof(Tests).Assembly.GetName().Name && DocumentTextLayouts.Default(DocumentTextLayouts.Lines) is not null
                  && DocumentTextLayouts.Arrange(runs, 612f, 792f, options).Text == "HELLO WORLD", "the app's layout answers");
            DocumentTextLayouts.Register(new TestLayout(failing: true));
            try
            {
                DocumentTextLayouts.Arrange(runs, 612f, 792f, options);
                Check(false, "the app's failure did not reach the caller");
            }
            catch (InvalidOperationException)
            {
                // Throw, the default policy.
            }

            DocumentTextLayouts.SetPolicy(DocumentTextLayouts.Lines, SlotPolicy.FallBack);
            Check(DocumentTextLayouts.Arrange(runs, 612f, 792f, options).Text == expected, "under FallBack the library's lines answers");
            DocumentTextLayouts.Register(new TestLayout(failing: false));
            DocumentTextLayouts.SetPolicy(DocumentTextLayouts.Lines, SlotPolicy.Shadow, shadowRate: 1);
            Check(DocumentTextLayouts.Arrange(runs, 612f, 792f, options).Text == expected, "under Shadow the library's lines answers");
            var row = Overrides.Report().Single(o => o.Registry == nameof(DocumentTextLayouts) && o.Name == DocumentTextLayouts.Lines);
            Check(row.Compared >= 1 && row.Differed >= 1, $"compared {row.Compared}, differed {row.Differed}");
        }
        finally
        {
            DocumentTextLayouts.Unregister(DocumentTextLayouts.Lines);
            DocumentTextLayouts.SetPolicy(DocumentTextLayouts.Lines, SlotPolicy.Throw);
        }

        Check(DocumentTextLayouts.Origin(DocumentTextLayouts.Lines) == Overrides.Library && DocumentTextLayouts.Arrange(runs, 612f, 792f, options).Text == expected, "the library's lines is back");
    }
}

// An app's "lines" for the layout guard: the library's text upper-cased, or a failure.
internal sealed class TestLayout(bool failing) : IDocumentTextLayout
{
    public string Name => DocumentTextLayouts.Lines;

    public string Summary => "the library's lines, upper-cased (a test's)";

    public DocumentText Arrange(IReadOnlyList<DocumentTextRun> runs, float width, float height, DocumentOptions options)
    {
        if (failing)
        {
            throw new InvalidOperationException("The test's layout fails, on purpose.");
        }

        var text = DocumentTextLayouts.Default(DocumentTextLayouts.Lines)!.Arrange(runs, width, height, options);
        return new DocumentText([.. text.Lines.Select(l => l with { Text = l.Text.ToUpperInvariant() })], text.Text.ToUpperInvariant());
    }
}

// A format over the outside toy that breaks one rule: pages changed by a function, or a damaged file failing with an
// exception other than InvalidDataException.
internal sealed class TweakedFormat(IDocumentFormat inner, Func<DocumentPage, DocumentPage>? tweak, bool damagedThrows = false) : IDocumentFormat
{
    public string Name => inner.Name;

    public IReadOnlyCollection<string> Extensions => inner.Extensions;

    public IReadOnlyCollection<string> MediaTypes => inner.MediaTypes;

    public bool Accepts(ReadOnlySpan<byte> header) => inner.Accepts(header);

    public IDocument Open(Stream stream, string name, DocumentOptions options)
    {
        try
        {
            return new TweakedDocument(inner.Open(stream, name, options), tweak);
        }
        catch (InvalidDataException e) when (damagedThrows && !e.Message.Contains("Max", StringComparison.Ordinal))
        {
            throw new IndexOutOfRangeException("a damaged file, reported the wrong way", e);
        }
    }

    private sealed class TweakedDocument(IDocument document, Func<DocumentPage, DocumentPage>? change) : IDocument
    {
        public string Format => document.Format;

        public DocumentInfo Info => document.Info;

        public int PageCount => document.PageCount;

        public IReadOnlyList<DocumentAttachment> Attachments => document.Attachments;

        public IReadOnlyList<DocumentField> Fields => document.Fields;

        public IReadOnlyList<DocumentOutline> Outline => document.Outline;

        public IReadOnlyList<DocumentSignature> Signatures => document.Signatures;

        public DocumentPage ReadPage(int index, DocumentParts parts) => change is null ? document.ReadPage(index, parts) : change(document.ReadPage(index, parts));

        public DocumentDrawing DrawPage(int index, DocumentDrawOptions options) => document.DrawPage(index, options);

        public string? PageLabel(int index) => document.PageLabel(index);

        public void Dispose() => document.Dispose();
    }
}

// A writer over the outside toy that writes one more blank line after the header with every document: other bytes for the
// same input, still readable.
internal sealed class CountingWriter(IDocumentWriter inner) : IDocumentWriter
{
    private int _documents;

    public string Name => inner.Name;

    public IReadOnlyCollection<string> Extensions => inner.Extensions;

    public IReadOnlyCollection<string> MediaTypes => inner.MediaTypes;

    public IDocumentBuilder Create(Stream destination, DocumentWriteOptions options) =>
        inner.Create(new AfterFirstWrite(destination, new byte[Interlocked.Increment(ref _documents)].Select(_ => (byte)'\n').ToArray()), options);

    public IDocumentEditor Edit(IDocument source, Stream destination, DocumentWriteOptions options) => inner.Edit(source, destination, options);

    private sealed class AfterFirstWrite(Stream destination, byte[] extra) : Stream
    {
        private bool _written;

        public override bool CanRead => false;

        public override bool CanSeek => false;

        public override bool CanWrite => true;

        public override long Length => throw new NotSupportedException();

        public override long Position
        {
            get => throw new NotSupportedException();
            set => throw new NotSupportedException();
        }

        public override void Write(byte[] buffer, int offset, int count) => Write(buffer.AsSpan(offset, count));

        public override void Write(ReadOnlySpan<byte> buffer)
        {
            destination.Write(buffer);
            if (!_written)
            {
                _written = true;
                destination.Write(extra);
            }
        }

        public override void Flush() => destination.Flush();

        public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();

        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();

        public override void SetLength(long value) => throw new NotSupportedException();
    }
}

// A page drawer of the test's own, enough for the drawer suite: parts in content order onto the background, fills through
// the registered rasterizer (even-odd by sampling each pixel's middle), strokes as a filled quad per flattened segment,
// glyph runs through their faces' outlines, images by their nearest pixel. Options: throw for a missing font (the suite must
// catch it), fail every call, or draw inverted (the guards' proofs).
internal sealed class TestPageDrawer(string name = "test-drawer", bool throwOnMissingFont = false, bool failing = false, bool inverted = false) : IDocumentPageDrawer
{
    public string Name => name;

    public string Summary => "a test's drawer: nearest pixels, sampled even-odd, quads for strokes";

    public DocumentDrawing Draw(DocumentPage page, DocumentDrawOptions options)
    {
        if (failing)
        {
            throw new InvalidOperationException("The test's drawer fails, on purpose.");
        }

        var (w, h) = DocumentPageDrawers.PixelSize(page, options);
        int channels = options.Colour == DocumentDrawColour.Grey ? 1 : 3;
        float ppp = options.PixelsPerPoint;
        var pixels = new float[channels * w * h];
        var coverage = new float[w * h];
        coverage.AsSpan().Fill(1f);
        Blend(pixels, channels, coverage, options.Background);
        var rasterizer = OutlineRasterizers.Get(options.Rasterizer);
        var parts = new List<(int Order, int Kind, int Index)>();
        if ((options.Parts & DocumentParts.Paths) != 0)
        {
            parts.AddRange(page.Paths.Select((p, i) => (p.Order, 0, i)));
        }

        if ((options.Parts & DocumentParts.Images) != 0)
        {
            parts.AddRange(page.Images.Select((p, i) => (p.Order, 1, i)));
        }

        if ((options.Parts & DocumentParts.Text) != 0)
        {
            parts.AddRange(page.Runs.Select((p, i) => (p.Order, 2, i)));
        }

        parts.Sort();
        int missing = 0;
        foreach (var (_, kind, index) in parts)
        {
            Array.Clear(coverage);
            switch (kind)
            {
                case 0:
                {
                    var path = page.Paths[index];
                    if (path.Fill == DocumentFill.NonZero)
                    {
                        rasterizer.Fill(Outline(path.Verbs.ToArray(), path.Points.ToArray()), ppp, 0f, 0f, coverage, w, h);
                        Blend(pixels, channels, coverage, path.FillColour);
                    }
                    else if (path.Fill == DocumentFill.EvenOdd)
                    {
                        EvenOdd(Flatten(path), ppp, coverage, w, h);
                        Blend(pixels, channels, coverage, path.FillColour);
                    }

                    if (path.Stroke is { } stroke)
                    {
                        Array.Clear(coverage);
                        foreach (var contour in Flatten(path))
                        {
                            for (int k = 1; k < contour.Count; k++)
                            {
                                Quad(contour[k - 1], contour[k], MathF.Max(stroke, 1f / ppp), rasterizer, ppp, coverage, w, h);
                            }
                        }

                        Blend(pixels, channels, coverage, path.StrokeColour);
                    }

                    break;
                }

                case 1:
                    PlaceImage(page.Images[index], pixels, channels, w, h, ppp);
                    break;
                default:
                {
                    var run = page.Runs[index];
                    if (run.Hidden && !options.Hidden)
                    {
                        break;
                    }

                    if (run.Face is not { } face)
                    {
                        missing++;
                        if (throwOnMissingFont)
                        {
                            throw new InvalidOperationException($"No font for \"{run.Text}\".");
                        }

                        break;
                    }

                    float scale = run.FontSize / face.Metrics.UnitsPerEm * ppp;
                    foreach (var glyph in run.Glyphs.Span)
                    {
                        rasterizer.Fill(face.Outline(glyph.Glyph), scale, glyph.X * ppp, glyph.Y * ppp, coverage, w, h);
                    }

                    Blend(pixels, channels, coverage, run.Colour);
                    break;
                }
            }
        }

        if (inverted)
        {
            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = 1f - pixels[i];
            }
        }

        DocumentIssue[] issues = missing > 0 ? [new DocumentIssue(DocumentIssue.MissingFont, $"{missing} runs whose font is not there", missing)] : [];
        return new DocumentDrawing(new ImageData(pixels, channels, h, w), issues);
    }

    private static void Blend(float[] pixels, int channels, float[] coverage, DocumentColour colour)
    {
        int area = coverage.Length;
        for (int c = 0; c < channels; c++)
        {
            float value = channels == 1 ? colour.Luma : c switch { 0 => colour.Red, 1 => colour.Green, _ => colour.Blue };
            for (int i = 0; i < area; i++)
            {
                float a = coverage[i] * colour.Alpha;
                if (a > 0f)
                {
                    pixels[c * area + i] = pixels[c * area + i] * (1f - a) + value * a;
                }
            }
        }
    }

    // Page points (y down) as an outline the rasterizer takes (y up), placed by a scale of the pixels per point at the origin.
    private static GlyphOutline Outline(OutlineVerb[] verbs, float[] points)
    {
        for (int i = 1; i < points.Length; i += 2)
        {
            points[i] = -points[i];
        }

        return new GlyphOutline(verbs, points);
    }

    private static List<List<(float X, float Y)>> Flatten(DocumentPath path)
    {
        var contours = new List<List<(float X, float Y)>>();
        List<(float X, float Y)>? current = null;
        var p = path.Points.Span;
        int k = 0;
        (float X, float Y) last = default, start = default;
        foreach (var verb in path.Verbs.Span)
        {
            switch (verb)
            {
                case OutlineVerb.Move:
                    current = [];
                    contours.Add(current);
                    last = start = (p[k], p[k + 1]);
                    current.Add(last);
                    k += 2;
                    break;
                case OutlineVerb.Line:
                    last = (p[k], p[k + 1]);
                    current?.Add(last);
                    k += 2;
                    break;
                case OutlineVerb.Quadratic:
                    for (int s = 1; s <= 16; s++)
                    {
                        float t = s / 16f, u = 1f - t;
                        current?.Add((u * u * last.X + 2 * u * t * p[k] + t * t * p[k + 2], u * u * last.Y + 2 * u * t * p[k + 1] + t * t * p[k + 3]));
                    }

                    last = (p[k + 2], p[k + 3]);
                    k += 4;
                    break;
                case OutlineVerb.Cubic:
                    for (int s = 1; s <= 16; s++)
                    {
                        float t = s / 16f, u = 1f - t;
                        current?.Add((u * u * u * last.X + 3 * u * u * t * p[k] + 3 * u * t * t * p[k + 2] + t * t * t * p[k + 4],
                            u * u * u * last.Y + 3 * u * u * t * p[k + 1] + 3 * u * t * t * p[k + 3] + t * t * t * p[k + 5]));
                    }

                    last = (p[k + 4], p[k + 5]);
                    k += 6;
                    break;
                default:
                    current?.Add(start);
                    last = start;
                    break;
            }
        }

        return contours;
    }

    // Each pixel whose middle a ray to the right from it crosses the contours an odd number of times is covered.
    private static void EvenOdd(List<List<(float X, float Y)>> contours, float ppp, float[] coverage, int w, int h)
    {
        for (int y = 0; y < h; y++)
        {
            float py = (y + 0.5f) / ppp;
            for (int x = 0; x < w; x++)
            {
                float px = (x + 0.5f) / ppp;
                bool inside = false;
                foreach (var contour in contours)
                {
                    for (int i = 0, j = contour.Count - 1; i < contour.Count; j = i++)
                    {
                        var (a, b) = (contour[i], contour[j]);
                        if ((a.Y > py) != (b.Y > py) && px < a.X + (py - a.Y) * (b.X - a.X) / (b.Y - a.Y))
                        {
                            inside = !inside;
                        }
                    }
                }

                coverage[y * w + x] = inside ? 1f : 0f;
            }
        }
    }

    private static void Quad((float X, float Y) a, (float X, float Y) b, float width, IOutlineRasterizer rasterizer, float ppp, float[] coverage, int w, int h)
    {
        float dx = b.X - a.X, dy = b.Y - a.Y, length = MathF.Sqrt(dx * dx + dy * dy);
        if (length == 0f)
        {
            return;
        }

        float nx = -dy / length * width / 2f, ny = dx / length * width / 2f;
        rasterizer.Fill(Outline([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close],
            [a.X + nx, a.Y + ny, b.X + nx, b.Y + ny, b.X - nx, b.Y - ny, a.X - nx, a.Y - ny]), ppp, 0f, 0f, coverage, w, h);
    }

    private static void PlaceImage(DocumentImage image, float[] pixels, int channels, int w, int h, float ppp)
    {
        var data = image.Decode();
        var box = image.Box;
        int area = w * h;
        for (int y = Math.Max(0, (int)(box.Y * ppp)); y < Math.Min(h, (int)MathF.Ceiling(box.Bottom * ppp)); y++)
        {
            for (int x = Math.Max(0, (int)(box.X * ppp)); x < Math.Min(w, (int)MathF.Ceiling(box.Right * ppp)); x++)
            {
                float u = ((x + 0.5f) / ppp - box.X) / box.Width, v = ((y + 0.5f) / ppp - box.Y) / box.Height;
                if (u is < 0f or >= 1f || v is < 0f or >= 1f)
                {
                    continue;
                }

                int ix = Math.Min(data.Width - 1, (int)(u * data.Width)), iy = Math.Min(data.Height - 1, (int)(v * data.Height));
                float Channel(int c) => data.Pixels[(Math.Min(c, data.Channels - 1) * data.Height + iy) * data.Width + ix];
                if (channels == 1)
                {
                    pixels[y * w + x] = data.Channels == 1 ? Channel(0) : 0.299f * Channel(0) + 0.587f * Channel(1) + 0.114f * Channel(2);
                }
                else
                {
                    for (int c = 0; c < 3; c++)
                    {
                        pixels[c * area + y * w + x] = Channel(c);
                    }
                }
            }
        }
    }
}

// A font writer of the test's own, enough for the font writer suite: the face's metrics, every glyph's advance and outline
// (a subset's other glyphs empty), the characters of the Basic Multilingual Plane it maps and its layout tables, as lines
// of text; Read gives the face back. Options: drop every outline (the suite must catch it), or fail (the guards' proofs).
internal sealed class TestFontWriter(string name = "test-fonts", bool dropOutlines = false, bool failing = false) : IFontWriter
{
    private static readonly string[] LayoutTables = ["GSUB", "GPOS", "GDEF"];

    public string Name => name;

    public IReadOnlyCollection<string> Extensions { get; } = [".testfont"];

    public void Write(IFontFace face, Stream destination, FontWriteOptions options)
    {
        if (failing)
        {
            destination.Write("half a font"u8);
            throw new InvalidOperationException("The test's font writer fails, on purpose.");
        }

        var kept = options.Glyphs is { } glyphs ? glyphs.ToHashSet() : null;
        var text = new StringBuilder("IDRAK-TEST-FONT 1\n");
        text.Append(B64(face.Family)).Append(' ').Append(B64(face.Style)).Append(' ').Append(I(face.Metrics.UnitsPerEm)).Append(' ').Append(I(face.Metrics.Ascender)).Append(' ')
            .Append(I(face.Metrics.Descender)).Append(' ').Append(I(face.Metrics.LineGap)).Append(' ').Append(I(face.GlyphCount)).Append('\n');
        for (int g = 0; g < face.GlyphCount; g++)
        {
            var outline = dropOutlines || (kept is not null && !kept.Contains(g)) ? GlyphOutline.Empty : face.Outline(g);
            text.Append(I(face.Advance(g))).Append(' ').Append(outline.IsEmpty ? "-" : string.Concat(outline.Verbs.ToArray().Select(v => (char)('0' + (int)v)))).Append(' ')
                .Append(outline.IsEmpty ? "-" : string.Join(',', outline.Points.ToArray().Select(p => p.ToString("R", CultureInfo.InvariantCulture)))).Append('\n');
        }

        var characters = new StringBuilder();
        for (int code = 0; code <= 0xFFFF; code++)
        {
            int glyph = face.Glyph(code);
            if (glyph != 0 && (kept is null || kept.Contains(glyph)))
            {
                characters.Append(characters.Length > 0 ? ";" : "").Append(I(code)).Append(':').Append(I(glyph));
            }
        }

        text.Append(characters.Length > 0 ? characters.ToString() : "-").Append('\n');
        string[] tags = options.KeepLayout ? LayoutTables : [];
        foreach (string tag in tags)
        {
            var table = face.Table(tag);
            if (!table.IsEmpty)
            {
                text.Append(tag).Append(' ').Append(Convert.ToBase64String(table.Span)).Append('\n');
            }
        }

        destination.Write(Encoding.UTF8.GetBytes(text.ToString()));
    }

    public static IFontFace Read(ReadOnlyMemory<byte> file)
    {
        string[] lines = Encoding.UTF8.GetString(file.Span).Split('\n');
        if (lines[0] != "IDRAK-TEST-FONT 1")
        {
            throw new InvalidDataException("not a test font");
        }

        string[] head = lines[1].Split(' ');
        int count = int.Parse(head[6], CultureInfo.InvariantCulture);
        var advances = new int[count];
        var outlines = new GlyphOutline[count];
        for (int g = 0; g < count; g++)
        {
            string[] f = lines[2 + g].Split(' ');
            advances[g] = int.Parse(f[0], CultureInfo.InvariantCulture);
            outlines[g] = f[1] == "-" ? GlyphOutline.Empty
                : new GlyphOutline([.. f[1].Select(c => (OutlineVerb)(c - '0'))], [.. f[2].Split(',').Select(p => float.Parse(p, CultureInfo.InvariantCulture))]);
        }

        var characters = new Dictionary<int, int>();
        if (lines[2 + count] != "-")
        {
            foreach (string pair in lines[2 + count].Split(';'))
            {
                string[] parts = pair.Split(':');
                characters[int.Parse(parts[0], CultureInfo.InvariantCulture)] = int.Parse(parts[1], CultureInfo.InvariantCulture);
            }
        }

        var tables = lines.Skip(3 + count).Where(l => l.Length > 0).Select(l => l.Split(' ')).ToDictionary(f => f[0], f => (ReadOnlyMemory<byte>)Convert.FromBase64String(f[1]));
        var metrics = new FontMetrics(int.Parse(head[2], CultureInfo.InvariantCulture), int.Parse(head[3], CultureInfo.InvariantCulture), int.Parse(head[4], CultureInfo.InvariantCulture),
            int.Parse(head[5], CultureInfo.InvariantCulture));
        return new TestFontFace(Unbase(head[0]), Unbase(head[1]), metrics, advances, outlines, characters, tables);
    }

    private static string I(int value) => value.ToString(CultureInfo.InvariantCulture);

    private static string B64(string text) => Convert.ToBase64String(Encoding.UTF8.GetBytes(text));

    private static string Unbase(string text) => Encoding.UTF8.GetString(Convert.FromBase64String(text));

    private sealed class TestFontFace(string family, string style, FontMetrics metrics, int[] advances, GlyphOutline[] outlines, Dictionary<int, int> characters,
        Dictionary<string, ReadOnlyMemory<byte>> tables) : IFontFace
    {
        public string Family => family;

        public string Style => style;

        public string Format => "test-fonts";

        public FontMetrics Metrics => metrics;

        public int GlyphCount => advances.Length;

        public int Glyph(int codePoint) => characters.GetValueOrDefault(codePoint);

        public int Advance(int glyph) => (uint)glyph < (uint)advances.Length ? advances[glyph] : 0;

        public GlyphOutline Outline(int glyph) => (uint)glyph < (uint)outlines.Length ? outlines[glyph] : GlyphOutline.Empty;

        public ReadOnlyMemory<byte> Table(string tag) => tables.GetValueOrDefault(tag);
    }
}
