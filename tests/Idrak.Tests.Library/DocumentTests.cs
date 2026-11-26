// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using Idrak.Imaging;
using Idrak.Imaging.Shaping;

// Plan 28, phase D0: the document contract's options and their variables, its values, and the library's layouts.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] DocumentContractGroup =
    [
        ("documents contracts: the options: the size limits measured from the free memory unless set, every variable read (IDRAK_DOCUMENT_*, IDRAK_FONT_CURVE_TOLERANCE), a bad value refused naming its variable, passwords never printed", DocumentsOptions),
        ("documents contracts: colours (grey, RGB, CMYK and their RGB, opacity), matrices, boxes, paths (points checked against their steps), page labels (roman, letters, prefixes), a check's validity", DocumentsValues),
    ];

    internal static readonly (string Name, Action<Device> Run)[] DocumentLayoutGroup =
    [
        ("document layouts: lines: runs grouped by baseline and size into lines from the top, ordered by place whatever their content order, a space where a gap passes the word gap (an option), none inside a word, a superscript kept on its line", LayoutsLinesLeftToRight),
        ("document layouts: lines: a right-to-left line of runs in visual order comes out in logical order, a number and a left-to-right word inside it and a right-to-left word in a left-to-right line through bidi; runs in logical order give the same text", LayoutsLinesBothDirections),
        ("document layouts: lines: each line's direction from its first strong character, a line with none taking the page's; the direction option sets every line; runs without a place are lines of their own, after the placed ones", LayoutsLinesDirection),
        ("document layouts: lines: once warm, a page of left-to-right and right-to-left lines allocates only what it gives back (its lines, each line's runs and text, the page's text) and the content-order text of each line of several runs, which its direction is read from", LayoutsLinesAllocation),
        ("document layouts: lines: once warm, a page of mixed-direction lines (about 20, 80 and 300 characters: numbers and Latin words in right-to-left lines, Arabic words in a left-to-right one) allocates only what it gives back and the texts it asks the reading order about, and the bidi order's ToColumns and FromColumns on each line only the string they give back", LayoutsLinesMixedAllocation),
        ("document layouts: content: runs in content order, joined; the runs reading order; the layout named by its variable; a line's direction from its first strong character", LayoutsContentAndNames),
    ];

    private static void DocumentsOptions(Device device)
    {
        CpuOnly(device);
        string[] variables =
        [
            DocumentOptions.LayoutVariable, DocumentOptions.MaxStreamBytesVariable, DocumentOptions.CacheBytesVariable,
            DocumentWriteOptions.ImageEncodingVariable, DocumentWriteOptions.JpegQualityVariable, DocumentWriteOptions.EmbedFontsVariable, DocumentWriteOptions.SaveModeVariable,
            DocumentDrawOptions.DpiVariable, DocumentDrawOptions.FontsVariable, SignatureCheckOptions.OnlineVariable, FontWriteOptions.CurveToleranceVariable,
            SignatureCheckOptions.MinimumRsaKeyBitsVariable, SignatureCheckOptions.MaxChainLengthVariable, SignatureCheckOptions.InitialPoliciesVariable,
            SignatureCheckOptions.RequireExplicitPolicyVariable, SignatureCheckOptions.InhibitPolicyMappingVariable, SignatureCheckOptions.InhibitAnyPolicyVariable,
            SignatureCheckOptions.FreshnessGraceVariable, SignatureCheckOptions.MaxNameConstraintChecksVariable,
        ];
        Check(variables.All(v => v.StartsWith("IDRAK_DOCUMENT_", StringComparison.Ordinal) || v == "IDRAK_FONT_CURVE_TOLERANCE") && variables.Distinct().Count() == 19, "the nineteen variables");

        // Unset: the defaults, the size limits measured from the free memory.
        WithEnvironment(variables.ToDictionary(v => v, _ => (string?)null), () =>
        {
            var options = new DocumentOptions();
            long total = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
            Check(options.Layout == DocumentTextLayouts.Lines && options.IncludeHidden && options.Direction == DocumentTextDirection.Auto
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
            Check(new SignatureCheckOptions().MinimumRsaKeyBits == 2048 && new SignatureCheckOptions().MaxChainLength == 12, "a check takes RSA keys of 2048 bits or more, paths of 12 certificates or fewer");
            var trust = new SignatureCheckOptions();
            Check(trust.InitialPolicies.Count == 0 && !trust.RequireExplicitPolicy && !trust.InhibitPolicyMapping && !trust.InhibitAnyPolicy && trust.FreshnessGrace == TimeSpan.Zero
                  && trust.MaxNameConstraintChecks == 1 << 20, $"RFC 5280's policy defaults (any policy, every flag off), no grace, 1,048,576 name comparisons: {trust}");
        });

        // Set: each variable read.
        string fonts = Path.Combine(Path.GetTempPath(), "fonts for documents");
        WithEnvironment(new Dictionary<string, string?>
        {
            [DocumentOptions.LayoutVariable] = "content", [DocumentOptions.MaxStreamBytesVariable] = "256m",
            [DocumentOptions.CacheBytesVariable] = "1GiB", [DocumentWriteOptions.ImageEncodingVariable] = "Jpeg", [DocumentWriteOptions.JpegQualityVariable] = "75",
            [DocumentWriteOptions.EmbedFontsVariable] = "whole", [DocumentWriteOptions.SaveModeVariable] = "rewrite", [DocumentDrawOptions.DpiVariable] = "300",
            [DocumentDrawOptions.FontsVariable] = fonts, [SignatureCheckOptions.OnlineVariable] = "on", [FontWriteOptions.CurveToleranceVariable] = "0.25",
            [SignatureCheckOptions.MinimumRsaKeyBitsVariable] = "1024", [SignatureCheckOptions.MaxChainLengthVariable] = "20",
            [SignatureCheckOptions.InitialPoliciesVariable] = "2.999.1.2, 2.999.1.3", [SignatureCheckOptions.RequireExplicitPolicyVariable] = "on",
            [SignatureCheckOptions.InhibitPolicyMappingVariable] = "yes", [SignatureCheckOptions.InhibitAnyPolicyVariable] = "1",
            [SignatureCheckOptions.FreshnessGraceVariable] = "12h", [SignatureCheckOptions.MaxNameConstraintChecksVariable] = "4096",
        }, () =>
        {
            var options = new DocumentOptions();
            Check(options.Layout == "content" && options.MaxStreamBytes == 256L << 20 && options.CacheBytes == 1L << 30, $"read: {options}");
            var write = new DocumentWriteOptions();
            Check(write.ImageEncoding == DocumentImageEncoding.Jpeg && write.JpegQuality == 75 && write.EmbedFonts == DocumentFontEmbedding.Whole && write.SaveMode == DocumentSaveMode.Rewrite,
                $"read: {write}");
            var draw = new DocumentDrawOptions();
            Check(MathF.Abs(draw.Dpi - 300f) < 1e-3f && draw.Fonts == fonts && new SignatureCheckOptions().Online && new FontWriteOptions().CurveTolerance == 0.25f, $"read: {draw}");
            var signatures = new SignatureCheckOptions();
            Check(signatures.MinimumRsaKeyBits == 1024 && signatures.MaxChainLength == 20, $"read: {signatures}");
            Check(signatures.InitialPolicies.SequenceEqual(["2.999.1.2", "2.999.1.3"]) && signatures.RequireExplicitPolicy && signatures.InhibitPolicyMapping && signatures.InhibitAnyPolicy
                  && signatures.FreshnessGrace == TimeSpan.FromHours(12) && signatures.MaxNameConstraintChecks == 4096, $"read: {signatures}");
        });
        WithEnvironment(new Dictionary<string, string?> { [SignatureCheckOptions.InitialPoliciesVariable] = "any", [SignatureCheckOptions.FreshnessGraceVariable] = "90" }, () =>
        {
            var signatures = new SignatureCheckOptions();
            Check(signatures.InitialPolicies.Count == 0 && signatures.FreshnessGrace == TimeSpan.FromSeconds(90), $"any policy, a grace in seconds: {signatures}");
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
                     (SignatureCheckOptions.MinimumRsaKeyBitsVariable, "-1", () => new SignatureCheckOptions()),
                     (SignatureCheckOptions.MaxChainLengthVariable, "0", () => new SignatureCheckOptions()),
                     (SignatureCheckOptions.InitialPoliciesVariable, "policy-a", () => new SignatureCheckOptions()),
                     (SignatureCheckOptions.RequireExplicitPolicyVariable, "sometimes", () => new SignatureCheckOptions()),
                     (SignatureCheckOptions.InhibitPolicyMappingVariable, "2", () => new SignatureCheckOptions()),
                     (SignatureCheckOptions.InhibitAnyPolicyVariable, "maybe", () => new SignatureCheckOptions()),
                     (SignatureCheckOptions.FreshnessGraceVariable, "-1h", () => new SignatureCheckOptions()),
                     (SignatureCheckOptions.MaxNameConstraintChecksVariable, "0", () => new SignatureCheckOptions()),
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
                     () => new FontWriteOptions { CurveTolerance = 0f }, () => new SignatureCheckOptions { MinimumRsaKeyBits = -1 },
                     () => new SignatureCheckOptions { MaxChainLength = 0 }, () => new SignatureCheckOptions { MaxChainLength = 65 },
                     () => new SignatureCheckOptions { FreshnessGrace = TimeSpan.FromSeconds(-1) }, () => new SignatureCheckOptions { FreshnessGrace = TimeSpan.FromDays(3651) },
                     () => new SignatureCheckOptions { MaxNameConstraintChecks = 0 },
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

        try
        {
            _ = new SignatureCheckOptions { InitialPolicies = ["2.999.1.2", "not an identifier"] };
            Check(false, "a policy that is not an object identifier was taken");
        }
        catch (ArgumentException e)
        {
            Check(e.Message.Contains("not an identifier", StringComparison.Ordinal), e.Message);
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

    private static DocumentTextRun Placed(string text, float x, float baseline, float size, int order, float? width = null, bool logical = false) =>
        new(text, new DocumentBox(x, baseline - size * 0.8f, width ?? text.Length * size * 0.5f, size), size) { Baseline = baseline, Order = order, Logical = logical };

    private static void LayoutsLinesLeftToRight(Device device)
    {
        CpuOnly(device);
        var options = new DocumentOptions();
        var layout = new LinesLayout(new BidiReadingOrder());
        DocumentTextRun[] runs =
        [
            Placed("world", 100f, 50f, 10f, 0, 25f),
            Placed("Hello", 50f, 50f, 10f, 1, 25f),
            Placed("Sec", 50f, 70f, 10f, 2, 15f),
            Placed("ond", 65.5f, 70f, 10f, 3, 15f),
            Placed("line", 100f, 70f, 10f, 4, 20f),
            Placed("2", 125.5f, 46.5f, 6f, 5, 3f),
        ];
        var text = layout.Arrange(runs, 612f, 792f, options);
        Check(text.Text == "Hello world2\nSecond line", $"the page's text: \"{text.Text}\"");
        Check(text.Lines.Count == 2 && text.Lines[0].Runs.SequenceEqual([1, 0, 5]) && text.Lines[1].Runs.SequenceEqual([2, 3, 4]) && !text.Lines[0].RightToLeft
              && text.Lines[0].Box.X == 50f && MathF.Abs(text.Lines[0].Box.Right - 128.5f) < 1e-4f, $"the lines: {string.Join(" | ", text.Lines)}");
        var wide = layout.Arrange(runs, 612f, 792f, options with { WordGap = 5f });
        Check(wide.Text == "Helloworld2\nSecondline", $"with a wider word gap: \"{wide.Text}\"");
        var tight = layout.Arrange(runs, 612f, 792f, options with { LineTolerance = 0.1f });
        Check(tight.Lines.Count == 3 && tight.Text.StartsWith("2\n", StringComparison.Ordinal), $"a tighter line tolerance puts the superscript on its own line: \"{tight.Text}\"");
        Check(layout.Arrange([], 612f, 792f, options) == DocumentText.Empty && layout.Arrange([Placed("", 1f, 1f, 1f, 0)], 612f, 792f, options).Lines.Count == 0,
            "no runs, no text");
    }

    private static void LayoutsLinesBothDirections(Device device)
    {
        CpuOnly(device);
        var options = new DocumentOptions();
        var layout = new LinesLayout(new BidiReadingOrder());

        // "قرأت 3 كتب" (I read 3 books) as a page shows it: the words' glyphs left to right, so each word's text reversed.
        DocumentTextRun[] visual =
        [
            Placed(LinesLayout.Reversed("قرأت"), 200f, 50f, 10f, 0, 40f),
            Placed("3", 180f, 50f, 10f, 1, 10f),
            Placed(LinesLayout.Reversed("كتب"), 140f, 50f, 10f, 2, 30f),
        ];
        var line = layout.Arrange(visual, 612f, 792f, options).Lines.Single();
        Check(line.Text == "قرأت 3 كتب" && line.RightToLeft && line.Runs.SequenceEqual([0, 1, 2]), $"a right-to-left line: \"{line.Text}\", right to left {line.RightToLeft}");

        DocumentTextRun[] logical =
        [
            Placed("قرأت", 200f, 50f, 10f, 0, 40f, logical: true),
            Placed("3", 180f, 50f, 10f, 1, 10f, logical: true),
            Placed("كتب", 140f, 50f, 10f, 2, 30f, logical: true),
        ];
        Check(layout.Arrange(logical, 612f, 792f, options).Text == "قرأت 3 كتب", "runs in logical order give the same text");

        // "افتح Idrak الآن" (open Idrak now): a left-to-right word in a right-to-left line.
        DocumentTextRun[] mixed =
        [
            Placed(LinesLayout.Reversed("افتح"), 300f, 50f, 10f, 0, 40f),
            Placed("Idrak", 240f, 50f, 10f, 1, 50f),
            Placed(LinesLayout.Reversed("الآن"), 190f, 50f, 10f, 2, 40f),
        ];
        string mixedText = layout.Arrange(mixed, 612f, 792f, options).Text;
        Check(mixedText == "افتح Idrak الآن", $"a left-to-right word in a right-to-left line: \"{mixedText}\"");

        // A right-to-left word in a left-to-right line: "Send the كتب today".
        DocumentTextRun[] english =
        [
            Placed("Send", 50f, 50f, 10f, 0, 20f),
            Placed("the", 75f, 50f, 10f, 1, 15f),
            Placed(LinesLayout.Reversed("كتب"), 95f, 50f, 10f, 2, 15f),
            Placed("today", 115f, 50f, 10f, 3, 25f),
        ];
        var left = layout.Arrange(english, 612f, 792f, options).Lines.Single();
        Check(left.Text == "Send the كتب today" && !left.RightToLeft, $"a right-to-left word in a left-to-right line: \"{left.Text}\"");
    }

    private static void LayoutsLinesDirection(Device device)
    {
        CpuOnly(device);
        var options = new DocumentOptions();
        var layout = new LinesLayout(new BidiReadingOrder());

        // A line of digits alone takes the page's direction: right to left under an Arabic line, left to right under a Latin one.
        DocumentTextRun[] arabic = [Placed(LinesLayout.Reversed("كتب"), 100f, 50f, 10f, 0, 30f), Placed("12", 160f, 70f, 10f, 1, 10f), Placed("34", 140f, 70f, 10f, 2, 10f)];
        var lines = layout.Arrange(arabic, 612f, 792f, options).Lines;
        Check(lines.Count == 2 && lines[0].RightToLeft && lines[1].RightToLeft && lines[1].Text == "12 34", $"digits under Arabic: {string.Join(" | ", lines)}");
        DocumentTextRun[] latin = [Placed("Total", 100f, 50f, 10f, 0, 25f), Placed("12", 140f, 70f, 10f, 1, 10f), Placed("34", 160f, 70f, 10f, 2, 10f)];
        lines = layout.Arrange(latin, 612f, 792f, options).Lines;
        Check(lines.Count == 2 && !lines[1].RightToLeft && lines[1].Text == "12 34", $"digits under Latin: {string.Join(" | ", lines)}");

        // The option sets every line.
        DocumentTextRun[] words = [Placed("Hello", 50f, 50f, 10f, 0, 25f), Placed("world", 100f, 50f, 10f, 1, 25f)];
        var forced = layout.Arrange(words, 612f, 792f, options with { Direction = DocumentTextDirection.RightToLeft }).Lines.Single();
        Check(forced.RightToLeft && forced.Text == "Hello world" && forced.Runs.SequenceEqual([1, 0]), $"right to left by the option: {forced}");

        // Runs without a place: lines of their own, in content order, after the placed ones.
        DocumentTextRun[] notes = [new("note two", DocumentBox.Empty, 10f) { Order = 5 }, Placed("Title", 50f, 50f, 10f, 0), new("note one", DocumentBox.Empty, 10f) { Order = 3 }];
        string text = layout.Arrange(notes, 612f, 792f, options).Text;
        Check(text == "Title\nnote one\nnote two", $"runs without a place: \"{text}\"");
    }

    private static void LayoutsLinesAllocation(Device device)
    {
        CpuOnly(device);
        var options = new DocumentOptions();
        var layout = new LinesLayout(new BidiReadingOrder());

        // A report's heading in Arabic words alone ("التقرير السنوي", the annual report, its glyphs left to right), two
        // lines of Latin words and numbers, and a line of one run: none that the reading order reorders.
        DocumentTextRun[] runs =
        [
            Placed(LinesLayout.Reversed("السنوي"), 120f, 30f, 12f, 0, 40f),
            Placed(LinesLayout.Reversed("التقرير"), 170f, 30f, 12f, 1, 45f),
            Placed("The", 50f, 50f, 10f, 2, 15f),
            Placed("quarterly", 70f, 50f, 10f, 3, 45f),
            Placed("report", 120f, 50f, 10f, 4, 30f),
            Placed("is", 155f, 50f, 10f, 5, 10f),
            Placed("ready.", 170f, 50f, 10f, 6, 30f),
            Placed("Sales", 50f, 70f, 10f, 7, 25f),
            Placed("rose", 80f, 70f, 10f, 8, 20f),
            Placed("12%", 105f, 70f, 10f, 9, 15f),
            Placed("Contents", 50f, 90f, 10f, 10, 40f),
        ];
        var text = layout.Arrange(runs, 612f, 792f, options);
        Check(text.Text == "التقرير السنوي\nThe quarterly report is ready.\nSales rose 12%\nContents" && text.Lines[0].RightToLeft && !text.Lines[1].RightToLeft,
            $"the page's text: \"{text.Text}\"");

        // What it gives back and asks, made by hand: the same objects of the same lengths.
        int[] lengths = [.. text.Lines.Select(l => l.Text.Length)];
        int[] counts = [.. text.Lines.Select(l => l.Runs.Count)];
        int[] asked = [.. text.Lines.Select(l => l.Runs.Count > 1 ? l.Runs.Sum(r => runs[r].Text.Length) : 0)];
        int page = text.Text.Length;
        object Given()
        {
            var lines = new DocumentLine[lengths.Length];
            for (int i = 0; i < lines.Length; i++)
            {
                string question = asked[i] > 0 ? new string('x', asked[i]) : "";
                lines[i] = new DocumentLine(new string('x', lengths[i]), DocumentBox.Empty, question.Length > 0, new int[counts[i]]);
            }

            return new DocumentText(lines, new string('x', page));
        }

        object Arranged() => layout.Arrange(runs, 612f, 792f, options);
        static long Allocated(Func<object> call)
        {
            for (int i = 0; i < 300; i++)
            {
                call();
            }

            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 1_000; i++)
            {
                call();
            }

            return GC.GetAllocatedBytesForCurrentThread() - before;
        }

        long given = Allocated(Given), arranged = Allocated(Arranged);
        Check(arranged <= given, $"{runs.Length} runs in {text.Lines.Count} lines arranged 1,000 times allocated {arranged} bytes once warm, {given} for what it gives back and asks"
            + (arranged <= given ? "" : $" ({AllocatedTypes(() => Arranged())})"));
    }

    private static void LayoutsContentAndNames(Device device)
    {
        CpuOnly(device);
        DocumentTextRun[] runs = [Placed("c", 10f, 50f, 10f, 2), Placed("a", 30f, 90f, 10f, 0), Placed("b", 50f, 10f, 10f, 1)];
        var content = new ContentLayout().Arrange(runs, 612f, 792f, new DocumentOptions());
        Check(content.Text == "abc" && content.Lines.Count == 3 && content.Lines[0].Runs.Single() == 1 && content.Lines.All(l => !l.RightToLeft), $"content order: \"{content.Text}\"");
        Check(new LinesLayout(new RunsReadingOrder()).Arrange(runs, 612f, 792f, new DocumentOptions()).Text == "b\nc\na", "the runs reading order");

        WithEnvironment(new Dictionary<string, string?> { [DocumentOptions.LayoutVariable] = DocumentTextLayouts.Content }, () =>
        {
            var options = new DocumentOptions();
            Check(options.Layout == DocumentTextLayouts.Content, "the layout from its variable");
        });

        Check(new BidiReadingOrder().RightToLeft("123 كتب abc") == true && new BidiReadingOrder().RightToLeft("(12) abc كتب") == false
              && new BidiReadingOrder().RightToLeft("12, 34") is null && new RunsReadingOrder().RightToLeft("שלום world") == true
              && new BidiReadingOrder().RightToLeft("⁧abc⁩ كتب") == true, "the first strong character, an isolate's skipped");
    }
}

