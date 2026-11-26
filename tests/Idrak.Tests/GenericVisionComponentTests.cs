// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Abstraction.Testing;
using Idrak.Vision.Plugins.Gemma;
using Idrak.Vision.Plugins.IdrakVision;
using Idrak.Imaging;
using Idrak.Imaging.Shaping;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Text.Abstractions;
using Idrak.Vision;
using Idrak.Vision.Abstractions;

// Plan 26, phase P3: the generic components of the vision samples moved into the library (G8-G14), each proven against
// what it replaced: the image-block attention rule (IdrakVision's and Gemma 3's copies), pixel noise (the Arabic OCR
// app's), line segmentation, text matching and the reading order (the Arabic OCR app's), and the line preparation of the
// line-recognition task.
internal abstract partial class DesktopTests : Tests
{
    internal static readonly (string Name, Action<Device> Run)[] GenericVisionComponentGroup =
    [
        ("generic components: image blocks: the library's image-blocks rule sees each image whole and keeps the window (KeySpans.ImageBlocks); IdrakVision and Gemma 3 name it", GenericImageBlocks),
        ("generic components: noise: PixelNoise (std, p) gives the Arabic OCR app's pixels for the same seed, clamped to [0, 1], keeps the objects, does nothing at p=0 and passes the augmentation suite", GenericNoise),
        ("generic components: line segmentation: the projection segmenter passes the line segmenter suite; its options (no deskew, padding) apply and are checked", GenericLineSegmentation),
        ("generic components: text matching: the semi-global matcher passes the text matcher suite (with and without word boundaries); Align snaps the Arabic OCR app's drafts as it did, MaxError applies", GenericTextMatching),
        ("generic components: reading order: the runs (its own inverse) and bidi (UAX #9) orders pass the reading order suite; both give the Arabic OCR app's columns for its line; bidi maps a number before or after a Latin word to the same columns and orders Arabic-Indic digits after Latin apart, so the app keeps runs; a line neither reorders comes back as the same string", GenericReadingOrder),
        ("generic components: line recognition task: ModelLineReader prepares a line as the OCR app did (ink high on 0, stretched, scaled to the height, flipped right to left)", GenericLineRecognitionTask),
    ];

    // ------------------------------------------------------------------ G8: the image-block rule

    private static void GenericImageBlocks(Device device)
    {
        _ = device;
        var rule = new ImageBlocksRule();
        Check(rule.Name == "image-blocks" && !rule.Causal, "the library's image-blocks rule");
        (int Start, int Length)[] blocks = [(2, 3), (7, 2)];
        foreach (int window in new[] { 0, 3, 12 })
        {
            var spans = rule.Spans(12, blocks, window);
            var expected = KeySpans.ImageBlocks(12, blocks, window);
            Check(spans.Starts.SequenceEqual(expected.Starts) && spans.Ends.SequenceEqual(expected.Ends), $"window {window}: the spans of KeySpans.ImageBlocks");
        }

        var windowed = rule.Spans(12, blocks, 3);
        Check(windowed.Ends[2] == 5 && windowed.Ends[7] == 9 && windowed.Starts[4] == 2 && windowed.Ends[10] == 11, "a row of a block sees its whole block; the window bounds the start");

        // The plug-ins name the library's rule.
        Check(Gemma3VisionPlugin.ImageBlocks == ImageAttentionRules.ImageBlocks && IdrakVisionPlugin.ImageBlocks == ImageAttentionRules.ImageBlocks, "the plug-ins' rule names");
    }

    // ------------------------------------------------------------------ G9: pixel noise

    // The Arabic OCR app's LineNoise, as it was (the reference the library's noise must match pixel for pixel).
    private static float[] AppLineNoise(ImageData image, double standardDeviation, double probability, Random random)
    {
        if (random.NextDouble() >= probability)
        {
            return image.Pixels;
        }

        var pixels = (float[])image.Pixels.Clone();
        for (int i = 0; i < pixels.Length; i++)
        {
            double u = 1 - random.NextDouble(), v = random.NextDouble();
            double normal = Math.Sqrt(-2 * Math.Log(u)) * Math.Cos(2 * Math.PI * v);
            pixels[i] = Math.Clamp(pixels[i] + (float)(normal * standardDeviation), 0f, 1f);
        }

        return pixels;
    }

    private static void GenericNoise(Device device)
    {
        _ = device;
        var sample = AugmentationSuite.Synthetic(new Random(1), 24, 18, 3, 2, masks: false, pixelClasses: false);
        IAugmentation noise = new PixelNoise(0.03);
        Check(noise is PixelNoise { StandardDeviation: 0.03, Probability: 1 } && noise.Name == "noise", $"noise(std=0.03): {noise}");
        Check(new PixelNoise() is { StandardDeviation: 0.05 }, "the default std is the app's (0.05)");
        foreach (int seed in new[] { 1, 7, 42 })
        {
            var expected = AppLineNoise(sample.Image, 0.03, 1, new Random(seed));
            var noisy = noise.Apply(sample, new AugmentationContext(new Random(seed)));
            Check(noisy.Image.Pixels.AsSpan().SequenceEqual(expected), $"seed {seed}: the app's pixels");
            Check(noisy.Boxes.SequenceEqual(sample.Boxes) && noisy.Labels.SequenceEqual(sample.Labels), "the objects kept");
        }

        var line = new ImageData([.. Enumerable.Range(0, 64).Select(i => (i % 2) * 0.9f)], 1, 4, 16);
        var app = AppLineNoise(line, 0.05, 1, new Random(3));
        Check(new PixelNoise().Apply(new AnnotatedImage(line), new AugmentationContext(new Random(3))).Image.Pixels.AsSpan().SequenceEqual(app), "a grey line as the app noised it");
        var loud = new PixelNoise(2).Apply(sample, new AugmentationContext(new Random(5)));
        Check(loud.Image.Pixels.All(v => v is >= 0f and <= 1f), "clamped to [0, 1]");
        Check(ReferenceEquals(new PixelNoise(0.1, 0).Apply(sample, new AugmentationContext(new Random(5))), sample), "p=0: the sample as it is");
        Conformance.Check(noise, new AugmentationSuite { ColorsKept = false }, new ContractCheckOptions { RandomCases = 8 }).ThrowIfFailed();
    }

    // ------------------------------------------------------------------ G11: line segmentation

    // Three lines of dark strokes on a light page, a dot above the second.
    private static ImageData StrokePage()
    {
        int width = 160, height = 110;
        var pixels = new float[width * height];
        Array.Fill(pixels, 0.95f);
        foreach (int top in new[] { 15, 48, 80 })
        {
            for (int x = 12; x < 148; x += 7)
            {
                for (int y = top; y < top + 12; y++)
                {
                    pixels[y * width + x] = pixels[y * width + x + 1] = pixels[y * width + x + 2] = 0.1f;
                }
            }
        }

        pixels[44 * width + 40] = pixels[44 * width + 41] = pixels[45 * width + 40] = pixels[45 * width + 41] = 0.1f;
        return new ImageData(pixels, 1, height, width);
    }

    private static SegmentedPage Segmented(PageLines lines) =>
        new([.. lines.Lines.Select(l => new SegmentedLine(l.X, l.Y, l.Width, l.Height, l.Image))], lines.Skew);

    private static void GenericLineSegmentation(Device device)
    {
        _ = device;
        var projection = new ProjectionLineSegmenter();
        Conformance.CheckTextLineSegmenter(page => Segmented(projection.Find(page, new LineSegmentationOptions()))).ThrowIfFailed();

        var page = StrokePage();
        var lines = TextLineSegmenters.Segment(page);
        Check(lines.Lines.Count == 3 && lines.Skew == 0 && lines.Lines.Select(l => l.Index).SequenceEqual([1, 2, 3]), $"three lines: {lines.Lines.Count}, skew {lines.Skew}");
        Check(lines.Lines[1].Y <= 44 && lines.Lines[1].Y + lines.Lines[1].Height >= 60, $"the dot joined to the second line: {lines.Lines[1].Y}..{lines.Lines[1].Y + lines.Lines[1].Height}");
        var tight = TextLineSegmenters.Segment(page, new LineSegmentationOptions { Padding = 0, MinPadding = 0, MaxSkew = 0 });
        Check(tight.Lines[0].Y == 15 && tight.Lines[0].Height == 12 && tight.Lines[0].X == 12 && tight.Lines[0].Width < lines.Lines[0].Width,
            $"no padding: the ink's box ({tight.Lines[0].X}, {tight.Lines[0].Y}) {tight.Lines[0].Width} x {tight.Lines[0].Height}");
        ThrowsNaming<ArgumentOutOfRangeException>(() => TextLineSegmenters.Segment(page, new LineSegmentationOptions { SkewStep = 0 }), "a zero skew step", ["SkewStep"]);
    }

    // ------------------------------------------------------------------ G12: text matching

    private static TextMatchSearch Search(ITextMatcher matcher, TextMatchOptions? options = null) =>
        (draft, text, from) => matcher.Best(draft, text, from, options ?? new TextMatchOptions()) is { } span ? (span.Start, span.Length, span.Distance) : null;

    private static void GenericTextMatching(Device device)
    {
        _ = device;
        var library = new Idrak.Text.SemiGlobalMatcher();
        Conformance.CheckTextMatcher(Search(library)).ThrowIfFailed();
        var free = new TextMatchOptions { WordBoundaries = false };
        Conformance.CheckTextMatcher(Search(library, free), wordBoundaries: false).ThrowIfFailed();

        // The OCR app's drafts snapped to its page's text, as its alignment did.
        string page = "the first line here then a second line and the third one";
        var aligned = TextMatchers.Align(["the frst line hre", "zzzzzzzzzzzzzzzzzzzz", "a secnd line", "the thrd one"], page);
        Check(aligned[0]?.Text.Trim() == "the first line here" && aligned[1] is null && aligned[2]?.Text.Trim() == "a second line" && aligned[3]?.Text.Trim() == "the third one",
            $"aligned: {string.Join(" | ", aligned.Select(a => a?.Text ?? "(none)"))}");
        Check(aligned[0]!.Match is > 0.8 and < 1 && aligned[0]!.Span.Distance == 2 && aligned[2]!.Span.Start >= aligned[0]!.Span.Start + aligned[0]!.Span.Length,
            $"match {aligned[0]!.Match}, distance {aligned[0]!.Span.Distance}; in order");
        var exact = TextMatchers.Align(["the first line here", "a secnd line"], page, new TextMatchOptions { MaxError = 0 });
        Check(exact[0] is { Match: 1 } && exact[1] is null, "MaxError 0: only exact drafts align");
        var span = TextMatchers.Best("secnd", page, 0, free)!.Value;
        Check(page.Substring(span.Start, span.Length) == "second" && span.Distance == 1, $"without word boundaries: '{page.Substring(span.Start, span.Length)}'");
        Check(TextMatchers.Best("", page) is null && TextMatchers.Best("line", page, page.Length) is null, "nothing to match");
    }

    // ------------------------------------------------------------------ G13: reading order

    private static void GenericReadingOrder(Device device)
    {
        _ = device;
        IReadingOrder runs = new RunsReadingOrder(), bidi = new BidiReadingOrder();
        Conformance.Check(runs, new ReadingOrderSuite(invertible: true)).ThrowIfFailed();
        Conformance.Check(bidi, new ReadingOrderSuite()).ThrowIfFailed();

        // The OCR app's test line: both give its columns; runs undoes itself.
        string logical = "با 123 س abc de م ١٢", columns = "با 321 س ed cba م ٢١";
        Check(runs.ToColumns(logical, true) == columns && bidi.ToColumns(logical, true) == columns, $"runs '{runs.ToColumns(logical, true)}', bidi '{bidi.ToColumns(logical, true)}'");
        Check(runs.FromColumns(columns, true) == logical && bidi.FromColumns(columns, true) == logical, "both read the line back");
        Check(runs.ToColumns("the river", false) == "the river" && bidi.ToColumns("the river", false) == "the river", "a left-to-right line of Latin as it is");
        Check(runs.ToColumns("the باب سم end", false) == "the مس باب end", "runs: Arabic reversed in a left-to-right line");

        // A line neither order reorders (Latin and European digits left to right, Arabic words alone right to left) comes back
        // as the same string, nothing allocated.
        string latin = "the river 12%", arabic = "باب سم";
        foreach (var order in new[] { runs, bidi })
        {
            Check(ReferenceEquals(order.ToColumns(latin, false), latin) && ReferenceEquals(order.FromColumns(latin, false), latin)
                  && ReferenceEquals(order.ToColumns(arabic, true), arabic) && ReferenceEquals(order.FromColumns(arabic, true), arabic), $"{order.Name}: an unchanged line is not given back as it is");
        }

        // Where UAX #9 differs from the app's order: a number after a Latin word reads back before it (both texts give the
        // same columns), and Arabic-Indic digits after Latin letters are a run of their own.
        string latinNumber = "س abc 123", latinColumns = "س 321 cba";
        Check(runs.ToColumns(latinNumber, true) == latinColumns && bidi.ToColumns(latinNumber, true) == latinColumns, "a Latin word and a number: the same columns");
        Check(runs.FromColumns(latinColumns, true) == latinNumber && bidi.FromColumns(latinColumns, true) == "س 123 abc"
              && bidi.ToColumns("س 123 abc", true) == latinColumns, $"bidi reads them back as '{bidi.FromColumns(latinColumns, true)}'");
        Check(runs.ToColumns("abc ١٢", true) == "٢١ cba" && bidi.ToColumns("abc ١٢", true) == "cba ٢١",
            $"Arabic-Indic digits after Latin: runs '{runs.ToColumns("abc ١٢", true)}', bidi '{bidi.ToColumns("abc ١٢", true)}'");
    }

    // ------------------------------------------------------------------ G14: the line-recognition task

    private static void GenericLineRecognitionTask(Device device)
    {
        _ = device;

        // Prepared as the OCR app prepared its lines: ink high on 0, stretched, scaled to the height, flipped for rtl.
        var image = LineImage("ab");
        var prepared = ModelLineReader.Prepare(image, new LineRecognitionSettings { Height = 8, Stride = 4 });
        Check(prepared.Height == 8 && prepared.Width == Math.Max(4, (int)Math.Round(image.Width * 8.0 / image.Height)) && prepared.Pixels.Max() > 0.9f && prepared.Pixels.Min() == 0f,
            $"prepared {prepared.Width} wide, values {prepared.Pixels.Min()}..{prepared.Pixels.Max()}");
        var flipped = ModelLineReader.Prepare(image, new LineRecognitionSettings { Height = 8, Stride = 4, RightToLeft = true });
        Check(Enumerable.Range(0, prepared.Width).All(x => flipped.Pixels[3 * prepared.Width + x] == prepared.Pixels[3 * prepared.Width + prepared.Width - 1 - x]), "right to left: flipped");
    }
}
