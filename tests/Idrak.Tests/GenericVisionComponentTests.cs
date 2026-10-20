// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Abstraction.Testing;
using Idrak.Gemma3Vision;
using Idrak.IdrakVision;
using Idrak.Imaging;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Text.Abstractions;
using Idrak.Vision;
using Idrak.Vision.Abstractions;

// Plan 26, phase P3: the generic components of the vision samples moved into the library (G8-G14), each proven against
// what it replaced: the image-block attention rule (IdrakVision's and Gemma 3's copies), pixel noise (the Arabic OCR
// app's), line segmentation, text matching and the reading order (the Arabic OCR app's), and the line-recognition task of
// the image model families (its predictor, its suite, idrak predict and idrak train).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] GenericVisionComponentGroup =
    [
        ("generic components: image blocks: the library's image-blocks rule beside causal sees each image whole and keeps the window (KeySpans.ImageBlocks); IdrakVision and Gemma 3 use it and register no rule of their own; an app's rule over it falls back under FallBack", GenericImageBlocks),
        ("generic components: noise: Augmentations' noise (std, p) gives the Arabic OCR app's pixels for the same seed, clamped to [0, 1], keeps the objects, does nothing at p=0, refuses unknown options and passes the augmentation suite", GenericNoise),
        ("generic components: line segmentation: TextLineSegmenters' projection passes the line segmenter suite; its options (no deskew, padding) apply and are checked; an app's segmenter falls back on it; an unknown name is refused naming the registry", GenericLineSegmentation),
        ("generic components: text matching: TextMatchers' semi-global passes the text matcher suite (with and without word boundaries); Align snaps the Arabic OCR app's drafts as it did, MaxError applies; an app's matcher falls back on it; an unknown name is refused", GenericTextMatching),
        ("generic components: reading order: ReadingOrders' runs (its own inverse) and bidi (UAX #9) pass the reading order suite; both give the Arabic OCR app's columns for its line; bidi maps a number before or after a Latin word to the same columns and orders Arabic-Indic digits after Latin apart, so the app keeps runs", GenericReadingOrder),
        ("generic components: line recognition task: a line-recognition family loads with its line settings and labels, reads lines through ModelLineReader (prepared as the OCR app did, alone as in a batch, text and labels round trip with a reading order), passes the image model family suite, and a description without line settings, labels or a blank in range is refused", GenericLineRecognitionTask),
        ("generic components: line recognition cli: idrak train --task line-recognition trains a builder network on line images with .txt beside them (CTC, cer and wer) and fine-tunes a family; idrak predict reads lines with a package or a family and --weights; wrong options and a network of the wrong width are refused", GenericLineRecognitionCli),
    ];

    // ------------------------------------------------------------------ G8: the image-block rule

    private sealed class FailingBlocksRule : IImageAttentionRule
    {
        public string Name => ImageAttentionRules.ImageBlocks;

        public bool Causal => false;

        public KeySpans Spans(int rows, IReadOnlyList<(int Start, int Length)> blocks, int window) => throw new InvalidOperationException("broken on purpose");
    }

    private static void GenericImageBlocks(Device device)
    {
        _ = device;
        var rule = ImageAttentionRules.Get(ImageAttentionRules.ImageBlocks);
        Check(ImageAttentionRules.Origin(ImageAttentionRules.ImageBlocks) == Overrides.Library && rule.Name == "image-blocks" && !rule.Causal
              && ImageAttentionRules.Names.Take(2).SequenceEqual([ImageAttentionRules.Causal, ImageAttentionRules.ImageBlocks]), "the library registers image-blocks beside causal");
        (int Start, int Length)[] blocks = [(2, 3), (7, 2)];
        foreach (int window in new[] { 0, 3, 12 })
        {
            var spans = rule.Spans(12, blocks, window);
            var expected = KeySpans.ImageBlocks(12, blocks, window);
            Check(spans.Starts.SequenceEqual(expected.Starts) && spans.Ends.SequenceEqual(expected.Ends), $"window {window}: the spans of KeySpans.ImageBlocks");
        }

        var windowed = rule.Spans(12, blocks, 3);
        Check(windowed.Ends[2] == 5 && windowed.Ends[7] == 9 && windowed.Starts[4] == 2 && windowed.Ends[10] == 11, "a row of a block sees its whole block; the window bounds the start");

        // The plug-ins name the library's rule and register none of their own.
        Check(Gemma3VisionPlugin.ImageBlocks == ImageAttentionRules.ImageBlocks && IdrakVisionPlugin.ImageBlocks == ImageAttentionRules.ImageBlocks, "the plug-ins' rule names");
        string[] before = [.. ImageAttentionRules.Names];
        Gemma3VisionPlugin.Register();
        IdrakVisionPlugin.Register();
        try
        {
            Check(ImageAttentionRules.Names.SequenceEqual(before) && ImageAttentionRules.Origin(ImageAttentionRules.ImageBlocks) == Overrides.Library,
                $"registering the plug-ins adds no rule: {string.Join(", ", ImageAttentionRules.Names)}");
        }
        finally
        {
            IdrakVisionPlugin.Unregister();
        }

        // An app's rule over the library's: under FallBack a failing call is answered by the library's.
        ImageAttentionRules.Register(new FailingBlocksRule());
        ImageAttentionRules.SetPolicy(ImageAttentionRules.ImageBlocks, SlotPolicy.FallBack);
        try
        {
            Check(ImageAttentionRules.Origin(ImageAttentionRules.ImageBlocks) != Overrides.Library, "the app's rule takes the name");
            var spans = ImageAttentionRules.Get(ImageAttentionRules.ImageBlocks).Spans(12, blocks, 0);
            Check(spans.Ends.SequenceEqual(KeySpans.ImageBlocks(12, blocks).Ends), "a failing app rule falls back on the library's");
        }
        finally
        {
            ImageAttentionRules.Unregister(ImageAttentionRules.ImageBlocks);
            ImageAttentionRules.SetPolicy(ImageAttentionRules.ImageBlocks, SlotPolicy.Throw);
        }

        Check(ImageAttentionRules.Origin(ImageAttentionRules.ImageBlocks) == Overrides.Library, "unregistered: the library's again");
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
        var noise = Augmentations.Parse("noise(std=0.03)").Single();
        Check(noise is PixelNoise { StandardDeviation: 0.03, Probability: 1 } && noise.Name == "noise", $"noise(std=0.03): {noise}");
        Check(Augmentations.Parse("noise").Single() is PixelNoise { StandardDeviation: 0.05 }, "the default std is the app's (0.05)");
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
        ThrowsNaming<ArgumentException>(() => Augmentations.Parse("noise(sigma=1)"), "an unknown option", ["sigma"]);
        Conformance.Check(noise, new AugmentationSuite { ColorsKept = false }, new ContractCheckOptions { RandomCases = 8 }).ThrowIfFailed();
    }

    // ------------------------------------------------------------------ G11: line segmentation

    private sealed class FailingSegmenter : ITextLineSegmenter
    {
        public string Name => TextLineSegmenters.Projection;

        public PageLines Find(ImageData page, LineSegmentationOptions options) => throw new InvalidOperationException("broken on purpose");
    }

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
        Check(TextLineSegmenters.Names.Contains(TextLineSegmenters.Projection) && TextLineSegmenters.Origin(TextLineSegmenters.Projection) == Overrides.Library
              && TextLineSegmenters.Default(TextLineSegmenters.Projection) is ProjectionLineSegmenter, "the library registers projection");
        var projection = TextLineSegmenters.Get(TextLineSegmenters.Projection);
        var library = TextLineSegmenters.Default(TextLineSegmenters.Projection)!;
        Conformance.CheckTextLineSegmenter(page => Segmented(projection.Find(page, new LineSegmentationOptions())),
            page => Segmented(library.Find(page, new LineSegmentationOptions()))).ThrowIfFailed();

        var page = StrokePage();
        var lines = TextLineSegmenters.Segment(page);
        Check(lines.Lines.Count == 3 && lines.Skew == 0 && lines.Lines.Select(l => l.Index).SequenceEqual([1, 2, 3]), $"three lines: {lines.Lines.Count}, skew {lines.Skew}");
        Check(lines.Lines[1].Y <= 44 && lines.Lines[1].Y + lines.Lines[1].Height >= 60, $"the dot joined to the second line: {lines.Lines[1].Y}..{lines.Lines[1].Y + lines.Lines[1].Height}");
        var tight = TextLineSegmenters.Segment(page, new LineSegmentationOptions { Padding = 0, MinPadding = 0, MaxSkew = 0 });
        Check(tight.Lines[0].Y == 15 && tight.Lines[0].Height == 12 && tight.Lines[0].X == 12 && tight.Lines[0].Width < lines.Lines[0].Width,
            $"no padding: the ink's box ({tight.Lines[0].X}, {tight.Lines[0].Y}) {tight.Lines[0].Width} x {tight.Lines[0].Height}");
        ThrowsNaming<ArgumentOutOfRangeException>(() => TextLineSegmenters.Segment(page, new LineSegmentationOptions { SkewStep = 0 }), "a zero skew step", ["SkewStep"]);
        ThrowsNaming<NotSupportedException>(() => TextLineSegmenters.Get("nobody"), "an unknown segmenter", ["'nobody' is not registered (registered: projection", "TextLineSegmenters.Register"]);

        TextLineSegmenters.Register(new FailingSegmenter());
        TextLineSegmenters.SetPolicy(TextLineSegmenters.Projection, SlotPolicy.FallBack);
        try
        {
            Check(TextLineSegmenters.Origin(TextLineSegmenters.Projection) != Overrides.Library && TextLineSegmenters.Segment(page).Lines.Count == 3,
                "a failing app segmenter falls back on the library's");
        }
        finally
        {
            TextLineSegmenters.Unregister(TextLineSegmenters.Projection);
            TextLineSegmenters.SetPolicy(TextLineSegmenters.Projection, SlotPolicy.Throw);
        }
    }

    // ------------------------------------------------------------------ G12: text matching

    private sealed class FailingMatcher : ITextMatcher
    {
        public string Name => TextMatchers.SemiGlobal;

        public TextMatch? Best(string draft, string text, int from, TextMatchOptions options) => throw new InvalidOperationException("broken on purpose");
    }

    private static TextMatchSearch Search(ITextMatcher matcher, TextMatchOptions? options = null) =>
        (draft, text, from) => matcher.Best(draft, text, from, options ?? new TextMatchOptions()) is { } span ? (span.Start, span.Length, span.Distance) : null;

    private static void GenericTextMatching(Device device)
    {
        _ = device;
        Check(TextMatchers.Origin(TextMatchers.SemiGlobal) == Overrides.Library && TextMatchers.Default(TextMatchers.SemiGlobal) is Idrak.Text.SemiGlobalMatcher, "the library registers semi-global");
        var library = TextMatchers.Default(TextMatchers.SemiGlobal)!;
        Conformance.CheckTextMatcher(Search(TextMatchers.Get(TextMatchers.SemiGlobal)), Search(library)).ThrowIfFailed();
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
        ThrowsNaming<NotSupportedException>(() => TextMatchers.Get("nobody"), "an unknown matcher", ["'nobody' is not registered (registered: semi-global", "TextMatchers.Register"]);

        TextMatchers.Register(new FailingMatcher());
        TextMatchers.SetPolicy(TextMatchers.SemiGlobal, SlotPolicy.FallBack);
        try
        {
            Check(TextMatchers.Origin(TextMatchers.SemiGlobal) != Overrides.Library && TextMatchers.Best("a secnd line", page)?.Distance == 1, "a failing app matcher falls back on the library's");
        }
        finally
        {
            TextMatchers.Unregister(TextMatchers.SemiGlobal);
            TextMatchers.SetPolicy(TextMatchers.SemiGlobal, SlotPolicy.Throw);
        }
    }

    // ------------------------------------------------------------------ G13: reading order

    private static void GenericReadingOrder(Device device)
    {
        _ = device;
        Check(ReadingOrders.Names.Contains(ReadingOrders.Bidi) && ReadingOrders.Names.Contains(ReadingOrders.Runs)
              && ReadingOrders.Origin(ReadingOrders.Bidi) == Overrides.Library && ReadingOrders.Origin(ReadingOrders.Runs) == Overrides.Library, "the library registers bidi and runs");
        Conformance.CheckReadingOrder(ReadingOrders.Runs, invertible: true).ThrowIfFailed();
        Conformance.CheckReadingOrder(ReadingOrders.Bidi).ThrowIfFailed();
        Conformance.CheckReadingOrder(ReadingOrders.Bidi, ReadingOrders.Get(ReadingOrders.Bidi)).ThrowIfFailed();
        var runs = ReadingOrders.Get(ReadingOrders.Runs);
        var bidi = ReadingOrders.Get(ReadingOrders.Bidi);

        // The OCR app's test line: both give its columns; runs undoes itself.
        string logical = "با 123 س abc de م ١٢", columns = "با 321 س ed cba م ٢١";
        Check(runs.ToColumns(logical, true) == columns && bidi.ToColumns(logical, true) == columns, $"runs '{runs.ToColumns(logical, true)}', bidi '{bidi.ToColumns(logical, true)}'");
        Check(runs.FromColumns(columns, true) == logical && bidi.FromColumns(columns, true) == logical, "both read the line back");
        Check(runs.ToColumns("the river", false) == "the river" && bidi.ToColumns("the river", false) == "the river", "a left-to-right line of Latin as it is");
        Check(runs.ToColumns("the باب سم end", false) == "the مس باب end", "runs: Arabic reversed in a left-to-right line");

        // Where UAX #9 differs from the app's order: a number after a Latin word reads back before it (both texts give the
        // same columns), and Arabic-Indic digits after Latin letters are a run of their own.
        string latinNumber = "س abc 123", latinColumns = "س 321 cba";
        Check(runs.ToColumns(latinNumber, true) == latinColumns && bidi.ToColumns(latinNumber, true) == latinColumns, "a Latin word and a number: the same columns");
        Check(runs.FromColumns(latinColumns, true) == latinNumber && bidi.FromColumns(latinColumns, true) == "س 123 abc"
              && bidi.ToColumns("س 123 abc", true) == latinColumns, $"bidi reads them back as '{bidi.FromColumns(latinColumns, true)}'");
        Check(runs.ToColumns("abc ١٢", true) == "٢١ cba" && bidi.ToColumns("abc ١٢", true) == "cba ٢١",
            $"Arabic-Indic digits after Latin: runs '{runs.ToColumns("abc ١٢", true)}', bidi '{bidi.ToColumns("abc ١٢", true)}'");
        ThrowsNaming<NotSupportedException>(() => ReadingOrders.Get("nobody"), "an unknown order", ["'nobody' is not registered (registered: ", "bidi", "ReadingOrders.Register"]);
    }

    // ------------------------------------------------------------------ G14: the line-recognition task

    private const string LineFamily = "IdrakTestLines";

    // The test's line recognizer: conv 3x3 → ReLU → average over the height → columns as a sequence → a score per class.
    private sealed class TestLineFamily(string name = LineFamily, Func<ImageModel, ImageModel>? change = null) : IImageModelFamily
    {
        public string Name => name;

        public ImageModel Read(ImageCheckpoint checkpoint)
        {
            var tensors = checkpoint.RequireTensors();
            int height = (int?)checkpoint.Config["line_height"] ?? 8;
            var network = new Sequential(
                StoredWeights.Conv2d(tensors, "conv", padding: (1, 1), device: checkpoint.Device), new ReLU(), new AvgPool2d((height, 1)),
                new Lambda(x => x.Permute(0, 3, 1, 2).Reshape(x.Shape[0], x.Shape[3], x.Shape[1] * x.Shape[2]), "ColumnsToSequence"),
                StoredWeights.Linear(tensors, "fc", checkpoint.Options.Weights, checkpoint.Device));
            var model = new ImageModel
            {
                Architecture = checkpoint.Architecture,
                Task = ImageTask.LineRecognition,
                Network = network,
                Preprocessor = new ImagePreprocessor { Height = height, Width = height, Normalize = false },
                Channels = 1,
                Labels = checkpoint.Labels(),
                Lines = new LineRecognitionSettings
                {
                    Height = height, RightToLeft = (string?)checkpoint.Config["direction"] == "rtl", ReadingOrder = (string?)checkpoint.Config["reading_order"],
                },
            };
            return change is null ? model : change(model);
        }
    }

    // A line recognizer checkpoint of the test family: blank, a, b, c.
    private static string LineCheckpoint(string folder, string architecture = LineFamily)
    {
        Directory.CreateDirectory(folder);
        var random = new Random(21);
        SafeTensorsWriter.Write(Path.Combine(folder, "model.safetensors"),
        [
            ("conv.weight", [4, 1, 3, 3], RandomArray(random, 36)), ("conv.bias", [4], RandomArray(random, 4)),
            ("fc.weight", [4, 4], RandomArray(random, 16)), ("fc.bias", [4], RandomArray(random, 4)),
        ], SafeTensorType.F32);
        File.WriteAllText(Path.Combine(folder, "config.json"),
            $$$"""{"architectures": ["{{{architecture}}}"], "line_height": 8, "id2label": {"0": "", "1": "a", "2": "b", "3": "c"}}""");
        return folder;
    }

    // A line of dark marks on a light page: a full stroke for a, the upper half for b, the lower half for c, a gap between.
    private static ImageData LineImage(string text, int height = 12)
    {
        int width = 2 + text.Length * 5;
        var pixels = new float[width * height];
        Array.Fill(pixels, 0.95f);
        for (int i = 0; i < text.Length; i++)
        {
            int x = 2 + i * 5;
            var (top, bottom) = text[i] switch { 'a' => (1, height - 1), 'b' => (1, height / 2), 'c' => (height / 2, height - 1), _ => (0, 0) };
            for (int y = top; y < bottom; y++)
            {
                pixels[y * width + x] = pixels[y * width + x + 1] = pixels[y * width + x + 2] = 0.1f;
            }
        }

        return new ImageData(pixels, 1, height, width);
    }

    private static ImageModelUnderTest LineModelUnderTest(string path, Device device)
    {
        var model = ImageModels.Load(path, new ImageModelOptions { Device = device });
        var reader = model.LineReader();
        return new ImageModelUnderTest("line-recognition", model.Labels, images => [.. reader.Read(images).Select(l => l.Labels.Select(v => (float)v).ToArray())], model);
    }

    private static void GenericLineRecognitionTask(Device device)
    {
        string folder = TempFolder("lines-family");
        ImageModelFamilies.Register(new TestLineFamily());
        ImageModelFamilies.Register(new TestLineFamily("IdrakTestLinesNoSettings", m => m with { Lines = null }));
        ImageModelFamilies.Register(new TestLineFamily("IdrakTestLinesBlank", m => m with { Lines = m.Lines! with { Blank = 9 } }));
        ImageModelFamilies.Register(new TestLineFamily("IdrakTestLinesNoLabels", m => m with { Labels = null }));
        try
        {
            string path = LineCheckpoint(Path.Combine(folder, "lines"));
            using (var model = ImageModels.Load(path, new ImageModelOptions { Device = device }))
            {
                Check(model.Task == ImageTask.LineRecognition && model.Lines is { Height: 8, Stride: 1, Blank: 0, RightToLeft: false } && model.InputShape is null
                      && model.Labels!.SequenceEqual(["", "a", "b", "c"]), "the description");
                var reader = model.LineReader();
                ImageData[] lines = [LineImage("abc"), LineImage("cab a"), LineImage("b", 16)];
                var read = reader.Read(lines);
                Check(read.Count == 3 && read.All(r => r.Text.All(c => c is 'a' or 'b' or 'c') && r.Confidence is >= 0 and <= 1 && r.Labels.All(l => l is >= 1 and <= 3)),
                    $"readings: {string.Join(" | ", read.Select(r => $"{r.Text} {r.Confidence}"))}");
                for (int i = 0; i < lines.Length; i++)
                {
                    var alone = reader.Read([lines[i]])[0];
                    Check(alone.Text == read[i].Text && alone.Labels.SequenceEqual(read[i].Labels) && Math.Abs(alone.LogProbability - read[i].LogProbability) < 1e-4,
                        $"line {i + 1} alone as in the batch");
                }

                ExpectRefusal<InvalidOperationException>(() => model.Classifier(), "is a line-recognition model, not a classification one", "the wrong predictor");
                ExpectRefusal<NotSupportedException>(() => model.LineReader("nobody"), "No CTC decoder 'nobody'", "an unknown decoder");

                // Prepared as the OCR app prepared its lines: ink high on 0, stretched, scaled to the height, flipped for rtl.
                var image = LineImage("ab");
                var prepared = ModelLineReader.Prepare(image, new LineRecognitionSettings { Height = 8, Stride = 4 });
                Check(prepared.Height == 8 && prepared.Width == Math.Max(4, (int)Math.Round(image.Width * 8.0 / image.Height)) && prepared.Pixels.Max() > 0.9f && prepared.Pixels.Min() == 0f,
                    $"prepared {prepared.Width} wide, values {prepared.Pixels.Min()}..{prepared.Pixels.Max()}");
                var flipped = ModelLineReader.Prepare(image, new LineRecognitionSettings { Height = 8, Stride = 4, RightToLeft = true });
                Check(Enumerable.Range(0, prepared.Width).All(x => flipped.Pixels[3 * prepared.Width + x] == prepared.Pixels[3 * prepared.Width + prepared.Width - 1 - x]), "right to left: flipped");

                // A reading order: the labels in column order, the text logical.
                var rtl = new ModelLineReader(model.Network, model.Labels!, model.Lines! with { RightToLeft = true, ReadingOrder = ReadingOrders.Runs }, device: device);
                var (ids, unknown) = rtl.Encode("ab cz");
                Check(ids.SequenceEqual([3, 2, 1]) && unknown.SequenceEqual(["z", " "]), $"encoded {string.Join(",", ids)}, unknown {string.Join(",", unknown)}");
                Check(rtl.Text([1, 2, 3]) == "cba" && rtl.Text([0, 3, 2, 1, 9]) == "abc", "decoded to logical order; the blank and ids out of range add nothing");
            }

            var samples = new[]
            {
                new ImageFamilySample(path, [LineImage("abc"), LineImage("ca"), LineImage("bbb", 10)], Tolerance: 0f),
                new ImageFamilySample(LineCheckpoint(Path.Combine(folder, "nobody"), "NobodysLines"), Error: "Image model family 'NobodysLines' is not registered"),
            };
            Conformance.CheckImageModelFamily(p => LineModelUnderTest(p, device), samples, new ContractCheckOptions { RandomCases = 4 }).ThrowIfFailed();

            var options = new ImageModelOptions { Device = device };
            ExpectRefusal<InvalidOperationException>(() => ImageModels.Load(path, options with { Architecture = "IdrakTestLinesNoSettings" }), "a line recognizer without line settings", "no settings");
            ExpectRefusal<InvalidOperationException>(() => ImageModels.Load(path, options with { Architecture = "IdrakTestLinesBlank" }), "blank 9 outside the 4 classes", "the blank");
            ExpectRefusal<InvalidOperationException>(() => ImageModels.Load(path, options with { Architecture = "IdrakTestLinesNoLabels" }), "a line recognizer without its labels", "no labels");
        }
        finally
        {
            foreach (string name in new[] { LineFamily, "IdrakTestLinesNoSettings", "IdrakTestLinesBlank", "IdrakTestLinesNoLabels" })
            {
                ImageModelFamilies.Unregister(name);
            }

            Directory.Delete(folder, recursive: true);
        }
    }

    private static void GenericLineRecognitionCli(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                         // the commands' wiring, not the device
        }

        string d = device.ToString(), folder = TempFolder("lines-cli"), cache = Path.Combine(folder, "cache"), data = Path.Combine(folder, "lines");
        ImageModelFamilies.Register(new TestLineFamily());
        try
        {
            var random = new Random(4);
            Directory.CreateDirectory(data);
            for (int i = 0; i < 12; i++)
            {
                string text = new([.. Enumerable.Range(0, random.Next(2, 6)).Select(_ => "abc"[random.Next(3)])]);
                ImageEncoders.Save(Path.Combine(data, $"line-{i:D2}.png"), LineImage(text));
                File.WriteAllText(Path.Combine(data, $"line-{i:D2}.txt"), text + "\n");
            }

            File.WriteAllText(Path.Combine(data, "empty.txt"), "");
            ImageEncoders.Save(Path.Combine(data, "empty.png"), LineImage("ab"));          // not transcribed: left out

            // A builder network ending in [steps, classes]: the data's three characters and the blank.
            string spec = Path.Combine(folder, "lines.json"), package = Path.Combine(folder, "lines.ikm");
            File.WriteAllText(spec, Network.Image(1, 8, 16).Conv2d(4, 3, padding: 1).ReLU().AvgPool2d((8, 1)).ColumnsToSequence().Linear(4).Named("lines").ToJson().ToJsonString());
            var trained = TrainCliJson("train", spec, "--task", "line-recognition", "--data", data, "--direction", "rtl", "--epochs", "2", "--batch", "4", "--validation", "0.25",
                "--augment", "noise(std=0.02)", "-o", package, "--cache", cache, "-d", d);
            Check((string?)trained["task"] == "line-recognition" && trained["classes"]!.AsArray().Select(c => (string?)c).SequenceEqual(["", "a", "b", "c"])
                  && (int)trained["trainingLines"]! == 9 && (int)trained["evaluationLines"]! == 3 && trained["metrics"]!["cer"] is not null && trained["metrics"]!["wer"] is not null
                  && File.Exists(package), $"train: {trained}");
            var predicted = TrainCliJson("predict", package, data, "-d", d);
            var rows = predicted["predictions"]!.AsArray();
            Check((string?)predicted["task"] == "line-recognition" && rows.Count == 13 && rows.All(r => ((string)r!["text"]!).All(c => c is 'a' or 'b' or 'c'))
                  && rows.All(r => (double)r!["confidence"]! is >= 0 and <= 1), $"predict with the package: {predicted}");
            var beam = TrainCliJson("predict", package, Path.Combine(data, "line-00.png"), "--decoder", "beam", "-d", d);
            Check((string?)beam["decoder"] == "beam" && beam["predictions"]!.AsArray().Count == 1, $"--decoder beam: {beam}");
            var badDecoder = TrainCli("predict", package, data, "--decoder", "nobody", "-d", d);
            Check(badDecoder.Code == 2 && badDecoder.Err.Contains("greedy", StringComparison.Ordinal), $"an unknown decoder: {badDecoder.Err}");

            // A family fine-tuned (its weights written), then read with --weights.
            string checkpoint = LineCheckpoint(Path.Combine(folder, "family")), tuned = Path.Combine(folder, "tuned.ikw");
            var fine = TrainCliJson("train", checkpoint, "--data", data, "--epochs", "1", "--batch", "4", "--validation", "0", "-o", tuned, "--cache", cache, "-d", d);
            Check((string?)fine["task"] == "line-recognition" && File.Exists(tuned) && (int)fine["evaluationLines"]! == 0, $"fine-tune: {fine}");
            var read = TrainCliJson("predict", checkpoint, data, "--weights", tuned, "-d", d);
            Check(read["predictions"]!.AsArray().Count == 13, $"predict with the family and --weights: {read}");

            // Refusals: options that are not a line recognizer's, a network of the wrong width, --direction with a family.
            var badLoss = TrainCli("train", spec, "--task", "line-recognition", "--data", data, "--loss", "dice", "--cache", cache, "-d", d);
            Check(badLoss.Code == 2 && badLoss.Err.Contains("CTC", StringComparison.Ordinal), $"--loss on lines: {badLoss.Err}");
            string wide = Path.Combine(folder, "wide.json");
            File.WriteAllText(wide, Network.Image(1, 8, 16).Conv2d(4, 3, padding: 1).AvgPool2d((8, 1)).ColumnsToSequence().Linear(6).ToJson().ToJsonString());
            var badWidth = TrainCli("train", wide, "--task", "line-recognition", "--data", data, "--epochs", "1", "--cache", cache, "-d", d);
            Check(badWidth.Code == 1 && badWidth.Err.Contains("make its last layer 4 wide", StringComparison.Ordinal), $"6 classes for 3 characters: {badWidth.Code} {badWidth.Err}");
            var badDirection = TrainCli("train", checkpoint, "--data", data, "--direction", "rtl", "--cache", cache, "-d", d);
            Check(badDirection.Code == 2 && badDirection.Err.Contains("--direction", StringComparison.Ordinal), $"--direction with a family: {badDirection.Err}");
        }
        finally
        {
            ImageModelFamilies.Unregister(LineFamily);
            Directory.Delete(folder, recursive: true);
        }
    }
}
