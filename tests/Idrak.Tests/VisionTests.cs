// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Data;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Vision;

// Idrak.Vision: page segmentation, character framing, reading order, and the text recognizer end to end.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] VisionGroup =
    [
        ("vision: segmentation finds lines, characters and spaces; dots join their line; specks drop; either polarity", VisionSegmentation),
        ("vision: GlyphFrame centres and scales a character; a page's character frames as its image does; Reframe transform", VisionGlyphFrame),
        ("vision: WritingScript from Unicode; right-to-left reading order keeps numbers left to right", VisionTextOrder),
        ("vision: TextRecognizer reads two scripts: one script per line, look-alikes in words, right to left, batches, Load", VisionRecognizer),
    ];

    // Shapes drawn into 20 x 20 images (ink 1): the test alphabet. Look-alikes share a shape, with small differences.
    private static GlyphImage VisionShape(string name)
    {
        const int n = 20;
        var p = new float[n * n];
        void Fill(int x0, int y0, int x1, int y1)
        {
            for (int y = y0; y < y1; y++)
            {
                for (int x = x0; x < x1; x++)
                {
                    p[y * n + x] = 1f;
                }
            }
        }

        switch (name)
        {
            case "bar": Fill(8, 1, 12, 19); break;                                         // 1 and ١ exactly
            case "bar+serif": Fill(8, 1, 12, 19); Fill(6, 1, 14, 3); break;                // I
            case "bar+tail": Fill(8, 1, 12, 19); Fill(12, 16, 15, 19); break;              // ا
            case "ring": Fill(3, 3, 17, 17); for (int y = 6; y < 14; y++) for (int x = 6; x < 14; x++) p[y * n + x] = 0; break;   // 0 and ٥
            case "ring+serif": Fill(3, 3, 17, 17); for (int y = 6; y < 14; y++) for (int x = 6; x < 14; x++) p[y * n + x] = 0; Fill(15, 0, 19, 3); break;   // O
            case "ring+notch": Fill(3, 3, 17, 17); for (int y = 6; y < 14; y++) for (int x = 6; x < 14; x++) p[y * n + x] = 0; Fill(9, 17, 11, 20); break;  // ه
            case "square": Fill(2, 2, 18, 18); break;                                     // A
            case "bowl": Fill(2, 6, 18, 10); Fill(2, 2, 5, 10); Fill(15, 2, 18, 10); Fill(9, 15, 12, 18); break;   // ب, with its dot below
            default: throw new ArgumentException(name);
        }

        return GlyphImage.Crop(p, n, n);
    }

    private static readonly Dictionary<char, string> VisionShapeOf = new()
    {
        ['1'] = "bar", ['١'] = "bar", ['I'] = "bar+serif", ['ا'] = "bar+tail",
        ['0'] = "ring", ['٥'] = "ring", ['O'] = "ring+serif", ['ه'] = "ring+notch",
        ['A'] = "square", ['ب'] = "bowl",
    };

    private static ImageData VisionPage(string text, int scale = 2) =>
        PageComposer.Compose(text, (c, _) => VisionShapeOf.TryGetValue(c, out var s) ? VisionShape(s) : null, new PageCompositionOptions { Scale = scale });

    private static void VisionSegmentation(Device device)
    {
        var layout = PageSegmenter.Segment(VisionPage("AIA 10\nبا ب"));
        Check(layout.DarkOnLight, "dark ink on light paper is detected");
        Check(layout.Lines.Count == 2, $"two lines, not {layout.Lines.Count}: the dot under ب joins its line");
        Check(layout.Lines[0].Glyphs.Count == 5 && layout.Lines[1].Glyphs.Count == 3, $"characters per line: {string.Join(", ", layout.Lines.Select(l => l.Glyphs.Count))}");
        Check(layout.Lines[0].Glyphs.Select(g => g.SpaceBefore).SequenceEqual([false, false, false, true, false]), "the space before 10");
        Check(layout.Lines[1].Glyphs.Select(g => g.SpaceBefore).SequenceEqual([false, true, false]), "the space in the Arabic line");
        Check(layout.Lines[0].Glyphs.Zip(layout.Lines[0].Glyphs.Skip(1)).All(p => p.First.Box.Right <= p.Second.Box.X), "characters left to right, apart");
        var bowl = layout.Lines[1].Glyphs[0].Box;
        Check(bowl.Height > 14 * 2, $"ب keeps its dot: {bowl.Height} rows");

        // Light ink on dark paper, and a speck: the same characters.
        var page = VisionPage("AIA 10\nبا ب");
        var inverted = page.Pixels.Select(v => 1f - v).ToArray();
        inverted[5 * page.Width + 5] = 1f;   // a one-pixel speck in the margin
        var dark = PageSegmenter.Segment(new ImageData(inverted, 1, page.Height, page.Width));
        Check(!dark.DarkOnLight && dark.Lines.Count == 2 && dark.GlyphCount == layout.GlyphCount, $"light on dark: {dark.Lines.Count} lines, {dark.GlyphCount} characters");

        float otsu = PageSegmenter.Otsu([0.1f, 0.12f, 0.08f, 0.9f, 0.88f, 0.92f]);
        Check(otsu > 0.12f && otsu < 0.88f, $"Otsu's threshold separates the groups: {otsu}");
    }

    private static void VisionGlyphFrame(Device device)
    {
        // A 10-row, 4-column bar off-centre in a 28 x 28 image: centred, 26 rows high, at full contrast.
        var image = new float[28 * 28];
        for (int y = 2; y < 12; y++) for (int x = 3; x < 7; x++) image[y * 28 + x] = 0.5f;
        var framed = new float[28 * 28];
        GlyphFrame.Fit(image, 28, 28, framed);
        var rows = Enumerable.Range(0, 28).Where(y => Enumerable.Range(0, 28).Any(x => framed[y * 28 + x] > 0.5f)).ToArray();
        var cols = Enumerable.Range(0, 28).Where(x => Enumerable.Range(0, 28).Any(y => framed[y * 28 + x] > 0.5f)).ToArray();
        Check(MathF.Abs(framed.Max() - 1f) < 1e-6f, "full contrast");
        Check(rows.Length >= 25 && rows[0] <= 2 && rows[^1] >= 25, $"26 rows high: {rows.First()}..{rows.Last()}");
        Check(Math.Abs(cols[0] + cols[^1] - 27) <= 1 && cols.Length is >= 9 and <= 12, $"centred, aspect kept: columns {cols.First()}..{cols.Last()}");

        // A character cut from a page (scale 1) frames as its own image does.
        var ring = VisionShape("ring");
        var page = PageComposer.Compose("0", (_, _) => ring, new PageCompositionOptions { Scale = 1 });
        var layout = PageSegmenter.Segment(page);
        var fromPage = new float[28 * 28];
        GlyphFrame.Extract(layout, layout.Lines[0].Glyphs[0].Box, fromPage);
        var fromImage = new float[28 * 28];
        GlyphFrame.Fit(ring.Pixels, ring.Height, ring.Width, fromImage);
        AssertClose(fromImage, fromPage, 1e-5f, "a page's character and its image frame alike");

        // As a DataLoader transform, in place.
        var sample = image.ToArray();
        GlyphFrame.Reframe().Apply(sample, Span<float>.Empty, [1, 28, 28], new Random(1));
        AssertClose(framed, sample, 0f, "Reframe = Fit");
    }

    private static void VisionTextOrder(Device device)
    {
        Check(WritingScript.Of('ب') == WritingScript.Arabic && WritingScript.Of('٣') == WritingScript.Arabic && WritingScript.Arabic.RightToLeft, "Arabic letters and digits");
        Check(WritingScript.Of('7') == WritingScript.Latin && WritingScript.Of('é') == WritingScript.Latin && WritingScript.Latin.HasCase, "Latin");
        Check(WritingScript.Of('ש').RightToLeft && WritingScript.Of('Ж') == WritingScript.Cyrillic && WritingScript.Of("  ب1") == WritingScript.Arabic, "Hebrew, Cyrillic, a text's first script");
        // Page order (left to right) of "بيت ٢٠٢ مفتوح" is: حوتفم ٢٠٢ تيب; reading order back.
        Check(TextOrder.Reorder(["حوتفم", "٢٠٢", "تيب"], rightToLeft: true) == "بيت ٢٠٢ مفتوح", "words right to left, numbers left to right");
        Check(TextOrder.Reorder(["AB", "12"], rightToLeft: false) == "AB 12", "left to right unchanged");
    }

    private static void VisionRecognizer(Device device)
    {
        // A template classifier: logit c = 30 * <frame, template_c> / |template_c|, so each shape scores highest on its own
        // template; look-alikes (1/I/١/ا, 0/O/٥/ه) score close, the exact templates (1, ١, 0, ٥) highest on a plain bar or ring.
        string[] classes = ["A", "I", "O", "1", "0", "ب", "ا", "ه", "١", "٥"];
        var weights = new float[28 * 28 * classes.Length];
        for (int c = 0; c < classes.Length; c++)
        {
            var shape = VisionShape(VisionShapeOf[classes[c][0]]);
            var frame = new float[28 * 28];
            GlyphFrame.Fit(shape.Pixels, shape.Height, shape.Width, frame);
            float norm = MathF.Sqrt(frame.Sum(v => v * v));
            for (int p = 0; p < frame.Length; p++)
            {
                weights[p * classes.Length + c] = 30f * frame[p] / norm;
            }
        }

        using var model = Network.Image(1, 28, 28).OnDevice(device).Flatten().Linear(classes.Length, bias: false).Build();
        model.Parameters().First().CopyFrom(weights);

        // "AAI" is a letter word: its bar reads 1 by shape, I by context. "10" is a number. The Arabic line has ب, so its
        // bars read ١ (not 1); in "باب" the bar becomes ا by context; "١٥" stays a number. Page order is right to left.
        const string drawn = "AA1 10\nب١ب ١٥", text = "AAI 10\nباب ١٥";
        var page = VisionPage(drawn);
        using (var ocr = TextRecognizer.For(model).Characters(classes).Build())
        {
            var read = ocr.Read(page);
            Check(read.Text == text, $"read:\n{read.Text}\nexpected:\n{text}");
            Check(read.Lines[0].Script == WritingScript.Latin && read.Lines[1].Script == WritingScript.Arabic && read.Lines[1].RightToLeft, "scripts per line");
            var bar = read.Lines[0].Characters[2];
            Check(bar.Text == "I" && bar.Candidates[0].Text is "1" or "١", $"context changed {bar.Candidates[0].Text} to {bar.Text}");
            Check(read.Lines.SelectMany(l => l.Characters).All(c => c.Confidence is >= 0 and <= 1 && c.Box.Width > 0 && c.Candidates.Count == 3), "confidences, boxes and candidates");

            // Without context, the shapes alone: the bar in AAI stays a digit, and Arabic still keeps Arabic classes.
            using var raw = TextRecognizer.For(model).Characters(classes).UseWordContext(false).Build();
            var plain = raw.Read(page).Lines;
            Check(plain[0].Characters[2].Text == "1" && plain[1].Characters.All(c => WritingScript.Of(c.Text) == WritingScript.Arabic), $"no context: {plain[0].Text} / {plain[1].Text}");

            // Batches of 3 give the same text.
            using var small = TextRecognizer.For(model).Characters(classes).BatchSize(3).Build();
            Check(small.Read(page).Text == text, "batches of 3");
        }

        // From a package written by Predictor.Save: its classes and input shape.
        string path = Path.Combine(Path.GetTempPath(), $"idrak-ocr-{Guid.NewGuid():N}.ikm");
        try
        {
            using (var predictor = Predictor.For(model).InputShape(1, 28, 28).Softmax().Classes(classes).Build())
            {
                predictor.Save(path);
            }

            using var loaded = TextRecognizer.Load(path, device).Build();
            Check(loaded.Classes.SequenceEqual(classes) && loaded.Read(page).Text == text, "TextRecognizer.Load reads the same");
        }
        finally
        {
            File.Delete(path);
        }
    }
}
