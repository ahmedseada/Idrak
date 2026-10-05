// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Data;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Training;
using Idrak.Vision;

// Idrak.Vision: foreground, connected regions, framing, boxes and suppression, channel normalization, segmentation,
// region classification and model-based detection.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] VisionGroup =
    [
        ("vision: foreground (polarity, Otsu); connected components (4 and 8 neighbours, areas, boxes, centres, minimum area)", VisionRegions),
        ("vision: ContentFrame centres and scales; a region of an image frames as its own image does; Reframe transform", VisionContentFrame),
        ("vision: boxes (IoU, corners, centre, scale, clip); non-maximum suppression (per class, across classes, scores, limit)", VisionBoxes),
        ("vision: Normalize layer is (x - mean) / std per channel, survives JSON and packages; ChannelStatistics", VisionNormalize),
        ("vision: masks from logits, resize and metrics; PixelCrossEntropy matches the manual loss and trains; ModelSegmenter", VisionSegmentation),
        ("vision: RegionClassifier classifies a canvas's shapes in batches, from a package too; ModelDetector decodes, rescales, suppresses", VisionClassifyAndDetect),
    ];

    // Dark shapes (0) on a light canvas (1), as listed: (x, y, width, height, hole).
    private static ImageData Canvas(int width, int height, params (int X, int Y, int W, int H, bool Hole)[] shapes)
    {
        var p = Enumerable.Repeat(1f, width * height).ToArray();
        foreach (var (x0, y0, w, h, hole) in shapes)
        {
            for (int y = y0; y < y0 + h; y++)
            {
                for (int x = x0; x < x0 + w; x++)
                {
                    bool inside = hole && x > x0 + 1 && x < x0 + w - 2 && y > y0 + 1 && y < y0 + h - 2;
                    p[y * width + x] = inside ? 1f : 0f;
                }
            }
        }

        return new ImageData(p, 1, height, width);
    }

    private static void VisionRegions(Device device)
    {
        // A 5 x 5 square, two pixels touching only at a corner, and a one-pixel speck.
        var canvas = Canvas(40, 30, (2, 2, 5, 5, false), (15, 5, 1, 1, false), (16, 6, 1, 1, false), (30, 25, 1, 1, false));
        var fg = Foreground.Extract(canvas);
        Check(fg.Inverted && fg.IsForeground(3, 3) && !fg.IsForeground(10, 10), "dark on light is inverted to foreground high");
        Check(!Foreground.Extract(canvas, Polarity.LightOnDark).IsForeground(3, 3), "a forced polarity is kept");
        float otsu = Foreground.Otsu([0.1f, 0.12f, 0.08f, 0.9f, 0.88f, 0.92f]);
        Check(otsu > 0.12f && otsu < 0.88f, $"Otsu separates the groups: {otsu}");

        var eight = ConnectedComponents.Find(fg);
        Check(eight.Regions.Count == 3, $"8 neighbours: square, diagonal pair, speck ({eight.Regions.Count})");
        var square = eight.Regions[0];
        Check(square.Label == 1 && square.Box == new PixelBox(2, 2, 5, 5) && square.Area == 25 && square.CenterX == 4.5f && square.CenterY == 4.5f,
            $"the square: {square}");
        Check(eight.Regions[1].Area == 2 && eight.LabelAt(15, 5) == 2 && eight.LabelAt(16, 6) == 2 && eight.LabelAt(0, 0) == 0, "the diagonal pair is one region");

        var four = ConnectedComponents.Find(fg, Connectivity.Four, minArea: 2);
        Check(four.Regions.Count == 1 && four.LabelAt(15, 5) == 0 && four.LabelAt(30, 25) == 0, "4 neighbours and a minimum area of 2 leave the square alone");
        Check(four.Labels.ToArray().Count(l => l == 1) == 25, "its 25 pixels keep its label");
    }

    private static void VisionContentFrame(Device device)
    {
        // A 10 x 4 bar off-centre in a 28 x 28 image: centred, 26 rows high, full contrast, aspect kept.
        var image = new float[28 * 28];
        for (int y = 2; y < 12; y++) for (int x = 3; x < 7; x++) image[y * 28 + x] = 0.5f;
        var framed = new float[28 * 28];
        ContentFrame.Fit(image, 28, 28, framed);
        var rows = Enumerable.Range(0, 28).Where(y => Enumerable.Range(0, 28).Any(x => framed[y * 28 + x] > 0.5f)).ToArray();
        var cols = Enumerable.Range(0, 28).Where(x => Enumerable.Range(0, 28).Any(y => framed[y * 28 + x] > 0.5f)).ToArray();
        Check(MathF.Abs(framed.Max() - 1f) < 1e-6f, "full contrast");
        Check(rows.Length >= 25 && rows[0] <= 2 && rows[^1] >= 25 && Math.Abs(rows[0] + rows[^1] - 27) <= 1, $"26 rows, centred: {rows.First()}..{rows.Last()}");
        Check(Math.Abs(cols[0] + cols[^1] - 27) <= 1 && cols.Length is >= 9 and <= 12, $"centred, aspect kept: {cols.First()}..{cols.Last()}");

        // A region of a larger image frames as the same object on its own.
        var canvas = Canvas(50, 40, (20, 10, 9, 14, true));
        var fg = Foreground.Extract(canvas);
        var region = ConnectedComponents.Find(fg).Regions.Single();
        var fromCanvas = new float[28 * 28];
        ContentFrame.Extract(fg, region.Box, fromCanvas);
        var alone = new float[region.Box.Area];
        for (int y = 0; y < region.Box.Height; y++)
            for (int x = 0; x < region.Box.Width; x++)
                alone[y * region.Box.Width + x] = fg.Values[(region.Box.Y + y) * fg.Width + region.Box.X + x];
        var fromImage = new float[28 * 28];
        ContentFrame.Fit(alone, region.Box.Height, region.Box.Width, fromImage, threshold: fg.Threshold);
        AssertClose(fromImage, fromCanvas, 1e-6f, "a region and its own image frame alike");

        bool threw = false;
        try { ContentFrame.Extract(fg, new PixelBox(45, 35, 10, 10), fromCanvas); } catch (ArgumentOutOfRangeException) { threw = true; }
        Check(threw, "a box outside the image is refused");

        // A small object grows smooth, not in blocks of repeated pixels: a 4 x 4 ramp enlarged to 26 pixels changes
        // between neighbouring output pixels along its middle row.
        var ramp = new float[4 * 4];
        for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++) ramp[y * 4 + x] = 0.4f + 0.2f * x;
        var grown = new float[28 * 28];
        ContentFrame.Fit(ramp, 4, 4, grown);
        var middle = grown.AsSpan(14 * 28 + 2, 24).ToArray();
        int longestRun = 1, run = 1;
        for (int i = 1; i < middle.Length; i++) { run = middle[i] == middle[i - 1] ? run + 1 : 1; longestRun = Math.Max(longestRun, run); }
        Check(longestRun <= 2 && middle.Zip(middle.Skip(1)).All(p => p.Second >= p.First - 1e-6f),
            $"enlarging is smooth and keeps the ramp's order: longest run of equal pixels {longestRun}");

        var sample = image.ToArray();
        ContentFrame.Reframe().Apply(sample, Span<float>.Empty, [1, 28, 28], new Random(1));
        AssertClose(framed, sample, 0f, "Reframe = Fit, in place");
    }

    private static void VisionBoxes(Device device)
    {
        var a = new BoundingBox(0, 0, 10, 10);
        Check(a.IntersectionOverUnion(a) == 1f && a.IntersectionOverUnion(new BoundingBox(20, 20, 5, 5)) == 0f, "IoU of a box with itself and with a far one");
        Check(MathF.Abs(a.IntersectionOverUnion(new BoundingBox(5, 0, 10, 10)) - 50f / 150f) < 1e-6f, "IoU of half-overlapping boxes");
        Check(BoundingBox.FromCorners(10, 8, 2, 4) == new BoundingBox(2, 4, 8, 4) && BoundingBox.FromCenter(5, 5, 4, 2) == new BoundingBox(3, 4, 4, 2), "corners and centre");
        Check(a.Scale(2, 3) == new BoundingBox(0, 0, 20, 30) && new BoundingBox(-5, 2, 10, 20).Clip(8, 10) == new BoundingBox(0, 2, 5, 8), "scale and clip");
        Check(new BoundingBox(1.2f, 2.5f, 3f, 1f).ToPixelBox() == new PixelBox(1, 2, 4, 2), "to whole pixels, outwards");

        List<Detection> found =
        [
            new(new BoundingBox(0, 0, 10, 10), 0, 0.9f),
            new(new BoundingBox(1, 1, 10, 10), 0, 0.8f),     // overlaps the first (IoU 0.68): suppressed
            new(new BoundingBox(1, 1, 10, 10), 1, 0.7f),     // same box, another class: kept per class
            new(new BoundingBox(30, 30, 5, 5), 0, 0.6f),
            new(new BoundingBox(50, 50, 5, 5), 0, 0.05f),    // below the minimum score
        ];
        var perClass = NonMaxSuppression.Apply(found, 0.5f, minScore: 0.1f);
        Check(perClass.Select(d => d.Score).SequenceEqual([0.9f, 0.7f, 0.6f]), $"per class: {string.Join(", ", perClass.Select(d => d.Score))}");
        var across = NonMaxSuppression.Apply(found, 0.5f, minScore: 0.1f, perClass: false);
        Check(across.Select(d => d.Score).SequenceEqual([0.9f, 0.6f]), "across classes");
        Check(NonMaxSuppression.Apply(found, 0.5f, maxDetections: 1).Count == 1, "at most one");
    }

    private static void VisionNormalize(Device device)
    {
        float[] mean = [0.5f, 0.2f, 0.8f], std = [0.25f, 0.5f, 0.1f];
        var network = Network.Image(3, 2, 2).OnDevice(device).Normalize(mean, std);
        using var model = network.Build();
        var random = new Random(2);
        var values = Enumerable.Range(0, 2 * 3 * 4).Select(_ => (float)random.NextDouble()).ToArray();
        using var x = Tensor.From(values, [2, 3, 2, 2], device);
        using var y = model.Predict(x);
        var expected = values.Select((v, i) => (v - mean[i / 4 % 3]) / std[i / 4 % 3]).ToArray();
        AssertClose(expected, y.ToArray(), 1e-5f, "(x - mean) / std per channel");
        Check(model.Parameters().Count() == 0 && model.Buffers().Count() == 2, "nothing to train; mean and std kept as buffers");

        using var rebuilt = Network.FromJson(network.ToJson()).OnDevice(device).Build();
        using var y2 = rebuilt.Predict(x);
        AssertClose(expected, y2.ToArray(), 1e-5f, "the JSON step rebuilds it");

        string path = Path.Combine(Path.GetTempPath(), $"idrak-normalize-{Guid.NewGuid():N}.ikm");
        try
        {
            using (var predictor = Predictor.For(model).InputShape(3, 2, 2).Build())
            {
                predictor.Save(path);
            }

            using var loaded = Predictor.Load(path, device).Build();
            AssertClose(expected[..12], loaded.Predict(values[..12]), 1e-5f, "a package keeps it");
            // A model with buffers and no parameters: inputs go to its device, not to the default one.
            using var inMemory = Predictor.For(model).InputShape(3, 2, 2).Build();
            Check(loaded.Device == device && inMemory.Device == device, $"inputs go to the model's device: {loaded.Device}, {inMemory.Device}");
        }
        finally
        {
            File.Delete(path);
        }

        // Two samples of [2, 2, 2]: channel 0 holds 1..4 and 5..8, channel 1 is constant 3.
        var features = new float[2, 8];
        for (int s = 0; s < 2; s++)
        {
            for (int p = 0; p < 4; p++)
            {
                features[s, p] = s * 4 + p + 1;
                features[s, 4 + p] = 3;
            }
        }

        var stats = ChannelStatistics.Compute(Dataset.FromArrays(features, new float[2, 1]).WithFeatureShape(2, 2, 2));
        Check(MathF.Abs(stats.Mean[0] - 4.5f) < 1e-5f && MathF.Abs(stats.Std[0] - MathF.Sqrt(5.25f)) < 1e-5f, $"channel 0: {stats.Mean[0]}, {stats.Std[0]}");
        Check(stats.Mean[1] == 3f && stats.Std[1] == 1f, "a constant channel gets std 1");
        Check(ChannelStatistics.ImageNet.Mean.Count == 3 && ChannelStatistics.ImageNet.Std[2] == 0.225f, "ImageNet's statistics");
    }

    private static void VisionSegmentation(Device device)
    {
        // Masks from logits [1, 2, 2, 2]: class 1 wins where its logit is higher.
        var mask = SegmentationMask.FromLogits([0f, 5f, 1f, 0f, /* class 1: */ 1f, 0f, 0f, 2f], 2, 2, 2);
        Check(mask.Labels.ToArray().SequenceEqual([1, 0, 0, 1]) && mask.Counts().SequenceEqual([2, 2]), "argmax per pixel");
        var big = mask.Resize(4, 4);
        Check(big.LabelAt(0, 0) == 1 && big.LabelAt(1, 1) == 1 && big.LabelAt(3, 0) == 0 && big.LabelAt(3, 3) == 1, "nearest-neighbour resize");
        var truth = new SegmentationMask([1, 0, 1, 1], 2, 2, 2);
        var score = SegmentationMetrics.Compute(mask, truth);
        Check(score.PixelAccuracy == 0.75 && score.ClassIoU[0] == 0.5 && Math.Abs(score.ClassIoU[1] - 2.0 / 3) < 1e-12, $"metrics: {score}");

        // PixelCrossEntropy against the loss computed by hand.
        var random = new Random(5);
        var logits = Enumerable.Range(0, 2 * 3 * 4).Select(_ => (float)(random.NextDouble() * 4 - 2)).ToArray();
        var labels = Enumerable.Range(0, 2 * 4).Select(_ => (float)random.Next(3)).ToArray();
        double manual = 0;
        for (int n = 0; n < 2; n++)
        {
            for (int p = 0; p < 4; p++)
            {
                var z = Enumerable.Range(0, 3).Select(c => (double)logits[n * 12 + c * 4 + p]).ToArray();
                double logSum = Math.Log(z.Sum(v => Math.Exp(v - z.Max()))) + z.Max();
                manual += logSum - z[(int)labels[n * 4 + p]];
            }
        }

        using (var lx = Tensor.From(logits, [2, 3, 2, 2], device))
        using (var lt = Tensor.From(labels, [2, 4], device))
        using (var loss = Losses.PixelCrossEntropy(lx, lt))
        {
            Check(Math.Abs(loss.Item() - manual / 8) < 1e-4, $"PixelCrossEntropy {loss.Item()} vs {manual / 8}");
        }

        // A one-layer network learns to mark bright pixels, then segments a larger image through ModelSegmenter.
        const int count = 64;
        var features = new float[count, 16];
        var targets = new float[count, 16];
        for (int i = 0; i < count; i++)
        {
            for (int p = 0; p < 16; p++)
            {
                bool on = random.NextDouble() < 0.4;
                features[i, p] = on ? 0.6f + 0.4f * (float)random.NextDouble() : 0.4f * (float)random.NextDouble();
                targets[i, p] = on ? 1 : 0;
            }
        }

        using var net = Network.Image(1, 4, 4).OnDevice(device).Seed(3).Conv2d(2, 1).Build();
        var history = new TrainingRun
        {
            Model = net, Loss = Losses.PixelCrossEntropy, Optimizer = p => new Adam(p, 0.1f),
            Train = Dataset.FromArrays(features, targets).WithFeatureShape(1, 4, 4).Batches(16, shuffle: true, device: device, seed: 1), Epochs = 40,
        }.Fit();
        Check(history.Epochs[^1].Loss < 0.25 * history.Epochs[0].Loss, $"the loss falls: {history.Epochs[0].Loss:F3} -> {history.Epochs[^1].Loss:F3}");

        var pixels = Enumerable.Range(0, 64).Select(i => i % 8 < 4 ? 0.9f : 0.1f).ToArray();   // left half bright
        var segmented = new ModelSegmenter(net, 1, 4, 4, device).Segment(new ImageData(pixels, 1, 8, 8));
        var expectedMask = new SegmentationMask([.. pixels.Select(v => v > 0.5f ? 1 : 0)], 8, 8, 2);
        Check(segmented.Width == 8 && SegmentationMetrics.Compute(segmented, expectedMask).PixelAccuracy == 1.0, "ModelSegmenter: every pixel right, at the image's size");
    }

    private static void VisionClassifyAndDetect(Device device)
    {
        // A template classifier for three shapes framed by ContentFrame: logit c = 30 * <frame, template_c> / |template_c|.
        var shapes = new (string Name, (int, int, int, int, bool) Shape)[] { ("square", (0, 0, 10, 10, false)), ("bar", (0, 0, 3, 12, false)), ("ring", (0, 0, 12, 12, true)) };
        var weights = new float[28 * 28 * shapes.Length];
        for (int c = 0; c < shapes.Length; c++)
        {
            var (_, (_, _, w, h, hole)) = shapes[c];
            var alone = Foreground.Extract(Canvas(w, h, (0, 0, w, h, hole)), Polarity.DarkOnLight, 0.5f);
            var frame = new float[28 * 28];
            ContentFrame.Fit(alone.Values, h, w, frame, threshold: 0.5f);
            float norm = MathF.Sqrt(frame.Sum(v => v * v));
            for (int p = 0; p < frame.Length; p++)
            {
                weights[p * shapes.Length + c] = 30f * frame[p] / norm;
            }
        }

        using var model = Network.Image(1, 28, 28).OnDevice(device).Flatten().Linear(shapes.Length, bias: false).Build();
        model.Parameters().First().CopyFrom(weights);
        string[] classes = [.. shapes.Select(s => s.Name)];

        // Shapes of other sizes on a canvas, in raster order: a ring, a bar, a square.
        var canvas = Canvas(80, 40, (5, 3, 16, 16, true), (40, 4, 5, 20, false), (60, 20, 14, 14, false));
        using (var classifier = RegionClassifier.For(model).Classes(classes).Build())
        {
            var found = classifier.Classify(canvas, new ComponentProposer());
            Check(found.Count == 3 && Enumerable.Range(0, 3).Select(found.Label).SequenceEqual(["ring", "bar", "square"]),
                $"labels: {string.Join(", ", Enumerable.Range(0, found.Count).Select(found.Label))}");
            Check(Enumerable.Range(0, 3).All(i => found.Confidence(i) > 0.9f) && found.Top(0, 2)[0].Class == "ring", "confident, with the top classes");
            Check(found.Boxes[1] == new PixelBox(40, 4, 5, 20), $"the bar's box: {found.Boxes[1]}");

            using var oneByOne = RegionClassifier.For(model).Classes(classes).BatchSize(1).Build();
            var again = oneByOne.Classify(canvas);
            Check(Enumerable.Range(0, 3).All(i => again.Label(i) == found.Label(i) && MathF.Abs(again.Confidence(i) - found.Confidence(i)) < 1e-5f), "batches of one agree");
        }

        string path = Path.Combine(Path.GetTempPath(), $"idrak-regions-{Guid.NewGuid():N}.ikm");
        try
        {
            using (var predictor = Predictor.For(model).InputShape(1, 28, 28).Softmax().Classes(classes).Build())
            {
                predictor.Save(path);
            }

            using var loaded = RegionClassifier.Load(path, device).Build();
            var found = loaded.Classify(canvas);
            Check(loaded.Classes.SequenceEqual(classes) && found.Label(2) == "square", "RegionClassifier.Load reads the package");
        }
        finally
        {
            File.Delete(path);
        }

        // ModelDetector: the decoder sees the outputs of each image and gives boxes in input pixels (8 x 8); they come back
        // scaled to the image (16 x 16), labelled, and suppressed.
        using var flatten = Network.Image(1, 8, 8).OnDevice(device).Flatten().Build();
        IReadOnlyList<int>? seenShape = null;
        IEnumerable<Detection> Decode(ReadOnlySpan<float> outputs, IReadOnlyList<int> shape)
        {
            seenShape = shape;
            float brightest = outputs.ToArray().Max();
            return
            [
                new(new BoundingBox(0, 0, 4, 4), 0, 0.9f * brightest),
                new(new BoundingBox(0.5f, 0.5f, 4, 4), 0, 0.8f),
                new(new BoundingBox(4, 4, 6, 6), 1, 0.7f),     // clipped at the image's edge
                new(new BoundingBox(1, 6, 1, 1), 1, 0.1f),     // below the minimum score
            ];
        }

        var detector = new ModelDetector(flatten, 1, 8, 8, Decode, new DetectorOptions { Classes = ["cat", "dog"] }, device);
        var image = new ImageData(Enumerable.Repeat(1f, 16 * 16).ToArray(), 1, 16, 16);
        var detections = detector.Detect(image);
        Check(seenShape is [64], $"the decoder sees one image's outputs: {string.Join(", ", seenShape ?? [])}");
        Check(detections.Count == 2 && detections[0].Box == new BoundingBox(0, 0, 8, 8) && detections[0].Label == "cat",
            $"scaled and labelled: {string.Join("; ", detections)}");
        Check(detections[1].Box == new BoundingBox(8, 8, 8, 8) && detections[1].Label == "dog", $"clipped: {detections[1].Box}");
        Check(detector.Detect([image, image]).All(d => d.Count == 2), "two images in one batch");
    }
}
