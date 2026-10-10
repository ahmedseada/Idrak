// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Data;
using Idrak.Data.Abstractions;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Onnx;
using Idrak.PluginTests;
using Idrak.Vision;
using Idrak.Vision.Abstractions;

// Plan 13, step 7: the command line for image models. idrak predict and idrak train with the families of the outside
// plug-in (tests/Idrak.PluginTests, loaded with -P as an app's plug-in is): classes, boxes and masks against the reference
// and the library; a classifier and a segmenter trained from builder JSON (augmentations, metrics, packages predicted
// again), the plug-in's classifier and grid detector fine-tuned (COCO training data, YOLO evaluation data, COCO's mAP);
// unregistered families, decoders and heads refused with their registry's message.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] CliVisionGroup =
    [
        ("cli vision: ImageEncoders write PNG and PGM that the codecs read back exactly (an unknown extension refused naming the registry); explain knows every step; kernels lists the vision operations", CliVisionBasics),
        ("cli vision: predict -P with the plug-in's families: the classifier's classes and probabilities as PyTorch's (--top, -j, --format csv, -o rows), the grid detector's boxes as PyTorch's (--threshold, --iou, --decoder by name), a segmenter's pixels and mask PNGs (--out) as the library's, a backbone's .npy features; an unregistered family or decoder exits 1 with its registry's message and -P", CliVisionPredict),
        ("cli vision: train a classifier on class folders with --augment (accuracy, run.json, log, the package predicted), fine-tune the plug-in's classifier (.ikw read by predict --weights); a segmenter on image and mask folders (--metric miou, --loss dice, mask PNGs predicted from the package)", CliVisionTrainClassesMasks),
        ("cli vision: fine-tune the plug-in's grid detector on a tiny COCO set evaluated on YOLO files (--metric coco reported, --matcher hungarian with --loss ciou, predict --weights); an unregistered family, head or loss refused", CliVisionTrainDetector),
    ];

    private static string ImagePluginPath => typeof(ImageFamilyPlugin).Assembly.Location;

    private static string[] FamilyImages => [.. new[] { "a.png", "b.png", "c.png" }.Select(n => TestData($"image-families/images/{n}"))];

    private static void CliVisionBasics(Device device)
    {
        _ = device;
        string folder = TempFolder("cli-vision-basics");
        try
        {
            // PNG (grey and colour) and binary PGM/PPM: the codecs read back the same 8-bit levels.
            var random = new Random(4);
            foreach (var (channels, name) in new[] { (1, "grey.png"), (3, "colour.png"), (1, "grey.pgm"), (3, "colour.ppm") })
            {
                var pixels = Enumerable.Range(0, channels * 5 * 7).Select(_ => random.Next(256) / 255f).ToArray();
                string path = Path.Combine(folder, name);
                ImageEncoders.Save(path, new ImageData(pixels, channels, 5, 7));
                var read = ImageCodecs.Decode(path);
                Check(read.Channels == channels && read.Height == 5 && read.Width == 7 && read.Pixels.Zip(pixels).All(p => MathF.Abs(p.First - p.Second) < 1e-6f),
                    $"{name}: read back as written");
            }

            Check(ImageEncoders.Names.Contains("png") && ImageEncoders.Origin("png") == Overrides.Library && ImageEncoders.Default("netpbm") is not null, "the library's encoders");
            ExpectRefusal<NotSupportedException>(() => ImageEncoders.Save(Path.Combine(folder, "x.tiff"), new ImageData(new float[4], 1, 2, 2)), "ImageEncoders.Register", "an unknown extension");

            // explain: the normalize step is known (no step built just to count it).
            string network = Path.Combine(folder, "normalized.json");
            File.WriteAllText(network, Network.Image(3, 8, 8).Normalize([0.5f, 0.5f, 0.5f], [0.25f, 0.25f, 0.25f]).Conv2d(4, 3, padding: 1).ReLU().GlobalAveragePool2d().Linear(2)
                .ToJson().ToJsonString());
            var explained = TrainCliJson("explain", network);
            Check(explained["unknownSteps"]!.AsArray().Count == 0, $"explain knows every step: {explained["unknownSteps"]}");

            // kernels: the convolution, resampling, CTC and detection loss operations of plan 13 are listed.
            var kernels = TrainCliJson("kernels", "-d", "cpu");
            var names = kernels["operations"]!.AsArray().Select(o => (string)o!["name"]!).ToHashSet(StringComparer.Ordinal);
            string[] expected = ["Convolution", "ConvolutionBackwardInput", "ConvolutionBackwardWeight", "AvgPool", "AvgPoolBackward", "Interpolate2d", "ResizeNormalize",
                "AdaptiveAvgPool", "AdaptiveMaxPool", "AdaptiveMaxPoolBackward", "CtcLoss", "CtcLossBackward", "BoxIouLoss", "SigmoidFocalLoss"];
            Check(expected.All(names.Contains), $"kernels lists {string.Join(", ", expected.Where(n => !names.Contains(n)))}");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    private static void CliVisionPredict(Device device)
    {
        string d = device.ToString(), families = TestData("image-families"), plugin = ImagePluginPath;
        string resnet = Path.Combine(families, "tiny-resnet"), detector = Path.Combine(families, "tiny-detector");
        var reference = JsonNode.Parse(File.ReadAllText(Path.Combine(families, "reference.json")))!;
        string[] images = FamilyImages;
        string folder = TempFolder("cli-vision-predict");
        ImageFamilyPlugin.Unregister();
        try
        {
            // Without the plug-in: the registry's message and the remedy, exit 1.
            var refused = TrainCli("predict", resnet, images[0], "-d", d);
            Check(refused.Code == 1 && refused.Err.Contains("Image model family 'OutsideTinyResNetForImageClassification' is not registered", StringComparison.Ordinal)
                  && refused.Err.Contains("-P", StringComparison.Ordinal), $"predict without the plug-in: {refused.Code} {refused.Err}");

            // The classifier: PyTorch's best class and its probability, the three most likely.
            var classified = TrainCliJson("predict", "-P", plugin, resnet, Path.Combine(families, "images"), "--top", "3", "-d", d);
            Check((string?)classified["task"] == "classification" && (int)classified["images"]! == 3, $"predict a classifier: {classified}");
            string[] labels = ["circle", "square", "triangle", "star", "ring"];
            var predictions = classified["predictions"]!.AsArray();
            for (int i = 0; i < 3; i++)
            {
                var logits = reference["classifier"]![Path.GetFileName(images[i])]!["logits"]!.AsArray().Select(v => (double)v!).ToArray();
                double max = logits.Max(), sum = logits.Sum(v => Math.Exp(v - max));
                int best = Array.IndexOf(logits, max);
                var row = predictions[i]!;
                Check(Path.GetFileName((string)row["file"]!) == Path.GetFileName(images[i]) && (string?)row["class"] == labels[best]
                      && Math.Abs((double)row["probability"]! - 1 / sum) < 1e-4 && row["top"]!.AsArray().Count == 3,
                    $"{images[i]}: {row.ToJsonString()}, PyTorch's best {labels[best]} at {1 / sum:F6}");
            }

            var csv = TrainCli(["predict", "-P", plugin, resnet, .. images, "--top", "2", "--format", "csv", "-d", d]);
            var lines = csv.Out.ReplaceLineEndings("\n").Split('\n', StringSplitOptions.RemoveEmptyEntries);
            Check(csv.Code == 0 && lines[0] == "File,Class,Probability,Top" && lines.Length == 4 && lines[1].StartsWith("a.png,", StringComparison.Ordinal),
                $"--format csv: {csv.Out} {csv.Err}");
            string rows = Path.Combine(folder, "classes.jsonl");
            var written = TrainCli("predict", "-P", plugin, resnet, Path.Combine(families, "images", "*.png"), "-o", rows, "-d", d);
            Check(written.Code == 0 && File.ReadAllLines(rows).Length == 3, $"-o rows from a pattern: {written.Out} {written.Err}");

            // The grid detector: PyTorch's decoded, clipped and suppressed boxes, from the family's decoder and by name.
            var detected = TrainCliJson(["predict", "-P", plugin, detector, .. images, "--threshold", "0.3", "--iou", "0.5", "-d", d]);
            CheckBoxes(detected, reference, images, "the family's decoder");
            DetectionDecoders.Register("cli-test-grid", GridDecoder.Create);
            var named = TrainCliJson(["predict", "-P", plugin, detector, .. images, "--threshold", "0.3", "--iou", "0.5", "--decoder", "cli-test-grid", "-d", d]);
            CheckBoxes(named, reference, images, "--decoder cli-test-grid");
            var table = TrainCli("predict", "-P", plugin, detector, images[0], "--threshold", "0.3", "-d", d);
            Check(table.Code == 0 && System.Text.RegularExpressions.Regex.IsMatch(table.Out, @"File +Label +Score +X +Y +Width +Height") && table.Out.ReplaceLineEndings("\n").Contains("boxes in 1 image\n", StringComparison.Ordinal), $"the boxes' table: {table.Out}");
            var unknown = TrainCli("predict", "-P", plugin, detector, images[0], "--decoder", "nobody", "-d", d);
            Check(unknown.Code == 1 && unknown.Err.Contains("Detection decoder 'nobody' is not registered", StringComparison.Ordinal) && unknown.Err.Contains("-P", StringComparison.Ordinal),
                $"an unregistered decoder: {unknown.Code} {unknown.Err}");
            var wrongTask = TrainCli("predict", "-P", plugin, resnet, images[0], "--decoder", "cli-test-grid", "-d", d);
            Check(wrongTask.Code == 2 && wrongTask.Err.Contains("--decoder is for detectors", StringComparison.Ordinal), $"--decoder on a classifier: {wrongTask.Err}");

            // A segmenter (the plug-in's ONNX family over a small network exported here): pixels per class and mask PNGs.
            string segmenter = Path.Combine(folder, "segmenter");
            Directory.CreateDirectory(segmenter);
            using (var network = Network.Image(3, 16, 16).OnDevice(Device.Cpu).Seed(5).Conv2d(4, 3, padding: 1).ReLU().Conv2d(3, 1).Build())
            {
                network.ExportOnnx(Path.Combine(segmenter, "model.onnx"), 3, 16, 16);
            }

            File.WriteAllText(Path.Combine(segmenter, "config.json"), new JsonObject
            {
                ["architectures"] = new JsonArray(ImageFamilyPlugin.Segmenter), ["image_size"] = new JsonArray(16, 16),
                ["id2label"] = new JsonObject { ["0"] = "background", ["1"] = "ink", ["2"] = "edge" },
            }.ToJsonString());
            string masks = Path.Combine(folder, "masks");
            var segmented = TrainCliJson(["predict", "-P", plugin, segmenter, .. images, "-o", masks, "-d", d]);
            using var model = ImageModels.Load(segmenter, new ImageModelOptions { Device = device });
            var library = model.Segmenter(device).Segment([.. images.Select(ImageCodecs.Decode)]);
            for (int i = 0; i < 3; i++)
            {
                var row = segmented["predictions"]![i]!;
                var mask = ImageCodecs.Decode((string)row["mask"]!);
                int[] levels = [.. mask.Pixels.Select(v => (int)MathF.Round(v * 255))];
                var counts = row["classes"]!.AsArray().ToDictionary(c => (int)c!["index"]!, c => (int)c!["pixels"]!);
                Check(mask.Width == library[i].Width && mask.Height == library[i].Height && levels.SequenceEqual(library[i].Labels.ToArray()),
                    $"{images[i]}: the mask PNG is the library's mask");
                Check(Enumerable.Range(0, 3).All(c => counts.GetValueOrDefault(c) == levels.Count(l => l == c)), $"{images[i]}: the pixels per class {row.ToJsonString()}");
            }

            var markdown = TrainCli("predict", "-P", plugin, segmenter, images[0], "--format", "md", "-d", d);
            Check(markdown.Code == 0 && markdown.Out.Contains("| File | Class | Pixels | Share |", StringComparison.Ordinal), $"--format md: {markdown.Out}");

            // A backbone (the plug-in's ONNX features family): .npy files with the network's outputs, or the values in -j.
            string backbone = Path.Combine(folder, "backbone");
            Directory.CreateDirectory(backbone);
            using (var network = Network.Image(3, 16, 16).OnDevice(Device.Cpu).Seed(6).Conv2d(4, 3, stride: 2, padding: 1).ReLU().Build())
            {
                network.ExportOnnx(Path.Combine(backbone, "model.onnx"), 3, 16, 16);
            }

            File.WriteAllText(Path.Combine(backbone, "config.json"), new JsonObject
            {
                ["architectures"] = new JsonArray(ImageFamilyPlugin.Backbone), ["image_size"] = new JsonArray(16, 16),
            }.ToJsonString());
            string features = Path.Combine(folder, "features");
            var described = TrainCliJson(["predict", "-P", plugin, backbone, .. images, "-o", features, "-d", d]);
            using var backboneModel = ImageModels.Load(backbone, new ImageModelOptions { Device = device });
            for (int i = 0; i < 3; i++)
            {
                var row = described["predictions"]![i]!;
                using var x = Tensor.From(backboneModel.Preprocessor.Pixels(ImageCodecs.Decode(images[i])), [1, 3, 16, 16], device);
                using var y = backboneModel.Network.Predict(x);
                float[] expected = y.ToArray();
                var (shape, values) = ReadNpy((string)row["npy"]!);
                Check((string?)described["task"] == "features" && shape.SequenceEqual([4, 8, 8]) && values.Length == expected.Length
                      && values.Zip(expected).All(p => MathF.Abs(p.First - p.Second) < 1e-5f), $"{images[i]}: the .npy file holds the network's features ({string.Join(", ", shape)})");
            }

            var inline = TrainCliJson("predict", "-P", plugin, backbone, images[0], "-d", d);
            Check(inline["predictions"]![0]!["values"]!.AsArray().Count == 4 * 8 * 8, "-j without --out gives the values");
            var featureCsv = TrainCli("predict", "-P", plugin, backbone, images[0], "--format", "csv", "-d", d);
            Check(featureCsv.Code == 0 && featureCsv.Out.ReplaceLineEndings("\n").Split('\n')[0] == "File,Shape,Size,Norm", $"features as CSV: {featureCsv.Out}");
        }
        finally
        {
            ImageFamilyPlugin.Unregister();
            DetectionDecoders.Unregister("cli-test-grid");
            Directory.Delete(folder, true);
        }
    }

    // A NumPy .npy file of float32 values (format 1.0, little-endian, C order): its shape and values.
    private static (int[] Shape, float[] Values) ReadNpy(string path)
    {
        var bytes = File.ReadAllBytes(path);
        Check(bytes.AsSpan(0, 8).SequenceEqual(new byte[] { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y', 1, 0 }), $"{path}: the .npy magic and version 1.0");
        int length = BitConverter.ToUInt16(bytes, 8);
        string header = System.Text.Encoding.ASCII.GetString(bytes, 10, length);
        Check(header.Contains("'descr': '<f4'", StringComparison.Ordinal) && header.Contains("'fortran_order': False", StringComparison.Ordinal) && (10 + length) % 64 == 0,
            $"{path}: header {header}");
        var dims = System.Text.RegularExpressions.Regex.Match(header, @"'shape': \(([^)]*)\)").Groups[1].Value;
        int[] shape = [.. dims.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(int.Parse)];
        var values = new float[(bytes.Length - 10 - length) / 4];
        Buffer.BlockCopy(bytes, 10 + length, values, 0, values.Length * 4);
        return (shape, values);
    }

    // A predict -j document's boxes against the reference's (x, y, width, height, class, score per box).
    private static void CheckBoxes(JsonObject document, JsonNode reference, string[] images, string what)
    {
        for (int i = 0; i < images.Length; i++)
        {
            var expected = reference["detector"]![Path.GetFileName(images[i])]!;
            float[] values = [.. expected["detections"]!.AsArray().Select(v => (float)v!)];
            var boxes = document["predictions"]![i]!["boxes"]!.AsArray();
            Check(boxes.Count == (int)expected["count"]!, $"{what}, {images[i]}: {boxes.Count} boxes, PyTorch keeps {expected["count"]}");
            for (int k = 0; k < boxes.Count; k++)
            {
                var box = boxes[k]!;
                float[] got = [(float)box["x"]!, (float)box["y"]!, (float)box["width"]!, (float)box["height"]!, (int)box["class"]!, (float)box["score"]!];
                Check(got.Select((v, j) => MathF.Abs(v - values[6 * k + j])).Max() < 2e-3f, $"{what}, {images[i]} box {k}: {box.ToJsonString()}");
            }
        }
    }

    // A grey image of 8 x 8 bright on its left or right half, with noise, written as PGM through the library's encoder.
    private static void WriteHalf(string path, bool left, Random random)
    {
        var pixels = new float[64];
        for (int y = 0; y < 8; y++)
        {
            for (int x = 0; x < 8; x++)
            {
                bool bright = left ? x < 4 : x >= 4;
                pixels[y * 8 + x] = Math.Clamp((bright ? 200 : 40) + random.Next(-30, 30), 0, 255) / 255f;
            }
        }

        ImageEncoders.Save(path, new ImageData(pixels, 1, 8, 8));
    }

    private static void CliVisionTrainClassesMasks(Device device)
    {
        string d = device.ToString(), folder = TempFolder("cli-vision-train"), cache = Path.Combine(folder, "cache");
        try
        {
            // A classifier from builder JSON on class folders, augmented (vertical flips keep the halves; brightness).
            var random = new Random(9);
            foreach (string label in new[] { "left", "right" })
            {
                for (int n = 0; n < 24; n++)
                {
                    WriteHalf(Path.Combine(folder, "halves", label, $"{n}.pgm"), label == "left", random);
                }
            }

            string spec = Path.Combine(folder, "cnn.json");
            File.WriteAllText(spec, Network.Image(1, 8, 8).Conv2d(4, 3, padding: 1).ReLU().MaxPool2d(2).Flatten().Linear(2).Named("halves").ToJson().ToJsonString());
            string package = Path.Combine(folder, "halves.ikm"), run = Path.Combine(folder, "run-halves");
            var trained = TrainCliJson("train", spec, "--data", Path.Combine(folder, "halves"), "--augment", "flip(horizontal=false, vertical=true), color-jitter(brightness=0.1)",
                "--epochs", "30", "--lr", "0.01", "--batch", "8", "-o", package, "--run", run, "--cache", cache, "-d", d);
            Check((string?)trained["task"] == "classification" && (string?)trained["metric"] == "accuracy" && (double)trained["metrics"]!["accuracy"]! >= 0.8
                  && File.Exists(package), $"train with --augment: {trained}");
            var runJson = JsonNode.Parse(File.ReadAllText(Path.Combine(run, "run.json")))!;
            Check((string?)runJson["image"]?["augment"] == "flip(horizontal=false, vertical=true), color-jitter(brightness=0.1)" && File.Exists(Path.Combine(run, "best.ikw"))
                  && File.ReadAllLines(Path.Combine(run, "log.jsonl")).Length >= 3, "run.json keeps the image options; checkpoints and log written");
            var predicted = TrainCliJson("predict", package, "-i", Path.Combine(folder, "halves", "right"), "-d", d);
            int right = predicted["predictions"]!.AsArray().Count(r => (string?)r!["prediction"] == "right");
            Check(right >= 20, $"the package predicts the right halves: {right} of 24");
            var shown = TrainCliJson("runs", "show", run);
            Check((int)shown["epochs"]! > 0 && shown["history"]!.AsArray().Count == (int)shown["epochs"]!, $"runs show reads the image run's log: {shown}");
            var badAugment = TrainCli("train", spec, "--data", Path.Combine(folder, "halves"), "--augment", "mosaic", "--cache", cache, "-d", d);
            Check(badAugment.Code == 2 && badAugment.Err.Contains("one class per image", StringComparison.Ordinal), $"mosaic on a classifier: {badAugment.Err}");
            var badLoss = TrainCli("train", spec, "--data", Path.Combine(folder, "halves"), "--loss", "nope", "--cache", cache, "-d", d);
            Check(badLoss.Code == 2 && badLoss.Err.Contains("--loss nope: use cross-entropy, ", StringComparison.Ordinal) && badLoss.Err.Contains("focal", StringComparison.Ordinal),
                $"an unknown loss names the registered ones: {badLoss.Err}");

            // Fine-tune the plug-in's classifier on folders named after two of its labels; predict reads the .ikw over the checkpoint.
            string resnet = TestData("image-families/tiny-resnet"), shapes = Path.Combine(folder, "shapes");
            var images = FamilyImages;
            for (int n = 0; n < 4; n++)
            {
                foreach (var (label, image) in new[] { ("circle", images[0]), ("square", images[1]) })
                {
                    Directory.CreateDirectory(Path.Combine(shapes, label));
                    File.Copy(image, Path.Combine(shapes, label, $"{n}.png"));
                }
            }

            string tuned = Path.Combine(folder, "tuned.ikw");
            try
            {
                var fine = TrainCliJson("train", "-P", ImagePluginPath, resnet, "--data", shapes, "--epochs", "2", "--batch", "4", "--lr", "0.001", "--validation", "0",
                    "-o", tuned, "--run", Path.Combine(folder, "run-tuned"), "--cache", cache, "-d", d);
                Check((string?)fine["task"] == "classification" && fine["classes"]!.AsArray().Count == 5 && File.Exists(tuned), $"fine-tune the plug-in's classifier: {fine}");
                var before = TrainCliJson("predict", "-P", ImagePluginPath, resnet, images[0], "-d", d);
                var after = TrainCliJson("predict", "-P", ImagePluginPath, resnet, images[0], "--weights", tuned, "-d", d);
                Check((double)before["predictions"]![0]!["probability"]! != (double)after["predictions"]![0]!["probability"]!, "--weights reads the fine-tuned weights");
            }
            finally
            {
                ImageFamilyPlugin.Unregister();
            }

            // A segmenter from builder JSON on images/ and masks/ (a bright rectangle is class 1), scored by mean IoU.
            string scans = Path.Combine(folder, "scans");
            var truth = new Dictionary<string, int[]>();
            for (int n = 0; n < 24; n++)
            {
                int x0 = random.Next(0, 9), y0 = random.Next(0, 9), w = random.Next(4, 8), h = random.Next(4, 8);
                var pixels = new float[3 * 256];
                var classes = new int[256];
                for (int y = 0; y < 16; y++)
                {
                    for (int x = 0; x < 16; x++)
                    {
                        bool inside = x >= x0 && x < x0 + w && y >= y0 && y < y0 + h;
                        classes[y * 16 + x] = inside ? 1 : 0;
                        for (int c = 0; c < 3; c++)
                        {
                            pixels[c * 256 + y * 16 + x] = Math.Clamp((inside ? 210 : 50) + random.Next(-25, 25), 0, 255) / 255f;
                        }
                    }
                }

                ImageEncoders.Save(Path.Combine(scans, "images", $"{n}.png"), new ImageData(pixels, 3, 16, 16));
                ImageEncoders.Save(Path.Combine(scans, "masks", $"{n}.png"), new ImageData([.. classes.Select(c => c / 255f)], 1, 16, 16));
                truth[$"{n}"] = classes;
            }

            File.WriteAllLines(Path.Combine(scans, "classes.txt"), ["background", "box"]);
            string unet = Path.Combine(folder, "segmenter.json"), segmenter = Path.Combine(folder, "scans.ikm");
            File.WriteAllText(unet, Network.Image(3, 16, 16).Conv2d(8, 3, padding: 1).ReLU().Conv2d(2, 1).Named("scans").ToJson().ToJsonString());
            var segmented = TrainCliJson("train", unet, "--data", scans, "--epochs", "25", "--lr", "0.01", "--batch", "8", "--metric", "miou", "-o", segmenter,
                "--cache", cache, "-d", d);
            Check((string?)segmented["task"] == "segmentation" && (double)segmented["metrics"]!["miou"]! > 0.6 && segmented["metrics"]!["iou/box"] is not null,
                $"train a segmenter: {segmented}");
            var dice = TrainCliJson("train", unet, "--data", scans, "--data-format", "masks", "--loss", "dice", "--epochs", "2", "--batch", "8", "-o", Path.Combine(folder, "dice.ikm"),
                "--cache", cache, "-d", d);
            Check((string?)dice["task"] == "segmentation" && dice["metrics"]!["miou"] is not null, $"--loss dice: {dice}");

            string masks = Path.Combine(folder, "predicted");
            string[] some = [.. new[] { "0", "1", "2" }.Select(n => Path.Combine(scans, "images", $"{n}.png"))];
            var predictedMasks = TrainCliJson(["predict", segmenter, .. some, "-o", masks, "-d", d]);
            int agree = 0;
            for (int i = 0; i < 3; i++)
            {
                var mask = ImageCodecs.Decode((string)predictedMasks["predictions"]![i]!["mask"]!);
                agree += mask.Pixels.Select((v, k) => (int)MathF.Round(v * 255) == truth[$"{i}"][k] ? 1 : 0).Sum();
            }

            Check((string?)predictedMasks["task"] == "segmentation" && agree > 0.9 * 3 * 256, $"the package's masks: {agree} of {3 * 256} pixels right");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    // A 64 x 64 colour image with one or two of the tiny detector's shapes (square, disc, bar), and its objects.
    private static ImageAnnotations WriteShapes(string path, Random random)
    {
        var pixels = new float[3 * 64 * 64];
        Array.Fill(pixels, 0.1f);
        var objects = new List<ObjectAnnotation>();
        int count = random.Next(1, 3);
        for (int o = 0; o < count; o++)
        {
            int kind = random.Next(3), size = random.Next(12, 20);
            int width = kind == 2 ? size + 8 : size, height = kind == 2 ? 6 : size;
            int x0 = o == 0 ? random.Next(2, 28 - width / 2) : random.Next(34, 62 - width), y0 = random.Next(2, 62 - height);
            x0 = Math.Clamp(x0, 0, 64 - width);
            for (int y = y0; y < y0 + height; y++)
            {
                for (int x = x0; x < x0 + width; x++)
                {
                    float dx = x + 0.5f - (x0 + width / 2f), dy = y + 0.5f - (y0 + height / 2f);
                    if (kind == 1 && dx * dx + dy * dy > width * width / 4f)
                    {
                        continue;
                    }

                    for (int c = 0; c < 3; c++)
                    {
                        pixels[c * 4096 + y * 64 + x] = kind == c ? 0.9f : 0.3f;
                    }
                }
            }

            objects.Add(new ObjectAnnotation(new BoundingBox(x0, y0, width, height), kind));
        }

        ImageEncoders.Save(path, new ImageData(pixels, 3, 64, 64));
        return new ImageAnnotations(Path.GetFileName(path), 64, 64, objects);
    }

    private static void CliVisionTrainDetector(Device device)
    {
        string d = device.ToString(), folder = TempFolder("cli-vision-detector"), cache = Path.Combine(folder, "cache");
        string detector = TestData("image-families/tiny-detector");
        string[] classes = ["square", "disc", "bar"];
        try
        {
            // Training images and a COCO file beside them; evaluation images in YOLO's images/ and labels/.
            var random = new Random(13);
            string coco = Path.Combine(folder, "coco");
            var train = Enumerable.Range(0, 12).Select(n => WriteShapes(Path.Combine(coco, $"{n}.png"), random)).ToList();
            AnnotationFormats.Write(new AnnotatedDataset(classes, train), Path.Combine(coco, "train.json"), "coco");
            string yolo = Path.Combine(folder, "yolo");
            var test = Enumerable.Range(0, 6).Select(n => WriteShapes(Path.Combine(yolo, "images", $"{n}.png"), random)).ToList();
            AnnotationFormats.Write(new AnnotatedDataset(classes, test) { ImagesRoot = Path.Combine(yolo, "images") }, yolo, "yolo");

            // Without the plug-in: refused before training, naming the registry.
            ImageFamilyPlugin.Unregister();
            var refused = TrainCli("train", detector, "--data", Path.Combine(coco, "train.json"), "--cache", cache, "-d", d);
            Check(refused.Code == 1 && refused.Err.Contains("Image model family 'OutsideGridDetector' is not registered", StringComparison.Ordinal)
                  && refused.Err.Contains("-P", StringComparison.Ordinal), $"train without the plug-in: {refused.Code} {refused.Err}");

            string tuned = Path.Combine(folder, "detector.ikw"), run = Path.Combine(folder, "run");
            var trained = TrainCliJson("train", "-P", ImagePluginPath, detector, "--data", Path.Combine(coco, "train.json"), "--eval", yolo, "--metric", "coco",
                "--augment", "flip", "--epochs", "3", "--batch", "4", "--lr", "0.001", "-o", tuned, "--run", run, "--cache", cache, "-d", d);
            var metrics = trained["metrics"]!;
            Check((string?)trained["task"] == "detection" && (string?)trained["metric"] == "coco" && (int)trained["trainingImages"]! == 12 && (int)trained["evaluationImages"]! == 6
                  && metrics["map"] is JsonValue map && (double)map >= 0 && (double)map <= 1 && metrics["map50"] is not null && File.Exists(tuned),
                $"fine-tune the grid detector: {trained}");
            var runJson = JsonNode.Parse(File.ReadAllText(Path.Combine(run, "run.json")))!;
            Check((string?)runJson["image"]?["model"] == Path.GetFullPath(detector) && File.Exists(Path.Combine(run, "training.json")), "run.json names the fine-tuned model");
            var text = TrainCli("train", "-P", ImagePluginPath, detector, "--data", Path.Combine(coco, "train.json"), "--eval", yolo, "--matcher", "hungarian", "--loss", "ciou",
                "--epochs", "1", "--batch", "6", "-o", Path.Combine(folder, "set.ikw"), "--cache", cache, "-d", d);
            Check(text.Code == 0 && text.Out.Contains("matcher hungarian", StringComparison.Ordinal) && text.Out.Contains("Result    evaluation coco: map ", StringComparison.Ordinal),
                $"--matcher hungarian --loss ciou: {text.Out} {text.Err}");
            var boxes = TrainCliJson("predict", "-P", ImagePluginPath, detector, Path.Combine(yolo, "images"), "--weights", tuned, "--threshold", "0.05", "-d", d);
            Check((string?)boxes["task"] == "detection" && (int)boxes["images"]! == 6, $"predict --weights with the fine-tuned detector: {boxes}");

            // A builder network naming a decoder no plug-in gives a head for; a loss no registry holds.
            string spec = Path.Combine(folder, "grid.json");
            File.WriteAllText(spec, Network.Image(3, 64, 64).Conv2d(8, 3, stride: 8, padding: 1).ReLU().Conv2d(8, 1).ToJson().ToJsonString());
            var noHead = TrainCli("train", spec, "--data", Path.Combine(coco, "train.json"), "--decoder", "nobody-grid", "--epochs", "1", "--cache", cache, "-d", d);
            Check(noHead.Code == 1 && noHead.Err.Contains("Detection head 'nobody-grid' is not registered", StringComparison.Ordinal) && noHead.Err.Contains("-P", StringComparison.Ordinal),
                $"an unregistered head: {noHead.Code} {noHead.Err}");
            var noLoss = TrainCli("train", "-P", ImagePluginPath, detector, "--data", Path.Combine(coco, "train.json"), "--loss", "nope", "--cache", cache, "-d", d);
            Check(noLoss.Code == 2 && noLoss.Err.Contains("--loss nope: use ", StringComparison.Ordinal) && noLoss.Err.Contains("giou", StringComparison.Ordinal), $"an unknown loss: {noLoss.Err}");
        }
        finally
        {
            ImageFamilyPlugin.Unregister();
            Directory.Delete(folder, true);
        }
    }
}
