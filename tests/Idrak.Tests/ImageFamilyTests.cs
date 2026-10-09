// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Abstraction.Testing;
using Idrak.Data;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Onnx;
using Idrak.Vision;
using Idrak.Vision.Abstractions;

// Plan 13, step 6: image model families as plug-ins. The library registers no family and no decoder for an architecture;
// these tests register small families of their own (the outside plug-in tests register two checked against PyTorch).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] ImageFamilyGroup =
    [
        ("image families: the library registers no family; an unregistered architecture is refused naming ImageModelFamilies before any weight is opened; a config without an architecture, a folder without config.json and an ONNX file naming none are refused", ImageFamiliesRefuse),
        ("image families: a family of the test reads a safetensors folder with StoredWeights (Conv2d, BatchNorm, Linear kept bfloat16 as stored; float32 when asked), matches the same layers built by hand, carries labels, preprocessing and notes; a description without a decoder or network is refused", ImageFamiliesLoadTensors),
        ("image families: an ONNX file finds its family by its \"architecture\" and \"config\" metadata (or the options), its network imported once and owned by the model", ImageFamiliesLoadOnnx),
        ("image families: DetectionDecoders has only the generic \"boxes-scores\" (rows and columns, xyxy, xywh, cxcywh, normalized, logits); an unknown name is refused naming the registry; an app's decoder over it falls back under FallBack", ImageFamiliesDecoders),
        ("image families: ModelDetector with a family's preprocessing maps boxes back through the resize and the center crop, per image size in one call, and takes a decoder by name; ModelSegmenter with a preprocessor (a crop refused); predictors refuse the wrong task", ImageFamiliesPredictors),
        ("conformance kit: the image model family suite passes a consistent family and fails one whose outputs change between loads, one whose images leak into each other in a batch, and one that loads an invalid checkpoint", ImageFamilySuiteCatches),
    ];

    private const string TestFamily = "IdrakTestImageNet", OnnxFamily = "IdrakTestOnnxNet";

    // Values that bfloat16 holds exactly (multiples of 1/64 within ±1), so a bfloat16 checkpoint stores them as written.
    private static float[] Sixty4ths(Random random, int count) => [.. Enumerable.Range(0, count).Select(_ => random.Next(-64, 65) / 64f)];

    private static void ExpectRefusal<TException>(Action action, string text, string what) where TException : Exception
    {
        try
        {
            action();
        }
        catch (TException e)
        {
            Check(e.Message.Contains(text, StringComparison.Ordinal), $"{what}: \"{e.Message}\" does not say \"{text}\"");
            return;
        }

        Check(false, $"{what}: no {typeof(TException).Name}");
    }

    private static string TempFolder(string what)
    {
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-{what}-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        return folder;
    }

    private static void ImageFamiliesRefuse(Device device)
    {
        Check(ImageModelFamilies.Names.All(n => ImageModelFamilies.Origin(n) != Overrides.Library) && ImageModelFamilies.Default("ResNetForImageClassification") is null,
            $"the library registers no image model family ({string.Join(", ", ImageModelFamilies.Names)})");
        string folder = TempFolder("image-refuse");
        try
        {
            // No weights at all: the family is looked up first, so the refusal names the registry, not a missing file.
            File.WriteAllText(Path.Combine(folder, "config.json"), """{"architectures": ["NobodysNet"]}""");
            ExpectRefusal<NotSupportedException>(() => ImageModels.Load(folder), "Image model family 'NobodysNet' is not registered (registered: ", "unregistered");
            ExpectRefusal<NotSupportedException>(() => ImageModels.Load(folder), "register it with ImageModelFamilies.Register (a plug-in or the app that brings the family; the library registers none)", "the remedy");
            ExpectRefusal<NotSupportedException>(() => ImageModels.Load(folder, new ImageModelOptions { Architecture = "AlsoNobody" }), "'AlsoNobody' is not registered", "the options' architecture");
            File.WriteAllText(Path.Combine(folder, "config.json"), """{"model_type": "mystery"}""");
            ExpectRefusal<InvalidDataException>(() => ImageModels.Load(folder), "names no architecture", "a config without an architecture");
            File.Delete(Path.Combine(folder, "config.json"));
            ExpectRefusal<InvalidDataException>(() => ImageModels.Load(folder), "has no config.json", "a folder without config.json");

            using (var net = new Sequential(new Flatten(), new Linear(12, 2, device: Device.Cpu, random: new Random(1))))
            {
                OnnxExport.For(net).Input(3, 2, 2).Save(Path.Combine(folder, "plain.onnx"));
            }

            ExpectRefusal<InvalidDataException>(() => ImageModels.Load(Path.Combine(folder, "plain.onnx")), "names no architecture", "an ONNX file naming none");
            ExpectRefusal<NotSupportedException>(() => ImageModels.Load(folder, new ImageModelOptions { Architecture = "NobodysOnnx" }), "'NobodysOnnx' is not registered",
                "a folder holding one .onnx file is read as it");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    // A small convolutional classifier's tensors (PyTorch's names and layouts) and the same network built by hand.
    private static (List<(string Name, int[] Shape, float[] Values)> Tensors, Sequential Network) SmallClassifier(Device device)
    {
        var random = new Random(11);
        float[] conv = Sixty4ths(random, 4 * 3 * 3 * 3), convBias = Sixty4ths(random, 4), gamma = Sixty4ths(random, 4), beta = Sixty4ths(random, 4),
            mean = Sixty4ths(random, 4), variance = [.. Sixty4ths(random, 4).Select(v => 0.5f + MathF.Abs(v))], fc = Sixty4ths(random, 2 * 4), fcBias = Sixty4ths(random, 2);
        var tensors = new List<(string, int[], float[])>
        {
            ("conv.weight", [4, 3, 3, 3], conv), ("conv.bias", [4], convBias),
            ("bn.weight", [4], gamma), ("bn.bias", [4], beta), ("bn.running_mean", [4], mean), ("bn.running_var", [4], variance),
            ("fc.weight", [2, 4], fc), ("fc.bias", [2], fcBias),
        };
        var norm = new BatchNorm(4, device: device);
        norm.Gamma.Load(gamma);
        norm.Beta.Load(beta);
        norm.RunningMean.Load(mean);
        norm.RunningVariance.Load(variance);
        var transposed = new float[8];
        for (int o = 0; o < 2; o++)
        {
            for (int i = 0; i < 4; i++)
            {
                transposed[i * 2 + o] = fc[o * 4 + i];
            }
        }

        var network = new Sequential(
            Conv2d.FromWeights(Tensor.From(conv, [4, 3, 3, 3], device), Tensor.From(convBias, [4], device), 3, 1, 1), norm, new ReLU(), new GlobalAveragePool2d(),
            Linear.FromWeights(Tensor.From(transposed, [4, 2], device), Tensor.From(fcBias, [2], device)));
        return (tensors, network);
    }

    // The test's family: conv → batch norm → ReLU → global average pooling → fc, read with StoredWeights.
    private sealed class SmallFamily(string name, Func<ImageModel, ImageModel>? change = null) : IImageModelFamily
    {
        public string Name => name;

        public ImageModel Read(ImageCheckpoint checkpoint)
        {
            var tensors = checkpoint.RequireTensors();
            var network = new Sequential(
                StoredWeights.Conv2d(tensors, "conv", padding: (1, 1), device: checkpoint.Device), StoredWeights.BatchNorm(tensors, "bn", device: checkpoint.Device),
                new ReLU(), new GlobalAveragePool2d(), StoredWeights.Linear(tensors, "fc", checkpoint.Options.Weights, checkpoint.Device));
            checkpoint.Notes.Add("read by the test's family");
            var model = new ImageModel
            {
                Architecture = checkpoint.Architecture,
                Task = ImageTask.Classification,
                Network = network,
                Preprocessor = checkpoint.Preprocessor(),
                Labels = checkpoint.Labels(),
                Notes = ["a note of the family's own"],
            };
            return change is null ? model : change(model);
        }
    }

    private static void ImageFamiliesLoadTensors(Device device)
    {
        string folder = TempFolder("image-tensors");
        var (tensors, reference) = SmallClassifier(device);
        try
        {
            SafeTensorsWriter.Write(Path.Combine(folder, "model.safetensors"), tensors, SafeTensorType.BF16);
            File.WriteAllText(Path.Combine(folder, "config.json"), $$$"""{"architectures": ["{{{TestFamily}}}"], "id2label": {"1": "dog", "0": "cat"}}""");
            File.WriteAllText(Path.Combine(folder, "preprocessor_config.json"), """{"do_resize": true, "size": {"height": 8, "width": 8}, "resample": 2, "do_normalize": false}""");
            ImageModelFamilies.Register(new SmallFamily(TestFamily));
            ImageModelFamilies.Register(new SmallFamily("IdrakTestNoDecoder", m => m with { Task = ImageTask.Detection }));
            ImageModelFamilies.Register(new SmallFamily("IdrakTestNoLabels", m => m with { Labels = [] }));
            try
            {
                var options = new ImageModelOptions { Device = device };
                using var model = ImageModels.Load(folder, options);
                Check(model.Architecture == TestFamily && model.Task == ImageTask.Classification && model.Labels!.SequenceEqual(["cat", "dog"])
                      && model.InputShape!.SequenceEqual([3, 8, 8]) && model.Preprocessor is { Height: 8, Normalize: false }, "the description");
                Check(model.Notes.SequenceEqual(["read by the test's family", "a note of the family's own"]), $"notes: {string.Join("; ", model.Notes)}");
                Check(!model.Network.IsTraining, "the network is in evaluation mode");
                var fc = (Linear)model.Network.Children().Last();
                Check(fc.BFloat16 is not null, "a bfloat16 checkpoint's fc weight stays bfloat16 (as stored)");

                var x = Tensor.From(RandomArray(new Random(3), 2 * 3 * 8 * 8), [2, 3, 8, 8], device);
                AssertClose(reference.Predict(x).ToArray(), model.Network.Predict(x).ToArray(), 1e-5f, "the family's network and the same layers by hand");
                using (var wide = ImageModels.Load(Path.Combine(folder, "model.safetensors"), options with { Weights = EncoderWeights.Float32 }))
                {
                    Check(((Linear)wide.Network.Children().Last()).BFloat16 is null, "float32 asked for");
                    AssertClose(reference.Predict(x).ToArray(), wide.Network.Predict(x).ToArray(), 1e-5f, "float32 weights; a .safetensors file names its folder");
                }

                ExpectRefusal<InvalidOperationException>(() => ImageModels.Load(folder, options with { Architecture = "IdrakTestNoDecoder" }), "a detector without a decoder name", "no decoder");
                ExpectRefusal<InvalidOperationException>(() => ImageModels.Load(folder, options with { Architecture = "IdrakTestNoLabels" }), "an empty list of labels", "empty labels");

                // A tensor missing: the error names it.
                SafeTensorsWriter.Write(Path.Combine(folder, "model.safetensors"), tensors.Where(t => t.Name != "bn.running_var"), SafeTensorType.F32);
                ExpectRefusal<InvalidDataException>(() => ImageModels.Load(folder, options), "'bn.running_var'", "a missing tensor");
            }
            finally
            {
                ImageModelFamilies.Unregister(TestFamily);
                ImageModelFamilies.Unregister("IdrakTestNoDecoder");
                ImageModelFamilies.Unregister("IdrakTestNoLabels");
            }

            Check(!ImageModelFamilies.Names.Contains(TestFamily), "unregistered");
        }
        finally
        {
            reference.Dispose();
            Directory.Delete(folder, recursive: true);
        }
    }

    // An ONNX family: the network imported as it is, labels from the config the file carries.
    private sealed class OnnxTestFamily : IImageModelFamily
    {
        public int Reads { get; private set; }

        public string Name => OnnxFamily;

        public ImageModel Read(ImageCheckpoint checkpoint)
        {
            Reads++;
            var first = checkpoint.RequireOnnx();
            Check(ReferenceEquals(first, checkpoint.RequireOnnx()), "the ONNX network is imported once");
            return new ImageModel
            {
                Architecture = checkpoint.Architecture,
                Task = ImageTask.Features,
                Network = first.Model,
                Preprocessor = checkpoint.Preprocessor(new ImagePreprocessor { Height = 8, Width = 8, Normalize = false }),
                Labels = checkpoint.Labels(),
            };
        }
    }

    private static void ImageFamiliesLoadOnnx(Device device)
    {
        string folder = TempFolder("image-onnx");
        var (_, network) = SmallClassifier(Device.Cpu);
        var family = new OnnxTestFamily();
        try
        {
            string file = Path.Combine(folder, "net.onnx");
            OnnxExport.For(network).Input(3, 8, 8).Metadata("architecture", OnnxFamily).Metadata("config", """{"id2label": {"0": "cat", "1": "dog"}}""").Save(file);
            ExpectRefusal<NotSupportedException>(() => ImageModels.Load(file), $"Image model family '{OnnxFamily}' is not registered", "unregistered ONNX family");
            ImageModelFamilies.Register(family);
            var x = Tensor.From(RandomArray(new Random(4), 3 * 3 * 8 * 8), [3, 3, 8, 8], Device.Cpu);
            var expected = network.Predict(x).ToArray();
            using (var model = ImageModels.Load(file, new ImageModelOptions { Device = device }))
            {
                Check(model.Task == ImageTask.Features && model.Labels!.SequenceEqual(["cat", "dog"]) && model.InputShape!.SequenceEqual([3, 8, 8]), "the description from the metadata");
                Check(model.Notes.Count == 0 || model.Notes.All(n => n.Length > 0), "the importer's notes carried");
                AssertClose(expected, model.Network.Predict(x.To(device)).ToArray(), 1e-5f, "the imported network");
                using var outputs = model.Outputs().Build();
                Check(outputs.Predict([new ImageData(new float[3 * 16 * 16], 3, 16, 16)])[0].Length == 2, "the outputs predictor");
            }

            // A config.json beside the file takes precedence over the metadata; the folder alone finds its one .onnx file.
            File.WriteAllText(Path.Combine(folder, "config.json"), $$$"""{"architectures": ["{{{OnnxFamily}}}"], "id2label": {"0": "left", "1": "right"}}""");
            using (var model = ImageModels.Load(folder, new ImageModelOptions { Device = device }))
            {
                Check(model.Labels!.SequenceEqual(["left", "right"]), "config.json beside the file");
            }

            Check(family.Reads == 2, $"read {family.Reads} times");
        }
        finally
        {
            ImageModelFamilies.Unregister(OnnxFamily);
            network.Dispose();
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void ImageFamiliesDecoders(Device device)
    {
        Check(DetectionDecoders.Names.Where(n => DetectionDecoders.Origin(n) == Overrides.Library).SequenceEqual([DetectionDecoders.BoxesScores]),
            $"the library's decoders: {string.Join(", ", DetectionDecoders.Names.Where(n => DetectionDecoders.Origin(n) == Overrides.Library))}");
        static IReadOnlyList<Detection> Decode(string name, DetectionDecoderContext context, float[] outputs, params int[] shape) =>
            [.. DetectionDecoders.Create(name, context)(outputs, shape)];

        float[] rows = [10, 20, 30, 40, 0.1f, 0.7f, 0, 0, 5, 5, 0.9f, 0.2f];
        var plain = new DetectionDecoderContext(0, 0, null, []);
        var found = Decode(DetectionDecoders.BoxesScores, plain, rows, 2, 6);
        Check(found.Count == 2 && found[0] == new Detection(new BoundingBox(10, 20, 20, 20), 1, 0.7f) && found[1] == new Detection(new BoundingBox(0, 0, 5, 5), 0, 0.9f),
            $"rows, xyxy: {string.Join(", ", found)}");
        float[] columns = new float[12];
        for (int r = 0; r < 2; r++)
        {
            for (int j = 0; j < 6; j++)
            {
                columns[j * 2 + r] = rows[r * 6 + j];
            }
        }

        Check(Decode(DetectionDecoders.BoxesScores, plain with { Settings = new JsonObject { ["layout"] = "columns" } }, columns, 6, 2).SequenceEqual(found), "columns");
        var xywh = Decode(DetectionDecoders.BoxesScores, plain with { Settings = new JsonObject { ["box"] = "xywh" } }, rows, 2, 6);
        Check(xywh[0].Box == new BoundingBox(10, 20, 30, 40), "xywh");
        var centre = Decode(DetectionDecoders.BoxesScores, new DetectionDecoderContext(100, 200, ["only"],
            new JsonObject { ["box"] = "cxcywh", ["normalized"] = true, ["scores"] = "logits" }), [0.5f, 0.5f, 0.1f, 0.2f, 2f], 1, 5);
        Check(centre.Single().Box == new BoundingBox(90, 40, 20, 20) && MathF.Abs(centre[0].Score - 1f / (1f + MathF.Exp(-2f))) < 1e-6f, $"cxcywh, normalized, logits: {centre[0]}");
        ExpectRefusal<ArgumentException>(() => Decode(DetectionDecoders.BoxesScores, new DetectionDecoderContext(0, 0, ["a", "b", "c"], []), rows, 2, 6), "for 3 classes", "classes disagree");
        ExpectRefusal<NotSupportedException>(() => DetectionDecoders.Create(DetectionDecoders.BoxesScores, plain with { Settings = new JsonObject { ["box"] = "polar" } }), "box 'polar'", "a box layout");
        ExpectRefusal<NotSupportedException>(() => DetectionDecoders.Create("yolo-of-nobody", plain), "Detection decoder 'yolo-of-nobody' is not registered", "an unknown decoder");
        ExpectRefusal<NotSupportedException>(() => DetectionDecoders.Create("yolo-of-nobody", plain), "register it with DetectionDecoders.Register", "the remedy");

        // An app's decoder over the library's: guarded, it falls back when it fails under FallBack.
        DetectionDecoders.Register(DetectionDecoders.BoxesScores, _ => (_, _) => throw new InvalidOperationException("the app's decoder fails"));
        try
        {
            ExpectRefusal<InvalidOperationException>(() => Decode(DetectionDecoders.BoxesScores, plain, rows, 2, 6), "the app's decoder fails", "Throw: the error reaches the caller");
            DetectionDecoders.SetPolicy(DetectionDecoders.BoxesScores, SlotPolicy.FallBack);
            Check(Decode(DetectionDecoders.BoxesScores, plain, rows, 2, 6).SequenceEqual(found), "FallBack: the library's detections");
        }
        finally
        {
            DetectionDecoders.SetPolicy(DetectionDecoders.BoxesScores, SlotPolicy.Throw);
            DetectionDecoders.Unregister(DetectionDecoders.BoxesScores);
        }

        Check(DetectionDecoders.Origin(DetectionDecoders.BoxesScores) == Overrides.Library, "the library's decoder is back");
    }

    // A network that answers every image with the same values (a [N, ...] tensor of them).
    private sealed class ConstantOutputs(float[] values, int[] shape) : Module
    {
        protected override Tensor ForwardCore(Tensor input) =>
            Tensor.From([.. Enumerable.Range(0, input.Shape[0]).SelectMany(_ => values)], [input.Shape[0], .. shape], input.Device);
    }

    private static void ImageFamiliesPredictors(Device device)
    {
        // One box (0, 0)-(32, 32) in the network's 32 x 32 input, score 0.9.
        using var boxes = new ConstantOutputs([0, 0, 32, 32, 0.9f], [1, 5]);
        var crop = new ImagePreprocessor { ShortestEdge = 40, CenterCrop = true, CropHeight = 32, CropWidth = 32, Normalize = false };
        var detector = new ModelDetector(boxes, crop, DetectionDecoders.BoxesScores, options: new DetectorOptions { MinScore = 0.5f, Classes = ["thing"] }, device: device);
        ImageData Blank(int width, int height) => new(new float[3 * width * height], 3, height, width);
        // 160 x 80 → 80 x 40 (shortest edge 40) → crop from (24, 4): the box is (24, 4, 32, 32) there, (48, 8, 64, 64) in the image.
        // 80 x 160 → 40 x 80 → crop from (4, 24): (8, 48, 64, 64).
        var found = detector.Detect([Blank(160, 80), Blank(80, 160), Blank(160, 80)]);
        Check(found.All(f => f.Count == 1 && f[0].Label == "thing"), "one labelled detection per image");
        Check(found[0][0].Box == new BoundingBox(48, 8, 64, 64) && found[2][0].Box == found[0][0].Box, $"wide image: {found[0][0].Box}");
        Check(found[1][0].Box == new BoundingBox(8, 48, 64, 64), $"tall image: {found[1][0].Box}");
        var resize = new ImagePreprocessor { Height = 32, Width = 32, Normalize = false };
        Check(new ModelDetector(boxes, resize, DetectionDecoders.BoxesScores, device: device).Detect(Blank(64, 48)).Single().Box == new BoundingBox(0, 0, 64, 48), "a resize alone");
        Check(new ModelDetector(boxes, 3, 32, 32, DetectionDecoders.BoxesScores, device: device).Detect(Blank(96, 96)).Single().Box == new BoundingBox(0, 0, 96, 96), "by name, without a preprocessor");
        ExpectRefusal<NotSupportedException>(() => _ = new ModelDetector(boxes, resize, "nobodys-decoder"), "'nobodys-decoder' is not registered", "an unknown decoder name");

        // A segmenter: class 1 on the left half of a 4 x 4 map, resized back to the image.
        var logits = new float[2 * 16];
        for (int y = 0; y < 4; y++)
        {
            for (int x = 0; x < 2; x++)
            {
                logits[16 + y * 4 + x] = 1f;
            }
        }

        using var map = new ConstantOutputs(logits, [2, 4, 4]);
        var mask = new ModelSegmenter(map, resize, device).Segment(Blank(8, 6));
        Check(mask.Width == 8 && mask.Height == 6 && mask.LabelAt(0, 0) == 1 && mask.LabelAt(3, 5) == 1 && mask.LabelAt(4, 0) == 0, "the mask, back to the image");
        ExpectRefusal<NotSupportedException>(() => _ = new ModelSegmenter(map, crop), "cannot center-crop", "a segmenter's crop");

        // The predictors check the task, the labels and that every image becomes one size.
        var classifier = new ImageModel { Architecture = "t", Task = ImageTask.Classification, Network = boxes, Preprocessor = resize };
        ExpectRefusal<InvalidOperationException>(() => classifier.Detector(), "not a detection one", "a classifier as a detector");
        ExpectRefusal<InvalidOperationException>(() => classifier.Classifier(), "names no classes", "a classifier without labels");
        ExpectRefusal<NotSupportedException>(() => (classifier with { Preprocessor = new ImagePreprocessor { ShortestEdge = 20 } }).Outputs(), "different sizes", "sizes that follow the image");
        ExpectRefusal<InvalidOperationException>(() => RegionClassifier.For(classifier), "with labels", "a region classifier without labels");
        Check(classifier.Classifier(["a", "b", "c", "d", "e"]) is not null && (classifier with { Task = ImageTask.Segmentation }).Segmenter() is not null, "labels given; a segmenter");
    }

    private static void ImageFamilySuiteCatches(Device device)
    {
        var image = new ImageData([.. Enumerable.Range(0, 3 * 5 * 7).Select(i => i % 11 / 10f)], 3, 5, 7);
        var samples = new[] { new ImageFamilySample("good", [image, new ImageData(new float[16], 1, 4, 4)]) };
        static float[] Mean(ImageData image) => [image.Pixels.Average(), image.Pixels.Max()];
        ImageModelUnderTest Wrap(Func<IReadOnlyList<ImageData>, IReadOnlyList<float[]>> predict) => new("features", null, predict);
        var options = new ContractCheckOptions { RandomCases = 3 };

        var good = Conformance.CheckImageModelFamily(_ => Wrap(images => [.. images.Select(Mean)]), samples, options);
        Check(good.Passed, $"a consistent family passes: {good}");

        int loads = 0;
        var changing = Conformance.CheckImageModelFamily(_ =>
        {
            float offset = Interlocked.Increment(ref loads);
            return Wrap(images => [.. images.Select(i => Mean(i).Select(v => v + offset).ToArray())]);
        }, samples, options);
        Check(changing.Failures.Any(f => f.Check == "the same outputs after loading again"), $"outputs that change between loads: {changing}");

        var leaking = Conformance.CheckImageModelFamily(_ => Wrap(images => [.. images.Select(i => Mean(i).Select(v => v + images.Count).ToArray())]), samples, options);
        Check(leaking.Failures.Any(f => f.Check == "each image's outputs alone as in the batch"), $"images leaking into each other: {leaking}");

        var lenient = Conformance.CheckImageModelFamily(_ => Wrap(images => [.. images.Select(Mean)]), [new ImageFamilySample("broken", Error: "not registered")], options);
        Check(lenient.Failures.Any(f => f.Check == "refuses an invalid checkpoint"), $"an invalid checkpoint loaded: {lenient}");

        var reference = Conformance.CheckImageModelFamily(_ => Wrap(images => [.. images.Select(Mean)]),
            [new ImageFamilySample("good", [image], [[0f, 0f]])], options with { RandomCases = 0 });
        Check(reference.Failures.Any(f => f.Check == "the reference's outputs"), $"outputs that are not the reference's: {reference}");
    }
}
