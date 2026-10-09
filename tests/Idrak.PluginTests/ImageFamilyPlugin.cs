// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Data;
using Idrak.Data.Abstractions;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Vision;
using Idrak.Vision.Abstractions;

namespace Idrak.PluginTests;

/// <summary>
/// Two image model families registered from outside the library (plan 13, step 6), checked against PyTorch
/// (tests/Idrak.Tests/data/image-families, written by tools/pytorch/image_families_reference.py):
/// <list type="bullet">
/// <item><see cref="TinyResNetFamily"/>: a ResNet-style classifier read from safetensors and config.json with torchvision's
/// tensor names, its layers built here (basic residual blocks are this assembly's own module), the weights read one tensor at
/// a time, the fc weight kept as bfloat16 as stored;</item>
/// <item><see cref="GridDetectorFamily"/>: an anchor-free grid detector read from ONNX through Idrak's importer, with its own
/// decoder (<see cref="GridDecoder"/>, registered in <see cref="DetectionDecoders"/>) and the library's non-maximum
/// suppression.</item>
/// </list>
/// The library registers neither: unregistered, their checkpoints are refused naming the registry.
/// </summary>
public static class ImageFamilyPluginTests
{
    /// <summary>The classifier family's architecture name.</summary>
    public const string Classifier = "OutsideTinyResNetForImageClassification";

    /// <summary>The detector family's architecture name.</summary>
    public const string Detector = "OutsideGridDetector";

    /// <summary>The detector's decoder name.</summary>
    public const string Decoder = "outside-grid";

    private static readonly string[] ImageNames = ["a.png", "b.png", "c.png"];

    /// <summary>
    /// The classifier family: refused unregistered; registered, it loads (pixel values and logits as PyTorch's, the fc weight
    /// as bfloat16), classifies through a predictor and a region classifier, and passes the testing kit.
    /// </summary>
    public static void TinyResNet(Device device)
    {
        string folder = Path.Combine(DataFolder(), "tiny-resnet");
        var reference = Reference()["classifier"]!.AsObject();
        var images = ImageNames.Select(n => ImageCodecs.Decode(Path.Combine(DataFolder(), "images", n))).ToList();
        var options = new ImageModelOptions { Device = device };
        Refused(() => ImageModels.Load(folder, options), Classifier);

        ImageModelFamilies.Register(new TinyResNetFamily());
        try
        {
            Check(ImageModelFamilies.Origin(Classifier) == typeof(TinyResNetFamily).Assembly.GetName().Name && ImageModelFamilies.Default(Classifier) is null,
                "registered from this assembly, with no library default");
            using var model = ImageModels.Load(folder, options);
            Check(model.Task == ImageTask.Classification && model.Labels!.SequenceEqual(["circle", "square", "triangle", "star", "ring"])
                  && model.InputShape!.SequenceEqual([3, 32, 32]) && model.Preprocessor is { ShortestEdge: 36, CenterCrop: true, Resampling: ImageResampling.Bicubic },
                $"the description: {model.Task}, [{string.Join(", ", model.Labels ?? [])}], [{string.Join(", ", model.InputShape ?? [])}]");
            var fc = model.Network.Children().OfType<Linear>().Single();
            Check(fc.BFloat16 is not null, "the fc weight, stored as bfloat16, stays bfloat16");
            Check(model.Notes.Any(n => n.Contains("float32", StringComparison.Ordinal)), $"the notes say what was widened: {string.Join("; ", model.Notes)}");

            for (int i = 0; i < images.Count; i++)
            {
                CheckClose(Floats(reference[ImageNames[i]]!["pixel_values"]!), model.Preprocessor.Pixels(images[i]), 1e-6f, $"{ImageNames[i]}: pixel values");
            }

            using var outputs = model.Outputs().Build();
            var logits = outputs.Predict(images);
            for (int i = 0; i < images.Count; i++)
            {
                CheckClose(Floats(reference[ImageNames[i]]!["logits"]!), logits[i], 1e-4f, $"{ImageNames[i]}: logits");
            }

            using var classifier = model.Classifier().Build();
            var predictions = classifier.Predict(images);
            for (int i = 0; i < images.Count; i++)
            {
                var expected = Floats(reference[ImageNames[i]]!["logits"]!);
                int best = Array.IndexOf(expected, expected.Max());
                Check(predictions[i].Index == best && predictions[i].Class == model.Labels![best] && MathF.Abs(predictions[i].Scores.Sum(s => s.Score) - 1f) < 1e-5f,
                    $"{ImageNames[i]}: classified as {predictions[i].Class}, PyTorch's best is {model.Labels![best]}");
            }

            // Float32 weights throughout: the same values, so the same logits.
            using (var wide = ImageModels.Load(folder, options with { Weights = EncoderWeights.Float32 }))
            using (var wideOutputs = wide.Outputs().Build())
            {
                Check(wide.Network.Children().OfType<Linear>().Single().BFloat16 is null, "Float32 asked for: the fc weight is float32");
                var wideLogits = wideOutputs.Predict(images);
                for (int i = 0; i < images.Count; i++)
                {
                    CheckClose(logits[i], wideLogits[i], 1e-5f, $"{ImageNames[i]}: float32 weights");
                }
            }

            // The region classifier frames regions in the model's square input, rescaled and normalized as its preprocessing.
            using (var regions = RegionClassifier.For(model).Build())
            {
                var found = regions.Classify(regions.Foreground(images[0]), [new PixelBox(4, 4, 20, 16), new PixelBox(0, 0, 48, 40)]);
                Check(found.Count == 2 && found.Classes.SequenceEqual(model.Labels!)
                      && Enumerable.Range(0, 2).All(r => MathF.Abs(found.Probabilities(r).ToArray().Sum() - 1f) < 1e-5f), "the region classifier's probabilities");
            }

            // The testing kit: the reference's logits, consistency over random images; a wrong config and an unregistered name refused.
            string broken = CopyWith(folder, c => c["hidden_sizes"] = new JsonArray(8, 32));
            string unknown = CopyWith(folder, c => c["architectures"] = new JsonArray("NobodysResNet"));
            try
            {
                var samples = new[]
                {
                    new ImageFamilySample(folder, images, [.. ImageNames.Select(n => Floats(reference[n]!["logits"]!))]),
                    new ImageFamilySample(broken, Error: "layer2.0.conv1.weight"),
                    new ImageFamilySample(unknown, Error: "Image model family 'NobodysResNet' is not registered"),
                };
                Conformance.CheckImageModelFamily(path =>
                {
                    var loaded = ImageModels.Load(path, options);
                    var predictor = loaded.Outputs().Build();
                    return new ImageModelUnderTest("classification", loaded.Labels, predictor.Predict, loaded);
                }, samples).ThrowIfFailed();
            }
            finally
            {
                Directory.Delete(broken, recursive: true);
                Directory.Delete(unknown, recursive: true);
            }
        }
        finally
        {
            ImageModelFamilies.Unregister(Classifier);
        }

        Refused(() => ImageModels.Load(folder, options), Classifier);
    }

    /// <summary>
    /// The detector family: refused unregistered; registered, it imports the ONNX network (outputs as PyTorch's), refuses to
    /// detect until its decoder is registered, then gives PyTorch's decoded and suppressed boxes, from the folder and from
    /// the bare .onnx file (architecture from its metadata), and passes the testing kit.
    /// </summary>
    public static void GridDetector(Device device)
    {
        string folder = Path.Combine(DataFolder(), "tiny-detector");
        var reference = Reference()["detector"]!.AsObject();
        var images = ImageNames.Select(n => ImageCodecs.Decode(Path.Combine(DataFolder(), "images", n))).ToList();
        var options = new ImageModelOptions { Device = device };
        var filter = new DetectorOptions { MinScore = 0.3f, IouThreshold = 0.5f };
        Refused(() => ImageModels.Load(folder, options), Detector);

        ImageModelFamilies.Register(new GridDetectorFamily());
        string bare = Path.Combine(Path.GetTempPath(), $"idrak-outside-detector-{Guid.NewGuid():N}");
        try
        {
            using var model = ImageModels.Load(folder, options);
            Check(model.Task == ImageTask.Detection && model.Decoder == Decoder && model.Labels!.SequenceEqual(["square", "disc", "bar"])
                  && model.InputShape!.SequenceEqual([3, 64, 64]), "the description");
            using (var outputs = model.Outputs().Build())
            {
                var raw = outputs.Predict(images);
                for (int i = 0; i < images.Count; i++)
                {
                    CheckClose(Floats(reference[ImageNames[i]]!["outputs"]!), raw[i], 1e-4f, $"{ImageNames[i]}: the network's outputs");
                }
            }

            try
            {
                model.Detector(filter);
                Check(false, "a detector whose decoder is not registered is refused");
            }
            catch (NotSupportedException e)
            {
                Check(e.Message.Contains($"Detection decoder '{Decoder}' is not registered", StringComparison.Ordinal) && e.Message.Contains("DetectionDecoders.Register", StringComparison.Ordinal),
                    $"refused naming the registry: {e.Message}");
            }

            DetectionDecoders.Register(Decoder, GridDecoder.Create);
            Check(DetectionDecoders.Origin(Decoder) == typeof(GridDecoder).Assembly.GetName().Name && DetectionDecoders.Default(Decoder) is null, "the decoder is this assembly's");
            var detector = model.Detector(filter);
            var found = detector.Detect(images);
            for (int i = 0; i < images.Count; i++)
            {
                CheckDetections(reference[ImageNames[i]]!, found[i], ImageNames[i]);
                Check(found[i].All(d => d.Label == model.Labels![d.Class]), $"{ImageNames[i]}: labelled by the model's labels");
                CheckDetections(reference[ImageNames[i]]!, detector.Detect(images[i]), $"{ImageNames[i]} alone");
            }

            // The .onnx file alone: its "architecture" metadata finds the family, which falls back on its own defaults.
            Directory.CreateDirectory(bare);
            File.Copy(Path.Combine(folder, "model.onnx"), Path.Combine(bare, "model.onnx"));
            using (var alone = ImageModels.Load(Path.Combine(bare, "model.onnx"), options))
            {
                Check(alone.Labels is null && alone.InputShape!.SequenceEqual([3, 64, 64]), "the bare file: no labels, the family's default preprocessing");
                var bareFound = alone.Detector(filter).Detect(images);
                for (int i = 0; i < images.Count; i++)
                {
                    CheckDetections(reference[ImageNames[i]]!, bareFound[i], $"{ImageNames[i]} from the bare .onnx file");
                }
            }

            string unknown = CopyWith(folder, c => c["architectures"] = new JsonArray("NobodysDetector"));
            try
            {
                var samples = new[]
                {
                    new ImageFamilySample(folder, images, [.. ImageNames.Select(n => Floats(reference[n]!["detections"]!))]),
                    new ImageFamilySample(unknown, Error: "Image model family 'NobodysDetector' is not registered"),
                };
                Conformance.CheckImageModelFamily(path =>
                {
                    var loaded = ImageModels.Load(path, options);
                    var boxes = loaded.Detector(filter);
                    return new ImageModelUnderTest("detection", loaded.Labels, batch => [.. boxes.Detect(batch).Select(Flatten)], loaded);
                }, samples).ThrowIfFailed();
            }
            finally
            {
                Directory.Delete(unknown, recursive: true);
            }
        }
        finally
        {
            DetectionDecoders.Unregister(Decoder);
            ImageModelFamilies.Unregister(Detector);
            if (Directory.Exists(bare))
            {
                Directory.Delete(bare, recursive: true);
            }
        }

        Refused(() => ImageModels.Load(folder, options), Detector);
    }

    // Detections as PyTorch's rows: [x, y, width, height, class, score] each.
    private static float[] Flatten(IReadOnlyList<Detection> detections) =>
        [.. detections.SelectMany(d => new[] { d.Box.X, d.Box.Y, d.Box.Width, d.Box.Height, d.Class, d.Score })];

    private static void CheckDetections(JsonNode expected, IReadOnlyList<Detection> found, string what)
    {
        int count = (int)expected["count"]!;
        Check(found.Count == count, $"{what}: {found.Count} detections, PyTorch keeps {count}");
        CheckClose(Floats(expected["detections"]!), Flatten(found), 1e-4f, $"{what}: detections");
    }

    private static void Refused(Func<ImageModel> load, string architecture)
    {
        try
        {
            load().Dispose();
            Check(false, $"an unregistered image model family ({architecture}) is refused");
        }
        catch (NotSupportedException e)
        {
            Check(e.Message.Contains($"Image model family '{architecture}' is not registered", StringComparison.Ordinal)
                  && e.Message.Contains("ImageModelFamilies.Register", StringComparison.Ordinal) && e.Message.Contains("the library registers none", StringComparison.Ordinal),
                $"refused naming the registry: {e.Message}");
        }
    }

    // A copy of a checkpoint folder with its config.json changed.
    private static string CopyWith(string source, Action<JsonObject> change)
    {
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-image-family-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        foreach (string file in Directory.GetFiles(source))
        {
            File.Copy(file, Path.Combine(folder, Path.GetFileName(file)));
        }

        var config = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "config.json")))!.AsObject();
        change(config);
        File.WriteAllText(Path.Combine(folder, "config.json"), config.ToJsonString());
        return folder;
    }

    private static JsonObject Reference() => JsonNode.Parse(File.ReadAllText(Path.Combine(DataFolder(), "reference.json")))!.AsObject();

    private static float[] Floats(JsonNode node) => [.. node.AsArray().Select(v => (float)v!)];

    // tests/Idrak.Tests/data/image-families, found from the build output folder.
    private static string DataFolder()
    {
        for (var directory = new DirectoryInfo(AppContext.BaseDirectory); directory is not null; directory = directory.Parent)
        {
            string data = Path.Combine(directory.FullName, "tests", "Idrak.Tests", "data", "image-families");
            if (Directory.Exists(data))
            {
                return data;
            }
        }

        throw new DirectoryNotFoundException("tests/Idrak.Tests/data/image-families");
    }

    private static void Check(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }

    private static void CheckClose(float[] expected, float[] actual, float tolerance, string what)
    {
        if (Comparisons.Difference(expected, actual, tolerance) is { } difference)
        {
            throw new InvalidOperationException($"{what}: {difference}");
        }
    }
}

/// <summary>
/// A ResNet-style classifier family with torchvision's tensor names (conv1, bn1, layer{i}.{j}.conv1 / bn1 / conv2 / bn2 /
/// downsample.0 / downsample.1, fc) and a config.json giving "embedding_size", "hidden_sizes", "depths" and "id2label": the
/// stem (3x3 convolution, batch norm, ReLU, 3x3 max pool of stride 2), the stages of basic blocks (the first block of each
/// stage after the first strided by 2, with a 1x1 strided downsample), global average pooling and the fc layer. Every
/// tensor is read once, as its layer is made.
/// </summary>
public sealed class TinyResNetFamily : IImageModelFamily
{
    /// <inheritdoc />
    public string Name => ImageFamilyPluginTests.Classifier;

    /// <inheritdoc />
    public ImageModel Read(ImageCheckpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        var tensors = checkpoint.RequireTensors();
        var config = checkpoint.Config;
        var device = checkpoint.Device;
        int stem = (int?)config["embedding_size"] ?? 64;
        int[] widths = [.. (config["hidden_sizes"]?.AsArray() ?? throw new InvalidDataException("config.json has no hidden_sizes.")).Select(v => (int)v!)];
        int[] depths = [.. (config["depths"]?.AsArray() ?? throw new InvalidDataException("config.json has no depths.")).Select(v => (int)v!)];
        float epsilon = (float?)config["layer_norm_eps"] ?? 1e-5f;
        var layers = new List<Module>();
        try
        {
            layers.Add(Expect(StoredWeights.Conv2d(tensors, "conv1", padding: (1, 1), device: device), stem, "conv1.weight"));
            layers.Add(StoredWeights.BatchNorm(tensors, "bn1", epsilon, device: device));
            layers.Add(new ReLU());
            layers.Add(new MaxPool2d(3, 2, 1));
            int channels = stem;
            for (int stage = 0; stage < widths.Length; stage++)
            {
                for (int block = 0; block < depths[stage]; block++)
                {
                    int stride = stage > 0 && block == 0 ? 2 : 1;
                    layers.Add(BasicBlock.Read(tensors, $"layer{stage + 1}.{block}", channels, widths[stage], stride, epsilon, device));
                    channels = widths[stage];
                }
            }

            layers.Add(new GlobalAveragePool2d());
            layers.Add(StoredWeights.Linear(tensors, "fc", checkpoint.Options.Weights, device));
        }
        catch
        {
            foreach (var layer in layers)
            {
                layer.Dispose();
            }

            throw;
        }

        checkpoint.Notes.Add("convolutions and norms hold float32 copies of the stored values (they have no narrower form)");
        return new ImageModel
        {
            Architecture = checkpoint.Architecture,
            Task = ImageTask.Classification,
            Network = new Sequential(layers),
            Preprocessor = checkpoint.Preprocessor(new ImagePreprocessor { ShortestEdge = 256, CenterCrop = true, CropHeight = 224, CropWidth = 224,
                Mean = [0.485f, 0.456f, 0.406f], Std = [0.229f, 0.224f, 0.225f] }),
            Labels = checkpoint.Labels(),
        };
    }

    // A convolution whose filters the configuration fixes.
    private static Conv2d Expect(Conv2d conv, int filters, string tensor)
    {
        if (conv.OutChannels != filters)
        {
            conv.Dispose();
            throw new InvalidDataException($"The checkpoint's '{tensor}' has {conv.OutChannels} filters; the configuration says {filters}.");
        }

        return conv;
    }

    /// <summary>A basic residual block: relu(bn2(conv2(relu(bn1(conv1(x))))) + shortcut(x)).</summary>
    private sealed class BasicBlock(Conv2d conv1, BatchNorm bn1, Conv2d conv2, BatchNorm bn2, Conv2d? down, BatchNorm? downNorm) : Module
    {
        public static BasicBlock Read(ITensorStore tensors, string name, int inputs, int outputs, int stride, float epsilon, Device device)
        {
            var parts = new List<Module>();
            try
            {
                var conv1 = Add(Expect(StoredWeights.Conv2d(tensors, name + ".conv1", (stride, stride), (1, 1), device: device), outputs, name + ".conv1.weight"));
                if (conv1.InChannels != inputs)
                {
                    throw new InvalidDataException($"The checkpoint's '{name}.conv1.weight' reads {conv1.InChannels} channels; the block before gives {inputs}.");
                }

                var bn1 = Add(StoredWeights.BatchNorm(tensors, name + ".bn1", epsilon, device: device));
                var conv2 = Add(Expect(StoredWeights.Conv2d(tensors, name + ".conv2", padding: (1, 1), device: device), outputs, name + ".conv2.weight"));
                var bn2 = Add(StoredWeights.BatchNorm(tensors, name + ".bn2", epsilon, device: device));
                bool downsample = stride != 1 || inputs != outputs;
                var down = downsample ? Add(Expect(StoredWeights.Conv2d(tensors, name + ".downsample.0", (stride, stride), device: device), outputs, name + ".downsample.0.weight")) : null;
                var downNorm = downsample ? Add(StoredWeights.BatchNorm(tensors, name + ".downsample.1", epsilon, device: device)) : null;
                return new BasicBlock(conv1, bn1, conv2, bn2, down, downNorm);
            }
            catch
            {
                foreach (var part in parts)
                {
                    part.Dispose();
                }

                throw;
            }

            T Add<T>(T module)
                where T : Module
            {
                parts.Add(module);
                return module;
            }
        }

        public override IEnumerable<Module> Children() => down is null ? [conv1, bn1, conv2, bn2] : [conv1, bn1, conv2, bn2, down, downNorm!];

        protected override Tensor ForwardCore(Tensor input)
        {
            var y = bn2.Forward(conv2.Forward(bn1.Forward(conv1.Forward(input)).Relu()));
            var shortcut = down is null ? input : downNorm!.Forward(down.Forward(input));
            return (y + shortcut).Relu();
        }

        public override string ToString() => $"BasicBlock({conv1.InChannels} -> {conv1.OutChannels}{(conv1.StrideHeight > 1 ? $", stride {conv1.StrideHeight}" : "")})";
    }
}

/// <summary>
/// An anchor-free grid detector family read from ONNX: the network is imported as it is, and its config.json (or, for a bare
/// .onnx file, the family's defaults) gives the labels, the grid's stride and the preprocessing; its outputs are read by
/// <see cref="GridDecoder"/>, registered under <see cref="ImageFamilyPluginTests.Decoder"/>.
/// </summary>
public sealed class GridDetectorFamily : IImageModelFamily
{
    /// <inheritdoc />
    public string Name => ImageFamilyPluginTests.Detector;

    /// <inheritdoc />
    public ImageModel Read(ImageCheckpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        var settings = new JsonObject();
        if ((int?)checkpoint.Config["stride"] is { } stride)
        {
            settings["stride"] = stride;
        }

        var preprocessor = checkpoint.Preprocessor(new ImagePreprocessor { Height = 64, Width = 64, Resampling = ImageResampling.Bilinear, Normalize = false });
        var labels = checkpoint.Labels();
        var network = checkpoint.RequireOnnx();                                     // imported last: nothing to release if the rest fails
        return new ImageModel
        {
            Architecture = checkpoint.Architecture,
            Task = ImageTask.Detection,
            Network = network.Model,
            Preprocessor = preprocessor,
            Labels = labels,
            Decoder = ImageFamilyPluginTests.Decoder,
            DecoderSettings = settings,
        };
    }
}

/// <summary>
/// The grid detector's decoder (anchor-free, YOLO-style): outputs [5 + C, gh, gw] per image; for each cell (gx, gy) the
/// box centre ((gx + σ(tx)) · stride, (gy + σ(ty)) · stride), its size (e^tw · stride, e^th · stride), and the best class c
/// with score σ(objectness) · σ(class c). The stride is the "stride" setting, or the input height over the grid's.
/// </summary>
public static class GridDecoder
{
    /// <summary>The decoder for one detector.</summary>
    public static DetectionDecoder Create(DetectionDecoderContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        int? configured = context.Settings["stride"] is JsonValue v ? (int)v : null;
        return (outputs, shape) =>
        {
            if (shape.Count != 3 || shape[0] < 6)
            {
                throw new ArgumentException($"The grid decoder reads [5 + classes, rows, columns] per image, not [{string.Join(", ", shape)}].");
            }

            int classes = shape[0] - 5, rows = shape[1], columns = shape[2], cells = rows * columns;
            if (context.Classes > 0 && classes != context.Classes)
            {
                throw new ArgumentException($"The grid decoder found {classes} class outputs for {context.Classes} labels.");
            }

            float stride = configured ?? (context.InputHeight > 0 ? context.InputHeight / (float)rows
                : throw new InvalidOperationException("The grid decoder needs a stride setting or the network's input size."));
            var found = new List<Detection>(cells);
            for (int gy = 0; gy < rows; gy++)
            {
                for (int gx = 0; gx < columns; gx++)
                {
                    int cell = gy * columns + gx;
                    float objectness = Sigmoid(outputs[4 * cells + cell]);
                    int best = 0;
                    float score = objectness * Sigmoid(outputs[5 * cells + cell]);
                    for (int c = 1; c < classes; c++)
                    {
                        float s = objectness * Sigmoid(outputs[(5 + c) * cells + cell]);
                        if (s > score)
                        {
                            (best, score) = (c, s);
                        }
                    }

                    float cx = (gx + Sigmoid(outputs[cell])) * stride, cy = (gy + Sigmoid(outputs[cells + cell])) * stride;
                    float w = MathF.Exp(outputs[2 * cells + cell]) * stride, h = MathF.Exp(outputs[3 * cells + cell]) * stride;
                    found.Add(new Detection(BoundingBox.FromCenter(cx, cy, w, h), best, score));
                }
            }

            return found;
        };

        static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
    }
}
