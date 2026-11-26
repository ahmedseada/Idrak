// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Data;
using Idrak.Data.Abstractions;
using Idrak.Imaging;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Vision;
using Idrak.Vision.Abstractions;

namespace Idrak.PluginTests;

/// <summary>
/// The names of two image model families written outside the library (plan 13, step 6), whose checkpoints are
/// tests/Idrak.Tests/data/image-families (written by tools/pytorch/image_families_reference.py):
/// <list type="bullet">
/// <item><see cref="TinyResNetFamily"/>: a ResNet-style classifier read from safetensors and config.json with torchvision's
/// tensor names, its layers built here (basic residual blocks are this assembly's own module), the weights read one tensor at
/// a time;</item>
/// <item><see cref="GridDetectorFamily"/>: an anchor-free grid detector read from ONNX through Idrak's importer, with its own
/// decoder (<see cref="GridDecoder"/>) and the library's non-maximum suppression.</item>
/// </list>
/// </summary>
public static class ImageFamilyPluginTests
{
    /// <summary>The classifier family's architecture name.</summary>
    public const string Classifier = "OutsideTinyResNetForImageClassification";

    /// <summary>The detector family's architecture name.</summary>
    public const string Detector = "OutsideGridDetector";

    /// <summary>The detector's decoder name.</summary>
    public const string Decoder = "outside-grid";
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
            layers.Add(StoredWeights.Linear(tensors, "fc", device));
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
/// <see cref="GridDecoder"/>, named <see cref="ImageFamilyPluginTests.Decoder"/>.
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
    public static DetectionDecoder Create(DetectionDecoderRequest context)
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

