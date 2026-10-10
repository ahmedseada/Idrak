// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Data;
using Idrak.Models.Abstractions;
using Idrak.Vision.Abstractions;

namespace Idrak.PluginTests;

/// <summary>
/// This assembly as a command-line plug-in for image models (<c>idrak predict -P</c>, <c>idrak train -P</c>, plan 13 step 7):
/// <see cref="RegisterIdrakPlugin"/> registers the two families of <see cref="ImageFamilyPluginTests"/>, an ONNX segmenter
/// family, the grid detector's decoder and its training head. The library registers none of them.
/// </summary>
public static class ImageFamilyPlugin
{
    /// <summary>The segmenter family's architecture name.</summary>
    public const string Segmenter = "OutsideOnnxSegmenter";

    /// <summary>Registers the families, the decoder and the head (what <c>-P</c> runs after loading the assembly).</summary>
    public static void RegisterIdrakPlugin()
    {
        ImageModelFamilies.Register(new TinyResNetFamily());
        ImageModelFamilies.Register(new GridDetectorFamily());
        ImageModelFamilies.Register(new OnnxSegmenterFamily());
        DetectionDecoders.Register(ImageFamilyPluginTests.Decoder, GridDecoder.Create);
        DetectionHeads.Register(ImageFamilyPluginTests.Decoder, GridHead.Create);
    }

    /// <summary>Removes everything <see cref="RegisterIdrakPlugin"/> registered (tests leave the registries as they found them).</summary>
    public static void Unregister()
    {
        ImageModelFamilies.Unregister(ImageFamilyPluginTests.Classifier);
        ImageModelFamilies.Unregister(ImageFamilyPluginTests.Detector);
        ImageModelFamilies.Unregister(Segmenter);
        DetectionDecoders.Unregister(ImageFamilyPluginTests.Decoder);
        DetectionHeads.Unregister(ImageFamilyPluginTests.Decoder);
    }
}

/// <summary>
/// A segmenter family read from ONNX: the network imported as it is (images [N, channels, h, w] to logits [N, classes, h',
/// w']), config.json's "image_size" [height, width] the resize, "num_channels" the channels (3 unless given), "id2label"
/// the classes; values in [0, 1], no normalization.
/// </summary>
public sealed class OnnxSegmenterFamily : IImageModelFamily
{
    /// <inheritdoc />
    public string Name => ImageFamilyPlugin.Segmenter;

    /// <inheritdoc />
    public ImageModel Read(ImageCheckpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        var size = checkpoint.Config["image_size"]?.AsArray() ?? throw new InvalidDataException("config.json has no image_size [height, width].");
        int channels = (int?)checkpoint.Config["num_channels"] ?? 3;
        var preprocessor = new ImagePreprocessor
        {
            Height = (int)size[0]!, Width = (int)size[1]!, Normalize = false, Grayscale = channels == 1, ConvertRgb = channels != 1,
        };
        var labels = checkpoint.Labels();
        var network = checkpoint.RequireOnnx();
        return new ImageModel
        {
            Architecture = checkpoint.Architecture,
            Task = ImageTask.Segmentation,
            Network = network.Model,
            Preprocessor = preprocessor,
            Channels = channels,
            Labels = labels,
        };
    }
}

/// <summary>
/// The grid detector's training head (<see cref="DetectionHeads"/>, under its decoder's name): the outputs [N, 5 + C, rows,
/// columns] read as <see cref="GridDecoder"/> reads them, as tensors: each cell's box centre ((g + σ(t)) · stride), size
/// (e^t · stride) as corners, its objectness logit and its class logits.
/// </summary>
public static class GridHead
{
    /// <summary>The head for one detector.</summary>
    public static DetectionHead Create(DetectionDecoderContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        int? configured = context.Settings["stride"] is JsonValue v ? (int)v : null;
        return outputs =>
        {
            if (outputs.Rank != 4 || outputs.Shape[1] < 6)
            {
                throw new ArgumentException($"The grid head reads [N, 5 + classes, rows, columns], not {Tensor.FormatShape(outputs.Shape)}.");
            }

            int n = outputs.Shape[0], width = outputs.Shape[1], rows = outputs.Shape[2], columns = outputs.Shape[3], cells = rows * columns;
            float stride = configured ?? (context.InputHeight > 0 ? context.InputHeight / (float)rows
                : throw new InvalidOperationException("The grid head needs a stride setting or the network's input size."));
            var flat = outputs.Reshape(n, width, cells);
            Tensor Row(int r) => flat.Narrow(1, r, 1).Reshape(n, cells);
            var gx = new float[n * cells];
            var gy = new float[n * cells];
            for (int i = 0; i < n * cells; i++)
            {
                (gx[i], gy[i]) = (i % cells % columns, i % cells / columns);
            }

            var cx = (Row(0).Sigmoid() + Tensor.From(gx, [n, cells], outputs.Device)) * stride;
            var cy = (Row(1).Sigmoid() + Tensor.From(gy, [n, cells], outputs.Device)) * stride;
            var halfWidth = Row(2).Exp() * (stride / 2);
            var halfHeight = Row(3).Exp() * (stride / 2);
            var boxes = Tensor.Stack([cx - halfWidth, cy - halfHeight, cx + halfWidth, cy + halfHeight], dim: 2);
            var classes = flat.Narrow(1, 5, width - 5).Permute(0, 2, 1);
            return new DetectionCandidates(boxes, classes, Row(4));
        };
    }
}
