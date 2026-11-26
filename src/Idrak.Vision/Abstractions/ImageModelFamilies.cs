// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Imaging;
using Idrak.Models.Abstractions;
using Idrak.Onnx;

namespace Idrak.Vision.Abstractions;

/// <summary>What an image model does with an image: the head its network ends in.</summary>
public enum ImageTask
{
    /// <summary>One score per class for the whole image ([N, classes] logits).</summary>
    Classification,

    /// <summary>Objects and their boxes: the network's outputs are read by a detection decoder (Idrak.Vision's <c>DetectionDecoders</c>).</summary>
    Detection,

    /// <summary>A class per pixel ([N, classes, h, w] logits).</summary>
    Segmentation,

    /// <summary>Features of the image (a backbone without a head): the network's outputs as they are.</summary>
    Features,

    /// <summary>
    /// A label sequence along the image's width (a line of text): [N, steps, classes] scores, a step per
    /// <see cref="LineRecognitionSettings.Stride"/> columns, CTC's blank among the classes, read by a CTC decoder
    ///. The model's <see cref="ImageModel.Lines"/> say how a line becomes the network's input.
    /// </summary>
    LineRecognition,
}

/// <summary>
/// How a line-recognition model (<see cref="ImageTask.LineRecognition"/>) reads a line: the line scaled to
/// <see cref="Height"/> keeping its aspect (at least <see cref="Stride"/> columns), its ink made high on a background of 0
/// and its contrast stretched to [0, 1] (<see cref="InkHigh"/>), flipped for a right-to-left line, then batched padded with
/// 0 on the right; the network's [N, steps, classes] scores decoded by CTC (blank <see cref="Blank"/>), the labels' text
/// turned from column order to logical order by <see cref="ReadingOrder"/>.
/// </summary>
public sealed record LineRecognitionSettings
{
    /// <summary>The height lines are scaled to (their width follows, keeping the aspect).</summary>
    public required int Height { get; init; }

    /// <summary>The image columns one step of the network's output stands for (its pooling along the width). Default 1.</summary>
    public int Stride { get; init; } = 1;

    /// <summary>The blank class of CTC (its label is empty). Default 0.</summary>
    public int Blank { get; init; }

    /// <summary>Whether lines read right to left: each line is flipped horizontally first, so its columns run in reading order.</summary>
    public bool RightToLeft { get; init; }

    /// <summary>
    /// The reading order (a <c>ReadingOrders</c> name, "runs" or "bidi") that turns a reading from column order to logical
    /// order and a transcription to column order for training; null: the labels' text as it is.
    /// </summary>
    public string? ReadingOrder { get; init; }

    /// <summary>
    /// Whether ink is made high on a background of 0 (a mostly light line inverted) and the contrast stretched to [0, 1]
    /// before scaling (default true); false keeps the grey values as decoded.
    /// </summary>
    public bool InkHigh { get; init; } = true;

    /// <summary>Whether the network's scores are log-probabilities already (else a log-softmax is taken over the classes).</summary>
    public bool LogProbabilities { get; init; }

    /// <summary>The settings as JSON (a package's training entry, <c>idrak train</c>'s output).</summary>
    public JsonObject ToJson() => new()
    {
        ["height"] = Height,
        ["stride"] = Stride,
        ["blank"] = Blank,
        ["direction"] = RightToLeft ? "rtl" : "ltr",
        ["reading_order"] = ReadingOrder,
        ["ink_high"] = InkHigh,
        ["log_probabilities"] = LogProbabilities,
    };

    /// <summary>Settings from <see cref="ToJson"/>'s form (absent keys take the defaults; "height" is required).</summary>
    /// <exception cref="InvalidDataException">"height" is missing.</exception>
    public static LineRecognitionSettings FromJson(JsonObject json)
    {
        ArgumentNullException.ThrowIfNull(json);
        return new LineRecognitionSettings
        {
            Height = (int?)json["height"] ?? throw new InvalidDataException("Line recognition settings need a \"height\"."),
            Stride = (int?)json["stride"] ?? 1,
            Blank = (int?)json["blank"] ?? 0,
            RightToLeft = (string?)json["direction"] == "rtl",
            ReadingOrder = (string?)json["reading_order"],
            InkHigh = (bool?)json["ink_high"] ?? true,
            LogProbabilities = (bool?)json["log_probabilities"] ?? false,
        };
    }

    /// <summary>Why these settings cannot read lines with <paramref name="classes"/> classes, or null when they can.</summary>
    public string? Problem(int classes) =>
        Height <= 0 ? $"line height {Height}"
        : Stride <= 0 ? $"stride {Stride}"
        : Blank < 0 || Blank >= classes ? $"blank {Blank} outside the {classes} classes"
        : null;
}

/// <summary>Settings for <see cref="ImageModels.Load"/>.</summary>
public sealed record ImageModelOptions
{
    /// <summary>Where the network is created (the default device when null).</summary>
    public Device? Device { get; init; }

    /// <summary>The architecture to read the checkpoint as, instead of the one its config.json (or ONNX metadata) names.</summary>
    public string? Architecture { get; init; }
}

/// <summary>
/// What an image model family reads a checkpoint from (<see cref="IImageModelFamily.Read"/>): a safetensors folder (or any
/// folder a checkpoint format reads) with its config.json, or an ONNX file.
/// </summary>
/// <param name="Architecture">The architecture name the checkpoint was matched by.</param>
/// <param name="Config">The model's config.json (an empty object for an ONNX file that came without one).</param>
/// <param name="Path">The path the caller gave.</param>
/// <param name="Folder">The model folder (where config.json and <c>preprocessor_config.json</c> are), or null.</param>
/// <param name="Tensors">
/// The checkpoint's tensors by their stored names, open while reading (do not keep it); read them one at a time, so a
/// tensor's float32 copy exists only while its layer is made. Null for an ONNX checkpoint.
/// </param>
/// <param name="Onnx">Imports the ONNX file through Idrak's importer (the same network on every call); null for a tensor checkpoint.</param>
/// <param name="Options">The caller's device and weight precision.</param>
/// <param name="Notes">Append anything approximated (shown in the loaded model's notes).</param>
public sealed record ImageCheckpoint(string Architecture, JsonObject Config, string Path, string? Folder, ITensorStore? Tensors, Func<ImportedNetwork>? Onnx,
    ImageModelOptions Options, List<string> Notes)
{
    /// <summary>The device the network is made on.</summary>
    public Device Device => Options.Device ?? Device.Default;

    /// <summary>The checkpoint's tensors.</summary>
    /// <exception cref="InvalidDataException">The checkpoint is an ONNX file (it has a network, not named tensors).</exception>
    public ITensorStore RequireTensors() => Tensors
        ?? throw new InvalidDataException($"{Path} is an ONNX file; family '{Architecture}' reads named tensors (a safetensors folder with config.json).");

    /// <summary>The imported ONNX network.</summary>
    /// <exception cref="InvalidDataException">The checkpoint is not an ONNX file.</exception>
    public ImportedNetwork RequireOnnx() => (Onnx ?? throw new InvalidDataException($"{Path} is not an ONNX file; family '{Architecture}' reads ONNX networks."))();

    /// <summary>
    /// The class names from config.json's "id2label" (index → name, Hugging Face's convention), in index order; null
    /// when it has none. Missing indices are refused.
    /// </summary>
    /// <exception cref="InvalidDataException">"id2label" does not name every index from 0 to its count - 1.</exception>
    public IReadOnlyList<string>? Labels()
    {
        if (Config["id2label"] is not JsonObject map || map.Count == 0)
        {
            return null;
        }

        var labels = new string?[map.Count];
        foreach (var (key, value) in map)
        {
            if (!int.TryParse(key, System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out int index)
                || index >= labels.Length || labels[index] is not null)
            {
                throw new InvalidDataException($"config.json's id2label has index '{key}'; it names each index from 0 to {labels.Length - 1} once.");
            }

            labels[index] = (string?)value ?? index.ToString(System.Globalization.CultureInfo.InvariantCulture);
        }

        return labels!;
    }

    /// <summary>
    /// The image steps of the folder's <c>preprocessor_config.json</c>, keys it does not give taking
    /// <paramref name="defaults"/>' values (the family's processor class's defaults); <paramref name="defaults"/> itself when
    /// the checkpoint has no such file.
    /// </summary>
    /// <exception cref="InvalidDataException">There is no preprocessor config and no defaults.</exception>
    public ImagePreprocessor Preprocessor(ImagePreprocessor? defaults = null)
    {
        string? file = Folder is null ? null : System.IO.Path.Combine(Folder, "preprocessor_config.json");
        return file is not null && File.Exists(file) ? ImagePreprocessor.FromConfig(file, defaults: defaults)
            : defaults ?? throw new InvalidDataException($"{Path} has no preprocessor_config.json and family '{Architecture}' gives no default preprocessing.");
    }
}

/// <summary>
/// An image model as its family describes it (<see cref="IImageModelFamily.Read"/>): the network, its task, how images
/// become its input, its labels, and for a detector the decoder that reads its outputs. Dispose it to release the network
/// (the predictors made from it, Idrak.Vision's <c>Classifier</c>, <c>Detector</c> and <c>Segmenter</c>, use it without
/// owning it).
/// </summary>
public sealed record ImageModel : IDisposable
{
    /// <summary>The architecture name (config.json's "architectures").</summary>
    public required string Architecture { get; init; }

    /// <summary>What the network's outputs are.</summary>
    public required ImageTask Task { get; init; }

    /// <summary>The network: images [N, <see cref="Channels"/>, height, width] (the preprocessor's output) in.</summary>
    public required Module Network { get; init; }

    /// <summary>How an image becomes the network's input (resize, crop, rescale, normalize), from the family's preprocessor config.</summary>
    public required ImagePreprocessor Preprocessor { get; init; }

    /// <summary>The network's input channels (3 unless the family says otherwise).</summary>
    public int Channels { get; init; } = 3;

    /// <summary>
    /// The class names in output order (classifiers, detectors, segmenters), or null when the checkpoint gives none. A line
    /// recognizer's are its labels' texts, one per class, the blank's empty.
    /// </summary>
    public IReadOnlyList<string>? Labels { get; init; }

    /// <summary>For <see cref="ImageTask.LineRecognition"/>: how a line becomes the network's input and its output text.</summary>
    public LineRecognitionSettings? Lines { get; init; }

    /// <summary>
    /// For <see cref="ImageTask.Detection"/>: the name of the decoder that reads the network's outputs (the family's
    /// plug-in's or the app's).
    /// </summary>
    public string? Decoder { get; init; }

    /// <summary>The decoder's settings (strides, box layout, ...), from the family's config; null when it needs none.</summary>
    public JsonObject? DecoderSettings { get; init; }

    /// <summary>What loading approximated or chose (precision kept, layers widened), with the checkpoint's notes.</summary>
    public IReadOnlyList<string> Notes { get; init; } = [];

    /// <summary>
    /// The network's input size, [channels, height, width], when every image becomes the same size (a resize to a height
    /// and width, or a center crop); null when the size follows the image (a resize by its shortest edge alone, or a line
    /// recognizer's lines, whose width follows each line).
    /// </summary>
    public IReadOnlyList<int>? InputShape =>
        Task == ImageTask.LineRecognition ? null
        : Preprocessor.CenterCrop ? [Channels, Preprocessor.CropHeight, Preprocessor.CropWidth]
        : Preprocessor.Resize && Preprocessor.ShortestEdge <= 0 ? [Channels, Preprocessor.Height, Preprocessor.Width]
        : null;

    /// <summary>Disposes the network.</summary>
    public void Dispose() => Network.Dispose();
}

/// <summary>
/// An image model family: how its checkpoints become an <see cref="ImageModel"/> (the network built from layers or imported
/// from ONNX, its task, preprocessing, labels and decoder), for the architecture name its checkpoints give.
/// </summary>
public interface IImageModelFamily
{
    /// <summary>Its architecture name.</summary>
    string Name { get; }

    /// <summary>The model the checkpoint holds, its network on <see cref="ImageCheckpoint.Device"/>.</summary>
    /// <exception cref="InvalidDataException">A tensor is missing or its shape disagrees with the configuration.</exception>
    ImageModel Read(ImageCheckpoint checkpoint);
}

