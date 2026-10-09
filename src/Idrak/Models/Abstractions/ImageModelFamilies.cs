// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using Idrak.Data;
using Idrak.Onnx;

namespace Idrak.Models.Abstractions;

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
}

/// <summary>Settings for <see cref="ImageModels.Load"/>.</summary>
public sealed record ImageModelOptions
{
    /// <summary>Where the network is created (the default device when null).</summary>
    public Device? Device { get; init; }

    /// <summary>
    /// The precision weights are kept in where the layers allow it: <see cref="EncoderWeights.AsStored"/> (the default)
    /// keeps a bfloat16 checkpoint's projection weights as bfloat16 (the same values, half the memory); a layer with no
    /// narrower form (a convolution, a norm) holds float32 copies of the stored values.
    /// </summary>
    public EncoderWeights Weights { get; init; } = EncoderWeights.AsStored;

    /// <summary>The architecture to read the checkpoint as, instead of the one its config.json (or ONNX metadata) names.</summary>
    public string? Architecture { get; init; }
}

/// <summary>
/// What an image model family reads a checkpoint from (<see cref="IImageModelFamily.Read"/>): a safetensors folder (or any
/// folder a registered <see cref="CheckpointFormats">checkpoint format</see> reads) with its config.json, or an ONNX file.
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

    /// <summary>The class names in output order (classifiers, detectors, segmenters), or null when the checkpoint gives none.</summary>
    public IReadOnlyList<string>? Labels { get; init; }

    /// <summary>
    /// For <see cref="ImageTask.Detection"/>: the name of the decoder that reads the network's outputs (registered in
    /// Idrak.Vision's <c>DetectionDecoders</c>, by the family's plug-in or the app).
    /// </summary>
    public string? Decoder { get; init; }

    /// <summary>The decoder's settings (strides, box layout, ...), from the family's config; null when it needs none.</summary>
    public JsonObject? DecoderSettings { get; init; }

    /// <summary>What loading approximated or chose (precision kept, layers widened), with the checkpoint's notes.</summary>
    public IReadOnlyList<string> Notes { get; init; } = [];

    /// <summary>
    /// The network's input size, [channels, height, width], when every image becomes the same size (a resize to a height
    /// and width, or a center crop); null when the size follows the image (a resize by its shortest edge alone).
    /// </summary>
    public IReadOnlyList<int>? InputShape =>
        Preprocessor.CenterCrop ? [Channels, Preprocessor.CropHeight, Preprocessor.CropWidth]
        : Preprocessor.Resize && Preprocessor.ShortestEdge <= 0 ? [Channels, Preprocessor.Height, Preprocessor.Width]
        : null;

    /// <summary>Disposes the network.</summary>
    public void Dispose() => Network.Dispose();
}

/// <summary>
/// An image model family: how its checkpoints become an <see cref="ImageModel"/> (the network built from layers or imported
/// from ONNX, its task, preprocessing, labels and decoder). Registered in <see cref="ImageModelFamilies"/> under the
/// architecture name its checkpoints give.
/// </summary>
public interface IImageModelFamily
{
    /// <summary>The architecture name it is registered under.</summary>
    string Name { get; }

    /// <summary>The model the checkpoint holds, its network on <see cref="ImageCheckpoint.Device"/>.</summary>
    /// <exception cref="InvalidDataException">A tensor is missing or its shape disagrees with the configuration.</exception>
    ImageModel Read(ImageCheckpoint checkpoint);
}

/// <summary>
/// The image model families (classifiers, detectors, segmenters, backbones), by architecture name (config.json's
/// "architectures", or an ONNX file's "architecture" metadata). The library registers none: a family (its configuration,
/// tensor names, layers, preprocessing defaults and decoder) is an application of these contracts, registered by the
/// plug-in or app that brings it (the test plug-ins in <c>tests/Idrak.PluginTests</c> register a small ResNet-style
/// classifier and a grid detector). A checkpoint is matched to its own family by that exact name; one whose name has no
/// family is refused, naming this registry: nothing falls back to another family. Load one with <see cref="ImageModels.Load"/>.
/// </summary>
public static class ImageModelFamilies
{
    private static readonly SlotTable<string, IImageModelFamily> Registry = new(nameof(ImageModelFamilies), comparer: StringComparer.Ordinal,
        unguarded: "a family's network, preprocessing, labels and decoder must agree with one another, so two registrations cannot be mixed");

    /// <summary>Registers <paramref name="family"/> under its <see cref="IImageModelFamily.Name"/>; under a library name it takes that family's place until <see cref="Unregister"/>.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IImageModelFamily family)
    {
        ArgumentNullException.ThrowIfNull(family);
        ArgumentException.ThrowIfNullOrEmpty(family.Name);
        Registry.Register(family.Name, family, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's family <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered architecture names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The family registered as <paramref name="name"/>, or null.</summary>
    public static IImageModelFamily? Find(string name) => Registry.Find(name);

    /// <summary>The family registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is; the message names <see cref="Register"/>.</exception>
    public static IImageModelFamily Get(string name) => Registry.Find(name) ?? throw NotRegistered(name);

    /// <summary>The library's family <paramref name="name"/>, whatever an app registered over it; null when the library has none (it has none).</summary>
    public static IImageModelFamily? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the family <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    internal static NotSupportedException NotRegistered(string name) =>
        new($"Image model family '{name}' is not registered (registered: {(Registry.Keys.Count == 0 ? "none" : string.Join(", ", Registry.Keys))}); "
            + "register it with ImageModelFamilies.Register (a plug-in or the app that brings the family; the library registers none).");
}
