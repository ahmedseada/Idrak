// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Data;
using Idrak.Generation;
using Idrak.Models;
using Idrak.Models.Abstractions;

namespace Idrak.Cli.Shared;

/// <summary>
/// Images given to a vision-language model on the command line (<c>run --image</c>, <c>chat --image</c>, <c>/image</c>):
/// the files as chat image parts, decoded by the library's codecs and turned upright by their EXIF orientation
/// (<see cref="ChatImageDecoder"/>, shared with the server's uploads and data URLs) as transformers' <c>load_image</c>
/// does for a path (<c>ImageOps.exif_transpose</c>). What a model does with them is its vision family's
/// (<see cref="PretrainedModel.Vision"/>, a <see cref="VisionFamilies"/> registration: a plug-in given with <c>-P</c>).
/// </summary>
internal static class ImageInputs
{
    /// <summary>The files as chat image parts; a usage error names a file that does not exist.</summary>
    public static IReadOnlyList<ChatImage> Read(IEnumerable<string> paths)
    {
        var images = new List<ChatImage>();
        foreach (string path in paths)
        {
            images.Add(File.Exists(path) ? ChatImage.FromFile(path) : throw new UsageException($"Image file not found: {path}"));
        }

        return images;
    }

    /// <summary>A chat image's pixels: decoded by the codec that reads its bytes, then turned by its EXIF orientation.</summary>
    /// <exception cref="InvalidDataException">No registered codec reads it.</exception>
    public static ImageData Decode(ChatImage image) => ChatImageDecoder.Decode(image);

    /// <summary>
    /// Makes the image encoder of a model: given the model and the options (device, vision options), its vision encoder. Null:
    /// the model's (<see cref="PretrainedModel.CreateVisionEncoder"/>: the family's encoder with the trained vision tensors an adapter brought). Tests set it to feed reference features.
    /// </summary>
    internal static Func<PretrainedModel, VisionEncoderOptions, IVisionEncoder>? EncoderFactory { get; set; }

    /// <summary>
    /// How a chat generator reads <paramref name="model"/>'s images (<see cref="ChatGenerator.Images"/>), or null with
    /// the reason when it reads none (a text model). Everything comes from the model's vision family: the encoder (built
    /// when the first image is read, disposed with the returned owner; <paramref name="transforms"/> run on every decoded
    /// image first, unless a request gives its own; <paramref name="visionOptions"/> are the family's own options for every image, their keys checked against
    /// the family's here), the prompt format and the attention rule.
    /// </summary>
    public static ModelImages? For(PretrainedModel model, ImageTransformPipeline transforms, VisionOptions? visionOptions, out string? reason)
    {
        ArgumentNullException.ThrowIfNull(model);
        reason = null;
        try
        {
            return ModelImages.For(model, transforms, visionOptions, EncoderFactory);
        }
        catch (ArgumentException ex)
        {
            throw new UsageException($"{ModelChoices.VisionOption}: {ex.Message}");
        }
    }
}
