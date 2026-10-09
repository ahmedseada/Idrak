// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Data;
using Idrak.Data.Abstractions;
using Idrak.Generation;
using Idrak.Generation.Abstractions;
using Idrak.Models;

namespace Idrak.Cli.Shared;

/// <summary>
/// Images given to a vision-language model on the command line (<c>run --image</c>, <c>chat --image</c>, <c>/image</c>):
/// the files as chat image parts, decoded by the library's codecs (<see cref="ImageCodecs"/>) and turned upright by their
/// EXIF orientation (<see cref="ChatImageDecoder"/>, shared with the server's uploads and data URLs) as transformers'
/// <c>load_image</c> does for a path (<c>ImageOps.exif_transpose</c>).
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

    /// <summary>A chat image's pixels: decoded by the codec that reads its bytes (<see cref="ImageCodecs"/>), then turned by its EXIF orientation.</summary>
    /// <exception cref="InvalidDataException">No registered codec reads it.</exception>
    public static ImageData Decode(ChatImage image) => ChatImageDecoder.Decode(image);

    /// <summary>
    /// Makes the image encoder of a model: given the model and the preprocessor its images are read with, a function from
    /// decoded images to their features [images, tokens per image, text width] on the model's device, and what to dispose
    /// when done. Null: the library's (<see cref="PretrainedVision.CreateEncoder"/>). Tests set it to feed reference features.
    /// </summary>
    internal static Func<PretrainedModel, ImagePreprocessor, (Func<IReadOnlyList<ImageData>, Tensor> Encode, IDisposable? Owner)>? EncoderFactory { get; set; }

    /// <summary>
    /// How a chat generator reads <paramref name="model"/>'s images (<see cref="ChatGenerator.Images"/>), or null with
    /// the reason when it reads none: a text model, or a family without a registered image prompt format. The encoder is
    /// built when the first image is read (and disposed with the returned owner): <paramref name="grayscale"/> turns
    /// images grey first (<see cref="PretrainedVision.Preprocessor"/>, Pillow's <c>convert("L")</c>).
    /// </summary>
    public static ModelImages? For(PretrainedModel model, bool grayscale, out string? reason)
    {
        ArgumentNullException.ThrowIfNull(model);
        reason = null;
        if (model.Vision is not { } vision)
        {
            return null;
        }

        string type = (string?)model.Config["model_type"] ?? "";
        if (ImagePromptFormats.Find(type) is not { } format)
        {
            reason = $"no image prompt format is registered for its model type '{type}' ({string.Join(", ", ImagePromptFormats.Names)})";
            return null;
        }

        return new ModelImages(model, vision, format, vision.Preprocessor(grayscale));
    }
}

/// <summary>A model's image reading for chat (<see cref="Images"/>), owning the encoder it builds on first use.</summary>
internal sealed class ModelImages : IDisposable
{
    private readonly Lock _lock = new();
    private (Func<IReadOnlyList<ImageData>, Tensor> Encode, IDisposable? Owner)? _encoder;

    public ModelImages(PretrainedModel model, PretrainedVision vision, IImagePromptFormat format, ImagePreprocessor preprocessor)
    {
        Preprocessor = preprocessor;
        Images = new ChatImages(vision.ImageTokens, format, images =>
        {
            Func<IReadOnlyList<ImageData>, Tensor> encode;
            lock (_lock)
            {
                _encoder ??= (ImageInputs.EncoderFactory ?? LibraryEncoder)(model, preprocessor);
                encode = _encoder.Value.Encode;
            }

            return encode([.. images.Select(ImageInputs.Decode)]);
        }) { Owner = this };                                            // the engine disposes it with the model it serves
    }

    /// <summary>How images become pixel values (the model's preprocessing, grey when asked).</summary>
    public ImagePreprocessor Preprocessor { get; }

    /// <summary>What the chat generator reads images with.</summary>
    public ChatImages Images { get; }

    private static (Func<IReadOnlyList<ImageData>, Tensor>, IDisposable?) LibraryEncoder(PretrainedModel model, ImagePreprocessor preprocessor)
    {
        var encoder = model.Vision!.CreateEncoder(model.Device, preprocessor);
        return (encoder.Encode, encoder);
    }

    public void Dispose()
    {
        lock (_lock)
        {
            _encoder?.Owner?.Dispose();
            _encoder = null;
        }
    }
}
