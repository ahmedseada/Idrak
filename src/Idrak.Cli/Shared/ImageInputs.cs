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
    /// Makes the image encoder of a model: given the model and the options (device, grayscale), its vision encoder. Null:
    /// the family's (<see cref="PretrainedVision.CreateEncoder"/>). Tests set it to feed reference features.
    /// </summary>
    internal static Func<PretrainedModel, VisionEncoderOptions, IVisionEncoder>? EncoderFactory { get; set; }

    /// <summary>
    /// How a chat generator reads <paramref name="model"/>'s images (<see cref="ChatGenerator.Images"/>), or null with
    /// the reason when it reads none (a text model). Everything comes from the model's vision family: the encoder (built
    /// when the first image is read, disposed with the returned owner; <paramref name="grayscale"/> turns images grey
    /// first), the prompt format and the attention rule.
    /// </summary>
    public static ModelImages? For(PretrainedModel model, bool grayscale, out string? reason)
    {
        ArgumentNullException.ThrowIfNull(model);
        reason = null;
        if (model.Vision is not { } vision)
        {
            return null;
        }

        return new ModelImages(model, vision, new VisionEncoderOptions { Device = model.Device, Grayscale = grayscale });
    }
}

/// <summary>A model's image reading for chat (<see cref="Images"/>), owning the encoder it builds on first use.</summary>
internal sealed class ModelImages : IDisposable
{
    private readonly LazyEncoder _encoder;

    public ModelImages(PretrainedModel model, PretrainedVision vision, VisionEncoderOptions options)
    {
        Options = options;
        _encoder = new LazyEncoder(() => (ImageInputs.EncoderFactory ?? ((_, o) => vision.CreateEncoder(o)))(model, options), vision.Width, model.Device);
        Images = new ChatImages(_encoder, vision.PromptFormat, vision.Attention) { Decode = ImageInputs.Decode, Owner = this };   // the engine disposes it with the model it serves
    }

    /// <summary>The device and grayscale setting its encoder is built with.</summary>
    public VisionEncoderOptions Options { get; }

    /// <summary>What the chat generator reads images with.</summary>
    public ChatImages Images { get; }

    public void Dispose() => _encoder.Dispose();

    // The family's encoder, built when an image is first read (its layout or its features), disposed with the model.
    private sealed class LazyEncoder(Func<IVisionEncoder> create, int width, Device device) : IVisionEncoder
    {
        private readonly Lock _lock = new();
        private IVisionEncoder? _built;

        public int Width => width;

        public Device Device => device;

        public ImageTokenLayout Layout(ImageData image) => Built().Layout(image);

        public IReadOnlyList<ImageFeatures> Encode(IReadOnlyList<ImageData> images) => Built().Encode(images);

        public void Dispose()
        {
            lock (_lock)
            {
                _built?.Dispose();
                _built = null;
            }
        }

        private IVisionEncoder Built()
        {
            lock (_lock)
            {
                return _built ??= create();
            }
        }
    }
}
