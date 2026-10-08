// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Data;
using Idrak.Data.Abstractions;
using Idrak.Layers;

namespace Idrak.Models;

/// <summary>
/// The image side of a vision-language model (Gemma 3): images become the embeddings of their soft tokens,
/// [images, <see cref="ImageTokenIds.TokensPerImage"/>, text width], which replace the <c>&lt;image_soft_token&gt;</c>
/// rows of the prompt's embeddings. Pixel values go through <see cref="Encoder"/> (SigLIP) and <see cref="Projector"/>;
/// files, encoded bytes and chat images are first decoded (<see cref="ImageCodecs"/>) and preprocessed
/// (<see cref="Preprocessor"/>, the model's <c>preprocessor_config.json</c>). Made by
/// <see cref="PretrainedVision.CreateEncoder"/>; as a module, its forward pass takes pixel values [images, channels,
/// size, size]. Every <c>Encode</c> runs without recording gradients and returns a tensor the caller disposes.
/// </summary>
public sealed class ImageEncoder : Module
{
    internal ImageEncoder(PretrainedVision vision, SiglipVisionEncoder encoder, ImageProjector projector, ImagePreprocessor preprocessor)
    {
        Vision = vision;
        Encoder = encoder;
        Projector = projector;
        Preprocessor = preprocessor;
    }

    /// <summary>The vision part this encoder was built from (its configuration and image token ids).</summary>
    public PretrainedVision Vision { get; }

    /// <summary>The SigLIP encoder: pixel values to patch outputs [images, patches, vision width].</summary>
    public SiglipVisionEncoder Encoder { get; }

    /// <summary>The projector: patch outputs to soft-token embeddings [images, tokens per image, text width].</summary>
    public ImageProjector Projector { get; }

    /// <summary>How images become pixel values (resize, rescale, normalize, optionally grayscale).</summary>
    public ImagePreprocessor Preprocessor { get; }

    /// <summary>
    /// The soft-token embeddings of pixel values [images, channels, size, size] or of one image [channels, size, size]
    /// (as <see cref="ImagePreprocessor.Process(ImageData, Device?)"/> gives them): [images, tokens per image, text width].
    /// </summary>
    public Tensor Encode(Tensor pixelValues)
    {
        ArgumentNullException.ThrowIfNull(pixelValues);
        var device = WeightsDevice ?? pixelValues.Device;
        if (pixelValues.Device != device)
        {
            using var moved = Tensor.From(pixelValues.ToArray(), pixelValues.Shape, device);
            return Predict(moved);
        }

        return Predict(pixelValues);
    }

    /// <summary>The soft-token embeddings of decoded images, [images.Count, tokens per image, text width].</summary>
    public Tensor Encode(IReadOnlyList<ImageData> images)
    {
        ArgumentNullException.ThrowIfNull(images);
        if (images.Count == 0)
        {
            throw new ArgumentException("No images to encode.", nameof(images));
        }

        var config = Vision.Encoder;
        int each = config.Channels * config.ImageSize * config.ImageSize;
        var values = new float[images.Count * each];
        for (int i = 0; i < images.Count; i++)
        {
            var pixels = Preprocessor.Pixels(images[i]);
            if (pixels.Length != each)
            {
                throw new InvalidOperationException($"The preprocessor gives {pixels.Length} pixel values per image; the encoder takes {config.Channels} x {config.ImageSize} x {config.ImageSize}.");
            }

            pixels.CopyTo(values, i * each);
        }

        using var input = Tensor.From(values, [images.Count, config.Channels, config.ImageSize, config.ImageSize], WeightsDevice);
        return Predict(input);
    }

    /// <summary>The soft-token embeddings of one decoded image, [1, tokens per image, text width].</summary>
    public Tensor Encode(ImageData image) => Encode([image]);

    /// <summary>The soft-token embeddings of an image file (any registered codec), [1, tokens per image, text width].</summary>
    public Tensor Encode(string path) => Encode(ImageCodecs.Decode(path));

    /// <summary>The soft-token embeddings of chat images (their encoded bytes, decoded by the registered codecs), [images.Count, tokens per image, text width].</summary>
    public Tensor Encode(IReadOnlyList<ChatImage> images)
    {
        ArgumentNullException.ThrowIfNull(images);
        return Encode([.. images.Select(image => ImageCodecs.Decode(image.Data.Span))]);
    }

    /// <summary>The soft-token embeddings of one chat image, [1, tokens per image, text width].</summary>
    public Tensor Encode(ChatImage image) => Encode([image]);

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => Projector.Forward(Encoder.Forward(input));

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => [Encoder, Projector];

    /// <inheritdoc />
    public override void Dispose()
    {
        Encoder.Dispose();
        Projector.Dispose();
        GC.SuppressFinalize(this);
    }

    /// <inheritdoc />
    public override string ToString() => $"ImageEncoder({Encoder}, {Projector})";
}
