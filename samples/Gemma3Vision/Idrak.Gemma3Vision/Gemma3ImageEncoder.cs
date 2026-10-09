// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Data;
using Idrak.Data.Abstractions;

namespace Idrak.Gemma3Vision;

/// <summary>
/// Gemma 3's image side: images become the embeddings of their soft tokens, [images, <see cref="Gemma3ImageTokens.TokensPerImage"/>,
/// text width], which replace the <c>&lt;image_soft_token&gt;</c> rows of the prompt's embeddings. Pixel values go
/// through <see cref="Encoder"/> (SigLIP) and <see cref="Projector"/>; files, encoded bytes and chat images are first
/// decoded (<see cref="ImageCodecs"/>) and preprocessed (<see cref="Preprocessor"/>, the model's
/// <c>preprocessor_config.json</c>). As Idrak's <see cref="IVisionEncoder"/>, every image takes the same layout: a square
/// grid of <see cref="Gemma3ImageTokens.TokensPerImage"/> tokens. Made by <see cref="Gemma3Vision.CreateEncoder(Device?, ImagePreprocessor?)"/>;
/// as a module, its forward pass takes pixel values [images, channels, size, size]. Every <c>Encode</c> runs without
/// recording gradients and returns tensors the caller disposes.
/// </summary>
public sealed class Gemma3ImageEncoder : Module, IVisionEncoder, IVisionEncoderStages
{
    private readonly ImageTokenLayout _layout;

    internal Gemma3ImageEncoder(Gemma3Vision vision, SiglipVisionEncoder encoder, Gemma3Projector projector, ImagePreprocessor preprocessor)
    {
        Vision = vision;
        Encoder = encoder;
        Projector = projector;
        Preprocessor = preprocessor;
        int side = (int)Math.Round(Math.Sqrt(vision.ImageTokens.TokensPerImage));
        _layout = new ImageTokenLayout(vision.ImageTokens.TokensPerImage) { Grid = [side, side] };
    }

    /// <summary>The vision part this encoder was built from (its configuration and image token ids).</summary>
    public Gemma3Vision Vision { get; }

    /// <summary>The SigLIP encoder: pixel values to patch outputs [images, patches, vision width].</summary>
    public SiglipVisionEncoder Encoder { get; }

    /// <summary>The projector: patch outputs to soft-token embeddings [images, tokens per image, text width].</summary>
    public Gemma3Projector Projector { get; }

    /// <summary>How images become pixel values (resize, rescale, normalize, optionally grayscale).</summary>
    public ImagePreprocessor Preprocessor { get; }

    /// <inheritdoc />
    public int Width => Vision.TextDim;

    /// <inheritdoc />
    public Device Device => WeightsDevice ?? Device.Default;

    /// <inheritdoc />
    public ImageTokenLayout Layout(ImageData image) => _layout;

    /// <inheritdoc />
    IReadOnlyList<ImageFeatures> IVisionEncoder.Encode(IReadOnlyList<ImageData> images)
    {
        using var batch = Encode(images);
        var result = new List<ImageFeatures>(images.Count);
        try
        {
            for (int i = 0; i < images.Count; i++)
            {
                result.Add(new ImageFeatures(batch.Narrow(0, i, 1).Reshape(_layout.Tokens, Width), _layout));
            }
        }
        catch
        {
            result.ForEach(f => f.Dispose());
            throw;
        }

        return result;
    }

    /// <inheritdoc />
    public Tensor PixelValues(ImageData image) => Preprocessor.Process(image, Device.Cpu);

    /// <inheritdoc />
    public Tensor Tower(Tensor pixelValues)
    {
        using var noGrad = Autograd.NoGrad();
        using var moved = OnDevice(pixelValues);
        return Encoder.Forward(moved ?? pixelValues);
    }

    /// <inheritdoc />
    public Tensor Features(Tensor pixelValues) => Encode(pixelValues);

    /// <summary>
    /// The soft-token embeddings of pixel values [images, channels, size, size] or of one image [channels, size, size]
    /// (as <see cref="ImagePreprocessor.Process(ImageData, Device?)"/> gives them): [images, tokens per image, text width].
    /// </summary>
    public Tensor Encode(Tensor pixelValues)
    {
        ArgumentNullException.ThrowIfNull(pixelValues);
        using var moved = OnDevice(pixelValues);
        return Predict(moved ?? pixelValues);
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
    public override string ToString() => $"Gemma3ImageEncoder({Encoder}, {Projector})";

    // The pixel values on the encoder's device: a copy when they are elsewhere, else null (use them as they are).
    private Tensor? OnDevice(Tensor pixelValues)
    {
        var device = WeightsDevice ?? pixelValues.Device;
        return pixelValues.Device == device ? null : Tensor.From(pixelValues.ToArray(), pixelValues.Shape, device);
    }
}
