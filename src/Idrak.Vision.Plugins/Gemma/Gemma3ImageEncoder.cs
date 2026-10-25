// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Imaging;
using Idrak.Models.Abstractions;
using Idrak.Vision;

namespace Idrak.Vision.Plugins.Gemma;

/// <summary>
/// Gemma 3's image side: images become the embeddings of their soft tokens, [images, <see cref="Gemma3ImageTokens.TokensPerImage"/>,
/// text width], which replace the <c>&lt;image_soft_token&gt;</c> rows of the prompt's embeddings. Pixel values go
/// through <see cref="Encoder"/> (SigLIP) and <see cref="Projector"/>; files, encoded bytes and chat images are first
/// decoded (<see cref="ImageCodecs"/>) and preprocessed (<see cref="Preprocessor"/>, the model's
/// <c>preprocessor_config.json</c>). As Idrak's <see cref="IVisionEncoder"/>, every block takes the same layout: a square
/// grid of <see cref="Gemma3ImageTokens.TokensPerImage"/> tokens; an image is one block, or with pan and scan
/// (<see cref="PanAndScan"/>, or a request's vision options) the whole image and then its crops, one block each.
/// Made by <see cref="Gemma3Vision.CreateEncoder(Device?, ImagePreprocessor?, Gemma3PanAndScan?, EncoderWeights)"/>;
/// as a module, its forward pass takes pixel values [images, channels, size, size]. Every <c>Encode</c> runs without
/// recording gradients and returns tensors the caller disposes.
/// </summary>
public sealed class Gemma3ImageEncoder : Module, ITrainableVisionEncoder, IVisionEncoderStages
{
    private readonly ImageTokenLayout _layout;

    internal Gemma3ImageEncoder(Gemma3Vision vision, VisionTransformerEncoder encoder, Gemma3Projector projector, ImagePreprocessor preprocessor, Gemma3PanAndScan panAndScan)
    {
        Vision = vision;
        Encoder = encoder;
        Projector = projector;
        Preprocessor = preprocessor;
        PanAndScan = panAndScan;
        int side = (int)Math.Round(Math.Sqrt(vision.ImageTokens.TokensPerImage));
        _layout = new ImageTokenLayout(vision.ImageTokens.TokensPerImage) { Grid = [side, side] };
    }

    /// <summary>The vision part this encoder was built from (its configuration and image token ids).</summary>
    public Gemma3Vision Vision { get; }

    // Training (the family's members over this encoder's modules, as ITrainableVisionEncoder).
    IPretrainedVision ITrainableVisionEncoder.Vision => Vision;

    IReadOnlyList<Tensor> ITrainableVisionEncoder.ParametersOf(string part) => Vision.Parameters(this, part);

    Tensor ITrainableVisionEncoder.FeaturesForTraining(Tensor towerOutput) => Vision.Features(this, towerOutput);

    Tensor ITrainableVisionEncoder.TowerForTraining(Tensor pixelValues) => Vision.Tower(this, pixelValues);

    IReadOnlyDictionary<string, Tensor> ITrainableVisionEncoder.ExportPart(string part) => Vision.Export(this, part);

    IReadOnlyCollection<string> ITrainableVisionEncoder.ImportParts(IReadOnlyDictionary<string, Tensor> tensors) => Vision.Import(this, tensors);

    /// <summary>The SigLIP encoder: pixel values to patch outputs [images, patches, vision width].</summary>
    public VisionTransformerEncoder Encoder { get; }

    /// <summary>The projector: patch outputs to soft-token embeddings [images, tokens per image, text width].</summary>
    public Gemma3Projector Projector { get; }

    /// <summary>How images become pixel values (resize, rescale, normalize, optionally grayscale).</summary>
    public ImagePreprocessor Preprocessor { get; }

    /// <summary>Pan and scan for every image unless a request's vision options say otherwise (<see cref="Gemma3PanAndScan.With"/>).</summary>
    public Gemma3PanAndScan PanAndScan { get; }

    /// <inheritdoc />
    public int Width => Vision.TextDim;

    /// <inheritdoc />
    public Device Device => WeightsDevice ?? Device.Default;

    /// <summary>One block's layout: a square grid of <see cref="Gemma3ImageTokens.TokensPerImage"/> tokens.</summary>
    public ImageTokenLayout BlockLayout => _layout;

    /// <summary>
    /// The images the encoder sees for <paramref name="image"/> under <paramref name="options"/> (over
    /// <see cref="PanAndScan"/>): the image itself, then its crops (<see cref="Gemma3PanAndScan.Crops"/>, cut from the
    /// decoded pixels before any resize), each one block.
    /// </summary>
    /// <exception cref="ArgumentException">An option Gemma 3 does not take, or a value out of range.</exception>
    public IReadOnlyList<ImageData> Views(ImageData image, VisionOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(image);
        var crops = PanAndScan.With(options).Crops(image.Height, image.Width);
        return crops.Count == 0 ? [image] : [image, .. crops.Select(c => Crop(image, c.Top, c.Left, c.Height, c.Width))];
    }

    /// <inheritdoc />
    public IReadOnlyList<ImageTokenLayout> Blocks(ImageData image, VisionOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(image);
        return [.. Enumerable.Repeat(_layout, 1 + PanAndScan.With(options).Crops(image.Height, image.Width).Count)];
    }

    /// <inheritdoc />
    IReadOnlyList<ImageFeatures> IVisionEncoder.Encode(IReadOnlyList<ImageData> images, VisionOptions? options)
    {
        ArgumentNullException.ThrowIfNull(images);
        _ = PanAndScan.With(options);                                                      // refused before anything runs
        var result = new List<ImageFeatures>(images.Count);
        try
        {
            // One image at a time (the whole image and its crops in one batch): the batch is at most 1 + MaxCrops images.
            foreach (var image in images)
            {
                var views = Views(image, options);
                using var batch = Encode(views);
                for (int i = 0; i < views.Count; i++)
                {
                    result.Add(new ImageFeatures(batch.Narrow(0, i, 1).Reshape(_layout.Tokens, Width), _layout));
                }
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
    public Tensor PixelValues(ImageData image, VisionOptions? options = null)
    {
        var views = Views(image, options);
        var config = Vision.Encoder;
        int each = config.Channels * config.ImageSize * config.ImageSize;
        float[] values = System.Buffers.ArrayPool<float>.Shared.Rent(views.Count * each);   // written, copied into the tensor, given back
        try
        {
            for (int i = 0; i < views.Count; i++)
            {
                Preprocessor.Pixels(views[i], values.AsSpan(i * each, each));
            }

            return Tensor.From(values.AsSpan(0, views.Count * each), [views.Count, config.Channels, config.ImageSize, config.ImageSize], Device.Cpu);
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(values);
        }
    }

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

        // Every view's pixel values as one batch on the encoder's device (ImagePreprocessor.ProcessBatch): on a GPU the views'
        // bytes are uploaded and resized there (the host's values bit for bit); on the CPU written into one pooled buffer,
        // so a page leaves no 9.6 MB arrays on the managed heap either way.
        Tensor input;
        try
        {
            input = Preprocessor.ProcessBatch(images, WeightsDevice);
        }
        catch (ArgumentException)
        {
            input = null!;
        }

        using (input)
        {
            if (input is null || !input.Shape.SequenceEqual([images.Count, config.Channels, config.ImageSize, config.ImageSize]))
            {
                var (h, w) = Preprocessor.OutputSize(images[0].Height, images[0].Width);
                throw new InvalidOperationException($"The preprocessor gives {h} x {w} pixel values per channel; the encoder takes {config.Channels} x {config.ImageSize} x {config.ImageSize}.");
            }

            return Predict(input);
        }
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

    // A rectangle of an image's decoded pixels (every channel): from its 8-bit planes when it holds them (an 8-bit crop
    // with the same values), else from its floats.
    private static ImageData Crop(ImageData image, int top, int left, int height, int width)
    {
        if (image.TryGetPlanes(out var planes, out var levels))
        {
            var cut = new byte[planes.Length][];
            for (int c = 0; c < planes.Length; c++)
            {
                var plane = cut[c] = GC.AllocateUninitializedArray<byte>(height * width);
                for (int y = 0; y < height; y++)
                {
                    planes[c].Span.Slice((top + y) * image.Width + left, width).CopyTo(plane.AsSpan(y * width, width));
                }
            }

            return new ImageData(cut, height, width, levels.ToArray());
        }

        var pixels = new float[image.Channels * height * width];
        for (int c = 0; c < image.Channels; c++)
        {
            for (int y = 0; y < height; y++)
            {
                Array.Copy(image.Pixels, (c * image.Height + top + y) * image.Width + left, pixels, (c * height + y) * width, width);
            }
        }

        return new ImageData(pixels, image.Channels, height, width);
    }

    // The pixel values on the encoder's device: a copy when they are elsewhere, else null (use them as they are).
    private Tensor? OnDevice(Tensor pixelValues)
    {
        var device = WeightsDevice ?? pixelValues.Device;
        return pixelValues.Device == device ? null : Tensor.From(pixelValues.ToArray(), pixelValues.Shape, device);
    }
}
