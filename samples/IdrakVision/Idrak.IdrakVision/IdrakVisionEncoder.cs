// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Imaging;
using Idrak.Models.Abstractions;

namespace Idrak.IdrakVision;

/// <summary>
/// An IdrakVision image encoder: an image becomes one block of <see cref="IdrakVisionTokens.TokensPerImage"/> tokens
/// (the pooled grid, row-major), through the preprocessing, the tower and the projector.
/// </summary>
public sealed class IdrakVisionEncoder : Module, IVisionEncoder, IVisionEncoderStages
{
    private readonly ImageTokenLayout _layout;

    internal IdrakVisionEncoder(IdrakVisionPart vision, IdrakVisionTower tower, IdrakVisionProjector projector, ImagePreprocessor preprocessor)
    {
        Vision = vision;
        Tower = tower;
        Projector = projector;
        Preprocessor = preprocessor;
        int side = vision.Config.PatchesPerSide / vision.Config.PoolSize;
        _layout = new ImageTokenLayout(side * side) { Grid = [side, side] };
    }

    /// <summary>The vision part that built it.</summary>
    public IdrakVisionPart Vision { get; }

    /// <summary>The tower.</summary>
    public IdrakVisionTower Tower { get; }

    /// <summary>The projector.</summary>
    public IdrakVisionProjector Projector { get; }

    /// <summary>The preprocessing (preprocessor_config.json).</summary>
    public ImagePreprocessor Preprocessor { get; }

    /// <inheritdoc />
    public int Width => Vision.Width;

    /// <inheritdoc />
    public Device Device => WeightsDevice ?? Device.Default;

    /// <inheritdoc />
    public IReadOnlyList<ImageTokenLayout> Blocks(ImageData image, VisionOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(image);
        options?.ThrowIfUnknown(IdrakVisionFamily.Architecture, []);
        return [_layout];
    }

    IReadOnlyList<ImageFeatures> IVisionEncoder.Encode(IReadOnlyList<ImageData> images, VisionOptions? options)
    {
        ArgumentNullException.ThrowIfNull(images);
        options?.ThrowIfUnknown(IdrakVisionFamily.Architecture, []);
        var result = new List<ImageFeatures>(images.Count);
        if (images.Count == 0)
        {
            return result;
        }

        using var batch = Encode(images);
        for (int i = 0; i < images.Count; i++)
        {
            result.Add(new ImageFeatures(batch.Narrow(0, i, 1).Reshape(_layout.Tokens, Width), _layout));
        }

        return result;
    }

    /// <inheritdoc />
    public Tensor PixelValues(ImageData image, VisionOptions? options = null)
    {
        options?.ThrowIfUnknown(IdrakVisionFamily.Architecture, []);
        var c = Vision.Config;
        var values = new float[c.Channels * c.ImageSize * c.ImageSize];
        Preprocessor.Pixels(image, values);
        return Tensor.From(values, [1, c.Channels, c.ImageSize, c.ImageSize], Device.Cpu);
    }

    /// <inheritdoc />
    Tensor IVisionEncoderStages.Tower(Tensor pixelValues)
    {
        using var noGrad = Autograd.NoGrad();
        using var moved = OnDevice(pixelValues);
        return Tower.Forward(moved ?? pixelValues);
    }

    /// <inheritdoc />
    public Tensor Features(Tensor pixelValues)
    {
        using var moved = OnDevice(pixelValues);
        return Predict(moved ?? pixelValues);
    }

    /// <summary>The features [images, tokens, width] of decoded images.</summary>
    public Tensor Encode(IReadOnlyList<ImageData> images)
    {
        var c = Vision.Config;
        using var input = Preprocessor.ProcessBatch(images, WeightsDevice);
        if (!input.Shape.SequenceEqual([images.Count, c.Channels, c.ImageSize, c.ImageSize]))
        {
            throw new InvalidOperationException($"The preprocessor gives {Tensor.FormatShape(input.Shape)}; the tower takes [{images.Count}, {c.Channels}, {c.ImageSize}, {c.ImageSize}].");
        }

        return Predict(input);
    }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => Projector.Forward(Tower.Forward(input));

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => [Tower, Projector];

    /// <inheritdoc />
    public override void Dispose()
    {
        Tower.Dispose();
        Projector.Dispose();
        GC.SuppressFinalize(this);
    }

    /// <inheritdoc />
    public override string ToString() => $"IdrakVisionEncoder({Tower}, {Projector})";

    private Tensor? OnDevice(Tensor pixelValues)
    {
        var device = WeightsDevice ?? pixelValues.Device;
        return pixelValues.Device == device ? null : Tensor.From(pixelValues.ToArray(), pixelValues.Shape, device);
    }
}
