// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.QwenVision;

/// <summary>
/// Qwen-VL's image side: images become the embeddings of their image tokens, [tokens, text width] per image, which
/// replace the <c>&lt;|image_pad|&gt;</c> rows of the prompt's embeddings. Each image is one block whose layout is its
/// merged grid (<see cref="QwenImageGrid.Layout"/>): blocks differ in size from image to image, unlike Gemma 3's fixed
/// square. Pixels go through <see cref="Processor"/>, <see cref="VisionTower"/> and <see cref="Merger"/>; several images in
/// one request run as one pass of the tower by their spans. As Idrak's <see cref="IVisionEncoder"/> and
/// <see cref="IVisionEncoderStages"/>, every <c>Encode</c> runs without recording gradients and returns tensors the
/// caller disposes. Made by <see cref="QwenVision.CreateEncoder"/>. Built in plan 16, step 5.
/// </summary>
public sealed class QwenImageEncoder : Module, IVisionEncoder, IVisionEncoderStages
{
    internal QwenImageEncoder(QwenVision vision, QwenVisionTower tower, QwenPatchMerger merger, QwenImageProcessor processor, Device device) =>
        (Vision, VisionTower, Merger, Processor, Device) = (vision, tower, merger, processor, device);

    /// <summary>The vision part this encoder was built from (its configuration and image token ids).</summary>
    public QwenVision Vision { get; }

    /// <summary>The vision transformer: pixel patches to patch outputs [patches, tower width].</summary>
    public QwenVisionTower VisionTower { get; }

    /// <summary>The merger: patch outputs to image-token embeddings [tokens, text width].</summary>
    public QwenPatchMerger Merger { get; }

    /// <summary>How images become pixel patches (the smart resize, rescale, normalize, patches and grid).</summary>
    public QwenImageProcessor Processor { get; }

    /// <inheritdoc />
    public int Width => Vision.TextDim;

    /// <inheritdoc />
    public Device Device { get; }

    /// <inheritdoc />
    public IReadOnlyList<ImageTokenLayout> Blocks(ImageData image, VisionOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(image);
        return [Processor.With(options).Grid(image).Layout(Vision.Config.MergeSize)];
    }

    /// <inheritdoc />
    IReadOnlyList<ImageFeatures> IVisionEncoder.Encode(IReadOnlyList<ImageData> images, VisionOptions? options)
    {
        ArgumentNullException.ThrowIfNull(images);
        throw QwenVisionPlugin.NotBuilt("encoding images", 5);
    }

    /// <inheritdoc />
    public Tensor PixelValues(ImageData image, VisionOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(image);
        throw QwenVisionPlugin.NotBuilt("pixel patches as a tensor", 5);
    }

    /// <inheritdoc />
    public Tensor Tower(Tensor pixelValues) => throw QwenVisionPlugin.NotBuilt("the tower stage", 5);

    /// <inheritdoc />
    public Tensor Features(Tensor pixelValues) => throw QwenVisionPlugin.NotBuilt("the features stage", 5);

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => Merger.Forward(VisionTower.Forward(input));

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => [VisionTower, Merger];

    /// <inheritdoc />
    public override void Dispose()
    {
        VisionTower.Dispose();
        Merger.Dispose();
        GC.SuppressFinalize(this);
    }

    /// <inheritdoc />
    public override string ToString() => $"QwenImageEncoder({VisionTower}, {Merger})";
}
