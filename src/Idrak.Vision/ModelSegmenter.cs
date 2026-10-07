// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Vision.Abstractions;

namespace Idrak.Vision;

/// <summary>
/// An <see cref="ISegmenter"/> over any network that maps images [N, channels, height, width] (values in [0, 1]; put
/// normalization in the network) to logits [N, classes, h, w]: each image is resized to the network's input, the
/// masks come from the logits, and are resized back to the image (nearest neighbour).
/// </summary>
public sealed class ModelSegmenter(Module model, int channels, int height, int width, Device? device = null) : ISegmenter
{
    private readonly Module _model = model ?? throw new ArgumentNullException(nameof(model));
    private readonly Device _device = device ?? model.WeightsDevice ?? Device.Default;

    /// <inheritdoc />
    public SegmentationMask Segment(ImageData image) => Segment([image])[0];

    /// <summary>The masks of several images, run through the network as one batch.</summary>
    public IReadOnlyList<SegmentationMask> Segment(IReadOnlyList<ImageData> images)
    {
        ArgumentNullException.ThrowIfNull(images);
        if (images.Count == 0)
        {
            return [];
        }

        int input = channels * height * width;
        var batch = new float[images.Count * input];
        for (int i = 0; i < images.Count; i++)
        {
            images[i].Resize(channels, height, width, batch.AsSpan(i * input, input));
        }

        using var x = Tensor.From(batch, [images.Count, channels, height, width], _device);
        using var logits = _model.Predict(x);
        var masks = SegmentationMask.FromLogits(logits);
        return [.. masks.Select((m, i) => m.Width == images[i].Width && m.Height == images[i].Height ? m : m.Resize(images[i].Width, images[i].Height))];
    }
}
