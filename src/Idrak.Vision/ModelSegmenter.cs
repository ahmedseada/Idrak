// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Data;
using Idrak.Vision.Abstractions;

namespace Idrak.Vision;

/// <summary>
/// An <see cref="ISegmenter"/> over any network that maps images [N, channels, height, width] to logits
/// [N, classes, h, w]: each image becomes the network's input (resized with values in [0, 1], normalization in the
/// network; or through a model family's <see cref="ImagePreprocessor"/>), the masks come from the logits, and are resized
/// back to the image (nearest neighbour).
/// </summary>
public sealed class ModelSegmenter : ISegmenter
{
    private readonly Module _model;
    private readonly Device _device;
    private readonly int _channels, _height, _width;
    private readonly ImagePreprocessor? _preprocessor;

    /// <summary>A segmenter over <paramref name="model"/>, its images resized to <paramref name="channels"/> x <paramref name="height"/> x <paramref name="width"/> (values in [0, 1]).</summary>
    public ModelSegmenter(Module model, int channels, int height, int width, Device? device = null)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        (_channels, _height, _width) = (channels, height, width);
        _device = device ?? model.WeightsDevice ?? Device.Default;
    }

    /// <summary>
    /// A segmenter whose images become the network's input through <paramref name="preprocessor"/> (a model family's
    /// resize and normalization). A center crop is refused: the mask would not cover the image's edges.
    /// </summary>
    /// <exception cref="NotSupportedException">The preprocessor crops.</exception>
    public ModelSegmenter(Module model, ImagePreprocessor preprocessor, Device? device = null)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _preprocessor = preprocessor ?? throw new ArgumentNullException(nameof(preprocessor));
        if (preprocessor.CenterCrop)
        {
            throw new NotSupportedException("A segmenter's preprocessing cannot center-crop: the masks would not cover the image's edges.");
        }

        _device = device ?? model.WeightsDevice ?? Device.Default;
    }

    /// <inheritdoc />
    public SegmentationMask Segment(ImageData image) => Segment([image])[0];

    /// <summary>
    /// The masks of several images, run through the network as one batch (with a preprocessor whose output size follows
    /// each image, images of different sizes run one at a time).
    /// </summary>
    public IReadOnlyList<SegmentationMask> Segment(IReadOnlyList<ImageData> images)
    {
        ArgumentNullException.ThrowIfNull(images);
        if (images.Count == 0)
        {
            return [];
        }

        var results = new SegmentationMask[images.Count];
        if (_preprocessor is null)
        {
            int input = _channels * _height * _width;
            var batch = new float[images.Count * input];
            for (int i = 0; i < images.Count; i++)
            {
                images[i].Resize(_channels, _height, _width, batch.AsSpan(i * input, input));
            }

            Run(images, 0, images.Count, batch, _channels, _height, _width, results);
            return results;
        }

        for (int start = 0; start < images.Count;)
        {
            var size = _preprocessor.OutputSize(images[start].Height, images[start].Width);
            int end = start + 1;
            while (end < images.Count && _preprocessor.OutputSize(images[end].Height, images[end].Width) == size)
            {
                end++;
            }

            float[]? batch = null;
            int channels = 0;
            for (int i = start; i < end; i++)
            {
                var pixels = _preprocessor.Pixels(images[i]);
                channels = pixels.Length / (size.Height * size.Width);
                batch ??= new float[(end - start) * pixels.Length];
                pixels.CopyTo(batch, (i - start) * pixels.Length);
            }

            Run(images, start, end, batch!, channels, size.Height, size.Width, results);
            start = end;
        }

        return results;
    }

    private void Run(IReadOnlyList<ImageData> images, int start, int end, float[] batch, int channels, int height, int width, SegmentationMask[] results)
    {
        using var x = Tensor.From(batch, [end - start, channels, height, width], _device);
        using var logits = _model.Predict(x);
        var masks = SegmentationMask.FromLogits(logits);
        for (int i = 0; i < masks.Length; i++)
        {
            var image = images[start + i];
            results[start + i] = masks[i].Width == image.Width && masks[i].Height == image.Height ? masks[i] : masks[i].Resize(image.Width, image.Height);
        }
    }
}
