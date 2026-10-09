// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Data;
using Idrak.Vision.Abstractions;

namespace Idrak.Vision;

/// <summary>How <see cref="ModelDetector"/> filters its detections.</summary>
public sealed record DetectorOptions
{
    /// <summary>Boxes overlapping a better one by more than this intersection over union are dropped (default 0.5).</summary>
    public float IouThreshold { get; init; } = 0.5f;

    /// <summary>Detections scoring lower are dropped (default 0.25).</summary>
    public float MinScore { get; init; } = 0.25f;

    /// <summary>Whether only boxes of the same class suppress each other (default true).</summary>
    public bool PerClass { get; init; } = true;

    /// <summary>The most detections kept per image (default 300).</summary>
    public int MaxDetections { get; init; } = 300;

    /// <summary>Class names, to label detections (by class index); null leaves them unlabelled.</summary>
    public IReadOnlyList<string>? Classes { get; init; }
}

/// <summary>
/// An <see cref="IObjectDetector"/> over any detection network: each image becomes the network's input (resized to
/// channels x height x width with values in [0, 1], normalization in the network, see <c>NetworkBuilder.Normalize</c>; or
/// through an <see cref="ImagePreprocessor"/>, a model family's resize, crop and normalization), the network runs on its
/// device, the <see cref="DetectionDecoder"/> (given, or registered in <see cref="DetectionDecoders"/> and named) reads its
/// outputs, and the boxes are mapped back to the image, clipped and filtered by non-maximum suppression.
/// </summary>
public sealed class ModelDetector : IObjectDetector
{
    private readonly Module _model;
    private readonly DetectionDecoder _decoder;
    private readonly DetectorOptions _options;
    private readonly Device _device;
    private readonly int _channels, _height, _width;
    private readonly ImagePreprocessor? _preprocessor;

    /// <summary>A detector over <paramref name="model"/>, its images resized to <paramref name="channels"/> x <paramref name="height"/> x <paramref name="width"/> (values in [0, 1]).</summary>
    public ModelDetector(Module model, int channels, int height, int width, DetectionDecoder decoder, DetectorOptions? options = null, Device? device = null)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _decoder = decoder ?? throw new ArgumentNullException(nameof(decoder));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        (_channels, _height, _width) = (channels, height, width);
        _options = options ?? new DetectorOptions();
        _device = device ?? model.WeightsDevice ?? Device.Default;
    }

    /// <summary>
    /// A detector whose decoder is the one registered as <paramref name="decoder"/> in <see cref="DetectionDecoders"/>,
    /// made for this input size, <paramref name="options"/>' classes and <paramref name="settings"/>.
    /// </summary>
    /// <exception cref="NotSupportedException">No decoder is registered under that name.</exception>
    public ModelDetector(Module model, int channels, int height, int width, string decoder, JsonObject? settings = null, DetectorOptions? options = null, Device? device = null)
        : this(model, channels, height, width, DetectionDecoders.Create(decoder, new DetectionDecoderContext(height, width, options?.Classes, settings ?? [])), options, device)
    {
    }

    /// <summary>
    /// A detector whose images become the network's input through <paramref name="preprocessor"/> (a model family's resize,
    /// crop and normalization); boxes are mapped back through the same resize and crop.
    /// </summary>
    public ModelDetector(Module model, ImagePreprocessor preprocessor, DetectionDecoder decoder, DetectorOptions? options = null, Device? device = null)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _preprocessor = preprocessor ?? throw new ArgumentNullException(nameof(preprocessor));
        _decoder = decoder ?? throw new ArgumentNullException(nameof(decoder));
        _options = options ?? new DetectorOptions();
        _device = device ?? model.WeightsDevice ?? Device.Default;
    }

    /// <summary>
    /// A detector with <paramref name="preprocessor"/> and the decoder registered as <paramref name="decoder"/> in
    /// <see cref="DetectionDecoders"/> (its context's input size is the preprocessor's fixed output size, or 0 when that
    /// follows the image).
    /// </summary>
    /// <exception cref="NotSupportedException">No decoder is registered under that name.</exception>
    public ModelDetector(Module model, ImagePreprocessor preprocessor, string decoder, JsonObject? settings = null, DetectorOptions? options = null, Device? device = null)
        : this(model, preprocessor, DetectionDecoders.Create(decoder, ContextFor(preprocessor, options, settings)), options, device)
    {
    }

    /// <inheritdoc />
    public IReadOnlyList<Detection> Detect(ImageData image) => Detect([image])[0];

    /// <summary>
    /// Detects objects in several images, run through the network as one batch (with a preprocessor whose output size
    /// follows each image, images of different sizes run one at a time).
    /// </summary>
    public IReadOnlyList<IReadOnlyList<Detection>> Detect(IReadOnlyList<ImageData> images)
    {
        ArgumentNullException.ThrowIfNull(images);
        if (images.Count == 0)
        {
            return [];
        }

        var results = new IReadOnlyList<Detection>[images.Count];
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

        // Runs of images of the same network input size go through as one batch.
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

    private void Run(IReadOnlyList<ImageData> images, int start, int end, float[] batch, int channels, int height, int width, IReadOnlyList<Detection>[] results)
    {
        int count = end - start;
        float[] outputs;
        int[] shape;
        using (var x = Tensor.From(batch, [count, channels, height, width], _device))
        using (var y = _model.Predict(x))
        {
            outputs = y.ToArray();
            shape = [.. y.Shape[1..]];
        }

        int per = outputs.Length / count;
        for (int i = 0; i < count; i++)
        {
            var image = images[start + i];
            var toImage = ToImage(image.Height, image.Width, height, width);
            var found = _decoder(outputs.AsSpan(i * per, per), shape)
                .Select(d => d with
                {
                    Box = toImage(d.Box).Clip(image.Width, image.Height),
                    Label = d.Label ?? (_options.Classes is { } names && d.Class >= 0 && d.Class < names.Count ? names[d.Class] : null),
                })
                .Where(d => d.Box.Area > 0)
                .ToList();
            results[start + i] = NonMaxSuppression.Apply(found, _options.IouThreshold, _options.MinScore, _options.PerClass, _options.MaxDetections);
        }
    }

    // Maps a box in the network's input pixels back to the image's: undoes the center crop (an offset), then the resize.
    private Func<BoundingBox, BoundingBox> ToImage(int imageHeight, int imageWidth, int height, int width)
    {
        if (_preprocessor is null)
        {
            float sx = imageWidth / (float)width, sy = imageHeight / (float)height;
            return box => box.Scale(sx, sy);
        }

        var (rh, rw) = _preprocessor.ResizedSize(imageHeight, imageWidth);
        float top = _preprocessor.CenterCrop ? MathF.Floor((rh - height) / 2f) : 0, left = _preprocessor.CenterCrop ? MathF.Floor((rw - width) / 2f) : 0;
        float scaleX = imageWidth / (float)rw, scaleY = imageHeight / (float)rh;
        return box => (box with { X = box.X + left, Y = box.Y + top }).Scale(scaleX, scaleY);
    }

    private static DetectionDecoderContext ContextFor(ImagePreprocessor preprocessor, DetectorOptions? options, JsonObject? settings)
    {
        ArgumentNullException.ThrowIfNull(preprocessor);
        var (h, w) = preprocessor.CenterCrop ? (preprocessor.CropHeight, preprocessor.CropWidth)
            : preprocessor.Resize && preprocessor.ShortestEdge <= 0 ? (preprocessor.Height, preprocessor.Width)
            : (0, 0);
        return new DetectionDecoderContext(h, w, options?.Classes, settings ?? []);
    }
}
