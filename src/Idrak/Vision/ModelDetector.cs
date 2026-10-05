// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Data;
using Idrak.Layers;

namespace Idrak.Vision;

/// <summary>
/// Turns one image's network outputs into detections, in the network's input pixels (width x height of
/// <see cref="ModelDetector"/>). This is the part that depends on the network (anchors, grid cells, box encoding,
/// how scores are stored); the detector does the rest.
/// </summary>
/// <param name="outputs">The outputs of one image (the model's output for a batch of one, flattened).</param>
/// <param name="outputShape">Their shape, without the batch dimension.</param>
public delegate IEnumerable<Detection> DetectionDecoder(ReadOnlySpan<float> outputs, IReadOnlyList<int> outputShape);

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
/// An <see cref="IObjectDetector"/> over any detection network: each image is resized to the network's input
/// (channels x height x width, values in [0, 1]; put normalization in the network, see
/// <see cref="NetworkBuilder.Normalize"/>), the network runs on its device, the <see cref="DetectionDecoder"/> reads
/// its outputs, and the boxes are scaled back to the image, clipped and filtered by non-maximum suppression.
/// </summary>
public sealed class ModelDetector(Module model, int channels, int height, int width, DetectionDecoder decoder, DetectorOptions? options = null, Device? device = null)
    : IObjectDetector
{
    private readonly Module _model = model ?? throw new ArgumentNullException(nameof(model));
    private readonly DetectionDecoder _decoder = decoder ?? throw new ArgumentNullException(nameof(decoder));
    private readonly DetectorOptions _options = options ?? new DetectorOptions();
    private readonly Device _device = device ?? model.WeightsDevice ?? Device.Default;

    /// <inheritdoc />
    public IReadOnlyList<Detection> Detect(ImageData image) => Detect([image])[0];

    /// <summary>Detects objects in several images, run through the network as one batch.</summary>
    public IReadOnlyList<IReadOnlyList<Detection>> Detect(IReadOnlyList<ImageData> images)
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

        float[] outputs;
        int[] shape;
        using (var x = Tensor.From(batch, [images.Count, channels, height, width], _device))
        using (var y = _model.Predict(x))
        {
            outputs = y.ToArray();
            shape = [.. y.Shape[1..]];
        }

        int per = outputs.Length / images.Count;
        var results = new IReadOnlyList<Detection>[images.Count];
        for (int i = 0; i < images.Count; i++)
        {
            float sx = images[i].Width / (float)width, sy = images[i].Height / (float)height;
            var found = _decoder(outputs.AsSpan(i * per, per), shape)
                .Select(d => d with
                {
                    Box = d.Box.Scale(sx, sy).Clip(images[i].Width, images[i].Height),
                    Label = d.Label ?? (_options.Classes is { } names && d.Class >= 0 && d.Class < names.Count ? names[d.Class] : null),
                })
                .Where(d => d.Box.Area > 0)
                .ToList();
            results[i] = NonMaxSuppression.Apply(found, _options.IouThreshold, _options.MinScore, _options.PerClass, _options.MaxDetections);
        }

        return results;
    }
}
