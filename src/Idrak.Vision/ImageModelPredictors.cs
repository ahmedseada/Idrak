// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Inference;
using Idrak.Models.Abstractions;

namespace Idrak.Vision;

/// <summary>
/// An image model loaded through its family (<c>ImageModels.Load</c> in Idrak) as a predictor ready for images, with the
/// family's preprocessing: a classifier (Idrak's <see cref="Predictor"/>), a <see cref="ModelDetector"/> with the family's
/// decoder, a <see cref="ModelSegmenter"/>, or the raw outputs. They use the model's network without owning it: dispose the
/// <see cref="ImageModel"/> when done.
/// </summary>
/// <remarks>
/// <code>
/// using var model = ImageModels.Load("models/tiny-detector");
/// var detector = model.Detector(new DetectorOptions { MinScore = 0.4f });
/// foreach (var d in detector.Detect(ImageCodecs.Decode("street.jpg"))) Console.WriteLine($"{d.Label} {d.Score:P0} at {d.Box}");
/// </code>
/// </remarks>
public static class ImageModelPredictors
{
    /// <summary>
    /// A classifier: images in, the most likely class (with every class's probability) out, through the family's
    /// preprocessing and a softmax. Add builder steps (batch size, device, warm-up) before <c>Build()</c>.
    /// </summary>
    /// <param name="model">A classification model.</param>
    /// <param name="labels">The class names, when the checkpoint gives none (or to rename them).</param>
    /// <exception cref="InvalidOperationException">The model is not a classifier, or has no labels and none are given.</exception>
    /// <exception cref="NotSupportedException">Its preprocessing gives images of different sizes (a classifier batches one size).</exception>
    public static PredictorBuilder<ImageData, ClassPrediction> Classifier(this ImageModel model, IReadOnlyList<string>? labels = null)
    {
        Require(model, ImageTask.Classification);
        var names = labels ?? model.Labels
            ?? throw new InvalidOperationException($"The {model.Architecture} classifier's checkpoint names no classes (config.json's id2label); pass the labels.");
        return Outputs(model).Softmax().Classes(names);
    }

    /// <summary>
    /// The network's outputs per image (features, logits) through the family's preprocessing, any task. Add builder steps
    /// before <c>Build()</c>.
    /// </summary>
    /// <exception cref="NotSupportedException">Its preprocessing gives images of different sizes.</exception>
    public static PredictorBuilder<ImageData, float[]> Outputs(this ImageModel model)
    {
        ArgumentNullException.ThrowIfNull(model);
        var shape = model.InputShape
            ?? throw new NotSupportedException($"The {model.Architecture} model's preprocessing gives images of different sizes (a resize by the shortest edge without a crop); a predictor batches one size.");
        var preprocessor = model.Preprocessor;
        return Predictor.For(model.Network).Input<ImageData>(preprocessor.Pixels).InputShape([.. shape]);
    }

    /// <summary>
    /// A detector over the model: the family's preprocessing, the decoder it names (registered in
    /// <see cref="Abstractions.DetectionDecoders"/>) made with its settings, boxes mapped back to each image and filtered by
    /// <paramref name="options"/> (labelled with the model's labels unless the options give classes).
    /// </summary>
    /// <exception cref="InvalidOperationException">The model is not a detector.</exception>
    /// <exception cref="NotSupportedException">Its decoder is not registered.</exception>
    public static ModelDetector Detector(this ImageModel model, DetectorOptions? options = null, Device? device = null)
    {
        Require(model, ImageTask.Detection);
        options ??= new DetectorOptions();
        options = options.Classes is null ? options with { Classes = model.Labels } : options;
        return new ModelDetector(model.Network, model.Preprocessor, model.Decoder!, model.DecoderSettings, options, device);
    }

    /// <summary>A segmenter over the model, with the family's preprocessing (a center crop is refused).</summary>
    /// <exception cref="InvalidOperationException">The model is not a segmenter.</exception>
    public static ModelSegmenter Segmenter(this ImageModel model, Device? device = null)
    {
        Require(model, ImageTask.Segmentation);
        return new ModelSegmenter(model.Network, model.Preprocessor, device);
    }

    private static void Require(ImageModel model, ImageTask task)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (model.Task != task)
        {
            throw new InvalidOperationException($"The {model.Architecture} model is a {model.Task.ToString().ToLowerInvariant()} model, not a {task.ToString().ToLowerInvariant()} one.");
        }
    }
}
