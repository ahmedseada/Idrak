// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using Idrak.Data;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Models.Abstractions;
using Idrak.Vision.Abstractions;

namespace Idrak.Vision;

/// <summary>Proposes each connected region of the foreground of at least <paramref name="minArea"/> pixels (raster order).</summary>
public sealed class ComponentProposer(Connectivity connectivity = Connectivity.Eight, int minArea = 6) : IRegionProposer
{
    /// <inheritdoc />
    public IReadOnlyList<PixelBox> Propose(ForegroundImage image) =>
        [.. ConnectedComponents.Find(image, connectivity, minArea).Regions.Select(r => r.Box)];
}

/// <summary>
/// The class probabilities of every region a <see cref="RegionClassifier"/> classified, kept in one array
/// ([regions, classes]); read them per region.
/// </summary>
public sealed class RegionPredictions
{
    private readonly float[] _probabilities;

    internal RegionPredictions(IReadOnlyList<string> classes, IReadOnlyList<PixelBox> boxes, float[] probabilities)
    {
        Classes = classes;
        Boxes = boxes;
        _probabilities = probabilities;
    }

    /// <summary>The classes, in the model's output order.</summary>
    public IReadOnlyList<string> Classes { get; }

    /// <summary>The regions' boxes, in the order they were classified.</summary>
    public IReadOnlyList<PixelBox> Boxes { get; }

    /// <summary>The number of regions.</summary>
    public int Count => Boxes.Count;

    /// <summary>A region's probability for every class, in class order.</summary>
    public ReadOnlySpan<float> Probabilities(int region) => _probabilities.AsSpan(region * Classes.Count, Classes.Count);

    /// <summary>A region's likeliest class index.</summary>
    public int Best(int region)
    {
        var p = Probabilities(region);
        int best = 0;
        for (int c = 1; c < p.Length; c++)
        {
            if (p[c] > p[best])
            {
                best = c;
            }
        }

        return best;
    }

    /// <summary>A region's likeliest class name.</summary>
    public string Label(int region) => Classes[Best(region)];

    /// <summary>A region's probability for its likeliest class.</summary>
    public float Confidence(int region) => Probabilities(region)[Best(region)];

    /// <summary>A region's <paramref name="count"/> likeliest classes, best first.</summary>
    public IReadOnlyList<ClassScore> Top(int region, int count)
    {
        var p = Probabilities(region);
        return [.. Enumerable.Range(0, p.Length).OrderByDescending(c => _probabilities[region * Classes.Count + c]).Take(count)
            .Select(c => new ClassScore(Classes[c], _probabilities[region * Classes.Count + c]))];
    }
}

/// <summary>
/// Classifies regions of an image with a classifier of single objects (images [N, 1, size, size] in, one logit per
/// class out): each region is framed by <see cref="ContentFrame"/> straight into one reused batch buffer, and the
/// batches go through the model on its device. Regions come from an <see cref="IRegionProposer"/> or from the caller.
/// </summary>
/// <remarks>
/// <code>
/// using var classifier = RegionClassifier.Load("shapes.ikm").Build();   // a package from Predictor.Save
/// var found = classifier.Classify(ImageCodecs.Decode("board.png"), new ComponentProposer());
/// for (int i = 0; i &lt; found.Count; i++) Console.WriteLine($"{found.Label(i)} at {found.Boxes[i]} ({found.Confidence(i):P0})");
/// </code>
/// Train the classifier on images framed the same way (<see cref="ContentFrame.Reframe"/> on its loader) so that
/// training images and regions look alike.
/// </remarks>
public sealed class RegionClassifier : IDisposable
{
    private readonly RegionClassifierBuilder _settings;
    private readonly Device _device;
    private readonly string[] _classes;

    internal RegionClassifier(RegionClassifierBuilder settings)
    {
        _settings = settings;
        _device = settings.Device ?? settings.Model.WeightsDevice ?? Device.Default;
        _classes = [.. settings.ClassNames ?? throw new InvalidOperationException("Give the model's classes in output order: RegionClassifier.For(model).Classes(...).")];
    }

    /// <summary>Starts a classifier over a model: images [N, 1, size, size] in, one logit per class out.</summary>
    public static RegionClassifierBuilder For(Module model) => new(model ?? throw new ArgumentNullException(nameof(model)), ownsModel: false);

    /// <summary>
    /// Starts a classifier over an image model loaded through its family (<c>ImageModels.Load</c>): its labels, its square
    /// input size (the frame size), and its preprocessing's rescale and normalization applied to the framed regions (grey,
    /// repeated to the model's channels). The model stays the caller's to dispose.
    /// </summary>
    /// <exception cref="InvalidOperationException">The model is not a classifier or names no classes.</exception>
    /// <exception cref="NotSupportedException">Its input is not square, or not one size for every image.</exception>
    public static RegionClassifierBuilder For(ImageModel model)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (model.Task != ImageTask.Classification || model.Labels is null)
        {
            throw new InvalidOperationException(
                $"A region classifier needs a classification model with labels; {model.Architecture} is a {model.Task} model{(model.Labels is null ? " without labels" : "")}.");
        }

        if (model.InputShape is not [int channels, int height, int width] || height != width)
        {
            throw new NotSupportedException(
                $"A region classifier frames regions in squares; the {model.Architecture} model's input is {(model.InputShape is { } s ? string.Join(" x ", s) : "each image's own size")}.");
        }

        var p = model.Preprocessor;
        var scale = new float[channels];
        var shift = new float[channels];
        for (int c = 0; c < channels; c++)
        {
            // A frame value v in [0, 1] is the byte 255 v: rescaled as the preprocessor rescales a byte, then normalized.
            float value = p.Rescale ? (float)(255 * p.RescaleFactor) : 255f, offset = 0f;
            if (p.Normalize)
            {
                float mean = p.Mean[p.Mean.Count == 1 ? 0 : c], std = p.Std[p.Std.Count == 1 ? 0 : c];
                (value, offset) = (value / std, -mean / std);
            }

            (scale[c], shift[c]) = (value, offset);
        }

        return new RegionClassifierBuilder(model.Network, ownsModel: false) { InputScale = scale, InputShift = shift }.Classes(model.Labels).InputSize(height);
    }

    /// <summary>
    /// Starts a classifier from a package written by <see cref="Predictor{TIn, TOut}.Save"/>: its model, class names and
    /// input shape (which sets the frame size). The classifier disposes the model.
    /// </summary>
    public static RegionClassifierBuilder Load(string path, Device? device = null)
    {
        var settings = Predictor.Load(path, device);
        var builder = new RegionClassifierBuilder(settings.Model!, ownsModel: true) { Device = device };
        if (settings.StoredClasses is { } classes)
        {
            builder.Classes(classes);
        }

        if (settings.SampleShape is [1, int h, int w] && h == w)
        {
            builder.InputSize(h);
        }

        return builder;
    }

    /// <summary>The classes, in the model's output order.</summary>
    public IReadOnlyList<string> Classes => _classes;

    /// <summary>The foreground of an image, with this classifier's polarity and threshold settings.</summary>
    public ForegroundImage Foreground(ImageData image) => Vision.Foreground.Extract(image, _settings.ImagePolarity, _settings.ForegroundThreshold);

    /// <summary>Classifies the regions <paramref name="proposer"/> finds in <paramref name="image"/> (default: its connected regions).</summary>
    public RegionPredictions Classify(ImageData image, IRegionProposer? proposer = null)
    {
        var foreground = Foreground(image);
        return Classify(foreground, (proposer ?? new ComponentProposer()).Propose(foreground));
    }

    /// <summary>Classifies the given regions of an image's foreground.</summary>
    public RegionPredictions Classify(ForegroundImage image, IReadOnlyList<PixelBox> regions)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(regions);
        int size = _settings.Size, frame = size * size, classes = _classes.Length, count = regions.Count;
        int channels = _settings.InputScale?.Length ?? 1, sample = channels * frame;
        var probabilities = new float[count * classes];
        if (count == 0)
        {
            return new RegionPredictions(_classes, regions, probabilities);
        }

        int batch = Math.Min(_settings.Batch, count);
        var buffer = ArrayPool<float>.Shared.Rent(batch * sample);
        try
        {
            for (int start = 0; start < count; start += batch)
            {
                int n = Math.Min(batch, count - start);
                for (int i = 0; i < n; i++)
                {
                    var target = buffer.AsSpan(i * sample, sample);
                    ContentFrame.Extract(image, regions[start + i], target[..frame], size, _settings.Border);
                    if (_settings.InputScale is { } scale)
                    {
                        // An image model's input: each channel the frame rescaled and normalized as its preprocessing does
                        // (the first channel last, since the frame is there).
                        for (int c = channels - 1; c >= 0; c--)
                        {
                            float a = scale[c], b = _settings.InputShift![c];
                            var plane = target.Slice(c * frame, frame);
                            for (int k = 0; k < frame; k++)
                            {
                                plane[k] = target[k] * a + b;
                            }
                        }
                    }
                }

                using var x = Tensor.From(buffer.AsSpan(0, n * sample), [n, channels, size, size], _device);
                using var logits = _settings.Model.Predict(x);
                if (logits.Size != n * classes)
                {
                    throw new InvalidOperationException($"The model gave {logits.Size / n} outputs per region for {classes} classes.");
                }

                using var p = logits.Softmax();
                p.ToArray().CopyTo(probabilities, start * classes);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
        }

        return new RegionPredictions(_classes, regions, probabilities);
    }

    /// <summary>Disposes the model when the classifier loaded it (<see cref="Load"/>).</summary>
    public void Dispose()
    {
        if (_settings.OwnsModel)
        {
            _settings.Model.Dispose();
        }
    }
}

/// <summary>Sets up a <see cref="RegionClassifier"/>; every setting but the classes has a working default.</summary>
public sealed class RegionClassifierBuilder
{
    internal RegionClassifierBuilder(Module model, bool ownsModel)
    {
        Model = model;
        OwnsModel = ownsModel;
    }

    internal Module Model { get; }

    internal bool OwnsModel { get; }

    internal IReadOnlyList<string>? ClassNames { get; private set; }

    internal Device? Device { get; set; }

    internal int Size { get; private set; } = 28;

    internal int Border { get; private set; } = 1;

    internal int Batch { get; private set; } = 512;

    internal Polarity ImagePolarity { get; private set; } = Polarity.Auto;

    internal float? ForegroundThreshold { get; private set; }

    // For an image model's input: per channel, frame value · scale + shift (null: the frame as it is, one channel).
    internal float[]? InputScale { get; init; }

    internal float[]? InputShift { get; init; }

    /// <summary>The model's classes, in output order. Required with <see cref="RegionClassifier.For(Module)"/>.</summary>
    public RegionClassifierBuilder Classes(IReadOnlyList<string> classes)
    {
        ArgumentNullException.ThrowIfNull(classes);
        if (classes.Count == 0)
        {
            throw new ArgumentException("Give at least one class.", nameof(classes));
        }

        ClassNames = classes;
        return this;
    }

    /// <summary>The device the model runs on (default: where its weights are).</summary>
    public RegionClassifierBuilder OnDevice(Device device)
    {
        Device = device;
        return this;
    }

    /// <summary>The model's input: size x size frames, the object <paramref name="border"/> pixels from the edge (default 28 and 1).</summary>
    public RegionClassifierBuilder InputSize(int size, int border = 1)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(size, 2);
        Size = size;
        Border = border >= 0 && 2 * border < size ? border : throw new ArgumentOutOfRangeException(nameof(border));
        return this;
    }

    /// <summary>Regions per batch (default 512): larger is faster on a GPU and holds more memory.</summary>
    public RegionClassifierBuilder BatchSize(int regions)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(regions, 1);
        Batch = regions;
        return this;
    }

    /// <summary>How images' foreground is told from their background (default: polarity from the image, Otsu's threshold).</summary>
    public RegionClassifierBuilder Foreground(Polarity polarity, float? threshold = null)
    {
        ImagePolarity = polarity;
        ForegroundThreshold = threshold;
        return this;
    }

    /// <summary>Creates the classifier.</summary>
    public RegionClassifier Build() => new(this);
}
