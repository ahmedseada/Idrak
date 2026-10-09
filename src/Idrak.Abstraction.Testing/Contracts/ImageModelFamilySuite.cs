// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Data;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// An image model loaded through its family, as the family checks see it. The contracts live with the packages that use
/// them (core's <c>IImageModelFamily</c> and <c>ImageModel</c>, Idrak.Vision's predictors) and this kit depends on
/// Idrak.Abstraction alone, so a test wraps the loaded model in these delegates: its task, its labels, and each image's
/// outputs (logits for a classifier, the decoded detections flattened for a detector, the mask for a segmenter, the
/// features for a backbone; whatever the test compares with its reference).
/// </summary>
/// <param name="task">"classification", "detection", "segmentation" or "features".</param>
/// <param name="labels">The class names, or null when the checkpoint gives none.</param>
/// <param name="predict">Each image's outputs, in the order given (the images run as one batch where the model can).</param>
/// <param name="owner">Disposed with this (the loaded model), or null.</param>
public sealed class ImageModelUnderTest(string task, IReadOnlyList<string>? labels, Func<IReadOnlyList<ImageData>, IReadOnlyList<float[]>> predict, IDisposable? owner = null)
    : IDisposable
{
    /// <summary>The model's task.</summary>
    public string Task { get; } = task;

    /// <summary>The class names, or null.</summary>
    public IReadOnlyList<string>? Labels { get; } = labels;

    /// <summary>Each image's outputs.</summary>
    public IReadOnlyList<float[]> Predict(IReadOnlyList<ImageData> images) => predict(images);

    /// <summary>Disposes the loaded model.</summary>
    public void Dispose() => owner?.Dispose();
}

/// <summary>A checkpoint an image model family is checked on (<see cref="ImageModelFamilySuite"/>).</summary>
/// <param name="Path">The checkpoint (a model folder or an .onnx file).</param>
/// <param name="Images">Images to predict on (random ones are added in the random cases); null for an invalid checkpoint.</param>
/// <param name="Expected">Each image's outputs from a reference (PyTorch's), in the order of <paramref name="Images"/>; null to check consistency only.</param>
/// <param name="Tolerance">How far the outputs may be from <paramref name="Expected"/>, and an image alone from the same image in a batch.</param>
/// <param name="Error">
/// When set, the checkpoint is not readable: loading it must fail with a <see cref="NotSupportedException"/> or an
/// <see cref="InvalidDataException"/> whose message contains this text (an unregistered architecture names its registry).
/// </param>
public sealed record ImageFamilySample(string Path, IReadOnlyList<ImageData>? Images = null, IReadOnlyList<float[]>? Expected = null, float Tolerance = 1e-4f,
    string? Error = null);

/// <summary>
/// The checks of an image model family (core's <c>IImageModelFamily</c>, registered in <c>ImageModelFamilies</c>) on
/// checkpoints of its own. The implementation is the loading, as a delegate: a checkpoint's path in, the loaded model out
/// (<see cref="ImageModelUnderTest"/>; <c>path =&gt; Wrap(ImageModels.Load(path))</c>). For each sample:
/// <list type="bullet">
/// <item>it loads; its task is one of the four, its labels (when given) are not empty, and a classifier gives one output
/// per label;</item>
/// <item>every image gets outputs, all finite, the same on a second prediction and from two predictions at once;</item>
/// <item>each image's outputs are the same alone as in the batch (within the tolerance): images do not leak into each
/// other, whatever their sizes;</item>
/// <item>loading again gives the same outputs (nothing random or left over in loading);</item>
/// <item>with <see cref="ImageFamilySample.Expected"/>, the reference's outputs within the tolerance;</item>
/// <item>an invalid checkpoint (<see cref="ImageFamilySample.Error"/>) is refused with an error saying why.</item>
/// </list>
/// Random cases predict on random images of many sizes and channel counts, mixed with the sample's own.
/// </summary>
/// <param name="samples">The checkpoints, of the family checked.</param>
public sealed class ImageModelFamilySuite(IReadOnlyList<ImageFamilySample> samples) : ContractSuite<Func<string, ImageModelUnderTest>>
{
    private static readonly HashSet<string> Tasks = new(StringComparer.Ordinal) { "classification", "detection", "segmentation", "features" };

    private readonly IReadOnlyList<ImageFamilySample> _samples = samples is { Count: > 0 } ? samples
        : throw new ArgumentException("An image model family is checked on checkpoints of its own: give at least one sample.", nameof(samples));

    /// <inheritdoc />
    public override string Name => "image-model-family";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases() =>
        _samples.Select((s, i) => Case(i, s.Error is null ? [.. Enumerable.Range(0, s.Images?.Count ?? 0)] : [], [], expected: true));

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        var valid = Enumerable.Range(0, _samples.Count).Where(i => _samples[i].Error is null).ToList();
        if (valid.Count == 0)
        {
            return Case(random.Next(_samples.Count), [], [], expected: false);
        }

        int index = valid[random.Next(valid.Count)];
        int own = _samples[index].Images?.Count ?? 0;
        var picked = Enumerable.Range(0, random.Next(0, 3)).Where(_ => own > 0).Select(_ => random.Next(own)).ToArray();
        int max = large ? 160 : 48;
        var generated = Enumerable.Range(0, random.Next(picked.Length == 0 ? 1 : 0, 4))
            .Select(_ => new[] { random.Next(2) == 0 ? 1 : 3, random.Next(2, max), random.Next(2, max), random.Next() }).ToArray();
        return Case(index, picked, generated, expected: false);
    }

    /// <inheritdoc />
    public override void Run(Func<string, ImageModelUnderTest> implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var sample = _samples[(int)@case.Data["sample"]!];
        string name = System.IO.Path.GetFileName(sample.Path.TrimEnd('/', '\\'));
        if (sample.Error is { } error)
        {
            try
            {
                implementation(sample.Path).Dispose();
                checks.Fail("refuses an invalid checkpoint", $"loaded {name} without an error (expected one about \"{error}\")");
            }
            catch (Exception e) when (e is NotSupportedException or InvalidDataException)
            {
                checks.Check("an invalid checkpoint's error says why", e.Message.Contains(error, StringComparison.Ordinal), $"\"{e.Message}\" does not say \"{error}\"");
            }

            return;
        }

        var images = Images(sample, @case.Data);
        using var model = implementation(sample.Path);
        checks.Check("the task is classification, detection, segmentation or features", Tasks.Contains(model.Task), $"task \"{model.Task}\"");
        checks.Check("labels, when given, are not empty", model.Labels is not { Count: 0 }, "an empty list of labels");
        if (images.Count == 0)
        {
            checks.Skip("outputs", "no images in this case");
            return;
        }

        var first = model.Predict(images);
        checks.Check("one output per image", first.Count == images.Count, $"{first.Count} outputs for {images.Count} images");
        if (first.Count != images.Count)
        {
            return;
        }

        checks.Check("outputs are finite", first.All(o => o.All(float.IsFinite)), "a NaN or infinite output");
        if (model.Task == "classification" && model.Labels is { } labels)
        {
            checks.Check("a classifier gives one output per label", first.All(o => o.Length == labels.Count),
                $"{string.Join(", ", first.Select(o => o.Length).Distinct())} outputs for {labels.Count} labels");
        }

        checks.Compare("the same outputs on a second prediction", Different(first, model.Predict(images), 0f));
        var parallel = new IReadOnlyList<float[]>[2];
        Parallel.For(0, 2, i => parallel[i] = model.Predict(images));
        checks.Compare("the same outputs from two predictions at once", Different(first, parallel[0], sample.Tolerance) ?? Different(first, parallel[1], sample.Tolerance));
        if (images.Count > 1)
        {
            var alone = images.Select(image => model.Predict([image])[0]).ToList();
            checks.Compare("each image's outputs alone as in the batch", Different(first, alone, sample.Tolerance));
        }
        else
        {
            checks.Skip("each image's outputs alone as in the batch", "one image");
        }

        using (var again = implementation(sample.Path))
        {
            checks.Compare("the same outputs after loading again", Different(first, again.Predict(images), sample.Tolerance));
        }

        if ((bool)@case.Data["expected"]! && sample.Expected is { } expected)
        {
            checks.Compare("the reference's outputs", Different(expected, first, sample.Tolerance));
        }
        else
        {
            checks.Skip("the reference's outputs", sample.Expected is null ? "the sample gives none" : "a random case");
        }
    }

    // The case's images: the sample's own picked by index, then random ones of the sizes drawn.
    private static List<ImageData> Images(ImageFamilySample sample, JsonObject data)
    {
        var images = data["picked"]!.AsArray().Select(i => sample.Images![(int)i!]).ToList();
        foreach (var spec in data["generated"]!.AsArray())
        {
            var s = spec!.AsArray();
            int channels = (int)s[0]!, height = (int)s[1]!, width = (int)s[2]!;
            var random = new Random((int)s[3]!);
            var pixels = new float[channels * height * width];
            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = random.Next(256) / 255f;                                   // whole bytes, as decoded images have
            }

            images.Add(new ImageData(pixels, channels, height, width));
        }

        return images;
    }

    // Where two lists of outputs differ beyond the tolerance, or null.
    private static string? Different(IReadOnlyList<float[]> expected, IReadOnlyList<float[]> actual, float tolerance)
    {
        if (expected.Count != actual.Count)
        {
            return $"{actual.Count} outputs, expected {expected.Count}";
        }

        for (int i = 0; i < expected.Count; i++)
        {
            if (expected[i].Length != actual[i].Length)
            {
                return $"image {i + 1}: {actual[i].Length} values, expected {expected[i].Length}";
            }

            if (Comparisons.Difference(expected[i], actual[i], tolerance) is { } difference)
            {
                return $"image {i + 1}: {difference}";
            }
        }

        return null;
    }

    private ContractCase Case(int sample, int[] picked, int[][] generated, bool expected)
    {
        var s = _samples[sample];
        string file = System.IO.Path.GetFileName(s.Path.TrimEnd('/', '\\'));
        string what = s.Error is not null ? "refused"
            : $"{picked.Length} of its images{(generated.Length > 0 ? $" and {generated.Length} random ({string.Join(", ", generated.Select(g => $"{g[0]}x{g[1]}x{g[2]}"))})" : "")}";
        return new ContractCase($"{file}, {what}", new JsonObject
        {
            ["sample"] = sample,
            ["picked"] = new JsonArray([.. picked.Select(i => (JsonNode)i)]),
            ["generated"] = new JsonArray([.. generated.Select(g => (JsonNode)new JsonArray([.. g.Select(v => (JsonNode)v)]))]),
            ["expected"] = expected,
        });
    }
}
