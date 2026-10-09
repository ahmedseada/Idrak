// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Data;

namespace Idrak.Abstraction.Testing;

/// <summary>A detector's answer the metric checks take: a box, its class and its score.</summary>
/// <param name="Box">The box.</param>
/// <param name="Class">The class.</param>
/// <param name="Score">The detector's confidence.</param>
public readonly record struct ScoredBox(BoundingBox Box, int Class, float Score);

/// <summary>One image of a detection metric check: the detector's answers and the true objects.</summary>
/// <param name="Predictions">The detections.</param>
/// <param name="Truths">The true objects (crowd and difficult ones count neither way).</param>
public sealed record DetectionImage(IReadOnlyList<ScoredBox> Predictions, IReadOnlyList<ObjectAnnotation> Truths);

/// <summary>
/// The checks of a detection metric (Idrak.Vision's <c>IDetectionMetric</c>, given as a delegate: images in, its headline
/// value out, such as mean average precision; the contract lives with the package that uses it and this kit depends on
/// Idrak.Abstraction alone) on random scenes:
/// <list type="bullet">
/// <item>a value in [0, 1], the same every time; perfect detections score 1, none score 0, and a false detection ranked above
/// the right ones lowers the score;</item>
/// <item>only the ranking of the scores counts (scores changed by an increasing map give the same value), not the order of
/// the images;</item>
/// <item>a false detection ranked below every other leaves it as it is, and so do detections of a class with no truths and
/// a top-scored detection of a difficult object;</item>
/// <item>given a reference (the library's metric of that name), the same value within 1e-9.</item>
/// </list>
/// </summary>
/// <param name="reference">The library's metric to agree with, or null.</param>
public sealed class DetectionMetricSuite(Func<IReadOnlyList<DetectionImage>, double>? reference = null) : ContractSuite<Func<IReadOnlyList<DetectionImage>, double>>
{
    /// <inheritdoc />
    public override string Name => "detection-metric";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        yield return Case("one image, one class", 1, 1, 1, 3);
        yield return Case("several images and classes", 2, 5, 3, 6);
        yield return Case("crowded", 3, 3, 2, 15);
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        return Case("random", random.Next(), random.Next(1, large ? 50 : 8), random.Next(1, 5), random.Next(1, large ? 30 : 10));
    }

    /// <inheritdoc />
    public override void Run(Func<IReadOnlyList<DetectionImage>, double> implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var d = @case.Data;
        var random = new Random((int)d["seed"]!);
        var images = Scene(random, (int)d["images"]!, (int)d["classes"]!, (int)d["objects"]!);
        double value = implementation(images);
        checks.Check("a value in [0, 1]", value is >= -1e-12 and <= 1 + 1e-12, $"{value}");
        checks.Check("the same value every time", implementation(images).Equals(value), "two calls differ");

        var perfect = images.Select(i => i with { Predictions = [.. i.Truths.Where(t => !t.Difficult && !t.Crowd).Select(t => new ScoredBox(t.Box, t.Class, 0.9f))] }).ToList();
        checks.Check("perfect detections score 1", Math.Abs(implementation(perfect) - 1) <= 1e-9, $"{implementation(perfect)}");
        var classes = images.SelectMany(i => i.Truths.Where(t => !t.Difficult && !t.Crowd).Select(t => t.Class)).Distinct().ToList();
        var falseFirst = perfect.Select((i, k) => k == 0 ? i with { Predictions = [.. i.Predictions, .. classes.Select(c => new ScoredBox(new BoundingBox(900, 900, 10, 10), c, 1f))] } : i).ToList();
        checks.Check("a false detection ranked above the right ones lowers it", classes.Count == 0 || implementation(falseFirst) < 1 - 1e-9,
            $"{implementation(falseFirst)} with a false detection first in every class");
        var nothing = images.Select(i => i with { Predictions = [] }).ToList();
        checks.Check("no detections score 0", implementation(nothing) == 0, $"{implementation(nothing)}");

        var rescored = images.Select(i => i with { Predictions = [.. i.Predictions.Select(p => p with { Score = p.Score * p.Score * 0.5f + 0.25f })] }).ToList();
        checks.Check("only the ranking of the scores counts", Math.Abs(implementation(rescored) - value) <= 1e-9, $"{implementation(rescored)} after rescoring, {value} before");
        var reversed = images.AsEnumerable().Reverse().ToList();
        checks.Check("the order of the images does not count", Math.Abs(implementation(reversed) - value) <= 1e-9, $"{implementation(reversed)} reversed, {value} before");

        var last = images.Select((i, k) => k == 0 ? i with { Predictions = [.. i.Predictions, new ScoredBox(new BoundingBox(500, 500, 10, 10), i.Truths.FirstOrDefault()?.Class ?? 0, 1e-6f)] } : i).ToList();
        checks.Check("a false detection ranked last changes nothing", Math.Abs(implementation(last) - value) <= 1e-9, $"{implementation(last)} with it, {value} without");
        var stranger = images.Select((i, k) => k == 0 ? i with { Predictions = [.. i.Predictions, new ScoredBox(new BoundingBox(1, 1, 10, 10), 99, 0.99f)] } : i).ToList();
        checks.Check("a detection of a class with no truths changes nothing", Math.Abs(implementation(stranger) - value) <= 1e-9, $"{implementation(stranger)} with it, {value} without");
        var hard = images.Select((i, k) => k == 0 ? new DetectionImage([.. i.Predictions, new ScoredBox(new BoundingBox(700, 700, 20, 20), 0, 1f)],
            [.. i.Truths, new ObjectAnnotation(new BoundingBox(700, 700, 20, 20), 0) { Difficult = true }]) : i).ToList();
        checks.Check("a detected difficult object changes nothing", Math.Abs(implementation(hard) - value) <= 1e-9, $"{implementation(hard)} with it, {value} without");

        if (reference is null)
        {
            checks.Skip("agrees with the reference", "no reference given");
        }
        else
        {
            double expected = reference(images);
            checks.Check("agrees with the reference", Math.Abs(expected - value) <= 1e-9 || double.IsNaN(expected) && double.IsNaN(value), $"{value}, reference {expected}");
        }
    }

    // Images of truths (some difficult, none crowds) and detections: jittered copies of truths, misplaced classes and stray boxes.
    private static List<DetectionImage> Scene(Random random, int images, int classes, int objects)
    {
        var list = new List<DetectionImage>();
        for (int i = 0; i < images; i++)
        {
            var truths = Enumerable.Range(0, random.Next(1, objects + 1)).Select(_ => new ObjectAnnotation(
                new BoundingBox(random.NextSingle() * 200, random.NextSingle() * 200, 8 + random.NextSingle() * 120, 8 + random.NextSingle() * 120), random.Next(classes))
            {
                Difficult = random.Next(10) == 0,
            }).ToList();
            var predictions = new List<ScoredBox>();
            foreach (var t in truths)
            {
                for (int n = random.Next(0, 3); n > 0; n--)
                {
                    float jitter = random.NextSingle() * 0.4f;
                    var box = new BoundingBox(t.Box.X + (random.NextSingle() - 0.5f) * jitter * t.Box.Width, t.Box.Y + (random.NextSingle() - 0.5f) * jitter * t.Box.Height,
                        t.Box.Width * (1 + (random.NextSingle() - 0.5f) * jitter), t.Box.Height * (1 + (random.NextSingle() - 0.5f) * jitter));
                    predictions.Add(new ScoredBox(box, random.Next(6) == 0 ? random.Next(classes) : t.Class, random.NextSingle()));
                }
            }

            for (int n = random.Next(0, 3); n > 0; n--)
            {
                predictions.Add(new ScoredBox(new BoundingBox(random.NextSingle() * 300, random.NextSingle() * 300, 5 + random.NextSingle() * 60, 5 + random.NextSingle() * 60),
                    random.Next(classes), random.NextSingle()));
            }

            list.Add(new DetectionImage(predictions, truths));
        }

        return list;
    }

    private static ContractCase Case(string name, int seed, int images, int classes, int objects) =>
        new($"{name} ({images} images, {classes} classes, up to {objects} objects)", new JsonObject { ["seed"] = seed, ["images"] = images, ["classes"] = classes, ["objects"] = objects });
}

/// <summary>One image of a segmentation metric check: the predicted and true class of every pixel, row by row.</summary>
/// <param name="Predicted">The predicted classes.</param>
/// <param name="Expected">The true classes.</param>
/// <param name="Width">The width.</param>
/// <param name="Height">The height.</param>
/// <param name="Classes">The number of classes.</param>
public sealed record SegmentationImage(int[] Predicted, int[] Expected, int Width, int Height, int Classes);

/// <summary>
/// The checks of a segmentation metric (Idrak.Vision's <c>ISegmentationMetric</c>, given as a delegate: images in, its
/// headline value out, such as the mean IoU): a value in [0, 1], the same every time; a perfect prediction scores 1, every
/// pixel wrong scores 0; swapping prediction and truth or the order of the images changes nothing; given a reference, the
/// same value within 1e-9.
/// </summary>
/// <param name="reference">The library's metric to agree with, or null.</param>
public sealed class SegmentationMetricSuite(Func<IReadOnlyList<SegmentationImage>, double>? reference = null) : ContractSuite<Func<IReadOnlyList<SegmentationImage>, double>>
{
    /// <inheritdoc />
    public override string Name => "segmentation-metric";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        yield return Case("two classes", 1, 1, 2);
        yield return Case("several images and classes", 2, 4, 5);
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        return Case("random", random.Next(), random.Next(1, large ? 20 : 5), random.Next(2, large ? 30 : 8));
    }

    /// <inheritdoc />
    public override void Run(Func<IReadOnlyList<SegmentationImage>, double> implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var d = @case.Data;
        var random = new Random((int)d["seed"]!);
        int classes = (int)d["classes"]!;
        var images = Enumerable.Range(0, (int)d["images"]!).Select(_ =>
        {
            int w = random.Next(4, 40), h = random.Next(4, 40);
            int[] expected = [.. Enumerable.Range(0, w * h).Select(_ => random.Next(classes))];
            int[] predicted = [.. expected.Select(c => random.Next(3) == 0 ? random.Next(classes) : c)];
            return new SegmentationImage(predicted, expected, w, h, classes);
        }).ToList();
        double value = implementation(images);
        checks.Check("a value in [0, 1]", value is >= -1e-12 and <= 1 + 1e-12, $"{value}");
        checks.Check("the same value every time", implementation(images).Equals(value), "two calls differ");
        var perfect = images.Select(i => i with { Predicted = i.Expected }).ToList();
        checks.Check("a perfect prediction scores 1", Math.Abs(implementation(perfect) - 1) <= 1e-12, $"{implementation(perfect)}");
        var wrong = images.Select(i => i with { Predicted = [.. i.Expected.Select(c => (c + 1) % classes)] }).ToList();
        checks.Check("every pixel wrong scores 0", implementation(wrong) == 0, $"{implementation(wrong)}");
        var swapped = images.Select(i => i with { Predicted = i.Expected, Expected = i.Predicted }).ToList();
        checks.Check("prediction and truth swapped, the same value", Math.Abs(implementation(swapped) - value) <= 1e-12, $"{implementation(swapped)} swapped, {value} before");
        var reversed = images.AsEnumerable().Reverse().ToList();
        checks.Check("the order of the images does not count", Math.Abs(implementation(reversed) - value) <= 1e-12, $"{implementation(reversed)} reversed, {value} before");
        if (reference is null)
        {
            checks.Skip("agrees with the reference", "no reference given");
        }
        else
        {
            double expected = reference(images);
            checks.Check("agrees with the reference", Math.Abs(expected - value) <= 1e-9, $"{value}, reference {expected}");
        }
    }

    private static ContractCase Case(string name, int seed, int images, int classes) =>
        new($"{name} ({images} images, {classes} classes)", new JsonObject { ["seed"] = seed, ["images"] = images, ["classes"] = classes });
}
