// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Testing;

/// <summary>What a detection or segmentation loss takes (<see cref="VisionLossSuite"/>).</summary>
public enum VisionLossInput
{
    /// <summary>Predicted and target boxes as corners, [N, 4] (box overlap and coordinate losses).</summary>
    Boxes,

    /// <summary>Logits and 0/1 targets of one shape, [N, classes] (the focal loss).</summary>
    Logits,

    /// <summary>Class probabilities and one-hot targets, [N, classes, H, W] (dice).</summary>
    Probabilities,
}

/// <summary>
/// The checks of a detection or segmentation loss (Idrak.Vision's <c>VisionLoss</c>, given as a delegate: predictions and
/// targets in, its losses before any reduction out; the contract lives with the package that uses it and this kit depends
/// on Idrak.Abstraction alone):
/// <list type="bullet">
/// <item>the losses are finite and not negative, and the same every time;</item>
/// <item>a perfect prediction (the target itself; for logits, large ones of the right sign) loses (almost) nothing;</item>
/// <item>the first sample alone loses what it loses in the batch;</item>
/// <item>the gradient of the summed losses with respect to the predictions agrees with central differences;</item>
/// <item>given a reference (the library's loss of that name), the same losses within <see cref="Tolerance"/>.</item>
/// </list>
/// </summary>
/// <param name="input">What the loss takes.</param>
/// <param name="reference">The library's loss to agree with, or null.</param>
public sealed class VisionLossSuite(VisionLossInput input, Func<Tensor, Tensor, Tensor>? reference = null) : ContractSuite<Func<Tensor, Tensor, Tensor>>
{
    /// <inheritdoc />
    public override string Name => "vision-loss";

    /// <summary>What the loss takes.</summary>
    public VisionLossInput Input { get; } = input;

    /// <summary>The agreement wanted with the reference, relative to each loss (at least 1).</summary>
    public float Tolerance { get; init; } = 1e-5f;

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        yield return Case("one sample", 1, 1, 2);
        yield return Case("a few samples", 2, 4, 3);
        yield return Case("a batch", 3, 9, 5);
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        return Case("random", random.Next(), random.Next(1, large ? 64 : 9), random.Next(2, large ? 12 : 6));
    }

    /// <inheritdoc />
    public override void Run(Func<Tensor, Tensor, Tensor> implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        int seed = (int)@case.Data["seed"]!, n = (int)@case.Data["samples"]!, classes = (int)@case.Data["classes"]!;
        var (predicted, target, shape) = Inputs(new Random(seed), n, classes);
        using var scope = new TensorScope();
        float[] Losses(float[] p, float[] t, int[] s)
        {
            using (Autograd.NoGrad())
            {
                return implementation(Tensor.From(p, s), Tensor.From(t, s)).ToArray();
            }
        }

        var losses = Losses(predicted, target, shape);
        checks.Check("losses finite and not negative", losses.Length > 0 && losses.All(v => float.IsFinite(v) && v >= -1e-6f), $"losses {string.Join(", ", losses.Take(8))}");
        checks.Compare("the same losses every time", Comparisons.Difference(losses, Losses(predicted, target, shape), 0f));

        var perfect = Input == VisionLossInput.Logits ? [.. target.Select(t => t > 0.5f ? 30f : -30f)] : (float[])target.Clone();
        var none = Losses(perfect, target, shape);
        checks.Check("a perfect prediction loses nothing", none.All(v => Math.Abs(v) <= 1e-4f), $"losses {string.Join(", ", none.Take(8))} for a perfect prediction");

        int per = predicted.Length / n;
        int[] one = [1, .. shape[1..]];
        var alone = Losses(predicted[..per], target[..per], one);
        checks.Compare("a sample alone loses what it loses in the batch", Comparisons.Difference(losses[..alone.Length], alone, 1e-5f));

        if (reference is not null)
        {
            float[] expected;
            using (Autograd.NoGrad())
            {
                expected = reference(Tensor.From(predicted, shape), Tensor.From(target, shape)).ToArray();
            }

            checks.Compare("agrees with the reference", Comparisons.Difference(expected, losses, Tolerance));
        }
        else
        {
            checks.Skip("agrees with the reference", "no reference given");
        }

        // The gradient of the summed losses against central differences, at a few coordinates.
        var x = Tensor.From(predicted, shape, requiresGrad: true);
        implementation(x, Tensor.From(target, shape)).Sum().Backward();
        var gradient = x.Grad?.ToArray() ?? new float[predicted.Length];
        var random = new Random(seed + 1);
        string? wrong = null;
        for (int probe = 0; probe < Math.Min(12, predicted.Length) && wrong is null; probe++)
        {
            int i = random.Next(predicted.Length);
            const float h = 1e-3f;
            var plus = (float[])predicted.Clone();
            var minus = (float[])predicted.Clone();
            plus[i] += h;
            minus[i] -= h;
            double numeric = (Losses(plus, target, shape).Sum(v => (double)v) - Losses(minus, target, shape).Sum(v => (double)v)) / (2 * h);
            if (Math.Abs(numeric - gradient[i]) > 2e-2 * Math.Max(1, Math.Abs(numeric)))
            {
                wrong = $"element {i}: gradient {gradient[i]}, central differences {numeric:G6}";
            }
        }

        checks.Compare("the gradient agrees with central differences", wrong);
    }

    // Predictions and targets as the input kind wants them.
    private (float[] Predicted, float[] Target, int[] Shape) Inputs(Random random, int n, int classes)
    {
        switch (Input)
        {
            case VisionLossInput.Boxes:
            {
                float[] Boxes()
                {
                    var b = new float[n * 4];
                    for (int i = 0; i < n; i++)
                    {
                        float x = random.NextSingle() * 20, y = random.NextSingle() * 20;
                        (b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]) = (x, y, x + 1 + random.NextSingle() * 15, y + 1 + random.NextSingle() * 15);
                    }

                    return b;
                }

                var target = Boxes();
                var predicted = target.Select(v => v + (random.NextSingle() - 0.5f) * 6).ToArray();
                for (int i = 0; i < n; i++)                                    // keep each predicted box the right way round
                {
                    (predicted[4 * i + 2], predicted[4 * i + 3]) = (Math.Max(predicted[4 * i + 2], predicted[4 * i] + 0.5f), Math.Max(predicted[4 * i + 3], predicted[4 * i + 1] + 0.5f));
                }

                return (predicted, target, [n, 4]);
            }

            case VisionLossInput.Logits:
                return ([.. Enumerable.Range(0, n * classes).Select(_ => (random.NextSingle() - 0.5f) * 8)],
                    [.. Enumerable.Range(0, n * classes).Select(_ => random.Next(4) == 0 ? 1f : 0f)], [n, classes]);
            default:
            {
                int h = random.Next(2, 6), w = random.Next(2, 6), plane = h * w;
                var probabilities = new float[n * classes * plane];
                var target = new float[n * classes * plane];
                for (int s = 0; s < n; s++)
                {
                    for (int k = 0; k < plane; k++)
                    {
                        double total = 0;
                        var e = new double[classes];
                        for (int c = 0; c < classes; c++)
                        {
                            total += e[c] = Math.Exp((random.NextDouble() - 0.5) * 4);
                        }

                        int truth = random.Next(classes);
                        for (int c = 0; c < classes; c++)
                        {
                            probabilities[(s * classes + c) * plane + k] = (float)(e[c] / total);
                            target[(s * classes + c) * plane + k] = c == truth ? 1f : 0f;
                        }
                    }
                }

                return (probabilities, target, [n, classes, h, w]);
            }
        }
    }

    private static ContractCase Case(string name, int seed, int samples, int classes) =>
        new($"{name} ({samples} samples, {classes} classes)", new JsonObject { ["seed"] = seed, ["samples"] = samples, ["classes"] = classes });
}
