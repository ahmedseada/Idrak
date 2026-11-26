// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Reflection;
using System.Runtime.CompilerServices;
using Idrak;
using Idrak.Generation;
using Idrak.Layers;

namespace Idrak.PluginTests;

/// <summary>
/// Plug-ins written with Idrak's public API alone, as an outside package writes them: an optimizer (<see cref="Lion"/>)
/// and a function with its own backward step. The test runner in tests/Idrak.Tests runs <see cref="All"/> on every
/// device; a failed check throws.
/// </summary>
public static class PluginTests
{
    /// <summary>The tests, by name.</summary>
    public static IReadOnlyList<(string Name, Action<Device> Run)> All { get; } =
    [
        ("outside plug-in: Idrak grants this assembly no internal access", NoInternalAccess),
        ("outside plug-in: a Lion optimizer makes the reference update and trains a linear model", LionTrains),
        ("outside plug-in: softplus through Autograd.Function matches finite differences", SoftplusStep),
        ("outside plug-in: a sample source computed when read trains a classifier, to the weights of its in-memory copy", DataPluginTests.SourceTrains),
        ("outside plug-in: a batch source making tensors itself trains a linear model through Trainer.Fit", DataPluginTests.BatchSourceTrains),
        ("outside plug-in: a message part of a kind of its own (audio) is refused by a text-only model and answered by one that takes the kind", ChatPartPluginTests.AudioKind),
    ];

    /// <summary>softplus(x) = log(1 + e^x), with its derivative sigmoid(x) as the backward step.</summary>
    public static DifferentiableFunction Softplus { get; } =
        Autograd.Function("softplus", x => (x[0].Exp() + 1f).Log(), (x, y, g) => [g * x[0].Sigmoid()]);

    private static void NoInternalAccess(Device device)
    {
        // Idrak names its friends (the first-party packages and tests); this assembly is not one of them.
        string self = typeof(PluginTests).Assembly.GetName().Name!;
        Check(typeof(Sequential).Assembly.GetCustomAttributes<InternalsVisibleToAttribute>().Any(), "Idrak lists its friend assemblies");
        foreach (var assembly in new[] { typeof(Device).Assembly, typeof(Sequential).Assembly })
        {
            var friends = assembly.GetCustomAttributes<InternalsVisibleToAttribute>().Select(a => a.AssemblyName).ToArray();
            Check(!friends.Contains(self), $"{assembly.GetName().Name} friends: {string.Join(", ", friends)}");
        }
    }

    private static void LionTrains(Device device)
    {
        // One step against the formula.
        float[] p0 = [0.5f, -1f, 2f, 0.25f], g0 = [0.1f, -0.3f, 0f, 2f];
        void FirstStep(Func<IReadOnlyList<Tensor>, Optimizer> make, string what)
        {
            using var p = Tensor.Persistent(p0, [4], device, requiresGrad: true);
            using var lion = make([p]);
            using (var scope = new TensorScope())
            {
                using var target = Tensor.From(g0, [4], device);
                (p * target).Sum().Backward();                                          // the gradient is g0
            }

            lion.Step();
            lion.ZeroGrad();
            var expected = p0.Select((v, i) => v * (1f - 0.1f * 0.5f) - 0.1f * MathF.Sign((1f - 0.9f) * g0[i])).ToArray();
            CheckClose(expected, p.ToArray(), 1e-6f, what);
        }

        FirstStep(ps => new Lion(ps, learningRate: 0.1f, weightDecay: 0.5f), "the first Lion step");

        // A noisy linear function, learned.
        var random = new Random(3);
        const int Samples = 64;
        float[] w = [1.5f, -2f, 0.5f];
        var xs = new float[Samples * 3];
        var ys = new float[Samples];
        for (int i = 0; i < Samples; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                xs[i * 3 + j] = random.NextSingle() * 2f - 1f;
                ys[i] += w[j] * xs[i * 3 + j];
            }

            ys[i] += 0.3f + 0.01f * (random.NextSingle() - 0.5f);
        }

        using var x = Tensor.From(xs, [Samples, 3], device);
        using var y = Tensor.From(ys, [Samples, 1], device);
        using var model = new Linear(3, 1, device: device, random: new Random(4));
        using var optimizer = new Lion(model.Parameters(), learningRate: 0.02f);
        var (first, last) = Train(model, optimizer, x, y, 300);
        Check(last < 0.02f * first && last < 0.01f, $"Lion: loss {first} → {last}");
    }

    private static void SoftplusStep(Device device)
    {
        float[] values = [-2.5f, -1.1f, -0.3f, 0.2f, 0.9f, 1.7f, 3.2f];
        using (var scope = new TensorScope())
        {
            var x = Tensor.From(values, [values.Length], device, requiresGrad: true);
            (Softplus.Apply(x * 1.5f) * x).Sum().Backward();
            var analytic = x.Grad!.ToArray();
            var numeric = new float[values.Length];
            using (Autograd.NoGrad())
            {
                for (int i = 0; i < values.Length; i++)
                {
                    float At(float delta)
                    {
                        var shifted = (float[])values.Clone();
                        shifted[i] += delta;
                        var t = Tensor.From(shifted, [values.Length], device);
                        return (Softplus.Apply(t * 1.5f) * t).Sum().Item();
                    }

                    numeric[i] = (At(1e-2f) - At(-1e-2f)) / 2e-2f;
                }
            }

            CheckClose(numeric, analytic, 2e-2f, "softplus gradient");
            CheckClose([.. values.Select(v => MathF.Log(1f + MathF.Exp(1.5f * v)))], Softplus.Apply(Tensor.From(values, [values.Length], device) * 1.5f).ToArray(), 1e-5f, "softplus");
        }
    }

    // Full-batch training; the first and the last loss.
    private static (float First, float Last) Train(Idrak.Abstraction.Module model, Idrak.Abstraction.Training.Optimizer optimizer, Tensor x, Tensor y, int steps)
    {
        float first = 0f, last = 0f;
        for (int step = 0; step < steps; step++)
        {
            using var scope = new TensorScope();
            var loss = Losses.MeanSquaredError(model.Forward(x), y);
            optimizer.ZeroGrad();
            loss.Backward();
            optimizer.Step();
            last = loss.Item();
            if (step == 0)
            {
                first = last;
            }
        }

        return (first, last);
    }

    private static void Check(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }

    private static void CheckClose(float[] expected, float[] actual, float tolerance, string what)
    {
        Check(expected.Length == actual.Length, $"{what}: {actual.Length} values, expected {expected.Length}");
        for (int i = 0; i < expected.Length; i++)
        {
            Check(MathF.Abs(expected[i] - actual[i]) <= tolerance * MathF.Max(1f, MathF.Abs(expected[i])), $"{what}: element {i} is {actual[i]}, expected {expected[i]}");
        }
    }
}
