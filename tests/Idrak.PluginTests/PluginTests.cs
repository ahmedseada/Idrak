// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using Idrak.Generation;
using Idrak.Layers.Abstractions;
using Idrak.Layers;
using Idrak.Onnx.Abstractions;
using Idrak.Onnx;
using Idrak;

namespace Idrak.PluginTests;

/// <summary>
/// Plug-ins written with Idrak's public API alone, as an outside package writes them: an optimizer (<see cref="Lion"/>),
/// an operation with its own backward step, a network-builder step and an ONNX import operator built on it, a packed
/// weight format (<see cref="RoundedWeight"/>) and a key/value cache format (<see cref="ScaledLayout"/>). The test runner
/// in tests/Idrak.Tests runs <see cref="All"/> on every device; a failed check throws.
/// </summary>
public static class PluginTests
{
    /// <summary>The tests, by name.</summary>
    public static IReadOnlyList<(string Name, Action<Device> Run)> All { get; } =
    [
        ("outside plug-in: Idrak grants this assembly no internal access", NoInternalAccess),
        ("outside plug-in: a Lion optimizer makes the reference update and trains a linear model", LionTrains),
        ("outside plug-in: softplus through Autograd.Function matches finite differences, registers as a network step, trains with Lion and imports from ONNX", SoftplusStep),
        ("outside plug-in: a packed weight format multiplies like its expanded weights, with the input's gradient", PackedFormat),
        ("outside plug-in: a key/value cache format generates the float32 cache's greedy text", CacheFormat),
        ("outside plug-in: a sample source computed when read trains a classifier, to the weights of its in-memory copy", DataPluginTests.SourceTrains),
        ("outside plug-in: a batch source making tensors itself trains a linear model through Trainer.Fit", DataPluginTests.BatchSourceTrains),
    ];

    /// <summary>The ONNX operator and network step softplus is exported, imported and replayed as.</summary>
    public const string SoftplusOperator = "OutsideSoftplus", SoftplusStepName = "outside-softplus";

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
        // One step against the formula, on the host.
        float[] p0 = [0.5f, -1f, 2f, 0.25f], g0 = [0.1f, -0.3f, 0f, 2f];
        using (var p = Tensor.Persistent(p0, [4], device, requiresGrad: true))
        using (var lion = new Lion([p], learningRate: 0.1f, weightDecay: 0.5f))
        {
            using (var scope = new TensorScope())
            {
                using var target = Tensor.From(g0, [4], device);
                (p * target).Sum().Backward();                                          // the gradient is g0
            }

            lion.Step();
            lion.ZeroGrad();
            var expected = p0.Select((v, i) => v * (1f - 0.1f * 0.5f) - 0.1f * MathF.Sign((1f - 0.9f) * g0[i])).ToArray();
            CheckClose(expected, p.ToArray(), 1e-6f, "the first Lion step");
        }

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

        // A network step of its own, trained with Lion.
        NetworkOps.Register(SoftplusStepName, (b, a) => b.Lambda(t => Softplus.Apply(t), "Softplus", [.. b.CurrentShape]));
        OnnxImportOps.Register(SoftplusOperator, c => c.Add(SoftplusStepName));
        try
        {
            var random = new Random(5);
            const int Samples = 48;
            var xs = Enumerable.Range(0, Samples * 2).Select(_ => random.NextSingle() * 4f - 2f).ToArray();
            var ys = Enumerable.Range(0, Samples).Select(i => MathF.Abs(xs[2 * i]) - 0.5f * xs[2 * i + 1]).ToArray();
            using var x = Tensor.From(xs, [Samples, 2], device);
            using var y = Tensor.From(ys, [Samples, 1], device);
            var builder = Network.Input(2).Seed(6).Linear(16).Op(SoftplusStepName).Linear(1);
            using var model = builder.OnDevice(device).Build();
            using var optimizer = new Lion(model.Parameters(), learningRate: 0.01f);
            var (first, last) = Train(model, optimizer, x, y, 200);
            Check(last < 0.2f * first, $"a network with the softplus step: loss {first} → {last}");

            // Exported with the step as a custom operator, and imported back onto the step.
            model.Eval();
            var bytes = OnnxExport.For(model).Input(2).Lambda("Softplus", (g, _, input, shape) => g.Node(SoftplusOperator, [input], shape)).ToBytes();
            using var imported = OnnxImport.Load(bytes, device);
            using var expected = model.Predict(x);
            using var actual = imported.Model.Predict(x);
            CheckClose(expected.ToArray(), actual.ToArray(), 1e-5f, "the imported network");
            Check(JsonNode.DeepEquals(Network.FromJson(builder.ToJson()).ToJson(), builder.ToJson()), "the step replays from JSON");
        }
        finally
        {
            NetworkOps.Unregister(SoftplusStepName);
            OnnxImportOps.Unregister(SoftplusOperator);
        }
    }

    private static void PackedFormat(Device device)
    {
        PackedWeight.Register(RoundedWeight.FormatName, RoundedWeight.Pack);
        Check(PackedWeight.FormatNames.Contains(RoundedWeight.FormatName), "registered");
        var random = new Random(7);
        const int Rows = 24, Columns = 10, Batch = 5;
        var values = Enumerable.Range(0, Rows * Columns).Select(_ => random.NextSingle() * 0.2f - 0.1f).ToArray();
        var inputs = Enumerable.Range(0, Batch * Rows).Select(_ => random.NextSingle() * 2f - 1f).ToArray();
        var bias = Enumerable.Range(0, Columns).Select(i => 0.01f * i).ToArray();
        using var linear = Linear.FromPacked(PackedWeight.FromValues(RoundedWeight.FormatName, values, Rows, Columns, device), Tensor.Persistent(bias, [Columns], device));
        var weight = (RoundedWeight)linear.PackedWeight!;
        using var scope = new TensorScope();
        var expanded = weight.Dequantize();
        var x = Tensor.From(inputs, [Batch, Rows], device, requiresGrad: true);
        var reference = Tensor.From(inputs, [Batch, Rows], device, requiresGrad: true);
        var y = linear.Forward(x);
        var expected = reference.MatMul(expanded) + Tensor.From(bias, [Columns], device);
        CheckClose(expected.ToArray(), y.ToArray(), 1e-4f, "the product");
        y.Sum().Backward();
        expected.Sum().Backward();
        CheckClose(reference.Grad!.ToArray(), x.Grad!.ToArray(), 1e-4f, "the input's gradient");
        Check(weight.Products > 0 && linear.ToString()!.Contains(RoundedWeight.FormatName, StringComparison.Ordinal), $"through the format: {linear}");
    }

    private static void CacheFormat(Device device)
    {
        var layout = new ScaledLayout();
        KeyValueLayouts.Register(ScaledLayout.FormatName, layout);
        Check(KeyValueLayouts.Get(ScaledLayout.FormatName) == layout, "registered");
        var tokenizer = new CharTokenizer("abcdefghijklmnopqrstuvwxyz .,");
        var random = new Random(4);
        using var model = new Sequential
        {
            new Embedding(tokenizer.VocabularySize, 16, device, random),
            new PositionalEncoding(32, 16, device),
            new TransformerEncoderLayer(16, 2, dropout: 0f, causal: true, device: device, random: random),
            new LayerNorm(16, device: device),
            new Linear(16, tokenizer.VocabularySize, device: device, random: random),
        };
        var options = new GenerationOptions { Temperature = 0f, TopK = 1, RepeatPenalty = 1f, NumPredict = 10 };
        string expected = new TextGenerator(model, tokenizer, 32) { KeepCache = false }.Generate("abc", options).Text;
        string actual = new TextGenerator(model, tokenizer, 32) { KeepCache = false, CacheLayout = layout }.Generate("abc", options).Text;
        Check(actual == expected && layout.Writes > 0, $"with the format '{actual}', float32 cache '{expected}'");
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
