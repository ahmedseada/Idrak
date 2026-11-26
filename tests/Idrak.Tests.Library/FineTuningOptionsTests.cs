// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Data;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Text;
using Idrak.Text.Abstractions;
using Idrak.Optimizers;

internal abstract partial class Tests
{
    internal static DecoderSpec TinyChatSpec() => new()
    {
        Vocabulary = 260, Dim = 32, Layers = 2, Heads = 2, KvHeads = 1, HeadDim = 16, FfDim = 64, MaxPositions = 256,
        Rope = new RopeSettings(10000f), NormEpsilon = 1e-6f, QkNorm = true, TieEmbeddings = true,
    };

    internal static (List<float> Losses, float[] Adapters, string Trace) TrainTiny(string folder, Device device, IReadOnlyList<TrainingSequence> sequences,
        FineTuningOptions options)
    {
        using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
        var losses = new List<float>();
        var trace = new System.Text.StringBuilder();
        FineTuner.Train(model, sequences, null, options, progress: new SynchronousProgress<FineTuningProgress>(p => losses.Add(p.Loss)),
            trace: line => trace.AppendLine(line));
        return (losses, [.. model.Network.TrainableParameters().SelectMany(p => p.ToArray())], trace.ToString());
    }

    internal static void PreferenceFormulas(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                              // the losses run on the CPU on every device
        }

        int[] counts = [3, 2, 4, 1, 2, 2];
        var random = new Random(9);
        float[] policy = [.. Enumerable.Range(0, counts.Sum()).Select(_ => -0.2f - 2f * random.NextSingle())];
        float[] reference = [.. policy.Select(v => v + 0.6f * (random.NextSingle() - 0.5f))];
        double[] Sums(float[] v)
        {
            var sums = new double[counts.Length];
            for (int s = 0, t = 0; s < counts.Length; s++)
            {
                for (int i = 0; i < counts[s]; i++, t++)
                {
                    sums[s] += v[t];
                }
            }

            return sums;
        }

        static double LogSigmoid(double x) => -Math.Log(1 + Math.Exp(-x));
        var sp = Sums(policy);
        var sr = Sums(reference);
        double[] means = [.. sp.Select((v, s) => v / counts[s])];
        int pairs = counts.Length / 2;
        double dpo = 0, dpoSmooth = 0, orpo = 0, simpo = 0;
        for (int p = 0; p < pairs; p++)
        {
            double h = (sp[2 * p] - sr[2 * p]) - (sp[2 * p + 1] - sr[2 * p + 1]);
            dpo += -LogSigmoid(0.1 * h);
            dpoSmooth += -0.9 * LogSigmoid(0.1 * h) - 0.1 * LogSigmoid(-0.1 * h);
            double Odds(double a) => a - Math.Log(1 - Math.Exp(a));
            orpo += -means[2 * p] - 0.25 * LogSigmoid(Odds(means[2 * p]) - Odds(means[2 * p + 1]));
            simpo += -LogSigmoid(2 * (means[2 * p] - means[2 * p + 1]) - 0.5);
        }

        float Run(IFineTuningLoss loss, out float[] gradient)
        {
            using var scope = new TensorScope();
            var leaf = Tensor.From(policy, [policy.Length], Device.Cpu, requiresGrad: true);
            var input = new FineTuningLossInput(leaf, counts, Tensor.From(reference, [reference.Length], Device.Cpu), pairs: true);
            var value = loss.Compute(input);
            value.Backward();
            gradient = leaf.Grad!.ToArray();
            return value.Item();
        }

        AssertClose([(float)(dpo / pairs)], [Run(FineTuningLosses.Dpo(0.1f), out var dpoGradient)], 1e-5f, "DPO against its formula");
        AssertClose([(float)(dpoSmooth / pairs)], [Run(FineTuningLosses.Dpo(0.1f, 0.1f), out _)], 1e-5f, "conservative DPO against its formula");
        AssertClose([(float)(orpo / pairs)], [Run(FineTuningLosses.Orpo(0.25f), out _)], 1e-5f, "ORPO against its formula");
        AssertClose([(float)(simpo / pairs)], [Run(FineTuningLosses.SimPo(2f, 0.5f), out _)], 1e-5f, "SimPO against its formula");

        // DPO's gradient: −β σ(−βh) / pairs on each chosen token, the opposite on each rejected one.
        var expected = new float[policy.Length];
        for (int s = 0, t = 0; s < counts.Length; s++)
        {
            int p = s / 2;
            double h = (sp[2 * p] - sr[2 * p]) - (sp[2 * p + 1] - sr[2 * p + 1]);
            double g = -0.1 / (1 + Math.Exp(0.1 * h)) / pairs * (s % 2 == 0 ? 1 : -1);
            for (int i = 0; i < counts[s]; i++, t++)
            {
                expected[t] = (float)g;
            }
        }

        AssertClose(expected, dpoGradient, 1e-6f, "DPO gradient");
        using (var scope = new TensorScope())
        {
            var x = Tensor.From([0f, -100f, 100f, 1.5f], [4], Device.Cpu, requiresGrad: true);
            var y = Losses.LogSigmoid(x);
            y.Sum().Backward();
            AssertClose([MathF.Log(0.5f), -100f, 0f, (float)LogSigmoid(1.5)], y.ToArray(), 1e-5f, "log-sigmoid values (no overflow)");
            AssertClose([0.5f, 1f, 0f, (float)(1 / (1 + Math.Exp(1.5)))], x.Grad!.ToArray(), 1e-5f, "log-sigmoid gradient (½ at 0)");
        }
        using (var scope = new TensorScope())
        {
            var plain = new FineTuningLossInput(Tensor.From(policy, [policy.Length], Device.Cpu), counts);
            Check(Refused(() => FineTuningLosses.Dpo().Compute(plain)), "DPO without pairs is refused");
        }
    }

    internal static bool Refused(Action action)
    {
        try
        {
            action();
            return false;
        }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or InvalidDataException or ArgumentException)
        {
            return true;
        }
    }

    internal static void DoraLayer(Device device)
    {
        const int In = 24, Out = 20, Rank = 3, Rows = 5;
        const float Scale = 0.7f;
        var random = new Random(31);
        float[] Values(int n, float range) => [.. Enumerable.Range(0, n).Select(_ => (random.NextSingle() * 2f - 1f) * range)];
        float[] w = Values(In * Out, 0.3f), bias = Values(Out, 0.1f), a = Values(In * Rank, 0.4f), b = Values(Rank * Out, 0.4f), x = Values(Rows * In, 1f);
        float[] m = [.. Enumerable.Range(0, Out).Select(_ => 0.5f + random.NextSingle())], c = Values(Rows * Out, 1f);

        {
            using var scope = new TensorScope();
            var layer = Linear.FromWeights(Tensor.Persistent(w, [In, Out], device, requiresGrad: false), Tensor.Persistent(bias, [Out], device, requiresGrad: false));

            float[] dense = layer.WeightValues();                        // the weights as the layer holds them
            var dora = new DoraAdapter(Tensor.Persistent(a, [In, Rank], device, requiresGrad: true), Tensor.Persistent(b, [Rank, Out], device, requiresGrad: true),
                Tensor.Persistent(m, [Out], device, requiresGrad: true), Rank, Scale);
            layer.Adapter = dora;
            var input = Tensor.From(x, [Rows, In], device);
            var y = layer.Forward(input);
            (y * Tensor.From(c, [Rows, Out], device)).Sum().Backward();

            // The reference on the host, in double: z = x·W + s·x·A·B, n = ‖W + s·A·B‖ per column, y = z·m/n + b;
            // d/dm (with the norm a constant, as peft) = Σ_rows c·z / n.
            var expected = new float[Rows * Out];
            var magnitudeGradient = new float[Out];
            var norms = new double[Out];
            for (int o = 0; o < Out; o++)
            {
                for (int i = 0; i < In; i++)
                {
                    double ab = 0;
                    for (int r = 0; r < Rank; r++)
                    {
                        ab += a[i * Rank + r] * b[r * Out + o];
                    }

                    double v = dense[i * Out + o] + Scale * ab;
                    norms[o] += v * v;
                }

                norms[o] = Math.Sqrt(norms[o]);
            }

            for (int row = 0; row < Rows; row++)
            {
                for (int o = 0; o < Out; o++)
                {
                    double z = 0;
                    for (int i = 0; i < In; i++)
                    {
                        double ab = 0;
                        for (int r = 0; r < Rank; r++)
                        {
                            ab += a[i * Rank + r] * b[r * Out + o];
                        }

                        z += x[row * In + i] * (dense[i * Out + o] + Scale * ab);
                    }

                    expected[row * Out + o] = (float)(z * m[o] / norms[o] + bias[o]);
                    magnitudeGradient[o] += (float)(c[row * Out + o] * z / norms[o]);
                }
            }

            string kind = "float base";
            CloseByNorm(expected, y.ToArray(), 2e-3f, $"DoRA output, {kind}");
            CloseByNorm(magnitudeGradient, dora.Magnitude.Grad!.ToArray(), 2e-3f, $"DoRA magnitude gradient, {kind}");
            Check(dora.A.Grad!.ToArray().Any(v => v != 0f) && dora.B.Grad!.ToArray().Any(v => v != 0f), $"A and B receive gradients, {kind}");
            float[] adapted;
            using (Autograd.NoGrad())
            {
                adapted = layer.Forward(input).ToArray();
                layer.MergeAdapter();
                Check(layer.Adapter is null, "merged: no adapter");
                CloseByNorm(adapted, layer.Forward(input).ToArray(), 1e-5f, "merged DoRA weight gives the adapted output");
            }
        }
    }
}
