// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections;
using Idrak.Data.Abstractions;
using Idrak.Data;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Training;
using Idrak;

namespace Idrak.PluginTests;

/// <summary>
/// Samples computed when read: points on two rings around the origin, labelled by ring (one-hot). Nothing is stored,
/// so any number of samples costs no memory.
/// </summary>
public sealed class RingSource(int count, int seed) : ISampleSource
{
    private static readonly int[] Features = [2], Targets = [2];

    /// <inheritdoc />
    public int Count => count;

    /// <inheritdoc />
    public IReadOnlyList<int> FeatureShape => Features;

    /// <inheritdoc />
    public IReadOnlyList<int> TargetShape => Targets;

    /// <inheritdoc />
    public void Read(int index, Span<float> features, Span<float> targets)
    {
        // The same sample for the same index, from any thread.
        var random = new Random(seed * 1_000_003 + index);
        int ring = index % 2;
        double angle = random.NextDouble() * 2 * Math.PI, radius = (ring == 0 ? 0.5 : 1.5) + (random.NextDouble() - 0.5) * 0.3;
        features[0] = (float)(radius * Math.Cos(angle));
        features[1] = (float)(radius * Math.Sin(angle));
        targets.Clear();
        targets[ring] = 1f;
    }
}

/// <summary>
/// Batches made directly as tensors on the device, without a <see cref="DataLoader"/>: y = 2 x0 - 3 x1 + 0.5, a fresh
/// random batch every step, a fixed number of steps per epoch.
/// </summary>
public sealed class LinearBatches(Device device, int batches, int size, int seed) : IBatchSource
{
    private readonly Random _random = new(seed);

    /// <summary>Batches per epoch.</summary>
    public int? BatchCount => batches;

    /// <summary>Samples per batch.</summary>
    public int? BatchSize => size;

    /// <inheritdoc />
    public IEnumerator<Batch> GetEnumerator()
    {
        for (int b = 0; b < batches; b++)
        {
            var x = new float[size * 2];
            var y = new float[size];
            for (int i = 0; i < size; i++)
            {
                x[2 * i] = _random.NextSingle() * 2 - 1;
                x[2 * i + 1] = _random.NextSingle() * 2 - 1;
                y[i] = 2 * x[2 * i] - 3 * x[2 * i + 1] + 0.5f;
            }

            yield return new Batch(Tensor.From(x, [size, 2], device), Tensor.From(y, [size, 1], device), b);
        }
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}

/// <summary>The data plug-in tests: a sample source and a batch source written outside the library train models.</summary>
public static class DataPluginTests
{
    /// <summary>A custom source trains a classifier through a <see cref="DataLoader"/>, to the same weights as its in-memory copy.</summary>
    public static void SourceTrains(Device device)
    {
        var rings = new RingSource(400, seed: 3);
        float[] Train(ISampleSource source, out double accuracy)
        {
            var init = new Random(5);
            using var model = new Sequential { new Linear(2, 16, device: device, random: init), new ReLU(), new Linear(16, 2, device: device, random: init) };
            using var optimizer = new Adam(model.Parameters(), 0.02f);
            using var trainer = new Trainer(model, optimizer, (p, t) => Losses.CrossEntropy(p, t)) { Metrics = { Metric.Accuracy } };
            trainer.Fit(new DataLoader(source, 32, shuffle: true, device: device, seed: 9), 40);
            accuracy = trainer.Evaluate(new DataLoader(new RingSource(200, seed: 4), 100, device: device)).Metrics["accuracy"];
            return [.. model.Parameters().SelectMany(p => p.ToArray())];
        }

        var fromSource = Train(rings, out double accuracy);
        Check(accuracy > 0.95, $"rings classified on fresh samples: accuracy {accuracy:P1}");
        var memory = Dataset.FromSource(rings);
        Check(memory.Count == 400 && memory.FeatureShape.SequenceEqual([2]), "the in-memory copy");
        var fromMemory = Train(memory, out _);
        for (int i = 0; i < fromSource.Length; i++)
        {
            Check(MathF.Abs(fromSource[i] - fromMemory[i]) <= 1e-6f * Math.Max(1f, MathF.Abs(fromMemory[i])), $"weight {i}: {fromSource[i]} from the source, {fromMemory[i]} from memory");
        }

        // A view of it, with a transform of its own.
        var (train, _) = rings.Split(0.5, seed: 1);
        var noisy = new DataLoader(train, 50, device: device, seed: 2) { Transforms = [new GaussianNoise(0.01f)] };
        int seen = 0;
        foreach (var batch in noisy)
        {
            using (batch)
            {
                seen += batch.Size;
            }
        }

        Check(seen == 200, $"the split view gave {seen} samples");
    }

    /// <summary>A custom batch source trains a linear model through <see cref="Trainer.Fit"/>, with its counts in the history.</summary>
    public static void BatchSourceTrains(Device device)
    {
        using var model = new Linear(2, 1, device: device, random: new Random(1));
        using var optimizer = new Adam(model.Parameters(), 0.05f);
        using var trainer = new Trainer(model, optimizer, Losses.MeanSquaredError);
        var history = trainer.Fit(new LinearBatches(device, batches: 20, size: 32, seed: 7), 15);
        Check(history.Epochs.Count == 15 && history.Epochs[^1].Loss < 0.01 * history.Epochs[0].Loss, $"loss {history.Epochs[0].Loss} -> {history.Epochs[^1].Loss}");
        var weights = model.Parameters().SelectMany(p => p.ToArray()).ToArray();
        float[] expected = [2f, -3f, 0.5f];
        for (int i = 0; i < 3; i++)
        {
            Check(MathF.Abs(weights[i] - expected[i]) < 0.05f, $"learned [{string.Join(", ", weights)}], expected [{string.Join(", ", expected)}]");
        }

        var evaluation = trainer.Evaluate(new LinearBatches(device, batches: 2, size: 50, seed: 8));
        Check(evaluation.Samples == 100 && evaluation.Loss < 0.01, $"evaluated on a batch source: {evaluation}");
    }

    private static void Check(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }
}
