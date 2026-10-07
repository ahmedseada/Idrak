// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Data.Abstractions;

namespace Idrak.Data;

/// <summary>Scales each column to zero mean and unit variance: (x - mean) / std.</summary>
public sealed class StandardScaler : IScaler
{
    private StandardScaler(float[] mean, float[] std)
    {
        Mean = mean;
        Std = std;
    }

    /// <summary>Per-column means.</summary>
    public IReadOnlyList<float> Mean { get; }

    /// <summary>Per-column standard deviations (1 for constant columns).</summary>
    public IReadOnlyList<float> Std { get; }

    /// <summary>Computes column statistics of row-major <paramref name="data"/>.</summary>
    public static StandardScaler Fit(ReadOnlySpan<float> data, int columns)
    {
        int rows = data.Length / columns;
        var sum = new double[columns];
        var sumSq = new double[columns];
        for (int r = 0; r < rows; r++)
        {
            var row = data.Slice(r * columns, columns);
            for (int c = 0; c < columns; c++)
            {
                sum[c] += row[c];
                sumSq[c] += (double)row[c] * row[c];
            }
        }

        var mean = new float[columns];
        var std = new float[columns];
        for (int c = 0; c < columns; c++)
        {
            double m = sum[c] / Math.Max(rows, 1);
            double variance = Math.Max(sumSq[c] / Math.Max(rows, 1) - m * m, 0);
            mean[c] = (float)m;
            std[c] = variance > 1e-12 ? (float)Math.Sqrt(variance) : 1f;
        }

        return new StandardScaler(mean, std);
    }

    /// <summary>Fits on a dataset's features.</summary>
    public static StandardScaler FitFeatures(Dataset dataset) => Fit(dataset.Features, dataset.FeatureCount);

    /// <summary>Fits on a dataset's targets.</summary>
    public static StandardScaler FitTargets(Dataset dataset) => Fit(dataset.Targets, dataset.TargetCount);

    /// <inheritdoc />
    public void Transform(Span<float> data, int columns)
    {
        Check(columns);
        for (int i = 0; i < data.Length; i++)
        {
            int c = i % columns;
            data[i] = (data[i] - Mean[c]) / Std[c];
        }
    }

    /// <inheritdoc />
    public void InverseTransform(Span<float> data, int columns)
    {
        Check(columns);
        for (int i = 0; i < data.Length; i++)
        {
            int c = i % columns;
            data[i] = data[i] * Std[c] + Mean[c];
        }
    }

    /// <summary>Saves the statistics as text (one "mean std" pair per line).</summary>
    public void Save(string path) =>
        File.WriteAllLines(path, Lines());

    /// <summary>Writes the same text as <see cref="Save(string)"/> to <paramref name="writer"/>.</summary>
    public void Save(TextWriter writer)
    {
        foreach (var line in Lines())
        {
            writer.WriteLine(line);
        }
    }

    private IEnumerable<string> Lines() =>
        Mean.Zip(Std, (m, s) => $"{m.ToString("R", CultureInfo.InvariantCulture)} {s.ToString("R", CultureInfo.InvariantCulture)}");

    /// <summary>Loads statistics written by <see cref="Save(string)"/>.</summary>
    public static StandardScaler Load(string path) => Parse(File.ReadAllLines(path));

    /// <summary>Reads statistics written by <see cref="Save(TextWriter)"/>.</summary>
    public static StandardScaler Load(TextReader reader) => Parse(reader.ReadToEnd().Split('\n'));

    private static StandardScaler Parse(IEnumerable<string> lines)
    {
        var pairs = lines.Select(l => l.Trim()).Where(l => l.Length > 0).Select(l => l.Split(' ')).ToArray();
        return new StandardScaler(
            [.. pairs.Select(p => float.Parse(p[0], CultureInfo.InvariantCulture))],
            [.. pairs.Select(p => float.Parse(p[1], CultureInfo.InvariantCulture))]);
    }

    private void Check(int columns)
    {
        if (columns != Mean.Count)
        {
            throw new ArgumentException($"The scaler was fitted on {Mean.Count} columns, not {columns}.");
        }
    }
}

/// <summary>Scales each column linearly into [0, 1] using the fitted minimum and maximum.</summary>
public sealed class MinMaxScaler : IScaler
{
    private MinMaxScaler(float[] min, float[] range)
    {
        Min = min;
        Range = range;
    }

    /// <summary>Per-column minimums.</summary>
    public IReadOnlyList<float> Min { get; }

    /// <summary>Per-column max - min (1 for constant columns).</summary>
    public IReadOnlyList<float> Range { get; }

    /// <summary>Computes column ranges of row-major <paramref name="data"/>.</summary>
    public static MinMaxScaler Fit(ReadOnlySpan<float> data, int columns)
    {
        var min = Enumerable.Repeat(float.PositiveInfinity, columns).ToArray();
        var max = Enumerable.Repeat(float.NegativeInfinity, columns).ToArray();
        for (int i = 0; i < data.Length; i++)
        {
            int c = i % columns;
            min[c] = MathF.Min(min[c], data[i]);
            max[c] = MathF.Max(max[c], data[i]);
        }

        var range = new float[columns];
        for (int c = 0; c < columns; c++)
        {
            range[c] = max[c] > min[c] ? max[c] - min[c] : 1f;
        }

        return new MinMaxScaler(min, range);
    }

    /// <inheritdoc />
    public void Transform(Span<float> data, int columns)
    {
        for (int i = 0; i < data.Length; i++)
        {
            int c = i % columns;
            data[i] = (data[i] - Min[c]) / Range[c];
        }
    }

    /// <inheritdoc />
    public void InverseTransform(Span<float> data, int columns)
    {
        for (int i = 0; i < data.Length; i++)
        {
            int c = i % columns;
            data[i] = data[i] * Range[c] + Min[c];
        }
    }

    /// <summary>Saves the ranges as text (one "min range" pair per line), like <see cref="StandardScaler.Save(string)"/>.</summary>
    public void Save(string path) => File.WriteAllLines(path, Lines());

    /// <summary>Writes the same text as <see cref="Save(string)"/> to <paramref name="writer"/>.</summary>
    public void Save(TextWriter writer)
    {
        foreach (var line in Lines())
        {
            writer.WriteLine(line);
        }
    }

    private IEnumerable<string> Lines() =>
        Min.Zip(Range, (m, r) => $"{m.ToString("R", CultureInfo.InvariantCulture)} {r.ToString("R", CultureInfo.InvariantCulture)}");

    /// <summary>Loads ranges written by <see cref="Save(string)"/>.</summary>
    public static MinMaxScaler Load(string path) => Parse(File.ReadAllLines(path));

    /// <summary>Reads ranges written by <see cref="Save(TextWriter)"/>.</summary>
    public static MinMaxScaler Load(TextReader reader) => Parse(reader.ReadToEnd().Split('\n'));

    private static MinMaxScaler Parse(IEnumerable<string> lines)
    {
        var pairs = lines.Select(l => l.Trim()).Where(l => l.Length > 0).Select(l => l.Split(' ')).ToArray();
        return new MinMaxScaler(
            [.. pairs.Select(p => float.Parse(p[0], CultureInfo.InvariantCulture))],
            [.. pairs.Select(p => float.Parse(p[1], CultureInfo.InvariantCulture))]);
    }
}
