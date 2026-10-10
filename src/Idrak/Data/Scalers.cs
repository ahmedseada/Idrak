// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Data.Abstractions;

namespace Idrak.Data;

/// <summary>Scales each column to zero mean and unit variance: (x - mean) / std.</summary>
public sealed class StandardScaler : IScaler
{
    private readonly float[] _mean, _std;

    private StandardScaler(float[] mean, float[] std)
    {
        _mean = mean;
        _std = std;
    }

    /// <summary>Per-column means.</summary>
    public IReadOnlyList<float> Mean => _mean;

    /// <summary>Per-column standard deviations (1 for constant columns).</summary>
    public IReadOnlyList<float> Std => _std;

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
        int width = RowWidth(columns, data.Length);
        for (int start = 0; start < data.Length; start += width)
        {
            var row = data.Slice(start, Math.Min(width, data.Length - start));
            for (int c = 0; c < row.Length; c++)
            {
                row[c] = (row[c] - _mean[c]) / _std[c];
            }
        }
    }

    /// <inheritdoc />
    public void InverseTransform(Span<float> data, int columns)
    {
        Check(columns);
        int width = RowWidth(columns, data.Length);
        for (int start = 0; start < data.Length; start += width)
        {
            var row = data.Slice(start, Math.Min(width, data.Length - start));
            for (int c = 0; c < row.Length; c++)
            {
                row[c] = row[c] * _std[c] + _mean[c];
            }
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

    // The values of a row: the column of value i is i % columns (row by row instead of a division per value, and the
    // same answer for any columns, 0 failing as the division did).
    internal static int RowWidth(int columns, int length) =>
        columns == 0 && length > 0 ? throw new DivideByZeroException()
        : columns == int.MinValue ? int.MaxValue
        : Math.Max(Math.Abs(columns), 1);
}

/// <summary>Scales each column linearly into [0, 1] using the fitted minimum and maximum.</summary>
public sealed class MinMaxScaler : IScaler
{
    private readonly float[] _min, _range;

    private MinMaxScaler(float[] min, float[] range)
    {
        _min = min;
        _range = range;
    }

    /// <summary>Per-column minimums.</summary>
    public IReadOnlyList<float> Min => _min;

    /// <summary>Per-column max - min (1 for constant columns).</summary>
    public IReadOnlyList<float> Range => _range;

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
        int width = Width(columns, data.Length);
        for (int start = 0; start < data.Length; start += width)
        {
            var row = data.Slice(start, Math.Min(width, data.Length - start));
            for (int c = 0; c < row.Length; c++)
            {
                row[c] = (row[c] - _min[c]) / _range[c];
            }
        }
    }

    /// <inheritdoc />
    public void InverseTransform(Span<float> data, int columns)
    {
        int width = Width(columns, data.Length);
        for (int start = 0; start < data.Length; start += width)
        {
            var row = data.Slice(start, Math.Min(width, data.Length - start));
            for (int c = 0; c < row.Length; c++)
            {
                row[c] = row[c] * _range[c] + _min[c];
            }
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

    // StandardScaler.RowWidth, with the error a column past the fitted ones gave when each value was looked up in the lists.
    private int Width(int columns, int length)
    {
        int width = StandardScaler.RowWidth(columns, length);
        return Math.Min(width, length) <= _min.Length ? width
            : throw new ArgumentOutOfRangeException("index", "Index was out of range. Must be non-negative and less than the size of the collection.");
    }

    private static MinMaxScaler Parse(IEnumerable<string> lines)
    {
        var pairs = lines.Select(l => l.Trim()).Where(l => l.Length > 0).Select(l => l.Split(' ')).ToArray();
        return new MinMaxScaler(
            [.. pairs.Select(p => float.Parse(p[0], CultureInfo.InvariantCulture))],
            [.. pairs.Select(p => float.Parse(p[1], CultureInfo.InvariantCulture))]);
    }
}
