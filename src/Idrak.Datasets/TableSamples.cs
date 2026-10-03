// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Data;
using TensorData = Idrak.Data.Dataset;

namespace Idrak.Datasets;

/// <summary>
/// Columns of data files (JSON Lines, JSON, CSV, TSV, Parquet, compressed or in archives: whatever
/// <see cref="DataFiles.Read"/> reads) as training samples: the feature columns become the features and the target
/// columns the targets. A cell is a number, a boolean (1 or 0), a number in text, or an array of numbers (an embedding,
/// a pixel row), which gives as many values as the first row's array has. With <c>classes</c>, the single target column
/// holds class names (or numbers) and becomes one-hot rows. A missing or unreadable cell is an error naming the row and
/// the column.
/// </summary>
/// <example>
/// <code>
/// var data = TableSamples.Load("houses.parquet", features: ["area", "rooms", "age"], targets: ["price"]);
/// var (train, test) = data.Split(0.8).StandardizeFeatures();
/// using var rows = new DataLoader(TableSamples.Stream("huge.jsonl", ["x"], ["y"]), batchSize: 256, shuffleBuffer: 10_000);
/// </code>
/// </example>
public static class TableSamples
{
    /// <summary>Reads the columns of <paramref name="path"/> into an in-memory dataset.</summary>
    /// <param name="path">The data file.</param>
    /// <param name="features">The feature columns, in order.</param>
    /// <param name="targets">The target columns, in order (one column with <paramref name="classes"/>).</param>
    /// <param name="classes">The class names of a single target column, in label order; null for numeric targets.</param>
    /// <param name="options">How the file is read (format, CSV types); null for the defaults.</param>
    public static TensorData Load(string path, IReadOnlyList<string> features, IReadOnlyList<string> targets,
        IReadOnlyList<string>? classes = null, ReadOptions? options = null) =>
        FromRows(DataFiles.Read(path, options ?? ReadOptions.Default), features, targets, classes, path);

    /// <summary>The columns of rows already read (from <see cref="DataFiles"/>, a <see cref="Dataset"/> or your own code) as an in-memory dataset.</summary>
    public static TensorData FromRows(IEnumerable<JsonObject> rows, IReadOnlyList<string> features, IReadOnlyList<string> targets,
        IReadOnlyList<string>? classes = null) =>
        FromRows(rows, features, targets, classes, "rows");

    /// <summary>
    /// The columns of <paramref name="path"/> read in order, again on every pass, without holding them: for files
    /// larger than memory (batch it with <see cref="DataLoader(ISampleStream, int, int, bool, Device?, int?)"/>). The
    /// first row is read now, for the shapes.
    /// </summary>
    public static ISampleStream Stream(string path, IReadOnlyList<string> features, IReadOnlyList<string> targets,
        IReadOnlyList<string>? classes = null, ReadOptions? options = null)
    {
        var read = options ?? ReadOptions.Default;
        var first = DataFiles.Read(path, read).FirstOrDefault() ?? throw new InvalidDataException($"{path} has no rows.");
        var layout = new Layout(first, features, targets, classes, path);
        return new RowStream(() => DataFiles.Read(path, read), layout);
    }

    /// <summary>
    /// A factory for <see cref="SampleSources.Register"/> (register it as "table"): options <c>features</c> and
    /// <c>target</c> (comma-separated column names; <c>features</c> defaults to every other column of the first row) and
    /// <c>classes</c> (comma-separated class names).
    /// </summary>
    public static SampleSourceFactory Factory { get; } = (path, options) =>
    {
        foreach (string key in options.Keys)
        {
            if (key is not ("features" or "target" or "classes"))
            {
                throw new ArgumentException($"The table source has no option '{key}' (features, target, classes).");
            }
        }

        string[] List(string text) => text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        string[] targets = List(options.GetValueOrDefault("target") ?? throw new ArgumentException("The table source needs the option 'target' (the target columns, comma-separated)."));
        var rows = DataFiles.Read(path, ReadOptions.Default);
        string[] features = options.TryGetValue("features", out var f) ? List(f)
            : [.. (rows.FirstOrDefault() ?? throw new InvalidDataException($"{path} has no rows.")).Select(p => p.Key).Where(k => !targets.Contains(k, StringComparer.Ordinal))];
        return FromRows(rows, features, targets, options.TryGetValue("classes", out var c) ? List(c) : null, path);
    };

    private static TensorData FromRows(IEnumerable<JsonObject> rows, IReadOnlyList<string> features, IReadOnlyList<string> targets, IReadOnlyList<string>? classes, string source)
    {
        Layout? layout = null;
        var x = new List<float>();
        var y = new List<float>();
        int count = 0;
        float[]? fx = null, fy = null;
        foreach (var row in rows)
        {
            layout ??= new Layout(row, features, targets, classes, source);
            fx ??= new float[layout.FeatureNames.Count];
            fy ??= new float[layout.TargetNames.Count];
            layout.Fill(row, count, fx, fy);
            x.AddRange(fx);
            y.AddRange(fy);
            count++;
        }

        if (layout is null)
        {
            throw new InvalidDataException($"{source} has no rows.");
        }

        return TensorData.FromFlat([.. x], [.. y], count, layout.FeatureNames, layout.TargetNames);
    }

    // The values each column gives (from the first row) and how a row becomes features and targets.
    private sealed class Layout
    {
        private readonly (string Column, int Width)[] _features, _targets;
        private readonly Dictionary<string, int>? _classes;
        private readonly string _source;

        public Layout(JsonObject first, IReadOnlyList<string> features, IReadOnlyList<string> targets, IReadOnlyList<string>? classes, string source)
        {
            _source = source;
            if (features.Count == 0 || targets.Count == 0)
            {
                throw new ArgumentException("Name at least one feature column and one target column.");
            }

            (string, int) Column(string name)
            {
                if (!first.ContainsKey(name))
                {
                    throw new ArgumentException($"Column '{name}' not found in {source}. Columns: {string.Join(", ", first.Select(p => p.Key))}.");
                }

                return (name, first[name] is JsonArray array ? array.Count : 1);
            }

            _features = [.. features.Select(Column)];
            if (classes is not null)
            {
                if (targets.Count != 1 || classes.Count == 0)
                {
                    throw new ArgumentException("Classes need exactly one target column and at least one class name.");
                }

                _classes = [];
                for (int i = 0; i < classes.Count; i++)
                {
                    _classes.TryAdd(classes[i], i);
                }

                Column(targets[0]);
                _targets = [(targets[0], classes.Count)];
                TargetNames = [.. classes];
            }
            else
            {
                _targets = [.. targets.Select(Column)];
                TargetNames = Names(_targets);
            }

            FeatureNames = Names(_features);
            FeatureShape = [FeatureNames.Count];
            TargetShape = [TargetNames.Count];
        }

        public IReadOnlyList<string> FeatureNames { get; }

        public IReadOnlyList<string> TargetNames { get; }

        public int[] FeatureShape { get; }

        public int[] TargetShape { get; }

        public void Fill(JsonObject row, int index, Span<float> features, Span<float> targets)
        {
            Values(row, index, _features, features);
            if (_classes is null)
            {
                Values(row, index, _targets, targets);
                return;
            }

            string column = _targets[0].Column;
            string key = Key(row[column]) ?? throw Missing(index, column);
            targets.Clear();
            targets[_classes.TryGetValue(key, out int label) ? label
                : throw new FormatException($"{_source} row {index + 1}, column '{column}': '{key}' is not one of the classes ({string.Join(", ", _classes.Keys)}).")] = 1f;
        }

        private void Values(JsonObject row, int index, (string Column, int Width)[] columns, Span<float> destination)
        {
            int at = 0;
            foreach (var (column, width) in columns)
            {
                var cell = row[column];
                if (cell is JsonArray array)
                {
                    if (array.Count != width)
                    {
                        throw new FormatException($"{_source} row {index + 1}, column '{column}': {array.Count} values; the first row has {width}.");
                    }

                    foreach (var item in array)
                    {
                        destination[at++] = Number(item) ?? throw new FormatException($"{_source} row {index + 1}, column '{column}': {item?.ToJsonString() ?? "null"} is not a number.");
                    }
                }
                else
                {
                    if (width != 1)
                    {
                        throw new FormatException($"{_source} row {index + 1}, column '{column}': one value; the first row has {width}.");
                    }

                    destination[at++] = Number(cell) ?? throw (cell is null ? Missing(index, column)
                        : new FormatException($"{_source} row {index + 1}, column '{column}': {cell.ToJsonString()} is not a number."));
                }
            }
        }

        private FormatException Missing(int index, string column) => new($"{_source} row {index + 1}: no value in column '{column}'.");

        private static string[] Names((string Column, int Width)[] columns) =>
            [.. columns.SelectMany(c => c.Width == 1 ? [c.Column] : Enumerable.Range(0, c.Width).Select(i => $"{c.Column}[{i}]"))];

        private static float? Number(JsonNode? cell) => cell switch
        {
            JsonValue v when v.GetValueKind() == JsonValueKind.Number => float.Parse(v.ToJsonString(), NumberStyles.Float, CultureInfo.InvariantCulture),
            JsonValue v when v.GetValueKind() is JsonValueKind.True or JsonValueKind.False => v.GetValueKind() == JsonValueKind.True ? 1f : 0f,
            JsonValue v when v.TryGetValue(out string? s) && float.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out float p) => p,
            _ => null,
        };

        private static string? Key(JsonNode? cell) => cell switch
        {
            null => null,
            JsonValue v when v.TryGetValue(out string? s) => string.IsNullOrWhiteSpace(s) ? null : s,
            JsonValue v when v.GetValueKind() == JsonValueKind.Number => double.Parse(v.ToJsonString(), NumberStyles.Float, CultureInfo.InvariantCulture).ToString(CultureInfo.InvariantCulture),
            _ => cell.ToJsonString(),
        };
    }

    private sealed class RowStream(Func<IEnumerable<JsonObject>> read, Layout layout) : ISampleStream
    {
        public IReadOnlyList<int> FeatureShape => layout.FeatureShape;

        public IReadOnlyList<int> TargetShape => layout.TargetShape;

        public ISampleReader Open() => new Reader(read().GetEnumerator(), layout);
    }

    private sealed class Reader(IEnumerator<JsonObject> rows, Layout layout) : ISampleReader
    {
        private int _index;

        public bool Read(Span<float> features, Span<float> targets)
        {
            if (!rows.MoveNext())
            {
                return false;
            }

            layout.Fill(rows.Current, _index++, features, targets);
            return true;
        }

        public void Dispose() => rows.Dispose();
    }
}
