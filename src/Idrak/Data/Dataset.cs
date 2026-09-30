using System.Globalization;

namespace Idrak.Data;

/// <summary>Options for <see cref="Dataset.LoadCsv"/>.</summary>
public sealed record CsvOptions
{
    /// <summary>Names (or zero-based indices, as text) of the columns to predict.</summary>
    public required IReadOnlyList<string> TargetColumns { get; init; }

    /// <summary>Columns to skip entirely, e.g. an id column.</summary>
    public IReadOnlyList<string> IgnoreColumns { get; init; } = [];

    /// <summary>Field separator.</summary>
    public char Delimiter { get; init; } = ',';

    /// <summary>Whether the first line holds column names. Without a header, columns are named "0", "1", ...</summary>
    public bool HasHeader { get; init; } = true;

    /// <summary>Culture for parsing numbers; invariant by default (dot as decimal separator).</summary>
    public CultureInfo Culture { get; init; } = CultureInfo.InvariantCulture;
}

/// <summary>
/// An in-memory table of samples: a [Count, FeatureCount] feature matrix and a [Count, TargetCount]
/// target matrix, both row-major float32. Datasets are immutable; transformations return new ones.
/// </summary>
public sealed class Dataset
{
    private readonly float[] _features;
    private readonly float[] _targets;

    private Dataset(float[] features, float[] targets, int count, IReadOnlyList<string> featureNames, IReadOnlyList<string> targetNames, int[]? featureShape = null)
    {
        _features = features;
        _targets = targets;
        Count = count;
        FeatureNames = featureNames;
        TargetNames = targetNames;
        FeatureShape = featureShape ?? [featureNames.Count];
    }

    /// <summary>
    /// The shape of one sample's features: [FeatureCount] for tabular data, [C, H, W] for images, [T] for
    /// token sequences. Batches from a <see cref="DataLoader"/> have shape [batch, ..FeatureShape].
    /// </summary>
    public IReadOnlyList<int> FeatureShape { get; }

    /// <summary>Returns the same data with each sample's features viewed as <paramref name="shape"/> (e.g. [1, 28, 28]).</summary>
    public Dataset WithFeatureShape(params int[] shape)
    {
        if (shape.Aggregate(1, (a, b) => a * b) != FeatureCount)
        {
            throw new ArgumentException($"Feature shape [{string.Join(", ", shape)}] does not hold {FeatureCount} values.");
        }

        return new Dataset(_features, _targets, Count, FeatureNames, TargetNames, shape);
    }

    /// <summary>
    /// Creates a classification dataset: <paramref name="labels"/> are class indices in [0, classes) and become
    /// one-hot target rows, ready for <see cref="Losses.CrossEntropy(Tensor, Tensor)"/> and <see cref="Training.Metric.Accuracy"/>.
    /// </summary>
    public static Dataset FromClassLabels(float[,] features, ReadOnlySpan<int> labels, int classes, IReadOnlyList<string>? classNames = null)
    {
        if (features.GetLength(0) != labels.Length)
        {
            throw new ArgumentException($"features has {features.GetLength(0)} rows but there are {labels.Length} labels.");
        }

        var targets = OneHot(labels, classes);
        return FromFlat(MemoryMarshalHelpers.Flatten(features), targets, labels.Length,
            DefaultNames("x", features.GetLength(1)), classNames ?? DefaultNames("class", classes));
    }

    /// <summary>
    /// Converts a single target column of class indices (e.g. loaded from CSV) into one-hot targets.
    /// </summary>
    public Dataset ToOneHot(int classes, IReadOnlyList<string>? classNames = null)
    {
        if (TargetCount != 1)
        {
            throw new InvalidOperationException($"ToOneHot needs exactly one target column of class indices, found {TargetCount}.");
        }

        var labels = new int[Count];
        for (int i = 0; i < Count; i++)
        {
            float v = _targets[i];
            labels[i] = (int)v;
            if (labels[i] != v)
            {
                throw new FormatException($"Target {v} of sample {i} is not a class index.");
            }
        }

        return new Dataset(_features, OneHot(labels, classes), Count, FeatureNames, classNames ?? DefaultNames("class", classes), [.. FeatureShape]);
    }

    private static float[] OneHot(ReadOnlySpan<int> labels, int classes)
    {
        var targets = new float[labels.Length * classes];
        for (int i = 0; i < labels.Length; i++)
        {
            if ((uint)labels[i] >= (uint)classes)
            {
                throw new ArgumentOutOfRangeException(nameof(labels), $"Label {labels[i]} of sample {i} is outside [0, {classes}).");
            }

            targets[i * classes + labels[i]] = 1f;
        }

        return targets;
    }

    /// <summary>Number of samples (rows).</summary>
    public int Count { get; }

    /// <summary>Number of input columns per sample.</summary>
    public int FeatureCount => FeatureNames.Count;

    /// <summary>Number of target columns per sample.</summary>
    public int TargetCount => TargetNames.Count;

    /// <summary>Input column names.</summary>
    public IReadOnlyList<string> FeatureNames { get; }

    /// <summary>Target column names.</summary>
    public IReadOnlyList<string> TargetNames { get; }

    /// <summary>All features, row-major ([Count * FeatureCount]).</summary>
    public ReadOnlySpan<float> Features => _features;

    /// <summary>All targets, row-major ([Count * TargetCount]).</summary>
    public ReadOnlySpan<float> Targets => _targets;

    /// <summary>The features of one sample.</summary>
    public ReadOnlySpan<float> GetFeatures(int index) => _features.AsSpan(index * FeatureCount, FeatureCount);

    /// <summary>The targets of one sample.</summary>
    public ReadOnlySpan<float> GetTargets(int index) => _targets.AsSpan(index * TargetCount, TargetCount);

    /// <summary>Creates a dataset from rectangular arrays with one row per sample.</summary>
    public static Dataset FromArrays(float[,] features, float[,] targets, IReadOnlyList<string>? featureNames = null, IReadOnlyList<string>? targetNames = null)
    {
        if (features.GetLength(0) != targets.GetLength(0))
        {
            throw new ArgumentException($"features has {features.GetLength(0)} rows but targets has {targets.GetLength(0)}.");
        }

        return FromFlat(
            MemoryMarshalHelpers.Flatten(features), MemoryMarshalHelpers.Flatten(targets), features.GetLength(0),
            featureNames ?? DefaultNames("x", features.GetLength(1)), targetNames ?? DefaultNames("y", targets.GetLength(1)));
    }

    /// <summary>Creates a dataset from row-major arrays (the arrays are used directly, not copied).</summary>
    public static Dataset FromFlat(float[] features, float[] targets, int count, IReadOnlyList<string> featureNames, IReadOnlyList<string> targetNames)
    {
        if (features.Length != count * featureNames.Count || targets.Length != count * targetNames.Count)
        {
            throw new ArgumentException("Array lengths do not match count × column counts.");
        }

        return new Dataset(features, targets, count, featureNames, targetNames);
    }

    /// <summary>
    /// Loads a numeric CSV file. Lines are parsed in parallel (bounded by <see cref="ComputeResources.MaxCpuThreads"/>).
    /// Empty lines are skipped; quoted fields are unquoted but may not contain the delimiter.
    /// </summary>
    /// <exception cref="FormatException">A value is not a number; the message names the line and column.</exception>
    public static Dataset LoadCsv(string path, CsvOptions options) => ParseCsv(File.ReadAllLines(path), options, path);

    /// <summary>Parses CSV text already in memory (see <see cref="LoadCsv"/>).</summary>
    public static Dataset ParseCsv(string text, CsvOptions options) =>
        ParseCsv(text.Split(["\r\n", "\n"], StringSplitOptions.None), options, "text");

    private static Dataset ParseCsv(string[] lines, CsvOptions options, string source)
    {
        int first = 0;
        while (first < lines.Length && string.IsNullOrWhiteSpace(lines[first]))
        {
            first++;
        }

        if (first == lines.Length)
        {
            throw new FormatException($"{source} is empty.");
        }

        string[] header = SplitLine(lines[first], options.Delimiter);
        if (options.HasHeader)
        {
            first++;
        }
        else
        {
            header = [.. Enumerable.Range(0, header.Length).Select(i => i.ToString(CultureInfo.InvariantCulture))];
        }

        int Resolve(string column)
        {
            int index = Array.FindIndex(header, h => string.Equals(h, column, StringComparison.OrdinalIgnoreCase));
            if (index < 0 && int.TryParse(column, out int numeric) && numeric >= 0 && numeric < header.Length)
            {
                index = numeric;
            }

            return index >= 0 ? index : throw new ArgumentException($"Column '{column}' not found in {source}. Columns: {string.Join(", ", header)}.");
        }

        int[] targetColumns = [.. options.TargetColumns.Select(Resolve)];
        var ignored = options.IgnoreColumns.Select(Resolve).Concat(targetColumns).ToHashSet();
        int[] featureColumns = [.. Enumerable.Range(0, header.Length).Where(i => !ignored.Contains(i))];
        if (targetColumns.Length == 0)
        {
            throw new ArgumentException("At least one target column is required.", nameof(options));
        }

        var rows = new List<int>(lines.Length - first);
        for (int i = first; i < lines.Length; i++)
        {
            if (!string.IsNullOrWhiteSpace(lines[i]))
            {
                rows.Add(i);
            }
        }

        int count = rows.Count, f = featureColumns.Length, t = targetColumns.Length;
        var features = new float[count * f];
        var targets = new float[count * t];
        try
        {
            ParseRows();
        }
        catch (AggregateException ex) when (ex.InnerExceptions.Count > 0)
        {
            // Surface the first parse error itself rather than the parallel loop's wrapper.
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ex.InnerExceptions[0]).Throw();
        }

        return new Dataset(features, targets, count, [.. featureColumns.Select(i => header[i])], [.. targetColumns.Select(i => header[i])]);

        void ParseRows() => Parallel.For(0, count, ComputeResources.ParallelOptions, () => new float[header.Length], (row, _, values) =>
        {
            int lineIndex = rows[row];
            ReadOnlySpan<char> line = lines[lineIndex];
            int column = 0;
            foreach (var range in line.Split(options.Delimiter))
            {
                if (column >= values.Length)
                {
                    throw new FormatException($"{source} line {lineIndex + 1}: more than {values.Length} fields.");
                }

                var field = line[range].Trim().Trim('"');
                if (!ignored.Contains(column) || targetColumns.Contains(column))
                {
                    if (!float.TryParse(field, NumberStyles.Float, options.Culture, out values[column]))
                    {
                        throw new FormatException($"{source} line {lineIndex + 1}, column '{header[column]}': '{field}' is not a number.");
                    }
                }

                column++;
            }

            if (column != values.Length)
            {
                throw new FormatException($"{source} line {lineIndex + 1}: expected {values.Length} fields, found {column}.");
            }

            for (int j = 0; j < f; j++)
            {
                features[row * f + j] = values[featureColumns[j]];
            }

            for (int j = 0; j < t; j++)
            {
                targets[row * t + j] = values[targetColumns[j]];
            }

            return values;
        }, _ => { });
    }

    private static string[] SplitLine(string line, char delimiter) => [.. line.Split(delimiter).Select(s => s.Trim().Trim('"'))];

    private static string[] DefaultNames(string prefix, int count) => [.. Enumerable.Range(0, count).Select(i => $"{prefix}{i}")];

    /// <summary>Returns the samples at <paramref name="indices"/>, in that order.</summary>
    public Dataset Subset(ReadOnlySpan<int> indices)
    {
        int f = FeatureCount, t = TargetCount;
        var features = new float[indices.Length * f];
        var targets = new float[indices.Length * t];
        for (int i = 0; i < indices.Length; i++)
        {
            GetFeatures(indices[i]).CopyTo(features.AsSpan(i * f, f));
            GetTargets(indices[i]).CopyTo(targets.AsSpan(i * t, t));
        }

        return new Dataset(features, targets, indices.Length, FeatureNames, TargetNames, [.. FeatureShape]);
    }

    /// <summary>Shuffles the samples and splits them into a training and a test set.</summary>
    /// <param name="trainFraction">Share of samples for the training set, e.g. 0.8.</param>
    /// <param name="seed">Shuffle seed, for reproducible splits.</param>
    public (Dataset Train, Dataset Test) Split(double trainFraction, int seed = 0)
    {
        if (trainFraction is <= 0 or >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(trainFraction), trainFraction, "Must be between 0 and 1.");
        }

        int[] order = [.. Enumerable.Range(0, Count)];
        new Random(seed).Shuffle(order);
        int trainCount = (int)Math.Round(Count * trainFraction);
        return (Subset(order.AsSpan(0, trainCount)), Subset(order.AsSpan(trainCount)));
    }

    /// <summary>Returns a copy with features and/or targets transformed by fitted scalers.</summary>
    public Dataset Scale(IScaler? features = null, IScaler? targets = null)
    {
        var f = (float[])_features.Clone();
        var t = (float[])_targets.Clone();
        features?.Transform(f, FeatureCount);
        targets?.Transform(t, TargetCount);
        return new Dataset(f, t, Count, FeatureNames, TargetNames, [.. FeatureShape]);
    }

    /// <summary>The features as a [Count, FeatureCount] array.</summary>
    public float[,] FeaturesToArray() => To2D(_features, Count, FeatureCount);

    /// <summary>The targets as a [Count, TargetCount] array.</summary>
    public float[,] TargetsToArray() => To2D(_targets, Count, TargetCount);

    private static float[,] To2D(float[] flat, int rows, int cols)
    {
        var result = new float[rows, cols];
        MemoryMarshalHelpers.Unflatten(flat, result);
        return result;
    }

    /// <inheritdoc />
    public override string ToString() => $"Dataset({Count:N0} samples, features [{string.Join(", ", FeatureNames)}] -> targets [{string.Join(", ", TargetNames)}])";
}
