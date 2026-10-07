// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Runtime.InteropServices;
using Idrak.Data.Abstractions;

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
/// target matrix, both row-major float32. Datasets are immutable; transformations return new ones. A dataset is the
/// in-memory <see cref="ISampleSource"/>; <see cref="FromSource"/> reads any other source into one.
/// </summary>
public sealed class Dataset : ISampleSource
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
        return FromFlat(ArrayLayout.Flatten(features), targets, labels.Length,
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

    /// <summary>The shape of one sample's targets: [<see cref="TargetCount"/>].</summary>
    public IReadOnlyList<int> TargetShape => _targetShape ??= [TargetCount];

    private int[]? _targetShape;

    /// <summary>Copies sample <paramref name="index"/> into the spans (<see cref="ISampleSource.Read"/>).</summary>
    public void Read(int index, Span<float> features, Span<float> targets)
    {
        GetFeatures(index).CopyTo(features);
        GetTargets(index).CopyTo(targets);
    }

    /// <summary>
    /// Reads every sample of <paramref name="source"/> into memory (several at once), keeping its feature shape; the
    /// targets of a sample become one row (the product of the target shape). Column names default to x0, x1, ... and
    /// y0, y1, ....
    /// </summary>
    /// <param name="source">The samples; it must allow concurrent reads (the built-in sources do).</param>
    /// <param name="featureNames">One name per feature value, or null.</param>
    /// <param name="targetNames">One name per target value (class names for one-hot targets), or null.</param>
    public static Dataset FromSource(ISampleSource source, IReadOnlyList<string>? featureNames = null, IReadOnlyList<string>? targetNames = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (source is Dataset dataset && featureNames is null && targetNames is null)
        {
            return dataset;
        }

        int count = source.Count, f = SampleSourceExtensions.Size(source.FeatureShape), t = SampleSourceExtensions.Size(source.TargetShape);
        if (featureNames is not null && featureNames.Count != f || targetNames is not null && targetNames.Count != t)
        {
            throw new ArgumentException($"The source has {f} feature and {t} target values per sample; the names do not match.");
        }

        var features = new float[checked(count * f)];
        var targets = new float[checked(count * t)];
        try
        {
            Parallel.For(0, count, ComputeResources.ParallelOptions, i => source.Read(i, features.AsSpan(i * f, f), targets.AsSpan(i * t, t)));
        }
        catch (AggregateException ex) when (ex.InnerExceptions.Count > 0)
        {
            // Surface the first read error itself rather than the parallel loop's wrapper.
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ex.InnerExceptions[0]).Throw();
        }

        return new Dataset(features, targets, count, featureNames ?? DefaultNames("x", f), targetNames ?? DefaultNames("y", t), [.. source.FeatureShape]);
    }

    /// <summary>Creates a dataset from rectangular arrays with one row per sample.</summary>
    public static Dataset FromArrays(float[,] features, float[,] targets, IReadOnlyList<string>? featureNames = null, IReadOnlyList<string>? targetNames = null)
    {
        if (features.GetLength(0) != targets.GetLength(0))
        {
            throw new ArgumentException($"features has {features.GetLength(0)} rows but targets has {targets.GetLength(0)}.");
        }

        return FromFlat(
            ArrayLayout.Flatten(features), ArrayLayout.Flatten(targets), features.GetLength(0),
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
    public static Dataset LoadCsv(string path, CsvOptions options)
    {
        string text = File.ReadAllText(path);
        return ParseCsv(text, Lines(text, alsoCarriageReturn: true), options, path);
    }

    /// <summary>Parses CSV text already in memory (see <see cref="LoadCsv"/>).</summary>
    public static Dataset ParseCsv(string text, CsvOptions options) => ParseCsv(text, Lines(text, alsoCarriageReturn: false), options, "text");

    // Where each line starts and ends, rather than a string per line. Lines end at \r\n or \n, as string.Split(["\r\n", "\n"]) splits
    // them, or also at a lone \r with no empty line after the last break, as File.ReadAllLines splits them.
    private static List<Range> Lines(string text, bool alsoCarriageReturn)
    {
        var lines = new List<Range>();
        int start = 0;
        while (true)
        {
            var rest = text.AsSpan(start);
            int found = alsoCarriageReturn ? rest.IndexOfAny('\r', '\n') : rest.IndexOf('\n');
            if (found < 0)
            {
                if (!alsoCarriageReturn || start < text.Length)
                {
                    lines.Add(start..text.Length);
                }

                return lines;
            }

            int end = start + found;
            if (!alsoCarriageReturn && end > start && text[end - 1] == '\r')
            {
                lines.Add(start..(end - 1));
            }
            else
            {
                lines.Add(start..end);
            }

            start = end + 1;
            if (alsoCarriageReturn && text[end] == '\r' && start < text.Length && text[start] == '\n')
            {
                start++;
            }
        }
    }

    private static Dataset ParseCsv(string text, List<Range> lines, CsvOptions options, string source)
    {
        int first = 0;
        while (first < lines.Count && text.AsSpan()[lines[first]].IsWhiteSpace())
        {
            first++;
        }

        if (first == lines.Count)
        {
            throw new FormatException($"{source} is empty.");
        }

        string[] header = SplitLine(text[lines[first]], options.Delimiter);
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
        // The columns parsed: features and targets (ignored ones are skipped).
        var parsed = new bool[header.Length];
        foreach (int i in featureColumns.Concat(targetColumns))
        {
            parsed[i] = true;
        }

        if (targetColumns.Length == 0)
        {
            throw new ArgumentException("At least one target column is required.", nameof(options));
        }

        var rows = new List<int>(lines.Count - first);
        for (int i = first; i < lines.Count; i++)
        {
            if (!text.AsSpan()[lines[i]].IsWhiteSpace())
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
            var line = text.AsSpan()[lines[lineIndex]];
            int column = 0;
            foreach (var range in line.Split(options.Delimiter))
            {
                if (column >= values.Length)
                {
                    throw new FormatException($"{source} line {lineIndex + 1}: more than {values.Length} fields.");
                }

                var field = line[range].Trim().Trim('"');
                if (parsed[column])
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

    /// <summary>
    /// Shuffles the samples and splits them into a training and a test set. By default the rows are deduplicated first
    /// (<see cref="Deduplicate"/> with <paramref name="duplicates"/>, or its default rules: rows with the same features
    /// count once, the first kept), so no row is in both sets and the test score is on rows the model did not see.
    /// <paramref name="removeDuplicates"/> false turns the whole step off and keeps every row, identical copies included
    /// (for augmentation, or when how often a row repeats matters).
    /// </summary>
    /// <param name="trainFraction">Share of samples for the training set, e.g. 0.8.</param>
    /// <param name="seed">Shuffle seed, for reproducible splits.</param>
    /// <param name="removeDuplicates">Deduplicate before splitting (the default); false keeps every row.</param>
    /// <param name="duplicates">The deduplication rules (null: the defaults of <see cref="DeduplicationOptions"/>).</param>
    public (Dataset Train, Dataset Test) Split(double trainFraction, int seed = 0, bool removeDuplicates = true, DeduplicationOptions? duplicates = null)
    {
        if (trainFraction is <= 0 or >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(trainFraction), trainFraction, "Must be between 0 and 1.");
        }

        if (removeDuplicates)
        {
            var unique = Deduplicate(duplicates);
            if (!ReferenceEquals(unique, this))
            {
                return unique.Split(trainFraction, seed, removeDuplicates: false);
            }
        }

        int[] order = [.. Enumerable.Range(0, Count)];
        new Random(seed).Shuffle(order);
        int trainCount = (int)Math.Round(Count * trainFraction);
        return (Subset(order.AsSpan(0, trainCount)), Subset(order.AsSpan(trainCount)));
    }

    /// <summary>
    /// The rows without duplicates, in the original order (the first of each group is kept); the same dataset when there
    /// are none (datasets are not changed in place, so nothing is copied). Rows are compared bit for
    /// bit (0 and -0 differ; a NaN equals the same NaN); what counts as a duplicate, and what happens to rows whose
    /// features repeat with other targets, is set by <paramref name="options"/>; with an embedding, near duplicates go
    /// too. <see cref="Split"/> runs it by default (<c>removeDuplicates: false</c> keeps every row); on its own it gives
    /// the deduplicated rows for other uses, and <see cref="DeduplicateWithReport"/> says what was removed.
    /// </summary>
    public Dataset Deduplicate(DeduplicationOptions? options = null) => DeduplicateWithReport(options).Data;

    /// <summary>
    /// <see cref="Deduplicate"/>, with which rows were kept (their indices here) and how many were removed for each reason.
    /// </summary>
    public DeduplicationResult DeduplicateWithReport(DeduplicationOptions? options = null)
    {
        options ??= new DeduplicationOptions();
        if (options.Embedding is not null && options.SimilarityThreshold is not (> -1f and <= 1f))
        {
            throw new ArgumentOutOfRangeException(nameof(options), options.SimilarityThreshold, "SimilarityThreshold must be in (-1, 1].");
        }

        // Exact duplicates, without an allocation per row: the rows' hashes (in parallel), then each row joins the group
        // of the first earlier row equal to it. Groups are named by their first row; rows with the same hash but other
        // contents are chained through nextFirst.
        bool withTargets = options.Match == DuplicateMatch.FeaturesAndTargets;
        int count = Count;
        var hashes = new int[count];
        Parallel.For(0, count, row =>
        {
            var hash = new HashCode();
            hash.AddBytes(MemoryMarshal.AsBytes(GetFeatures(row)));
            if (withTargets)
            {
                hash.AddBytes(MemoryMarshal.AsBytes(GetTargets(row)));
            }

            hashes[row] = hash.ToHashCode();
        });

        var group = new int[count];                                    // the first row of each row's group
        var nextFirst = new int[count];                                // the next group's first row with the same hash, or -1
        var firstByHash = new Dictionary<int, int>(count);
        for (int row = 0; row < count; row++)
        {
            ref int head = ref CollectionsMarshal.GetValueRefOrAddDefault(firstByHash, hashes[row], out bool exists);
            if (!exists)
            {
                head = row;
                group[row] = row;
                nextFirst[row] = -1;
                continue;
            }

            int candidate = head, last = -1;
            while (candidate >= 0 && !SameRow(candidate, row, withTargets))
            {
                (last, candidate) = (candidate, nextFirst[candidate]);
            }

            if (candidate >= 0)
            {
                group[row] = candidate;
            }
            else
            {
                group[row] = row;
                nextFirst[row] = -1;
                nextFirst[last] = row;
            }
        }

        // Groups whose rows have other targets (only when matching on the features alone).
        var conflict = new bool[count];
        int exact = 0, conflicting = 0;
        for (int row = 0; row < count; row++)
        {
            int first = group[row];
            if (first != row)
            {
                exact++;
                if (!withTargets && !conflict[first] && !SameTargets(first, row))
                {
                    conflict[first] = true;
                }
            }
        }

        var keep = new bool[count];                                    // reuses nothing else: one byte per row
        for (int row = 0; row < count; row++)
        {
            keep[row] = group[row] == row && !(conflict[row] && options.Conflicts == DuplicateConflicts.DropAll);
        }

        if (options.Conflicts != DuplicateConflicts.KeepFirst && !withTargets)
        {
            var members = new Dictionary<int, List<int>>();            // only the conflicting groups (usually few)
            for (int row = 0; row < count; row++)
            {
                if (conflict[group[row]])
                {
                    ref var list = ref CollectionsMarshal.GetValueRefOrAddDefault(members, group[row], out _);
                    (list ??= []).Add(row);
                }
            }

            foreach (var (first, rows) in members)
            {
                if (options.Conflicts == DuplicateConflicts.DropAll)
                {
                    conflicting += rows.Count;
                    exact -= rows.Count - 1;
                    continue;
                }

                // KeepMostFrequent: the first row of the targets most rows have (the earliest such targets on a tie). One
                // pass: rows are counted under the first row with their targets, found through the targets' hash.
                var tally = new Dictionary<int, List<(int First, int Votes)>>();
                foreach (int row in rows)
                {
                    var hash = new HashCode();
                    hash.AddBytes(MemoryMarshal.AsBytes(GetTargets(row)));
                    ref var entries = ref CollectionsMarshal.GetValueRefOrAddDefault(tally, hash.ToHashCode(), out _);
                    entries ??= [];
                    int at = entries.FindIndex(e => SameTargets(e.First, row));
                    if (at < 0)
                    {
                        entries.Add((row, 1));
                    }
                    else
                    {
                        entries[at] = (entries[at].First, entries[at].Votes + 1);
                    }
                }

                int best = first, bestVotes = 0;
                foreach (var entries in tally.Values)
                {
                    foreach (var (candidate, votes) in entries)
                    {
                        if (votes > bestVotes || (votes == bestVotes && candidate < best))
                        {
                            (best, bestVotes) = (candidate, votes);
                        }
                    }
                }

                keep[first] = false;
                keep[best] = true;
            }
        }

        int near = 0;
        if (options.Embedding is { } embed)
        {
            near = DropNearDuplicates(keep, embed, options.SimilarityThreshold, withTargets);
        }

        int kept = 0;
        foreach (bool k in keep)
        {
            kept += k ? 1 : 0;
        }

        if (kept == count)
        {
            return new DeduplicationResult(this, [.. Enumerable.Range(0, count)], 0, 0, 0);   // nothing removed: no copy
        }

        var indices = new int[kept];
        for (int row = 0, i = 0; row < count; row++)
        {
            if (keep[row])
            {
                indices[i++] = row;
            }
        }

        return new DeduplicationResult(Subset(indices), indices, exact, conflicting, near);
    }

    // Greedy, in order: a kept row goes when its embedding is close enough to one kept before it. Rows are taken in blocks:
    // a block's rows are compared, in parallel, with the kept embeddings one cache-sized tile at a time (a row stops at
    // its first match), then in order with the rows of the block kept before them. The kept embeddings, normalized, sit in
    // one growing buffer.
    private int DropNearDuplicates(bool[] keep, Func<int, float[]> embed, float threshold, bool withTargets)
    {
        const int Block = 512, Tile = 1024;
        int[] candidates = [.. Enumerable.Range(0, keep.Length).Where(r => keep[r])];
        if (candidates.Length == 0)
        {
            return 0;
        }

        int dimension = -1, stored = 0, near = 0;
        float[] vectors = [], block = [];
        int[] owners = new int[Math.Min(candidates.Length, 4096)];
        var found = new bool[Block];
        var scores = new float[Block * Tile];
        var inBlock = new float[Block * Block];
        var blockIndex = new int[Block];                                // position in the block of each row it kept
        var tiles = new List<Tensor>();                                 // full tiles of kept embeddings, as tensors
        try
        {
        for (int start = 0; start < candidates.Length; start += Block)
        {
            int size = Math.Min(Block, candidates.Length - start);
            for (int c = 0; c < size; c++)
            {
                int row = candidates[start + c];
                float[] vector = embed(row);                            // the caller's function: not assumed thread-safe
                if (dimension < 0)
                {
                    dimension = vector.Length;
                    block = new float[Block * dimension];
                    vectors = new float[owners.Length * dimension];
                }
                else if (vector.Length != dimension)
                {
                    throw new ArgumentException($"Embeddings of different lengths ({dimension} and {vector.Length}, row {row}).");
                }

                float norm = MathF.Sqrt(Dot(vector, vector));
                if (norm == 0 || float.IsNaN(norm))
                {
                    throw new ArgumentException($"The embedding of row {row} has no length (all zeros or NaN).");
                }

                var target = block.AsSpan(c * dimension, dimension);
                for (int i = 0; i < dimension; i++)
                {
                    target[i] = vector[i] / norm;
                }
            }

            // The block against the kept embeddings, one tile at a time, as one matrix product per tile (the CPU kernels):
            // scores[c, k] = cosine of block row c and kept row k. Full tiles never change, so their tensors are kept.
            Array.Clear(found);
            if (stored > 0)
            {
                using var blockTensor = Tensor.From(block.AsSpan(0, size * dimension), [size, dimension], Device.Cpu);
                for (int tile = 0, t = 0; tile < stored; tile += Tile, t++)
                {
                    int length = Math.Min(Tile, stored - tile);
                    Tensor tileTensor;
                    if (length == Tile)
                    {
                        if (t == tiles.Count)
                        {
                            tiles.Add(Tensor.From(vectors.AsSpan(tile * dimension, Tile * dimension), [Tile, dimension], Device.Cpu));
                        }

                        tileTensor = tiles[t];
                    }
                    else
                    {
                        tileTensor = Tensor.From(vectors.AsSpan(tile * dimension, length * dimension), [length, dimension], Device.Cpu);
                    }

                    try
                    {
                        using var product = blockTensor.MatMul(tileTensor, transposeB: true);
                        product.CopyTo(scores.AsSpan(0, size * length));
                    }
                    finally
                    {
                        if (length != Tile)
                        {
                            tileTensor.Dispose();
                        }
                    }

                    int first = tile, offset = start, width = length;
                    Parallel.For(0, size, c =>
                    {
                        if (found[c])
                        {
                            return;
                        }

                        var row = scores.AsSpan(c * width, width);
                        for (int k = 0; k < width; k++)
                        {
                            if (row[k] >= threshold && (!withTargets || SameTargets(candidates[offset + c], owners[first + k])))
                            {
                                found[c] = true;
                                return;
                            }
                        }
                    });
                }
            }

            // Then in order within the block: a row goes when it is close to a row of the block kept before it.
            using (var own = Tensor.From(block.AsSpan(0, size * dimension), [size, dimension], Device.Cpu))
            using (var similar = own.MatMul(own, transposeB: true))
            {
                similar.CopyTo(inBlock.AsSpan(0, size * size));
            }

            int blockFirst = stored;                                    // this block's kept rows start here
            for (int c = 0; c < size; c++)
            {
                int row = candidates[start + c];
                var candidate = block.AsSpan(c * dimension, dimension);
                for (int k = blockFirst; k < stored && !found[c]; k++)
                {
                    found[c] = inBlock[c * size + (blockIndex[k - blockFirst])] >= threshold && (!withTargets || SameTargets(row, owners[k]));
                }

                if (found[c])
                {
                    keep[row] = false;
                    near++;
                    continue;
                }

                if (stored == owners.Length)
                {
                    Array.Resize(ref owners, Math.Min(candidates.Length, owners.Length * 2));
                    Array.Resize(ref vectors, owners.Length * dimension);
                }

                candidate.CopyTo(vectors.AsSpan(stored * dimension));
                blockIndex[stored - blockFirst] = c;
                owners[stored++] = row;
            }
        }
        }
        finally
        {
            foreach (var tile in tiles)
            {
                tile.Dispose();
            }
        }

        return near;

        static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
        {
            var sum = System.Numerics.Vector<float>.Zero;
            int width = System.Numerics.Vector<float>.Count, i = 0;
            for (; i <= a.Length - width; i += width)
            {
                sum += new System.Numerics.Vector<float>(a.Slice(i)) * new System.Numerics.Vector<float>(b.Slice(i));
            }

            float dot = System.Numerics.Vector.Sum(sum);
            for (; i < a.Length; i++)
            {
                dot += a[i] * b[i];
            }

            return dot;
        }
    }

    // Bit for bit, so a NaN matches the same NaN and the comparison agrees with the hash.
    private bool SameRow(int a, int b, bool withTargets) =>
        MemoryMarshal.AsBytes(GetFeatures(a)).SequenceEqual(MemoryMarshal.AsBytes(GetFeatures(b))) && (!withTargets || SameTargets(a, b));

    private bool SameTargets(int a, int b) => MemoryMarshal.AsBytes(GetTargets(a)).SequenceEqual(MemoryMarshal.AsBytes(GetTargets(b)));

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
        ArrayLayout.Unflatten(flat, result);
        return result;
    }

    /// <inheritdoc />
    public override string ToString() => $"Dataset({Count:N0} samples, features [{string.Join(", ", FeatureNames)}] -> targets [{string.Join(", ", TargetNames)}])";
}

// Row-major copies between two-dimensional arrays and flat ones.
internal static class ArrayLayout
{
    public static float[] Flatten(float[,] values)
    {
        var flat = new float[values.Length];
        System.Runtime.InteropServices.MemoryMarshal.CreateReadOnlySpan(
            ref System.Runtime.CompilerServices.Unsafe.As<byte, float>(ref System.Runtime.InteropServices.MemoryMarshal.GetArrayDataReference(values)), values.Length).CopyTo(flat);
        return flat;
    }

    public static void Unflatten(float[] flat, float[,] destination) =>
        flat.AsSpan().CopyTo(System.Runtime.InteropServices.MemoryMarshal.CreateSpan(
            ref System.Runtime.CompilerServices.Unsafe.As<byte, float>(ref System.Runtime.InteropServices.MemoryMarshal.GetArrayDataReference(destination)), destination.Length));
}
