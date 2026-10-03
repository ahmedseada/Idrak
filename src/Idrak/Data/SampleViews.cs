// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Data;

/// <summary>
/// Views of any <see cref="ISampleSource"/>: subsets, train/test splits, shuffles and concatenations that copy no
/// samples (each read goes to the underlying source), and <see cref="ToDataset"/> to hold them all in memory.
/// <see cref="Dataset"/>'s own <see cref="Dataset.Subset"/> and <see cref="Dataset.Split"/> (which copy, and deduplicate
/// before splitting) take precedence on a variable typed as <see cref="Dataset"/>.
/// </summary>
public static class SampleSourceExtensions
{
    /// <summary>The samples at <paramref name="indices"/>, in that order (an index may repeat).</summary>
    public static ISampleSource Subset(this ISampleSource source, ReadOnlySpan<int> indices)
    {
        ArgumentNullException.ThrowIfNull(source);
        foreach (int i in indices)
        {
            if ((uint)i >= (uint)source.Count)
            {
                throw new ArgumentOutOfRangeException(nameof(indices), $"Index {i} is outside the source's {source.Count} samples.");
            }
        }

        return new SubsetSource(source, indices.ToArray());
    }

    /// <summary>The samples in an order shuffled by <paramref name="seed"/>.</summary>
    public static ISampleSource Shuffle(this ISampleSource source, int seed = 0)
    {
        int[] order = [.. Enumerable.Range(0, source.Count)];
        new Random(seed).Shuffle(order);
        return new SubsetSource(source, order);
    }

    /// <summary>
    /// Shuffles the samples by <paramref name="seed"/> and splits them into a training and a test view, as
    /// <see cref="Dataset.Split"/> does with <c>removeDuplicates: false</c> (the same samples in the same order).
    /// </summary>
    /// <param name="source">The samples.</param>
    /// <param name="trainFraction">Share of samples for the training view, e.g. 0.8.</param>
    /// <param name="seed">Shuffle seed, for reproducible splits.</param>
    public static (ISampleSource Train, ISampleSource Test) Split(this ISampleSource source, double trainFraction, int seed = 0)
    {
        if (trainFraction is <= 0 or >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(trainFraction), trainFraction, "Must be between 0 and 1.");
        }

        int[] order = [.. Enumerable.Range(0, source.Count)];
        new Random(seed).Shuffle(order);
        int trainCount = (int)Math.Round(source.Count * trainFraction);
        return (new SubsetSource(source, order[..trainCount]), new SubsetSource(source, order[trainCount..]));
    }

    /// <summary>The samples of <paramref name="source"/> followed by those of <paramref name="others"/> (same shapes).</summary>
    public static ISampleSource Concat(this ISampleSource source, params ISampleSource[] others) => new ConcatSource([source, .. others]);

    /// <summary>Reads every sample into memory: <see cref="Dataset.FromSource"/> with default column names.</summary>
    public static Dataset ToDataset(this ISampleSource source) => Dataset.FromSource(source);

    /// <summary><c>new DataLoader(source, batchSize, shuffle, dropLast, device, seed)</c>, with the constructor's defaults.</summary>
    public static DataLoader Batches(this ISampleSource source, int batchSize = 32, bool shuffle = false, bool dropLast = false,
        Device? device = null, int? seed = null) =>
        new(source, batchSize, shuffle, dropLast, device, seed);

    /// <summary>The number of values in a sample of <paramref name="shape"/>.</summary>
    internal static int Size(IReadOnlyList<int> shape)
    {
        int size = 1;
        foreach (int d in shape)
        {
            size *= d;
        }

        return size;
    }

    private sealed class SubsetSource(ISampleSource source, int[] indices) : ISampleSource
    {
        public int Count => indices.Length;

        public IReadOnlyList<int> FeatureShape => source.FeatureShape;

        public IReadOnlyList<int> TargetShape => source.TargetShape;

        public void Read(int index, Span<float> features, Span<float> targets) => source.Read(indices[index], features, targets);
    }

    private sealed class ConcatSource : ISampleSource
    {
        private readonly ISampleSource[] _parts;
        private readonly int[] _starts;

        public ConcatSource(ISampleSource[] parts)
        {
            foreach (var part in parts)
            {
                ArgumentNullException.ThrowIfNull(part);
                if (!part.FeatureShape.SequenceEqual(parts[0].FeatureShape) || !part.TargetShape.SequenceEqual(parts[0].TargetShape))
                {
                    throw new ArgumentException($"Sources of different shapes cannot be concatenated: [{string.Join(", ", parts[0].FeatureShape)}] -> [{string.Join(", ", parts[0].TargetShape)}] "
                        + $"and [{string.Join(", ", part.FeatureShape)}] -> [{string.Join(", ", part.TargetShape)}].");
                }
            }

            _parts = parts;
            _starts = new int[parts.Length + 1];
            for (int i = 0; i < parts.Length; i++)
            {
                _starts[i + 1] = checked(_starts[i] + parts[i].Count);
            }
        }

        public int Count => _starts[^1];

        public IReadOnlyList<int> FeatureShape => _parts[0].FeatureShape;

        public IReadOnlyList<int> TargetShape => _parts[0].TargetShape;

        public void Read(int index, Span<float> features, Span<float> targets)
        {
            if ((uint)index >= (uint)Count)
            {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            int part = Array.BinarySearch(_starts, index);
            part = part >= 0 ? part : ~part - 1;
            while (_starts[part + 1] == index)
            {
                part++;                                                                 // past empty parts
            }

            _parts[part].Read(index - _starts[part], features, targets);
        }
    }
}
