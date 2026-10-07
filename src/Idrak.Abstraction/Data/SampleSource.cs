// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Abstraction.Data;

/// <summary>
/// Samples with random access, one at a time: what a <c>DataLoader</c> batches. <c>Dataset</c> is one
/// (every sample in memory); <c>CsvSource</c>, <c>ImageFolderSource</c>, <c>TokenFileSource</c> and
/// <c>NpySource</c> read files lazily; the views of <c>SampleSourceExtensions</c> (subsets, splits,
/// shuffles, concatenation) copy nothing. Write your own for any other storage.
/// </summary>
/// <remarks>
/// A <c>DataLoader</c> calls <see cref="Read"/> from one thread at a time, though not always the thread that
/// created it (the next batch is gathered on a worker thread while the model trains). A source read by several loaders
/// at the same time must allow concurrent reads; the built-in sources and views do.
/// </remarks>
public interface ISampleSource
{
    /// <summary>The number of samples.</summary>
    int Count { get; }

    /// <summary>The shape of one sample's features: [n] for tabular data, [C, H, W] for images, [T] for token windows.</summary>
    IReadOnlyList<int> FeatureShape { get; }

    /// <summary>The shape of one sample's targets: [n] for values or one-hot classes, [T] for next tokens.</summary>
    IReadOnlyList<int> TargetShape { get; }

    /// <summary>
    /// Writes sample <paramref name="index"/> (0 to <see cref="Count"/> - 1) into <paramref name="features"/> and
    /// <paramref name="targets"/>, row-major, sized as the products of <see cref="FeatureShape"/> and <see cref="TargetShape"/>.
    /// </summary>
    void Read(int index, Span<float> features, Span<float> targets);
}

/// <summary>
/// Samples in order, for data whose count is unknown or too large to index (a file read front to back, rows from a
/// network stream, generated data). <c>DataLoader</c> batches it,
/// shuffling within a buffer.
/// </summary>
public interface ISampleStream
{
    /// <summary>The shape of one sample's features.</summary>
    IReadOnlyList<int> FeatureShape { get; }

    /// <summary>The shape of one sample's targets.</summary>
    IReadOnlyList<int> TargetShape { get; }

    /// <summary>Starts a pass over the samples from the first; every epoch opens one and disposes it at the end.</summary>
    ISampleReader Open();
}

/// <summary>One pass over an <see cref="ISampleStream"/>.</summary>
public interface ISampleReader : IDisposable
{
    /// <summary>Writes the next sample into <paramref name="features"/> and <paramref name="targets"/>; false at the end.</summary>
    bool Read(Span<float> features, Span<float> targets);
}

/// <summary>
/// A change made to each sample as it is batched (flips, shifts, rotations, noise, masking), set on
/// <c>DataLoader.Transforms</c>. The random numbers it is given are seeded from the loader's seed, the epoch and
/// the sample (its index in a source, its position in a stream), so a run repeats exactly and every epoch differs.
/// </summary>
public interface ISampleTransform
{
    /// <summary>Changes one sample in place; <paramref name="featureShape"/> is the source's <see cref="ISampleSource.FeatureShape"/>.</summary>
    void Apply(Span<float> features, Span<float> targets, IReadOnlyList<int> featureShape, Random random);
}

/// <summary>Opens a sample source from a path and named options (see <c>SampleSources</c> in Idrak).</summary>
/// <param name="path">The file or folder.</param>
/// <param name="options">Options by name (names ignore case); a factory rejects names it does not know.</param>
public delegate ISampleSource SampleSourceFactory(string path, IReadOnlyDictionary<string, string> options);
