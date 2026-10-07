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

/// <summary>
/// Mini-batches for <c>Trainer.Fit</c>: each enumeration is one epoch. <c>DataLoader</c> is
/// one; write your own to batch in another way (variable lengths, sampling by class, batches made on the device). The
/// trainer disposes every batch after its step. The counts are for telemetry and progress; leave them null when unknown.
/// </summary>
public interface IBatchSource : IEnumerable<Batch>
{
    /// <summary>Batches per epoch, or null when unknown.</summary>
    int? BatchCount => null;

    /// <summary>Samples per epoch, or null when unknown.</summary>
    int? SampleCount => null;

    /// <summary>Samples per batch (the last may be smaller), or null when it varies.</summary>
    int? BatchSize => null;
}

/// <summary>Opens a sample source from a path and named options (see <see cref="SampleSources"/>).</summary>
/// <param name="path">The file or folder.</param>
/// <param name="options">Options by name (names ignore case); a factory rejects names it does not know.</param>
public delegate ISampleSource SampleSourceFactory(string path, IReadOnlyDictionary<string, string> options);

/// <summary>
/// Sample sources by name (ignoring case), for tools and configuration files that name their data. The built-ins:
/// <list type="bullet">
/// <item><c>csv</c>: <c>CsvSource</c>; options <c>target</c> (required, comma-separated), <c>ignore</c>,
/// <c>delimiter</c>, <c>header</c> (true or false).</item>
/// <item><c>images</c>: <c>ImageFolderSource</c> over class folders; options <c>channels</c>, <c>height</c>,
/// <c>width</c> (each defaults to the first image's).</item>
/// <item><c>tokens</c>: <c>TokenFileSource</c>; options <c>length</c> (required), <c>stride</c>, <c>type</c>
/// (uint16, int32 or uint32).</item>
/// <item><c>npy</c>: <c>NpySource</c>; options <c>targets</c> (a second .npy file), <c>classes</c>.</item>
/// </list>
/// Add your own with <see cref="Register"/> (Idrak.Data offers <c>TableSamples.Factory</c> for JSON Lines, Parquet
/// and CSV columns).
/// </summary>
public static class SampleSources
{
    private static readonly Dictionary<string, SampleSourceFactory> Registry = new(StringComparer.OrdinalIgnoreCase);

    // Idrak's built-in sources (csv, images, tokens, npy) are registered before the first use.
    static SampleSources() => LibraryDefaults.Ensure();

    /// <summary>Registers (or replaces) the source <paramref name="name"/> (names ignore case).</summary>
    public static void Register(string name, SampleSourceFactory factory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(factory);
        lock (Registry)
        {
            Registry[name] = factory;
        }
    }

    /// <summary>Removes the source <paramref name="name"/>; returns whether it was registered.</summary>
    public static bool Unregister(string name)
    {
        lock (Registry)
        {
            return Registry.Remove(name);
        }
    }

    /// <summary>The registered source names.</summary>
    public static IReadOnlyCollection<string> Names
    {
        get
        {
            lock (Registry)
            {
                return [.. Registry.Keys];
            }
        }
    }

    /// <summary>Whether a source is registered as <paramref name="name"/>.</summary>
    public static bool Contains(string name)
    {
        lock (Registry)
        {
            return Registry.ContainsKey(name);
        }
    }

    /// <summary>The factory registered as <paramref name="name"/> (any case).</summary>
    public static SampleSourceFactory Get(string name)
    {
        lock (Registry)
        {
            return Registry.TryGetValue(name, out var factory) ? factory
                : throw new NotSupportedException($"No sample source '{name}' is registered ({string.Join(", ", Registry.Keys)}); add it with SampleSources.Register.");
        }
    }

    /// <summary>Opens <paramref name="path"/> with the source registered as <paramref name="name"/>.</summary>
    public static ISampleSource Open(string name, string path, IReadOnlyDictionary<string, string>? options = null) =>
        Get(name)(path, options is null ? new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            : new Dictionary<string, string>(options, StringComparer.OrdinalIgnoreCase));
}
