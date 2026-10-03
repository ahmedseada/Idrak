// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Data;

/// <summary>
/// Samples with random access, one at a time: what a <see cref="DataLoader"/> batches. <see cref="Dataset"/> is one
/// (every sample in memory); <see cref="CsvSource"/>, <see cref="ImageFolderSource"/>, <see cref="TokenFileSource"/> and
/// <see cref="NpySource"/> read files lazily; the views of <see cref="SampleSourceExtensions"/> (subsets, splits,
/// shuffles, concatenation) copy nothing. Write your own for any other storage.
/// </summary>
/// <remarks>
/// A <see cref="DataLoader"/> calls <see cref="Read"/> from one thread at a time, though not always the thread that
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
/// network stream, generated data). <see cref="DataLoader(ISampleStream, int, int, bool, Device?, int?)"/> batches it,
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
/// <see cref="DataLoader.Transforms"/>. The random numbers it is given are seeded from the loader's seed, the epoch and
/// the sample (its index in a source, its position in a stream), so a run repeats exactly and every epoch differs.
/// </summary>
public interface ISampleTransform
{
    /// <summary>Changes one sample in place; <paramref name="featureShape"/> is the source's <see cref="ISampleSource.FeatureShape"/>.</summary>
    void Apply(Span<float> features, Span<float> targets, IReadOnlyList<int> featureShape, Random random);
}

/// <summary>
/// Mini-batches for <see cref="Training.Trainer.Fit"/>: each enumeration is one epoch. <see cref="DataLoader"/> is
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
/// <item><c>csv</c>: <see cref="CsvSource"/>; options <c>target</c> (required, comma-separated), <c>ignore</c>,
/// <c>delimiter</c>, <c>header</c> (true or false).</item>
/// <item><c>images</c>: <see cref="ImageFolderSource"/> over class folders; options <c>channels</c>, <c>height</c>,
/// <c>width</c> (each defaults to the first image's).</item>
/// <item><c>tokens</c>: <see cref="TokenFileSource"/>; options <c>length</c> (required), <c>stride</c>, <c>type</c>
/// (uint16, int32 or uint32).</item>
/// <item><c>npy</c>: <see cref="NpySource"/>; options <c>targets</c> (a second .npy file), <c>classes</c>.</item>
/// </list>
/// Add your own with <see cref="Register"/> (Idrak.Datasets offers <c>TableSamples.Factory</c> for JSON Lines, Parquet
/// and CSV columns).
/// </summary>
public static class SampleSources
{
    private static readonly Dictionary<string, SampleSourceFactory> Registry = new(StringComparer.OrdinalIgnoreCase)
    {
        ["csv"] = OpenCsv,
        ["images"] = OpenImages,
        ["tokens"] = OpenTokens,
        ["npy"] = OpenNpy,
    };

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

    private static CsvSource OpenCsv(string path, IReadOnlyDictionary<string, string> options)
    {
        Known(options, "csv", "target", "ignore", "delimiter", "header");
        string target = options.GetValueOrDefault("target") ?? throw new ArgumentException("The csv source needs the option 'target' (the target columns, comma-separated).");
        return CsvSource.Open(path, new CsvOptions
        {
            TargetColumns = List(target),
            IgnoreColumns = options.TryGetValue("ignore", out var ignore) ? List(ignore) : [],
            Delimiter = options.TryGetValue("delimiter", out var d) ? (d == "\\t" ? '\t' : d.Length == 1 ? d[0] : throw new ArgumentException($"delimiter '{d}': one character.")) : ',',
            HasHeader = !options.TryGetValue("header", out var header) || bool.Parse(header),
        });
    }

    private static ImageFolderSource OpenImages(string path, IReadOnlyDictionary<string, string> options)
    {
        Known(options, "images", "channels", "height", "width");
        int? Number(string name) => options.TryGetValue(name, out var v) ? int.Parse(v, CultureInfo.InvariantCulture) : null;
        return ImageFolderSource.Open(path, Number("channels"), Number("height"), Number("width"));
    }

    private static TokenFileSource OpenTokens(string path, IReadOnlyDictionary<string, string> options)
    {
        Known(options, "tokens", "length", "stride", "type");
        int length = int.Parse(options.GetValueOrDefault("length") ?? throw new ArgumentException("The tokens source needs the option 'length' (tokens per window)."), CultureInfo.InvariantCulture);
        int? stride = options.TryGetValue("stride", out var s) ? int.Parse(s, CultureInfo.InvariantCulture) : null;
        var type = options.TryGetValue("type", out var t) ? Enum.Parse<TokenType>(t, ignoreCase: true) : TokenType.UInt16;
        return new TokenFileSource(path, length, stride, type);
    }

    private static NpySource OpenNpy(string path, IReadOnlyDictionary<string, string> options)
    {
        Known(options, "npy", "targets", "classes");
        int? classes = options.TryGetValue("classes", out var c) ? int.Parse(c, CultureInfo.InvariantCulture) : null;
        return new NpySource(path, options.GetValueOrDefault("targets"), classes);
    }

    private static void Known(IReadOnlyDictionary<string, string> options, string source, params string[] names)
    {
        foreach (string key in options.Keys)
        {
            if (!names.Contains(key, StringComparer.OrdinalIgnoreCase))
            {
                throw new ArgumentException($"The {source} source has no option '{key}' ({string.Join(", ", names)}).");
            }
        }
    }

    private static string[] List(string text) => text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
}
