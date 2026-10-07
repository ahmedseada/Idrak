// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Data.Abstractions;

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
    static SampleSources() => LibrarySampleSources.RegisterAll();   // the built-in sources, on first use

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
