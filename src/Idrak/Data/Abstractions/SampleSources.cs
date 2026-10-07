// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Runtime.CompilerServices;

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
    // Opening a source falls back when it fails; under Shadow both open the data and agree when their sizes and shapes do.
    private static readonly SlotTable<string, SampleSourceFactory> Registry = new(nameof(SampleSources),
        (slot, app, library) => (path, options) => slot.Call(() => app(path, options), () => library(path, options), (a, b) => Comparisons.Exact(Shape(a), Shape(b))),
        StringComparer.OrdinalIgnoreCase);

    // Idrak's built-in sources (csv, images, tokens, npy) are registered before the first use.
    static SampleSources() => Overrides.AsLibraryDefaults(LibrarySampleSources.RegisterAll);   // the built-in sources, on first use

    /// <summary>
    /// Registers the source <paramref name="name"/> (names ignore case); under a built-in name it overrides the library's,
    /// which stays behind it (see <see cref="SetPolicy"/>) until <see cref="Unregister"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(string name, SampleSourceFactory factory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(factory);
        Registry.Register(name, factory, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's source <paramref name="name"/> (a built-in name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered source names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>Whether a source is registered as <paramref name="name"/>.</summary>
    public static bool Contains(string name) => Registry.Contains(name);

    /// <summary>The factory registered as <paramref name="name"/> (any case).</summary>
    public static SampleSourceFactory Get(string name) =>
        Registry.TryGet(name, out var factory) ? factory
            : throw new NotSupportedException($"No sample source '{name}' is registered ({string.Join(", ", Registry.Keys)}); add it with SampleSources.Register.");

    /// <summary>The library's factory <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static SampleSourceFactory? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the source <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>What happens when the app's source <paramref name="name"/> fails to open (<see cref="SlotPolicy.Throw"/> unless set: the error reaches the caller; <see cref="SlotPolicy.FallBack"/> retries on the library's).</summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(name, policy, shadowRate);

    private static string Shape(ISampleSource source) =>
        $"{source.Count} samples of [{string.Join(", ", source.FeatureShape)}] -> [{string.Join(", ", source.TargetShape)}]";

    /// <summary>Opens <paramref name="path"/> with the source registered as <paramref name="name"/>.</summary>
    public static ISampleSource Open(string name, string path, IReadOnlyDictionary<string, string>? options = null) =>
        Get(name)(path, options is null ? new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            : new Dictionary<string, string>(options, StringComparer.OrdinalIgnoreCase));
}
