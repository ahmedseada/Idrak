// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;

namespace Idrak.Data.Abstractions;

/// <summary>
/// The rows a <see cref="IDatasetSource"/> opened (JSON objects, column name → value), read lazily: each enumeration reads
/// the source again. <c>DatasetRows</c> (Idrak.Data) implements it and wraps any other implementation, so the filters,
/// mixes and splits apply to rows from every source.
/// </summary>
public interface IDatasetRows : IEnumerable<JsonObject>
{
    /// <summary>A name for messages and reports (the source the rows are read from).</summary>
    string Name { get; }

    /// <summary>
    /// The local files the rows are read from, downloading remote ones into the cache first (without reading rows); throws
    /// <see cref="InvalidOperationException"/> for rows that are not read from files of their own (derived from other rows).
    /// </summary>
    IReadOnlyList<string> Download();
}

/// <summary>
/// A kind of <see cref="DatasetSpec"/> source (such as <c>hf:org/name</c> or a local folder): which sources it opens and how
/// they become rows. Register new ones with <see cref="DatasetSources.Register"/>.
/// </summary>
public interface IDatasetSource
{
    /// <summary>The source kind's name (the built-ins: hf, github, kaggle, zenodo, http, folder, file).</summary>
    string Name { get; }

    /// <summary>Options this source reads beyond the ones every source has (<see cref="DatasetSpec.CommonOptions"/>); others are refused.</summary>
    IReadOnlyCollection<string> Options => [];

    /// <summary>Whether this kind opens <paramref name="source"/> (the spec's source without options), usually by its prefix.</summary>
    bool CanOpen(string source);

    /// <summary>
    /// The rows of <paramref name="spec"/>, read with <paramref name="options"/> (built from the spec's options);
    /// <paramref name="downloader"/> fetches remote files (null: the library's shared one).
    /// </summary>
    IDatasetRows Open(DatasetSpec spec, ReadOptions options, IDownloader? downloader);
}

/// <summary>
/// The sources a <see cref="DatasetSpec"/> opens, tried in order. Idrak.Data registers Hugging Face (<c>hf:</c>), GitHub
/// (<c>github:</c>), Kaggle (<c>kaggle:</c>), Zenodo (<c>zenodo:</c>), http(s) URLs, then local folders and files. Add others
/// with <see cref="Register"/>: a new source is tried before the ones already registered.
/// </summary>
public static class DatasetSources
{
    private static readonly SlotTable<string, IDatasetSource> Registry = new(nameof(DatasetSources), (slot, app, library) => new GuardedSource(slot, app, library),
        StringComparer.OrdinalIgnoreCase, newestFirst: true);

    static DatasetSources() => Overrides.AsLibraryDefaults(LibraryDatasetSources.RegisterAll);   // the built-in sources, on first use

    /// <summary>
    /// Registers <paramref name="source"/>. One with the name of a registered source takes its place (a built-in stays
    /// behind it as its fallback, see <see cref="SetPolicy"/>); a new one is tried first, before the sources already registered.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IDatasetSource source)
    {
        ArgumentNullException.ThrowIfNull(source);
        Registry.Register(source.Name, source, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's source <paramref name="name"/> (a built-in name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered source names, in the order they are tried.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The source registered as <paramref name="name"/> (ignoring case).</summary>
    public static IDatasetSource Get(string name) => Registry.Find(name)
        ?? throw new NotSupportedException($"No dataset source '{name}' is registered ({string.Join(", ", Names)}); add it with DatasetSources.Register.");

    /// <summary>The first registered source that opens <paramref name="source"/>, or null.</summary>
    public static IDatasetSource? Find(string source) => Registry.Values.FirstOrDefault(s => s.CanOpen(source));

    /// <summary>The library's source <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IDatasetSource? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the source <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>
    /// What happens when the app's source <paramref name="name"/> fails (<see cref="SlotPolicy.FallBack"/> to the library's
    /// unless set). Under <see cref="SlotPolicy.Shadow"/> only <see cref="IDatasetSource.CanOpen"/> is compared: opening
    /// downloads files, which is not done twice.
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(name, policy, shadowRate);

    private sealed class GuardedSource(Slot slot, IDatasetSource app, IDatasetSource library) : IDatasetSource
    {
        public string Name => app.Name;

        public IReadOnlyCollection<string> Options => app.Options;

        public bool CanOpen(string source) => slot.Call(() => app.CanOpen(source), () => library.CanOpen(source), Comparisons.Exact);

        public IDatasetRows Open(DatasetSpec spec, ReadOptions options, IDownloader? downloader) =>
            slot.Call(() => app.Open(spec, options, downloader), () => library.Open(spec, options, downloader), effects: true);
    }
}
