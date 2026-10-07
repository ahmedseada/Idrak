// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Data;

/// <summary>
/// The rows a <see cref="IDatasetSource"/> opened (JSON objects, column name → value), read lazily: each enumeration reads
/// the source again. <c>Dataset</c> (Idrak.Datasets) implements it and wraps any other implementation, so the filters,
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
/// The sources a <see cref="DatasetSpec"/> opens, tried in order. Idrak.Datasets registers Hugging Face (<c>hf:</c>), GitHub
/// (<c>github:</c>), Kaggle (<c>kaggle:</c>), Zenodo (<c>zenodo:</c>), http(s) URLs, then local folders and files. Add others
/// with <see cref="Register"/>: a new source is tried before the ones already registered.
/// </summary>
public static class DatasetSources
{
    private static readonly List<IDatasetSource> Registry = [];

    // The built-ins of the first-party assemblies (Idrak.Datasets) are registered before the first use.
    static DatasetSources() => LibraryDefaults.Ensure();

    /// <summary>
    /// Registers <paramref name="source"/>. One with the name of a registered source replaces it in its place; a new one is
    /// tried first, before the sources already registered.
    /// </summary>
    public static void Register(IDatasetSource source)
    {
        ArgumentNullException.ThrowIfNull(source);
        lock (Registry)
        {
            int index = Registry.FindIndex(s => string.Equals(s.Name, source.Name, StringComparison.OrdinalIgnoreCase));
            if (index >= 0)
            {
                Registry[index] = source;
            }
            else
            {
                Registry.Insert(0, source);
            }
        }
    }

    /// <summary>Removes the source registered as <paramref name="name"/> (ignoring case); returns whether there was one.</summary>
    public static bool Unregister(string name)
    {
        lock (Registry)
        {
            return Registry.RemoveAll(s => string.Equals(s.Name, name, StringComparison.OrdinalIgnoreCase)) > 0;
        }
    }

    /// <summary>The registered source names, in the order they are tried.</summary>
    public static IReadOnlyCollection<string> Names
    {
        get
        {
            lock (Registry)
            {
                return [.. Registry.Select(s => s.Name)];
            }
        }
    }

    /// <summary>The source registered as <paramref name="name"/> (ignoring case).</summary>
    public static IDatasetSource Get(string name)
    {
        lock (Registry)
        {
            return Registry.Find(s => string.Equals(s.Name, name, StringComparison.OrdinalIgnoreCase))
                ?? throw new NotSupportedException($"No dataset source '{name}' is registered ({string.Join(", ", Registry.Select(s => s.Name))}); add it with DatasetSources.Register.");
        }
    }

    /// <summary>The first registered source that opens <paramref name="source"/>, or null.</summary>
    public static IDatasetSource? Find(string source)
    {
        IDatasetSource[] sources;
        lock (Registry)
        {
            sources = [.. Registry];
        }

        return Array.Find(sources, s => s.CanOpen(source));
    }
}
