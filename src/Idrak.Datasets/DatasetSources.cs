// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Datasets;

/// <summary>
/// A kind of <see cref="DatasetSpec"/> source (such as <c>hf:org/name</c> or a local folder): which sources it opens and how
/// they become rows. Register new ones with <see cref="DatasetSources.Register"/>.
/// </summary>
public interface IDatasetSource
{
    /// <summary>The source kind's name (the built-ins: hf, github, kaggle, zenodo, http, folder, file).</summary>
    string Name { get; }

    /// <summary>Options this source reads beyond the ones every source has (see <see cref="DatasetSpec"/>); others are refused.</summary>
    IReadOnlyCollection<string> Options => [];

    /// <summary>Whether this kind opens <paramref name="source"/> (the spec's source without options), usually by its prefix.</summary>
    bool CanOpen(string source);

    /// <summary>The rows of <paramref name="spec"/>, read with <paramref name="options"/> (built from the spec's options); <paramref name="downloader"/> fetches remote files.</summary>
    Dataset Open(DatasetSpec spec, ReadOptions options, Downloader? downloader);
}

/// <summary>
/// The sources a <see cref="DatasetSpec"/> opens, tried in order: Hugging Face (<c>hf:</c>), GitHub (<c>github:</c>), Kaggle
/// (<c>kaggle:</c>), Zenodo (<c>zenodo:</c>), http(s) URLs, then local folders and files. Add others with
/// <see cref="Register"/>: a new source is tried before the ones already registered.
/// </summary>
public static class DatasetSources
{
    private static readonly List<IDatasetSource> Registry =
    [
        new Prefixed("hf", "hf:", (spec, rest, read, downloader) => HuggingFace.Dataset(rest, spec.String("config"), spec.String("split") ?? "train",
            spec.String("files"), spec.String("revision") ?? "main", maxFiles: spec.Int("max_files"), options: read, downloader: downloader)),
        new Prefixed("github", "github:", GitHubSource),
        new Prefixed("kaggle", "kaggle:", (spec, rest, read, downloader) => Kaggle.Dataset(rest, spec.String("files"), options: read, downloader: downloader)),
        new Prefixed("zenodo", "zenodo:", (spec, rest, read, downloader) => Zenodo.Record(rest, spec.String("files"), options: read, downloader: downloader)),
        new Local("http", source => source.StartsWith("http://", StringComparison.OrdinalIgnoreCase) || source.StartsWith("https://", StringComparison.OrdinalIgnoreCase),
            (spec, read, downloader) => Dataset.FromUrl(spec.Source, read, downloader)),
        new Local("folder", Directory.Exists, (spec, read, _) => Dataset.FromFolder(spec.Source, spec.String("files"), read)),
        new Local("file", File.Exists, (spec, read, _) => Dataset.FromFile(spec.Source, read with { Pattern = spec.String("files") })),
    ];

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

    // github:owner/repo[@ref]: a repository's files as documents, data files when files= names them, or release assets.
    private static Dataset GitHubSource(DatasetSpec spec, string repo, ReadOptions read, Downloader? downloader)
    {
        string? reference = spec.String("ref");
        int at = repo.IndexOf('@', StringComparison.Ordinal);
        if (at > 0)
        {
            reference = repo[(at + 1)..];
            repo = repo[..at];
        }

        bool dataFiles = spec.String("files") is { } files && DataFileFormats.Find(files, includeCode: false) is not null && !spec.Flag("documents");
        return spec.String("release") is { } release
            ? GitHub.Release(repo, spec.String("asset") ?? "*", release == "latest" ? null : release, options: read, downloader: downloader)
            : dataFiles
                ? GitHub.Files(repo, spec.String("files")!, reference, options: read, downloader: downloader)
                : GitHub.Repository(repo, reference, spec.String("files"), downloader: downloader);
    }

    // A source written as "prefix:rest" (the prefix ignores case).
    private sealed class Prefixed(string name, string prefix, Func<DatasetSpec, string, ReadOptions, Downloader?, Dataset> open) : IDatasetSource
    {
        public string Name => name;

        public bool CanOpen(string source) => source.StartsWith(prefix, StringComparison.OrdinalIgnoreCase);

        public Dataset Open(DatasetSpec spec, ReadOptions options, Downloader? downloader) => open(spec, spec.Source[prefix.Length..], options, downloader);
    }

    // A source recognised by a test on the whole string (a URL, a folder, a file).
    private sealed class Local(string name, Func<string, bool> canOpen, Func<DatasetSpec, ReadOptions, Downloader?, Dataset> open) : IDatasetSource
    {
        public string Name => name;

        public bool CanOpen(string source) => canOpen(source);

        public Dataset Open(DatasetSpec spec, ReadOptions options, Downloader? downloader) => open(spec, options, downloader);
    }
}
