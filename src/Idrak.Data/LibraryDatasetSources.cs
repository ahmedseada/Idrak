// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Data;

/// <summary>
/// The dataset sources this assembly ships, registered in <see cref="DatasetSources"/> (Idrak.Abstraction) by
/// <see cref="LibraryRegistrations"/>, tried in this order: hf, github, kaggle, zenodo, http, folder, file.
/// </summary>
internal static class LibraryDatasetSources
{
    public static void RegisterAll()
    {
        IDatasetSource[] sources =
        [
            new Prefixed("hf", "hf:", (spec, rest, read, downloader) => HuggingFace.Dataset(rest, spec.Get("config"), spec.Get("split") ?? "train",
                spec.Get("files"), spec.Get("revision") ?? "main", maxFiles: spec.GetInt32("max_files"), options: read, downloader: downloader)),
            new Prefixed("github", "github:", GitHubSource),
            new Prefixed("kaggle", "kaggle:", (spec, rest, read, downloader) => Kaggle.Dataset(rest, spec.Get("files"), options: read, downloader: downloader)),
            new Prefixed("zenodo", "zenodo:", (spec, rest, read, downloader) => Zenodo.Record(rest, spec.Get("files"), options: read, downloader: downloader)),
            new Local("http", source => source.StartsWith("http://", StringComparison.OrdinalIgnoreCase) || source.StartsWith("https://", StringComparison.OrdinalIgnoreCase),
                (spec, read, downloader) => DatasetRows.FromUrl(spec.Source, read, downloader)),
            new Local("folder", Directory.Exists, (spec, read, _) => DatasetRows.FromFolder(spec.Source, spec.Get("files"), read)),
            new Local("file", File.Exists, (spec, read, _) => DatasetRows.FromFile(spec.Source, read with { Pattern = spec.Get("files") })),
        ];

        // A new source is tried before the ones registered earlier, so register from the last to the first.
        for (int i = sources.Length - 1; i >= 0; i--)
        {
            DatasetSources.Register(sources[i]);
        }
    }

    // github:owner/repo[@ref]: a repository's files as documents, data files when files= names them, or release assets.
    private static DatasetRows GitHubSource(DatasetSpec spec, string repo, ReadOptions read, IDownloader? downloader)
    {
        string? reference = spec.Get("ref");
        int at = repo.IndexOf('@', StringComparison.Ordinal);
        if (at > 0)
        {
            reference = repo[(at + 1)..];
            repo = repo[..at];
        }

        bool dataFiles = spec.Get("files") is { } files && DataFileFormats.Find(files, includeCode: false) is not null && !spec.Flag("documents");
        return spec.Get("release") is { } release
            ? GitHub.Release(repo, spec.Get("asset") ?? "*", release == "latest" ? null : release, options: read, downloader: downloader)
            : dataFiles
                ? GitHub.Files(repo, spec.Get("files")!, reference, options: read, downloader: downloader)
                : GitHub.Repository(repo, reference, spec.Get("files"), downloader: downloader);
    }

    // A source written as "prefix:rest" (the prefix ignores case).
    private sealed class Prefixed(string name, string prefix, Func<DatasetSpec, string, ReadOptions, IDownloader?, DatasetRows> open) : IDatasetSource
    {
        public string Name => name;

        public bool CanOpen(string source) => source.StartsWith(prefix, StringComparison.OrdinalIgnoreCase);

        public IDatasetRows Open(DatasetSpec spec, ReadOptions options, IDownloader? downloader) => open(spec, spec.Source[prefix.Length..], options, downloader);
    }

    // A source recognised by a test on the whole string (a URL, a folder, a file).
    private sealed class Local(string name, Func<string, bool> canOpen, Func<DatasetSpec, ReadOptions, IDownloader?, DatasetRows> open) : IDatasetSource
    {
        public string Name => name;

        public bool CanOpen(string source) => canOpen(source);

        public IDatasetRows Open(DatasetSpec spec, ReadOptions options, IDownloader? downloader) => open(spec, options, downloader);
    }
}
