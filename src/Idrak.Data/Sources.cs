// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Idrak.Data.Abstractions;

namespace Idrak.Data;

/// <summary>A file in a remote repository.</summary>
/// <param name="Path">Its path in the repository.</param>
/// <param name="Size">Its size in bytes (0 when unknown).</param>
public sealed record RepoFile(string Path, long Size);

// A dataset over files a source resolves and downloads once (on first use), then reads from the cache.
internal static class RemoteFiles
{
    public static DatasetRows Over(string name, Func<IReadOnlyList<string>> resolve, ReadOptions? options, Func<JsonObject, JsonObject>? post = null,
        IDownloader? downloader = null)
    {
        var files = new Lazy<IReadOnlyList<string>>(() =>
        {
            downloader?.Log?.Invoke($"{name}: resolving files");
            var list = resolve();
            downloader?.Log?.Invoke($"{name}: {list.Count} file{(list.Count == 1 ? "" : "s")} ready ({Downloader.Size(list.Sum(f => new FileInfo(f).Length))})");
            return list;
        }, LazyThreadSafetyMode.ExecutionAndPublication);
        return new DatasetRows(() =>
        {
            var rows = files.Value.SelectMany((f, i) =>
            {
                if (files.Value.Count > 1)
                {
                    downloader?.Log?.Invoke($"{name}: reading {Path.GetFileName(f)} ({i + 1} of {files.Value.Count})");
                }

                return DataFiles.Read(f, options ?? ReadOptions.Default);
            });
            return post is null ? rows : rows.Select(post);
        }, name) { Files = () => files.Value };
    }

    public static Dictionary<string, string> Bearer(string? token, Dictionary<string, string>? extra = null)
    {
        var headers = extra ?? [];
        if (!string.IsNullOrEmpty(token))
        {
            headers["Authorization"] = $"Bearer {token}";
        }

        return headers;
    }

    public static string EscapePath(string path) => string.Join('/', path.Split('/').Select(Uri.EscapeDataString));

    // Files whose names mark them as belonging to a split: "train", "data/train-00000-of-00002.parquet", "test.jsonl" …
    private static readonly Dictionary<string, string[]> SplitWords = new(StringComparer.OrdinalIgnoreCase)
    {
        ["train"] = ["train", "training"],
        ["validation"] = ["validation", "valid", "val", "dev"],
        ["test"] = ["test", "testing", "eval", "evaluation"],
    };

    public static IReadOnlyList<string> ForSplit(IReadOnlyList<string> paths, string split)
    {
        static string[] Words(string path) => Regex.Split(path.ToLowerInvariant(), "[^a-z]+").Where(w => w.Length > 0).ToArray();
        string[] Keywords(string s) => SplitWords.TryGetValue(s, out var words) ? words : [s.ToLowerInvariant()];
        var any = SplitWords.Values.SelectMany(w => w).ToHashSet();
        bool structured = paths.Any(p => Words(p).Any(any.Contains));
        if (!structured)
        {
            return split.Equals("train", StringComparison.OrdinalIgnoreCase) ? paths : [];
        }

        if (!SplitWords.ContainsKey(split))
        {
            // A split with its own name ("train_sft", "test_gen"): the name between separators in the path.
            var named = new Regex($"(^|[^a-z0-9]){Regex.Escape(split.ToLowerInvariant())}([^a-z0-9]|$)");
            return [.. paths.Where(p => named.IsMatch(p.ToLowerInvariant()))];
        }

        var wanted = Keywords(split);
        return [.. paths.Where(p => Words(p).Any(wanted.Contains))];
    }

    // Among data files, the ones in the preferred format (so a dataset shipped as both Parquet and JSON is read once).
    public static IReadOnlyList<string> BestFormat(IEnumerable<string> paths)
    {
        static int Rank(string p) => DataFiles.IsArchive(p) ? 6 : DataFiles.FormatOf(p, includeCode: false) switch
        {
            DataFormat.Parquet => 0,
            DataFormat.JsonLines => 1,
            DataFormat.Json => 2,
            DataFormat.Csv => 3,
            DataFormat.Tsv => 4,
            DataFormat.Text => 5,
            _ => 99,
        };
        var ranked = paths.Select(p => (Path: p, Rank: Rank(p))).Where(p => p.Rank < 99).ToList();
        if (ranked.Count == 0)
        {
            return [];
        }

        int best = ranked.Min(p => p.Rank);
        return [.. ranked.Where(p => p.Rank == best).Select(p => p.Path).Order(StringComparer.Ordinal)];
    }

    private static readonly HashSet<string> NotData = new(["dataset_info.json", "dataset_infos.json", "dataset_dict.json", "state.json", "config.json",
        "package.json", "package-lock.json", "tsconfig.json", "readme.md", "license", "license.md", "license.txt", "requirements.txt", "citation.txt"],
        StringComparer.OrdinalIgnoreCase);

    public static bool IsDataFile(string path) =>
        !NotData.Contains(System.IO.Path.GetFileName(path)) && !path.StartsWith('.') && !path.Contains("/.", StringComparison.Ordinal)
        && (DataFiles.IsArchive(path) || DataFiles.FormatOf(path, includeCode: false) is { } f && (f != DataFormat.Text || !path.EndsWith(".md", StringComparison.OrdinalIgnoreCase)));
}

/// <summary>
/// Datasets (and other files) from the Hugging Face Hub. Private and gated repositories need a token: pass one, or set
/// <c>HF_TOKEN</c>, or sign in with <c>huggingface-cli login</c> (the token file is read). <c>HF_ENDPOINT</c> selects a
/// mirror. Files are downloaded once into the <see cref="Downloader"/> cache.
/// </summary>
public static class HuggingFace
{
    /// <summary><c>HF_ENDPOINT</c>, or https://huggingface.co.</summary>
    public static string Endpoint => (Environment.GetEnvironmentVariable("HF_ENDPOINT") ?? "https://huggingface.co").TrimEnd('/');

    /// <summary>The token to use: <paramref name="token"/>, else <c>HF_TOKEN</c>, else the token saved by huggingface-cli login.</summary>
    public static string? Token(string? token = null)
    {
        if (!string.IsNullOrWhiteSpace(token))
        {
            return token.Trim();
        }

        if ((Environment.GetEnvironmentVariable("HF_TOKEN") ?? Environment.GetEnvironmentVariable("HUGGING_FACE_HUB_TOKEN")) is { Length: > 0 } variable)
        {
            return variable.Trim();
        }

        string home = Environment.GetEnvironmentVariable("HF_HOME")
                      ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "huggingface");
        string file = Environment.GetEnvironmentVariable("HF_TOKEN_PATH") ?? Path.Combine(home, "token");
        return File.Exists(file) ? File.ReadAllText(file).Trim() : null;
    }

    /// <summary>
    /// A split of a Hub dataset (<paramref name="repo"/> such as "HuggingFaceH4/ultrachat_200k"). The repository's own data
    /// files are used (Parquet, JSON Lines, JSON, CSV, text; those of <paramref name="config"/> when it names a folder),
    /// chosen by <paramref name="split"/> the way the datasets library does ("train-00000-of-00004.parquet", "test.jsonl" …);
    /// otherwise the Parquet copy the Hub makes of every dataset. <paramref name="files"/> (a glob such as
    /// "data/train-*.parquet") picks files directly; <paramref name="maxFiles"/> reads only the first files of a large dataset.
    /// </summary>
    public static DatasetRows Dataset(string repo, string? config = null, string split = "train", string? files = null, string revision = "main",
        string? token = null, int? maxFiles = null, ReadOptions? options = null, IDownloader? downloader = null)
    {
        var d = downloader ?? Downloader.Shared;
        string name = $"hf:{repo}{(config is null ? "" : "/" + config)}[{split}]";
        return RemoteFiles.Over(name, () => ResolveAsync(repo, config, split, files, revision, token, maxFiles, d).GetAwaiter().GetResult(), options, downloader: d);
    }

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, string> Commits = new(StringComparer.Ordinal);

    /// <summary>
    /// The commit a branch or tag of a repository points at (a commit hash is returned as it is). Files are cached per
    /// commit, so an updated dataset is downloaded again rather than read stale from the cache.
    /// </summary>
    public static async Task<string> ResolveRevisionAsync(string repo, string kind = "datasets", string revision = "main", string? token = null,
        IDownloader? downloader = null, CancellationToken cancellationToken = default)
    {
        if (revision.Length == 40 && revision.All(char.IsAsciiHexDigitLower))
        {
            return revision;
        }

        string key = $"{Endpoint}|{kind}|{repo}|{revision}";
        if (Commits.TryGetValue(key, out var known))
        {
            return known;
        }

        var d = downloader ?? Downloader.Shared;
        var json = JsonNode.Parse(await d.GetStringAsync($"{Endpoint}/api/{kind}/{repo}/revision/{Uri.EscapeDataString(revision)}", RemoteFiles.Bearer(Token(token)),
            cancellationToken).ConfigureAwait(false));
        string sha = (string?)json?["sha"] ?? throw new InvalidDataException($"hf:{repo}: no commit for revision '{revision}'.");
        d.Log?.Invoke($"hf:{repo}: {revision} is at commit {sha[..Math.Min(7, sha.Length)]}");
        return Commits[key] = sha;
    }

    /// <summary>The files of a repository (<paramref name="kind"/>: "datasets" or "models").</summary>
    public static async Task<IReadOnlyList<RepoFile>> ListFilesAsync(string repo, string kind = "datasets", string revision = "main", string? token = null,
        IDownloader? downloader = null, CancellationToken cancellationToken = default)
    {
        var d = downloader ?? Downloader.Shared;
        string commit = await ResolveRevisionAsync(repo, kind, revision, token, d, cancellationToken).ConfigureAwait(false);
        string url = $"{Endpoint}/api/{kind}/{repo}/tree/{commit}?recursive=true";
        var list = new List<RepoFile>();
        foreach (var page in await d.GetPagesAsync(url, RemoteFiles.Bearer(Token(token)), cancellationToken).ConfigureAwait(false))
        {
            foreach (var item in JsonNode.Parse(page) as JsonArray ?? [])
            {
                if ((string?)item?["type"] == "file" && (string?)item["path"] is { } path)
                {
                    list.Add(new RepoFile(path, item["size"] is JsonValue v && v.TryGetValue<long>(out long size) ? size : 0));
                }
            }
        }

        return list;
    }

    /// <summary>
    /// Downloads one file of a repository (or finds it in the cache) and returns its local path:
    /// huggingface/&lt;kind&gt;/&lt;owner&gt;/&lt;name&gt;/&lt;commit&gt;/&lt;path&gt; in the cache.
    /// </summary>
    public static async Task<string> DownloadFileAsync(string repo, string path, string kind = "datasets", string revision = "main", string? token = null,
        IDownloader? downloader = null, CancellationToken cancellationToken = default)
    {
        var d = downloader ?? Downloader.Shared;
        string commit = await ResolveRevisionAsync(repo, kind, revision, token, d, cancellationToken).ConfigureAwait(false);
        string prefix = kind == "models" ? "" : kind + "/";
        string url = $"{Endpoint}/{prefix}{repo}/resolve/{commit}/{RemoteFiles.EscapePath(path)}";
        return await d.DownloadAsync(url, RemoteFiles.Bearer(Token(token)), $"huggingface/{kind}/{repo}/{commit[..Math.Min(12, commit.Length)]}/{path}",
            cancellationToken).ConfigureAwait(false);
    }

    private static async Task<IReadOnlyList<string>> ResolveAsync(string repo, string? config, string split, string? pattern, string revision, string? token,
        int? maxFiles, IDownloader downloader)
    {
        var all = (await ListFilesAsync(repo, "datasets", revision, token, downloader).ConfigureAwait(false)).Select(f => f.Path).ToList();
        IReadOnlyList<string> chosen;
        if (pattern is not null)
        {
            var glob = DataFiles.Glob(pattern);
            chosen = [.. all.Where(p => glob.IsMatch(p)).Order(StringComparer.Ordinal)];
            if (chosen.Count == 0)
            {
                throw new FileNotFoundException($"hf:{repo}: no files match '{pattern}'. Files: {string.Join(", ", all.Take(20))}{(all.Count > 20 ? " …" : "")}");
            }
        }
        else
        {
            var data = all.Where(RemoteFiles.IsDataFile).ToList();
            if (config is not null && config != "default")
            {
                data = [.. data.Where(p => p.StartsWith(config + "/", StringComparison.Ordinal) || p.Contains("/" + config + "/", StringComparison.Ordinal))];
            }

            chosen = RemoteFiles.BestFormat(RemoteFiles.ForSplit(data, split));
        }

        downloader.Log?.Invoke($"hf:{repo}: {all.Count} files in the repository; {chosen.Count} data file{(chosen.Count == 1 ? "" : "s")} for split '{split}'"
                               + (maxFiles is { } m && m < chosen.Count ? $", reading the first {m}" : ""));
        if (chosen.Count == 0)
        {
            downloader.Log?.Invoke($"hf:{repo}: no plain data files for '{split}'; using the Hub's Parquet copy");
            return await ConvertedParquetAsync(repo, config, split, token, maxFiles, downloader).ConfigureAwait(false);
        }

        var paths = new List<string>();
        foreach (var path in chosen.Take(maxFiles ?? int.MaxValue))
        {
            paths.Add(await DownloadFileAsync(repo, path, "datasets", revision, token, downloader).ConfigureAwait(false));
        }

        return paths;
    }

    // The Hub's Parquet copy of a dataset (refs/convert/parquet), for datasets without plain data files.
    private static async Task<IReadOnlyList<string>> ConvertedParquetAsync(string repo, string? config, string split, string? token, int? maxFiles, IDownloader downloader)
    {
        var headers = RemoteFiles.Bearer(Token(token));
        var index = JsonNode.Parse(await downloader.GetStringAsync($"{Endpoint}/api/datasets/{repo}/parquet", headers).ConfigureAwait(false)) as JsonObject
                    ?? throw new InvalidDataException($"hf:{repo}: unexpected answer listing its Parquet files.");
        string chosenConfig = config ?? (index.ContainsKey("default") ? "default" : index.Select(p => p.Key).FirstOrDefault()
                                         ?? throw new FileNotFoundException($"hf:{repo} has no data files."));
        if (index[chosenConfig] is not JsonObject splits)
        {
            throw new FileNotFoundException($"hf:{repo} has no configuration '{chosenConfig}'. Configurations: {string.Join(", ", index.Select(p => p.Key))}.");
        }

        if (splits[split] is not JsonArray urls)
        {
            throw new FileNotFoundException($"hf:{repo} ({chosenConfig}) has no split '{split}'. Splits: {string.Join(", ", splits.Select(p => p.Key))}.");
        }

        var paths = new List<string>();
        int number = 0;
        foreach (var url in urls.Select(u => (string)u!).Take(maxFiles ?? int.MaxValue))
        {
            paths.Add(await downloader.DownloadAsync(url, headers, $"huggingface/datasets/{repo}/parquet/{chosenConfig}/{split}/{number++:D5}.parquet").ConfigureAwait(false));
        }

        return paths;
    }
}

/// <summary>
/// Data from GitHub: a whole repository as a code dataset, data files in a repository, or release assets. Set
/// <c>GITHUB_TOKEN</c> (or pass a token) for private repositories and higher rate limits.
/// </summary>
public static class GitHub
{
    /// <summary>The token to use: <paramref name="token"/>, else <c>GITHUB_TOKEN</c> or <c>GH_TOKEN</c>.</summary>
    public static string? Token(string? token = null) =>
        !string.IsNullOrWhiteSpace(token) ? token : Environment.GetEnvironmentVariable("GITHUB_TOKEN") ?? Environment.GetEnvironmentVariable("GH_TOKEN");

    /// <summary>The API root (<c>GITHUB_API_URL</c> for GitHub Enterprise, else https://api.github.com).</summary>
    public static string Api => (Environment.GetEnvironmentVariable("GITHUB_API_URL") ?? "https://api.github.com").TrimEnd('/');

    private static Dictionary<string, string> Headers(string? token, string accept = "application/vnd.github+json") =>
        RemoteFiles.Bearer(Token(token), new Dictionary<string, string> { ["Accept"] = accept, ["X-GitHub-Api-Version"] = "2022-11-28" });

    /// <summary>
    /// A repository's files at <paramref name="reference"/> (branch, tag or commit; default: the default branch), one row
    /// per text file (source code, docs and data files alike): {"text", "path", "language", "repo"}. Dependency and build folders, binary files and files over
    /// <see cref="ReadOptions.MaxDocumentBytes"/> are skipped; <paramref name="pattern"/> (e.g. "src/**/*.cs") narrows the files.
    /// </summary>
    public static DatasetRows Repository(string repo, string? reference = null, string? pattern = null, string? token = null, IDownloader? downloader = null)
    {
        var d = downloader ?? Downloader.Shared;
        var options = new ReadOptions { Documents = true, Text = TextRows.Document, Pattern = pattern is null ? null : pattern.Contains('/') ? "*/" + pattern : pattern };
        return RemoteFiles.Over($"github:{repo}{(reference is null ? "" : "@" + reference)}",
            () =>
            {
                // The branch or tag names a commit; snapshots are cached per commit, so a new push is fetched again.
                string commit = d.GetStringAsync($"{Api}/repos/{repo}/commits/{Uri.EscapeDataString(reference ?? "HEAD")}", Headers(token, "application/vnd.github.sha"))
                    .GetAwaiter().GetResult().Trim();
                d.Log?.Invoke($"github:{repo}: {reference ?? "default branch"} is at commit {commit[..Math.Min(7, commit.Length)]}");
                string url = $"{Api}/repos/{repo}/tarball/{commit}";
                return [d.DownloadAsync(url, Headers(token), $"github/{repo}/{commit[..Math.Min(12, commit.Length)]}.tar.gz").GetAwaiter().GetResult()];
            },
            options,
            post: row =>
            {
                // GitHub's tarballs hold everything under "<owner>-<repo>-<commit>/".
                if ((string?)row["path"] is { } path && path.IndexOf('/') is var slash and > 0)
                {
                    row["path"] = path[(slash + 1)..];
                }

                row["repo"] = repo;
                return row;
            },
            downloader: d);
    }

    /// <summary>The data files of a repository matching <paramref name="pattern"/> (e.g. "data/*.jsonl"), read as data.</summary>
    public static DatasetRows Files(string repo, string pattern, string? reference = null, string? token = null, ReadOptions? options = null, IDownloader? downloader = null)
    {
        var d = downloader ?? Downloader.Shared;
        return RemoteFiles.Over($"github:{repo}/{pattern}", () => ResolveFilesAsync(repo, pattern, reference, token, d).GetAwaiter().GetResult(), options, downloader: d);
    }

    /// <summary>The assets of a release (<paramref name="tag"/>, or the latest) whose names match <paramref name="assetPattern"/>.</summary>
    public static DatasetRows Release(string repo, string assetPattern, string? tag = null, string? token = null, ReadOptions? options = null, IDownloader? downloader = null)
    {
        var d = downloader ?? Downloader.Shared;
        return RemoteFiles.Over($"github:{repo} release {tag ?? "latest"}", () => ResolveReleaseAsync(repo, assetPattern, tag, token, d).GetAwaiter().GetResult(), options, downloader: d);
    }

    private static async Task<IReadOnlyList<string>> ResolveFilesAsync(string repo, string pattern, string? reference, string? token, IDownloader downloader)
    {
        string tree = (await downloader.GetStringAsync($"{Api}/repos/{repo}/commits/{Uri.EscapeDataString(reference ?? "HEAD")}", Headers(token, "application/vnd.github.sha"))
            .ConfigureAwait(false)).Trim();
        downloader.Log?.Invoke($"github:{repo}: {reference ?? "default branch"} is at commit {tree[..Math.Min(7, tree.Length)]}");
        var json = JsonNode.Parse(await downloader.GetStringAsync($"{Api}/repos/{repo}/git/trees/{Uri.EscapeDataString(tree)}?recursive=1", Headers(token)).ConfigureAwait(false));
        var glob = DataFiles.Glob(pattern);
        var paths = (json?["tree"] as JsonArray ?? []).Where(n => (string?)n?["type"] == "blob").Select(n => (string)n!["path"]!)
            .Where(p => glob.IsMatch(p)).Order(StringComparer.Ordinal).ToList();
        if (paths.Count == 0)
        {
            throw new FileNotFoundException($"github:{repo}: no files match '{pattern}'.");
        }

        var local = new List<string>();
        foreach (var path in paths)
        {
            string url = $"{Api}/repos/{repo}/contents/{RemoteFiles.EscapePath(path)}?ref={Uri.EscapeDataString(tree)}";
            local.Add(await downloader.DownloadAsync(url, Headers(token, "application/vnd.github.raw"), $"github/{repo}/{tree[..Math.Min(12, tree.Length)]}/{path}").ConfigureAwait(false));
        }

        return local;
    }

    private static async Task<IReadOnlyList<string>> ResolveReleaseAsync(string repo, string assetPattern, string? tag, string? token, IDownloader downloader)
    {
        string url = tag is null ? $"{Api}/repos/{repo}/releases/latest" : $"{Api}/repos/{repo}/releases/tags/{Uri.EscapeDataString(tag)}";
        var release = JsonNode.Parse(await downloader.GetStringAsync(url, Headers(token)).ConfigureAwait(false));
        var glob = DataFiles.Glob(assetPattern);
        var assets = (release?["assets"] as JsonArray ?? []).Where(a => glob.IsMatch((string?)a?["name"] ?? "")).ToList();
        if (assets.Count == 0)
        {
            throw new FileNotFoundException($"github:{repo}: release {tag ?? "latest"} has no asset matching '{assetPattern}'.");
        }

        var local = new List<string>();
        foreach (var asset in assets)
        {
            local.Add(await downloader.DownloadAsync((string)asset!["url"]!, Headers(token, "application/octet-stream"),
                $"github/{repo}/releases/{(string?)release?["tag_name"] ?? tag ?? "latest"}/{(string)asset["name"]!}").ConfigureAwait(false));
        }

        return local;
    }
}

/// <summary>
/// Datasets from Kaggle. Needs an API token: <c>KAGGLE_USERNAME</c> and <c>KAGGLE_KEY</c>, or the kaggle.json file from
/// your Kaggle account settings in ~/.kaggle (or <c>KAGGLE_CONFIG_DIR</c>).
/// </summary>
public static class Kaggle
{
    /// <summary>A Kaggle dataset ("owner/dataset"); <paramref name="pattern"/> picks files inside it (e.g. "*.csv").</summary>
    public static DatasetRows Dataset(string dataset, string? pattern = null, int? version = null, ReadOptions? options = null, IDownloader? downloader = null)
    {
        var d = downloader ?? Downloader.Shared;
        string url = $"https://www.kaggle.com/api/v1/datasets/download/{dataset}{(version is null ? "" : $"?datasetVersionNumber={version}")}";
        return RemoteFiles.Over($"kaggle:{dataset}",
            () => [d.DownloadAsync(url, Credentials(), $"kaggle/{dataset}/{(version is null ? "latest" : $"v{version}")}/{dataset.Split('/')[^1]}.zip").GetAwaiter().GetResult()],
            (options ?? ReadOptions.Default) with { Pattern = pattern ?? options?.Pattern }, downloader: d);
    }

    private static Dictionary<string, string> Credentials()
    {
        string? user = Environment.GetEnvironmentVariable("KAGGLE_USERNAME"), key = Environment.GetEnvironmentVariable("KAGGLE_KEY");
        if (user is null || key is null)
        {
            string folder = Environment.GetEnvironmentVariable("KAGGLE_CONFIG_DIR")
                            ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".kaggle");
            string file = Path.Combine(folder, "kaggle.json");
            if (File.Exists(file) && JsonNode.Parse(File.ReadAllText(file)) is JsonObject json)
            {
                user ??= (string?)json["username"];
                key ??= (string?)json["key"];
            }
        }

        if (user is null || key is null)
        {
            throw new InvalidOperationException("Kaggle needs an API token: set KAGGLE_USERNAME and KAGGLE_KEY, or put kaggle.json (Kaggle → Settings → API → Create New Token) in ~/.kaggle.");
        }

        return new Dictionary<string, string> { ["Authorization"] = "Basic " + Convert.ToBase64String(Encoding.UTF8.GetBytes($"{user}:{key}")) };
    }
}

/// <summary>Research datasets from Zenodo (zenodo.org records); <c>ZENODO_TOKEN</c> for restricted ones.</summary>
public static class Zenodo
{
    /// <summary>The files of record <paramref name="record"/> (its number) matching <paramref name="pattern"/>.</summary>
    public static DatasetRows Record(string record, string? pattern = null, string? token = null, ReadOptions? options = null, IDownloader? downloader = null)
    {
        var d = downloader ?? Downloader.Shared;
        return RemoteFiles.Over($"zenodo:{record}", () => ResolveAsync(record, pattern, token, d).GetAwaiter().GetResult(), options, downloader: d);
    }

    private static async Task<IReadOnlyList<string>> ResolveAsync(string record, string? pattern, string? token, IDownloader downloader)
    {
        var headers = RemoteFiles.Bearer(token ?? Environment.GetEnvironmentVariable("ZENODO_TOKEN"));
        var json = JsonNode.Parse(await downloader.GetStringAsync($"https://zenodo.org/api/records/{Uri.EscapeDataString(record)}", headers).ConfigureAwait(false));
        var glob = pattern is null ? null : DataFiles.Glob(pattern);
        var files = (json?["files"] as JsonArray ?? []).Select(f => (Name: (string?)f?["key"] ?? "", Url: (string?)f?["links"]?["self"] ?? ""))
            .Where(f => f.Url.Length > 0 && (glob?.IsMatch(f.Name) ?? RemoteFiles.IsDataFile(f.Name))).ToList();
        if (files.Count == 0)
        {
            throw new FileNotFoundException($"zenodo:{record}: no data files{(pattern is null ? "" : $" matching '{pattern}'")}.");
        }

        var local = new List<string>();
        foreach (var (name, url) in files)
        {
            local.Add(await downloader.DownloadAsync(url, headers, $"zenodo/{record}/{name}").ConfigureAwait(false));
        }

        return local;
    }
}
