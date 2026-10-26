// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Net;
using System.Net.Http.Headers;
using Idrak.Data;
using Idrak.Maui.Abstractions;
using Idrak.Models;

namespace Idrak.Maui;

/// <summary>
/// The model store over the library's downloads (<see cref="IModelStore"/>): <see cref="HuggingFaceModels.ChooseFiles"/>
/// picks a model's files, <see cref="HuggingFace.DownloadFileAsync"/> brings each one (or the one .gguf file asked for)
/// through a <see cref="Downloader"/> into <see cref="ComputeResources.CacheFolder"/>, where loading a model by its id finds
/// it; the downloader resumes a partial file where it stopped. While a pull runs, its revision folder holds
/// <see cref="PullMarker"/>, so a cancelled or failed pull is listed as unfinished even between two files.
/// </summary>
public sealed class ModelStore : IModelStore
{
    /// <summary>The file a pull keeps in a model's revision folder until every file is in (not counted in its size): the library's own (<see cref="HuggingFaceModels.PullMarker"/>).</summary>
    public const string PullMarker = HuggingFaceModels.PullMarker;

    private const string Models = "models";
    private readonly IdrakMauiOptions _options;
    private readonly Lazy<HttpClient> _http;
    private readonly Func<string, StorageSpace?> _measure;

    /// <summary>A store pulling with <paramref name="options"/>' token and storage headroom.</summary>
    /// <param name="options">The app's options (read at each pull: a token saved in the settings applies to the next pull).</param>
    /// <param name="http">The client downloads go through; null: one of the store's own, made at the first pull and kept.</param>
    /// <param name="measureSpace">Measures the storage a folder is on; null: <see cref="Measure"/>.</param>
    public ModelStore(IdrakMauiOptions options, HttpClient? http = null, Func<string, StorageSpace?>? measureSpace = null)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _http = http is null ? new Lazy<HttpClient>(Client) : new Lazy<HttpClient>(http);
        _measure = measureSpace ?? Measure;
    }

    /// <inheritdoc />
    public string Folder => Path.Combine(ComputeResources.CacheFolder, "downloads", "huggingface", Models);

    /// <inheritdoc />
    public bool HasToken => HuggingFace.Token(_options.HuggingFaceToken) is not null;

    /// <inheritdoc />
    public async Task<StoredModel> PullAsync(string model, string? revision = null, IProgressReporter? progress = null, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(model);
        model = model.Trim();
        string repo, asked;
        string? file;
        try
        {
            (repo, file) = Split(model);
            asked = Revision(revision);
        }
        catch (ArgumentException e)
        {
            progress?.Finish($"{model}: not pulled ({e.Message.Split('\n')[0]})");    // a caller's item ends with the refusal
            throw;
        }

        string name = asked == "main" ? model : $"{model}@{asked}";
        var reporter = progress ?? ProgressReporters.Start($"pull {name}", null, ProgressUnit.Bytes);
        var counter = new Counter(reporter);
        var downloader = new Downloader(_http.Value) { Progress = counter };
        string? token = HuggingFace.Token(_options.HuggingFaceToken);
        try
        {
            string commit = await HuggingFace.ResolveRevisionAsync(repo, Models, asked, token, downloader, cancellationToken).ConfigureAwait(false);
            string shortCommit = commit[..Math.Min(12, commit.Length)];
            var files = await HuggingFace.ListFilesAsync(repo, Models, commit, token, downloader, cancellationToken).ConfigureAwait(false);
            IReadOnlyList<RepoFile> wanted = file is null ? HuggingFaceModels.ChooseFiles(repo, files)
                : files.Where(f => f.Path == file).ToList() is { Count: 1 } one ? one
                : throw new FileNotFoundException($"{repo} has no file {file} at {shortCommit}.");
            string root = downloader.PathFor($"huggingface/{Models}/{repo}/{shortCommit}");

            // What is in already (a finished file counts whole, a partial one by its bytes) and what is still to come.
            long done = 0, needed = 0;
            var missing = new List<RepoFile>();
            foreach (var f in wanted)
            {
                string path = Path.Combine(root, f.Path);
                if (File.Exists(path))
                {
                    done += f.Size;
                    continue;
                }

                missing.Add(f);
                needed += Math.Max(0, f.Size - (File.Exists(path + ".part") ? new FileInfo(path + ".part").Length : 0));
            }

            reporter.Total = wanted.Sum(f => f.Size);
            CheckSpace(name, root, needed);
            counter.Start(done);
            string marker = Path.Combine(root, PullMarker);
            if (missing.Count > 0)
            {
                Directory.CreateDirectory(root);
                await File.WriteAllTextAsync(marker, commit, cancellationToken).ConfigureAwait(false);
                foreach (var f in missing)
                {
                    await HuggingFace.DownloadFileAsync(repo, f.Path, Models, commit, token, downloader, cancellationToken).ConfigureAwait(false);
                }
            }

            File.Delete(marker);                                                // also one an earlier pull left after its last file
            var stored = Read(repo, root);
            reporter.Finish($"{name}: {ProgressText.Bytes(stored.Bytes)} in {stored.Folder}");
            return stored;
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            reporter.Finish($"{name}: cancelled (pulling again resumes it)");
            throw;
        }
        catch (Exception e)
        {
            reporter.Finish($"{name}: not pulled ({e.Message.Split('\n')[0]})");
            throw;
        }
    }

    /// <inheritdoc />
    public IReadOnlyList<StoredModel> List()
    {
        var models = new List<StoredModel>();
        if (!Directory.Exists(Folder))
        {
            return models;
        }

        foreach (string owner in Directory.EnumerateDirectories(Folder))
        {
            foreach (string name in Directory.EnumerateDirectories(owner))
            {
                foreach (string revision in Directory.EnumerateDirectories(name))
                {
                    models.Add(Read($"{Path.GetFileName(owner)}/{Path.GetFileName(name)}", revision));
                }
            }
        }

        return [.. models.OrderBy(m => m.Id, StringComparer.OrdinalIgnoreCase).ThenByDescending(m => m.Modified)];
    }

    /// <inheritdoc />
    public bool Remove(string id)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        int at = id.IndexOf('@', StringComparison.Ordinal);
        string repo = at < 0 ? id.Trim() : id[..at].Trim();
        string? revision = at < 0 ? null : id[(at + 1)..].Trim();
        if (!ModelSource.IsModelId(repo) || revision is { } r && (r.Length == 0 || !r.All(char.IsAsciiLetterOrDigit)))
        {
            throw new ArgumentException($"'{id}' is not a model id (owner/name, or owner/name@revision).", nameof(id));
        }

        string root = Path.GetFullPath(Folder);
        string folder = Path.GetFullPath(revision is null ? Path.Combine(root, repo) : Path.Combine(root, repo, revision));
        if (!folder.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.Ordinal) || !Directory.Exists(folder))
        {
            return false;
        }

        Directory.Delete(folder, recursive: true);
        foreach (string parent in (string[])[Path.Combine(root, repo), Path.GetDirectoryName(Path.Combine(root, repo))!])
        {
            if (Directory.Exists(parent) && !Directory.EnumerateFileSystemEntries(parent).Any())
            {
                Directory.Delete(parent);                                       // the model's and the owner's folders, once empty
            }
        }

        return true;
    }

    /// <inheritdoc />
    public StorageSpace? FreeSpace() => _measure(Folder);

    /// <summary>
    /// The space of the storage <paramref name="folder"/> is on (the nearest folder of it that exists), through
    /// <see cref="DriveInfo"/>: the folder's own file system on Linux and Android, its drive on Windows. Null where it
    /// cannot be read.
    /// </summary>
    public static StorageSpace? Measure(string folder)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(folder);
        try
        {
            var existing = new DirectoryInfo(Path.GetFullPath(folder));
            while (existing is { Exists: false })
            {
                existing = existing.Parent;
            }

            if (existing is null)
            {
                return null;
            }

            var drive = new DriveInfo(OperatingSystem.IsWindows() ? Path.GetPathRoot(existing.FullName)! : existing.FullName);
            return new StorageSpace(existing.FullName, drive.AvailableFreeSpace, drive.TotalSize);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or ArgumentException or PlatformNotSupportedException)
        {
            return null;
        }
    }

    // The files still to come must fit, with the room the settings keep free; a storage that cannot be measured is not
    // refused (a full one then fails on writing, which the page names too).
    private void CheckSpace(string model, string root, long needed)
    {
        if (needed <= 0 || _measure(root) is not { } space)
        {
            return;
        }

        long kept = (_options.StorageHeadroomMiB ?? 0) << 20;
        if (space.FreeBytes - needed < kept)
        {
            throw new NotEnoughSpaceException(model, needed, space.FreeBytes, kept, space.Folder);
        }
    }

    // The store's own client, as the downloader makes one (no decompression, no time limit on a large file), made once.
    private static HttpClient Client()
    {
        var http = new HttpClient(new HttpClientHandler { AutomaticDecompression = DecompressionMethods.None }) { Timeout = Timeout.InfiniteTimeSpan };
        http.DefaultRequestHeaders.UserAgent.Add(new ProductInfoHeaderValue("Idrak", "1.0"));
        return http;
    }

    // "owner/name" (the library's files) or "owner/name/path/file.gguf" (that file).
    private static (string Repo, string? File) Split(string model)
    {
        string[] parts = model.Split('/');
        if (parts.Length >= 3 && model.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) && parts.All(p => p.Length > 0 && p != "." && p != ".."))
        {
            return ($"{parts[0]}/{parts[1]}", string.Join('/', parts[2..]));
        }

        return ModelSource.IsModelId(model) ? (model, null)
            : throw new ArgumentException($"'{model}' is not a Hugging Face id (owner/name) or an id and a .gguf file in it (owner/name/file.gguf).", nameof(model));
    }

    // A branch, a tag ("v1.0", "refs/pr/3") or a commit; empty is "main".
    private static string Revision(string? revision)
    {
        string text = revision?.Trim() ?? "";
        if (text.Length == 0)
        {
            return "main";
        }

        return text.Any(c => char.IsWhiteSpace(c) || char.IsControl(c)) || text.Split('/').Any(p => p is "" or "." or "..")
            ? throw new ArgumentException($"'{revision}' is not a revision (a branch, a tag or a commit).", nameof(revision))
            : text;
    }

    private static StoredModel Read(string id, string folder)
    {
        long bytes = 0;
        int files = 0;
        bool complete = true;
        var modified = DateTimeOffset.MinValue;
        if (Directory.Exists(folder))
        {
            foreach (var info in new DirectoryInfo(folder).EnumerateFiles("*", SearchOption.AllDirectories))
            {
                modified = info.LastWriteTimeUtc > modified ? info.LastWriteTimeUtc : modified;
                if (info.Name == PullMarker)
                {
                    complete = false;
                    continue;
                }

                bytes += info.Length;
                files++;
                complete &= !info.Name.EndsWith(".part", StringComparison.Ordinal);
            }
        }

        return new StoredModel(id, Path.GetFileName(folder), folder, bytes, files, complete, modified);
    }

    // The bytes of a pull across its files: those done (and those cached) plus the current file's.
    private sealed class Counter(IProgressReporter reporter) : IProgress<DownloadProgress>
    {
        private long _done;

        public void Start(long cached)
        {
            _done = cached;
            reporter.Report(cached);
        }

        public void Report(DownloadProgress value)
        {
            if (value.Completed)
            {
                _done += value.Received;
                reporter.Report(_done);
            }
            else
            {
                reporter.Report(_done + value.Received);
            }
        }
    }
}
