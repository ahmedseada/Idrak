// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Data;
using Idrak.Maui.Abstractions;
using Idrak.Models;

namespace Idrak.Maui;

/// <summary>
/// The model store over the library's downloads (<see cref="IModelStore"/>): <see cref="HuggingFaceModels"/> for a model's
/// files, <see cref="HuggingFace.DownloadFileAsync"/> for one .gguf file, both through a <see cref="Downloader"/> into
/// <see cref="ComputeResources.CacheFolder"/>, where loading a model by its id finds it.
/// </summary>
public sealed class ModelStore(IdrakMauiOptions options, HttpClient? http = null) : IModelStore
{
    private const string Models = "models";

    /// <inheritdoc />
    public string Folder => Path.Combine(ComputeResources.CacheFolder, "downloads", "huggingface", Models);

    /// <inheritdoc />
    public async Task<StoredModel> PullAsync(string model, IProgressReporter? progress = null, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(model);
        var (repo, file) = Split(model.Trim());
        var reporter = progress ?? ProgressReporters.Start($"pull {model}", null, ProgressUnit.Bytes);
        var counter = new Counter(reporter);
        var downloader = new Downloader(http) { Progress = counter };
        string? token = HuggingFace.Token(options.HuggingFaceToken);
        try
        {
            string commit = await HuggingFace.ResolveRevisionAsync(repo, Models, "main", token, downloader, cancellationToken).ConfigureAwait(false);
            var files = await HuggingFace.ListFilesAsync(repo, Models, commit, token, downloader, cancellationToken).ConfigureAwait(false);
            IReadOnlyList<RepoFile> wanted = file is null ? HuggingFaceModels.ChooseFiles(repo, files)
                : files.Where(f => f.Path == file).ToList() is { Count: 1 } one ? one
                : throw new FileNotFoundException($"{repo} has no file {file} at {commit[..Math.Min(12, commit.Length)]}.");
            string root = downloader.PathFor($"huggingface/{Models}/{repo}/{commit[..Math.Min(12, commit.Length)]}");
            reporter.Total = wanted.Sum(f => f.Size);
            counter.Start(wanted.Where(f => File.Exists(Path.Combine(root, f.Path))).Sum(f => f.Size));   // cached files count as done
            if (file is null)
            {
                await HuggingFaceModels.DownloadAsync(repo, commit, token, downloader, cancellationToken).ConfigureAwait(false);
            }
            else
            {
                await HuggingFace.DownloadFileAsync(repo, file, Models, commit, token, downloader, cancellationToken).ConfigureAwait(false);
            }

            var stored = Read(repo, root);
            reporter.Finish($"{model}: {ProgressText.Bytes(stored.Bytes)} in {stored.Folder}");
            return stored;
        }
        catch (Exception e)
        {
            reporter.Finish($"{model}: not pulled ({e.Message.Split('\n')[0]})");
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
                bytes += info.Length;
                files++;
                complete &= !info.Name.EndsWith(".part", StringComparison.Ordinal);
                modified = info.LastWriteTimeUtc > modified ? info.LastWriteTimeUtc : modified;
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
