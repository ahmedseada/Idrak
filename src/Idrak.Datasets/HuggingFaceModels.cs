// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models;

namespace Idrak.Datasets;

/// <summary>
/// Hugging Face models by id ("Qwen/Qwen3-0.6B"), the model source this package registers with
/// <see cref="ModelSources"/> (as "huggingface"), so <c>ModelSource.Resolve</c> and <c>PretrainedModel.Load</c> take
/// an id. An id is looked up in Hugging Face's own cache (models fetched with transformers or huggingface-cli), then in
/// Idrak's download cache, and downloaded otherwise: only the files the library reads (config, tokenizer, chat template,
/// generation config and the safetensors weights), into downloads/huggingface/models/&lt;owner&gt;/&lt;name&gt;/&lt;commit&gt;/.
/// Gated and private models need a token (HF_TOKEN or huggingface-cli login).
/// </summary>
public static class HuggingFaceModels
{
    private static readonly string[] Wanted = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "added_tokens.json", "chat_template.jinja", "chat_template.json", "model.safetensors.index.json"];

    // Registered after Idrak's own sources, so asked before them: it leaves them the names they own (a .gguf file, "store:…").
    internal static IModelSource Source { get; } = new HubSource();

    /// <summary>Downloads the files the library reads of a Hugging Face model (once) and returns their folder.</summary>
    public static async Task<string> DownloadAsync(string repo, string revision = "main", string? token = null, IDownloader? downloader = null,
        CancellationToken cancellationToken = default)
    {
        var d = downloader ?? Downloader.Shared;
        IReadOnlyList<RepoFile> files;
        string commit;
        try
        {
            commit = await HuggingFace.ResolveRevisionAsync(repo, "models", revision, token, d, cancellationToken).ConfigureAwait(false);
            files = await HuggingFace.ListFilesAsync(repo, "models", commit, token, d, cancellationToken).ConfigureAwait(false);
        }
        catch (HttpRequestException ex) when (ex.StatusCode is null && NewestDownloaded(repo, d) is { } offline)
        {
            d.Log?.Invoke($"{repo}: Hugging Face cannot be reached ({ex.Message.Split('\n')[0]}); using the copy downloaded before: {offline}");
            return offline;
        }

        var chosen = Choose(repo, files);
        string folder = d.PathFor($"huggingface/models/{repo}/{commit[..Math.Min(12, commit.Length)]}");
        var missing = chosen.Where(f => !File.Exists(Path.Combine(folder, f.Path)) || d is Downloader { Refresh: true }).ToList();
        if (missing.Count == 0)
        {
            d.Log?.Invoke($"{repo}: all {chosen.Count} files cached ({Downloader.Size(chosen.Sum(f => new FileInfo(Path.Combine(folder, f.Path)).Length))})");
            return folder;
        }

        d.Log?.Invoke($"{repo}: {chosen.Count} of {files.Count} files needed ({Downloader.Size(chosen.Sum(f => f.Size))}), {missing.Count} to download");
        foreach (var file in missing)
        {
            await HuggingFace.DownloadFileAsync(repo, file.Path, "models", commit, token, d, cancellationToken).ConfigureAwait(false);
        }

        return folder;
    }

    // A Hugging Face id: the Hugging Face cache, then Idrak's downloads (or a download).
    private static string Resolve(string model, ModelSourceOptions options)
    {
        var downloader = options.Downloader ?? Downloader.Shared;
        if (options.Revision == "main" && HuggingFaceCache(model) is { } cached)
        {
            downloader.Log?.Invoke($"{model}: found in the Hugging Face cache ({cached})");
            return cached;
        }

        if (!options.Download)
        {
            return NewestDownloaded(model, downloader) ?? throw new DirectoryNotFoundException($"{model} is not in a local cache.");
        }

        return DownloadAsync(model, options.Revision, options.Token, downloader).GetAwaiter().GetResult();
    }

    // The files a model needs: its configuration, tokenizer and chat template, and the safetensors weights (the shards the
    // index names, or the single file). PyTorch .bin files, ONNX exports and subfolders (original/, onnx/ …) are skipped.
    private static List<RepoFile> Choose(string repo, IReadOnlyList<RepoFile> files)
    {
        var root = files.Where(f => !f.Path.Contains('/')).ToList();
        if (!root.Any(f => f.Path == "config.json"))
        {
            throw new FileNotFoundException(root.Any(f => f.Path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
                ? $"{repo} holds GGUF files, not a transformers model (config.json and safetensors)."
                : $"{repo} has no config.json at its top level; it is not a transformers model.");
        }

        var chosen = root.Where(f => Wanted.Contains(f.Path)).ToList();
        var safetensors = root.Where(f => f.Path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase)).ToList();
        if (safetensors.Count == 0)
        {
            throw new FileNotFoundException(root.Any(f => f.Path.EndsWith(".bin", StringComparison.OrdinalIgnoreCase))
                ? $"{repo} has only PyTorch .bin weights; Idrak reads safetensors (look for a safetensors version of the model)."
                : $"{repo} has no safetensors weights.");
        }

        // A sharded model lists its shards in the index; otherwise prefer the transformers file over a "consolidated" copy.
        bool sharded = root.Any(f => f.Path == "model.safetensors.index.json");
        chosen.AddRange(sharded
            ? safetensors.Where(f => !f.Path.StartsWith("consolidated", StringComparison.Ordinal))
            : safetensors.Any(f => f.Path == "model.safetensors") ? safetensors.Where(f => f.Path == "model.safetensors") : safetensors);
        return chosen;
    }

    // A snapshot in Hugging Face's cache (HF_HOME or ~/.cache/huggingface/hub) with a config and safetensors weights.
    private static string? HuggingFaceCache(string repo)
    {
        string home = Environment.GetEnvironmentVariable("HF_HUB_CACHE")
                      ?? Path.Combine(Environment.GetEnvironmentVariable("HF_HOME")
                                      ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "huggingface"), "hub");
        string snapshots = Path.Combine(home, "models--" + repo.Replace("/", "--", StringComparison.Ordinal), "snapshots");
        if (!Directory.Exists(snapshots))
        {
            return null;
        }

        return Directory.GetDirectories(snapshots)
            .Where(d => File.Exists(Path.Combine(d, "config.json")) && Directory.EnumerateFiles(d, "*.safetensors").Any())
            .OrderByDescending(Directory.GetLastWriteTimeUtc)
            .FirstOrDefault();
    }

    // The most recent download of a model in Idrak's cache.
    private static string? NewestDownloaded(string repo, IDownloader? downloader)
    {
        string folder = Path.Combine([(downloader ?? Downloader.Shared).CacheFolder, "huggingface", "models", .. repo.Split('/')]);
        return Directory.Exists(folder)
            ? Directory.GetDirectories(folder).Where(d => File.Exists(Path.Combine(d, "config.json"))).OrderByDescending(Directory.GetLastWriteTimeUtc).FirstOrDefault()
            : null;
    }

    private sealed class HubSource : IModelSource
    {
        public string Name => "huggingface";

        public bool CanResolve(string model) =>
            ModelSource.IsModelId(model) && !File.Exists(model) && !model.StartsWith("store:", StringComparison.OrdinalIgnoreCase);

        public string Resolve(string model, ModelSourceOptions options) => HuggingFaceModels.Resolve(model, options);
    }
}
