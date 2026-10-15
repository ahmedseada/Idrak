// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models;

namespace Idrak.Data;

/// <summary>
/// Hugging Face models by id ("Qwen/Qwen3-0.6B"), the model source this package registers with
/// <see cref="ModelSources"/> (as "huggingface"), so <c>ModelSource.Resolve</c> and <c>PretrainedModel.Load</c> take
/// an id. An id is looked up in Hugging Face's own cache (models fetched with transformers or huggingface-cli), then in
/// Idrak's download cache, and downloaded otherwise: only the files the library reads (config, tokenizer, chat template,
/// generation config and the safetensors weights, else one PyTorch checkpoint read without running it), into
/// downloads/huggingface/models/&lt;owner&gt;/&lt;name&gt;/&lt;commit&gt;/ (a subfolder's model below it, <see cref="DownloadFolderAsync"/>).
/// Gated and private models need a token (HF_TOKEN or huggingface-cli login).
/// </summary>
public static class HuggingFaceModels
{
    private static readonly string[] Wanted = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "added_tokens.json", "chat_template.jinja", "chat_template.json", "model.safetensors.index.json",
        "preprocessor_config.json", "processor_config.json",
        "vocab.json", "merges.txt", "vocab.txt", "tokenizer.model"];   // the tokenizers without a tokenizer.json (VITS's vocab.json, BPE, WordPiece, SentencePiece)

    // Registered after Idrak's own sources, so asked before them: it leaves them the names they own (a .gguf file, "store:…").
    internal static IModelSource Source { get; } = new HubSource();

    /// <summary>Downloads the files the library reads of a Hugging Face model (once) and returns their folder.</summary>
    public static Task<string> DownloadAsync(string repo, string revision = "main", string? token = null, IDownloader? downloader = null,
        CancellationToken cancellationToken = default) => DownloadFolderAsync(repo, "", revision, token, downloader, cancellationToken);

    /// <summary>
    /// Downloads the files the library reads of the model in <paramref name="subfolder"/> of a Hugging Face repository
    /// (once; "" for its top level) and returns their folder: a repository that keeps a model below its top level
    /// (myshell-ai/OpenVoiceV2's converter/) is read as one that keeps it at the top.
    /// </summary>
    public static async Task<string> DownloadFolderAsync(string repo, string subfolder, string revision = "main", string? token = null, IDownloader? downloader = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(subfolder);
        subfolder = subfolder.Trim('/');
        var d = downloader ?? Downloader.Shared;
        IReadOnlyList<RepoFile> files;
        string commit;
        try
        {
            commit = await HuggingFace.ResolveRevisionAsync(repo, "models", revision, token, d, cancellationToken).ConfigureAwait(false);
            files = await HuggingFace.ListFilesAsync(repo, "models", commit, token, d, cancellationToken).ConfigureAwait(false);
        }
        catch (HttpRequestException ex) when (ex.StatusCode is null && NewestDownloaded(repo, d) is { } offline
                                              && Directory.Exists(Path.Combine(offline, subfolder)))
        {
            d.Log?.Invoke($"{repo}: Hugging Face cannot be reached ({ex.Message.Split('\n')[0]}); using the copy downloaded before: {offline}");
            return Path.Combine(offline, subfolder);
        }

        var chosen = ChooseFiles(repo, files, subfolder);
        string root = d.PathFor($"huggingface/models/{repo}/{commit[..Math.Min(12, commit.Length)]}");
        string folder = subfolder.Length == 0 ? root : Path.Combine(root, subfolder);
        var missing = chosen.Where(f => !File.Exists(Path.Combine(root, f.Path)) || d is Downloader { Refresh: true }).ToList();
        if (missing.Count == 0)
        {
            d.Log?.Invoke($"{repo}: all {chosen.Count} files cached ({Downloader.Size(chosen.Sum(f => new FileInfo(Path.Combine(root, f.Path)).Length))})");
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

    /// <summary>
    /// The files the library reads of the model in <paramref name="subfolder"/> of a repository ("" for its top level),
    /// by their paths in the repository: its configuration, tokenizer, chat template and processor files, and its weights:
    /// the safetensors (the shards the index names, or the single file), else one PyTorch checkpoint (.pth, .pt, .bin,
    /// .ckpt), which <see cref="PyTorchCheckpointReader"/> reads without running it. ONNX exports and deeper folders are skipped.
    /// </summary>
    /// <exception cref="FileNotFoundException">The level holds no config.json, no weights, sharded PyTorch weights, or several checkpoints.</exception>
    public static IReadOnlyList<RepoFile> ChooseFiles(string repo, IReadOnlyList<RepoFile> files, string subfolder = "")
    {
        ArgumentNullException.ThrowIfNull(files);
        string prefix = subfolder.Trim('/') is { Length: > 0 } sub ? sub + "/" : "";
        string where = prefix.Length == 0 ? repo : $"{repo}/{prefix.TrimEnd('/')}";
        var level = files.Where(f => f.Path.StartsWith(prefix, StringComparison.Ordinal) && !f.Path.AsSpan(prefix.Length).Contains('/')).ToList();
        string Name(RepoFile f) => f.Path[prefix.Length..];
        if (!level.Any(f => Name(f) == "config.json"))
        {
            var folders = files.Where(f => f.Path.StartsWith(prefix, StringComparison.Ordinal) && f.Path.EndsWith("/config.json", StringComparison.Ordinal))
                .Select(f => f.Path[..^"/config.json".Length]).Take(6).ToList();
            throw new FileNotFoundException(level.Any(f => Name(f).EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
                ? $"{where} holds GGUF files, not a transformers model (config.json and safetensors)."
                : $"{where} has no config.json at its top level; it is not a model"
                  + (folders.Count > 0 ? $" (a config.json is in {string.Join(", ", folders)}: pull {repo}/{folders[0]})." : "."));
        }

        var chosen = level.Where(f => Wanted.Contains(Name(f))).ToList();
        var safetensors = level.Where(f => Name(f).EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase)).ToList();
        if (safetensors.Count > 0)
        {
            // A sharded model lists its shards in the index; otherwise prefer the transformers file over a "consolidated" copy.
            bool sharded = level.Any(f => Name(f) == "model.safetensors.index.json");
            chosen.AddRange(sharded
                ? safetensors.Where(f => !Name(f).StartsWith("consolidated", StringComparison.Ordinal))
                : safetensors.Any(f => Name(f) == "model.safetensors") ? safetensors.Where(f => Name(f) == "model.safetensors") : safetensors);
            return chosen;
        }

        var checkpoints = level.Where(f => PyTorchCheckpointReader.Extensions.Contains(Path.GetExtension(f.Path), StringComparer.OrdinalIgnoreCase)).ToList();
        if (level.Any(f => Name(f).EndsWith(".bin.index.json", StringComparison.OrdinalIgnoreCase)))
        {
            throw new FileNotFoundException($"{where} has sharded PyTorch weights ({string.Join(", ", checkpoints.Take(4).Select(Name))}…); Idrak reads safetensors or one "
                + "PyTorch checkpoint (look for a safetensors version of the model).");
        }

        var single = checkpoints.Count == 1 ? checkpoints[0] : checkpoints.FirstOrDefault(f => Name(f) == "pytorch_model.bin");
        if (single is null)
        {
            throw new FileNotFoundException(checkpoints.Count == 0
                ? $"{where} has no safetensors or PyTorch weights."
                : $"{where} holds several PyTorch checkpoints ({string.Join(", ", checkpoints.Select(Name))}) and no safetensors; download the one you want with its own name.");
        }

        chosen.Add(single);
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
