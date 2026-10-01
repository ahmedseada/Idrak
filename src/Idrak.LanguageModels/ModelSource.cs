// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Datasets;

namespace Idrak.LanguageModels;

/// <summary>
/// Where a model comes from: a local folder, a GGUF file, an Ollama model ("ollama:qwen3:8b", read from Ollama's own
/// store), or a Hugging Face model id such as "Qwen/Qwen3-0.6B". An id is looked up in
/// Hugging Face's own cache (models fetched with transformers or huggingface-cli), then in Idrak's download cache,
/// and downloaded otherwise: only the files the library reads (config, tokenizer, chat template, generation config and
/// the safetensors weights), into downloads/huggingface/models/&lt;owner&gt;/&lt;name&gt;/&lt;commit&gt;/. Gated and private models
/// need a token (HF_TOKEN or huggingface-cli login).
/// </summary>
public static class ModelSource
{
    private static readonly string[] Wanted = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "added_tokens.json", "chat_template.jinja", "chat_template.json", "model.safetensors.index.json"];

    /// <summary>
    /// The GGUF file of an Ollama model ("qwen3:8b", "llama3.2" for :latest, "user/model:tag", "hf.co/owner/repo:tag") in
    /// Ollama's store (OLLAMA_MODELS, or ~/.ollama/models): the manifest names the model layer's blob.
    /// </summary>
    public static string OllamaModel(string name)
    {
        string store = Environment.GetEnvironmentVariable("OLLAMA_MODELS")
                       ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".ollama", "models");
        int colon = name.LastIndexOf(':');
        string tag = colon > name.LastIndexOf('/') && colon > 0 ? name[(colon + 1)..] : "latest";
        string model = colon > name.LastIndexOf('/') && colon > 0 ? name[..colon] : name;
        var parts = model.Split('/');
        string[] path = parts.Length switch
        {
            1 => ["registry.ollama.ai", "library", parts[0]],
            2 => ["registry.ollama.ai", parts[0], parts[1]],
            _ => parts,
        };
        string manifest = Path.Combine([store, "manifests", .. path, tag]);
        if (!File.Exists(manifest))
        {
            string known = Directory.Exists(Path.Combine(store, "manifests"))
                ? string.Join(", ", Directory.EnumerateFiles(Path.Combine(store, "manifests"), "*", SearchOption.AllDirectories)
                    .Select(f => Path.GetRelativePath(Path.Combine(store, "manifests"), f).Replace('\\', '/'))
                    .Select(f => f.StartsWith("registry.ollama.ai/library/", StringComparison.Ordinal) ? f["registry.ollama.ai/library/".Length..] : f)
                    .Select(f => f[..f.LastIndexOf('/')] + ":" + f[(f.LastIndexOf('/') + 1)..]).Order().Take(30))
                : "none";
            throw new FileNotFoundException($"Ollama model '{name}' not found ({manifest}). Models in {store}: {known}.");
        }

        var layers = System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(manifest))?["layers"] as System.Text.Json.Nodes.JsonArray ?? [];
        string digest = layers.Where(l => (string?)l?["mediaType"] == "application/vnd.ollama.image.model").Select(l => (string?)l!["digest"]).FirstOrDefault()
                        ?? throw new InvalidDataException($"The manifest of Ollama model '{name}' has no model layer.");
        string blob = Path.Combine(store, "blobs", digest.Replace(':', '-'));
        return File.Exists(blob) ? blob : throw new FileNotFoundException($"Ollama model '{name}': its weights {blob} are missing (pull the model again).");
    }

    /// <summary>Whether <paramref name="model"/> reads as a Hugging Face id ("owner/name") rather than a folder.</summary>
    public static bool IsModelId(string model) =>
        !Directory.Exists(model) && !Path.IsPathRooted(model) && model.Count(c => c == '/') == 1 && !model.StartsWith('.') && !model.Contains('\\');

    /// <summary>
    /// The local folder of <paramref name="model"/>: the folder itself, or a Hugging Face id found in a cache or
    /// downloaded (see the class summary). With <paramref name="download"/> false, only the caches are searched.
    /// </summary>
    public static string Resolve(string model, string revision = "main", string? token = null, Downloader? downloader = null, bool download = true)
    {
        if (Directory.Exists(model))
        {
            return model;
        }

        if (model.StartsWith("ollama:", StringComparison.OrdinalIgnoreCase))
        {
            string blob = OllamaModel(model[7..]);
            (downloader ?? Downloader.Shared).Log?.Invoke($"{model}: Ollama's model file {blob}");
            return GgufModel.Prepare(blob);
        }

        if (File.Exists(model) && model.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
        {
            return GgufModel.Prepare(model);
        }

        if (!IsModelId(model))
        {
            throw new DirectoryNotFoundException($"'{model}' is neither a folder nor a Hugging Face model id (owner/name).");
        }

        if (revision == "main" && HuggingFaceCache(model) is { } cached)
        {
            (downloader ?? Downloader.Shared).Log?.Invoke($"{model}: found in the Hugging Face cache ({cached})");
            return cached;
        }

        if (!download)
        {
            return NewestDownloaded(model, downloader) ?? throw new DirectoryNotFoundException($"{model} is not in a local cache.");
        }

        return DownloadAsync(model, revision, token, downloader).GetAwaiter().GetResult();
    }

    /// <summary>Downloads the files the library reads of a Hugging Face model (once) and returns their folder.</summary>
    public static async Task<string> DownloadAsync(string repo, string revision = "main", string? token = null, Downloader? downloader = null,
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
        var missing = chosen.Where(f => !File.Exists(Path.Combine(folder, f.Path)) || d.Refresh).ToList();
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
    private static string? NewestDownloaded(string repo, Downloader? downloader)
    {
        string folder = Path.Combine([(downloader ?? Downloader.Shared).CacheFolder, "huggingface", "models", .. repo.Split('/')]);
        return Directory.Exists(folder)
            ? Directory.GetDirectories(folder).Where(d => File.Exists(Path.Combine(d, "config.json"))).OrderByDescending(Directory.GetLastWriteTimeUtc).FirstOrDefault()
            : null;
    }
}
