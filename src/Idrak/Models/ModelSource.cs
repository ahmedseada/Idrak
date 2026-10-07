// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Models;

/// <summary>
/// Where a model comes from: a local folder, a GGUF file, a model of the local model store ("store:qwen3:8b", see
/// <see cref="LocalStoreModel"/>), or a Hugging Face model id such as "Qwen/Qwen3-0.6B". Idrak registers the folder,
/// store and .gguf sources of <see cref="ModelSources"/>; Idrak.Datasets, which downloads from the Hub, registers the
/// Hugging Face one (<c>HuggingFaceModels</c>).
/// </summary>
public static class ModelSource
{
    /// <summary>
    /// The GGUF file of a model in the local model store that another local model server keeps its pulled models in
    /// ("qwen3:8b", "llama3.2" for :latest, "user/model:tag", "hf.co/owner/repo:tag"): the folder its OLLAMA_MODELS
    /// variable names, or that server's default folder in the home folder. The model's manifest names the blob holding
    /// the weights, which is read in place.
    /// </summary>
    public static string LocalStoreModel(string name)
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
            throw new FileNotFoundException($"Model '{name}' is not in the local model store ({manifest}). Models in {store}: {known}.");
        }

        var layers = System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(manifest))?["layers"] as System.Text.Json.Nodes.JsonArray ?? [];
        string digest = layers.Where(l => (string?)l?["mediaType"] == "application/vnd.ollama.image.model").Select(l => (string?)l!["digest"]).FirstOrDefault()
                        ?? throw new InvalidDataException($"The manifest of '{name}' in the local model store has no model layer.");
        string blob = Path.Combine(store, "blobs", digest.Replace(':', '-'));
        return File.Exists(blob) ? blob : throw new FileNotFoundException($"'{name}' in the local model store: its weights {blob} are missing (pull the model again).");
    }

    // The length of the local model store's prefix on a model name ("store:"), else 0.
    private static int StorePrefix(string model) => model.StartsWith("store:", StringComparison.OrdinalIgnoreCase) ? 6 : 0;

    /// <summary>Whether <paramref name="model"/> reads as a Hugging Face id ("owner/name") rather than a folder.</summary>
    public static bool IsModelId(string model) =>
        !Directory.Exists(model) && !Path.IsPathRooted(model) && model.Count(c => c == '/') == 1 && !model.StartsWith('.') && !model.Contains('\\');

    /// <summary>
    /// The local folder of <paramref name="model"/>: the folder itself, a .gguf file's prepared folder, or what another
    /// source registered with <see cref="ModelSources"/> makes of it (a Hugging Face id found in a cache or downloaded,
    /// with Idrak.Datasets). With <paramref name="download"/> false, only the caches are searched.
    /// </summary>
    public static string Resolve(string model, string revision = "main", string? token = null, IDownloader? downloader = null, bool download = true) =>
        ModelSources.For(model)?.Resolve(model, Options(revision, token, downloader, download))
        ?? throw new DirectoryNotFoundException(IsModelId(model) && !ModelSources.Names.Contains("huggingface")
            ? $"'{model}' reads as a Hugging Face model id, but no source resolves those; reference Idrak.Datasets, which adds it."
            : $"'{model}' is neither a folder nor a Hugging Face model id (owner/name).");

    /// <summary>The options <see cref="Resolve"/> passes to a source: the revision, token, downloader and download switch.</summary>
    public static ModelSourceOptions Options(string revision = "main", string? token = null, IDownloader? downloader = null, bool download = true) =>
        new() { Revision = revision, Token = token, Download = download, Downloader = downloader };

    // The built-in sources, in the order they are asked (see ModelSources); LibraryModelFormats registers them.
    internal static IModelSource[] BuiltIn =>
    [
        new DelegateModelSource("folder", Directory.Exists, (model, _) => model),
        new DelegateModelSource("store", model => StorePrefix(model) > 0, (model, options) =>
        {
            string blob = LocalStoreModel(model[StorePrefix(model)..]);
            options.Downloader?.Log?.Invoke($"{model}: the local model store's file {blob}");
            return GgufModel.Prepare(blob);
        }),
        new DelegateModelSource("gguf", model => File.Exists(model) && model.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase), (model, _) => GgufModel.Prepare(model)),
    ];

    private sealed class DelegateModelSource(string name, Func<string, bool> canResolve, Func<string, ModelSourceOptions, string> resolve) : IModelSource
    {
        public string Name => name;

        public bool CanResolve(string model) => canResolve(model);

        public string Resolve(string model, ModelSourceOptions options) => resolve(model, options);
    }
}
