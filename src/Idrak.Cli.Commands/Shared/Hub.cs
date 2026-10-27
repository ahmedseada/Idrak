// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Data;

namespace Idrak.Cli.Shared;

/// <summary>How <c>idrak pull</c> reads a model name: "owner/name", "owner/name:TAG" (a GGUF file by its quantization tag), "owner/name/file.gguf", "owner/name/folder" (a model below the top level), a URL.</summary>
internal static class PullTarget
{
    /// <summary>"owner/name:Q4_K_M" → ("owner/name", "Q4_K_M"); a name without a tag → (name, null).</summary>
    public static (string Repo, string? Tag) SplitTag(string model)
    {
        int colon = model.LastIndexOf(':');
        return colon > model.LastIndexOf('/') && colon > 0 && !model.Contains("://", StringComparison.Ordinal) && model.Count(c => c == '/') == 1
            ? (model[..colon], model[(colon + 1)..])
            : (model, null);
    }

    /// <summary>"owner/name/sub/file.gguf" → ("owner/name", "sub/file.gguf"), or null when <paramref name="model"/> is not one.</summary>
    public static (string Repo, string File)? SplitFile(string model)
    {
        if (model.Contains("://", StringComparison.Ordinal) || Path.IsPathRooted(model) || model.StartsWith('.') || File.Exists(model))
        {
            return null;
        }

        var parts = model.Split('/');
        return parts.Length > 2 && model.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) ? ($"{parts[0]}/{parts[1]}", string.Join('/', parts[2..])) : null;
    }

    /// <summary>"owner/name/sub/folder" → ("owner/name", "sub/folder"), or null when <paramref name="model"/> is not one (a .gguf file, a path, a URL, "owner/name").</summary>
    public static (string Repo, string Folder)? SplitFolder(string model)
    {
        if (model.Contains("://", StringComparison.Ordinal) || Path.IsPathRooted(model) || model.StartsWith('.') || File.Exists(model) || Directory.Exists(model)
            || model.Contains('\\') || model.Contains(':') || model.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        var parts = model.Trim('/').Split('/');
        return parts.Length > 2 && parts.All(p => p.Length > 0) ? ($"{parts[0]}/{parts[1]}", string.Join('/', parts[2..])) : null;
    }
}

/// <summary>Hugging Face hub calls the model commands share (pull, search, verify), through the library's <see cref="Downloader"/>.</summary>
internal static class Hub
{
    // The files HuggingFaceModels.DownloadAsync takes of a transformers model (kept the same, so a pull is what loading reads).
    /// <summary>A file of a hub repository with its size and, for large (LFS) files, its SHA-256.</summary>
    public sealed record HubFile(string Path, long Size, string? Sha256);

    /// <summary>The token: <c>--token</c>, the config's "hf_token", else HF_TOKEN or the saved login.</summary>
    public static string? Token(CommandContext context) => HuggingFace.Token(context.Option("--token") ?? context.Config.Get("hf_token"));

    /// <summary>
    /// The files loading reads of the model in <paramref name="subfolder"/> ("" for the top level): the library's own
    /// choice (<see cref="HuggingFaceModels.ChooseFiles"/>: config, tokenizer, chat template, and the safetensors or one
    /// PyTorch checkpoint); null when the level holds no config.json (a GGUF repository, say).
    /// </summary>
    public static List<HubFile>? TransformersFiles(string repo, IReadOnlyList<HubFile> files, string subfolder = "")
    {
        string prefix = subfolder.Trim('/') is { Length: > 0 } sub ? sub + "/" : "";
        if (!files.Any(f => f.Path == prefix + "config.json"))
        {
            return null;
        }

        var byPath = files.ToDictionary(f => f.Path, StringComparer.Ordinal);
        try
        {
            return [.. HuggingFaceModels.ChooseFiles(repo, [.. files.Select(f => new RepoFile(f.Path, f.Size))], subfolder).Select(f => byPath[f.Path])];
        }
        catch (FileNotFoundException e)
        {
            throw new InvalidOperationException(e.Message, e);
        }
    }

    /// <summary>Every file of a model repository at <paramref name="commit"/>, with sizes and LFS hashes.</summary>
    public static async Task<List<HubFile>> FilesAsync(string repo, string commit, string? token, Downloader downloader)
    {
        var list = new List<HubFile>();
        string url = $"{HuggingFace.Endpoint}/api/models/{repo}/tree/{commit}?recursive=true";
        foreach (string page in await downloader.GetPagesAsync(url, Bearer(token)).ConfigureAwait(false))
        {
            foreach (var item in JsonNode.Parse(page) as JsonArray ?? [])
            {
                if ((string?)item?["type"] == "file" && (string?)item["path"] is { } path)
                {
                    long size = item["size"] is JsonValue v && v.TryGetValue(out long s) ? s : 0;
                    list.Add(new HubFile(path, size, (string?)item["lfs"]?["oid"]));
                }
            }
        }

        return list;
    }

    /// <summary>The hub's search for <paramref name="query"/>, most downloaded first, with what tells whether Idrak can load each model.</summary>
    public static async Task<JsonArray> SearchAsync(string query, int limit, string? token, Downloader downloader)
    {
        string url = $"{HuggingFace.Endpoint}/api/models?search={Uri.EscapeDataString(query)}&limit={limit}&sort=downloads&direction=-1"
                     + "&expand[]=downloads&expand[]=likes&expand[]=tags&expand[]=config&expand[]=gguf&expand[]=library_name&expand[]=pipeline_tag";
        return JsonNode.Parse(await downloader.GetStringAsync(url, Bearer(token)).ConfigureAwait(false)) as JsonArray
               ?? throw new InvalidDataException("The hub's search answered with something other than a list.");
    }

    private static Dictionary<string, string> Bearer(string? token) => string.IsNullOrEmpty(token) ? [] : new() { ["Authorization"] = $"Bearer {token}" };
}
