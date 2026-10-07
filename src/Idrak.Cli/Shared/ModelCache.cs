// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Data;
using Idrak.Models.Abstractions;
using Idrak.Models;

namespace Idrak.Cli.Shared;

/// <summary>
/// The models in the cache folder, as the library lays them out (<see cref="ModelSource"/> and <see cref="Downloader"/>):
/// Hugging Face models under downloads/huggingface/models/&lt;owner&gt;/&lt;name&gt;/&lt;commit&gt;/, GGUF files pulled from the hub
/// beside them or from a URL under downloads/urls/, and the folders <see cref="GgufModel.Prepare"/> writes under gguf/.
/// Also: finding a model argument's local files without downloading, the last use of each model, and confirmations.
/// </summary>
internal static class ModelCache
{
    /// <summary>A cached model (or a GGUF file, a partial download, a prepared folder).</summary>
    /// <param name="Name">What commands take: "owner/name", "owner/name/file.gguf", "host/path/file.gguf".</param>
    /// <param name="Kind">"huggingface", "gguf", "partial" (an interrupted download), "prepared" or "hub cache".</param>
    /// <param name="Path">The folder or file.</param>
    /// <param name="Revision">The commit (its first 12 characters) of a hub model.</param>
    /// <param name="Bytes">Size on disk.</param>
    /// <param name="Format">The weights' format and element type ("safetensors BF16", "gguf Q8_0").</param>
    /// <param name="LastUse">When a command last used it (or the files' newest access time).</param>
    /// <param name="Removable">Whether <c>idrak rm</c> may delete it (false for another tool's cache).</param>
    public sealed record Entry(string Name, string Kind, string Path, string? Revision, long Bytes, string Format, DateTime LastUse, bool Removable);

    /// <summary>The download cache under <paramref name="cacheFolder"/> (what <see cref="Downloader.CacheFolder"/> is by default).</summary>
    public static string Downloads(string cacheFolder) => System.IO.Path.Combine(cacheFolder, "downloads");

    /// <summary>The folder Hugging Face models are downloaded into.</summary>
    public static string HubModels(string cacheFolder) => System.IO.Path.Combine(Downloads(cacheFolder), "huggingface", "models");

    /// <summary>
    /// A downloader for the cache folder: logs to the verbose output, draws a progress line on a terminal, and honours
    /// <c>--offline</c> and <c>--timeout</c> (<see cref="Http.Downloader"/>).
    /// </summary>
    public static Downloader Downloader(CommandContext context, bool refresh = false) => Http.Downloader(context, refresh: refresh);

    /// <summary>
    /// Every model in the cache, newest use first: Idrak's downloads and GGUF files; with <paramref name="all"/> also the
    /// folders prepared from GGUF files and the models in Hugging Face's own cache.
    /// </summary>
    public static List<Entry> Scan(string cacheFolder, bool all = false)
    {
        var entries = new List<Entry>();
        var uses = LastUses(cacheFolder);
        string hub = HubModels(cacheFolder);
        if (Directory.Exists(hub))
        {
            foreach (string owner in Directory.GetDirectories(hub))
            {
                foreach (string name in Directory.GetDirectories(owner))
                {
                    string id = $"{System.IO.Path.GetFileName(owner)}/{System.IO.Path.GetFileName(name)}";
                    foreach (string revision in Directory.GetDirectories(name))
                    {
                        string commit = System.IO.Path.GetFileName(revision);
                        var ggufs = Directory.GetFiles(revision, "*.gguf", SearchOption.AllDirectories);
                        foreach (string gguf in ggufs)
                        {
                            string file = System.IO.Path.GetRelativePath(revision, gguf).Replace('\\', '/');
                            entries.Add(new Entry($"{id}/{file}", "gguf", gguf, commit, new FileInfo(gguf).Length, GgufFormat(gguf), LastUse(uses, gguf), true));
                        }

                        bool complete = File.Exists(System.IO.Path.Combine(revision, "config.json"))
                                        && Directory.EnumerateFiles(revision, "*.safetensors").Any()
                                        && !Directory.EnumerateFiles(revision, "*.part", SearchOption.AllDirectories).Any();
                        if (complete)
                        {
                            entries.Add(new Entry(id, "huggingface", revision, commit, FolderBytes(revision), SafeTensorsFormat(revision), LastUse(uses, revision), true));
                        }
                        else if (ggufs.Length == 0 || Directory.EnumerateFiles(revision, "*.part", SearchOption.AllDirectories).Any())
                        {
                            entries.Add(new Entry(id, "partial", revision, commit, FolderBytes(revision), "partial download", LastUse(uses, revision), true));
                        }
                    }
                }
            }
        }

        string urls = System.IO.Path.Combine(Downloads(cacheFolder), "urls");
        if (Directory.Exists(urls))
        {
            foreach (string gguf in Directory.GetFiles(urls, "*.gguf", SearchOption.AllDirectories))
            {
                string name = System.IO.Path.GetRelativePath(urls, gguf).Replace('\\', '/');
                entries.Add(new Entry(name, "gguf", gguf, null, new FileInfo(gguf).Length, GgufFormat(gguf), LastUse(uses, gguf), true));
            }

            foreach (string part in Directory.GetFiles(urls, "*.gguf.part", SearchOption.AllDirectories))
            {
                string name = System.IO.Path.GetRelativePath(urls, part).Replace('\\', '/')[..^5];
                entries.Add(new Entry(name, "partial", part, null, new FileInfo(part).Length, "partial download", LastUse(uses, part), true));
            }
        }

        if (all)
        {
            string prepared = System.IO.Path.Combine(cacheFolder, "gguf");
            if (Directory.Exists(prepared))
            {
                foreach (string folder in Directory.GetDirectories(prepared).Where(GgufModel.IsPrepared))
                {
                    string source = SourceOf(folder) ?? "?";
                    entries.Add(new Entry(System.IO.Path.GetFileName(folder), "prepared", folder, null, FolderBytes(folder), $"prepared from {Path.GetFileName(source)}", LastUse(uses, folder), true));
                }
            }

            string hfCache = HuggingFaceCacheFolder();
            if (Directory.Exists(hfCache))
            {
                foreach (string repo in Directory.GetDirectories(hfCache, "models--*"))
                {
                    string snapshots = System.IO.Path.Combine(repo, "snapshots");
                    string id = System.IO.Path.GetFileName(repo)["models--".Length..].Replace("--", "/", StringComparison.Ordinal);
                    foreach (string snapshot in Directory.Exists(snapshots) ? Directory.GetDirectories(snapshots) : [])
                    {
                        if (File.Exists(System.IO.Path.Combine(snapshot, "config.json")))
                        {
                            string commit = System.IO.Path.GetFileName(snapshot);
                            entries.Add(new Entry(id, "hub cache", snapshot, commit[..Math.Min(12, commit.Length)], FolderBytes(snapshot), SafeTensorsFormat(snapshot),
                                LastUse(uses, snapshot), false));
                        }
                    }
                }
            }
        }

        return [.. entries.OrderByDescending(e => e.LastUse).ThenBy(e => e.Name, StringComparer.Ordinal)];
    }

    /// <summary>The entries <paramref name="name"/> names: by name ("owner/name", "owner/name@commit"), or by path.</summary>
    public static List<Entry> Match(IEnumerable<Entry> entries, string name)
    {
        string? revision = null;
        int at = name.LastIndexOf('@');
        if (at > 0)
        {
            (name, revision) = (name[..at], name[(at + 1)..]);
        }

        string full = System.IO.Path.GetFullPath(name);
        return [.. entries.Where(e => (e.Name.Equals(name, StringComparison.OrdinalIgnoreCase) && (revision is null || e.Revision?.StartsWith(revision, StringComparison.OrdinalIgnoreCase) == true))
                                      || string.Equals(System.IO.Path.GetFullPath(e.Path), full, OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal))];
    }

    /// <summary>What a model argument is on disk.</summary>
    /// <param name="Name">The name as given, after an alias.</param>
    /// <param name="Folder">The folder in the Hugging Face layout (config.json …); a GGUF file's prepared folder.</param>
    /// <param name="File">The GGUF file, when the model is one.</param>
    public sealed record Local(string Name, string Folder, string? File);

    /// <summary>
    /// The local files of a model argument (an alias, a folder, a .gguf file, a Hugging Face id, a pulled GGUF file),
    /// without downloading: a model that is not cached is an error naming <c>idrak pull</c>.
    /// </summary>
    public static Local Locate(CommandContext context, string name)
    {
        string model = ModelChoices.Choose(context, name).Model;
        return TryLocate(context, model) ?? throw new InvalidOperationException(context.Offline
            ? $"{model} is not in the cache ({context.CacheFolder}) and --offline allows no download; run 'idrak pull {model}' while online."
            : $"{model} is not in the cache ({context.CacheFolder}). Download it first: idrak pull {model}");
    }

    /// <summary>
    /// The local files of <paramref name="model"/> (a folder, a .gguf file, a cached Hugging Face id or pulled GGUF
    /// file, or a model in Hugging Face's own cache), or null when nothing local matches; no network. A cached model's
    /// last use is recorded (<see cref="Touch"/>).
    /// </summary>
    public static Local? TryLocate(CommandContext context, string model)
    {
        if (Directory.Exists(model))
        {
            return new Local(model, model, null);
        }

        if (System.IO.File.Exists(model))
        {
            return model.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)
                ? new Local(model, GgufModel.Prepare(model, context.CacheFolder), model)
                : throw new UsageException($"{model} is a file but not a .gguf model; give the model's folder, a .gguf file or a Hugging Face id.");
        }

        var cached = Scan(context.CacheFolder).Where(e => e.Kind is "huggingface" or "gguf").ToList();
        var found = Match(cached, model);
        if (found.Count == 0)
        {
            // A GGUF repository pulled as "owner/name" or "owner/name:TAG": its one file (or the one the tag names).
            var (repo, tag) = PullTarget.SplitTag(model);
            var files = cached.Where(e => e.Kind == "gguf" && e.Name.StartsWith(repo + "/", StringComparison.OrdinalIgnoreCase)
                                          && (tag is null || System.IO.Path.GetFileName(e.Name).Contains(tag, StringComparison.OrdinalIgnoreCase))).ToList();
            found = files.Count == 1 ? files : found;
            if (files.Count > 1)
            {
                throw new UsageException($"{model} has {files.Count} GGUF files in the cache ({string.Join(", ", files.Select(f => f.Name))}); name one.");
            }
        }

        if (found.FirstOrDefault() is { } entry)
        {
            Touch(context.CacheFolder, entry.Path);
            return entry.Kind == "gguf" ? new Local(model, GgufModel.Prepare(entry.Path, context.CacheFolder), entry.Path) : new Local(model, entry.Path, null);
        }

        if (ModelSource.IsModelId(model))
        {
            try
            {
                // Hugging Face's own cache too (models fetched with other tools), as loading does.
                string folder = ModelSource.Resolve(model, downloader: Downloader(context), download: false);
                return new Local(model, folder, null);
            }
            catch (DirectoryNotFoundException)
            {
            }
        }

        return null;
    }

    /// <summary>The checkpoint's tensors by Hugging Face name (safetensors or GGUF), with their shapes.</summary>
    public static ITensorStore OpenTensors(string path)
    {
        var format = CheckpointFormats.For(path);
        return format.Open(format.Prepare(path));
    }

    /// <summary>Total elements of the tensors in a store.</summary>
    public static long Parameters(ITensorStore store) => store.Names.Sum(n => store.ShapeOf(n).Aggregate(1L, (a, b) => a * b));

    /// <summary>Records that a command used the model at <paramref name="path"/> now (for <c>idrak list</c>).</summary>
    public static void Touch(string cacheFolder, string path)
    {
        try
        {
            string file = System.IO.Path.Combine(cacheFolder, "models-last-use.json");
            var uses = System.IO.File.Exists(file) ? JsonNode.Parse(System.IO.File.ReadAllText(file)) as JsonObject ?? [] : [];
            uses[System.IO.Path.GetFullPath(path)] = DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture);
            Directory.CreateDirectory(cacheFolder);
            System.IO.File.WriteAllText(file, uses.ToJsonString());
        }
        catch (IOException)
        {
            // Last use is a convenience; a read-only cache still works.
        }
        catch (UnauthorizedAccessException)
        {
        }
    }

    /// <summary>Forgets the last use of <paramref name="path"/> (after it is removed).</summary>
    public static void Forget(string cacheFolder, string path)
    {
        string file = System.IO.Path.Combine(cacheFolder, "models-last-use.json");
        if (System.IO.File.Exists(file) && JsonNode.Parse(System.IO.File.ReadAllText(file)) is JsonObject uses && uses.Remove(System.IO.Path.GetFullPath(path)))
        {
            System.IO.File.WriteAllText(file, uses.ToJsonString());
        }
    }

    /// <summary>Bytes of every file under <paramref name="folder"/>.</summary>
    public static long FolderBytes(string folder) =>
        Directory.Exists(folder) ? Directory.EnumerateFiles(folder, "*", SearchOption.AllDirectories).Sum(f => new FileInfo(f).Length) : 0;

    /// <summary>Hugging Face's own cache of hub models (HF_HUB_CACHE, or HF_HOME/hub, or ~/.cache/huggingface/hub).</summary>
    public static string HuggingFaceCacheFolder() => Environment.GetEnvironmentVariable("HF_HUB_CACHE")
        ?? System.IO.Path.Combine(Environment.GetEnvironmentVariable("HF_HOME")
            ?? System.IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "huggingface"), "hub");

    /// <summary>The GGUF file a prepared folder was made from, or null.</summary>
    public static string? SourceOf(string folder)
    {
        try
        {
            return GgufModel.SourceOf(folder);
        }
        catch (Exception e) when (e is IOException or InvalidDataException or System.Text.Json.JsonException)
        {
            return null;
        }
    }

    /// <summary>The element types of a folder's safetensors weights ("safetensors BF16").</summary>
    public static string SafeTensorsFormat(string folder)
    {
        try
        {
            using var reader = SafeTensorsReader.Open(folder);
            var types = reader.Tensors.Values.GroupBy(t => t.Type).OrderByDescending(g => g.Sum(t => t.Count)).Select(g => g.Key.ToString());
            return "safetensors " + string.Join("+", types);
        }
        catch (Exception e) when (e is IOException or InvalidDataException or NotSupportedException or System.Text.Json.JsonException or KeyNotFoundException or NullReferenceException)
        {
            return "safetensors";
        }
    }

    /// <summary>The main element type of a GGUF file's tensors ("gguf Q8_0"), the one holding the most values.</summary>
    public static string GgufFormat(string path)
    {
        try
        {
            using var file = GgufFile.Open(path);
            var main = file.Tensors.Values.GroupBy(t => GgufFile.TypeName(t.Type)).OrderByDescending(g => g.Sum(t => t.Count)).Select(g => g.Key).FirstOrDefault();
            return main is null ? "gguf" : $"gguf {main}";
        }
        catch (Exception e) when (e is IOException or InvalidDataException or NotSupportedException or EndOfStreamException or OverflowException or ArgumentException)
        {
            return "gguf (unreadable)";
        }
    }

    private static Dictionary<string, DateTime> LastUses(string cacheFolder)
    {
        var result = new Dictionary<string, DateTime>(StringComparer.Ordinal);
        string file = System.IO.Path.Combine(cacheFolder, "models-last-use.json");
        try
        {
            if (System.IO.File.Exists(file) && JsonNode.Parse(System.IO.File.ReadAllText(file)) is JsonObject uses)
            {
                foreach (var (path, value) in uses)
                {
                    if ((string?)value is { } text && DateTime.TryParse(text, CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind, out var time))
                    {
                        result[path] = time.ToUniversalTime();
                    }
                }
            }
        }
        catch (Exception e) when (e is IOException or System.Text.Json.JsonException)
        {
        }

        return result;
    }

    // The recorded last use, else the newest access (or write) time of the files.
    private static DateTime LastUse(Dictionary<string, DateTime> uses, string path)
    {
        if (uses.TryGetValue(System.IO.Path.GetFullPath(path), out var used))
        {
            return used;
        }

        IEnumerable<string> files = Directory.Exists(path) ? Directory.EnumerateFiles(path, "*", SearchOption.AllDirectories) : [path];
        return files.Select(f => new FileInfo(f)).Select(f => f.LastAccessTimeUtc > f.LastWriteTimeUtc ? f.LastAccessTimeUtc : f.LastWriteTimeUtc)
            .DefaultIfEmpty(DateTime.MinValue).Max();
    }
}
