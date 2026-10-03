// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Datasets;
using Idrak.LanguageModels;

namespace Idrak.Cli.Commands;

/// <summary>
/// <c>idrak pull MODEL</c>: downloads a Hugging Face model (the files loading reads) or a GGUF file into the cache, with
/// progress; an interrupted pull resumes where it stopped (the downloader keeps the partial file).
/// </summary>
internal sealed class PullCommand : Command
{
    public override string Name => "pull";

    public override string Summary => "Download a Hugging Face model or a GGUF file into the cache, with progress and resume";

    public override string Usage => """
        MODEL [--file NAME] [--revision REV] [--token TOKEN] [-f] [--dry-run]

          MODEL               owner/name (a transformers model: config, tokenizer, chat template, safetensors weights),
                              owner/name:TAG or owner/name/FILE.gguf (one GGUF file of a repository), a URL of a .gguf
                              file, or an alias; a local folder or .gguf file needs no download (a GGUF file's
                              description is prepared in the cache)
              --file NAME     the GGUF file to take from a repository holding several
              --revision REV  a branch, tag or commit (default main)
              --token TOKEN   for gated and private models (default: the config's hf_token, HF_TOKEN or the saved login)
          -f, --force         download again even when cached
              --dry-run       list what would be downloaded, and its size, without downloading

        An interrupted pull (Ctrl+C, a lost connection) keeps its partial files; pulling again resumes them.

        Examples:
          idrak pull Qwen/Qwen3-0.6B
          idrak pull Qwen/Qwen3-0.6B-GGUF:Q8_0
          idrak pull unsloth/Qwen3-0.6B-GGUF --file Qwen3-0.6B-Q4_K_M.gguf --dry-run
        """;

    public override IReadOnlyCollection<string> ValueOptions => ["--file", "--revision", "--token"];

    public override IReadOnlyCollection<string> Flags => ["--force", "--dry-run"];

    public override IReadOnlyDictionary<string, string> ShortForms => new Dictionary<string, string> { ["-f"] = "--force" };

    public override int Run(CommandContext context)
    {
        string name = context.Argument(0, "MODEL (a Hugging Face id such as Qwen/Qwen3-0.6B, or a GGUF file)");
        if (context.Positional.Count > 1)
        {
            throw new UsageException($"pull takes one model; '{context.Positional[1]}' is extra.");
        }

        string model = Shared.Models.Choose(context, name).Model;
        bool dryRun = context.Flag("--dry-run"), force = context.Flag("--force");
        if (Directory.Exists(model) || File.Exists(model))
        {
            return Local(context, model, dryRun);
        }

        if (context.Offline)
        {
            // Cache only: a cached model is reported, anything else is an error naming what is missing.
            var cached = ModelCache.Locate(context, model);
            var path = cached.File ?? cached.Folder;
            context.Write($"{model} is already in the cache: {Units.Bytes(cached.File is null ? ModelCache.FolderBytes(path) : new FileInfo(path).Length)} in {path} (offline)");
            context.WriteJson(new JsonObject { ["model"] = model, ["path"] = path, ["downloaded"] = 0, ["offline"] = true });
            return ExitCodes.Ok;
        }

        // A progress line per file on a terminal (none with --quiet, --json, --plain or captured output); --timeout applies.
        var downloader = ModelCache.Downloader(context, force);
        if (model.StartsWith("https://", StringComparison.OrdinalIgnoreCase) || model.StartsWith("http://", StringComparison.OrdinalIgnoreCase))
        {
            return Url(context, model, downloader, dryRun);
        }

        return Hub(context, model, downloader, dryRun).GetAwaiter().GetResult();
    }

    // A folder or file on disk: nothing to download; a GGUF file is prepared (its config, tokenizer and template written once).
    private static int Local(CommandContext context, string path, bool dryRun)
    {
        string? prepared = null;
        if (File.Exists(path) && path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) && !dryRun)
        {
            prepared = GgufModel.Prepare(path, context.CacheFolder);
        }

        context.Write($"{path} is on disk; nothing to download.{(prepared is null ? "" : $" Prepared for loading in {prepared}.")}");
        context.WriteJson(new JsonObject { ["model"] = path, ["path"] = Path.GetFullPath(path), ["downloaded"] = 0, ["bytes"] = 0, ["prepared"] = prepared, ["dryRun"] = dryRun });
        return ExitCodes.Ok;
    }

    private static int Url(CommandContext context, string url, Downloader downloader, bool dryRun)
    {
        if (!url.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
        {
            throw new UsageException($"{url} is not a .gguf file; pull a Hugging Face model by its id (owner/name) or a URL ending in .gguf.");
        }

        if (dryRun)
        {
            context.Write($"Would download {url} into {downloader.CacheFolder}.");
            context.WriteJson(new JsonObject { ["model"] = url, ["dryRun"] = true, ["files"] = new JsonArray(new JsonObject { ["name"] = url }) });
            return ExitCodes.Ok;
        }

        var clock = System.Diagnostics.Stopwatch.StartNew();
        string path = downloader.Download(url);
        ModelCache.Touch(context.CacheFolder, path);
        long bytes = new FileInfo(path).Length;
        Report(context, url, null, path, [(Path.GetFileName(path), bytes, true)], clock.Elapsed, Prepare(context, path));
        return ExitCodes.Ok;
    }

    private static async Task<int> Hub(CommandContext context, string model, Downloader downloader, bool dryRun)
    {
        string? file = context.Option("--file");
        string? tag = null;
        string repo = model;
        if (PullTarget.SplitFile(model) is var (r, f))
        {
            (repo, file) = (r, f);
        }
        else
        {
            (repo, tag) = PullTarget.SplitTag(model);
        }

        if (!ModelSource.IsModelId(repo))
        {
            throw new UsageException($"'{model}' is not a folder, a file or a Hugging Face id (owner/name). Check the name, or 'idrak search {model}' to find one.");
        }

        string? token = Shared.Hub.Token(context);
        string revision = context.Option("--revision") ?? "main";
        string commit = await HuggingFace.ResolveRevisionAsync(repo, "models", revision, token, downloader).ConfigureAwait(false);
        var files = await Shared.Hub.FilesAsync(repo, commit, token, downloader).ConfigureAwait(false);
        string folder = downloader.PathFor($"huggingface/models/{repo}/{commit[..Math.Min(12, commit.Length)]}");
        var wanted = file is null && tag is null ? Shared.Hub.TransformersFiles(repo, files) : null;
        bool gguf = wanted is null;
        wanted ??= GgufFiles(repo, files, file, tag);

        var missing = wanted.Where(w => downloader.Refresh || !File.Exists(Path.Combine(folder, w.Path))).ToList();
        long total = wanted.Sum(w => w.Size);
        if (dryRun)
        {
            context.Write($"{repo} at {commit[..Math.Min(12, commit.Length)]}: {wanted.Count} files ({Units.Bytes(total)}), {missing.Count} to download"
                          + (missing.Count > 0 ? $" ({Units.Bytes(missing.Sum(m => m.Size))})" : "") + $" into {folder}");
            context.Table(["File", "Size", "State"], wanted.Select(w => (IReadOnlyList<string>)[w.Path, Units.Bytes(w.Size), missing.Contains(w) ? "to download" : "cached"]));
            context.WriteJson(new JsonObject
            {
                ["model"] = repo,
                ["revision"] = commit,
                ["path"] = folder,
                ["dryRun"] = true,
                ["bytes"] = total,
                ["files"] = new JsonArray([.. wanted.Select(w => (JsonNode)new JsonObject { ["name"] = w.Path, ["bytes"] = w.Size, ["cached"] = !missing.Contains(w) })]),
            });
            return ExitCodes.Ok;
        }

        var clock = System.Diagnostics.Stopwatch.StartNew();
        string? prepared = null;
        if (!gguf)
        {
            // The library's own download, so a pull fetches exactly what loading reads, into the folder loading looks in.
            folder = await ModelSource.DownloadAsync(repo, commit, token, downloader).ConfigureAwait(false);
        }
        else
        {
            foreach (var w in missing)
            {
                await HuggingFace.DownloadFileAsync(repo, w.Path, "models", commit, token, downloader).ConfigureAwait(false);
            }

            prepared = Prepare(context, Path.Combine(folder, wanted[0].Path));
        }

        ModelCache.Touch(context.CacheFolder, gguf ? Path.Combine(folder, wanted[0].Path) : folder);
        Report(context, repo, commit, folder, [.. wanted.Select(w => (w.Path, w.Size, missing.Contains(w)))], clock.Elapsed, prepared);
        return ExitCodes.Ok;
    }

    // The GGUF file(s) to take: --file, or the one whose name holds the tag, or the only one; a split file's parts together.
    private static List<Shared.Hub.HubFile> GgufFiles(string repo, IReadOnlyList<Shared.Hub.HubFile> files, string? file, string? tag)
    {
        var ggufs = files.Where(f => f.Path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)).ToList();
        if (ggufs.Count == 0)
        {
            throw new InvalidOperationException($"{repo} has neither config.json (a transformers model) nor .gguf files; it is not a model Idrak reads.");
        }

        List<Shared.Hub.HubFile> chosen = file is not null
            ? [.. ggufs.Where(f => f.Path.Equals(file, StringComparison.OrdinalIgnoreCase) || Path.GetFileName(f.Path).Equals(file, StringComparison.OrdinalIgnoreCase))]
            : tag is not null
                ? [.. ggufs.Where(f => Path.GetFileName(f.Path).Contains(tag, StringComparison.OrdinalIgnoreCase))]
                : ggufs;
        string list = string.Join(", ", ggufs.Select(g => $"{g.Path} ({Units.Bytes(g.Size)})"));
        if (chosen.Count == 0)
        {
            throw new UsageException($"{repo} has no GGUF file {(file is not null ? $"named {file}" : $"tagged {tag}")}. Its GGUF files: {list}");
        }

        // The parts of one split file (name-00001-of-00003.gguf) belong together; anything else must be one file.
        string Stem(string path) => System.Text.RegularExpressions.Regex.Replace(path, @"-\d{5}-of-\d{5}\.gguf$", "", System.Text.RegularExpressions.RegexOptions.IgnoreCase);
        if (chosen.Select(c => Stem(c.Path)).Distinct(StringComparer.OrdinalIgnoreCase).Count() > 1)
        {
            throw new UsageException($"{repo} has {chosen.Count} GGUF files: {string.Join(", ", chosen.Select(g => $"{g.Path} ({Units.Bytes(g.Size)})"))}. "
                                     + $"Choose one with --file NAME or {repo}:TAG (e.g. {repo}:Q4_K_M).");
        }

        return [.. chosen.OrderBy(c => c.Path, StringComparer.Ordinal)];
    }

    // A GGUF file's description for loading, written once; an architecture Idrak does not read is a warning, not a failure.
    private static string? Prepare(CommandContext context, string gguf)
    {
        try
        {
            return GgufModel.Prepare(gguf, context.CacheFolder);
        }
        catch (Exception e) when (e is NotSupportedException or InvalidDataException or IOException)
        {
            context.Error($"warning: {Path.GetFileName(gguf)} was downloaded but cannot be loaded yet: {e.Message}");
            return null;
        }
    }

    private static void Report(CommandContext context, string model, string? commit, string path, IReadOnlyList<(string Name, long Bytes, bool Downloaded)> files,
        TimeSpan elapsed, string? prepared)
    {
        long bytes = files.Sum(f => f.Bytes), fetched = files.Where(f => f.Downloaded).Sum(f => f.Bytes);
        int downloaded = files.Count(f => f.Downloaded);
        context.Write(downloaded == 0
            ? $"{model} is already in the cache: {files.Count} files, {Units.Bytes(bytes)} in {path}"
            : $"Pulled {model}{(commit is null ? "" : $" ({commit[..Math.Min(12, commit.Length)]})")}: {downloaded} of {files.Count} files downloaded "
              + $"({Units.Bytes(fetched)} in {elapsed.TotalSeconds:F1} s), {Units.Bytes(bytes)} in {path}");
        context.Detail(string.Join(Environment.NewLine, files.Select(f => $"  {f.Name}  {Units.Bytes(f.Bytes)}{(f.Downloaded ? "" : "  (cached)")}")));
        context.WriteJson(new JsonObject
        {
            ["model"] = model,
            ["revision"] = commit,
            ["path"] = path,
            ["prepared"] = prepared,
            ["bytes"] = bytes,
            ["downloaded"] = downloaded,
            ["seconds"] = Math.Round(elapsed.TotalSeconds, 2),
            ["files"] = new JsonArray([.. files.Select(f => (JsonNode)new JsonObject { ["name"] = f.Name, ["bytes"] = f.Bytes, ["downloaded"] = f.Downloaded })]),
        });
    }
}
