// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands;

/// <summary><c>idrak rm MODEL</c>: removes cached models (every revision, or one), GGUF files and partial downloads.</summary>
internal sealed class RmCommand : Command
{
    public override string Name => "rm";

    public override string Summary => "Remove a cached model";

    public override string Usage => """
        MODEL... [-y] [--dry-run] [-f]

        Arguments:
          MODEL  a name as 'idrak list' shows it: owner/name (every revision), owner/name@REVISION,
                 owner/name/FILE.gguf, a partial download, or the path of a cached folder or file

        Options:
          -y, --yes      remove without asking
              --dry-run  show what would be removed
          -f, --force    no error when a name matches nothing

        A GGUF file's prepared folders go with it. Models in Hugging Face's own cache are not removed (another tool
        owns them).

        Examples:
          idrak rm Qwen/Qwen3-0.6B --dry-run
          idrak rm Qwen/Qwen3-0.6B -y
        """;

    public override IReadOnlyCollection<string> Flags => ["--yes", "--dry-run", "--force"];

    public override IReadOnlyDictionary<string, string> ShortForms => new Dictionary<string, string> { ["-y"] = "--yes", ["-f"] = "--force" };

    public override int Run(CommandContext context)
    {
        if (context.Positional.Count == 0)
        {
            throw new UsageException("Missing MODEL (a name as 'idrak list' shows it).");
        }

        var all = ModelCache.Scan(context.CacheFolder, all: true);
        var chosen = new List<ModelCache.Entry>();
        foreach (string name in context.Positional)
        {
            string model = context.Config.Object("aliases")?[name] is JsonObject ? Shared.Models.Choose(context, name).Model : name;
            var matches = ModelCache.Match(all, model);
            if (matches.Count == 0)
            {
                // A GGUF repository pulled as owner/name: its files.
                matches = [.. all.Where(e => e.Kind is "gguf" or "partial" && e.Name.StartsWith(model.TrimEnd('/') + "/", StringComparison.OrdinalIgnoreCase))];
            }

            if (matches.Count == 0 && !context.Flag("--force"))
            {
                throw new InvalidOperationException($"No cached model named '{model}'. 'idrak list' shows the names (or pass --force to ignore).");
            }

            if (matches.Count > 0 && matches.All(m => !m.Removable))
            {
                throw new InvalidOperationException($"'{model}' is in Hugging Face's own cache ({matches[0].Path}); remove it with that tool, not idrak.");
            }

            chosen.AddRange(matches.Where(m => m.Removable && !chosen.Contains(m)));
        }

        // A GGUF file takes its prepared folders with it.
        foreach (var gguf in chosen.Where(c => c.Kind == "gguf").ToList())
        {
            chosen.AddRange(all.Where(e => e.Kind == "prepared" && !chosen.Contains(e)
                                           && ModelCache.SourceOf(e.Path) is { } source && Path.GetFullPath(source) == Path.GetFullPath(gguf.Path)));
        }

        long bytes = chosen.Sum(c => c.Bytes);
        bool dryRun = Terminal.DryRun(context);
        foreach (var c in chosen)
        {
            context.Write($"{(dryRun ? "would remove" : "removing")} {c.Name}{(c.Revision is null || c.Kind == "gguf" ? "" : "@" + c.Revision)}  {Units.Bytes(c.Bytes)}  {c.Path}");
        }

        bool go = !dryRun && chosen.Count > 0
                  && Terminal.Confirm(context, $"Remove {chosen.Count} {(chosen.Count == 1 ? "entry" : "entries")} ({Units.Bytes(bytes)})?");
        if (go)
        {
            foreach (var c in chosen)
            {
                Delete(c.Path);
                ModelCache.Forget(context.CacheFolder, c.Path);
                PruneEmpty(Path.GetDirectoryName(c.Path)!, context.CacheFolder);
            }
        }

        context.Write(dryRun ? $"{chosen.Count} would be removed, {Units.Bytes(bytes)} freed (dry run)."
            : go ? $"Removed {chosen.Count}, {Units.Bytes(bytes)} freed."
            : chosen.Count == 0 ? "Nothing to remove." : "Nothing removed.");
        context.WriteJson(new JsonObject
        {
            ["dryRun"] = dryRun,
            ["removed"] = go || dryRun ? chosen.Count : 0,
            ["bytes"] = go || dryRun ? bytes : 0,
            ["entries"] = new JsonArray([.. chosen.Select(c => (JsonNode)new JsonObject { ["name"] = c.Name, ["kind"] = c.Kind, ["revision"] = c.Revision, ["bytes"] = c.Bytes, ["path"] = c.Path })]),
        });
        return ExitCodes.Ok;
    }

    private static void Delete(string path)
    {
        if (Directory.Exists(path))
        {
            Directory.Delete(path, recursive: true);
        }
        else if (File.Exists(path))
        {
            File.Delete(path);
            if (File.Exists(path + ".part"))
            {
                File.Delete(path + ".part");
            }
        }
    }

    // Empty owner and name folders left behind, up to (not including) the cache folder.
    private static void PruneEmpty(string folder, string cacheFolder)
    {
        string root = Path.GetFullPath(cacheFolder);
        for (var dir = new DirectoryInfo(folder); dir is not null && dir.FullName.Length > root.Length && dir.FullName.StartsWith(root, StringComparison.Ordinal); dir = dir.Parent)
        {
            if (!dir.Exists || dir.EnumerateFileSystemInfos().Any())
            {
                break;
            }

            dir.Delete();
        }
    }
}
