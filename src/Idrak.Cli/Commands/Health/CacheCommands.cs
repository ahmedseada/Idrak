// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands.Health;

/// <summary><c>idrak cache info</c>: the cache folder's parts with their paths and sizes.</summary>
internal sealed class CacheInfoCommand : Command
{
    public override string Name => "cache info";

    public override string Summary => "Sizes and paths of the caches (models, tuning, kernels, downloads)";

    public override string Usage => """

        The cache folder is --cache, the config's "cache", IDRAK_CACHE or ~/.cache/idrak.

        Examples:
          idrak cache info
          idrak cache info -j
          idrak cache info --cache /data/idrak
        """;

    public override int Run(CommandContext context)
    {
        string root = Path.GetFullPath(context.CacheFolder);
        var parts = CacheLayout.Parts.Select(p => (Part: p, Size: CacheLayout.Size(root, p), Paths: CacheLayout.Existing(root, p).ToList())).ToList();
        long total = CacheLayout.Size(root);
        context.Write($"Cache folder: {root}{(Directory.Exists(root) ? "" : " (not created yet)")}");
        context.Table(["Kind", "Size", "Holds", "Paths"], parts.Select(p => (IReadOnlyList<string>)[p.Part.Kind, ProgressLine.Bytes(p.Size), p.Part.Meaning,
            p.Paths.Count == 0 ? "-" : string.Join(", ", p.Paths.Select(x => Path.GetRelativePath(root, x)))]));
        context.Write($"Total: {ProgressLine.Bytes(total)}");
        context.WriteJson(new JsonObject
        {
            ["folder"] = root,
            ["exists"] = Directory.Exists(root),
            ["totalBytes"] = total,
            ["parts"] = new JsonArray([.. parts.Select(p => (JsonNode)new JsonObject
            {
                ["kind"] = p.Part.Kind,
                ["bytes"] = p.Size,
                ["holds"] = p.Part.Meaning,
                ["paths"] = new JsonArray([.. p.Paths.Select(x => (JsonNode)x)]),
            })]),
        });
        return ExitCodes.Ok;
    }
}

/// <summary><c>idrak cache clear [models|tuning|kernels|all]</c>: deletes one part of the cache or all of it.</summary>
internal sealed class CacheClearCommand : Command
{
    private static readonly string[] Kinds = ["models", "tuning", "kernels", "all"];

    public override string Name => "cache clear";

    public override string Summary => "Clears one part of the cache (models, tuning, kernels) or all of it";

    public override string Usage => """
        [models|tuning|kernels|all] [--yes] [--dry-run]

          models     downloaded models
          tuning     kernel choices measured on each device (measured again at the next run)
          kernels    compiled kernels (compiled again when needed)
          all        the whole cache folder (the default)
          -y, --yes  no question
          --dry-run  say what would be deleted, delete nothing

        Examples:
          idrak cache clear tuning
          idrak cache clear --dry-run
          idrak cache clear models -y
        """;

    public override IReadOnlyCollection<string> Flags => Terminal.ConfirmFlags;

    public override IReadOnlyDictionary<string, string> ShortForms => Terminal.ConfirmShortForms;

    public override int Run(CommandContext context)
    {
        string kind = context.Positional.Count > 0 ? context.Positional[0] : "all";
        if (!Kinds.Contains(kind) || context.Positional.Count > 1)
        {
            throw new UsageException($"cache clear takes one of {string.Join(", ", Kinds)}, not '{string.Join(' ', context.Positional)}'.");
        }

        string root = Path.GetFullPath(context.CacheFolder);
        var paths = kind == "all" ? (Directory.Exists(root) ? [root] : new List<string>())
            : CacheLayout.Existing(root, CacheLayout.Parts.First(p => p.Kind == kind)).ToList();
        long size = paths.Sum(CacheLayout.Size);
        var json = new JsonObject { ["kind"] = kind, ["paths"] = new JsonArray([.. paths.Select(p => (JsonNode)p)]), ["bytes"] = size };
        if (paths.Count == 0)
        {
            context.Write($"Nothing to clear: no {kind} cache under {root}.");
            json["cleared"] = false;
            context.WriteJson(json);
            return ExitCodes.Ok;
        }

        if (Terminal.DryRun(context))
        {
            context.Write($"Would delete {ProgressLine.Bytes(size)}:");
            paths.ForEach(p => context.Write($"  {p}"));
            json["cleared"] = false;
            json["dryRun"] = true;
            context.WriteJson(json);
            return ExitCodes.Ok;
        }

        if (!Terminal.Confirm(context, $"Delete the {kind} cache ({ProgressLine.Bytes(size)} in {string.Join(", ", paths)})?"))
        {
            context.Write("Nothing deleted.");
            json["cleared"] = false;
            context.WriteJson(json);
            return ExitCodes.Ok;
        }

        long freed = paths.Sum(CacheLayout.Delete);
        context.Write($"Freed {ProgressLine.Bytes(freed)}.");
        json["cleared"] = true;
        context.WriteJson(json);
        return ExitCodes.Ok;
    }
}
