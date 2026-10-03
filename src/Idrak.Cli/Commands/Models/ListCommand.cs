// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands;

/// <summary><c>idrak list</c> (<c>ls</c>): the models in the cache with their sizes, formats and last use.</summary>
internal sealed class ListCommand : Command
{
    public override string Name => "list";

    public override IReadOnlyCollection<string> Aliases => ["ls"];

    public override string Summary => "Cached models with sizes, formats and last use";

    public override string Usage => """
        [FILTER] [--all] [--sort used|name|size]

        Arguments:
          FILTER  only models whose name contains this text

        Options:
              --all       also the folders prepared from GGUF files and the models in Hugging Face's own cache
              --sort KEY  used (most recent first, the default), name or size (largest first)

        Examples:
          idrak ls
          idrak list qwen --sort size
          idrak list --all --json
        """;

    public override IReadOnlyCollection<string> ValueOptions => ["--sort"];

    public override IReadOnlyCollection<string> Flags => ["--all"];

    public override int Run(CommandContext context)
    {
        string? filter = context.Positional.Count > 0 ? context.Positional[0] : null;
        if (context.Positional.Count > 1)
        {
            throw new UsageException($"list takes one filter; '{context.Positional[1]}' is extra.");
        }

        var entries = ModelCache.Scan(context.CacheFolder, context.Flag("--all"))
            .Where(e => filter is null || e.Name.Contains(filter, StringComparison.OrdinalIgnoreCase)).ToList();
        entries = (context.Option("--sort") ?? "used") switch
        {
            "used" => entries,
            "name" => [.. entries.OrderBy(e => e.Name, StringComparer.OrdinalIgnoreCase)],
            "size" => [.. entries.OrderByDescending(e => e.Bytes)],
            var other => throw new UsageException($"--sort takes used, name or size, not '{other}'."),
        };

        string[] headers = ["Model", "Size", "Format", "Last use", "Kind"];
        if (entries.Count == 0)
        {
            context.Write(filter is null
                ? $"No models in {context.CacheFolder}. Download one with: idrak pull Qwen/Qwen3-0.6B"
                : $"No cached model matches '{filter}'.");
            if (context.Format is OutputFormat.Csv)
            {
                context.Table(headers, []);                                     // the header line, for scripts
            }
        }
        else
        {
            context.Table(headers,
                entries.Select(e => (IReadOnlyList<string>)[e.Name + (e.Revision is null || e.Kind == "gguf" ? "" : $"@{e.Revision}"), Units.Bytes(e.Bytes), e.Format,
                    e.LastUse == DateTime.MinValue ? "-" : Units.Ago(e.LastUse), e.Kind]));
            context.Write($"{entries.Count} {(entries.Count == 1 ? "entry" : "entries")}, {Units.Bytes(entries.Sum(e => e.Bytes))} in {context.CacheFolder}");
        }

        foreach (var e in entries)
        {
            context.Detail($"  {e.Name}: {e.Path}");
        }

        context.WriteJson(new JsonObject
        {
            ["cache"] = context.CacheFolder,
            ["bytes"] = entries.Sum(e => e.Bytes),
            ["models"] = new JsonArray([.. entries.Select(e => (JsonNode)new JsonObject
            {
                ["name"] = e.Name,
                ["kind"] = e.Kind,
                ["revision"] = e.Revision,
                ["format"] = e.Format,
                ["bytes"] = e.Bytes,
                ["lastUse"] = e.LastUse == DateTime.MinValue ? null : e.LastUse.ToString("o", CultureInfo.InvariantCulture),
                ["path"] = e.Path,
                ["removable"] = e.Removable,
            })]),
        });
        return ExitCodes.Ok;
    }
}
