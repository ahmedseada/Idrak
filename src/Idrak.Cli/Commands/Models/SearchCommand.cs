// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Models.Abstractions;

namespace Idrak.Cli.Commands;

/// <summary>
/// <c>idrak search QUERY</c>: searches the Hugging Face hub and keeps the models Idrak can load: transformers models with
/// safetensors weights of a registered family, and GGUF files of a registered GGUF architecture.
/// </summary>
internal sealed class SearchCommand : Command
{
    public override string Name => "search";

    public override string Summary => "Search the Hugging Face hub for models Idrak can load";

    public override string Usage => """
        QUERY [--limit N] [--kind safetensors|gguf] [--all] [--token TOKEN]

        Arguments:
          QUERY  words in the model's name (e.g. qwen3 0.6b)

        Options:
              --limit N      at most N results (default 20)
              --kind K       only transformers models with safetensors weights, or only GGUF repositories
              --all          also models Idrak cannot load (marked)
              --token TOKEN  for private models (default: the config's hf_token, HF_TOKEN or the saved login)

        A model is loadable when its family (config.json "architectures") is registered, or, for GGUF, its
        architecture is; plug-ins (--plugin) add families. Then: idrak pull NAME.

        Examples:
          idrak search qwen3
          idrak search llama --kind gguf --limit 5 --json
        """;

    public override IReadOnlyCollection<string> ValueOptions => ["--limit", "--kind", "--token"];

    public override IReadOnlyCollection<string> Flags => ["--all"];

    public override int Run(CommandContext context)
    {
        if (context.Positional.Count == 0)
        {
            throw new UsageException("Missing QUERY (words in the model's name, e.g. qwen3).");
        }

        if (context.Offline)
        {
            throw new InvalidOperationException("search asks the Hugging Face hub, and --offline allows no network; 'idrak list' shows the cached models.");
        }

        string query = string.Join(' ', context.Positional);
        int limit = context.IntOption("--limit", 20);
        if (limit <= 0)
        {
            throw new UsageException("--limit needs a positive number.");
        }

        string? format = context.Option("--kind")?.ToLowerInvariant();
        if (format is not (null or "safetensors" or "gguf"))
        {
            throw new UsageException($"--kind takes safetensors or gguf, not '{format}'.");
        }

        bool all = context.Flag("--all");
        var found = Shared.Hub.SearchAsync(query, Math.Min(1000, all ? limit : Math.Max(50, limit * 5)), Shared.Hub.Token(context), ModelCache.Downloader(context))
            .GetAwaiter().GetResult();
        var families = PretrainedArchitectures.Names.ToHashSet(StringComparer.Ordinal);
        var ggufArchitectures = GgufArchitectures.Names.ToHashSet(StringComparer.OrdinalIgnoreCase);
        var rows = new List<Result>();
        foreach (var item in found.OfType<JsonObject>())
        {
            string id = (string?)item["id"] ?? (string?)item["modelId"] ?? "?";
            var tags = (item["tags"] as JsonArray ?? []).Select(t => (string?)t).OfType<string>().ToHashSet(StringComparer.OrdinalIgnoreCase);
            string? family = (string?)item["config"]?["architectures"]?[0];
            string? ggufArchitecture = (string?)item["gguf"]?["architecture"];
            bool isGguf = tags.Contains("gguf") || ggufArchitecture is not null;
            bool isSafetensors = tags.Contains("safetensors") && !isGguf;
            string kind = isGguf ? "gguf" : isSafetensors ? "safetensors" : "other";
            if (format is not null && kind != format)
            {
                continue;
            }

            string? why = isGguf
                ? ggufArchitecture is null ? "GGUF architecture not reported" : ggufArchitectures.Contains(ggufArchitecture) ? null : $"GGUF architecture {ggufArchitecture} not registered"
                : isSafetensors
                    ? family is null ? "no family in config.json" : families.Contains(family) ? null : $"family {family} not registered"
                    : "neither safetensors nor GGUF weights";
            if (why is not null && !all)
            {
                continue;
            }

            long downloads = item["downloads"] is JsonValue d && d.TryGetValue(out long n) ? n : 0;
            long likes = item["likes"] is JsonValue l && l.TryGetValue(out long k) ? k : 0;
            rows.Add(new Result(id, kind, isGguf ? ggufArchitecture : family, downloads, likes, why));
            if (rows.Count == limit)
            {
                break;
            }
        }

        if (rows.Count == 0)
        {
            context.Write($"No loadable models match '{query}'{(all ? "" : " (--all shows the others)")}.");
        }
        else
        {
            context.Table(["Model", "Format", "Family", "Downloads", "Likes", "Loadable"],
                rows.Select(r => (IReadOnlyList<string>)[r.Id, r.Kind, r.Family ?? "-", r.Downloads.ToString("N0", System.Globalization.CultureInfo.InvariantCulture),
                    r.Likes.ToString("N0", System.Globalization.CultureInfo.InvariantCulture), r.Problem is null ? "yes" : $"no: {r.Problem}"]));
            context.Write($"Download one with: idrak pull {rows[0].Id}{(rows[0].Kind == "gguf" ? ":Q4_K_M" : "")}");
        }

        context.WriteJson(new JsonObject
        {
            ["query"] = query,
            ["models"] = new JsonArray([.. rows.Select(r => (JsonNode)new JsonObject
            {
                ["id"] = r.Id,
                ["format"] = r.Kind,
                ["family"] = r.Family,
                ["downloads"] = r.Downloads,
                ["likes"] = r.Likes,
                ["loadable"] = r.Problem is null,
                ["problem"] = r.Problem,
            })]),
        });
        return ExitCodes.Ok;
    }

    private sealed record Result(string Id, string Kind, string? Family, long Downloads, long Likes, string? Problem);
}
