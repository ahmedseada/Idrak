// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands.Data;

/// <summary>
/// <c>idrak data show|count|download|build|cache</c>: the dataset tool (idrak-data) as a subcommand. The cache is
/// idrak's (<c>--cache</c>, the config or <c>IDRAK_CACHE</c>), downloads under its <c>downloads</c> folder.
/// </summary>
internal sealed class DataCommand : Command
{
    public override string Name => "data";

    public override string Summary => "Datasets: show, count, download, build (and preview, validate, stats, convert, dedupe, split, sample, mix)";

    public override string Usage => "show|count|download|build|cache SPEC... [options]\n\n"
        + DataTool.Usage.Replace("idrak-data: inspect, download and assemble datasets", "Inspect, download and assemble datasets.", StringComparison.Ordinal)
            .Replace("idrak-data ", "idrak data ", StringComparison.Ordinal)
            .Replace("hf:openai/gsm8k", "hf:owner/qa-set", StringComparison.Ordinal)
        + """

        idrak data also takes -o for --out and -s for --system. Its other commands, each with its own help:
          idrak data preview FILE                       first rows, columns and their types
          idrak data validate FILE --as chat|preference|table
          idrak data stats FILE [-m MODEL]              lengths in characters, words and tokens
          idrak data convert IN OUT                     between CSV, JSON Lines and Parquet (read), chat layouts to chat rows
          idrak data dedupe FILE / data split FILE / data sample FILE -n N / data mix RECIPE.json

        Examples:
          idrak data show "hf:owner/qa-set?config=main"
          idrak data build "data.csv?user={question}&assistant={answer}" -o chats.jsonl --eval-fraction 0.02
          idrak data cache --json

        Environment: IDRAK_CACHE, HF_TOKEN, HF_ENDPOINT, GITHUB_TOKEN, KAGGLE_USERNAME, KAGGLE_KEY, ZENODO_TOKEN
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } = [.. DataTool.ValueOptions.Where(o => o != "--cache")];

    public override IReadOnlyCollection<string> Flags => DataTool.Flags;

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-o"] = "--out", ["-s"] = "--system" };

    public override int Run(CommandContext context)
    {
        var args = new List<string>(context.Positional);
        foreach (string option in ValueOptions)
        {
            foreach (string value in context.Options(option))
            {
                args.AddRange([option, value]);
            }
        }

        args.AddRange(Flags.Where(context.Flag));
        args.AddRange(["--cache", context.CacheFolder]);

        var captured = context.Json ? new StringWriter() : null;
        var output = captured ?? (context.Quiet ? TextWriter.Null : context.Output);
        var console = new ToolConsole(output, context.ErrorOutput, Console.In, live: captured is null && !context.Quiet && ReferenceEquals(context.Output, Console.Out));
        var tool = new DataTool(console);
        try
        {
            if (!tool.Parse(args))
            {
                context.Output.Write(Help.For(this));                       // idrak data help
                return ExitCodes.Ok;
            }
        }
        catch (Exception e) when (e is ArgumentException or FormatException)
        {
            throw new UsageException(e.Message);
        }

        if (tool.Problem() is { } problem)
        {
            throw new UsageException(problem);
        }

        int code;
        try
        {
            code = tool.Execute();
        }
        catch (Exception e) when (DataTool.IsDataError(e))
        {
            console.Clear();
            throw;
        }

        if (captured is not null)
        {
            context.WriteJson(new JsonObject
            {
                ["command"] = tool.Positional[0],
                ["specs"] = new JsonArray([.. tool.Positional.Skip(1).Select(s => (JsonNode)s)]),
                ["exitCode"] = code,
                ["output"] = new JsonArray([.. captured.ToString().Split('\n').Select(l => l.TrimEnd('\r')).Where(l => l.Length > 0).Select(l => (JsonNode)l)]),
            });
        }

        return code;
    }
}
