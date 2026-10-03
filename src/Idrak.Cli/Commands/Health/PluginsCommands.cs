// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Cli.Commands.Health;

/// <summary>
/// <c>idrak plugins list</c>: everything registered (formats, families, RoPE scalings, tool-call formats, graph ops,
/// layer types, ONNX ops, devices); with <c>--plugin</c>, what each loaded assembly added.
/// </summary>
internal sealed class PluginsListCommand : Command
{
    private IReadOnlyList<Registries.Category>? _before;

    public override string Name => "plugins list";

    public override string Summary => "Everything registered: formats, families, RoPE scalings, tool-call formats, ops, devices";

    public override string Usage => """
        [--added]

          --added   only what the --plugin assemblies (and the config's "plugins") added

        With --plugin PATH, each name an assembly added is marked with "+".

        Examples:
          idrak plugins list
          idrak plugins list -P ./MyFormat.dll --added
          idrak plugins list -j
        """;

    public override IReadOnlyCollection<string> Flags => ["--added"];

    public override void BeforePlugins(CommandContext context) => _before = Registries.Snapshot();

    public override int Run(CommandContext context)
    {
        var all = Registries.Snapshot();
        var added = Registries.Added(_before ?? all, all);
        bool onlyAdded = context.Flag("--added");
        Print(context, onlyAdded ? added : all, added);
        context.WriteJson(Json(onlyAdded ? added : all, added));
        return ExitCodes.Ok;
    }

    /// <summary>One section per category: its title and the names, those added marked with "+".</summary>
    internal static void Print(CommandContext context, IReadOnlyList<Registries.Category> categories, IReadOnlyList<Registries.Category> added)
    {
        if (context.Format is OutputFormat.Csv or OutputFormat.Markdown)
        {
            context.Table(["Kind", "Count", "Names"], categories.Where(c => c.Names.Count > 0).Select(c => (IReadOnlyList<string>)
                [c.Title, c.Names.Count.ToString(System.Globalization.CultureInfo.InvariantCulture),
                 string.Join(", ", c.Names.Select(n => added.First(a => a.Key == c.Key).Names.Contains(n) ? $"+{n}" : n))]));
            return;
        }

        foreach (var c in categories.Where(c => c.Names.Count > 0))
        {
            var mine = added.First(a => a.Key == c.Key).Names;
            context.Write($"{c.Title} ({c.Names.Count}):");
            context.Write("  " + string.Join(", ", c.Names.Select(n => mine.Contains(n) ? $"+{n}" : n)));
        }

        if (added.Any(a => a.Names.Count > 0))
        {
            context.Write("+ added by a plug-in");
        }
    }

    internal static JsonObject Json(IReadOnlyList<Registries.Category> categories, IReadOnlyList<Registries.Category> added) =>
        new([.. categories.Select(c => KeyValuePair.Create(c.Key, (JsonNode?)new JsonObject
        {
            ["title"] = c.Title,
            ["names"] = new JsonArray([.. c.Names.Select(n => (JsonNode)n)]),
            ["added"] = new JsonArray([.. added.First(a => a.Key == c.Key).Names.Select(n => (JsonNode)n)]),
        }))]);
}

/// <summary><c>idrak formats</c>: every registered weight, KV cache, checkpoint, dataset and tool-call format.</summary>
internal sealed class FormatsCommand : Command
{
    private IReadOnlyList<Registries.Category>? _before;

    public override string Name => "formats";

    public override string Summary => "Every registered weight, KV cache, checkpoint, dataset and tool-call format";

    public override string Usage => """

        Built-in formats and those a plug-in registers (marked "+" when loaded with --plugin). Weight formats go to
        -w/--weights, KV cache formats to -k/--kv.

        Examples:
          idrak formats
          idrak formats -P ./MyFormat.dll
          idrak formats --format md
        """;

    public override void BeforePlugins(CommandContext context) => _before = Registries.Snapshot();

    public override int Run(CommandContext context)
    {
        var all = Registries.Snapshot().Where(c => c.Format).ToList();
        var added = Registries.Added(_before ?? all, all);
        context.Table(["Kind", "Formats"], all.Select(c => (IReadOnlyList<string>)[c.Title,
            string.Join(", ", c.Names.Select(n => added.First(a => a.Key == c.Key).Names.Contains(n) ? $"+{n}" : n))]));
        context.WriteJson(PluginsListCommand.Json(all, added));
        return ExitCodes.Ok;
    }
}
