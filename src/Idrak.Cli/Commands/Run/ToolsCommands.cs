// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Generation;

namespace Idrak.Cli.Commands.Run;

/// <summary><c>idrak tools list TOOLS.dll</c>: the tools an assembly registers.</summary>
internal sealed class ToolsListCommand : Command
{
    public override string Name => "tools list";

    public override string Summary => "The tools an assembly registers: names, descriptions and parameters";

    public override string Usage => """
        TOOLS.dll [TOOLS.dll ...]

        Tools are the public methods marked [Tool(name, description)] (Idrak.Generation) on the assembly's public types,
        and its public static Tool properties and fields; chat, run and agent take the same assemblies with --tools.

        Examples:
          idrak tools list ./MyTools.dll
          idrak tools list ./MyTools.dll --json
        """ + "\n\nEnvironment:\n  IDRAK_CONFIG, IDRAK_TRACE   the config file, error stacks\n";

    public override int Run(CommandContext context)
    {
        _ = context.Argument(0, "TOOLS.dll");
        var tools = ToolAssemblies.Load(context.Positional);
        context.Table(["tool", "parameters", "description"], tools.Select(t => (IReadOnlyList<string>)
            [t.Definition.Name, Parameters(t.Definition.Parameters), t.Definition.Description ?? ""]));
        context.WriteJson(new JsonObject { ["tools"] = ChatJson.Tools([.. tools.Select(t => t.Definition)]) });
        return ExitCodes.Ok;
    }

    // "city: string, days?: integer" (optional parameters marked with ?).
    internal static string Parameters(JsonNode? schema)
    {
        if (schema?["properties"] is not JsonObject properties || properties.Count == 0)
        {
            return "-";
        }

        var required = (schema["required"] as JsonArray)?.Select(n => (string?)n).ToHashSet() ?? [];
        return string.Join(", ", properties.Select(p => $"{p.Key}{(required.Contains(p.Key) ? "" : "?")}: {p.Value?["type"]?.ToString() ?? "any"}"));
    }
}

/// <summary><c>idrak tools test TOOLS.dll</c>: calls each tool of an assembly with sample arguments.</summary>
internal sealed class ToolsTestCommand : Command
{
    public override string Name => "tools test";

    public override string Summary => "Calls each tool of an assembly with sample arguments and shows the results";

    public override string Usage => """
        TOOLS.dll [TOOLS.dll ...] [options]

        Each tool is called once with arguments made from its schema (defaults, examples, the first enum value, or
        "test", 1, 1.5, true, [] by type), through the same validation the model's calls go through. Exits with 1 when
        a call fails.

        Options:
              --tool NAME        test only this tool (repeatable)
              --args JSON        the arguments to use instead of the samples (with one --tool)
              --timeout S        seconds each call may take (default 30)

        Examples:
          idrak tools test ./MyTools.dll
          idrak tools test ./MyTools.dll --tool get_weather --args '{"city": "Cairo"}'
        """ + "\n\nEnvironment:\n  IDRAK_CONFIG, IDRAK_TRACE   the config file, error stacks\n";

    public override IReadOnlyCollection<string> ValueOptions => ["--tool", "--args", "--timeout"];

    public override int Run(CommandContext context)
    {
        _ = context.Argument(0, "TOOLS.dll");
        var tools = ToolAssemblies.Load(context.Positional);
        var only = context.Options("--tool");
        var unknown = only.Where(n => tools.All(t => t.Definition.Name != n)).ToList();
        if (unknown.Count > 0)
        {
            throw new UsageException($"No tool named {string.Join(", ", unknown)}; the tools are {string.Join(", ", tools.Select(t => t.Definition.Name))}.");
        }

        JsonObject? given = null;
        if (context.Option("--args") is { } args)
        {
            if (only.Count != 1)
            {
                throw new UsageException("--args needs exactly one --tool NAME.");
            }

            given = (JsonNode.Parse(args) as JsonObject) ?? throw new UsageException("--args needs a JSON object.");
        }

        var chosen = tools.Where(t => only.Count == 0 || only.Contains(t.Definition.Name)).ToList();
        var registry = ToolRegistry.Create().Add(chosen).Timeout(TimeSpan.FromSeconds(context.IntOption("--timeout", 30))).Build();
        var results = new JsonArray();
        int failed = 0;
        foreach (var tool in chosen)
        {
            var arguments = given ?? ToolAssemblies.SampleArguments(tool.Definition.Parameters);
            var result = registry.InvokeAsync(new ToolCall(tool.Definition.Name, arguments)).GetAwaiter().GetResult();
            failed += result.Succeeded ? 0 : 1;
            string content = result.Succeeded ? result.Content : result.Error ?? "failed";
            context.Write($"{(result.Succeeded ? "ok  " : "FAIL")} {tool.Definition.Name}({arguments.ToJsonString()}) in {result.Duration.TotalMilliseconds:F0} ms");
            context.Write("     " + (content.Length > 300 ? content[..300] + "..." : content).Replace("\n", "\n     "));
            results.Add((JsonNode)new JsonObject
            {
                ["tool"] = tool.Definition.Name,
                ["arguments"] = arguments.DeepClone(),
                ["succeeded"] = result.Succeeded,
                ["result"] = result.Succeeded ? result.Content : null,
                ["error"] = result.Error,
                ["milliseconds"] = Math.Round(result.Duration.TotalMilliseconds, 2),
            });
        }

        context.Write($"{chosen.Count - failed} of {chosen.Count} tools answered.");
        context.WriteJson(new JsonObject { ["tools"] = chosen.Count, ["failed"] = failed, ["results"] = results });
        return failed == 0 ? ExitCodes.Ok : ExitCodes.Failed;
    }
}
