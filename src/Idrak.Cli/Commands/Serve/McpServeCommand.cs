// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Linq.Expressions;
using System.Reflection;
using System.Text.Json.Nodes;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.Server;
using Idrak.Generation;
using Idrak.Cli.Shared;
using Idrak.Mcp;
using Tool = Idrak.Generation.Tool;

namespace Idrak.Cli.Commands.Serve;

/// <summary><c>idrak mcp serve TOOLS.dll [...]</c>: serves an assembly's tools to MCP clients over standard input/output.</summary>
internal sealed class McpServeCommand : Command
{
    public override string Name => "mcp serve";

    public override string Summary => "Serve the tools of an assembly over MCP (standard input/output)";

    public override string Usage => """
        TOOLS.dll [TOOLS.dll...] [options]

        Loads each assembly and serves every public method marked [Tool] (Idrak.Generation.ToolAttribute) on its public
        types (instance methods need a parameterless constructor) to an MCP client over standard input and output,
        until the client disconnects. Status lines go to the error output, since the output carries the protocol.
              --list   print the tools (name and description) and exit, without serving

        Examples:
          idrak mcp serve ./MyTools.dll
          idrak mcp serve ./MyTools.dll --list --json
        A client's configuration starts it as: {"command": "idrak", "args": ["mcp", "serve", "/path/MyTools.dll"]}
        """;

    public override IReadOnlyCollection<string> Flags => ["--list"];

    /// <summary>The protocol's input and output (tests replace them with pipes).</summary>
    internal static Func<(Stream Input, Stream Output)> Streams { get; set; } = () => (Console.OpenStandardInput(), Console.OpenStandardOutput());

    public override int Run(CommandContext context)
    {
        if (context.Positional.Count == 0)
        {
            throw new UsageException("Name at least one assembly (TOOLS.dll) whose [Tool] methods to serve.");
        }

        var builder = ToolRegistry.Create();
        int count = 0;
        var listed = new JsonArray();
        foreach (string path in context.Positional)
        {
            foreach (var tool in ToolAssemblies.Load(path))
            {
                builder.Add(tool);
                listed.Add(new JsonObject { ["name"] = tool.Definition.Name, ["description"] = tool.Definition.Description, ["assembly"] = Path.GetFileName(path) });
                count++;
            }
        }

        if (count == 0)
        {
            throw new InvalidOperationException($"no public method marked [Tool] was found in {string.Join(", ", context.Positional)}.");
        }

        if (context.Flag("--list"))
        {
            context.Table(["Tool", "Description"], listed.Select(t => (IReadOnlyList<string>)[(string)t!["name"]!, (string?)t["description"] ?? ""]));
            context.WriteJson(new JsonObject { ["tools"] = listed });
            return ExitCodes.Ok;
        }

        var registry = builder.Build();
        var options = new McpServerOptions { ServerInfo = new Implementation { Name = "idrak", Version = typeof(Tensor).Assembly.GetName().Version?.ToString() ?? "0" }, ToolCollection = [] };
        foreach (var tool in McpTools.ServerTools(registry))
        {
            options.ToolCollection.Add(tool);
        }

        var (input, output) = Streams();
        if (!context.Quiet)
        {
            context.Error($"Serving {count} tool{(count == 1 ? "" : "s")} over MCP on standard input/output: {string.Join(", ", registry.Definitions.Select(d => d.Name))}");
        }

        using var interrupt = new Interrupt(context);           // Ctrl+C or --timeout ends the session
        RunServer(input, output, options, interrupt.Token).GetAwaiter().GetResult();

        return ExitCodes.Ok;
    }

    private static async Task RunServer(Stream input, Stream output, McpServerOptions options, CancellationToken token)
    {
        await using var server = McpServer.Create(new StreamServerTransport(input, output, "idrak"), options);
        try
        {
            await server.RunAsync(token);
        }
        catch (OperationCanceledException)
        {
            // Ctrl+C
        }
    }
}

/// <summary>Reads the [Tool] methods of an assembly's public types as tools.</summary>
internal static class ToolAssemblies
{
    public static IReadOnlyList<Tool> Load(string path)
    {
        string full = Path.GetFullPath(path);
        if (!File.Exists(full))
        {
            throw new UsageException($"Tools assembly not found: {full}");
        }

        var assembly = Assembly.LoadFrom(full);
        var tools = new List<Tool>();
        foreach (var type in assembly.GetExportedTypes().Where(t => t.IsClass && !t.IsGenericTypeDefinition))
        {
            var methods = type.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly)
                .Select(m => (Method: m, Attribute: m.GetCustomAttribute<ToolAttribute>())).Where(m => m.Attribute is not null).ToList();
            if (methods.Count == 0)
            {
                continue;
            }

            object? instance = null;
            if (methods.Any(m => !m.Method.IsStatic))
            {
                instance = type.GetConstructor(Type.EmptyTypes) is { } constructor ? constructor.Invoke(null)
                    : throw new InvalidOperationException($"{type.FullName} has [Tool] instance methods but no public parameterless constructor.");
            }

            foreach (var (method, attribute) in methods)
            {
                var delegateType = Expression.GetDelegateType([.. method.GetParameters().Select(p => p.ParameterType), method.ReturnType]);
                var function = method.IsStatic ? method.CreateDelegate(delegateType) : method.CreateDelegate(delegateType, instance);
                tools.Add(Tool.FromDelegate(attribute!.Name, attribute.Description, function));
            }
        }

        return tools;
    }
}
