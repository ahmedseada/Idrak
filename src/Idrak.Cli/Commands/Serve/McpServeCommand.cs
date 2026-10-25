// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.Server;
using Idrak.Cli.Shared;
using Idrak.Mcp;

namespace Idrak.Cli.Commands.Serve;

/// <summary><c>idrak mcp serve TOOLS.dll [...]</c>: serves an assembly's tools to MCP clients over standard input/output.</summary>
internal sealed class McpServeCommand : Command
{
    public override string Name => "mcp serve";

    public override string Summary => "Serve the tools of an assembly over MCP (standard input/output)";

    public override string Usage => """
        TOOLS.dll [TOOLS.dll...] [options]

        Loads each assembly and serves the tools it declares ([assembly: ToolProvider(typeof(X))], X an IToolProvider
        of Idrak.Abstraction.Generation, whose tools are ITool: its [Tool] methods or types of its own) to an MCP
        client over standard input and output, until the client disconnects. Status lines go to the error output,
        since the output carries the protocol. To serve models as MCP tools too (generate, chat): idrak serve MODEL
        --mcp [--tools TOOLS.dll].

        Options:
              --list  print the tools (name and description) and exit, without serving

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
            throw new UsageException("Name at least one assembly (TOOLS.dll) whose tools to serve.");
        }

        var builder = ToolRegistry.Create();
        var listed = new JsonArray();
        foreach (string path in context.Positional)
        {
            foreach (var tool in ToolAssemblies.Load(path))
            {
                builder.Add(tool);
                listed.Add(new JsonObject { ["name"] = tool.Definition.Name, ["description"] = tool.Definition.Description, ["assembly"] = Path.GetFileName(path) });
            }
        }

        if (context.Flag("--list"))
        {
            context.Table(["Tool", "Description"], listed.Select(t => (IReadOnlyList<string>)[(string)t!["name"]!, (string?)t["description"] ?? ""]));
            context.WriteJson(new JsonObject { ["tools"] = listed });
            return ExitCodes.Ok;
        }

        return Serve(context, builder.Build());
    }

    /// <summary>
    /// Serves <paramref name="registry"/> to an MCP client over <see cref="Streams"/> until the client disconnects, or
    /// Ctrl+C or --timeout ends the session (also used by <c>idrak serve --mcp</c>). The status line goes to the error output.
    /// </summary>
    public static int Serve(CommandContext context, IToolRegistry registry)
    {
        var options = new McpServerOptions { ServerInfo = new Implementation { Name = "idrak", Version = typeof(Idrak.Layers.Sequential).Assembly.GetName().Version?.ToString() ?? "0" }, ToolCollection = [] };
        foreach (var tool in McpTools.ServerTools(registry))
        {
            options.ToolCollection.Add(tool);
        }

        var (input, output) = Streams();
        if (!context.Quiet)
        {
            int count = registry.Definitions.Count;
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
