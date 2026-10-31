// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Mcp;
using Idrak.Mcp.Abstractions;

namespace Idrak.Cli.Commands.Serve;

/// <summary><c>idrak mcp serve TOOLS.dll [...]</c>: serves an assembly's tools to MCP clients over a transport (standard input/output, or http).</summary>
internal sealed class McpServeCommand : Command
{
    public override string Name => "mcp serve";

    public override string Summary => "Serve the tools of an assembly over MCP (standard input/output, or http)";

    /// <inheritdoc />
    public override CommandNeeds Needs => CommandNeeds.ConsoleInput;

    /// <inheritdoc />
    public override CommandNeeds NeedsFor(CommandContext context) => NeedsOf(context);

    /// <summary>What serving MCP needs with the command's transport: the console's input for stdio, a listener for any other.</summary>
    public static CommandNeeds NeedsOf(CommandContext context) =>
        TransportName(context).Equals(McpServerTransports.Stdio, StringComparison.OrdinalIgnoreCase) ? CommandNeeds.ConsoleInput : CommandNeeds.NetworkListener;

    /// <summary>The options that choose the transport and where it listens (mcp serve, and serve with --mcp).</summary>
    public const string TransportHelp = """
              --transport NAME       how clients connect: stdio (default: the client starts this command and talks
                                     over its input and output) or http (streamable HTTP on this machine, for a
                                     client that connects to a running server); or a registered one
              --mcp-address H:P      where http listens (default 127.0.0.1:0: a port the system picks, printed)
              --mcp-token TOKEN      the bearer token http requires (default: a random one, printed); none: no token
              --mcp-allow-remote     let http listen on an address other machines reach (off: loopback only)
              --mcp-max-sessions N   the most http sessions open at once (default 1024; 0: no limit); a new
                                     one over it is answered 503 with Retry-After (clients of the 2026-07-28
                                     revision hold no session and are not counted)
              --mcp-replace-idle     at that limit, close the idle session unused the longest for the new one
              --verbose              print each session, request (method, tool, status, bytes, time), tool call
                                     and refusal on the error output (never on the output: over stdio it is
                                     the protocol); --quiet keeps them and drops the status lines
              --profile              at exit, the profile report with the session's requests and tool calls by
                                     method and tool (counts, errors, p50 and p99 times, bytes) and refusals
        """;

    /// <summary>The value options of <see cref="TransportHelp"/>.</summary>
    public static IReadOnlyCollection<string> TransportOptions => ["--transport", "--mcp-address", "--mcp-token", "--mcp-max-sessions"];

    /// <summary>The flags of <see cref="TransportHelp"/>.</summary>
    public static IReadOnlyCollection<string> TransportFlags => ["--mcp-allow-remote", "--mcp-replace-idle"];

    public override string Usage => $$"""
        TOOLS.dll [TOOLS.dll...] [options]

        Loads each assembly and serves the tools it declares ([assembly: ToolProvider(typeof(X))], X an IToolProvider
        of Idrak.Abstraction.Generation, whose tools are ITool: its [Tool] methods or types of its own) to MCP
        clients. Over stdio (the default) it serves the client that started it, until that client disconnects; over
        http it serves any client on this machine that sends its token, until Ctrl+C or --timeout, and prints its
        address and token. Status lines go to the error output, since the output carries the protocol over stdio.
        To serve models as MCP tools too (generate, chat): idrak serve MODEL --mcp [--tools TOOLS.dll].

        Options:
              --list                 print the tools (name and description) and exit, without serving
        {{TransportHelp}}

        Examples:
          idrak mcp serve ./MyTools.dll
          idrak mcp serve ./MyTools.dll --list --json
          idrak mcp serve ./MyTools.dll --transport http --mcp-address 127.0.0.1:8765
        A client's configuration starts it as: {"command": "idrak", "args": ["mcp", "serve", "/path/MyTools.dll"]};
        over http it takes the printed URL and sends "Authorization: Bearer TOKEN" (idrak chat MODEL --mcp URL
        --mcp-token TOKEN).
        """;

    public override IReadOnlyCollection<string> ValueOptions => TransportOptions;

    public override IReadOnlyCollection<string> Flags => ["--list", .. TransportFlags];

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
                listed.Add(item: new JsonObject { ["name"] = tool.Definition.Name, ["description"] = tool.Definition.Description, ["assembly"] = Path.GetFileName(path) });
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

    /// <summary>The transport the command asked for: --transport, else IDRAK_MCP_TRANSPORT, else stdio.</summary>
    public static string TransportName(CommandContext context) => context.Option("--transport") is { Length: > 0 } name ? name : McpServerTransports.DefaultName;

    /// <summary>
    /// How the command's host listens: --mcp-address, --mcp-token and --mcp-max-sessions over IDRAK_MCP_ADDRESS,
    /// IDRAK_MCP_TOKEN and IDRAK_MCP_MAX_SESSIONS, --mcp-replace-idle over IDRAK_MCP_REPLACE_IDLE, --mcp-allow-remote, and
    /// <see cref="Streams"/> for stdio.
    /// </summary>
    public static McpHostOptions HostOptions(CommandContext context, string transport)
    {
        McpHostOptions environment;
        try
        {
            environment = McpHostOptions.FromEnvironment();
        }
        catch (ArgumentException e)
        {
            throw new UsageException(e.Message);
        }

        var options = environment.WithToken(context.Option("--mcp-token")) with
        {
            AllowRemote = context.Flag("--mcp-allow-remote"),
            ReplaceIdleSessions = environment.ReplaceIdleSessions || context.Flag("--mcp-replace-idle"),
            ServerVersion = typeof(Idrak.Layers.Sequential).Assembly.GetName().Version?.ToString(),
        };
        if (context.Option("--mcp-max-sessions") is not null)
        {
            int max = context.IntOption("--mcp-max-sessions", options.MaxSessions);
            options = options with { MaxSessions = max >= 0 ? max : throw new UsageException("--mcp-max-sessions takes 0 (no limit) or more.") };
        }

        if (context.Option("--mcp-address") is { Length: > 0 } address)
        {
            options = options with { Address = address };
        }

        if (transport.Equals(McpServerTransports.Stdio, StringComparison.OrdinalIgnoreCase))
        {
            var (input, output) = Streams();
            options = options with { Input = input, Output = output };
        }

        return options;
    }

    /// <summary>
    /// Serves <paramref name="registry"/> to MCP clients over the command's transport (<see cref="TransportName"/>) until
    /// the stdio client disconnects, or Ctrl+C or --timeout ends it (also used by <c>idrak serve --mcp</c>). The status
    /// line, with an http server's address and token, goes to the error output.
    /// </summary>
    public static int Serve(CommandContext context, IToolRegistry registry)
    {
        string transport = TransportName(context);
        var options = HostOptions(context, transport);
        using var interrupt = new Interrupt(context);           // Ctrl+C or --timeout ends the session
        McpHost host;
        try
        {
            host = McpHost.StartAsync(transport, registry, options, interrupt.Token).GetAwaiter().GetResult();
        }
        catch (Exception e) when (e is ArgumentException or NotRegisteredException)
        {
            throw new UsageException(e.Message);                 // a bad address, a host off this machine, a transport not registered
        }

        try
        {
            int count = registry.Definitions.Count;
            string tools = $"{count} tool{(count == 1 ? "" : "s")} over MCP", names = string.Join(", ", registry.Definitions.Select(d => d.Name));
            if (host.Address == McpServerTransports.Stdio)
            {
                if (!context.Quiet)
                {
                    context.Error($"Serving {tools} on standard input/output: {names}");
                }
            }
            else
            {
                // The address and token are what a client needs: printed even with --quiet.
                context.Error($"Serving {tools} at {host.Address}: {names}");
                context.Error(host.Token is null ? "No token is required (--mcp-token none); stop with Ctrl+C."
                    : $"Clients send \"Authorization: Bearer {host.Token}\" (idrak chat MODEL --mcp {host.Address} --mcp-token {host.Token}); stop with Ctrl+C.");
            }

            host.Completion.GetAwaiter().GetResult();
        }
        finally
        {
            host.DisposeAsync().AsTask().GetAwaiter().GetResult();
        }

        return ExitCodes.Ok;
    }
}
