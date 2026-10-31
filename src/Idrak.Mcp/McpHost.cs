// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Mcp.Abstractions;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.Server;

namespace Idrak.Mcp;

/// <summary>
/// How an <see cref="McpHost"/> serves: the server's name, and where and how its transport listens. Every field has a
/// default; <see cref="FromEnvironment"/> takes the address and token from <see cref="AddressVariable"/> and <see cref="TokenVariable"/>.
/// </summary>
public sealed record McpHostOptions
{
    /// <summary>The variable that gives the http transport's address (host:port) when an option does not: "IDRAK_MCP_ADDRESS".</summary>
    public const string AddressVariable = "IDRAK_MCP_ADDRESS";

    /// <summary>The variable that gives the http transport's bearer token when an option does not ("none": no token): "IDRAK_MCP_TOKEN".</summary>
    public const string TokenVariable = "IDRAK_MCP_TOKEN";

    /// <summary>The token value that turns the token off ("none"), in <see cref="TokenVariable"/> and on the command line.</summary>
    public const string NoToken = "none";

    /// <summary>The server's name, as clients see it ("idrak").</summary>
    public string ServerName { get; init; } = "idrak";

    /// <summary>The server's version, as clients see it (null: Idrak.Mcp's).</summary>
    public string? ServerVersion { get; init; }

    /// <summary>Where a network transport listens, host:port (null: 127.0.0.1 on a port the system picks).</summary>
    public string? Address { get; init; }

    /// <summary>The path a network transport answers on ("/mcp").</summary>
    public string Path { get; init; } = "/mcp";

    /// <summary>The bearer token clients must send (null: a random one, see <see cref="McpHost.Token"/>).</summary>
    public string? Token { get; init; }

    /// <summary>Whether a network transport requires the token (true).</summary>
    public bool RequireToken { get; init; } = true;

    /// <summary>Whether a network transport may listen where other machines reach it (false: loopback only).</summary>
    public bool AllowRemote { get; init; }

    /// <summary>How long an http session may go unused before it is closed (two hours); zero or less: never.</summary>
    public TimeSpan SessionIdleTimeout { get; init; } = TimeSpan.FromHours(2);

    /// <summary>The largest request body a network transport reads, in bytes (32 MiB).</summary>
    public long MaxRequestBytes { get; init; } = 32L << 20;

    /// <summary>The stdio transport's input (null: standard input).</summary>
    public Stream? Input { get; init; }

    /// <summary>The stdio transport's output (null: standard output).</summary>
    public Stream? Output { get; init; }

    /// <summary>Options of the transport's own (<see cref="IConfigurableKind.Keys"/>); null for none.</summary>
    public IReadOnlyDictionary<string, string>? TransportOptions { get; init; }

    /// <summary>
    /// The defaults with <see cref="Address"/> from <see cref="AddressVariable"/> and <see cref="Token"/> from
    /// <see cref="TokenVariable"/> when they are set ("none" in the token's turns the token off).
    /// </summary>
    public static McpHostOptions FromEnvironment()
    {
        string? address = Environment.GetEnvironmentVariable(AddressVariable) is { } a && a.Trim().Length > 0 ? a.Trim() : null;
        string? token = Environment.GetEnvironmentVariable(TokenVariable) is { } t && t.Trim().Length > 0 ? t.Trim() : null;
        return new McpHostOptions().WithToken(token) with { Address = address };
    }

    /// <summary>These options with <paramref name="token"/>: null keeps them, <see cref="NoToken"/> turns the token off, anything else is the token.</summary>
    public McpHostOptions WithToken(string? token) => token switch
    {
        null => this,
        _ when token.Equals(NoToken, StringComparison.OrdinalIgnoreCase) => this with { Token = null, RequireToken = false },
        _ => this with { Token = token, RequireToken = true },
    };
}

/// <summary>
/// An MCP server running in this process over a registered transport (<see cref="McpServerTransports"/>): "stdio" for a
/// client that started the process, "http" for any client that can reach a loopback port, including while an app runs.
/// Dispose it (or cancel the token it started with) to stop it: sessions close and calls in flight are cancelled.
/// </summary>
/// <example>
/// <code>
/// await using var host = await McpHost.StartAsync("http", registry);
/// Console.WriteLine($"{host.Address} (Authorization: Bearer {host.Token})");
/// await host.Completion;
/// </code>
/// </example>
public sealed class McpHost : IAsyncDisposable
{
    private readonly IMcpListener _listener;
    private readonly CancellationTokenRegistration _stopping;

    private McpHost(string transport, IMcpListener listener, int tools, CancellationToken cancellationToken)
    {
        Transport = transport;
        _listener = listener;
        ToolCount = tools;
        _stopping = cancellationToken.Register(static s => _ = ((IMcpListener)s!).StopAsync(), listener);
    }

    /// <summary>The transport's name.</summary>
    public string Transport { get; }

    /// <summary>Where clients reach the server, as a client's configuration takes it ("stdio", "http://127.0.0.1:PORT/mcp").</summary>
    public string Address => _listener.Address;

    /// <summary>The bearer token clients send (http with a token); null when none is required.</summary>
    public string? Token => _listener.Token;

    /// <summary>The tools served.</summary>
    public int ToolCount { get; }

    /// <summary>Completes when the server stops (the stdio client left, or the host was stopped).</summary>
    public Task Completion => _listener.Completion;

    /// <summary>
    /// Serves <paramref name="registry"/>'s tools (its validation, rules and timeout apply) over the transport
    /// <paramref name="transport"/>; returns once clients can connect.
    /// </summary>
    /// <exception cref="NotRegisteredException">No transport is registered as <paramref name="transport"/>.</exception>
    public static Task<McpHost> StartAsync(string transport, IToolRegistry registry, McpHostOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(registry);
        return StartAsync(transport, registry, null, options, cancellationToken);
    }

    /// <summary>
    /// Serves the tools of <paramref name="registry"/> and the models of <paramref name="models"/> (the tools generate, chat
    /// and embed, <see cref="McpTools.ServerTools(IModelCatalog)"/>), either may be null, over <paramref name="transport"/>.
    /// </summary>
    /// <exception cref="NotRegisteredException">No transport is registered as <paramref name="transport"/>.</exception>
    public static Task<McpHost> StartAsync(string transport, IToolRegistry? registry, IModelCatalog? models, McpHostOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new McpHostOptions();
        var server = new McpServerOptions
        {
            ServerInfo = new Implementation { Name = options.ServerName, Version = options.ServerVersion ?? typeof(McpHost).Assembly.GetName().Version?.ToString() ?? "0" },
            ToolCollection = [],
        };
        if (registry is not null)
        {
            foreach (var tool in McpTools.ServerTools(registry))
            {
                server.ToolCollection.Add(tool);
            }
        }

        if (models is not null)
        {
            foreach (var tool in McpTools.ServerTools(models))
            {
                server.ToolCollection.Add(tool);
            }
        }

        return StartAsync(transport, server, options, cancellationToken);
    }

    /// <summary>
    /// Serves <paramref name="server"/> (the SDK's options: tools, prompts, resources, handlers) over <paramref name="transport"/>;
    /// the name and version in <paramref name="options"/> are not applied (the server's own are).
    /// </summary>
    /// <exception cref="NotRegisteredException">No transport is registered as <paramref name="transport"/>.</exception>
    public static async Task<McpHost> StartAsync(string transport, McpServerOptions server, McpHostOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(transport);
        ArgumentNullException.ThrowIfNull(server);
        options ??= new McpHostOptions();
        var kind = McpServerTransports.Get(transport);
        var listener = await kind.StartAsync(new McpTransportRequest
        {
            Server = server,
            Address = options.Address,
            Path = options.Path,
            Token = options.Token,
            RequireToken = options.RequireToken,
            AllowRemote = options.AllowRemote,
            SessionIdleTimeout = options.SessionIdleTimeout,
            MaxRequestBytes = options.MaxRequestBytes,
            Input = options.Input,
            Output = options.Output,
            Options = options.TransportOptions,
        }, cancellationToken).ConfigureAwait(false);
        return new McpHost(kind.Name, listener, server.ToolCollection?.Count ?? 0, cancellationToken);
    }

    /// <summary>Stops the server: sessions close, calls in flight are cancelled; completes when it has.</summary>
    public Task StopAsync() => _listener.StopAsync();

    /// <summary>Stops the server (<see cref="StopAsync"/>).</summary>
    public async ValueTask DisposeAsync()
    {
        await _stopping.DisposeAsync().ConfigureAwait(false);
        await _listener.DisposeAsync().ConfigureAwait(false);
    }
}
