// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using ModelContextProtocol.Server;

namespace Idrak.Mcp.Abstractions;

/// <summary>
/// What a transport is asked to serve with (<see cref="IMcpServerTransport.StartAsync"/>): the MCP server's options and
/// where and how to listen. A field added later never breaks a transport; a transport ignores the fields that do not
/// apply to it (stdio has no address, http no streams) and refuses option keys it does not take.
/// </summary>
public sealed record McpTransportRequest
{
    /// <summary>The MCP server every session runs (its name, tools, handlers); shared by the sessions of one listener.</summary>
    public required McpServerOptions Server { get; init; }

    /// <summary>
    /// Where a network transport listens, as <c>host:port</c> (<c>127.0.0.1:0</c> when null: the loopback address, a port
    /// the system picks). A host off this machine is refused unless <see cref="AllowRemote"/> is set.
    /// </summary>
    public string? Address { get; init; }

    /// <summary>The path a network transport answers on ("/mcp").</summary>
    public string Path { get; init; } = "/mcp";

    /// <summary>
    /// The bearer token a network transport requires (<c>Authorization: Bearer TOKEN</c>); null makes a random one, which
    /// the listener reports (<see cref="IMcpListener.Token"/>). Ignored when <see cref="RequireToken"/> is false.
    /// </summary>
    public string? Token { get; init; }

    /// <summary>Whether a network transport requires the token (true); false answers any request on the address.</summary>
    public bool RequireToken { get; init; } = true;

    /// <summary>Whether a network transport may listen on an address other machines reach (false: loopback only).</summary>
    public bool AllowRemote { get; init; }

    /// <summary>How long a session may go without a request before it is closed (two hours); <see cref="TimeSpan.Zero"/> or less: never.</summary>
    public TimeSpan SessionIdleTimeout { get; init; } = TimeSpan.FromHours(2);

    /// <summary>The largest request body a network transport reads, in bytes (32 MiB); a larger one is answered 413.</summary>
    public long MaxRequestBytes { get; init; } = 32L << 20;

    /// <summary>
    /// The most sessions a network transport keeps open at once (<see cref="DefaultMaxSessions"/>); 0 or less: no limit. A new
    /// session over it is refused with 503 and <c>Retry-After</c>, unless <see cref="ReplaceIdleSessions"/> closes an idle one
    /// for it. Requests of the sessionless protocol revisions (2026-07-28 and later) hold no session and do not count.
    /// </summary>
    public int MaxSessions { get; init; } = DefaultMaxSessions;

    /// <summary>
    /// Whether a new session at <see cref="MaxSessions"/> closes the session unused the longest (one with no request in
    /// flight and no event stream open) to take its place (false: the new session is refused with 503).
    /// </summary>
    public bool ReplaceIdleSessions { get; init; }

    /// <summary>The default <see cref="MaxSessions"/>: 1024 (a session holds a few KB until it is deleted or goes idle).</summary>
    public const int DefaultMaxSessions = 1024;

    /// <summary>The stream a stream transport reads the client's messages from (null: the process's standard input).</summary>
    public Stream? Input { get; init; }

    /// <summary>The stream a stream transport writes its messages to (null: the process's standard output).</summary>
    public Stream? Output { get; init; }

    /// <summary>Options of the transport's own, by key (<see cref="IConfigurableKind.Keys"/> names the ones it takes); null for none.</summary>
    public IReadOnlyDictionary<string, string>? Options { get; init; }
}

/// <summary>
/// A running MCP server on one transport (<see cref="IMcpServerTransport.StartAsync"/>): where clients reach it, and its
/// end. Disposing it stops it: the sessions close, in-flight calls are cancelled and the listener is released.
/// </summary>
public interface IMcpListener : IAsyncDisposable
{
    /// <summary>Where a client reaches it, as a client's configuration takes it: "stdio", or "http://127.0.0.1:PORT/mcp".</summary>
    string Address { get; }

    /// <summary>The bearer token clients send (a network transport that requires one); null when none is required.</summary>
    string? Token { get; }

    /// <summary>Completes when it stops: its client left (stdio), or it was stopped or cancelled.</summary>
    Task Completion { get; }

    /// <summary>Stops serving: closes the sessions (cancelling calls in flight) and releases the listener; completes when it has.</summary>
    Task StopAsync();
}

/// <summary>
/// A way to serve MCP sessions to clients, registered in <see cref="McpServerTransports"/> under its
/// <see cref="IKind.Name"/>: "stdio" (the process's standard input and output, or given streams) and "http" (streamable
/// HTTP on a loopback port) come with Idrak.Mcp; a host registers others (Idrak.AspNetCore, Kestrel).
/// </summary>
public interface IMcpServerTransport : IConfigurableKind
{
    /// <summary>
    /// Starts serving <paramref name="request"/>'s server; returns once clients can connect (a network transport has bound
    /// its address). Cancelling <paramref name="cancellationToken"/> while it starts throws; after it, use the listener.
    /// </summary>
    /// <exception cref="ArgumentException">An option key the transport does not take, a bad address, or a host off this machine without <see cref="McpTransportRequest.AllowRemote"/>.</exception>
    ValueTask<IMcpListener> StartAsync(McpTransportRequest request, CancellationToken cancellationToken = default);
}

/// <summary>
/// The MCP server transports by name (ignoring case). Built in (Idrak.Mcp): <see cref="Stdio"/> and <see cref="Http"/>.
/// A transport that is not registered is refused with "not registered"; nothing falls back to another. Register another
/// with <see cref="Register"/>; one under a library name takes the library's place, which <see cref="Unregister"/> brings back.
/// </summary>
public static class McpServerTransports
{
    /// <summary>The process's standard input and output (or the request's streams): one client, the one that started the process.</summary>
    public const string Stdio = "stdio";

    /// <summary>Streamable HTTP on a loopback port, with a bearer token: any number of clients, each its own session.</summary>
    public const string Http = "http";

    /// <summary>The variable that names the transport when an option does not: "IDRAK_MCP_TRANSPORT".</summary>
    public const string EnvironmentVariable = "IDRAK_MCP_TRANSPORT";

    private static readonly KindTable<IMcpServerTransport> Table = BuiltIn();

    private static KindTable<IMcpServerTransport> BuiltIn()
    {
        var table = new KindTable<IMcpServerTransport>(nameof(McpServerTransports), "MCP server transport", "add it with McpServerTransports.Register",
            comparer: StringComparer.OrdinalIgnoreCase,
            unguarded: "a transport is where clients connect and what they may reach: falling back to another would listen where nobody asked");
        table.RegisterDefault(new Transports.StdioServerTransportKind());
        table.RegisterDefault(new Transports.HttpServerTransportKind());
        return table;
    }

    /// <summary>The transport named by <see cref="EnvironmentVariable"/>, or <see cref="Stdio"/> when it is not set.</summary>
    public static string DefaultName => Environment.GetEnvironmentVariable(EnvironmentVariable) is { } name && name.Trim().Length > 0 ? name.Trim() : Stdio;

    /// <summary>Registers <paramref name="transport"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IMcpServerTransport transport) => Table.Register(transport, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's transport <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Names;

    /// <summary>The transport registered as <paramref name="name"/>, or null.</summary>
    public static IMcpServerTransport? Find(ReadOnlySpan<char> name) => Table.Find(name);

    /// <summary>The transport registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotRegisteredException">None is: the message names the registered ones.</exception>
    public static IMcpServerTransport Get(ReadOnlySpan<char> name) => Table.Get(name);

    /// <summary>The library's transport <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IMcpServerTransport? Default(string name) => Table.Default(name);

    /// <summary>Who registered <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered transport as "name: summary", one per line.</summary>
    public static string Describe() => Table.Describe();
}
