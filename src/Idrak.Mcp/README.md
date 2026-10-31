# Idrak.Mcp

Model Context Protocol for Idrak: use the tools of MCP servers in conversations and the inference engine, and serve
Idrak tools and the inference engine's models to MCP clients. It depends on `Idrak.Abstraction` (the tool and serving
contracts) and the MCP SDK only.

- **Client:** `McpTools.ConnectStdioAsync`, `ConnectHttpAsync` or `ConnectAsync`; `ListToolsAsync` returns the server's
  tools as Idrak `Tool`s, to add to a `ToolRegistry` (or any `IToolRegistry`).
- **Server, tools:** `McpTools.ServerTools(registry)` serves an `IToolRegistry` (its validation, rules and timeout apply).
- **Server, models:** `McpTools.ServerTools(models)` serves the models of an `IModelCatalog` (an `InferenceEngine`) as
  the tools `generate` (text models), `chat` (chat models) and `embed` (models whose kind implements `IEmbedder`), each
  with a `model` argument; `McpTools.ModelTools(models)` gives the same as `Tool`s, to combine with other tools.

- **Server, hosted:** `McpHost.StartAsync(transport, registry, options)` runs the server in any process (a console
  tool, a GUI app that is already running) over a transport registered in `McpServerTransports`
  (`Idrak.Mcp.Abstractions`; `IMcpServerTransport`, an unknown name fails with "not registered"):
  - `stdio`: the process's standard input and output (or `McpHostOptions.Input`/`Output`): the one client that
    started the process.
  - `http`: MCP streamable HTTP on its own socket (no web framework): `127.0.0.1` on a port the system picks unless
    `Address` says otherwise (a host off this machine only with `AllowRemote`), a bearer token required (a random one
    unless `Token` gives it; `RequireToken = false` turns it off), each client its own session (`Mcp-Session-Id`), a
    call cancelled when its client cancels or disconnects, idle sessions closed after `SessionIdleTimeout`.
  - Protocol revisions: a client of the 2026-07-28 revision (the SDK's default) is served without a session, as the
    revision defines: `server/discover` and every request stand alone (no `initialize`, no `Mcp-Session-Id`; GET and
    DELETE answer 405), so it connects in one request; a client of an earlier revision gets its session from
    `initialize`.
  - Sessions: at most `MaxSessions` open at once (1024 by default, 0: no limit; `IDRAK_MCP_MAX_SESSIONS`); a new one over
    it is answered 503 with `Retry-After` (the seconds until an idle session closes), unless `ReplaceIdleSessions`
    (`IDRAK_MCP_REPLACE_IDLE=1`) closes the idle session unused the longest for it. Sessionless requests do not count.
  `host.Address` is what a client's configuration takes ("stdio", `http://127.0.0.1:PORT/mcp`), `host.Token` the token
  it sends; dispose the host (or cancel its token) to stop it. A host registers another transport (an ASP.NET Core one)
  with `McpServerTransports.Register`.

```csharp
await using var engine = await InferenceEngine.Create().ChatModel("my-gpt", "models/chat.ikm", "chars").BuildAsync();
var registry = ToolRegistry.Create().Add(McpTools.ModelTools(engine)).Add(myTools).Build();
await using var host = await McpHost.StartAsync("http", registry);
Console.WriteLine($"{host.Address}, Authorization: Bearer {host.Token}");

// A client: McpTools.ConnectHttpAsync(new Uri(host.Address), host.Token)
// Or the SDK's options, for anything beyond tools: new McpServerOptions { ToolCollection = [.. McpTools.ServerTools(registry)] }
```

- **Cancellation:** a call `McpToolSource.CallAsync` (and the tools `ListToolsAsync` gives) whose caller cancels it sends
  the server `notifications/cancelled` with the call's request id, over stdio as over http, so the server's tool stops.
- **Telemetry:** at `TelemetryLevel.Mcp` (`McpEvent`): sessions opened and closed (transport, address, client), requests
  received and answered (method, tool, HTTP status, bytes in and out, time), server tool calls started, finished,
  cancelled or failed, refusals (400, 401, 403, 404, 405, 409, 413, 431, 503) with their reason, and the cancellations a
  client sends. Nothing is measured when no hook listens (one static read and a bitwise AND per site). The profile probe
  "mcp" sums them for a run: requests by method and tool (count, errors, p50, p99, bytes), tool calls, sessions, refusals.
  Over stdio a request's bytes are not split out: the session's bytes read and written come with its close.
  `idrak trace --levels mcp -- mcp serve ...` writes them as JSON Lines.

From the command line: `idrak mcp serve TOOLS.dll` serves tools, `idrak serve MODEL --mcp [--tools TOOLS.dll]` models
(and tools); both over stdio by default, or `--transport http` (with `--mcp-address HOST:PORT`, `--mcp-token TOKEN` or
`none`, `--mcp-allow-remote`; `--mcp-max-sessions N`, `--mcp-replace-idle`; `IDRAK_MCP_TRANSPORT`, `IDRAK_MCP_ADDRESS`, `IDRAK_MCP_TOKEN`,
`IDRAK_MCP_MAX_SESSIONS`, `IDRAK_MCP_REPLACE_IDLE`), which prints the address and token. `--verbose` prints each
session, request, tool call and refusal on the error output (never on the output, which carries the protocol over
stdio), `--profile` the session's report at exit. A model calls such a server's tools with `idrak chat MODEL --mcp http://127.0.0.1:PORT/mcp --mcp-token TOKEN`.

## Install

```bash
dotnet add package Idrak.Mcp
```

Part of [Idrak](https://www.nuget.org/packages/Idrak), a self-contained deep-learning library for .NET.

## Documentation

Guides, samples and the full API overview: https://github.com/ahmedseada/Idrak

License: Apache 2.0.
