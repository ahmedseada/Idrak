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

From the command line: `idrak mcp serve TOOLS.dll` serves tools, `idrak serve MODEL --mcp [--tools TOOLS.dll]` models
(and tools); both over stdio by default, or `--transport http` (with `--mcp-address HOST:PORT`, `--mcp-token TOKEN` or
`none`, `--mcp-allow-remote`; `IDRAK_MCP_TRANSPORT`, `IDRAK_MCP_ADDRESS`, `IDRAK_MCP_TOKEN`), which prints the address
and token. A model calls such a server's tools with `idrak chat MODEL --mcp http://127.0.0.1:PORT/mcp --mcp-token TOKEN`.

## Install

```bash
dotnet add package Idrak.Mcp
```

Part of [Idrak](https://www.nuget.org/packages/Idrak), a self-contained deep-learning library for .NET.

## Documentation

Guides, samples and the full API overview: https://github.com/ahmedseada/Idrak

License: Apache 2.0.
