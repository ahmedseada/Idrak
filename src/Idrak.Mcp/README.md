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

```csharp
await using var engine = await InferenceEngine.Create().ChatModel("my-gpt", "models/chat.ikm", "chars").BuildAsync();
var registry = ToolRegistry.Create().Add(McpTools.ModelTools(engine)).Add(myTools).Build();
var options = new McpServerOptions { ToolCollection = [.. McpTools.ServerTools(registry)] };
```

From the command line: `idrak serve MODEL --mcp [--tools TOOLS.dll]` serves models (and tools) over standard
input/output; `idrak mcp serve TOOLS.dll` serves tools alone.

## Install

```bash
dotnet add package Idrak.Mcp
```

Part of [Idrak](https://www.nuget.org/packages/Idrak), a self-contained deep-learning library for .NET.

## Documentation

Guides, samples and the full API overview: https://github.com/ahmedseada/Idrak

License: Apache 2.0.
