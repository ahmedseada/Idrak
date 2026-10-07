// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using ModelContextProtocol.Client;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.Server;
using McpTool = ModelContextProtocol.Protocol.Tool;
using Tool = Idrak.Abstraction.Generation.Tool;

namespace Idrak.Mcp;

/// <summary>
/// Connects to Model Context Protocol servers (<see cref="ConnectAsync"/>, <see cref="ConnectStdioAsync"/>,
/// <see cref="ConnectHttpAsync"/>) so their tools can be added to a <see cref="ToolRegistry"/>, and serves tools to MCP
/// clients: a registry's (<see cref="ServerTools(IToolRegistry)"/>) and the models of an inference engine
/// (<see cref="ServerTools(IModelCatalog)"/>, <see cref="ModelTools"/>).
/// </summary>
public static partial class McpTools
{
    /// <summary>Connects over any MCP client transport.</summary>
    public static async Task<McpToolSource> ConnectAsync(IClientTransport transport, CancellationToken cancellationToken = default) =>
        new(await McpClient.CreateAsync(transport, cancellationToken: cancellationToken).ConfigureAwait(false));

    /// <summary>Starts <paramref name="command"/> with <paramref name="arguments"/> and talks to it over standard input/output.</summary>
    public static Task<McpToolSource> ConnectStdioAsync(string command, IEnumerable<string> arguments, CancellationToken cancellationToken = default) =>
        ConnectAsync(new StdioClientTransport(new StdioClientTransportOptions { Command = command, Arguments = [.. arguments] }), cancellationToken);

    /// <summary>Connects to an MCP server over HTTP (streamable HTTP, falling back to server-sent events).</summary>
    public static Task<McpToolSource> ConnectHttpAsync(Uri endpoint, CancellationToken cancellationToken = default) =>
        ConnectAsync(new HttpClientTransport(new HttpClientTransportOptions { Endpoint = endpoint }), cancellationToken);

    /// <summary>
    /// The tools of <paramref name="registry"/> as MCP server tools (add them to <c>McpServerOptions.ToolCollection</c>).
    /// Calls go through the registry, so its validation, allow rules, approvals and timeout apply; a failed call is
    /// returned with <c>isError</c> set.
    /// </summary>
    public static IReadOnlyList<McpServerTool> ServerTools(IToolRegistry registry)
    {
        ArgumentNullException.ThrowIfNull(registry);
        return [.. registry.Definitions.Select(d => (McpServerTool)new RegistryServerTool(registry, d))];
    }

    internal static JsonObject ToJsonObject(IDictionary<string, JsonElement>? arguments)
    {
        var result = new JsonObject();
        foreach (var (name, value) in arguments ?? new Dictionary<string, JsonElement>())
        {
            result[name] = JsonNode.Parse(JsonMarshal.GetRawUtf8Value(value));      // the element's own UTF-8, no string copy
        }

        return result;
    }

    internal static Dictionary<string, JsonElement> ToElements(JsonObject arguments)
    {
        // Each value written as ToJsonString() writes it (default writer options), but as UTF-8 into one reused buffer.
        var result = new Dictionary<string, JsonElement>();
        var buffer = new ArrayBufferWriter<byte>();
        using var writer = new Utf8JsonWriter(buffer);
        foreach (var (name, value) in arguments)
        {
            buffer.ResetWrittenCount();
            writer.Reset();
            if (value is null)
            {
                writer.WriteNullValue();
            }
            else
            {
                value.WriteTo(writer);
            }

            writer.Flush();
            var reader = new Utf8JsonReader(buffer.WrittenSpan);
            result[name] = JsonElement.ParseValue(ref reader);
        }

        return result;
    }

    internal static string TextOf(IEnumerable<ContentBlock> content)
    {
        var text = new StringBuilder();
        foreach (var block in content)
        {
            if (text.Length > 0)
            {
                text.Append('\n');
            }

            text.Append(block switch
            {
                TextContentBlock t => t.Text,
                ImageContentBlock i => $"[image {i.MimeType}]",
                AudioContentBlock a => $"[audio {a.MimeType}]",
                EmbeddedResourceBlock { Resource: TextResourceContents r } => r.Text,
                EmbeddedResourceBlock e => $"[resource {e.Resource.Uri}]",
                ResourceLinkBlock l => $"[resource {l.Uri}]",
                _ => $"[{block.Type}]",
            });
        }

        return text.ToString();
    }

    private sealed class RegistryServerTool : McpServerTool
    {
        private readonly IToolRegistry _registry;
        private readonly McpTool _tool;

        public RegistryServerTool(IToolRegistry registry, ToolDefinition definition)
        {
            _registry = registry;
            using var schema = JsonDocument.Parse(definition.Parameters?.ToJsonString() ?? """{"type":"object","properties":{}}""");
            _tool = new McpTool { Name = definition.Name, Description = definition.Description, InputSchema = schema.RootElement.Clone() };
        }

        public override McpTool ProtocolTool => _tool;

        public override IReadOnlyList<object> Metadata => [];

        public override async ValueTask<CallToolResult> InvokeAsync(RequestContext<CallToolRequestParams> request, CancellationToken cancellationToken = default)
        {
            var result = await _registry.InvokeAsync(new ToolCall(_tool.Name, ToJsonObject(request.Params?.Arguments)), cancellationToken).ConfigureAwait(false);
            return new CallToolResult
            {
                Content = [new TextContentBlock { Text = result.Succeeded ? result.Content : result.Error ?? result.Content }],
                IsError = !result.Succeeded,
            };
        }
    }
}

/// <summary>A connection to an MCP server whose tools Idrak models can call. Dispose it to disconnect.</summary>
public sealed class McpToolSource : IAsyncDisposable
{
    internal McpToolSource(McpClient client) => Client = client;

    /// <summary>The MCP client, for anything beyond tools (resources, prompts).</summary>
    public McpClient Client { get; }

    /// <summary>The server's name, as it reported it.</summary>
    public string? ServerName => Client.ServerInfo?.Name;

    /// <summary>
    /// The server's tools as Idrak tools (same names, descriptions and JSON schemas; <paramref name="prefix"/>
    /// is put before each name when set, to keep tools of several servers apart). A call runs the tool on the server
    /// and returns its text content; a result the server marks as an error becomes a failed call.
    /// </summary>
    public async Task<IReadOnlyList<Tool>> ListToolsAsync(string? prefix = null, CancellationToken cancellationToken = default)
    {
        var tools = await Client.ListToolsAsync(cancellationToken: cancellationToken).ConfigureAwait(false);
        return [.. tools.Select(t => Tool.Create(prefix + t.Name, t.Description ?? "", JsonNode.Parse(t.JsonSchema.GetRawText())!,
            (arguments, ct) => CallAsync(t.Name, arguments, ct)))];
    }

    /// <summary>Calls the server tool <paramref name="name"/> and returns its text content (throws if the server reports an error).</summary>
    public async Task<string> CallAsync(string name, JsonObject arguments, CancellationToken cancellationToken = default)
    {
        var result = await Client.CallToolAsync(new CallToolRequestParams { Name = name, Arguments = McpTools.ToElements(arguments) }, cancellationToken)
            .ConfigureAwait(false);
        string text = McpTools.TextOf(result.Content);
        return result.IsError == true ? throw new InvalidOperationException(text.Length > 0 ? text : $"The MCP tool '{name}' failed.") : text;
    }

    /// <summary>Disconnects (and stops a server started with <see cref="McpTools.ConnectStdioAsync"/>).</summary>
    public ValueTask DisposeAsync() => Client.DisposeAsync();
}
