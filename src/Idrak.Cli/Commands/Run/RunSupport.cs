// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Generation;
using Idrak.Models;
using Idrak.Nlp;

namespace Idrak.Cli.Commands.Run;

/// <summary>A model loaded for chat: the model, the choice it came from, its chat generator, context window and tools.</summary>
internal sealed class LoadedChat(PretrainedModel model, Models.ModelChoice choice, ChatGenerator chat, int context, IReadOnlyList<Tool> tools) : IDisposable
{
    private readonly List<Idrak.Mcp.McpToolSource> _servers = [];

    public PretrainedModel Model { get; } = model;

    public Models.ModelChoice Choice { get; } = choice;

    public ChatGenerator Chat { get; } = chat;

    public int Context { get; } = context;

    public IReadOnlyList<Tool> Tools { get; } = tools;

    /// <summary>The tools as a registry, or null.</summary>
    public ToolRegistry? Registry { get; } = ToolAssemblies.Registry(tools);

    /// <summary>Loads the model named <paramref name="name"/> (alias, id, folder or file) with the command's options.</summary>
    public static LoadedChat Load(CommandContext context, string name, GenerationSettings settings, Models.ModelChoice? choice = null)
    {
        var tools = ToolAssemblies.Load(settings.ToolAssemblies).ToList();
        var servers = new List<Idrak.Mcp.McpToolSource>();
        try
        {
            // MCP servers (--mcp, on the commands that take it): their tools join the assemblies' ones.
            var addresses = context.Options("--mcp");
            foreach (string server in addresses)
            {
                var source = Connect(server);
                servers.Add(source);
                tools.AddRange(source.ListToolsAsync(addresses.Count > 1 ? $"{source.ServerName ?? "mcp" + servers.Count}_" : null).GetAwaiter().GetResult());
                context.Detail($"MCP server {source.ServerName ?? server}: {tools.Count} tools in all");
            }

            choice ??= Models.Choose(context, name);
            var model = Models.Load(context, choice);
            try
            {
                var loaded = new LoadedChat(model, choice, Models.CreateChat(model, choice), Models.ContextLength(choice, model), tools);
                loaded._servers.AddRange(servers);
                return loaded;
            }
            catch
            {
                model.Dispose();
                throw;
            }
        }
        catch
        {
            foreach (var server in servers)
            {
                server.DisposeAsync().AsTask().GetAwaiter().GetResult();
            }

            throw;
        }
    }

    // An http(s) URL is a server to call over HTTP; anything else is a command line that starts a server (over stdio).
    private static Idrak.Mcp.McpToolSource Connect(string server)
    {
        try
        {
            if (Uri.TryCreate(server, UriKind.Absolute, out var url) && url.Scheme is "http" or "https")
            {
                return Idrak.Mcp.McpTools.ConnectHttpAsync(url).GetAwaiter().GetResult();
            }

            var words = server.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            return words.Length == 0 ? throw new UsageException("--mcp needs a URL or a command that starts the server.")
                : Idrak.Mcp.McpTools.ConnectStdioAsync(words[0], words.Skip(1)).GetAwaiter().GetResult();
        }
        catch (Exception e) when (e is not UsageException)
        {
            throw new InvalidOperationException($"Cannot reach the MCP server '{server}': {e.Message}", e);
        }
    }

    /// <summary>"name (LlamaForCausalLM, 1.2M parameters) on cpu".</summary>
    public string Describe(CommandContext context) =>
        $"{Choice.Model} ({Model.Config["architectures"]?[0]}, {Parameters(Model.Spec.ParameterCount)} parameters) on {context.Device}";

    /// <summary>A parameter count for people (1.2M, 7.6B).</summary>
    public static string Parameters(long count) => count >= 1e9 ? $"{count / 1e9:F1}B" : count >= 1e6 ? $"{count / 1e6:F1}M" : $"{count / 1e3:F0}k";

    public void Dispose()
    {
        foreach (var server in _servers)
        {
            server.DisposeAsync().AsTask().GetAwaiter().GetResult();
        }

        Model.Dispose();
    }
}

/// <summary>Conversations saved and loaded by chat (<c>--history</c>, <c>/save</c>, <c>/load</c>) in the chat JSON layout fine-tuning reads.</summary>
internal static class ChatHistory
{
    /// <summary>Writes <paramref name="messages"/> to <paramref name="path"/> (<c>{"messages": [...], "tools": [...]}</c>).</summary>
    public static void Save(string path, IReadOnlyList<ChatMessage> messages, IReadOnlyList<ToolDefinition>? tools, bool? think)
    {
        if (Path.GetDirectoryName(Path.GetFullPath(path)) is { } folder)
        {
            Directory.CreateDirectory(folder);
        }

        File.WriteAllText(path, ChatJson.Transcript(messages, tools, think).ToJsonString(CommandContext.JsonOutput) + "\n");
    }

    /// <summary>Reads a conversation saved by <see cref="Save"/> (or any file with a "messages" list in the chat layout).</summary>
    public static List<ChatMessage> Load(string path)
    {
        JsonNode? root;
        try
        {
            root = JsonNode.Parse(File.ReadAllText(path));
        }
        catch (JsonException e)
        {
            throw new InvalidOperationException($"{path} is not JSON: {e.Message}");
        }

        var list = root as JsonArray ?? root?["messages"] as JsonArray
            ?? throw new InvalidOperationException($"{path} has no \"messages\" list.");
        return [.. list.Select(Message)];
    }

    /// <summary>One message of the chat layout.</summary>
    public static ChatMessage Message(JsonNode? json)
    {
        if (json is not JsonObject m || (string?)m["role"] is not { } role)
        {
            throw new InvalidOperationException($"A message needs a \"role\": {json?.ToJsonString()}");
        }

        List<ToolCall>? calls = null;
        if (m["tool_calls"] is JsonArray array)
        {
            calls = [];
            foreach (var call in array)
            {
                var function = call?["function"] ?? call;
                var arguments = function?["arguments"] switch
                {
                    JsonObject o => (JsonObject)o.DeepClone(),
                    JsonValue v when v.TryGetValue(out string? text) => JsonNode.Parse(text) as JsonObject ?? [],
                    _ => [],
                };
                calls.Add(new ToolCall((string?)function?["name"] ?? "", arguments));
            }
        }

        return new ChatMessage(role, m["content"] is JsonValue c && c.TryGetValue(out string? content) ? content : "",
            (string?)m["reasoning_content"] ?? (string?)m["thinking"], calls is { Count: > 0 } ? calls : null, (string?)m["name"]);
    }
}

/// <summary>Prompt text from arguments, <c>--input</c> files and piped standard input (run, complete, tokenize).</summary>
internal static class PromptInput
{
    /// <summary>
    /// The prompt: the arguments from <paramref name="first"/> on, joined by spaces, followed by the text of the
    /// <c>--input</c> files and then of piped standard input (each after a blank line). A usage error when all are empty.
    /// </summary>
    public static string Read(CommandContext context, int first, string what = "a prompt")
    {
        var parts = new List<string>();
        if (context.Positional.Count > first)
        {
            parts.Add(string.Join(' ', context.Positional.Skip(first)));
        }

        foreach (string file in context.Options("--input"))
        {
            parts.Add(File.Exists(file) ? File.ReadAllText(file).TrimEnd() : throw new UsageException($"Input file not found: {file}"));
        }

        if (StandardInput.IsRedirected && StandardInput.Reader.ReadToEnd().TrimEnd() is { Length: > 0 } piped)
        {
            parts.Add(piped);
        }

        parts.RemoveAll(p => p.Length == 0);
        return parts.Count > 0 ? string.Join("\n\n", parts)
            : throw new UsageException($"Missing {what}: give it as an argument, with --input FILE, or pipe it in (cat notes.txt | idrak run MODEL \"Summarize\").");
    }
}
