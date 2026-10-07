// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using ModelContextProtocol.Server;
using Tool = Idrak.Abstraction.Generation.Tool;

namespace Idrak.Mcp;

public static partial class McpTools
{
    /// <summary>The name of the tool that continues a prompt with a text model (<see cref="ITextModel"/>).</summary>
    public const string GenerateTool = "generate";

    /// <summary>The name of the tool that answers a conversation with a chat model (<see cref="IChatModel"/>).</summary>
    public const string ChatTool = "chat";

    /// <summary>The name of the tool that turns texts into vectors with an embedding model (<see cref="IEmbedder"/>).</summary>
    public const string EmbedTool = "embed";

    /// <summary>
    /// The models of <paramref name="models"/> (an inference engine, for example) as MCP server tools: <see cref="ModelTools"/>
    /// served through a <see cref="ToolRegistry"/>, so arguments are validated and a failed call is returned with
    /// <c>isError</c> set.
    /// </summary>
    public static IReadOnlyList<McpServerTool> ServerTools(IModelCatalog models) =>
        ServerTools(ToolRegistry.Create().Add(ModelTools(models)).Build());

    /// <summary>
    /// The models of <paramref name="models"/> as tools, one per contract a model offers, each with a <c>model</c>
    /// argument naming one of the models that offer it (optional when only one does):
    /// <list type="bullet">
    /// <item><c>generate</c> (<see cref="ITextModel"/>): <c>prompt</c>, and optionally <c>max_tokens</c>, <c>temperature</c>
    /// and <c>seed</c>; returns the generated text.</item>
    /// <item><c>chat</c> (<see cref="IChatModel"/>): <c>messages</c> (a list of <c>{role, content}</c>), and optionally
    /// <c>think</c>, <c>max_tokens</c>, <c>temperature</c> and <c>seed</c>; returns the assistant's answer.</item>
    /// <item><c>embed</c> (<see cref="IEmbedder"/>): <c>input</c> (a list of texts); returns <c>{"embeddings": [[...], ...]}</c>
    /// in input order.</item>
    /// </list>
    /// A tool is left out when no model offers its contract. The tools ask the catalog for the model on each call, so
    /// an engine's queueing, timeouts and statistics apply.
    /// </summary>
    public static IReadOnlyList<Tool> ModelTools(IModelCatalog models)
    {
        ArgumentNullException.ThrowIfNull(models);
        var tools = new List<Tool>();
        if (Offering<ITextModel>(models) is { Count: > 0 } text)
        {
            tools.Add(Tool.Create(GenerateTool, $"Continue a prompt with a local language model and return the generated text. Models: {string.Join(", ", text)}.",
                Schema(text, new JsonObject { ["prompt"] = Property("string", "The text to continue.") }, ["prompt"], generation: true),
                async (args, token) =>
                {
                    var (generated, _, _) = await models.Model<ITextModel>(Pick(args, text)).GenerateAsync((string)args["prompt"]!, Options(args), token)
                        .ConfigureAwait(false);
                    return generated;
                }));
        }

        if (Offering<IChatModel>(models) is { Count: > 0 } chat)
        {
            var message = new JsonObject
            {
                ["type"] = "object",
                ["properties"] = new JsonObject
                {
                    ["role"] = new JsonObject { ["type"] = "string", ["enum"] = new JsonArray("system", "user", "assistant") },
                    ["content"] = Property("string", "The message's text."),
                },
                ["required"] = new JsonArray("role", "content"),
            };
            var properties = new JsonObject
            {
                ["messages"] = new JsonObject { ["type"] = "array", ["items"] = message, ["description"] = "The conversation so far, ending with the user's message." },
                ["think"] = Property("boolean", "Whether the model reasons before answering (models that can)."),
            };
            tools.Add(Tool.Create(ChatTool, $"Answer a conversation with a local chat model and return the assistant's reply. Models: {string.Join(", ", chat)}.",
                Schema(chat, properties, ["messages"], generation: true),
                async (args, token) =>
                {
                    var request = new ChatRequest(Messages(args["messages"]!.AsArray()), Think: (bool?)args["think"], Options: Options(args));
                    var reply = await models.Model<IChatModel>(Pick(args, chat)).ChatAsync(request, token).ConfigureAwait(false);
                    return reply.Message?.Content ?? "";
                }));
        }

        if (Offering<IEmbedder>(models) is { Count: > 0 } embed)
        {
            var properties = new JsonObject
            {
                ["input"] = new JsonObject { ["type"] = "array", ["items"] = new JsonObject { ["type"] = "string" }, ["description"] = "The texts to embed." },
            };
            tools.Add(Tool.Create(EmbedTool, $"Turn texts into embedding vectors with a local model; returns {{\"embeddings\": [[...], ...]}} in input order. Models: {string.Join(", ", embed)}.",
                Schema(embed, properties, ["input"], generation: false),
                async (args, token) =>
                {
                    var texts = args["input"]!.AsArray().Select(t => t?.GetValueKind() == JsonValueKind.String ? (string)t! : throw new ArgumentException("input must be a list of texts")).ToList();
                    return Embeddings(await models.Model<IEmbedder>(Pick(args, embed)).EmbedAsync(texts, token).ConfigureAwait(false));
                }));
        }

        return tools;
    }

    // The names of the models whose kind implements T, in the catalog's order.
    private static List<string> Offering<T>(IModelCatalog models)
        where T : class => [.. models.Names.Where(n => models.TryGetModel<T>(n, out _))];

    private static JsonObject Property(string type, string description) => new() { ["type"] = type, ["description"] = description };

    // The tool's schema: "model" (one of names; required when there are several), the tool's own properties, and the
    // generation options for generate and chat.
    private static JsonObject Schema(List<string> names, JsonObject own, string[] required, bool generation)
    {
        var properties = new JsonObject
        {
            ["model"] = new JsonObject
            {
                ["type"] = "string",
                ["enum"] = new JsonArray([.. names.Select(n => (JsonNode)n)]),
                ["description"] = names.Count == 1 ? $"The model (default {names[0]})." : "The model.",
            },
        };
        foreach (var (name, value) in own.ToList())
        {
            own.Remove(name);
            properties[name] = value;
        }

        if (generation)
        {
            properties["max_tokens"] = Property("integer", "The most tokens to generate.");
            properties["temperature"] = Property("number", "Sampling temperature (0 picks the likeliest token).");
            properties["seed"] = Property("integer", "Random seed, for repeatable sampling.");
        }

        var list = new JsonArray([.. required.Select(r => (JsonNode)r)]);
        if (names.Count > 1)
        {
            list.Insert(0, "model");
        }

        return new JsonObject { ["type"] = "object", ["properties"] = properties, ["required"] = list };
    }

    private static string Pick(JsonObject args, List<string> names) => (string?)args["model"] ?? names[0];

    private static GenerationOptions Options(JsonObject args)
    {
        var options = new GenerationOptions();
        if (args["max_tokens"] is { } max)
        {
            options = options with { NumPredict = (int)max };
        }

        if (args["temperature"] is { } temperature)
        {
            options = options with { Temperature = (float)temperature };
        }

        if (args["seed"] is { } seed)
        {
            options = options with { Seed = (int)seed };
        }

        return options;
    }

    private static List<ChatMessage> Messages(JsonArray messages)
    {
        var list = new List<ChatMessage>();
        foreach (var node in messages)
        {
            if (node is not JsonObject m || m["role"] is not JsonValue role || role.GetValueKind() != JsonValueKind.String
                || m["content"] is not JsonValue content || content.GetValueKind() != JsonValueKind.String)
            {
                throw new ArgumentException("each message needs a role and a content, both strings");
            }

            list.Add(new ChatMessage((string)role!, (string)content!));
        }

        return list.Count > 0 ? list : throw new ArgumentException("messages must not be empty");
    }

    // {"embeddings": [[...], ...]}, written as UTF-8 at once rather than through a node per number.
    private static string Embeddings(float[][] vectors)
    {
        var buffer = new ArrayBufferWriter<byte>();
        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            writer.WriteStartArray("embeddings");
            foreach (var vector in vectors)
            {
                writer.WriteStartArray();
                foreach (float value in vector)
                {
                    writer.WriteNumberValue(value);
                }

                writer.WriteEndArray();
            }

            writer.WriteEndArray();
            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(buffer.WrittenSpan);
    }
}
