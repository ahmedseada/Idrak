// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Generation;

/// <summary>
/// Conversations in the OpenAI / Hugging Face chat JSON, the layout fine-tuning reads
/// (<c>{"messages": [...], "tools": [...]}</c>). Tool calls get ids (call_1, call_2, …)
/// and each tool message the id of the call it answers, matched in order by name.
/// </summary>
public static class ChatJson
{
    /// <summary>One transcript line: messages, tools and, when set, the reasoning mode ("enable_thinking").</summary>
    public static JsonObject Transcript(IReadOnlyList<ChatMessage> messages, IReadOnlyList<ToolDefinition>? tools = null, bool? think = null)
    {
        var json = new JsonObject { ["messages"] = Messages(messages) };
        if (tools is { Count: > 0 })
        {
            json["tools"] = Tools(tools);
        }

        if (think is { } t)
        {
            json["enable_thinking"] = t;
        }

        return json;
    }

    /// <summary>The messages; assistant reasoning goes in "reasoning_content", call arguments are JSON objects.</summary>
    public static JsonArray Messages(IReadOnlyList<ChatMessage> messages)
    {
        var array = new JsonArray();
        var pending = new List<(string Id, string Name)>();
        int calls = 0;
        foreach (var message in messages)
        {
            var json = new JsonObject { ["role"] = message.Role, ["content"] = message.Content };
            if (!string.IsNullOrEmpty(message.Thinking))
            {
                json["reasoning_content"] = message.Thinking;
            }

            if (message.ToolCalls is { Count: > 0 })
            {
                var list = new JsonArray();
                foreach (var call in message.ToolCalls)
                {
                    string id = $"call_{++calls}";
                    pending.Add((id, call.Name));
                    list.Add((JsonNode)new JsonObject
                    {
                        ["id"] = id,
                        ["type"] = "function",
                        ["function"] = new JsonObject
                        {
                            ["name"] = call.Name,
                            ["arguments"] = call.Arguments.DeepClone(),
                        },
                    });
                }

                json["tool_calls"] = list;
            }

            if (message.Role == "tool")
            {
                int match = pending.FindIndex(p => p.Name == message.ToolName);
                match = match < 0 && message.ToolName is null ? 0 : match;
                if (match >= 0 && match < pending.Count)
                {
                    json["tool_call_id"] = pending[match].Id;
                    pending.RemoveAt(match);
                }

                if (message.ToolName is { } name)
                {
                    json["name"] = name;
                }
            }

            array.Add((JsonNode)json);
        }

        return array;
    }

    /// <summary>Tool definitions as <c>[{"type": "function", "function": {"name", "description", "parameters"}}]</c>.</summary>
    public static JsonArray Tools(IReadOnlyList<ToolDefinition> tools)
    {
        var array = new JsonArray();
        foreach (var tool in tools)
        {
            var function = new JsonObject { ["name"] = tool.Name };
            if (tool.Description is { } description)
            {
                function["description"] = description;
            }

            function["parameters"] = tool.Parameters?.DeepClone() ?? new JsonObject { ["type"] = "object", ["properties"] = new JsonObject() };
            array.Add((JsonNode)new JsonObject { ["type"] = "function", ["function"] = function });
        }

        return array;
    }
}
