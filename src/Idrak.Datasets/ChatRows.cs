// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Frozen;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace Idrak.Datasets;

/// <summary>
/// How rows become conversations: templates over the row's columns, such as <c>User = "{instruction}\n\n{input}"</c>,
/// <c>Assistant = "{output}"</c>. A placeholder names a column; empty or missing columns leave no blank lines behind.
/// </summary>
public sealed record ChatMapping
{
    /// <summary>The system message (a template), or null.</summary>
    public string? System { get; init; }

    /// <summary>The user message (a template).</summary>
    public required string User { get; init; }

    /// <summary>The assistant's answer (a template).</summary>
    public required string Assistant { get; init; }
}

/// <summary>What <see cref="ChatRows"/> makes of rows.</summary>
public enum RowKind
{
    /// <summary>Conversations when a row can be read as one, else plain text when it has text, else dropped.</summary>
    Auto,

    /// <summary>Only conversations ({"messages": [...]}); other rows are dropped.</summary>
    Chat,

    /// <summary>Only plain text ({"text": ...}); conversations are flattened to their text.</summary>
    Text,

    /// <summary>
    /// Only preference pairs, as TRL's conversational preference rows ({"prompt": [...], "chosen": [...], "rejected": [...]},
    /// see <see cref="ChatRows.Preference"/>); other rows are dropped.
    /// </summary>
    Preference,
}

/// <summary>
/// Turns rows of common dataset layouts into the fine-tuning format: <c>{"messages": [{"role", "content"}, ...], "tools": [...]}</c>
/// for conversations, <c>{"text": ...}</c> for plain text. Recognized: OpenAI / Hugging Face messages (also as
/// "conversation(s)" or TRL's "prompt"/"completion" message lists and "chosen" preference pairs), ShareGPT
/// ({"from": "human" | "gpt", "value"}), Alpaca (instruction, input, output), and question/answer pairs (question →
/// answer, prompt → completion / response, query → response, problem → solution, input → output / target).
/// </summary>
public static partial class ChatRows
{
    // Role names as datasets spell them (matched ignoring case), as user, assistant, system or tool.
    private static readonly FrozenDictionary<string, string> Roles = new Dictionary<string, string>
    {
        ["human"] = "user", ["user"] = "user", ["prompter"] = "user",
        ["gpt"] = "assistant", ["assistant"] = "assistant", ["model"] = "assistant", ["bot"] = "assistant", ["chatgpt"] = "assistant", ["bard"] = "assistant",
        ["system"] = "system",
        ["tool"] = "tool", ["function"] = "tool", ["observation"] = "tool", ["function_response"] = "tool", ["ipython"] = "tool",
        ["function_call"] = "assistant",
    }.ToFrozenDictionary(StringComparer.OrdinalIgnoreCase);

    private static readonly string[] MessageFields = ["reasoning_content", "thinking", "reasoning", "tool_calls", "name", "tool_call_id"];

    private static readonly string[] TextColumns = ["text", "content", "document", "body", "code"];

    private static readonly string[][] Pairs =
    [
        ["instruction", "output"], ["instruction", "response"], ["question", "answer"], ["question", "response"], ["prompt", "completion"],
        ["prompt", "response"], ["prompt", "answer"], ["prompt", "output"], ["query", "response"], ["query", "answer"], ["problem", "solution"],
        ["input", "output"], ["input", "target"], ["context", "response"], ["src", "tgt"],
    ];

    /// <summary>Rows as conversations or text (see the class summary); <paramref name="mapping"/> overrides the detection.</summary>
    public static Dataset Normalize(Dataset data, RowKind kind = RowKind.Auto, ChatMapping? mapping = null, string? system = null) =>
        data.Select(row => Normalize(row, kind, mapping, system));

    /// <summary>One row as a conversation or text, or null when it is neither.</summary>
    public static JsonObject? Normalize(JsonObject row, RowKind kind = RowKind.Auto, ChatMapping? mapping = null, string? system = null)
    {
        if (kind == RowKind.Preference)
        {
            return Preference(row, system);
        }

        var messages = mapping is not null ? Mapped(row, mapping) : Conversation(row);
        if (messages is not null && messages.Count > 0)
        {
            if (system is not null && (string?)messages[0]?["role"] != "system")
            {
                messages.Insert(0, new JsonObject { ["role"] = "system", ["content"] = system });
            }

            if (kind == RowKind.Text)
            {
                return new JsonObject { ["text"] = string.Join("\n\n", messages.Select(m => (string?)m?["content"] ?? "").Where(t => t.Length > 0)) };
            }

            var result = new JsonObject { ["messages"] = messages };
            if (Tools(row) is { } tools)
            {
                result["tools"] = tools;
            }

            return result;
        }

        if (kind != RowKind.Chat && Text(row) is { Length: > 0 } text)
        {
            return new JsonObject { ["text"] = text };
        }

        return null;
    }

    /// <summary>
    /// A preference row as <c>{"prompt": [messages], "chosen": [messages], "rejected": [messages]}</c> (with "tools" when the
    /// row has them), or null when the row is not a preference pair. Read: TRL's explicit-prompt layouts (a prompt as a
    /// message list or a string, the answers as message lists or strings) and its implicit-prompt layout (chosen and
    /// rejected as whole conversations: the messages they share are the prompt, the rest are the answers, as when a dataset
    /// repeats the prompt in both). Message lists are read as <see cref="Normalize(JsonObject, RowKind, ChatMapping?, string?)"/>
    /// reads them (role names, ShareGPT turns, lists stored as JSON strings); a "system" column, or
    /// <paramref name="system"/>, starts a prompt that has no system message.
    /// </summary>
    public static JsonObject? Preference(JsonObject row, string? system = null)
    {
        ArgumentNullException.ThrowIfNull(row);
        if (Turns(row["chosen"], "assistant") is not { Count: > 0 } chosen || Turns(row["rejected"], "assistant") is not { Count: > 0 } rejected)
        {
            return null;
        }

        var prompt = Turns(row["prompt"], "user");
        // A prefix the two share is the prompt (implicit prompts, or a dataset that repeats the prompt in both answers).
        int shared = 0;
        while (shared < chosen.Count - 1 && shared < rejected.Count - 1 && JsonNode.DeepEquals(chosen[shared], rejected[shared]))
        {
            shared++;
        }

        if (shared > 0)
        {
            prompt = [];
            for (int i = 0; i < shared; i++)
            {
                prompt.Add(chosen[0]!.DeepClone());
                chosen.RemoveAt(0);
                rejected.RemoveAt(0);
            }
        }

        if (prompt is not { Count: > 0 })
        {
            return null;
        }

        system ??= Has(row, "system") ? Str(row["system"]) : null;
        if (system is not null && (string?)prompt[0]?["role"] != "system")
        {
            prompt.Insert(0, (JsonNode)Message("system", system));
        }

        var result = new JsonObject { ["prompt"] = prompt, ["chosen"] = chosen, ["rejected"] = rejected };
        if (Tools(row) is { } tools)
        {
            result["tools"] = tools;
        }

        return result;
    }

    // Messages from a message list (or one stored as a JSON string), or a plain string as one message of `role`.
    private static JsonArray? Turns(JsonNode? node, string role)
    {
        switch (node)
        {
            case JsonArray list when list.Count > 0 && list.All(m => m is JsonObject):
                return Messages(list);
            case JsonValue v when v.TryGetValue<string>(out var text):
                if (text.TrimStart().StartsWith('['))
                {
                    try
                    {
                        if (JsonNode.Parse(text) is JsonArray parsed && parsed.Count > 0 && parsed.All(m => m is JsonObject))
                        {
                            return Messages(parsed);
                        }
                    }
                    catch (JsonException)
                    {
                    }
                }

                return text.Length > 0 ? [(JsonNode)Message(role, text)] : null;
            default:
                return null;
        }
    }

    /// <summary>The layout detected in a row, for reports: "messages", "sharegpt", "alpaca", "question/answer" …, "text", or null.</summary>
    public static string? Describe(JsonObject row)
    {
        if (row.ContainsKey("chosen") && row.ContainsKey("rejected"))
        {
            return "preference (chosen/rejected)";
        }

        foreach (var name in new[] { "messages", "conversations", "conversation", "chosen" })
        {
            if (row[name] is JsonArray a && a.Count > 0 && a[0] is JsonObject first)
            {
                return first.ContainsKey("from") ? $"sharegpt ({name})" : name;
            }
        }

        if (row["prompt"] is JsonArray)
        {
            return "prompt/completion messages";
        }

        foreach (var pair in Pairs)
        {
            if (Has(row, pair[0]) && Has(row, pair[1]))
            {
                return pair[0] == "instruction" ? "alpaca" : $"{pair[0]}/{pair[1]}";
            }
        }

        return Text(row) is not null ? "text" : null;
    }

    private static JsonArray? Conversation(JsonObject row)
    {
        foreach (var name in new[] { "messages", "conversations", "conversation", "chosen", "dialogue", "dialog" })
        {
            if (row[name] is JsonArray list && list.Count > 0 && list.All(m => m is JsonObject))
            {
                return Messages(list);
            }

            if (row[name] is JsonValue v && v.TryGetValue<string>(out var json) && json.TrimStart().StartsWith('['))
            {
                try
                {
                    if (JsonNode.Parse(json) is JsonArray parsed && parsed.Count > 0)
                    {
                        return Messages(parsed);
                    }
                }
                catch (JsonException)
                {
                }
            }
        }

        // TRL's conversational prompt-completion: both are message lists.
        if (row["prompt"] is JsonArray prompt && (row["completion"] ?? row["chosen"]) is JsonArray completion)
        {
            var joined = Messages(prompt)!;
            foreach (var m in Messages(completion)!)
            {
                joined.Add((JsonNode)m!.DeepClone());
            }

            return joined;
        }

        foreach (var pair in Pairs)
        {
            if (!Has(row, pair[1]) || !Has(row, pair[0]))
            {
                continue;
            }

            string user = Str(row[pair[0]]);
            if (pair[0] == "instruction" && Has(row, "input"))
            {
                user = $"{user}\n\n{Str(row["input"])}";           // Alpaca: the instruction, then its input
            }

            var messages = new JsonArray();
            if (Has(row, "system"))
            {
                messages.Add((JsonNode)Message("system", Str(row["system"])));
            }

            messages.Add((JsonNode)Message("user", user));
            messages.Add((JsonNode)Message("assistant", Str(row[pair[1]])));
            return messages;
        }

        return null;
    }

    private static JsonArray? Messages(JsonArray list)
    {
        var messages = new JsonArray();
        foreach (var node in list)
        {
            if (node is not JsonObject m)
            {
                return null;
            }

            string role = Role((string?)m["role"] ?? (string?)m["from"] ?? (string?)m["speaker"] ?? "");
            if (role.Length == 0)
            {
                return null;
            }

            var copy = new JsonObject { ["role"] = role, ["content"] = Content(m["content"] ?? m["value"] ?? m["text"]) };
            foreach (var key in MessageFields)
            {
                if (m[key] is { } value)
                {
                    copy[key] = value.DeepClone();
                }
            }

            messages.Add((JsonNode)copy);
        }

        return messages;
    }

    // A known role's name, or the name in lower case. Ignoring case matches lowering it for ASCII names, the usual ones.
    private static string Role(string name)
    {
        if (Ascii.IsValid(name) && Roles.TryGetValue(name, out var role))
        {
            return role;
        }

        string lower = name.ToLowerInvariant();
        return Ascii.IsValid(lower) && Roles.TryGetValue(lower, out role) ? role : lower;
    }

    private static JsonArray Mapped(JsonObject row, ChatMapping mapping)
    {
        var messages = new JsonArray();
        if (mapping.System is { } system && Fill(system, row) is { Length: > 0 } s)
        {
            messages.Add((JsonNode)Message("system", s));
        }

        messages.Add((JsonNode)Message("user", Fill(mapping.User, row)));
        messages.Add((JsonNode)Message("assistant", Fill(mapping.Assistant, row)));
        return messages;
    }

    /// <summary><paramref name="template"/> with each {column} replaced by the row's value; lines left empty are removed.</summary>
    public static string Fill(string template, JsonObject row)
    {
        string text = Placeholder().Replace(template, m => row.TryGetPropertyValue(m.Groups[1].Value, out var v) ? Str(v) : m.Value);
        text = text.Replace("\\n", "\n", StringComparison.Ordinal);
        return BlankLines().Replace(text, "\n\n").Trim();
    }

    [GeneratedRegex(@"\{([A-Za-z0-9_.\-]+)\}")]
    private static partial Regex Placeholder();

    [GeneratedRegex(@"\n{3,}")]
    private static partial Regex BlankLines();

    private static JsonNode? Tools(JsonObject row) => row["tools"] switch
    {
        JsonArray a when a.Count > 0 => a.DeepClone(),
        JsonValue v when v.TryGetValue<string>(out var json) && json.TrimStart().StartsWith('[') => TryParse(json),
        _ => null,
    };

    private static JsonNode? TryParse(string json)
    {
        try
        {
            return JsonNode.Parse(json) is JsonArray { Count: > 0 } a ? a : null;
        }
        catch (JsonException)
        {
            return null;
        }
    }

    private static string? Text(JsonObject row)
    {
        foreach (var name in TextColumns)
        {
            if (row[name] is JsonValue v && v.TryGetValue<string>(out var s))
            {
                return s;
            }
        }

        return null;
    }

    private static JsonObject Message(string role, string content) => new() { ["role"] = role, ["content"] = content };

    private static bool Has(JsonObject row, string column) => row[column] is JsonValue v && Str(v).Length > 0;

    private static string Str(JsonNode? node) => node switch
    {
        null => "",
        JsonValue v when v.TryGetValue<string>(out var s) => s,
        _ => node.ToJsonString(),
    };

    // Content as text: a string, or the text parts of a multi-part content list.
    private static string Content(JsonNode? node) => node switch
    {
        JsonArray parts => string.Concat(parts.Select(p => p is JsonObject o ? (string?)o["text"] ?? "" : Str(p))),
        _ => Str(node),
    };
}
