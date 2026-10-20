// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Text.Plugins.Qwen;

/// <summary>
/// Qwen3-Coder's XML calls: <c>&lt;tool_call&gt;\n&lt;function=name&gt;\n&lt;parameter=key&gt;\nvalue\n&lt;/parameter&gt;\n&lt;/function&gt;\n&lt;/tool_call&gt;</c>.
/// Values are text; they are typed by the tool's schema (numbers, booleans, objects and arrays as JSON), else kept as
/// strings (objects and arrays still read as JSON).
/// </summary>
internal sealed class Qwen3CoderToolCallParser(ToolCallContext context) : MarkedToolCallParser(context)
{
    private const string FunctionOpen = "<function=", FunctionClose = "</function>", ParameterOpen = "<parameter=", ParameterClose = "</parameter>";

    protected override string Open => "<tool_call>";

    protected override string Close => "</tool_call>";

    protected override List<ToolCall>? Parse(string body)
    {
        var calls = new List<ToolCall>();
        int at = body.IndexOf(FunctionOpen, StringComparison.Ordinal);
        if (at < 0 || body.AsSpan(0, at).Trim().Length > 0)
        {
            return null;
        }

        while (at >= 0)
        {
            int nameEnd = body.IndexOf('>', at);
            if (nameEnd < 0)
            {
                return null;
            }

            string name = body[(at + FunctionOpen.Length)..nameEnd].Trim();
            int end = body.IndexOf(FunctionClose, nameEnd, StringComparison.Ordinal);
            int next = body.IndexOf(FunctionOpen, nameEnd, StringComparison.Ordinal);
            int stop = end >= 0 && (next < 0 || end < next) ? end : next >= 0 ? next : body.Length;
            if (name.Length == 0)
            {
                return null;
            }

            var properties = Context.Tool(name)?.Parameters?["properties"] as JsonObject;
            var arguments = new JsonObject();
            string inner = body[(nameEnd + 1)..stop];
            for (int p = inner.IndexOf(ParameterOpen, StringComparison.Ordinal); p >= 0;)
            {
                int keyEnd = inner.IndexOf('>', p);
                if (keyEnd < 0)
                {
                    return null;
                }

                string key = inner[(p + ParameterOpen.Length)..keyEnd].Trim();
                int close = inner.IndexOf(ParameterClose, keyEnd, StringComparison.Ordinal);
                int following = inner.IndexOf(ParameterOpen, keyEnd, StringComparison.Ordinal);
                int valueEnd = close >= 0 && (following < 0 || close < following) ? close : following >= 0 ? following : inner.Length;
                arguments[key] = Value(Unwrap(inner[(keyEnd + 1)..valueEnd]), properties?[key]?["type"]);
                p = following;
            }

            calls.Add(new ToolCall(name, arguments));
            at = next;
        }

        return calls;
    }

    // The template writes a newline after the opening and before the closing tag.
    private static string Unwrap(string value)
    {
        if (value.StartsWith('\n'))
        {
            value = value[1..];
        }

        return value.EndsWith('\n') ? value[..^1] : value;
    }

    private static JsonNode? Value(string text, JsonNode? type)
    {
        var types = type switch
        {
            JsonValue v when v.TryGetValue<string>(out var t) => [t],
            JsonArray a => a.Select(t => t?.ToString() ?? "").ToArray(),
            _ => Array.Empty<string>(),
        };
        if (types.Contains("string") || types.Length == 0)
        {
            // Text, unless no type is known and it reads as a JSON object or array.
            return types.Length == 0 && text.TrimStart() is ['{' or '[', ..] && ToolCallText.TryParse(text) is { } json ? json : JsonValue.Create(text);
        }

        string trimmed = text.Trim();
        if (types.Contains("null") && trimmed is "null" or "None")
        {
            return null;
        }

        if (types.Contains("boolean") && trimmed.ToLowerInvariant() is "true" or "false")
        {
            return JsonValue.Create(trimmed.Equals("true", StringComparison.OrdinalIgnoreCase));
        }

        return ToolCallText.TryParse(trimmed) ?? JsonValue.Create(text);
    }
}
