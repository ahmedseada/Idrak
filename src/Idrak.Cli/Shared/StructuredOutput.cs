// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json;
using System.Text.Json.Nodes;

namespace Idrak.Cli.Shared;

/// <summary>
/// JSON answers that follow a JSON schema (<c>run --schema</c>; serve may reuse it). The library has no constrained
/// sampling yet (plans/plug-in.md gap 13), so the schema is given to the model as an instruction and the answer is
/// checked afterwards: it must hold one JSON value whose types, required properties, items and enums match.
/// </summary>
internal static class StructuredOutput
{
    /// <summary>Reads a JSON schema file (an object).</summary>
    public static JsonObject Read(string path)
    {
        if (!File.Exists(path))
        {
            throw new UsageException($"Schema file not found: {path}");
        }

        try
        {
            return JsonNode.Parse(File.ReadAllText(path)) as JsonObject ?? throw new UsageException($"{path} is not a JSON schema (an object).");
        }
        catch (JsonException e)
        {
            throw new UsageException($"{path} is not JSON: {e.Message}");
        }
    }

    /// <summary>The system prompt with the instruction to answer in the schema's shape.</summary>
    public static string Instruction(string? system, JsonObject schema) =>
        (system is { Length: > 0 } ? system + "\n\n" : "")
        + "Answer with one JSON value that matches this JSON schema, and nothing else (no explanation, no code fence):\n"
        + schema.ToJsonString();

    /// <summary>The JSON value in <paramref name="answer"/> and the first way it breaks <paramref name="schema"/> (null when it matches).</summary>
    public static (JsonNode? Value, string? Problem) Check(string answer, JsonObject schema)
    {
        string text = answer.Trim();
        if (text.StartsWith("```", StringComparison.Ordinal))
        {
            int start = text.IndexOf('\n'), end = text.LastIndexOf("```", StringComparison.Ordinal);
            text = start > 0 && end > start ? text[(start + 1)..end].Trim() : text;
        }

        int first = text.IndexOfAny(['{', '[']);
        int last = text.LastIndexOfAny(['}', ']']);
        if (first > 0 && last > first && !LooksLikeScalar(text))
        {
            text = text[first..(last + 1)];
        }

        JsonNode? value;
        try
        {
            value = JsonNode.Parse(text);
        }
        catch (JsonException e)
        {
            return (null, $"the answer is not JSON ({e.Message})");
        }

        return (value, Validate(schema, value, "$"));
    }

    private static bool LooksLikeScalar(string text) => text is "true" or "false" or "null" || text.StartsWith('"') || double.TryParse(text, out _);

    // The first mismatch of value against schema, as "path: what" (null when it matches).
    internal static string? Validate(JsonNode? schema, JsonNode? value, string path)
    {
        if (schema is not JsonObject s)
        {
            return null;
        }

        if (s["enum"] is JsonArray options && !options.Any(o => JsonNode.DeepEquals(o, value)))
        {
            return $"{path} must be one of {string.Join(", ", options.Select(o => o?.ToJsonString() ?? "null"))}";
        }

        var types = s["type"] switch
        {
            JsonValue v when v.TryGetValue(out string? one) => [one],
            JsonArray many => many.Select(t => (string?)t).OfType<string>().ToList(),
            _ => new List<string>(),
        };
        if (types.Count > 0 && !types.Any(t => Is(t, value)))
        {
            return $"{path} must be {string.Join(" or ", types)}, not {Kind(value)}";
        }

        if (value is JsonObject obj)
        {
            foreach (var name in (s["required"] as JsonArray ?? []).Select(n => (string?)n).OfType<string>())
            {
                if (!obj.ContainsKey(name))
                {
                    return $"{path} lacks the required property '{name}'";
                }
            }

            if (s["properties"] is JsonObject properties)
            {
                foreach (var (name, child) in obj)
                {
                    if (properties[name] is { } property && Validate(property, child, $"{path}.{name}") is { } problem)
                    {
                        return problem;
                    }
                }
            }
        }

        if (value is JsonArray array && s["items"] is JsonObject items)
        {
            for (int i = 0; i < array.Count; i++)
            {
                if (Validate(items, array[i], $"{path}[{i}]") is { } problem)
                {
                    return problem;
                }
            }
        }

        return null;
    }

    private static string Kind(JsonNode? value) => value is null ? "null" : value.GetValueKind().ToString().ToLowerInvariant();

    private static bool Is(string type, JsonNode? value)
    {
        var kind = value?.GetValueKind() ?? JsonValueKind.Null;
        return type switch
        {
            "object" => kind == JsonValueKind.Object,
            "array" => kind == JsonValueKind.Array,
            "string" => kind == JsonValueKind.String,
            "boolean" => kind is JsonValueKind.True or JsonValueKind.False,
            "number" => kind == JsonValueKind.Number,
            "integer" => kind == JsonValueKind.Number && value!.GetValue<double>() is var d && d == Math.Floor(d),
            "null" => kind == JsonValueKind.Null,
            _ => true,
        };
    }
}
