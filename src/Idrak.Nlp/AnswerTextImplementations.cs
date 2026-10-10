// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Nlp.Abstractions;

namespace Idrak.Nlp;

// The library's answer texts (AnswerTexts).
internal sealed class RawAnswerText : IAnswerText
{
    public string Name => AnswerTexts.Raw;

    public string Summary => "the answer as it is";

    public string Text(string answer) => answer;
}

internal sealed class JsonValuesAnswerText : IAnswerText
{
    public string Name => AnswerTexts.Values;

    public string Summary => "the string and number values of the answer's JSON in document order, one a line (the JSON read leniently; an answer without JSON as it is)";

    public string Text(string answer)
    {
        if (JsonRepairs.Parse(answer) is not { } found)
        {
            return answer;
        }

        var values = new List<string>();
        Collect(found.Json, values);
        return string.Join("\n", values);
    }

    private static void Collect(JsonNode? node, List<string> values)
    {
        switch (node)
        {
            case JsonObject obj:
                foreach (var (_, value) in obj)
                {
                    Collect(value, values);
                }

                break;
            case JsonArray array:
                foreach (var item in array)
                {
                    Collect(item, values);
                }

                break;
            case JsonValue value when value.GetValueKind() is JsonValueKind.String:
                if (value.GetValue<string>() is { Length: > 0 } s)
                {
                    values.Add(s);
                }

                break;
            case JsonValue value when value.GetValueKind() is JsonValueKind.Number:
                values.Add(value.ToJsonString());
                break;
        }
    }
}
