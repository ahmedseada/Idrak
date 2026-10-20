// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Text.Plugins.Mistral;

/// <summary>
/// Mistral's calls after [TOOL_CALLS], to the end of the turn: a JSON list of {"name", "arguments"} (v3 tokenizers) or,
/// per call, <c>[TOOL_CALLS]name[ARGS]{…}</c> (newer ones; a <c>[CALL_ID]id</c> before [ARGS] is skipped).
/// </summary>
internal sealed class MistralToolCallParser(ToolCallContext context) : MarkedToolCallParser(context)
{
    internal const string Marker = "[TOOL_CALLS]";
    private const string Args = "[ARGS]", CallId = "[CALL_ID]";

    protected override string Open => Marker;

    protected override string Close => "";

    protected override List<ToolCall>? Parse(string body)
    {
        var calls = new List<ToolCall>();
        foreach (var part in body.Split(Marker))
        {
            string text = part.Trim();
            if (text.Length == 0)
            {
                continue;
            }

            if (text[0] is '[' or '{')
            {
                if (ToolCallText.JsonCalls(text, "arguments", null) is not { } listed)
                {
                    return null;
                }

                calls.AddRange(listed);
                continue;
            }

            int args = text.IndexOf(Args, StringComparison.Ordinal);
            if (args <= 0)
            {
                return null;
            }

            string name = text[..args];
            int id = name.IndexOf(CallId, StringComparison.Ordinal);
            name = (id >= 0 ? name[..id] : name).Trim();
            string json = text[(args + Args.Length)..].Trim();
            int end = ToolCallText.JsonValueEnd(json, 0);
            if (name.Length == 0 || end < 0 || ToolCallText.Arguments(ToolCallText.TryParse(json[..end])) is not { } arguments)
            {
                return null;
            }

            calls.Add(new ToolCall(name, arguments));
        }

        return calls.Count > 0 ? calls : null;
    }
}
