// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Text.Plugins.DeepSeek;

/// <summary>
/// DeepSeek's special-token calls: <c>&lt;｜tool▁calls▁begin｜&gt;</c>, then per call
/// <c>&lt;｜tool▁call▁begin｜&gt;function&lt;｜tool▁sep｜&gt;name\n```json\n{…}\n```&lt;｜tool▁call▁end｜&gt;</c> (V3, R1)
/// or <c>&lt;｜tool▁call▁begin｜&gt;name&lt;｜tool▁sep｜&gt;{…}&lt;｜tool▁call▁end｜&gt;</c> (V3.1), then <c>&lt;｜tool▁calls▁end｜&gt;</c>.
/// </summary>
internal sealed class DeepSeekToolCallParser(ToolCallContext context) : MarkedToolCallParser(context)
{
    internal const string CallsBegin = "<｜tool▁calls▁begin｜>", CallsEnd = "<｜tool▁calls▁end｜>";
    internal const string CallBegin = "<｜tool▁call▁begin｜>", CallEnd = "<｜tool▁call▁end｜>", Separator = "<｜tool▁sep｜>";

    protected override string Open => CallsBegin;

    protected override string Close => CallsEnd;

    protected override List<ToolCall>? Parse(string body)
    {
        var calls = new List<ToolCall>();
        for (int at = body.IndexOf(CallBegin, StringComparison.Ordinal); at >= 0; at = body.IndexOf(CallBegin, at, StringComparison.Ordinal))
        {
            at += CallBegin.Length;
            int end = body.IndexOf(CallEnd, at, StringComparison.Ordinal);
            string call = end >= 0 ? body[at..end] : body[at..];
            int separator = call.IndexOf(Separator, StringComparison.Ordinal);
            if (separator < 0)
            {
                return null;
            }

            string name = call[..separator].Trim(), rest = call[(separator + Separator.Length)..];
            if (name == "function")
            {
                // V3 and R1: the type, then the name on its own line and the arguments in a fenced block.
                int line = rest.IndexOf('\n');
                name = (line < 0 ? rest : rest[..line]).Trim();
                rest = line < 0 ? "" : rest[(line + 1)..];
            }

            string json = rest.Trim();
            if (json.StartsWith("```", StringComparison.Ordinal))
            {
                int line = json.IndexOf('\n');
                json = line < 0 ? "" : json[(line + 1)..];
                json = (json.EndsWith("```", StringComparison.Ordinal) ? json[..^3] : json).Trim();
            }

            if (name.Length == 0 || ToolCallText.Arguments(ToolCallText.TryParse(json)) is not { } arguments)
            {
                return null;
            }

            calls.Add(new ToolCall(name, arguments));
            if (end < 0)
            {
                break;
            }

            at = end + CallEnd.Length;
        }

        return calls.Count > 0 ? calls : null;
    }
}
