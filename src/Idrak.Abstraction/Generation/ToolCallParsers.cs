// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Generation;

/// <summary>
/// The shape most formats share, as a base for a format's parser (the library's "json" and "pythonic", and plug-ins'):
/// answer text, then calls between an opening and a closing marker (the closing one may be missing: the calls run to the
/// end of the turn). A format without an opening marker writes a call as the whole answer, decided by how the answer
/// starts (<see cref="Decide"/>) and held to the end of the turn. Text that may still turn out to be a marker is held back
/// until it can be decided; a body that does not parse (<see cref="Parse"/> returns null) stays answer text.
/// </summary>
public abstract class MarkedToolCallParser : IToolCallParser
{
    private enum Mode { Text, Call, Whole }

    private readonly CharBuffer _pending = new(), _call = new();
    private readonly StringBuilder _content = new();
    private int _searched;                                      // leading characters of _call known not to hold the closing marker
    private Mode _mode;
    private bool _decided;                                      // whole-answer calls may only open the answer

    /// <summary>A parser for one reply made for <paramref name="context"/>.</summary>
    protected MarkedToolCallParser(ToolCallContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        Context = context;
    }

    /// <summary>What the parser was made for: the template and the request's tools.</summary>
    protected ToolCallContext Context { get; }

    /// <summary>The text before calls; empty: calls are only written as the whole answer.</summary>
    protected abstract string Open { get; }

    /// <summary>The text after calls; empty: they run to the end of the turn.</summary>
    protected abstract string Close { get; }

    /// <summary>The calls in the text between the markers (or in the whole answer), or null when it is not calls (it stays text).</summary>
    protected abstract List<ToolCall>? Parse(string body);

    /// <summary>For a format without an opening marker, from the answer's first characters (leading whitespace removed): 1 when it is a call, 0 when not, -1 when more text is needed.</summary>
    protected virtual int Decide(ReadOnlySpan<char> start) => 0;

    /// <inheritdoc/>
    public ChatDelta Feed(ReadOnlySpan<char> text)
    {
        _pending.Append(text);
        return Drain(final: false);
    }

    /// <inheritdoc/>
    public ChatDelta Finish() => Drain(final: true);

    private ChatDelta Drain(bool final)
    {
        var content = _content.Clear();
        List<ToolCall>? calls = null;
        while (true)
        {
            var p = _pending.Span;
            if (!_decided)
            {
                if (Open.Length > 0)
                {
                    _decided = true;
                    continue;
                }

                var trimmed = p.TrimStart();
                if (trimmed.Length == 0 && !final)
                {
                    return Result();
                }

                int decision = trimmed.Length == 0 ? 0 : Decide(trimmed);
                if (decision < 0 && !final)
                {
                    return Result();                                        // still undecided
                }

                _decided = true;
                if (decision > 0)
                {
                    _pending.Remove(p.Length - trimmed.Length);
                    _mode = Mode.Whole;
                }

                continue;
            }

            switch (_mode)
            {
                case Mode.Whole:
                {
                    if (!final)
                    {
                        return Result();
                    }

                    string answer = p.ToString();
                    _pending.Clear();
                    if (Parse(answer) is { } parsed)
                    {
                        (calls ??= []).AddRange(parsed);
                    }
                    else
                    {
                        content.Append(answer);
                    }

                    _mode = Mode.Text;
                    return Result();
                }

                case Mode.Call:
                {
                    // Accumulate first, so a closing marker split across pieces is still found; only the new text (and
                    // the marker's length before it) can hold the first match.
                    string close = Close;
                    _call.Append(p);
                    _pending.Clear();
                    var all = _call.Span;
                    int at = -1;
                    if (close.Length > 0)
                    {
                        int from = Math.Max(0, _searched - close.Length + 1);
                        at = all[from..].IndexOf(close, StringComparison.Ordinal);
                        at = at < 0 ? -1 : at + from;
                    }

                    if (at < 0 && !final)
                    {
                        _searched = all.Length;
                        return Result();
                    }

                    var body = at >= 0 ? all[..at] : all;
                    if (at >= 0)
                    {
                        _pending.Append(all[(at + close.Length)..]);
                    }

                    if (Parse(body.ToString()) is { } parsed)
                    {
                        (calls ??= []).AddRange(parsed);
                    }
                    else
                    {
                        content.Append(Open).Append(body).Append(at >= 0 ? close : "");    // not a valid call: keep as text
                    }

                    _call.Clear();
                    _searched = 0;
                    _mode = Mode.Text;
                    if (_pending.Length == 0)
                    {
                        return Result();
                    }

                    continue;
                }

                default:
                {
                    if (p.Length == 0)
                    {
                        return Result();
                    }

                    string open = Open;
                    if (open.Length == 0)
                    {
                        content.Append(p);
                        _pending.Clear();
                        return Result();
                    }

                    int at = p.IndexOf(open, StringComparison.Ordinal);
                    if (at >= 0)
                    {
                        content.Append(p[..at]);
                        _pending.Remove(at + open.Length);
                        _mode = Mode.Call;
                        continue;
                    }

                    int safe = final ? p.Length : ToolCallText.SafeLength(p, open);
                    content.Append(p[..safe]);
                    _pending.Remove(safe);
                    return Result();
                }
            }
        }

        ChatDelta Result() => new(content.ToString(), "", calls ?? (IReadOnlyList<ToolCall>)[]);
    }
}

/// <summary>JSON calls in the template's <see cref="ToolCallFormat"/>: between tags, as the whole answer, or as a list.</summary>
internal sealed class JsonToolCallParser(ToolCallContext context) : MarkedToolCallParser(context)
{
    private readonly ToolCallFormat _format = context.Template.ToolCalls;

    protected override string Open => _format.Open;

    protected override string Close => _format.Close;

    // A bare answer is a call when it starts with '{' or '[' (or Llama 3.1's <|python_tag|>).
    protected override int Decide(ReadOnlySpan<char> start) =>
        start[0] is '{' or '[' || start.StartsWith(ToolCallText.PythonTag, StringComparison.Ordinal) ? 1
        : ToolCallText.PythonTag.AsSpan().StartsWith(start, StringComparison.Ordinal) ? -1 : 0;

    protected override List<ToolCall>? Parse(string body)
    {
        if (_format.Open.Length > 0)
        {
            return ToolCallText.JsonCalls(body, _format.ArgumentsKey, null);
        }

        // The whole answer: one call or a list; Llama 3.1 may put <|python_tag|> first and separate calls with ';', and
        // calls its built-in tools as name.call(key="value").
        string text = body.Trim();
        bool tagged = text.StartsWith(ToolCallText.PythonTag, StringComparison.Ordinal);
        if (tagged)
        {
            text = text[ToolCallText.PythonTag.Length..].Trim();
        }

        var calls = ToolCallText.JsonCalls(text, _format.ArgumentsKey, Context);
        if (calls is null && text.Length > 0 && text[0] == '{')
        {
            calls = ToolCallText.JsonCallSequence(text, _format.ArgumentsKey, Context);
        }

        if (calls is null && tagged && PythonCalls.Parse(text, builtIn: true) is { } builtIn && builtIn.All(c => Context.Allows(c.Name)))
        {
            calls = builtIn;
        }

        return calls;
    }
}

/// <summary>A Python list of calls as the whole answer: <c>[get_weather(city="Paris", days=3), now()]</c>.</summary>
internal sealed class PythonicToolCallParser(ToolCallContext context) : MarkedToolCallParser(context)
{
    protected override string Open => "";

    protected override string Close => "";

    protected override int Decide(ReadOnlySpan<char> start) =>
        start[0] == '[' || start.StartsWith(ToolCallText.PythonTag, StringComparison.Ordinal) ? 1
        : ToolCallText.PythonTag.AsSpan().StartsWith(start, StringComparison.Ordinal) ? -1 : 0;

    protected override List<ToolCall>? Parse(string body)
    {
        string text = body.Trim();
        if (text.StartsWith(ToolCallText.PythonTag, StringComparison.Ordinal))
        {
            text = text[ToolCallText.PythonTag.Length..].Trim();
        }

        return PythonCalls.Parse(text, builtIn: false) is { } calls && calls.All(c => Context.Allows(c.Name)) ? calls : null;
    }
}

/// <summary>Helpers the tool-call parsers share (the library's and plug-ins').</summary>
public static class ToolCallText
{
    /// <summary>The token Llama 3.1 may write before a call.</summary>
    public const string PythonTag = "<|python_tag|>";

    /// <summary>Length of <paramref name="text"/> that cannot be the start of <paramref name="tag"/>.</summary>
    public static int SafeLength(ReadOnlySpan<char> text, string tag)
    {
        for (int keep = Math.Min(tag.Length - 1, text.Length); keep > 0; keep--)
        {
            if (tag.AsSpan().StartsWith(text[^keep..], StringComparison.Ordinal))
            {
                return text.Length - keep;
            }
        }

        return text.Length;
    }

    /// <summary>The JSON value in <paramref name="text"/>, or null when it does not parse.</summary>
    public static JsonNode? TryParse(string text)
    {
        try
        {
            return JsonNode.Parse(text);
        }
        catch (JsonException)
        {
            return null;
        }
    }

    /// <summary>Arguments as an object: an object, or a string holding one; null for anything else.</summary>
    public static JsonObject? Arguments(JsonNode? node) => node switch
    {
        JsonObject o => o,
        JsonValue v when v.TryGetValue<string>(out var s) && TryParse(s) is JsonObject parsed => parsed,
        _ => null,
    };

    /// <summary>
    /// One call ({"name", <paramref name="argumentsKey"/>}) or a list of them; null when the text is not calls, or when
    /// <paramref name="names"/> is given and a call names a tool it does not allow.
    /// </summary>
    public static List<ToolCall>? JsonCalls(string json, string argumentsKey, ToolCallContext? names)
    {
        try
        {
            var node = JsonNode.Parse(json.Trim());
            return Calls(node is JsonArray array ? [.. array] : [node], argumentsKey, names);
        }
        catch (JsonException)
        {
        }
        catch (InvalidOperationException)
        {
        }

        return null;
    }

    /// <summary>Calls written one after another, separated by ';' or whitespace: {…}; {…}.</summary>
    public static List<ToolCall>? JsonCallSequence(string text, string argumentsKey, ToolCallContext? names)
    {
        var items = new List<JsonNode?>();
        int at = 0;
        while (at < text.Length)
        {
            int end = JsonValueEnd(text, at);
            if (end < 0 || TryParse(text[at..end]) is not JsonObject item)
            {
                return null;
            }

            items.Add(item);
            at = end;
            while (at < text.Length && (char.IsWhiteSpace(text[at]) || text[at] == ';'))
            {
                at++;
            }
        }

        return items.Count > 1 ? Calls(items, argumentsKey, names) : null;
    }

    private static List<ToolCall>? Calls(List<JsonNode?> items, string argumentsKey, ToolCallContext? names)
    {
        var calls = new List<ToolCall>();
        foreach (var item in items)
        {
            if (item is not JsonObject o || o["name"] is not JsonValue nameValue || !nameValue.TryGetValue<string>(out var name) || name.Length == 0
                || (names is not null && !names.Allows(name)))
            {
                return null;
            }

            var args = (o[argumentsKey] ?? o["arguments"] ?? o["parameters"]) switch
            {
                JsonObject a => (JsonObject)a.DeepClone(),
                JsonValue v when v.TryGetValue<string>(out var s) && JsonNode.Parse(s) is JsonObject parsed => parsed,
                _ => new JsonObject(),
            };
            calls.Add(new ToolCall(name, args));
        }

        return calls.Count > 0 ? calls : null;
    }

    /// <summary>The index just past the JSON object or array that starts at <paramref name="start"/>, or -1 when it does not close there.</summary>
    public static int JsonValueEnd(string text, int start)
    {
        if (start >= text.Length || text[start] is not ('{' or '['))
        {
            return -1;
        }

        int depth = 0;
        bool inString = false;
        for (int i = start; i < text.Length; i++)
        {
            char c = text[i];
            if (inString)
            {
                if (c == '\\')
                {
                    i++;
                }
                else if (c == '"')
                {
                    inString = false;
                }
            }
            else if (c == '"')
            {
                inString = true;
            }
            else if (c is '{' or '[')
            {
                depth++;
            }
            else if (c is '}' or ']' && --depth == 0)
            {
                return i + 1;
            }
        }

        return -1;
    }
}

/// <summary>
/// Python call syntax: <c>[f(a=1, b="x"), g()]</c> or one call, keyword arguments only, values as Python literals
/// (strings, numbers, True/False/None, lists, tuples and dicts; JSON's true/false/null too).
/// </summary>
internal static class PythonCalls
{
    /// <summary>
    /// The calls in <paramref name="text"/>, or null when it is not calls. With <paramref name="builtIn"/> a name ending in
    /// ".call" loses it (Llama 3.1's built-in tools: <c>brave_search.call(query="…")</c>).
    /// </summary>
    public static List<ToolCall>? Parse(string text, bool builtIn)
    {
        int at = 0;
        try
        {
            var calls = new List<ToolCall>();
            Skip(text, ref at);
            bool list = at < text.Length && text[at] == '[';
            if (list)
            {
                at++;
            }

            while (true)
            {
                Skip(text, ref at);
                if (list && at < text.Length && text[at] == ']' && calls.Count > 0)
                {
                    at++;
                    break;
                }

                int nameStart = at;
                while (at < text.Length && (char.IsLetterOrDigit(text[at]) || text[at] is '_' or '.' or '-'))
                {
                    at++;
                }

                string name = text[nameStart..at];
                if (name.Length == 0 || char.IsDigit(name[0]))
                {
                    return null;
                }

                if (builtIn && name.EndsWith(".call", StringComparison.Ordinal))
                {
                    name = name[..^5];
                }

                Skip(text, ref at);
                Expect(text, ref at, '(');
                var arguments = new JsonObject();
                while (true)
                {
                    Skip(text, ref at);
                    if (Peek(text, at) == ')')
                    {
                        at++;
                        break;
                    }

                    int keyStart = at;
                    while (at < text.Length && (char.IsLetterOrDigit(text[at]) || text[at] == '_'))
                    {
                        at++;
                    }

                    string key = text[keyStart..at];
                    Skip(text, ref at);
                    if (key.Length == 0 || Peek(text, at) != '=')
                    {
                        return null;                                         // positional arguments have no name to map to
                    }

                    at++;
                    arguments[key] = Value(text, ref at);
                    Skip(text, ref at);
                    if (Peek(text, at) == ',')
                    {
                        at++;
                    }
                    else if (Peek(text, at) != ')')
                    {
                        return null;
                    }
                }

                calls.Add(new ToolCall(name, arguments));
                Skip(text, ref at);
                if (!list)
                {
                    break;
                }

                if (Peek(text, at) == ',')
                {
                    at++;
                }
                else if (Peek(text, at) != ']')
                {
                    return null;
                }
            }

            Skip(text, ref at);
            return at == text.Length && calls.Count > 0 ? calls : null;
        }
        catch (FormatException)
        {
            return null;
        }
    }

    private static JsonNode? Value(string text, ref int at)
    {
        Skip(text, ref at);
        char c = Peek(text, at);
        switch (c)
        {
            case '"' or '\'':
                return JsonValue.Create(String(text, ref at));
            case '[' or '(':
            {
                char close = c == '[' ? ']' : ')';
                at++;
                var array = new JsonArray();
                while (true)
                {
                    Skip(text, ref at);
                    if (Peek(text, at) == close)
                    {
                        at++;
                        return array;
                    }

                    array.Add(Value(text, ref at));
                    Skip(text, ref at);
                    if (Peek(text, at) == ',')
                    {
                        at++;
                    }
                    else if (Peek(text, at) != close)
                    {
                        throw new FormatException();
                    }
                }
            }

            case '{':
            {
                at++;
                var obj = new JsonObject();
                while (true)
                {
                    Skip(text, ref at);
                    if (Peek(text, at) == '}')
                    {
                        at++;
                        return obj;
                    }

                    var key = Value(text, ref at);
                    Skip(text, ref at);
                    Expect(text, ref at, ':');
                    obj[key?.ToString() ?? "None"] = Value(text, ref at);
                    Skip(text, ref at);
                    if (Peek(text, at) == ',')
                    {
                        at++;
                    }
                    else if (Peek(text, at) != '}')
                    {
                        throw new FormatException();
                    }
                }
            }

            default:
            {
                int start = at;
                while (at < text.Length && (char.IsLetterOrDigit(text[at]) || text[at] is '_' or '.' or '-' or '+'))
                {
                    at++;
                }

                string word = text[start..at];
                return word switch
                {
                    "True" or "true" => JsonValue.Create(true),
                    "False" or "false" => JsonValue.Create(false),
                    "None" or "null" => null,
                    _ when long.TryParse(word, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out long l) => JsonValue.Create(l),
                    _ when word.Length > 0 && (char.IsDigit(word[0]) || word[0] is '-' or '+' or '.')
                        && double.TryParse(word, NumberStyles.Float, CultureInfo.InvariantCulture, out double d) => JsonValue.Create(d),
                    _ => throw new FormatException(),
                };
            }
        }
    }

    // A quoted string ('…' or "…", or triple-quoted), with Python's common escapes.
    private static string String(string text, ref int at)
    {
        char quote = text[at];
        bool triple = at + 2 < text.Length && text[at + 1] == quote && text[at + 2] == quote;
        at += triple ? 3 : 1;
        var sb = new StringBuilder();
        while (at < text.Length)
        {
            char c = text[at];
            if (c == quote && (!triple || (at + 2 < text.Length && text[at + 1] == quote && text[at + 2] == quote)))
            {
                at += triple ? 3 : 1;
                return sb.ToString();
            }

            if (c == '\\' && at + 1 < text.Length)
            {
                char e = text[++at];
                switch (e)
                {
                    case 'n': sb.Append('\n'); break;
                    case 't': sb.Append('\t'); break;
                    case 'r': sb.Append('\r'); break;
                    case '0': sb.Append('\0'); break;
                    case 'u' when at + 4 < text.Length && int.TryParse(text.AsSpan(at + 1, 4), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out int code):
                        sb.Append((char)code);
                        at += 4;
                        break;
                    default: sb.Append(e); break;
                }

                at++;
                continue;
            }

            sb.Append(c);
            at++;
        }

        throw new FormatException();
    }

    private static char Peek(string text, int at) => at < text.Length ? text[at] : '\0';

    private static void Expect(string text, ref int at, char c)
    {
        if (Peek(text, at) != c)
        {
            throw new FormatException();
        }

        at++;
    }

    private static void Skip(string text, ref int at)
    {
        while (at < text.Length && char.IsWhiteSpace(text[at]))
        {
            at++;
        }
    }
}

/// <summary>
/// Text held back between pieces of a streamed answer (for tool-call parsers): a char array with a moving start, so
/// taking from the front copies nothing.
/// </summary>
public sealed class CharBuffer
{
    private char[] _chars = new char[64];
    private int _start;

    /// <summary>The characters held.</summary>
    public int Length { get; private set; }

    /// <summary>The text held (valid until the next change).</summary>
    public ReadOnlySpan<char> Span => _chars.AsSpan(_start, Length);

    /// <summary>Adds <paramref name="text"/> at the end.</summary>
    public void Append(ReadOnlySpan<char> text)
    {
        if (_start + Length + text.Length > _chars.Length)
        {
            var chars = Length + text.Length > _chars.Length ? new char[Math.Max(_chars.Length * 2, Length + text.Length)] : _chars;
            _chars.AsSpan(_start, Length).CopyTo(chars);
            (_chars, _start) = (chars, 0);
        }

        text.CopyTo(_chars.AsSpan(_start + Length));
        Length += text.Length;
    }

    /// <summary>Takes the first <paramref name="count"/> characters away.</summary>
    public void Remove(int count)
    {
        _start = count == Length ? 0 : _start + count;
        Length -= count;
    }

    /// <summary>Takes every character away.</summary>
    public void Clear() => (_start, Length) = (0, 0);
}
