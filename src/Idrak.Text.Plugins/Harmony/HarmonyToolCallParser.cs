// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

namespace Idrak.Text.Plugins.Harmony;

/// <summary>
/// GPT-OSS's harmony messages: <c>&lt;|channel|&gt;analysis&lt;|message|&gt;…&lt;|end|&gt;</c> is reasoning,
/// <c>&lt;|start|&gt;assistant&lt;|channel|&gt;final&lt;|message|&gt;…</c> (or commentary without a recipient) is answer text, and a
/// message to a recipient (<c>&lt;|channel|&gt;commentary to=functions.name &lt;|constrain|&gt;json&lt;|message|&gt;{…}&lt;|call|&gt;</c>,
/// or with <c>to=</c> before the channel as the template writes it) is a call to that function. Text without headers
/// is answer text.
/// </summary>
internal sealed class HarmonyToolCallParser : IToolCallParser
{
    private const string Start = "<|start|>", Channel = "<|channel|>", Message = "<|message|>";
    private static readonly string[] Ends = ["<|end|>", "<|call|>", "<|return|>"];
    private static readonly int EndReach = Ends.Max(e => e.Length) - 1;      // how far back an end tag can start before new text
    private static readonly System.Buffers.SearchValues<char> WordEnd = System.Buffers.SearchValues.Create(" \t\n<");

    private enum Mode { Header, Plain, Thinking, Content, Call }

    private readonly CharBuffer _pending = new(), _call = new();
    private readonly StringBuilder _content = new(), _thinking = new();
    private Mode _mode;
    private string _recipient = "";
    private int _searched;                                      // leading characters of _call known not to hold an end tag

    public ChatDelta Feed(ReadOnlySpan<char> text)
    {
        _pending.Append(text);
        return Drain(final: false);
    }

    public ChatDelta Finish() => Drain(final: true);

    private ChatDelta Drain(bool final)
    {
        var content = _content.Clear();
        var thinking = _thinking.Clear();
        List<ToolCall>? calls = null;
        while (true)
        {
            var p = _pending.Span;
            switch (_mode)
            {
                case Mode.Header:
                {
                    var trimmed = p.TrimStart();
                    if (trimmed.Length == 0)
                    {
                        if (final)
                        {
                            _pending.Clear();
                        }

                        return Result();
                    }

                    // A header starts a message, or continues the prompt's "<|start|>assistant" with a recipient or a channel.
                    if (!Could(trimmed, Start) && !Could(trimmed, Channel) && !Could(trimmed, "to="))
                    {
                        _mode = Mode.Plain;
                        continue;
                    }

                    int message = p.IndexOf(Message, StringComparison.Ordinal);
                    if (message < 0)
                    {
                        if (final)
                        {
                            _pending.Clear();                                    // a header cut off by the end of the turn
                        }

                        return Result();
                    }

                    var header = p[..message];
                    _recipient = Recipient(header);
                    string channel = Word(header, Channel);
                    _mode = _recipient.Length > 0 ? Mode.Call : channel == "analysis" ? Mode.Thinking : Mode.Content;
                    _pending.Remove(message + Message.Length);
                    continue;
                }

                case Mode.Plain:
                {
                    int start = Math.Min(Index(p, Start), Index(p, Channel));
                    if (start < p.Length)
                    {
                        content.Append(p[..start]);
                        _pending.Remove(start);
                        _mode = Mode.Header;
                        continue;
                    }

                    int safe = final ? p.Length : Math.Min(ToolCallText.SafeLength(p, Start), ToolCallText.SafeLength(p, Channel));
                    content.Append(p[..safe]);
                    _pending.Remove(safe);
                    return Result();
                }

                case Mode.Call:
                {
                    _call.Append(p);
                    _pending.Clear();
                    var all = _call.Span;

                    // Only the new text (and an end tag's length before it) can hold the first end: the call is not
                    // searched again from its start at every piece.
                    int from = Math.Max(0, _searched - EndReach);
                    var (end, length) = FirstEnd(all[from..]);
                    end = end < 0 ? -1 : end + from;
                    if (end < 0 && !final)
                    {
                        _searched = all.Length;
                        return Result();
                    }

                    var body = end >= 0 ? all[..end] : all;
                    string name = _recipient.StartsWith("functions.", StringComparison.Ordinal) ? _recipient["functions.".Length..] : _recipient;
                    if (ToolCallText.Arguments(ToolCallText.TryParse(body.ToString())) is { } arguments)
                    {
                        (calls ??= []).Add(new ToolCall(name, arguments));
                    }
                    else
                    {
                        content.Append(body);                                    // not JSON arguments: keep as text
                    }

                    if (end >= 0)
                    {
                        _pending.Append(all[(end + length)..]);
                    }

                    _call.Clear();
                    _searched = 0;
                    _mode = Mode.Header;
                    continue;
                }

                default:
                {
                    var target = _mode == Mode.Thinking ? thinking : content;
                    var (end, length) = FirstEnd(p);
                    if (end >= 0)
                    {
                        target.Append(p[..end]);
                        _pending.Remove(end + length);
                        _mode = Mode.Header;
                        continue;
                    }

                    int safe = p.Length;
                    if (!final)
                    {
                        foreach (var tag in Ends)
                        {
                            safe = Math.Min(safe, ToolCallText.SafeLength(p, tag));
                        }
                    }

                    target.Append(p[..safe]);
                    _pending.Remove(safe);
                    return Result();
                }
            }
        }

        ChatDelta Result() => new(content.ToString(), thinking.ToString(), calls ?? (IReadOnlyList<ToolCall>)[]);
    }

    // Whether `text` starts with `tag` or could still become it.
    private static bool Could(ReadOnlySpan<char> text, string tag) =>
        text.StartsWith(tag, StringComparison.Ordinal) || tag.AsSpan().StartsWith(text, StringComparison.Ordinal);

    private static int Index(ReadOnlySpan<char> text, string tag)
    {
        int at = text.IndexOf(tag, StringComparison.Ordinal);
        return at < 0 ? text.Length : at;
    }

    private static (int At, int Length) FirstEnd(ReadOnlySpan<char> text)
    {
        int at = -1, length = 0;
        foreach (var tag in Ends)
        {
            int i = text.IndexOf(tag, StringComparison.Ordinal);
            if (i >= 0 && (at < 0 || i < at))
            {
                (at, length) = (i, tag.Length);
            }
        }

        return (at, length);
    }

    // The word after `marker` in a header (to whitespace or the next special token).
    private static string Word(ReadOnlySpan<char> header, string marker)
    {
        int at = header.IndexOf(marker, StringComparison.Ordinal);
        if (at < 0)
        {
            return "";
        }

        var rest = header[(at + marker.Length)..].TrimStart();
        int end = rest.IndexOfAny(WordEnd);
        return (end < 0 ? rest : rest[..end]).ToString();
    }

    // The message's recipient ("functions.get_weather"), or empty.
    private static string Recipient(ReadOnlySpan<char> header)
    {
        for (int at = header.IndexOf("to=", StringComparison.Ordinal); at >= 0;)
        {
            if (at == 0 || header[at - 1] is ' ' or '\t' or '\n' or '>')
            {
                var rest = header[(at + 3)..];
                int end = rest.IndexOfAny(WordEnd);
                return (end < 0 ? rest : rest[..end]).ToString();
            }

            int next = header[(at + 3)..].IndexOf("to=", StringComparison.Ordinal);
            at = next < 0 ? -1 : at + 3 + next;
        }

        return "";
    }
}
