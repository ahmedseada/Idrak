// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Idrak.Generation;

/// <summary>One message of a conversation.</summary>
/// <param name="Role">"system", "user", "assistant" or "tool".</param>
/// <param name="Content">The text.</param>
/// <param name="Thinking">An assistant's reasoning, kept apart from the answer.</param>
/// <param name="ToolCalls">Functions an assistant asked to call.</param>
/// <param name="ToolName">For role "tool": which function produced this result.</param>
public sealed record ChatMessage(string Role, string Content = "", string? Thinking = null, IReadOnlyList<ToolCall>? ToolCalls = null, string? ToolName = null);

/// <summary>A function the model may call.</summary>
/// <param name="Name">Function name.</param>
/// <param name="Description">What it does.</param>
/// <param name="Parameters">JSON schema of its arguments.</param>
public sealed record ToolDefinition(string Name, string? Description = null, JsonNode? Parameters = null);

/// <summary>A call the model asked for: a function name and JSON arguments.</summary>
public sealed record ToolCall(string Name, JsonObject Arguments);

/// <summary>
/// How a model family writes tool calls in its answers: <paramref name="Open"/> before the call's JSON (empty: the answer
/// itself is the JSON, as Llama 3 writes it), <paramref name="Close"/> after it (empty: the call runs to the end of the
/// turn), whether the JSON is a list of calls (Mistral's [TOOL_CALLS] [{…}, …]), and the key of the arguments
/// ("arguments", or "parameters" for Llama 3). Each call is {"name": …, key: {…}}.
/// </summary>
public sealed record ToolCallFormat(string Open, string Close, bool List = false, string ArgumentsKey = "arguments")
{
    /// <summary>&lt;tool_call&gt;{"name": …, "arguments": {…}}&lt;/tool_call&gt;, one block per call (Hermes, Qwen).</summary>
    public static ToolCallFormat Tagged { get; } = new("<tool_call>", "</tool_call>");
}

/// <summary>Formats a conversation (and the available tools) as the prompt text a chat model was trained on.</summary>
public abstract class ChatTemplate
{
    /// <summary>Opening and closing tags of the reasoning block.</summary>
    public virtual (string Open, string Close) ThinkTags => ("<think>", "</think>");

    /// <summary>How the model writes tool calls (by default &lt;tool_call&gt;{"name": …, "arguments": {…}}&lt;/tool_call&gt;).</summary>
    public virtual ToolCallFormat ToolCalls => ToolCallFormat.Tagged;

    /// <summary>Text that ends an assistant turn; used as a stop sequence.</summary>
    public abstract IReadOnlyList<string> StopSequences { get; }

    /// <summary>
    /// The prompt for the next assistant turn. <paramref name="think"/>: true asks for reasoning, false suppresses it
    /// (the template closes an empty reasoning block), null leaves it to the model.
    /// </summary>
    public abstract string Render(IReadOnlyList<ChatMessage> messages, IReadOnlyList<ToolDefinition> tools, bool? think);
}

/// <summary>
/// The ChatML layout used by Qwen-family models: &lt;|im_start|&gt;role\ncontent&lt;|im_end|&gt;\n per message, tools
/// listed in the system turn inside &lt;tools&gt;&lt;/tools&gt;, tool calls as &lt;tool_call&gt;JSON&lt;/tool_call&gt;, tool
/// results as a user turn with &lt;tool_response&gt;&lt;/tool_response&gt;, and reasoning in &lt;think&gt;&lt;/think&gt;.
/// </summary>
public sealed class ChatMLTemplate : ChatTemplate
{
    private const string Start = "<|im_start|>", End = "<|im_end|>";

    /// <inheritdoc />
    public override IReadOnlyList<string> StopSequences { get; } = [End, Start];

    /// <inheritdoc />
    public override string Render(IReadOnlyList<ChatMessage> messages, IReadOnlyList<ToolDefinition> tools, bool? think)
    {
        var sb = new StringBuilder();
        using var json = new JsonText(sb);
        string system = messages.FirstOrDefault(m => m.Role == "system")?.Content ?? "";
        if (system.Length > 0 || tools.Count > 0)
        {
            sb.Append(Start).Append("system\n").Append(system);
            if (tools.Count > 0)
            {
                sb.Append(system.Length > 0 ? "\n\n" : "").Append("# Tools\n\nYou may call one or more functions. Function signatures:\n<tools>\n");
                for (int i = 0; i < tools.Count; i++)
                {
                    if (i > 0)
                    {
                        sb.Append('\n');
                    }

                    var tool = tools[i];
                    var w = json.Begin();
                    w.WriteStartObject();
                    w.WriteString("type", "function");
                    w.WriteStartObject("function");
                    w.WriteString("name", tool.Name);
                    w.WriteString("description", tool.Description);
                    w.WritePropertyName("parameters");
                    JsonText.Write(w, tool.Parameters);
                    w.WriteEndObject();
                    w.WriteEndObject();
                    json.End();
                }

                sb.Append("\n</tools>\n\nFor each call, return ").Append(ToolCalls.Open)
                  .Append("{\"name\": <function-name>, \"arguments\": <args-json-object>}").Append(ToolCalls.Close);
            }

            sb.Append(End).Append('\n');
        }

        foreach (var m in messages.Where(m => m.Role != "system"))
        {
            switch (m.Role)
            {
                case "tool":
                    sb.Append(Start).Append("user\n<tool_response>\n").Append(m.Content).Append("\n</tool_response>").Append(End).Append('\n');
                    break;
                case "assistant":
                    sb.Append(Start).Append("assistant\n").Append(m.Content);
                    foreach (var call in m.ToolCalls ?? [])
                    {
                        sb.Append('\n').Append(ToolCalls.Open);
                        var w = json.Begin();
                        w.WriteStartObject();
                        w.WriteString("name", call.Name);
                        w.WritePropertyName("arguments");
                        JsonText.Write(w, call.Arguments);
                        w.WriteEndObject();
                        json.End();
                        sb.Append(ToolCalls.Close);
                    }

                    sb.Append(End).Append('\n');
                    break;
                default:
                    sb.Append(Start).Append(m.Role).Append('\n').Append(m.Content).Append(End).Append('\n');
                    break;
            }
        }

        sb.Append(Start).Append("assistant\n");
        if (think == false)
        {
            sb.Append(ThinkTags.Open).Append("\n\n").Append(ThinkTags.Close).Append("\n\n");
        }

        return sb.ToString();
    }

    // Writes JSON (as JsonNode.ToJsonString would) straight into the prompt, reusing one buffer and writer.
    private sealed class JsonText(StringBuilder sb) : IDisposable
    {
        private readonly System.Buffers.ArrayBufferWriter<byte> _buffer = new();
        private Utf8JsonWriter? _writer;

        public Utf8JsonWriter Begin()
        {
            _buffer.ResetWrittenCount();
            if (_writer is null)
            {
                _writer = new Utf8JsonWriter(_buffer, new JsonWriterOptions { MaxDepth = 64, SkipValidation = true });   // as ToJsonString
            }
            else
            {
                _writer.Reset(_buffer);
            }

            return _writer;
        }

        public void End()
        {
            _writer!.Flush();
            var bytes = _buffer.WrittenSpan;
            char[] chars = System.Buffers.ArrayPool<char>.Shared.Rent(Encoding.UTF8.GetMaxCharCount(bytes.Length));
            sb.Append(chars, 0, Encoding.UTF8.GetChars(bytes, chars));
            System.Buffers.ArrayPool<char>.Shared.Return(chars);
        }

        public static void Write(Utf8JsonWriter writer, JsonNode? node)
        {
            if (node is null)
            {
                writer.WriteNullValue();
            }
            else
            {
                node.WriteTo(writer);
            }
        }

        public void Dispose() => _writer?.Dispose();
    }
}

/// <summary>What one piece of streamed assistant output added.</summary>
/// <param name="Content">New answer text.</param>
/// <param name="Thinking">New reasoning text.</param>
/// <param name="ToolCalls">Tool calls completed in this piece.</param>
public sealed record ChatDelta(string Content, string Thinking, IReadOnlyList<ToolCall> ToolCalls)
{
    /// <summary>True when the piece added nothing.</summary>
    public bool IsEmpty => Content.Length == 0 && Thinking.Length == 0 && ToolCalls.Count == 0;
}

/// <summary>
/// Splits streamed assistant text into reasoning (inside the think tags), answer text and tool calls, in the template's
/// <see cref="ToolCallFormat"/>. Feed text as it arrives; partial tags are held back until they can be decided. When the
/// format has no opening text (the answer is the call's JSON), an answer that starts with '{' or '[' is held to the end
/// of the turn and becomes calls if it parses as calls to one of <paramref name="toolNames"/> (any name when null).
/// </summary>
public sealed class ChatOutputParser(ChatTemplate template, bool separateThinking = true, IReadOnlyCollection<string>? toolNames = null)
{
    private enum Mode { Start, Content, Thinking, ToolCall }

    private readonly CharBuffer _pending = new();
    private readonly CharBuffer _toolText = new();
    private int _toolSearched;                                  // leading characters of _toolText known not to hold the closing text
    private readonly StringBuilder _content = new(), _thinking = new();
    private Mode _mode = Mode.Start;
    private bool _afterThinking;
    private bool _answerStarted;                                // bare-JSON calls may only open the answer
    private readonly ToolCallFormat _format = template.ToolCalls;

    /// <summary>Processes new text.</summary>
    public ChatDelta Feed(string text)
    {
        _pending.Append(text);
        return Drain(final: false);
    }

    /// <summary>Flushes everything held back at the end of the stream.</summary>
    public ChatDelta Finish() => Drain(final: true);

    private ChatDelta Drain(bool final)
    {
        var content = _content.Clear();
        var thinking = _thinking.Clear();
        var calls = new List<ToolCall>();
        var (thinkOpen, thinkClose) = template.ThinkTags;
        string callOpen = _format.Open, callClose = _format.Close;

        while (_pending.Length > 0 || (final && _mode == Mode.ToolCall && _toolText.Length > 0))
        {
            var p = _pending.Span;
            switch (_mode)
            {
                case Mode.Start:
                {
                    // Reasoning may only open the turn (after optional whitespace).
                    var trimmed = p.TrimStart();
                    if (trimmed.StartsWith(thinkOpen, StringComparison.Ordinal))
                    {
                        _pending.Remove(p.Length - trimmed.Length + thinkOpen.Length);
                        _mode = Mode.Thinking;
                        continue;
                    }

                    if (!final && (trimmed.Length == 0 || thinkOpen.AsSpan().StartsWith(trimmed, StringComparison.Ordinal)))
                    {
                        return Result();                                          // still undecided
                    }

                    _mode = Mode.Content;
                    continue;
                }

                case Mode.Thinking:
                {
                    int close = p.IndexOf(thinkClose, StringComparison.Ordinal);
                    if (close >= 0)
                    {
                        thinking.Append(p[..close]);
                        _pending.Remove(close + thinkClose.Length);
                        _mode = Mode.Content;
                        _afterThinking = true;                               // drop the blank lines that follow reasoning
                        continue;
                    }

                    int safe = final ? p.Length : SafeLength(p, thinkClose);
                    thinking.Append(p[..safe]);
                    _pending.Remove(safe);
                    return Result();
                }

                case Mode.ToolCall:
                {
                    // Accumulate first, so a closing tag split across chunks is still found; only the new text (and
                    // the closing text's length before it) can hold the first match.
                    _toolText.Append(p);
                    _pending.Clear();
                    var all = _toolText.Span;
                    int from = Math.Max(0, _toolSearched - callClose.Length + 1);
                    int close = callClose.Length > 0 ? all[from..].IndexOf(callClose, StringComparison.Ordinal) : -1;
                    close = close < 0 ? -1 : close + from;
                    if (close < 0 && !final)
                    {
                        _toolSearched = all.Length;
                        return Result();                                   // no closing text: the call runs to the end
                    }

                    var json = close >= 0 ? all[..close] : all;
                    if (close >= 0)
                    {
                        _pending.Append(all[(close + callClose.Length)..]);
                    }

                    if (TryParseCalls(json.ToString()) is { } parsed)
                    {
                        calls.AddRange(parsed);
                    }
                    else
                    {
                        content.Append(callOpen).Append(json).Append(close >= 0 ? callClose : "");   // not a valid call: keep as text
                    }

                    _toolText.Clear();
                    _toolSearched = 0;
                    _mode = Mode.Content;
                    continue;
                }

                default:
                {
                    if (_afterThinking)
                    {
                        int skip = p.IndexOfAnyExcept('\n', '\r');
                        if (skip < 0)
                        {
                            _pending.Clear();
                            return Result();                                   // only newlines so far: keep waiting
                        }

                        _pending.Remove(skip);
                        _afterThinking = false;
                        p = _pending.Span;
                    }

                    if (callOpen.Length == 0)
                    {
                        // The answer itself may be the call's JSON: decided by its first character.
                        if (!_answerStarted)
                        {
                            var trimmed = p.TrimStart();
                            if (trimmed.Length == 0 && !final)
                            {
                                return Result();
                            }

                            _answerStarted = true;
                            if (trimmed.Length > 0 && trimmed[0] is '{' or '[')
                            {
                                _pending.Remove(p.Length - trimmed.Length);
                                _mode = Mode.ToolCall;
                                continue;
                            }
                        }

                        content.Append(p);
                        _pending.Clear();
                        return Result();
                    }

                    int open = p.IndexOf(callOpen, StringComparison.Ordinal);
                    if (open >= 0)
                    {
                        content.Append(p[..open]);
                        _pending.Remove(open + callOpen.Length);
                        _mode = Mode.ToolCall;
                        continue;
                    }

                    int safe = final ? p.Length : SafeLength(p, callOpen);
                    content.Append(p[..safe]);
                    _pending.Remove(safe);
                    return Result();
                }
            }
        }

        return Result();

        ChatDelta Result()
        {
            if (!separateThinking && thinking.Length > 0)
            {
                return new ChatDelta(thinking.Append(content).ToString(), "", calls);
            }

            return new ChatDelta(content.ToString(), thinking.ToString(), calls);
        }
    }

    /// <summary>Length of <paramref name="text"/> that cannot be the start of <paramref name="tag"/>.</summary>
    private static int SafeLength(ReadOnlySpan<char> text, string tag)
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

    // Text held back between chunks: a char array with a moving start, so taking from the front copies nothing.
    private sealed class CharBuffer
    {
        private char[] _chars = new char[64];
        private int _start;

        public int Length { get; private set; }

        public ReadOnlySpan<char> Span => _chars.AsSpan(_start, Length);

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

        public void Remove(int count)
        {
            _start = count == Length ? 0 : _start + count;
            Length -= count;
        }

        public void Clear() => (_start, Length) = (0, 0);
    }

    /// <summary>
    /// Calls written in the answer instead of in the template's format: small models often reply with the call's JSON
    /// ({"name", "arguments"}, or a list of them) as the whole answer, bare or in a ``` block, or end the answer with such
    /// a block after a sentence ("I'll run it: ```json …```"). They count as calls only when the request has tools and
    /// every name is one of them; the text before the block is kept. Otherwise null (the answer stays text).
    /// </summary>
    public (IReadOnlyList<ToolCall> Calls, string Text)? CallsInAnswer(string answer)
    {
        if (toolNames is not { Count: > 0 })
        {
            return null;
        }

        string text = answer.Trim(), before = "";
        if (text.Length >= 8 && text.EndsWith("```", StringComparison.Ordinal))
        {
            // The last fenced block, which must end the answer: ```json\n…\n```
            int open = text.LastIndexOf("```", text.Length - 4, StringComparison.Ordinal);
            int line = open < 0 ? -1 : text.IndexOf('\n', open);
            if (line < 0 || line > text.Length - 4)
            {
                return null;
            }

            before = text[..open].Trim();
            text = text[(line + 1)..^3].Trim();
        }

        if (text.Length == 0 || text[0] is not ('{' or '['))
        {
            return null;
        }

        var calls = TryParseCalls(text);
        return calls is not null && calls.All(c => toolNames.Contains(c.Name)) ? (calls, before) : null;
    }

    // One call ({"name", arguments}) or a list of them; null when the text is not calls (then it stays answer text).
    private List<ToolCall>? TryParseCalls(string json)
    {
        try
        {
            var node = JsonNode.Parse(json.Trim());
            var items = node is JsonArray array ? array.ToList() : [node];
            var calls = new List<ToolCall>();
            foreach (var item in items)
            {
                if (item is not JsonObject o || o["name"] is not JsonValue nameValue || !nameValue.TryGetValue<string>(out var name) || name.Length == 0
                    || (_format.Open.Length == 0 && toolNames is not null && !toolNames.Contains(name)))
                {
                    return null;
                }

                var args = (o[_format.ArgumentsKey] ?? o["arguments"] ?? o["parameters"]) switch
                {
                    JsonObject a => (JsonObject)a.DeepClone(),
                    JsonValue v when v.TryGetValue<string>(out var s) && JsonNode.Parse(s) is JsonObject parsed => parsed,
                    _ => new JsonObject(),
                };
                calls.Add(new ToolCall(name, args));
            }

            return calls.Count > 0 ? calls : null;
        }
        catch (JsonException)
        {
        }
        catch (InvalidOperationException)
        {
        }

        return null;
    }
}
