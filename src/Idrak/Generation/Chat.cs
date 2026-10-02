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
/// ("arguments", or "parameters" for Llama 3). Each call is {"name": …, key: {…}}. This is the shape of the "json"
/// tool-call format; formats that are not JSON are registered in <see cref="ToolCallFormats"/>.
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

    /// <summary>
    /// The shape of JSON tool calls, used by the "json" format (by default
    /// &lt;tool_call&gt;{"name": …, "arguments": {…}}&lt;/tool_call&gt;).
    /// </summary>
    public virtual ToolCallFormat ToolCalls => ToolCallFormat.Tagged;

    /// <summary>The name of the registered tool-call format (see <see cref="ToolCallFormats"/>) the model writes calls in; "json" by default.</summary>
    public virtual string ToolCallFormatName => ToolCallFormats.Json;

    /// <summary>
    /// A parser for the tool calls of one reply: by default one of the format named by <see cref="ToolCallFormatName"/>.
    /// Override it to parse calls without registering a format.
    /// </summary>
    public virtual IToolCallParser CreateToolCallParser(ToolCallContext context) => ToolCallFormats.Create(ToolCallFormatName, context);

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
/// Splits streamed assistant text into reasoning (inside the think tags), answer text and tool calls. The reasoning is
/// taken out here; the answer goes through the template's tool-call parser (<see cref="ChatTemplate.CreateToolCallParser"/>,
/// one of the <see cref="ToolCallFormats"/>), which finds the calls in the model's format. Feed text as it arrives;
/// partial tags are held back until they can be decided. In the default JSON format without opening text (the answer
/// is the call's JSON), an answer that starts with '{' or '[' is held to the end of the turn and becomes calls if it
/// parses as calls to one of the request's tools (any name when none are known).
/// </summary>
public sealed class ChatOutputParser
{
    private enum Mode { Start, Content, Thinking }

    private readonly ChatTemplate _template;
    private readonly bool _separateThinking;
    private readonly IReadOnlyCollection<string>? _toolNames;
    private readonly IToolCallParser _calls;
    private readonly CharBuffer _pending = new();
    private readonly StringBuilder _content = new(), _thinking = new();
    private Mode _mode = Mode.Start;
    private bool _afterThinking;

    /// <summary>A parser for one reply rendered with <paramref name="template"/>.</summary>
    /// <param name="template">The template the prompt was rendered with.</param>
    /// <param name="separateThinking">false: reasoning is returned as answer text.</param>
    /// <param name="toolNames">The names calls may have (any name when null).</param>
    public ChatOutputParser(ChatTemplate template, bool separateThinking = true, IReadOnlyCollection<string>? toolNames = null)
        : this(template, null, toolNames, separateThinking)
    {
    }

    /// <summary>
    /// A parser for one reply to a request with <paramref name="tools"/> (calls must name one of them when given; formats
    /// whose arguments are text, such as "qwen3-coder", type them by the tools' schemas).
    /// </summary>
    /// <param name="template">The template the prompt was rendered with.</param>
    /// <param name="tools">The request's tools; null when it has none (calls may then name anything).</param>
    /// <param name="separateThinking">false: reasoning is returned as answer text.</param>
    public ChatOutputParser(ChatTemplate template, IReadOnlyList<ToolDefinition>? tools, bool separateThinking = true)
        : this(template, tools, tools?.Select(t => t.Name).ToHashSet(), separateThinking)
    {
    }

    private ChatOutputParser(ChatTemplate template, IReadOnlyList<ToolDefinition>? tools, IReadOnlyCollection<string>? toolNames, bool separateThinking)
    {
        ArgumentNullException.ThrowIfNull(template);
        _template = template;
        _separateThinking = separateThinking;
        _toolNames = toolNames;
        _calls = template.CreateToolCallParser(new ToolCallContext(template, tools ?? [], toolNames));
    }

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
        List<ToolCall>? calls = null;
        var (thinkOpen, thinkClose) = _template.ThinkTags;

        while (_pending.Length > 0)
        {
            var p = _pending.Span;
            if (_mode == Mode.Start)
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

            if (_mode == Mode.Thinking)
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

                int safe = final ? p.Length : ToolCallText.SafeLength(p, thinkClose);
                thinking.Append(p[..safe]);
                _pending.Remove(safe);
                break;
            }

            if (_afterThinking)
            {
                int skip = p.IndexOfAnyExcept('\n', '\r');
                if (skip < 0)
                {
                    _pending.Clear();
                    break;                                                 // only newlines so far: keep waiting
                }

                _pending.Remove(skip);
                _afterThinking = false;
                p = _pending.Span;
            }

            // The answer: the tool-call parser decides what is text and what is a call.
            Add(_calls.Feed(p.ToString()));
            _pending.Clear();
        }

        if (final)
        {
            Add(_calls.Finish());
        }

        return Result();

        void Add(ChatDelta delta)
        {
            content.Append(delta.Content);
            thinking.Append(delta.Thinking);
            if (delta.ToolCalls.Count > 0)
            {
                (calls ??= []).AddRange(delta.ToolCalls);
            }
        }

        ChatDelta Result()
        {
            IReadOnlyList<ToolCall> completed = calls ?? (IReadOnlyList<ToolCall>)[];
            if (!_separateThinking && thinking.Length > 0)
            {
                return new ChatDelta(thinking.Append(content).ToString(), "", completed);
            }

            return new ChatDelta(content.ToString(), thinking.ToString(), completed);
        }
    }

    /// <summary>
    /// Calls written in the answer instead of in the template's format: small models often reply with the call's JSON
    /// ({"name", "arguments"}, or a list of them) as the whole answer, bare or in a ``` block, or end the answer with such
    /// a block after a sentence ("I'll run it: ```json …```"). They count as calls only when the request has tools and
    /// every name is one of them; the text before the block is kept. Otherwise null (the answer stays text).
    /// </summary>
    public (IReadOnlyList<ToolCall> Calls, string Text)? CallsInAnswer(string answer)
    {
        if (_toolNames is not { Count: > 0 })
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

        var calls = ToolCallText.JsonCalls(text, _template.ToolCalls.ArgumentsKey, null);
        return calls is not null && calls.All(c => _toolNames.Contains(c.Name)) ? (calls, before) : null;
    }
}
