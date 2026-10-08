// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Generation;

namespace Idrak.Nlp;

/// <summary>
/// A model's own chat template: the Jinja source from its tokenizer_config.json (or chat_template.jinja), rendered the
/// way Hugging Face's <c>apply_chat_template</c> renders it. Messages become dicts with role, content,
/// reasoning_content and tool_calls ([{type: "function", function: {name, arguments}}]); tools become
/// [{type: "function", function: {name, description, parameters}}]; <c>enable_thinking</c> is set when reasoning is
/// requested or suppressed. Because the model's template decides the layout, prompts (and fine-tuning data) match what
/// the model was trained on, whatever the family.
/// </summary>
public sealed class JinjaChatTemplate : ChatTemplate
{
    private readonly JinjaTemplate _template;
    private readonly JinjaTemplate? _toolTemplate;

    /// <summary>A template from its Jinja source.</summary>
    /// <param name="source">The template.</param>
    /// <param name="stopSequences">Text that ends an assistant turn (usually the end-of-turn and end-of-sequence tokens).</param>
    /// <param name="bosToken">The value of <c>bos_token</c>.</param>
    /// <param name="eosToken">The value of <c>eos_token</c>.</param>
    /// <param name="toolSource">A separate template used when tools are given (some models name one "tool_use").</param>
    public JinjaChatTemplate(string source, IReadOnlyList<string> stopSequences, string bosToken = "", string eosToken = "", string? toolSource = null)
    {
        Source = source;
        _template = JinjaTemplate.Parse(source);
        _toolTemplate = toolSource is null ? null : JinjaTemplate.Parse(toolSource);
        StopSequences = stopSequences;
        BosToken = bosToken;
        EosToken = eosToken;
    }

    /// <summary>The Jinja source.</summary>
    public string Source { get; }

    /// <summary>The value of <c>bos_token</c> in the template.</summary>
    public string BosToken { get; }

    /// <summary>The value of <c>eos_token</c> in the template.</summary>
    public string EosToken { get; }

    /// <inheritdoc />
    public override IReadOnlyList<string> StopSequences { get; }

    /// <summary>The reasoning tags the model writes (&lt;think&gt;…&lt;/think&gt; unless set).</summary>
    public (string Open, string Close) ReasoningTags { get; init; } = ("<think>", "</think>");

    /// <summary>
    /// The shape of the model's JSON tool calls (for the "json" format); when not set, read off the template itself by
    /// rendering a probe call (see <see cref="DetectToolCallFormat"/>), so any family's JSON format is parsed as its
    /// template writes it. Setting it selects the "json" format unless <see cref="CallFormatName"/> says otherwise.
    /// </summary>
    public ToolCallFormat? CallFormat { get; init; }

    /// <summary>
    /// The registered tool-call format the model writes calls in (see <see cref="ToolCallFormats"/>); when not set,
    /// detected from the template (<see cref="ToolCallFormats.Detect"/> of <see cref="ProbeToolCalls"/>).
    /// </summary>
    public string? CallFormatName { get; init; }

    private ToolCallFormat? _detectedCalls;
    private string? _detectedFormatName;
    private (string Text, int Shared, int SharedTail)? _probe;
    private bool _probed;

    /// <inheritdoc />
    public override (string Open, string Close) ThinkTags => ReasoningTags;

    /// <inheritdoc />
    public override ToolCallFormat ToolCalls => CallFormat ?? (_detectedCalls ??= DetectToolCallFormat());

    /// <inheritdoc />
    public override string ToolCallFormatName =>
        CallFormatName ?? (CallFormat is not null ? ToolCallFormats.Json : _detectedFormatName ??= ToolCallFormats.Detect(ProbeToolCalls()));

    /// <summary>
    /// What tool-call format detectors see of this template: its source, and the assistant turn rendered with one probe
    /// call, cut to where it differs from the same turn with a plain answer (null when the template renders no calls).
    /// </summary>
    public ToolCallProbe ProbeToolCalls() =>
        new(Source, RenderProbe() is var (text, shared, sharedTail) ? text[shared..(text.Length - sharedTail)] : null);

    /// <summary>
    /// The template's JSON tool-call format: a conversation whose answer is one probe call is rendered next to one whose
    /// answer is plain text; the text they share before and after the answer is the turn's layout, and around the call's
    /// JSON remain its opening and closing text (Qwen/Hermes &lt;tool_call&gt;…&lt;/tool_call&gt;, Mistral [TOOL_CALLS] [ … ],
    /// Llama 3 nothing: the answer is the JSON, with "parameters"). <see cref="ToolCallFormat.Tagged"/> when the template
    /// does not render tool calls as JSON.
    /// </summary>
    public ToolCallFormat DetectToolCallFormat()
    {
        const string Name = ToolCallProbe.FunctionName, Argument = ToolCallProbe.ArgumentName;
        if (RenderProbe() is not var (call, shared, sharedTail))
        {
            return ToolCallFormat.Tagged;
        }

        try
        {
            int nameAt = call.IndexOf(Name, shared, StringComparison.Ordinal);
            if (nameAt < 0)
            {
                return ToolCallFormat.Tagged;
            }

            // The call's JSON: the innermost object around the name; a list when an array encloses it.
            for (int start = call.LastIndexOf('{', nameAt); start >= shared; start = start > 0 ? call.LastIndexOf('{', start - 1) : -1)
            {
                int end = JsonEnd(call, start);
                if (end <= nameAt || JsonNode.Parse(call[start..end]) is not JsonObject o || o["name"]?.ToString() != Name)
                {
                    continue;
                }

                string key = o.FirstOrDefault(p => p.Key != "name" && (p.Value?.ToJsonString().Contains(Argument, StringComparison.Ordinal) ?? false)).Key ?? "arguments";
                bool list = false;
                int before = start - 1;
                while (before >= shared && char.IsWhiteSpace(call[before]))
                {
                    before--;
                }

                if (before >= shared && call[before] == '[' && JsonEnd(call, before) is int listEnd && listEnd > end && JsonNode.Parse(call[before..listEnd]) is JsonArray)
                {
                    (start, end, list) = (before, listEnd, true);
                }

                return new ToolCallFormat(call[shared..start].Trim(), call[end..(call.Length - sharedTail)].Trim(), list, key);
            }
        }
        catch (Exception ex) when (ex is JsonException or InvalidOperationException or ArgumentException)
        {
        }

        return ToolCallFormat.Tagged;
    }

    // The conversation whose answer is the probe call, rendered, with the length it shares with the plain answer's
    // rendering before and after the answer; null when the template renders no call (or fails to render).
    private (string Text, int Shared, int SharedTail)? RenderProbe()
    {
        if (_probed)
        {
            return _probe;
        }

        const string Name = ToolCallProbe.FunctionName, Argument = ToolCallProbe.ArgumentName, Answer = "ns-probe-answer";
        try
        {
            var tool = new ToolDefinition(Name, "Probe.", new JsonObject
            {
                ["type"] = "object",
                ["properties"] = new JsonObject { [Argument] = new JsonObject { ["type"] = "string" } },
            });
            var user = new ChatMessage("user", "ns-probe-question");
            string plain = Render([user, new ChatMessage("assistant", Answer)], [tool], null, addGenerationPrompt: false);
            string call = Render([user, new ChatMessage("assistant", "", ToolCalls: [new ToolCall(Name, new JsonObject { [Argument] = ToolCallProbe.ArgumentValue })])],
                [tool], null, addGenerationPrompt: false);
            int answerAt = plain.IndexOf(Answer, StringComparison.Ordinal);
            int shared = 0;
            while (shared < answerAt && shared < call.Length && plain[shared] == call[shared])
            {
                shared++;
            }

            int tail = plain.Length - answerAt - Answer.Length, sharedTail = 0;
            while (sharedTail < tail && sharedTail < call.Length - shared && plain[^(sharedTail + 1)] == call[^(sharedTail + 1)])
            {
                sharedTail++;
            }

            if (answerAt >= 0 && call.IndexOf(Name, shared, StringComparison.Ordinal) >= 0)
            {
                _probe = (call, shared, sharedTail);
            }
        }
        catch (Exception ex) when (ex is JsonException or InvalidOperationException or ArgumentException or FormatException or KeyNotFoundException)
        {
        }

        _probed = true;
        return _probe;
    }

    // The index just past the JSON value that starts at `start` (an object or array), or -1 when it does not close.
    private static int JsonEnd(string text, int start)
    {
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

    /// <summary>Extra template variables (for example a date or a model-specific switch).</summary>
    public IReadOnlyDictionary<string, object?> Variables { get; init; } = new Dictionary<string, object?>();

    /// <inheritdoc />
    public override string Render(IReadOnlyList<ChatMessage> messages, IReadOnlyList<ToolDefinition> tools, bool? think) =>
        Render(messages, tools, think, addGenerationPrompt: true);

    /// <summary>
    /// Renders the conversation. With <paramref name="addGenerationPrompt"/> false the text ends after the last message,
    /// as training data is rendered.
    /// </summary>
    public string Render(IReadOnlyList<ChatMessage> messages, IReadOnlyList<ToolDefinition> tools, bool? think, bool addGenerationPrompt)
    {
        ArgumentNullException.ThrowIfNull(messages);
        ChatParts.ThrowIfUnsupported(messages, PartKinds, "The chat template " + nameof(JinjaChatTemplate));
        return RenderValues(ToValues(messages), tools, think, addGenerationPrompt);
    }

    /// <summary>
    /// The kinds of message parts the template renders: text, and "image" when it writes something for an image part. A
    /// message with an image is given to the template as Hugging Face gives it, its content a list of parts
    /// (<c>{"type": "image"}</c>, <c>{"type": "text", "text": ...}</c>); a text-only message keeps a string. Found by
    /// rendering a probe: a user turn whose content is one text part must render as the same text given as a string (the
    /// template reads lists of parts), and adding an image part before it must change the turn (Gemma 3's writes
    /// <c>&lt;start_of_image&gt;</c>).
    /// </summary>
    public override IReadOnlySet<string> PartKinds => _partKinds ??= DetectPartKinds();

    private IReadOnlySet<string>? _partKinds;

    private IReadOnlySet<string> DetectPartKinds()
    {
        const string Probe = "ns-probe-text";
        try
        {
            static List<object?> Turn(object? content) => [new Dictionary<string, object?> { ["role"] = "user", ["content"] = content }];
            var text = new Dictionary<string, object?> { ["type"] = "text", ["text"] = Probe };
            string plain = RenderValues(Turn(Probe), [], null, addGenerationPrompt: false);
            string parts = RenderValues(Turn(new List<object?> { text }), [], null, addGenerationPrompt: false);
            string image = RenderValues(Turn(new List<object?> { new Dictionary<string, object?> { ["type"] = "image" }, text }), [], null, addGenerationPrompt: false);
            if (plain == parts && image != parts && image.Contains(Probe, StringComparison.Ordinal))
            {
                return System.Collections.Frozen.FrozenSet.Create(StringComparer.Ordinal, ChatParts.Text, ChatParts.Image);
            }
        }
        catch (Exception ex) when (ex is JsonException or InvalidOperationException or ArgumentException or FormatException or KeyNotFoundException or InvalidCastException)
        {
        }

        return ChatParts.TextOnly;
    }

    private string RenderValues(List<object?> messages, IReadOnlyList<ToolDefinition> tools, bool? think, bool addGenerationPrompt)
    {
        var variables = new Dictionary<string, object?>(Variables)
        {
            ["messages"] = messages,
            ["tools"] = tools.Count > 0 ? tools.Select(ToValue).ToList<object?>() : null,
            ["add_generation_prompt"] = addGenerationPrompt,
            ["bos_token"] = BosToken,
            ["eos_token"] = EosToken,
        };
        if (think is bool enabled)
        {
            variables["enable_thinking"] = enabled;
        }

        return (tools.Count > 0 && _toolTemplate is not null ? _toolTemplate : _template).Render(variables);
    }

    /// <summary>
    /// Loads the chat template of the model in <paramref name="folder"/> (chat_template.jinja, chat_template.json or the
    /// "chat_template" of tokenizer_config.json), or null when it has none. Stop sequences are the end-of-sequence token
    /// and every token listed as eos_token_id in generation_config.json or config.json.
    /// </summary>
    public static JinjaChatTemplate? Load(string folder, ITokenizer? tokenizer = null)
    {
        var config = ReadJson(Path.Combine(folder, "tokenizer_config.json"));
        string? source = null, toolSource = null;
        string jinjaFile = Path.Combine(folder, "chat_template.jinja");
        if (File.Exists(jinjaFile))
        {
            source = File.ReadAllText(jinjaFile);
        }
        else if (ReadJson(Path.Combine(folder, "chat_template.json"))?["chat_template"] is { } separate)
        {
            (source, toolSource) = Choose(separate);
        }
        else if (config?["chat_template"] is { } inline)
        {
            (source, toolSource) = Choose(inline);
        }

        if (source is null)
        {
            return null;
        }

        string bos = TokenText(config?["bos_token"]), eos = TokenText(config?["eos_token"]);
        var stops = new List<string>();
        if (eos.Length > 0)
        {
            stops.Add(eos);
        }

        foreach (var file in new[] { "generation_config.json", "config.json" })
        {
            if (ReadJson(Path.Combine(folder, file))?["eos_token_id"] is { } ids && tokenizer is not null)
            {
                foreach (var id in ids is JsonArray array ? array.Select(i => (int)i!) : [(int)ids])
                {
                    if (id >= 0 && id < tokenizer.VocabularySize && tokenizer.TokenOf(id) is { Length: > 0 } token && !stops.Contains(token))
                    {
                        stops.Add(token);
                    }
                }
            }
        }

        return new JinjaChatTemplate(source, stops, bos, eos, toolSource);
    }

    private static (string? Default, string? Tools) Choose(JsonNode node)
    {
        if (node is JsonArray named)
        {
            // [{"name": "default", "template": …}, {"name": "tool_use", "template": …}]
            string? Named(string name) => (string?)named.FirstOrDefault(n => (string?)n?["name"] == name)?["template"];
            return (Named("default") ?? (string?)named[0]?["template"], Named("tool_use"));
        }

        return ((string?)node, null);
    }

    private static string TokenText(JsonNode? node) => node switch
    {
        JsonObject o => (string?)o["content"] ?? "",
        JsonValue v when v.TryGetValue<string>(out var s) => s,
        _ => "",
    };

    private static JsonObject? ReadJson(string path) =>
        File.Exists(path) ? JsonNode.Parse(File.ReadAllText(path)) as JsonObject : null;

    // Tool calls get ids ("call00000", nine characters as some templates require), and each tool result the id of the
    // earliest call not yet answered.
    private static List<object?> ToValues(IReadOnlyList<ChatMessage> messages)
    {
        var values = new List<object?>();
        var pending = new Queue<string>();
        int next = 0;
        foreach (var message in messages)
        {
            var value = ToValue(message, () =>
            {
                string id = $"call{next++:D5}";
                pending.Enqueue(id);
                return id;
            });
            if (message.Role == "tool" && pending.TryDequeue(out var answered))
            {
                value["tool_call_id"] = answered;
            }

            values.Add(value);
        }

        return values;
    }

    private static Dictionary<string, object?> ToValue(ChatMessage message, Func<string> newId)
    {
        // A message with a part other than text gets its content as a list of parts, as Hugging Face's processors give it.
        object? content = message.Parts.All(p => p is ChatText) ? message.Content : message.Parts.Select(PartValue).ToList();
        var value = new Dictionary<string, object?> { ["role"] = message.Role, ["content"] = content };
        if (message.Thinking is not null)
        {
            value["reasoning_content"] = message.Thinking;
            value["thinking"] = message.Thinking;                   // the name GPT-OSS's template reads
        }

        if (message.ToolCalls is { Count: > 0 } calls)
        {
            value["tool_calls"] = calls.Select(c => (object?)new Dictionary<string, object?>
            {
                ["type"] = "function",
                ["id"] = newId(),
                ["function"] = new Dictionary<string, object?> { ["name"] = c.Name, ["arguments"] = FromJson(c.Arguments) },
            }).ToList();
        }

        if (message.ToolName is not null)
        {
            value["name"] = message.ToolName;
        }

        return value;
    }

    // A content part as templates read it: {"type": "text", "text": ...}, {"type": "image"}, or {"type": kind} for others.
    private static object? PartValue(ChatPart part) => part switch
    {
        ChatText text => new Dictionary<string, object?> { ["type"] = ChatParts.Text, ["text"] = text.Text },
        _ => new Dictionary<string, object?> { ["type"] = part.Kind },
    };

    private static Dictionary<string, object?> ToValue(ToolDefinition tool)
    {
        var function = new Dictionary<string, object?> { ["name"] = tool.Name };
        if (tool.Description is not null)
        {
            function["description"] = tool.Description;
        }

        if (tool.Parameters is not null)
        {
            function["parameters"] = FromJson(tool.Parameters);
        }

        return new Dictionary<string, object?> { ["type"] = "function", ["function"] = function };
    }

    /// <summary>A JSON value as a template value (dicts keep their key order).</summary>
    internal static object? FromJson(JsonNode? node) => node switch
    {
        null => null,
        JsonObject o => o.ToDictionary(p => p.Key, p => FromJson(p.Value)),
        JsonArray a => a.Select(FromJson).ToList(),
        JsonValue v => v.GetValueKind() switch
        {
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            JsonValueKind.String => v.GetValue<string>(),
            JsonValueKind.Number => Number(v),
            _ => null,
        },
        _ => node.ToJsonString(),
    };

    // A long when the number's JSON text is an integer that fits, else a double; parsed JSON is read as its raw UTF-8 text.
    private static object Number(JsonValue v)
    {
        if (v.TryGetValue(out JsonElement element))
        {
            var utf8 = System.Runtime.InteropServices.JsonMarshal.GetRawUtf8Value(element);
            return long.TryParse(utf8, NumberStyles.Integer, CultureInfo.InvariantCulture, out long l)
                ? (object)l : double.Parse(utf8, NumberStyles.Float, CultureInfo.InvariantCulture);
        }

        string text = v.ToJsonString();
        return long.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out long n)
            ? (object)n : double.Parse(text, NumberStyles.Float, CultureInfo.InvariantCulture);
    }
}
