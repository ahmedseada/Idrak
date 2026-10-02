// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Generation;
using Idrak.LanguageModels;
using Formats = Idrak.Generation.ToolCallFormats;

// Tool-call formats: each built-in format detected from a family's published chat template (cut to the parts that
// render tools, messages and calls), parsed from realistic output streamed in pieces split anywhere, and the parsed
// calls rendered back through the template and parsed again.
internal static partial class Tests
{
    private static readonly ToolDefinition[] ParserTools =
    [
        new("get_weather", "Weather.", JsonNode.Parse("""{"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}, "metric": {"type": "boolean"}}}""")),
        new("get_time", "Time.", JsonNode.Parse("""{"type": "object", "properties": {"timezone": {"type": "string"}}}""")),
        new("write_file", "Write.", JsonNode.Parse("""{"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}""")),
        new("brave_search", "Search.", JsonNode.Parse("""{"type": "object", "properties": {"query": {"type": "string"}}}""")),
    ];

    private static void ToolCallFormatsDetectAndParse(Device device)
    {
        _ = device;
        var llama = new JinjaChatTemplate(LlamaStyleTemplate, ["<|eot_id|>"], "<|begin_of_text|>");
        var pythonic = new JinjaChatTemplate(LlamaPythonicTemplate, ["<|eot_id|>"], "<|begin_of_text|>");
        var qwenCoder = new JinjaChatTemplate(Qwen3CoderTemplate, ["<|im_end|>"]);
        var mistralV3 = new JinjaChatTemplate(MistralStyleTemplate, ["</s>"], "<s>", "</s>");
        var mistral = new JinjaChatTemplate(MistralArgsTemplate, ["</s>"], "<s>", "</s>");
        var harmony = new JinjaChatTemplate(HarmonyTemplate, ["<|return|>", "<|call|>"]);
        var deepSeek31 = new JinjaChatTemplate(DeepSeekV31Template, ["<｜end▁of▁sentence｜>"], "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>");
        var deepSeekR1 = new JinjaChatTemplate(DeepSeekR1Template, ["<｜end▁of▁sentence｜>"], "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>");
        var qwen = new JinjaChatTemplate(Qwen3Template, ["<|im_end|>"]);

        foreach (var (name, template, expected) in new[]
        {
            ("qwen", qwen, Formats.Json), ("llama", llama, Formats.Json), ("pythonic", pythonic, Formats.Pythonic),
            ("qwen3-coder", qwenCoder, Formats.Qwen3Coder), ("mistral v3", mistralV3, Formats.Mistral), ("mistral", mistral, Formats.Mistral),
            ("harmony", harmony, Formats.Harmony), ("deepseek v3.1", deepSeek31, Formats.DeepSeek), ("deepseek r1", deepSeekR1, Formats.DeepSeek),
        })
        {
            Check(template.ToolCallFormatName == expected, $"{name}: detected '{template.ToolCallFormatName}', expected '{expected}'");
        }

        var weather = ParsedCall("get_weather", """{"city": "San Francisco", "days": 3}""");
        var time = ParsedCall("get_time", """{"timezone": "Europe/Paris"}""");
        var cases = new (string Name, ChatTemplate Template, string Output, string Content, string Thinking, ToolCall[] Calls)[]
        {
            ("llama, python tag", llama, "<|python_tag|>{\"name\": \"get_weather\", \"parameters\": {\"city\": \"San Francisco\", \"days\": 3}}", "", "", [weather]),
            ("llama, two calls", llama, "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"San Francisco\", \"days\": 3}}; {\"name\": \"get_time\", \"parameters\": {\"timezone\": \"Europe/Paris\"}}", "", "", [weather, time]),
            ("llama, built-in tool", llama, "<|python_tag|>brave_search.call(query=\"weather in Paris\")", "", "", [ParsedCall("brave_search", """{"query": "weather in Paris"}""")]),
            ("llama, text", llama, "It is sunny {mostly}.", "It is sunny {mostly}.", "", []),
            ("pythonic", pythonic, "[get_weather(city=\"San Francisco\", days=3), get_time(timezone='Europe/Paris')]", "", "", [weather, time]),
            ("pythonic, literals", pythonic, " [get_weather(city=\"Rome\\n\", days=-2, metric=True), write_file(path='a.json', content=None)]", "", "",
                [ParsedCall("get_weather", """{"city": "Rome\n", "days": -2, "metric": true}"""), ParsedCall("write_file", """{"path": "a.json", "content": null}""")]),
            ("pythonic, nested", pythonic, "<|python_tag|>[write_file(path=\"x\", content={'a': [1, 2.5, \"b\"], 'c': False})]", "", "",
                [ParsedCall("write_file", """{"path": "x", "content": {"a": [1, 2.5, "b"], "c": false}}""")]),
            ("pythonic, unknown tool stays text", pythonic, "[delete_all(force=True)]", "[delete_all(force=True)]", "", []),
            ("pythonic, list stays text", pythonic, "[1, 2, 3] are the numbers.", "[1, 2, 3] are the numbers.", "", []),
            ("pythonic, text", pythonic, "The answer is [sunny].", "The answer is [sunny].", "", []),
            ("qwen3-coder", qwenCoder, "I'll check the weather.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nSan Francisco\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=write_file>\n<parameter=path>\nsrc/a.py\n</parameter>\n<parameter=content>\ndef f():\n    return 1 < 2\n\n</parameter>\n</function>\n</tool_call>",
                "I'll check the weather.\n\n\n", "", [weather, ParsedCall("write_file", """{"path": "src/a.py", "content": "def f():\n    return 1 < 2\n"}""")]),
            ("qwen3-coder, typed", qwenCoder, "<tool_call>\n<function=get_weather>\n<parameter=city>\n42\n</parameter>\n<parameter=days>\n7\n</parameter>\n<parameter=metric>\nfalse\n</parameter>\n</function>\n</tool_call>", "", "",
                [ParsedCall("get_weather", """{"city": "42", "days": 7, "metric": false}""")]),
            ("mistral", mistral, "[TOOL_CALLS]get_weather[ARGS]{\"city\": \"San Francisco\", \"days\": 3}[TOOL_CALLS]get_time[ARGS]{\"timezone\": \"Europe/Paris\"}", "", "", [weather, time]),
            ("mistral, call id", mistral, "Sure.[TOOL_CALLS]get_time[CALL_ID]a1b2c3d4e[ARGS]{\"timezone\": \"Europe/Paris\"}", "Sure.", "", [time]),
            ("mistral, list", mistral, "[TOOL_CALLS][{\"name\": \"get_weather\", \"arguments\": {\"city\": \"San Francisco\", \"days\": 3}}, {\"name\": \"get_time\", \"arguments\": {\"timezone\": \"Europe/Paris\"}}]", "", "", [weather, time]),
            ("harmony, call", harmony, "<|channel|>analysis<|message|>The user wants the weather. Call the tool.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"San Francisco\",\"days\":3}",
                "", "The user wants the weather. Call the tool.", [weather]),
            ("harmony, call token", harmony, "<|channel|>analysis<|message|>Time.<|end|><|start|>assistant to=functions.get_time<|channel|>commentary json<|message|>{\"timezone\": \"Europe/Paris\"}<|call|>", "", "Time.", [time]),
            ("harmony, final", harmony, "<|channel|>analysis<|message|>Simple greeting.<|end|><|start|>assistant<|channel|>final<|message|>Hello! How can I help?", "Hello! How can I help?", "Simple greeting.", []),
            ("harmony, plain text", harmony, "Hello <b>there</b>.", "Hello <b>there</b>.", "", []),
            ("deepseek v3.1", deepSeek31, "Let me check.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"city\": \"San Francisco\", \"days\": 3}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"timezone\": \"Europe/Paris\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
                "Let me check.", "", [weather, time]),
            ("deepseek r1", deepSeekR1, "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\": \"San Francisco\", \"days\": 3}\n```<｜tool▁call▁end｜>\n<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_time\n```json\n{\"timezone\": \"Europe/Paris\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
                "", "", [weather, time]),
            ("deepseek, broken call stays text", deepSeek31, "<｜tool▁calls▁begin｜>oops<｜tool▁calls▁end｜>", "<｜tool▁calls▁begin｜>oops<｜tool▁calls▁end｜>", "", []),
        };

        foreach (var (name, template, output, content, thinking, calls) in cases)
        {
            foreach (int size in new[] { 1, 2, 3, 5, 7, 1000 })
            {
                var (gotContent, gotThinking, gotCalls) = ParseStreamed(template, output, size, ParserTools);
                Check(gotContent == content, $"{name}, pieces of {size}: content '{gotContent}', expected '{content}'");
                Check(gotThinking == thinking, $"{name}, pieces of {size}: thinking '{gotThinking}'");
                Check(SameCalls(gotCalls, calls), $"{name}, pieces of {size}: calls {Show(gotCalls)}, expected {Show(calls)}");
            }
        }

        // The parsed calls go back into each template's message shape, and what the template writes parses again.
        var trips = new (string Name, JinjaChatTemplate Template, ToolCall[] Calls)[]
        {
            ("qwen", qwen, [weather, time]), ("llama", llama, [weather]), ("pythonic", pythonic, [ParsedCall("get_weather", """{"days": 3, "metric": true}"""), ParsedCall("get_time", "{}")]),
            ("qwen3-coder", qwenCoder, [weather, ParsedCall("write_file", """{"path": "a.py", "content": "x = 1\ny = [1, 2]"}""")]),
            ("mistral v3", mistralV3, [weather, time]), ("mistral", mistral, [weather, time]), ("harmony", harmony, [weather]),
            ("deepseek v3.1", deepSeek31, [weather, time]), ("deepseek r1", deepSeekR1, [weather, time]),
        };
        foreach (var (name, template, calls) in trips)
        {
            var user = new ChatMessage("user", "What is the weather?");
            string plain = template.Render([user, new ChatMessage("assistant", "rt-answer")], ParserTools, null, addGenerationPrompt: false);
            string written = template.Render([user, new ChatMessage("assistant", "", ToolCalls: calls)], ParserTools, null, addGenerationPrompt: false);
            int shared = 0;
            while (shared < plain.Length && shared < written.Length && plain[shared] == written[shared])
            {
                shared++;
            }

            // As generated: from where the turn differs to the first stop sequence.
            string turn = written[shared..];
            foreach (var stop in template.StopSequences)
            {
                int at = turn.IndexOf(stop, StringComparison.Ordinal);
                turn = at >= 0 ? turn[..at] : turn;
            }

            var (_, _, back) = ParseStreamed(template, turn, 4, ParserTools);
            Check(SameCalls(back, calls), $"{name}: round trip of {Show(calls)} through '{turn}' gave {Show(back)}");
        }

        // A thinking reply's reasoning stays apart from harmony's analysis only by channel; merged when not separated.
        var merged = new ChatOutputParser(harmony, ParserTools, separateThinking: false);
        var m1 = merged.Feed("<|channel|>analysis<|message|>Hmm.<|end|><|start|>assistant<|channel|>final<|message|>Hi.");
        var m2 = merged.Finish();
        Check(m1.Content + m2.Content == "Hmm.Hi." && m1.Thinking + m2.Thinking == "", $"harmony, merged reasoning: '{m1.Content + m2.Content}'");
    }

    private static void ToolCallFormatsRegistry(Device device)
    {
        _ = device;
        string[] builtIn = [Formats.DeepSeek, Formats.Harmony, Formats.Mistral, Formats.Qwen3Coder, Formats.Pythonic, Formats.Json];
        Check(Formats.Names.SequenceEqual(builtIn), $"built-in formats in the order asked: {string.Join(", ", Formats.Names)}");

        // A format of its own: <<name|{json}>>, detected from the template's rendered probe call.
        const string Template = """
            {%- for m in messages %}{{ m.role }}: {{ m.content }}{%- if m.tool_calls %}{%- for c in m.tool_calls %}<<{{ c.function.name }}|{{ c.function.arguments | tojson }}>>{%- endfor %}{%- endif %}
            {% endfor %}
            """;
        var template = new JinjaChatTemplate(Template, ["\n"]);
        Check(template.ToolCallFormatName == Formats.Json, $"before registering: {template.ToolCallFormatName}");
        try
        {
            Formats.Register("angle", probe => probe.Call?.Contains("<<" + ToolCallProbe.FunctionName + "|", StringComparison.Ordinal) ?? false, context => new AngleParser(context));
            Check(Formats.Names.First() == "angle", "a new format is asked first");
            var detected = new JinjaChatTemplate(Template, ["\n"]);
            Check(detected.ToolCallFormatName == "angle", $"detected: {detected.ToolCallFormatName}");
            Check(detected.ProbeToolCalls().Call?.Contains("<<ns_probe_function|{\"ns_probe_argument\": \"ns-probe-value\"}>>", StringComparison.Ordinal) == true,
                $"probe call: '{detected.ProbeToolCalls().Call}'");
            var (content, _, calls) = ParseStreamed(detected, "Checking <<get_time|{\"timezone\": \"UTC\"}>> now", 2, ParserTools);
            Check(content == "Checking  now" && calls is [{ Name: "get_time" }], $"registered parser: '{content}' {Show(calls)}");

            // Named explicitly, a template skips detection.
            var named = new JinjaChatTemplate(Template, ["\n"]) { CallFormatName = Formats.Pythonic };
            Check(named.ToolCallFormatName == Formats.Pythonic, "named format");
            var shaped = new JinjaChatTemplate(Template, ["\n"]) { CallFormat = new ToolCallFormat("<<", ">>") };
            Check(shaped.ToolCallFormatName == Formats.Json, "a JSON shape selects the json format");
        }
        finally
        {
            Check(Formats.Unregister("angle"), "unregister");
        }

        Check(!Formats.Unregister("angle"), "unregister twice");
        Check(Formats.Names.SequenceEqual(builtIn), "built-ins left as they were");
        try
        {
            Formats.Create("nope", new ToolCallContext(new ChatMLTemplate(), [], null));
            Check(false, "unknown format accepted");
        }
        catch (NotSupportedException ex)
        {
            Check(ex.Message.Contains("'nope'", StringComparison.Ordinal) && ex.Message.Contains("qwen3-coder, pythonic", StringComparison.Ordinal), ex.Message);
        }

        // A template may make its own parser without registering anything.
        var own = new OwnParserTemplate();
        var (ownContent, _, ownCalls) = ParseStreamed(own, "<think>ok</think>CALL get_time", 3, ParserTools);
        Check(ownContent == "" && ownCalls is [{ Name: "get_time" }], $"template's own parser: '{ownContent}' {Show(ownCalls)}");
    }

    // <<name|{json}>> anywhere in the answer; text around kept.
    private sealed class AngleParser(ToolCallContext context) : IToolCallParser
    {
        private string _text = "";

        public ChatDelta Feed(string text)
        {
            _text += text;
            return new ChatDelta("", "", []);
        }

        public ChatDelta Finish()
        {
            var calls = new List<ToolCall>();
            string content = System.Text.RegularExpressions.Regex.Replace(_text, @"<<(\w+)\|(\{.*?\})>>", m =>
            {
                if (context.Allows(m.Groups[1].Value))
                {
                    calls.Add(new ToolCall(m.Groups[1].Value, (JsonObject)JsonNode.Parse(m.Groups[2].Value)!));
                    return "";
                }

                return m.Value;
            });
            return new ChatDelta(content, "", calls);
        }
    }

    private sealed class OwnParserTemplate : ChatTemplate
    {
        public override IReadOnlyList<string> StopSequences => ["\n"];

        public override string Render(IReadOnlyList<ChatMessage> messages, IReadOnlyList<ToolDefinition> tools, bool? think) => "";

        public override IToolCallParser CreateToolCallParser(ToolCallContext context) => new Command();

        private sealed class Command : IToolCallParser
        {
            private string _text = "";

            public ChatDelta Feed(string text)
            {
                _text += text;
                return new ChatDelta("", "", []);
            }

            public ChatDelta Finish() => _text.StartsWith("CALL ", StringComparison.Ordinal)
                ? new ChatDelta("", "", [new ToolCall(_text[5..], [])]) : new ChatDelta(_text, "", []);
        }
    }

    private static ToolCall ParsedCall(string name, string arguments) => new(name, (JsonObject)JsonNode.Parse(arguments)!);

    private static (string Content, string Thinking, List<ToolCall> Calls) ParseStreamed(ChatTemplate template, string output, int size, IReadOnlyList<ToolDefinition>? tools)
    {
        var parser = new ChatOutputParser(template, tools);
        string content = "", thinking = "";
        var calls = new List<ToolCall>();
        for (int i = 0; i < output.Length; i += size)
        {
            var d = parser.Feed(output.Substring(i, Math.Min(size, output.Length - i)));
            content += d.Content;
            thinking += d.Thinking;
            calls.AddRange(d.ToolCalls);
        }

        var f = parser.Finish();
        return (content + f.Content, thinking + f.Thinking, [.. calls, .. f.ToolCalls]);
    }

    private static bool SameCalls(IReadOnlyList<ToolCall> a, IReadOnlyList<ToolCall> b) =>
        a.Count == b.Count && a.Zip(b).All(p => p.First.Name == p.Second.Name && JsonNode.DeepEquals(p.First.Arguments, p.Second.Arguments));

    private static string Show(IEnumerable<ToolCall> calls) => "[" + string.Join(", ", calls.Select(c => c.Name + c.Arguments.ToJsonString())) + "]";

    // Llama 3.2's pythonic tool-call layout (the community template served with pythonic parsers, cut to the parts that
    // render tools, messages and calls): a Python list of calls, values written with Python's str().
    private const string LlamaPythonicTemplate = """
{{- bos_token }}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "You are a helpful assistant with tool calling capabilities." %}
{%- endif %}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if tools is not none %}
    {{- "Environment: ipython\n" }}
    {{- "You have access to the following functions. To call functions, please respond with a python list of the calls. " }}
    {{- 'Respond in the format [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)] ' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}
{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
    {%- elif 'tool_calls' in message %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n[' -}}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- tool_call.name + '(' -}}
            {%- for param in tool_call.arguments %}
                {{- param + '=' -}}
                {{- "%s" | format(tool_call.arguments[param]) -}}
                {% if not loop.last %}, {% endif %}
            {%- endfor %}
            {{- ')' -}}
            {% if not loop.last %}, {% endif %}
        {%- endfor %}
        {{- ']<|eot_id|>' -}}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping %}
            {{- message.content | tojson }}
        {%- else %}
            {{- { "output": message.content } | tojson }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
""";

    // Qwen3-Coder's chat template, as published.
    private const string Qwen3CoderTemplate = """
{% macro render_extra_keys(json_dict, handled_keys) %}
    {%- if json_dict is mapping %}
        {%- for json_key in json_dict if json_key not in handled_keys %}
            {%- if json_dict[json_key] is mapping or (json_dict[json_key] is sequence and json_dict[json_key] is not string) %}
                {{- '\n<' ~ json_key ~ '>' ~ (json_dict[json_key] | tojson | safe) ~ '</' ~ json_key ~ '>' }}
            {%- else %}
                {{-'\n<' ~ json_key ~ '>' ~ (json_dict[json_key] | string) ~ '</' ~ json_key ~ '>' }}
            {%- endif %}
        {%- endfor %}
    {%- endif %}
{% endmacro %}

{%- if messages[0]["role"] == "system" %}
    {%- set system_message = messages[0]["content"] %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set loop_messages = messages %}
{%- endif %}

{%- if not tools is defined %}
    {%- set tools = [] %}
{%- endif %}

{%- if system_message is defined %}
    {{- "<|im_start|>system\n" + system_message }}
{%- else %}
    {%- if tools is iterable and tools | length > 0 %}
        {{- "<|im_start|>system\nYou are Qwen, a helpful AI assistant that can interact with a computer to solve tasks." }}
    {%- endif %}
{%- endif %}
{%- if tools is iterable and tools | length > 0 %}
    {{- "\n\n# Tools\n\nYou have access to the following tools:\n\n" }}
    {{- "<tools>" }}
    {%- for tool in tools %}
        {%- if tool.function is defined %}
            {%- set tool = tool.function %}
        {%- endif %}
        {{- "\n<function>\n<name>" ~ tool.name ~ "</name>" }}
        {%- if tool.description is defined %}
            {{- '\n<description>' ~ (tool.description | trim) ~ '</description>' }}
        {%- endif %}
        {{- '\n<parameters>' }}
        {%- if tool.parameters is defined and tool.parameters is mapping and tool.parameters.properties is defined and tool.parameters.properties is mapping %}
            {%- for param_name, param_fields in tool.parameters.properties|items %}
                {{- '\n<parameter>' }}
                {{- '\n<name>' ~ param_name ~ '</name>' }}
                {%- if param_fields.type is defined %}
                    {{- '\n<type>' ~ (param_fields.type | string) ~ '</type>' }}
                {%- endif %}
                {%- if param_fields.description is defined %}
                    {{- '\n<description>' ~ (param_fields.description | trim) ~ '</description>' }}
                {%- endif %}
                {%- set handled_keys = ['name', 'type', 'description'] %}
                {{- render_extra_keys(param_fields, handled_keys) }}
                {{- '\n</parameter>' }}
            {%- endfor %}
        {%- endif %}
        {% set handled_keys = ['type', 'properties'] %}
        {{- render_extra_keys(tool.parameters, handled_keys) }}
        {{- '\n</parameters>' }}
        {%- set handled_keys = ['type', 'name', 'description', 'parameters'] %}
        {{- render_extra_keys(tool, handled_keys) }}
        {{- '\n</function>' }}
    {%- endfor %}
    {{- "\n</tools>" }}
    {{- '\n\nIf you choose to call a tool ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nvalue_2\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: the tool calling block MUST begin with an opening <tool_call> tag and end with a closing </tool_call> tag.\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>' }}
{%- endif %}
{%- if system_message is defined %}
    {{- '<|im_end|>\n' }}
{%- else %}
    {%- if tools is iterable and tools | length > 0 %}
        {{- '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in loop_messages %}
    {%- if message.role == "assistant" and message.tool_calls is defined and message.tool_calls is iterable and message.tool_calls | length > 0 %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content is defined and message.content is string and message.content | trim | length > 0 %}
            {{- '\n' + message.content | trim + '\n' }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
            {%- if tool_call.arguments is defined %}
                {%- for args_name, args_value in tool_call.arguments|items %}
                    {{- '<parameter=' + args_name + '>\n' }}
                    {%- set args_value = args_value | tojson | safe if args_value is mapping or (args_value is sequence and args_value is not string) else args_value | string %}
                    {{- args_value }}
                    {{- '\n</parameter>\n' }}
                {%- endfor %}
            {%- endif %}
            {{- '</function>\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "user" or message.role == "system" or message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.previtem and loop.previtem.role != "tool" %}
            {{- '<|im_start|>user\n' }}
        {%- endif %}
        {{- '<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>\n' }}
        {%- if not loop.last and loop.nextitem.role != "tool" %}
            {{- '<|im_end|>\n' }}
        {%- elif loop.last %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- else %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
""";

    // Mistral's newer layout (v7 tokenizers), cut to the parts that render tools, messages and calls (the long default
    // system prompt dropped): [TOOL_CALLS]name[CALL_ID]id[ARGS]{…} per call.
    private const string MistralArgsTemplate = """
{%- set default_system_message = "You are a helpful assistant." %}
{{- bos_token }}
{%- set system_prompt = default_system_message %}
{%- set loop_messages = messages %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}
{%- if messages|length > 0 and messages[0]['role'] == 'system' %}
    {%- set system_prompt = messages[0]['content'] %}
    {%- set loop_messages = messages[1:] %}
{%- endif %}
{%- set user_messages = loop_messages | selectattr("role", "equalto", "user") | list %}
{{- '[SYSTEM_PROMPT]' + system_prompt + '[/SYSTEM_PROMPT]' }}
{%- for message in loop_messages %}
    {%- if message['role'] == 'user' %}
        {%- if tools is not none and (message == user_messages[-1]) %}
            {{- '[AVAILABLE_TOOLS]' + tools|tojson + '[/AVAILABLE_TOOLS]' }}
        {%- endif %}
        {{- '[INST]' + message['content'] + '[/INST]' }}
    {%- elif message['role'] == 'assistant' %}
        {%- if message.get('tool_calls') %}
            {%- for tool_call in message.tool_calls %}
                {{- '[TOOL_CALLS]' + tool_call.function.name }}
                {%- if not tool_call.id is defined or tool_call.id is not string or tool_call.id|length != 9 %}
                    {{- raise_exception("Tool call IDs should be alphanumeric strings with length 9!") }}
                {%- endif %}
                {{- '[CALL_ID]' + tool_call.id }}
                {{- '[ARGS]' + tool_call['function']['arguments']|tojson }}
            {%- endfor %}
            {{- eos_token }}
        {%- else %}
            {{- message['content'] + eos_token }}
        {%- endif %}
    {%- elif message['role'] == 'tool_results' or message['role'] == 'tool' %}
        {{- '[TOOL_RESULTS]' + message.tool_call_id + '[TOOL_CONTENT]' + message.content|string + '[/TOOL_RESULTS]' }}
    {%- endif %}
{%- endfor %}
""";

    // GPT-OSS's harmony layout, cut to the parts that render messages and calls (the system message and the tool
    // namespace simplified; the message loop as published).
    private const string HarmonyTemplate = """
{{- "<|start|>system<|message|>Reasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message." }}
{%- if tools -%}
    {{- "\nCalls to these tools must go to the commentary channel: 'functions'." }}
{%- endif -%}
{{- "<|end|>" }}
{%- if messages[0].role == "developer" or messages[0].role == "system" %}
    {%- set developer_message = messages[0].content %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set developer_message = "" %}
    {%- set loop_messages = messages %}
{%- endif %}
{%- if developer_message or tools %}
    {{- "<|start|>developer<|message|>" }}
    {%- if developer_message %}
        {{- "# Instructions\n\n" }}
        {{- developer_message }}
        {{- "\n\n" }}
    {%- endif %}
    {%- if tools -%}
        {{- "# Tools\n\n" }}
        {{- tools | tojson }}
    {%- endif -%}
    {{- "<|end|>" }}
{%- endif %}
{%- set last_tool_call = namespace(name=none) %}
{%- for message in loop_messages -%}
    {%- if message.role == 'assistant' -%}
        {%- if "tool_calls" in message %}
            {%- set future_final_message = namespace(found=false) %}
            {%- for future_message in loop_messages[loop.index:] %}
                {%- if future_message.role == 'assistant' and "tool_calls" not in future_message %}
                    {%- set future_final_message.found = true %}
                {%- endif %}
            {%- endfor %}
            {%- set tool_call = message.tool_calls[0] %}
            {%- if tool_call.function %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {%- if message.content and message.thinking %}
                {{- raise_exception("Cannot pass both content and thinking in an assistant message with tool calls! Put the analysis message in one or the other, but not both.") }}
            {%- elif message.content and not future_final_message.found %}
                {{- "<|start|>assistant<|channel|>analysis<|message|>" + message.content + "<|end|>" }}
            {%- elif message.thinking and not future_final_message.found %}
                {{- "<|start|>assistant<|channel|>analysis<|message|>" + message.thinking + "<|end|>" }}
            {%- endif %}
            {{- "<|start|>assistant to=" }}
            {{- "functions." + tool_call.name + "<|channel|>commentary " }}
            {{- (tool_call.content_type if tool_call.content_type is defined else "json") + "<|message|>" }}
            {{- tool_call.arguments|tojson }}
            {{- "<|call|>" }}
            {%- set last_tool_call.name = tool_call.name %}
        {%- elif loop.last and not add_generation_prompt %}
            {%- if "thinking" in message %}
                {{- "<|start|>assistant<|channel|>analysis<|message|>" + message.thinking + "<|end|>" }}
            {%- endif %}
            {{- "<|start|>assistant<|channel|>final<|message|>" + message.content + "<|return|>" }}
        {%- else %}
            {{- "<|start|>assistant<|channel|>final<|message|>" + message.content + "<|end|>" }}
            {%- set last_tool_call.name = none %}
        {%- endif %}
    {%- elif message.role == 'tool' -%}
        {%- if last_tool_call.name is none %}
            {{- raise_exception("Message has tool role, but there was no previous assistant message with a tool call!") }}
        {%- endif %}
        {{- "<|start|>functions." + last_tool_call.name }}
        {{- " to=assistant<|channel|>commentary<|message|>" + message.content|tojson + "<|end|>" }}
    {%- elif message.role == 'user' -%}
        {{- "<|start|>user<|message|>" + message.content + "<|end|>" }}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
<|start|>assistant
{%- endif -%}
""";

    // DeepSeek V3.1's chat template, as published: name<｜tool▁sep｜>{…} per call.
    private const string DeepSeekV31Template = """
{% if not add_generation_prompt is defined -%}
  {%- set add_generation_prompt = false -%}
{%- endif -%}
{%- if not thinking is defined -%}
  {%- if enable_thinking is defined -%}
    {%- set thinking = enable_thinking -%}
    {%- else -%}
    {%- set thinking = false -%}
  {%- endif -%}
{%- endif -%}
{%- set ns = namespace(is_first=false, is_tool=false, system_prompt='', is_first_sp=true, is_last_user=false) -%}
{%- for message in messages -%}
  {%- if message['role'] == 'system' -%}
    {%- if ns.is_first_sp -%}
      {%- set ns.system_prompt = ns.system_prompt + message['content'] -%}
      {%- set ns.is_first_sp = false -%}
      {%- else -%}
      {%- set ns.system_prompt = ns.system_prompt + '

' + message['content'] -%}
    {%- endif -%}
  {%- endif -%}
{%- endfor -%}{{ bos_token }}{{ ns.system_prompt }}
{%- for message in messages -%}
  {%- if message['role'] == 'user' -%}
    {%- set ns.is_tool = false -%}
    {%- set ns.is_first = false -%}
    {%- set ns.is_last_user = true -%}{{'<｜User｜>' + message['content']}}
  {%- endif -%}
  {%- if message['role'] == 'assistant' and message['tool_calls'] -%}
    {%- if ns.is_last_user -%}{{'<｜Assistant｜><think></think>'}}
    {%- endif -%}
    {%- set ns.is_last_user = false -%}
    {%- set ns.is_first = false -%}
    {%- set ns.is_tool = false -%}
    {%- for tool in message['tool_calls'] -%}
      {%- if not ns.is_first -%}
        {%- if not message['content'] -%}{{'<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>'+ tool['function']['name'] + '<｜tool▁sep｜>' + tool['function']['arguments'] | tojson + '<｜tool▁call▁end｜>'}}
          {%- else -%}{{message['content'] + '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['function']['name'] + '<｜tool▁sep｜>' + tool['function']['arguments'] | tojson + '<｜tool▁call▁end｜>'}}
        {%- endif -%}
        {%- set ns.is_first = true -%}
        {%- else -%}{{'<｜tool▁call▁begin｜>'+ tool['function']['name'] + '<｜tool▁sep｜>' + tool['function']['arguments'] | tojson + '<｜tool▁call▁end｜>'}}
      {%- endif -%}
    {%- endfor -%}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}
  {%- endif -%}
  {%- if message['role'] == 'assistant' and not message['tool_calls'] -%}
    {%- if ns.is_last_user -%}{{'<｜Assistant｜>'}}
      {%- if message['prefix'] is defined and message['prefix'] and thinking -%}{{'<think>'}}
        {%- else -%}{{'<think></think>'}}
      {%- endif -%}
    {%- endif -%}
    {%- set ns.is_last_user = false -%}
    {%- if ns.is_tool -%}{{message['content'] + '<｜end▁of▁sentence｜>'}}
      {%- set ns.is_tool = false -%}
      {%- else -%}
      {%- set content = message['content'] -%}
      {%- if '</think>' in content -%}
        {%- set content = content.split('</think>', 1)[1] -%}
      {%- endif -%}{{content + '<｜end▁of▁sentence｜>'}}
    {%- endif -%}
  {%- endif -%}
  {%- if message['role'] == 'tool' -%}
    {%- set ns.is_last_user = false -%}
    {%- set ns.is_tool = true -%}{{'<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}
  {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt and ns.is_last_user and not ns.is_tool -%}{{'<｜Assistant｜>'}}
  {%- if not thinking -%}{{'<think></think>'}}
  {%- else -%}{{'<think>'}}
  {%- endif -%}
{%- endif %}
""";

    // DeepSeek R1's chat template (as its distilled models publish it): function<｜tool▁sep｜>name and a fenced JSON block.
    private const string DeepSeekR1Template = """
{% if not add_generation_prompt is defined -%}
  {%- set add_generation_prompt = false -%}
{%- endif -%}
{%- set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') -%}
{%- for message in messages -%}
  {%- if message['role'] == 'system' -%}
    {%- set ns.system_prompt = message['content'] -%}
  {%- endif -%}
{%- endfor -%}{{bos_token}}{{ns.system_prompt}}
{%- for message in messages -%}
  {%- if message['role'] == 'user' -%}
    {%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}
  {%- endif -%}
  {%- if message['role'] == 'assistant' and message['tool_calls'] -%}
    {%- set ns.is_tool = false -%}
    {%- for tool in message['tool_calls']-%}
      {%- if not ns.is_first -%}
        {{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] | tojson + '\n' + '```' + '<｜tool▁call▁end｜>'}}
        {%- set ns.is_first = true -%}
      {%- else -%}
        {{'\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] | tojson + '\n' + '```' + '<｜tool▁call▁end｜>'}}
      {%- endif -%}
    {%- endfor -%}
    {{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}
  {%- endif -%}
  {%- if message['role'] == 'assistant' and message['content'] is not none -%}
    {%- if ns.is_tool -%}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}
      {%- set ns.is_tool = false -%}
    {%- else -%}
      {%- set content = message['content'] -%}
      {%- if '</think>' in content -%}
        {%- set content = content.split('</think>')[-1] -%}
      {%- endif -%}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}
    {%- endif -%}
  {%- endif -%}
  {%- if message['role'] == 'tool' -%}
    {%- set ns.is_tool = true -%}
    {%- if ns.is_output_first -%}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}
      {%- set ns.is_output_first = false -%}
      {%- else -%}{{'\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}
    {%- endif -%}
  {%- endif -%}
{%- endfor -%}
{%- if ns.is_tool -%}{{'<｜tool▁outputs▁end｜>'}}
{%- endif -%}
{%- if add_generation_prompt and not ns.is_tool -%}{{'<｜Assistant｜><think>\n'}}{% if not enable_thinking %}{{- '</think>' -}}{% endif %}
{%- endif %}
""";
}
