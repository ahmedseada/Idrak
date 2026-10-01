// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Generation;
using Idrak.LanguageModels;

// Chat templates: the Jinja interpreter and model templates, checked against outputs of Python's jinja2 rendered the way
// Hugging Face's apply_chat_template renders them (trim_blocks, lstrip_blocks, tojson as json.dumps).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] ChatTemplates =
    [
        ("chat templates: Jinja expressions, filters, tests, loops, macros and whitespace control match jinja2", JinjaMatchesPython),
        ("chat templates: a Qwen3 template read from a model folder renders tools, tool calls and reasoning as transformers does", ModelChatTemplate),
        ("chat templates: each family's tool-call format is read off its template (tags, bare JSON, lists) and parsed from streamed output", ToolCallFormats),
    ];

    // Llama 3.1's tool-call layout (its published template, cut to the parts that render messages and calls): the
    // answer is the call's JSON, with "parameters".
    private const string LlamaStyleTemplate = """
        {{- bos_token }}
        {%- if messages[0]['role'] == 'system' %}{%- set system_message = messages[0]['content']|trim %}{%- set messages = messages[1:] %}{%- else %}{%- set system_message = "" %}{%- endif %}
        {{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
        {%- if tools is not none %}{{- "Environment: ipython\n" }}{%- endif %}
        {{- system_message }}
        {{- "<|eot_id|>" }}
        {%- for message in messages %}
            {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
                {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
            {%- elif 'tool_calls' in message %}
                {%- set tool_call = message.tool_calls[0].function %}
                {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
                {{- '{"name": "' + tool_call.name + '", ' }}
                {{- '"parameters": ' }}
                {{- tool_call.arguments | tojson }}
                {{- "}" }}
                {{- "<|eot_id|>" }}
            {%- else %}
                {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
                {{- message.content }}
                {{- "<|eot_id|>" }}
            {%- endif %}
        {%- endfor %}
        {%- if add_generation_prompt %}{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{%- endif %}
        """;

    // Mistral's (v3 tokenizer) layout: [TOOL_CALLS] and a JSON list of calls, the turn ended by the end-of-sequence token.
    private const string MistralStyleTemplate = """
        {{- bos_token }}
        {%- for message in messages %}
            {%- if message.role == 'user' %}
                {%- if tools is not none and loop.last %}{{- '[AVAILABLE_TOOLS] ' + tools | tojson + '[/AVAILABLE_TOOLS]' }}{%- endif %}
                {{- '[INST] ' + message.content + '[/INST]' }}
            {%- elif message.tool_calls is defined and message.tool_calls is not none %}
                {{- '[TOOL_CALLS] [' }}
                {%- for tool_call in message.tool_calls %}
                    {{- '{"name": "' + tool_call.function.name + '", "arguments": ' + tool_call.function.arguments | tojson + ', "id": "' + tool_call.id + '"}' }}
                    {%- if not loop.last %}{{- ', ' }}{%- endif %}
                {%- endfor %}
                {{- ']' + eos_token }}
            {%- elif message.role == 'assistant' %}
                {{- ' ' + message.content + eos_token }}
            {%- elif message.role == 'tool' %}
                {{- '[TOOL_RESULTS] {"content": ' + message.content | tojson + ', "call_id": "' + message.tool_call_id + '"}[/TOOL_RESULTS]' }}
            {%- endif %}
        {%- endfor %}
        """;

    private static void ToolCallFormats(Device device)
    {
        _ = device;
        var families = new (string Name, JinjaChatTemplate Template, ToolCallFormat Expected, string Output)[]
        {
            ("qwen", new JinjaChatTemplate(Qwen3Template, ["<|im_end|>"]), new ToolCallFormat("<tool_call>", "</tool_call>"),
                "Let me read it.\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"src/app.ts\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"search\", \"arguments\": \"{\\\"pattern\\\": \\\"Component\\\"}\"}\n</tool_call>"),
            ("llama", new JinjaChatTemplate(LlamaStyleTemplate, ["<|eot_id|>"], "<|begin_of_text|>"), new ToolCallFormat("", "", false, "parameters"),
                "{\"name\": \"read_file\", \"parameters\": {\"path\": \"src/app.ts\"}}"),
            ("mistral", new JinjaChatTemplate(MistralStyleTemplate, ["</s>"], "<s>", "</s>"), new ToolCallFormat("[TOOL_CALLS]", "", true),
                "[TOOL_CALLS] [{\"name\": \"read_file\", \"arguments\": {\"path\": \"src/app.ts\"}, \"id\": \"abc123def\"}, {\"name\": \"search\", \"arguments\": {\"pattern\": \"Component\"}}]"),
        };
        string[] toolNames = ["read_file", "search"];
        foreach (var (name, template, expected, output) in families)
        {
            var format = template.ToolCalls;
            Check(format == expected, $"{name}: detected {format}, expected {expected}");
            foreach (int size in new[] { 1, 3, 1000 })
            {
                var parser = new ChatOutputParser(template, toolNames: toolNames);
                string content = "";
                var calls = new List<ToolCall>();
                for (int i = 0; i < output.Length; i += size)
                {
                    var d = parser.Feed(output.Substring(i, Math.Min(size, output.Length - i)));
                    content += d.Content;
                    calls.AddRange(d.ToolCalls);
                }

                var f = parser.Finish();
                content += f.Content;
                calls.AddRange(f.ToolCalls);
                Check(calls.Count >= 1 && calls[0].Name == "read_file" && calls[0].Arguments["path"]!.GetValue<string>() == "src/app.ts",
                    $"{name}, chunks of {size}: first call");
                Check(name == "llama" || (calls.Count == 2 && calls[1].Name == "search" && calls[1].Arguments["pattern"]!.GetValue<string>() == "Component"),
                    $"{name}, chunks of {size}: second call");
                Check(content.Trim() == (name == "qwen" ? "Let me read it." : ""), $"{name}, chunks of {size}: content '{content}'");
            }
        }

        // Bare JSON (Llama): an answer that is JSON but no call to a known tool, or plain text, stays text.
        var llama = families[1].Template;
        foreach (string answer in new[] { "{\"name\": \"unknown_tool\", \"parameters\": {}}", "{\"result\": 42}", "The answer is {not JSON}." })
        {
            var parser = new ChatOutputParser(llama, toolNames: toolNames);
            var a = parser.Feed(answer);
            var b = parser.Finish();
            Check(a.ToolCalls.Count + b.ToolCalls.Count == 0 && a.Content + b.Content == answer, $"llama: '{answer}' stays text");
        }

        // A template that renders no tool calls falls back to the tags.
        var plain = new JinjaChatTemplate("{%- for m in messages %}{{ m.role }}: {{ m.content }}\n{%- endfor %}", ["\n"]);
        Check(plain.ToolCalls == ToolCallFormat.Tagged, $"no tool calls in the template: {plain.ToolCalls}");
    }

    // Qwen3's chat template, as published in its tokenizer_config.json.
    private const string Qwen3Template = """
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
""";

    private static readonly (string Template, string Expected)[] JinjaCases =
    [
        ("{{ s.strip() }}|{{ s|trim }}|{{ s.lstrip() }}|{{ s.rstrip() }}|{{ s.split(',') }}|{{ s.split() }}|{{ s.upper() }}|{{ s|lower }}",
         "Hello, World|Hello, World|Hello, World  |  Hello, World|['  Hello', ' World  ']|['Hello,', 'World']|  HELLO, WORLD  |  hello, world  "),
        ("{{ n + 1 }} {{ n / 2 }} {{ n // 2 }} {{ n % 3 }} {{ -n % 3 }} {{ n ** 2 }} {{ f * 2 }} {{ f }} {{ 1.0 }} {{ 1e-5 }} {{ 3 / 1 }} {{ 10 // 4.0 }}",
         "8 3.5 3 1 2 49 5.0 2.5 1.0 1e-05 3.0 2.0"),
        ("{{ xs }} {{ xs|sort }} {{ xs|reverse|list }} {{ xs|length }} {{ xs|first }} {{ xs|last }} {{ xs|join('-') }} {{ xs[1:] }} {{ xs[::-1] }} {{ xs[-1] }} {{ xs|max }} {{ xs|min }} {{ xs|sum }}",
         "[3, 1, 2] [1, 2, 3] [2, 1, 3] 3 3 2 3-1-2 [1, 2] [2, 1, 3] 2 3 1 6"),
        ("{{ d|tojson }} {{ d|tojson(indent=2) }} {{ d.a }} {{ d['b'] }} {{ d.missing }}|{{ d.get('b') }} {{ d.get('z', 5) }} {{ d|dictsort }} {{ d.items()|list }} {{ d.keys()|list }}",
         "{\"b\": 1, \"a\": [1, 2, {\"c\": null}]} {\n  \"b\": 1,\n  \"a\": [\n    1,\n    2,\n    {\n      \"c\": null\n    }\n  ]\n} [1, 2, {'c': None}] 1 |1 5 [('a', [1, 2, {'c': None}]), ('b', 1)] [('b', 1), ('a', [1, 2, {'c': None}])] ['b', 'a']"),
        ("{{ none }} {{ t }} {{ none is none }} {{ t is true }} {{ n is odd }} {{ n is divisibleby 7 }} {{ s is string }} {{ d is mapping }} {{ xs is iterable }} {{ q is defined }} {{ q is undefined }} {{ 'a' in 'cat' }} {{ 3 not in xs }}",
         "None  True False True True True True True False True True False"),
        ("{% for u in users if u.age > 0 %}{{ loop.index }}/{{ loop.length }}:{{ u.name }}{% if not loop.last %}, {% endif %}{% else %}none{% endfor %}\n{% for k, v in d.items() %}{{ k }}={{ v }};{% endfor %}\n{% for x in [] %}x{% else %}empty{% endfor %}",
         "1/2:b, 2/2:ab=1;a=[1, 2, {'c': None}];empty"),
        ("{{ users|map(attribute='name')|join(',') }} {{ users|selectattr('age', 'gt', 1)|list|length }} {{ users|rejectattr('x')|map(attribute='name')|list }} {{ users|sort(attribute='name')|map(attribute='age')|list }} {{ users|selectattr('x')|map(attribute='x.y')|first }}",
         "b,a 1 ['b'] [1, 3] 2"),
        ("{% set ns = namespace(total=0) %}{% for x in xs %}{% set ns.total = ns.total + x %}{% endfor %}{{ ns.total }}\n{% set y = 1 %}{% for x in xs %}{% set y = y + x %}{% endfor %}{{ y }}\n{% macro greet(name, punct='!') %}Hi {{ name }}{{ punct }}{% endmacro %}{{ greet('a') }} {{ greet('b', '?') }} {{ greet(name='c') }}",
         "6\n1\nHi a! Hi b? Hi c!"),
        ("{% set block %}inside {{ n }}{% endset %}[{{ block }}]\n{% if n > 5 %}big{% elif n > 2 %}mid{% else %}small{% endif %}\n{{ 'yes' if t else 'no' }} {{ 'x' if none }}|{{ none or 'fallback' }} {{ n and 'and' }}",
         "[inside 7]\nbigno |fallback and"),
        ("{{ \"a'b\" }} {{ ['x', \"y'z\"] }} {{ {'k': [1, 'v']} }} {{ (1, 2) }} {{ \"%s is %d\" % ('n', 3) }} {{ '{} + {}'.format(1, 2) }}",
         "a'b ['x', \"y'z\"] {'k': [1, 'v']} (1, 2) n is 3 1 + 2"),
        ("{{ 'abc'[0] }}{{ 'abc'[-1] }}{{ 'abcdef'[1:4] }} {{ 'a b c'.split(' ', 1) }} {{ 'x,y'.replace(',', ';') }} {{ 'Hello World'.startswith('Hell') }} {{ 'Hello'.endswith(('lo', 'x')) }} {{ 'ab'.count('a') }} {{ 'hello world'|title }} {{ 'HELLO'|capitalize }}",
         "acbcd ['a', 'b c'] x;y True True 1 Hello World Hello"),
        ("{% for x in xs %}{% if x == 1 %}{% continue %}{% endif %}{% if x == 2 %}{% break %}{% endif %}{{ x }}{% endfor %}|{{ range(3)|list }} {{ range(1, 7, 2)|list }} {{ xs|unique|list }} {{ [1, 1, 2]|unique|list }} {{ 3.7|int }} {{ '4'|int + 1 }} {{ 2.5|round }} {{ 2.567|round(2) }} {{ -3|abs }}",
         "3|[0, 1, 2] [1, 3, 5] [3, 1, 2] [1, 2] 3 5 2.0 2.57 3"),
        ("{% if xs|selectattr('missing', 'defined')|list %}a{% else %}b{% endif %} {{ xs|select('odd')|list }} {{ xs|reject('odd')|list }} {{ [none, 1]|select|list }} {{ 'a' ~ 1 ~ none }} {{ [1] + [2] }} {{ 'ab' * 2 }}\n",
         "b [3, 1] [2] [1] a1None [1, 2] abab"),
    ];

    private static void JinjaMatchesPython(Device device)
    {
        _ = device;
        var variables = new Dictionary<string, object?>
        {
            ["n"] = 7L, ["f"] = 2.5, ["xs"] = new List<object?> { 3L, 1L, 2L }, ["s"] = "  Hello, World  ",
            ["d"] = new Dictionary<string, object?> { ["b"] = 1L, ["a"] = new List<object?> { 1L, 2L, new Dictionary<string, object?> { ["c"] = null } } },
            ["users"] = new List<object?>
            {
                new Dictionary<string, object?> { ["name"] = "b", ["age"] = 3L },
                new Dictionary<string, object?> { ["name"] = "a", ["age"] = 1L, ["x"] = new Dictionary<string, object?> { ["y"] = 2L } },
            },
        };
        foreach (var (template, expected) in JinjaCases)
        {
            string actual = JinjaTemplate.Parse(template).Render(variables);
            Check(actual == expected, $"{template}\n  expected: {expected}\n  actual:   {actual}");
        }

        try
        {
            JinjaTemplate.Parse("{{ raise_exception('bad input') }}").Render(new Dictionary<string, object?>());
            Check(false, "raise_exception throws");
        }
        catch (InvalidOperationException ex)
        {
            Check(ex.Message.Contains("bad input"), ex.Message);
        }
    }

    private static void ModelChatTemplate(Device device)
    {
        _ = device;
        string folder = TempFolder();
        try
        {
            File.WriteAllText(Path.Combine(folder, "tokenizer_config.json"), new JsonObject
            {
                ["chat_template"] = Qwen3Template, ["eos_token"] = "<|im_end|>", ["bos_token"] = null,
            }.ToJsonString());
            File.WriteAllText(Path.Combine(folder, "generation_config.json"), """{"eos_token_id": [2, 1]}""");
            var vocab = new JsonObject { ["a"] = 0 };
            var tokenizer = BpeTokenizer.FromJson(new JsonObject
            {
                ["added_tokens"] = new JsonArray(
                    new JsonObject { ["id"] = 1, ["content"] = "<|endoftext|>", ["special"] = true },
                    new JsonObject { ["id"] = 2, ["content"] = "<|im_end|>", ["special"] = true }),
                ["model"] = new JsonObject { ["type"] = "BPE", ["vocab"] = vocab, ["merges"] = new JsonArray() },
            });
            var template = JinjaChatTemplate.Load(folder, tokenizer) ?? throw new InvalidOperationException("no template");
            Check(template.StopSequences.SequenceEqual(["<|im_end|>", "<|endoftext|>"]), $"stop sequences: {string.Join(", ", template.StopSequences)}");

            var tools = new List<ToolDefinition>
            {
                new("read_file", "Read a file — \"quoted\"", JsonNode.Parse("""{"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}""")),
            };
            var messages = new List<ChatMessage>
            {
                new("system", "You are a coding agent."),
                new("user", "Fix the bug in main.py"),
                new("assistant", "", "Look first.", [new ToolCall("read_file", new JsonObject { ["path"] = "main.py", ["lines"] = 10 })]),
                new("tool", "print(1/0)", ToolName: "read_file"),
                new("assistant", "The divisor is zero.", "Found it."),
            };
            string prompt = template.Render(messages[..4], tools, think: null);
            Check(prompt == "<|im_start|>system\nYou are a coding agent.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"read_file\", \"description\": \"Read a file — \\\"quoted\\\"\", \"parameters\": {\"type\": \"object\", \"properties\": {\"path\": {\"type\": \"string\"}}, \"required\": [\"path\"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\nFix the bug in main.py<|im_end|>\n<|im_start|>assistant\n<think>\nLook first.\n</think>\n\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"main.py\", \"lines\": 10}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\nprint(1/0)\n</tool_response><|im_end|>\n<|im_start|>assistant\n", $"prompt with tools and a tool call:\n{prompt}");
            string training = template.Render(messages, tools, think: null, addGenerationPrompt: false);
            Check(training == "<|im_start|>system\nYou are a coding agent.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"read_file\", \"description\": \"Read a file — \\\"quoted\\\"\", \"parameters\": {\"type\": \"object\", \"properties\": {\"path\": {\"type\": \"string\"}}, \"required\": [\"path\"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\nFix the bug in main.py<|im_end|>\n<|im_start|>assistant\n<think>\nLook first.\n</think>\n\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"main.py\", \"lines\": 10}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\nprint(1/0)\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\nFound it.\n</think>\n\nThe divisor is zero.<|im_end|>\n", $"whole conversation:\n{training}");
            string noThinking = template.Render([new ChatMessage("user", "hi")], [], think: false);
            Check(noThinking == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", $"enable_thinking=false:\n{noThinking}");

            // The model's answer in this format parses back into reasoning, text and a tool call.
            var parser = new ChatOutputParser(template);
            var first = parser.Feed("<think>\nCheck.\n</think>\n\nReading.\n<tool_call>\n{\"name\": \"read_file\", \"argu");
            var rest = parser.Feed("ments\": {\"path\": \"a.py\"}}\n</tool_call>");
            var end = parser.Finish();
            string content = first.Content + rest.Content + end.Content, thinking = first.Thinking + rest.Thinking + end.Thinking;
            var calls = first.ToolCalls.Concat(rest.ToolCalls).Concat(end.ToolCalls).ToList();
            Check(thinking.Trim() == "Check." && content.Trim() == "Reading." && calls.Count == 1 && calls[0].Name == "read_file"
                  && (string?)calls[0].Arguments["path"] == "a.py", $"parsed output: '{thinking}' '{content}' {calls.Count}");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }
}
