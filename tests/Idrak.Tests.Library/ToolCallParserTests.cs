// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Generation;
using Idrak.Text;

// Tool calls written as text: read when they name an offered tool, parsed from output streamed in pieces split anywhere.
internal abstract partial class Tests
{
    // The coding agent's tools, by name (what the reply below may call).
    internal static readonly ToolDefinition[] CodingToolNames =
    [
        new("list_files"), new("read_file", "Read.", JsonNode.Parse("""{"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}""")),
        new("search"), new("edit_file"),
        new("write_file", "Write.", JsonNode.Parse("""{"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}""")),
        new("run_command"),
    ];

    // A 7B coder model's reply to "Create C# class For House with float 3 props and + float price" (the owner's run,
    // 2026-11-01, the shape kept): it describes the calls in markdown json fences instead of its template's
    // <tool_call> tags, the first with one '}' too many, and ends claiming the work is done.
    internal const string HouseReplyWithFencedCalls = """
        To create a C# class for a `House` with three `float` properties and a `float` price, you can use the `write_file` tool to create a new file named `House.cs`. Here's how you can do it:

        ```json
        {
          "name": "write_file",
          "arguments": {
            "path": "House.cs",
            "content": "public class House\n{\n    public float Width { get; set; }\n    public float Length { get; set; }\n    public float Height { get; set; }\n    public float Price { get; set; }\n}\n"
          }
        }}
        ```

        After creating the file, you can use the `read_file` tool to verify that the class was written correctly:

        ```json
        {
          "name": "read_file",
          "arguments": {
            "path": "House.cs"
          }
        }
        ```

        Finally, you can build the project with the `run_command` tool to make sure everything compiles:

        ```json
        {
          "name": "run_command",
          "arguments": {
            "command": "dotnet build"
          }
        }
        ```

        You created a new House.cs file with the `House` class, which has three float properties (`Width`, `Length`, `Height`) and a float `Price` property. The project builds without errors.
        """;

    internal static void ToolCallsWrittenAsText(Device device)
    {
        if (device != Device.Cpu)
        {
            return;
        }

        var template = new ChatMLTemplate();
        var context = new ToolCallContext(template, CodingToolNames, CodingToolNames.Select(t => t.Name).ToHashSet(StringComparer.Ordinal));

        // The owner's reply: the template's parser finds no call (no <tool_call> tags); read as text, the three fences
        // give the three calls in order, the extra brace after the first ignored.
        var (content, _, parsed) = ParseStreamed(template, HouseReplyWithFencedCalls, 13, CodingToolNames);
        Check(parsed.Count == 0 && content.Contains("You created a new House.cs", StringComparison.Ordinal), $"the template's parser reads no call from fences ({parsed.Count})");
        var calls = ToolCallText.CallsInText(HouseReplyWithFencedCalls, context);
        Check(calls is { Count: 3 } && calls[0].Name == "write_file" && calls[1].Name == "read_file" && calls[2].Name == "run_command",
            $"the owner's reply gives write_file, read_file, run_command: {(calls is null ? "none" : Show(calls))}");
        Check((string?)calls![0].Arguments["path"] == "House.cs" && ((string?)calls[0].Arguments["content"])!.Contains("public float Price { get; set; }\n}", StringComparison.Ordinal)
              && (string?)calls[1].Arguments["path"] == "House.cs" && (string?)calls[2].Arguments["command"] == "dotnet build",
            $"each call keeps its arguments (the file's text with its newlines): {Show(calls)}");
        Check(ToolCallText.LooksLikeCall(HouseReplyWithFencedCalls, context), "the reply looks like calls");

        var cases = new (string Name, string Text, ToolCall[] Calls)[]
        {
            ("an example naming no offered tool stays text", "For example:\n```json\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n```\nThat is how a call looks.", []),
            ("a list in one fence", "```json\n[{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.txt\"}}, {\"name\": \"list_files\", \"arguments\": {}}]\n```",
                [ParsedCall("read_file", """{"path": "a.txt"}"""), ParsedCall("list_files", "{}")]),
            ("a list naming one tool that is not offered stays text", "```\n[{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.txt\"}}, {\"name\": \"delete_all\", \"arguments\": {}}]\n```", []),
            ("a bare fence, a trailing comma and arguments as a string", "```\n{\"name\": \"read_file\", \"arguments\": \"{\\\"path\\\": \\\"b.txt\\\"}\",}\n```",
                [ParsedCall("read_file", """{"path": "b.txt"}""")]),
            ("a call in the prose", "I will write it now. {\"name\": \"write_file\", \"arguments\": {\"path\": \"a.txt\", \"content\": \"x\"}} Then I am done.",
                [ParsedCall("write_file", """{"path": "a.txt", "content": "x"}""")]),
            ("two calls in the prose, \"parameters\" as the key", "First {\"name\": \"read_file\", \"parameters\": {\"path\": \"a\"}} then {\"name\": \"read_file\", \"parameters\": {\"path\": \"b\"}}.",
                [ParsedCall("read_file", """{"path": "a"}"""), ParsedCall("read_file", """{"path": "b"}""")]),
            ("a call in the template's tags with one '}' too many", "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.txt\"}}}\n</tool_call>",
                [ParsedCall("read_file", """{"path": "a.txt"}""")]),
            ("code in a C# fence stays text", "Here is the class:\n```csharp\n[Serializable]\npublic class House { public float Price { get; set; } }\n```\nDone.", []),
            ("a call that does not close stays text", "```json\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"a.txt\", \"content\": \"x\"}\n```", []),
            ("plain prose", "The class is done; it has three float properties and a price.", []),
        };
        foreach (var (name, text, expected) in cases)
        {
            var got = ToolCallText.CallsInText(text, context) ?? [];
            Check(SameCalls(got, expected), $"{name}: expected {Show(expected)}, got {Show(got)}");
        }

        var tagged = ParseStreamed(template, cases[6].Text, 5, CodingToolNames);
        Check(tagged.Calls.Count == 0, "the template's parser leaves a tagged call with one '}' too many as text (the text read recovers it)");

        // Call-like text with no call read: what the agent corrects.
        foreach (var (name, text, like) in new (string, string, bool)[]
        {
            ("a json fence of a call that does not close", cases[8].Text, true),
            ("a json fence naming no offered tool", cases[0].Text, true),
            ("\"name\" with an offered tool's name in prose", "I called \"name\": \"write_file\" with the path.", true),
            ("\"name\" with another name", "Use \"name\": \"write_files\" for that.", false),
            ("a C# fence", cases[7].Text, false),
            ("plain prose", cases[9].Text, false),
        })
        {
            Check(ToolCallText.LooksLikeCall(text, context) == like, $"looks like a call: {name} should be {like}");
        }

        // Rule 74: one sync pass over the reply's chars; a value is checked as UTF-8 in a pooled buffer before anything is
        // built, so text with no call, or an example naming no offered tool, allocates nothing once warm.
        foreach (var (name, text) in new[] { ("a C# fence", cases[7].Text), ("an example naming no offered tool", cases[0].Text), ("plain prose", cases[9].Text) })
        {
            for (int i = 0; i < 64; i++)
            {
                _ = ToolCallText.CallsInText(text, context);
                _ = ToolCallText.LooksLikeCall(text, context);
            }

            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 100; i++)
            {
                _ = ToolCallText.CallsInText(text, context);
                _ = ToolCallText.LooksLikeCall(text, context);
            }

            long bytes = GC.GetAllocatedBytesForCurrentThread() - before;
            Check(bytes == 0, $"{name}: reading calls as text allocated {bytes} bytes over 100 warm runs (expected none)");
        }
    }

    internal static ToolCall ParsedCall(string name, string arguments) => new(name, (JsonObject)JsonNode.Parse(arguments)!);

    internal static (string Content, string Thinking, List<ToolCall> Calls) ParseStreamed(IChatTemplate template, string output, int size, IReadOnlyList<ToolDefinition>? tools)
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

    internal static bool SameCalls(IReadOnlyList<ToolCall> a, IReadOnlyList<ToolCall> b) =>
        a.Count == b.Count && a.Zip(b).All(p => p.First.Name == p.Second.Name && JsonNode.DeepEquals(p.First.Arguments, p.Second.Arguments));

    internal static string Show(IEnumerable<ToolCall> calls) => "[" + string.Join(", ", calls.Select(c => c.Name + c.Arguments.ToJsonString())) + "]";
}
