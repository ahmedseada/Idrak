// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Text;
using Idrak.Text.Generation;

// From Craft's agent bench on the owner's GPU (Qwen2.5-7B-Instruct, Qwen2.5-Coder-7B-Instruct): a call that is not valid
// JSON lost beside calls that were run, code shown in the answer with no file changed, and a reply in another script than
// the request's.
internal abstract partial class DesktopTests
{
    // Qwen2.5-7B-Instruct's two calls, as written: a single-quoted value, and ';' where ',' belongs.
    private const string SingleQuotedCall = "```json\n<tool_call>\n{\"name\": \"edit_file\", \"arguments\": {\"path\": \"Program.cs\", \"old_text\": \"Console.WriteLine(\\\"Hello World!\\\");\", "
                                            + "\"new_text\": 'using Calculations;\\n\\nConsole.WriteLine(Calculations.Calculator.Multiply(6, 7));'}}\n</tool_call>\n";

    private const string SemicolonCall = "<tool_call>\n{\"name\": \"edit_file\", \"arguments\": {\"path\": \"Program.cs\", \"old_text\": \"Console.WriteLine(\\\"Hello World\\\"\"; "
                                         + "\"new_text\": \"Console.WriteLine(6 * 7);\"}}\n</tool_call>";

    private static void AgentToldMalformedCalls(Device device)
    {
        _ = device;
        InAgentWorkspace(root =>
        {
            File.WriteAllText(Path.Combine(root, "Program.cs"), "Console.WriteLine(\"Hello, World!\");\n");
            var tools = new CodingAgent(FakeChatModel.Script(FakeChatModel.Answer("Done."))).RunAsync("x", new CodingTools(root)).GetAwaiter().GetResult().Tools;
            var context = CodingAgent.CallContext(null, tools);

            // Where reading stopped, and the rule: the quote, then the semicolon.
            var quoted = ToolCallText.MalformedCall(SingleQuotedCall, context);
            string quotedJson = SingleQuotedCall[SingleQuotedCall.IndexOf('{')..];
            Check(quoted is { Name: "edit_file" } && quotedJson[quoted.Position] == '\'' && quoted.Hint.Contains("double quotes", StringComparison.Ordinal)
                  && quoted.Describe().StartsWith("your edit_file call is not valid JSON at character " + (quoted.Position + 1), StringComparison.Ordinal),
                $"a single-quoted value is found at its quote: {quoted?.Describe()}");
            var semicolon = ToolCallText.MalformedCall(SemicolonCall, context);
            string semicolonJson = SemicolonCall[SemicolonCall.IndexOf('{')..];
            Check(semicolon is { Name: "edit_file" } && semicolonJson[semicolon.Position] == ';' && semicolon.Hint.Contains("separated by commas, not ';'", StringComparison.Ordinal),
                $"';' between arguments is found where it stands: {semicolon?.Describe()}");
            Check(ToolCallText.MalformedCall("```json\n{\"name\": \"edit_file\", \"arguments\": {\"path\": \"a\",}}\n```", context) is null
                  && ToolCallText.MalformedCall("<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"a\", \"content\": \"x\ny\"}}\n</tool_call>", context) is null
                  && ToolCallText.MalformedCall("Use {\"name\": \"edit_file\"} as you like; the code is ```csharp\nvar x = new { a = 'b' };\n```", context) is null
                  && ToolCallText.MalformedCall("{\"name\": \"not_a_tool\", \"arguments\": {'a': 1}}", context) is null,
                "trailing commas, raw line breaks in strings, valid JSON, another language's fence and a tool not offered are not reported");
            Check(ToolCallText.MalformedCall("{\"name\": \"read_file\", \"arguments\": {\"path\": \"a\"", context) is { Name: "read_file" } open && open.Hint.Contains("closed", StringComparison.Ordinal),
                "an unclosed bare call is reported as not closed");

            // Beside a call that was read: the correction comes after the results, the model writes it again, the file changes.
            var fake = FakeChatModel.Script(new ChatMessage("assistant", "Now Program.cs:\n" + SingleQuotedCall, ToolCalls: [new ToolCall("read_file", new JsonObject { ["path"] = "Program.cs" })]),
                FakeChatModel.ToolCall("edit_file", new JsonObject { ["path"] = "Program.cs", ["old_text"] = "Hello, World!", ["new_text"] = "42" }), FakeChatModel.Answer("Program.cs prints 42."));
            var run = new CodingAgent(fake).RunAsync("Change Program.cs", new CodingTools(root)).GetAwaiter().GetResult();
            Check(run is { End: CodingAgentEnd.Answered, Nudges: 1, ToolCalls: 2 } && fake.Requests[1].Messages[^2].Role == "tool"
                  && fake.Requests[1].Messages[^1] is { Role: "user" } told && told.Content.StartsWith("One tool call in your last reply could not be read", StringComparison.Ordinal)
                  && told.Content.Contains("your edit_file call is not valid JSON at character", StringComparison.Ordinal)
                  && File.ReadAllText(Path.Combine(root, "Program.cs")).Contains("42", StringComparison.Ordinal),
                $"a malformed call beside a read one is told after the results and then made ({run.End}, {run.Nudges}, {string.Join(" | ", fake.Requests.Count > 1 ? fake.Requests[1].Messages.Select(m => m.Content) : [])})");

            // Alone: the call-format correction names the error too.
            string? correction = CodingAgent.Correction(FakeChatModel.Answer(SemicolonCall), context);
            Check(correction is not null && correction.Contains("Your edit_file call is not valid JSON at character", StringComparison.Ordinal) && correction.Contains("not ';'", StringComparison.Ordinal),
                $"the correction of a reply with no readable call names the parse error: {correction}");
        });
    }

    private static void AgentRemindsUnchangedCode(Device device)
    {
        _ = device;
        InAgentWorkspace(root =>
        {
            // Qwen2.5-Coder-7B's run: the code in fences, one call that changes nothing, "done". Reminded once; it then writes.
            const string shown = "Here is Calculator.cs:\n```csharp\nstatic class Calculator { }\n```\nThe project builds.";
            var fake = FakeChatModel.Script(FakeChatModel.ToolCall("list_files", new JsonObject()), FakeChatModel.Answer(shown),
                FakeChatModel.ToolCall("write_file", new JsonObject { ["path"] = "Calculator.cs", ["content"] = "static class Calculator { }\n" }), FakeChatModel.Answer(shown));
            var run = new CodingAgent(fake).RunAsync("Add a Calculator class", new CodingTools(root)).GetAwaiter().GetResult();
            Check(run is { End: CodingAgentEnd.Answered, NoChangeNudges: 1, NoToolNudges: 0, Rounds: 4 }
                  && fake.Requests[2].Messages[^1] is { Role: "user" } note && note.Content == CodingAgent.NoChangeReminder(["edit_file", "write_file"])
                  && note.Content.Contains("write_file", StringComparison.Ordinal) && File.Exists(Path.Combine(root, "Calculator.cs"))
                  && run.ToJson()["no_change_nudges"]?.GetValue<int>() == 1,
                $"code shown with no file changed is reminded once; after a write the same answer ends the run ({run.End}, {run.NoChangeNudges}, {run.Rounds})");

            // No tool at all: the no-change reminder is sent instead of the no-tool one, not both.
            var none = FakeChatModel.Script(FakeChatModel.Answer(shown), FakeChatModel.Answer(shown));
            Check(new CodingAgent(none).RunAsync("Add a Calculator class", new CodingTools(root)).GetAwaiter().GetResult() is { End: CodingAgentEnd.Answered, NoChangeNudges: 1, NoToolNudges: 0, Rounds: 2 },
                "with no tool at all one reminder is sent, the no-change one");

            // 0 sends none; an answer without code is not reminded.
            var off = FakeChatModel.Script(FakeChatModel.ToolCall("list_files", new JsonObject()), FakeChatModel.Answer(shown));
            Check(new CodingAgent(off, new AgentOptions { NoChangeNudges = 0 }).RunAsync("x", new CodingTools(root)).GetAwaiter().GetResult() is { End: CodingAgentEnd.Answered, NoChangeNudges: 0, Rounds: 2 },
                "NoChangeNudges 0: none");
            var words = FakeChatModel.Script(FakeChatModel.ToolCall("list_files", new JsonObject()), FakeChatModel.Answer("Nothing needs changing."));
            Check(new CodingAgent(words).RunAsync("x", new CodingTools(root)).GetAwaiter().GetResult() is { End: CodingAgentEnd.Answered, NoChangeNudges: 0, Rounds: 2 },
                "an answer without code is not reminded");
        });
    }

    private static void AgentRemindsReplyLanguage(Device device)
    {
        _ = device;
        InAgentWorkspace(root =>
        {
            // Qwen2.5-7B-Instruct's run 2: Chinese prose for an English request, with calls. Reminded after the results.
            const string chinese = "对其真实请求进行详细的分析和理解：首先，我将列出所有文件以确认当前的文件结构，然后创建并编辑相应的文件。\n```json\n{\"name\": \"list_files\", \"arguments\": {}}\n```";
            var fake = FakeChatModel.Script(new ChatMessage("assistant", chinese, ToolCalls: [new ToolCall("list_files", new JsonObject())]), FakeChatModel.Answer("Listed the files; nothing to change."));
            var run = new CodingAgent(fake, new AgentOptions { NoToolNudges = 0 }).RunAsync("List the project's files and tell me what is there.", new CodingTools(root)).GetAwaiter().GetResult();
            Check(run is { End: CodingAgentEnd.Answered, LanguageNudges: 1, Rounds: 2 } && fake.Requests[1].Messages[^2].Role == "tool"
                  && fake.Requests[1].Messages[^1] is { Role: "user" } note && note.Content == CodingAgent.LanguageReminder("Han")
                  && run.ToJson()["language_nudges"]?.GetValue<int>() == 1,
                $"Han prose for a Latin request is reminded once, in English, after the results ({run.End}, {run.LanguageNudges})");

            // A final answer: reminded instead of ending, once a run.
            const string answer = "已经创建了文件，并修改了程序，使其包含调用的代码。项目构建成功，所有更改都已完成。";
            var twice = FakeChatModel.Script(FakeChatModel.Answer(answer), FakeChatModel.Answer(answer));
            Check(new CodingAgent(twice, new AgentOptions { NoToolNudges = 0 }).RunAsync("Explain the project.", new CodingTools(root)).GetAwaiter().GetResult() is { End: CodingAgentEnd.Answered, LanguageNudges: 1, Rounds: 2 },
                "a final answer in another script is reminded once, then ends the run");

            // The request's own script, a Japanese request (kana and Han) with Han prose, English prose with Chinese only in
            // code, too few letters, and 0: none.
            (string Request, string Reply, int Nudges)[] quiet =
            [
                ("请解释这个项目的结构和用途。", answer, 1),
                ("このプロジェクトの構造を説明してください。", answer, 1),
                ("Explain the project.", "The project prints a greeting.\n```csharp\n// 打印问候语并且返回结果，然后结束程序的运行过程\nConsole.WriteLine(\"你好\");\n```\nIt has one file, `程序的运行过程和结果都在这里说明`.", 1),
                ("Explain the project.", "完成。", 1),
                ("Explain the project.", answer, 0),
            ];
            foreach (var (request, reply, nudges) in quiet)
            {
                var model = FakeChatModel.Script(FakeChatModel.Answer(reply));
                var quietRun = new CodingAgent(model, new AgentOptions { NoToolNudges = 0, NoChangeNudges = 0, LanguageNudges = nudges }).RunAsync(request, new CodingTools(root)).GetAwaiter().GetResult();
                Check(quietRun is { End: CodingAgentEnd.Answered, LanguageNudges: 0, Rounds: 1 }, $"no language reminder for '{request}' / '{reply[..Math.Min(20, reply.Length)]}' at {nudges} ({quietRun.End}, {quietRun.Note})");
            }
        });
    }
}
