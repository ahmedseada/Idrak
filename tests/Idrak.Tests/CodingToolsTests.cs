// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Generation;
using Idrak.Reinforcement.Plugins.Coding;
using Idrak.Text;
using Idrak.Text.Generation;

// Coding tools: the workspace sandbox, reading, searching, exact edits, writes and allowlisted commands.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] CodingToolsGroup =
    [
        ("coding tools: list, read, search, edit and write stay inside the workspace and explain mistakes", CodingToolsFiles),
        ("coding tools: read_file and search number and cut lines as the string-splitting reference (CRLF, lone CR, final newline, long lines); web text collapses white space as the regex did", d => { if (d == Device.Cpu) CodingToolsLinesMatchReference(); }),
        ("coding tools: run_command starts allowlisted programs without a shell and trims long output", CodingToolsCommands),
        ("coding tools: edit_file on the bench's Program.cs (byte order mark, CRLF): a guessed old_text's error quotes it and the file; read_file's line numbers, trailing white space and the final line break tolerated when the match is unique, said so; the encoding kept; the option off is exact", CodingEditTolerance),
        ("coding agent: an answer with a failed change left (no later success on its path) gets one reminder listing it; a change made good, a failed read or the option at 0 gets none", AgentRemindsFailedChanges),
        ("coding agent: transcripts written as OpenAI chat JSON read back for fine-tuning", AgentTranscriptJson),
        ("coding agent: a task is copied, run, verified by its commands and recorded", AgentRunsTask),
        ("coding agent: a call repeated right after itself is not run again, and the model is told to use its result", AgentSkipsRepeatedCalls),
        ("coding agent: a steering note given while it runs is read before its next request and stays in the conversation", AgentReadsSteeringNotes),
        ("coding agent: extra tools are offered after the coding tools and called; one with a coding tool's name is refused", AgentOffersExtraTools),
        ("coding agent: ContinueAsync keeps the earlier run's conversation (its task, replies and tool results) and adds the next message after it", AgentContinuesConversation),
        ("coding agent: calls written in fences are read and run (the owner's House reply: the file is written, read back, the build asked for)", AgentReadsFencedCalls),
        ("coding agent: a malformed call is nudged, the scripted model answers properly and the file is written; strict calls nudge a fenced one", AgentNudgesMalformedCall),
        ("coding agent: the nudge limit ends the run with CallsNotRead; an answer in words or with a code fence ends as answered", AgentNudgeLimit),
        ("coding agent: each call's result is told as that call ends (before the next one waits), and a turn cut by the time limit still tells the calls that finished", AgentReportsEachResultAsItEnds),
        ("coding agent: three ```json fences (write_file with escaped newlines and braces, edit_file, run_command) are all read and run, in order", AgentRunsEveryFencedCall),
        ("coding tools: a refusal or error (old_text not found or not unique, a path outside the folder, a missing file, a refused command) is a failed result with its message; the model reads it as before", CodingToolErrorsFail),
        ("coding agent: edit_file refuses a file the task has not read or written (read it first), a new task reads again, the option off allows it; the prompt asks for the read", AgentReadsBeforeEditing),
    ];

    private static void AgentReadsBeforeEditing(Device device)
    {
        _ = device;
        InAgentWorkspace(root =>
        {
            string program = Path.Combine(root, "Program.cs");
            File.WriteAllText(program, "class Program { static void Main() { } }\n");
            var guess = new JsonObject { ["path"] = "Program.cs", ["old_text"] = "static void Main() { }", ["new_text"] = "static void Main() => System.Console.WriteLine(1);" };
            // A guessed edit first (refused, nothing changed), then the read and the same edit (done).
            var fake = FakeChatModel.Script(FakeChatModel.ToolCall("edit_file", guess), FakeChatModel.ToolCall("read_file", new JsonObject { ["path"] = "Program.cs" }),
                FakeChatModel.ToolCall("edit_file", (JsonObject)guess.DeepClone()), FakeChatModel.Answer("Main prints 1."));
            var tools = new CodingTools(root);
            var told = new List<ToolResult>();
            var run = new CodingAgent(fake) { OnToolResult = told.Add }.RunAsync("Make Main print 1", tools).GetAwaiter().GetResult();
            Check(told is [{ Succeeded: false } refused, { Succeeded: true }, { Succeeded: true }] && refused.Error!.StartsWith("read Program.cs first (read_file)", StringComparison.Ordinal)
                  && fake.Requests[1].Messages[^1].Content.StartsWith("Error: read Program.cs first", StringComparison.Ordinal),
                $"the unread edit is refused saying to read it first, then read and edited ({string.Join("; ", told.Select(r => r.Error ?? "ok"))})");
            Check(File.ReadAllText(program).Contains("WriteLine(1)", StringComparison.Ordinal) && run is { End: CodingAgentEnd.Answered, ToolErrors: 1 }, "the edit after the read is made");

            // A new task forgets the reads; a file the task wrote may be edited; the option off allows any edit.
            var again = FakeChatModel.Script(FakeChatModel.ToolCall("edit_file", new JsonObject { ["path"] = "Program.cs", ["old_text"] = "WriteLine(1)", ["new_text"] = "WriteLine(2)" }), FakeChatModel.Answer("x"));
            var second = new CodingAgent(again).RunAsync("Print 2", tools).GetAwaiter().GetResult();
            Check(second.ToolErrors == 1 && File.ReadAllText(program).Contains("WriteLine(1)", StringComparison.Ordinal), "the next task must read the file again");
            var written = FakeChatModel.Script(FakeChatModel.ToolCall("write_file", new JsonObject { ["path"] = "New.cs", ["content"] = "class A { }\n" }),
                FakeChatModel.ToolCall("edit_file", new JsonObject { ["path"] = "New.cs", ["old_text"] = "class A", ["new_text"] = "class B" }), FakeChatModel.Answer("x"));
            Check(new CodingAgent(written).RunAsync("Make B", tools).GetAwaiter().GetResult().ToolErrors == 0 && File.ReadAllText(Path.Combine(root, "New.cs")).Contains("class B", StringComparison.Ordinal),
                "a file the task wrote can be edited");
            var free = new CodingTools(root, new CodingToolOptions { ReadBeforeEdit = false });
            var direct = FakeChatModel.Script(FakeChatModel.ToolCall("edit_file", new JsonObject { ["path"] = "Program.cs", ["old_text"] = "WriteLine(1)", ["new_text"] = "WriteLine(3)" }), FakeChatModel.Answer("x"));
            Check(new CodingAgent(direct).RunAsync("Print 3", free).GetAwaiter().GetResult().ToolErrors == 0 && File.ReadAllText(program).Contains("WriteLine(3)", StringComparison.Ordinal),
                "ReadBeforeEdit off: an unread edit is made");
            Check(CodingAgent.DefaultSystemPrompt.Contains("Read a file with read_file before you edit it", StringComparison.Ordinal), "the default prompt asks for the read");
        });
    }

    private static void CodingToolErrorsFail(Device device)
    {
        _ = device;
        InAgentWorkspace(root =>
        {
            File.WriteAllText(Path.Combine(root, "Program.cs"), "class Program { int a; int a2; }\n");
            var tools = new CodingTools(root, null, new RecordingSandbox(new CommandResult(1, false, "build failed", TimeSpan.Zero, false)));
            var registry = tools.Registry();
            ToolResult Run(string name, JsonObject arguments) => registry.InvokeAsync(new ToolCall(name, arguments)).GetAwaiter().GetResult();
            _ = Run("read_file", new() { ["path"] = "Program.cs" });               // edits need the file read first
            var failures = new (string Name, JsonObject Arguments, string Says)[]
            {
                ("edit_file", new() { ["path"] = "Program.cs", ["old_text"] = "int b;", ["new_text"] = "int c;" }, "not found"),
                ("edit_file", new() { ["path"] = "Program.cs", ["old_text"] = "int a", ["new_text"] = "int c" }, "occurs"),
                ("read_file", new() { ["path"] = "../outside.cs" }, "outside the workspace"),
                ("read_file", new() { ["path"] = "Missing.cs" }, "does not exist"),
                ("run_command", new() { ["command"] = "dotnet build && dotnet test" }, "not run by a shell"),
            };
            foreach (var (name, arguments, says) in failures)
            {
                var result = Run(name, arguments);
                Check(!result.Succeeded && result.Error is { } error && error.Contains(says, StringComparison.Ordinal) && !error.StartsWith("Error:", StringComparison.Ordinal)
                      && result.Content.Contains(says, StringComparison.Ordinal),
                    $"{name} {arguments.ToJsonString()}: a failed result saying '{says}' ({(result.Succeeded ? "succeeded" : "failed")}: {result.Error ?? result.Content})");
            }

            Check(Run("read_file", new() { ["path"] = "Program.cs" }) is { Succeeded: true } read && read.Content.Contains("class Program", StringComparison.Ordinal)
                  && Run("run_command", new() { ["command"] = "dotnet build" }) is { Succeeded: true } built && built.Content.Contains("build failed", StringComparison.Ordinal),
                "a read and a command that ran (exit code 1 included) succeed");

            // In the agent: the model reads "Error: …" as before, and the run counts the error.
            var fake = FakeChatModel.Script(FakeChatModel.ToolCall("read_file", new JsonObject { ["path"] = "Program.cs" }),
                FakeChatModel.ToolCall("edit_file", new JsonObject { ["path"] = "Program.cs", ["old_text"] = "int b;", ["new_text"] = "int c;" }), FakeChatModel.Answer("It failed."));
            var told = new List<ToolResult>();
            var run = new CodingAgent(fake) { OnToolResult = told.Add }.RunAsync("Rename b", tools).GetAwaiter().GetResult();
            Check(run is { ToolCalls: 2, ToolErrors: 1 } && told is [{ Succeeded: true }, { Succeeded: false }]
                  && fake.Requests[2].Messages[^1] is { Role: "tool" } message && message.Content.StartsWith("Error: old_text was not found", StringComparison.Ordinal),
                $"the agent: one failed call, the model told 'Error: …' ({fake.Requests[2].Messages[^1].Content})");
        });
    }

    // A 7B coder's reply: three json fences, write_file's content with \n escapes, quotes and braces.
    private const string ThreeFencedCalls = """
        I'll create the program, then change its greeting and run it.

        ```json
        {"name": "write_file", "arguments": {"path": "Program.cs", "content": "using System;\n\nclass Program\n{\n    static void Main()\n    {\n        Console.WriteLine(\"Hello\");\n    }\n}\n"}}
        ```

        ```json
        {"name": "edit_file", "arguments": {"path": "Program.cs", "old_text": "Console.WriteLine(\"Hello\");", "new_text": "Console.WriteLine(\"Hello, {0}!\", \"World\");"}}
        ```

        ```json
        {"name": "run_command", "arguments": {"command": "dotnet run"}}
        ```
        """;

    private static void AgentRunsEveryFencedCall(Device device)
    {
        _ = device;
        InAgentWorkspace(root =>
        {
            var fake = FakeChatModel.Script(FakeChatModel.Answer(ThreeFencedCalls), FakeChatModel.Answer("Program.cs prints Hello, World!"));
            var sandbox = new RecordingSandbox(new CommandResult(0, false, "Hello, World!", TimeSpan.Zero, false));
            var tools = new CodingTools(root, null, sandbox);
            var results = new List<ToolResult>();
            var run = new CodingAgent(fake) { OnToolResult = results.Add }.RunAsync("Write a hello program and run it", tools).GetAwaiter().GetResult();
            string program = Path.Combine(root, "Program.cs");
            string text = File.Exists(program) ? File.ReadAllText(program) : "(missing)";
            Check(results.Select(r => r.Call.Name).SequenceEqual(["write_file", "edit_file", "run_command"]) && results.All(r => r.Succeeded),
                $"the three calls ran in order: {string.Join(", ", results.Select(r => $"{r.Call.Name} {(r.Succeeded ? "ok" : r.Error)}"))}");
            Check(text.Contains("Console.WriteLine(\"Hello, {0}!\", \"World\");", StringComparison.Ordinal) && text.Contains("class Program\n{", StringComparison.Ordinal),
                $"Program.cs written with real newlines and braces, then edited: {text}");
            Check(sandbox.Calls.Count == 1 && run is { End: CodingAgentEnd.Answered, ToolCalls: 3, Rounds: 2 },
                $"the command ran once; {run.End}, {run.ToolCalls} calls, {run.Rounds} rounds");

            // The same reply with the file's line breaks written raw inside the JSON string (not valid JSON): before, only
            // the one-line run_command was read; now the strings' control characters are escaped and all three run.
            File.Delete(program);
            string raw = ThreeFencedCalls.Replace("\\n", "\n", StringComparison.Ordinal);
            var context = CodingAgent.CallContext(null, [.. tools.All.Select(t => t.Definition)]);
            var read = ToolCallText.CallsInText(raw, context);
            Check(read is { Count: 3 } && read[0].Name == "write_file" && (string?)read[0].Arguments["content"] == "using System;\n\nclass Program\n{\n    static void Main()\n    {\n        Console.WriteLine(\"Hello\");\n    }\n}\n"
                  && read[1].Name == "edit_file" && read[2].Name == "run_command",
                $"raw line breaks inside a string: the three calls read, the content's lines kept ({(read is null ? "none" : string.Join(", ", read.Select(c => c.Name)))})");
            var valid = ToolCallText.CallsInText(ThreeFencedCalls, context)!;
            Check(valid.Count == 3 && valid.Zip(read!).All(p => p.First.Name == p.Second.Name && JsonNode.DeepEquals(p.First.Arguments, p.Second.Arguments)),
                "valid JSON reads as before, and the escaped reply gives the same calls");
        });
    }

    private static void AgentReportsEachResultAsItEnds(Device device)
    {
        _ = device;
        InAgentWorkspace(root =>
        {
            // write_file, then a tool that waits until the turn is cancelled (an approval nobody gives, cut by the time limit).
            var results = new List<ToolResult>();
            bool toldBeforeWaiting = false;
            var wait = Tool.Create("wait_for_user", "Waits until cancelled.", new JsonObject { ["type"] = "object", ["properties"] = new JsonObject() },
                async (_, token) =>
                {
                    toldBeforeWaiting = results.Any(r => r.Call.Name == "write_file");
                    await Task.Delay(Timeout.Infinite, token);
                    return "";
                });
            var reply = new ChatMessage("assistant", ToolCalls: [new ToolCall("write_file", new JsonObject { ["path"] = "a.txt", ["content"] = "alpha" }), new ToolCall("wait_for_user", [])]);
            var fake = FakeChatModel.Script(reply, FakeChatModel.Answer("never asked"));
            var agent = new CodingAgent(fake, new AgentOptions { ExtraTools = [wait], TimeLimit = TimeSpan.FromSeconds(1) }) { OnToolResult = results.Add };
            var run = agent.RunAsync("Write a.txt", new CodingTools(root)).GetAwaiter().GetResult();
            Check(toldBeforeWaiting, "write_file's result was told before the next call began to wait");
            Check(run.End == CodingAgentEnd.OutOfBudget && File.Exists(Path.Combine(root, "a.txt")) && results.Count == 1 && results[0] is { Succeeded: true, Call.Name: "write_file" },
                $"the time limit cut the turn; the written file's result was told ({run.End}: {run.Note}; {results.Count} told)");
            Check(run is { ToolCalls: 0, Rounds: 1 } && !run.Messages.Any(m => m.Role == "tool"),
                "the run's counts and messages are as before: the cut turn adds no tool message");

            // Calls that all end: each told once, in call order, the same results as the turn's.
            results.Clear();
            var two = FakeChatModel.Script(new ChatMessage("assistant", ToolCalls: [new ToolCall("write_file", new JsonObject { ["path"] = "b.txt", ["content"] = "beta" }),
                new ToolCall("read_file", new JsonObject { ["path"] = "b.txt" })]), FakeChatModel.Answer("Done."));
            var done = new CodingAgent(two) { OnToolResult = results.Add }.RunAsync("Write and read b.txt", new CodingTools(root)).GetAwaiter().GetResult();
            Check(done is { End: CodingAgentEnd.Answered, ToolCalls: 2 } && results.Select(r => r.Call.Name).SequenceEqual(["write_file", "read_file"])
                  && results[1].Content.Contains("beta", StringComparison.Ordinal) && two.Requests[1].Messages.Count(m => m.Role == "tool") == 2,
                $"two calls told in order, their results sent on ({string.Join(", ", results.Select(r => r.Call.Name))})");
        });
    }

    // A workspace for one agent test, deleted after it.
    private static void InAgentWorkspace(Action<string> test)
    {
        string root = Path.Combine(Path.GetTempPath(), "idrak-calls-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            test(root);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    // A json fence whose call does not close: what the agent cannot read and corrects.
    private const string UnclosedFencedCall = "I'll create the file.\n```json\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"House.cs\", \"content\": \"class House {}\"}\n```\nThe file is created.";

    private static void AgentReadsFencedCalls(Device device)
    {
        _ = device;
        InAgentWorkspace(root =>
        {
            // The owner's reply: the agent reads write_file, read_file and run_command from the fences and runs them in
            // order (the build goes to a recording sandbox, so no build runs here).
            var fake = FakeChatModel.Script(FakeChatModel.Answer(HouseReplyWithFencedCalls), FakeChatModel.Answer("House.cs is written with Width, Length, Height and Price."));
            var sandbox = new RecordingSandbox(new CommandResult(0, false, "Build succeeded.", TimeSpan.Zero, false));
            var tools = new CodingTools(root, null, sandbox);
            var run = new CodingAgent(fake).RunAsync("Create C# class For House with float 3 props and + float price", tools).GetAwaiter().GetResult();
            Check(sandbox.Calls.Count == 1 && string.Join(' ', sandbox.Calls[0].Command) == "dotnet build", $"the fenced run_command asked for the build: {string.Join(" | ", sandbox.Calls.Select(c => string.Join(' ', c.Command)))}");
            string house = Path.Combine(root, "House.cs");
            Check(File.Exists(house) && File.ReadAllText(house).Contains("public float Price { get; set; }", StringComparison.Ordinal),
                $"the fenced write_file ran: House.cs {(File.Exists(house) ? "holds " + File.ReadAllText(house) : "is missing")}");
            var reply = run.Messages.First(m => m.Role == "assistant");
            Check(reply.ToolCalls is { Count: 3 } calls && calls[0].Name == "write_file" && calls[1].Name == "read_file" && calls[2].Name == "run_command"
                  && reply.Content == HouseReplyWithFencedCalls,
                $"the reply keeps its text and carries the three calls read from it: {(reply.ToolCalls is null ? "none" : Show(reply.ToolCalls))}");
            Check(run is { End: CodingAgentEnd.Answered, Rounds: 2, ToolCalls: 3, Nudges: 0 },
                $"the run goes on after the calls and ends with the answer ({run.End}, {run.Rounds} rounds, {run.ToolCalls} calls, {run.Nudges} nudges, {run.Note})");
            var sent = fake.Requests[1].Messages;
            Check(sent.Count(m => m.Role == "tool") == 3 && sent.Any(m => m.Role == "tool" && m.Content.Contains("public float Price", StringComparison.Ordinal)),
                "the model's next request carries the three results, read_file's showing the class");

            // Off: the same reply is not read; it looks like calls, so it is corrected, and the answer after it ends the run.
            File.Delete(house);
            var strict = FakeChatModel.Script(FakeChatModel.Answer(HouseReplyWithFencedCalls), FakeChatModel.Answer("I cannot call tools."));
            var off = new CodingAgent(strict, new AgentOptions { CallsInText = false }).RunAsync("Create the House class", tools).GetAwaiter().GetResult();
            Check(!File.Exists(house) && off is { End: CodingAgentEnd.Answered, Nudges: 1, ToolCalls: 0 },
                $"CallsInText off: nothing run, one correction ({off.End}, {off.Nudges} nudges, {off.ToolCalls} calls)");
        });
    }

    private static void AgentNudgesMalformedCall(Device device)
    {
        _ = device;
        InAgentWorkspace(root =>
        {
            var fake = FakeChatModel.Script(
                FakeChatModel.Answer(UnclosedFencedCall),
                FakeChatModel.ToolCall("write_file", new JsonObject { ["path"] = "House.cs", ["content"] = "public class House { public float Price { get; set; } }\n" }),
                FakeChatModel.Answer("House.cs is written."));
            var tools = new CodingTools(root);
            var run = new CodingAgent(fake, new AgentOptions { Template = new ChatMLTemplate() }).RunAsync("Create the House class", tools).GetAwaiter().GetResult();
            string house = Path.Combine(root, "House.cs");
            Check(File.Exists(house) && File.ReadAllText(house).Contains("float Price", StringComparison.Ordinal), "after the correction the model's proper call wrote House.cs");
            Check(run is { End: CodingAgentEnd.Answered, Rounds: 3, ToolCalls: 1, Nudges: 1 },
                $"one correction, then the call and the answer ({run.End}, {run.Rounds} rounds, {run.ToolCalls} calls, {run.Nudges} nudges)");
            var correction = fake.Requests[1].Messages[^1];
            Check(correction.Role == "user" && correction.Content.Contains("nothing in it was run", StringComparison.Ordinal)
                  && correction.Content.Contains("<tool_call>\n{\"name\":\"write_file\",\"arguments\":{\"path\":\"...\",\"content\":\"...\"}}\n</tool_call>", StringComparison.Ordinal)
                  && correction.Content.Contains("run_command", StringComparison.Ordinal),
                $"the correction says nothing ran and shows the template's own call format with an offered tool: {correction.Content}");
            Check(fake.Requests[1].Messages[^2] is { Role: "assistant", Content: UnclosedFencedCall }, "the malformed reply stays in the conversation before the correction");

            // A template whose answer is the call (bare JSON) and one with no known template: the example follows each.
            var llama = CodingAgent.Correction(FakeChatModel.Answer(UnclosedFencedCall),
                new ToolCallContext(new FixedCallsTemplate(new ToolCallFormat("", "", false, "parameters")), tools.All.Select(t => t.Definition).ToList(), null));
            Check(llama is not null && llama.Contains("as your whole reply", StringComparison.Ordinal) && llama.Contains("\"parameters\":", StringComparison.Ordinal) && !llama.Contains("<tool_call>", StringComparison.Ordinal),
                $"a bare-JSON template's correction: {llama}");
            var generic = CodingAgent.Correction(FakeChatModel.Answer(UnclosedFencedCall), CodingAgent.CallContext(null, [.. tools.All.Select(t => t.Definition)]));
            Check(generic is not null && generic.Contains("own tool-call format", StringComparison.Ordinal), $"no known template: a generic correction: {generic}");
            Check(CodingAgent.Correction(FakeChatModel.Answer("Done: House.cs has three floats and a price."), CodingAgent.CallContext(null, [.. tools.All.Select(t => t.Definition)])) is null,
                "an answer in words needs no correction");
        });
    }

    private static void AgentNudgeLimit(Device device)
    {
        _ = device;
        InAgentWorkspace(root =>
        {
            var tools = new CodingTools(root);
            var fake = FakeChatModel.Script(FakeChatModel.Answer(UnclosedFencedCall), FakeChatModel.Answer(UnclosedFencedCall), FakeChatModel.Answer(UnclosedFencedCall));
            var run = new CodingAgent(fake).RunAsync("Create the House class", tools).GetAwaiter().GetResult();
            Check(run is { End: CodingAgentEnd.CallsNotRead, Rounds: 3, Nudges: 2, ToolCalls: 0 } && run.Note.Contains("after 2 corrections", StringComparison.Ordinal)
                  && run.Messages[^1].Role == "assistant",
                $"two corrections (the default), then the run ends saying why ({run.End}, {run.Rounds} rounds, {run.Nudges} nudges: {run.Note})");
            Check(!File.Exists(Path.Combine(root, "House.cs")), "nothing was written");
            Check((int?)run.ToJson()["nudges"] == 2, "the transcript's figures count the corrections");

            var none = new CodingAgent(FakeChatModel.Script(FakeChatModel.Answer(UnclosedFencedCall)), new AgentOptions { Nudges = 0 }).RunAsync("x", tools).GetAwaiter().GetResult();
            Check(none is { End: CodingAgentEnd.CallsNotRead, Rounds: 1, Nudges: 0 } && none.Note.Contains("no correction", StringComparison.Ordinal),
                $"Nudges 0: the first such reply ends the run ({none.End}, {none.Rounds}: {none.Note})");

            // A correction in a row is counted again after a reply whose calls were read.
            var reset = FakeChatModel.Script(FakeChatModel.Answer(UnclosedFencedCall), FakeChatModel.ToolCall("list_files", []), FakeChatModel.Answer(UnclosedFencedCall), FakeChatModel.Answer("Done."));
            var again = new CodingAgent(reset, new AgentOptions { Nudges = 1 }).RunAsync("x", tools).GetAwaiter().GetResult();
            Check(again is { End: CodingAgentEnd.Answered, Nudges: 2, Rounds: 4 }, $"the count restarts after a call ({again.End}, {again.Nudges} nudges, {again.Rounds} rounds)");

            foreach (var answer in new[] { "Done: House.cs has three floats and a price.", "Here is the class:\n```csharp\npublic class House { public float Price { get; set; } }\n```" })
            {
                var plain = new CodingAgent(FakeChatModel.Script(FakeChatModel.Answer(answer))).RunAsync("x", tools).GetAwaiter().GetResult();
                Check(plain is { End: CodingAgentEnd.Answered, Rounds: 1, Nudges: 0 }, $"no call-like text, no correction: '{answer[..20]}…' ended {plain.End} after {plain.Rounds} rounds");
            }

            ThrowsNaming<ArgumentOutOfRangeException>(() => new CodingAgent(FakeChatModel.Script(), new AgentOptions { Nudges = -1 }).RunAsync("x", tools).GetAwaiter().GetResult(),
                "a negative Nudges", "Nudges");
        });
    }

    // A template with a fixed tool-call shape and the "json" format (for the correction's example).
    private sealed class FixedCallsTemplate(ToolCallFormat calls) : ChatTemplate
    {
        public override ToolCallFormat ToolCalls => calls;

        public override IReadOnlyList<string> StopSequences => [];

        public override void Render(IReadOnlyList<ChatMessage> messages, IReadOnlyList<ToolDefinition> tools, bool? think, System.Text.StringBuilder prompt) => throw new NotSupportedException();
    }

    private static void AgentContinuesConversation(Device device)
    {
        _ = device;
        string root = Path.Combine(Path.GetTempPath(), "idrak-continue-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        File.WriteAllText(Path.Combine(root, "a.txt"), "alpha");
        try
        {
            var fake = FakeChatModel.Script(
                FakeChatModel.ToolCall("read_file", new JsonObject { ["path"] = "a.txt" }),
                FakeChatModel.Answer("It says alpha."),
                FakeChatModel.Answer("You asked what a.txt says; it says alpha."));
            var agent = new CodingAgent(fake);
            var tools = new CodingTools(root);
            var first = agent.RunAsync("What does a.txt say?", tools).GetAwaiter().GetResult();
            var second = agent.ContinueAsync(first, "What did I ask you?", tools).GetAwaiter().GetResult();
            var sent = fake.Requests[2].Messages;
            Check(sent.Count == first.Messages.Count + 1 && sent.Take(first.Messages.Count).SequenceEqual(first.Messages),
                $"the next request carries the whole earlier conversation ({first.Messages.Count} messages) then the new one; it sent {sent.Count}");
            Check(sent[0].Role == "system" && sent.Count(m => m.Role == "system") == 1, "one system prompt, the earlier run's");
            Check(sent[^1] is { Role: "user", Content: "What did I ask you?" } && sent.Any(m => m is { Role: "tool" } && m.Content.Contains("alpha", StringComparison.Ordinal)),
                "the new message last, the earlier tool result kept");
            Check(second is { End: CodingAgentEnd.Answered, Rounds: 1 } && second.Messages.Count == sent.Count + 1 && second.Messages[^1].Content.StartsWith("You asked", StringComparison.Ordinal),
                $"the continued run holds the whole conversation and counts its own rounds ({second.End}, {second.Rounds} rounds, {second.Messages.Count} messages)");
            var third = new CodingAgent(FakeChatModel.Script(FakeChatModel.Answer("ok")))
                .ContinueAsync(first with { Messages = [.. first.Messages.Skip(1)] }, "again", tools).GetAwaiter().GetResult();
            Check(third.Messages[0].Role == "system" && third.Messages.Count(m => m.Role == "system") == 1, "a conversation without its system prompt gets one first");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    // CodingTools cuts lines as places in the text and matches spans; the reference splits the text into strings.
    private static void CodingToolsLinesMatchReference()
    {
        static List<string> Lines(string text)
        {
            var lines = text.Replace("\r\n", "\n", StringComparison.Ordinal).Split('\n').ToList();
            if (lines.Count > 0 && lines[^1].Length == 0)
            {
                lines.RemoveAt(lines.Count - 1);
            }

            return lines;
        }

        string root = Path.Combine(Path.GetTempPath(), "idrak-lines-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            string[] texts =
            [
                "", "\n", "\r\n", "\r", "one", "one\n", "one\r\n", "one\r", "a\r\r\nb\n\r\nc\r", "\n\nx\n\n", "  match me  \r\nno\r\n\tmatch\tagain\t\n",
                string.Concat(Enumerable.Range(1, 1234).Select(i => i % 7 == 0 ? $"line {i} match {new string('x', i % 300)}\r\n" : $"line {i}\n")),
            ];
            var tools = new CodingTools(root, new CodingToolOptions { MaxReadLines = 50, MaxSearchResults = 2000 });
            for (int f = 0; f < texts.Length; f++)
            {
                string name = $"f{f}.txt";
                File.WriteAllText(Path.Combine(root, name), texts[f]);
                var lines = Lines(texts[f]);
                foreach (var (start, end) in new (int?, int?)[] { (null, null), (1, 3), (2, null), (40, 1300), (1200, null) })
                {
                    string expected;
                    int first = Math.Max(1, start ?? 1);
                    if (lines.Count == 0)
                    {
                        expected = "(empty file)";
                    }
                    else if (first > lines.Count)
                    {
                        expected = $"Error: the file has {lines.Count} lines.";
                    }
                    else
                    {
                        int last = Math.Min(lines.Count, Math.Min(end ?? int.MaxValue, first + 50 - 1));
                        int width = last.ToString(System.Globalization.CultureInfo.InvariantCulture).Length;
                        var sb = new System.Text.StringBuilder();
                        for (int i = first; i <= last; i++)
                        {
                            sb.Append(i.ToString(System.Globalization.CultureInfo.InvariantCulture).PadLeft(width)).Append("| ").Append(lines[i - 1]).Append('\n');
                        }

                        if (last < lines.Count && (end is null || last < end))
                        {
                            sb.Append($"… lines {last + 1}-{lines.Count} not shown (read_file with start_line {last + 1})\n");
                        }

                        expected = sb.ToString().TrimEnd('\n');
                    }

                    Check(tools.ReadFile(name, start, end) == expected, $"read_file {name} {start}-{end}");
                }

                foreach (string pattern in new[] { "match", "^line", "\\r", "^$", "e$" })
                {
                    var regex = new System.Text.RegularExpressions.Regex(pattern, System.Text.RegularExpressions.RegexOptions.CultureInvariant);
                    var found = lines.Select((l, i) => (Line: l.Trim(), Number: i + 1)).Where((p, i) => regex.IsMatch(lines[i]))
                        .Select(p => $"{name}:{p.Number}: {(p.Line.Length > 200 ? p.Line[..200] + " …" : p.Line)}").ToList();
                    Check(tools.Search(pattern, name) == (found.Count == 0 ? "(no matches)" : string.Join('\n', found)), $"search {pattern} in {name}");
                }
            }
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }

        string[] pieces = ["a", "bc", " ", "\t", "\r\n", "\u00a0", "\u2028", "\u3000", "\u0085", "\u200b", "\ufeff", "é"];
        var random = new Random(3);
        for (int n = 0; n < 2000; n++)
        {
            string text = string.Concat(Enumerable.Range(0, random.Next(0, 30)).Select(_ => pieces[random.Next(pieces.Length)]));
            int max = random.Next(1, 40);
            string collapsed = System.Text.RegularExpressions.Regex.Replace(text, @"\s+", " ").Trim();
            Check(WebTools.CollapseSpaces(text, max) == (collapsed.Length <= max ? collapsed : collapsed[..max]), $"collapse '{text}' to {max}");
        }

        for (int c = 0; c <= char.MaxValue; c++)
        {
            string text = $"a{(char)c}b";
            Check(WebTools.CollapseSpaces(text, 10) == System.Text.RegularExpressions.Regex.Replace(text, @"\s+", " "), $"white space U+{c:X4}");
        }
    }

    // Small models ask for the same call again instead of answering from its result (Qwen2.5-Coder-1.5B ran
    // dotnet --version three times): the repeat is answered with a note, not run; the same call later still runs.
    private static void AgentSkipsRepeatedCalls(Device device)
    {
        _ = device;
        string root = Path.Combine(Path.GetTempPath(), "idrak-agent-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        File.WriteAllText(Path.Combine(root, "a.txt"), "alpha");
        File.WriteAllText(Path.Combine(root, "b.txt"), "beta");
        try
        {
            JsonObject Read(string file) => new() { ["path"] = file };
            var fake = FakeChatModel.Script(
                FakeChatModel.ToolCall("read_file", Read("a.txt")),
                FakeChatModel.ToolCall("read_file", Read("a.txt")),                 // repeated: not run
                FakeChatModel.ToolCall("read_file", Read("a.txt")),                 // again: not run
                FakeChatModel.ToolCall("read_file", Read("b.txt")),
                FakeChatModel.ToolCall("read_file", Read("a.txt")),                 // after another call: runs
                FakeChatModel.Answer("a.txt says alpha."));
            var results = new List<ToolResult>();
            var agent = new CodingAgent(fake) { OnToolResult = results.Add };
            var run = agent.RunAsync("What does a.txt say?", new CodingTools(root)).GetAwaiter().GetResult();
            Check(run is { End: CodingAgentEnd.Answered, Rounds: 6, ToolCalls: 5, ToolErrors: 2 }, $"counts {run.End} {run.Rounds} {run.ToolCalls} {run.ToolErrors}");
            Check(results.Select(r => r.Succeeded).SequenceEqual([true, false, false, true, true]), "run, skipped, skipped, run, run");
            Check(results[1].Error!.StartsWith("Not run again", StringComparison.Ordinal), $"the note: {results[1].Error}");
            Check(fake.Requests[2].Messages[^1] is { Role: "tool", ToolName: "read_file" } note && note.Content.Contains("Not run again", StringComparison.Ordinal),
                "the model sees the note in place of a result");
            Check(results[4].Content.Contains("alpha", StringComparison.Ordinal), "the later repeat reads the file");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void AgentReadsSteeringNotes(Device device)
    {
        _ = device;
        string root = Path.Combine(Path.GetTempPath(), "idrak-steer-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        File.WriteAllText(Path.Combine(root, "a.txt"), "alpha");
        try
        {
            var fake = FakeChatModel.Script(
                FakeChatModel.ToolCall("read_file", new JsonObject { ["path"] = "a.txt" }),
                FakeChatModel.ToolCall("list_files", []),
                FakeChatModel.Answer("Done."));
            CodingAgent? agent = null;
            // The note is given while the first tool call runs: the next request carries it, and the one after still has it.
            agent = new CodingAgent(fake) { OnToolResult = r => { if (r.Call.Name == "read_file") agent!.Steer("  Only read, do not edit.  "); } };
            var run = agent.RunAsync("What does a.txt say?", new CodingTools(root)).GetAwaiter().GetResult();
            Check(run is { End: CodingAgentEnd.Answered, Rounds: 3 }, $"counts {run.End} {run.Rounds}");
            Check(!fake.Requests[0].Messages.Any(m => m.Content.Contains("Note from the user", StringComparison.Ordinal)), "no note before it was given");
            Check(fake.Requests[1].Messages[^1] is { Role: "user", Content: "Note from the user: Only read, do not edit." },
                $"the next request ends with the note, trimmed (it ends with '{fake.Requests[1].Messages[^1].Content}')");
            Check(fake.Requests[2].Messages.Count(m => m.Content == "Note from the user: Only read, do not edit.") == 1, "the note stays in the conversation, once");
            Check(agent.PendingNotes == 0, "every note was read");
            bool refused = false;
            try
            {
                agent.Steer(" ");
            }
            catch (ArgumentException)
            {
                refused = true;
            }

            Check(refused, "an empty note is refused");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void AgentOffersExtraTools(Device device)
    {
        _ = device;
        string root = Path.Combine(Path.GetTempPath(), "idrak-extra-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            var search = Tool.Create("search_code", "Searches the code by meaning.", new JsonObject { ["type"] = "object", ["properties"] = new JsonObject() },
                (_, _) => Task.FromResult("Shapes.cs:4 class Circle"));
            var fake = FakeChatModel.Script(FakeChatModel.ToolCall("search_code", new JsonObject { ["query"] = "circle" }), FakeChatModel.Answer("In Shapes.cs."));
            var agent = new CodingAgent(fake, new AgentOptions { ExtraTools = [search] });
            var run = agent.RunAsync("Where is the circle?", new CodingTools(root)).GetAwaiter().GetResult();
            var offered = fake.Requests[0].Tools!.Select(t => t.Name).ToList();
            Check(offered[^1] == "search_code" && offered.Contains("read_file"), $"the extra tool is offered after the coding tools ({string.Join(", ", offered)})");
            Check(run is { End: CodingAgentEnd.Answered, ToolCalls: 1, ToolErrors: 0 }, $"the call ran ({run.End}, {run.ToolCalls} calls, {run.ToolErrors} errors)");
            Check(fake.Requests[1].Messages[^1] is { Role: "tool", Content: "Shapes.cs:4 class Circle" }, $"its result reached the model ({fake.Requests[1].Messages[^1].Content})");
            var clash = Tool.Create("read_file", "another read", new JsonObject { ["type"] = "object" }, (_, _) => Task.FromResult(""));
            string? refusal = null;
            try
            {
                new CodingAgent(FakeChatModel.Script(FakeChatModel.Answer("x")), new AgentOptions { ExtraTools = [clash] })
                    .RunAsync("x", new CodingTools(root)).GetAwaiter().GetResult();
            }
            catch (ArgumentException e)
            {
                refusal = e.Message;
            }

            Check(refusal is not null && refusal.Contains("read_file", StringComparison.Ordinal), $"a coding tool's name is refused, naming it ({refusal})");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void AgentTranscriptJson(Device device)
    {
        _ = device;
        var tools = new CodingTools(Path.GetTempPath(), new CodingToolOptions { ReadOnly = true });
        ChatMessage[] messages =
        [
            new("system", "Be brief."), new("user", "Where is Main?"),
            new("assistant", "", "Search first.", [new ToolCall("search", new JsonObject { ["pattern"] = "Main" }), new ToolCall("read_file", new JsonObject { ["path"] = "a.cs" })]),
            new("tool", "a.cs:1: static void Main()", ToolName: "search"), new("tool", "1| static void Main()", ToolName: "read_file"),
            new("assistant", "In a.cs."),
        ];
        var json = ChatJson.Transcript(messages, [.. tools.All.Select(t => t.Definition)], think: true);
        var back = ChatTranscript.FromJson(JsonNode.Parse(json.ToJsonString())!.AsObject());
        Check(back.Messages.Count == 6 && back.Think == true && back.Tools.Count == 3, "counts");
        Check(back.Messages[2] is { Thinking: "Search first.", ToolCalls: [{ Name: "search" }, { Name: "read_file" }] }, "calls and reasoning");
        Check(back.Messages[2].ToolCalls![0].Arguments["pattern"]!.GetValue<string>() == "Main", "arguments");
        Check(back.Messages[4].ToolName == "read_file" && (string?)json["messages"]![4]!["tool_call_id"] == "call_2", "tool results matched to calls");
        Check(back.Tools[0].Parameters!["required"]!.AsArray().Count == 0 && back.Tools[1].Parameters!["required"]![0]!.GetValue<string>() == "path", "tool schemas");
    }

    private static void AgentRunsTask(Device device)
    {
        _ = device;
        string root = Path.Combine(Path.GetTempPath(), "idrak-agent-" + Guid.NewGuid().ToString("N"));
        string task = Path.Combine(root, "suite", "csharp", "fix-greeting");
        Directory.CreateDirectory(Path.Combine(task, "workspace", "bin"));
        Directory.CreateDirectory(Path.Combine(task, "verify"));
        File.WriteAllText(Path.Combine(task, "task.json"), "{\"prompt\":\"Make Greet say Hello.\",\"language\":\"csharp\",\"verify\":[\"git grep -q Hello -- Greeter.cs\"],\"tags\":[\"bug\"]}");
        File.WriteAllText(Path.Combine(task, "workspace", "Greeter.cs"), "static class Greeter\n{\n    public static string Greet() => \"Bye\";\n}\n");
        File.WriteAllText(Path.Combine(task, "workspace", "bin", "old.dll"), "x");
        File.WriteAllText(Path.Combine(task, "verify", "GreeterTests.cs"), "// hidden test\n");
        try
        {
            var suite = CodingSuite.Load(Path.Combine(root, "suite"));
            Check(suite is [{ Id: "csharp/fix-greeting", Language: "csharp", Verify: ["git grep -q Hello -- Greeter.cs"] }], $"suite: {string.Join(",", suite.Select(t => t.Id))}");
            var fake = FakeChatModel.Script(
                FakeChatModel.ToolCall("read_file", new JsonObject { ["path"] = "Greeter.cs" }),
                FakeChatModel.ToolCall("edit_file", new JsonObject { ["path"] = "Greeter.cs", ["old_text"] = "\"Bye\"", ["new_text"] = "\"Hello\"" }),
                FakeChatModel.ToolCall("read_file", new JsonObject { ["path"] = "Missing.cs" }),
                FakeChatModel.Answer("Greet now returns Hello.", thinking: "Done."));
            var tools = new List<string>();
            var agent = new CodingAgent(fake, new AgentOptions { Think = true }) { OnToolResult = r => tools.Add(r.Call.Name) };
            var verifier = new CodingVerifier();
            var run = verifier.RunAsync(agent, suite[0], Path.Combine(root, "run1")).GetAwaiter().GetResult();
            Check(run.Outcome == CodingOutcome.Passed, $"passed: {run.Outcome} {run.VerifyOutput}");
            Check(run.Agent is { Rounds: 4, ToolCalls: 3, ToolErrors: 1 } && tools.Count == 3, $"counts {run.Agent.Rounds} {run.Agent.ToolCalls} {run.Agent.ToolErrors}");
            Check(!Directory.Exists(Path.Combine(root, "run1", "bin")) && File.Exists(Path.Combine(root, "run1", "GreeterTests.cs")), "build output skipped, hidden files copied");
            Check(fake.Requests[0].Messages is [{ Role: "system" }, { Role: "user", Content: "Make Greet say Hello." }] && fake.Requests[0].Tools!.Count == 6, "first request");
            Check(fake.Requests[1].Messages[^1] is { Role: "tool", ToolName: "read_file" } m && m.Content.Contains("\"Bye\"", StringComparison.Ordinal), "tool results reach the model");

            var line = run.ToJson();
            Check((string?)line["outcome"] == "Passed" && (string?)line["task"] == "csharp/fix-greeting" && (int?)line["rounds"] == 4, "run fields");
            var transcript = ChatTranscript.FromJson(JsonNode.Parse(line.ToJsonString())!.AsObject());
            Check(transcript.Messages.Count == 9 && transcript.Tools.Count == 6 && transcript.Messages[^1].Content == "Greet now returns Hello.", "transcript reads back");

            var lazy = FakeChatModel.Script(FakeChatModel.Answer("Done!"));
            var failed = verifier.RunAsync(new CodingAgent(lazy), suite[0], Path.Combine(root, "run2")).GetAwaiter().GetResult();
            Check(failed.Outcome == CodingOutcome.Failed && failed.VerifyOutput.Contains("exit code 1", StringComparison.Ordinal), $"verification fails: {failed.VerifyOutput}");

            var looping = FakeChatModel.Script([.. Enumerable.Repeat(FakeChatModel.ToolCall("list_files", []), 5)]);
            var budget = verifier.RunAsync(new CodingAgent(looping, new AgentOptions { MaxRounds = 3 }), suite[0], Path.Combine(root, "run3")).GetAwaiter().GetResult();
            Check(budget.Outcome == CodingOutcome.OutOfBudget && budget.Agent.Rounds == 3, "round limit");
        }
        finally
        {
            foreach (var file in Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories))
            {
                File.SetAttributes(file, FileAttributes.Normal);   // git's read-only objects
            }

            Directory.Delete(root, true);
        }
    }

    private static void CodingToolsFiles(Device device)
    {
        _ = device;
        string root = Path.Combine(Path.GetTempPath(), "ns-coding-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(root, "src", "app"));
        Directory.CreateDirectory(Path.Combine(root, "node_modules", "lib"));
        Directory.CreateDirectory(Path.Combine(root, "bin"));
        try
        {
            File.WriteAllText(Path.Combine(root, "src", "app", "user.service.ts"), "export class UserService {\n  getUsers() {\n    return [];\n  }\n}\n");
            File.WriteAllText(Path.Combine(root, "src", "Program.cs"), "class Program\r\n{\r\n    static void Main() { }\r\n}\r\n");
            File.WriteAllText(Path.Combine(root, "node_modules", "lib", "index.ts"), "export class UserService {}\n");
            File.WriteAllText(Path.Combine(root, "bin", "app.dll"), "x\0y");
            File.WriteAllText(Path.Combine(root, "long.txt"), string.Join('\n', Enumerable.Range(1, 1000).Select(i => $"line {i}")));
            var tools = new CodingTools(root, new CodingToolOptions { MaxReadLines = 50 });
            var registry = tools.Registry();
            string Run(string name, string json)
            {
                var result = registry.InvokeAsync(Call(name, json)).GetAwaiter().GetResult();
                return result.Content ?? result.Error!;
            }

            Check(tools.All.Select(t => t.Definition.Name).SequenceEqual(["list_files", "read_file", "search", "edit_file", "write_file", "run_command"]), "tool names");
            Check(Run("list_files", "{}") == "long.txt\nsrc/Program.cs\nsrc/app/user.service.ts", $"list skips ignored folders: {Run("list_files", "{}")}");
            Check(Run("list_files", "{\"pattern\":\"*.ts\"}") == "src/app/user.service.ts", "list pattern");
            Check(Run("read_file", "{\"path\":\"src/app/user.service.ts\",\"start_line\":2,\"end_line\":3}") == "2|   getUsers() {\n3|     return [];", "read range");
            string longRead = Run("read_file", "{\"path\":\"long.txt\"}");
            Check(longRead.Contains("50| line 50", StringComparison.Ordinal) && !longRead.Contains("line 51\n", StringComparison.Ordinal)
                  && longRead.EndsWith("start_line 51)", StringComparison.Ordinal), "long files come in parts");
            Check(Run("read_file", "{\"path\":\"../secret.txt\"}").Contains("outside the workspace", StringComparison.Ordinal), "no escaping the workspace");
            Check(Run("read_file", "{\"path\":\"/etc/passwd\"}").Contains("outside the workspace", StringComparison.Ordinal), "no absolute paths out");
            Check(Run("read_file", "{\"path\":\"missing.cs\"}").Contains("does not exist", StringComparison.Ordinal), "missing file");

            Check(Run("search", "{\"pattern\":\"class \\\\w+Service\"}") == "src/app/user.service.ts:1: export class UserService {", "search skips node_modules");
            Check(Run("search", "{\"pattern\":\"MAIN\",\"ignore_case\":true,\"glob\":\"*.cs\"}") == "src/Program.cs:3: static void Main() { }", "search options");
            Check(Run("search", "{\"pattern\":\"(\"}").Contains("invalid regular expression", StringComparison.Ordinal), "bad regex");

            string edited = Run("edit_file", "{\"path\":\"src/app/user.service.ts\",\"old_text\":\"    return [];\",\"new_text\":\"    return this.http.get('/api/users');\"}");
            Check(edited.StartsWith("Edited src/app/user.service.ts", StringComparison.Ordinal) && edited.Contains("3|     return this.http.get", StringComparison.Ordinal), $"edit shows the result: {edited}");
            Check(Run("edit_file", "{\"path\":\"src/app/user.service.ts\",\"old_text\":\"return [];\",\"new_text\":\"x\"}").Contains("not found", StringComparison.Ordinal), "stale edit");
            Check(Run("edit_file", "{\"path\":\"src/app/user.service.ts\",\"old_text\":\"getUsers() {\\n return\",\"new_text\":\"x\"}").Contains("whitespace is ignored", StringComparison.Ordinal), "whitespace hint");
            Check(Run("edit_file", "{\"path\":\"src/app/user.service.ts\",\"old_text\":\"s\",\"new_text\":\"S\"}").Contains("occurs", StringComparison.Ordinal), "ambiguous edit");

            // Line endings: the model writes \n; a CRLF file keeps CRLF (read first: edit_file refuses a file not read).
            Run("read_file", "{\"path\":\"src/Program.cs\"}");
            Run("edit_file", "{\"path\":\"src/Program.cs\",\"old_text\":\"{\\n    static void Main() { }\\n}\",\"new_text\":\"{\\n    static void Main() => System.Console.WriteLine(1);\\n}\"}");
            Check(File.ReadAllText(Path.Combine(root, "src", "Program.cs")) == "class Program\r\n{\r\n    static void Main() => System.Console.WriteLine(1);\r\n}\r\n", "CRLF kept");

            Check(Run("write_file", "{\"path\":\"src/app/user.ts\",\"content\":\"export interface User { id: number; }\\n\"}") == "Created src/app/user.ts (1 lines).", "write");
            Check(File.Exists(Path.Combine(root, "src", "app", "user.ts")), "written");
            Check(Run("write_file", "{\"path\":\"../x.ts\",\"content\":\"\"}").Contains("outside", StringComparison.Ordinal), "no writing outside");

            var readOnly = new CodingTools(root, new CodingToolOptions { ReadOnly = true });
            Check(readOnly.All.Count == 3, "read-only toolset");
        }
        finally
        {
            Directory.Delete(root, true);
        }
    }

    private static void CodingToolsCommands(Device device)
    {
        _ = device;
        string root = Path.Combine(Path.GetTempPath(), "ns-coding-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            var tools = new CodingTools(root, new CodingToolOptions { Commands = ["dotnet", "git"], MaxOutputCharacters = 300 });
            string Run(string command, int? timeout = null) => tools.RunCommandAsync(command, ".", timeout).GetAwaiter().GetResult();

            Check(Run("rm -rf /").Contains("not an allowed program", StringComparison.Ordinal), "allowlist");
            Check(Run("/usr/bin/dotnet --version").Contains("not an allowed program", StringComparison.Ordinal), "no paths to programs");
            Check(Run("dotnet build && rm x").Contains("not run by a shell", StringComparison.Ordinal), "no shell operators");
            Check(Run("git push").Contains("only git", StringComparison.Ordinal), "git read-only");

            string version = Run("dotnet --version");
            Check(version.StartsWith("$ dotnet --version\nexit code 0", StringComparison.Ordinal), $"dotnet runs: {version}");
            string help = Run("dotnet --help");
            Check(help.Contains("characters cut", StringComparison.Ordinal) && help.Length < 500, "long output is trimmed");
            string failed = Run("dotnet \"no such command\"");
            Check(!failed.Contains("exit code 0", StringComparison.Ordinal), $"failure exit code: {failed}");
        }
        finally
        {
            Directory.Delete(root, true);
        }
    }
}
