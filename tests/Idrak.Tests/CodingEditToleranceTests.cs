// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json.Nodes;
using Idrak.Text.Generation;

// edit_file's tolerance (CodingToolOptions.EditTolerance), its error that quotes old_text and the file, the encoding kept,
// and the agent's reminder of failed changes (AgentOptions.FailedCallNudges): from Craft's agent bench, where a 7B
// Instruct model sent an old_text it had guessed before reading Program.cs (a byte order mark, CRLF, a blank last
// line) and was told only to read the file again, ten times; and answered "done" with its edits failed.
internal abstract partial class DesktopTests
{
    // The bench's Program.cs, byte for byte: a UTF-8 byte order mark, two lines and a blank one, CRLF.
    private static readonly byte[] BenchProgram = [0xEF, 0xBB, 0xBF, .. Encoding.UTF8.GetBytes("Console.WriteLine(\"Hello, World!\");\r\nConsole.WriteLine(\"Hello World\");\r\n\r\n")];

    private static void CodingEditTolerance(Device device)
    {
        _ = device;
        InAgentWorkspace(root =>
        {
            string program = Path.Combine(root, "Program.cs");
            var tools = new CodingTools(root);
            byte[] Bytes(string text) => [0xEF, 0xBB, 0xBF, .. Encoding.UTF8.GetBytes(text)];
            string Edit(string oldText, string newText)
            {
                _ = tools.ReadFile("Program.cs");
                return tools.EditFile("Program.cs", oldText, newText);
            }

            // The bench's call: a guessed old_text. Not found, and the error shows it and the file's text to copy from.
            File.WriteAllBytes(program, BenchProgram);
            string missed = Edit("// This is a comment", "Console.WriteLine(42);");
            Check(missed.StartsWith("Error: old_text was not found in Program.cs. old_text was: \"// This is a comment\".", StringComparison.Ordinal)
                  && missed.Contains("not in the file", StringComparison.Ordinal) && missed.Contains("```\nConsole.WriteLine(\"Hello, World!\");\nConsole.WriteLine(\"Hello World\");\n\n```", StringComparison.Ordinal)
                  && File.ReadAllBytes(program).SequenceEqual(BenchProgram),
                $"a guessed old_text: the error quotes it and the file's lines, nothing changes ({missed})");

            // Exact first and unchanged: the byte order mark and CRLF kept.
            string exact = Edit("Console.WriteLine(\"Hello World\");", "Console.WriteLine(1);");
            Check(exact.StartsWith("Edited Program.cs (1 replacement):", StringComparison.Ordinal)
                  && File.ReadAllBytes(program).SequenceEqual(Bytes("Console.WriteLine(\"Hello, World!\");\r\nConsole.WriteLine(1);\r\n\r\n")),
                $"an exact edit keeps the byte order mark and CRLF, and names no tolerance ({exact})");

            // read_file's line numbers copied into old_text (and new_text): taken off, said so.
            File.WriteAllBytes(program, BenchProgram);
            string numbered = Edit("1| Console.WriteLine(\"Hello, World!\");", "1| Console.WriteLine(2);");
            Check(numbered.Contains("old_text matched after removing read_file's line numbers", StringComparison.Ordinal)
                  && File.ReadAllBytes(program).SequenceEqual(Bytes("Console.WriteLine(2);\r\nConsole.WriteLine(\"Hello World\");\r\n\r\n")),
                $"line numbers taken off old_text and new_text ({numbered})");

            // A read_file range copied whole (right-aligned numbers, two lines) matches; numbers out of sequence do not.
            File.WriteAllText(program, string.Concat(Enumerable.Range(1, 12).Select(i => $"int v{i} = {i};\n")));
            _ = tools.ReadFile("Program.cs");
            string range = tools.ReadFile("Program.cs", 9, 10);
            string ranged = tools.EditFile("Program.cs", range, "int nine = 9;\nint ten = 10;");
            Check(range == " 9| int v9 = 9;\n10| int v10 = 10;" && ranged.Contains("after removing read_file's line numbers", StringComparison.Ordinal)
                  && File.ReadAllText(program).Contains("int v8 = 8;\nint nine = 9;\nint ten = 10;\nint v11 = 11;\n", StringComparison.Ordinal),
                $"a copied read_file range ({range}) replaces its lines ({ranged})");
            string skipped = tools.EditFile("Program.cs", "1| int v1 = 1;\n3| int v3 = 3;", "x");
            Check(skipped.StartsWith("Error: old_text was not found", StringComparison.Ordinal) && File.ReadAllText(program).StartsWith("int v1 = 1;\n", StringComparison.Ordinal),
                $"line numbers out of sequence are not taken off ({skipped})");

            // Trailing white space and the final line break: matched line by line, the file's own line break kept.
            File.WriteAllBytes(program, BenchProgram);
            string spaced = Edit("Console.WriteLine(\"Hello World\");   \n", "Console.WriteLine(3);\n");
            Check(spaced.Contains("old_text matched with trailing white space and the final line break ignored", StringComparison.Ordinal)
                  && File.ReadAllBytes(program).SequenceEqual(Bytes("Console.WriteLine(\"Hello, World!\");\r\nConsole.WriteLine(3);\r\n\r\n")),
                $"trailing spaces and the final line break ignored ({spaced})");
            File.WriteAllText(program, "a();\nb();");
            string last = Edit("b();\n", "c();\n");
            Check(last.Contains("final line break ignored", StringComparison.Ordinal) && File.ReadAllText(program) == "a();\nc();",
                $"an extra final line break in old_text, the file's last line without one ({last})");

            // Two tolerant matches: refused, nothing changed.
            File.WriteAllText(program, "a();  \nb();\na(); \nb();\n");
            string twice = Edit("a();\nb();", "c();");
            Check(twice.StartsWith("Error: old_text was not found exactly in Program.cs;", StringComparison.Ordinal) && twice.Contains("2 or more times", StringComparison.Ordinal)
                  && File.ReadAllText(program) == "a();  \nb();\na(); \nb();\n",
                $"an ambiguous tolerant match is refused ({twice})");

            // The option off: the exact text only.
            File.WriteAllBytes(program, BenchProgram);
            var strict = new CodingTools(root, new CodingToolOptions { EditTolerance = false, EditQuotedLines = 0 });
            _ = strict.ReadFile("Program.cs");
            string refused = strict.EditFile("Program.cs", "1| Console.WriteLine(\"Hello, World!\");", "x");
            Check(refused.StartsWith("Error: old_text was not found", StringComparison.Ordinal) && !refused.Contains("```", StringComparison.Ordinal)
                  && File.ReadAllBytes(program).SequenceEqual(BenchProgram),
                $"EditTolerance off: numbered old_text is not found, EditQuotedLines 0 quotes no lines ({refused})");

            // write_file over a file with a byte order mark keeps it.
            _ = tools.WriteFile("Program.cs", "Console.WriteLine(4);\n");
            Check(File.ReadAllBytes(program).SequenceEqual(Bytes("Console.WriteLine(4);\r\n")), "write_file keeps the byte order mark and CRLF");
        });
    }

    private static void AgentRemindsFailedChanges(Device device)
    {
        _ = device;
        InAgentWorkspace(root =>
        {
            string program = Path.Combine(root, "Program.cs");
            var read = FakeChatModel.ToolCall("read_file", new JsonObject { ["path"] = "Program.cs" });
            JsonObject Guess() => new() { ["path"] = "Program.cs", ["old_text"] = "// This is a comment", ["new_text"] = "Console.WriteLine(42);" };
            const string Reminder = "These calls failed and did not take effect:\n- edit_file Program.cs: old_text was not found in Program.cs.";

            // The bench's run: a failed edit, then "done": reminded once, and the second answer says what was not done.
            File.WriteAllBytes(program, BenchProgram);
            var fake = FakeChatModel.Script(read, FakeChatModel.ToolCall("edit_file", Guess()), FakeChatModel.Answer("Done, I changed Program.cs."),
                FakeChatModel.Answer("Program.cs was not changed: the text I tried to replace is not in it."));
            var run = new CodingAgent(fake).RunAsync("Print 42", new CodingTools(root)).GetAwaiter().GetResult();
            var sent = run.Messages.Where(m => m.Role == "user" && m.Content.StartsWith("These calls failed", StringComparison.Ordinal)).ToList();
            Check(run is { End: CodingAgentEnd.Answered, FailureNudges: 1 } && sent is [{ } note] && note.Content.StartsWith(Reminder, StringComparison.Ordinal)
                  && note.Content.EndsWith("Fix them, or say plainly what was not done.", StringComparison.Ordinal) && run.ToJson()["failure_nudges"]?.GetValue<int>() == 1,
                $"a failed edit left at the answer is reminded once ({run.End}, {run.FailureNudges}: {string.Join(" | ", sent.Select(m => m.Content))})");

            // One reminder a run: the model fails again and answers; the run ends.
            var stubborn = FakeChatModel.Script(read, FakeChatModel.ToolCall("edit_file", Guess()), FakeChatModel.Answer("Done."),
                FakeChatModel.ToolCall("edit_file", Guess()), FakeChatModel.Answer("Done."));
            var again = new CodingAgent(stubborn).RunAsync("Print 42", new CodingTools(root)).GetAwaiter().GetResult();
            Check(again is { End: CodingAgentEnd.Answered, FailureNudges: 1, Rounds: 5 }, $"at most one reminder a run by default ({again.End}, {again.FailureNudges}, {again.Rounds} rounds)");

            // A later successful change on the path makes the failure good (by write_file too): no reminder.
            var fixedUp = FakeChatModel.Script(read, FakeChatModel.ToolCall("edit_file", Guess()),
                FakeChatModel.ToolCall("edit_file", new JsonObject { ["path"] = "./Program.cs", ["old_text"] = "Console.WriteLine(\"Hello World\");", ["new_text"] = "Console.WriteLine(42);" }),
                FakeChatModel.Answer("Done."));
            var good = new CodingAgent(fixedUp).RunAsync("Print 42", new CodingTools(root)).GetAwaiter().GetResult();
            Check(good is { End: CodingAgentEnd.Answered, FailureNudges: 0, ToolErrors: 1 }, $"a failure made good later is not reminded ({good.End}, {good.FailureNudges})");

            // A failed read (not a changing tool) is not reminded; FailedCallNudges 0 sends none.
            var missing = FakeChatModel.Script(FakeChatModel.ToolCall("read_file", new JsonObject { ["path"] = "Missing.cs" }), FakeChatModel.Answer("There is no Missing.cs."));
            Check(new CodingAgent(missing).RunAsync("Read Missing.cs", new CodingTools(root)).GetAwaiter().GetResult() is { End: CodingAgentEnd.Answered, FailureNudges: 0 },
                "a failed read is not a failed change");
            File.WriteAllBytes(program, BenchProgram);
            var quiet = FakeChatModel.Script(read, FakeChatModel.ToolCall("edit_file", Guess()), FakeChatModel.Answer("Done."));
            Check(new CodingAgent(quiet, new AgentOptions { FailedCallNudges = 0 }).RunAsync("Print 42", new CodingTools(root)).GetAwaiter().GetResult()
                    is { End: CodingAgentEnd.Answered, FailureNudges: 0 },
                "FailedCallNudges 0: the answer ends the run");
        });
    }
}
