using System.Text.Json.Nodes;
using Idrak;
using Idrak.Generation;
using Idrak.LanguageModels;

// Coding tools: the workspace sandbox, reading, searching, exact edits, writes and allowlisted commands.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] CodingToolsGroup =
    [
        ("coding tools: list, read, search, edit and write stay inside the workspace and explain mistakes", CodingToolsFiles),
        ("coding tools: read_file and search number and cut lines as the string-splitting reference (CRLF, lone CR, final newline, long lines); web text collapses white space as the regex did", d => { if (d == Device.Cpu) CodingToolsLinesMatchReference(); }),
        ("coding tools: run_command starts allowlisted programs without a shell and trims long output", CodingToolsCommands),
        ("coding agent: transcripts written as OpenAI chat JSON read back for fine-tuning", AgentTranscriptJson),
        ("coding agent: a task is copied, run, verified by its commands and recorded", AgentRunsTask),
        ("coding agent: a call repeated right after itself is not run again, and the model is told to use its result", AgentSkipsRepeatedCalls),
    ];

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
            var run = agent.RunAsync(new AgentTask("typed", "What does a.txt say?"), new CodingTools(root)).GetAwaiter().GetResult();
            Check(run is { Outcome: AgentOutcome.Passed, Rounds: 6, ToolCalls: 5, ToolErrors: 2 }, $"counts {run.Outcome} {run.Rounds} {run.ToolCalls} {run.ToolErrors}");
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
            var suite = AgentTask.LoadSuite(Path.Combine(root, "suite"));
            Check(suite is [{ Id: "csharp/fix-greeting", Language: "csharp", Verify: ["git grep -q Hello -- Greeter.cs"] }], $"suite: {string.Join(",", suite.Select(t => t.Id))}");
            var fake = FakeChatModel.Script(
                FakeChatModel.ToolCall("read_file", new JsonObject { ["path"] = "Greeter.cs" }),
                FakeChatModel.ToolCall("edit_file", new JsonObject { ["path"] = "Greeter.cs", ["old_text"] = "\"Bye\"", ["new_text"] = "\"Hello\"" }),
                FakeChatModel.ToolCall("read_file", new JsonObject { ["path"] = "Missing.cs" }),
                FakeChatModel.Answer("Greet now returns Hello.", thinking: "Done."));
            var tools = new List<string>();
            var agent = new CodingAgent(fake, new AgentOptions { Think = true }) { OnToolResult = r => tools.Add(r.Call.Name) };
            var run = agent.RunAsync(suite[0], Path.Combine(root, "run1")).GetAwaiter().GetResult();
            Check(run.Outcome == AgentOutcome.Passed, $"passed: {run.Outcome} {run.VerifyOutput}");
            Check(run is { Rounds: 4, ToolCalls: 3, ToolErrors: 1 } && tools.Count == 3, $"counts {run.Rounds} {run.ToolCalls} {run.ToolErrors}");
            Check(!Directory.Exists(Path.Combine(root, "run1", "bin")) && File.Exists(Path.Combine(root, "run1", "GreeterTests.cs")), "build output skipped, hidden files copied");
            Check(fake.Requests[0].Messages is [{ Role: "system" }, { Role: "user", Content: "Make Greet say Hello." }] && fake.Requests[0].Tools!.Count == 6, "first request");
            Check(fake.Requests[1].Messages[^1] is { Role: "tool", ToolName: "read_file" } m && m.Content.Contains("\"Bye\"", StringComparison.Ordinal), "tool results reach the model");

            var line = run.ToJson();
            Check((string?)line["outcome"] == "Passed" && (string?)line["task"] == "csharp/fix-greeting" && (int?)line["rounds"] == 4, "run fields");
            var transcript = ChatTranscript.FromJson(JsonNode.Parse(line.ToJsonString())!.AsObject());
            Check(transcript.Messages.Count == 9 && transcript.Tools.Count == 6 && transcript.Messages[^1].Content == "Greet now returns Hello.", "transcript reads back");

            var lazy = FakeChatModel.Script(FakeChatModel.Answer("Done!"));
            var failed = new CodingAgent(lazy).RunAsync(suite[0], Path.Combine(root, "run2")).GetAwaiter().GetResult();
            Check(failed.Outcome == AgentOutcome.Failed && failed.VerifyOutput.Contains("exit code 1", StringComparison.Ordinal), $"verification fails: {failed.VerifyOutput}");

            var looping = FakeChatModel.Script([.. Enumerable.Repeat(FakeChatModel.ToolCall("list_files", []), 5)]);
            var budget = new CodingAgent(looping, new AgentOptions { MaxRounds = 3 }).RunAsync(suite[0], Path.Combine(root, "run3")).GetAwaiter().GetResult();
            Check(budget.Outcome == AgentOutcome.OutOfBudget && budget.Rounds == 3, "round limit");
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

            // Line endings: the model writes \n; a CRLF file keeps CRLF.
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
