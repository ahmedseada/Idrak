// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Reinforcement.Plugins.Coding;
using Idrak.Text.Abstractions;
using Idrak.Text.Generation;

// Coding tools: the workspace sandbox, reading, searching, exact edits, writes and allowlisted commands.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] CodingToolsGroup =
    [
        ("coding tools: web text collapses white space as the regex did", d => { if (d == Device.Cpu) CodingToolsLinesMatchReference(); }),
        ("coding tools: write policy: allow is the coding tools' default; IDRAK_WRITE_POLICY names the policy and IDRAK_AGENT_COMMANDS the programs", CodingToolsWritePolicy),
        ("coding tools: edit_file on the bench's Program.cs (byte order mark, CRLF): a guessed old_text's error quotes it and the file; read_file's line numbers, trailing white space and the final line break tolerated when the match is unique, said so; the encoding kept; the option off is exact", CodingEditTolerance),
        ("coding agent: an answer with a failed change left (no later success on its path) gets one reminder listing it; a change made good, a failed read or the option at 0 gets none", AgentRemindsFailedChanges),
        ("coding agent: a suite's task folder is read (its id, language and verify commands)", AgentRunsTask),
        ("coding agent: the default prompt asks for the read before an edit", AgentReadsBeforeEditing),
        ("coding agent: a template that renders no tools gets them in the system message with the fenced-json call format the agent reads back; results as user messages; a template with tools renders them as before", ToolsWithoutTemplateSupport),
        ("coding tools: tool arguments written as strings (\"false\" in any case, \"12\", \"2.5\") are read as the schema's boolean, integer and number; anything else is refused naming the argument, the type and the value; the call keeps what was sent", ToolArgumentsCoerced),
        ("coding agent: a template that renders one call at once (a probe, no family names; Llama 3.1's real one when it is in the local store) gets a reply of several calls as replies of one call, each with its result, in order", ToolsOneCallAtOnce),
        ("coding agent: deepseek-coder's real chat template (from the local store) renders no tools, and ChatGenerator's prompt describes every coding tool in its system text", DeepSeekPromptCarriesTools),
        ("coding agent: an edit_file in the same reply as the file's first read_file is refused until its result is seen; a file read before, or written in that reply, is edited; tools called without replies count a read at once", AgentEditsAfterTheReadIsSeen),
        ("coding tools: a missed edit on a short file quotes the whole file and offers write_file; a second miss on the file leads with write_file; a success or a new task starts the count again", CodingEditMissedTwice),
        ("coding agent: the default prompt asks for the user's language; an answer without any tool call in the run gets one reminder (in English), at most one a run, none at 0", AgentRemindsAnswerWithoutTools),
        ("coding agent: a tool call that is not valid JSON (a single-quoted value, ';' for ',') is told with where and why, beside read calls after their results, alone in the call-format correction", AgentToldMalformedCalls),
        ("coding agent: code shown in the final answer with no file changed in the run gets one reminder to write it (instead of the no-tool one), none after a change or at 0", AgentRemindsUnchangedCode),
        ("coding agent: a reply mostly in a script the request does not use (Han against Latin) gets one reminder in English, after the results or instead of ending; not for the request's script, code, few letters or 0", AgentRemindsReplyLanguage),
    ];

    private static void AgentReadsBeforeEditing(Device device)
    {
        _ = device;
        Check(CodingAgent.DefaultSystemPrompt.Contains("Read a file with read_file before you edit it", StringComparison.Ordinal), "the default prompt asks for the read");
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

    // Web text collapses white space as the regex did.
    private static void CodingToolsLinesMatchReference()
    {
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

    private static void CodingToolsWritePolicy(Device device)
    {
        _ = device;
        Check(Environment.GetEnvironmentVariable(WritePolicies.EnvironmentVariable) is not null || new CodingToolOptions().WritePolicy == WritePolicies.Allow,
            "allow is the coding tools' default (IDRAK_WRITE_POLICY unset)");
        WithEnvironment(new Dictionary<string, string?> { [WritePolicies.EnvironmentVariable] = "coding-test-ask", [CodingToolOptions.CommandsEnvironmentVariable] = "dotnet, node" }, () =>
            Check(new CodingToolOptions() is { WritePolicy: "coding-test-ask", Commands: { } commands } && commands.SequenceEqual(["dotnet", "node"]),
                "IDRAK_WRITE_POLICY names the policy and IDRAK_AGENT_COMMANDS the programs"));
    }
}
