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
        ("coding agent: a suite's task folder is read (its id, language and verify commands)", AgentRunsTask),
        ("coding agent: the default prompt asks for the read before an edit", AgentReadsBeforeEditing),
        ("coding tools: tool arguments written as strings (\"false\" in any case, \"12\", \"2.5\") are read as the schema's boolean, integer and number; anything else is refused naming the argument, the type and the value; the call keeps what was sent", ToolArgumentsCoerced),
    ];

    private static void AgentReadsBeforeEditing(Device device)
    {
        _ = device;
        Check(CodingAgent.DefaultSystemPrompt.Contains("Read a file with read_file before you edit it", StringComparison.Ordinal), "the default prompt asks for the read");
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
