// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json.Nodes;
using Idrak.Cli;
using Idrak.Data;
using Idrak.Data.Abstractions;
using Idrak.Data.Code;
using Idrak.Reinforcement.Plugins.Coding;
using Idrak.Text.Plugins.CSharp;

// Plan 36, G1: idrak data code on the fixture repository (data/code-repo: a library and a test program whose two named
// tests reach Area.Rectangle and Words.CountWords): file-text rows, method-body and test-subject drafts, proven by a
// scripted verifier or written unproven, the manifest, and the command.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] DataCodeGroup =
    [
        ("data code: the fixture repository gives a file-text row for every source file ({text, path, language}), method-body and test-subject drafts for the methods its two tests reach (each body removed byte for byte, its signature and documentation kept, the check the tests by name), none for a method no test reaches, a name defined twice, a body too small or one already unwritten, and a manifest with the counts", d => { CpuOnly(d); CodeDataFixture(); }),
        ("data code: a registered verifier proves the drafts: a proven one becomes a task folder, a rejected one is counted by reason with its proof's output, and --max-tasks stops the run with what it kept", d => { CpuOnly(d); CodeDataProven(); }),
        ("data code: with no verifier registered the drafts are written unproven (drafts/, marked), never as tasks; an unknown family or verifier is refused with \"not registered\", a row format that cannot be written naming why", d => { CpuOnly(d); CodeDataUnproven(); }),
        ("data code: the command writes the fixture's rows, drafts and manifest with every setting a flag (the manifest records them), and --list names the registered families and verifiers", d => { CpuOnly(d); CodeDataCommand(); }),
    ];

    private const string AreaTest = "area: a rectangle's area is its width times its height", WordsTest = "words: runs of spaces and punctuation separate words";

    private static readonly string[] CodeFamilies = [FileTextFamily.FamilyName, CSharpTaskFamilies.MethodBodyName, CSharpTaskFamilies.TestSubjectName];

    private static CodeExtraction ExtractCode(string output, CodeExtractorOptions options)
    {
        CSharpLanguagePlugin.Register();                                    // the C# families (once: a second call changes nothing)
        return new CodeExtractor(options).RunAsync(new CodeSource(TestData("code-repo")), output).GetAwaiter().GetResult();
    }

    // The task folders under a kind ("tasks", "drafts") of a family, by their task.json.
    private static List<(string Folder, JsonObject Task)> TaskFolders(string output, string kind, string family)
    {
        string folder = Path.Combine(output, kind, family);
        return !Directory.Exists(folder) ? []
            : [.. Directory.GetDirectories(folder).Order(StringComparer.Ordinal).Select(f => (f, (JsonObject)JsonNode.Parse(File.ReadAllText(Path.Combine(f, "task.json")))!))];
    }

    private static void CodeDataFixture()
    {
        string output = TempFolder();
        try
        {
            string repository = Path.GetFullPath(TestData("code-repo"));
            var result = ExtractCode(output, new CodeExtractorOptions { Families = CodeFamilies, Verifier = TaskVerifiers.None, Concurrency = 1 });

            // Rows: every file whose extension names a language (the README has none), its text as stored.
            string[] expectedPaths = ["src/Shapes/Area.cs", "src/Shapes/Shapes.csproj", "src/Shapes/Words.cs", "tests/Shapes.Tests/Program.cs", "tests/Shapes.Tests/Shapes.Tests.csproj"];
            var rows = File.ReadAllLines(Path.Combine(output, "rows", "file-text.jsonl")).Select(l => (JsonObject)JsonNode.Parse(l)!).ToList();
            Check(rows.Select(r => (string)r["path"]!).Order(StringComparer.Ordinal).SequenceEqual(expectedPaths) && result.Rows == 5,
                $"a row per source file: {string.Join(", ", rows.Select(r => (string)r["path"]!))}");
            var area = rows.Single(r => (string)r["path"]! == "src/Shapes/Area.cs");
            Check((string)area["language"]! == "csharp" && (string)area["text"]! == File.ReadAllText(Path.Combine(repository, "src/Shapes/Area.cs")),
                "a row holds the file's text and language");

            // Drafts: written unproven, one per reached method and one per named test.
            var bodies = TaskFolders(output, "drafts", "method-body");
            var subjects = TaskFolders(output, "drafts", "test-subject");
            Check(bodies.Select(b => (string)b.Task["id"]!).SequenceEqual(["src/Shapes/Area.cs#Area.Rectangle", "src/Shapes/Words.cs#Words.CountWords"]),
                $"method-body drafts the two reached methods (not Circle, unreached; not Describe, twice; not Zero, small; not Triangle, unwritten): {string.Join(", ", bodies.Select(b => (string)b.Task["id"]!))}");
            Check(subjects.Select(s => (string)s.Task["id"]!).SequenceEqual([AreaTest, WordsTest]), $"test-subject drafts each named test: {string.Join(", ", subjects.Select(s => (string)s.Task["id"]!))}");
            Check(result.Unproven == 4 && result.Tasks == 0 && !Directory.Exists(Path.Combine(output, "tasks")) && bodies.All(b => (bool?)b.Task["proven"] == false),
                $"without a verifier every draft is unproven and none is a task: {result.Unproven} unproven, {result.Tasks} tasks");

            var (folder, task) = bodies[0];
            Check((string)task["base"]! == repository && (string)task["language"]! == "csharp"
                  && task["verify"]!.AsArray().Select(v => (string)v!).SequenceEqual(["dotnet run --project tests/Shapes.Tests/Shapes.Tests.csproj"])
                  && (string)task["environment"]!["IDRAK_FILTER"]! == AreaTest && (string)task["environment"]!["IDRAK_DEVICES"]! == "cpu"
                  && (string)task["provenance"]!["test"]! == AreaTest && (string)task["provenance"]!["group"]! == "area",
                $"the check is the test program run with the reaching test's name: {task.ToJsonString()}");
            byte[] original = File.ReadAllBytes(Path.Combine(repository, "src/Shapes/Area.cs"));
            byte[] workspace = File.ReadAllBytes(Path.Combine(folder, "workspace", "src/Shapes/Area.cs"));
            Check(File.ReadAllBytes(Path.Combine(folder, "solution", "src/Shapes/Area.cs")).AsSpan().SequenceEqual(original), "the solution is the file as it was");
            int prefix = original.AsSpan().CommonPrefixLength(workspace);
            int suffix = 0;
            while (suffix < original.Length - prefix && suffix < workspace.Length - prefix && original[^(suffix + 1)] == workspace[^(suffix + 1)])
            {
                suffix++;
            }

            string removed = Encoding.UTF8.GetString(original, prefix, original.Length - prefix - suffix);
            string inserted = Encoding.UTF8.GetString(workspace, prefix, workspace.Length - prefix - suffix);
            string text = Encoding.UTF8.GetString(workspace);
            Check(removed.Contains("return width * height;", StringComparison.Ordinal) && !removed.Contains("Circle", StringComparison.Ordinal)
                  && inserted.Contains("throw new global::System.NotImplementedException()", StringComparison.Ordinal)
                  && text.Contains("/// <summary>The area of a rectangle: its width times its height.</summary>", StringComparison.Ordinal)
                  && text.Contains("public static double Rectangle(double width, double height)", StringComparison.Ordinal),
                $"only the body changed (its signature and documentation kept): removed '{removed}', inserted '{inserted}'");
            Check(((string)task["prompt"]!).Contains("Area.Rectangle", StringComparison.Ordinal) && ((string)task["prompt"]!).Contains("public static double Rectangle(double width, double height)", StringComparison.Ordinal),
                $"the prompt names the method and its signature: {(string)task["prompt"]!}");

            var words = subjects[1];
            Check((string)words.Task["environment"]!["IDRAK_FILTER"]! == WordsTest
                  && File.ReadAllText(Path.Combine(words.Folder, "workspace", "src/Shapes/Words.cs")).Contains("NotImplementedException", StringComparison.Ordinal)
                  && ((string)words.Task["prompt"]!).Contains(WordsTest, StringComparison.Ordinal),
                "test-subject removes the code the words test reaches through its helper (CountWords, past strings and braces that look like code), checked by that test");

            // The manifest: every family with its counts, the totals and the settings.
            var manifest = (JsonObject)JsonNode.Parse(File.ReadAllText(result.ManifestPath))!;
            var families = manifest["families"]!.AsArray().ToDictionary(f => (string)f!["name"]!, f => (JsonObject)f!);
            Check((long?)families["file-text"]["rows"]!["written"] == 5 && (long?)families["method-body"]["drafts"]!["unproven"] == 2 && (long?)families["test-subject"]["drafts"]!["asked"] == 2
                  && (long?)manifest["totals"]!["unproven"] == 4 && (long?)manifest["files"]!["walked"] == 6 && manifest["verifier"] is null
                  && (string?)manifest["options"]!["verifier"] == "none" && (double?)manifest["seconds"] >= 0,
                $"the manifest counts what each family asked and kept: {manifest.ToJsonString()}");
        }
        finally
        {
            Directory.Delete(output, recursive: true);
        }
    }

    // A verifier of the test: a draft made from Words.cs passes on its workspace (rejected), any other is proven.
    private sealed class ScriptedVerifier : ITaskVerifier
    {
        public string Name => "code-test-verifier";

        public string Summary => "proves every draft but those of Words.cs";

        public Task<TaskProof> ProveAsync(string taskFolder, string workFolder, CancellationToken cancellationToken = default)
        {
            var task = JsonNode.Parse(File.ReadAllText(Path.Combine(taskFolder, "task.json")))!;
            bool words = ((string?)task["provenance"]!["path"])?.EndsWith("Words.cs", StringComparison.Ordinal) == true;
            return Task.FromResult(words ? new TaskProof(false, true, "scripted: the check passed on the workspace") : new TaskProof(true, true, "scripted: proven"));
        }
    }

    private static void CodeDataProven()
    {
        string output = TempFolder();
        TaskVerifiers.Register(new ScriptedVerifier());
        try
        {
            var result = ExtractCode(output, new CodeExtractorOptions { Families = CodeFamilies, Verifier = "code-test-verifier", Concurrency = 1 });
            var tasks = TaskFolders(output, "tasks", "method-body").Concat(TaskFolders(output, "tasks", "test-subject")).ToList();
            Check(result.Tasks == 2 && result.Rejected == 2 && result.Unproven == 0 && tasks.Select(t => (string)t.Task["id"]!).SequenceEqual(["src/Shapes/Area.cs#Area.Rectangle", AreaTest])
                  && tasks.All(t => (bool?)t.Task["proven"] == true) && !Directory.Exists(Path.Combine(output, "drafts")),
                $"the proven drafts are task folders: {result.Tasks} tasks, {result.Rejected} rejected ({string.Join(", ", tasks.Select(t => (string)t.Task["id"]!))})");
            var manifest = result.Manifest;
            string reason = new TaskProof(false, true, "").Reason!;
            Check((string?)manifest["verifier"] == "code-test-verifier"
                  && manifest["families"]!.AsArray().Where(f => (string)f!["name"]! != "file-text").All(f => (long?)f!["drafts"]!["rejected"]![reason] == 1),
                $"each rejection is counted by its reason ('{reason}'): {manifest["families"]!.ToJsonString()}");
            string proof = Path.Combine(output, "rejected", "method-body", CodeExtractor.SafeName("src/Shapes/Words.cs#Words.CountWords") + ".txt");
            Check(File.Exists(proof) && File.ReadAllText(proof).Contains("scripted: the check passed on the workspace", StringComparison.Ordinal), $"the rejected draft's proof is kept: {proof}");

            string limited = Path.Combine(output, "limited");
            var stopped = ExtractCode(limited, new CodeExtractorOptions { Families = CodeFamilies, Verifier = "code-test-verifier", Concurrency = 1, MaxTasks = 1 });
            Check(stopped.Tasks == 1 && stopped.Stopped == "max tasks" && (string?)stopped.Manifest["stopped"] == "max tasks",
                $"--max-tasks 1 keeps one task and says why it stopped: {stopped.Tasks} tasks, stopped {stopped.Stopped}");
        }
        finally
        {
            TaskVerifiers.Unregister("code-test-verifier");
            Directory.Delete(output, recursive: true);
        }
    }

    private static void CodeDataUnproven()
    {
        string output = TempFolder();
        bool coding = CodingPlugin.IsRegistered;
        if (coding)
        {
            CodingPlugin.Unregister();                                       // its verifier: none is registered for this run
        }

        try
        {
            Check(TaskVerifiers.Names.Count == 0 && TaskVerifiers.Choose(null) is null, $"no verifier registered: {string.Join(", ", TaskVerifiers.Names)}");
            var result = ExtractCode(output, new CodeExtractorOptions { Families = CodeFamilies, Concurrency = 1 });
            var drafts = TaskFolders(output, "drafts", "method-body").Concat(TaskFolders(output, "drafts", "test-subject")).ToList();
            Check(result.Unproven == 4 && result.Tasks == 0 && drafts.Count == 4 && drafts.All(d => (bool?)d.Task["proven"] == false)
                  && result.Manifest["verifier"] is null && !Directory.Exists(Path.Combine(output, "tasks")),
                $"no verifier: the drafts are written unproven under drafts/, none under tasks/: {result.Unproven} unproven, {result.Tasks} tasks");
        }
        finally
        {
            if (coding)
            {
                CodingPlugin.Register();
            }
        }

        Check(!coding || TaskVerifiers.Find(CodingTaskVerifier.VerifierName) is not null, "the coding verifier is back with its plug-in");
        try
        {
            ThrowsNaming<NotRegisteredException>(() => ExtractCode(Path.Combine(output, "a"), new CodeExtractorOptions { Families = ["code-nobody"] }),
                "an unknown family", "'code-nobody'", "not registered", "TaskFamilies.Register");
            ThrowsNaming<NotRegisteredException>(() => ExtractCode(Path.Combine(output, "b"), new CodeExtractorOptions { Verifier = "code-nobody-verifier" }),
                "an unknown verifier", "'code-nobody-verifier'", "not registered", "TaskVerifiers.Register");
            ThrowsNaming<NotSupportedException>(() => ExtractCode(Path.Combine(output, "c"), new CodeExtractorOptions { Formats = ["Parquet"] }),
                "Parquet rows", "Parquet", "read only");
        }
        finally
        {
            Directory.Delete(output, recursive: true);
        }
    }

    private static void CodeDataCommand()
    {
        string output = TempFolder();
        try
        {
            using var stdout = new StringWriter();
            using var stderr = new StringWriter();
            string[] args =
            [
                "data", "code", TestData("code-repo"), "-o", output, "--families", string.Join(",", CodeFamilies), "--verifier", "none", "--skip", ".git,bin,obj",
                "--max-file-bytes", "200000", "--dedupe", "0.9", "--seed", "3", "--split-by", "test-group", "--eval-fraction", "0.2", "--eval-groups", "area",
                "--max-tasks", "0", "--max-time", "0", "--formats", "JsonLines", "--concurrency", "1",
            ];
            int code = CommandLine.Run(args, stdout, stderr);
            Check(code == 0 && File.Exists(Path.Combine(output, "manifest.json")), $"idrak data code ran: exit {code}\n{stdout}\n{stderr}");
            var manifest = (JsonObject)JsonNode.Parse(File.ReadAllText(Path.Combine(output, "manifest.json")))!;
            var options = manifest["options"]!;
            Check((long?)options["maxFileBytes"] == 200000 && (double?)options["dedupe"] == 0.9 && (long?)options["seed"] == 3 && (string?)options["splitBy"] == "test-group"
                  && (double?)options["evalFraction"] == 0.2 && options["evalGroups"]!.AsArray().Select(g => (string)g!).SequenceEqual(["area"]) && options["maxTasks"] is null
                  && options["maxTime"] is null && (long?)options["concurrency"] == 1 && (string?)options["verifier"] == "none"
                  && options["skippedFolders"]!.AsArray().Select(f => (string)f!).SequenceEqual([".git", "bin", "obj"]),
                $"every flag reaches the run (the manifest records it): {options.ToJsonString()}");
            Check((long?)manifest["totals"]!["rows"] == 5 && (long?)manifest["totals"]!["unproven"] == 4 && stdout.ToString().Contains("4 unproven", StringComparison.Ordinal),
                $"the command's rows and drafts: {stdout}");

            using var listed = new StringWriter();
            Check(CommandLine.Run(["data", "code", "--list"], listed, TextWriter.Null) == 0 && listed.ToString().Contains("file-text:", StringComparison.Ordinal)
                  && listed.ToString().Contains("method-body:", StringComparison.Ordinal) && listed.ToString().Contains("test-subject:", StringComparison.Ordinal)
                  && listed.ToString().Contains("coding:", StringComparison.Ordinal),
                $"--list names the families and verifiers: {listed}");
        }
        finally
        {
            Directory.Delete(output, recursive: true);
        }
    }
}
