// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Abstraction.Plugins;
using Idrak.Abstraction.Reinforcement;
using Idrak.Abstraction.Testing;
using Idrak.Reinforcement.Plugins;
using Idrak.Reinforcement.Plugins.Coding;
using Idrak.Reinforcement.Plugins.CSharp;
using Idrak.Text.Abstractions;
using Idrak.Text.Plugins.CSharp;

// Plan 27, R4b part 2: Idrak.Reinforcement.Plugins.CSharp, agentic coding on C# projects: the csharp environment (the
// coding environment with dotnet's defaults) on a tiny solution written here (a library and an xunit test project, built
// and tested by the real dotnet), its rewards from the build's logs and the tests' TRX results, the shared package folder,
// no build nodes left running, the protected files, and the plug-in's registration over the coding plug-in and the C#
// language. The model is a script of replies: no download but the test project's NuGet packages (once, into a kept folder).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] CSharpCodingGroup =
    [
        ("plugins: csharp: CSharpCodingPlugin is the plug-in \"csharp\" of ReinforcementPlugins.All; Register twice registers the csharp environment and its two rewards once, and the coding plug-in and the C# language when missing; Unregister takes back only what it registered (coding stays as found); the rewards pass the reward suite and read the TRX results and the build's logs", d => { if (d == Device.Cpu) CSharpPluginRegisters(); }),
        ("plugins: csharp: on a C# solution (a library, an xunit test project): a task with no commands gets dotnet's defaults; a partly right answer gets the share of tests it passes; a build with errors gets csharp-build's penalty per error; edits to the test project and the build's shared files are refused when asked; two episodes in parallel share the package folder and leave no build nodes running", d => { if (d == Device.Cpu) CSharpEpisodes(); }),
    ];

    private static readonly ChatMessage CSharpFixAdd = new("assistant", "Fixing Add.",
        ToolCalls: [new ToolCall("edit_file", new JsonObject { ["path"] = "Lib/Calc.cs", ["old_text"] = "Add(int a, int b) => a - b", ["new_text"] = "Add(int a, int b) => a + b" })]);
    private static readonly ChatMessage CSharpBreak = new("assistant", "",
        ToolCalls: [new ToolCall("write_file", new JsonObject { ["path"] = "Lib/Broken.cs", ["content"] = "namespace Lib;\nstatic class Broken\n{\n    static int X() => y;\n    static int Y() => z;\n}\n" })]);
    private static readonly ChatMessage CSharpEditTests = new("assistant", "",
        ToolCalls:
        [
            new ToolCall("edit_file", new JsonObject { ["path"] = "Lib.Tests/Lib.Tests.csproj", ["old_text"] = "<IsPackable>false</IsPackable>", ["new_text"] = "<IsPackable>false</IsPackable><!-- quiet -->" }),
            new ToolCall("write_file", new JsonObject { ["path"] = "Directory.Build.props", ["content"] = "<Project />\n" }),
        ]);
    private static readonly ChatMessage CSharpDone = new("assistant", "Done.");

    // The kept package folder of these tests: the test project's packages are downloaded once per machine.
    private static string CSharpTestPackages => Path.Combine(Path.GetTempPath(), "idrak-csharp-test-packages");

    // A suite of one task: Calc (Add and Sub wrong, Twice right) and four hidden xunit tests; no setup or verification of its own.
    private static string WriteCSharpSolutionSuite(string root)
    {
        string task = Path.Combine(root, "suite", "calc");
        string workspace = Path.Combine(task, "workspace");
        Directory.CreateDirectory(Path.Combine(workspace, "Lib"));
        Directory.CreateDirectory(Path.Combine(workspace, "Lib.Tests"));
        Directory.CreateDirectory(Path.Combine(task, "verify", "Lib.Tests"));
        File.WriteAllText(Path.Combine(task, "task.json"), """{"prompt": "Fix Calc so its tests pass.", "language": "csharp"}""");
        File.WriteAllText(Path.Combine(workspace, "Calc.slnx"), "<Solution>\n  <Project Path=\"Lib/Lib.csproj\" />\n  <Project Path=\"Lib.Tests/Lib.Tests.csproj\" />\n</Solution>\n");
        File.WriteAllText(Path.Combine(workspace, "Lib", "Lib.csproj"),
            "<Project Sdk=\"Microsoft.NET.Sdk\">\n  <PropertyGroup>\n    <TargetFramework>net10.0</TargetFramework>\n  </PropertyGroup>\n</Project>\n");
        File.WriteAllText(Path.Combine(workspace, "Lib", "Calc.cs"),
            "namespace Lib;\n\npublic static class Calc\n{\n    public static int Add(int a, int b) => a - b;\n\n    public static int Sub(int a, int b) => a + b;\n\n    public static int Twice(int a) => a * 2;\n}\n");
        File.WriteAllText(Path.Combine(workspace, "Lib.Tests", "Lib.Tests.csproj"), """
            <Project Sdk="Microsoft.NET.Sdk">
              <PropertyGroup>
                <TargetFramework>net10.0</TargetFramework>
                <IsPackable>false</IsPackable>
              </PropertyGroup>
              <ItemGroup>
                <PackageReference Include="Microsoft.NET.Test.Sdk" Version="17.12.0" />
                <PackageReference Include="xunit" Version="2.9.2" />
                <PackageReference Include="xunit.runner.visualstudio" Version="2.8.2" />
              </ItemGroup>
              <ItemGroup>
                <ProjectReference Include="../Lib/Lib.csproj" />
              </ItemGroup>
            </Project>
            """);
        File.WriteAllText(Path.Combine(workspace, "Lib.Tests", "Placeholder.cs"), "namespace Lib.Tests;\n\ninternal static class Placeholder\n{\n}\n");
        File.WriteAllText(Path.Combine(task, "verify", "Lib.Tests", "CalcTests.cs"), """
            using Xunit;

            namespace Lib.Tests;

            public class CalcTests
            {
                [Fact]
                public void Adds() => Assert.Equal(5, Calc.Add(2, 3));

                [Fact]
                public void Subtracts() => Assert.Equal(2, Calc.Sub(5, 3));

                [Fact]
                public void Doubles() => Assert.Equal(4, Calc.Twice(2));

                [Fact]
                public void AddsZero() => Assert.Equal(0, Calc.Add(0, 0));
            }
            """);
        return Path.Combine(root, "suite");
    }

    // The processes (other than the compiler server) whose variables hold `marker` as NUGET_PACKAGES: on Linux from /proc
    // (elsewhere none are found, and the check is the variable reaching the commands).
    private static List<string> CSharpLeftovers(string marker)
    {
        var found = new List<string>();
        if (!OperatingSystem.IsLinux())
        {
            return found;
        }

        foreach (var folder in Directory.GetDirectories("/proc"))
        {
            if (!int.TryParse(Path.GetFileName(folder), out int pid) || pid == Environment.ProcessId)
            {
                continue;
            }

            try
            {
                if (!File.ReadAllText(Path.Combine(folder, "environ")).Split('\0').Contains("NUGET_PACKAGES=" + marker))
                {
                    continue;
                }

                string command = File.ReadAllText(Path.Combine(folder, "cmdline")).Replace('\0', ' ');
                if (!command.Contains("VBCSCompiler", StringComparison.OrdinalIgnoreCase))
                {
                    found.Add($"{pid}: {command}");
                }
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
            }
        }

        return found;
    }

    private static Dictionary<string, string> CSharpInfoOf(int total, int passed, bool built, int errors, int warnings) => new()
    {
        [CSharpInfo.TestsTotal] = $"{total}", [CSharpInfo.TestsPassed] = $"{passed}", [CSharpInfo.BuildPassed] = built ? "true" : "false",
        [CSharpInfo.BuildErrors] = $"{errors}", [CSharpInfo.BuildWarnings] = $"{warnings}",
    };

    private static void CSharpPluginRegisters()
    {
        var plugin = CSharpCodingPlugin.Plugin;
        Check(plugin.Name == "csharp" && ReinforcementPlugins.All.Contains(plugin) && plugin.GetType().Namespace == "Idrak.Reinforcement.Plugins.CSharp",
            "the plug-in \"csharp\" of ReinforcementPlugins.All, in Idrak.Reinforcement.Plugins.CSharp");
        bool codingWas = CodingPlugin.IsRegistered, csharpWas = CSharpCodingPlugin.IsRegistered, languageWas = CSharpLanguagePlugin.IsRegistered;
        string State() => $"{string.Join(",", EnvironmentKinds.Names)} | {string.Join(",", RewardFunctions.Names)} | {string.Join(",", CodingLanguages.Names)} | {string.Join(",", IdrakPlugins.Names)}";
        try
        {
            IdrakPlugins.Unregister(plugin);
            IdrakPlugins.Unregister(CodingPlugin.Plugin);
            IdrakPlugins.Unregister(CSharpLanguagePlugin.Plugin);
            string bare = State();
            CSharpCodingPlugin.Register();
            CSharpCodingPlugin.Register();
            Check(CodingPlugin.IsRegistered && CSharpLanguagePlugin.IsRegistered, "registering csharp registers the coding plug-in and the C# language it needs");
            Check(EnvironmentKinds.Names.Count(n => n == "csharp") == 1 && EnvironmentKinds.Get("csharp") is CodingEnvironmentKind { Variant: CSharpEnvironmentVariant },
                "the csharp environment once: the coding environment with the C# variant");
            Check(new[] { CSharpRewards.Tests, CSharpRewards.Build }.All(n => RewardFunctions.Names.Count(r => r == n) == 1 && RewardFunctions.Origin(n) == "Idrak.Reinforcement.Plugins"),
                "the two rewards once, from Idrak.Reinforcement.Plugins");
            CSharpCodingPlugin.Unregister();
            Check(State() == bare && !CodingPlugin.IsRegistered && !CSharpLanguagePlugin.IsRegistered, $"Unregister takes back what it registered:\n{bare}\nnow:\n{State()}");

            CodingPlugin.Register();
            CSharpLanguagePlugin.Register();
            string withCoding = State();
            CSharpCodingPlugin.Register();
            CSharpCodingPlugin.Unregister();
            Check(State() == withCoding && CodingPlugin.IsRegistered && EnvironmentKinds.Get("coding") is CodingEnvironmentKind { Variant: null },
                $"unregistering csharp leaves coding as it found it:\n{withCoding}\nnow:\n{State()}");
            CSharpCodingPlugin.Register();

            // The rewards: the reward suite, and their scores from the environment's info.
            RewardInput Input(int total, int passed, bool built, int errors, int warnings) => new("Fix it.", "Done.", null, CSharpInfoOf(total, passed, built, errors, warnings));
            RewardInput[] samples = [Input(10, 7, true, 0, 2), Input(4, 4, true, 0, 0), Input(0, 0, false, 3, 1), Input(5, 0, true, 0, 0)];
            foreach (var reward in CSharpRewards.All)
            {
                Conformance.CheckRewardFunction(reward, samples).ThrowIfFailed();
            }

            float[] Score(string name, IReadOnlyDictionary<string, string>? options = null) => RewardFunctions.Get(name).ScoreAsync(samples, options).AsTask().GetAwaiter().GetResult();
            AssertClose([0.7f, 1f, 0f, 0f], Score(CSharpRewards.Tests), 1e-6f, "csharp-tests: the share passed (seven of ten is 0.7), none run is 0");
            AssertClose([1f, 1f, -0.3f, 1f], Score(CSharpRewards.Build), 1e-6f, "csharp-build: 1 built, less 0.1 an error");
            AssertClose([0.9f, 1f, -0.35f, 1f], Score(CSharpRewards.Build, new Dictionary<string, string> { ["warning"] = "0.05" }), 1e-6f, "csharp-build with a weight per warning");
            AssertClose([1f, 1f, -0.2f, 1f], Score(CSharpRewards.Build, new Dictionary<string, string> { ["floor"] = "-0.2" }), 1e-6f, "csharp-build at least its floor");
            ThrowsNaming<ArgumentException>(() => Score(CSharpRewards.Build, new Dictionary<string, string> { ["weight"] = "1" }), "an option csharp-build does not take", "'weight'", "error, warning, floor");
            ThrowsNaming<ArgumentException>(() => RewardFunctions.Get(CSharpRewards.Tests).ScoreAsync([new RewardInput("p", "a")]).AsTask().GetAwaiter().GetResult(),
                "an answer without the csharp environment's info", CSharpInfo.TestsTotal, "csharp environment");

            // The machine-readable output: TRX counters summed over the files, MSBuild's error and warning lines counted once.
            string folder = Path.Combine(Path.GetTempPath(), "idrak-csharp-results-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(Path.Combine(folder, "results", "more"));
            try
            {
                const string Trx = "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<TestRun xmlns=\"http://microsoft.com/schemas/VisualStudio/TeamTest/2010\"><ResultSummary outcome=\"Failed\">"
                                   + "<Counters total=\"{0}\" executed=\"{0}\" passed=\"{1}\" failed=\"1\" notExecuted=\"0\" /></ResultSummary></TestRun>";
                File.WriteAllText(Path.Combine(folder, "results", "a.trx"), string.Format(System.Globalization.CultureInfo.InvariantCulture, Trx, 6, 5));
                File.WriteAllText(Path.Combine(folder, "results", "more", "b.trx"), string.Format(System.Globalization.CultureInfo.InvariantCulture, Trx, 4, 2));
                File.WriteAllText(Path.Combine(folder, "results", "broken.trx"), "<TestRun><Counters total=");
                Check(CSharpResults.Tests(Path.Combine(folder, "results")) == (10, 7), $"TRX: {CSharpResults.Tests(Path.Combine(folder, "results"))}");
                Check(CSharpResults.Tests(Path.Combine(folder, "none")) == (0, 0), "no results: none ran");
                File.WriteAllLines(Path.Combine(folder, "errors.log"),
                [
                    "   2:2>/src/Lib/Broken.cs(4,23): error CS0103: The name 'y' does not exist in the current context [/src/Lib/Lib.csproj]",
                    "/src/Lib/Broken.cs(4,23): error CS0103: The name 'y' does not exist in the current context [/src/Lib/Lib.csproj]",
                    "   2:2>/src/Lib/Broken.cs(5,23): error CS0103: The name 'z' does not exist in the current context [/src/Lib/Lib.csproj]",
                    "MSBUILD : error MSB1009: Project file does not exist.",
                    "/src/Lib/A.cs(1,1): warning CS0219: an error: in a warning's text",
                ]);
                Check(CSharpResults.Diagnostics(Path.Combine(folder, "errors.log"), "error") == 3 && CSharpResults.Diagnostics(Path.Combine(folder, "errors.log"), "warning") == 1,
                    "errors counted once each (a node's prefix aside), a warning that says error is a warning");
                Check(CSharpResults.Diagnostics(Path.Combine(folder, "missing.log"), "error") == 0, "no log: none");
            }
            finally
            {
                Directory.Delete(folder, recursive: true);
            }

            var variant = new CSharpEnvironmentVariant();
            var variables = variant.Variables(new Dictionary<string, string>());
            Check(variables.Count == 1 && variables["MSBUILDDISABLENODEREUSE"] == "1" || Environment.GetEnvironmentVariable(CSharpEnvironmentVariant.PackagesVariable) is not null,
                "by default: build nodes end with their build, the machine's package folder, the shared compiler server");
            Check(variant.Variables(new Dictionary<string, string> { ["node-reuse"] = "true", ["build-server"] = "false" }) is { } other && !other.ContainsKey("MSBUILDDISABLENODEREUSE")
                  && other["UseSharedCompilation"] == "false", "node-reuse and build-server as settings");
            ThrowsNaming<ArgumentException>(() => variant.Check(new Dictionary<string, string> { ["node-reuse"] = "sometimes" }), "a bad flag", "node-reuse=sometimes");
            Check(variant.Defaults()["reward"] == CSharpRewards.Tests && variant.Defaults()["languages"] == "csharp"
                  && (Environment.GetEnvironmentVariable(CSharpEnvironmentVariant.ProtectVariable) is not null || variant.Defaults()["protect"] == string.Join(",", CSharpEnvironmentVariant.DefaultProtect)),
                "its defaults for the coding options");
            WithEnvironment(new Dictionary<string, string?> { [CSharpEnvironmentVariant.ProtectVariable] = "a/**; b.props" }, () =>
                Check(CSharpEnvironmentVariant.Protect.SequenceEqual(["a/**", "b.props"]), "IDRAK_CSHARP_PROTECT replaces the protected globs"));
        }
        finally
        {
            IdrakPlugins.Unregister(plugin);
            if (codingWas)
            {
                CodingPlugin.Register();
            }
            else
            {
                IdrakPlugins.Unregister(CodingPlugin.Plugin);
            }

            if (languageWas)
            {
                CSharpLanguagePlugin.Register();
            }

            WithoutLanguages(() => { });                         // the languages back in TextPlugins.All's order
            if (csharpWas)
            {
                CSharpCodingPlugin.Register();
            }
        }
    }

    private static void CSharpEpisodes()
    {
        bool codingWas = CodingPlugin.IsRegistered, csharpWas = CSharpCodingPlugin.IsRegistered;
        CSharpCodingPlugin.Register();
        string root = Path.Combine(Path.GetTempPath(), "idrak-csharp-test-" + Guid.NewGuid().ToString("N"));
        string suite = WriteCSharpSolutionSuite(root), work = Path.Combine(root, "work"), packages = CSharpTestPackages;
        var kind = EnvironmentKinds.Get("csharp");
        var options = new Dictionary<string, string>
        {
            ["suite"] = suite, ["system"] = "Fix the code.", ["rounds"] = "6", ["reward"] = "csharp-tests:1,csharp-build:1", ["work"] = work, ["workspaces"] = "2", ["packages"] = packages,
        };
        var first = (CodingEnvironment)kind.Create(options);
        var second = (CodingEnvironment)kind.Create(options);
        CodingEnvironment? refusing = null;
        try
        {
            // A task with no commands: dotnet's restore, then the build and the tests, writing their results to files.
            var observation = first.Reset(0);
            var task = first.Task!;
            Check(task.Setup is ["dotnet restore"] && task.Verify is [var build, var test]
                  && build == "dotnet build --no-restore -fl1 -flp1:errorsonly -fl2 -flp2:warningsonly" && test == "dotnet test --no-build --logger trx --results-directory .idrak-results",
                $"the C# defaults: setup {string.Join(" | ", task.Setup ?? [])}, verify {string.Join(" | ", task.Verify ?? [])}");
            Check(observation.Tools!.Single(t => t.Name == "run_command").Description!.Contains("Allowed programs: dotnet, git", StringComparison.Ordinal),
                "the episode's tools allow the C# language's programs");
            Check(first.BrokenTasks.Count == 0, $"the setup restored: {string.Join("; ", first.BrokenTasks.Values)}");
            string assets = Path.Combine(work, "prepared", "000", "Lib.Tests", "obj", "project.assets.json");
            Check(File.Exists(assets) && (File.ReadAllText(assets).Contains(packages.Replace('\\', '/').TrimEnd('/'), StringComparison.OrdinalIgnoreCase)
                                          || File.ReadAllText(assets).Contains(packages.Replace("\\", "\\\\").TrimEnd('\\'), StringComparison.OrdinalIgnoreCase)),
                "the restore used the shared package folder");
            Check(Directory.Exists(Path.Combine(packages, "xunit")), "the package folder holds the test project's packages");

            // Partly right: Add fixed, Sub not: three of the four hidden tests pass.
            var partly = first.StepAsync(CSharpFixAdd).AsTask().GetAwaiter().GetResult();
            var end = first.StepAsync(CSharpDone).AsTask().GetAwaiter().GetResult();
            Check(!partly.Done && end.Done, "two steps");
            Check(end.Info![CSharpInfo.TestsTotal] == "4" && end.Info[CSharpInfo.TestsPassed] == "3" && end.Info[CSharpInfo.BuildPassed] == "true" && end.Info[CSharpInfo.BuildErrors] == "0"
                  && end.Info[CodingInfo.VerifyPassedCommands] == "1" && end.Info[CodingInfo.VerifyCommands] == "2",
                $"a partly right answer: {ShowInfo(end.Info)}\n{first.Verification?.Output}");
            AssertClose([0.75f + 1f], [end.Reward], 1e-6f, "the reward: the share of the tests (0.75) and the build (1)");

            // A build with errors: no tests ran; csharp-build's penalty per error.
            var broken = PlayCoding(first, 1, [CSharpBreak, CSharpDone])[^1];
            Check(broken.Info![CSharpInfo.BuildPassed] == "false" && broken.Info[CSharpInfo.BuildErrors] == "2" && broken.Info[CSharpInfo.TestsTotal] == "0",
                $"a build with two errors: {ShowInfo(broken.Info)}");
            AssertClose([-0.2f], [broken.Reward], 1e-6f, "the reward: no tests, 0 less 0.1 for each of the two errors");

            // Edits to the test project and to the build's shared files: reported, and refused when asked.
            refusing = (CodingEnvironment)kind.Create(new Dictionary<string, string>(options) { ["refuse-test-edits"] = "true", ["work"] = Path.Combine(root, "work-refusing") });
            var refused = PlayCoding(refusing, 0, [CSharpEditTests, CSharpDone])[^1];
            Check(refused.Info![CodingInfo.Refused] == "test-edits" && refused.Info[CodingInfo.TestEdits] == "Directory.Build.props,Lib.Tests/Lib.Tests.csproj"
                  && refused.Info[CSharpInfo.TestsTotal] == "0", $"the protected edits refused: {ShowInfo(refused.Info)}");

            // Two episodes at once (the collector's way): the same rewards, one package folder, no build node left running.
            var episodes = new[] { first, second }.Select(environment => Task.Run(() => PlayCoding(environment, 0, [CSharpFixAdd, CSharpDone])[^1])).ToArray();
            Task.WaitAll(episodes);
            Check(episodes.All(e => Math.Abs(e.Result.Reward - 1.75f) < 1e-6f), $"both episodes: {string.Join(", ", episodes.Select(e => e.Result.Reward))}");
            var leftovers = CSharpLeftovers(packages);
            for (var watch = Stopwatch.StartNew(); leftovers.Count > 0 && watch.Elapsed < TimeSpan.FromSeconds(10); leftovers = CSharpLeftovers(packages))
            {
                Thread.Sleep(200);
            }

            Check(leftovers.Count == 0, $"no build node of the episodes left running:\n{string.Join("\n", leftovers)}");
        }
        finally
        {
            first.Dispose();
            second.Dispose();
            refusing?.Dispose();
            CodingWorkspace.Delete(root);
            if (!csharpWas)
            {
                CSharpCodingPlugin.Unregister();
            }

            if (codingWas && !CodingPlugin.IsRegistered)
            {
                CodingPlugin.Register();
            }
        }
    }
}
