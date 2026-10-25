// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Abstraction.Plugins;
using Idrak.Abstraction.Reinforcement;
using Idrak.Abstraction.Testing;
using Idrak.Generation;
using Idrak.Models;
using Idrak.Reinforcement;
using Idrak.Reinforcement.Plugins;
using Idrak.Reinforcement.Plugins.Coding;
using Idrak.Text;
using Idrak.Text.Generation;
using Idrak.Text.Reinforcement;

// Plan 27, R4: the command sandboxes (process, Idrak.Text's; container, the coding plug-in's) and the coding plug-in of
// Idrak.Reinforcement.Plugins: its environment on task suites written here (a tiny C# project with a hidden test built by
// the real dotnet, and a quick suite checked by git grep), its rewards, its episodes in parallel, and R3's policy with a
// teacher term trained on it by R2's trainer. The chat model is a script (or the tests' tiny decoder): no download.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] CodingPluginGroup =
    [
        ("command sandboxes: process is Idrak.Text's in CommandSandboxes; it runs a program without a shell, enforces the programs allowed, its time, output and memory limits (stopping the command), refuses network off, processes and CPUs naming container, and stops on cancellation; the coding tools run every command through the sandbox their options name", d => { if (d == Device.Cpu) ProcessSandboxLimits(); }),
        ("command sandboxes: container refuses when no program is found or no image is set (naming IDRAK_SANDBOX_PROGRAM and IDRAK_SANDBOX_IMAGE), and builds its command line: the workspace mounted, the network off unless allowed, the memory, CPU and process limits, the user, the variables, the image, then the command", d => { if (d == Device.Cpu) ContainerSandboxCommandLine(); }),
        ("plugins: coding: CodingPlugin is the plug-in \"coding\" of ReinforcementPlugins.All; Register twice registers the coding environment, its four rewards and the container sandbox once, from Idrak.Reinforcement.Plugins; Unregister leaves the registries as found; the rewards pass the reward suite", d => { if (d == Device.Cpu) CodingPluginRegisters(); }),
        ("plugins: coding: on a C# suite (a project, a hidden test, a restore) and a scripted model: a seeded episode replays the same; the reward follows verification (solved, not solved, out of rounds); setup runs once per task and is reused; a broken setup is left out and reported; a test edit is overwritten by the hidden tests and reported, and refused when asked", d => { if (d == Device.Cpu) CodingEpisodes(); }),
        ("plugins: coding: the coding environment passes the environment suite; episodes stepped in parallel (R2's collector) give the rewards, info and conversations of one at a time, measured both ways", d => { if (d == Device.Cpu) CodingSuiteAndParallel(); }),
        ("plugins: coding: verification and tool commands are limited by the task's own verification measured once when it is prepared (timeout-scale times it, at least timeout-least, at most 600 and 300 s), so a program that never ends costs seconds; verify-timeout and command-timeout given are used as they are", d => { if (d == Device.Cpu) CodingMeasuredLimits(); }),
        ("plugins: coding: R3's on-policy distillation on the coding environment under R2's trainer (grpo and a teacher term; no coding code of its own): only the assistant's tokens train, the prompts and the tools' results never, in one-turn and multi-turn episodes", CodingDistillation),
    ];

    // The scripted replies: each is made once, so a replay offers the very same messages (their tool-call lists compare by reference).
    private static readonly ChatMessage CodingRead = new("assistant", "Reading.", ToolCalls: [new ToolCall("read_file", new JsonObject { ["path"] = "Greeter.cs" })]);
    private static readonly ChatMessage CodingMissing = new("assistant", "", ToolCalls: [new ToolCall("read_file", new JsonObject { ["path"] = "Missing.cs" })]);
    private static readonly ChatMessage CodingFix = new("assistant", "Fixing.",
        ToolCalls: [new ToolCall("edit_file", new JsonObject { ["path"] = "Greeter.cs", ["old_text"] = "\"Bye\"", ["new_text"] = "\"Hello\"" })]);
    private static readonly ChatMessage CodingTestEdit = new("assistant", "",
        ToolCalls: [new ToolCall("write_file", new JsonObject { ["path"] = "tests/Check.cs", ["content"] = "return 0;\n" })]);
    private static readonly ChatMessage CodingDone = new("assistant", "Done.");

    // A suite of three C# tasks: hello and hi (Greet must say it; the hidden tests/Check.cs exits 0 when it does), whose
    // setup restores the project, and broken, whose setup fails.
    private static string WriteCSharpSuite(string root)
    {
        string suite = Path.Combine(root, "suite");
        const string Project = "<Project Sdk=\"Microsoft.NET.Sdk\">\n  <PropertyGroup>\n    <OutputType>Exe</OutputType>\n    <TargetFramework>net10.0</TargetFramework>\n"
                               + "    <UseAppHost>false</UseAppHost>\n    <ImplicitUsings>enable</ImplicitUsings>\n  </PropertyGroup>\n</Project>\n";
        foreach (var (id, word) in new[] { ("hello", "Hello"), ("hi", "Hi") })
        {
            string task = Path.Combine(suite, "csharp", id);
            Directory.CreateDirectory(Path.Combine(task, "workspace"));
            Directory.CreateDirectory(Path.Combine(task, "verify", "tests"));
            File.WriteAllText(Path.Combine(task, "task.json"), $$"""
                {"prompt": "Make Greet say {{word}}.", "language": "csharp", "setup": ["dotnet restore -nodeReuse:false"],
                 "verify": ["dotnet build --no-restore -nodeReuse:false", "dotnet bin/Debug/net10.0/Greeter.dll"]}
                """);
            File.WriteAllText(Path.Combine(task, "workspace", "Greeter.csproj"), Project);
            File.WriteAllText(Path.Combine(task, "workspace", "Greeter.cs"), "static class Greeter\n{\n    public static string Greet() => \"Bye\";\n}\n");
            File.WriteAllText(Path.Combine(task, "verify", "tests", "Check.cs"), $"return Greeter.Greet() == \"{word}\" ? 0 : 1;\n");
        }

        string broken = Path.Combine(suite, "csharp", "broken");
        Directory.CreateDirectory(Path.Combine(broken, "workspace"));
        File.WriteAllText(Path.Combine(broken, "task.json"), """{"prompt": "Anything.", "setup": ["dotnet build missing.csproj -nodeReuse:false"], "verify": []}""");
        File.WriteAllText(Path.Combine(broken, "workspace", "a.txt"), "x");
        return suite;
    }

    // A quick suite: one task whose verification is git grep (no build), for runs of many episodes.
    private static string WriteQuickSuite(string root)
    {
        string task = Path.Combine(root, "quick", "greet");
        Directory.CreateDirectory(Path.Combine(task, "workspace"));
        Directory.CreateDirectory(Path.Combine(task, "verify", "tests"));
        File.WriteAllText(Path.Combine(task, "task.json"), """{"prompt": "Make Greet say Hello.", "verify": ["git grep -q Hello -- Greeter.cs"]}""");
        File.WriteAllText(Path.Combine(task, "workspace", "Greeter.cs"), "static class Greeter\n{\n    public static string Greet() => \"Bye\";\n}\n");
        File.WriteAllText(Path.Combine(task, "verify", "tests", "GreeterTests.cs"), "// hidden test\n");
        return Path.Combine(root, "quick");
    }

    private static string CodingTemp() => Path.Combine(Path.GetTempPath(), "idrak-coding-test-" + Guid.NewGuid().ToString("N"));

    // Registers the coding plug-in for a test; gives back whether it was registered before (so the test leaves it as found).
    private static bool CodingRegistered()
    {
        bool before = CodingPlugin.IsRegistered;
        CodingPlugin.Register();
        return before;
    }

    // A command sandbox that records what it is asked and answers as told.
    private sealed class RecordingSandbox(CommandResult answer) : ICommandSandbox
    {
        public List<(IReadOnlyList<string> Command, string Directory, CommandLimits Limits)> Calls { get; } = [];

        public string Name => "coding-test-recording";

        public string Summary => "records the commands (the coding tests)";

        public string? Check(CommandLimits limits) => null;

        public ValueTask<CommandResult> RunAsync(IReadOnlyList<string> command, string directory, CommandLimits limits, CancellationToken cancellationToken = default)
        {
            Calls.Add((command, directory, limits));
            return ValueTask.FromResult(answer);
        }
    }

    private static void ProcessSandboxLimits()
    {
        var registered = CommandSandboxes.Get(CommandSandboxes.Process);
        Check(registered is ProcessCommandSandbox && CommandSandboxes.Origin(CommandSandboxes.Process) == Overrides.Library, "process is the library's, from Idrak.Text");
        Check(Environment.GetEnvironmentVariable(CommandSandboxes.EnvironmentVariable) is not null || new CodingToolOptions().Sandbox == CommandSandboxes.Process,
            "the coding tools' default sandbox is process (IDRAK_SANDBOX unset)");
        ThrowsNaming<NotSupportedException>(() => CommandSandboxes.Get("coding-test-nobody"), "an unregistered sandbox", "not registered", "process", "Idrak.Reinforcement.Plugins");

        string root = CodingTemp();
        Directory.CreateDirectory(root);
        try
        {
            var sandbox = new ProcessCommandSandbox { PollInterval = TimeSpan.FromMilliseconds(1) };
            CommandResult Run(CommandLimits limits, params string[] command) => sandbox.RunAsync(command, root, limits).AsTask().GetAwaiter().GetResult();

            var version = Run(new CommandLimits { Programs = ["dotnet"] }, "dotnet", "--version");
            Check(version is { Succeeded: true, TimedOut: false, OutOfMemory: false, Refused: false } && version.Output.Length > 0, $"dotnet --version runs: {version}");
            Check(Run(new CommandLimits { Programs = ["git"] }, "dotnet", "--version") is { Refused: true } refused && refused.Output.Contains("not an allowed program", StringComparison.Ordinal),
                "a program not allowed is refused");
            Check(Run(new CommandLimits(), "coding-test-no-such-program") is { Refused: true } missing && missing.Output.Contains("not installed", StringComparison.Ordinal),
                "a program not on the PATH is refused");
            var slow = Run(new CommandLimits { Time = TimeSpan.FromMilliseconds(1) }, "dotnet", "--info");
            Check(slow is { TimedOut: true, ExitCode: null, Succeeded: false }, $"the time limit stops the command: {slow.TimedOut} {slow.ExitCode}");
            var cut = Run(new CommandLimits { MaxOutputCharacters = 300 }, "dotnet", "--help");
            Check(cut.Output.Contains("characters cut", StringComparison.Ordinal) && cut.Output.Length < 500, "the output is cut to its cap");
            var heavy = Run(new CommandLimits { MaxMemoryBytes = 1 }, "dotnet", "--info");
            Check(heavy is { OutOfMemory: true, TimedOut: false, ExitCode: null, Succeeded: false }, $"the memory limit stops the command: {heavy.OutOfMemory} {heavy.ExitCode}");
            var roomy = Run(new CommandLimits { MaxMemoryBytes = 64L << 30 }, "dotnet", "--version");
            Check(roomy is { Succeeded: true, OutOfMemory: false }, "within the memory limit it runs to its end");
            using (var cancelled = new CancellationTokenSource())
            {
                cancelled.Cancel();
                ThrowsNaming<OperationCanceledException>(() => sandbox.RunAsync(["dotnet", "--info"], root, new CommandLimits(), cancelled.Token).AsTask().GetAwaiter().GetResult(),
                    "a cancelled command");
            }

            foreach (var (limits, what) in new[] { (new CommandLimits { Network = false }, "turn the network off"), (new CommandLimits { MaxProcesses = 8 }, "limit the processes"),
                         (new CommandLimits { MaxCpus = 1 }, "limit the CPUs") })
            {
                Check(sandbox.Check(limits) is { } why && why.Contains(what, StringComparison.Ordinal) && why.Contains("container", StringComparison.Ordinal), $"process cannot {what}, naming container");
                ThrowsNaming<NotSupportedException>(() => Run(limits, "dotnet", "--version"), $"a command asking process to {what}", "container");
            }

            ThrowsNaming<NotSupportedException>(() => _ = new CodingTools(root, new CodingToolOptions { Network = false }), "coding tools with the network off on process", "container", "network");
            ThrowsNaming<NotSupportedException>(() => _ = new CodingTools(root, new CodingToolOptions { Sandbox = "coding-test-nobody" }), "coding tools on an unregistered sandbox", "not registered");

            // The coding tools give every command, split into words with its folder and limits, to their sandbox.
            var recorder = new RecordingSandbox(new CommandResult(null, false, "", TimeSpan.Zero, false) { OutOfMemory = true });
            Directory.CreateDirectory(Path.Combine(root, "sub"));
            var tools = new CodingTools(root, new CodingToolOptions { Commands = ["dotnet"], CommandTimeout = TimeSpan.FromSeconds(30), MaxMemoryBytes = 1L << 20,
                Environment = new Dictionary<string, string> { ["CODING_TEST"] = "1" } }, recorder);
            string text = tools.RunCommandAsync("dotnet build \"my app.csproj\" -c Release", "sub", 5).GetAwaiter().GetResult();
            var call = recorder.Calls.Single();
            Check(call.Command.SequenceEqual(["dotnet", "build", "my app.csproj", "-c", "Release"]) && call.Directory == Path.Combine(tools.Workspace, "sub"),
                $"the words and the folder: {string.Join(" | ", call.Command)} in {call.Directory}");
            Check(call.Limits is { Time.TotalSeconds: 5, MaxMemoryBytes: 1L << 20, Network: true } && call.Limits.Programs!.SequenceEqual(["dotnet"]) && call.Limits.Root == tools.Workspace
                  && call.Limits.Environment!["CODING_TEST"] == "1" && call.Limits.Environment["CI"] == "1", "the limits: the call's time, the memory, the programs, the root, the variables");
            Check(text.Contains("stopped: it used more than the memory limit (1 MB)", StringComparison.Ordinal), $"run_command says a command was stopped for its memory: {text}");
            Check(tools.RunCommandAsync("python x.py").GetAwaiter().GetResult().Contains("not an allowed program", StringComparison.Ordinal) && recorder.Calls.Count == 1,
                "a program not allowed never reaches the sandbox");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void ContainerSandboxCommandLine()
    {
        var limits = new CommandLimits();
        var none = new ContainerSandbox(new ContainerSandboxOptions { Program = null, Programs = ["coding-test-no-such-container-program"], Image = "coding-test:1" });
        Check(none.FindProgram() is null && none.Check(limits) is { } why && why.Contains("coding-test-no-such-container-program", StringComparison.Ordinal)
              && why.Contains(ContainerSandboxOptions.ProgramVariable, StringComparison.Ordinal), "no program on the PATH: refused, naming IDRAK_SANDBOX_PROGRAM");
        string root = CodingTemp();
        Directory.CreateDirectory(Path.Combine(root, "src"));
        try
        {
            ThrowsNaming<NotSupportedException>(() => none.RunAsync(["dotnet", "test"], root, limits).AsTask().GetAwaiter().GetResult(), "running without a program", "IDRAK_SANDBOX_PROGRAM");
            var given = new ContainerSandbox(new ContainerSandboxOptions { Program = Path.Combine(root, "no-such-docker"), Image = "coding-test:1" });
            Check(given.Check(limits) is { } missing && missing.Contains("is not found", StringComparison.Ordinal), "a program given that does not exist: refused");

            // A program found (dotnet stands in: the command line is checked, not run), but no image: refused, naming the variable.
            var noImage = new ContainerSandbox(new ContainerSandboxOptions { Program = null, Programs = ["dotnet"], Image = null });
            Check(noImage.FindProgram() is not null && noImage.Check(limits) is { } image && image.Contains(ContainerSandboxOptions.ImageVariable, StringComparison.Ordinal),
                "no image: refused, naming IDRAK_SANDBOX_IMAGE");
            ThrowsNaming<NotSupportedException>(() => _ = new CodingTools(root, new CodingToolOptions(), noImage), "coding tools on a container without an image", "IDRAK_SANDBOX_IMAGE");

            var sandbox = new ContainerSandbox(new ContainerSandboxOptions { Program = null, Image = "coding-test/sdk:1", Network = null, User = "1000:1000", ExtraArguments = ["--cap-drop", "ALL"] });
            var full = new CommandLimits
            {
                Root = root, MaxMemoryBytes = 512L << 20, MaxCpus = 1.5, MaxProcesses = 64, Environment = new Dictionary<string, string> { ["NO_COLOR"] = "1", ["CI"] = "1" },
            };
            var line = sandbox.CommandLine("docker", ["dotnet", "test", "--no-build"], Path.Combine(root, "src"), full, "coding-test-name");
            string[] expected =
            [
                "docker", "run", "--rm", "--name", "coding-test-name", "--network", "none", "--mount", $"type=bind,source={Path.GetFullPath(root)},target=/workspace",
                "--workdir", "/workspace/src", "--memory", "536870912b", "--memory-swap", "536870912b", "--cpus", "1.5", "--pids-limit", "64", "--user", "1000:1000",
                "--env", "CI=1", "--env", "NO_COLOR=1", "--cap-drop", "ALL", "coding-test/sdk:1", "dotnet", "test", "--no-build",
            ];
            Check(line.SequenceEqual(expected), $"the command line:\n{string.Join(" ", line)}\nexpected:\n{string.Join(" ", expected)}");
            Check(sandbox.CommandLine("podman", ["dotnet"], root, new CommandLimits(), "n").SequenceEqual(
                ["podman", "run", "--rm", "--name", "n", "--network", "none", "--mount", $"type=bind,source={Path.GetFullPath(root)},target=/workspace", "--workdir", "/workspace",
                 "--user", "1000:1000", "--cap-drop", "ALL", "coding-test/sdk:1", "dotnet"]), "the folder itself as the root, no limits: the network still off");
            var on = new ContainerSandbox(new ContainerSandboxOptions { Image = "i", Network = "on" });
            Check(!on.CommandLine("docker", ["x"], root, new CommandLimits(), "n").Contains("--network"), "IDRAK_SANDBOX_NETWORK=on: the program's default network");
            Check(on.CommandLine("docker", ["x"], root, new CommandLimits { Network = false }, "n").Contains("none"), "a command asked to stay offline stays offline");
            var named = new ContainerSandbox(new ContainerSandboxOptions { Image = "i", Network = "coding-net" }).CommandLine("docker", ["x"], root, new CommandLimits(), "n");
            Check(named.SkipWhile(a => a != "--network").Skip(1).First() == "coding-net", "a network by name");
            ThrowsNaming<ArgumentException>(() => sandbox.CommandLine("docker", ["x"], Path.GetTempPath(), new CommandLimits { Root = root }, "n"), "a folder outside the root", "not inside");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }

        bool before = CodingRegistered();
        try
        {
            Check(CommandSandboxes.Get(CommandSandboxes.Container) is ContainerSandbox && CommandSandboxes.Origin(CommandSandboxes.Container) == "Idrak.Reinforcement.Plugins",
                $"container is the coding plug-in's: {CommandSandboxes.Origin(CommandSandboxes.Container)}");
        }
        finally
        {
            if (!before)
            {
                CodingPlugin.Unregister();
            }
        }
    }

    private static void CodingPluginRegisters()
    {
        Check(CodingPlugin.Plugin.Name == "coding" && ReinforcementPlugins.All.Contains(CodingPlugin.Plugin), "the plug-in \"coding\" of ReinforcementPlugins.All");
        bool before = CodingPlugin.IsRegistered;
        string State() => $"{string.Join(",", EnvironmentKinds.Names)} | {string.Join(",", RewardFunctions.Names)} | {string.Join(",", CommandSandboxes.Names)}";
        try
        {
            CodingPlugin.Unregister();
            string found = State();
            CodingPlugin.Register();
            CodingPlugin.Register();
            Check(EnvironmentKinds.Names.Count(n => n == "coding") == 1 && EnvironmentKinds.Get("coding") is CodingEnvironmentKind, "the coding environment once");
            Check(new[] { "tests-passed", "verify-fraction", "rounds", "tool-errors" }.All(n => RewardFunctions.Names.Count(r => r == n) == 1 && RewardFunctions.Origin(n) == "Idrak.Reinforcement.Plugins"),
                "the four rewards once, from Idrak.Reinforcement.Plugins");
            Check(CommandSandboxes.Names.Count(n => n == "container") == 1, "the container sandbox once");
            CodingPlugin.Unregister();
            Check(State() == found, $"Unregister leaves the registries as found:\n{found}\nnow:\n{State()}");
            CodingPlugin.Register();

            RewardInput Input(bool passed, int commands, int passedCommands, int rounds, int errors) => new("Fix it.", "Done.", null, new Dictionary<string, string>
            {
                [CodingInfo.VerifyPassed] = passed ? "true" : "false", [CodingInfo.VerifyCommands] = $"{commands}", [CodingInfo.VerifyPassedCommands] = $"{passedCommands}",
                [CodingInfo.Rounds] = $"{rounds}", [CodingInfo.ToolErrors] = $"{errors}", [CodingInfo.ToolCalls] = $"{rounds}",
            });
            RewardInput[] samples = [Input(true, 2, 2, 3, 0), Input(false, 2, 1, 5, 2), Input(true, 0, 0, 1, 0), Input(false, 4, 0, 40, 7)];
            foreach (var reward in CodingRewards.All)
            {
                Conformance.CheckRewardFunction(reward, samples).ThrowIfFailed();
            }

            float[] Score(string name, IReadOnlyDictionary<string, string>? options = null) => RewardFunctions.Get(name).ScoreAsync(samples, options).AsTask().GetAwaiter().GetResult();
            Check(Score("tests-passed").SequenceEqual([1f, 0f, 1f, 0f]), "tests-passed");
            Check(Score("verify-fraction").SequenceEqual([1f, 0.5f, 1f, 0f]), "verify-fraction: in order up to the first failure; none and passed is 1");
            AssertClose([-0.03f, -0.05f, -0.01f, -0.4f], Score("rounds"), 1e-6f, "rounds: 0.01 a reply");
            AssertClose([0f, -0.2f, 0f, -0.7f], Score("tool-errors", new Dictionary<string, string> { ["penalty"] = "0.1" }), 1e-6f, "tool-errors with a penalty of 0.1");
            AssertClose([-0.01f, -0.03f, 0f, -0.38f], Score("rounds", new Dictionary<string, string> { ["free"] = "2" }), 1e-6f, "rounds past 2 free");
            ThrowsNaming<ArgumentException>(() => Score("rounds", new Dictionary<string, string> { ["weight"] = "1" }), "an option rounds does not take", "'weight'", "penalty, free");
            ThrowsNaming<ArgumentException>(() => RewardFunctions.Get("tests-passed").ScoreAsync([new RewardInput("p", "a")]).AsTask().GetAwaiter().GetResult(),
                "an answer without the coding environment's info", "verify_passed", "coding environment");
        }
        finally
        {
            if (!before)
            {
                CodingPlugin.Unregister();
            }
        }
    }

    // An episode played with the actions given (one after another, until it ends): every step.
    private static List<EnvironmentStep<ChatObservation>> PlayCoding(IEnvironment<ChatObservation, ChatMessage> environment, int seed, IReadOnlyList<ChatMessage> actions)
    {
        environment.Reset(seed);
        var steps = new List<EnvironmentStep<ChatObservation>>();
        foreach (var action in actions)
        {
            steps.Add(environment.StepAsync(action).AsTask().GetAwaiter().GetResult());
            if (steps[^1].Done)
            {
                break;
            }
        }

        return steps;
    }

    private static string ShowInfo(IReadOnlyDictionary<string, string>? info) =>
        info is null ? "" : string.Join(", ", info.OrderBy(p => p.Key, StringComparer.Ordinal).Select(p => $"{p.Key}={p.Value}"));

    private static void CodingEpisodes()
    {
        bool before = CodingRegistered();
        string root = CodingTemp();
        string suite = WriteCSharpSuite(root), work = Path.Combine(root, "work");
        var kind = EnvironmentKinds.Get("coding");
        var options = new Dictionary<string, string>
        {
            ["suite"] = suite, ["system"] = "Fix the code.", ["rounds"] = "6", ["reward"] = "tests-passed:1,rounds:1,tool-errors:1", ["work"] = work, ["workspaces"] = "2",
        };
        var first = (CodingEnvironment)kind.Create(new EnvironmentRequest(options));
        var second = (CodingEnvironment)kind.Create(new EnvironmentRequest(options));
        CodingEnvironment? refusing = null;
        try
        {
            Check(first.ObservationSpace == Space.Conversation && first.ActionSpace == Space.Conversation, "conversations in, replies out");

            // Seeds that draw each task: an order of the tasks from the seed, the first whose setup passes.
            int hello = -1, hi = -1;
            for (int seed = 0; seed < 64 && (hello < 0 || hi < 0 || first.BrokenTasks.Count == 0); seed++)
            {
                var observation = first.Reset(seed);
                Check(observation.Messages is [{ Role: "system", Content: "Fix the code." }, { Role: "user" }] && observation.Tools!.Count == 6, "the first observation: the system prompt, the task, the tools");
                (hello, hi) = first.Task!.Id switch { "csharp/hello" when hello < 0 => (seed, hi), "csharp/hi" when hi < 0 => (hello, seed), _ => (hello, hi) };
                Check(first.Task.Id != "csharp/broken", "a task whose setup failed is never played");
            }

            Check(hello >= 0 && hi >= 0, $"seeds for both tasks: {hello}, {hi}");
            Check(first.BrokenTasks.Keys.SequenceEqual(["csharp/broken"]) && first.BrokenTasks["csharp/broken"].Contains("missing.csproj", StringComparison.Ordinal),
                $"the broken setup is left out and reported: {string.Join(", ", first.BrokenTasks.Keys)}");

            // Solved, with a failed call on the way; replayed in another environment from the same seed.
            ChatMessage[] solve = [CodingRead, CodingMissing, CodingFix, CodingDone];
            var played = PlayCoding(first, hello, solve);
            string prepared = Path.Combine(work, "prepared", "001");
            string assets = Path.Combine(prepared, "obj", "project.assets.json");
            var restored = File.GetLastWriteTimeUtc(assets);
            var replay = PlayCoding(second, hello, solve);
            Check(played.Count == 4 && replay.Count == 4, $"four steps: {played.Count}, {replay.Count}");
            for (int i = 0; i < played.Count; i++)
            {
                Check(played[i].Observation.Equals(replay[i].Observation) && played[i].Reward == replay[i].Reward && played[i].Terminated == replay[i].Terminated
                      && ShowInfo(played[i].Info) == ShowInfo(replay[i].Info), $"step {i} replays the same: {ShowInfo(played[i].Info)} | {ShowInfo(replay[i].Info)}");
            }

            Check(played.Take(3).All(s => s is { Reward: 0f, Done: false }), "no reward before the end");
            var end = played[^1];
            Check(end is { Terminated: true, Truncated: false }, "an answer without a tool call ends the episode");
            Check(ShowInfo(end.Info) == "broken_tasks=csharp/broken, outcome=Passed, rounds=4, task=csharp/hello, test_edits=, tool_calls=3, tool_errors=1, verify_commands=2, verify_passed=true, verify_passed_commands=2",
                $"the end's info: {ShowInfo(end.Info)}");
            AssertClose([1f - 0.04f - 0.05f], [end.Reward], 1e-6f, "the reward: tests passed, less 0.01 a round and 0.05 a failed call");
            Check(end.Observation.Messages.Count == 2 + 4 + 3 && end.Observation.Messages[3] is { Role: "tool", ToolName: "read_file" } read && read.Content.Contains("\"Bye\"", StringComparison.Ordinal)
                  && end.Observation.Messages[5].Content.StartsWith("Error:", StringComparison.Ordinal), "the tools' results are the observations: the file, the error");
            Check(first.Verification!.Output.Contains("dotnet build", StringComparison.Ordinal), "the verification's output");
            Check(File.GetLastWriteTimeUtc(assets) == restored && Directory.GetDirectories(Path.Combine(work, "prepared")).Length == 3,
                "setup ran once per task: the episodes cloned the prepared copy, restored once");
            Check(!Directory.EnumerateFileSystemEntries(Path.Combine(work, "episodes")).Any(), "an ended episode's workspace is deleted");

            // Not solved: the build passes, the hidden test fails.
            var wrong = PlayCoding(first, hi, [CodingFix, CodingDone])[^1];
            Check(wrong.Info![CodingInfo.VerifyPassed] == "false" && wrong.Info[CodingInfo.VerifyPassedCommands] == "1" && wrong.Info[CodingInfo.Outcome] == "Failed",
                $"not solved: {ShowInfo(wrong.Info)}");
            AssertClose([-0.02f], [wrong.Reward], 1e-6f, "not solved: the penalties alone");
            Check(RewardFunctions.Get("verify-fraction").ScoreAsync([new RewardInput("", "", null, wrong.Info)]).AsTask().GetAwaiter().GetResult()[0] == 0.5f, "half the verification passed");

            // Out of rounds: still calling tools at the last one; verification still runs.
            var budget = PlayCoding(first, hello, [CodingRead, CodingMissing, CodingRead, CodingMissing, CodingRead, CodingMissing, CodingDone]);
            Check(budget.Count == 6 && budget[^1] is { Truncated: true, Terminated: false } && budget[^1].Info![CodingInfo.Outcome] == "OutOfBudget",
                $"the rounds run out: {budget.Count} steps, {ShowInfo(budget[^1].Info)}");

            // A test edit: the hidden tests overwrite it (the task passes), and it is reported.
            var edited = PlayCoding(first, hello, [CodingFix, CodingTestEdit, CodingDone])[^1];
            Check(edited.Info![CodingInfo.TestEdits] == "tests/Check.cs" && edited.Info[CodingInfo.VerifyPassed] == "true" && !edited.Info.ContainsKey(CodingInfo.Refused),
                $"a test edit reported, the hidden tests copied over it: {ShowInfo(edited.Info)}");

            // Refused when asked: no verification, counted as failed.
            refusing = (CodingEnvironment)kind.Create(new EnvironmentRequest(new Dictionary<string, string>(options) { ["refuse-test-edits"] = "true", ["work"] = Path.Combine(root, "work-refusing") }));
            var refused = PlayCoding(refusing, hello, [CodingFix, CodingTestEdit, CodingDone])[^1];
            Check(refused.Info![CodingInfo.Refused] == "test-edits" && refused.Info[CodingInfo.VerifyPassed] == "false" && refused.Info[CodingInfo.TestEdits] == "tests/Check.cs",
                $"a test edit refused: {ShowInfo(refused.Info)}");
            AssertClose([-0.03f], [refused.Reward], 1e-6f, "a refused episode: the penalties alone");
            Check(refusing.Verification!.Output.StartsWith("refused", StringComparison.Ordinal), "the verification did not run");

            ThrowsNaming<InvalidOperationException>(() => first.StepAsync(CodingDone).AsTask().GetAwaiter().GetResult(), "a step after the end", "reset");
            ThrowsNaming<ArgumentException>(() => kind.Create(new EnvironmentRequest(new Dictionary<string, string> { ["suite"] = suite, ["rounds"] = "many" })), "a bad option", "rounds=many");
            ThrowsNaming<ArgumentException>(() => kind.Create(new EnvironmentRequest(new Dictionary<string, string> { ["suite"] = suite, ["filter"] = "nothing-matches" })), "a filter that leaves nothing", "nothing-matches");
        }
        finally
        {
            first.Dispose();
            second.Dispose();
            refusing?.Dispose();
            Check(!Directory.Exists(work) && !Directory.Exists(Path.Combine(root, "work-refusing")), "the last environment of a run deletes its copies");
            CodingWorkspace.Delete(root);
            if (!before)
            {
                CodingPlugin.Unregister();
            }
        }
    }

    // A policy that plays a script: its first reply fixes Greeter.cs, its second answers (the coding tests' collector runs).
    private sealed class ScriptedCoder : IPolicy<ChatObservation, ChatMessage>
    {
        public ValueTask<PolicyActions<ChatMessage>> ActAsync(IReadOnlyList<ChatObservation> observations, ActionSampling sampling, CancellationToken cancellationToken = default)
        {
            var actions = observations.Select(o => o.Messages.Count(m => m.Role == "assistant") == 0 ? CodingFix : CodingDone).ToList();
            return ValueTask.FromResult(new PolicyActions<ChatMessage>(actions, [.. actions.Select(_ => new ActionRecord([1], [-1f], float.NaN, [1]))]));
        }

        public UpdateStats Update(RolloutBatch<ChatObservation, ChatMessage> batch, IPolicyLoss loss, IReadOnlyList<int>? episodes = null, IReadOnlyDictionary<string, string>? lossOptions = null) =>
            throw new NotSupportedException("a script does not learn");

        public void Save(string path) => throw new NotSupportedException();

        public void Load(string path) => throw new NotSupportedException();

        public void Dispose()
        {
        }
    }

    private static void CodingMeasuredLimits()
    {
        bool before = CodingRegistered();
        string root = CodingTemp();
        var recorder = new RecordingSandbox(new CommandResult(0, false, "", TimeSpan.Zero, false));
        CommandSandboxes.Register(recorder);
        try
        {
            var kind = EnvironmentKinds.Get("coding");
            int made = 0;

            // The episode's verification limit and a tool command's, the quick suite's verification answered at once by the recorder.
            (TimeSpan Verify, TimeSpan Command) Limits(Dictionary<string, string> extra)
            {
                var options = new Dictionary<string, string> { ["suite"] = WriteQuickSuite(Path.Combine(root, $"suite-{made++}")), ["system"] = "Fix it.", ["rounds"] = "4", ["sandbox"] = recorder.Name };
                foreach (var (key, value) in extra)
                {
                    options[key] = value;
                }

                using var environment = (CodingEnvironment)kind.Create(new EnvironmentRequest(options));
                environment.Reset(1);
                var command = new ChatMessage("assistant", "", ToolCalls: [new ToolCall("run_command", new JsonObject { ["command"] = "git status" })]);
                environment.StepAsync(command).AsTask().GetAwaiter().GetResult();
                var tool = recorder.Calls[^1].Limits.Time;
                environment.StepAsync(CodingDone).AsTask().GetAwaiter().GetResult();
                return (recorder.Calls[^1].Limits.Time, tool);
            }

            var measured = Limits([]);
            Check(measured == (TimeSpan.FromSeconds(30), TimeSpan.FromSeconds(30)), $"a verification measured at once: both limits the least, 30 s: {measured}");
            var least = Limits(new() { ["timeout-least"] = "2" });
            Check(least == (TimeSpan.FromSeconds(2), TimeSpan.FromSeconds(2)), $"timeout-least 2: {least}");
            var most = Limits(new() { ["timeout-least"] = "1000" });
            Check(most == (TimeSpan.FromSeconds(600), TimeSpan.FromSeconds(300)), $"never past 600 s for verification and 300 s for a tool command: {most}");
            var given = Limits(new() { ["verify-timeout"] = "7", ["command-timeout"] = "9" });
            Check(given == (TimeSpan.FromSeconds(7), TimeSpan.FromSeconds(9)), $"verify-timeout and command-timeout as given: {given}");
            ThrowsNaming<ArgumentException>(() => kind.Create(new EnvironmentRequest(new Dictionary<string, string> { ["suite"] = WriteQuickSuite(Path.Combine(root, "bad")), ["timeout-scale"] = "x" })),
                "a timeout-scale that is not a number", "timeout-scale");
        }
        finally
        {
            CommandSandboxes.Unregister(recorder.Name);
            if (!before)
            {
                CodingPlugin.Unregister();
            }

            if (Directory.Exists(root))
            {
                Directory.Delete(root, true);
            }
        }
    }

    private static void CodingSuiteAndParallel()
    {
        bool before = CodingRegistered();
        string root = CodingTemp();
        try
        {
            // The environment suite on the quick suite, the actions drawn from the scripted replies.
            ChatMessage[] replies = [CodingDone, CodingRead, CodingFix, CodingMissing];
            var quick = new Dictionary<string, string> { ["suite"] = WriteQuickSuite(root), ["system"] = "Fix it.", ["rounds"] = "4" };
            Conformance.CheckEnvironmentKind<ChatObservation, ChatMessage>(EnvironmentKinds.Get("coding"), (random, _) => replies[random.Next(replies.Length)], quick, maxSteps: 6,
                new ContractCheckOptions { RandomCases = 1 }).ThrowIfFailed();

            // Episodes in parallel give what they give one at a time; the time of both ways is measured.
            var options = new Dictionary<string, string> { ["suite"] = WriteCSharpSuite(root), ["filter"] = "csharp/h", ["system"] = "Fix the code.", ["workspaces"] = "4" };
            var kind = EnvironmentKinds.Get("coding");
            // Two seeds for each task (hello solved, hi not), by the environment's rule: the tasks in an order drawn from the seed.
            int Drawn(int seed)
            {
                int[] order = [0, 1];
                new Random(seed).Shuffle(order);
                return order[0];
            }

            int[] seeds = [.. Enumerable.Range(1, 200).Where(s => Drawn(s) == 0).Take(2), .. Enumerable.Range(1, 200).Where(s => Drawn(s) == 1).Take(2)];
            EpisodeStart[] starts = [.. seeds.Select((seed, i) => new EpisodeStart(i, seed, i))];
            using var policy = new ScriptedCoder();
            using var sequential = new RolloutCollector<ChatObservation, ChatMessage>(kind, options, seed: 1, environments: 1, parallelSteps: false);
            using var parallel = new RolloutCollector<ChatObservation, ChatMessage>(kind, options, seed: 1, environments: 4, parallelSteps: true);
            using (sequential.CollectAsync(policy, [starts[0], starts[2]], new ActionSampling()).AsTask().GetAwaiter().GetResult())   // both tasks' setup, outside the timing
            {
            }

            var clock = Stopwatch.StartNew();
            using var one = sequential.CollectAsync(policy, starts, new ActionSampling()).AsTask().GetAwaiter().GetResult();
            var oneTime = clock.Elapsed;
            clock.Restart();
            using var many = parallel.CollectAsync(policy, starts, new ActionSampling()).AsTask().GetAwaiter().GetResult();
            var manyTime = clock.Elapsed;
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"    coding episodes (4, each a dotnet build and run): one at a time {oneTime.TotalSeconds:F1} s, four at once {manyTime.TotalSeconds:F1} s, {Environment.ProcessorCount} cores"));
            Check(one.Rewards.ToArray().SequenceEqual(many.Rewards.ToArray()), $"the same rewards: {string.Join(", ", one.Rewards.ToArray())} | {string.Join(", ", many.Rewards.ToArray())}");
            Check(one.Rewards.ToArray().Distinct().Count() == 2, $"both tasks drawn, one solved and one not: {string.Join(", ", one.Rewards.ToArray())}");
            for (int e = 0; e < starts.Length; e++)
            {
                Check(ShowInfo(one.Trajectories[e].Info) == ShowInfo(many.Trajectories[e].Info)
                      && one.Trajectories[e].Observations.SequenceEqual(many.Trajectories[e].Observations), $"episode {e}: the same info and conversation");
            }
        }
        finally
        {
            CodingWorkspace.Delete(root);
            if (!before)
            {
                CodingPlugin.Unregister();
            }
        }
    }

    private static void CodingDistillation(Device device)
    {
        bool before = CodingRegistered();
        string root = CodingTemp();
        string folder = WriteChatModelWith(TinyChatSpec(), ChatMlTemplate);
        try
        {
            string quick = WriteQuickSuite(root);
            using var studentModel = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
            using var teacherModel = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
            using var teacher = DistillationTeachers.FromModel(teacherModel);
            var session = new FineTuningSession(studentModel, SessionOptions(1e-2f) with { Teacher = teacher }, totalSteps: 8);
            using var policy = new LanguageModelPolicy(session, new LanguageModelPolicyOptions { MaxNewTokens = 6, TeacherWeight = 1f });
            var environmentOptions = new Dictionary<string, string> { ["suite"] = quick, ["system"] = "Fix it.", ["rounds"] = "2" };

            // The student's own episodes in the coding environment, R2's trainer with grpo and the teacher term.
            int checkedEpisodes = 0;
            var trainer = new ReinforcementTrainer<ChatObservation, ChatMessage>(EnvironmentKinds.Get("coding"), policy, new ReinforcementOptions
            {
                Algorithm = "grpo", EpisodesPerIteration = 4, GroupSize = 4, Iterations = 2, Seed = 3, Environments = 1, ParallelSteps = false, EnvironmentOptions = environmentOptions,
            })
            {
                OnBatch = (_, batch) =>
                {
                    var layouts = policy.Lay(batch);
                    for (int e = 0; e < batch.Episodes; e++)
                    {
                        var trained = layouts[e].Sequences.SelectMany(s => s.Tokens.Where((_, t) => s.Trained[t]));
                        Check(trained.SequenceEqual(batch.Trajectories[e].Records.SelectMany(r => r.Tokens)), $"episode {e}: the replies' ids alone are trained");
                        checkedEpisodes++;
                    }
                },
            };
            var result = trainer.Train();
            Check(result.Iterations == 2 && checkedEpisodes == 8 && result.History.All(h => double.IsFinite(h.Loss)), $"two iterations on the coding environment: {result.Iterations}, {checkedEpisodes} episodes");

            // Multi-turn: a reply that reads a file, then an answer; the file's text comes back as a tool message, never trained.
            var kind = EnvironmentKinds.Get("coding");
            var tokenizer = studentModel.Tokenizer!;
            var trajectories = new List<Trajectory<ChatObservation, ChatMessage>>();
            foreach (var (script, index) in new[] { (new[] { CodingFix, CodingDone }, 0), (new[] { CodingRead, CodingDone }, 1) })
            {
                using var environment = (IEnvironment<ChatObservation, ChatMessage>)kind.Create(new EnvironmentRequest(environmentOptions));
                var observations = new List<ChatObservation> { environment.Reset(7) };
                var records = new List<ActionRecord>();
                var rewards = new List<float>();
                EnvironmentStep<ChatObservation> step = default;
                foreach (var reply in script)
                {
                    var observation = observations[^1];
                    int[] prompt = [.. tokenizer.Encode(policy.Chat.RenderPrompt(new ChatRequest(observation.Messages, observation.Tools, observation.Think)))];
                    int[] ids = [.. tokenizer.Encode(reply.Content + "<|im_end|>")];
                    records.Add(Turn(prompt, ids));
                    step = environment.StepAsync(reply).AsTask().GetAwaiter().GetResult();
                    observations.Add(step.Observation);
                    rewards.Add(step.Reward);
                }

                Check(step.Observation.Messages.Any(m => m.Role == "tool") && step.Done, $"script {index}: a tool's result in the conversation, then the end");
                trajectories.Add(new Trajectory<ChatObservation, ChatMessage>(index, 7, observations, script, records, rewards, step.Terminated, step.Truncated, group: 0));
            }

            using var multi = new RolloutBatch<ChatObservation, ChatMessage>(trajectories);
            Check(multi.Rewards.ToArray().Max() == 1f && multi.Rewards.ToArray().Sum() == 1f, $"one solved, one not: {string.Join(", ", multi.Rewards.ToArray())}");
            multi.Estimate(AdvantageEstimators.Get("group"));
            var turns = policy.Lay(multi);
            for (int e = 0; e < turns.Count; e++)
            {
                var trained = turns[e].Sequences.SelectMany(s => s.Tokens.Where((_, t) => s.Trained[t]));
                Check(trained.SequenceEqual(trajectories[e].Records.SelectMany(r => r.Tokens)), $"multi-turn episode {e} ({turns[e].Mode}): the two replies' ids alone are trained");
            }

            var stats = policy.Update(multi, PolicyLosses.Get("grpo"));
            Check(float.IsFinite(stats.Loss) && stats.Units == trajectories.Sum(t => t.Units), $"grpo and the teacher term over the multi-turn episodes: {stats}");
        }
        finally
        {
            Directory.Delete(folder, true);
            CodingWorkspace.Delete(root);
            if (!before)
            {
                CodingPlugin.Unregister();
            }
        }
    }
}
