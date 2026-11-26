// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Abstraction.Reinforcement;
using Idrak.Abstraction.Testing;
using Idrak.Cluster;
using Idrak.Generation;
using Idrak.Models;
using Idrak.Reinforcement;
using Idrak.Reinforcement.Plugins;
using Idrak.Reinforcement.Plugins.Coding;
using Idrak.Text;
using Idrak.Text.Generation;
using Idrak.Text.Reinforcement;

// Plan 27, R4: the command sandboxes (process, Idrak.Text's; container, the coding plug-in's): a program run within its
// limits, a command started interactively, who asks for it, and the container's command line.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] CodingPluginGroup =
    [
        ("command sandboxes: process runs a program without a shell, enforces the programs allowed, its time, output and memory limits (stopping the command), hands each line of output to CommandLimits.Output as it is written, refuses network off, processes and CPUs naming container, and stops on cancellation; a program not allowed is refused with the sandbox's words", d => { if (d == Device.Cpu) ProcessSandboxLimits(); }),
        ("command sandboxes: interactive start: process starts a command with its standard input, output and error as streams (a child of this program echoes each line sent to it on its output, an error line on its error stream, and exits with its code at the end of its input; its variables set); the memory stays flat over thousands of lines; Stop and a dispose end its whole process tree, the time limit stops it; a program not allowed or not found is a refused start, a limit it cannot keep throws naming container, a cancelled start throws; a sandbox without an interactive start (container, an app's) refuses with the reason", d => { if (d == Device.Cpu) ProcessSandboxInteractive(); }),
        ("command sandboxes: requester: CommandLimits.Requester (who asks, as the words of a sentence, for a sandbox that asks the user first) is \"the agent\" by default, Purpose (what the command is for) is none by default; a copy keeps the ones given, and they change nothing about the run (the process sandbox runs the same command with another requester)", d => { if (d == Device.Cpu) CommandRequester(); }),
        ("command sandboxes: container refuses when no program is found or no image is set (naming IDRAK_SANDBOX_PROGRAM and IDRAK_SANDBOX_IMAGE), and builds its command line: the workspace mounted, the network off unless allowed, the memory, CPU and process limits, the user, the variables, the image, then the command", d => { if (d == Device.Cpu) ContainerSandboxCommandLine(); }),
    ];

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
        string root = CodingTemp();
        Directory.CreateDirectory(root);
        try
        {
            var sandbox = new ProcessCommandSandbox { PollInterval = TimeSpan.FromMilliseconds(1) };
            CommandResult Run(CommandLimits limits, params string[] command) => sandbox.RunAsync(command, root, limits).AsTask().GetAwaiter().GetResult();

            var version = Run(new CommandLimits { Programs = ["dotnet"] }, "dotnet", "--version");
            Check(version is { Succeeded: true, TimedOut: false, OutOfMemory: false, Refused: false } && version.Output.Length > 0, $"dotnet --version runs: {version}");
            var lines = new List<string>();
            var watched = Run(new CommandLimits { Programs = ["dotnet"], MaxOutputCharacters = 40, Output = line => { lock (lines) { lines.Add(line); } } }, "dotnet", "--help");
            Check(watched.Succeeded && lines.Count > 1 && string.Join('\n', lines).Length > 40 && watched.Output.Contains("characters cut", StringComparison.Ordinal),
                $"CommandLimits.Output sees every line as it is written ({lines.Count} lines), untrimmed, while the result is cut to its cap");
            Check(Run(new CommandLimits { Output = _ => throw new InvalidOperationException("a watcher that throws") }, "dotnet", "--version").Succeeded,
                "a watcher that throws does not stop the command");
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

            var direct = sandbox.RunAsync(["python", "x.py"], root, new CommandLimits { Programs = ["dotnet"] }).AsTask().GetAwaiter().GetResult();
            Check(direct.Refused && direct.Output == "Error: 'python' is not an allowed program. Allowed: dotnet.", $"the process sandbox refuses it, in its words: {direct.Output}");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    // Plan 1 (Craft), X part 6b: a command started interactively through the process sandbox, its streams the caller's.
    private static void ProcessSandboxInteractive() => ProcessSandboxInteractiveAsync().GetAwaiter().GetResult();

    private static async Task ProcessSandboxInteractiveAsync()
    {
        var limits = new CommandLimits();
        var sandbox = new ProcessCommandSandbox { PollInterval = TimeSpan.FromMilliseconds(5) };
        Check(sandbox.CheckStart(limits) is null, "the process sandbox starts commands interactively (CheckStart is null)");
        var (file, prefix) = LocalhostCluster.ProgramOf(typeof(DesktopTests).Assembly);
        string[] Child(string scenario) => [file, .. prefix, "--sandbox-child", scenario];
        var wait = TimeSpan.FromSeconds(60);
        string root = CodingTemp();
        Directory.CreateDirectory(root);
        try
        {
            var start = await sandbox.StartAsync(Child("echo"), root, new CommandLimits { Environment = new Dictionary<string, string> { ["IDRAK_SANDBOX_TEST"] = "7" } });
            Check(start.Started, $"the echoing child starts: {start.Refusal}");
            var session = start.Session!;
            await using (session)
            {
                using var output = new StreamReader(session.Output, Encoding.UTF8);
                using var error = new StreamReader(session.Error, Encoding.UTF8);
                Check(session.ProcessId > 0 && await output.ReadLineAsync().WaitAsync(wait) == "ready 7", "it starts in its folder with its variable, and says so on its output");
                foreach (var text in new[] { "first line", "second line, ünïcode and 数字" })
                {
                    await session.Input.WriteAsync(Encoding.UTF8.GetBytes(text + "\n"));
                    string? back = await output.ReadLineAsync().WaitAsync(wait);
                    Check(back == "echo: " + text, $"a line written to its input comes back on its output as it is sent ({back})");
                }

                await session.Input.WriteAsync("err oops\n"u8.ToArray());
                Check(await error.ReadLineAsync().WaitAsync(wait) == "error: oops", "the error stream is its own");

                // Memory flat: thousands of lines through the streams keep nothing between them and the caller.
                var line = "a line of some length, sent and read back again\n"u8.ToArray();
                for (int i = 0; i < 200; i++)
                {
                    await session.Input.WriteAsync(line);
                    await output.ReadLineAsync();
                }

                long before = GC.GetTotalMemory(forceFullCollection: true);
                for (int i = 0; i < 5_000; i++)
                {
                    await session.Input.WriteAsync(line);
                    await output.ReadLineAsync();
                }

                long grown = GC.GetTotalMemory(forceFullCollection: true) - before;
                Console.WriteLine($"    5,000 lines through the streams: the managed memory grew by {grown} bytes");
                Check(grown < 256 * 1024, $"the memory stays flat over 5,000 lines ({grown} bytes more after them)");

                await session.Input.DisposeAsync();                    // the end of its input: it says bye and exits with 3
                Check(await output.ReadLineAsync().WaitAsync(wait) == "bye" && await output.ReadLineAsync().WaitAsync(wait) is null, "closing its input ends it: its last line, then the end of its output");
                var ended = await session.Completion.WaitAsync(wait);
                Check(ended is { ExitCode: 3, TimedOut: false, OutOfMemory: false, Stopped: false, Refused: false, Output.Length: 0 }, $"its exit code at its end, no output kept (the streams were the caller's): {ended}");
            }

            // Stop ends the whole tree: the child and the grandchild it started.
            var tree = (await sandbox.StartAsync(Child("tree"), root, limits)).Session!;
            await using (tree)
            {
                using var output = new StreamReader(tree.Output, Encoding.UTF8);
                string said = await output.ReadLineAsync().WaitAsync(wait) ?? "";
                Check(said.StartsWith("grandchild ", StringComparison.Ordinal), $"the child started a grandchild ({said})");
                int grandchild = int.Parse(said["grandchild ".Length..], CultureInfo.InvariantCulture);
                tree.Stop();
                var stopped = await tree.Completion.WaitAsync(wait);
                Check(stopped is { Stopped: true, ExitCode: null, TimedOut: false }, $"Stop: stopped, no exit code ({stopped})");
                bool gone = false;
                for (int i = 0; i < 200 && !gone; i++)
                {
                    gone = Ended(grandchild);
                    if (!gone)
                    {
                        await Task.Delay(50);
                    }
                }

                Check(gone, $"the grandchild {grandchild} ended with its parent (the process tree)");
                tree.Stop();                                           // again, after its end: nothing
            }

            var slow = (await sandbox.StartAsync(Child("sleep"), root, new CommandLimits { Time = TimeSpan.FromMilliseconds(300) })).Session!;
            await using (slow)
            {
                var timed = await slow.Completion.WaitAsync(wait);
                Check(timed is { TimedOut: true, ExitCode: null, Stopped: false }, $"the time limit stops it ({timed})");
            }

            var disposed = (await sandbox.StartAsync(Child("sleep"), root, new CommandLimits { Time = Timeout.InfiniteTimeSpan })).Session!;
            int pid = disposed.ProcessId;
            await disposed.DisposeAsync();
            Check(disposed.Completion.IsCompleted && disposed.Completion.Result.Stopped && Ended(pid), "a dispose of a running command stops it");

            Check((await sandbox.StartAsync(["dotnet", "--version"], root, new CommandLimits { Programs = ["git"] })) is { Started: false } notAllowed
                  && notAllowed.Refusal!.Contains("not an allowed program", StringComparison.Ordinal), "a program not allowed is a refused start");
            Check((await sandbox.StartAsync(["coding-test-no-such-program"], root, limits)) is { Started: false } missing
                  && missing.Refusal!.Contains("not installed", StringComparison.Ordinal), "a program not on the PATH is a refused start");
            ThrowsNaming<NotSupportedException>(() => sandbox.StartAsync(Child("echo"), root, new CommandLimits { Network = false }).AsTask().GetAwaiter().GetResult(),
                "an interactive start with the network off on process", "container");
            using (var cancelled = new CancellationTokenSource())
            {
                cancelled.Cancel();
                ThrowsNaming<OperationCanceledException>(() => sandbox.StartAsync(Child("echo"), root, limits, cancelled.Token).AsTask().GetAwaiter().GetResult(), "a cancelled start");
            }

            // A sandbox without an interactive start refuses, saying why and naming one that can.
            foreach (ICommandSandbox without in new ICommandSandbox[] { new RecordingSandbox(new CommandResult(0, false, "", TimeSpan.Zero, false)), new ContainerSandbox(new ContainerSandboxOptions { Image = "coding-test:1" }) })
            {
                Check(without.CheckStart(limits) is { } why && why.Contains("cannot start a command interactively", StringComparison.Ordinal) && why.Contains(sandbox.Name, StringComparison.Ordinal),
                    $"{without.Name}: CheckStart says it cannot, naming process");
                ThrowsNaming<NotSupportedException>(() => without.StartAsync(["dotnet"], root, limits).AsTask().GetAwaiter().GetResult(), $"{without.Name}: an interactive start", "cannot start a command interactively");
            }
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }

        static bool Ended(int pid)
        {
            try
            {
                // An orphan whose parent was killed may stay a zombie where nothing reaps it (a container's first process): ended.
                string stat = $"/proc/{pid}/stat";
                if (OperatingSystem.IsLinux() && File.Exists(stat) && File.ReadAllText(stat).Contains(") Z ", StringComparison.Ordinal))
                {
                    return true;
                }

                using var process = Process.GetProcessById(pid);
                return process.HasExited;
            }
            catch (IOException)
            {
                return true;                                           // its /proc entry went while it was read
            }
            catch (ArgumentException)
            {
                return true;                                           // no such process any more
            }
            catch (InvalidOperationException)
            {
                return true;
            }
        }
    }

    // The children of the interactive start's test (Program.cs: --sandbox-child): echo, tree, sleep.
    internal static int SandboxChild(string scenario)
    {
        switch (scenario)
        {
            case "echo":
                Console.Out.WriteLine("ready " + Environment.GetEnvironmentVariable("IDRAK_SANDBOX_TEST"));
                Console.Out.Flush();
                using (var input = new StreamReader(Console.OpenStandardInput(), Encoding.UTF8))
                using (var output = new StreamWriter(Console.OpenStandardOutput(), new UTF8Encoding(false)) { AutoFlush = true })
                using (var error = new StreamWriter(Console.OpenStandardError(), new UTF8Encoding(false)) { AutoFlush = true })
                {
                    while (input.ReadLine() is { } line)
                    {
                        if (line.StartsWith("err ", StringComparison.Ordinal))
                        {
                            error.WriteLine("error: " + line[4..]);
                        }
                        else
                        {
                            output.WriteLine("echo: " + line);
                        }
                    }

                    output.WriteLine("bye");
                }

                return 3;
            case "tree":
                var (file, prefix) = LocalhostCluster.ProgramOf(typeof(DesktopTests).Assembly);
                var start = new ProcessStartInfo(file) { UseShellExecute = false };
                foreach (var word in prefix)
                {
                    start.ArgumentList.Add(word);
                }

                start.ArgumentList.Add("--sandbox-child");
                start.ArgumentList.Add("sleep");
                using (var grandchild = Process.Start(start)!)
                {
                    Console.Out.WriteLine("grandchild " + grandchild.Id.ToString(CultureInfo.InvariantCulture));
                    Console.Out.Flush();
                    Thread.Sleep(Timeout.Infinite);
                }

                return 0;
            case "sleep":
                Thread.Sleep(Timeout.Infinite);
                return 0;
            default:
                Console.Error.WriteLine("unknown sandbox child: " + scenario);
                return 2;
        }
    }

    private static void CommandRequester()
    {
        Check(new CommandLimits().Requester == "the agent", $"the default requester: {new CommandLimits().Requester}");
        Check(new CommandLimits().Purpose is null, "no purpose by default: the command line says it");
        var npm = new CommandLimits { Requester = "the npm plug-in", Purpose = "the script build: ng build", Programs = ["dotnet"] };
        Check((npm with { Time = TimeSpan.FromSeconds(30) }) is { Requester: "the npm plug-in", Purpose: "the script build: ng build" }, "a copy keeps the requester and the purpose given");
        Check(npm != new CommandLimits { Programs = npm.Programs }, "the requester is part of the limits' equality");
        string root = CodingTemp();
        Directory.CreateDirectory(root);
        try
        {
            var sandbox = new ProcessCommandSandbox { PollInterval = TimeSpan.FromMilliseconds(1) };
            var asAgent = sandbox.RunAsync(["dotnet", "--version"], root, npm with { Requester = "the agent", Purpose = null }).AsTask().GetAwaiter().GetResult();
            var asNpm = sandbox.RunAsync(["dotnet", "--version"], root, npm).AsTask().GetAwaiter().GetResult();
            Check(asAgent.Succeeded && asNpm.Succeeded && asAgent.Output == asNpm.Output, $"the same run whoever asks: {asAgent.Output} / {asNpm.Output}");
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
    }
}
