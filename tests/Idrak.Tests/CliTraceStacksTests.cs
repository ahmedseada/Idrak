// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Cli;
using Idrak.Cli.Commands.Developer.Stacks;

// idrak trace --stacks: the .NET runtime's trace of a run read and reported (NetTrace, StackProfile, StackReport). A child
// of this runner (--stacks-child) runs a known workload, so the report's answer is known; Microsoft's own reader
// (dotnet-trace, where installed) reads the same file as the reference; the command itself runs a child (--cli-child).
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] TraceStacksGroup =
    [
        ("cli dev: trace --stacks reads the runtime's trace of a known run: each spinning method's own time and its caller, the allocations by type and the code that made them, the collections, the printed names (lambdas, local functions, state machines, generic arguments), and a damaged file refused naming what is wrong", TraceStacksKnown),
        ("cli dev: trace --stacks reads a trace as Microsoft's reader does (dotnet-trace convert, where installed): the run's thread time within 2% and every one of its ten largest methods' own time within 1% of it", TraceStacksReference),
        ("cli dev: trace --stacks runs a command in a child process and reports it as tables and as JSON (focus, timeline, threads, a comparison, the speedscope file), and refuses a stray or mixed option", TraceStacksCommand),
    ];

    // The known run's lengths: the main thread and three workers each spin this long; 64 arrays of 1 MiB; 3 collections.
    private const int StacksSpinMs = 700, StacksWorkers = 3, StacksArrays = 64, StacksCollections = 3;

    /// <summary>The traced child: known hot methods (--stacks-child spin).</summary>
    internal static int StacksChild(string scenario)
    {
        if (scenario != "spin")
        {
            Console.Error.WriteLine($"unknown stacks scenario '{scenario}'");
            return 2;
        }

        var workers = Enumerable.Range(0, StacksWorkers).Select(_ => new Thread(() => SpinWorkerForStacks(StacksSpinMs)) { IsBackground = true }).ToList();
        workers.ForEach(t => t.Start());
        double sink = SpinForStacks(StacksSpinMs);
        workers.ForEach(t => t.Join());
        long kept = AllocateForStacks(StacksArrays);
        for (int i = 0; i < StacksCollections; i++)
        {
            GC.Collect();
        }

        Console.WriteLine($"stacks child: {sink:F0} {kept}");
        return 0;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static double SpinForStacks(int ms)
    {
        var clock = Stopwatch.StartNew();
        double x = 1;
        while (clock.ElapsedMilliseconds < ms)
        {
            for (int i = 0; i < 20000; i++)
            {
                x = x * 1.0000001 + 1e-9;
            }
        }

        return x;
    }

    // Not a tail call (its frame stays on the stack, as the caller check needs).
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static double SpinWorkerForStacks(int ms)
    {
        double x = SpinForStacks(ms);
        return x + 1;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static long AllocateForStacks(int count)
    {
        long sum = 0;
        for (int i = 0; i < count; i++)
        {
            var array = new float[1 << 18];
            array[i] = i;
            sum += array.Length + (long)array[i];
        }

        return sum;
    }

    // Records the known run once a test run (the traces are read by the three tests).
    private static string? _stacksTrace;

    private static string StacksTrace()
    {
        if (_stacksTrace is { } made)
        {
            return made;
        }

        string path = Path.Combine(DevTemp("stacks"), "spin.nettrace");
        var output = new StringWriter();
        var error = new StringWriter();
        int code = StackRecorder.Record(StackRecorder.ThisProgram(["--stacks-child", "spin"]), path, output, error);
        Check(code == 0 && output.ToString().Contains("stacks child:", StringComparison.Ordinal), $"the traced child: exit {code}\n{output}\n{error}");
        return _stacksTrace = path;
    }

    private static void TraceStacksKnown(Device device)
    {
        _ = device;
        string path = StacksTrace();
        var trace = NetTrace.Read(path);
        var p = StackProfile.Of(trace);
        Check(trace.Listed, "the trace holds the runtime's end-of-run method listing");
        double Own(string method, Func<Moment, bool>? which = null) =>
            trace.Seconds(p.Moments.Where(m => m.Frames.Length > 0 && p.Names[m.Frames[0]].EndsWith(method, StringComparison.Ordinal) && (which?.Invoke(m) ?? true)).Sum(m => m.Length));

        // The main thread's spin, and the workers' (theirs through SpinWorkerForStacks: its caller, not its own time).
        double spin = Own(".SpinForStacks"), running = Own(".SpinForStacks", m => m.Kind == TimeKind.Running);
        double expected = (1 + StacksWorkers) * StacksSpinMs / 1000.0;
        Check(spin >= 0.75 * expected && spin <= 1.25 * expected, $"SpinForStacks's own time {spin:F3} s (expected about {expected:F2} s: {1 + StacksWorkers} threads × {StacksSpinMs} ms)");
        Check(running >= 0.9 * spin, $"SpinForStacks runs .NET code: {running:F3} of {spin:F3} s running");
        var callers = p.Moments.Where(m => m.Frames.Length > 1 && p.Names[m.Frames[0]].EndsWith(".SpinForStacks", StringComparison.Ordinal))
            .GroupBy(m => p.Names[m.Frames[1]]).ToDictionary(g => g.Key, g => trace.Seconds(g.Sum(m => m.Length)));
        double fromMain = callers.Where(c => c.Key.EndsWith(".StacksChild", StringComparison.Ordinal)).Sum(c => c.Value);
        double fromWorkers = callers.Where(c => c.Key.EndsWith(".SpinWorkerForStacks", StringComparison.Ordinal)).Sum(c => c.Value);
        Check(fromMain >= 0.75 * StacksSpinMs / 1000.0 && fromWorkers >= 0.75 * StacksWorkers * StacksSpinMs / 1000.0,
            $"SpinForStacks called from StacksChild {fromMain:F3} s and from SpinWorkerForStacks {fromWorkers:F3} s ({string.Join(", ", callers.Select(c => $"{c.Key} {c.Value:F3}"))})");
        int spinThreads = p.Moments.Where(m => m.Frames.Length > 0 && p.Names[m.Frames[0]].EndsWith(".SpinForStacks", StringComparison.Ordinal)).Select(m => m.Thread).Distinct().Count();
        Check(spinThreads == 1 + StacksWorkers, $"SpinForStacks on {spinThreads} threads (expected {1 + StacksWorkers})");

        // The allocations: the 1 MiB float arrays, made in AllocateForStacks (the first frame of the program's own code).
        long arrays = p.Allocations.Where(a => a.Report.Type == "System.Single[]" && a.Frames.Any(f => p.Names[f].EndsWith(".AllocateForStacks", StringComparison.Ordinal))).Sum(a => a.Report.Bytes);
        Check(arrays >= 0.75 * StacksArrays * (1L << 20), $"the float arrays reported as made in AllocateForStacks: {arrays / 1048576.0:F1} MiB (made {StacksArrays} MiB)");
        int collections = trace.Suspensions.Count(s => s.ForCollection);
        Check(collections >= StacksCollections, $"{collections} collections' suspensions (the run asked {StacksCollections})");
        long busy = p.Moments.Where(m => m.Kind is TimeKind.Running or TimeKind.Outside or TimeKind.Spinning).Sum(m => m.Length);
        long unknown = p.Moments.Where(m => m.Kind is TimeKind.Running or TimeKind.Outside or TimeKind.Spinning && m.Frames.Length > 0 && m.Frames[0] == 0).Sum(m => m.Length);
        Check(unknown <= 0.05 * busy, $"busy time in unknown code: {trace.Seconds(unknown):F3} of {trace.Seconds(busy):F3} s");

        // The printed names.
        foreach (var (type, method, display) in new[]
        {
            ("Idrak.Abstraction.Devices.Cpu.CpuBackend+<>c__DisplayClass433_0", "<StoredMatMulKernel>b__0", "CpuBackend.StoredMatMulKernel (lambda)"),
            ("Idrak.Abstraction.Devices.Cpu.CpuMatMul+<>c__DisplayClass14_0", "<Tiled>g__RunTile|0", "CpuMatMul.Tiled.RunTile"),
            ("System.Collections.Generic.List`1[System.Double]", "Grow", "List`1[Double].Grow"),
            ("Idrak.Cli.Program+<Main>d__0", "MoveNext", "Program.Main (async)"),
            ("System.Threading.Tasks.Parallel+<>c__DisplayClass19_0`2[System.__Canon,System.Int32]", "<ForWorker>b__1", "Parallel.ForWorker (lambda) [__Canon,Int32]"),
        })
        {
            string got = StackProfile.DisplayName(type, method);
            Check(got == display, $"{type}.{method} prints as '{got}', not '{display}'");
        }

        // A damaged file: cut short, not a trace, a format version this reader does not read.
        byte[] bytes = File.ReadAllBytes(path);
        string folder = Path.GetDirectoryName(path)!;
        void Refused(string name, byte[] content, string says)
        {
            string file = Path.Combine(folder, name);
            File.WriteAllBytes(file, content);
            try
            {
                _ = NetTrace.Read(file);
                Check(false, $"{name}: read without an error");
            }
            catch (InvalidDataException e)
            {
                Check(e.Message.Contains(says, StringComparison.Ordinal), $"{name}: '{e.Message}' does not say '{says}'");
            }
        }

        Refused("cut.nettrace", bytes[..(bytes.Length / 2)], "the file ends inside");
        Refused("other.nettrace", [.. "Notatrace"u8, .. bytes[9..]], "is not a .nettrace file");
        var newer = (byte[])bytes.Clone();
        BitConverter.GetBytes(9).CopyTo(newer, 35);                               // the Trace object's version: after the magic, the header and three tags
        Refused("newer.nettrace", newer, "trace format version 9");
    }

    private static void TraceStacksReference(Device device)
    {
        _ = device;
        string path = StacksTrace();
        string reference = Path.Combine(Path.GetDirectoryName(path)!, "reference");
        Process? convert;
        try
        {
            var info = new ProcessStartInfo("dotnet-trace") { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false };
            foreach (string a in new[] { "convert", path, "--format", "speedscope", "-o", reference })
            {
                info.ArgumentList.Add(a);
            }

            convert = Process.Start(info);
        }
        catch (System.ComponentModel.Win32Exception)
        {
            Skip("dotnet-trace is not installed (dotnet tool install --global dotnet-trace): the reference reader is Microsoft's");
            return;
        }

        string said = convert!.StandardOutput.ReadToEnd() + convert.StandardError.ReadToEnd();
        convert.WaitForExit();
        string file = reference + ".speedscope.json";
        Check(convert.ExitCode == 0 && File.Exists(file), $"dotnet-trace convert: exit {convert.ExitCode}\n{said}");

        // The reference's own time a method: the frame under its marker frames (CPU_TIME, UNMANAGED_CODE_TIME), by name
        // without the module and the parameters.
        var json = JsonNode.Parse(File.ReadAllText(file))!;
        var frames = json["shared"]!["frames"]!.AsArray().Select(f => Plain((string)f!["name"]!)).ToArray();
        var expected = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (var profile in json["profiles"]!.AsArray())
        {
            var stack = new List<int>();
            double? last = null;
            foreach (var e in profile!["events"]!.AsArray())
            {
                double at = (double)e!["at"]!;
                if (stack.Count > 0 && last is { } before)
                {
                    string top = frames[stack[^1]];
                    string leaf = top is "CPU_TIME" or "UNMANAGED_CODE_TIME" ? stack.Count > 1 ? frames[stack[^2]] : "-" : top;
                    expected[leaf] = expected.GetValueOrDefault(leaf) + (at - before);
                }

                last = at;
                if ((string)e["type"]! == "O")
                {
                    stack.Add((int)e["frame"]!);
                }
                else
                {
                    stack.RemoveAt(stack.Count - 1);
                }
            }
        }

        var p = StackProfile.Of(NetTrace.Read(path));
        var ours = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (var m in p.Moments)
        {
            string leaf = m.Raw.Length > 0 ? p.Names[m.Raw[0]] : "-";                // as the trace has it (the reference keeps a GC poll's frames)
            ours[leaf] = ours.GetValueOrDefault(leaf) + m.Length * 1000.0 / p.Trace.Frequency;
        }

        double total = expected.Values.Sum(), mine = ours.Values.Sum();
        Check(Math.Abs(mine - total) <= 0.02 * total, $"thread time {mine:F1} ms here, {total:F1} ms by dotnet-trace");
        foreach (var (method, ms) in expected.OrderByDescending(x => x.Value).Take(10))
        {
            double here = ours.GetValueOrDefault(method);
            Check(Math.Abs(here - ms) <= 0.01 * total, $"{method}: own time {here:F1} ms here, {ms:F1} ms by dotnet-trace (of {total:F1} ms)");
        }

        static string Plain(string name)
        {
            name = name[(name.IndexOf('!', StringComparison.Ordinal) + 1)..];
            int open = name.IndexOf('(', StringComparison.Ordinal);
            return open > 0 ? name[..open] : name;
        }
    }

    private static void TraceStacksCommand(Device device)
    {
        _ = device;
        string folder = DevTemp("stacks-command");
        var launcher = StackRecorder.Launcher;
        StackRecorder.Launcher = args => StackRecorder.ThisProgram(["--cli-child", .. args]);
        try
        {
            string path = Path.Combine(folder, "version.nettrace");
            var (code, output, error) = DevRun("trace", "--stacks", "-o", path, "--", "version");
            Check(code == 0 && output.Contains("trace: version exited with 0", StringComparison.Ordinal) && output.Contains("where the threads' time went", StringComparison.Ordinal)
                  && output.Contains("methods by their own time", StringComparison.Ordinal) && output.Contains("allocations (", StringComparison.Ordinal)
                  && output.Contains("files: the trace", StringComparison.Ordinal), $"trace --stacks -- version: exit {code}\n{output}\n{error}");
            Check(File.Exists(path) && File.Exists(Path.ChangeExtension(path, ".speedscope.json")), "the trace and its speedscope file are written");

            var json = DevJson("trace", "--read", path, "--focus", "CommandLine", "--timeline", "--by-thread", "--compare", StacksTrace(), "-j");
            var stacks = json["stacks"]!;
            Check((int)json["exitCode"]! == 0 && (double)stacks["seconds"]! > 0 && stacks["byOwnTime"]!.AsArray().Count > 0, $"the JSON report: {json}");
            Check((double)stacks["focus"]!["ms"]! > 0 && stacks["focus"]!["calledFrom"]!.AsArray().Count > 0, $"--focus CommandLine: {stacks["focus"]}");
            Check(stacks["timeline"]!.AsArray().Count == 20 && stacks["byThread"]!.AsArray().Count > 0, "--timeline gives 20 slices, --by-thread the threads");
            Check(stacks["compare"]!["methods"]!.AsArray().Count > 0, $"--compare: {stacks["compare"]}");

            var viewer = JsonNode.Parse(File.ReadAllText(Path.ChangeExtension(path, ".speedscope.json")))!;
            Check(viewer["profiles"]!.AsArray().Count > 0 && viewer["profiles"]![0]!["samples"]!.AsArray().Count == viewer["profiles"]![0]!["weights"]!.AsArray().Count,
                "the speedscope file has a sampled profile a thread, a weight a sample");

            (code, output, _) = DevRun("trace", "--read", path, "--from", "0", "--to", "100", "--waits", "--all-frames", "--top", "5");
            Check(code == 0 && output.Contains("methods by their own time (all time", StringComparison.Ordinal), $"a window, all time, every frame:\n{output}");

            foreach (var (args, why) in new (string[], string)[]
            {
                (["trace", "--focus", "Main", "version"], "a stacks option without --stacks"),
                (["trace", "--stacks"], "no command"),
                (["trace", "--read", path, "version"], "--read with a command"),
                (["trace", "--stacks", "--levels", "layers", "--", "version"], "--stacks with the telemetry's --levels"),
                (["trace", "--read", path, "--from", "soon"], "--from not in seconds"),
                (["trace", "--read", path, "--from", "3", "--to", "2"], "--to before --from"),
                (["trace", "--read", path, "--top", "0"], "--top 0"),
            })
            {
                Check(DevRun(args).Code == 2, $"{why}: a usage error");
            }

            (code, _, error) = DevRun("trace", "--read", Path.Combine(folder, "none.nettrace"));
            Check(code == 1 && error.Contains("No trace at", StringComparison.Ordinal), $"a missing trace: exit {code}: {error}");
        }
        finally
        {
            StackRecorder.Launcher = launcher;
            Directory.Delete(folder, recursive: true);
        }
    }
}
