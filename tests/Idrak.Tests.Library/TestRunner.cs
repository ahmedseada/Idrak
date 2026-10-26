// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Diagnostics;
using Idrak.Gpu.Cuda;

namespace Idrak.Tests.Runner;

/// <summary>What happened to one test on one device (<see cref="TestEvent.Kind"/>).</summary>
public enum TestEventKind
{
    /// <summary>The test started.</summary>
    Started,

    /// <summary>It passed.</summary>
    Passed,

    /// <summary>It failed (<see cref="TestEvent.Message"/> says why), or was counted as failed on a device lost earlier.</summary>
    Failed,

    /// <summary>Its data is not here (the owner's references, written by a tool's run); the message names the run that writes it.</summary>
    Skipped,

    /// <summary>It needs the repository (a source file, an API dump, a plan, the test data) and there is none here, as on a phone.</summary>
    NotRunHere,

    /// <summary>It was still running after the run's timeout.</summary>
    Hung,
}

/// <summary>One test's event: what happened (<paramref name="Kind"/>) to <paramref name="Name"/> on <paramref name="Device"/>.</summary>
/// <param name="Kind">What happened.</param>
/// <param name="Device">The device it ran on.</param>
/// <param name="Name">The test's name.</param>
/// <param name="Message">Why it failed, was skipped, was not run here or hung; null when it started or passed.</param>
/// <param name="Elapsed">How long it ran (zero when it started).</param>
public sealed record TestEvent(TestEventKind Kind, Device Device, string Name, string? Message, TimeSpan Elapsed);

/// <summary>The counts of one device's tests in a run.</summary>
/// <param name="Device">The device.</param>
/// <param name="Passed">Tests that passed.</param>
/// <param name="Failed">Tests that failed (with those counted as failed after the device was lost).</param>
/// <param name="Skipped">Tests whose data is not here.</param>
/// <param name="NotRunHere">Tests that need the repository, where there is none.</param>
public sealed record DeviceTestCounts(Device Device, int Passed, int Failed, int Skipped, int NotRunHere);

/// <summary>What a run did: its counts, as a whole and per device, the test that hung (if one did) and how long it took.</summary>
/// <param name="Passed">Tests that passed, over every device.</param>
/// <param name="Failed">Tests that failed.</param>
/// <param name="Skipped">Tests whose data is not here.</param>
/// <param name="NotRunHere">Tests that need the repository, where there is none.</param>
/// <param name="Hung">"[device] name" of the test still running after the timeout, or null.</param>
/// <param name="Cancelled">Whether the run was cancelled before its last test.</param>
/// <param name="Elapsed">How long the run took.</param>
/// <param name="Devices">The counts per device, in the order they ran.</param>
public sealed record TestRunSummary(int Passed, int Failed, int Skipped, int NotRunHere, string? Hung, bool Cancelled, TimeSpan Elapsed, IReadOnlyList<DeviceTestCounts> Devices)
{
    /// <summary>The console runner's exit code: 3 when a test hung, 1 when one failed, else 0 (not run here and skipped are not failures).</summary>
    public int ExitCode => Hung is not null ? 3 : Failed == 0 ? 0 : 1;

    /// <summary>"12 passed, 1 failed", with ", 2 skipped" and ", 3 not run here" when there are any: the runner's summary line.</summary>
    public string Counts => TestRunner.Counts(Passed, Failed, Skipped, NotRunHere);
}

/// <summary>
/// How <see cref="TestRunner.Run"/> runs: which tests (<see cref="Filters"/>) on which devices, where its lines go, the
/// failures file, the progress line, the timeout, where the repository and the test data are, and the per-test events.
/// Every setting has a default that the console runner (tests/Idrak.Tests) maps to its flags and variables.
/// </summary>
public sealed class TestRunOptions
{
    /// <summary>Texts a test's name must contain one of (ignoring case); null or empty runs every test (<c>IDRAK_FILTER</c>, split at "|").</summary>
    public IReadOnlyList<string>? Filters { get; init; }

    /// <summary>The devices, in order; null for every device found (<see cref="Device.Available"/>), with a line when CUDA is not there (<c>IDRAK_DEVICES</c>).</summary>
    public IReadOnlyList<Device>? Devices { get; init; }

    /// <summary>Seconds a test may run before it is reported as hung (<c>IDRAK_TIMEOUT</c>, default 300).</summary>
    public int TimeoutSeconds { get; init; } = 300;

    /// <summary>
    /// The repository's root (the folder holding Idrak.slnx); null to find it above the program's folder
    /// (<c>IDRAK_TEST_REPOSITORY</c>). A folder without Idrak.slnx, or none found, is no repository: a test that needs it
    /// is reported as not run here, naming what it needs.
    /// </summary>
    public string? RepositoryRoot { get; init; }

    /// <summary>The test data (tests/Idrak.Tests/data, or a copy a phone carries); null for the repository's (<c>IDRAK_TEST_DATA</c>).</summary>
    public string? DataFolder { get; init; }

    /// <summary>Where the runner's own lines go (the header, device headers, PASS, FAIL, SKIP, NOT RUN, HANG, the summary); null for the console's output.</summary>
    public TextWriter? Output { get; init; }

    /// <summary>The lock <see cref="Progress"/> draws under (the console reporter's: the console's error writer), so a line never lands inside the progress line; null for <see cref="Output"/>.</summary>
    public object? OutputLock { get; init; }

    /// <summary>Only the header, the device headers, the FAIL, HANG, SKIP and NOT RUN lines and the summary (no PASS lines, the tests' own lines dropped).</summary>
    public bool FailuresOnly { get; init; }

    /// <summary>A file the header, the device headers, the failures, the tests not run here and the summary are written to as they happen (its folder made); null for none.</summary>
    public string? FailuresFile { get; init; }

    /// <summary>A writer for the same lines as <see cref="FailuresFile"/> (an app's share sheet, a view); null for none.</summary>
    public TextWriter? FailuresWriter { get; init; }

    /// <summary>The live line: its label, total, amount and detail (the counts and the test running) are set by the runner; null for none. The runner clears it at the end.</summary>
    public IProgressReporter? Progress { get; init; }

    /// <summary>Where the tests' own console lines go while they run (a writer that keeps them off the progress line); null leaves the console's output as it is. Ignored with <see cref="FailuresOnly"/>, which drops them.</summary>
    public TextWriter? TestOutput { get; init; }

    /// <summary>The first line of the run and of the failures file (what the run is on); null for none.</summary>
    public string? Header { get; init; }

    /// <summary>The whole exception (its stack) in a FAIL line, not only its message (<c>IDRAK_TRACE=1</c>).</summary>
    public bool FullErrors { get; init; }

    /// <summary>
    /// The library's work as it happens (--verbose): made once for the run from a writer of the runner's lines, and
    /// disposed at its end; null for none. Each test then starts with a "test:" line.
    /// </summary>
    public Func<TextWriter, IDisposable>? Verbose { get; init; }

    /// <summary>One profile report per test (--profile), written as the runner's lines; null for none.</summary>
    public ProfileOptions? Profile { get; init; }

    /// <summary>Each test's events (started, passed, failed, skipped, not run here, hung), as they happen, on the thread that runs the tests (the watchdog's for hung).</summary>
    public Action<TestEvent>? Events { get; init; }

    /// <summary>
    /// Called from the watchdog when a test hangs, after its HANG line and the counts (the console runner ends the
    /// process with exit code 3); null to wait for the test, then end the run with <see cref="TestRunSummary.Hung"/> set.
    /// </summary>
    public Action<TestRunSummary>? Hung { get; init; }

    /// <summary>Stops the run before its next test.</summary>
    public CancellationToken Cancellation { get; init; }
}

/// <summary>
/// The test runner (plan 34, M2): the console program in tests/Idrak.Tests and an app both run test groups through
/// <see cref="Run"/>, with the same filter, devices, lines, failures file, timeout and counts.
/// </summary>
public static class TestRunner
{
    /// <summary>Splits an <c>IDRAK_FILTER</c> value at "|" (each text a filter of its own); null or empty for none.</summary>
    public static IReadOnlyList<string>? ParseFilters(string? text) => string.IsNullOrEmpty(text) ? null : text.Split('|', StringSplitOptions.RemoveEmptyEntries);

    /// <summary>The tests whose name contains one of <paramref name="filters"/> (ignoring case), in order; every test without filters.</summary>
    public static IReadOnlyList<(string Name, Action<Device> Run)> Select(IEnumerable<(string Name, Action<Device> Run)> tests, IReadOnlyList<string>? filters)
    {
        ArgumentNullException.ThrowIfNull(tests);
        return filters is not { Count: > 0 } ? [.. tests] : [.. tests.Where(t => filters.Any(f => t.Name.Contains(f, StringComparison.OrdinalIgnoreCase)))];
    }

    /// <summary>The names <see cref="Select"/> keeps, in order (the console runner's --list).</summary>
    public static IReadOnlyList<string> List(IEnumerable<(string Name, Action<Device> Run)> tests, IReadOnlyList<string>? filters) => [.. Select(tests, filters).Select(t => t.Name)];

    /// <summary>The summary line's counts: passed and failed, then the skipped and the not run here when there are any.</summary>
    public static string Counts(int passed, int failed, int skipped, int notRunHere)
    {
        string counts = $"{passed} passed, {failed} failed";
        if (skipped > 0)
        {
            counts += $", {skipped} skipped";
        }

        return notRunHere > 0 ? counts + $", {notRunHere} not run here" : counts;
    }

    /// <summary>
    /// Runs <paramref name="tests"/> (those <see cref="TestRunOptions.Filters"/> keeps) on every device of
    /// <paramref name="options"/>, one after another on the calling thread, each in a tensor scope: a line per test
    /// (PASS with its time, FAIL with the error, SKIP, NOT RUN when it needs the repository and there is none), an event
    /// per test, the progress line, and the counts at the end. A test still running after the timeout is reported as
    /// HANG. A CUDA context lost in a test (an illegal address, a launch failure) counts that device's later tests as failed.
    /// </summary>
    public static TestRunSummary Run(IEnumerable<(string Name, Action<Device> Run)> tests, TestRunOptions options)
    {
        ArgumentNullException.ThrowIfNull(tests);
        ArgumentNullException.ThrowIfNull(options);
        var selected = Select(tests, options.Filters);
        var results = options.Output ?? Console.Out;
        object screen = options.OutputLock ?? results;
        var bar = options.Progress;
        TestPaths.Configure(options.RepositoryRoot, options.DataFolder);

        string? failuresPath = options.FailuresFile;
        if (failuresPath is not null)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(failuresPath))!);
        }

        using var failuresFile = failuresPath is null ? null : new StreamWriter(failuresPath, append: false, new System.Text.UTF8Encoding(false)) { AutoFlush = true };
        var failuresWriter = options.FailuresWriter;

        // A line of the runner's own, with the progress line off the screen while it is written (the test's end shows it
        // again); one that is not a pass also goes to the failures file.
        void Report(string line, bool pass = false, bool toResults = true)
        {
            lock (screen)
            {
                bar?.Erase();
                if (toResults)
                {
                    results.WriteLine(line);
                    results.Flush();
                }

                if (!pass)
                {
                    failuresFile?.WriteLine(line);
                    failuresWriter?.WriteLine(line);
                    failuresWriter?.Flush();
                }
            }
        }

        if (options.Header is { } header)
        {
            Report(header);
        }

        // Every device: the CPU and each GPU found, unless the run names its own.
        var devices = options.Devices?.ToList();
        if (devices is null)
        {
            devices = [.. Device.Available];
            if (!Device.IsCudaAvailable)
            {
                Report($"CUDA not available ({CudaBackend.UnavailableReason}); testing {string.Join(", ", devices)}.", pass: true);
            }
        }

        long total = (long)selected.Count * devices.Count, done = 0;
        int failed = 0, passed = 0, skipped = 0, notRunHere = 0;
        string? running = null, hung = null;
        var perDevice = new List<DeviceTestCounts>();
        var runStarted = Stopwatch.StartNew();
        string Counts() => TestRunner.Counts(passed, failed, skipped, notRunHere);
        if (bar is not null)
        {
            bar.Label = global::Tests.ProgressLabel(0, total);
            bar.Total = total;
            bar.Detail = () => $"{Counts()}  {Volatile.Read(ref running)}";
        }

        // The tests' own lines: dropped when only failures show, else where the options say (around the progress line).
        var console = Console.Out;
        if (options.FailuresOnly)
        {
            Console.SetOut(TextWriter.Null);
        }
        else if (options.TestOutput is { } testOutput)
        {
            Console.SetOut(testOutput);
        }

        // The verbose lines name the thread they came from (work two threads do at once tells apart); the reports are the runner's.
        var runnerLines = new global::Tests.LineWriter(line => Report("    " + line));
        using var verboseLines = options.Verbose?.Invoke(new global::Tests.LineWriter(line => Report($"    [thread {Environment.CurrentManagedThreadId}] {line}")));
        bool namesTests = options.Verbose is not null || options.Profile is not null;

        void Ended()
        {
            long now = Interlocked.Increment(ref done);
            if (bar is not null)
            {
                bar.Label = global::Tests.ProgressLabel(now, total);
                bar.Advance();
            }
        }

        void Raise(TestEventKind kind, Device device, string name, string? message, TimeSpan elapsed) => options.Events?.Invoke(new TestEvent(kind, device, name, message, elapsed));

        TestRunSummary Summary(bool cancelled) =>
            new(passed, failed, skipped, notRunHere, Volatile.Read(ref hung), cancelled, runStarted.Elapsed, [.. perDevice]);

        // The time so far and the time left move during a long test too (Advance(0) shows at the reporter's own interval).
        using var clock = bar is null ? null : new Timer(_ => bar.Advance(0), null, 1000, 1000);
        var runningSince = Stopwatch.StartNew();
        Device? current = null;
        string? currentName = null;
        bool finished = false;
        var watchdog = new Thread(() =>
        {
            while (!Volatile.Read(ref finished))
            {
                Thread.Sleep(1000);
                if (Volatile.Read(ref running) is { } now && runningSince.Elapsed.TotalSeconds > options.TimeoutSeconds && Volatile.Read(ref hung) is null)
                {
                    Volatile.Write(ref hung, now);
                    bar?.Clear();
                    Report($"  HANG {now}: still running after {options.TimeoutSeconds} s");
                    Report($"{Counts()}, 1 hung");
                    Raise(TestEventKind.Hung, current!, currentName!, $"still running after {options.TimeoutSeconds} s", runningSince.Elapsed);
                    options.Hung?.Invoke(Summary(cancelled: false));
                    return;
                }
            }
        }) { IsBackground = true, Name = "test watchdog" };
        watchdog.Start();
        bool cancelled = false;
        try
        {
            foreach (var device in devices)
            {
                Report($"== {device}: {device.Name}");
                int devicePassed = passed, deviceFailed = failed, deviceSkipped = skipped, deviceNotRun = notRunHere;
                string? lostIn = null;
                foreach (var (name, test) in selected)
                {
                    if (options.Cancellation.IsCancellationRequested || Volatile.Read(ref hung) is not null)
                    {
                        cancelled = hung is null;
                        break;
                    }

                    if (lostIn is not null)
                    {
                        failed++;
                        Raise(TestEventKind.Failed, device, name, $"not run: the CUDA context was lost in {lostIn}", TimeSpan.Zero);
                        Ended();
                        continue;                                                            // counted, not run: the device is unusable
                    }

                    var sw = Stopwatch.StartNew();
                    current = device;
                    currentName = name;
                    runningSince.Restart();
                    Volatile.Write(ref running, $"[{device}] {name}");
                    if (namesTests)
                    {
                        Report($"  test: {name}");
                    }

                    Raise(TestEventKind.Started, device, name, null, TimeSpan.Zero);
                    var watch = options.Profile is null ? null : Profiler.Watch(name, device, options.Profile);
                    try
                    {
                        using (new TensorScope())
                        {
                            test(device);
                        }

                        passed++;
                        if (!options.FailuresOnly)
                        {
                            Report($"  PASS {name} ({sw.ElapsedMilliseconds} ms)", pass: true);
                        }

                        Raise(TestEventKind.Passed, device, name, null, sw.Elapsed);
                    }
                    catch (global::Tests.SkippedException ex)
                    {
                        skipped++;
                        Report($"  SKIP {name}: {ex.Message}", pass: true);
                        Raise(TestEventKind.Skipped, device, name, ex.Message, sw.Elapsed);
                    }
                    catch (Exception ex) when (TestNeedsRepositoryException.Find(ex) is { } needs)
                    {
                        notRunHere++;
                        Report($"  NOT RUN {name}: {needs.Message}");
                        Raise(TestEventKind.NotRunHere, device, name, needs.Message, sw.Elapsed);
                    }
                    catch (Exception ex)
                    {
                        failed++;
                        string message = options.FullErrors ? ex.ToString() : ex.Message;
                        Report($"  FAIL {name}: {message}");
                        Raise(TestEventKind.Failed, device, name, message, sw.Elapsed);

                        // A kernel that faults (illegal or misaligned address, illegal instruction, launch failure) ruins the CUDA
                        // context: every later call on it fails the same way. Name the test and skip the rest on this device.
                        if (device.Type == DeviceType.Cuda && ex.Message is var text
                            && (text.Contains("ILLEGAL", StringComparison.Ordinal) || text.Contains("MISALIGNED", StringComparison.Ordinal)
                                || text.Contains("LAUNCH_FAILED", StringComparison.Ordinal) || text.Contains("HARDWARE_STACK_ERROR", StringComparison.Ordinal)))
                        {
                            lostIn = name;
                            Report($"  the CUDA context was lost in this test: the remaining {device} tests are skipped and counted as failed.");
                            Report("  (the fault is reported where the device is next used; IDRAK_CUDA_DEBUG=1 names the kernel that faulted)");
                        }
                    }

                    watch?.End().Write(runnerLines);
                    Ended();
                }

                perDevice.Add(new DeviceTestCounts(device, passed - devicePassed, failed - deviceFailed, skipped - deviceSkipped, notRunHere - deviceNotRun));
                if (cancelled || hung is not null)
                {
                    break;
                }
            }
        }
        finally
        {
            Volatile.Write(ref finished, true);
            Volatile.Write(ref running, null);
            clock?.Dispose();
            bar?.Clear();
            Console.SetOut(console);
        }

        var summary = Summary(cancelled);
        if (summary.Hung is null)
        {
            Report(bar is null ? Counts() : $"{Counts()} in {ProgressText.Duration(runStarted.Elapsed)}");
        }

        return summary;
    }
}

/// <summary>
/// A test that needs the repository (a source file, an API dump, a plan, the test data) where there is none, as in an
/// app on a phone: thrown by the tests' repository and data lookups, and reported by <see cref="TestRunner"/> as not run
/// here (counted apart from passed and failed), never as a failure.
/// </summary>
/// <param name="what">What the test needs from the repository ("api/Idrak.txt", "tests/Idrak.Tests/data").</param>
public sealed class TestNeedsRepositoryException(string what) : Exception($"not run here: needs the repository ({what})")
{
    /// <summary>What the test needs from the repository.</summary>
    public string What { get; } = what;

    /// <summary>This exception in <paramref name="exception"/> or what it wraps (a type initializer's, a task's, a reflected call's); null when it is not there.</summary>
    public static TestNeedsRepositoryException? Find(Exception exception)
    {
        for (var e = exception; e is not null; e = e.InnerException)
        {
            if (e is TestNeedsRepositoryException needs)
            {
                return needs;
            }

            if (e is AggregateException aggregate)
            {
                foreach (var inner in aggregate.InnerExceptions)
                {
                    if (Find(inner) is { } found)
                    {
                        return found;
                    }
                }
            }
        }

        return null;
    }
}

/// <summary>The groups of this library (every group but the console runner's own, which need the desktop), and what a run of them needs first.</summary>
public static class TestSuite
{
    /// <summary>Every test of the library's groups, in the console runner's order.</summary>
    public static IReadOnlyList<(string Name, Action<Device> Run)> All => global::Tests.All;

    /// <summary>
    /// What every run registers first, as an app does: the text model families (Idrak.Text.Plugins; the library registers
    /// none), which a test of a family's registration takes away and puts back.
    /// </summary>
    public static void Prepare() => Idrak.Text.Plugins.TextPlugins.RegisterAll();
}
