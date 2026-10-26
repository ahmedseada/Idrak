// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Tests.Runner;

// Plan 34, M2: the runner the console program and an app share (TestRunner.Run): its filter and listing, its lines, the
// failures writer, the per-test events and the summary, and a test that needs the repository where there is none
// reported as not run here (counted apart, never a failure).
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] TestRunnerGroup =
    [
        ("test runner: the filter keeps the tests whose name holds any of its texts (ignoring case), in their order; the listing names those; no filter keeps every test", TestRunnerFilter),
        ("test runner: a run reports PASS, FAIL, SKIP and NOT RUN lines, the failures writer gets all but the passes, the events come per test, the summary counts not run here apart (per device too)", TestRunnerOutcomes),
        ("test runner: without the repository a test that reads it is not run here, naming what it needs (a repository path, the checkout, the test data); with a root and a data folder of its own the same tests run", TestRunnerRepository),
        ("test runner: a cancelled run runs nothing more; a test past the timeout is a HANG line, an event and the summary's hung test", TestRunnerStops),
    ];

    internal static readonly Action<Device> Nothing = _ => { };

    internal static void TestRunnerFilter(Device device)
    {
        _ = device;
        (string, Action<Device>)[] tests = [("alpha: one", Nothing), ("beta: two", Nothing), ("Gamma: three", Nothing), ("alphabet: four", Nothing)];
        Check(TestRunner.ParseFilters("ALPHA|gam") is ["ALPHA", "gam"] && TestRunner.ParseFilters("") is null && TestRunner.ParseFilters(null) is null
              && TestRunner.ParseFilters("a||b") is ["a", "b"], "IDRAK_FILTER splits at |, empty texts dropped, nothing for none");
        Check(TestRunner.List(tests, ["ALPHA", "gam"]).SequenceEqual(["alpha: one", "Gamma: three", "alphabet: four"]), $"filtered: {string.Join("; ", TestRunner.List(tests, ["ALPHA", "gam"]))}");
        Check(TestRunner.List(tests, null).Count == 4 && TestRunner.List(tests, []).Count == 4 && TestRunner.List(tests, ["no such"]).Count == 0, "no filter keeps every test; a filter nothing matches keeps none");
        Check(TestRunner.Select(tests, ["beta"]) is [("beta: two", _)], "Select keeps the bodies");
        Check(TestSuite.All.Any(t => t.Name == "gradient: sigmoid, sum") && TestSuite.All.Any(t => t.Name.StartsWith("test runner: ", StringComparison.Ordinal))
              && TestSuite.All.Select(t => t.Name).Distinct(StringComparer.Ordinal).Count() == TestSuite.All.Count, $"the library's {TestSuite.All.Count} tests (the basic group's and the runner's among them) have names of their own");
    }

    // A nested run of `tests` on `device` (the outer run's settings for the repository put back after it).
    internal static (TestRunSummary Summary, string Output, string Failures, List<TestEvent> Events) NestedRun(Device device, (string, Action<Device>)[] tests,
        string? root, string? data, int timeout = 300, CancellationToken cancel = default)
    {
        string? outerRoot = TestPaths.GivenRoot, outerData = TestPaths.GivenData;
        var output = new StringWriter();
        var failures = new StringWriter();
        var events = new List<TestEvent>();
        try
        {
            var summary = TestRunner.Run(tests, new TestRunOptions
            {
                Devices = [device], Output = output, FailuresWriter = failures, Header = "header line", RepositoryRoot = root, DataFolder = data,
                TimeoutSeconds = timeout, Cancellation = cancel, Events = e => { lock (events) { events.Add(e); } },
            });
            return (summary, output.ToString().ReplaceLineEndings("\n"), failures.ToString().ReplaceLineEndings("\n"), events);
        }
        finally
        {
            TestPaths.Configure(outerRoot, outerData);
        }
    }

    internal static string RunnerFolder(bool repository)
    {
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-runner-{Guid.NewGuid():N}");
        Directory.CreateDirectory(Path.Combine(folder, "api"));
        File.WriteAllText(Path.Combine(folder, "api", "x.txt"), "x");
        if (repository)
        {
            File.WriteAllText(Path.Combine(folder, "Idrak.slnx"), "<Solution />");
            Directory.CreateDirectory(Path.Combine(folder, "tests", "Idrak.Tests", "data"));
            File.WriteAllText(Path.Combine(folder, "tests", "Idrak.Tests", "data", "d.txt"), "d");
        }

        return folder;
    }

    internal static void TestRunnerOutcomes(Device device)
    {
        string folder = RunnerFolder(repository: false);
        try
        {
            (string, Action<Device>)[] tests =
            [
                ("passes", Nothing),
                ("fails", _ => Check(false, "the reason")),
                ("skips", _ => Skip("written by a run")),
                ("needs the repository", _ => File.ReadAllText(RepositoryPath("api", "x.txt"))),
            ];
            var (summary, output, failures, events) = NestedRun(device, tests, folder, null);
            string[] lines = output.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            Check(lines.Length == 7 && lines[0] == "header line" && lines[1] == $"== {device}: {device.Name}" && lines[2].StartsWith("  PASS passes (", StringComparison.Ordinal)
                  && lines[3] == "  FAIL fails: the reason" && lines[4] == "  SKIP skips: written by a run"
                  && lines[5] == "  NOT RUN needs the repository: not run here: needs the repository (api/x.txt)"
                  && lines[6] == "1 passed, 1 failed, 1 skipped, 1 not run here", $"the run's lines:\n{output}");
            Check(failures == $"header line\n== {device}: {device.Name}\n  FAIL fails: the reason\n  NOT RUN needs the repository: not run here: needs the repository (api/x.txt)\n1 passed, 1 failed, 1 skipped, 1 not run here\n",
                $"the failures writer: every line but the passes and the skips:\n{failures}");
            Check(summary is { Passed: 1, Failed: 1, Skipped: 1, NotRunHere: 1, Hung: null, Cancelled: false, ExitCode: 1 }
                  && summary.Devices is [{ Passed: 1, Failed: 1, Skipped: 1, NotRunHere: 1 } counts] && counts.Device == device
                  && summary.Counts == "1 passed, 1 failed, 1 skipped, 1 not run here", $"the summary: {summary}");
            Check(events.Select(e => (e.Kind, e.Name)).SequenceEqual([
                    (TestEventKind.Started, "passes"), (TestEventKind.Passed, "passes"), (TestEventKind.Started, "fails"), (TestEventKind.Failed, "fails"),
                    (TestEventKind.Started, "skips"), (TestEventKind.Skipped, "skips"), (TestEventKind.Started, "needs the repository"), (TestEventKind.NotRunHere, "needs the repository")])
                  && events.All(e => e.Device == device) && events[3].Message == "the reason" && events[7].Message == "not run here: needs the repository (api/x.txt)",
                $"the events: {string.Join("; ", events.Select(e => $"{e.Kind} {e.Name}: {e.Message}"))}");

            // Not run here and skipped are not failures: a run of only those exits 0.
            var (quiet, _, _, _) = NestedRun(device, [tests[0], tests[2], tests[3]], folder, null);
            Check(quiet is { Failed: 0, ExitCode: 0, NotRunHere: 1 } && quiet.Counts == "1 passed, 0 failed, 1 skipped, 1 not run here", $"without failures: {quiet}");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    internal static void TestRunnerRepository(Device device)
    {
        string bare = RunnerFolder(repository: false), checkout = RunnerFolder(repository: true), data = Path.Combine(bare, "data");
        Directory.CreateDirectory(data);
        File.WriteAllText(Path.Combine(data, "d.txt"), "carried");
        try
        {
            (string, Action<Device>)[] tests =
            [
                ("root", _ => Check(File.Exists(Path.Combine(RepositoryRoot(), "Idrak.slnx")), "the root holds Idrak.slnx")),
                ("path", _ => Check(File.ReadAllText(RepositoryPath("api", "x.txt")) == "x", "a repository path")),
                ("data", _ => Check(File.ReadAllText(TestData("d.txt")).Length > 0, "a data file")),
                ("data folder", _ => Check(Directory.Exists(DataFolder()), "the data folder")),
            ];

            // No repository and no data folder: every one is not run here, naming what it needed.
            var (none, _, _, events) = NestedRun(device, tests, bare, null);
            var reasons = events.Where(e => e.Kind == TestEventKind.NotRunHere).Select(e => e.Message).ToList();
            Check(none is { NotRunHere: 4, Passed: 0, Failed: 0 } && reasons.Count == 4
                  && reasons[0] == $"not run here: needs the repository (the checkout, read by {nameof(TestRunnerRepository)}; no Idrak.slnx in {bare})"
                  && reasons[1] == "not run here: needs the repository (api/x.txt)"
                  && reasons[2] == "not run here: needs the repository (tests/Idrak.Tests/data)" && reasons[3] == reasons[2], $"without the repository: {string.Join("; ", reasons)}");

            // A data folder of the run's own (as a phone carries): its files are read; one it lacks is not run here, naming it.
            var (carried, _, _, carriedEvents) = NestedRun(device, [tests[2], tests[3], ("missing", _ => TestData("gone/e.txt"))], bare, data);
            Check(carried is { Passed: 2, NotRunHere: 1, Failed: 0 }
                  && carriedEvents.Last().Message == $"not run here: needs the repository (tests/Idrak.Tests/data/gone/e.txt, not in the data folder {data})",
                $"a data folder of its own: {carried}; {carriedEvents.Last().Message}");

            // A checkout (Idrak.slnx at its root): every test runs; a file its data folder lacks fails, as on the desktop.
            var (all, output, _, _) = NestedRun(device, [.. tests, ("missing", _ => TestData("gone.txt"))], checkout, null);
            Check(all is { Passed: 4, Failed: 1, NotRunHere: 0 } && output.Contains("  FAIL missing: Test data 'gone.txt' not found in ", StringComparison.Ordinal), $"with the repository: {output}");

            // Wrapped (a type initializer's, a task's) it is still found.
            var wrapped = new TypeInitializationException("T", new AggregateException(new TestNeedsRepositoryException("x")));
            Check(TestNeedsRepositoryException.Find(wrapped)?.What == "x" && TestNeedsRepositoryException.Find(new InvalidOperationException("y")) is null, "found inside what wraps it");
        }
        finally
        {
            Directory.Delete(bare, true);
            Directory.Delete(checkout, true);
        }
    }

    internal static void TestRunnerStops(Device device)
    {
        using var cancel = new CancellationTokenSource();
        int ran = 0;
        (string, Action<Device>)[] tests = [("first", _ => { ran++; cancel.Cancel(); }), ("second", _ => ran++)];
        var (stopped, output, _, _) = NestedRun(device, tests, null, null, cancel: cancel.Token);
        Check(ran == 1 && stopped is { Cancelled: true, Passed: 1, Failed: 0 } && output.EndsWith("1 passed, 0 failed\n", StringComparison.Ordinal), $"cancelled after the first test: {stopped}, {ran} ran\n{output}");

        var (hung, hungOutput, hungFailures, events) = NestedRun(device, [("slow", _ => Thread.Sleep(3500)), ("after", Nothing)], null, null, timeout: 1);
        Check(hung is { Hung: var name, ExitCode: 3, Passed: 1 } && name == $"[{device}] slow" && hungOutput.Contains($"  HANG [{device}] slow: still running after 1 s\n", StringComparison.Ordinal)
              && hungFailures.Contains("  HANG ", StringComparison.Ordinal) && events.Any(e => e.Kind == TestEventKind.Hung && e.Name == "slow")
              && !events.Any(e => e.Name == "after"), $"a test past the timeout: {hung}\n{hungOutput}");
    }
}
