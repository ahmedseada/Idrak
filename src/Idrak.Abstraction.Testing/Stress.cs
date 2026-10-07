// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text;
using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Testing.Devices;

namespace Idrak.Abstraction.Testing;

/// <summary>How long and how hard <see cref="Stress"/> runs.</summary>
public sealed record StressOptions
{
    /// <summary>Stop after this long (a long run); with <see cref="Iterations"/> too, whichever comes first.</summary>
    public TimeSpan? Duration { get; init; }

    /// <summary>
    /// Stop after this many iterations (an operation call replayed, or a contract case run); 200 when neither this nor
    /// <see cref="Duration"/> is set.
    /// </summary>
    public int? Iterations { get; init; }

    /// <summary>Threads running iterations at once on the one implementation (or device).</summary>
    public int Threads { get; init; } = Math.Clamp(Environment.ProcessorCount, 1, 4);

    /// <summary>The seed of the generated inputs.</summary>
    public int Seed { get; init; } = 1;

    /// <summary>Large inputs as well as odd ones (a million elements, long texts, wide vocabularies).</summary>
    public bool Large { get; init; } = true;

    /// <summary>The bytes the memory in use (on the device, or the CPU) may grow by over the run; more is a failure (a leak).</summary>
    public long MemoryGrowthLimit { get; init; } = 1L << 20;

    /// <summary>Cancels the run itself after this long, to check that a run stopped half-way leaves no memory behind.</summary>
    public TimeSpan? CancelAfter { get; init; }

    /// <summary>The failing iterations reported; the others are counted.</summary>
    public int MaxFailures { get; init; } = 10;

    internal int IterationLimit => Iterations ?? (Duration is null ? 200 : int.MaxValue);
}

/// <summary>What a stress run found: the iterations run, the failures, the slowest cases and the memory before and after.</summary>
public sealed class StressReport
{
    internal StressReport(string subject, int iterations, int threads, TimeSpan elapsed, bool cancelled, long memoryBefore, long memoryAfter,
        IReadOnlyList<ConformanceFailure> failures, int failed, IReadOnlyList<(string Case, TimeSpan Time)> slowest)
    {
        Subject = subject;
        Iterations = iterations;
        Threads = threads;
        Elapsed = elapsed;
        Cancelled = cancelled;
        MemoryBefore = memoryBefore;
        MemoryAfter = memoryAfter;
        Failures = failures;
        Failed = failed;
        Slowest = slowest;
    }

    /// <summary>What ran.</summary>
    public string Subject { get; }

    /// <summary>The iterations run.</summary>
    public int Iterations { get; }

    /// <summary>The threads that ran them.</summary>
    public int Threads { get; }

    /// <summary>How long the run took.</summary>
    public TimeSpan Elapsed { get; }

    /// <summary>Whether the run was cancelled before its budget was spent.</summary>
    public bool Cancelled { get; }

    /// <summary>The bytes in use before the run (<see cref="MemoryUsage.InUse"/>).</summary>
    public long MemoryBefore { get; }

    /// <summary>The bytes in use after the run.</summary>
    public long MemoryAfter { get; }

    /// <summary>How much the memory in use grew.</summary>
    public long MemoryGrowth => MemoryAfter - MemoryBefore;

    /// <summary>The failures reported (at most <see cref="StressOptions.MaxFailures"/>, plus one for memory growth).</summary>
    public IReadOnlyList<ConformanceFailure> Failures { get; }

    /// <summary>The iterations that failed, reported or not.</summary>
    public int Failed { get; }

    /// <summary>The slowest cases, slowest first.</summary>
    public IReadOnlyList<(string Case, TimeSpan Time)> Slowest { get; }

    /// <summary>True when no iteration failed and the memory did not grow past the limit.</summary>
    public bool Passed => Failures.Count == 0;

    /// <summary>Throws a <see cref="ConformanceException"/> carrying the report when something failed.</summary>
    /// <exception cref="ConformanceException">Something failed.</exception>
    public void ThrowIfFailed()
    {
        if (!Passed)
        {
            throw new ConformanceException(ToString());
        }
    }

    /// <summary>Saves every failure that carries a case into <paramref name="folder"/>; returns the files written.</summary>
    public IReadOnlyList<string> SaveFailures(string folder) =>
        [.. Failures.Where(f => f.Repro is not null).Select(f => Regression.Save(Path.Combine(folder, Regression.FileName(f.Repro!)), f.Repro!))];

    /// <summary>The summary line, the failures and the slowest cases.</summary>
    public override string ToString()
    {
        var text = new StringBuilder();
        text.Append($"{Subject}: {(Passed ? "passed" : "FAILED")}, {Iterations} iterations on {Threads} threads in {Elapsed.TotalSeconds:F1} s"
                    + $"{(Cancelled ? " (cancelled)" : "")}; {Failed} failed; memory in use {MemoryBefore:N0} -> {MemoryAfter:N0} bytes");
        foreach (var failure in Failures)
        {
            text.Append("\n  FAIL ").Append(failure);
        }

        foreach (var (name, time) in Slowest)
        {
            text.Append($"\n  slow {time.TotalMilliseconds:F1} ms {name}");
        }

        return text.ToString();
    }
}

/// <summary>
/// Stress runs: an implementation driven by generated inputs (odd and large shapes, many threads at once, for a time or
/// a number of iterations) and compared with the library default as the conformance check compares, with the memory in
/// use watched for growth. A run can be cancelled; what it ran by then is reported, and its memory must still come back.
/// </summary>
public static class Stress
{
    /// <summary>
    /// Replays the device check's operation calls (recorded on the CPU from the random cases, large ones included when
    /// <see cref="StressOptions.Large"/>) on <paramref name="device"/> from several threads, in random order, comparing
    /// each with the CPU; then checks that the device's memory in use came back.
    /// </summary>
    public static StressReport Run(Device device, StressOptions? options = null, CancellationToken cancellation = default)
    {
        ArgumentNullException.ThrowIfNull(device);
        return Run(device.Backend, $"{device} ({device.Name})", options ?? new StressOptions(), cancellation);
    }

    /// <summary>The same as <see cref="Run(Device, StressOptions?, CancellationToken)"/> for a backend that is not registered as a device.</summary>
    public static StressReport Run(Backend backend, StressOptions? options = null, CancellationToken cancellation = default)
    {
        ArgumentNullException.ThrowIfNull(backend);
        return Run(backend, $"{backend.Kind} ({backend.Name})", options ?? new StressOptions(), cancellation);
    }

    /// <summary>
    /// Runs random cases of <paramref name="suite"/> on <paramref name="implementation"/> from several threads at once
    /// (the one implementation shared, as an app shares it), as the conformance check runs them; then checks that the
    /// memory in use on the suite's device came back.
    /// </summary>
    public static StressReport Run<T>(T implementation, ContractSuite<T> suite, StressOptions? options = null, CancellationToken cancellation = default)
        where T : class
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(suite);
        options ??= new StressOptions();
        var checks = new CaseChecks(suite.Name, options.MaxFailures);
        var device = suite.Device;
        var report = Loop($"{implementation.GetType().Name} ({suite.Name})", device.Backend, options, cancellation, (iteration, timings) =>
        {
            var @case = suite.RandomCase(new Random(Seeds.Mix(options.Seed, iteration)), options.Large);
            int before = checks.Failures.Count;
            var time = suite.RunOne(implementation, @case, checks);
            timings.Add((@case.Name, time));
            return checks.Failures.Count == before ? null : [];
        });
        return report.With(checks.Failures);
    }

    private static StressReport Run(Backend backend, string subject, StressOptions options, CancellationToken cancellation)
    {
        var failures = new List<ConformanceFailure>();
        var calls = DeviceConformance.Record(new DeviceCheckOptions { Seed = options.Seed, Large = options.Large, RandomRuns = 2 }, failures, out _);
        if (calls.Count == 0)
        {
            throw new InvalidOperationException("No operation call was recorded.");
        }

        var defaults = new DeviceCheckOptions();
        var order = new Random(options.Seed);
        int[] picks = [.. Enumerable.Range(0, Math.Min(options.IterationLimit, 1 << 20)).Select(_ => order.Next(calls.Count))];
        var report = Loop(subject, backend, options, cancellation, (iteration, timings) =>
        {
            var call = calls[iteration < picks.Length ? picks[iteration] : (int)((uint)Seeds.Mix(options.Seed, iteration) % (uint)calls.Count)];
            var started = Stopwatch.GetTimestamp();
            string? problem = DeviceConformance.Run(backend, call, defaults.ToleranceFor(call.Operation), out _);
            timings.Add(($"{call.Operation} [{call.Case}]", Stopwatch.GetElapsedTime(started)));
            return problem is null ? null : [new ConformanceFailure(call.Operation.Name, call.Case, problem, OperationCases.ToCase(call, backend.Kind, problem))];
        });
        return report.With([.. failures, .. report.Failures]);
    }

    // Runs `iteration` on the threads until the budget is spent or the run is cancelled, then measures the memory.
    private static StressReport Loop(string subject, Backend backend, StressOptions options, CancellationToken cancellation,
        Func<int, List<(string, TimeSpan)>, IReadOnlyList<ConformanceFailure>?> iteration)
    {
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(cancellation);
        if (options.CancelAfter is { } cancelAfter)
        {
            linked.CancelAfter(cancelAfter);
        }

        backend.Synchronize();
        long before = backend.GetMemoryUsage().InUse;
        var started = Stopwatch.GetTimestamp();
        int next = -1, failed = 0, limit = options.IterationLimit;
        var failures = new List<ConformanceFailure>();
        var timings = new List<(string Case, TimeSpan Time)>();
        var threads = Enumerable.Range(0, Math.Max(1, options.Threads)).Select(_ => new Thread(() =>
        {
            var mine = new List<(string, TimeSpan)>();
            while (!linked.IsCancellationRequested && (options.Duration is not { } duration || Stopwatch.GetElapsedTime(started) < duration))
            {
                int i = Interlocked.Increment(ref next);
                if (i >= limit)
                {
                    break;
                }

                IReadOnlyList<ConformanceFailure>? problems;
                try
                {
                    problems = iteration(i, mine);
                }
                catch (Exception ex)
                {
                    problems = [new ConformanceFailure("iteration", $"iteration {i}", $"threw {ex.GetType().Name}: {ex.Message}")];
                }

                if (problems is not null)
                {
                    lock (failures)
                    {
                        failed++;
                        failures.AddRange(problems.Take(Math.Max(0, options.MaxFailures - failures.Count)));
                    }
                }
            }

            lock (timings)
            {
                timings.AddRange(mine);
            }
        })).ToList();
        threads.ForEach(t => t.Start());
        threads.ForEach(t => t.Join());

        var elapsed = Stopwatch.GetElapsedTime(started);
        backend.Synchronize();
        long after = backend.GetMemoryUsage().InUse;
        if (after - before > options.MemoryGrowthLimit)
        {
            failures.Add(new ConformanceFailure("memory", subject, $"the memory in use grew by {after - before:N0} bytes over the run "
                                                                    + $"({before:N0} -> {after:N0}; the limit is {options.MemoryGrowthLimit:N0})"));
        }

        int run = Math.Min(Volatile.Read(ref next) + 1, limit);
        var slowest = timings.GroupBy(t => t.Case).Select(g => (g.Key, g.Max(t => t.Time))).OrderByDescending(t => t.Item2).Take(5).ToList();
        return new StressReport(subject, run, Math.Max(1, options.Threads), elapsed, linked.IsCancellationRequested && run < limit, before, after,
            failures, failed, slowest);
    }

    // The report with these failures (from the suite's checks) before its own (memory).
    private static StressReport With(this StressReport report, IReadOnlyList<ConformanceFailure> failures) =>
        new(report.Subject, report.Iterations, report.Threads, report.Elapsed, report.Cancelled, report.MemoryBefore, report.MemoryAfter,
            [.. failures, .. report.Failures.Where(f => !failures.Contains(f))], report.Failed, report.Slowest);
}
