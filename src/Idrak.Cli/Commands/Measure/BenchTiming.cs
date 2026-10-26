// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;

namespace Idrak.Cli.Commands.Measure;

/// <summary>
/// How a kernel benchmark warms up before it times a measurement (<c>idrak bench</c> without a model): warm-up rounds
/// of at least <see cref="Round"/> each run until the medians of the last <see cref="Rounds"/> rounds agree within
/// <see cref="Tolerance"/> and the last round compiled no method (the tiered JIT and on-stack replacement have
/// finished), or until <see cref="Budget"/> or <see cref="MaxCalls"/> is spent. Each is set by an option and an
/// environment variable (the option wins).
/// </summary>
/// <param name="Tolerance">How far apart the last rounds' medians may be, as a fraction of the smallest (0.05: 5%).</param>
/// <param name="Rounds">How many rounds in a row must agree.</param>
/// <param name="Round">The shortest round (a round is at least one call).</param>
/// <param name="Budget">The longest warm-up of one measurement; zero: no warm-up beyond the first (cold) call.</param>
/// <param name="MaxCalls">The most warm-up calls of one measurement.</param>
internal sealed record BenchSettle(double Tolerance, int Rounds, TimeSpan Round, TimeSpan Budget, int MaxCalls)
{
    /// <summary>The variable of <see cref="Tolerance"/>, in percent (<c>--settle</c>).</summary>
    public const string SettleVariable = "IDRAK_BENCH_SETTLE";

    /// <summary>The variable of <see cref="Rounds"/> (<c>--settle-rounds</c>).</summary>
    public const string RoundsVariable = "IDRAK_BENCH_SETTLE_ROUNDS";

    /// <summary>The variable of <see cref="Round"/> (<c>--warmup-round</c>).</summary>
    public const string RoundVariable = "IDRAK_BENCH_WARMUP_ROUND";

    /// <summary>The variable of <see cref="Budget"/> (<c>--warmup</c>).</summary>
    public const string BudgetVariable = "IDRAK_BENCH_WARMUP";

    /// <summary>The variable of <see cref="MaxCalls"/> (<c>--warmup-calls</c>).</summary>
    public const string CallsVariable = "IDRAK_BENCH_WARMUP_CALLS";

    /// <summary>5%, 3 rounds of at least 100 ms, at most 2 s or 100,000 calls (the time is the usual limit; the calls stay above three rounds of a 10 µs kernel).</summary>
    public static BenchSettle Default { get; } = new(0.05, 3, TimeSpan.FromMilliseconds(100), TimeSpan.FromSeconds(2), 100_000);
}

/// <summary>One timed measurement.</summary>
/// <param name="ColdMicros">The first call, in µs: what a first token pays (compiling methods, building and tuning kernels).</param>
/// <param name="Micros">The median of the timed calls after the warm-up, in µs.</param>
/// <param name="WarmupCalls">The warm-up calls between the first call and the timed ones.</param>
/// <param name="Settled">Whether the warm-up settled (false: it stopped at its budget or call limit).</param>
internal readonly record struct BenchTime(double ColdMicros, double Micros, int WarmupCalls, bool Settled);

/// <summary>The kernel benchmarks' timing: the first call apart (cold), a warm-up until the timing settles, then the median.</summary>
internal static class BenchTiming
{
    /// <summary>
    /// The method's name, kept in saved results (<c>timing</c>) so <c>--compare</c> can tell results timed another way
    /// (saved before it, without the field: the mean of the calls after one warm-up call).
    /// </summary>
    public const string Method = "settled-median";

    /// <summary>The statistic of the timed calls, as the results name it.</summary>
    public const string Statistic = "median";

    /// <summary>
    /// Times <paramref name="run"/>: its first call alone (cold), then warm-up rounds until <paramref name="settle"/>
    /// says the timing settled, then the median of <paramref name="repeats"/> calls. Every call is timed alone, the
    /// device drained (<paramref name="drain"/>) before and after it; <paramref name="between"/> runs untimed before
    /// each call (resetting state the call changes, e.g. a cache's length).
    /// </summary>
    /// <param name="timestamp">The clock (default <see cref="Stopwatch.GetTimestamp"/>); a test passes a fake one.</param>
    /// <param name="ticksPerSecond">The clock's ticks per second (default <see cref="Stopwatch.Frequency"/>).</param>
    /// <param name="compiledMethods">The methods compiled so far in the process (default the JIT's count).</param>
    public static BenchTime Measure(Action run, Action drain, int repeats, BenchSettle settle, Action? between = null,
        Func<long>? timestamp = null, double ticksPerSecond = 0, Func<long>? compiledMethods = null)
    {
        timestamp ??= Stopwatch.GetTimestamp;
        if (ticksPerSecond <= 0)
        {
            ticksPerSecond = Stopwatch.Frequency;
        }

        compiledMethods ??= static () => System.Runtime.JitInfo.GetCompiledMethodCount(currentThread: false);
        double Call()
        {
            between?.Invoke();
            drain();
            long start = timestamp();
            run();
            drain();
            return (timestamp() - start) * 1e6 / ticksPerSecond;
        }

        double cold = Call();
        int calls = 0;
        bool settled = false;
        double budget = settle.Budget.TotalSeconds * ticksPerSecond, round = settle.Round.TotalSeconds * ticksPerSecond;
        var medians = new List<double>();
        var times = new List<double>();
        long warmStart = timestamp();
        while (!settled && calls < settle.MaxCalls && timestamp() - warmStart < budget)
        {
            times.Clear();
            long compiled = compiledMethods(), roundStart = timestamp();
            do
            {
                times.Add(Call());
                calls++;
            }
            while (timestamp() - roundStart < round && calls < settle.MaxCalls);

            medians.Add(Median(times));
            settled = compiledMethods() == compiled && Agree(medians, settle.Rounds, settle.Tolerance);
        }

        times.Clear();
        for (int i = 0; i < Math.Max(1, repeats); i++)
        {
            times.Add(Call());
        }

        return new BenchTime(cold, Median(times), calls, settled);
    }

    /// <summary>Whether the last <paramref name="rounds"/> values lie within <paramref name="tolerance"/> of the smallest of them.</summary>
    public static bool Agree(IReadOnlyList<double> values, int rounds, double tolerance)
    {
        if (values.Count < Math.Max(1, rounds))
        {
            return false;
        }

        double low = double.MaxValue, high = double.MinValue;
        for (int i = values.Count - Math.Max(1, rounds); i < values.Count; i++)
        {
            low = Math.Min(low, values[i]);
            high = Math.Max(high, values[i]);
        }

        return high <= low * (1 + tolerance);
    }

    /// <summary>The median (the mean of the middle two for an even count); 0 for none. Sorts <paramref name="values"/>.</summary>
    public static double Median(List<double> values)
    {
        values.Sort();
        int n = values.Count;
        return n == 0 ? 0 : n % 2 == 1 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
    }
}
