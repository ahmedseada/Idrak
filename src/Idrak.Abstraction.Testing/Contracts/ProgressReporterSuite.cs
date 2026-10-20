// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The checks of a progress reporter (a factory registered in <see cref="ProgressReporters"/>), each case one piece of
/// work reported to a <see cref="StringWriter"/> with an interval of an hour (so only the first report and explicit
/// showings show anything):
/// <list type="bullet">
/// <item>it starts with the label, total and unit asked for and nothing done;</item>
/// <item>reports set the amount and advances add to it, from one thread or several at once;</item>
/// <item>between showings a report allocates nothing and does not ask for the detail text;</item>
/// <item>the next stage takes its label, total and unit and starts from nothing done;</item>
/// <item>erasing, finishing twice, clearing and disposing never throw, and nothing is written after the end;</item>
/// <item>a reporter that is not enabled writes nothing.</item>
/// </list>
/// </summary>
public sealed class ProgressReporterSuite : ContractSuite<Func<ProgressRequest, IProgressReporter>>
{
    private static readonly ProgressUnit[] Units = [ProgressUnit.Items, ProgressUnit.Bytes, ProgressUnit.Elapsed];

    /// <inheritdoc />
    public override string Name => "progress-reporter";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        foreach (var unit in Units)
        {
            foreach (long? total in new long?[] { null, 0, 100, 1L << 40 })
            {
                foreach (var (reports, threads) in new[] { (0, 1), (1, 1), (1000, 1), (4000, 4) })
                {
                    yield return Case(unit, total, reports, threads);
                }
            }
        }
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        long? total = random.Next(3) == 0 ? null : random.NextInt64(0, large ? 1L << 50 : 1L << 20);
        return Case(Units[random.Next(Units.Length)], total, random.Next(0, large ? 200_000 : 3000), random.Next(1, 5));
    }

    /// <inheritdoc />
    public override void Run(Func<ProgressRequest, IProgressReporter> implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var data = @case.Data;
        var unit = Enum.Parse<ProgressUnit>((string)data["unit"]!);
        long? total = (long?)data["total"];
        int reports = (int)data["reports"]!, threads = (int)data["threads"]!;
        var output = new StringWriter();
        var reporter = implementation(new ProgressRequest("work", total, unit, output, TimeSpan.FromHours(1)));
        try
        {
            checks.Check("starts as asked", reporter.Label == "work" && reporter.Total == total && reporter.Unit == unit && reporter.Done == 0,
                $"label '{reporter.Label}', total {reporter.Total}, unit {reporter.Unit}, done {reporter.Done}");

            if (threads == 1)
            {
                for (int i = 1; i <= reports; i++)
                {
                    reporter.Report(i);
                }

                checks.Check("reports set the amount", reporter.Done == reports, $"done {reporter.Done} after reporting {reports}");
            }
            else
            {
                Parallel.For(0, reports, new ParallelOptions { MaxDegreeOfParallelism = threads }, _ => reporter.Advance());
                checks.Check("advances from several threads add up", reporter.Done == reports, $"done {reporter.Done} after {reports} advances on {threads} threads");
            }

            // Between showings (the interval is an hour and the first report has shown, when any did): nothing allocated,
            // the detail not asked for.
            reporter.Report(reports);
            int asked = 0;
            reporter.Detail = () =>
            {
                Interlocked.Increment(ref asked);
                return "detail";
            };
            for (int i = 0; i < 200; i++)
            {
                reporter.Report(reports + i);                                    // warm: the methods' first calls are compiled
            }

            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 1000; i++)
            {
                reporter.Report(reports + i);
                reporter.Advance(0);
            }

            long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            checks.Check("a report between showings allocates nothing", allocated == 0, $"{allocated} bytes over 2,000 reports");
            checks.Check("the detail is asked for only when shown", Volatile.Read(ref asked) == 0 || unit == ProgressUnit.Elapsed,
                $"asked {asked} times between showings");

            reporter.NextStage("next", 50, ProgressUnit.Items, keepLine: false);
            checks.Check("the next stage starts from nothing done", reporter.Label == "next" && reporter.Total == 50 && reporter.Unit == ProgressUnit.Items && reporter.Done == 0,
                $"label '{reporter.Label}', total {reporter.Total}, unit {reporter.Unit}, done {reporter.Done}");
            reporter.Report(7);
            reporter.Show();
            reporter.Erase();
            reporter.Finish("finished");
            int written = output.ToString().Length;
            reporter.Finish();
            reporter.Report(9);
            reporter.Show();
            reporter.Clear();
            checks.Check("nothing is written after the end", output.ToString().Length == written, $"{output.ToString().Length - written} characters after Finish");
            checks.Check("a reporter that is not enabled writes nothing", reporter.Enabled || output.ToString().Length == 0, $"wrote '{output}'");
        }
        finally
        {
            reporter.Dispose();
        }
    }

    private static ContractCase Case(ProgressUnit unit, long? total, int reports, int threads) =>
        new($"{unit.ToString().ToLowerInvariant()}, total {total?.ToString(System.Globalization.CultureInfo.InvariantCulture) ?? "unknown"}, {reports} reports on {threads} thread(s)",
            new JsonObject { ["unit"] = unit.ToString(), ["total"] = total, ["reports"] = reports, ["threads"] = threads });
}
