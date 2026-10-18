// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Cluster;
using Idrak.Cli.Shared;
using Idrak.Cluster;

namespace Idrak.Cli.Commands.Cluster;

/// <summary>
/// <c>idrak cluster replay --topology FILE --seed S --step N [-- COMMAND …]</c> (plan 23, phase C8, over C3's
/// <see cref="ClusterReplay"/>): the fault schedule a simulated run of the seed draws through step N, and, with a training
/// command after <c>--</c>, that run again through step N (every rank sits out the later steps) with each rank's clock as
/// it began the step; <c>--from</c> compares both with a run's event log, bit for bit.
/// </summary>
internal sealed class ClusterReplayCommand : Command
{
    public override string Name => "cluster replay";

    public override string Summary => "Replays a simulated run's fault schedule and clock through a step, and reruns the step";

    public override string Usage => """
        --topology FILE --step N [--seed S] [--failures M] [--from LOG] [-- COMMAND ARGS]

        A simulated run draws its faults from (seed, step) alone and charges its virtual clock from each message's identity,
        so a run of the same seed through the same step is the same bit for bit. Without a command, prints the faults the
        failure model draws on the topology from the seed for steps 0 to N, with the replay key a fault's message names.
        With a training command after -- (as idrak cluster sim takes it, into a run folder of its own), runs it again
        through step N: every rank sits out the steps after it, and each rank's clock as it began step N is printed, read
        from the run's event log. --from LOG (a run's cluster-events.idrklog, which idrak cluster sim writes in its run
        folder) compares the faults and, after a rerun, every rank's clock through step N with that run's, bit for bit,
        and names the first that differs.

        Options:
              --topology FILE    the topology file the run used
              --step N           the step to replay through (IDRAK_CLUSTER_REPLAY_STEP)
              --failures MODEL   the failure model (the file's; IDRAK_CLUSTER_FAILURES)
              --from LOG         a run's event log to compare with (its seed, failure model and clocks)
              --event-log FILE   where the rerun's event log goes (its run folder's cluster-events.idrklog)
              --seed S           the run's seed (the file's; the log's with --from)

        Examples:
          idrak cluster replay --topology two.json --seed 7 --step 12
          idrak cluster replay --topology two.json --step 12 --from run/cluster-events.idrklog
          idrak cluster replay --topology two.json --seed 7 --step 3 --from run/cluster-events.idrklog -- pretrain my-model
              --tokens tokens.bin --steps 100 -o replay
        """;

    public override IReadOnlyCollection<string> ValueOptions => ["--topology", "--step", "--failures", "--from", "--event-log"];

    public override int Run(CommandContext context)
    {
        string path = context.Option("--topology") ?? throw new UsageException("Give the topology file with --topology (the one the run used).");
        long step = context.Option("--step") is { } stepText
            ? long.TryParse(stepText, NumberStyles.None, CultureInfo.InvariantCulture, out long s) ? s : throw new UsageException($"--step is a step number, 0 or more, not '{stepText}'.")
            : throw new UsageException("Give the step to replay through with --step N (a fault's message names it).");
        EventLog? original = null;
        if (context.Option("--from") is { } from)
        {
            try
            {
                original = EventLog.Load(from);
            }
            catch (Exception e) when (e is InvalidDataException or FileNotFoundException or DirectoryNotFoundException)
            {
                context.Error(e.Message);
                return ExitCodes.Failed;
            }
        }

        ClusterTopology topology;
        try
        {
            topology = ClusterTopology.Load(path);
        }
        catch (Exception e) when (e is InvalidDataException or NotSupportedException or FileNotFoundException or DirectoryNotFoundException)
        {
            context.Error(e.Message);
            return ExitCodes.Failed;
        }

        int seed = context.Seed ?? original?.Seed ?? topology.Simulation.Seed;
        string failures = context.Option("--failures") ?? original?.Failures ?? topology.Simulation.Failures;
        if (original is not null && (original.Seed != seed || !string.Equals(original.Failures, failures, StringComparison.OrdinalIgnoreCase) || original.WorldSize != topology.WorldSize))
        {
            throw new UsageException($"{context.Option("--from")} is a run of seed {original.Seed}, failures {original.Failures} on {original.WorldSize} rank(s); "
                                     + $"the replay asks seed {seed}, failures {failures} on {topology.WorldSize}: give the run's.");
        }

        IReadOnlyList<Fault> schedule;
        try
        {
            schedule = ClusterReplay.Schedule(topology, failures, seed, step);
        }
        catch (NotSupportedException e)
        {
            context.Error(e.Message);
            return ExitCodes.Failed;
        }

        var inv = CultureInfo.InvariantCulture;
        context.Write($"{path}: {topology.WorldSize} rank(s), failures {failures}, seed {seed.ToString(inv)}, through step {step.ToString(inv)}");
        context.Write(schedule.Count == 0
            ? "fault schedule: no fault drawn"
            : $"fault schedule ({schedule.Count}): " + string.Join("; ", schedule.Select(f => $"{f.Kind} at step {f.Step.ToString(inv)} on rank(s) {string.Join(",", f.Ranks)}"
                                                                                             + (f.Duration > TimeSpan.Zero ? $" for {f.Duration.TotalMilliseconds.ToString("F1", inv)} ms" : ""))));
        var problems = new List<string>();
        if (original is not null)
        {
            var logged = original.Faults.Where(f => f.Step <= step).ToList();
            if (!Same(logged, schedule))
            {
                problems.Add($"the run's log holds {logged.Count} fault(s) through step {step}, the schedule {schedule.Count}: they differ");
            }
        }

        var json = new JsonObject
        {
            ["topology"] = Path.GetFullPath(path),
            ["seed"] = seed,
            ["failures"] = failures,
            ["step"] = step,
            ["faults"] = new JsonArray([.. schedule.Select(f => (JsonNode)new JsonObject
            {
                ["kind"] = f.Kind.ToString(), ["step"] = f.Step, ["ranks"] = new JsonArray([.. f.Ranks.Select(r => (JsonNode)r)]), ["duration_ms"] = f.Duration.TotalMilliseconds,
            })]),
        };

        var inner = context.Positional;
        if (inner.Count > 0)
        {
            var (options, simTopology) = ClusterSimCommand.Prepare(context, path, inner);
            if (simTopology is null)
            {
                return ExitCodes.Failed;
            }

            options = options with { Simulation = options.Simulation with { Seed = seed, Failures = failures, ReplayStep = step, Mode = SimulationMode.InProcess } };
            options = ClusterSimCommand.WithEventLog(context, options, inner);
            context.Write($"rerunning through step {step.ToString(inv)} on {simTopology.WorldSize} rank(s): idrak {string.Join(' ', inner)}");
            var (codes, result) = ClusterSimCommand.Simulate(context, simTopology, options, inner);
            if (!result.Succeeded || result.Log is null)
            {
                context.Error(result.Log is null ? "the rerun kept no event log (it ran no simulated transport)" : "the rerun failed");
                return result.Succeeded ? ExitCodes.Failed : ClusterSimCommand.Outcome(context, result, codes);
            }

            var clocks = new JsonArray();
            for (int rank = 0; rank < simTopology.WorldSize; rank++)
            {
                double? at = result.Log.ClockAt(rank, step);
                clocks.Add(at);
                context.Write($"rank {rank}: " + (at is { } t ? $"clock {(t * 1e3).ToString("F6", inv)} ms of simulated network time as it began step {step.ToString(inv)}" : $"did not begin step {step.ToString(inv)}"));
            }

            json["clocks_seconds"] = clocks;
            json["event_log"] = options.Simulation.EventLog;
            if (!Same(result.Log.Faults.Where(f => f.Step <= step).ToList(), schedule))
            {
                problems.Add("the rerun drew other faults than the schedule");
            }

            if (original is not null && FirstDifference(original, result.Log, step) is { } difference)
            {
                problems.Add(difference);
            }
        }

        json["matches"] = original is null ? null : problems.Count == 0;
        if (original is not null)
        {
            context.Write(problems.Count == 0
                ? $"matches {context.Option("--from")} bit for bit through step {step.ToString(inv)}"
                : "differs from " + context.Option("--from") + ": " + string.Join("; ", problems));
        }

        context.WriteJson(json);
        return problems.Count == 0 ? ExitCodes.Ok : ExitCodes.Failed;
    }

    private static bool Same(IReadOnlyList<Fault> a, IReadOnlyList<Fault> b) =>
        a.Count == b.Count && a.Zip(b).All(p => p.First.Kind == p.Second.Kind && p.First.Step == p.Second.Step && p.First.Duration == p.Second.Duration
                                                && p.First.Factor.Equals(p.Second.Factor) && p.First.Ranks.SequenceEqual(p.Second.Ranks));

    // The first rank and step whose clock differs between two logs through `step`, or null when every one is the same.
    private static string? FirstDifference(EventLog original, EventLog rerun, long step)
    {
        foreach (var clock in original.Clocks.Where(c => c.Step <= step).OrderBy(c => c.Step).ThenBy(c => c.Rank))
        {
            double? again = rerun.ClockAt(clock.Rank, clock.Step);
            if (again is not { } value || BitConverter.DoubleToInt64Bits(value) != BitConverter.DoubleToInt64Bits(clock.Seconds))
            {
                return string.Create(CultureInfo.InvariantCulture,
                    $"rank {clock.Rank}'s clock as it began step {clock.Step}: {clock.Seconds:R} s in the run, {(again is { } v ? v.ToString("R", CultureInfo.InvariantCulture) + " s" : "none")} in the rerun");
            }
        }

        return null;
    }
}
