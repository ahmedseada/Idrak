// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Cluster;

/// <summary>
/// The seeded fault injector of a simulated run (plan 23, phase C3). The faults of step s are drawn once, when the first
/// rank begins it, from a generator seeded with (seed, s) alone, so they are the same on every run and on a replay that
/// starts nowhere near step 0; each is written to the event log and published as <see cref="FaultInjected"/> with its seed
/// and step. A rank applies the faults that concern it as it begins the step (its state changes logged and published as
/// <see cref="RankStateChanged"/> with the seed and step), and the virtual clock asks it, for each transfer, what the
/// sender's active faults do to it: a slow worker's transfers stretched, a lossy link's extra loss, a partition's hold.
/// A rank's own state is touched by that rank's thread alone; the drawn faults are shared under a lock.
/// </summary>
internal sealed class FaultInjector
{
    private readonly IFailureModel _model;
    private readonly IClusterTopology _topology;
    private readonly TimeSpan _timeout;
    private readonly EventLogWriter _log;
    private readonly Action<Exception> _breakGroup;
    private readonly ClusterRunControl? _control;
    private readonly Lock _gate = new();
    private readonly Dictionary<long, Fault[]> _byStep = [];
    private readonly List<Fault> _lateJoins = [];
    private readonly RankFaults[] _ranks;
    private long _drawnThrough = -1;

    public FaultInjector(IFailureModel model, IClusterTopology topology, int seed, TimeSpan timeout, EventLogWriter log, Action<Exception> breakGroup,
        ClusterRunControl? control = null)
    {
        _model = model;
        _control = control;
        _topology = topology;
        Seed = seed;
        _timeout = timeout;
        _log = log;
        _breakGroup = breakGroup;
        _ranks = [.. Enumerable.Range(0, topology.WorldSize).Select(_ => new RankFaults())];
    }

    public int Seed { get; }

    public string Model => _model.Name;

    // The rank whose dropped-worker fault broke the group, or -1.
    public int Dropped { get; private set; } = -1;

    /// <summary>The faults <paramref name="model"/> draws for steps 0 to <paramref name="throughStep"/>, as a run of the seed draws them.</summary>
    public static IReadOnlyList<Fault> Schedule(IFailureModel model, IClusterTopology topology, int seed, long throughStep)
    {
        var faults = new List<Fault>();
        for (long step = 0; step <= throughStep; step++)
        {
            faults.AddRange(DrawStep(model, topology, seed, step));
        }

        return faults;
    }

    // Draws every step up to `step` not yet drawn, in order: logged and published once each.
    public void DrawThrough(long step)
    {
        lock (_gate)
        {
            for (long s = _drawnThrough + 1; s <= step; s++)
            {
                // The run's own model draws from (seed, s) as it would unsteered; the live faults (a steered run's, phase C9)
                // follow, taking nothing from the generator, so the two compose.
                var drawn = DrawStep(_model, _topology, Seed, s);
                bool ownLive = _model is LiveFailures;
                Fault[] live = _control is null ? [] : Checked(_control.Live, _topology, s, _control.Live.Draw(s, _topology, Random.Shared));
                Fault[] faults = live.Length == 0 ? drawn : [.. drawn, .. live];
                if (faults.Length > 0)
                {
                    _byStep[s] = faults;
                }

                for (int i = 0; i < faults.Length; i++)
                {
                    var fault = faults[i];
                    if (fault.Kind == FaultKind.LateJoin)
                    {
                        _lateJoins.Add(fault);
                    }

                    bool isLive = ownLive || i >= drawn.Length;
                    _log.Fault(fault, isLive);
                    if (Telemetry.IsEnabled(TelemetryLevel.Network))
                    {
                        Telemetry.Fault(new FaultInjected(Seed, s, fault.Kind, fault.Ranks, fault.Duration, fault.Factor));
                    }

                    if (isLive)
                    {
                        _control?.Drawn(fault);
                    }
                }

                _drawnThrough = s;
            }
        }
    }

    // The ranks taking part in `step` (every rank but those a late join keeps out), in order. Step -1 is the start: all.
    public int[] Members(long step)
    {
        lock (_gate)
        {
            if (step < 0 || _lateJoins.Count == 0)
            {
                return [.. Enumerable.Range(0, _topology.WorldSize)];
            }

            var absent = new bool[_topology.WorldSize];
            foreach (var join in _lateJoins)
            {
                if (join.Step <= step && step < join.Step + (long)join.Factor)
                {
                    absent[join.Ranks[0]] = true;
                }
            }

            return [.. Enumerable.Range(0, absent.Length).Where(r => !absent[r])];
        }
    }

    // The step's late joiner `rank` is: the step it joins at, or null.
    public long? JoinsAt(int rank, long step)
    {
        lock (_gate)
        {
            foreach (var join in _lateJoins)
            {
                if (join.Ranks[0] == rank && join.Step <= step && step < join.Step + (long)join.Factor)
                {
                    return join.Step + (long)join.Factor;
                }
            }

            return null;
        }
    }

    // Rank `rank` begins `step` at simulated time `now`: the faults that end are cleared, those that start are applied,
    // and each change of its state is logged and published. A dropped worker breaks the group and throws.
    public void Begin(int rank, long step, double now)
    {
        var mine = _ranks[rank];
        if (mine.Slow is { } slow && now >= mine.SlowUntil)
        {
            (mine.Slow, mine.SlowFactor) = (null, 1);
            State(rank, step, RankState.Rejoined, $"no longer slow (the slow worker of step {slow.Step} ended)");
        }

        mine.Losses.RemoveAll(l => now >= l.Until);
        if (mine.Partitions.Count > 0 && mine.Partitions.RemoveAll(p => now >= p.Until) > 0 && mine.Partitions.Count == 0)
        {
            State(rank, step, RankState.Rejoined, "the partition healed");
        }

        Fault[] faults;
        lock (_gate)
        {
            faults = _byStep.GetValueOrDefault(step) ?? [];
            foreach (var join in _lateJoins)
            {
                if (join.Ranks[0] == rank && join.Step + (long)join.Factor == step)
                {
                    State(rank, step, RankState.Joined, string.Create(CultureInfo.InvariantCulture, $"connected {join.Factor:G} step(s) late (the late join of step {join.Step})"));
                }
            }
        }

        foreach (var fault in faults)
        {
            double until = fault.Duration > TimeSpan.Zero ? now + fault.Duration.TotalSeconds : double.PositiveInfinity;
            switch (fault.Kind)
            {
                case FaultKind.SlowWorker when fault.Ranks.Contains(rank):
                    (mine.Slow, mine.SlowFactor, mine.SlowUntil) = (fault, fault.Factor, until);
                    State(rank, step, RankState.Slow, string.Create(CultureInfo.InvariantCulture,
                        $"its transfers take {fault.Factor:G4} times as long {(fault.Duration > TimeSpan.Zero ? $"for {fault.Duration.TotalMilliseconds:F1} ms" : "for the rest of the run")}"));
                    break;
                case FaultKind.DroppedWorker when fault.Ranks.Contains(rank):
                    State(rank, step, RankState.Dropped, "a dropped-worker fault: its links closed");
                    Dropped = rank;
                    var dropped = new InvalidOperationException($"rank {rank} was dropped at step {step}: its links closed (a dropped-worker fault; replay with --seed {Seed} --step {step})");
                    _breakGroup(dropped);
                    throw dropped;
                case FaultKind.LinkLoss when fault.Ranks.Contains(rank):
                    mine.Losses.Add((fault.Ranks[0] == rank ? fault.Ranks[1] : fault.Ranks[0], fault.Factor, until));
                    break;
                case FaultKind.Partition:
                    var side = new bool[_topology.WorldSize];
                    foreach (int r in fault.Ranks)
                    {
                        side[r] = true;
                    }

                    bool fatal = fault.Duration <= TimeSpan.Zero || fault.Duration >= _timeout;
                    mine.Partitions.Add((side, until, fatal, fault));
                    State(rank, step, RankState.Partitioned, $"{Sides(fault)} {(fatal ? "for longer than the timeout" : string.Create(CultureInfo.InvariantCulture, $"for {fault.Duration.TotalMilliseconds:F1} ms"))}");
                    break;
            }
        }
    }

    // What the faults active on `from` at its time `sent` do to a transfer to `to` (called under the clock's lock, by
    // `from`'s own thread).
    public FaultEffect Effect(int from, int to, double sent)
    {
        var mine = _ranks[from];
        if (mine.Slow is null && mine.Losses.Count == 0 && mine.Partitions.Count == 0)
        {
            return FaultEffect.None;
        }

        double factor = mine.Slow is not null && sent < mine.SlowUntil ? mine.SlowFactor : 1;
        double kept = 1;
        foreach (var (peer, share, until) in mine.Losses)
        {
            if (peer == to && sent < until)
            {
                kept *= 1 - share;
            }
        }

        double hold = 0;
        Fault? partition = null;
        foreach (var (side, until, fatal, fault) in mine.Partitions)
        {
            if (side[from] != side[to] && sent < until)
            {
                hold = Math.Max(hold, fatal ? double.PositiveInfinity : until);
                partition = fault;
            }
        }

        return new FaultEffect(factor, 1 - kept, hold, partition);
    }

    // The timeout a partition that never opens causes on the receiver, naming both sides.
    public TimeoutException PartitionTimeout(int from, int to, int tag, Fault partition) =>
        new($"rank {to} waited {_timeout.TotalSeconds.ToString("F0", CultureInfo.InvariantCulture)} s for rank {from} (tag {tag}): a partition at step {partition.Step} separates {Sides(partition)} (replay with --seed {Seed} --step {partition.Step})");

    private string Sides(Fault partition)
    {
        var a = partition.Ranks.Order().ToArray();
        var b = Enumerable.Range(0, _topology.WorldSize).Except(a);
        return $"rank(s) {string.Join(", ", a)} from rank(s) {string.Join(", ", b)}";
    }

    private void State(int rank, long step, RankState state, string detail)
    {
        _log.State(rank, step, state, detail);
        if (Telemetry.IsEnabled(TelemetryLevel.Network))
        {
            Telemetry.RankState(new RankStateChanged(rank, state, step, detail, Seed));
        }
    }

    // The faults of one step, from (seed, step) alone, checked.
    private static Fault[] DrawStep(IFailureModel model, IClusterTopology topology, int seed, long step)
    {
        var random = new Random(unchecked((int)VirtualClock.Mix(VirtualClock.Mix((ulong)(uint)seed) ^ (ulong)step)));
        return Checked(model, topology, step, model.Draw(step, topology, random));
    }

    private static Fault[] Checked(IFailureModel model, IClusterTopology topology, long step, IReadOnlyList<Fault> faults)
    {
        foreach (var f in faults)
        {
            string? problem = f.Step != step ? $"is for step {f.Step}, drawn at step {step}" : Problem(f, topology.WorldSize);
            if (problem is not null)
            {
                throw new InvalidOperationException($"Failure model '{model.Name}' drew a {f.Kind} fault that {problem}.");
            }
        }

        return [.. faults];
    }

    // What makes `f` a fault the simulator cannot apply on `world` ranks (whatever its step), or null.
    internal static string? Problem(Fault f, int world) =>
        f.Ranks is null || f.Ranks.Count == 0 || f.Ranks.Any(r => (uint)r >= (uint)world) || f.Ranks.Distinct().Count() != f.Ranks.Count ? $"names ranks outside 0 to {world - 1}, none, or one twice"
        : f.Duration < TimeSpan.Zero || !double.IsFinite(f.Factor) ? "has a negative duration or a factor that is not a number"
        : f.Kind switch
        {
            FaultKind.SlowWorker when f.Ranks.Count != 1 || f.Factor < 1 => "is a slow worker that is not one rank with a factor of 1 or more",
            FaultKind.DroppedWorker when f.Ranks.Count != 1 => "drops more than one rank",
            FaultKind.LinkLoss when f.Ranks.Count != 2 || f.Factor is < 0 or > 1 => "is a lossy link that is not two ranks with a share from 0 to 1",
            FaultKind.Partition when f.Ranks.Count >= world => "leaves no rank on the other side",
            FaultKind.LateJoin when f.Ranks.Count != 1 || f.Factor < 1 || f.Factor != Math.Floor(f.Factor) => "is a late join that is not one rank and a whole number of steps",
            FaultKind.SlowWorker or FaultKind.DroppedWorker or FaultKind.LinkLoss or FaultKind.Partition or FaultKind.LateJoin => null,
            _ => $"is of a kind the simulator does not know ({(int)f.Kind})",
        };

    // One rank's active faults: written and read by that rank's thread alone.
    private sealed class RankFaults
    {
        public Fault? Slow;
        public double SlowFactor = 1;
        public double SlowUntil;
        public readonly List<(int Peer, double Share, double Until)> Losses = [];
        public readonly List<(bool[] SideA, double Until, bool Fatal, Fault Fault)> Partitions = [];
    }
}
