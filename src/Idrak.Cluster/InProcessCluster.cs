// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;

namespace Idrak.Cluster;

/// <summary>One rank of a simulated cluster as its work sees it.</summary>
/// <param name="Rank">Its rank.</param>
/// <param name="Device">Its device, carved from the physical one its topology entry names, with its cap.</param>
/// <param name="Transport">Its end of the transport joining every rank ("sim" in a simulated run).</param>
/// <param name="Collectives">Its collectives: over every rank, or over the ranks that take part in the current step.</param>
/// <param name="Context">The topology, its rank and the options.</param>
public sealed record ClusterRank(int Rank, Device Device, ITransport Transport, ICollectives Collectives, ClusterContext Context)
{
    private long _step = -1;

    internal SimStepper? Stepper { get; init; }

    /// <summary>The step this rank last began (<see cref="BeginStep"/>), or -1 before the first.</summary>
    public long Step => Stepper?.Step ?? _step;

    /// <summary>
    /// Begins <paramref name="step"/> on this rank (a training loop calls it at the start of each step; steps go forward).
    /// In a simulated run the step's faults are drawn and applied here (a dropped worker throws, naming itself, and every
    /// other rank's step ends naming it), and the clock as the step begins goes to the event log. Returns false when this
    /// rank sits the step out: a late joiner before it joins, or every rank past the step a replay stops at
    /// (<see cref="SimulationOptions.ReplayStep"/>). While a rank sits out, <see cref="Collectives"/> of the others runs over
    /// the ranks that take part.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">A step not after the last one begun.</exception>
    public bool BeginStep(long step)
    {
        if (Stepper is { } stepper)
        {
            return stepper.BeginStep(step);
        }

        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(step, _step);
        _step = step;
        return Context.Options.Simulation.ReplayStep is not { } last || step <= last;
    }
}

/// <summary>What a simulated run gave: each rank's failure, if any, and its time on the virtual clock, and the wall time.</summary>
/// <param name="Errors">Each rank's exception, or null when it finished.</param>
/// <param name="Simulated">Each rank's network time on the virtual clock.</param>
/// <param name="Wall">The time the run took here.</param>
public sealed record ClusterRunResult(IReadOnlyList<Exception?> Errors, IReadOnlyList<TimeSpan> Simulated, TimeSpan Wall)
{
    /// <summary>Whether every rank finished.</summary>
    public bool Succeeded => Errors.All(e => e is null);

    /// <summary>The run's event log (faults, each rank's clock per step, state changes, transfers), or null when no rank ran on "sim".</summary>
    public EventLog? Log { get; init; }

    /// <summary>Throws when a rank failed, naming every rank that did.</summary>
    /// <exception cref="AggregateException">The ranks' exceptions.</exception>
    public void ThrowIfFailed()
    {
        if (!Succeeded)
        {
            throw new AggregateException("Ranks failed: " + string.Join("; ", Errors.Select((e, r) => e is null ? null : $"rank {r}: {e.Message}").Where(m => m is not null)),
                Errors.Where(e => e is not null).Select(e => e!));
        }
    }
}

/// <summary>
/// Runs a cluster inside this process (plan 23, the in-process mode): every rank a thread of its own on a device carved
/// from the physical device its topology entry names (<see cref="SimulatedDevices.Carve"/>, the entry's memory as its cap;
/// <see cref="SimulationOptions.CarveFrom"/> carves every one from one device), joined by the transport the plan chooses
/// per pair (<see cref="TransportPlan"/>: "sim" here unless one is forced; each choice published once as
/// <see cref="TransportChosen"/>) with the algorithm <see cref="ClusterOptions.CollectiveFor"/> names (hierarchical across
/// several nodes, else ring). The failure model's faults are injected step by step (<see cref="ClusterRank.BeginStep"/>)
/// and every run keeps an <see cref="EventLog"/>. Deterministic for a seed: the virtual clock counts network time only.
/// </summary>
public static class InProcessCluster
{
    /// <summary>Runs <paramref name="work"/> on every rank of <paramref name="topology"/> and waits for them all.</summary>
    /// <exception cref="NotSupportedException">
    /// The collective algorithm, the transport or the failure model is not registered, or faults are asked of a transport
    /// that is not "sim". A plan that joins pairs over several transports runs them routed per pair (phase C5).
    /// </exception>
    public static ClusterRunResult Run(IClusterTopology topology, ClusterOptions? options, Action<ClusterRank> work) => Run(topology, options, work, null);

    /// <summary>
    /// Runs <paramref name="work"/> on every rank of <paramref name="topology"/>, steered by <paramref name="control"/> (plan
    /// 23, phase C9): paused and resumed at step boundaries, stopped after a step, faults injected live. Steering needs the
    /// "sim" transport (its steps are where the ranks wait and its faults are the ones injected).
    /// </summary>
    /// <exception cref="NotSupportedException">A control given while the plan joins the ranks by another transport, or as above.</exception>
    public static ClusterRunResult Run(IClusterTopology topology, ClusterOptions? options, Action<ClusterRank> work, ClusterRunControl? control)
    {
        ArgumentNullException.ThrowIfNull(topology);
        ArgumentNullException.ThrowIfNull(work);
        string run = "sim://" + Guid.NewGuid().ToString("N", CultureInfo.InvariantCulture);   // this run's own rendezvous
        options = (options ?? new ClusterOptions()) with
        {
            Coordinator = run,
            Simulation = (options?.Simulation ?? new SimulationOptions()) with { Mode = SimulationMode.InProcess },
        };
        if (control is null)
        {
            return Start(topology, options, work);
        }

        control.Attach(topology.WorldSize);
        RunControls.Add(run, control);
        try
        {
            return Start(topology, options, work, steered: true);
        }
        finally
        {
            RunControls.Remove(run);
        }
    }

    private static ClusterRunResult Start(IClusterTopology topology, ClusterOptions options, Action<ClusterRank> work, bool steered = false)
    {
        var plan = TransportPlan.Make(topology, options);
        // One transport for every pair, or (phase C5) several routed per pair, each carrier connected once.
        string? transportName = plan.Single ?? (plan.Used.Count == 0
            ? (string.Equals(options.Transport, "auto", StringComparison.OrdinalIgnoreCase) ? "sim" : options.Transport)
            : null);
        var transports = transportName is null ? null : Transports.Get(transportName);
        string failures = options.Simulation.Failures ?? topology.Simulation.Failures;
        if (steered && !string.Equals(transportName, "sim", StringComparison.OrdinalIgnoreCase))
        {
            throw new NotSupportedException($"A run steered live (paused, stopped, faults injected) needs the sim transport; this run uses '{transportName ?? string.Join("+", plan.Used)}': "
                                            + "set IDRAK_CLUSTER_TRANSPORT=auto (or sim).");
        }

        if (!string.Equals(transportName, "sim", StringComparison.OrdinalIgnoreCase) && !string.Equals(failures, "none", StringComparison.OrdinalIgnoreCase))
        {
            throw new NotSupportedException($"Faults ('{failures}') are injected by the sim transport; this run uses '{transportName ?? string.Join("+", plan.Used)}': set IDRAK_CLUSTER_FAILURES=none, or let the selector choose sim.");
        }

        var collectives = Collectives.Get(options.CollectiveFor(topology));
        int world = topology.WorldSize;
        var devices = new Device[world];
        for (int r = 0; r < world; r++)
        {
            var (node, local) = topology.Place(r);
            var spec = topology.Nodes[node].Devices[local];
            devices[r] = SimulatedDevices.Carve(Device.Parse(options.Simulation.CarveFrom ?? spec.Device), spec.MemoryBytes, r);
        }

        var errors = new Exception?[world];
        var simulated = new TimeSpan[world];
        var worlds = new SimWorld?[world];
        var threads = new Thread[world];
        long start = Stopwatch.GetTimestamp();
        for (int r = 0; r < world; r++)
        {
            int rank = r;
            threads[r] = new Thread(() =>
            {
                var context = new ClusterContext(topology, rank, options);
                ITransport? transport = null;
                ICollectives? group = null;
                ClusterRank? me = null;
                try
                {
                    transport = transports is not null ? transports.Connect(context) : plan.Connect(context);
                    SimStepper? stepper = null;
                    if (transport is SimTransport sim)
                    {
                        worlds[rank] = sim.Hub.World;
                        group = stepper = new SimStepper(sim, devices[rank], context, collectives);
                    }
                    else
                    {
                        group = collectives.Create(transport, devices[rank], context);
                    }

                    State(rank, RankState.Joined, -1, null, worlds[rank]?.Seed);
                    me = new ClusterRank(rank, devices[rank], transport, group, context) { Stepper = stepper };
                    work(me);
                    simulated[rank] = transport.SimulatedTime;
                    stepper?.Finish();
                    State(rank, RankState.Finished, me.Step, null, worlds[rank]?.Seed);
                }
                catch (Exception e)
                {
                    errors[rank] = e;
                    simulated[rank] = transport?.SimulatedTime ?? TimeSpan.Zero;
                    if (worlds[rank]?.Faults.Dropped != rank)               // a dropped worker said so as it dropped
                    {
                        State(rank, RankState.Dropped, me?.Step ?? -1, e.Message, worlds[rank]?.Seed);
                    }
                }
                finally
                {
                    group?.Dispose();
                    transport?.Dispose();
                }
            })
            { IsBackground = true, Name = $"sim rank {rank}" };
        }

        foreach (var thread in threads)
        {
            thread.Start();
        }

        foreach (var thread in threads)
        {
            thread.Join();
        }

        var wall = Stopwatch.GetElapsedTime(start);
        EventLog? log = null;
        if (worlds.FirstOrDefault(w => w is not null) is { } simWorld)
        {
            try
            {
                log = simWorld.Log.Seal();
            }
            finally
            {
                simWorld.Log.Dispose();
            }

            if (options.Simulation.EventLog is { } path)
            {
                log.Save(path);
            }
        }

        return new ClusterRunResult(errors, simulated, wall) { Log = log };
    }

    private static void State(int rank, RankState state, long step, string? detail, int? seed)
    {
        if (Telemetry.IsEnabled(TelemetryLevel.Network))
        {
            Telemetry.RankState(new RankStateChanged(rank, state, step, detail, seed));
        }
    }
}

/// <summary>
/// A rank's steps in a simulated run, and its collectives: over every rank, or, while a late joiner is away, over a group
/// of the ranks that take part (its own queues over the same clock, numbered from 0, with the topology of those ranks).
/// When the group changes its ranks meet at a barrier, so a late joiner's clock moves to theirs as it joins.
/// </summary>
internal sealed class SimStepper : ICollectives, IStepGroup, IHostValueCollectives
{
    private readonly SimTransport _root;
    private readonly SimWorld _world;
    private readonly ICollectivesProvider _provider;
    private readonly ClusterContext _context;
    private readonly Dictionary<SimHub, (SimTransport Transport, ICollectives Collectives)> _groups = [];
    private (SimTransport Transport, ICollectives Collectives) _current;
    private bool _present = true;
    private long? _joinsAt;

    public SimStepper(SimTransport root, Device device, ClusterContext context, ICollectivesProvider provider)
    {
        _root = root;
        _world = root.Hub.World;
        _provider = provider;
        _context = context;
        Device = device;
        _current = (root, provider.Create(root, device, context));
        _groups[root.Hub] = _current;
    }

    public long Step { get; private set; } = -1;

    public bool Present => _present;

    public bool Changed { get; private set; }

    public int Holder { get; private set; }

    public TimeSpan SimulatedTime => _current.Transport.SimulatedTime;

    // Every group's collectives come from one provider: they all take host values, or none does.
    public bool SupportsHostValues => _current.Collectives is IHostValueCollectives { SupportsHostValues: true };

    public string Name => _current.Collectives.Name;

    public int Rank => _current.Collectives.Rank;

    public int WorldSize => _current.Collectives.WorldSize;

    public Device Device { get; }

    // The current group's transport and context, for messages of one's own (a pipeline's activations, plan 23 C7).
    public ITransport? Transport => _current.Transport;

    public ClusterContext? Context => ReferenceEquals(_current.Transport, _root) ? _context : null;

    private int Real => _root.Real;

    private ICollectives Current => _present
        ? _current.Collectives
        : throw new InvalidOperationException(_joinsAt is { } at
            ? $"rank {Real} sits out step {Step}: it joins at step {at} (a late join)"
            : $"rank {Real} sits out step {Step}: the replay stops at step {_world.Options.Simulation.ReplayStep}");

    public bool BeginStep(long step)
    {
        if (step <= Step)
        {
            throw new ArgumentOutOfRangeException(nameof(step), step, $"rank {Real} began step {Step}; steps go forward");
        }

        _world.Control?.Gate(Real, step, Device);           // a steered run waits here while paused, and stops here (phase C9)
        if (_world.Broken is not null)
        {
            throw _world.Failure();
        }

        Step = step;
        (Changed, Holder) = (false, 0);
        if (_world.Options.Simulation.ReplayStep is { } last && step > last)
        {
            (_present, _joinsAt) = (false, null);
            return false;
        }

        _world.Faults.DrawThrough(step);
        var members = _world.Faults.Members(step);
        if (Array.BinarySearch(members, Real) < 0)
        {
            (_present, _joinsAt) = (false, _world.Faults.JoinsAt(Real, step));
            return false;
        }

        _present = true;
        var hub = _world.GroupOf(members);
        if (!_groups.TryGetValue(hub, out var group))
        {
            int rank = hub.SubOf(Real)!.Value;
            var transport = new SimTransport(hub, rank);
            group = (transport, _provider.Create(transport, Device, new ClusterContext(hub.Topology, rank, _context.Options)));
            _groups[hub] = group;
        }

        _current = group;
        var before = _world.Faults.Members(step - 1);
        if (!members.AsSpan().SequenceEqual(before))
        {
            group.Transport.Barrier();                         // the group changed: its ranks meet, their clocks at the latest
            // The trained state is a rank's that took part in the step before too: the first of them, in the group's numbering.
            int holder = Array.FindIndex(members, r => Array.BinarySearch(before, r) >= 0);
            (Changed, Holder) = (true, Math.Max(0, holder));
        }

        _world.Faults.Begin(Real, step, _world.Clock.Now(Real));
        _world.Log.Clock(Real, step, _world.Clock.Now(Real));
        return true;
    }

    // The rank finished its work: its clock goes to the log.
    public void Finish() => _world.Log.End(Real, _world.Clock.Now(Real));

    public void AllReduce(Storage buffer, int count, ReduceOp op, WireType wire = WireType.Float32) => Current.AllReduce(buffer, count, op, wire);

    public void AllGather(Storage input, int count, Storage output) => Current.AllGather(input, count, output);

    public void ReduceScatter(Storage input, int count, Storage output, ReduceOp op) => Current.ReduceScatter(input, count, output, op);

    public void Broadcast(Storage buffer, int count, int root) => Current.Broadcast(buffer, count, root);

    public void Barrier() => Current.Barrier();

    public bool TryAllReduce(Span<float> values, ReduceOp op, WireType wire) => Current is IHostValueCollectives host && host.TryAllReduce(values, op, wire);

    public bool TryAllGather(Span<float> all, int count) => Current is IHostValueCollectives host && host.TryAllGather(all, count);

    public bool TryBroadcast(Span<float> values, int root) => Current is IHostValueCollectives host && host.TryBroadcast(values, root);

    public void Dispose()
    {
        foreach (var (transport, collectives) in _groups.Values)
        {
            collectives.Dispose();
            if (!ReferenceEquals(transport, _root))
            {
                transport.Dispose();
            }
        }

        _groups.Clear();
    }
}
