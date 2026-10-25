// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using Idrak.Cluster.Abstractions;

namespace Idrak.Cluster;

/// <summary>Where a steered run stands (<see cref="ClusterRunControl.State"/>).</summary>
public enum ClusterRunState
{
    /// <summary>Training.</summary>
    Running,

    /// <summary>Asked to pause: the ranks finish the step they are in and wait at the next boundary.</summary>
    Pausing,

    /// <summary>Every rank waits at the step boundary <see cref="ClusterRunControl.PauseAt"/>.</summary>
    Paused,

    /// <summary>Asked to stop: the ranks finish the step they are in, then stop with the trainer's checkpoint.</summary>
    Stopping,
}

/// <summary>A change of a steered run, as <see cref="ClusterRunControl.Changed"/> tells it.</summary>
/// <param name="State">The run's state after the change.</param>
/// <param name="Step">The step concerned: the boundary a pause waits at, the last step a stop lets finish, a fault's step once drawn.</param>
/// <param name="Detail">What happened, in a line.</param>
/// <param name="Fault">The fault queued or drawn, for a live fault; else null.</param>
public readonly record struct ClusterRunChange(ClusterRunState State, long Step, string Detail, Fault? Fault = null);

/// <summary>One rank as it began a step of a steered run, told while someone observes (<see cref="ClusterRunControl.Observed"/>).</summary>
/// <param name="Rank">The rank.</param>
/// <param name="Step">The step it began.</param>
/// <param name="Wall">The wall time since it began the step before (zero at its first).</param>
/// <param name="Memory">Its device's memory as its accountant counts it (in use, peak, its cap's figures).</param>
public readonly record struct ClusterRankStep(int Rank, long Step, TimeSpan Wall, MemoryUsage Memory);

/// <summary>
/// Steers an in-process simulated run while it goes (plan 23, phase C9: the dashboard's controls, through the library):
/// <see cref="Pause"/> and <see cref="Resume"/> take effect at a step boundary (every rank's <see cref="ClusterRank.BeginStep"/>
/// waits while paused, all at the same step), <see cref="Stop"/> lets the step in progress finish and then stops every rank
/// as it begins the next (a <see cref="ClusterRunStoppedException"/> the trainer answers with its checkpoint), and
/// <see cref="Inject"/> queues a fault to the <see cref="Live"/> model, which hands it to the next step beside the faults
/// of the run's own failure model. Every live fault is written to the run's event log marked live
/// (<see cref="EventLog.LiveFaults"/>), so <see cref="ClusterReplay"/> replays a steered run bit for bit (given its event
/// log). Give it to <see cref="InProcessCluster"/>'s Run; one control steers one run.
/// </summary>
public sealed class ClusterRunControl
{
    private readonly object _sync = new();
    private readonly Dictionary<int, long> _lastBegun = [];
    private long _last = -1;
    private long? _pauseAt;
    private long? _stopAfter;
    private int _waiting;
    private int _world;
    private bool _pausedTold;
    private int _attached;

    /// <summary>A control with nothing queued.</summary>
    public ClusterRunControl()
        : this(null)
    {
    }

    /// <summary>
    /// A control whose live model hands out <paramref name="scheduled"/> at their own steps (a replay of a steered run: the
    /// live faults its event log recorded), beside anything queued later.
    /// </summary>
    public ClusterRunControl(IEnumerable<Fault>? scheduled) => Live = new LiveFailures(scheduled);

    /// <summary>The faults asked for while the run goes; the run's own model draws beside them.</summary>
    public LiveFailures Live { get; }

    /// <summary>The highest step any rank began, or -1 before the first.</summary>
    public long LastStep
    {
        get
        {
            lock (_sync)
            {
                return _last;
            }
        }
    }

    /// <summary>The step boundary a pause waits at, or null while not pausing.</summary>
    public long? PauseAt
    {
        get
        {
            lock (_sync)
            {
                return _pauseAt;
            }
        }
    }

    /// <summary>The last step a stop lets finish, or null while not stopping.</summary>
    public long? StopAfter
    {
        get
        {
            lock (_sync)
            {
                return _stopAfter;
            }
        }
    }

    /// <summary>Where the run stands.</summary>
    public ClusterRunState State
    {
        get
        {
            lock (_sync)
            {
                return StateLocked();
            }
        }
    }

    /// <summary>
    /// Whether someone watches each rank's steps (<see cref="RankStepped"/>, each one reading the rank's memory figures); off,
    /// a step boundary costs one lock and nothing is measured (rule 78 of docs/optimization.md).
    /// </summary>
    public bool Observed { get; set; }

    /// <summary>Told of every change: a pause asked and reached, a resume, a stop, a fault queued and drawn.</summary>
    public event Action<ClusterRunChange>? Changed;

    /// <summary>Told as each rank begins a step, while <see cref="Observed"/>.</summary>
    public event Action<ClusterRankStep>? RankStepped;

    /// <summary>
    /// Pauses the run at the next step boundary: the steps begun go on to their end, and every rank waits as it begins the
    /// one after (the same step for every rank). Returns that step.
    /// </summary>
    public long Pause()
    {
        lock (_sync)
        {
            if (_pauseAt is not null || _stopAfter is not null)
            {
                return _pauseAt ?? _stopAfter!.Value + 1;
            }

            _pauseAt = _last + 1;
            _pausedTold = false;
            Changed?.Invoke(new ClusterRunChange(ClusterRunState.Pausing, _pauseAt.Value,
                string.Create(CultureInfo.InvariantCulture, $"pausing: the ranks finish step {_last} and wait before step {_pauseAt}")));
            return _pauseAt.Value;
        }
    }

    /// <summary>Lets a paused (or pausing) run go on.</summary>
    public void Resume()
    {
        lock (_sync)
        {
            if (_pauseAt is not { } at)
            {
                return;
            }

            _pauseAt = null;
            _pausedTold = false;
            Monitor.PulseAll(_sync);
            Changed?.Invoke(new ClusterRunChange(StateLocked(), at, string.Create(CultureInfo.InvariantCulture, $"resumed at step {at}")));
        }
    }

    /// <summary>
    /// Stops the run after the step in progress: every rank finishes the highest step begun and stops as it begins the next
    /// (paused ranks stop where they wait). Returns that last step (-1: before the first).
    /// </summary>
    public long Stop()
    {
        lock (_sync)
        {
            if (_stopAfter is { } after)
            {
                return after;
            }

            _stopAfter = _last;
            _pauseAt = null;
            Monitor.PulseAll(_sync);
            Changed?.Invoke(new ClusterRunChange(ClusterRunState.Stopping, _last,
                _last < 0 ? "stopping before the first step" : string.Create(CultureInfo.InvariantCulture, $"stopping after step {_last}; the trainer writes its checkpoint")));
            return _last;
        }
    }

    /// <summary>
    /// Queues a fault for the next step the run begins (<see cref="LiveFailures.Queue"/>): it is drawn beside the run's own
    /// model's faults, applied as every rank begins that step, written to the event log marked live and published as
    /// <see cref="FaultInjected"/> with the run's seed and its step, the replay key.
    /// </summary>
    /// <exception cref="ArgumentException">A fault the simulator cannot apply (ranks out of range, a factor out of its kind's range); the message says why.</exception>
    public Fault Inject(FaultKind kind, IReadOnlyList<int> ranks, TimeSpan duration, double factor)
    {
        ArgumentNullException.ThrowIfNull(ranks);
        int world;
        lock (_sync)
        {
            world = _world;
        }

        var fault = new Fault(kind, -1, [.. ranks], duration, factor);
        if (world > 0 && FaultInjector.Problem(fault, world) is { } problem)
        {
            throw new ArgumentException($"A live {kind} fault that {problem}.", nameof(ranks));
        }

        lock (_sync)
        {
            Live.Queue(fault);
            Changed?.Invoke(new ClusterRunChange(StateLocked(), _last + 1,
                $"{kind} on rank(s) {string.Join(", ", fault.Ranks)} queued for the next step", fault));
        }

        return fault;
    }

    // The run this control steers starts on `world` ranks.
    internal void Attach(int world)
    {
        if (Interlocked.Exchange(ref _attached, 1) != 0)
        {
            throw new InvalidOperationException("This run control already steers a run; make one per run.");
        }

        lock (_sync)
        {
            _world = world;
        }
    }

    // A live fault drawn at its step (by the injector): told as a change.
    internal void Drawn(Fault fault)
    {
        lock (_sync)
        {
            Changed?.Invoke(new ClusterRunChange(StateLocked(), fault.Step, string.Create(CultureInfo.InvariantCulture,
                $"{fault.Kind} on rank(s) {string.Join(", ", fault.Ranks)} injected at step {fault.Step}"), fault));
        }
    }

    // Rank `rank` begins `step` (on its own thread): waits while the run is paused at or before it, throws once a stop's last
    // step is behind it, and tells an observer the rank's step and memory (Hold).
    internal void Gate(int rank, long step, Device device)
    {
        if (Hold(rank, step, device) is { } after)
        {
            throw new ClusterRunStoppedException(step, string.Create(CultureInfo.InvariantCulture,
                $"the run was stopped after step {after} as asked; rank {rank} does not begin step {step}"));
        }
    }

    // Rank `rank` reaches the boundary before `step` (on its own thread): waits while the run is paused at or before it; once
    // a stop's last step is behind it, returns that last step (the rank does not begin `step`), else null, the step counted as
    // begun and, while observed, the rank's step and memory told. Changes are told under the lock, in the order they happen;
    // a listener only records them (it never waits on a rank). Gate for a step gate, and a trainer's own boundaries (Plan 27,
    // R11: InProcessSteering).
    internal long? Hold(int rank, long step, Device device)
    {
        TimeSpan wall = TimeSpan.Zero;
        bool observed;
        lock (_sync)
        {
            bool counted = false;
            try
            {
                while (true)
                {
                    if (_stopAfter is { } after && step > after)
                    {
                        return after;
                    }

                    if (_pauseAt is not { } at || step < at)
                    {
                        break;
                    }

                    if (!counted)
                    {
                        _waiting++;
                        counted = true;
                    }

                    if (!_pausedTold && _waiting >= _world)
                    {
                        _pausedTold = true;
                        Changed?.Invoke(new ClusterRunChange(ClusterRunState.Paused, at,
                            string.Create(CultureInfo.InvariantCulture, $"paused: every rank waits before step {at}")));
                    }

                    Monitor.Wait(_sync);
                }
            }
            finally
            {
                if (counted)
                {
                    _waiting--;
                }
            }

            _last = Math.Max(_last, step);
            observed = Observed;
            if (observed)
            {
                long now = Stopwatch.GetTimestamp();
                if (_lastBegun.TryGetValue(rank, out long before))
                {
                    wall = Stopwatch.GetElapsedTime(before, now);
                }

                _lastBegun[rank] = now;
            }
        }

        if (observed && RankStepped is { } stepped)
        {
            stepped(new ClusterRankStep(rank, step, wall, device.Backend.GetMemoryUsage()));
        }

        return null;
    }

    private ClusterRunState StateLocked() =>
        _stopAfter is not null ? ClusterRunState.Stopping
        : _pauseAt is null ? ClusterRunState.Running
        : _pausedTold ? ClusterRunState.Paused
        : ClusterRunState.Pausing;
}

/// <summary>
/// The "in-process" run steering (Plan 27, R11), registered in <see cref="RunSteerings"/>: a simulated run's rank steered
/// through the run's <see cref="ClusterRunControl"/> (the dashboard's in-process control) at its trainer's boundaries.
/// </summary>
internal sealed class InProcessSteeringProvider : IRunSteeringProvider
{
    public string Name => "in-process";

    public string Summary => "a simulated run's ranks held and stopped at their trainer's boundaries through the run's control (the dashboard's in-process run)";

    public IRunSteering? Create(ICollectives? collectives, ClusterOptions options) =>
        collectives is SimStepper { World.Control: { } control } stepper ? new InProcessSteering(control, stepper.RealRank, stepper.Device) : null;
}

/// <summary>
/// One rank of a steered in-process run at its trainer's boundaries (Plan 27, R11): the run control's hold, as its step gate
/// holds a step (every rank waits at the same boundary while paused, the boundaries counted as the control's steps), and a
/// stop told on every rank at the boundary after the highest one begun when it was asked. The control's own step gate lets
/// this rank's gradient steps through meanwhile (<see cref="RunSteerings.Steered"/>), so a stop never cuts an iteration.
/// </summary>
internal sealed class InProcessSteering(ClusterRunControl control, int rank, Device device) : IRunSteering
{
    public string Name => "in-process";

    // A paused rank waits for the dashboard (a resume or a stop): its token is the trainer's to see once it goes on.
    public ClusterControlCommand Boundary(long step, CancellationToken cancellationToken = default) =>
        control.Hold(rank, step, device) is null ? ClusterControlCommand.Run : ClusterControlCommand.Stop;
}

/// <summary>
/// The "live" failure model (plan 23, phase C9): faults asked for while a run goes (the dashboard's "inject at next step",
/// <see cref="ClusterRunControl.Inject"/>, or <see cref="Queue"/> on the instance <see cref="FailureModels"/> holds), each
/// handed to the next step drawn, and faults scheduled at their own steps (a replay of a steered run). A steered run draws
/// its own model's faults and this model's beside them, so the two compose; named on its own ("live") a run has the queued
/// faults alone. Its draws take nothing from the seeded generator, so the run's own model draws what it would unsteered.
/// </summary>
public sealed class LiveFailures : ILiveFailureModel
{
    private readonly Lock _gate = new();
    private readonly Queue<Fault> _queued = new();
    private readonly SortedDictionary<long, List<Fault>> _scheduled = [];

    /// <summary>A live model with nothing queued.</summary>
    public LiveFailures()
        : this(null)
    {
    }

    /// <summary>A live model with <paramref name="scheduled"/> handed out at their own steps.</summary>
    public LiveFailures(IEnumerable<Fault>? scheduled)
    {
        foreach (var fault in scheduled ?? [])
        {
            ArgumentOutOfRangeException.ThrowIfNegative(fault.Step, nameof(scheduled));
            if (!_scheduled.TryGetValue(fault.Step, out var list))
            {
                _scheduled[fault.Step] = list = [];
            }

            list.Add(fault);
        }
    }

    /// <inheritdoc />
    public string Name => "live";

    /// <inheritdoc />
    public string Summary => "faults asked for while the run goes (the dashboard), each at the next step, beside the run's own model";

    /// <summary>Faults queued and not yet drawn.</summary>
    public int Pending
    {
        get
        {
            lock (_gate)
            {
                return _queued.Count;
            }
        }
    }

    /// <summary>Queues <paramref name="fault"/> for the next step drawn (its own step is replaced by that step's).</summary>
    public void Queue(Fault fault)
    {
        ArgumentNullException.ThrowIfNull(fault.Ranks);
        lock (_gate)
        {
            _queued.Enqueue(fault);
        }
    }

    /// <inheritdoc />
    public IReadOnlyList<Fault> Draw(long step, IClusterTopology topology, Random seeded)
    {
        lock (_gate)
        {
            if (_queued.Count == 0 && !_scheduled.ContainsKey(step))
            {
                return [];
            }

            var faults = new List<Fault>();
            if (_scheduled.TryGetValue(step, out var scheduled))
            {
                faults.AddRange(scheduled);
                _scheduled.Remove(step);
            }

            while (_queued.TryDequeue(out var fault))
            {
                faults.Add(fault with { Step = step });
            }

            return faults;
        }
    }
}

/// <summary>
/// Steers a run of processes a launcher started (plan 23, phase C9: a localhost simulation started from the dashboard)
/// through their control file (<see cref="ClusterControlFile"/>): rank 0 reads it as each step begins and every rank pauses,
/// resumes or stops at that boundary, the stop with the trainer's checkpoint. Faults are not sent this way: they belong to
/// the simulated transport (an in-process run, <see cref="ClusterRunControl"/>). Give it to the launch
/// (<see cref="LocalhostLaunch.Control"/>).
/// </summary>
public sealed class LocalhostControl
{
    /// <summary>A control through the file at <paramref name="path"/> (created as "run").</summary>
    public LocalhostControl(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        Path = System.IO.Path.GetFullPath(path);
        ClusterControlFile.Write(Path, ClusterControlCommand.Run);
        File.Delete(ClusterControlFile.StatePath(Path));
    }

    /// <summary>The control file.</summary>
    public string Path { get; }

    /// <summary>What the file says now.</summary>
    public ClusterControlCommand Command => ClusterControlFile.Read(Path);

    /// <summary>What rank 0 last reported ("paused", "running", "stopped" and the step), or null before any report.</summary>
    public (string State, long Step)? Reported => ClusterControlFile.ReadState(Path);

    /// <summary>Asks the ranks to wait at the next step boundary.</summary>
    public void Pause() => ClusterControlFile.Write(Path, ClusterControlCommand.Pause);

    /// <summary>Lets them go on.</summary>
    public void Resume() => ClusterControlFile.Write(Path, ClusterControlCommand.Run);

    /// <summary>Asks them to stop at the next step boundary, the trainer writing its checkpoint.</summary>
    public void Stop() => ClusterControlFile.Write(Path, ClusterControlCommand.Stop);
}

// The run controls of the in-process runs going on, by their rendezvous name: the simulated transport's world finds its own.
internal static class RunControls
{
    private static readonly ConcurrentDictionary<string, ClusterRunControl> ByRun = new(StringComparer.Ordinal);

    public static void Add(string run, ClusterRunControl control) => ByRun[run] = control;

    public static void Remove(string run) => ByRun.TryRemove(run, out _);

    public static ClusterRunControl? Find(string? run) => run is not null && ByRun.TryGetValue(run, out var control) ? control : null;
}
