// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading.Channels;
using Idrak.Abstraction.Cluster;
using Idrak.Cluster;

namespace Idrak.Cli.Commands.Cluster;

/// <summary>How a run on the dashboard runs.</summary>
internal enum RunKind
{
    /// <summary>Every rank a thread of this process on the simulated transport: steered through the library's run control.</summary>
    InProcess,

    /// <summary>Every rank a process of this machine (a localhost simulation): pause, resume and stop through the launcher's control file.</summary>
    Localhost,

    /// <summary>A run folder another command writes (<c>cluster start</c> across machines): followed read-only.</summary>
    Followed,
}

/// <summary>
/// One run the dashboard shows (plan 23, phase C9): its kind, command and folder, its state, the controls it takes, and its
/// records: the run's own changes (started, paused, a live fault queued and drawn, stopped), the library's telemetry while
/// a page listens (an in-process run's), and the lines its processes write (a localhost run's). Records are JSON objects
/// with an "event" key, the telemetry ones as <see cref="TelemetryJson"/> writes them in a run's log.jsonl, so the page
/// reads a live run and a followed folder alike. The last <see cref="DashboardSettings.History"/> are kept for a page that
/// connects late; each listener gets them, then every new one, in order.
/// </summary>
internal sealed class DashboardRun
{
    private static readonly string[] Ended = ["stopped", "finished", "failed"];

    private readonly object _gate = new();
    private readonly Queue<string> _history = new();
    private readonly List<Channel<string>> _listeners = [];
    private readonly DashboardRuns _owner;
    private readonly Dictionary<(int From, int To), LinkTotals> _links = [];
    private readonly SortedDictionary<int, RankTotals> _collectives = [];
    private (string State, long Step)? _reported;

    public DashboardRun(DashboardRuns owner, string id, RunKind kind, string? topology, IReadOnlyList<string> command, string? folder, int world, int seed, string failures)
    {
        _owner = owner;
        Id = id;
        Kind = kind;
        Topology = topology is null ? null : Path.GetFullPath(topology);
        Command = [.. command];
        Folder = folder is null ? null : Path.GetFullPath(folder);
        World = world;
        Seed = seed;
        Failures = failures;
        State = kind == RunKind.Followed ? "following" : "starting";
        if (Command.Count > 0 && Command[0] == "reinforce" && OptionOf(Command, "-o", "--out") is { } output)
        {
            // A reinforce run (Plan 27, R11): its -o is the adapter's folder; its iterations come from its JSON log there (or
            // where IDRAK_RL_LOG puts it), from this run's first line on.
            OutputFolder = Path.GetFullPath(output);
            Figures = new ReinforceFigures(Environment.GetEnvironmentVariable("IDRAK_RL_LOG") is { Length: > 0 } log ? Path.GetFullPath(log.Trim())
                : Path.Combine(OutputFolder, ReinforceCommandLog));
        }

        _control = Path.Combine(Folder ?? Path.GetTempPath(), Folder is null ? $"idrak-cluster-control-{Guid.NewGuid():N}" : "cluster-control");
    }

    // The JSON log idrak reinforce writes in its folder (ReinforceCommand.LogFile).
    private const string ReinforceCommandLog = Commands.Train.ReinforceCommand.LogFile;

    private readonly string _control;

    public string Id { get; }

    public RunKind Kind { get; }

    public string? Topology { get; }

    public IReadOnlyList<string> Command { get; }

    public string? Folder { get; }

    public int World { get; }

    public int Seed { get; }

    public string Failures { get; }

    /// <summary>The folder a reinforce run writes (its -o: the adapter, the checkpoints, the log), or null for another command.</summary>
    public string? OutputFolder { get; }

    /// <summary>A reinforce run's iterations, rewards and pass rate, read from its JSON log; null for another command.</summary>
    public ReinforceFigures? Figures { get; }

    /// <summary>What the run counts: a reinforce run's iterations, another's steps.</summary>
    public string Unit => Figures is null ? "step" : "iteration";

    public DateTimeOffset Started { get; } = DateTimeOffset.UtcNow;

    public string State { get; private set; }

    public long Step { get; private set; } = -1;

    public string? Detail { get; private set; }

    public ClusterRunControl? Control { get; private set; }

    public LocalhostControl? Localhost { get; private set; }

    /// <summary>Ends the run's processes (a localhost run) when the dashboard closes.</summary>
    public CancellationTokenSource Cancel { get; } = new();

    /// <summary>The thread running a run the page started, or null.</summary>
    public Thread? Worker { get; set; }

    public bool Going => !Ended.Contains(State) && Kind != RunKind.Followed;

    /// <summary>The event log of an in-process run, in its run folder.</summary>
    public string? EventLogPath => Folder is null ? null : Path.Combine(Folder, ClusterSimCommand.EventLogFile);

    /// <summary>The control file of a localhost run, in its run folder (a file of its own in the temporary folder for a run without one: reinforce's -o is an adapter's).</summary>
    public string ControlPath => _control;

    /// <summary>Why <paramref name="action"/> (pause, resume, stop, faults) is refused on this run, or null when it is taken.</summary>
    public string? Refusal(string action)
    {
        if (Kind == RunKind.Followed)
        {
            return $"this run is followed read-only from its run folder {Folder} (a cluster start run's, on the machines that run it): "
                   + "pause, resume and stop it there; faults belong to simulated runs";
        }

        if (!Going)
        {
            return $"the run is {State}: there is nothing to {(action == "faults" ? "inject a fault into" : action)}";
        }

        if (action == "faults" && Kind == RunKind.Localhost)
        {
            return "faults belong to the simulated transport: a localhost run joins its processes by the real transports (start the run in-process to inject faults)";
        }

        return null;
    }

    /// <summary>The run as the runs list shows it.</summary>
    public JsonObject Summary() => new()
    {
        ["id"] = Id,
        ["kind"] = Kind switch { RunKind.InProcess => "in-process", RunKind.Localhost => "localhost", _ => "followed" },
        ["state"] = State,
        ["step"] = Step,
        ["detail"] = Detail,
        ["command"] = string.Join(' ', Command),
        ["folder"] = Folder,
        ["topology"] = Topology,
        ["world"] = World,
        ["seed"] = Seed,
        ["failures"] = Failures,
        ["unit"] = Unit,
        ["figures"] = Figures?.Read(),
        ["started"] = Started.ToString("O", CultureInfo.InvariantCulture),
        ["controls"] = new JsonObject
        {
            ["pause"] = Refusal("pause"), ["resume"] = Refusal("resume"), ["stop"] = Refusal("stop"), ["faults"] = Refusal("faults"),
        },
    };

    /// <summary>Takes a run control's changes (and, while a page listens, each rank's steps) as records.</summary>
    public void Bind(ClusterRunControl control)
    {
        Control = control;
        control.Observed = _owner.Listening > 0;
        control.Changed += change => Changed(change);
        control.RankStepped += step => Record(new JsonObject
        {
            ["event"] = "rank_step", ["rank"] = step.Rank, ["step"] = step.Step, ["wall_ms"] = step.Wall.TotalMilliseconds,
            ["in_use"] = step.Memory.InUse, ["peak"] = step.Memory.Peak, ["limit"] = step.Memory.Limit, ["offloaded"] = step.Memory.Offloaded,
        });
        SetState("running", null);
    }

    /// <summary>Steers a localhost run through its control file.</summary>
    public void Bind(LocalhostControl control)
    {
        Localhost = control;
        SetState("running", null);
    }

    /// <summary>A line a rank's process wrote (a localhost run): a record, and the step its progress names.</summary>
    public void Line(LocalhostLine line) => Record(new JsonObject
    {
        ["event"] = "rank_line", ["rank"] = line.Rank, ["error"] = line.IsError, ["text"] = line.Text,
    });

    /// <summary>A writer of rank 0's output: each line a record, and passed on to <paramref name="also"/>.</summary>
    public TextWriter Output(TextWriter? also) => new LineRecorder(this, also, false);

    /// <summary>A writer of rank 0's errors: each line a record.</summary>
    public TextWriter Errors(TextWriter? also) => new LineRecorder(this, also, true);

    /// <summary>The run ended as <paramref name="state"/> ("failed", …) for <paramref name="detail"/>.</summary>
    public void Finish(string state, string? detail) => SetState(state, detail);

    /// <summary>An in-process run ended: stopped as asked, finished, or failed naming each rank's error; its faults with their replay key.</summary>
    public void Finish(ClusterRunResult result, ClusterRunControl? control)
    {
        var inv = CultureInfo.InvariantCulture;
        var faults = result.Log?.Faults ?? [];
        var live = result.Log?.LiveFaults ?? [];
        Record(new JsonObject
        {
            ["event"] = "run_faults",
            ["seed"] = Seed,
            ["faults"] = new JsonArray([.. faults.Select(f => (JsonNode)new JsonObject
            {
                ["kind"] = f.Kind.ToString(), ["step"] = f.Step, ["ranks"] = new JsonArray([.. f.Ranks.Select(r => (JsonNode)r)]), ["live"] = live.Contains(f),
                ["key"] = string.Create(inv, $"--seed {Seed} --step {f.Step}"),
            })]),
            ["event_log"] = result.Log is null ? null : EventLogPath,
            ["simulated_seconds"] = new JsonArray([.. result.Simulated.Select(t => (JsonNode)t.TotalSeconds)]),
            ["wall_seconds"] = result.Wall.TotalSeconds,
        });
        if (!result.Succeeded)
        {
            SetState("failed", string.Join("; ", result.Errors.Select((e, r) => e is null ? null : $"rank {r}: {e.Message}").Where(m => m is not null)));
        }
        else if (control?.StopAfter is { } after)
        {
            SetState("stopped", Figures is not null ? Stopped()
                : after < 0 ? "stopped before the first step" : string.Create(inv, $"stopped after step {after}; its checkpoint is in {Folder}"));
        }
        else
        {
            SetState("finished", string.Create(inv, $"{faults.Count} fault(s); wall {result.Wall.TotalSeconds:F1} s"));
        }
    }

    /// <summary>A localhost run ended: stopped as asked (rank 0 reported it), finished, or failed naming why.</summary>
    public void Finish(LocalhostRunResult result)
    {
        PollState();
        bool stopped = Localhost?.Reported is ("stopped", _);
        SetState(!result.Succeeded ? "failed" : stopped ? "stopped" : "finished",
            !result.Succeeded ? result.Failure ?? string.Join("; ", result.Ranks.Where(r => r.ExitCode != 0).Select(r => $"rank {r.Rank}: {r.Errors.LastOrDefault(l => l.Length > 0) ?? "ended"}"))
            : (stopped && Figures is not null ? Stopped() + "; " : "") + string.Create(CultureInfo.InvariantCulture, $"wall {result.Wall.TotalSeconds:F1} s"));
    }

    // A reinforce run that stopped: the iterations its log holds (the ranks agreed on the last), and where its checkpoint is.
    private string Stopped() => Figures!.Read()["iteration"] is { } done && (int)done > 0
        ? string.Create(CultureInfo.InvariantCulture, $"stopped after iteration {(int)done}; its checkpoint is in {Path.Combine(OutputFolder!, "checkpoints")}, the adapter so far in {OutputFolder}")
        : "stopped before the first iteration";

    /// <summary>A localhost run: what rank 0 reported through the control file, as a record when it changed.</summary>
    public void PollState()
    {
        (string State, long Step) reported;
        lock (_gate)
        {
            if (Localhost?.Reported is not { } now || now == _reported)
            {
                return;
            }

            _reported = reported = now;
        }

        if (Going)
        {
            SetState(reported.State switch { "paused" => "paused", "stopped" => "stopping", _ => "running" }, $"rank 0 reports the ranks {reported.State} at step {reported.Step}", reported.Step);
        }
    }

    /// <summary>Asks the run to pause at its next step boundary; the step it will wait at (-1 for a localhost run: rank 0 says).</summary>
    public long Pause()
    {
        if (Control is { } control)
        {
            return control.Pause();
        }

        Localhost?.Pause();
        SetState("pausing", "asked to pause at the next step boundary");
        return -1;
    }

    public void Resume()
    {
        if (Control is { } control)
        {
            control.Resume();
            return;
        }

        Localhost?.Resume();
        SetState("running", "asked to go on");
    }

    public long Stop()
    {
        if (Control is { } control)
        {
            return control.Stop();
        }

        Localhost?.Stop();
        SetState("stopping", "asked to stop at the next step boundary");
        return -1;
    }

    /// <summary>Records <paramref name="record"/> (an "event" object) for every listener and the history.</summary>
    public void Record(JsonObject record)
    {
        record["time"] ??= DateTimeOffset.UtcNow.ToString("O", CultureInfo.InvariantCulture);
        Record(record.ToJsonString());
    }

    /// <summary>Records one JSON object already written.</summary>
    public void Record(string json)
    {
        lock (_gate)
        {
            _history.Enqueue(json);
            while (_history.Count > _owner.Settings.History)
            {
                _history.Dequeue();
            }

            foreach (var listener in _listeners)
            {
                listener.Writer.TryWrite(json);
            }
        }
    }

    // A listener joins: every record kept, then each new one on the channel (in one lock, so none is lost or doubled).
    internal (string[] Kept, Channel<string> Channel) Join()
    {
        var channel = Channel.CreateBounded<string>(new BoundedChannelOptions(Math.Max(1, _owner.Settings.History))
        {
            FullMode = BoundedChannelFullMode.DropOldest, SingleReader = true, SingleWriter = false,
        });
        lock (_gate)
        {
            _listeners.Add(channel);
            return ([.. _history], channel);
        }
    }

    internal void Leave(Channel<string> channel)
    {
        lock (_gate)
        {
            _listeners.Remove(channel);
        }

        channel.Writer.TryComplete();
    }

    // The telemetry of an in-process run while a page listens: transfers and collectives added up per pair and per rank,
    // sent as one "links" and one "collectives" record before each step's batch record.
    internal void Transfer(in TransferCompleted e)
    {
        lock (_gate)
        {
            var key = (e.From, e.To);
            var totals = _links.GetValueOrDefault(key);
            _links[key] = totals with
            {
                Count = totals.Count + 1, Bytes = totals.Bytes + e.Bytes, Simulated = totals.Simulated + e.Simulated.TotalSeconds, Wall = totals.Wall + e.Wall.TotalSeconds,
            };
        }
    }

    internal void Collective(in CollectiveCompleted e)
    {
        lock (_gate)
        {
            var totals = _collectives.GetValueOrDefault(e.Rank);
            _collectives[e.Rank] = new RankTotals(totals.Count + 1, totals.Sent + e.BytesSent, totals.Received + e.BytesReceived, totals.Blocked + e.Blocked.TotalSeconds,
                totals.Simulated + e.Simulated.TotalSeconds, totals.Wall + e.Wall.TotalSeconds, e.SlowestRank, e.Collective);
        }
    }

    internal void Batch(long step, string json)
    {
        JsonObject links, collectives;
        lock (_gate)
        {
            links = new JsonObject
            {
                ["event"] = "links", ["step"] = step,
                ["pairs"] = new JsonArray([.. _links.OrderBy(l => l.Key).Select(l => (JsonNode)new JsonObject
                {
                    ["from"] = l.Key.From, ["to"] = l.Key.To, ["messages"] = l.Value.Count, ["bytes"] = l.Value.Bytes,
                    ["simulated_ms"] = l.Value.Simulated * 1e3, ["wall_ms"] = l.Value.Wall * 1e3,
                })]),
            };
            collectives = new JsonObject
            {
                ["event"] = "collectives", ["step"] = step,
                ["ranks"] = new JsonArray([.. _collectives.Select(c => (JsonNode)new JsonObject
                {
                    ["rank"] = c.Key, ["count"] = c.Value.Count, ["bytes_sent"] = c.Value.Sent, ["bytes_received"] = c.Value.Received,
                    ["blocked_ms"] = c.Value.Blocked * 1e3, ["simulated_ms"] = c.Value.Simulated * 1e3, ["wall_ms"] = c.Value.Wall * 1e3,
                    ["slowest_rank"] = c.Value.Slowest, ["collective"] = c.Value.Name,
                })]),
            };
            _collectives.Clear();                               // a step's collectives; the links add up over the run
            Step = Math.Max(Step, step);
            Record(links);                                      // the three together, nothing between them
            Record(collectives);
            Record(json);
        }
    }

    private void Changed(ClusterRunChange change)
    {
        if (change.Fault is { } fault)
        {
            bool drawn = fault.Step >= 0;
            Record(new JsonObject
            {
                ["event"] = "live_fault", ["stage"] = drawn ? "injected" : "queued", ["kind"] = fault.Kind.ToString(),
                ["ranks"] = new JsonArray([.. fault.Ranks.Select(r => (JsonNode)r)]), ["step"] = drawn ? fault.Step : change.Step,
                ["duration_ms"] = fault.Duration.TotalMilliseconds, ["factor"] = fault.Factor, ["detail"] = change.Detail,
                ["key"] = drawn ? string.Create(CultureInfo.InvariantCulture, $"--seed {Seed} --step {fault.Step}") : null,
            });
            return;
        }

        SetState(change.State switch
        {
            ClusterRunState.Pausing => "pausing",
            ClusterRunState.Paused => "paused",
            ClusterRunState.Stopping => "stopping",
            _ => "running",
        }, change.Detail, change.Step);
    }

    private void SetState(string state, string? detail, long? step = null)
    {
        lock (_gate)
        {
            State = state;
            Detail = detail;
        }

        Record(new JsonObject { ["event"] = "run", ["id"] = Id, ["state"] = state, ["step"] = step ?? Step, ["detail"] = detail });
    }

    // The value of an option of `command` (NAME VALUE, or --NAME=VALUE), or null.
    private static string? OptionOf(IReadOnlyList<string> command, params string[] names)
    {
        for (int i = 1; i < command.Count; i++)
        {
            foreach (string name in names)
            {
                if (command[i] == name && i + 1 < command.Count)
                {
                    return command[i + 1];
                }

                if (name.StartsWith("--", StringComparison.Ordinal) && command[i].StartsWith(name + "=", StringComparison.Ordinal))
                {
                    return command[i][(name.Length + 1)..];
                }
            }
        }

        return null;
    }

    private readonly record struct LinkTotals(long Count, long Bytes, double Simulated, double Wall);

    private readonly record struct RankTotals(long Count, long Sent, long Received, double Blocked, double Simulated, double Wall, int Slowest, string Name);

    // Rank 0's output, line by line, as records.
    private sealed class LineRecorder(DashboardRun run, TextWriter? also, bool error) : TextWriter
    {
        private readonly StringBuilder _line = new();

        public override Encoding Encoding => Encoding.UTF8;

        public override void Write(char value)
        {
            also?.Write(value);
            lock (_line)
            {
                if (value == '\n')
                {
                    Flush(_line);
                }
                else if (value != '\r')
                {
                    _line.Append(value);
                }
            }
        }

        public override void Write(string? value)
        {
            if (value is null)
            {
                return;
            }

            also?.Write(value);
            lock (_line)
            {
                foreach (char c in value)
                {
                    if (c == '\n')
                    {
                        Flush(_line);
                    }
                    else if (c != '\r')
                    {
                        _line.Append(c);
                    }
                }
            }
        }

        public override void WriteLine(string? value)
        {
            Write(value);
            Write('\n');
        }

        public override void Flush() => also?.Flush();

        private void Flush(StringBuilder line)
        {
            if (line.Length > 0)
            {
                run.Record(new JsonObject { ["event"] = "output", ["rank"] = 0, ["error"] = error, ["text"] = line.ToString() });
                line.Clear();
            }
        }
    }
}

/// <summary>
/// The dashboard's runs (plan 23, phase C9) and the telemetry they take: one hook, subscribed only while a page listens
/// to a run's events (rule 78: nothing measured for nobody), whose records go to the in-process run that is going (one at a
/// time: the library's telemetry is the process's).
/// </summary>
internal sealed class DashboardRuns : IDisposable
{
    private readonly object _gate = new();
    private readonly List<DashboardRun> _runs = [];
    private readonly Hook _hook;
    private IDisposable? _subscription;
    private int _listening;
    private int _next;

    public DashboardRuns(DashboardSettings settings)
    {
        Settings = settings;
        _hook = new Hook(this);
    }

    public DashboardSettings Settings { get; }

    /// <summary>Pages listening to a run's events now.</summary>
    public int Listening => Volatile.Read(ref _listening);

    /// <summary>The in-process run going on, or null.</summary>
    public DashboardRun? Active
    {
        get
        {
            lock (_gate)
            {
                return _runs.LastOrDefault(r => r.Kind == RunKind.InProcess && r.Going);
            }
        }
    }

    public IReadOnlyList<DashboardRun> All
    {
        get
        {
            lock (_gate)
            {
                return [.. _runs];
            }
        }
    }

    public DashboardRun? Find(string id)
    {
        lock (_gate)
        {
            return _runs.FirstOrDefault(r => r.Id == id);
        }
    }

    /// <summary>Adds a run started here or elsewhere (a terminal's <c>cluster sim --dashboard</c>) of <paramref name="kind"/>.</summary>
    /// <exception cref="InvalidOperationException">An in-process run while another is going (they would share the process's telemetry).</exception>
    public DashboardRun Attach(RunKind kind, string topologyPath, IReadOnlyList<string> command, ClusterOptions options, IClusterTopology topology)
    {
        lock (_gate)
        {
            if (kind == RunKind.InProcess && Active is { } going)
            {
                throw new InvalidOperationException($"Run {going.Id} is going in this process: stop it first (an in-process run's telemetry is the process's).");
            }

            var run = new DashboardRun(this, NewId(), kind, topologyPath, command, ClusterSimCommand.RunFolder(command), topology.WorldSize,
                options.Simulation.Seed ?? topology.Simulation.Seed, options.Simulation.Failures ?? topology.Simulation.Failures);
            _runs.Add(run);
            return run;
        }
    }

    /// <summary>Follows a run folder another command writes (rank 0's log.jsonl; cluster start's ranks.jsonl), read-only.</summary>
    /// <exception cref="ArgumentException">Not a run folder; the message says what is missing.</exception>
    public DashboardRun Follow(string folder)
    {
        string full = Path.GetFullPath(folder);
        if (!File.Exists(Path.Combine(full, "log.jsonl")) && !File.Exists(Path.Combine(full, DashboardTail.RanksFile)))
        {
            throw new ArgumentException($"{full} is not a run folder being written: it has no log.jsonl (rank 0's log) and no {DashboardTail.RanksFile} (cluster start's rank lines).");
        }

        lock (_gate)
        {
            var run = new DashboardRun(this, NewId(), RunKind.Followed, null, [], full, 0, 0, "none");
            _runs.Add(run);
            return run;
        }
    }

    /// <summary>A page listens to <paramref name="run"/>: the telemetry is subscribed while any page does.</summary>
    public (string[] Kept, Channel<string> Channel, IDisposable Leave) Listen(DashboardRun run)
    {
        var (kept, channel) = run.Join();
        lock (_gate)
        {
            if (_listening++ == 0)
            {
                _subscription = Telemetry.Subscribe(_hook);
                Observe(true);
            }
        }

        return (kept, channel, new Leaving(this, run, channel));
    }

    public void Dispose()
    {
        lock (_gate)
        {
            _subscription?.Dispose();
            _subscription = null;
        }

        foreach (var run in All.Where(r => r.Going))
        {
            run.Stop();
            run.Cancel.Cancel();
        }

        foreach (var run in All)
        {
            if (run.Worker is { } worker && !worker.Join(TimeSpan.FromSeconds(60)))
            {
                run.Finish("failed", "the dashboard closed while the run went on");
            }

            run.Cancel.Dispose();
        }
    }

    private void Left(DashboardRun run, Channel<string> channel)
    {
        run.Leave(channel);
        lock (_gate)
        {
            if (--_listening == 0)
            {
                _subscription?.Dispose();
                _subscription = null;
                Observe(false);
            }
        }
    }

    private void Observe(bool on)
    {
        foreach (var run in _runs)
        {
            if (run.Control is { } control)
            {
                control.Observed = on;
            }
        }
    }

    private string NewId() => string.Create(CultureInfo.InvariantCulture, $"run-{++_next}");

    private sealed class Leaving(DashboardRuns owner, DashboardRun run, Channel<string> channel) : IDisposable
    {
        private int _done;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _done, 1) == 0)
            {
                owner.Left(run, channel);
            }
        }
    }

    // The library's telemetry for the in-process run going on, as the run's records.
    private sealed class Hook(DashboardRuns owner) : ITelemetryHook
    {
        public TelemetryLevel Levels => TelemetryLevel.Training | TelemetryLevel.Batches | TelemetryLevel.Network | TelemetryLevel.Suggestions;

        public void OnTrainingStarted(in TrainingStarted e) => Pass(e);

        // A setting the library suggested from the run's first warm steps (plan 23, decision 4): an event on the timeline.
        public void OnSuggestion(in SettingSuggestion e) => Pass(e);

        public void OnTrainingCompleted(in TrainingCompleted e) => Pass(e);

        public void OnBatchCompleted(in BatchCompleted e) => owner.Active?.Batch(e.Step, Json(e));

        public void OnTransfer(in TransferCompleted e) => owner.Active?.Transfer(e);

        public void OnCollective(in CollectiveCompleted e)
        {
            if (e.Operation == "step-summary")
            {
                Pass(e);
            }
            else
            {
                owner.Active?.Collective(e);
            }
        }

        public void OnGradientSync(in GradientsSynced e) => Pass(e);

        public void OnPipelineStep(in PipelineStepped e) => Pass(e);

        public void OnTransportChosen(in TransportChosen e) => Pass(e);

        public void OnRankState(in RankStateChanged e) => Pass(e);

        public void OnFault(in FaultInjected e) => Pass(e);

        private void Pass(object e) => owner.Active?.Record(Json(e));

        private static string Json(object e)
        {
            var buffer = new ArrayBufferWriter<byte>(256);
            using (var writer = new Utf8JsonWriter(buffer))
            {
                TelemetryJson.Write(writer, new TelemetryRecord(DateTimeOffset.UtcNow, e));
            }

            return Encoding.UTF8.GetString(buffer.WrittenSpan);
        }
    }
}

/// <summary>
/// Reads the new lines of a run's files as a listener waits (a followed or localhost run's folder: rank 0's log.jsonl and
/// cluster start's ranks.jsonl; a reinforce run's JSON log), from where the last read stopped, as bytes (rule 74), a partial
/// last line kept for the next.
/// </summary>
internal sealed class DashboardTail
{
    /// <summary>The file <c>cluster start</c> writes its ranks' lines to in the run folder.</summary>
    public const string RanksFile = "ranks.jsonl";

    private readonly (string Path, long Offset, byte[] Leftover)[] _files;

    /// <summary>The run folder's log.jsonl and ranks.jsonl, from their start.</summary>
    public DashboardTail(string folder)
        : this([(Path.Combine(folder, "log.jsonl"), 0), (Path.Combine(folder, RanksFile), 0)])
    {
    }

    /// <summary>These files, each from its offset (the bytes before it were another run's).</summary>
    public DashboardTail(IReadOnlyList<(string Path, long Offset)> files) => _files = [.. files.Select(f => (f.Path, f.Offset, Array.Empty<byte>()))];

    /// <summary>The complete lines written since the last read, each a JSON object.</summary>
    public List<string> Read()
    {
        var lines = new List<string>();
        for (int i = 0; i < _files.Length; i++)
        {
            var (path, offset, rest) = _files[i];
            try
            {
                using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
                if (stream.Length <= offset)
                {
                    continue;
                }

                stream.Position = offset;
                byte[] bytes = new byte[rest.Length + (stream.Length - offset)];
                rest.CopyTo(bytes, 0);
                int read = stream.ReadAtLeast(bytes.AsSpan(rest.Length), bytes.Length - rest.Length, throwOnEndOfStream: false);
                var span = bytes.AsSpan(0, rest.Length + read);
                int start = 0;
                while (span[start..].IndexOf((byte)'\n') is var next and >= 0)
                {
                    var line = span.Slice(start, next).TrimEnd((byte)'\r');
                    if (line.Length > 0 && line[0] == (byte)'{')
                    {
                        lines.Add(Encoding.UTF8.GetString(line));
                    }

                    start += next + 1;
                }

                _files[i] = (path, offset + read, span[start..].ToArray());
            }
            catch (IOException)
            {
                // Not written yet: read on the next turn.
            }
        }

        return lines;
    }
}

/// <summary>
/// A reinforce run's figures for the dashboard (Plan 27, R11), from its JSON log (a line per iteration, the trainer's): the
/// iterations done of the run's, the mean reward (and the first one's), its spread, the pass rate, the loss and the last
/// evaluation. The log is read as bytes from where the last read stopped (rule 74), from this run's first line on (lines
/// already there when the run was attached were another run's, or the run it resumes).
/// </summary>
internal sealed class ReinforceFigures
{
    private static readonly string[] Kept = ["iteration", "iterations", "episodes", "reward_mean", "reward_spread", "reward_min", "reward_max", "pass_rate", "loss", "evaluation_reward"];

    private readonly Lock _gate = new();
    private readonly DashboardTail _tail;
    private JsonObject _figures;

    public ReinforceFigures(string log)
    {
        Log = log;
        try
        {
            Start = File.Exists(log) ? new FileInfo(log).Length : 0;
        }
        catch (IOException)
        {
            Start = 0;
        }

        _tail = new DashboardTail([(log, Start)]);
        _figures = new JsonObject { ["log"] = log, ["iteration"] = 0 };
    }

    /// <summary>The run's JSON log.</summary>
    public string Log { get; }

    /// <summary>Where this run's lines begin in it.</summary>
    public long Start { get; }

    /// <summary>A reader of this run's lines as they come (each listener its own).</summary>
    public DashboardTail Tail() => new([(Log, Start)]);

    /// <summary>The figures after the lines written so far (a copy).</summary>
    public JsonObject Read()
    {
        lock (_gate)
        {
            foreach (string line in _tail.Read())
            {
                if (JsonNode.Parse(line) is not JsonObject record || (string?)record["event"] is not "iteration")
                {
                    continue;
                }

                var figures = new JsonObject { ["log"] = Log };
                foreach (string key in Kept)
                {
                    if (record[key] is JsonValue value)
                    {
                        figures[key] = value.DeepClone();
                    }
                }

                figures["first_reward_mean"] = _figures["first_reward_mean"]?.DeepClone() ?? record["reward_mean"]?.DeepClone();
                if (record["evaluation_reward"] is null && _figures["evaluation_reward"] is { } evaluation)
                {
                    figures["evaluation_reward"] = evaluation.DeepClone();            // the last evaluation stays shown between them
                }

                _figures = figures;
            }

            return (JsonObject)_figures.DeepClone();
        }
    }
}
