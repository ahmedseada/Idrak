// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Cluster;
using Idrak.Cli.Shared;
using Idrak.Cluster;

namespace Idrak.Cli.Commands.Cluster;

/// <summary>
/// The rank of a simulated cluster the current thread runs (<c>idrak cluster sim</c>), or null: the training commands
/// train on its device with its collectives (plan 23, phase C2).
/// </summary>
internal static class ClusterSession
{
    private static readonly AsyncLocal<ClusterRank?> Rank = new();

    /// <summary>This thread's rank, or null outside a simulated cluster.</summary>
    public static ClusterRank? Current
    {
        get => Rank.Value;
        set => Rank.Value = value;
    }

    /// <summary>
    /// Joins this process to the cluster it was started for (phase C5: a rank process of <c>idrak cluster start</c> or of
    /// <c>cluster sim --mode localhost</c>, which set <c>IDRAK_RANK</c> and <c>IDRAK_CLUSTER_TOPOLOGY</c>) and makes it the
    /// current rank until disposed; null when this process is not such a rank, or already is one.
    /// </summary>
    public static IDisposable? JoinStarted()
    {
        if (Current is not null || Environment.GetEnvironmentVariable("IDRAK_RANK") is not { Length: > 0 }
            || Environment.GetEnvironmentVariable("IDRAK_CLUSTER_TOPOLOGY") is not { Length: > 0 })
        {
            return null;
        }

        var joined = LocalhostCluster.Join();
        Current = joined.Rank;
        return new Leave(joined);
    }

    private sealed class Leave(LocalhostRank joined) : IDisposable
    {
        public void Dispose()
        {
            Current = null;
            joined.Dispose();
        }
    }
}

/// <summary>
/// <c>idrak cluster sim --topology FILE -- pretrain|tune|train|resume|reinforce …</c> (plan 23, phases C2 and C8; reinforce
/// from Plan 27, R3c): runs a training
/// command once per rank of a topology file, every rank a thread of this process on a device carved from the physical
/// device its entry names, joined by the simulated transport; rank 0 writes the run and prints, and each rank's network
/// time on the virtual clock is reported beside the wall time. The run's event log goes to its run folder, flushed as it
/// goes. <c>--profile</c> profiles the run (kernel, network and cluster probes; the predicted step beside the measured).
/// </summary>
internal sealed class ClusterSimCommand : Command
{
    // The commands whose training takes a cluster's rank (ClusterSession), and the option naming the folder rank 0 writes.
    private static readonly (string Command, string[] Output, string Hint)[] Trainers =
    [
        ("pretrain", ["-o", "--out"], "-o RUN"),
        ("tune", ["-o", "--out"], "-o DIR"),
        ("train", ["--run"], "--run DIR"),
        ("resume", [], "RUN"),
        ("reinforce", ["-o", "--out"], "-o DIR"),                           // Plan 27, R3c: rank 0 writes the adapter, its checkpoints and log
    ];

    /// <summary>The event log's file name in a run folder.</summary>
    internal const string EventLogFile = "cluster-events.idrklog";

    public override string Name => "cluster sim";

    public override string Summary => "Runs training on a simulated cluster: every rank of a topology file in this process";

    public override string Usage => """
        --topology FILE [options] -- pretrain|tune|train|resume|reinforce ARGS

        Runs the training command after -- once per rank of the topology file: every rank a thread of this process on
        a device carved from the physical device its entry names (the entry's memory as a hard cap), the ranks joined
        by the simulated transport. Each step's rows are shared across the ranks and the gradients summed after the
        backward pass, so the run trains as one process would; rank 0 writes the run (pretrain -o RUN, tune -o DIR,
        train --run DIR, resume RUN, reinforce -o DIR) and its output is shown. reinforce spreads each rollout's episodes
        over the ranks by shards (--shards, --spread; IDRAK_RL_SHARDS, IDRAK_RL_SPREAD) and gathers them, so with the same
        shard count a seeded run plays the episodes one process plays; rank 0 writes the adapter, its manifest, the
        checkpoints and the log, and --resume goes on from the checkpoint. Each training step draws the failure model's
        faults from the seed (a slow worker, a lossy link, a late joiner that sits out its first steps while the others
        train them, a dropped worker that ends the run naming it); the faults injected are listed with their replay key,
        and the event log (faults, each rank's clock per step) is written to the run folder as it goes (idrak cluster
        replay reads it; tune and reinforce write none into their adapter folder). At the end each rank's network time
        on the virtual clock (what the cluster the file describes would spend on its links) is printed beside the wall
        time. With --mode localhost every rank is a process of its own,
        joined by the real transports (shared memory inside a node, TCP between nodes) paced to the links' profiles
        (IDRAK_CLUSTER_THROTTLE=0 lets them run free); one line per rank shows its progress; faults are the simulated
        transport's, so a localhost run has none. With --profile the run is profiled (in-process): cold and warm apart
        (IDRAK_PROFILE_WARM warm runs, 1 here; the later runs train into folders of their own, removed after), the
        probes' rows (cpu, memory, the device's kernels where it times them, the network's collectives, transfers and
        gradient buckets, the cluster's simulated links) and the step measured here beside the step predicted for the
        topology's links. The other cluster settings come from the IDRAK_CLUSTER_ variables (idrak env lists them). The
        settings the library suggests from the first warm steps (a bf16 wire, local SGD, gradient accumulation, the bucket
        size, top-k, the hierarchical collective; never applied) are shown once each above the progress line and kept in
        the run's log (idrak suggest RUN lists them). With
        --dashboard the run is shown in a browser while it goes (idrak cluster ui's page): its ranks, links, losses and
        faults; pause, resume, stop with a checkpoint, and faults injected at the next step (in-process; a localhost run
        takes no faults). A run the dashboard steers is pretrain's (or resume of a pretrain run): its stop writes the
        step's checkpoint, which train and tune do not yet.

        Options:
              --topology FILE    the topology file (idrak cluster init writes one)
              --mode M           in-process (threads, the virtual clock) or localhost (processes) (IDRAK_CLUSTER_MODE)
              --collective NAME  the collective algorithm: ring, naive or hierarchical (hierarchical with several nodes)
              --wire W           how gradients travel: f32 or bf16 (f32)
              --local-sgd K      average the weights every K steps instead of the gradients every step
              --bucket-bytes N   a gradient bucket's size, such as 64KiB (measured on the first step)
              --topk F           send each rank's largest share F of the gradient values (0 to 1)
              --failures MODEL   the failure model: none, slow-worker, dropped-worker, link-loss, partition,
                                 late-join or mixed (the file's)
              --seed N           the seed of the simulation's jitter and faults (the file's)
              --placement P      single or pipeline: the whole model on every rank, or a stage each (pretrain)
              --stages N         a pipeline's stages, dividing the ranks (one per rank)
              --event-log FILE   where the event log goes (the run folder's cluster-events.idrklog)
              --profile          profile the run: the probes' rows and the predicted step beside the measured
              --dashboard        show and steer the run in a browser (127.0.0.1, IDRAK_CLUSTER_UI_HOST)
              --port N           the dashboard's port (IDRAK_CLUSTER_UI_PORT; 0: one the system picks)
              --no-browser       print the dashboard's address instead of opening it

        Examples:
          idrak cluster init -y --nodes 1 --devices cpu,cpu -o two.json
          idrak cluster sim --topology two.json -- pretrain my-model --tokens tokens.bin --steps 100 -o run
          idrak cluster sim --topology two.json --mode localhost -- pretrain my-model --tokens tokens.bin -o run
          idrak cluster sim --topology two.json --failures slow-worker --seed 7 -- pretrain my-model --tokens t.bin -o run
          idrak cluster sim --topology two.json --placement pipeline -- pretrain my-model --tokens t.bin -o run
          idrak cluster sim --topology two.json --profile -- train network.json --data houses.csv --run run
          idrak cluster sim --topology two.json -- tune train my-model chats.jsonl -o adapters/chat
          idrak cluster sim --topology two.json -- reinforce my-model --data prompts.jsonl --algorithm grpo --shards 4 -o rl
          idrak cluster sim --topology two.json --dashboard -- pretrain my-model --tokens tokens.bin --steps 500 -o run
        """;

    public override IReadOnlyCollection<string> ValueOptions =>
        ["--topology", "--collective", "--wire", "--local-sgd", "--bucket-bytes", "--topk", "--mode", "--failures", "--placement", "--stages", "--event-log", "--port"];   // --seed: a common option

    public override IReadOnlyCollection<string> Flags => ["--profile", "--dashboard", "--no-browser"];

    /// <summary>
    /// Runs <paramref name="inner"/> once per rank of <paramref name="topology"/> in this process (every rank a thread on its
    /// carved device), rank 0's output to <paramref name="output"/>, steered by <paramref name="control"/> when given: each
    /// rank's exit code and the error text of the ranks after rank 0.
    /// </summary>
    public static (ClusterRunResult Result, int[] Codes, string[] Errors) RunRanks(IClusterTopology topology, ClusterOptions options, IReadOnlyList<string> inner,
        TextWriter output, TextWriter error, ClusterRunControl? control)
    {
        var inv = CultureInfo.InvariantCulture;
        int world = topology.WorldSize;
        var codes = new int[world];
        var errors = new StringWriter[world];
        using var shown = SuggestionLine.Show(output, error);           // the settings suggested while the ranks train, once each (decision 4)
        var result = InProcessCluster.Run(topology, options, rank =>
        {
            errors[rank.Rank] = new StringWriter(inv);
            ClusterSession.Current = rank;
            try
            {
                // Rank 0's output is the run's; the others' errors are kept to name a rank that failed.
                codes[rank.Rank] = CommandLine.Run(inner, rank.Rank == 0 ? output : TextWriter.Null, rank.Rank == 0 ? error : errors[rank.Rank]);
            }
            finally
            {
                ClusterSession.Current = null;
            }

            if (codes[rank.Rank] != ExitCodes.Ok)
            {
                throw new InvalidOperationException($"{inner[0]} ended with exit code {codes[rank.Rank]}" + (errors[rank.Rank].ToString().Trim() is { Length: > 0 } text ? $": {text.Split('\n')[0]}" : ""));
            }
        }, control);
        return (result, codes, [.. errors.Select(e => e?.ToString() ?? "")]);
    }

    /// <summary>
    /// The checks every simulated run takes before it starts: the training command after -- and the folder rank 0 writes;
    /// a run the dashboard steers (<paramref name="steered"/>) must answer a stop with a checkpoint (pretrain, or resume of one).
    /// </summary>
    public static void CheckTrainer(IReadOnlyList<string> inner, bool steered = true)
    {
        if (inner.Count > 0 && inner[0] == "distill")
        {
            // Plan 27, R3c: on-policy distillation samples every step's prompts on every rank, without an exchange (not spread).
            throw new UsageException("cluster sim does not run distill: on-policy distillation samples every prompt on every rank, and its other ways are teacher passes "
                                     + "of their own. Give reinforce with a policy loss and --teacher (its episodes spread over the ranks), or tune.");
        }

        if (inner.Count == 0 || Trainers.All(t => t.Command != inner[0]))
        {
            throw new UsageException($"Give the training command after --: {string.Join(", ", Trainers.Select(t => t.Command))} (got {(inner.Count == 0 ? "nothing" : inner[0])}).");
        }

        var trainer = Trainers.First(t => t.Command == inner[0]);
        if (trainer.Output.Length > 0 && !inner.Any(a => trainer.Output.Contains(a) || trainer.Output.Any(o => o.StartsWith("--", StringComparison.Ordinal) && a.StartsWith(o + "=", StringComparison.Ordinal))))
        {
            throw new UsageException($"Give the folder rank 0 writes after {inner[0]} ({trainer.Hint}): every rank trains into one run, rank 0 writes it.");
        }

        if (steered && !Steerable(inner))
        {
            // A stop asked from the dashboard ends every rank as it begins the next step; the command answers it with that step's
            // checkpoint (pretrain since C9; train and tune since C8b).
            throw new UsageException($"The dashboard steers pretrain, tune and train runs (and resume of one): a stop ends with that step's checkpoint; {inner[0]} does not "
                                     + "write one on a stop. Run it with idrak cluster sim without --dashboard.");
        }
    }

    // Whether a run can be steered (paused, resumed, stopped with a checkpoint): pretrain, tune and train (plan 23, C8b:
    // the Trainer and the fine-tuner write the last finished step's checkpoint on a stop), and resume of a run of them.
    private static bool Steerable(IReadOnlyList<string> inner) => inner[0] is "pretrain" or "tune" or "train" or "resume";

    public override int Run(CommandContext context)
    {
        string path = context.Option("--topology") ?? throw new UsageException("Give the topology file with --topology (idrak cluster init writes one).");
        var inner = context.Positional;
        if (!context.Flag("--dashboard") && (context.Option("--port") is not null || context.Flag("--no-browser")))
        {
            throw new UsageException("--port and --no-browser are the dashboard's: add --dashboard.");
        }

        if (context.Flag("--dashboard"))
        {
            CheckTrainer(inner);
            if (context.Flag("--profile"))
            {
                throw new UsageException("--profile and --dashboard are one at a time: a profile runs the training more than once.");
            }
        }

        var (options, topology) = Prepare(context, path, inner);
        if (topology is null)
        {
            return ExitCodes.Failed;
        }

        if (context.Option("--mode") is { } mode)
        {
            options = options with
            {
                Simulation = options.Simulation with
                {
                    Mode = mode.ToLowerInvariant() switch
                    {
                        "in-process" or "inprocess" => SimulationMode.InProcess,
                        "localhost" => SimulationMode.Localhost,
                        _ => throw new UsageException($"--mode is in-process or localhost, not '{mode}'."),
                    },
                },
            };
        }

        if (options.Simulation.Mode == SimulationMode.Localhost
            && !string.Equals(options.Simulation.Failures ?? topology.Simulation.Failures, "none", StringComparison.OrdinalIgnoreCase))
        {
            // Faults are drawn and applied by the simulated transport's clock; real transports between processes have none.
            throw new UsageException($"--failures {options.Simulation.Failures ?? topology.Simulation.Failures} needs the simulated transport (--mode in-process); "
                                     + "a localhost run joins its processes by the real transports, which inject no faults.");
        }

        if (options.Simulation.Mode == SimulationMode.Localhost && context.Flag("--profile"))
        {
            throw new UsageException("--profile needs --mode in-process: the probes read this process's telemetry, and a localhost run's ranks are processes of their own.");
        }

        using var dashboard = context.Flag("--dashboard") ? Dashboard.Open(context, path) : null;
        if (options.Simulation.Mode == SimulationMode.Localhost)
        {
            return Localhost(context, path, topology, options, inner, dashboard);
        }

        ClusterRunControl? control = null;
        DashboardRun? watched = null;
        if (dashboard is not null)
        {
            control = new ClusterRunControl();
            watched = dashboard.Runs.Attach(RunKind.InProcess, path, inner, options, topology);
            watched.Bind(control);
            options = options with { Simulation = options.Simulation with { EventLog = options.Simulation.EventLog ?? watched.EventLogPath } };
        }

        var inv = CultureInfo.InvariantCulture;
        string model = options.Simulation.Failures ?? topology.Simulation.Failures;
        int runSeed = options.Simulation.Seed ?? topology.Simulation.Seed;
        options = WithEventLog(context, options, inner);
        context.Write($"simulating {topology.WorldSize} rank(s) of {path} in this process ({options.CollectiveFor(topology)}, {(options.Wire == WireType.BFloat16 ? "bf16" : "f32")} wire, "
                      + $"failures {model}, seed {runSeed.ToString(inv)}{(options.Pipelined ? $", pipeline of {options.Stages ?? topology.WorldSize} stage(s)" : "")}): idrak {string.Join(' ', inner)}");
        int[] codes;
        ClusterRunResult result;
        JsonNode? profile = null;
        if (context.Flag("--profile"))
        {
            (codes, result, profile) = Profiled(context, topology, options, inner);
        }
        else
        {
            try
            {
                (codes, result) = Simulate(context, topology, options, inner, control, watched?.Output(context.Output));
            }
            catch (Exception e) when (watched is not null)
            {
                watched.Finish("failed", e.Message);
                throw;
            }

            watched?.Finish(result, control);
        }

        var faults = result.Log?.Faults ?? [];
        if (faults.Count > 0)
        {
            context.Write($"faults injected ({faults.Count}): " + string.Join("; ", faults.Select(f =>
                $"{f.Kind} at step {f.Step.ToString(inv)} on rank(s) {string.Join(",", f.Ranks)} (replay with --seed {runSeed.ToString(inv)} --step {f.Step.ToString(inv)})")));
        }

        context.Write("network time on the virtual clock: " + string.Join(", ", result.Simulated.Select((t, r) => $"rank {r} {t.TotalSeconds.ToString("F3", inv)} s"))
                      + $"; wall {result.Wall.TotalSeconds.ToString("F1", inv)} s");
        if (options.Simulation.EventLog is { } eventLog && result.Log is not null)
        {
            context.Write($"event log: {eventLog} (idrak cluster replay --topology {path} --seed {runSeed.ToString(inv)} --step N --from {eventLog})");
        }

        context.WriteJson(new JsonObject
        {
            ["topology"] = Path.GetFullPath(path),
            ["ranks"] = topology.WorldSize,
            ["command"] = string.Join(' ', inner),
            ["failures"] = model,
            ["seed"] = runSeed,
            ["faults"] = new JsonArray([.. faults.Select(f => (JsonNode)new JsonObject
            {
                ["kind"] = f.Kind.ToString(),
                ["step"] = f.Step,
                ["ranks"] = new JsonArray([.. f.Ranks.Select(r => (JsonNode)r)]),
            })]),
            ["simulated_seconds"] = new JsonArray([.. result.Simulated.Select(t => (JsonNode)t.TotalSeconds)]),
            ["wall_seconds"] = result.Wall.TotalSeconds,
            ["exit_codes"] = new JsonArray([.. codes.Select(c => (JsonNode)c)]),
            ["event_log"] = options.Simulation.EventLog is { } log ? Path.GetFullPath(log) : null,
            ["profile"] = profile,
        });
        return Outcome(context, result, codes);
    }

    /// <summary>
    /// The options and topology a simulated run of <paramref name="inner"/> takes (the file, the collective, wire, seed,
    /// failures, placement and stages given); null topology when the file cannot be read (the error is written).
    /// </summary>
    internal static (ClusterOptions Options, ClusterTopology? Topology) Prepare(CommandContext context, string path, IReadOnlyList<string> inner)
    {
        CheckTrainer(inner, steered: false);

        ClusterOptions options;
        try
        {
            options = ClusterOptions.FromEnvironment() with { Topology = path };
        }
        catch (ArgumentException e)
        {
            throw new UsageException(e.Message);
        }

        if (context.Option("--collective") is { } collective)
        {
            options = options with { Collective = collective };
        }

        if (context.Option("--wire") is { } wire)
        {
            options = options with
            {
                Wire = wire.ToLowerInvariant() switch
                {
                    "f32" or "float32" => WireType.Float32,
                    "bf16" or "bfloat16" => WireType.BFloat16,
                    _ => throw new UsageException($"--wire is f32 or bf16, not '{wire}'."),
                },
            };
        }

        if (context.Option("--local-sgd") is { } localSgd)
        {
            options = options with { LocalSgdEvery = int.TryParse(localSgd, NumberStyles.None, CultureInfo.InvariantCulture, out int k) ? k : throw new UsageException($"--local-sgd is a whole number of steps, 0 or more, not '{localSgd}'.") };
        }

        if (context.Option("--bucket-bytes") is { } bucket)
        {
            long bytes;
            try
            {
                bytes = ClusterTopology.ParseMemory(bucket);
            }
            catch (FormatException)
            {
                bytes = 0;
            }

            options = options with { BucketBytes = bytes > 0 ? bytes : throw new UsageException($"--bucket-bytes is a size above 0 with its unit (64KiB, 4MB), not '{bucket}'.") };
        }

        if (context.Option("--topk") is { } topk)
        {
            options = options with
            {
                TopK = double.TryParse(topk, NumberStyles.Float, CultureInfo.InvariantCulture, out double share) && share is >= 0 and <= 1 ? share : throw new UsageException($"--topk is a share from 0 to 1, not '{topk}'."),
            };
        }

        if (context.Option("--seed") is { } seed)
        {
            options = options with { Simulation = options.Simulation with { Seed = int.TryParse(seed, NumberStyles.Integer, CultureInfo.InvariantCulture, out int s) ? s : throw new UsageException($"--seed is a whole number, not '{seed}'.") } };
        }

        if (context.Option("--failures") is { } failures)
        {
            options = options with { Simulation = options.Simulation with { Failures = failures } };
        }

        if (context.Option("--placement") is { } placement)
        {
            options = options with
            {
                Placement = placement.ToLowerInvariant() is "single" or "pipeline" ? placement.ToLowerInvariant() : throw new UsageException($"--placement is single or pipeline, not '{placement}'."),
            };
        }

        if (context.Option("--stages") is { } stages)
        {
            options = options with { Stages = int.TryParse(stages, NumberStyles.None, CultureInfo.InvariantCulture, out int n) && n >= 1 ? n : throw new UsageException($"--stages is a whole number, 1 or more, not '{stages}'.") };
        }

        if (options.Pipelined && inner[0] != "pretrain" && !(inner[0] == "resume"))
        {
            throw new UsageException($"--placement pipeline trains with pretrain (and resumes its runs); {inner[0]} trains the whole model on every rank.");
        }

        if (context.Option("--event-log") is { } eventLog)
        {
            options = options with { Simulation = options.Simulation with { EventLog = Path.GetFullPath(eventLog) } };
        }

        try
        {
            var topology = ClusterTopology.Load(path);
            _ = Collectives.Get(options.CollectiveFor(topology));
            _ = FailureModels.Get(options.Simulation.Failures ?? topology.Simulation.Failures);
            if (options.Stages is { } count && topology.WorldSize % count != 0)
            {
                throw new UsageException($"--stages {count} does not divide the {topology.WorldSize} rank(s) of {path}.");
            }

            return (options, topology);
        }
        catch (Exception e) when (e is InvalidDataException or NotSupportedException or FileNotFoundException or DirectoryNotFoundException)
        {
            context.Error(e.Message);
            return (options, null);
        }
    }

    // The event log in the run folder rank 0 writes, unless IDRAK_CLUSTER_EVENT_LOG or --event-log names one.
    internal static ClusterOptions WithEventLog(CommandContext context, ClusterOptions options, IReadOnlyList<string> inner)
    {
        if (options.Simulation.EventLog is not null || RunFolder(inner) is not { } folder)
        {
            return options;
        }

        Directory.CreateDirectory(folder);
        return options with { Simulation = options.Simulation with { EventLog = Path.Combine(Path.GetFullPath(folder), EventLogFile) } };
    }

    /// <summary>
    /// The run folder a training command writes (pretrain -o, train --run, resume RUN; another command's -o, as cluster start
    /// runs it, "idrak" first or not), or null (tune's adapter folder).
    /// </summary>
    public static string? RunFolder(IReadOnlyList<string> inner)
    {
        if (inner.Count > 0 && string.Equals(inner[0], "idrak", StringComparison.OrdinalIgnoreCase))
        {
            inner = [.. inner.Skip(1)];                                    // a command cluster start runs: idrak pretrain …
        }

        if (inner.Count == 0)
        {
            return null;
        }

        string? Value(params string[] names)
        {
            for (int i = 1; i < inner.Count; i++)
            {
                foreach (string name in names)
                {
                    if (inner[i] == name && i + 1 < inner.Count)
                    {
                        return inner[i + 1];
                    }

                    if (name.StartsWith("--", StringComparison.Ordinal) && inner[i].StartsWith(name + "=", StringComparison.Ordinal))
                    {
                        return inner[i][(name.Length + 1)..];
                    }
                }
            }

            return null;
        }

        return inner[0] switch
        {
            "pretrain" => Value("-o", "--out"),
            "train" => Value("--run"),
            "resume" => inner.Skip(1).FirstOrDefault(a => !a.StartsWith('-')) is { } run && Directory.Exists(run) ? run : null,
            "tune" or "reinforce" => null,                                      // its -o is the adapter's folder, not a run's
            _ => Value("-o", "--out"),                                          // another command cluster start runs
        };
    }

    /// <summary>
    /// Runs <paramref name="inner"/> on every rank of <paramref name="topology"/> in this process (rank 0's output is the
    /// run's; the others' errors are kept to name a rank that failed); each rank's exit code beside the run's result.
    /// </summary>
    internal static (int[] Codes, ClusterRunResult Result) Simulate(CommandContext context, ClusterTopology topology, ClusterOptions options, IReadOnlyList<string> inner,
        ClusterRunControl? control = null, TextWriter? output = null)
    {
        var (result, codes, _) = RunRanks(topology, options, inner, output ?? context.Output, context.ErrorOutput, control);
        return (codes, result);
    }

    // The run under the profiler: the first (cold) run trains into the run folder, each later one into a folder of its own,
    // removed after; the report's rows go to the output (JSON with -j), with the predicted step beside the measured.
    private static (int[] Codes, ClusterRunResult Result, JsonNode? Profile) Profiled(CommandContext context, ClusterTopology topology, ClusterOptions options,
        IReadOnlyList<string> inner)
    {
        ProfileOptions profileOptions;
        try
        {
            var probes = ProfileProbes.Names.Where(n => ProfileProbes.Find(n) is { Intrusive: false } || n == "kernels").ToList();
            profileOptions = ProfileOptions.FromEnvironment(new ProfileOptions { WarmRuns = 1, Probes = probes });
        }
        catch (ArgumentException e)
        {
            throw new UsageException(e.Message);
        }

        int[] codes = [];
        ClusterRunResult? first = null;
        var extra = new List<string>();
        string? folder = RunFolder(inner) ?? Value(inner, "-o", "--out");
        var device = ClusterDevice(topology, options);
        ProfileReport report;
        try
        {
            report = Profiler.Run($"cluster sim {inner[0]}", device, run =>
            {
                var args = inner.ToList();
                var runOptions = options;
                if (run.Index > 0 && folder is not null)
                {
                    string other = $"{Path.TrimEndingDirectorySeparator(Path.GetFullPath(folder))}.profile-{run.Index.ToString(CultureInfo.InvariantCulture)}";
                    extra.Add(other);
                    args = [.. args.Select(a => a == folder ? other : a)];
                    runOptions = options with { Simulation = options.Simulation with { EventLog = null } };
                }

                var (runCodes, result) = Simulate(context, topology, runOptions, args);
                if (first is null)
                {
                    (codes, first) = (runCodes, result);
                }

                if (!result.Succeeded)
                {
                    throw new InvalidOperationException(string.Join("; ", result.Errors.Select((e, r) => e is null ? null : $"rank {r}: {e.Message}").OfType<string>()));
                }
            }, profileOptions);
        }
        catch (InvalidOperationException) when (first is not null && !first.Succeeded)
        {
            return (codes, first, null);
        }
        finally
        {
            foreach (string other in extra.Where(Directory.Exists))
            {
                Directory.Delete(other, recursive: true);
            }
        }

        var inv = CultureInfo.InvariantCulture;
        using var text = new StringWriter(inv);
        report.Write(text, context.Json ? ProfileWriters.Json : ProfileWriters.Table);
        if (!context.Json)
        {
            context.Write(text.ToString().TrimEnd());
        }

        if (report["measured step"] is { } measured && report["predicted step"] is { } predicted)
        {
            context.Write($"step: measured {Ms(measured.Value ?? measured.Cold)} here, predicted {Ms(predicted.Value ?? predicted.Cold)} on the topology's links "
                          + $"(cold {Ms(measured.Cold)} and {Ms(predicted.Cold)})");
        }

        JsonNode? json = null;
        if (context.Json)
        {
            try
            {
                json = JsonNode.Parse(text.ToString());
            }
            catch (System.Text.Json.JsonException)
            {
                json = text.ToString();
            }
        }

        return (codes, first!, json);

        static string Ms(double? ms) => ms is { } v ? v.ToString("F2", CultureInfo.InvariantCulture) + " ms" : "-";
    }

    private static string? Value(IReadOnlyList<string> inner, params string[] names)
    {
        for (int i = 1; i + 1 < inner.Count; i++)
        {
            if (names.Contains(inner[i]))
            {
                return inner[i + 1];
            }
        }

        return null;
    }

    // The device the profile is reported for: rank 0's physical device.
    private static Device ClusterDevice(ClusterTopology topology, ClusterOptions options) =>
        Device.Parse(options.Simulation.CarveFrom ?? topology.Nodes[0].Devices[0].Device);

    /// <summary>The exit code of a simulated run, naming every rank that failed.</summary>
    internal static int Outcome(CommandContext context, ClusterRunResult result, int[] codes)
    {
        if (result.Succeeded)
        {
            return ExitCodes.Ok;
        }

        foreach (var (error, rank) in result.Errors.Select((e, r) => (e, r)).Where(p => p.e is not null && p.r > 0))
        {
            context.Error($"rank {rank}: {error!.Message}");
        }

        return codes.Length > 0 && codes[0] != ExitCodes.Ok ? codes[0] : ExitCodes.Failed;
    }

    /// <summary>The settings a localhost run's processes take from this command beside the environment they inherit.</summary>
    public static Dictionary<string, string> LocalhostEnvironment(ClusterOptions options)
    {
        var environment = new Dictionary<string, string>(StringComparer.Ordinal) { ["IDRAK_CLUSTER_WIRE"] = options.Wire == WireType.BFloat16 ? "bf16" : "f32" };
        if (options.Collective is { } collective)
        {
            environment["IDRAK_CLUSTER_COLLECTIVE"] = collective;
        }

        if (options.Simulation.Seed is { } seed)
        {
            environment["IDRAK_CLUSTER_SEED"] = seed.ToString(CultureInfo.InvariantCulture);
        }

        if (options.LocalSgdEvery > 0)
        {
            environment["IDRAK_CLUSTER_LOCAL_SGD"] = options.LocalSgdEvery.ToString(CultureInfo.InvariantCulture);
        }

        if (options.BucketBytes is { } bucketBytes)
        {
            environment["IDRAK_CLUSTER_BUCKET_BYTES"] = bucketBytes.ToString(CultureInfo.InvariantCulture) + "B";
        }

        if (options.TopK > 0)
        {
            environment["IDRAK_CLUSTER_TOPK"] = options.TopK.ToString("R", CultureInfo.InvariantCulture);
        }

        environment["IDRAK_CLUSTER_PLACEMENT"] = options.Placement;
        if (options.Stages is { } stages)
        {
            environment["IDRAK_CLUSTER_STAGES"] = stages.ToString(CultureInfo.InvariantCulture);
        }

        return environment;
    }

    // --mode localhost: every rank a process of its own (this tool again), joined by the real transports.
    private static int Localhost(CommandContext context, string path, ClusterTopology topology, ClusterOptions options, IReadOnlyList<string> inner, DashboardServer? dashboard)
    {
        var inv = CultureInfo.InvariantCulture;
        var environment = LocalhostEnvironment(options);

        int world = topology.WorldSize;
        var ranks = Enumerable.Range(0, world).ToList();
        var devices = ranks.Select(r => options.Simulation.CarveFrom ?? topology.Nodes[topology.Place(r).Node].Devices[topology.Place(r).Local].Device).ToList();
        context.Write($"simulating {world} rank(s) of {path} as processes on this machine (real transports{(options.Simulation.Throttle ? ", paced to the links" : "")}): "
                      + $"idrak {string.Join(' ', inner)}");
        using var board = new RankBoard(context, $"{inner[0]} on {world} rank(s)", ranks, devices, line => line.Rank == 0);
        DashboardRun? watched = dashboard?.Runs.Attach(RunKind.Localhost, path, inner, options, topology);
        var steer = watched is null ? null : new LocalhostControl(watched.ControlPath);
        watched?.Bind(steer!);
        LocalhostRunResult result;
        try
        {
            result = LocalhostCluster.Run(path, options, ClusterLaunch.Program(["idrak", .. inner], environment) with { Control = steer }, null,
                line =>
                {
                    board.Line(line);
                    watched?.Line(line);
                }, context.TimeoutToken);
        }
        catch (Exception e) when (e is NotSupportedException or InvalidOperationException or ArgumentException)
        {
            watched?.Finish("failed", e.Message);
            context.Error(e.Message);
            return ExitCodes.Failed;
        }

        watched?.Finish(result);
        board.Finish(result.Ranks);
        context.Write($"localhost: {world} process(es), wall {result.Wall.TotalSeconds.ToString("F1", inv)} s; "
                      + string.Join(", ", result.Ranks.Select(r => $"rank {r.Rank} {(r.ExitCode switch { 0 => "done", null => "ended", { } code => $"exit {code}" })}")));
        context.WriteJson(new JsonObject
        {
            ["topology"] = Path.GetFullPath(path),
            ["ranks"] = world,
            ["mode"] = "localhost",
            ["command"] = string.Join(' ', inner),
            ["wall_seconds"] = result.Wall.TotalSeconds,
            ["exit_codes"] = new JsonArray([.. result.Ranks.Select(r => (JsonNode?)r.ExitCode)]),
        });
        if (result.Succeeded)
        {
            return ExitCodes.Ok;
        }

        context.Error(result.Failure ?? "a rank failed");
        foreach (var rank in result.Ranks.Where(r => r.ExitCode != 0 && r.Rank > 0))
        {
            context.Error($"rank {rank.Rank}: " + (rank.Errors.LastOrDefault(l => l.Length > 0) ?? "ended"));
        }

        return ClusterLaunch.ExitCode(result);
    }
}
