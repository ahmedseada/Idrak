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
/// <c>idrak cluster sim --topology FILE -- pretrain …</c> (plan 23, phase C2): runs a training command once per rank of a
/// topology file, every rank a thread of this process on a device carved from the physical device its entry names, joined
/// by the simulated transport; rank 0 writes the run folder and prints, and each rank's network time on the virtual clock
/// is reported beside the wall time.
/// </summary>
internal sealed class ClusterSimCommand : Command
{
    // The commands whose training takes a simulated rank (ClusterSession); the others join with --cluster in phase C8.
    private static readonly string[] Trainers = ["pretrain"];

    public override string Name => "cluster sim";

    public override string Summary => "Runs training on a simulated cluster: every rank of a topology file in this process";

    public override string Usage => """
        --topology FILE [options] -- pretrain MODEL [options] -o RUN

        Runs the training command after -- once per rank of the topology file: every rank a thread of this process on
        a device carved from the physical device its entry names (the entry's memory as a hard cap), the ranks joined
        by the simulated transport. Each step's rows are shared across the ranks and the gradients summed after the
        backward pass, so the run trains as one process would; rank 0 writes the run folder and its output is shown.
        Each training step draws the failure model's faults from the seed (a slow worker, a lossy link, a late joiner
        that sits out its first steps while the others train them, a dropped worker that ends the run naming it); the
        faults injected are listed with their replay key. At the end each rank's network time on the virtual clock
        (what the cluster the file describes would spend on its links) is printed beside the wall time. With --mode
        localhost every rank is a process of its own, joined by the real transports (shared memory inside a node, TCP
        between nodes) paced to the links' profiles (IDRAK_CLUSTER_THROTTLE=0 lets them run free); one line per rank
        shows its progress; faults are the simulated transport's, so a localhost run has none. The other cluster
        settings come from the IDRAK_CLUSTER_ variables (idrak env lists them). With --dashboard the run is shown in
        a browser while it goes (idrak cluster ui's page): its ranks, links, losses and faults; pause, resume, stop
        with a checkpoint, and faults injected at the next step (in-process; a localhost run takes no faults).

        Options:
              --topology FILE    the topology file (idrak cluster init writes one)
              --mode M           in-process (threads, the virtual clock) or localhost (processes) (IDRAK_CLUSTER_MODE)
              --collective NAME  the collective algorithm: ring or naive (ring)
              --wire W           how gradients travel: f32 or bf16 (f32)
              --failures MODEL   the failure model: none, slow-worker, dropped-worker, link-loss, partition,
                                 late-join or mixed (the file's)
              --seed N           the seed of the simulation's jitter and faults (the file's)
              --dashboard        show and steer the run in a browser (127.0.0.1, IDRAK_CLUSTER_UI_HOST)
              --port N           the dashboard's port (IDRAK_CLUSTER_UI_PORT; 0: one the system picks)
              --no-browser       print the dashboard's address instead of opening it

        Examples:
          idrak cluster init -y --nodes 1 --devices cpu,cpu -o two.json
          idrak cluster sim --topology two.json -- pretrain my-model --tokens tokens.bin --steps 100 -o run
          idrak cluster sim --topology two.json --mode localhost -- pretrain my-model --tokens tokens.bin -o run
          idrak cluster sim --topology two.json --failures slow-worker --seed 7 -- pretrain my-model --tokens t.bin -o run
          idrak cluster sim --topology two.json --dashboard -- pretrain my-model --tokens tokens.bin --steps 500 -o run
        """;

    public override IReadOnlyCollection<string> ValueOptions => ["--topology", "--collective", "--wire", "--mode", "--failures", "--seed", "--port"];

    public override IReadOnlyCollection<string> Flags => ["--dashboard", "--no-browser"];

    /// <summary>The run folder a training command writes (its -o or --out), or null.</summary>
    public static string? RunFolder(IReadOnlyList<string> inner)
    {
        for (int i = 0; i < inner.Count; i++)
        {
            if (inner[i] is "-o" or "--out" && i + 1 < inner.Count)
            {
                return inner[i + 1];
            }

            if (inner[i].StartsWith("--out=", StringComparison.Ordinal))
            {
                return inner[i]["--out=".Length..];
            }
        }

        return null;
    }

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

    /// <summary>The checks every simulated run takes before it starts: the training command after -- and its run folder.</summary>
    public static void CheckTrainer(IReadOnlyList<string> inner)
    {
        if (inner.Count == 0 || !Trainers.Contains(inner[0], StringComparer.Ordinal))
        {
            throw new UsageException($"Give the training command after --: {string.Join(" or ", Trainers)} (got {(inner.Count == 0 ? "nothing" : inner[0])}).");
        }

        if (RunFolder(inner) is null)
        {
            throw new UsageException($"Give the run folder after {inner[0]} (-o RUN): every rank trains into one run, rank 0 writes it.");
        }
    }

    public override int Run(CommandContext context)
    {
        string path = context.Option("--topology") ?? throw new UsageException("Give the topology file with --topology (idrak cluster init writes one).");
        var inner = context.Positional;
        CheckTrainer(inner);
        if (!context.Flag("--dashboard") && (context.Option("--port") is not null || context.Flag("--no-browser")))
        {
            throw new UsageException("--port and --no-browser are the dashboard's: add --dashboard.");
        }

        var options = ClusterOptions.FromEnvironment() with { Topology = path };
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

        if (context.Option("--seed") is { } seed)
        {
            options = options with { Simulation = options.Simulation with { Seed = int.TryParse(seed, NumberStyles.Integer, CultureInfo.InvariantCulture, out int s) ? s : throw new UsageException($"--seed is a whole number, not '{seed}'.") } };
        }

        if (context.Option("--failures") is { } failures)
        {
            options = options with { Simulation = options.Simulation with { Failures = failures } };
        }

        ClusterTopology topology;
        try
        {
            topology = ClusterTopology.Load(path);
            _ = Collectives.Get(options.CollectiveFor(topology));
            _ = FailureModels.Get(options.Simulation.Failures ?? topology.Simulation.Failures);
        }
        catch (Exception e) when (e is InvalidDataException or NotSupportedException or FileNotFoundException)
        {
            context.Error(e.Message);
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
        context.Write($"simulating {topology.WorldSize} rank(s) of {path} in this process ({options.CollectiveFor(topology)}, {(options.Wire == WireType.BFloat16 ? "bf16" : "f32")} wire, "
                      + $"failures {model}, seed {runSeed.ToString(inv)}): idrak {string.Join(' ', inner)}");
        int world = topology.WorldSize;
        ClusterRunResult result;
        int[] codes;
        try
        {
            (result, codes, _) = RunRanks(topology, options, inner, watched?.Output(context.Output) ?? context.Output, context.ErrorOutput, control);
        }
        catch (Exception e) when (watched is not null)
        {
            watched.Finish("failed", e.Message);
            throw;
        }

        watched?.Finish(result, control);

        var faults = result.Log?.Faults ?? [];
        if (faults.Count > 0)
        {
            context.Write($"faults injected ({faults.Count}): " + string.Join("; ", faults.Select(f =>
                $"{f.Kind} at step {f.Step.ToString(inv)} on rank(s) {string.Join(",", f.Ranks)} (replay with --seed {runSeed.ToString(inv)} --step {f.Step.ToString(inv)})")));
        }

        context.Write("network time on the virtual clock: " + string.Join(", ", result.Simulated.Select((t, r) => $"rank {r} {t.TotalSeconds.ToString("F3", inv)} s"))
                      + $"; wall {result.Wall.TotalSeconds.ToString("F1", inv)} s");
        context.WriteJson(new JsonObject
        {
            ["topology"] = Path.GetFullPath(path),
            ["ranks"] = world,
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
        });
        if (result.Succeeded)
        {
            return ExitCodes.Ok;
        }

        foreach (var (error, rank) in result.Errors.Select((e, r) => (e, r)).Where(p => p.e is not null && p.r > 0))
        {
            context.Error($"rank {rank}: {error!.Message}");
        }

        return codes[0] != ExitCodes.Ok ? codes[0] : ExitCodes.Failed;
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
