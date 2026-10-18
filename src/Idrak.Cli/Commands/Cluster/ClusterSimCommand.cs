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
        --topology FILE [--collective NAME] [--wire f32|bf16] -- pretrain MODEL [options] -o RUN

        Runs the training command after -- once per rank of the topology file: every rank a thread of this process on
        a device carved from the physical device its entry names (the entry's memory as a hard cap), the ranks joined
        by the simulated transport. Each step's rows are shared across the ranks and the gradients summed after the
        backward pass, so the run trains as one process would; rank 0 writes the run folder and its output is shown.
        At the end each rank's network time on the virtual clock (what the cluster the file describes would spend on
        its links) is printed beside the wall time. The other cluster settings come from the IDRAK_CLUSTER_ variables
        (idrak env lists them).

        Options:
              --topology FILE    the topology file (idrak cluster init writes one)
              --collective NAME  the collective algorithm: ring or naive (ring)
              --wire W           how gradients travel: f32 or bf16 (f32)
              --seed N           the seed of the simulation's jitter and faults (the file's)

        Examples:
          idrak cluster init -y --nodes 1 --devices cpu,cpu -o two.json
          idrak cluster sim --topology two.json -- pretrain my-model --tokens tokens.bin --steps 100 -o run
        """;

    public override IReadOnlyCollection<string> ValueOptions => ["--topology", "--collective", "--wire"];

    public override int Run(CommandContext context)
    {
        string path = context.Option("--topology") ?? throw new UsageException("Give the topology file with --topology (idrak cluster init writes one).");
        var inner = context.Positional;
        if (inner.Count == 0 || !Trainers.Contains(inner[0], StringComparer.Ordinal))
        {
            throw new UsageException($"Give the training command after --: {string.Join(" or ", Trainers)} (got {(inner.Count == 0 ? "nothing" : inner[0])}).");
        }

        if (!inner.Any(a => a is "-o" or "--out" || a.StartsWith("--out=", StringComparison.Ordinal)))
        {
            throw new UsageException($"Give the run folder after {inner[0]} (-o RUN): every rank trains into one run, rank 0 writes it.");
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

        ClusterTopology topology;
        try
        {
            topology = ClusterTopology.Load(path);
            _ = Collectives.Get(options.Collective ?? "ring");
        }
        catch (Exception e) when (e is InvalidDataException or NotSupportedException or FileNotFoundException)
        {
            context.Error(e.Message);
            return ExitCodes.Failed;
        }

        var inv = CultureInfo.InvariantCulture;
        context.Write($"simulating {topology.WorldSize} rank(s) of {path} in this process ({options.Collective ?? "ring"}, {(options.Wire == WireType.BFloat16 ? "bf16" : "f32")} wire): "
                      + $"idrak {string.Join(' ', inner)}");
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
                codes[rank.Rank] = CommandLine.Run(inner, rank.Rank == 0 ? context.Output : TextWriter.Null, rank.Rank == 0 ? context.ErrorOutput : errors[rank.Rank]);
            }
            finally
            {
                ClusterSession.Current = null;
            }

            if (codes[rank.Rank] != ExitCodes.Ok)
            {
                throw new InvalidOperationException($"{inner[0]} ended with exit code {codes[rank.Rank]}" + (errors[rank.Rank].ToString().Trim() is { Length: > 0 } text ? $": {text.Split('\n')[0]}" : ""));
            }
        });

        context.Write("network time on the virtual clock: " + string.Join(", ", result.Simulated.Select((t, r) => $"rank {r} {t.TotalSeconds.ToString("F3", inv)} s"))
                      + $"; wall {result.Wall.TotalSeconds.ToString("F1", inv)} s");
        context.WriteJson(new JsonObject
        {
            ["topology"] = Path.GetFullPath(path),
            ["ranks"] = world,
            ["command"] = string.Join(' ', inner),
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
}
