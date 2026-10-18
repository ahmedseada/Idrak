// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Cluster;
using Idrak.Cli.Shared;
using Idrak.Cluster;
using Idrak.Networking.Bench;

namespace Idrak.Cli.Commands.Cluster;

/// <summary>
/// <c>idrak cluster bench --topology FILE [--bytes N] [--rounds N]</c> (plan 23, phase C5): the bandwidth and latency of every
/// pair of ranks over the transports the cluster would use, the first round apart from the warm ones. On each machine it
/// starts that node's ranks (as <c>cluster start</c> does), or with <c>--localhost</c> every rank of the file here, through
/// the throttle; each rank then runs <see cref="LinkBench"/>, and rank 0's machine prints the table.
/// </summary>
internal sealed class ClusterBenchCommand : Command
{
    /// <summary>The line rank 0 writes with the rows, for the command that started it.</summary>
    internal const string RowsMarker = "idrak-cluster-bench ";

    public override string Name => "cluster bench";

    public override string Summary => "Measures the bandwidth and latency of every pair of ranks of a cluster, cold and warm apart";

    public override string Usage => """
        --topology FILE [--bytes N] [--rounds N] [--node NAME | --localhost]

        Measures every pair of ranks of the topology file over the transport the cluster would use for it, one pair at a
        time: half the round trip of an empty message (latency) and a message of --bytes with its answer (bandwidth).
        The first round (cold: connections, compiling) is shown apart from the median of the warm ones. Run it on every
        machine of the cluster, as idrak cluster start runs a command: each starts its node's ranks, and the machine of
        rank 0 prints the table. With --localhost every rank of the file runs on this machine, the links paced to their
        profiles (IDRAK_CLUSTER_THROTTLE=0 measures this machine's own pipes). -j prints the rows as JSON, the input of a
        calibration.

        Options:
              --topology FILE  the topology file (idrak cluster init writes one)
              --bytes N        the message the bandwidth is measured with: B, KB, MB, GB (16MB)
              --rounds N       the warm rounds after the cold one (5)
              --node NAME      this machine's node (the node named after this host)
              --localhost      every rank of the file on this machine, paced to the links

        Examples:
          idrak cluster bench --topology lan.json
          idrak cluster bench --topology two.json --localhost --bytes 4MB --rounds 3 -j
        """;

    public override IReadOnlyCollection<string> ValueOptions => ["--topology", "--bytes", "--rounds", "--node"];

    public override IReadOnlyCollection<string> Flags => ["--localhost"];

    public override int Run(CommandContext context)
    {
        long bytes = context.Option("--bytes") is { } text ? Size(text) : new LinkBenchOptions().Bytes;
        int rounds = context.IntOption("--rounds", new LinkBenchOptions().Rounds);
        if (rounds < 1)
        {
            throw new UsageException($"--rounds is 1 or more, not {rounds}.");
        }

        var bench = new LinkBenchOptions { Bytes = bytes, Rounds = rounds };
        return Environment.GetEnvironmentVariable("IDRAK_RANK") is { Length: > 0 } ? Worker(context, bench) : Launch(context, bench);
    }

    // A rank started by the launcher: joins, measures, and (rank 0) hands the rows back on one line.
    private static int Worker(CommandContext context, LinkBenchOptions bench)
    {
        using var joined = LocalhostCluster.Join();
        int me = joined.Rank.Rank;
        var inv = CultureInfo.InvariantCulture;
        var progress = new Progress(p => context.Write(string.Create(inv,
            $"pair {p.Done} of {p.Total} (ranks {p.RankA} and {p.RankB}), {100.0 * p.Done / p.Total:F0}%, {p.Elapsed.TotalSeconds:F1} s, {p.Elapsed.TotalSeconds / p.Done * (p.Total - p.Done):F1} s left")));
        var rows = LinkBench.Run(joined.Rank.Transport, bench, (a, b) => joined.Plan[a, b].Transport, progress);
        if (me == 0)
        {
            context.Write(RowsMarker + Json(rows).ToJsonString());
        }

        return ExitCodes.Ok;
    }

    // This machine's ranks (or every rank, --localhost), each running this command as a worker.
    private static int Launch(CommandContext context, LinkBenchOptions bench)
    {
        var options = ClusterOptions.FromEnvironment();
        string path;
        ClusterTopology topology;
        try
        {
            (path, topology) = ClusterLaunch.Topology(context);
        }
        catch (Exception e) when (e is InvalidDataException or FileNotFoundException or NotSupportedException)
        {
            context.Error(e.Message);
            return ExitCodes.Failed;
        }

        bool localhost = context.Flag("--localhost");
        string? node = localhost ? null : ClusterLaunch.Node(context, topology, path, options);
        int[] ranks = node is null ? [.. Enumerable.Range(0, topology.WorldSize)] : [.. topology.Ranks(topology.NodeOf(node)!.Value)];
        var devices = ranks.Select(r => options.Simulation.CarveFrom ?? topology.Nodes[topology.Place(r).Node].Devices[topology.Place(r).Local].Device).ToList();
        var inv = CultureInfo.InvariantCulture;
        var launch = ClusterLaunch.Program(["idrak", "cluster", "bench", "--topology", Path.GetFullPath(path), "--bytes", bench.Bytes.ToString(inv), "--rounds", bench.Rounds.ToString(inv)]);
        context.Write($"benchmarking {topology.WorldSize * (topology.WorldSize - 1) / 2} pair(s) of {path}, {ClusterTopology.FormatMemory(bench.Bytes)} messages, "
                      + $"{bench.Rounds} warm round(s): " + (node is null ? $"every rank on this machine{(options.Simulation.Throttle ? ", paced to the links" : "")}" : $"ranks {string.Join(", ", ranks)} of node {node}"));
        JsonArray? rows = null;
        using var board = new RankBoard(context, "cluster bench", ranks, devices, _ => false);
        LocalhostRunResult result;
        try
        {
            result = LocalhostCluster.Run(path, options, launch, node, line =>
            {
                if (!line.IsError && line.Text.StartsWith(RowsMarker, StringComparison.Ordinal))
                {
                    rows = JsonNode.Parse(line.Text[RowsMarker.Length..]) as JsonArray;
                    return;
                }

                board.Line(line);
            }, context.TimeoutToken);
        }
        catch (Exception e) when (e is NotSupportedException or InvalidOperationException or ArgumentException)
        {
            context.Error(e.Message);
            return ExitCodes.Failed;
        }

        board.Finish(result.Ranks);
        if (!result.Succeeded)
        {
            context.Error(result.Failure ?? "a rank failed");
            foreach (var rank in result.Ranks.Where(r => r.ExitCode != 0))
            {
                context.Error($"rank {rank.Rank}: " + (rank.Errors.LastOrDefault(l => l.Length > 0) ?? "ended"));
            }

            return ClusterLaunch.ExitCode(result);
        }

        if (rows is null)
        {
            context.Write(ranks.Contains(0) ? "rank 0 gave no rows" : "the table prints on the machine of rank 0");
            context.WriteJson(new JsonObject { ["topology"] = Path.GetFullPath(path), ["rows"] = new JsonArray() });
            return ranks.Contains(0) ? ExitCodes.Failed : ExitCodes.Ok;
        }

        foreach (var row in rows.OfType<JsonObject>())
        {
            context.Write(string.Create(inv, $"ranks {(int)row["rank_a"]!}-{(int)row["rank_b"]!} {(string)row["transport"]!,-14} ")
                          + $"bandwidth {Rate((double)row["warm_bytes_per_second"]!)} warm, {Rate((double)row["cold_bytes_per_second"]!)} cold; "
                          + $"latency {Micro((double)row["warm_latency_seconds"]!)} warm, {Micro((double)row["cold_latency_seconds"]!)} cold");
        }

        context.WriteJson(new JsonObject
        {
            ["topology"] = Path.GetFullPath(path),
            ["bytes"] = bench.Bytes,
            ["rounds"] = bench.Rounds,
            ["throttled"] = node is null && options.Simulation.Throttle,
            ["rows"] = rows.DeepClone(),
        });
        return ExitCodes.Ok;
    }

    internal static JsonArray Json(IEnumerable<LinkBenchRow> rows) => new([.. rows.Select(r => (JsonNode)new JsonObject
    {
        ["rank_a"] = r.RankA, ["rank_b"] = r.RankB, ["transport"] = r.Transport, ["bytes"] = r.Bytes, ["rounds"] = r.Rounds,
        ["cold_bytes_per_second"] = r.ColdBandwidth, ["warm_bytes_per_second"] = r.WarmBandwidth,
        ["cold_latency_seconds"] = r.ColdLatency.TotalSeconds, ["warm_latency_seconds"] = r.WarmLatency.TotalSeconds,
    })]);

    private static long Size(string text)
    {
        try
        {
            return ClusterTopology.ParseMemory(text) is > 0 and var bytes ? bytes : throw new UsageException($"--bytes is above 0, not {text}.");
        }
        catch (FormatException)
        {
            throw new UsageException($"--bytes is a size (B, KB, MB, GB), not '{text}'.");
        }
    }

    private static string Rate(double bytesPerSecond) => string.Create(CultureInfo.InvariantCulture,
        $"{bytesPerSecond / (1 << 20):F1} MiB/s ({bytesPerSecond * 8 / 1e9:F2} Gb/s)");

    private static string Micro(double seconds) => string.Create(CultureInfo.InvariantCulture, $"{seconds * 1e6:F0} us");

    private sealed class Progress(Action<LinkBenchProgress> report) : IProgress<LinkBenchProgress>
    {
        public void Report(LinkBenchProgress value) => report(value);
    }
}
