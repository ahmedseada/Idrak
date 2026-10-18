// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Cluster;
using Idrak.Cli.Shared;
using Idrak.Cluster;

namespace Idrak.Cli.Commands.Cluster;

/// <summary>What the commands that start rank processes share: the program a rank runs, and this machine's node.</summary>
internal static class ClusterLaunch
{
    /// <summary>The program for <paramref name="command"/>: "idrak …" runs this very tool (whatever installs it), anything else as it is.</summary>
    public static LocalhostLaunch Program(IReadOnlyList<string> command, IReadOnlyDictionary<string, string>? environment = null)
    {
        if (command.Count == 0)
        {
            throw new UsageException("Give the command each rank runs after --.");
        }

        if (string.Equals(command[0], "idrak", StringComparison.OrdinalIgnoreCase))
        {
            var (file, prefix) = LocalhostCluster.ProgramOf(typeof(CommandLine).Assembly);
            return new LocalhostLaunch(file, [.. prefix, .. command.Skip(1)]) { Environment = environment };
        }

        return new LocalhostLaunch(command[0], [.. command.Skip(1)]) { Environment = environment };
    }

    /// <summary>This machine's node: --node, else IDRAK_CLUSTER_NODE, else the node named after this host.</summary>
    public static string Node(CommandContext context, ClusterTopology topology, string path, ClusterOptions options)
    {
        string? name = context.Option("--node") ?? options.Node;
        if (name is not null)
        {
            return topology.NodeOf(name) is not null
                ? name
                : throw new UsageException($"--node {name} is not a node of {path} (nodes: {string.Join(", ", topology.Nodes.Select(n => n.Name))}).");
        }

        return topology.NodeOf(Environment.MachineName) is { } node
            ? topology.Nodes[node].Name
            : throw new UsageException($"This host ({Environment.MachineName}) is not a node of {path}: give --node NAME (nodes: {string.Join(", ", topology.Nodes.Select(n => n.Name))}).");
    }

    /// <summary>The topology file, or a usage error naming --topology.</summary>
    public static (string Path, ClusterTopology Topology) Topology(CommandContext context)
    {
        string path = context.Option("--topology") ?? throw new UsageException("Give the topology file with --topology (idrak cluster init writes one).");
        return (path, ClusterTopology.Load(path));
    }

    /// <summary>Each rank's end, as JSON.</summary>
    public static JsonArray Ranks(LocalhostRunResult result) =>
        new([.. result.Ranks.Select(r => (JsonNode)new JsonObject { ["rank"] = r.Rank, ["device"] = r.Device, ["exit_code"] = r.ExitCode })]);

    /// <summary>The exit code of a run: 0 when every rank ended well, else the first failed rank's (or 1).</summary>
    public static int ExitCode(LocalhostRunResult result) =>
        result.Succeeded ? ExitCodes.Ok : result.Ranks.FirstOrDefault(r => r.ExitCode is { } code && code != 0)?.ExitCode ?? ExitCodes.Failed;
}

/// <summary>
/// <c>idrak cluster start --topology FILE [--node NAME] -- COMMAND …</c> (plan 23, phase C5): this machine's part of a real
/// cluster job. It finds this host's node, starts the command once per rank of that node with its rank, world size,
/// node, device and coordinator in its environment, shows one live line per rank, and ends them all when one fails. The
/// same command line on every machine.
/// </summary>
internal sealed class ClusterStartCommand : Command
{
    public override string Name => "cluster start";

    public override string Summary => "Starts this machine's ranks of a cluster: the command once per rank of its node";

    public override string Usage => """
        --topology FILE [--node NAME] -- COMMAND [ARGS]

        Starts COMMAND once per rank of this machine's node in the topology file (the node named after this host, or
        --node), each process given its rank, the world size, its node, its device, the coordinator and the run's
        name in its environment (IDRAK_RANK, IDRAK_WORLD_SIZE, IDRAK_CLUSTER_NODE, IDRAK_CLUSTER_DEVICE,
        IDRAK_COORDINATOR, IDRAK_CLUSTER_RUN); the cluster key travels in IDRAK_CLUSTER_KEY from this environment,
        never on a command line. In a terminal one line per rank shows its state, its pace and its last line (an idrak
        command's own progress and time left); otherwise every line prints as "rank N: ...". When one rank fails the
        others are ended, and none outlives this command. Run the same line on every machine of the cluster; an idrak
        command (pretrain) joins the cluster by itself.

        Options:
              --topology FILE  the topology file (idrak cluster init writes one; its first node's address is the coordinator)
              --node NAME      this machine's node (the node named after this host)

        Examples:
          idrak cluster start --topology lan.json -- idrak pretrain my-model --tokens tokens.bin --steps 1000 -o run
          idrak cluster start --topology lan.json --node mini -- idrak cluster bench --topology lan.json
        """;

    public override IReadOnlyCollection<string> ValueOptions => ["--topology", "--node"];

    public override int Run(CommandContext context)
    {
        var options = ClusterOptions.FromEnvironment();
        ClusterTopology topology;
        string path;
        try
        {
            (path, topology) = ClusterLaunch.Topology(context);
        }
        catch (Exception e) when (e is InvalidDataException or FileNotFoundException or NotSupportedException)
        {
            context.Error(e.Message);
            return ExitCodes.Failed;
        }

        var launch = ClusterLaunch.Program(context.Positional);
        string node = ClusterLaunch.Node(context, topology, path, options);
        int index = topology.NodeOf(node)!.Value;
        var ranks = topology.Ranks(index);
        var devices = ranks.Select(r => options.Simulation.CarveFrom ?? topology.Nodes[index].Devices[topology.Place(r).Local].Device).ToList();
        context.Write($"starting {ranks.Count} rank(s) of node {node} ({string.Join(", ", ranks.Select((r, i) => $"rank {r} on {devices[i]}"))}) of {topology.WorldSize}: "
                      + string.Join(' ', context.Positional));
        using var board = new RankBoard(context, $"{context.Positional[0]} on node {node}", ranks, devices, line => line.Rank == ranks[0] && !line.IsError);
        LocalhostRunResult result;
        try
        {
            result = LocalhostCluster.Run(path, options, launch, node, board.Line, context.TimeoutToken);
        }
        catch (Exception e) when (e is NotSupportedException or InvalidOperationException or ArgumentException)
        {
            context.Error(e.Message);
            return ExitCodes.Failed;
        }

        board.Finish(result.Ranks);
        foreach (var rank in result.Ranks)
        {
            string end = rank.ExitCode switch { 0 => "done", null => "ended by the launcher", { } code => $"exit code {code}" };
            context.Write($"rank {rank.Rank} ({rank.Device}): {end}" + (rank.ExitCode is not (0 or null) && rank.Errors.LastOrDefault(l => l.Length > 0) is { } last ? $": {last}" : ""));
        }

        if (result.Failure is { } failure)
        {
            context.Error(failure);
        }

        context.WriteJson(new JsonObject
        {
            ["topology"] = Path.GetFullPath(path),
            ["node"] = node,
            ["command"] = string.Join(' ', context.Positional),
            ["wall_seconds"] = result.Wall.TotalSeconds,
            ["failure"] = result.Failure,
            ["ranks"] = ClusterLaunch.Ranks(result),
        });
        return ClusterLaunch.ExitCode(result);
    }
}
