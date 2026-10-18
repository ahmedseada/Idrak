// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Abstraction.Cluster;
using Idrak.Cli.Shared;
using Idrak.Cluster;

namespace Idrak.Cli.Commands.Cluster;

/// <summary>
/// <c>--cluster FILE</c> on the training commands (plan 23, phase C8): a process started by hand joins a real cluster as
/// one rank. The rank and the world size come from the topology file (decision 3): the world size is the file's device
/// count, this machine's node is found by its host name (or <c>--node</c>), and a node of one worker gives its rank. A node
/// of several workers needs the rank: <c>--rank N</c> where the command has no <c>--rank</c> of its own (train, pretrain,
/// resume), else <c>IDRAK_RANK</c> (tune and distill keep <c>--rank</c> as the adapter's rank; nothing is renamed).
/// Without a file, <c>--coordinator HOST:PORT</c>, <c>--world-size N</c> and the rank join a cluster of one device per
/// rank. A world size that contradicts the file is refused naming both; <c>--world-size</c> without a coordinator names
/// it. The key comes only from <c>IDRAK_CLUSTER_KEY</c>.
/// </summary>
internal static class ClusterJoin
{
    /// <summary>The options every training command takes for it (with <c>--rank</c> where the command owns none).</summary>
    public static readonly string[] Options = ["--cluster", "--world-size", "--coordinator", "--node"];

    /// <summary>The placement options pretrain takes beside them.</summary>
    public static readonly string[] PipelineOptions = ["--placement", "--stages"];

    /// <summary>
    /// Joins this process to the cluster its options (or, as phase C5 had it, the environment a launcher set) name, making it
    /// <see cref="ClusterSession.Current"/> until disposed; null when it joins none (or already is a rank).
    /// </summary>
    /// <param name="context">The command's context.</param>
    /// <param name="ownsRank">Whether the command's own <c>--rank</c> means something else (tune and distill: the adapter's).</param>
    /// <exception cref="UsageException">A rank, world size, node or coordinator that is missing or contradicts the file; the message names both.</exception>
    public static IDisposable? Join(CommandContext context, bool ownsRank)
    {
        if (ClusterSession.Current is not null)
        {
            return null;                                                // a rank of idrak cluster sim
        }

        string? file = context.Option("--cluster"), coordinator = context.Option("--coordinator"), worldText = context.Option("--world-size");
        string? rankText = ownsRank ? null : context.Option("--rank");
        string? node = context.Option("--node"), placement = context.Option("--placement"), stagesText = context.Option("--stages");
        if (file is null && coordinator is null && worldText is null && rankText is null)
        {
            if (node is not null || placement is not null || stagesText is not null)
            {
                throw new UsageException($"{(node is not null ? "--node" : placement is not null ? "--placement" : "--stages")} is for a cluster run: give --cluster FILE (or --coordinator).");
            }

            return ClusterSession.JoinStarted();                          // a rank process of idrak cluster start (phase C5)
        }

        ClusterOptions options;
        try
        {
            options = ClusterOptions.FromEnvironment();
        }
        catch (ArgumentException e)
        {
            throw new UsageException(e.Message);
        }

        int? rank = rankText is null ? options.Rank : Whole("--rank", rankText, 0);
        if (rankText is not null && options.Rank is { } fromEnvironment && fromEnvironment != rank)
        {
            throw new UsageException($"--rank {rank} contradicts IDRAK_RANK={fromEnvironment}; give one.");
        }

        int? world = worldText is null ? null : Whole("--world-size", worldText, 1);
        if (placement is not null)
        {
            options = options with
            {
                Placement = placement.ToLowerInvariant() is "single" or "pipeline" ? placement.ToLowerInvariant() : throw new UsageException($"--placement is single or pipeline, not '{placement}'."),
            };
        }

        if (stagesText is not null)
        {
            options = options with { Stages = Whole("--stages", stagesText, 1) };
        }

        options = options with { Rank = rank, Coordinator = coordinator ?? options.Coordinator, Node = node ?? options.Node };
        string rankHint = ownsRank ? "IDRAK_RANK (--rank here is the adapter's rank)" : "--rank N (or IDRAK_RANK)";
        LocalhostRank joined;
        try
        {
            if (file is not null)
            {
                var topology = ClusterTopology.Load(file);
                if ((world ?? options.WorldSize) is { } claimed && claimed != topology.WorldSize)
                {
                    string named = world is not null ? $"--world-size {claimed}" : $"IDRAK_WORLD_SIZE={claimed}";
                    throw new UsageException($"{named} contradicts {file}, which has {topology.WorldSize} rank(s): the file is the world size; leave it out.");
                }

                int? here = options.Node is { } nodeName
                    ? topology.NodeOf(nodeName) ?? throw new UsageException($"--node {nodeName} is not a node of {file} ({string.Join(", ", topology.Nodes.Select(n => n.Name))}).")
                    : topology.NodeOf(Environment.MachineName);
                if (rank is null)
                {
                    if (here is not { } n)
                    {
                        throw new UsageException($"This host ({Environment.MachineName}) is not a node of {file}: give --node NAME (IDRAK_CLUSTER_NODE), or the rank with {rankHint}.");
                    }

                    var ranks = topology.Ranks(n);
                    if (ranks.Count > 1)
                    {
                        throw new UsageException($"Node {topology.Nodes[n].Name} of {file} has {ranks.Count} workers (ranks {string.Join(", ", ranks)}): give this process's rank with {rankHint}, "
                                                 + $"or start them all with idrak cluster start --topology {file}.");
                    }
                }
                else if ((uint)rank.Value >= (uint)topology.WorldSize)
                {
                    throw new UsageException($"Rank {rank} is not a rank of {file} (0 to {topology.WorldSize - 1}).");
                }

                joined = LocalhostCluster.Join(options with { Topology = Path.GetFullPath(file), WorldSize = null });
            }
            else
            {
                if (coordinator is null && options.Coordinator is null)
                {
                    throw new UsageException(world is not null
                        ? "--world-size needs --coordinator HOST:PORT (where rank 0 listens; IDRAK_COORDINATOR), or a topology file with --cluster FILE."
                        : "A cluster without a topology file needs --coordinator HOST:PORT (where rank 0 listens), --world-size N and the rank.");
                }

                int size = world ?? options.WorldSize ?? throw new UsageException("A cluster without a topology file needs --world-size N (IDRAK_WORLD_SIZE), or give --cluster FILE.");
                int me = rank ?? throw new UsageException($"A cluster without a topology file needs this process's rank: {rankHint}.");
                if (me >= size)
                {
                    throw new UsageException($"Rank {me} is not a rank of a world of {size} (0 to {size - 1}).");
                }

                // Every rank a node of its own on this process's device; the first node's address is the coordinator.
                string device = context.Device.ToString();
                var nodes = Enumerable.Range(0, size).Select(r => new ClusterNodeSpec(string.Create(CultureInfo.InvariantCulture, $"rank{r}"),
                    r == 0 ? options.Coordinator : null, [new ClusterDeviceSpec(device, null)])).ToList();
                var topology = ClusterTopology.Create(nodes, null, size > 1 ? "ethernet" : null);
                joined = LocalhostCluster.Join(topology, options with { WorldSize = size, Node = nodes[me].Name }, "--coordinator " + options.Coordinator);
            }
        }
        catch (Exception e) when (e is InvalidDataException or FileNotFoundException or DirectoryNotFoundException or NotSupportedException)
        {
            throw new UsageException(e.Message);
        }
        catch (InvalidOperationException e) when (e.Message.Contains("contradicts", StringComparison.Ordinal) || e.Message.Contains("not a node", StringComparison.Ordinal)
                                                   || e.Message.Contains("not a rank", StringComparison.Ordinal))
        {
            throw new UsageException(e.Message);
        }

        ClusterSession.Current = joined.Rank;
        return new Leave(joined);
    }

    private static int Whole(string option, string text, int min) =>
        int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) && value >= min
            ? value
            : throw new UsageException($"{option} needs a whole number, {min} or more, not '{text}'.");

    private sealed class Leave(LocalhostRank joined) : IDisposable
    {
        public void Dispose()
        {
            ClusterSession.Current = null;
            joined.Dispose();
        }
    }
}
