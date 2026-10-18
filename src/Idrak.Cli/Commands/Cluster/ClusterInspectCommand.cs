// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Cluster;
using Idrak.Cli.Shared;
using Idrak.Cluster;

namespace Idrak.Cli.Commands.Cluster;

/// <summary>
/// <c>idrak cluster inspect FILE</c> (plan 23): reads a topology file and shows which node this machine is, every rank with
/// its device and cap, each kind of link with its figures as this machine resolves them (measured links measured here),
/// and the network time an all-reduce would take on it with each algorithm.
/// </summary>
internal sealed class ClusterInspectCommand : Command
{
    public override string Name => "cluster inspect";

    public override string Summary => "Shows a topology file: its ranks, its links and the time an all-reduce would take on it";

    public override string Usage => """
        FILE [--bytes SIZE] [--unit UNIT]

        Reads the topology file (idrak cluster init writes one) and checks it: the node this machine is (by its host
        name), every rank with its node, device and memory cap, each kind of link the ranks use with its bandwidth and
        latency (links measured on this machine are measured now), and the network time an all-reduce of SIZE would
        take on the topology with each algorithm, played on the simulator's virtual clock.

        Arguments:
          FILE              the topology file

        Options:
              --bytes SIZE  the all-reduce whose time is predicted (100MB; B, KB, MB, GB, TB, powers of 1,024)
              --unit UNIT   the unit sizes and bandwidths are shown in: B, KB, MB, GB or TB (powers of 1,024; KiB and
                            the rest the same); default: per value, the largest unit it fills

        Examples:
          idrak cluster inspect topology.json
          idrak cluster inspect topology.json --bytes 1GB --unit GB
          idrak cluster inspect topology.json --bytes 1GB -j
        """;

    public override IReadOnlyCollection<string> ValueOptions => ["--bytes", "--unit"];

    private static readonly string[] Units = ["B", "KB", "MB", "GB", "TB"];

    public override int Run(CommandContext context)
    {
        string path = context.Argument(0, "the topology file");
        long bytes;
        try
        {
            bytes = ClusterTopology.ParseMemory(context.Option("--bytes") ?? "100MB");
        }
        catch (FormatException e)
        {
            throw new UsageException($"--bytes: {e.Message}");
        }

        int? unit = null;
        if (context.Option("--unit") is { } named)
        {
            string plain = named.Replace("i", "", StringComparison.OrdinalIgnoreCase);    // KiB is KB: both powers of 1,024
            int found = Array.FindIndex(Units, u => u.Equals(plain, StringComparison.OrdinalIgnoreCase));
            unit = found >= 0 ? found : throw new UsageException($"--unit: '{named}' is not a unit; use {string.Join(", ", Units)} (powers of 1,024).");
        }

        ClusterTopology topology;
        try
        {
            topology = ClusterTopology.Load(path);
        }
        catch (Exception e) when (e is InvalidDataException or NotSupportedException or FileNotFoundException)
        {
            context.Error(e.Message);
            return ExitCodes.Failed;
        }

        var inv = CultureInfo.InvariantCulture;
        int? here = topology.NodeOf(Environment.MachineName);
        context.Write($"{path}: {topology.Nodes.Count} node(s), {topology.WorldSize} rank(s); " + (here is { } n
            ? $"this machine is node {topology.Nodes[n].Name}, rank(s) {string.Join(", ", topology.Ranks(n))}"
            : $"this machine ({Environment.MachineName}) is not one of its nodes"));
        context.Write("");
        var ranks = Enumerable.Range(0, topology.WorldSize).Select(r =>
        {
            var (node, local) = topology.Place(r);
            var device = topology.Nodes[node].Devices[local];
            string cap = device.MemoryBytes is { } m ? Size(m, unit) : "the device's own";
            return (IReadOnlyList<string>)[r.ToString(inv), topology.Nodes[node].Name, device.Device, cap];
        }).ToList();
        context.Table(["rank", "node", "device", "cap"], ranks);
        context.Write("");

        IReadOnlyList<TopologyLink> links;
        try
        {
            links = TopologyInspection.Links(topology);
        }
        catch (Exception e) when (e is NotSupportedException or InvalidOperationException or ResourceLimitExceededException)
        {
            context.Error($"A link could not be resolved here: {e.Message}");
            return ExitCodes.Failed;
        }

        context.Table(["between", "link", "bandwidth", "latency", "figures from"], [.. links.Select(l => (IReadOnlyList<string>)
            [l.Between, l.Link, l.BandwidthBytesPerSecond > 0 ? Size(l.BandwidthBytesPerSecond, unit) + "/s" : "unknown",
             $"{l.Latency.TotalMicroseconds.ToString("F1", inv)} us", l.Source])]);
        context.Write("");

        var predictions = new JsonObject();
        foreach (string algorithm in new[] { "ring", "naive" })
        {
            try
            {
                var time = TopologyInspection.PredictAllReduce(topology, algorithm, bytes);
                predictions[algorithm] = time.TotalSeconds;
                context.Write($"all-reduce of {Size(bytes, unit)} with {algorithm}: {time.TotalMilliseconds.ToString("F2", inv)} ms of network time");
            }
            catch (InvalidOperationException e)
            {
                predictions[algorithm] = null;
                context.Write($"all-reduce with {algorithm}: not predicted ({e.Message})");
            }
        }

        context.Write("simulated: every rank in this process over the sim transport; real transports join in a later phase.");
        context.WriteJson(new JsonObject
        {
            ["path"] = Path.GetFullPath(path),
            ["nodes"] = topology.Nodes.Count,
            ["ranks"] = topology.WorldSize,
            ["node_here"] = here is { } h ? topology.Nodes[h].Name : null,
            ["links"] = new JsonArray([.. links.Select(l => (JsonNode)new JsonObject
            {
                ["between"] = l.Between, ["link"] = l.Link, ["bandwidth"] = l.BandwidthBytesPerSecond, ["latency_us"] = l.Latency.TotalMicroseconds, ["source"] = l.Source,
            })]),
            ["all_reduce_bytes"] = bytes,
            ["all_reduce_seconds"] = predictions,
        });
        return ExitCodes.Ok;
    }

    // A size in `unit` (an index into Units), or in the largest unit it fills; powers of 1,024 (decision 7).
    private static string Size(double bytes, int? unit)
    {
        int power = unit ?? 0;
        if (unit is null)
        {
            while (power < Units.Length - 1 && bytes >= Math.Pow(1024, power + 1))
            {
                power++;
            }
        }

        double value = bytes / Math.Pow(1024, power);
        string format = power == 0 || value == Math.Floor(value) ? "N0" : value >= 100 ? "N1" : value >= 1 ? "N2" : "G3";
        return value.ToString(format, CultureInfo.InvariantCulture) + " " + Units[power];
    }
}
