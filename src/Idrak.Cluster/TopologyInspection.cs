// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Cluster;

/// <summary>One kind of link a topology uses, as this machine resolves it.</summary>
/// <param name="Between">Which ranks it joins ("desk ranks on cuda:0", "desk and mini").</param>
/// <param name="FirstRank">One rank it joins.</param>
/// <param name="SecondRank">Another.</param>
/// <param name="Link">The profile's name.</param>
/// <param name="BandwidthBytesPerSecond">Its bandwidth (0: not known; give a rate or measure it).</param>
/// <param name="Latency">Its latency.</param>
/// <param name="Source">Where the figures come from ("measured on cuda:0", "the name's rate", "none").</param>
public sealed record TopologyLink(string Between, int FirstRank, int SecondRank, string Link, double BandwidthBytesPerSecond, TimeSpan Latency, string Source);

/// <summary>
/// What <c>idrak cluster inspect</c> shows of a topology (plan 23): each kind of link it uses with its figures as this
/// machine resolves them (measured links measured here), and the network time an all-reduce would take on it, played on
/// the <see cref="VirtualClock"/> without moving any data.
/// </summary>
public static class TopologyInspection
{
    /// <summary>One entry per pair of nodes (and per kind of pair inside a node) the topology joins.</summary>
    public static IReadOnlyList<TopologyLink> Links(IClusterTopology topology)
    {
        ArgumentNullException.ThrowIfNull(topology);
        var seen = new HashSet<(int, int, int)>();
        var links = new List<TopologyLink>();
        for (int a = 0; a < topology.WorldSize; a++)
        {
            for (int b = a + 1; b < topology.WorldSize; b++)
            {
                var (na, la) = topology.Place(a);
                var (nb, lb) = topology.Place(b);
                string deviceA = topology.Nodes[na].Devices[la].Device, deviceB = topology.Nodes[nb].Devices[lb].Device;
                int kind = na != nb ? 0 : string.Equals(deviceA, deviceB, StringComparison.OrdinalIgnoreCase) ? 1 : 2;
                if (!seen.Add((na, nb, kind)))
                {
                    continue;
                }

                var declared = topology.Link(a, b);
                var link = Resolve(declared, deviceA);
                string between = kind switch
                {
                    0 => $"{topology.Nodes[na].Name} and {topology.Nodes[nb].Name}",
                    1 => $"{topology.Nodes[na].Name} ranks on {deviceA}",
                    _ => $"{topology.Nodes[na].Name} devices",
                };
                string source = declared is MeasuredLink ? $"measured on {deviceA}" : link.BandwidthBytesPerSecond > 0 ? "the name's rate or a registered profile" : "none: give a rate";
                links.Add(new TopologyLink(between, a, b, link.Name, link.BandwidthBytesPerSecond, link.Latency, source));
            }
        }

        return links;
    }

    /// <summary>
    /// The network time an all-reduce of <paramref name="bytes"/> takes on <paramref name="topology"/> with
    /// <paramref name="algorithm"/> ("ring" or "naive"), played on a fresh virtual clock: the slowest rank's time.
    /// </summary>
    /// <exception cref="NotSupportedException">An algorithm this method cannot play.</exception>
    /// <exception cref="InvalidOperationException">A link the play crosses has no known bandwidth.</exception>
    public static TimeSpan PredictAllReduce(IClusterTopology topology, string algorithm, long bytes, int? seed = null)
    {
        ArgumentNullException.ThrowIfNull(topology);
        ArgumentOutOfRangeException.ThrowIfNegative(bytes);
        int world = topology.WorldSize;
        var devices = Enumerable.Range(0, world).Select(r =>
        {
            var (node, local) = topology.Place(r);
            return topology.Nodes[node].Devices[local].Device;
        }).ToArray();
        var clock = new VirtualClock(world, (a, b) => Resolve(topology.Link(a, b), devices[a]), seed ?? topology.Simulation.Seed);
        if (world == 1)
        {
            return TimeSpan.Zero;
        }

        switch (algorithm.ToLowerInvariant())
        {
            case "ring":
                long slice = (bytes + world - 1) / world;
                for (int step = 0; step < 2 * (world - 1); step++)
                {
                    var arrivals = new double[world];
                    for (int r = 0; r < world; r++)
                    {
                        arrivals[(r + 1) % world] = clock.Send(r, (r + 1) % world, slice, 0, step).Arrives;
                    }

                    for (int r = 0; r < world; r++)
                    {
                        clock.Arrive(r, arrivals[r]);
                    }
                }

                break;
            case "naive":
                for (int r = 1; r < world; r++)
                {
                    clock.Arrive(0, clock.Send(r, 0, bytes, 0, 0).Arrives);
                }

                for (int r = 1; r < world; r++)
                {
                    clock.Arrive(r, clock.Send(0, r, bytes, 1, 0).Arrives);
                }

                break;
            default:
                throw new NotSupportedException($"Cannot play '{algorithm}' on the clock (ring, naive).");
        }

        return TimeSpan.FromSeconds(Enumerable.Range(0, world).Max(clock.Now));
    }

    private static ILinkProfile Resolve(ILinkProfile link, string device) => MeasuredLink.Resolve(link, device);
}
