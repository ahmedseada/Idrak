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

    /// <summary>The collective algorithms <see cref="PredictAllReduce"/> plays.</summary>
    public static IReadOnlyList<string> Playable { get; } = ["ring", "naive", "hierarchical"];

    /// <summary>
    /// The network time an all-reduce of <paramref name="bytes"/> (float32 values) takes on <paramref name="topology"/> with
    /// <paramref name="algorithm"/> ("ring", "naive" or "hierarchical"), played on a fresh virtual clock without moving any
    /// data: the slowest rank's time. The play sends what the algorithm sends, message for message (the same pieces, tags,
    /// message numbers and chunks of <paramref name="chunkBytes"/>, null: one message a piece, as
    /// <see cref="ClusterOptions.ChunkBytes"/> unset), so it equals an in-process run's simulated time with no faults.
    /// </summary>
    /// <exception cref="NotSupportedException">An algorithm this method cannot play.</exception>
    /// <exception cref="InvalidOperationException">A link the play crosses has no known bandwidth.</exception>
    public static TimeSpan PredictAllReduce(IClusterTopology topology, string algorithm, long bytes, int? seed = null, long? chunkBytes = null)
    {
        ArgumentNullException.ThrowIfNull(topology);
        ArgumentNullException.ThrowIfNull(algorithm);
        ArgumentOutOfRangeException.ThrowIfNegative(bytes);
        long values = (bytes + sizeof(float) - 1) / sizeof(float);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(values, (long)int.MaxValue, nameof(bytes));
        int world = topology.WorldSize;
        var devices = Enumerable.Range(0, world).Select(r =>
        {
            var (node, local) = topology.Place(r);
            return topology.Nodes[node].Devices[local].Device;
        }).ToArray();
        var play = new Play(new VirtualClock(world, (a, b) => Resolve(topology.Link(a, b), devices[a]), seed ?? topology.Simulation.Seed), chunkBytes);
        if (world > 1)
        {
            switch (algorithm.ToLowerInvariant())
            {
                case "ring":
                    play.Ring([.. Enumerable.Range(0, world)], (int)values);
                    break;
                case "naive":
                    play.Naive(world, (int)values);
                    break;
                case "hierarchical":
                    play.Hierarchical(topology, (int)values);
                    break;
                default:
                    throw new NotSupportedException($"Cannot play '{algorithm}' on the clock ({string.Join(", ", Playable)}).");
            }
        }

        return TimeSpan.FromSeconds(Enumerable.Range(0, world).Max(play.Clock.Now));
    }

    private static ILinkProfile Resolve(ILinkProfile link, string device) => MeasuredLink.Resolve(link, device);

    // The algorithms of Idrak.Network's Collectives/Algorithms.cs and HierarchicalCollectives.cs, played on the clock: the
    // same pieces (ring's cut), the same tags (mirrored here; the two packages do not reference each other), each send
    // charged at the sender's time after what it received before it, as HostCollectives.Exchange interleaves chunks.
    private sealed class Play(VirtualClock clock, long? chunkBytes)
    {
        private const int NaiveToRoot = -1, NaiveFromRoot = -2, RingScatter = -11, RingGather = -12;
        private const int IntraScatter = -21, IntraGather = -22, CrossScatter = -23, CrossGather = -24;

        // HostCollectives' chunk: ChunkBytes, or what one array holds.
        private readonly long _chunk = Math.Clamp(chunkBytes ?? Array.MaxLength / 2, sizeof(float), Array.MaxLength / 2);
        private readonly Dictionary<(int, int, int), long> _sequence = [];

        public VirtualClock Clock => clock;

        public void Ring(int[] members, int values)
        {
            Scatter(members, s => Size(values, members.Length, s), RingScatter);
            Gather(members, s => Size(values, members.Length, s), RingGather);
        }

        public void Naive(int world, int values)
        {
            long per = PerChunk();
            var arrivals = new List<(int Rank, Charged Charged)>();
            for (int r = 1; r < world; r++)
            {
                foreach (long chunk in Chunks(values, per))
                {
                    arrivals.Add((r, Send(r, 0, chunk, NaiveToRoot)));
                }
            }

            foreach (var (_, charged) in arrivals)                 // rank 0 takes them in rank order
            {
                clock.Arrive(0, charged.Arrives, charged.OwedTo);
            }

            for (int r = 1; r < world; r++)
            {
                foreach (long chunk in Chunks(values, per))
                {
                    var charged = Send(0, r, chunk, NaiveFromRoot);
                    clock.Arrive(r, charged.Arrives, charged.OwedTo);
                }
            }
        }

        public void Hierarchical(IClusterTopology topology, int values)
        {
            int nodes = topology.Nodes.Count;
            var ranks = Enumerable.Range(0, nodes).Select(n => topology.Ranks(n).ToArray()).ToArray();
            foreach (var mine in ranks)
            {
                Scatter(mine, s => Size(values, mine.Length, s), IntraScatter);
            }

            if (nodes > 1)
            {
                foreach (var (start, end, members) in Columns(ranks, values))
                {
                    int span = end - start;
                    Scatter(members, s => Size(span, nodes, s), CrossScatter);
                    Gather(members, s => Size(span, nodes, s), CrossGather);
                }
            }

            foreach (var mine in ranks)
            {
                Gather(mine, s => Size(values, mine.Length, s), IntraGather);
            }
        }

        // HierarchicalCollectives' columns for an all-reduce: split at every node's slice boundary, one owner per node.
        private static IEnumerable<(int Start, int End, int[] Members)> Columns(int[][] ranks, int values)
        {
            var bounds = new SortedSet<int> { 0, values };
            foreach (var mine in ranks)
            {
                for (int s = 0; s <= mine.Length; s++)
                {
                    bounds.Add(Offset(values, mine.Length, s));
                }
            }

            int previous = -1;
            foreach (int bound in bounds)
            {
                if (previous >= 0 && bound > previous)
                {
                    int at = previous;
                    yield return (at, bound, [.. ranks.Select(mine => mine[SliceOf(at, values, mine.Length)])]);
                }

                previous = bound;
            }
        }

        // A ring's reduce-scatter: at step t the member at position q sends piece q - 1 - t (the play needs sizes only).
        private void Scatter(int[] members, Func<int, int> size, int tag)
        {
            for (int step = 0; step < members.Length - 1; step++)
            {
                int at = step;
                Exchange(members, q => size(Wrap(q - 1 - at, members.Length)), tag);
            }
        }

        // A ring's all-gather: at step t the member at position q sends piece q - t.
        private void Gather(int[] members, Func<int, int> size, int tag)
        {
            for (int step = 0; step < members.Length - 1; step++)
            {
                int at = step;
                Exchange(members, q => size(Wrap(q - at, members.Length)), tag);
            }
        }

        // One ring step: the member at position q sends values(q) to the next and takes the previous one's, chunk by chunk.
        private void Exchange(int[] members, Func<int, int> sending, int tag)
        {
            int m = members.Length;
            long per = PerChunk();
            var chunks = new long[m][];
            int most = 0;
            for (int q = 0; q < m; q++)
            {
                chunks[q] = [.. Chunks(sending(q), per)];
                most = Math.Max(most, chunks[q].Length);
            }

            var arrivals = new Charged[m];
            for (int k = 0; k < most; k++)
            {
                for (int q = 0; q < m; q++)
                {
                    if (k < chunks[q].Length)
                    {
                        arrivals[(q + 1) % m] = Send(members[q], members[(q + 1) % m], chunks[q][k], tag);
                    }
                }

                for (int q = 0; q < m; q++)
                {
                    if (k < chunks[(q - 1 + m) % m].Length)
                    {
                        clock.Arrive(members[q], arrivals[q].Arrives, arrivals[q].OwedTo);
                    }
                }
            }
        }

        private Charged Send(int from, int to, long bytes, int tag)
        {
            ref long sequence = ref System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrAddDefault(_sequence, (from, to, tag), out _);
            return clock.Charge(from, to, bytes, tag, sequence++);
        }

        private long PerChunk() => Math.Max(1, _chunk / sizeof(float));

        // The bytes of each chunk of `values` float32 values (none: one empty message).
        private static IEnumerable<long> Chunks(int values, long per)
        {
            if (values == 0)
            {
                yield return 0;
                yield break;
            }

            for (long at = 0; at < values; at += per)
            {
                yield return Math.Min(per, values - at) * sizeof(float);
            }
        }

        private static int Offset(int length, int parts, int s) => (s * (length / parts)) + Math.Min(s, length % parts);

        private static int Size(int length, int parts, int s) => (length / parts) + (s < length % parts ? 1 : 0);

        private static int SliceOf(int at, int length, int parts)
        {
            int slice = length / parts, extra = length % parts, longer = extra * (slice + 1);
            return at < longer ? at / (slice + 1) : extra + ((at - longer) / slice);
        }

        private static int Wrap(int value, int count) => ((value % count) + count) % count;
    }
}
