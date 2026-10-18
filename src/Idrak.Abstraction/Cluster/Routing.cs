// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Abstraction.Cluster;

/// <summary>
/// One group over several transports (plan 23, phase C5): each carrier the plan uses is connected once, for the ranks its
/// pairs join (a carrier over a part of the group sees those ranks as a group of their own, numbered from 0 in rank
/// order), and every message goes to the carrier of its pair. A barrier is made of messages to rank 0 and back over the
/// pairs' own carriers.
/// </summary>
internal sealed class RoutedTransport : ITransport
{
    private const int BarrierTag = int.MinValue + 7;   // below every collective's tag: never a program's or an algorithm's
    private readonly ITransport[] _carriers;
    private readonly (int Carrier, int Rank)[] _routes;  // for each peer: its carrier and its rank in that carrier's group

    private RoutedTransport(string name, int rank, int world, ITransport[] carriers, (int, int)[] routes)
    {
        Name = name;
        Rank = rank;
        WorldSize = world;
        _carriers = carriers;
        _routes = routes;
        var all = carriers.Select(c => c.Capabilities).ToArray();
        Capabilities = all.Length == 0
            ? new TransportCapabilities(0, TimeSpan.Zero, false, false, true, false, false)
            : new TransportCapabilities(all.Min(c => c.BandwidthBytesPerSecond), all.Max(c => c.Latency), all.All(c => c.ZeroCopy), all.All(c => c.GpuDirect),
                all.All(c => c.Reliable), all.Any(c => c.CrossProcess), all.Any(c => c.CrossMachine));
    }

    public string Name { get; }

    public int Rank { get; }

    public int WorldSize { get; }

    public TransportCapabilities Capabilities { get; }

    public TimeSpan SimulatedTime => _carriers.Length == 0 ? TimeSpan.Zero : _carriers.Max(c => c.SimulatedTime);

    public bool Simulated => _carriers.Any(c => c.Simulated);

    public static RoutedTransport Connect(TransportPlan plan, ClusterContext context, CancellationToken cancellationToken)
    {
        int world = plan.WorldSize, me = context.Rank;
        // The carriers, in the order of the plan's transports (the same on every rank).
        var carriers = new List<(string Carrier, ITransportProvider Provider, HashSet<string> Transports)>();
        foreach (string name in plan.Used)
        {
            var provider = Transports.Get(name);
            string carrier = provider.Carrier;
            int at = carriers.FindIndex(c => string.Equals(c.Carrier, carrier, StringComparison.OrdinalIgnoreCase));
            if (at < 0)
            {
                carriers.Add((carrier, provider, new HashSet<string>(StringComparer.OrdinalIgnoreCase) { name }));
            }
            else
            {
                carriers[at].Transports.Add(name);
            }
        }

        var connected = new List<ITransport>();
        var routes = new (int, int)[world];
        Array.Fill(routes, (-1, -1));
        try
        {
            foreach (var (carrier, provider, names) in carriers)
            {
                int[] members = Component(plan, world, me, names);
                if (members.Length < 2)
                {
                    continue;                                      // no pair of this rank rides on it
                }

                if (provider.Capabilities.CrossMachine && members[0] != 0)
                {
                    throw new NotSupportedException($"Transport '{carrier}' joins ranks {string.Join(", ", members)}, which leave out rank 0, the coordinator its rendezvous needs; force one transport with ClusterOptions.Transport (IDRAK_CLUSTER_TRANSPORT).");
                }

                ClusterContext sub = members.Length == world
                    ? context
                    : new ClusterContext(new SubsetTopology(context.Topology, members), Array.IndexOf(members, me),
                        context.Options with { RunId = $"{context.Options.RunId ?? context.Options.Coordinator ?? "run"}.{carrier}.{string.Join('-', members)}" });
                var transport = provider.Connect(sub, cancellationToken);
                connected.Add(transport);
                for (int p = 0; p < world; p++)
                {
                    if (p != me && names.Contains(plan[me, p].Transport))
                    {
                        routes[p] = (connected.Count - 1, Array.IndexOf(members, p));
                    }
                }
            }
        }
        catch
        {
            foreach (var transport in connected)
            {
                transport.Dispose();
            }

            throw;
        }

        return new RoutedTransport(string.Join("+", carriers.Select(c => c.Carrier)), me, world, [.. connected], routes);
    }

    // The ranks joined to `rank` by pairs over `transports`, in rank order.
    private static int[] Component(TransportPlan plan, int world, int rank, HashSet<string> transports)
    {
        var seen = new bool[world];
        var stack = new Stack<int>();
        stack.Push(rank);
        seen[rank] = true;
        while (stack.Count > 0)
        {
            int a = stack.Pop();
            for (int b = 0; b < world; b++)
            {
                if (!seen[b] && b != a && transports.Contains(plan[a, b].Transport))
                {
                    seen[b] = true;
                    stack.Push(b);
                }
            }
        }

        return [.. Enumerable.Range(0, world).Where(r => seen[r])];
    }

    public void Send(int to, ReadOnlySpan<byte> bytes, int tag)
    {
        var (carrier, rank) = Route(to, nameof(to));
        _carriers[carrier].Send(rank, bytes, tag);
    }

    public int Receive(int from, Span<byte> bytes, int tag)
    {
        var (carrier, rank) = Route(from, nameof(from));
        return _carriers[carrier].Receive(rank, bytes, tag);
    }

    public ValueTask SendAsync(int to, ReadOnlyMemory<byte> bytes, int tag, CancellationToken cancellationToken = default)
    {
        var (carrier, rank) = Route(to, nameof(to));
        return _carriers[carrier].SendAsync(rank, bytes, tag, cancellationToken);
    }

    public ValueTask<int> ReceiveAsync(int from, Memory<byte> bytes, int tag, CancellationToken cancellationToken = default)
    {
        var (carrier, rank) = Route(from, nameof(from));
        return _carriers[carrier].ReceiveAsync(rank, bytes, tag, cancellationToken);
    }

    public void Barrier()
    {
        // Every rank tells rank 0 it is here; rank 0 answers each once all have come.
        Span<byte> none = [];
        if (Rank != 0)
        {
            Send(0, none, BarrierTag);
            Receive(0, none, BarrierTag);
            return;
        }

        for (int r = 1; r < WorldSize; r++)
        {
            Receive(r, none, BarrierTag);
        }

        for (int r = 1; r < WorldSize; r++)
        {
            Send(r, none, BarrierTag);
        }
    }

    public void Dispose()
    {
        foreach (var carrier in _carriers)
        {
            carrier.Dispose();
        }
    }

    private (int Carrier, int Rank) Route(int peer, string name)
    {
        if (peer == Rank || (uint)peer >= (uint)WorldSize)
        {
            throw new ArgumentOutOfRangeException(name, peer, $"rank {Rank} reaches the other ranks of 0 to {WorldSize - 1}");
        }

        return _routes[peer] is { Carrier: >= 0 } route
            ? route
            : throw new InvalidOperationException(string.Create(CultureInfo.InvariantCulture, $"rank {Rank} has no transport to rank {peer}"));
    }
}

/// <summary>
/// Some ranks of a topology as a topology of their own, numbered from 0 in rank order: the nodes that hold them (each with
/// those ranks' devices only, in order) and the original's links between them.
/// </summary>
internal sealed class SubsetTopology : IClusterTopology
{
    private readonly IClusterTopology _whole;
    private readonly int[] _members;
    private readonly (int Node, int Local)[] _places;
    private readonly List<int>[] _ranks;
    private readonly string[] _names;

    public SubsetTopology(IClusterTopology whole, int[] members)
    {
        _whole = whole;
        _members = members;
        var nodes = new List<ClusterNodeSpec>();
        var nodeOf = new Dictionary<int, int>();
        var devices = new List<List<ClusterDeviceSpec>>();
        _places = new (int, int)[members.Length];
        for (int i = 0; i < members.Length; i++)
        {
            var (node, local) = whole.Place(members[i]);
            if (!nodeOf.TryGetValue(node, out int at))
            {
                at = nodes.Count;
                nodeOf[node] = at;
                nodes.Add(whole.Nodes[node]);
                devices.Add([]);
            }

            _places[i] = (at, devices[at].Count);
            devices[at].Add(whole.Nodes[node].Devices[local]);
        }

        Nodes = [.. nodes.Select((n, i) => n with { Devices = devices[i] })];
        _ranks = [.. Enumerable.Range(0, Nodes.Count).Select(n => Enumerable.Range(0, members.Length).Where(r => _places[r].Node == n).ToList())];
        _names = [.. Nodes.Select(n => n.Name)];
    }

    public IReadOnlyList<ClusterNodeSpec> Nodes { get; }

    public int WorldSize => _members.Length;

    public ClusterSimulationSpec Simulation => _whole.Simulation;

    public ILinkProfile Link(int rankA, int rankB) => _whole.Link(_members[rankA], _members[rankB]);

    public string? AddressOf(int rank, int from) => _whole.AddressOf(_members[rank], _members[from]);

    public (int Node, int Local) Place(int rank) => _places[rank];

    public IReadOnlyList<int> Ranks(int node) => _ranks[node];

    public int? NodeOf(string hostName)
    {
        int at = Array.FindIndex(_names, n => string.Equals(n, hostName, StringComparison.OrdinalIgnoreCase));
        return at < 0 ? null : at;
    }
}
