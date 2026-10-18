// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Cluster;

/// <summary>
/// The topology of some of a cluster's ranks (a group formed while a late joiner is away): the nodes that keep a rank,
/// each with the devices of its ranks in the group, numbered from 0 in the cluster's order, and the cluster's links between
/// them.
/// </summary>
internal sealed class GroupTopology : IClusterTopology
{
    private readonly IClusterTopology _cluster;
    private readonly int[] _real;
    private readonly (int Node, int Local)[] _place;
    private readonly int[][] _nodeRanks;
    private readonly Dictionary<string, int> _byName = new(StringComparer.OrdinalIgnoreCase);

    public GroupTopology(IClusterTopology cluster, int[] members)
    {
        _cluster = cluster;
        _real = members;
        var nodes = new List<ClusterNodeSpec>();
        var nodeRanks = new List<int[]>();
        _place = new (int, int)[members.Length];
        for (int n = 0; n < cluster.Nodes.Count; n++)
        {
            var kept = cluster.Ranks(n).Where(r => Array.BinarySearch(members, r) >= 0).ToArray();
            if (kept.Length == 0)
            {
                continue;
            }

            var spec = cluster.Nodes[n];
            var ranks = new int[kept.Length];
            for (int i = 0; i < kept.Length; i++)
            {
                ranks[i] = Array.BinarySearch(members, kept[i]);   // the group's rank for it: its place among the members
                _place[ranks[i]] = (nodes.Count, i);
            }

            _byName[spec.Name] = nodes.Count;
            nodes.Add(spec with { Devices = [.. kept.Select(r => spec.Devices[cluster.Place(r).Local])] });
            nodeRanks.Add(ranks);
        }

        Nodes = nodes;
        _nodeRanks = [.. nodeRanks];
    }

    public IReadOnlyList<ClusterNodeSpec> Nodes { get; }

    public int WorldSize => _real.Length;

    public ClusterSimulationSpec Simulation => _cluster.Simulation;

    public ILinkProfile Link(int rankA, int rankB) => _cluster.Link(_real[rankA], _real[rankB]);

    public string? AddressOf(int rank, int from) => _cluster.AddressOf(_real[rank], _real[from]);

    public (int Node, int Local) Place(int rank) => _place[rank];

    public IReadOnlyList<int> Ranks(int node) => _nodeRanks[node];

    public int? NodeOf(string hostName) => _byName.TryGetValue(hostName, out int n) ? n : null;
}
