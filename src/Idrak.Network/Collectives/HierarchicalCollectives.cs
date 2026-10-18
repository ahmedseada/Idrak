// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Networking.Abstractions;

namespace Idrak.Networking.Collectives;

/// <summary>
/// "hierarchical" (plan 23, phase C3): the topology's two levels used apart. An all-reduce reduce-scatters inside each
/// node over its own link (a ring of the node's ranks), so each rank holds its node's sum of one slice; then the ranks
/// that hold the same values in every node, one rank per node, run a ring across the nodes on those shards only; then
/// each node all-gathers inside. The slow links between machines carry 2(N−1)/N of a node's share instead of every rank's
/// whole buffer crossing them in turn, and the ranks of a node use them side by side.
///
/// Nodes of different sizes cut the values differently, so the values are split into columns at every node's slice
/// boundary: each column has one owner per node (the rank whose slice holds it) and its own ring across the nodes. Every
/// sum is made in an order fixed by the topology and the sizes (inside each node in ring order, then across the nodes in
/// ring order: rule 86), so every rank ends with identical bytes and a rerun gives them again; the order differs from
/// ring's and naive's, so the three agree within float rounding, exactly on whole numbers.
///
/// Reduce-scatter cuts the cross-node pieces where the destination nodes' slices begin, so the ring across the nodes
/// leaves each node's results in that node, and they are delivered inside it; all-gather is the same backwards; a
/// broadcast goes along a chain of one rank per node, then along a chain inside each node.
/// </summary>
internal sealed class HierarchicalCollectives : HostCollectives
{
    private const int IntraScatterTag = -21, IntraGatherTag = -22, CrossScatterTag = -23, CrossGatherTag = -24, DeliverTag = -25,
        CrossChainTag = -26, IntraChainTag = -27;

    private readonly int[][] _nodeRanks;   // each node's ranks, in order
    private readonly int[] _first;         // each node's first rank: a node's ranks are numbered together
    private readonly int[] _rankNode;      // each rank's node
    private readonly int _node;            // this rank's node
    private readonly int _position;        // this rank's place in its node
    private ColumnPlan? _plan;             // the last call's columns: a training run reduces the same sizes every step (rule 75)

    public HierarchicalCollectives(ITransport transport, Device device, ClusterContext context)
        : base(transport, device, context)
    {
        var topology = context.Topology;
        if (topology.WorldSize != transport.WorldSize)
        {
            throw new ArgumentException($"hierarchical needs the group's own topology: the transport joins {transport.WorldSize} rank(s), the topology has {topology.WorldSize}.", nameof(context));
        }

        int nodes = topology.Nodes.Count;
        _nodeRanks = new int[nodes][];
        _first = new int[nodes];
        _rankNode = new int[topology.WorldSize];
        for (int n = 0, expected = 0; n < nodes; n++)
        {
            var ranks = topology.Ranks(n);
            _nodeRanks[n] = [.. ranks];
            _first[n] = expected;
            for (int i = 0; i < ranks.Count; i++, expected++)
            {
                if (ranks[i] != expected)
                {
                    throw new ArgumentException($"hierarchical needs each node's ranks numbered together, node after node (as a topology file numbers them); node {n} holds {string.Join(", ", ranks)}.", nameof(context));
                }

                _rankNode[expected] = n;
            }
        }

        (_node, _position) = topology.Place(transport.Rank);
    }

    public override string Name => "hierarchical";

    private int Nodes => _nodeRanks.Length;

    private int[] Mine => _nodeRanks[_node];

    protected override void AllReduceHost(Span<float> values, ReduceOp op, WireType wire)
    {
        var travel = op == ReduceOp.Sum ? wire : WireType.Float32;
        int length = values.Length, local = Mine.Length, nodes = Nodes;
        RingScatter(values, Mine, _position, s => Offset(length, local, s), s => Size(length, local, s), op, travel, IntraScatterTag);
        if (nodes > 1)
        {
            foreach (var column in PlanFor(length, 0).Columns)
            {
                if (column.Owners[_node] != _position)
                {
                    continue;
                }

                int span = column.End - column.Start;
                var shard = values.Slice(column.Start, span);
                RingScatter(shard, column.Members, _node, s => Offset(span, nodes, s), s => Size(span, nodes, s), op, travel, CrossScatterTag);
                RingGather(shard, column.Members, _node, s => Offset(span, nodes, s), s => Size(span, nodes, s), CrossGatherTag);
            }
        }

        RingGather(values, Mine, _position, s => Offset(length, local, s), s => Size(length, local, s), IntraGatherTag);
    }

    protected override void ReduceScatterHost(Span<float> values, int count, ReduceOp op)
    {
        int total = values.Length, local = Mine.Length;
        RingScatter(values, Mine, _position, s => Offset(total, local, s), s => Size(total, local, s), op, WireType.Float32, IntraScatterTag);
        var plan = PlanFor(total, count);
        if (Nodes > 1)
        {
            foreach (var column in plan.Columns)
            {
                if (column.Owners[_node] == _position)
                {
                    int start = column.Start, span = column.End - start;
                    RingScatter(values.Slice(start, span), column.Members, _node, n => Piece(start, span, count, n).Offset, n => Piece(start, span, count, n).Size, op,
                        WireType.Float32, CrossScatterTag);
                }
            }
        }

        Deliver(values, plan, count, toOwners: false);
    }

    protected override void AllGatherHost(Span<float> all, int count)
    {
        int total = all.Length, local = Mine.Length;
        var plan = PlanFor(total, count);
        Deliver(all, plan, count, toOwners: true);
        if (Nodes > 1)
        {
            foreach (var column in plan.Columns)
            {
                if (column.Owners[_node] == _position)
                {
                    int start = column.Start, span = column.End - start;
                    RingGather(all.Slice(start, span), column.Members, _node, n => Piece(start, span, count, n).Offset, n => Piece(start, span, count, n).Size, CrossGatherTag);
                }
            }
        }

        RingGather(all, Mine, _position, s => Offset(total, local, s), s => Size(total, local, s), IntraGatherTag);
    }

    protected override void BroadcastHost(Span<float> values, int root)
    {
        // A chain of one rank per node (the root, then each other node's first rank) in node order from the root's, then a
        // chain inside each node from its rank on the first chain. No chain closes, so no rank waits on another in a circle.
        int nodes = Nodes, rootNode = _rankNode[root];
        int Leader(int n) => n == rootNode ? root : _nodeRanks[n][0];
        int leader = Leader(_node);
        if (Rank == leader)
        {
            int place = Wrap(_node - rootNode, nodes);
            if (place > 0)
            {
                Receive(Leader(Wrap(_node - 1, nodes)), values, CrossChainTag, WireType.Float32);
            }

            if (place < nodes - 1)
            {
                Send(Leader(Wrap(_node + 1, nodes)), values, CrossChainTag, WireType.Float32);
            }
        }

        var mine = Mine;
        int local = mine.Length;
        if (local == 1)
        {
            return;
        }

        int k = Wrap(_position - Array.IndexOf(mine, leader), local);   // this rank's place in its node's chain
        if (k > 0)
        {
            Receive(mine[Wrap(_position - 1, local)], values, IntraChainTag, WireType.Float32);
        }

        if (k < local - 1)
        {
            Send(mine[Wrap(_position + 1, local)], values, IntraChainTag, WireType.Float32);
        }
    }

    // Moves each rank's slice of a node's results between the rank that holds it after the ring across the nodes (its
    // column's owner in this node) and the rank it belongs to: to the rank for reduce-scatter, to the owner for all-gather.
    // Every rank of the node walks the pieces in one order, so every send meets its receive (no rank waits in a circle).
    private void Deliver(Span<float> values, ColumnPlan plan, int count, bool toOwners)
    {
        int blockStart = _first[_node] * count, blockEnd = (_first[_node] + Mine.Length) * count;
        foreach (var column in plan.Columns)
        {
            int from = Math.Max(column.Start, blockStart), to = Math.Min(column.End, blockEnd);
            int owner = Mine[column.Owners[_node]];
            for (int at = from; at < to;)
            {
                int target = at / count, end = Math.Min(to, (target + 1) * count);
                var piece = values[at..end];
                if (owner != target)
                {
                    int sender = toOwners ? target : owner, receiver = toOwners ? owner : target;
                    if (Rank == sender)
                    {
                        Send(receiver, piece, DeliverTag, WireType.Float32);
                    }
                    else if (Rank == receiver)
                    {
                        Receive(sender, piece, DeliverTag, WireType.Float32);
                    }
                }

                at = end;
            }
        }
    }

    // Node n's part of the column [start, start + span): the values destined for node n's ranks (reduce-scatter's slices
    // of `count`), relative to the column. Nodes are numbered in rank order, so the parts follow one another.
    private (int Offset, int Size) Piece(int start, int span, int count, int n)
    {
        long lo = Math.Clamp(((long)_first[n] * count) - start, 0, span), hi = Math.Clamp(((long)(_first[n] + _nodeRanks[n].Length) * count) - start, 0, span);
        return ((int)lo, (int)(hi - lo));
    }

    // The columns of `length` values: split at every node's slice boundary (and, with `count` > 0, at every node's block
    // of reduce-scatter slices), runs with the same owner in every node joined into one.
    private ColumnPlan PlanFor(int length, int count)
    {
        if (_plan is { } kept && kept.Length == length && kept.Count == count)
        {
            return kept;
        }

        int nodes = Nodes;
        var bounds = new SortedSet<int> { 0, length };
        foreach (var ranks in _nodeRanks)
        {
            for (int s = 0; s <= ranks.Length; s++)
            {
                bounds.Add(Offset(length, ranks.Length, s));
            }
        }

        if (count > 0)
        {
            foreach (int first in _first)
            {
                bounds.Add(first * count);
            }
        }

        var columns = new List<Column>();
        int previous = -1;
        int[]? owners = null;
        foreach (int bound in bounds)
        {
            if (previous >= 0 && bound > previous)
            {
                var here = new int[nodes];
                for (int n = 0; n < nodes; n++)
                {
                    here[n] = SliceOf(previous, length, _nodeRanks[n].Length);
                }

                if (owners is not null && here.AsSpan().SequenceEqual(owners))
                {
                    columns[^1] = columns[^1] with { End = bound };
                }
                else
                {
                    var members = new int[nodes];
                    for (int n = 0; n < nodes; n++)
                    {
                        members[n] = _nodeRanks[n][here[n]];
                    }

                    columns.Add(new Column(previous, bound, here, members));
                    owners = here;
                }
            }

            previous = bound;
        }

        return _plan = new ColumnPlan(length, count, [.. columns]);
    }

    // Slice s of `length` values cut into `parts` (the first length % parts slices one longer): ring's cut.
    private static int Offset(int length, int parts, int s) => (s * (length / parts)) + Math.Min(s, length % parts);

    private static int Size(int length, int parts, int s) => (length / parts) + (s < length % parts ? 1 : 0);

    // The slice of that cut holding value `at`.
    private static int SliceOf(int at, int length, int parts)
    {
        int slice = length / parts, extra = length % parts, longer = extra * (slice + 1);
        return at < longer ? at / (slice + 1) : extra + ((at - longer) / slice);
    }

    // A run of values with one owner per node: its owner's place in each node and that owner's rank.
    private readonly record struct Column(int Start, int End, int[] Owners, int[] Members);

    private sealed record ColumnPlan(int Length, int Count, Column[] Columns);
}
