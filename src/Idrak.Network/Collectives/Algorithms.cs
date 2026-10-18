// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using Idrak.Networking.Abstractions;

namespace Idrak.Networking.Collectives;

/// <summary>
/// "naive" (plan 23): every rank sends rank 0 its values, rank 0 combines them in rank order (0, 1, 2, …) and sends the
/// result back. The simplest correct algorithm and the reference the others agree with (within float rounding; exactly
/// on whole numbers). On a bfloat16 wire every rank's values are rounded to bfloat16 before they travel; the sum is made
/// in float32 and comes back as float32.
/// </summary>
internal sealed class NaiveCollectives(ITransport transport, Device device, ClusterContext context) : HostCollectives(transport, device, context)
{
    private const int ToRoot = -1, FromRoot = -2;

    public override string Name => "naive";

    protected override void AllReduceHost(Span<float> values, ReduceOp op, WireType wire)
    {
        var travel = op == ReduceOp.Sum ? wire : WireType.Float32;
        if (travel == WireType.BFloat16)
        {
            WireValues.RoundToBFloat16(values);                    // rank 0's own values too: every rank's travel alike
        }

        if (Rank != 0)
        {
            Send(0, values, ToRoot, travel);
            Receive(0, values, FromRoot, WireType.Float32);
            return;
        }

        var other = ArrayPool<float>.Shared.Rent(Math.Max(1, values.Length));
        try
        {
            for (int r = 1; r < WorldSize; r++)
            {
                Receive(r, other.AsSpan(0, values.Length), ToRoot, travel);
                Combine(values, other.AsSpan(0, values.Length), op);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(other);
        }

        for (int r = 1; r < WorldSize; r++)
        {
            Send(r, values, FromRoot, WireType.Float32);
        }
    }

    protected override void AllGatherHost(Span<float> all, int count)
    {
        if (Rank != 0)
        {
            Send(0, all.Slice(Rank * count, count), ToRoot, WireType.Float32);
            Receive(0, all, FromRoot, WireType.Float32);
            return;
        }

        for (int r = 1; r < WorldSize; r++)
        {
            Receive(r, all.Slice(r * count, count), ToRoot, WireType.Float32);
        }

        for (int r = 1; r < WorldSize; r++)
        {
            Send(r, all, FromRoot, WireType.Float32);
        }
    }

    protected override void ReduceScatterHost(Span<float> values, int count, ReduceOp op) => AllReduceHost(values, op, WireType.Float32);

    protected override void BroadcastHost(Span<float> values, int root)
    {
        if (Rank != root)
        {
            Receive(root, values, FromRoot, WireType.Float32);
            return;
        }

        for (int r = 0; r < WorldSize; r++)
        {
            if (r != root)
            {
                Send(r, values, FromRoot, WireType.Float32);
            }
        }
    }
}

/// <summary>
/// "ring" (plan 23): reduce-scatter then all-gather around the ranks, so each rank sends and receives 2·(N−1)/N of the
/// values instead of rank 0 handling them all. Slice j's reduction starts at rank j+1 and goes round, each rank adding
/// its own values to what arrived, and ends complete at rank j: an order fixed by the sizes alone (rule 86), so every rank
/// ends with identical bytes and a rerun gives them again; it differs from naive's rank order, so the two agree within
/// float rounding. On a bfloat16 wire the partial sums travel as bfloat16 (rounded once a hop), the finished slices as
/// float32. Each step sends and receives chunk by chunk in turn, so no rank waits on another in a circle.
/// </summary>
internal sealed class RingCollectives(ITransport transport, Device device, ClusterContext context) : HostCollectives(transport, device, context)
{
    private const int ScatterTag = -11, GatherTag = -12, BroadcastTag = -13;

    public override string Name => "ring";

    private int Next => (Rank + 1) % WorldSize;

    private int Previous => (Rank - 1 + WorldSize) % WorldSize;

    protected override void AllReduceHost(Span<float> values, ReduceOp op, WireType wire)
    {
        var travel = op == ReduceOp.Sum ? wire : WireType.Float32;
        int world = WorldSize, slice = values.Length / world, extra = values.Length % world;
        int Offset(int s) => (s * slice) + Math.Min(s, extra);
        int Size(int s) => slice + (s < extra ? 1 : 0);
        ScatterReduce(values, Offset, Size, op, travel);
        GatherAround(values, Offset, Size);
    }

    protected override void AllGatherHost(Span<float> all, int count) => GatherAround(all, s => s * count, _ => count);

    protected override void ReduceScatterHost(Span<float> values, int count, ReduceOp op) => ScatterReduce(values, s => s * count, _ => count, op, WireType.Float32);

    protected override void BroadcastHost(Span<float> values, int root)
    {
        // A chain from the root: each rank takes the values from the one before and passes them on, chunk by chunk.
        if (WorldSize == 1)
        {
            return;
        }

        if (Rank == root)
        {
            Send(Next, values, BroadcastTag, WireType.Float32);
        }
        else if (Next == root)
        {
            Receive(Previous, values, BroadcastTag, WireType.Float32);
        }
        else
        {
            Relay(values);
        }
    }

    // Reduce-scatter: after N−1 steps rank r holds slice r complete, reduced in the order r+1, r+2, …, r.
    private void ScatterReduce(Span<float> values, Func<int, int> offset, Func<int, int> size, ReduceOp op, WireType wire)
    {
        int world = WorldSize;
        if (world == 1)
        {
            return;
        }

        int largest = 0;
        for (int s = 0; s < world; s++)
        {
            largest = Math.Max(largest, size(s));
        }

        var arrived = ArrayPool<float>.Shared.Rent(Math.Max(1, largest));
        try
        {
            for (int step = 0; step < world - 1; step++)
            {
                int send = Mod(Rank - 1 - step), take = Mod(Rank - 2 - step);
                var outgoing = values.Slice(offset(send), size(send));
                if (wire == WireType.BFloat16 && step == 0)
                {
                    WireValues.RoundToBFloat16(outgoing);          // this rank's own values as they first travel
                }

                var incoming = arrived.AsSpan(0, size(take));
                Exchange(Next, outgoing, Previous, incoming, ScatterTag, wire);
                var mine = values.Slice(offset(take), size(take));
                if (wire == WireType.BFloat16)
                {
                    WireValues.RoundToBFloat16(mine);              // as every rank's own values travel
                }

                Combine(mine, incoming, op);                       // what arrived, then this rank's own
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(arrived);
        }
    }

    // All-gather: rank r holds slice r; after N−1 steps every rank holds every slice, passed round as float32.
    private void GatherAround(Span<float> values, Func<int, int> offset, Func<int, int> size)
    {
        for (int step = 0; step < WorldSize - 1; step++)
        {
            int send = Mod(Rank - step), take = Mod(Rank - 1 - step);
            Exchange(Next, values.Slice(offset(send), size(send)), Previous, values.Slice(offset(take), size(take)), GatherTag, WireType.Float32);
        }
    }

    // A middle rank of a broadcast chain: takes the values from the rank before, then passes them on (store and forward;
    // the chain never closes, so no rank waits on another in a circle).
    private void Relay(Span<float> values)
    {
        Receive(Previous, values, BroadcastTag, WireType.Float32);
        Send(Next, values, BroadcastTag, WireType.Float32);
    }

    private int Mod(int value) => ((value % WorldSize) + WorldSize) % WorldSize;
}
