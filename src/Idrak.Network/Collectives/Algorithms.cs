// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using Idrak.Networking.Abstractions;

namespace Idrak.Networking.Collectives;

/// <summary>
/// "naive" (plan 23): every rank sends rank 0 its values, rank 0 combines them in rank order (0, 1, 2, …) and sends the
/// result back. The simplest correct algorithm and the reference the others agree with (within float rounding; exactly
/// on whole numbers).
/// </summary>
internal sealed class NaiveCollectives(ITransport transport, Device device, ClusterContext context) : HostCollectives(transport, device, context)
{
    private const int ToRoot = -1, FromRoot = -2;

    public override string Name => "naive";

    protected override void AllReduceHost(Span<float> values, ReduceOp op)
    {
        if (Rank != 0)
        {
            Send(0, values, ToRoot);
            Receive(0, values, FromRoot);
            return;
        }

        var other = ArrayPool<float>.Shared.Rent(Math.Max(1, values.Length));
        try
        {
            for (int r = 1; r < WorldSize; r++)
            {
                Receive(r, other.AsSpan(0, values.Length), ToRoot);
                Combine(values, other.AsSpan(0, values.Length), op);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(other);
        }

        for (int r = 1; r < WorldSize; r++)
        {
            Send(r, values, FromRoot);
        }
    }

    protected override void AllGatherHost(Span<float> all, int count)
    {
        if (Rank != 0)
        {
            Send(0, all.Slice(Rank * count, count), ToRoot);
            Receive(0, all, FromRoot);
            return;
        }

        for (int r = 1; r < WorldSize; r++)
        {
            Receive(r, all.Slice(r * count, count), ToRoot);
        }

        for (int r = 1; r < WorldSize; r++)
        {
            Send(r, all, FromRoot);
        }
    }

    protected override void ReduceScatterHost(Span<float> values, int count, ReduceOp op) => AllReduceHost(values, op);

    protected override void BroadcastHost(Span<float> values, int root)
    {
        if (Rank != root)
        {
            Receive(root, values, FromRoot);
            return;
        }

        for (int r = 0; r < WorldSize; r++)
        {
            if (r != root)
            {
                Send(r, values, FromRoot);
            }
        }
    }
}

/// <summary>
/// "ring" (plan 23): reduce-scatter then all-gather around the ranks, so each rank sends and receives 2·(N−1)/N of the
/// values instead of rank 0 handling them all. Slice j's reduction starts at rank j+1 and goes round, each rank adding
/// its own values to what arrived, and ends complete at rank j: an order fixed by the sizes alone (rule 86), so every rank
/// ends with identical bytes and a rerun gives them again; it differs from naive's rank order, so the two agree within
/// float rounding. Each step sends and receives chunk by chunk in turn, so no rank waits on another in a circle.
/// <para>
/// Phase C6: over a transport that moves device memory (<see cref="HostCollectives.DeviceTransport"/>), a float32 sum,
/// all-gather, reduce-scatter and broadcast keep their pieces on the device and add them there, the same sums in the same
/// order (bit for bit the host path's); the pieces of a sharded run's buckets (each rank's own count) go round the same ring.
/// </para>
/// </summary>
internal sealed class RingCollectives(ITransport transport, Device device, ClusterContext context) : HostCollectives(transport, device, context)
{
    private const int ScatterTag = -11, GatherTag = -12, BroadcastTag = -13, DeviceTag = -14;

    private readonly int[] _ring = [.. Enumerable.Range(0, transport.WorldSize)];   // every rank, in rank order

    public override string Name => "ring";

    private int Next => (Rank + 1) % WorldSize;

    private int Previous => (Rank - 1 + WorldSize) % WorldSize;

    protected override void AllReduceHost(Span<float> values, ReduceOp op)
    {
        int world = WorldSize, slice = values.Length / world, extra = values.Length % world;
        int Offset(int s) => (s * slice) + Math.Min(s, extra);
        int Size(int s) => slice + (s < extra ? 1 : 0);
        RingScatter(values, _ring, Rank, Offset, Size, op, ScatterTag);
        RingGather(values, _ring, Rank, Offset, Size, GatherTag);
    }

    protected override void AllGatherHost(Span<float> all, int count) => RingGather(all, _ring, Rank, s => s * count, _ => count, GatherTag);

    protected override void ReduceScatterHost(Span<float> values, int count, ReduceOp op) =>
        RingScatter(values, _ring, Rank, s => s * count, _ => count, op, ScatterTag);

    // Pieces of each rank's own size (a sharded bucket): the same ring, each piece ending at its rank.
    protected override void ReduceScatterPiecesHost(Span<float> values, int[] counts, ReduceOp op)
    {
        var starts = Starts(counts);
        RingScatter(values, _ring, Rank, s => starts[s], s => counts[s], op, ScatterTag);
    }

    protected override void AllGatherPiecesHost(Span<float> values, int[] counts)
    {
        var starts = Starts(counts);
        RingGather(values, _ring, Rank, s => starts[s], s => counts[s], GatherTag);
    }

    protected override bool AllReduceDevice(Storage buffer, int count)
    {
        int world = WorldSize, slice = count / world, extra = count % world;
        int Offset(int s) => (s * slice) + Math.Min(s, extra);
        int Size(int s) => slice + (s < extra ? 1 : 0);
        DeviceRingScatter(buffer, _ring, Rank, Offset, Size, buffer, Offset(Rank), DeviceTag);
        DeviceRingGather(buffer, _ring, Rank, Offset, Size, DeviceTag);
        return true;
    }

    protected override bool AllGatherDevice(Storage input, int count, Storage output)
    {
        Device.Backend.Copy2D(input, 0, 0, output, Rank * count, 0, 1, count, accumulate: false);
        DeviceRingGather(output, _ring, Rank, s => s * count, _ => count, DeviceTag);
        return true;
    }

    protected override bool ReduceScatterDevice(Storage input, int count, Storage output)
    {
        DeviceRingScatter(input, _ring, Rank, s => s * count, _ => count, output, 0, DeviceTag);
        return true;
    }

    protected override bool BroadcastDevice(Storage buffer, int count, int root)
    {
        DeviceChainBroadcast(buffer, count, root, DeviceTag);
        return true;
    }

    protected override void BroadcastHost(Span<float> values, int root)
    {
        // A chain from the root: each rank takes the values from the one before and passes them on (store and forward;
        // the chain never closes, so no rank waits on another in a circle).
        if (WorldSize == 1)
        {
            return;
        }

        if (Rank == root)
        {
            Send(Next, values, BroadcastTag);
        }
        else if (Next == root)
        {
            Receive(Previous, values, BroadcastTag);
        }
        else
        {
            Receive(Previous, values, BroadcastTag);
            Send(Next, values, BroadcastTag);
        }
    }
}
