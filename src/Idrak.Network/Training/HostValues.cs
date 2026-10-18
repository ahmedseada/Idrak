// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Networking.Training;

/// <summary>
/// Collectives over values held on the host, for the gradient synchronisations and compressions (plan 23): straight from
/// the host buffer when the collectives take host values (<see cref="IHostValueCollectives"/>: the library's, and a
/// simulated rank over them), else through one device buffer made on first need and kept (a plug-in's collectives work on
/// device storage). Only the first kind may run off the training thread, so a synchronisation overlaps with the backward
/// pass only when <see cref="OnHost"/> is true.
/// </summary>
internal sealed class HostValues(ICollectives collectives) : IDisposable
{
    private readonly IHostValueCollectives? _host = collectives is IHostValueCollectives { SupportsHostValues: true } host ? host : null;
    private Tensor? _scratch;

    /// <summary>Whether the values travel from the host buffer itself (and so may from any thread).</summary>
    public bool OnHost => _host is not null;

    /// <summary>This rank's time on a simulated transport's clock (zero on a real one, or behind device-only collectives).</summary>
    public TimeSpan SimulatedTime => _host?.SimulatedTime ?? TimeSpan.Zero;

    /// <summary>Whether the transport under the collectives is a simulated one, behind device-only collectives too.</summary>
    public bool Simulated => _host?.Simulated ?? collectives.Transport?.Simulated ?? false;

    public ICollectives Collectives => collectives;

    /// <summary>Sums <paramref name="values"/> across the ranks in place.</summary>
    public void AllReduce(Span<float> values, WireType wire, ReduceOp op = ReduceOp.Sum)
    {
        if (values.IsEmpty || _host?.TryAllReduce(values, op, wire) == true)
        {
            return;
        }

        var scratch = Scratch(values.Length);
        var backend = collectives.Device.Backend;
        backend.Upload(values, scratch.Storage);
        collectives.AllReduce(scratch.Storage, values.Length, op, wire);
        backend.DownloadRange(scratch.Storage, 0, values);
    }

    /// <summary><paramref name="all"/> holds this rank's <paramref name="count"/> values at Rank · count; every rank's afterwards.</summary>
    public void AllGather(Span<float> all, int count)
    {
        if (count == 0 || _host?.TryAllGather(all, count) == true)
        {
            return;
        }

        int rank = collectives.Rank;
        var input = Scratch(count + all.Length);
        var backend = collectives.Device.Backend;
        using var output = Tensor.PersistentZeros([all.Length], collectives.Device);
        backend.Upload(all.Slice(rank * count, count), input.Storage);
        collectives.AllGather(input.Storage, count, output.Storage);
        backend.Download(output.Storage, all);
    }

    /// <summary>Copies rank <paramref name="root"/>'s <paramref name="values"/> to every rank's.</summary>
    public void Broadcast(Span<float> values, int root)
    {
        if (values.IsEmpty || _host?.TryBroadcast(values, root) == true)
        {
            return;
        }

        var scratch = Scratch(values.Length);
        var backend = collectives.Device.Backend;
        backend.Upload(values, scratch.Storage);
        collectives.Broadcast(scratch.Storage, values.Length, root);
        backend.DownloadRange(scratch.Storage, 0, values);
    }

    /// <summary>
    /// Reduces <paramref name="values"/> so this rank's piece (pieces of <paramref name="counts"/> values in rank order) holds
    /// the sum; through the device, every value is summed (each piece complete everywhere).
    /// </summary>
    public void ReduceScatter(Span<float> values, ReadOnlySpan<int> counts, WireType wire)
    {
        if (values.IsEmpty || _host?.TryReduceScatter(values, counts, ReduceOp.Sum, wire) == true)
        {
            return;
        }

        AllReduce(values, wire);
    }

    /// <summary>Every rank's piece of <paramref name="values"/> (pieces of <paramref name="counts"/> values in rank order) on every rank.</summary>
    public void AllGather(Span<float> values, ReadOnlySpan<int> counts)
    {
        if (values.IsEmpty || _host?.TryAllGather(values, counts) == true)
        {
            return;
        }

        int at = 0;
        for (int r = 0; r < counts.Length; r++)
        {
            if (counts[r] > 0)
            {
                Broadcast(values.Slice(at, counts[r]), r);
            }

            at += counts[r];
        }
    }

    public void Dispose()
    {
        _scratch?.Dispose();
        _scratch = null;
    }

    private Tensor Scratch(int values)
    {
        if (_scratch is null || _scratch.Size < values)
        {
            _scratch?.Dispose();
            _scratch = Tensor.PersistentZeros([Math.Max(1, values)], collectives.Device);
        }

        return _scratch;
    }
}
