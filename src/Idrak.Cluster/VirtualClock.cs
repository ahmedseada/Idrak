// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Cluster;

/// <summary>
/// The simulated network time of each rank (plan 23): a transfer from rank A to rank B over link L starts when A sends it
/// or when the A-to-B direction of L is free, whichever is later, holds that direction for bytes / bandwidth, and arrives
/// after the profile's latency and jitter; B's clock moves to the arrival when B receives it. A lost message is sent
/// again after another latency. The clock counts network time only, so it is the same on every run of a seed: jitter and
/// loss are drawn from a hash of (seed, sender, receiver, tag, message number), never from the order threads ran in.
/// A step's predicted time is its measured compute plus this.
/// </summary>
public sealed class VirtualClock
{
    private const int MaxResends = 64;

    private readonly Lock _gate = new();
    private readonly double[] _now;
    private readonly Dictionary<(int From, int To), double> _free = [];
    private readonly Func<int, int, ILinkProfile> _link;
    private readonly int _seed;

    /// <summary>A clock for <paramref name="worldSize"/> ranks whose links are <paramref name="link"/>(a, b).</summary>
    public VirtualClock(int worldSize, Func<int, int, ILinkProfile> link, int seed)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(worldSize, 1);
        ArgumentNullException.ThrowIfNull(link);
        _now = new double[worldSize];
        _link = link;
        _seed = seed;
    }

    /// <summary>The number of ranks.</summary>
    public int WorldSize => _now.Length;

    /// <summary>Rank <paramref name="rank"/>'s simulated time, seconds.</summary>
    public double Now(int rank)
    {
        lock (_gate)
        {
            return _now[rank];
        }
    }

    /// <summary>
    /// Charges a transfer of <paramref name="bytes"/> from <paramref name="from"/> to <paramref name="to"/>, the
    /// <paramref name="sequence"/>-th the sender sent the receiver under <paramref name="tag"/>: the sender's time when it
    /// was sent and the time it arrives, seconds.
    /// </summary>
    public (double Sent, double Arrives) Send(int from, int to, long bytes, int tag, long sequence)
    {
        var link = _link(from, to);
        double bandwidth = link.BandwidthBytesPerSecond;
        if (!(bandwidth > 0))
        {
            throw new InvalidOperationException($"The link between ranks {from} and {to} ('{link.Name}') has no known bandwidth: give the rate in its name (\"{link.Name}-10g\") or measure it with idrak cluster bench.");
        }

        double occupy = bytes / bandwidth;
        double travel = link.Latency.TotalSeconds;
        double jitter = link.Jitter.TotalSeconds * Uniform(from, to, tag, sequence, 0);
        int resends = 0;
        while (link.LossRate > 0 && resends < MaxResends && Uniform(from, to, tag, sequence, 1 + resends) < link.LossRate)
        {
            resends++;
        }

        lock (_gate)
        {
            double sent = _now[from];
            double start = Math.Max(sent, _free.GetValueOrDefault((from, to)));
            double busy = occupy * (1 + resends);                     // a lost message holds the link again
            _free[(from, to)] = start + busy;
            return (sent, start + busy + travel * (1 + resends) + jitter);
        }
    }

    /// <summary>Moves <paramref name="rank"/>'s time to <paramref name="arrives"/> when that is later (a message it received).</summary>
    public void Arrive(int rank, double arrives)
    {
        lock (_gate)
        {
            _now[rank] = Math.Max(_now[rank], arrives);
        }
    }

    /// <summary>Moves every rank's time to the latest (a barrier: nobody leaves before the last arrives).</summary>
    public void Synchronize()
    {
        lock (_gate)
        {
            double latest = _now.Max();
            Array.Fill(_now, latest);
        }
    }

    // A uniform draw in [0, 1) from the message's identity alone (SplitMix64 over the fields).
    private double Uniform(int from, int to, int tag, long sequence, int draw)
    {
        ulong x = (ulong)(uint)_seed;
        foreach (long field in (ReadOnlySpan<long>)[from, to, tag, sequence, draw])
        {
            x = Mix(x ^ (ulong)field);
        }

        return (x >> 11) * (1.0 / (1UL << 53));
    }

    private static ulong Mix(ulong z)
    {
        z += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
}
