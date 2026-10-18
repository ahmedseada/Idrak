// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Cluster;

/// <summary>
/// The simulated network time of each rank (plan 23): a transfer from rank A to rank B over link L starts when A sends it
/// or when the A-to-B direction of L is free, whichever is later, holds that direction for bytes / bandwidth, and arrives
/// after the profile's latency and jitter; B's clock moves to the arrival when B receives it. A lost message is sent
/// again after another latency. The clock counts network time only, so it is the same on every run of a seed: jitter and
/// loss are drawn from a hash of (seed, sender, receiver, tag, message number), never from the order threads ran in.
/// A step's predicted time is its measured compute plus this. In a simulated run with faults (phase C3) a slow worker's
/// transfers are stretched, a lossy link loses more, and a partition holds messages across it; the clock also keeps, per
/// rank, the rank whose delay last held it back (the critical path's origin), so a slow worker is named on every rank.
/// </summary>
public sealed class VirtualClock
{
    private const int MaxResends = 64;

    private readonly Lock _gate = new();
    private readonly double[] _now;
    private readonly int[] _blame;
    private readonly Dictionary<(int From, int To), double> _free = [];
    private readonly Func<int, int, ILinkProfile> _link;
    private readonly int _seed;

    /// <summary>A clock for <paramref name="worldSize"/> ranks whose links are <paramref name="link"/>(a, b).</summary>
    public VirtualClock(int worldSize, Func<int, int, ILinkProfile> link, int seed)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(worldSize, 1);
        ArgumentNullException.ThrowIfNull(link);
        _now = new double[worldSize];
        _blame = new int[worldSize];
        for (int r = 0; r < worldSize; r++)
        {
            _blame[r] = r;
        }

        _link = link;
        _seed = seed;
    }

    /// <summary>The number of ranks.</summary>
    public int WorldSize => _now.Length;

    // The faults a simulated run injects into its transfers, or null.
    internal FaultInjector? Faults { get; init; }

    /// <summary>Rank <paramref name="rank"/>'s simulated time, seconds.</summary>
    public double Now(int rank)
    {
        lock (_gate)
        {
            return _now[rank];
        }
    }

    /// <summary>The rank whose delay last held <paramref name="rank"/>'s clock back (itself when nothing did).</summary>
    public int CriticalRank(int rank)
    {
        lock (_gate)
        {
            return _blame[rank];
        }
    }

    /// <summary>
    /// Charges a transfer of <paramref name="bytes"/> from <paramref name="from"/> to <paramref name="to"/>, the
    /// <paramref name="sequence"/>-th the sender sent the receiver under <paramref name="tag"/>: the sender's time when it
    /// was sent and the time it arrives, seconds.
    /// </summary>
    public (double Sent, double Arrives) Send(int from, int to, long bytes, int tag, long sequence)
    {
        var charged = Charge(from, to, bytes, tag, sequence);
        return (charged.Sent, charged.Arrives);
    }

    // The transfer charged, with the resends a lost message cost, the rank its arrival time is owed to, and the fault (a
    // partition) that holds it; Arrives is +infinity when a partition never lets it through.
    internal Charged Charge(int from, int to, long bytes, int tag, long sequence)
    {
        var link = _link(from, to);
        double bandwidth = link.BandwidthBytesPerSecond;
        if (!(bandwidth > 0))
        {
            throw new InvalidOperationException($"The link between ranks {from} and {to} ('{link.Name}') has no known bandwidth: give the rate in its name (\"{link.Name}-10g\") or measure it with idrak cluster bench.");
        }

        double jitter = link.Jitter.TotalSeconds * Uniform(from, to, tag, sequence, 0);
        lock (_gate)
        {
            double sent = _now[from];
            var effect = Faults?.Effect(from, to, sent) ?? FaultEffect.None;
            double loss = effect.Loss > 0 ? 1 - ((1 - link.LossRate) * (1 - effect.Loss)) : link.LossRate;   // no fault: the profile's own, exactly
            int resends = 0;
            while (loss > 0 && resends < MaxResends && Uniform(from, to, tag, sequence, 1 + resends) < loss)
            {
                resends++;
            }

            double occupy = bytes / bandwidth * effect.Factor;
            double travel = link.Latency.TotalSeconds * effect.Factor;
            double start = Math.Max(sent, _free.GetValueOrDefault((from, to)));
            double busy = occupy * (1 + resends);                     // a lost message holds the link again
            _free[(from, to)] = start + busy;
            double arrives = Math.Max(start + busy + (travel * (1 + resends)) + jitter, effect.HoldUntil);
            return new Charged(sent, arrives, resends, effect.Factor > 1 ? from : _blame[from], effect.Partition);
        }
    }

    /// <summary>Moves <paramref name="rank"/>'s time to <paramref name="arrives"/> when that is later (a message it received).</summary>
    public void Arrive(int rank, double arrives) => Arrive(rank, arrives, -1);

    // A received message: when it moves the receiver's time, the rank it is owed to is the receiver's too.
    internal void Arrive(int rank, double arrives, int owedTo)
    {
        lock (_gate)
        {
            if (arrives > _now[rank])
            {
                _now[rank] = arrives;
                if (owedTo >= 0)
                {
                    _blame[rank] = owedTo;
                }
            }
        }
    }

    /// <summary>Moves every rank's time to the latest (a barrier: nobody leaves before the last arrives).</summary>
    public void Synchronize()
    {
        lock (_gate)
        {
            int latest = Array.IndexOf(_now, _now.Max());
            Array.Fill(_now, _now[latest]);
            Array.Fill(_blame, _blame[latest]);
        }
    }

    /// <summary>Moves the time of every rank of <paramref name="ranks"/> to the latest among them (a barrier of those ranks).</summary>
    public void Synchronize(ReadOnlySpan<int> ranks)
    {
        if (ranks.IsEmpty)
        {
            return;
        }

        lock (_gate)
        {
            int latest = ranks[0];
            foreach (int r in ranks)
            {
                if (_now[r] > _now[latest])
                {
                    latest = r;
                }
            }

            foreach (int r in ranks)
            {
                (_now[r], _blame[r]) = (_now[latest], _blame[latest]);
            }
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

    internal static ulong Mix(ulong z)
    {
        z += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
}

// One charged transfer (see VirtualClock.Charge).
internal readonly record struct Charged(double Sent, double Arrives, int Resends, int OwedTo, Fault? Partition);

// What the faults active on a sender do to one of its transfers.
internal readonly record struct FaultEffect(double Factor, double Loss, double HoldUntil, Fault? Partition)
{
    public static FaultEffect None { get; } = new(1, 0, 0, null);
}
