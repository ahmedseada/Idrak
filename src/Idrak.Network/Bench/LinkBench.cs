// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Binary;
using System.Diagnostics;

namespace Idrak.Networking.Bench;

/// <summary>What <see cref="LinkBench.Run"/> measures with.</summary>
public sealed record LinkBenchOptions
{
    /// <summary>The message the bandwidth is measured with, in bytes (<c>idrak cluster bench --bytes</c>).</summary>
    public long Bytes { get; init; } = 16L << 20;

    /// <summary>The warm rounds measured after the cold one, 1 or more (<c>--rounds</c>).</summary>
    public int Rounds { get; init; } = 5;
}

/// <summary>One pair's link as measured: the first (cold) round apart from the median of the later (warm) ones (rule 76).</summary>
/// <param name="RankA">The lower rank, which timed the pair.</param>
/// <param name="RankB">The higher rank.</param>
/// <param name="Transport">The transport joining them.</param>
/// <param name="Bytes">The message the bandwidth was measured with.</param>
/// <param name="Rounds">The warm rounds.</param>
/// <param name="ColdBandwidth">Bytes a second of the first round (A to B, until B's answer).</param>
/// <param name="WarmBandwidth">Bytes a second, the median of the warm rounds.</param>
/// <param name="ColdLatency">Half the first round trip of an empty message.</param>
/// <param name="WarmLatency">Half the round trip, the median of the warm rounds.</param>
public sealed record LinkBenchRow(int RankA, int RankB, string Transport, long Bytes, int Rounds, double ColdBandwidth, double WarmBandwidth,
    TimeSpan ColdLatency, TimeSpan WarmLatency);

/// <summary>
/// The pairs between two nodes measured at once (<see cref="LinkBench.RunConcurrent"/>): together they moved this many
/// bytes a second; against one pair alone it shows whether the machines' ranks share one network adapter.
/// </summary>
/// <param name="RankA">The first pair's sender (on the first node).</param>
/// <param name="RankB">The first pair's receiver (on the second node).</param>
/// <param name="Pairs">The pairs that sent at once.</param>
/// <param name="Transport">The transport joining the first pair.</param>
/// <param name="Bytes">The message each pair sent.</param>
/// <param name="Rounds">The warm rounds.</param>
/// <param name="ColdBandwidth">Bytes a second of every pair together in the first round (their bytes over the slowest pair's time).</param>
/// <param name="WarmBandwidth">The same over the warm rounds (each pair's median).</param>
public sealed record LinkBenchConcurrentRow(int RankA, int RankB, int Pairs, string Transport, long Bytes, int Rounds, double ColdBandwidth, double WarmBandwidth);

/// <summary>How far a bench is.</summary>
/// <param name="Done">The pairs measured.</param>
/// <param name="Total">Every pair.</param>
/// <param name="RankA">The pair just measured: its lower rank.</param>
/// <param name="RankB">Its higher rank.</param>
/// <param name="Elapsed">The time so far.</param>
public readonly record struct LinkBenchProgress(int Done, int Total, int RankA, int RankB, TimeSpan Elapsed);

/// <summary>
/// The bandwidth and latency of every pair of a connected group (plan 23, phase C5; <c>idrak cluster bench</c>), the input
/// a calibration needs. Pairs are measured one at a time (every rank meets at a barrier before each, so no other pair
/// shares the links), the lower rank timing: half the round trip of an empty message for the latency; a message of
/// <see cref="LinkBenchOptions.Bytes"/> and an empty answer for the bandwidth. The first round of each is reported apart
/// from the median of the rest (cold and warm, rule 76). Every rank calls it; rank 0 returns every pair's row, the others
/// none. One pooled buffer of the message size per rank, returned at the end.
/// </summary>
public static class LinkBench
{
    private const int PingTag = -101, DataTag = -102, AckTag = -103, RowTag = -104, TogetherTag = -105, TogetherAckTag = -106, TogetherRowTag = -107;
    private const int RowBytes = 4 * sizeof(double);

    /// <summary>Measures every pair of <paramref name="transport"/>'s group; rank 0 gets the rows.</summary>
    /// <param name="transport">This rank's end of the group.</param>
    /// <param name="options">The message size and the rounds.</param>
    /// <param name="transportOf">The transport joining a pair, for the rows (a group over several); null: the transport's name.</param>
    /// <param name="progress">Told on rank 0 after each pair.</param>
    /// <exception cref="ArgumentOutOfRangeException">Bytes below 1 or past what one message holds, or rounds below 1.</exception>
    public static IReadOnlyList<LinkBenchRow> Run(ITransport transport, LinkBenchOptions? options = null, Func<int, int, string>? transportOf = null,
        IProgress<LinkBenchProgress>? progress = null)
    {
        ArgumentNullException.ThrowIfNull(transport);
        options ??= new LinkBenchOptions();
        ArgumentOutOfRangeException.ThrowIfLessThan(options.Bytes, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(options.Bytes, Array.MaxLength);
        ArgumentOutOfRangeException.ThrowIfLessThan(options.Rounds, 1);
        int me = transport.Rank, world = transport.WorldSize, bytes = (int)options.Bytes, rounds = options.Rounds;
        int total = world * (world - 1) / 2, done = 0;
        var rows = new List<LinkBenchRow>(me == 0 ? total : 0);
        var buffer = ArrayPool<byte>.Shared.Rent(bytes);
        var times = new double[rounds + 1];
        Span<byte> row = stackalloc byte[RowBytes];
        long start = Stopwatch.GetTimestamp();
        try
        {
            buffer.AsSpan(0, bytes).Fill(0x5A);
            for (int a = 0; a < world; a++)
            {
                for (int b = a + 1; b < world; b++)
                {
                    transport.Barrier();                         // one pair at a time: nothing else on the links
                    if (me == a)
                    {
                        var (coldLatency, warmLatency) = Time(times, () =>
                        {
                            transport.Send(b, [], PingTag);
                            transport.Receive(b, [], PingTag);
                        }, half: true);
                        var (coldSeconds, warmSeconds) = Time(times, () =>
                        {
                            transport.Send(b, buffer.AsSpan(0, bytes), DataTag);
                            transport.Receive(b, [], AckTag);
                        }, half: false);
                        BinaryPrimitives.WriteDoubleLittleEndian(row, bytes / coldSeconds);
                        BinaryPrimitives.WriteDoubleLittleEndian(row[8..], bytes / warmSeconds);
                        BinaryPrimitives.WriteDoubleLittleEndian(row[16..], coldLatency);
                        BinaryPrimitives.WriteDoubleLittleEndian(row[24..], warmLatency);
                        if (me != 0)
                        {
                            transport.Send(0, row, RowTag);
                        }
                    }
                    else if (me == b)
                    {
                        for (int i = 0; i <= rounds; i++)
                        {
                            transport.Receive(a, [], PingTag);
                            transport.Send(a, [], PingTag);
                        }

                        for (int i = 0; i <= rounds; i++)
                        {
                            transport.Receive(a, buffer.AsSpan(0, bytes), DataTag);
                            transport.Send(a, [], AckTag);
                        }
                    }

                    if (me == 0)
                    {
                        if (a != 0)
                        {
                            transport.Receive(a, row, RowTag);
                        }

                        rows.Add(new LinkBenchRow(a, b, transportOf?.Invoke(a, b) ?? transport.Name, bytes, rounds,
                            BinaryPrimitives.ReadDoubleLittleEndian(row), BinaryPrimitives.ReadDoubleLittleEndian(row[8..]),
                            TimeSpan.FromSeconds(BinaryPrimitives.ReadDoubleLittleEndian(row[16..])), TimeSpan.FromSeconds(BinaryPrimitives.ReadDoubleLittleEndian(row[24..]))));
                        progress?.Report(new LinkBenchProgress(++done, total, a, b, Stopwatch.GetElapsedTime(start)));
                    }
                }
            }

            transport.Barrier();
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }

        return rows;
    }

    /// <summary>
    /// Measures, for every two nodes of <paramref name="topology"/> with two or more ranks each, as many pairs between them
    /// as the smaller holds at once (the i-th rank of one with the i-th of the other, every pair sending a message of
    /// <see cref="LinkBenchOptions.Bytes"/> and taking an empty answer, its rounds as <see cref="Run"/>'s): what they move
    /// together against one pair alone shows a network adapter the machines' ranks share (plan 23, phase C8: the input a
    /// calibration fits it from). Every rank calls it; rank 0 returns the rows. One pooled buffer per rank.
    /// </summary>
    public static IReadOnlyList<LinkBenchConcurrentRow> RunConcurrent(ITransport transport, IClusterTopology topology, LinkBenchOptions? options = null,
        Func<int, int, string>? transportOf = null)
    {
        ArgumentNullException.ThrowIfNull(transport);
        ArgumentNullException.ThrowIfNull(topology);
        options ??= new LinkBenchOptions();
        ArgumentOutOfRangeException.ThrowIfLessThan(options.Bytes, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(options.Bytes, Array.MaxLength);
        ArgumentOutOfRangeException.ThrowIfLessThan(options.Rounds, 1);
        int me = transport.Rank, bytes = (int)options.Bytes, rounds = options.Rounds;
        var rows = new List<LinkBenchConcurrentRow>();
        var buffer = ArrayPool<byte>.Shared.Rent(bytes);
        var times = new double[rounds + 1];
        Span<byte> row = stackalloc byte[2 * sizeof(double)];
        try
        {
            buffer.AsSpan(0, bytes).Fill(0x5A);
            for (int na = 0; na < topology.Nodes.Count; na++)
            {
                for (int nb = na + 1; nb < topology.Nodes.Count; nb++)
                {
                    var a = topology.Ranks(na);
                    var b = topology.Ranks(nb);
                    int pairs = Math.Min(a.Count, b.Count);
                    if (pairs < 2)
                    {
                        continue;
                    }

                    transport.Barrier();                         // every pair starts together
                    int sender = IndexOf(a, me, pairs), receiver = IndexOf(b, me, pairs);
                    double cold = 0, warm = 0;
                    if (sender >= 0)
                    {
                        int to = b[sender];
                        (cold, warm) = Time(times, () =>
                        {
                            transport.Send(to, buffer.AsSpan(0, bytes), TogetherTag);
                            transport.Receive(to, [], TogetherAckTag);
                        }, half: false);
                        if (me != 0)
                        {
                            BinaryPrimitives.WriteDoubleLittleEndian(row, cold);
                            BinaryPrimitives.WriteDoubleLittleEndian(row[8..], warm);
                            transport.Send(0, row, TogetherRowTag);
                        }
                    }
                    else if (receiver >= 0)
                    {
                        for (int i = 0; i <= rounds; i++)
                        {
                            transport.Receive(a[receiver], buffer.AsSpan(0, bytes), TogetherTag);
                            transport.Send(a[receiver], [], TogetherAckTag);
                        }
                    }

                    if (me == 0)
                    {
                        double slowestCold = 0, slowestWarm = 0;
                        for (int i = 0; i < pairs; i++)
                        {
                            double c = cold, w = warm;
                            if (a[i] != 0)
                            {
                                transport.Receive(a[i], row, TogetherRowTag);
                                (c, w) = (BinaryPrimitives.ReadDoubleLittleEndian(row), BinaryPrimitives.ReadDoubleLittleEndian(row[8..]));
                            }

                            (slowestCold, slowestWarm) = (Math.Max(slowestCold, c), Math.Max(slowestWarm, w));
                        }

                        rows.Add(new LinkBenchConcurrentRow(a[0], b[0], pairs, transportOf?.Invoke(a[0], b[0]) ?? transport.Name, bytes, rounds,
                            (double)pairs * bytes / slowestCold, (double)pairs * bytes / slowestWarm));
                    }
                }
            }

            transport.Barrier();
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }

        return rows;

        static int IndexOf(IReadOnlyList<int> ranks, int rank, int count)
        {
            for (int i = 0; i < count; i++)
            {
                if (ranks[i] == rank)
                {
                    return i;
                }
            }

            return -1;
        }
    }

    // The first call's seconds and the median of the rest (halved for a round trip's one way).
    private static (double Cold, double Warm) Time(double[] times, Action call, bool half)
    {
        for (int i = 0; i < times.Length; i++)
        {
            long start = Stopwatch.GetTimestamp();
            call();
            times[i] = Math.Max(Stopwatch.GetElapsedTime(start).TotalSeconds, 1e-9) * (half ? 0.5 : 1);
        }

        var warm = times.AsSpan(1);
        warm.Sort();
        return (times[0], warm[warm.Length / 2]);
    }
}
