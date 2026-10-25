// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Binary;
using System.Diagnostics;
using Idrak.Cluster.Abstractions;

namespace Idrak.Cluster;

/// <summary>
/// A token bucket that paces one direction of a real pipe to a link's bandwidth (plan 23, the localhost throttle): bytes
/// earn the right to leave at the link's rate, and a message leaves once the bytes before it have; the bucket holds no
/// more than one message's worth, so nothing bursts past the link. Positions are <see cref="Stopwatch"/> timestamps, which
/// every process of one machine reads from one clock.
/// </summary>
internal sealed class TokenBucket(double bytesPerSecond)
{
    private readonly Lock _gate = new();
    private long _free;                                        // when the link is free again (a timestamp)

    public double BytesPerSecond { get; } = bytesPerSecond;

    /// <summary>Reserves the link for <paramref name="bytes"/> and returns the timestamp their last byte leaves at.</summary>
    public long Reserve(long bytes)
    {
        long cost = (long)(bytes / BytesPerSecond * Stopwatch.Frequency);
        lock (_gate)
        {
            long start = Math.Max(Stopwatch.GetTimestamp(), _free);
            _free = start + cost;
            return _free;
        }
    }

    /// <summary>Waits until <paramref name="timestamp"/>: asleep until the last two milliseconds, spinning for them.</summary>
    public static void WaitUntil(long timestamp)
    {
        while (true)
        {
            long left = timestamp - Stopwatch.GetTimestamp();
            if (left <= 0)
            {
                return;
            }

            double milliseconds = left * 1000.0 / Stopwatch.Frequency;
            if (milliseconds > 2)
            {
                Thread.Sleep((int)(milliseconds - 1));
            }
            else
            {
                Thread.SpinWait(64);
            }
        }
    }
}

/// <summary>
/// A real transport paced to the topology's links (plan 23: localhost mode with <see cref="SimulationOptions.Throttle"/>),
/// so sockets and shared memory on one machine imitate the slower links of a cluster that is not there: each send to a
/// peer whose link declares a bandwidth waits for that pair's <see cref="TokenBucket"/>, and carries the time it would
/// arrive (its departure plus the link's latency) in 8 bytes ahead of its payload; the receiver holds it until then. A
/// pair whose link has no rate (one measured on this machine, or none declared) is not paced and carries nothing extra.
/// </summary>
internal sealed class ThrottledTransport : ITransport
{
    private readonly ITransport _inner;
    private readonly TokenBucket?[] _buckets;
    private readonly long[] _latency;                          // per peer, in Stopwatch ticks

    private ThrottledTransport(ITransport inner, TokenBucket?[] buckets, long[] latency)
    {
        _inner = inner;
        _buckets = buckets;
        _latency = latency;
    }

    public string Name => _inner.Name;

    public int Rank => _inner.Rank;

    public int WorldSize => _inner.WorldSize;

    public TransportCapabilities Capabilities => _inner.Capabilities;

    /// <summary>The pair's paced bandwidth, or null when it is not paced.</summary>
    public double? Bandwidth(int peer) => _buckets[peer]?.BytesPerSecond;

    /// <summary>
    /// <paramref name="inner"/> paced to <paramref name="context"/>'s links, or <paramref name="inner"/> itself when no link
    /// of this rank declares a bandwidth.
    /// </summary>
    public static ITransport Wrap(ITransport inner, ClusterContext context)
    {
        ArgumentNullException.ThrowIfNull(inner);
        ArgumentNullException.ThrowIfNull(context);
        int world = context.Topology.WorldSize;
        var buckets = new TokenBucket?[world];
        var latency = new long[world];
        bool any = false;
        for (int peer = 0; peer < world; peer++)
        {
            if (peer == context.Rank)
            {
                continue;
            }

            var link = context.Topology.Link(context.Rank, peer);
            if (link is IDeviceMeasuredLink || link.BandwidthBytesPerSecond <= 0)
            {
                continue;                                      // this machine's own speed, or no rate declared: nothing to imitate
            }

            buckets[peer] = new TokenBucket(link.BandwidthBytesPerSecond);
            latency[peer] = (long)(link.Latency.TotalSeconds * Stopwatch.Frequency);
            any = true;
        }

        return any ? new ThrottledTransport(inner, buckets, latency) : inner;
    }

    public void Send(int to, ReadOnlySpan<byte> bytes, int tag)
    {
        if (Bucket(to) is not { } bucket)
        {
            _inner.Send(to, bytes, tag);
            return;
        }

        long leaves = bucket.Reserve(bytes.Length);
        TokenBucket.WaitUntil(leaves);
        var framed = ArrayPool<byte>.Shared.Rent(bytes.Length + sizeof(long));
        try
        {
            BinaryPrimitives.WriteInt64LittleEndian(framed, leaves + _latency[to]);
            bytes.CopyTo(framed.AsSpan(sizeof(long)));
            _inner.Send(to, framed.AsSpan(0, bytes.Length + sizeof(long)), tag);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(framed);
        }
    }

    public int Receive(int from, Span<byte> bytes, int tag)
    {
        if (Bucket(from) is null)
        {
            return _inner.Receive(from, bytes, tag);
        }

        var framed = ArrayPool<byte>.Shared.Rent(bytes.Length + sizeof(long));
        try
        {
            int length = _inner.Receive(from, framed.AsSpan(0, bytes.Length + sizeof(long)), tag) - sizeof(long);
            TokenBucket.WaitUntil(BinaryPrimitives.ReadInt64LittleEndian(framed));
            framed.AsSpan(sizeof(long), length).CopyTo(bytes);
            return length;
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(framed);
        }
    }

    public ValueTask SendAsync(int to, ReadOnlyMemory<byte> bytes, int tag, CancellationToken cancellationToken = default)
    {
        if (Bucket(to) is null)
        {
            return _inner.SendAsync(to, bytes, tag, cancellationToken);
        }

        try
        {
            Send(to, bytes.Span, tag);
            return ValueTask.CompletedTask;
        }
        catch (Exception e)
        {
            return ValueTask.FromException(e);
        }
    }

    public ValueTask<int> ReceiveAsync(int from, Memory<byte> bytes, int tag, CancellationToken cancellationToken = default)
    {
        if (Bucket(from) is null)
        {
            return _inner.ReceiveAsync(from, bytes, tag, cancellationToken);
        }

        try
        {
            return ValueTask.FromResult(Receive(from, bytes.Span, tag));
        }
        catch (Exception e)
        {
            return ValueTask.FromException<int>(e);
        }
    }

    public void Barrier() => _inner.Barrier();

    public void Dispose() => _inner.Dispose();

    private TokenBucket? Bucket(int peer)
    {
        if (peer == Rank || (uint)peer >= (uint)WorldSize)
        {
            throw new ArgumentOutOfRangeException(nameof(peer), peer, $"rank {Rank} reaches the other ranks of 0 to {WorldSize - 1}");
        }

        return _buckets[peer];
    }
}
