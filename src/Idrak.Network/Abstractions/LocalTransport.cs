// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading.Channels;

namespace Idrak.Networking.Abstractions;

/// <summary>
/// The ranks of one group inside one process (plan 23, phase C5: devices of one machine driven by one process): they meet
/// by name (<see cref="ClusterOptions.RunId"/>, else the coordinator), each frame is handed to the receiving rank's queue
/// by a call (a copy into a pooled buffer), and device memory is handed over for the receiver to copy
/// (<see cref="IDeviceTransport"/>).
/// </summary>
internal abstract class LocalTransport : FramedTransport, IDeviceTransport
{
    private static readonly ConcurrentDictionary<string, LocalHub> Hubs = new(StringComparer.Ordinal);
    private readonly LocalHub _hub;
    private readonly ConcurrentDictionary<(int From, int Tag), Channel<Handoff>> _handoffs = new();

    protected LocalTransport(string name, LocalHub hub, int rank) : base(name, rank, hub.Members.Length, hub.Timeout) => _hub = hub;

    /// <summary>Joins the group <paramref name="context"/> names, waiting for every rank up to the timeout.</summary>
    internal static T JoinGroup<T>(ClusterContext context, string transport, Func<LocalHub, int, T> create, CancellationToken cancellationToken)
        where T : LocalTransport
    {
        ArgumentNullException.ThrowIfNull(context);
        string name = transport + "|" + (context.Options.RunId ?? context.Options.Coordinator ?? "local");
        var hub = Hubs.GetOrAdd(name, _ => new LocalHub(context.Topology.WorldSize, context.Options.Timeout));
        var me = create(hub, context.Rank);
        try
        {
            hub.Join(me, cancellationToken);
        }
        catch
        {
            me.Dispose();
            throw;
        }
        finally
        {
            if (hub.AllJoined)
            {
                Hubs.TryRemove(new KeyValuePair<string, LocalHub>(name, hub));   // the next group of that name starts afresh
            }
        }

        return me;
    }

    public void SendDevice(int to, Storage source, int count, int tag)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        if (to == Rank || (uint)to >= (uint)WorldSize)
        {
            throw new ArgumentOutOfRangeException(nameof(to), to, $"rank {Rank} reaches the other ranks of 0 to {WorldSize - 1}");
        }

        var peer = _hub.Members[to] ?? throw new InvalidOperationException($"rank {to} is not in rank {Rank}'s group");
        using var handoff = new Handoff(source, count);
        if (!peer.Handoffs(Rank, tag).Writer.TryWrite(handoff))
        {
            throw new InvalidOperationException($"rank {to} left the group before rank {Rank} handed it device memory (tag {tag})");
        }

        if (!handoff.Done.Wait(Timeout))
        {
            throw new TimeoutException($"rank {Rank} waited {Timeout.TotalSeconds:F0} s for rank {to} to take its device memory (tag {tag})");
        }

        if (handoff.Error is { } error)
        {
            throw new InvalidOperationException($"rank {to} could not take rank {Rank}'s device memory (tag {tag}): {error.Message}", error);
        }
    }

    public void ReceiveDevice(int from, Storage destination, int count, int tag)
    {
        ArgumentNullException.ThrowIfNull(destination);
        if (from == Rank || (uint)from >= (uint)WorldSize)
        {
            throw new ArgumentOutOfRangeException(nameof(from), from, $"rank {Rank} reaches the other ranks of 0 to {WorldSize - 1}");
        }

        var reader = Handoffs(from, tag).Reader;
        if (!reader.TryRead(out var handoff))
        {
            using var timeout = new CancellationTokenSource(Timeout);
            try
            {
                handoff = reader.ReadAsync(timeout.Token).AsTask().GetAwaiter().GetResult();
            }
            catch (OperationCanceledException)
            {
                throw new TimeoutException($"rank {Rank} waited {Timeout.TotalSeconds:F0} s for rank {from}'s device memory (tag {tag})");
            }
        }

        try
        {
            if (handoff.Count != count)
            {
                throw new InvalidDataException($"rank {from} handed {handoff.Count} values (tag {tag}) where rank {Rank} takes {count}");
            }

            long start = Telemetry.IsEnabled(TelemetryLevel.Network) ? Stopwatch.GetTimestamp() : 0;
            Copy(from, handoff.Source, destination, count);
            if (start != 0)
            {
                Telemetry.Transfer(new TransferCompleted(Name, from, Rank, tag, (long)count * sizeof(float), Since(start), TimeSpan.Zero));
            }
        }
        catch (Exception e)
        {
            handoff.Error = e;
            throw;
        }
        finally
        {
            handoff.Done.Set();
        }
    }

    /// <summary>Copies <paramref name="count"/> values of rank <paramref name="from"/>'s <paramref name="source"/> into <paramref name="destination"/>.</summary>
    protected abstract void Copy(int from, Storage source, Storage destination, int count);

    /// <summary>A copy down to host memory and up to the destination's device, in one pooled buffer: the path every machine has.</summary>
    protected static void CopyThroughHost(Storage source, Storage destination, int count)
    {
        if (count == 0)
        {
            return;
        }

        var host = ArrayPool<float>.Shared.Rent(count);
        try
        {
            source.Backend.DownloadRange(source, 0, host.AsSpan(0, count));
            destination.Backend.Upload(host.AsSpan(0, count), destination);
            destination.Backend.Synchronize();
        }
        finally
        {
            ArrayPool<float>.Shared.Return(host);
        }
    }

    /// <summary>The group this rank is in.</summary>
    protected LocalHub Hub => _hub;

    protected override void WriteFrame(int to, in WireHeader header, ReadOnlySpan<byte> payload)
    {
        var peer = _hub.Members[to] ?? throw new InvalidOperationException($"rank {to} is not in rank {Rank}'s group");
        var buffer = ArrayPool<byte>.Shared.Rent(Math.Max(1, payload.Length));
        payload.CopyTo(buffer);
        peer.Arrived(Rank, header, buffer, payload.Length, TimeSpan.Zero);
    }

    protected override void CloseCarriers()
    {
        foreach (var (_, channel) in _handoffs)
        {
            channel.Writer.TryComplete();
        }
    }

    private Channel<Handoff> Handoffs(int from, int tag) =>
        _handoffs.GetOrAdd((from, tag), static _ => Channel.CreateUnbounded<Handoff>(new UnboundedChannelOptions { SingleReader = false, SingleWriter = false }));

    /// <summary>One rank's device memory handed to another, and the receiver's word that it was copied.</summary>
    private sealed class Handoff(Storage source, int count) : IDisposable
    {
        public Storage Source { get; } = source;

        public int Count { get; } = count;

        public ManualResetEventSlim Done { get; } = new(false);

        public Exception? Error { get; set; }

        public void Dispose() => Done.Dispose();
    }
}

/// <summary>The ranks of one in-process group, and the pairs whose fallback was already reported.</summary>
internal sealed class LocalHub(int world, TimeSpan timeout)
{
    private readonly object _gate = new();
    private int _joined;

    public LocalTransport?[] Members { get; } = new LocalTransport?[world];

    public TimeSpan Timeout { get; } = timeout;

    public ConcurrentDictionary<(int, int), bool> Reported { get; } = new();

    public bool AllJoined
    {
        get
        {
            lock (_gate)
            {
                return _joined == Members.Length;
            }
        }
    }

    public void Join(LocalTransport rank, CancellationToken cancellationToken)
    {
        lock (_gate)
        {
            if (Members[rank.Rank] is not null)
            {
                throw new InvalidOperationException($"Rank {rank.Rank} joined the in-process group twice.");
            }

            Members[rank.Rank] = rank;
            _joined++;
            Monitor.PulseAll(_gate);
            long deadline = Stopwatch.GetTimestamp() + (long)(Timeout.TotalSeconds * Stopwatch.Frequency);
            while (_joined < Members.Length)
            {
                cancellationToken.ThrowIfCancellationRequested();
                long left = deadline - Stopwatch.GetTimestamp();
                if (left <= 0)
                {
                    Members[rank.Rank] = null;
                    _joined--;
                    var missing = Enumerable.Range(0, Members.Length).Where(r => Members[r] is null);
                    throw new TimeoutException($"rank {rank.Rank} waited {Timeout.TotalSeconds:F0} s for rank(s) {string.Join(", ", missing)} to join");
                }

                Monitor.Wait(_gate, TimeSpan.FromSeconds(Math.Min(1, left / (double)Stopwatch.Frequency)));
            }
        }
    }
}
