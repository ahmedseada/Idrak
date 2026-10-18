// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading.Channels;

namespace Idrak.Cluster;

/// <summary>
/// The "sim" transport (plan 23): ranks that are threads of one process, joined by bounded queues (one per sender,
/// receiver and tag, holding <see cref="ClusterOptions.InFlight"/> messages, so a rank that races ahead waits as it would
/// on a socket), every transfer charged on a <see cref="VirtualClock"/> from the topology's link profiles, with the faults
/// of the run's failure model injected (phase C3). Ranks rendezvous by <see cref="ClusterOptions.Coordinator"/> as a name
/// (any text: the in-process cluster gives each run its own), as network ranks do at an address.
/// </summary>
internal sealed class SimTransportProvider : ITransportProvider
{
    private static readonly ConcurrentDictionary<string, SimHub> Rendezvous = new(StringComparer.Ordinal);

    public string Name => "sim";

    public string Summary => "ranks inside one process, every transfer charged on a virtual clock from the topology's link profiles";

    public TransportCapabilities Capabilities => new(0, TimeSpan.Zero, ZeroCopy: false, GpuDirect: false, Reliable: true, CrossProcess: false, CrossMachine: false);

    public TransportAvailability Availability(ClusterContext context, int peer) =>
        context.Options.Simulation.Mode == SimulationMode.InProcess
            ? new(Name, true, "both ranks are threads of this process (a simulated cluster)")
            : new(Name, false, "sim joins the ranks of one process only (IDRAK_CLUSTER_MODE=in-process)");

    public ITransport Connect(ClusterContext context, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(context);
        string name = context.Options.Coordinator ?? "sim";
        var hub = Rendezvous.GetOrAdd(name, _ => new SimWorld(context).Root);
        try
        {
            hub.Join(context, cancellationToken);
        }
        finally
        {
            if (hub.AllJoined)
            {
                Rendezvous.TryRemove(new KeyValuePair<string, SimHub>(name, hub));   // the next group under this name starts afresh
            }
        }

        return new SimTransport(hub, context.Rank);
    }
}

// What a simulated run's ranks share: the clock, the fault injector, the event log, the groups its ranks form (every rank;
// fewer while a late joiner is away), and whether a fault broke it.
internal sealed class SimWorld
{
    private readonly Lock _gate = new();
    private readonly Dictionary<string, SimHub> _groups = new(StringComparer.Ordinal);
    private volatile Exception? _broken;

    public SimWorld(ClusterContext context)
    {
        Topology = context.Topology;
        Options = context.Options;
        World = Topology.WorldSize;
        Timeout = Options.Timeout;
        Capacity = Math.Max(1, Options.InFlight);
        Seed = Options.Simulation.Seed ?? Topology.Simulation.Seed;
        string? carveFrom = Options.Simulation.CarveFrom;
        var devices = Enumerable.Range(0, World).Select(r =>
        {
            var (node, local) = Topology.Place(r);
            return carveFrom ?? Topology.Nodes[node].Devices[local].Device;
        }).ToArray();
        var model = FailureModels.Get(Options.Simulation.Failures ?? Topology.Simulation.Failures);
        Log = new EventLogWriter(Seed, World, model.Name, Options.Simulation.LogTransfers, Options.Simulation.EventLog, Options.Simulation.EventLogFlushSteps,
            (Options.Suggest == true, Options.SuggestSteps, Options.SuggestSliceValues));   // the advisor's setting, which a replay takes (C8c)
        Control = RunControls.Find(Options.Coordinator);               // a run steered live (phase C9), or none
        Faults = new FaultInjector(model, Topology, Seed, Timeout, Log, Break, Control);
        Clock = new VirtualClock(World, (a, b) => MeasuredLink.Resolve(Topology.Link(a, b), devices[a]), Seed) { Faults = Faults };
        Root = new SimHub(this, [.. Enumerable.Range(0, World)], Topology);
    }

    public IClusterTopology Topology { get; }

    public ClusterOptions Options { get; }

    public int World { get; }

    public TimeSpan Timeout { get; }

    public int Capacity { get; }

    public int Seed { get; }

    public VirtualClock Clock { get; }

    public FaultInjector Faults { get; }

    // The control steering this run (pause, resume, stop, live faults), or null.
    public ClusterRunControl? Control { get; }

    public EventLogWriter Log { get; }

    public SimHub Root { get; }

    // The fault that ended the run for every rank, or null.
    public Exception? Broken => _broken;

    // Ends the run on every rank: every queue closes and every wait wakes to `reason`.
    public void Break(Exception reason)
    {
        SimHub[] hubs;
        lock (_gate)
        {
            if (_broken is not null)
            {
                return;
            }

            _broken = reason;
            hubs = [Root, .. _groups.Values];
        }

        foreach (var hub in hubs)
        {
            hub.Close();
        }
    }

    // The break as seen by another rank: the same kind of exception, the same message.
    public Exception Failure() => _broken switch
    {
        TimeoutException t => new TimeoutException(t.Message, t),
        { } e => new InvalidOperationException(e.Message, e),
        null => new InvalidOperationException("the simulated group is not broken"),
    };

    // The group of `members` (ranks in order): every rank's, or one of its own over the same clock.
    public SimHub GroupOf(int[] members)
    {
        if (members.Length == World)
        {
            return Root;
        }

        lock (_gate)
        {
            string key = string.Join(',', members);
            if (!_groups.TryGetValue(key, out var hub))
            {
                hub = new SimHub(this, members, new GroupTopology(Topology, members));
                _groups[key] = hub;
            }

            return hub;
        }
    }
}

// A group's queues and barrier: `Real` maps its ranks (0 to Members - 1) to the run's, whose clock it charges.
internal sealed class SimHub
{
    private readonly ConcurrentDictionary<(int From, int To, int Tag), Queue> _queues = new();
    private readonly object _gate = new();                     // a monitor: ranks wait on it to join and at barriers
    private readonly int[] _sub;
    private readonly bool[] _joined;
    private readonly bool[] _left;
    private readonly bool[] _atBarrier;
    private int _joinedCount;
    private int _barrierCount;
    private long _barrierGeneration;

    public SimHub(SimWorld world, int[] real, IClusterTopology topology)
    {
        World = world;
        Real = real;
        Topology = topology;
        _sub = new int[world.World];
        Array.Fill(_sub, -1);
        for (int i = 0; i < real.Length; i++)
        {
            _sub[real[i]] = i;
        }

        _joined = new bool[real.Length];
        _left = new bool[real.Length];
        _atBarrier = new bool[real.Length];
        if (real.Length != world.World)
        {
            Array.Fill(_joined, true);                         // a group formed during a run: its ranks are all there
            _joinedCount = real.Length;
        }
    }

    public SimWorld World { get; }

    public int[] Real { get; }

    public IClusterTopology Topology { get; }

    public int Members => Real.Length;

    public VirtualClock Clock => World.Clock;

    public bool AllJoined
    {
        get
        {
            lock (_gate)
            {
                return _joinedCount == Members;
            }
        }
    }

    // This group's rank for the run's rank `real`, or null when it is not in the group.
    public int? SubOf(int real) => (uint)real < (uint)_sub.Length && _sub[real] >= 0 ? _sub[real] : null;

    public void Join(ClusterContext context, CancellationToken cancellationToken)
    {
        int rank = context.Rank;
        lock (_gate)
        {
            if (context.Topology.WorldSize != Members)
            {
                throw new InvalidOperationException($"Rank {rank} joined a group of {Members} ranks with a topology of {context.Topology.WorldSize}.");
            }

            if ((uint)rank >= (uint)Members || _joined[rank])
            {
                throw new InvalidOperationException($"Rank {rank} joined twice, or is outside the group of {Members} ranks.");
            }

            _joined[rank] = true;
            _joinedCount++;
            Monitor.PulseAll(_gate);
            long deadline = Stopwatch.GetTimestamp() + (long)(World.Timeout.TotalSeconds * Stopwatch.Frequency);
            while (_joinedCount < Members)
            {
                cancellationToken.ThrowIfCancellationRequested();
                long left = deadline - Stopwatch.GetTimestamp();
                if (left <= 0)
                {
                    var missing = Enumerable.Range(0, Members).Where(r => !_joined[r]);
                    throw new TimeoutException($"rank {rank} waited {World.Timeout.TotalSeconds:F0} s for rank(s) {string.Join(", ", missing)} to join");
                }

                Monitor.Wait(_gate, TimeSpan.FromSeconds(Math.Min(1, left / (double)Stopwatch.Frequency)));
            }
        }
    }

    public Queue QueueOf(int from, int to, int tag) => _queues.GetOrAdd((from, to, tag), static (key, hub) =>
    {
        var queue = new Queue(hub.World.Capacity);
        if (hub.HasLeft(key.From) || hub.World.Broken is not null)
        {
            queue.Channel.Writer.TryComplete();
        }

        return queue;
    }, this);

    public bool HasLeft(int rank)
    {
        lock (_gate)
        {
            return _left[rank];
        }
    }

    public void Leave(int rank)
    {
        lock (_gate)
        {
            _left[rank] = true;
            Monitor.PulseAll(_gate);
        }

        foreach (var ((from, _, _), queue) in _queues)
        {
            if (from == rank)
            {
                queue.Channel.Writer.TryComplete();
            }
        }
    }

    // A broken run: every queue closes and every rank waiting here wakes.
    public void Close()
    {
        foreach (var (_, queue) in _queues)
        {
            queue.Channel.Writer.TryComplete();
        }

        lock (_gate)
        {
            Monitor.PulseAll(_gate);
        }
    }

    public void Barrier(int rank)
    {
        lock (_gate)
        {
            if (World.Broken is not null)
            {
                throw World.Failure();
            }

            long generation = _barrierGeneration;
            _atBarrier[rank] = true;
            if (++_barrierCount == Members)
            {
                Clock.Synchronize(Real);
                Array.Clear(_atBarrier);
                _barrierCount = 0;
                _barrierGeneration++;
                Monitor.PulseAll(_gate);
                return;
            }

            long deadline = Stopwatch.GetTimestamp() + (long)(World.Timeout.TotalSeconds * Stopwatch.Frequency);
            while (_barrierGeneration == generation)
            {
                var gone = Enumerable.Range(0, Members).Where(r => _left[r] && !_atBarrier[r]).ToList();
                long left = deadline - Stopwatch.GetTimestamp();
                if (World.Broken is not null || gone.Count > 0 || left <= 0)
                {
                    var missing = Enumerable.Range(0, Members).Where(r => !_atBarrier[r]);
                    _atBarrier[rank] = false;
                    _barrierCount--;
                    throw World.Broken is not null ? World.Failure()
                        : gone.Count > 0 ? new InvalidOperationException($"rank {rank} is at a barrier rank(s) {string.Join(", ", gone)} left before reaching")
                        : new TimeoutException($"rank {rank} waited {World.Timeout.TotalSeconds:F0} s at a barrier for rank(s) {string.Join(", ", missing)}");
                }

                Monitor.Wait(_gate, TimeSpan.FromSeconds(Math.Min(1, left / (double)Stopwatch.Frequency)));
            }
        }
    }

    // One sender's messages to one receiver under one tag: bounded, numbered in send order.
    internal sealed class Queue(int capacity)
    {
        public Channel<SimMessage> Channel { get; } = System.Threading.Channels.Channel.CreateBounded<SimMessage>(
            new BoundedChannelOptions(capacity) { FullMode = BoundedChannelFullMode.Wait, SingleReader = true, SingleWriter = false });

        public Lock Gate { get; } = new();

        public long Sent { get; set; }
    }
}

// A message in flight: its bytes (a pooled buffer the receiver returns), when it was sent here, its virtual times, and
// the run's rank its arrival time is owed to.
internal readonly record struct SimMessage(byte[] Buffer, int Length, long SentTimestamp, double SentAt, double Arrives, int OwedTo);

internal sealed class SimTransport(SimHub hub, int rank) : ITransport
{
    private bool _disposed;

    public string Name => "sim";

    public int Rank { get; } = rank;

    public int WorldSize => hub.Members;

    public TransportCapabilities Capabilities => new(0, TimeSpan.Zero, ZeroCopy: false, GpuDirect: false, Reliable: true, CrossProcess: false, CrossMachine: false);

    public TimeSpan SimulatedTime => TimeSpan.FromSeconds(hub.Clock.Now(Real));

    public bool Simulated => true;

    public int? SimulatedCriticalRank => hub.SubOf(hub.Clock.CriticalRank(Real));

    // This rank in the run's numbering.
    public int Real => hub.Real[Rank];

    public SimHub Hub => hub;

    public void Send(int to, ReadOnlySpan<byte> bytes, int tag)
    {
        var queue = Prepare(to, tag);
        var buffer = ArrayPool<byte>.Shared.Rent(Math.Max(1, bytes.Length));
        bytes.CopyTo(buffer);
        lock (queue.Gate)                                      // the message number and the queue order agree
        {
            var message = Charge(to, tag, queue, buffer, bytes.Length);
            if (!queue.Channel.Writer.TryWrite(message))
            {
                using var timeout = new CancellationTokenSource(hub.World.Timeout);
                try
                {
                    queue.Channel.Writer.WriteAsync(message, timeout.Token).AsTask().GetAwaiter().GetResult();
                }
                catch (OperationCanceledException)
                {
                    ArrayPool<byte>.Shared.Return(buffer);
                    throw new TimeoutException($"rank {Rank} waited {hub.World.Timeout.TotalSeconds:F0} s to send to rank {to} (tag {tag}): its {hub.World.Capacity} messages in flight were not received");
                }
                catch (ChannelClosedException)
                {
                    ArrayPool<byte>.Shared.Return(buffer);
                    throw hub.World.Broken is not null ? hub.World.Failure() : new InvalidOperationException($"rank {Rank} sent to rank {to} after the group closed");
                }
            }
        }
    }

    public int Receive(int from, Span<byte> bytes, int tag)
    {
        var reader = Check(from, tag);
        if (!reader.TryRead(out var message))
        {
            using var timeout = new CancellationTokenSource(hub.World.Timeout);
            try
            {
                message = reader.ReadAsync(timeout.Token).AsTask().GetAwaiter().GetResult();
            }
            catch (OperationCanceledException)
            {
                throw new TimeoutException($"rank {Rank} waited {hub.World.Timeout.TotalSeconds:F0} s for rank {from} (tag {tag})");
            }
            catch (ChannelClosedException)
            {
                throw Closed(from, tag);
            }
        }

        return Deliver(from, tag, message, bytes);
    }

    public async ValueTask SendAsync(int to, ReadOnlyMemory<byte> bytes, int tag, CancellationToken cancellationToken = default)
    {
        var queue = Prepare(to, tag);
        var buffer = ArrayPool<byte>.Shared.Rent(Math.Max(1, bytes.Length));
        bytes.Span.CopyTo(buffer);
        SimMessage message;
        lock (queue.Gate)
        {
            message = Charge(to, tag, queue, buffer, bytes.Length);
            if (queue.Channel.Writer.TryWrite(message))
            {
                return;
            }
        }

        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(hub.World.Timeout);
        try
        {
            await queue.Channel.Writer.WriteAsync(message, timeout.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            ArrayPool<byte>.Shared.Return(buffer);
            throw new TimeoutException($"rank {Rank} waited {hub.World.Timeout.TotalSeconds:F0} s to send to rank {to} (tag {tag}): its {hub.World.Capacity} messages in flight were not received");
        }
        catch (ChannelClosedException)
        {
            ArrayPool<byte>.Shared.Return(buffer);
            throw hub.World.Broken is not null ? hub.World.Failure() : new InvalidOperationException($"rank {Rank} sent to rank {to} after the group closed");
        }
    }

    public async ValueTask<int> ReceiveAsync(int from, Memory<byte> bytes, int tag, CancellationToken cancellationToken = default)
    {
        var reader = Check(from, tag);
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(hub.World.Timeout);
        SimMessage message;
        try
        {
            message = await reader.ReadAsync(timeout.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            throw new TimeoutException($"rank {Rank} waited {hub.World.Timeout.TotalSeconds:F0} s for rank {from} (tag {tag})");
        }
        catch (ChannelClosedException)
        {
            throw Closed(from, tag);
        }

        return Deliver(from, tag, message, bytes.Span);
    }

    public void Barrier()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        hub.Barrier(Rank);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            hub.Leave(Rank);
        }
    }

    private SimHub.Queue Prepare(int to, int tag)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (to == Rank || (uint)to >= (uint)WorldSize)
        {
            throw new ArgumentOutOfRangeException(nameof(to), to, $"rank {Rank} can send to the other ranks of 0 to {WorldSize - 1}");
        }

        if (hub.World.Broken is not null)
        {
            throw hub.World.Failure();
        }

        return hub.QueueOf(Rank, to, tag);
    }

    // Charges the message on the clock and records it; a partition that never opens ends the run naming both sides.
    private SimMessage Charge(int to, int tag, SimHub.Queue queue, byte[] buffer, int length)
    {
        int real = Real, realTo = hub.Real[to];
        long sequence = queue.Sent++;
        var charged = hub.Clock.Charge(real, realTo, length, tag, sequence);
        hub.World.Log.Transfer(real, realTo, tag, sequence, length, charged);
        if (double.IsPositiveInfinity(charged.Arrives))
        {
            ArrayPool<byte>.Shared.Return(buffer);
            var timeout = hub.World.Faults.PartitionTimeout(real, realTo, tag, charged.Partition!.Value);
            hub.World.Break(timeout);
            throw timeout;
        }

        return new SimMessage(buffer, length, Stopwatch.GetTimestamp(), charged.Sent, charged.Arrives, charged.OwedTo);
    }

    private ChannelReader<SimMessage> Check(int from, int tag)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (from == Rank || (uint)from >= (uint)WorldSize)
        {
            throw new ArgumentOutOfRangeException(nameof(from), from, $"rank {Rank} receives from the other ranks of 0 to {WorldSize - 1}");
        }

        var reader = hub.QueueOf(from, Rank, tag).Channel.Reader;
        if (hub.World.Broken is not null)
        {
            throw hub.World.Failure();
        }

        return reader;
    }

    private Exception Closed(int from, int tag) => hub.World.Broken is not null
        ? hub.World.Failure()
        : new InvalidOperationException($"rank {from} left the group before sending rank {Rank} its message (tag {tag})");

    private int Deliver(int from, int tag, SimMessage message, Span<byte> bytes)
    {
        try
        {
            if (message.Length > bytes.Length)
            {
                throw new InvalidDataException($"a {message.Length}-byte message from rank {from} (tag {tag}) does not fit rank {Rank}'s {bytes.Length}-byte buffer");
            }

            message.Buffer.AsSpan(0, message.Length).CopyTo(bytes);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(message.Buffer);
        }

        hub.Clock.Arrive(Real, message.Arrives, message.OwedTo);
        if (Telemetry.IsEnabled(TelemetryLevel.Network))
        {
            Telemetry.Transfer(new TransferCompleted(Name, hub.Real[from], Real, tag, message.Length, Stopwatch.GetElapsedTime(message.SentTimestamp),
                TimeSpan.FromSeconds(message.Arrives - message.SentAt)));
        }

        return message.Length;
    }
}
