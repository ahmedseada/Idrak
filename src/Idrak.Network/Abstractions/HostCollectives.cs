// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Diagnostics;
using Idrak.Networking.Collectives;

namespace Idrak.Networking.Abstractions;

/// <summary>
/// What every collective algorithm here shares (plan 23, phase C1): the buffer's values come down to one pooled host
/// buffer per call (returned in <c>finally</c>; device memory does not grow, rule 77), the algorithm works on it, and the
/// result goes back up once; messages are cut into chunks of <see cref="ClusterOptions.ChunkBytes"/> (unset: measured at
/// the first collective over a real transport and kept in the tuning cache, one message a piece on a virtual clock); each call is published as <see cref="CollectiveCompleted"/> with its bytes, wall and simulated
/// time, the time blocked on peers and the peer waited on longest, measured only while someone listens (rule 78); on a
/// simulated transport the blocked time is the virtual clock's and the slowest rank is the origin of the critical path
/// (<see cref="ITransport.SimulatedCriticalRank"/>), so a slow worker is named on every rank, the same on every run.
/// Collectives use negative tags, so a program's own messages (tags 0 and up) never mix with theirs. The ring steps
/// (<see cref="RingScatter"/>, <see cref="RingGather"/>) run over any list of ranks, so an algorithm builds rings inside a
/// node or across nodes from them.
/// <para>
/// Phase C6: over a transport that moves device memory itself (<see cref="IDeviceTransport"/>: "pcie-host", "nvlink" in
/// one process) an algorithm may keep each piece on the device and sum it there with the device's kernels
/// (<see cref="AllReduceDevice"/> and the others; ring does), the same sums in the same order, so the results are the same
/// bit for bit; two device buffers made on first need and kept serve every call (rule 77). Elsewhere, and for what an
/// algorithm keeps on the host, the values are staged through host memory; each call's <see cref="CollectiveCompleted"/>
/// says which ("device" or "host"). <see cref="ClusterOptions.DeviceCollectives"/> turns the device path off.
/// </para>
/// </summary>
internal abstract partial class HostCollectives : ICollectives, IHostValueCollectives
{
    private long _chunkBytes;
    private bool _chunkKnown;
    private bool _measuring;
    private long _sent;
    private long _received;
    private long _blockedTicks;
    private long _longestWait;
    private int _slowest;
    private TimeSpan _simulatedBlocked;
    private string? _staging;                                       // the current call's: "device", "host", or null for host values
    private readonly IDeviceTransport? _device;
    private Storage? _sendScratch;                                  // the device path's two buffers, kept (rule 77)
    private Storage? _receiveScratch;

    // Phase C5: the step's totals over every collective, kept where the ranks are separate processes (their telemetry is
    // their own), sent to rank 0 after each step (ReportStep) so rank 0's log describes the cluster.
    private const int SummaryTag = -90;
    private const int SummaryBytes = 44;
    private readonly bool _summarize;
    private long _stepSent;
    private long _stepReceived;
    private long _stepBlockedTicks;
    private long _stepWallTicks;
    private long _stepLongest;
    private int _stepSlowest;

    protected HostCollectives(ITransport transport, Device device, ClusterContext context)
    {
        Transport = transport ?? throw new ArgumentNullException(nameof(transport));
        Device = device ?? throw new ArgumentNullException(nameof(device));
        ArgumentNullException.ThrowIfNull(context);
        Context = context;
        // A message holds at most what one array can: the structural limit, not a tuning. Unset, the chunk is measured at the
        // first collective over a real transport (HostCollectives.Chunks.cs); on a virtual clock it stays one message a piece.
        _chunkBytes = Math.Clamp(context.Options.ChunkBytes ?? Array.MaxLength / 2, sizeof(float), Array.MaxLength / 2);
        _chunkKnown = context.Options.ChunkBytes is not null || transport.WorldSize < 2 || transport.SimulatedCriticalRank is not null
                      || string.Equals(transport.Name, "sim", StringComparison.OrdinalIgnoreCase);
        _summarize = transport.Capabilities.CrossProcess && transport.WorldSize > 1;
        _stepSlowest = transport.Rank;
        _device = context.Options.DeviceCollectives && transport is IDeviceTransport direct && transport.WorldSize > 1 ? direct : null;
    }

    /// <summary>The transport's device path when the device collectives are on (<see cref="ClusterOptions.DeviceCollectives"/>), else null.</summary>
    protected IDeviceTransport? DeviceTransport => _device;

    public abstract string Name { get; }

    public int Rank => Transport.Rank;

    public int WorldSize => Transport.WorldSize;

    public Device Device { get; }

    protected ITransport Transport { get; }

    // The transport for messages of one's own (tags 0 and up), and the context the collectives were made for (plan 23, C7).
    ITransport? ICollectives.Transport => Transport;

    public ClusterContext? Context { get; }

    public void AllReduce(Storage buffer, int count, ReduceOp op, WireType wire = WireType.Float32)
    {
        Check(buffer, count, nameof(buffer));
        if (_device is not null && op == ReduceOp.Sum && wire == WireType.Float32 && OnDevice("all-reduce", () => AllReduceDevice(buffer, count)))
        {
            return;
        }

        Measure("all-reduce", count, buffer, values => AllReduceHost(values, op, wire), upload: buffer, at: 0, length: count);
    }

    public void AllGather(Storage input, int count, Storage output)
    {
        Check(input, count, nameof(input));
        Check(output, checked(count * WorldSize), nameof(output));
        if (_device is not null && OnDevice("all-gather", () => AllGatherDevice(input, count, output)))
        {
            return;
        }

        _staging = "host";
        var host = ArrayPool<float>.Shared.Rent(Math.Max(1, count * WorldSize));
        try
        {
            var all = host.AsSpan(0, count * WorldSize);
            Device.Backend.Download(input, all.Slice(Rank * count, count));
            Time("all-gather", () => AllGatherHost(host.AsSpan(0, count * WorldSize), count));
            Device.Backend.Upload(all, output);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(host);
        }
    }

    public void ReduceScatter(Storage input, int count, Storage output, ReduceOp op)
    {
        Check(input, checked(count * WorldSize), nameof(input));
        Check(output, count, nameof(output));
        if (_device is not null && op == ReduceOp.Sum && OnDevice("reduce-scatter", () => ReduceScatterDevice(input, count, output)))
        {
            return;
        }

        Measure("reduce-scatter", count * WorldSize, input, values => ReduceScatterHost(values, count, op), upload: output, at: Rank * count, length: count);
    }

    public void Broadcast(Storage buffer, int count, int root)
    {
        Check(buffer, count, nameof(buffer));
        ArgumentOutOfRangeException.ThrowIfNegative(root);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(root, WorldSize);
        if (_device is not null && OnDevice("broadcast", () => BroadcastDevice(buffer, count, root)))
        {
            return;
        }

        Measure("broadcast", count, buffer, values => BroadcastHost(values, root), upload: Rank == root ? null : buffer, at: 0, length: count);
    }

    /// <summary>
    /// Reduces <paramref name="buffer"/> on the device (<see cref="DeviceTransport"/>: the pieces handed device to device and
    /// summed there in the order the host path sums them); false when the algorithm has no device path (the host path runs).
    /// </summary>
    protected virtual bool AllReduceDevice(Storage buffer, int count) => false;

    /// <summary><see cref="AllGather"/> on the device; false when the algorithm has no device path.</summary>
    protected virtual bool AllGatherDevice(Storage input, int count, Storage output) => false;

    /// <summary><see cref="ReduceScatter"/> (a sum) on the device; false when the algorithm has no device path.</summary>
    protected virtual bool ReduceScatterDevice(Storage input, int count, Storage output) => false;

    /// <summary><see cref="Broadcast"/> on the device; false when the algorithm has no device path.</summary>
    protected virtual bool BroadcastDevice(Storage buffer, int count, int root) => false;

    // Runs a device path timed as `operation`, published as staged on the device; false (nothing run) when it has none.
    private bool OnDevice(string operation, Func<bool> body)
    {
        bool timed = BeginTiming(out long start, out var simulated);
        bool done;
        try
        {
            done = body();
        }
        finally
        {
            _measuring = false;
        }

        if (done)
        {
            _staging = "device";
            EndTiming(timed, operation, start, simulated);
        }

        return done;
    }

    /// <summary>
    /// Reduce-scatter around the ring <paramref name="members"/> on the device: the member at position p ends with piece p
    /// summed over every member, written to <paramref name="target"/> at <paramref name="targetOffset"/>;
    /// <paramref name="source"/> is not changed. Each step hands the piece on (<see cref="IDeviceTransport.PostDevice"/>),
    /// takes the one before it and adds this rank's own values to what arrived, the order <see cref="RingScatter"/> sums in.
    /// </summary>
    protected void DeviceRingScatter(Storage source, ReadOnlySpan<int> members, int position, Func<int, int> offset, Func<int, int> size, Storage target, int targetOffset, int tag)
    {
        var device = _device!;
        int count = members.Length;
        var backend = Device.Backend;
        if (count <= 1)
        {
            backend.Copy2D(source, offset(position), 0, target, targetOffset, 0, 1, size(position), accumulate: false);
            return;
        }

        int next = members[(position + 1) % count], previous = members[(position - 1 + count) % count];
        int largest = 0;
        for (int s = 0; s < count; s++)
        {
            largest = Math.Max(largest, size(s));
        }

        Storage send = DeviceScratch(ref _sendScratch, largest), receive = DeviceScratch(ref _receiveScratch, largest);
        for (int step = 0; step < count - 1; step++)
        {
            int give = Wrap(position - 1 - step, count), take = Wrap(position - 2 - step, count);
            if (step == 0)
            {
                backend.Copy2D(source, offset(give), 0, send, 0, 0, 1, size(give), accumulate: false);   // this rank's own piece
            }

            var posted = device.PostDevice(next, send, size(give), tag);
            device.ReceiveDevice(previous, receive, size(take), tag);
            backend.Copy2D(source, offset(take), 0, receive, 0, 0, 1, size(take), accumulate: true);   // what arrived, then this rank's own
            device.WaitTaken(posted, next, tag);
            Counted(size(give), size(take));
            (send, receive) = (receive, send);                     // the sum goes on at the next step
        }

        backend.Copy2D(send, 0, 0, target, targetOffset, 0, 1, size(position), accumulate: false);
    }

    /// <summary>
    /// All-gather around the ring <paramref name="members"/> on the device: <paramref name="values"/> holds piece p at the
    /// member at position p; every piece on every member afterwards, handed device to device.
    /// </summary>
    protected void DeviceRingGather(Storage values, ReadOnlySpan<int> members, int position, Func<int, int> offset, Func<int, int> size, int tag)
    {
        var device = _device!;
        int count = members.Length;
        if (count <= 1)
        {
            return;
        }

        var backend = Device.Backend;
        int next = members[(position + 1) % count], previous = members[(position - 1 + count) % count];
        int largest = 0;
        for (int s = 0; s < count; s++)
        {
            largest = Math.Max(largest, size(s));
        }

        Storage send = DeviceScratch(ref _sendScratch, largest), receive = DeviceScratch(ref _receiveScratch, largest);
        backend.Copy2D(values, offset(position), 0, send, 0, 0, 1, size(position), accumulate: false);
        for (int step = 0; step < count - 1; step++)
        {
            int give = Wrap(position - step, count), take = Wrap(position - 1 - step, count);
            var posted = device.PostDevice(next, send, size(give), tag);
            device.ReceiveDevice(previous, receive, size(take), tag);
            backend.Copy2D(receive, 0, 0, values, offset(take), 0, 1, size(take), accumulate: false);
            device.WaitTaken(posted, next, tag);
            Counted(size(give), size(take));
            (send, receive) = (receive, send);                     // the piece that arrived goes on next
        }
    }

    /// <summary>A chain from <paramref name="root"/> on the device: each rank takes the values from the one before and hands them on.</summary>
    protected void DeviceChainBroadcast(Storage buffer, int count, int root, int tag)
    {
        var device = _device!;
        int world = WorldSize, next = (Rank + 1) % world, previous = (Rank - 1 + world) % world;
        if (Rank != root)
        {
            device.ReceiveDevice(previous, buffer, count, tag);
            Counted(0, count);
        }

        if (next != root)
        {
            device.WaitTaken(device.PostDevice(next, buffer, count, tag), next, tag);
            Counted(count, 0);
        }
    }

    private void Counted(int sentValues, int receivedValues)
    {
        if (_measuring)
        {
            _sent += (long)sentValues * sizeof(float);
            _received += (long)receivedValues * sizeof(float);
        }
    }

    private Storage DeviceScratch(ref Storage? slot, int values)
    {
        if (slot is null || slot.Length < values)
        {
            slot?.Release();
            slot = null;
            slot = Device.Backend.Allocate(Math.Max(1, values), zeroed: false);
        }

        return slot;
    }

    public void Barrier() => Time("barrier", Transport.Barrier);

    /// <summary>
    /// An all-reduce of values the caller already holds on the host (a gradient synchronisation's bucket, plan 23 C2), with
    /// no device copy: the same algorithm, chunks, sum order and telemetry as <see cref="AllReduce"/>.
    /// </summary>
    internal void AllReduceHostValues(Span<float> values, ReduceOp op, WireType wire)
    {
        _staging = null;
        bool timed = BeginTiming(out long start, out var simulated);
        try
        {
            AllReduceHost(values, op, wire);
        }
        finally
        {
            _measuring = false;
        }

        EndTiming(timed, "all-reduce", start, simulated);
    }

    public bool SupportsHostValues => true;

    public TimeSpan SimulatedTime => Transport.SimulatedTime;

    public bool TryAllReduce(Span<float> values, ReduceOp op, WireType wire)
    {
        AllReduceHostValues(values, op, wire);
        return true;
    }

    public bool TryAllGather(Span<float> all, int count)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        if (all.Length != checked(count * WorldSize))
        {
            throw new ArgumentException($"An all-gather of {count} values a rank over {WorldSize} ranks takes {count * WorldSize} values, not {all.Length}.", nameof(all));
        }

        _staging = null;
        bool timed = BeginTiming(out long start, out var simulated);
        try
        {
            AllGatherHost(all, count);
        }
        finally
        {
            _measuring = false;
        }

        EndTiming(timed, "all-gather", start, simulated);

        return true;
    }

    public bool TryBroadcast(Span<float> values, int root)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(root);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(root, WorldSize);
        _staging = null;
        bool timed = BeginTiming(out long start, out var simulated);
        try
        {
            BroadcastHost(values, root);
        }
        finally
        {
            _measuring = false;
        }

        EndTiming(timed, "broadcast", start, simulated);

        return true;
    }

    public bool TryReduceScatter(Span<float> values, ReadOnlySpan<int> counts, ReduceOp op, WireType wire)
    {
        var pieces = Pieces(values.Length, counts);
        _staging = null;
        bool timed = BeginTiming(out long start, out var simulated);
        try
        {
            ReduceScatterPiecesHost(values, pieces, op, op == ReduceOp.Sum ? wire : WireType.Float32);
        }
        finally
        {
            _measuring = false;
        }

        EndTiming(timed, "reduce-scatter", start, simulated);
        return true;
    }

    public bool TryAllGather(Span<float> values, ReadOnlySpan<int> counts)
    {
        var pieces = Pieces(values.Length, counts);
        _staging = null;
        bool timed = BeginTiming(out long start, out var simulated);
        try
        {
            AllGatherPiecesHost(values, pieces);
        }
        finally
        {
            _measuring = false;
        }

        EndTiming(timed, "all-gather", start, simulated);
        return true;
    }

    /// <summary>
    /// Reduces <paramref name="values"/> so this rank's piece (pieces of <paramref name="counts"/> values in rank order) holds
    /// the reduction. By default an all-reduce of every value (each piece complete on every rank, the reduction's order the
    /// algorithm's all-reduce); an algorithm with a cheaper way overrides it.
    /// </summary>
    protected virtual void ReduceScatterPiecesHost(Span<float> values, int[] counts, ReduceOp op, WireType wire) => AllReduceHost(values, op, wire);

    /// <summary>
    /// Every rank's piece of <paramref name="values"/> (pieces of <paramref name="counts"/> values in rank order) on every
    /// rank. By default the all-gather of equal pieces, or each rank's piece broadcast from it in rank order.
    /// </summary>
    protected virtual void AllGatherPiecesHost(Span<float> values, int[] counts)
    {
        if (counts.All(c => c == counts[0]))
        {
            AllGatherHost(values, counts[0]);
            return;
        }

        int at = 0;
        for (int r = 0; r < counts.Length; r++)
        {
            if (counts[r] > 0)
            {
                BroadcastHost(values.Slice(at, counts[r]), r);
            }

            at += counts[r];
        }
    }

    /// <summary>Where each piece of <paramref name="counts"/> starts.</summary>
    protected static int[] Starts(int[] counts)
    {
        var starts = new int[counts.Length];
        for (int r = 1; r < counts.Length; r++)
        {
            starts[r] = starts[r - 1] + counts[r - 1];
        }

        return starts;
    }

    private int[] Pieces(int length, ReadOnlySpan<int> counts)
    {
        if (counts.Length != WorldSize)
        {
            throw new ArgumentException($"{counts.Length} pieces for {WorldSize} ranks: give one count per rank.", nameof(counts));
        }

        long total = 0;
        foreach (int c in counts)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(c, nameof(counts));
            total += c;
        }

        if (total != length)
        {
            throw new ArgumentException($"The pieces hold {total} values; the buffer {length}.", nameof(counts));
        }

        return counts.ToArray();
    }

    /// <summary>
    /// Ends step <paramref name="step"/>'s summary where the ranks are separate processes: every rank sends rank 0 what its
    /// collectives moved during the step (bytes each way, wall time, time blocked on peers, the peer waited on longest),
    /// and rank 0 publishes one <see cref="CollectiveCompleted"/> per rank with the operation "step-summary", so rank 0's
    /// log (<c>log.jsonl</c>) holds every rank's step. Every rank calls it after each step it takes part in (the gradient
    /// syncs' <c>Stepped</c>); in one process (the simulator) it does nothing, the ranks sharing one telemetry already.
    /// </summary>
    internal void ReportStep(long step)
    {
        if (!_summarize)
        {
            return;
        }

        Span<byte> summary = stackalloc byte[SummaryBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteInt64LittleEndian(summary, step);
        System.Buffers.Binary.BinaryPrimitives.WriteInt64LittleEndian(summary[8..], _stepSent);
        System.Buffers.Binary.BinaryPrimitives.WriteInt64LittleEndian(summary[16..], _stepReceived);
        System.Buffers.Binary.BinaryPrimitives.WriteInt64LittleEndian(summary[24..], _stepBlockedTicks * TimeSpan.TicksPerSecond / Stopwatch.Frequency);
        System.Buffers.Binary.BinaryPrimitives.WriteInt64LittleEndian(summary[32..], _stepWallTicks * TimeSpan.TicksPerSecond / Stopwatch.Frequency);
        System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(summary[40..], _stepSlowest);
        (_stepSent, _stepReceived, _stepBlockedTicks, _stepWallTicks, _stepLongest, _stepSlowest) = (0, 0, 0, 0, 0, Rank);
        if (Rank != 0)
        {
            Transport.Send(0, summary, SummaryTag);
            return;
        }

        bool publish = Telemetry.IsEnabled(TelemetryLevel.Network);
        for (int r = 0; r < WorldSize; r++)
        {
            if (r > 0 && Transport.Receive(r, summary, SummaryTag) != SummaryBytes)
            {
                throw new InvalidDataException($"rank {r}'s step summary is not {SummaryBytes} bytes");
            }

            if (publish)
            {
                Telemetry.Collective(new CollectiveCompleted(Name, "step-summary", r, WorldSize,
                    System.Buffers.Binary.BinaryPrimitives.ReadInt64LittleEndian(summary[8..]), System.Buffers.Binary.BinaryPrimitives.ReadInt64LittleEndian(summary[16..]),
                    TimeSpan.FromTicks(System.Buffers.Binary.BinaryPrimitives.ReadInt64LittleEndian(summary[32..])), TimeSpan.Zero,
                    TimeSpan.FromTicks(System.Buffers.Binary.BinaryPrimitives.ReadInt64LittleEndian(summary[24..])), System.Buffers.Binary.BinaryPrimitives.ReadInt32LittleEndian(summary[40..])));
            }
        }
    }

    private void AddToStep(long wallTicks)
    {
        _stepSent += _sent;
        _stepReceived += _received;
        _stepBlockedTicks += _blockedTicks;
        _stepWallTicks += wallTicks;
        if (_longestWait > _stepLongest)
        {
            (_stepLongest, _stepSlowest) = (_longestWait, _slowest);
        }
    }

    public virtual void Dispose()
    {
        // The transport is the caller's: it may outlive these collectives.
        _sendScratch?.Release();
        _receiveScratch?.Release();
        (_sendScratch, _receiveScratch) = (null, null);
    }

    protected abstract void AllReduceHost(Span<float> values, ReduceOp op, WireType wire);

    // `all` holds this rank's values at Rank * count; every rank's afterwards.
    protected abstract void AllGatherHost(Span<float> all, int count);

    // `values` holds world · count values; this rank's slice of the reduction afterwards (at Rank * count).
    protected abstract void ReduceScatterHost(Span<float> values, int count, ReduceOp op);

    protected abstract void BroadcastHost(Span<float> values, int root);

    /// <summary>Sends <paramref name="values"/> to <paramref name="to"/> in chunks.</summary>
    protected void Send(int to, ReadOnlySpan<float> values, int tag, WireType wire)
    {
        int per = PerChunk(wire);
        var bytes = ArrayPool<byte>.Shared.Rent(Math.Max(1, Math.Min(values.Length, per) * WireValues.BytesPerValue(wire)));
        try
        {
            int at = 0;
            do
            {
                int length = Math.Min(per, values.Length - at);
                SendChunk(to, values.Slice(at, length), tag, wire, bytes);
                at += length;
            }
            while (at < values.Length);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(bytes);
        }
    }

    /// <summary>Receives <paramref name="into"/>.Length values from <paramref name="from"/> in chunks.</summary>
    protected void Receive(int from, Span<float> into, int tag, WireType wire)
    {
        int per = PerChunk(wire);
        var bytes = ArrayPool<byte>.Shared.Rent(Math.Max(1, Math.Min(into.Length, per) * WireValues.BytesPerValue(wire)));
        try
        {
            int at = 0;
            do
            {
                int length = Math.Min(per, into.Length - at);
                ReceiveChunk(from, into.Slice(at, length), tag, wire, bytes);
                at += length;
            }
            while (at < into.Length);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(bytes);
        }
    }

    /// <summary>
    /// Sends <paramref name="outgoing"/> to <paramref name="to"/> and receives <paramref name="incoming"/> from
    /// <paramref name="from"/> chunk by chunk in turn, so ranks that all send then receive never wait on each other in a
    /// circle (a ring's step).
    /// </summary>
    protected void Exchange(int to, ReadOnlySpan<float> outgoing, int from, Span<float> incoming, int tag, WireType wire)
    {
        int per = PerChunk(wire);
        int bytesPer = WireValues.BytesPerValue(wire);
        var sendBytes = ArrayPool<byte>.Shared.Rent(Math.Max(1, Math.Min(outgoing.Length, per) * bytesPer));
        var receiveBytes = ArrayPool<byte>.Shared.Rent(Math.Max(1, Math.Min(incoming.Length, per) * bytesPer));
        try
        {
            int sendChunks = Chunks(outgoing.Length, per), receiveChunks = Chunks(incoming.Length, per);
            for (int k = 0; k < Math.Max(sendChunks, receiveChunks); k++)
            {
                if (k < sendChunks)
                {
                    SendChunk(to, outgoing.Slice(k * per, Math.Min(per, outgoing.Length - (k * per))), tag, wire, sendBytes);
                }

                if (k < receiveChunks)
                {
                    ReceiveChunk(from, incoming.Slice(k * per, Math.Min(per, incoming.Length - (k * per))), tag, wire, receiveBytes);
                }
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(sendBytes);
            ArrayPool<byte>.Shared.Return(receiveBytes);
        }
    }

    /// <summary>
    /// Reduce-scatter around the ring <paramref name="members"/> (ranks in ring order; this rank at
    /// <paramref name="position"/>): after <c>members.Length - 1</c> steps the member at position p holds piece p (values
    /// [offset(p), offset(p) + size(p)) of <paramref name="values"/>) reduced over every member, in the order p+1, p+2, …,
    /// p: an order fixed by the sizes alone (rule 86). On a bfloat16 wire each member's own values are rounded as they
    /// first travel and the partial sums travel as bfloat16; the sums are made in float32.
    /// </summary>
    protected void RingScatter(Span<float> values, ReadOnlySpan<int> members, int position, Func<int, int> offset, Func<int, int> size, ReduceOp op,
        WireType wire, int tag)
    {
        int count = members.Length;
        if (count <= 1)
        {
            return;
        }

        int next = members[(position + 1) % count], previous = members[(position - 1 + count) % count];
        int largest = 0;
        for (int s = 0; s < count; s++)
        {
            largest = Math.Max(largest, size(s));
        }

        var arrived = ArrayPool<float>.Shared.Rent(Math.Max(1, largest));
        try
        {
            for (int step = 0; step < count - 1; step++)
            {
                int send = Wrap(position - 1 - step, count), take = Wrap(position - 2 - step, count);
                var outgoing = values.Slice(offset(send), size(send));
                if (wire == WireType.BFloat16 && step == 0)
                {
                    WireValues.RoundToBFloat16(outgoing);          // this rank's own values as they first travel
                }

                var incoming = arrived.AsSpan(0, size(take));
                Exchange(next, outgoing, previous, incoming, tag, wire);
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

    /// <summary>
    /// All-gather around the ring <paramref name="members"/>: the member at position p holds piece p; after
    /// <c>members.Length - 1</c> steps every member holds every piece, passed round as float32.
    /// </summary>
    protected void RingGather(Span<float> values, ReadOnlySpan<int> members, int position, Func<int, int> offset, Func<int, int> size, int tag)
    {
        int count = members.Length;
        if (count <= 1)
        {
            return;
        }

        int next = members[(position + 1) % count], previous = members[(position - 1 + count) % count];
        for (int step = 0; step < count - 1; step++)
        {
            int send = Wrap(position - step, count), take = Wrap(position - 1 - step, count);
            Exchange(next, values.Slice(offset(send), size(send)), previous, values.Slice(offset(take), size(take)), tag, WireType.Float32);
        }
    }

    /// <summary><paramref name="value"/> modulo <paramref name="count"/>, from 0.</summary>
    protected static int Wrap(int value, int count) => ((value % count) + count) % count;

    /// <summary>values[i] = values[i] (op) other[i].</summary>
    protected static void Combine(Span<float> values, ReadOnlySpan<float> other, ReduceOp op)
    {
        switch (op)
        {
            case ReduceOp.Sum:
                int i = 0;
                if (System.Numerics.Vector.IsHardwareAccelerated)
                {
                    int width = System.Numerics.Vector<float>.Count;
                    for (; i <= values.Length - width; i += width)
                    {
                        (new System.Numerics.Vector<float>(values[i..]) + new System.Numerics.Vector<float>(other[i..])).CopyTo(values[i..]);
                    }
                }

                for (; i < values.Length; i++)                              // the scalar tail
                {
                    values[i] += other[i];
                }

                break;
            case ReduceOp.Max:
                for (int j = 0; j < values.Length; j++)
                {
                    values[j] = Math.Max(values[j], other[j]);
                }

                break;
            default:
                for (int j = 0; j < values.Length; j++)
                {
                    values[j] = Math.Min(values[j], other[j]);
                }

                break;
        }
    }

    private static int Chunks(int values, int per) => values == 0 ? 1 : (values + per - 1) / per;

    private int PerChunk(WireType wire) => (int)Math.Max(1, _chunkBytes / WireValues.BytesPerValue(wire));

    private void SendChunk(int to, ReadOnlySpan<float> values, int tag, WireType wire, byte[] bytes)
    {
        int length = WireValues.Encode(values, bytes, wire);
        Transport.Send(to, bytes.AsSpan(0, length), tag);
        _sent += length;
    }

    private void ReceiveChunk(int from, Span<float> into, int tag, WireType wire, byte[] bytes)
    {
        int expected = into.Length * WireValues.BytesPerValue(wire);
        long start = _measuring ? Stopwatch.GetTimestamp() : 0;
        var simulatedBefore = _measuring ? Transport.SimulatedTime : TimeSpan.Zero;
        int length = Transport.Receive(from, bytes.AsSpan(0, Math.Max(1, expected)), tag);
        if (_measuring)
        {
            _simulatedBlocked += Transport.SimulatedTime - simulatedBefore;
            long waited = Stopwatch.GetTimestamp() - start;
            _blockedTicks += waited;
            if (waited > _longestWait)
            {
                (_longestWait, _slowest) = (waited, from);
            }
        }

        if (length != expected)
        {
            throw new InvalidDataException($"rank {Rank} received {length} bytes from rank {from} (tag {tag}) where {expected} were due: the ranks called different collectives or sizes");
        }

        WireValues.Decode(bytes.AsSpan(0, length), into, wire);
        _received += length;
    }

    // Downloads `count` values of `source`, runs `body` on them, uploads `length` of them from `at` into `upload` (if any).
    private void Measure(string operation, int count, Storage source, SpanAction body, Storage? upload, int at, int length)
    {
        var host = ArrayPool<float>.Shared.Rent(Math.Max(1, count));
        try
        {
            var values = host.AsSpan(0, count);
            Device.Backend.Download(source, values);
            _staging = "host";
            Time(operation, () => body(host.AsSpan(0, count)));
            if (upload is not null)
            {
                Device.Backend.Upload(host.AsSpan(at, length), upload);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(host);
        }
    }

    private delegate void SpanAction(Span<float> values);

    private void Time(string operation, Action body)
    {
        bool timed = BeginTiming(out long start, out var simulated);
        try
        {
            body();
        }
        finally
        {
            _measuring = false;
        }

        EndTiming(timed, operation, start, simulated);
    }

    // Starts counting a call's bytes, waits and times, only while someone listens (rule 78) or the step's summary is kept
    // for rank 0 (ranks in processes of their own, phase C5).
    private bool BeginTiming(out long start, out TimeSpan simulated)
    {
        if (!_chunkKnown)
        {
            _chunkKnown = true;                                     // before measuring: the measurement's own calls pass here
            _chunkBytes = MeasureChunk();
        }

        (start, simulated) = (0, TimeSpan.Zero);
        if (!Telemetry.IsEnabled(TelemetryLevel.Network) && !_summarize)
        {
            return false;
        }

        (_measuring, _sent, _received, _blockedTicks, _longestWait, _slowest, _simulatedBlocked) = (true, 0, 0, 0, 0, Rank, TimeSpan.Zero);
        simulated = Transport.SimulatedTime;
        start = Stopwatch.GetTimestamp();
        return true;
    }

    // Publishes the call as CollectiveCompleted: on a simulated transport the blocked time is the virtual clock's and the
    // slowest rank the critical path's origin, so a slow worker is named the same on every rank and run.
    private void EndTiming(bool timed, string operation, long start, TimeSpan simulated)
    {
        if (!timed)
        {
            return;
        }

        AddToStep(Stopwatch.GetTimestamp() - start);
        if (!Telemetry.IsEnabled(TelemetryLevel.Network))
        {
            return;
        }

        var wall = Stopwatch.GetElapsedTime(start);
        var simulatedTime = Transport.SimulatedTime - simulated;
        bool onClock = Transport.SimulatedTime > TimeSpan.Zero;       // a simulated transport (its clock has moved): the cluster's time, even for a call that took none
        Telemetry.Collective(new CollectiveCompleted(Name, operation, Rank, WorldSize, _sent, _received, wall, simulatedTime,
            onClock ? _simulatedBlocked : TimeSpan.FromSeconds(_blockedTicks / (double)Stopwatch.Frequency),
            onClock && Transport.SimulatedCriticalRank is { } critical ? critical : _slowest, operation == "barrier" ? null : _staging));
    }

    private void Check(Storage storage, int count, string name)
    {
        ArgumentNullException.ThrowIfNull(storage, name);
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        if (count > storage.Length)
        {
            throw new ArgumentException($"{name} holds {storage.Length} values; the call names {count}.", name);
        }

        if (!ReferenceEquals(storage.Backend, Device.Backend))
        {
            throw new ArgumentException($"{name} is not on rank {Rank}'s device {Device}.", name);
        }
    }
}
