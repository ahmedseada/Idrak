// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using Idrak.Abstraction.Cluster;
using Idrak.Abstraction.Devices;

// Plan 23, phase C0: a reference transport and reference collectives the kit's suites are checked against before the
// library's own exist (Idrak.Cluster's "sim" and Idrak.Network's algorithms come in C1). Each has a broken twin the
// suites must fail. Test-only: nothing here is shipped.
internal sealed class MemoryTransport : ITransport
{
    private readonly Shared _shared;
    private readonly bool _corrupt;

    private MemoryTransport(Shared shared, int rank, bool corrupt)
    {
        (_shared, Rank, _corrupt) = (shared, rank, corrupt);
    }

    public string Name => "memory";

    public int Rank { get; }

    public int WorldSize => _shared.World;

    public TransportCapabilities Capabilities => new(0, TimeSpan.Zero, ZeroCopy: false, GpuDirect: false, Reliable: true, CrossProcess: false, CrossMachine: false);

    // A connected group; with corrupt, every message of 2 bytes or more arrives with its last byte flipped.
    public static IReadOnlyList<ITransport> Connect(int world, TimeSpan timeout, bool corrupt = false)
    {
        var shared = new Shared(world, timeout);
        return [.. Enumerable.Range(0, world).Select(r => new MemoryTransport(shared, r, corrupt))];
    }

    public void Send(int to, ReadOnlySpan<byte> bytes, int tag)
    {
        if (to == Rank || (uint)to >= (uint)WorldSize)
        {
            throw new ArgumentOutOfRangeException(nameof(to), to, $"rank {Rank} can send to the other ranks of 0 to {WorldSize - 1}");
        }

        var copy = bytes.ToArray();
        if (_corrupt && copy.Length >= 2)
        {
            copy[^1] ^= 0xFF;
        }

        _shared.Queue(Rank, to, tag).Add(copy);
    }

    public int Receive(int from, Span<byte> bytes, int tag)
    {
        if (!_shared.Queue(from, Rank, tag).TryTake(out var message, _shared.Timeout))
        {
            throw new TimeoutException($"rank {Rank} waited {_shared.Timeout.TotalSeconds:F0} s for rank {from} (tag {tag})");
        }

        if (message.Length > bytes.Length)
        {
            throw new InvalidDataException($"a {message.Length}-byte message from rank {from} does not fit {bytes.Length} bytes");
        }

        message.CopyTo(bytes);
        return message.Length;
    }

    public ValueTask SendAsync(int to, ReadOnlyMemory<byte> bytes, int tag, CancellationToken cancellationToken = default)
    {
        Send(to, bytes.Span, tag);
        return ValueTask.CompletedTask;
    }

    public ValueTask<int> ReceiveAsync(int from, Memory<byte> bytes, int tag, CancellationToken cancellationToken = default) =>
        new(Task.Run(() => Receive(from, bytes.Span, tag), cancellationToken));

    public void Barrier()
    {
        if (!_shared.Gate.SignalAndWait(_shared.Timeout))
        {
            throw new TimeoutException($"rank {Rank} waited {_shared.Timeout.TotalSeconds:F0} s at a barrier");
        }
    }

    public void Dispose()
    {
    }

    private sealed class Shared(int world, TimeSpan timeout)
    {
        private readonly ConcurrentDictionary<(int, int, int), BlockingCollection<byte[]>> _queues = new();

        public int World { get; } = world;

        public TimeSpan Timeout { get; } = timeout;

        public Barrier Gate { get; } = new(world);

        public BlockingCollection<byte[]> Queue(int from, int to, int tag) => _queues.GetOrAdd((from, to, tag), _ => []);
    }
}

// Every rank to rank 0, which combines in rank order and sends the result back; the reference every algorithm agrees with
// within float rounding. With skipReduce, all-reduce leaves each rank its own values (the broken twin).
internal sealed class NaiveReferenceCollectives(ITransport transport, Device device, bool skipReduce = false) : ICollectives
{
    private const int ReduceTag = 1_000, GatherTag = 1_001, BroadcastTag = 1_002, ResultTag = 1_003;

    public string Name => "naive-reference";

    public int Rank => transport.Rank;

    public int WorldSize => transport.WorldSize;

    public Device Device { get; } = device;

    public static IReadOnlyList<ICollectives> Connect(int world, TimeSpan timeout, Device device, bool skipReduce = false) =>
        [.. MemoryTransport.Connect(world, timeout).Select(t => new NaiveReferenceCollectives(t, device, skipReduce))];

    public void AllReduce(Storage buffer, int count, ReduceOp op, WireType wire = WireType.Float32)
    {
        var values = Download(buffer, count);
        if (op == ReduceOp.Sum && wire == WireType.BFloat16)
        {
            RoundToBFloat16(values);
        }

        if (!skipReduce)
        {
            Reduce(values, op);
        }

        Device.Backend.Upload(values, buffer);
    }

    public void AllGather(Storage input, int count, Storage output)
    {
        var mine = Download(input, count);
        var all = new float[count * WorldSize];
        if (Rank == 0)
        {
            mine.CopyTo(all, 0);
            for (int r = 1; r < WorldSize; r++)
            {
                Receive(r, all.AsSpan(r * count, count), GatherTag);
            }

            for (int r = 1; r < WorldSize; r++)
            {
                transport.Send(r, MemoryMarshal.AsBytes(all.AsSpan()), ResultTag);
            }
        }
        else
        {
            transport.Send(0, MemoryMarshal.AsBytes(mine.AsSpan()), GatherTag);
            Receive(0, all, ResultTag);
        }

        Device.Backend.Upload(all, output);
    }

    public void ReduceScatter(Storage input, int count, Storage output, ReduceOp op)
    {
        var values = Download(input, count * WorldSize);
        Reduce(values, op);
        Device.Backend.Upload(values.AsSpan(Rank * count, count), output);
    }

    public void Broadcast(Storage buffer, int count, int root)
    {
        var values = Download(buffer, count);
        if (Rank == root)
        {
            for (int r = 0; r < WorldSize; r++)
            {
                if (r != root)
                {
                    transport.Send(r, MemoryMarshal.AsBytes(values.AsSpan()), BroadcastTag);
                }
            }
        }
        else
        {
            Receive(root, values, BroadcastTag);
            Device.Backend.Upload(values, buffer);
        }
    }

    public void Barrier() => transport.Barrier();

    public void Dispose() => transport.Dispose();

    // Rank 0 combines every rank's values in rank order (0, 1, 2, …) and sends the result back.
    private void Reduce(float[] values, ReduceOp op)
    {
        if (Rank == 0)
        {
            var other = new float[values.Length];
            for (int r = 1; r < WorldSize; r++)
            {
                Receive(r, other, ReduceTag);
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = op switch { ReduceOp.Max => Math.Max(values[i], other[i]), ReduceOp.Min => Math.Min(values[i], other[i]), _ => values[i] + other[i] };
                }
            }

            for (int r = 1; r < WorldSize; r++)
            {
                transport.Send(r, MemoryMarshal.AsBytes(values.AsSpan()), ResultTag);
            }
        }
        else
        {
            transport.Send(0, MemoryMarshal.AsBytes(values.AsSpan()), ReduceTag);
            Receive(0, values, ResultTag);
        }
    }

    private void Receive(int from, Span<float> into, int tag)
    {
        int length = transport.Receive(from, MemoryMarshal.AsBytes(into), tag);
        if (length != into.Length * sizeof(float))
        {
            throw new InvalidDataException($"rank {Rank} got {length} bytes from rank {from} where {into.Length * sizeof(float)} were due");
        }
    }

    private float[] Download(Storage storage, int count)
    {
        var values = new float[count];
        Device.Backend.Download(storage, values);
        return values;
    }

    // Round to nearest even at bfloat16's 8 bits of mantissa, as a bfloat16 wire sends the values.
    private static void RoundToBFloat16(Span<float> values)
    {
        foreach (ref float v in values)
        {
            uint bits = BitConverter.SingleToUInt32Bits(v);
            bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000;
            v = BitConverter.UInt32BitsToSingle(bits);
        }
    }
}
