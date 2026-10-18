// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Binary;
using System.Text;

namespace Idrak.Cluster;

/// <summary>A rank's simulated time when it began a step, from an <see cref="EventLog"/>.</summary>
/// <param name="Rank">The rank.</param>
/// <param name="Step">The step it began.</param>
/// <param name="Seconds">Its time on the virtual clock then.</param>
public readonly record struct ClockRecord(int Rank, long Step, double Seconds);

/// <summary>
/// The event log of a simulated run (plan 23, phase C3): a compact binary record (little-endian, written with
/// <see cref="BinaryPrimitives"/>) of the faults drawn, each rank's clock when it began each step and when it finished,
/// its state changes, and (unless <see cref="SimulationOptions.LogTransfers"/> is off) every transfer it sent with the
/// clock's times and resends. Each rank's records are kept in pooled blocks of its own, in the order it made them, and
/// written rank by rank, so a run of a seed gives the same bytes every time, whatever order the threads ran in; a replay
/// to a step (<see cref="ClusterReplay"/>) gives the same records up to there.
/// </summary>
public sealed class EventLog
{
    /// <summary>The format version this one writes and reads.</summary>
    public const int Version = 1;

    internal static ReadOnlySpan<byte> Magic => "IDRKLOG\0"u8;

    private readonly Dictionary<(int Rank, long Step), double> _clocks;

    private EventLog(byte[] bytes, int seed, int worldSize, string failures, IReadOnlyList<Fault> faults, IReadOnlyList<Fault> live, IReadOnlyList<ClockRecord> clocks,
        IReadOnlyList<RankStateChanged> states, IReadOnlyList<TransferRecord> transfers, IReadOnlyList<double?> ends)
    {
        LiveFaults = live;
        Bytes = bytes;
        Seed = seed;
        WorldSize = worldSize;
        Failures = failures;
        Faults = faults;
        Clocks = clocks;
        States = states;
        TransferRecords = transfers;
        EndClocks = ends;
        _clocks = clocks.ToDictionary(c => (c.Rank, c.Step), c => c.Seconds);
    }

    /// <summary>The log as written.</summary>
    public ReadOnlyMemory<byte> Bytes { get; }

    /// <summary>The run's seed.</summary>
    public int Seed { get; }

    /// <summary>The number of ranks.</summary>
    public int WorldSize { get; }

    /// <summary>The failure model's name.</summary>
    public string Failures { get; }

    /// <summary>The faults drawn, in step order (the live ones among them).</summary>
    public IReadOnlyList<Fault> Faults { get; }

    /// <summary>
    /// The faults asked for while the run went (phase C9: <see cref="ClusterRunControl.Inject"/>, the "live" model), each at
    /// the step it was applied at: what a replay of the steered run injects again (<see cref="ClusterReplay"/>).
    /// </summary>
    public IReadOnlyList<Fault> LiveFaults { get; }

    /// <summary>Each rank's clock as it began each step, rank by rank.</summary>
    public IReadOnlyList<ClockRecord> Clocks { get; }

    /// <summary>Each rank's state changes, with their step and the seed.</summary>
    public IReadOnlyList<RankStateChanged> States { get; }

    /// <summary>The transfers recorded (none when the log leaves them out).</summary>
    public long Transfers => TransferRecords.Count;

    /// <summary>The messages the transfers recorded were sent again for, lost on their link.</summary>
    public long Resends => TransferRecords.Sum(t => (long)t.Resends);

    /// <summary>Each rank's clock when it finished, or null for one that did not.</summary>
    public IReadOnlyList<double?> EndClocks { get; }

    internal IReadOnlyList<TransferRecord> TransferRecords { get; }

    /// <summary>Rank <paramref name="rank"/>'s clock as it began step <paramref name="step"/>, or null when it did not.</summary>
    public double? ClockAt(int rank, long step) => _clocks.TryGetValue((rank, step), out double seconds) ? seconds : null;

    /// <summary>Writes the log to <paramref name="path"/>.</summary>
    public void Save(string path) => File.WriteAllBytes(path, Bytes.ToArray());

    /// <summary>Reads the log at <paramref name="path"/>.</summary>
    /// <exception cref="InvalidDataException">It is not an event log of this version; the message names the file.</exception>
    public static EventLog Load(string path)
    {
        try
        {
            return Parse(File.ReadAllBytes(path));
        }
        catch (InvalidDataException e)
        {
            throw new InvalidDataException($"{path}: {e.Message}", e);
        }
    }

    /// <summary>Reads a log from its bytes.</summary>
    /// <exception cref="InvalidDataException">They are not an event log of this version, or end early.</exception>
    public static EventLog Parse(ReadOnlySpan<byte> bytes)
    {
        var r = new Reader(bytes);
        if (!r.Take(Magic.Length).SequenceEqual(Magic))
        {
            throw new InvalidDataException("not an Idrak cluster event log (no IDRKLOG header).");
        }

        int version = r.UInt16();
        if (version != Version)
        {
            throw new InvalidDataException($"event log version {version}; this Idrak reads version {Version}.");
        }

        int seed = r.Int32(), world = r.Int32();
        string failures = r.Text();
        var faults = new List<Fault>();
        var live = new List<Fault>();
        var faultBytes = new Reader(r.Take(checked((int)r.Int64())));
        while (!faultBytes.Done)
        {
            long step = faultBytes.Int64();
            int kindByte = faultBytes.Byte();
            var kind = (FaultKind)(kindByte & ~LiveMark);
            var duration = TimeSpan.FromTicks(faultBytes.Int64());
            double factor = faultBytes.Double();
            var ranks = new int[faultBytes.Int32()];
            for (int i = 0; i < ranks.Length; i++)
            {
                ranks[i] = faultBytes.Int32();
            }

            var fault = new Fault(kind, step, ranks, duration, factor);
            faults.Add(fault);
            if ((kindByte & LiveMark) != 0)
            {
                live.Add(fault);
            }
        }

        var clocks = new List<ClockRecord>();
        var states = new List<RankStateChanged>();
        var transfers = new List<TransferRecord>();
        var ends = new double?[world];
        for (int rank = 0; rank < world; rank++)
        {
            var records = new Reader(r.Take(checked((int)r.Int64())));
            while (!records.Done)
            {
                switch (records.Byte())
                {
                    case RecordClock:
                        long step = records.Int64();
                        clocks.Add(new ClockRecord(rank, step, records.Double()));
                        break;
                    case RecordState:
                        long at = records.Int64();
                        var state = (RankState)records.Byte();
                        string detail = records.Text();
                        states.Add(new RankStateChanged(rank, state, at, detail.Length == 0 ? null : detail, seed));
                        break;
                    case RecordTransfer:
                        transfers.Add(new TransferRecord(rank, records.Int32(), records.Int32(), records.Int64(), records.Int64(), records.Double(), records.Double(), records.Int32()));
                        break;
                    case RecordEnd:
                        ends[rank] = records.Double();
                        break;
                    default:
                        throw new InvalidDataException($"an unknown record in rank {rank}'s events.");
                }
            }
        }

        if (!r.Done)
        {
            throw new InvalidDataException("bytes follow the last rank's events.");
        }

        return new EventLog(bytes.ToArray(), seed, world, failures, faults, live, clocks, states, transfers, ends);
    }

    internal const byte RecordClock = 1, RecordState = 2, RecordTransfer = 3, RecordEnd = 4;

    // The high bit of a fault's kind: the fault was asked for while the run went (a live fault, phase C9). Logs without it read as before.
    internal const int LiveMark = 0x80;

    // Reads little-endian fields from the front of a span, refusing to read past its end.
    private ref struct Reader(ReadOnlySpan<byte> bytes)
    {
        private ReadOnlySpan<byte> _rest = bytes;

        public readonly bool Done => _rest.IsEmpty;

        public ReadOnlySpan<byte> Take(int count)
        {
            if (count < 0 || count > _rest.Length)
            {
                throw new InvalidDataException("the event log ends early.");
            }

            var taken = _rest[..count];
            _rest = _rest[count..];
            return taken;
        }

        public byte Byte() => Take(1)[0];

        public int UInt16() => BinaryPrimitives.ReadUInt16LittleEndian(Take(2));

        public int Int32() => BinaryPrimitives.ReadInt32LittleEndian(Take(4));

        public long Int64() => BinaryPrimitives.ReadInt64LittleEndian(Take(8));

        public double Double() => BinaryPrimitives.ReadDoubleLittleEndian(Take(8));

        public string Text() => Encoding.UTF8.GetString(Take(UInt16()));
    }
}

// One transfer a rank sent, as the virtual clock charged it.
internal readonly record struct TransferRecord(int From, int To, int Tag, long Sequence, long Bytes, double Sent, double Arrives, int Resends);

// Writes a run's event log while it runs: the faults (under the injector's lock) and each rank's records (by that rank's
// thread alone) into pooled blocks, joined at the end; the blocks go back to the pool when it is disposed.
internal sealed class EventLogWriter(int seed, int worldSize, string failures, bool transfers) : IDisposable
{
    private const int MaxDetailChars = 300;

    private readonly Blocks _faults = new();
    private readonly Blocks[] _ranks = [.. Enumerable.Range(0, worldSize).Select(_ => new Blocks())];

    public bool LogsTransfers => transfers;

    public void Fault(Fault fault, bool live = false)
    {
        var b = _faults.Take(8 + 1 + 8 + 8 + 4 + (4 * fault.Ranks.Count));
        BinaryPrimitives.WriteInt64LittleEndian(b, fault.Step);
        b[8] = (byte)((int)fault.Kind | (live ? EventLog.LiveMark : 0));
        BinaryPrimitives.WriteInt64LittleEndian(b[9..], fault.Duration.Ticks);
        BinaryPrimitives.WriteDoubleLittleEndian(b[17..], fault.Factor);
        BinaryPrimitives.WriteInt32LittleEndian(b[25..], fault.Ranks.Count);
        for (int i = 0; i < fault.Ranks.Count; i++)
        {
            BinaryPrimitives.WriteInt32LittleEndian(b[(29 + (4 * i))..], fault.Ranks[i]);
        }
    }

    public void Clock(int rank, long step, double seconds)
    {
        var b = _ranks[rank].Take(17);
        b[0] = EventLog.RecordClock;
        BinaryPrimitives.WriteInt64LittleEndian(b[1..], step);
        BinaryPrimitives.WriteDoubleLittleEndian(b[9..], seconds);
    }

    public void State(int rank, long step, RankState state, string? detail)
    {
        string text = detail is { Length: > MaxDetailChars } ? detail[..MaxDetailChars] : detail ?? "";
        int length = Encoding.UTF8.GetByteCount(text);
        var b = _ranks[rank].Take(12 + length);
        b[0] = EventLog.RecordState;
        BinaryPrimitives.WriteInt64LittleEndian(b[1..], step);
        b[9] = (byte)state;
        BinaryPrimitives.WriteUInt16LittleEndian(b[10..], (ushort)length);
        Encoding.UTF8.GetBytes(text, b[12..]);
    }

    public void Transfer(int from, int to, int tag, long sequence, long bytes, in Charged charged)
    {
        if (!transfers)
        {
            return;
        }

        var b = _ranks[from].Take(45);
        b[0] = EventLog.RecordTransfer;
        BinaryPrimitives.WriteInt32LittleEndian(b[1..], to);
        BinaryPrimitives.WriteInt32LittleEndian(b[5..], tag);
        BinaryPrimitives.WriteInt64LittleEndian(b[9..], sequence);
        BinaryPrimitives.WriteInt64LittleEndian(b[17..], bytes);
        BinaryPrimitives.WriteDoubleLittleEndian(b[25..], charged.Sent);
        BinaryPrimitives.WriteDoubleLittleEndian(b[33..], charged.Arrives);
        BinaryPrimitives.WriteInt32LittleEndian(b[41..], charged.Resends);
    }

    public void End(int rank, double seconds)
    {
        var b = _ranks[rank].Take(9);
        b[0] = EventLog.RecordEnd;
        BinaryPrimitives.WriteDoubleLittleEndian(b[1..], seconds);
    }

    // The log: the header, the faults, then each rank's records.
    public EventLog Seal()
    {
        int nameLength = Encoding.UTF8.GetByteCount(failures);
        long total = EventLog.Magic.Length + 2 + 4 + 4 + 2 + nameLength + 8 + _faults.Length + _ranks.Sum(s => 8 + s.Length);
        var bytes = new byte[checked((int)total)];
        var b = bytes.AsSpan();
        EventLog.Magic.CopyTo(b);
        int at = EventLog.Magic.Length;
        BinaryPrimitives.WriteUInt16LittleEndian(b[at..], EventLog.Version);
        BinaryPrimitives.WriteInt32LittleEndian(b[(at + 2)..], seed);
        BinaryPrimitives.WriteInt32LittleEndian(b[(at + 6)..], worldSize);
        BinaryPrimitives.WriteUInt16LittleEndian(b[(at + 10)..], (ushort)nameLength);
        at += 12 + Encoding.UTF8.GetBytes(failures, b[(at + 12)..]);
        foreach (var stream in (Blocks[])[_faults, .. _ranks])
        {
            BinaryPrimitives.WriteInt64LittleEndian(b[at..], stream.Length);
            at += 8;
            at += stream.CopyTo(b[at..]);
        }

        return EventLog.Parse(bytes);
    }

    public void Dispose()
    {
        _faults.Dispose();
        foreach (var stream in _ranks)
        {
            stream.Dispose();
        }
    }

    // Records in pooled blocks (rule 10: rented, returned when the log is disposed).
    private sealed class Blocks : IDisposable
    {
        private const int BlockBytes = 64 * 1024;

        private readonly List<(byte[] Block, int Used)> _blocks = [];

        public long Length { get; private set; }

        public Span<byte> Take(int count)
        {
            if (_blocks.Count == 0 || _blocks[^1].Block.Length - _blocks[^1].Used < count)
            {
                _blocks.Add((ArrayPool<byte>.Shared.Rent(Math.Max(BlockBytes, count)), 0));
            }

            var (block, used) = _blocks[^1];
            _blocks[^1] = (block, used + count);
            Length += count;
            return block.AsSpan(used, count);
        }

        public int CopyTo(Span<byte> target)
        {
            int at = 0;
            foreach (var (block, used) in _blocks)
            {
                block.AsSpan(0, used).CopyTo(target[at..]);
                at += used;
            }

            return at;
        }

        public void Dispose()
        {
            foreach (var (block, _) in _blocks)
            {
                ArrayPool<byte>.Shared.Return(block);
            }

            _blocks.Clear();
        }
    }
}
