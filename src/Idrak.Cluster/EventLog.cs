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
    /// <summary>
    /// The format version this one writes (and reads, with version 1): version 2 adds the cluster advisor's setting to the
    /// header (plan 23, C8c); a version 1 log was written before the advisor and reads as one with it off.
    /// </summary>
    public const int Version = 2;

    // The advisor's setting in the header: on (a byte), its warm steps and its slice's values (two Int32s).
    internal const int AdviceBytes = 9;

    internal static ReadOnlySpan<byte> Magic => "IDRKLOG\0"u8;

    // The file a run's log is appended to while it goes (SimulationOptions.EventLogFlushSteps): the same header, then
    // chunks of one stream each (-1: the faults; r: rank r's records) in the order they were written.
    internal static ReadOnlySpan<byte> JournalMagic => "IDRKJRN\0"u8;

    private readonly Dictionary<(int Rank, long Step), double> _clocks;

    private EventLog(byte[] bytes, int seed, int worldSize, string failures, (bool On, int Steps, int Slice) advice, IReadOnlyList<Fault> faults, IReadOnlyList<Fault> live,
        IReadOnlyList<ClockRecord> clocks, IReadOnlyList<RankStateChanged> states, IReadOnlyList<TransferRecord> transfers, IReadOnlyList<double?> ends)
    {
        (Suggest, SuggestSteps, SuggestSliceValues) = advice;
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

    /// <summary>
    /// Whether the run's ranks measured for the cluster setting advisor (<see cref="ClusterOptions.Suggest"/> as the run
    /// resolved it): its slice's two collectives are on the clock, so a replay from this log measures alike.
    /// </summary>
    public bool Suggest { get; }

    /// <summary>The warm steps the advisor measured (<see cref="ClusterOptions.SuggestSteps"/>).</summary>
    public int SuggestSteps { get; }

    /// <summary>The values of the advisor's slice (<see cref="ClusterOptions.SuggestSliceValues"/>).</summary>
    public int SuggestSliceValues { get; }

    /// <summary><paramref name="options"/> with the advisor's setting this log's run had, so a rerun charges the clock alike.</summary>
    public ClusterOptions AdviceOf(ClusterOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return options with { Suggest = Suggest, SuggestSteps = SuggestSteps, SuggestSliceValues = SuggestSliceValues };
    }

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

    /// <summary>Reads the log at <paramref name="path"/>: a finished run's, or what a run still going wrote so far.</summary>
    /// <exception cref="InvalidDataException">It is not an event log of this version; the message names the file.</exception>
    public static EventLog Load(string path)
    {
        try
        {
            // Shared for writing: a run still going appends to it (SimulationOptions.EventLogFlushSteps).
            using var file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
            var bytes = new byte[file.Length];
            file.ReadExactly(bytes);
            return Parse(bytes);
        }
        catch (InvalidDataException e)
        {
            throw new InvalidDataException($"{path}: {e.Message}", e);
        }
    }

    /// <summary>
    /// Reads a log from its bytes: a finished log, or the file of a run still going (what its ranks recorded so far, each
    /// rank's records in order; a chunk being written as it was read is left out).
    /// </summary>
    /// <exception cref="InvalidDataException">They are not an event log of this version, or end early.</exception>
    public static EventLog Parse(ReadOnlySpan<byte> bytes)
    {
        if (bytes.StartsWith(JournalMagic))
        {
            return Parse(FromJournal(bytes));
        }

        var r = new Reader(bytes);
        if (!r.Take(Magic.Length).SequenceEqual(Magic))
        {
            throw new InvalidDataException("not an Idrak cluster event log (no IDRKLOG header).");
        }

        int version = r.UInt16();
        if (version is not (1 or Version))
        {
            throw new InvalidDataException($"event log version {version}; this Idrak reads version 1 and version {Version}.");
        }

        int seed = r.Int32(), world = r.Int32();
        string failures = r.Text();
        var advice = ReadAdvice(ref r, version);
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

        return new EventLog(bytes.ToArray(), seed, world, failures, advice, faults, live, clocks, states, transfers, ends);
    }

    // The advisor's setting after the failure model's name (version 2), or a run before the advisor (version 1): off.
    private static (bool On, int Steps, int Slice) ReadAdvice(ref Reader r, int version)
    {
        if (version < 2)
        {
            var defaults = new ClusterOptions();
            return (false, defaults.SuggestSteps, defaults.SuggestSliceValues);
        }

        bool on = r.Byte() != 0;
        return (on, r.Int32(), r.Int32());
    }

    internal static void WriteAdvice(Span<byte> field, (bool On, int Steps, int Slice) advice)
    {
        field[0] = advice.On ? (byte)1 : (byte)0;
        BinaryPrimitives.WriteInt32LittleEndian(field[1..], advice.Steps);
        BinaryPrimitives.WriteInt32LittleEndian(field[5..], advice.Slice);
    }

    internal const byte RecordClock = 1, RecordState = 2, RecordTransfer = 3, RecordEnd = 4;

    // A journal's chunks joined into a finished log's layout: the header, the faults, then each rank's records.
    internal static byte[] FromJournal(ReadOnlySpan<byte> journal)
    {
        var r = new Reader(journal);
        r.Take(JournalMagic.Length);
        int version = r.UInt16();
        if (version is not (1 or Version))
        {
            throw new InvalidDataException($"event log version {version}; this Idrak reads version 1 and version {Version}.");
        }

        int seed = r.Int32(), world = r.Int32();
        string failures = r.Text();
        var advice = ReadAdvice(ref r, version);
        if (world is < 1 or > 1 << 20)
        {
            throw new InvalidDataException($"an event log of {world} ranks.");
        }

        var streams = new List<byte[]>[world + 1];
        for (int i = 0; i <= world; i++)
        {
            streams[i] = [];
        }

        while (r.Remaining >= 8)
        {
            int stream = BinaryPrimitives.ReadInt32LittleEndian(r.Peek(8));
            int length = BinaryPrimitives.ReadInt32LittleEndian(r.Peek(8)[4..]);
            if (stream < -1 || stream >= world || length < 0 || r.Remaining - 8 < length)
            {
                break;                                                     // a chunk still being written
            }

            r.Take(8);
            streams[stream + 1].Add(r.Take(length).ToArray());
        }

        var output = new MemoryStream();
        Span<byte> field = stackalloc byte[8];
        output.Write(Magic);
        BinaryPrimitives.WriteUInt16LittleEndian(field, Version);
        output.Write(field[..2]);
        BinaryPrimitives.WriteInt32LittleEndian(field, seed);
        output.Write(field[..4]);
        BinaryPrimitives.WriteInt32LittleEndian(field, world);
        output.Write(field[..4]);
        int nameLength = Encoding.UTF8.GetByteCount(failures);
        BinaryPrimitives.WriteUInt16LittleEndian(field, (ushort)nameLength);
        output.Write(field[..2]);
        output.Write(Encoding.UTF8.GetBytes(failures));
        Span<byte> adviceField = stackalloc byte[AdviceBytes];
        WriteAdvice(adviceField, advice);
        output.Write(adviceField);
        foreach (var stream in streams)
        {
            BinaryPrimitives.WriteInt64LittleEndian(field, stream.Sum(c => (long)c.Length));
            output.Write(field);
            foreach (var chunk in stream)
            {
                output.Write(chunk);
            }
        }

        return output.ToArray();
    }

    // The high bit of a fault's kind: the fault was asked for while the run went (a live fault, phase C9). Logs without it read as before.
    internal const int LiveMark = 0x80;

    // Reads little-endian fields from the front of a span, refusing to read past its end.
    private ref struct Reader(ReadOnlySpan<byte> bytes)
    {
        private ReadOnlySpan<byte> _rest = bytes;

        public readonly bool Done => _rest.IsEmpty;

        public readonly int Remaining => _rest.Length;

        public readonly ReadOnlySpan<byte> Peek(int count) => _rest[..count];

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

// Writes a run's event log while it runs: the faults (under a lock of their own) and each rank's records (by that rank's
// thread alone) into pooled blocks. With a file (SimulationOptions.EventLog) and a flush interval, each rank appends its
// records since its last flush to the file's journal every few steps it begins (Flush, on its own thread), so the file is
// readable while the run goes and the blocks go back to the pool; Seal joins the journal and what is left into the
// finished log, the same bytes as without flushing, and writes it over the journal.
internal sealed class EventLogWriter : IDisposable
{
    private const int MaxDetailChars = 300;

    private readonly int _seed;
    private readonly int _worldSize;
    private readonly string _failures;
    private readonly Blocks _faults = new();
    private readonly Blocks[] _ranks;
    private readonly long[] _began;
    private readonly Lock _faultGate = new();
    private readonly Lock _fileGate = new();
    private readonly string? _path;
    private readonly int _flushSteps;
    private FileStream? _journal;

    private readonly (bool On, int Steps, int Slice) _advice;

    public EventLogWriter(int seed, int worldSize, string failures, bool transfers, string? path = null, int flushSteps = 0, (bool On, int Steps, int Slice) advice = default)
    {
        _advice = advice;
        _seed = seed;
        _worldSize = worldSize;
        _failures = failures;
        LogsTransfers = transfers;
        _ranks = [.. Enumerable.Range(0, worldSize).Select(_ => new Blocks())];
        _began = new long[worldSize];
        _path = path;
        _flushSteps = path is null ? 0 : Math.Max(0, flushSteps);
    }

    public bool LogsTransfers { get; }

    public void Fault(Fault fault, bool live = false)
    {
        lock (_faultGate)
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
    }

    public void Clock(int rank, long step, double seconds)
    {
        var b = _ranks[rank].Take(17);
        b[0] = EventLog.RecordClock;
        BinaryPrimitives.WriteInt64LittleEndian(b[1..], step);
        BinaryPrimitives.WriteDoubleLittleEndian(b[9..], seconds);
        if (_flushSteps > 0 && ++_began[rank] % _flushSteps == 0)
        {
            Flush(rank);
        }
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
        if (!LogsTransfers)
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

    // Appends rank `rank`'s records (and the faults drawn) since the last flush to the journal; called on that rank's thread.
    public void Flush(int rank)
    {
        if (_path is null)
        {
            return;
        }

        lock (_fileGate)
        {
            var journal = Journal();
            lock (_faultGate)
            {
                Append(journal, -1, _faults);
            }

            Append(journal, rank, _ranks[rank]);
            journal.Flush();
        }
    }

    // The log: the header, the faults, then each rank's records (joined from the journal when the run kept one).
    public EventLog Seal()
    {
        if (_journal is not null)
        {
            byte[] canonical;
            lock (_fileGate)
            {
                lock (_faultGate)
                {
                    Append(_journal, -1, _faults);
                }

                for (int rank = 0; rank < _worldSize; rank++)
                {
                    Append(_journal, rank, _ranks[rank]);
                }

                _journal.Flush();
                _journal.Position = 0;
                var journal = new byte[_journal.Length];
                _journal.ReadExactly(journal);
                _journal.Dispose();
                _journal = null;
                canonical = EventLog.FromJournal(journal);
                File.WriteAllBytes(_path!, canonical);
            }

            return EventLog.Parse(canonical);
        }

        int nameLength = Encoding.UTF8.GetByteCount(_failures);
        long total = EventLog.Magic.Length + 2 + 4 + 4 + 2 + nameLength + EventLog.AdviceBytes + 8 + _faults.Length + _ranks.Sum(s => 8 + s.Length);
        var bytes = new byte[checked((int)total)];
        var b = bytes.AsSpan();
        EventLog.Magic.CopyTo(b);
        int at = EventLog.Magic.Length;
        BinaryPrimitives.WriteUInt16LittleEndian(b[at..], EventLog.Version);
        BinaryPrimitives.WriteInt32LittleEndian(b[(at + 2)..], _seed);
        BinaryPrimitives.WriteInt32LittleEndian(b[(at + 6)..], _worldSize);
        BinaryPrimitives.WriteUInt16LittleEndian(b[(at + 10)..], (ushort)nameLength);
        at += 12 + Encoding.UTF8.GetBytes(_failures, b[(at + 12)..]);
        EventLog.WriteAdvice(b[at..], _advice);
        at += EventLog.AdviceBytes;
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
        _journal?.Dispose();
        _journal = null;
        _faults.Dispose();
        foreach (var stream in _ranks)
        {
            stream.Dispose();
        }
    }

    // The journal, made with its header on the first flush (readers may open it meanwhile).
    private FileStream Journal()
    {
        if (_journal is not null)
        {
            return _journal;
        }

        string? folder = Path.GetDirectoryName(Path.GetFullPath(_path!));
        if (folder is not null)
        {
            Directory.CreateDirectory(folder);
        }

        var journal = new FileStream(_path!, FileMode.Create, FileAccess.ReadWrite, FileShare.Read);
        int nameLength = Encoding.UTF8.GetByteCount(_failures);
        var header = new byte[EventLog.JournalMagic.Length + 2 + 4 + 4 + 2 + nameLength + EventLog.AdviceBytes];
        EventLog.JournalMagic.CopyTo(header);
        int at = EventLog.JournalMagic.Length;
        BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(at), EventLog.Version);
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(at + 2), _seed);
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(at + 6), _worldSize);
        BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(at + 10), (ushort)nameLength);
        int written = Encoding.UTF8.GetBytes(_failures, header.AsSpan(at + 12));
        EventLog.WriteAdvice(header.AsSpan(at + 12 + written), _advice);
        journal.Write(header);
        _journal = journal;
        return journal;
    }

    // One chunk of a stream's records, then the stream's blocks go back to the pool.
    private static void Append(FileStream journal, int stream, Blocks blocks)
    {
        if (blocks.Length == 0)
        {
            return;
        }

        Span<byte> head = stackalloc byte[8];
        BinaryPrimitives.WriteInt32LittleEndian(head, stream);
        BinaryPrimitives.WriteInt32LittleEndian(head[4..], checked((int)blocks.Length));
        journal.Write(head);
        blocks.WriteTo(journal);
        blocks.Dispose();
    }

    // Records in pooled blocks (rule 10: rented, returned when written to the journal or when the log is disposed).
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

        public void WriteTo(Stream stream)
        {
            foreach (var (block, used) in _blocks)
            {
                stream.Write(block, 0, used);
            }
        }

        public void Dispose()
        {
            foreach (var (block, _) in _blocks)
            {
                ArrayPool<byte>.Shared.Return(block);
            }

            _blocks.Clear();
            Length = 0;
        }
    }
}
