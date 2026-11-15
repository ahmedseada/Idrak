// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;

namespace Idrak.Cli.Commands.Developer.Stacks;

/// <summary>What a stack observation is: a sample profiler's sample in .NET code, one outside it, or another event a thread raised.</summary>
internal enum ObservationKind : byte
{
    /// <summary>An event the thread raised itself (an allocation report, a method compiled, a GC it started): it was running.</summary>
    Event = 0,

    /// <summary>The sample profiler found the thread outside .NET code: in a native call, the runtime or a wait.</summary>
    Outside = 1,

    /// <summary>The sample profiler found the thread running .NET code.</summary>
    Managed = 2,
}

/// <summary>A thread's stack at one time: its return addresses, leaf first.</summary>
internal readonly record struct Observation(long Thread, long Time, ObservationKind Kind, ulong[] Stack);

/// <summary>An allocation report (the runtime's, about one each 100 KB a thread allocates): the bytes since the last, their type and the stack.</summary>
internal readonly record struct AllocationReport(long Thread, long Time, string Type, long Bytes, ulong[] Stack);

/// <summary>A time the runtime stopped every thread running .NET code (Reason 1 or 6: a garbage collection).</summary>
internal readonly record struct Suspension(long Start, long End, int Reason)
{
    public bool ForCollection => Reason is 1 or 6;
}

/// <summary>A compiled method's code: its first address, its length, and its type and name.</summary>
internal readonly record struct MethodCode(ulong Start, uint Size, int Method);

/// <summary>
/// A .NET runtime trace (.nettrace: what the runtime writes when DOTNET_EnableEventPipe is set), read for what
/// <c>idrak trace --stacks</c> reports: every thread's stacks over time (the sample profiler's samples and every other
/// event that carries a stack, as Microsoft's TraceEvent reads them), the compiled methods' code (the runtime's method-load
/// events and its end-of-run listing), the allocation reports and the times the runtime stopped every thread.
/// </summary>
/// <remarks>
/// The format: "Nettrace", a FastSerialization header, a Trace object (the clock, the pointer size, the processors, the
/// sampling interval), then blocks (MetadataBlock, StackBlock, EventBlock, SPBlock) up to a null tag. Event headers are
/// compressed: each field is relative to the block's previous event (times may go backward: a 64-bit wrap). A stack is
/// resolved when its event is read, since the runtime reuses stack ids after a sequence point; method names are resolved
/// after the whole file, since the end-of-run listing comes last.
/// </remarks>
internal sealed class NetTrace
{
    private const byte NullReference = 1, BeginObject = 5, EndObject = 6;

    private const string SampleProfiler = "Microsoft-DotNETCore-SampleProfiler", Runtime = "Microsoft-Windows-DotNETRuntime",
        Rundown = "Microsoft-Windows-DotNETRuntimeRundown";

    private static readonly ulong[] NoStack = [];

    private NetTrace(string path)
    {
        Path = path;
    }

    /// <summary>The file read.</summary>
    public string Path { get; }

    /// <summary>The clock's ticks a second (the times' unit).</summary>
    public long Frequency { get; private set; }

    /// <summary>The traced process's id.</summary>
    public int ProcessId { get; private set; }

    /// <summary>The machine's processors, as the runtime counted them.</summary>
    public int Processors { get; private set; }

    /// <summary>The sample profiler's asked interval, in nanoseconds (the timer the system gives may be coarser).</summary>
    public int SamplingInterval { get; private set; }

    /// <summary>Every stack observation, in file order (times are ticks since the session started).</summary>
    public List<Observation> Observations { get; } = [];

    /// <summary>The allocation reports.</summary>
    public List<AllocationReport> Allocations { get; } = [];

    /// <summary>The times the runtime stopped every thread, in time order.</summary>
    public List<Suspension> Suspensions { get; } = [];

    /// <summary>The compiled methods' code, by first address.</summary>
    public MethodCode[] Code { get; private set; } = [];

    /// <summary>Each method's type (its namespace included) and name; index 0 is the unknown code.</summary>
    public List<(string Type, string Name)> Methods { get; } = [("", "[unknown code]")];

    /// <summary>Whether the runtime's end-of-run listing is in the file (without it, code compiled ahead of time has no names).</summary>
    public bool Listed { get; private set; }

    /// <summary>Seconds of <paramref name="ticks"/>.</summary>
    public double Seconds(long ticks) => (double)ticks / Frequency;

    /// <summary>Reads <paramref name="path"/>; an unreadable file fails naming the part it could not read.</summary>
    public static NetTrace Read(string path)
    {
        byte[] data;
        try
        {
            data = File.ReadAllBytes(path);
        }
        catch (FileNotFoundException)
        {
            throw new InvalidDataException($"No trace at {path}.");
        }

        var trace = new NetTrace(path);
        trace.Parse(data);
        return trace;
    }

    /// <summary>The method whose code holds <paramref name="address"/> (0: none known).</summary>
    public int MethodAt(ulong address)
    {
        var code = Code;
        int lo = 0, hi = code.Length - 1, found = -1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >>> 1;
            if (code[mid].Start <= address)
            {
                found = mid;
                lo = mid + 1;
            }
            else
            {
                hi = mid - 1;
            }
        }

        return found >= 0 && address - code[found].Start < code[found].Size ? code[found].Method : 0;
    }

    private void Parse(byte[] data)
    {
        var cursor = new Cursor(data, Path);
        if (data.Length < 8 || !data.AsSpan(0, 8).SequenceEqual("Nettrace"u8))
        {
            throw new InvalidDataException($"{Path} is not a .nettrace file (it does not start with \"Nettrace\").");
        }

        cursor.Skip(8);
        int length = cursor.Int32("the serialization header");
        if (!cursor.Bytes(length, "the serialization header").SequenceEqual("!FastSerialization.1"u8))
        {
            throw new InvalidDataException($"{Path}: the serialization header is not FastSerialization.1.");
        }

        cursor.Tag(BeginObject, "the trace header");
        var (type, version, minimum) = cursor.TypeHeader("the trace header");
        if (type != "Trace")
        {
            throw new InvalidDataException($"{Path}: the first object is {type}, not the trace header.");
        }

        if (version is < 4 or > 5 || minimum > 5)
        {
            throw new InvalidDataException($"{Path}: trace format version {version} (read from version {minimum}); this reader reads versions 4 and 5.");
        }

        cursor.Skip(16);                                                         // the session's start as a SYSTEMTIME
        long start = cursor.Int64("the trace header");
        Frequency = cursor.Int64("the trace header");
        int pointer = cursor.Int32("the trace header");
        ProcessId = cursor.Int32("the trace header");
        Processors = cursor.Int32("the trace header");
        SamplingInterval = cursor.Int32("the trace header");
        cursor.Tag(EndObject, "the trace header");
        if (pointer is not (4 or 8) || Frequency <= 0)
        {
            throw new InvalidDataException($"{Path}: the trace header gives {pointer}-byte pointers and a clock of {Frequency} ticks a second.");
        }

        var metadata = new Dictionary<uint, (string Provider, int Event, int Version)>();
        var stacks = new Dictionary<uint, ulong[]>();
        var code = new List<MethodCode>();
        var names = new Dictionary<(string, string), int>();
        var begins = new List<(long Time, int Reason)>();
        var ends = new List<long>();
        while (true)
        {
            byte tag = cursor.Byte("a block's tag");
            if (tag == NullReference)
            {
                break;
            }

            if (tag != BeginObject)
            {
                throw new InvalidDataException($"{Path}: byte {cursor.Position - 1} is tag {tag}, not a block's start.");
            }

            var (block, _, _) = cursor.TypeHeader("a block's header");
            int size = cursor.Int32(block);
            cursor.Align(4);
            var content = cursor.Bytes(size, block);
            cursor.Tag(EndObject, block);
            switch (block)
            {
                case "MetadataBlock":
                    Events(content, block, (meta, thread, stack, time, payload) =>
                    {
                        // The metadata id, the provider, then the event's id and name (and more this reader does not need).
                        uint id = BinaryPrimitives.ReadUInt32LittleEndian(payload);
                        int at = 4;
                        string provider = Utf16(payload, ref at, block);
                        int eventId = BinaryPrimitives.ReadInt32LittleEndian(payload[at..]);
                        at += 4;
                        _ = Utf16(payload, ref at, block);
                        int eventVersion = at + 12 <= payload.Length ? BinaryPrimitives.ReadInt32LittleEndian(payload[(at + 8)..]) : 0;
                        metadata[id] = (provider, eventId, eventVersion);
                    });
                    break;
                case "StackBlock":
                {
                    uint first = BinaryPrimitives.ReadUInt32LittleEndian(content);
                    uint count = BinaryPrimitives.ReadUInt32LittleEndian(content[4..]);
                    int at = 8;
                    for (uint i = 0; i < count; i++)
                    {
                        int bytes = BinaryPrimitives.ReadInt32LittleEndian(content[at..]);
                        at += 4;
                        if (bytes < 0 || at + bytes > content.Length)
                        {
                            throw new InvalidDataException($"{Path}: stack {first + i} runs past its block.");
                        }

                        var addresses = new ulong[bytes / pointer];
                        if (pointer == 8)
                        {
                            MemoryMarshal.Cast<byte, ulong>(content.Slice(at, bytes)).CopyTo(addresses);
                        }
                        else
                        {
                            for (int k = 0; k < addresses.Length; k++)
                            {
                                addresses[k] = BinaryPrimitives.ReadUInt32LittleEndian(content[(at + 4 * k)..]);
                            }
                        }

                        stacks[first + i] = addresses;
                        at += bytes;
                    }

                    break;
                }

                case "EventBlock":
                    Events(content, block, (meta, thread, stackId, time, payload) =>
                    {
                        long when = time - start;
                        var stack = stackId != 0 && stacks.TryGetValue(stackId, out var s) ? s : NoStack;
                        var (provider, eventId, eventVersion) = metadata.TryGetValue(meta, out var m) ? m : ("", -1, 0);
                        var kind = ObservationKind.Event;
                        if (provider == SampleProfiler && eventId == 0 && payload.Length >= 4)
                        {
                            kind = BinaryPrimitives.ReadUInt32LittleEndian(payload) == 2 ? ObservationKind.Managed : ObservationKind.Outside;
                        }
                        else if (provider == Runtime || provider == Rundown)
                        {
                            RuntimeEvent(provider, eventId, eventVersion, thread, when, stack, payload, code, names, begins, ends, pointer, block);
                        }

                        if (stack.Length > 0)
                        {
                            Observations.Add(new Observation(thread, when, kind, stack));
                        }
                    });
                    break;
                default:
                    break;                                                       // SPBlock (sequence points) and later kinds: not needed
            }
        }

        code.Sort((a, b) => a.Start.CompareTo(b.Start));
        Code = [.. code];
        begins.Sort((a, b) => a.Time.CompareTo(b.Time));
        ends.Sort();
        int e = 0;
        foreach (var (time, reason) in begins)
        {
            while (e < ends.Count && ends[e] < time)
            {
                e++;
            }

            Suspensions.Add(new Suspension(time, e < ends.Count ? ends[e++] : time, reason));
        }
    }

    // The runtime's events this reader reads: compiled methods (143, and the listing's 144), allocation reports (10), and
    // the suspensions' begin (9) and end (3).
    private void RuntimeEvent(string provider, int eventId, int version, long thread, long when, ulong[] stack, ReadOnlySpan<byte> payload, List<MethodCode> code,
        Dictionary<(string, string), int> names, List<(long, int)> begins, List<long> ends, int pointer, string block)
    {
        if (eventId == 143 && provider == Runtime || eventId == 144 && provider == Rundown)
        {
            // Method id, module id, start address, size, token, flags, then the type (namespace included), name and signature.
            if (payload.Length < 36)
            {
                return;
            }

            ulong address = BinaryPrimitives.ReadUInt64LittleEndian(payload[16..]);
            uint size = BinaryPrimitives.ReadUInt32LittleEndian(payload[24..]);
            int at = 36;
            string type = Utf16(payload, ref at, block), name = Utf16(payload, ref at, block);
            if (!names.TryGetValue((type, name), out int method))
            {
                method = Methods.Count;
                Methods.Add((type, name));
                names[(type, name)] = method;
            }

            code.Add(new MethodCode(address, size, method));
            Listed |= provider == Rundown;
        }
        else if (eventId == 10 && provider == Runtime && payload.Length >= 10)
        {
            // Bytes (32-bit), kind, instance; from version 2 the 64-bit bytes, the type's id and its name.
            long bytes = BinaryPrimitives.ReadUInt32LittleEndian(payload);
            string type = "?";
            if (version >= 2 && payload.Length >= 18 + pointer)
            {
                bytes = (long)BinaryPrimitives.ReadUInt64LittleEndian(payload[10..]);
                int at = 18 + pointer;
                type = Utf16(payload, ref at, block);
            }

            Allocations.Add(new AllocationReport(thread, when, type, bytes, stack));
        }
        else if (eventId == 9 && provider == Runtime)
        {
            begins.Add((when, payload.Length >= 4 ? BinaryPrimitives.ReadInt32LittleEndian(payload) : 0));
        }
        else if (eventId == 3 && provider == Runtime)
        {
            ends.Add(when);
        }
    }

    private delegate void EventHandler(uint metadata, long thread, uint stack, long time, ReadOnlySpan<byte> payload);

    // A block's events: its header (its size, then flags: bit 0, compressed headers), then each event's header and payload.
    private void Events(ReadOnlySpan<byte> block, string name, EventHandler handle)
    {
        if (block.Length < 4)
        {
            throw new InvalidDataException($"{Path}: a {name} is shorter than its header.");
        }

        int header = BinaryPrimitives.ReadInt16LittleEndian(block);
        int flags = BinaryPrimitives.ReadInt16LittleEndian(block[2..]);
        if ((flags & 1) == 0)
        {
            throw new InvalidDataException($"{Path}: a {name} without compressed event headers (an older runtime's format), which this reader does not read.");
        }

        // The running header: each field as the previous event left it, from zero at the block's start.
        uint meta = 0, stack = 0, size = 0;
        long thread = 0;
        ulong time = 0;
        int at = header;
        while (at < block.Length)
        {
            byte f = block[at++];
            if ((f & 1) != 0)
            {
                meta = (uint)VarUInt(block, ref at, name);
            }

            if ((f & 2) != 0)
            {
                _ = VarUInt(block, ref at, name);                               // the sequence number, the capturing thread, its processor
                _ = VarUInt(block, ref at, name);
                _ = VarUInt(block, ref at, name);
            }

            if ((f & 4) != 0)
            {
                thread = (long)VarUInt(block, ref at, name);
            }

            if ((f & 8) != 0)
            {
                stack = (uint)VarUInt(block, ref at, name);
            }

            time = unchecked(time + VarUInt(block, ref at, name));
            at += ((f & 16) != 0 ? 16 : 0) + ((f & 32) != 0 ? 16 : 0);           // the activity ids
            if ((f & 128) != 0)
            {
                size = (uint)VarUInt(block, ref at, name);
            }

            if (at + (long)size > block.Length)
            {
                throw new InvalidDataException($"{Path}: an event runs past the end of its {name}.");
            }

            handle(meta, thread, stack, (long)time, block.Slice(at, (int)size));
            at += (int)size;
        }
    }

    private ulong VarUInt(ReadOnlySpan<byte> block, ref int at, string name)
    {
        ulong value = 0;
        for (int shift = 0; shift < 64; shift += 7)
        {
            if (at >= block.Length)
            {
                throw new InvalidDataException($"{Path}: an event header runs past the end of its {name}.");
            }

            byte b = block[at++];
            value |= (ulong)(b & 0x7F) << shift;
            if (b < 0x80)
            {
                return value;
            }
        }

        throw new InvalidDataException($"{Path}: an event header in a {name} holds a number longer than 64 bits.");
    }

    // A null-terminated UTF-16 string.
    private string Utf16(ReadOnlySpan<byte> payload, ref int at, string name)
    {
        int end = at;
        while (end + 1 < payload.Length && (payload[end] | payload[end + 1]) != 0)
        {
            end += 2;
        }

        if (end + 1 >= payload.Length)
        {
            throw new InvalidDataException($"{Path}: a name in a {name} has no end.");
        }

        string text = Encoding.Unicode.GetString(payload[at..end]);
        at = end + 2;
        return text;
    }

    // The file read in order: each read names what it was reading where the file ends inside it.
    private ref struct Cursor(byte[] data, string path)
    {
        private readonly ReadOnlySpan<byte> _data = data;

        public int Position { get; private set; }

        public void Skip(int count) => Bytes(count, "the trace header");

        public void Align(int to) => Position = (Position + to - 1) / to * to;

        public ReadOnlySpan<byte> Bytes(int count, string what)
        {
            if (count < 0 || Position + (long)count > _data.Length)
            {
                throw new InvalidDataException($"{path}: the file ends inside {Article(what)} at byte {_data.Length} (the run may have stopped before the runtime finished writing it).");
            }

            var bytes = _data.Slice(Position, count);
            Position += count;
            return bytes;
        }

        public byte Byte(string what) => Bytes(1, what)[0];

        public int Int32(string what) => BinaryPrimitives.ReadInt32LittleEndian(Bytes(4, what));

        public long Int64(string what) => BinaryPrimitives.ReadInt64LittleEndian(Bytes(8, what));

        public void Tag(byte expected, string what)
        {
            int at = Position;
            byte tag = Byte(what);
            if (tag != expected)
            {
                throw new InvalidDataException($"{path}: byte {at} is tag {tag} where {what} needs {expected}.");
            }
        }

        // An object's type: begin, a null type-of-type, the version, the oldest reader's version, the name, end.
        public (string Name, int Version, int Minimum) TypeHeader(string what)
        {
            Tag(BeginObject, what);
            Tag(NullReference, what);
            int version = Int32(what), minimum = Int32(what), length = Int32(what);
            if (length is < 0 or > 256)
            {
                throw new InvalidDataException($"{path}: {what} at byte {Position} names a type of {length} characters.");
            }

            string name = Encoding.UTF8.GetString(Bytes(length, what));
            Tag(EndObject, what);
            return (name, version, minimum);
        }

        private static string Article(string what) => what.StartsWith("the ", StringComparison.Ordinal) || what.StartsWith("a ", StringComparison.Ordinal) ? what : $"a {what}";
    }
}
