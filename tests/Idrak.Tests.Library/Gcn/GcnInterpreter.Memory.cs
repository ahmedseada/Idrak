// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Idrak.Tests.Gcn;

/// <summary>
/// Global memory as a flat byte space: storages mapped at addresses (from 4 GiB up, 1 MiB of unmapped space between
/// them, so a pointer cut to 32 bits or run past a storage lands nowhere). An access that is not wholly inside one
/// storage is reported, not ignored.
/// </summary>
internal sealed class GcnMemory
{
    /// <summary>A mapped storage, with its race tracking for the current dispatch (allocated when first touched).</summary>
    internal sealed class Storage(string name, ulong address, byte[] data)
    {
        public string Name => name;

        public ulong Address => address;

        public byte[] Data => data;

        public int[]? Writer { get; set; }

        public long[]? Reader { get; set; }
    }

    private readonly List<Storage> storages = [];
    private ulong next = 0x1_0000_0000;
    private Storage? last;

    /// <summary>Maps <paramref name="data"/> (the array itself: what the kernel writes lands in it) and returns its address.</summary>
    public ulong Map(string name, byte[] data)
    {
        var storage = new Storage(name, next, data);
        storages.Add(storage);
        next = (next + (ulong)data.Length + (1UL << 20) + 0xFFFF) & ~0xFFFFUL;
        return storage.Address;
    }

    /// <summary>Maps <paramref name="bytes"/> zeroed bytes.</summary>
    public ulong Map(string name, int bytes) => Map(name, new byte[bytes]);

    /// <summary>Maps a copy of <paramref name="values"/>.</summary>
    public ulong Map<T>(string name, ReadOnlySpan<T> values) where T : unmanaged => Map(name, MemoryMarshal.AsBytes(values).ToArray());

    /// <summary>The bytes of the storage mapped at <paramref name="address"/>.</summary>
    public byte[] Bytes(ulong address) => storages.First(s => s.Address == address).Data;

    /// <summary>A copy of <paramref name="count"/> values from <paramref name="address"/>.</summary>
    public T[] Read<T>(ulong address, int count) where T : unmanaged
    {
        var storage = Find(address, count * Marshal.SizeOf<T>()) ?? throw new ArgumentOutOfRangeException(nameof(address), $"0x{address:X}: {Describe(address)}");
        return MemoryMarshal.Cast<byte, T>(storage.Data.AsSpan((int)(address - storage.Address), count * Marshal.SizeOf<T>())).ToArray();
    }

    /// <summary>Unmaps the storage at <paramref name="address"/>.</summary>
    public void Unmap(ulong address)
    {
        storages.RemoveAll(s => s.Address == address);
        last = null;
    }

    /// <summary>The storage holding [address, address + size), or null.</summary>
    internal Storage? Find(ulong address, int size)
    {
        if (last is { } hit && address >= hit.Address && address + (ulong)size <= hit.Address + (ulong)hit.Data.Length)
        {
            return hit;
        }

        foreach (var storage in storages)
        {
            if (address >= storage.Address && address + (ulong)size <= storage.Address + (ulong)storage.Data.Length)
            {
                return last = storage;
            }
        }

        return null;
    }

    /// <summary>Where an address lies against the storages, for messages.</summary>
    internal string Describe(ulong address)
    {
        Storage? before = null, after = null;
        foreach (var s in storages)
        {
            if (s.Address <= address && (before is null || s.Address > before.Address))
            {
                before = s;
            }

            if (s.Address > address && (after is null || s.Address < after.Address))
            {
                after = s;
            }
        }

        if (before is not null && address < before.Address + (ulong)before.Data.Length)
        {
            return $"{address - before.Address} bytes into '{before.Name}' ({before.Data.Length} bytes), running past its end";
        }

        string text = before is null ? "below every storage" : $"{address - before.Address - (ulong)before.Data.Length} bytes past the end of '{before.Name}' ({before.Data.Length} bytes)";
        return after is null ? text : $"{text}, {after.Address - address} bytes before '{after.Name}'";
    }

    /// <summary>Starts a dispatch: the race tracking begins anew.</summary>
    internal void BeginDispatch()
    {
        foreach (var s in storages)
        {
            s.Writer = null;
            s.Reader = null;
        }
    }
}

internal sealed partial class GcnMachine
{
    // ---- Race tracking --------------------------------------------------------------------------------------------
    //
    // Per byte: the store that last wrote it (an index into Stores: its workgroup, wave, barrier interval, and whether a
    // wait retired it), and who read it (the workgroup, the barrier interval and a mask of the waves, or "several
    // workgroups"). Within a workgroup, another wave's bytes may be read or overwritten only after an s_barrier
    // separates the two and the store was retired by a wait; across workgroups, which run in no set order on a device,
    // no bytes one writes may be touched by another. Atomics are not tracked.

    private static long Reader(int group, int epoch, int mask) => (long)(group + 1) << 32 | (long)(epoch & 0xFFFF) << 16 | (uint)(mask & 0xFFFF);

    internal void TrackRead(int[] writer, long[] reader, int at, int size, GcnWave wave, GcnInstruction i, Func<int, string> where)
    {
        for (int b = at; b < at + size; b++)
        {
            int id = writer[b];
            if (id != 0)
            {
                var store = Stores[id];
                if (store.Group != Group)
                {
                    if (!Ordered(id, store, wave, i, read: true, out string why))
                    {
                        throw Fail(GcnFailure.Race, wave, i, $"it reads {where(b)}, which workgroup {store.Group} wrote ('{store.Instruction?.Text}' at 0x{store.Instruction?.Offset:X4}); workgroups run in no set order{why}");
                    }
                }
                else if (store.Wave != wave.Index)
                {
                    if (store.Epoch == Epoch)
                    {
                        throw Fail(GcnFailure.Race, wave, i,
                            $"it reads {where(b)}, which wave {store.Wave} wrote ('{store.Instruction?.Text}' at 0x{store.Instruction?.Offset:X4}) with no s_barrier between them");
                    }

                    if (!store.Retired)
                    {
                        throw Fail(GcnFailure.MissingWait, wave, i,
                            $"it reads {where(b)}, which wave {store.Wave}'s '{store.Instruction?.Text}' at 0x{store.Instruction?.Offset:X4} wrote, but no s_waitcnt retired that store before the s_barrier");
                    }
                }
            }

            long r = reader[b];
            int bit = 1 << (wave.Index & 15);
            if (r == 0)
            {
                reader[b] = Reader(Group, Epoch, bit);
            }
            else
            {
                int group = (int)(r >> 32) - 1;
                if (group == Group)
                {
                    reader[b] = ((r >> 16) & 0xFFFF) == (Epoch & 0xFFFF) ? r | (uint)bit : Reader(Group, Epoch, bit);
                }
                else if (group != -2)
                {
                    reader[b] = Reader(-2, 0, 0);
                }
            }
        }
    }

    internal void TrackWrite(int[] writer, long[] reader, int at, int size, GcnWave wave, GcnInstruction i, int store, Func<int, string> where)
    {
        for (int b = at; b < at + size; b++)
        {
            int id = writer[b];
            if (id != 0)
            {
                var other = Stores[id];
                if (other.Group != Group)
                {
                    if (!Ordered(id, other, wave, i, read: false, out string why))
                    {
                        throw Fail(GcnFailure.Race, wave, i, $"it writes {where(b)}, which workgroup {other.Group} also wrote ('{other.Instruction?.Text}' at 0x{other.Instruction?.Offset:X4}); workgroups run in no set order{why}");
                    }
                }
                else if (other.Wave != wave.Index && (other.Epoch == Epoch || !other.Retired))
                {
                    throw Fail(GcnFailure.Race, wave, i,
                        $"it writes {where(b)}, which wave {other.Wave} also wrote ('{other.Instruction?.Text}' at 0x{other.Instruction?.Offset:X4}) with no s_barrier (or no wait) between them");
                }
            }

            long r = reader[b];
            if (r != 0)
            {
                int group = (int)(r >> 32) - 1;
                if (group != Group && !(group >= 0 && acquired.TryGetValue(group, out var acquisition) && (acquisition.Wave == wave.Index || Epoch > acquisition.Epoch)))
                {
                    throw Fail(GcnFailure.Race, wave, i, $"it writes {where(b)}, which {(group == -2 ? "several workgroups" : $"workgroup {group}")} read; workgroups run in no set order");
                }

                int mask = (int)(r & 0xFFFF) & ~(1 << (wave.Index & 15));
                if (((r >> 16) & 0xFFFF) == (Epoch & 0xFFFF) && mask != 0)
                {
                    throw Fail(GcnFailure.Race, wave, i, $"it writes {where(b)}, which wave {BitOperations.TrailingZeroCount(mask)} read with no s_barrier between them");
                }
            }

            writer[b] = store;
        }
    }

    // ---- Global memory --------------------------------------------------------------------------------------------

    /// <summary>The storage and offset of [address, address + size), or an out-of-range failure.</summary>
    private GcnMemory.Storage Global(ulong address, int size, GcnWave wave, GcnInstruction i, int lane, out int offset)
    {
        var storage = memory.Find(address, size);
        if (storage is null)
        {
            string who = lane >= 0 ? $"lane {lane}" : "the wave";
            throw Fail(GcnFailure.OutOfRange, wave, i, $"{who} accesses {size} bytes at 0x{address:X}: {memory.Describe(address)}");
        }

        offset = (int)(address - storage.Address);
        return storage;
    }

    private void GlobalRead(ulong address, Span<byte> into, GcnWave wave, GcnInstruction i, int lane)
    {
        var storage = Global(address, into.Length, wave, i, lane, out int offset);
        storage.Data.AsSpan(offset, into.Length).CopyTo(into);
        if (TrackRaces)
        {
            storage.Writer ??= new int[storage.Data.Length];
            storage.Reader ??= new long[storage.Data.Length];
            TrackRead(storage.Writer, storage.Reader, offset, into.Length, wave, i, b => $"byte {b} of '{storage.Name}'");
        }
    }

    private void GlobalWrite(ulong address, ReadOnlySpan<byte> bytes, GcnWave wave, GcnInstruction i, int lane, int store)
    {
        var storage = Global(address, bytes.Length, wave, i, lane, out int offset);
        if (TrackRaces)
        {
            storage.Writer ??= new int[storage.Data.Length];
            storage.Reader ??= new long[storage.Data.Length];
            TrackWrite(storage.Writer, storage.Reader, offset, bytes.Length, wave, i, store, b => $"byte {b} of '{storage.Name}'");
        }

        bytes.CopyTo(storage.Data.AsSpan(offset));
    }

    // ---- SMEM -----------------------------------------------------------------------------------------------------

    internal static void ScalarLoad(GcnMachine m, GcnWave w, GcnInstruction i)
    {
        string name = i.Opcode.Base;
        int dwords = name.EndsWith("x16", StringComparison.Ordinal) ? 16 : name.EndsWith("x8", StringComparison.Ordinal) ? 8
            : name.EndsWith("x4", StringComparison.Ordinal) ? 4 : name.EndsWith("x2", StringComparison.Ordinal) ? 2 : 1;
        uint offset = i.Imm ? (uint)i.MemoryOffset : m.Scalar32(w, i.MemoryOffset, i);
        ulong address;
        if (name.StartsWith("s_buffer_load", StringComparison.Ordinal))
        {
            if (i.A % 4 != 0)
            {
                throw m.Fail(GcnFailure.Register, w, i, "a buffer resource is four SGPRs starting at a multiple of 4");
            }

            ulong low = m.ReadScalar64(w, i.A, i);
            uint records = m.ReadScalar(w, i.A + 2, i);
            _ = m.ReadScalar(w, i.A + 3, i);
            if (((low >> 48) & 0x3FFF) != 0)
            {
                throw m.Fail(GcnFailure.NotModelled, w, i, "a buffer resource with a stride");
            }

            if ((ulong)(offset & ~3u) + (ulong)(4 * dwords) > records)
            {
                throw m.Fail(GcnFailure.OutOfRange, w, i, $"it reads bytes {offset & ~3u} to {(offset & ~3u) + 4 * dwords} of a buffer of {records} bytes");
            }

            address = (low & 0xFFFFFFFFFFFF) + (offset & ~3u);
        }
        else
        {
            address = (m.ReadScalar64(w, i.A, i) + offset) & ~3UL;
        }

        int align = Math.Min(dwords, 4);
        if (i.D % align != 0 || i.D + dwords > 102 && i.D < 102)
        {
            throw m.Fail(GcnFailure.Register, w, i, $"s[{i.D}:{i.D + dwords - 1}]: {dwords} SGPRs start at a multiple of {align}, within s0-s101");
        }

        var values = new uint[dwords];
        Span<byte> bytes = stackalloc byte[4];
        for (int k = 0; k < dwords; k++)
        {
            m.GlobalRead(address + (ulong)(4 * k), bytes, w, i, -1);
            values[k] = BinaryPrimitives.ReadUInt32LittleEndian(bytes);
        }

        m.Issue(w, new GcnPending { Instruction = i, Lgkm = true, SFirst = i.D, SCount = dwords, SValues = values });
    }

    internal static void ScalarTime(GcnMachine m, GcnWave w, GcnInstruction i)
    {
        ulong time = (ulong)m.instructions;
        m.CheckPair(w, i.D, i);
        m.Issue(w, new GcnPending { Instruction = i, Lgkm = true, SFirst = i.D, SCount = 2, SValues = [(uint)time, (uint)(time >> 32)] });
    }

    // ---- LDS ------------------------------------------------------------------------------------------------------

    private int LdsAddress(uint address, int size, int align, GcnWave wave, GcnInstruction i, int lane)
    {
        if (i.Gds)
        {
            throw Fail(GcnFailure.NotModelled, wave, i, "GDS");
        }

        if ((ulong)address + (ulong)size > (ulong)Lds.Length)
        {
            throw Fail(GcnFailure.OutOfRange, wave, i, $"lane {lane} accesses LDS bytes {address} to {(ulong)address + (ulong)size - 1}, past the workgroup's {Lds.Length} bytes");
        }

        if (address % (uint)align != 0)
        {
            throw Fail(GcnFailure.OutOfRange, wave, i, $"lane {lane} accesses {size} LDS bytes at {address}, not {align}-byte aligned");
        }

        LdsTouched = Math.Max(LdsTouched, (int)address + size);
        return (int)address;
    }

    private void LdsRead(int at, Span<byte> into, GcnWave wave, GcnInstruction i)
    {
        Lds.AsSpan(at, into.Length).CopyTo(into);
        if (TrackRaces)
        {
            TrackRead(LdsWriter, LdsReader, at, into.Length, wave, i, b => $"LDS byte {b}");
        }
    }

    private void LdsWrite(int at, ReadOnlySpan<byte> bytes, GcnWave wave, GcnInstruction i, int store)
    {
        if (TrackRaces)
        {
            TrackWrite(LdsWriter, LdsReader, at, bytes.Length, wave, i, store, b => $"LDS byte {b}");
        }

        bytes.CopyTo(Lds.AsSpan(at));
    }

    private static int Alignment(int size) => size >= 12 ? 16 : size;

    /// <summary>ds_read_*: per active lane, from address + offset; the destination pending until lgkmcnt retires it.</summary>
    internal static void LdsLoad(GcnMachine m, GcnWave w, GcnInstruction i)
    {
        string name = i.Opcode.Base;
        ulong exec = m.ExecOf(w, i);
        m.ReadVector(w, i.A - 256, i, m.L0);
        bool two = name.StartsWith("ds_read2", StringComparison.Ordinal);
        int size = name.EndsWith("b128", StringComparison.Ordinal) ? 16 : name.EndsWith("b96", StringComparison.Ordinal) ? 12 : name.EndsWith("b64", StringComparison.Ordinal) ? 8
            : name.Contains("8", StringComparison.Ordinal) ? 1 : name.Contains("16", StringComparison.Ordinal) ? 2 : 4;
        int registers = two ? 2 * Math.Max(1, size / 4) : Math.Max(1, size / 4);
        var values = new uint[registers * 64];
        int merge = name.EndsWith("_d16_hi", StringComparison.Ordinal) ? 2 : name.EndsWith("_d16", StringComparison.Ordinal) ? 1 : 0;
        bool signed = name.Contains("_i8", StringComparison.Ordinal) || name.Contains("_i16", StringComparison.Ordinal);
        Span<byte> bytes = stackalloc byte[16];
        for (ulong mask = exec; mask != 0; mask &= mask - 1)
        {
            int lane = BitOperations.TrailingZeroCount(mask);
            uint address = m.L0[lane];
            int parts = two ? 2 : 1;
            for (int p = 0; p < parts; p++)
            {
                int scale = two ? size * (name.Contains("st64", StringComparison.Ordinal) ? 64 : 1) : 1;
                uint at = two ? address + (uint)((p == 0 ? i.Offset0 : i.Offset1) * scale) : address + (uint)i.MemoryOffset;
                int where = m.LdsAddress(at, size, Alignment(size), w, i, lane);
                m.LdsRead(where, bytes[..size], w, i);
                int regs = Math.Max(1, size / 4);
                for (int k = 0; k < regs; k++)
                {
                    uint value = size switch
                    {
                        1 => signed ? (uint)(sbyte)bytes[0] : bytes[0],
                        2 => signed ? (uint)BinaryPrimitives.ReadInt16LittleEndian(bytes) : BinaryPrimitives.ReadUInt16LittleEndian(bytes),
                        _ => BinaryPrimitives.ReadUInt32LittleEndian(bytes[(4 * k)..]),
                    };
                    values[(p * regs + k) * 64 + lane] = value;
                }
            }
        }

        m.Issue(w, new GcnPending { Instruction = i, Lgkm = true, InOrder = true, VFirst = i.D - 256, VCount = registers, VValues = values, Lanes = exec, Merge = merge });
    }

    /// <summary>ds_write_*: per active lane, at once in the LDS; counted by lgkmcnt until a wait retires it (other waves may read it only then).</summary>
    internal static void LdsStore(GcnMachine m, GcnWave w, GcnInstruction i)
    {
        string name = i.Opcode.Base;
        ulong exec = m.ExecOf(w, i);
        m.ReadVector(w, i.A - 256, i, m.L0);
        bool two = name.StartsWith("ds_write2", StringComparison.Ordinal);
        int size = name.EndsWith("b128", StringComparison.Ordinal) ? 16 : name.EndsWith("b96", StringComparison.Ordinal) ? 12 : name.EndsWith("b64", StringComparison.Ordinal) ? 8
            : name.Contains("_b8", StringComparison.Ordinal) ? 1 : name.Contains("_b16", StringComparison.Ordinal) ? 2 : 4;
        bool high = name.EndsWith("_d16_hi", StringComparison.Ordinal);
        int regs = Math.Max(1, size / 4);
        var data = new uint[2 * regs * 64];
        for (int k = 0; k < regs; k++)
        {
            m.ReadVector(w, i.S0 - 256 + k, i, data.AsSpan(k * 64, 64));
            if (two)
            {
                m.ReadVector(w, i.S1 - 256 + k, i, data.AsSpan((regs + k) * 64, 64));
            }
        }

        int store = m.NewStore(w, i, out var record);
        Span<byte> bytes = stackalloc byte[16];
        for (ulong mask = exec; mask != 0; mask &= mask - 1)
        {
            int lane = BitOperations.TrailingZeroCount(mask);
            int parts = two ? 2 : 1;
            for (int p = 0; p < parts; p++)
            {
                int scale = two ? size * (name.Contains("st64", StringComparison.Ordinal) ? 64 : 1) : 1;
                uint at = two ? m.L0[lane] + (uint)((p == 0 ? i.Offset0 : i.Offset1) * scale) : m.L0[lane] + (uint)i.MemoryOffset;
                int where = m.LdsAddress(at, size, Alignment(size), w, i, lane);
                for (int k = 0; k < regs; k++)
                {
                    uint value = data[(p * regs + k) * 64 + lane];
                    if (high)
                    {
                        value >>= 16;
                    }

                    BinaryPrimitives.WriteUInt32LittleEndian(bytes[(4 * k)..], value);
                }

                m.LdsWrite(where, bytes[..size], w, i, store);
            }
        }

        m.Issue(w, new GcnPending { Instruction = i, Lgkm = true, InOrder = true, Store = record });
    }

    /// <summary>ds_&lt;op&gt;[_rtn]_{u32,i32,b32,f32,u64}: atomic per lane in lane order; the _rtn forms return the old value.</summary>
    internal static void LdsAtomic(GcnMachine m, GcnWave w, GcnInstruction i)
    {
        string name = i.Opcode.Base;
        bool returns = name.Contains("_rtn_", StringComparison.Ordinal);
        string op = name[3..name.IndexOf('_', 3)];
        bool wide = name.EndsWith("64", StringComparison.Ordinal);
        string type = name[^3..];
        if (op is "min" or "max" && type is "i32" or "i64")
        {
            op = "s" + op;
        }

        ulong exec = m.ExecOf(w, i);
        m.ReadVector(w, i.A - 256, i, m.L0);
        m.ReadVector(w, i.S0 - 256, i, m.L1);
        if (wide)
        {
            m.ReadVector(w, i.S0 - 255, i, m.L2);
        }

        bool two = op is "cmpst" or "mskor";
        if (two)
        {
            m.ReadVector(w, i.S1 - 256, i, m.L3);
        }

        int size = wide ? 8 : 4;
        int regs = wide ? 2 : 1;
        var old = returns ? new uint[regs * 64] : null;
        Span<byte> bytes = stackalloc byte[8];
        for (ulong mask = exec; mask != 0; mask &= mask - 1)
        {
            int lane = BitOperations.TrailingZeroCount(mask);
            int at = m.LdsAddress(m.L0[lane] + (uint)i.MemoryOffset, size, size, w, i, lane);
            var cell = m.Lds.AsSpan(at, size);
            ulong before = wide ? BinaryPrimitives.ReadUInt64LittleEndian(cell) : BinaryPrimitives.ReadUInt32LittleEndian(cell);
            ulong data = m.L1[lane] | (wide ? (ulong)m.L2[lane] << 32 : 0);
            ulong after = Atomic(m, w, i, op, type, before, data, two ? m.L3[lane] : 0);
            if (wide)
            {
                BinaryPrimitives.WriteUInt64LittleEndian(cell, after);
            }
            else
            {
                BinaryPrimitives.WriteUInt32LittleEndian(cell, (uint)after);
            }

            if (old is not null)
            {
                old[lane] = (uint)before;
                if (wide)
                {
                    old[64 + lane] = (uint)(before >> 32);
                }
            }
        }

        m.Issue(w, new GcnPending { Instruction = i, Lgkm = true, InOrder = true, VFirst = returns ? i.D - 256 : -1, VCount = regs, VValues = old, Lanes = exec });
    }

    private static ulong Atomic(GcnMachine m, GcnWave w, GcnInstruction i, string op, string type, ulong before, ulong data, ulong data2)
    {
        bool f32 = type == "f32";
        switch (op)
        {
            case "add" when f32:
                return BitConverter.SingleToUInt32Bits(BitConverter.UInt32BitsToSingle((uint)before) + BitConverter.UInt32BitsToSingle((uint)data));
            case "add":
                return before + data;
            case "sub":
                return before - data;
            case "rsub":
                return data - before;
            case "inc":
                return before >= data ? 0 : before + 1;
            case "dec":
                return before == 0 || before > data ? data : before - 1;
            case "smin":
                return type == "i32" ? (uint)Math.Min((int)before, (int)data) : (ulong)Math.Min((long)before, (long)data);
            case "smax":
                return type == "i32" ? (uint)Math.Max((int)before, (int)data) : (ulong)Math.Max((long)before, (long)data);
            case "min" or "umin":
                return f32 ? BitConverter.SingleToUInt32Bits(GcnFloat.Min(BitConverter.UInt32BitsToSingle((uint)before), BitConverter.UInt32BitsToSingle((uint)data))) : Math.Min(before, data);
            case "max" or "umax":
                return f32 ? BitConverter.SingleToUInt32Bits(GcnFloat.Max(BitConverter.UInt32BitsToSingle((uint)before), BitConverter.UInt32BitsToSingle((uint)data))) : Math.Max(before, data);
            case "and":
                return before & data;
            case "or":
                return before | data;
            case "xor":
                return before ^ data;
            case "wrxchg" or "swap":
                return data;
            case "mskor":
                return (before & ~data) | data2;
            case "cmpst":
                return before == data ? data2 : before;
            default:
                throw m.Fail(GcnFailure.NotModelled, w, i, $"the atomic {op}");
        }
    }

    /// <summary>ds_swizzle_b32, ds_permute_b32, ds_bpermute_b32: lanes exchanged through the LDS hardware, counted by lgkmcnt.</summary>
    internal static void LdsLanes(GcnMachine m, GcnWave w, GcnInstruction i)
    {
        string name = i.Opcode.Base;
        ulong exec = m.ExecOf(w, i);
        var result = new uint[64];
        if (name == "ds_swizzle_b32")
        {
            m.ReadVector(w, i.A - 256, i, m.L1);
            int imm = i.MemoryOffset;
            if ((imm & 0x8000) != 0 && (imm & 0xFF00) != 0x8000)
            {
                throw m.Fail(GcnFailure.NotModelled, w, i, $"swizzle offset 0x{imm:X4}");
            }

            for (ulong mask = exec; mask != 0; mask &= mask - 1)
            {
                int lane = BitOperations.TrailingZeroCount(mask);
                int source;
                if ((imm & 0x8000) != 0)
                {
                    source = (lane & ~3) | (imm >> (2 * (lane & 3)) & 3);
                }
                else
                {
                    int and = imm & 0x1F, or = (imm >> 5) & 0x1F, xor = (imm >> 10) & 0x1F;
                    source = (lane & 0x20) | (((lane & 0x1F & and) | or) ^ xor);
                }

                result[lane] = (exec >> source & 1) != 0 ? m.L1[source] : 0;
            }
        }
        else
        {
            m.ReadVector(w, i.A - 256, i, m.L0);
            m.ReadVector(w, i.S0 - 256, i, m.L1);
            if (name == "ds_permute_b32")
            {
                var forward = new uint[64];
                for (ulong mask = exec; mask != 0; mask &= mask - 1)
                {
                    int lane = BitOperations.TrailingZeroCount(mask);
                    forward[(int)((m.L0[lane] + (uint)i.MemoryOffset) >> 2) & 63] = m.L1[lane];
                }

                for (ulong mask = exec; mask != 0; mask &= mask - 1)
                {
                    int lane = BitOperations.TrailingZeroCount(mask);
                    result[lane] = forward[lane];
                }
            }
            else
            {
                for (ulong mask = exec; mask != 0; mask &= mask - 1)
                {
                    int lane = BitOperations.TrailingZeroCount(mask);
                    int source = (int)((m.L0[lane] + (uint)i.MemoryOffset) >> 2) & 63;
                    result[lane] = (exec >> source & 1) != 0 ? m.L1[source] : 0;
                }
            }
        }

        m.Issue(w, new GcnPending { Instruction = i, Lgkm = true, InOrder = true, VFirst = i.D - 256, VCount = 1, VValues = result, Lanes = exec });
    }

    // ---- FLAT and GLOBAL ------------------------------------------------------------------------------------------

    private void Addresses(GcnWave wave, GcnInstruction i, ulong exec, Span<ulong> addresses)
    {
        if (i.Opcode.Format == GcnFormat.Scratch)
        {
            throw Fail(GcnFailure.NotModelled, wave, i, "scratch memory");
        }

        if (i.SAddr >= 0)
        {
            ulong baseAddress = ReadScalar64(wave, i.SAddr, i);
            ReadVector(wave, i.A - 256, i, L0);
            for (ulong mask = exec; mask != 0; mask &= mask - 1)
            {
                int lane = BitOperations.TrailingZeroCount(mask);
                addresses[lane] = baseAddress + L0[lane] + (ulong)(long)i.MemoryOffset;
            }
        }
        else
        {
            ReadVector(wave, i.A - 256, i, L0);
            ReadVector(wave, i.A - 255, i, L1);
            for (ulong mask = exec; mask != 0; mask &= mask - 1)
            {
                int lane = BitOperations.TrailingZeroCount(mask);
                addresses[lane] = (L0[lane] | (ulong)L1[lane] << 32) + (ulong)(long)i.MemoryOffset;
            }
        }
    }

    private static (int Size, int Registers, bool Signed, int Merge) Shape(string name)
    {
        int merge = name.EndsWith("_d16_hi", StringComparison.Ordinal) ? 2 : name.EndsWith("_d16", StringComparison.Ordinal) ? 1 : 0;
        bool signed = name.Contains("_sbyte", StringComparison.Ordinal) || name.Contains("_sshort", StringComparison.Ordinal);
        if (name.Contains("byte", StringComparison.Ordinal))
        {
            return (1, 1, signed, merge);
        }

        if (name.Contains("short", StringComparison.Ordinal))
        {
            return (2, 1, signed, merge);
        }

        int regs = name.EndsWith("x4", StringComparison.Ordinal) ? 4 : name.EndsWith("x3", StringComparison.Ordinal) ? 3 : name.EndsWith("x2", StringComparison.Ordinal) ? 2 : 1;
        return (4 * regs, regs, false, 0);
    }

    /// <summary>global_load_* and flat_load_*: per active lane; the destination pending until vmcnt (FLAT: or lgkmcnt) retires it.</summary>
    internal static void VectorLoad(GcnMachine m, GcnWave w, GcnInstruction i)
    {
        ulong exec = m.ExecOf(w, i);
        Span<ulong> addresses = stackalloc ulong[64];
        m.Addresses(w, i, exec, addresses);
        var (size, regs, signed, merge) = Shape(i.Opcode.Base);
        var values = new uint[regs * 64];
        Span<byte> bytes = stackalloc byte[16];
        for (ulong mask = exec; mask != 0; mask &= mask - 1)
        {
            int lane = BitOperations.TrailingZeroCount(mask);
            m.GlobalRead(addresses[lane], bytes[..size], w, i, lane);
            for (int k = 0; k < regs; k++)
            {
                values[k * 64 + lane] = size switch
                {
                    1 => signed ? (uint)(sbyte)bytes[0] : bytes[0],
                    2 => signed ? (uint)BinaryPrimitives.ReadInt16LittleEndian(bytes) : BinaryPrimitives.ReadUInt16LittleEndian(bytes),
                    _ => BinaryPrimitives.ReadUInt32LittleEndian(bytes[(4 * k)..]),
                };
            }
        }

        bool flat = i.Opcode.Format == GcnFormat.Flat;
        m.Issue(w, new GcnPending { Instruction = i, Vm = true, Lgkm = flat, VFirst = i.D - 256, VCount = regs, VValues = values, Lanes = exec, Merge = merge });
    }

    /// <summary>global_store_* and flat_store_*: per active lane, written at once; counted by vmcnt until a wait retires it.</summary>
    internal static void VectorStore(GcnMachine m, GcnWave w, GcnInstruction i)
    {
        ulong exec = m.ExecOf(w, i);
        Span<ulong> addresses = stackalloc ulong[64];
        m.Addresses(w, i, exec, addresses);
        var (size, regs, _, _) = Shape(i.Opcode.Base);
        bool high = i.Opcode.Base.EndsWith("_d16_hi", StringComparison.Ordinal);
        var data = new uint[regs * 64];
        for (int k = 0; k < regs; k++)
        {
            m.ReadVector(w, i.S0 - 256 + k, i, data.AsSpan(k * 64, 64));
        }

        int store = m.NewStore(w, i, out var record, global: true);
        Span<byte> bytes = stackalloc byte[16];
        for (ulong mask = exec; mask != 0; mask &= mask - 1)
        {
            int lane = BitOperations.TrailingZeroCount(mask);
            for (int k = 0; k < regs; k++)
            {
                uint value = data[k * 64 + lane];
                BinaryPrimitives.WriteUInt32LittleEndian(bytes[(4 * k)..], high ? value >> 16 : value);
            }

            m.GlobalWrite(addresses[lane], bytes[..size], w, i, lane, store);
        }

        m.Issue(w, new GcnPending { Instruction = i, Vm = true, Lgkm = i.Opcode.Format == GcnFormat.Flat, Store = record });
    }

    /// <summary>global_atomic_* and flat_atomic_*: atomic per lane in lane order; with glc the old value returns.</summary>
    internal static void VectorAtomic(GcnMachine m, GcnWave w, GcnInstruction i)
    {
        string name = i.Opcode.Base;
        ulong exec = m.ExecOf(w, i);
        Span<ulong> addresses = stackalloc ulong[64];
        m.Addresses(w, i, exec, addresses);
        int at = name.IndexOf("_atomic_", StringComparison.Ordinal) + 8;
        bool wide = name.EndsWith("_x2", StringComparison.Ordinal);
        string op = wide ? name[at..^3] : name[at..];
        string type = "u32";
        if (op.EndsWith("_f32", StringComparison.Ordinal))
        {
            op = op[..^4];
            type = "f32";
        }
        else if (op.StartsWith('s') && op is "smin" or "smax")
        {
            type = wide ? "i64" : "i32";
        }

        if (op == "pk_add_f16")
        {
            throw m.Fail(GcnFailure.NotModelled, w, i, "packed half atomics");
        }

        int size = wide ? 8 : 4;
        bool cmpswap = op == "cmpswap";
        int dataRegs = (wide ? 2 : 1) * (cmpswap ? 2 : 1);
        var data = new uint[dataRegs * 64];
        for (int k = 0; k < dataRegs; k++)
        {
            m.ReadVector(w, i.S0 - 256 + k, i, data.AsSpan(k * 64, 64));
        }

        int regs = wide ? 2 : 1;
        var old = i.Glc ? new uint[regs * 64] : null;
        Span<byte> bytes = stackalloc byte[8];
        for (ulong mask = exec; mask != 0; mask &= mask - 1)
        {
            int lane = BitOperations.TrailingZeroCount(mask);
            var storage = m.Global(addresses[lane], size, w, i, lane, out int offset);
            var cell = storage.Data.AsSpan(offset, size);
            ulong before = wide ? BinaryPrimitives.ReadUInt64LittleEndian(cell) : BinaryPrimitives.ReadUInt32LittleEndian(cell);
            ulong value = data[lane] | (wide ? (ulong)data[64 + lane] << 32 : 0);
            ulong compare = 0;
            if (cmpswap)
            {
                int c = wide ? 2 : 1;
                compare = data[c * 64 + lane] | (wide ? (ulong)data[(c + 1) * 64 + lane] << 32 : 0);
            }

            ulong after = cmpswap ? (before == compare ? value : before) : Atomic(m, w, i, op == "swap" ? "wrxchg" : op, type, before, value, 0);
            if (!wide)
            {
                after = (uint)after;
            }

            if (wide)
            {
                BinaryPrimitives.WriteUInt64LittleEndian(cell, after);
            }
            else
            {
                BinaryPrimitives.WriteUInt32LittleEndian(cell, (uint)after);
            }

            if (old is not null)
            {
                old[lane] = (uint)before;
                if (wide)
                {
                    old[64 + lane] = (uint)(before >> 32);
                }
            }
        }

        // A completion counter: the atomic releases this workgroup's retired stores; with glc it returns the count, and
        // once a wait retires it the workgroup acquires what the others released (CUDA's membar.gl and atom.global.add).
        var touched = new List<ulong>();
        for (ulong mask = exec; mask != 0; mask &= mask - 1)
        {
            ulong address = addresses[BitOperations.TrailingZeroCount(mask)];
            m.Release(address);
            touched.Add(address);
        }

        bool flat = i.Opcode.Format == GcnFormat.Flat;
        m.Issue(w, new GcnPending
        {
            Instruction = i, Vm = true, Lgkm = flat, VFirst = old is null ? -1 : i.D - 256, VCount = regs, VValues = old, Lanes = exec,
            OnRetire = old is null ? null : wave => touched.ForEach(a => m.Acquire(a, wave)),
        });
    }
}
