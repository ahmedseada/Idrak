// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Devices.Cpu;

// How the CPU's operations take their operands: through the operand's type (IElementType), never its name. A storage of
// the device's own floats, or of a type whose native kind is float32, is read and written by float32's own lane (V); an
// operand of another type is read through its type's description by the operation's described read where the operation
// has one (Gather's rows; the products of stored weights), else refused naming the operation and the type. Indices are
// read and written as their storage's type holds them (Indices): int32 words where the type's native kind is Int32,
// float values (exact up to 2^24) for the device's own floats.
internal sealed unsafe partial class CpuBackend
{
    // Whether the storage's values are float32's own: the device's floats, or a type whose native kind is float32.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool NativeFloats(Storage s) => !s.Typed || s.Type.Native == NativeKind.Float32;

    /// <summary>The storage's words for float32's own lane; another type is refused naming the operation (its described read not built).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[] V(Storage s) => NativeFloats(s) ? D(s) : throw NotNative(s);

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static NotSupportedException NotNative(Storage s) =>
        new($"The CPU's operation was given values of {s.Type.Name} (native kind {s.Type.Native}): this operation reads float32's own values; its described read for other types is not built yet.");

    /// <summary>How an index operand's words hold its indices: int32 or int64 words, or float values.</summary>
    internal readonly struct Indices
    {
        private readonly float* _floats;
        private readonly int* _ints;
        private readonly long* _longs;

        public Indices(float* floats, int* ints, long* longs = null)
        {
            _floats = floats;
            _ints = ints;
            _longs = longs;
        }

        /// <summary>Index <paramref name="i"/> (a float value truncated toward zero; NaN reads as -1; an int64 as it is).</summary>
        public long this[long i] => _ints != null ? _ints[i] : _longs != null ? _longs[i] : float.IsNaN(_floats[i]) ? -1 : (long)_floats[i];

        /// <summary>Writes index <paramref name="value"/> at <paramref name="i"/> as the words hold indices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Set(long i, long value)
        {
            if (_ints != null)
            {
                _ints[i] = (int)value;
            }
            else if (_longs != null)
            {
                _longs[i] = value;
            }
            else
            {
                _floats[i] = value;
            }
        }
    }

    // Whether an index storage holds int32 words (its type's native kind Int32), else float values (the device's floats,
    // or a type native float32); int64 words are read by IndicesOf; any other type is refused.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool IntIndices(Storage s) => s.Typed && s.Type.Native switch
    {
        NativeKind.Int32 => true,
        NativeKind.Float32 => false,
        _ => throw new NotSupportedException($"The CPU reads indices as int32, int64 or float32 values, not {s.Type.Name} (native kind {s.Type.Native})."),
    };

    // An index storage's words (pinned by the caller) as its indices: int32 or int64 words (never narrowed) or float values.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Indices IndicesOf(Storage s, float* words) => s.Typed && s.Type.Native == NativeKind.Int64 ? new Indices(null, null, (long*)words)
        : IntIndices(s) ? new Indices(null, (int*)words) : new Indices(words, null);

    // ------------------------------------------------------------------ typed inputs read a chunk at a time

    /// <summary>Values a typed input's chunk holds (64 KB of float32): the whole operand is never read wider at once.</summary>
    internal const int ChunkValues = 16384;

    /// <summary>
    /// Whether a type's values need float64 to be held exactly (a float of more than float32's 8 exponent or 23 mantissa
    /// bits, or an integer of more than 24 bits, as its code or any scale or minimum): such values are never read into
    /// float32 (no narrowing); the operations with a float64 lane read them there.
    /// </summary>
    internal static bool WideValues(IElementType type)
    {
        var d = type.Description;
        if (Wide(d.Code))
        {
            return true;
        }

        // By index (no enumerator made): asked on every call of an operation.
        var scales = d.Scales;
        for (int i = 0; scales is not null && i < scales.Count; i++)
        {
            if (Wide(scales[i].Type))
            {
                return true;
            }
        }

        var mins = d.Mins;
        for (int i = 0; mins is not null && i < mins.Count; i++)
        {
            if (Wide(mins[i].Type))
            {
                return true;
            }
        }

        return false;

        static bool Wide(NumberType t) => t.Kind switch
        {
            NumberKind.Float => t.Exponent > 8 || t.Mantissa > 23,
            NumberKind.Integer => t.Bits > 24,
            _ => false,
        };
    }

    // Each thread's pinned buffers for its operations' typed inputs (a slot an input), grown as asked, kept for the thread.
    [ThreadStatic]
    private static float[]?[]? t_stage;

    /// <summary>
    /// A typed input of an operation read a range of its values at a time into a buffer of its own (pinned, the thread's):
    /// through its type (IElementType.Read, the type's default), or, for a 16-bit float of a machine format, by its bits in
    /// the vector lanes where that is measured faster (the same bits: NativeRead); never the whole operand at once. A native
    /// float32 input is its own words.
    /// </summary>
    internal readonly struct TypedInput
    {
        private readonly Storage _storage;
        private readonly float* _words, _buffer;
        private readonly int _unit;

        /// <summary>The input <paramref name="s"/> (its words pinned by the caller at <paramref name="words"/>), read into slot <paramref name="slot"/> (0 to 3) for ranges of up to <paramref name="most"/> values.</summary>
        public TypedInput(Storage s, float* words, int slot, long most, string operation)
        {
            _storage = s;
            _words = words;
            if (NativeFloats(s))
            {
                return;
            }

            if (WideValues(s.Type))
            {
                throw new NotSupportedException($"The CPU's {operation} reads float32 values; {s.Type.Name}'s values need float64 (native kind {s.Type.Native}), "
                    + "which it never narrows: its float64 lane is not built.");
            }

            _unit = Math.Max(1, s.Type.Description.UnitValues);
            long size = most + _unit;
            if (size > int.MaxValue)
            {
                throw new NotSupportedException($"The CPU's {operation} reads {most} values of {s.Type.Name} at once: past one buffer.");
            }

            var buffers = t_stage ??= new float[4][];
            var buffer = buffers[slot];
            if (buffer is null || buffer.Length < size)
            {
                buffers[slot] = buffer = GC.AllocateUninitializedArray<float>((int)Math.Max(size, ChunkValues + _unit), pinned: true);
            }

            _buffer = (float*)Unsafe.AsPointer(ref System.Runtime.InteropServices.MemoryMarshal.GetArrayDataReference(buffer));
        }

        /// <summary>Whether the input is read through its type (else its words are float32's own).</summary>
        public bool Typed => _buffer != null;

        /// <summary>
        /// A pointer p such that p[i] is value i for i in [<paramref name="first"/>, <paramref name="first"/> +
        /// <paramref name="count"/>): the words themselves, or those values read into the buffer (from the unit holding
        /// <paramref name="first"/>), p placed so element i lands on its value.
        /// </summary>
        public float* Range(long first, long count)
        {
            if (_buffer == null)
            {
                return _words;
            }

            var type = _storage.Type;
            long start = first / _unit * _unit;
            int total = (int)(first + count - start);
            var bytes = new ReadOnlySpan<byte>((byte*)_words + type.ByteCount(start), (int)type.ByteCount(total));
            ReadValues(type, bytes, new Span<float>(_buffer, total));
            return _buffer - start;
        }
    }

    // ------------------------------------------------------------------ typed inputs read a range at a time where a kernel reads them

    // Each thread's pinned buffers for the ranges its kernels read (a slot an operand, and the slots of its working rows),
    // grown as asked and kept for the thread.
    [ThreadStatic]
    private static float[]?[]? t_ranges;

    // Slots a thread's kernels take: 0 to 7 an input's ranges (TypedRanges), 8 to 11 a kernel's working rows (WorkRow).
    private const int RangeSlots = 12;

    // The calling thread's pinned buffer of `slot`, at least `count` floats.
    private static float* ThreadBuffer(int slot, long count)
    {
        if (count > int.MaxValue)
        {
            throw new NotSupportedException($"The CPU reads {count} values of an operand at once: past one buffer.");
        }

        var buffers = t_ranges ??= new float[RangeSlots][];
        var buffer = buffers[slot];
        if (buffer is null || buffer.Length < count)
        {
            buffers[slot] = buffer = GC.AllocateUninitializedArray<float>((int)Math.Max(count, 256), pinned: true);
        }

        return (float*)Unsafe.AsPointer(ref System.Runtime.InteropServices.MemoryMarshal.GetArrayDataReference(buffer));
    }

    /// <summary>A working row of the calling thread (slot 0 to 3) of at least <paramref name="count"/> floats: a kernel's partial sums kept across the ranges it reads.</summary>
    internal static float* WorkRow(int slot, long count) => ThreadBuffer(8 + slot, count);

    /// <summary>
    /// An input an operation's kernels read a range at a time, where and when they read it (a row, a run of a group, a plane,
    /// a head's keys, a window of a strided view): float32's own words where they lie, or another type's values read
    /// through the type (its default read, or its native kind's lane where that is measured faster: the same bits) into
    /// the calling thread's own buffer of the operand's slot (0 to 7), from the unit holding the range's first value; never
    /// the whole operand at once. Each thread reads into its own buffers, so the kernels of every thread read at once; a
    /// range is valid until the same thread reads the same slot again.
    /// </summary>
    internal readonly struct TypedRanges
    {
        private readonly IElementType? _type;
        private readonly int _slot;
        private readonly long _unit, _count;

        /// <summary>The input <paramref name="s"/>, read into slot <paramref name="slot"/> where typed; refused where its values need float64 (named by <paramref name="operation"/>).</summary>
        public TypedRanges(Storage s, int slot, string operation)
        {
            _slot = slot;
            _count = s.ElementCount;
            if (NativeFloats(s))
            {
                return;
            }

            if (WideValues(s.Type))
            {
                throw new NotSupportedException($"The CPU's {operation} reads float32 values; {s.Type.Name}'s values need float64 (native kind {s.Type.Native}), "
                    + "which it never narrows: its float64 lane is not built.");
            }

            _type = s.Type;
            _unit = Math.Max(1, s.Type.Description.UnitValues);
        }

        /// <summary>Whether the input is read through its type (else its words are float32's own).</summary>
        public bool Typed => _type is not null;

        /// <summary>The values the input holds.</summary>
        public long Count => _count;

        /// <summary>
        /// A pointer p such that p[i] is value i for i in [<paramref name="first"/>, <paramref name="first"/> +
        /// <paramref name="count"/>): the words themselves (pinned by the caller at <paramref name="words"/>), or those
        /// values read into the thread's buffer, p placed so element i lands on its value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float* At(float* words, long first, long count) => _type is null ? words : Read(words, first, count);

        /// <summary>
        /// The array holding values [<paramref name="first"/>, <paramref name="first"/> + <paramref name="count"/>) at
        /// <paramref name="shift"/> + i for value i: the words themselves (shift 0), or the thread's buffer they were read
        /// into through the type. For kernels written over managed arrays.
        /// </summary>
        public float[] Values(float[] words, long first, long count, out long shift)
        {
            shift = 0;
            if (_type is null)
            {
                return words;
            }

            long start = first / _unit * _unit, total = first + count - start;
            ThreadBuffer(_slot, total);
            var buffer = t_ranges![_slot]!;
            var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(words.AsSpan()).Slice((int)_type.ByteCount(start), (int)_type.ByteCount(total));
            ReadValues(_type, bytes, buffer.AsSpan(0, (int)total));
            shift = -start;
            return buffer;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private float* Read(float* words, long first, long count)
        {
            long start = first / _unit * _unit;
            long total = first + count - start;
            float* buffer = ThreadBuffer(_slot, total);
            var type = _type!;
            var bytes = new ReadOnlySpan<byte>((byte*)words + type.ByteCount(start), (int)type.ByteCount(total));
            ReadValues(type, bytes, new Span<float>(buffer, (int)total));
            return buffer - start;
        }

        /// <summary>
        /// The input's value at <paramref name="at"/> through a window of up to <paramref name="window"/> values kept by
        /// the caller in [<paramref name="from"/>, <paramref name="to"/>) (<paramref name="values"/> placed as by
        /// <see cref="At"/>): read again from the unit holding <paramref name="at"/> when it falls outside, so a kernel
        /// whose addresses run on through the input reads each window once.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Windowed(float* words, long at, long window, ref float* values, ref long from, ref long to)
        {
            if (at < from || at >= to)
            {
                from = at / _unit * _unit;
                to = Math.Min(_count, from + Math.Max(window, _unit));
                values = At(words, from, to - from);
            }

            return values[at];
        }
    }

    /// <summary>
    /// The rows of an input from element <paramref name="offset"/> of its words (pinned at <paramref name="words"/>): a
    /// head's or a view's plane, read where it lies (float32's own) or through its type (<see cref="TypedRanges"/>).
    /// </summary>
    internal readonly struct RowsIn(TypedRanges reader, float* words, long offset)
    {
        /// <summary>Whether its values are read through their type.</summary>
        public bool Typed => reader.Typed;

        /// <summary>A pointer to element <paramref name="element"/> (from the offset) and the <paramref name="count"/> after it, as float32.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float* Row(long element, long count)
        {
            long at = offset + element;
            return reader.At(words, at, count) + at;
        }

        /// <summary>The rows from <paramref name="more"/> elements further on.</summary>
        public RowsIn Plus(long more) => new(reader, words, offset + more);
    }

    /// <summary>The elements an operation runs at a time: all <paramref name="n"/> where every input is float32's own, else a chunk.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static long Step(long n, bool typed) => typed ? ChunkValues : Math.Max(1, n);

    /// <summary>The rows of <paramref name="cols"/> an operation runs at a time: all <paramref name="rows"/> where every input is float32's own, else a chunk's worth (one row at least).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int RowStep(int rows, int cols, bool typed) => typed ? Math.Max(1, ChunkValues / Math.Max(1, cols)) : Math.Max(1, rows);

    /// <summary>
    /// Values [<paramref name="start"/>, + <paramref name="count"/>) of a typed storage's words into <paramref name="into"/>,
    /// read through its type from the unit holding the first (through <paramref name="temp"/>, of at least
    /// <paramref name="count"/> plus a unit's values, where the first is inside a unit). Thread-safe: nothing shared.
    /// </summary>
    internal static void ReadRange(IElementType type, float* words, long start, int count, float* into, float* temp)
    {
        long unit = Math.Max(1, type.Description.UnitValues), first = start / unit * unit;
        int total = (int)(start + count - first);
        var bytes = new ReadOnlySpan<byte>((byte*)words + type.ByteCount(first), (int)type.ByteCount(total));
        if (first == start)
        {
            ReadValues(type, bytes, new Span<float>(into, count));
            return;
        }

        ReadValues(type, bytes, new Span<float>(temp, total));
        new ReadOnlySpan<float>(temp + (start - first), count).CopyTo(new Span<float>(into, count));
    }

    // A typed range's values as float32: the type's default read, or its native kind's lane where measured faster.
    private static void ReadValues(IElementType type, ReadOnlySpan<byte> stored, Span<float> values)
    {
        var kind = type.Native;
        if (kind is NativeKind.BFloat16 or NativeKind.Float16 && NativeReadChosen(type, kind, stored, values.Length))
        {
            fixed (byte* s = stored)
            fixed (float* v = values)
            {
                NativeRead(type, kind, (ushort*)s, v, values.Length);
            }

            return;
        }

        type.Read(stored, values);
    }
}
