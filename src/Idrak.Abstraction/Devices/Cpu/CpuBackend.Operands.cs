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
