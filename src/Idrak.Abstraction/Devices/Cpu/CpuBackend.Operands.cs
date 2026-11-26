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

    /// <summary>How an index operand's words hold its indices: int32 words, or float values.</summary>
    internal readonly struct Indices
    {
        private readonly float* _floats;
        private readonly int* _ints;

        public Indices(float* floats, int* ints)
        {
            _floats = floats;
            _ints = ints;
        }

        /// <summary>Index <paramref name="i"/> (a float value truncated toward zero; NaN reads as -1).</summary>
        public long this[long i] => _ints != null ? _ints[i] : float.IsNaN(_floats[i]) ? -1 : (long)_floats[i];

        /// <summary>Writes index <paramref name="value"/> at <paramref name="i"/> as the words hold indices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Set(long i, long value)
        {
            if (_ints != null)
            {
                _ints[i] = (int)value;
            }
            else
            {
                _floats[i] = value;
            }
        }
    }

    // Whether an index storage holds int32 words (its type's native kind Int32), else float values (the device's floats,
    // or a type native float32); any other type is refused.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool IntIndices(Storage s) => s.Typed && s.Type.Native switch
    {
        NativeKind.Int32 => true,
        NativeKind.Float32 => false,
        _ => throw new NotSupportedException($"The CPU reads indices as int32 or float32 values, not {s.Type.Name} (native kind {s.Type.Native})."),
    };

    // An index storage's words (pinned by the caller) as its indices.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Indices IndicesOf(Storage s, float* words) => IntIndices(s) ? new Indices(null, (int*)words) : new Indices(words, null);
}
