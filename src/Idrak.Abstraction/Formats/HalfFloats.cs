// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;
using System.Runtime.InteropServices;

namespace Idrak.Abstraction.Formats;

/// <summary>
/// IEEE halves (float16) widened to float32, exactly for every bit pattern: the CPU device's float16 weights and the
/// checkpoint readers' float16 tensors both widen through it. Normal halves and subnormals give their exact value, an
/// infinity stays one, and a NaN keeps its payload (sign | 0x7F800000 | mantissa &lt;&lt; 13, never quieted), as the GPU
/// devices' float16 kernels widen one.
/// </summary>
public static class HalfFloats
{
    /// <summary>
    /// values[i] = the float32 value of the half whose bits are halves[i], for each of <paramref name="halves"/>: whole
    /// vectors at a time (the exponent rebiased, a subnormal through one subtraction of the smallest normal, an infinity or
    /// NaN given the top exponent), the rest value by value (<see cref="ToSingle"/>), the same values either way.
    /// </summary>
    /// <exception cref="ArgumentException"><paramref name="values"/> is shorter than <paramref name="halves"/>.</exception>
    public static void Widen(ReadOnlySpan<ushort> halves, Span<float> values)
    {
        if (values.Length < halves.Length)
        {
            throw new ArgumentException($"{values.Length} values cannot hold {halves.Length} halves.", nameof(values));
        }

        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            var bits = MemoryMarshal.Cast<float, uint>(values);
            for (; i <= halves.Length - Vector<ushort>.Count; i += Vector<ushort>.Count)
            {
                Vector.Widen(new Vector<ushort>(halves[i..]), out var low, out var high);
                Bits(low).CopyTo(bits[i..]);
                Bits(high).CopyTo(bits[(i + Vector<uint>.Count)..]);
            }
        }

        for (; i < halves.Length; i++)
        {
            values[i] = ToSingle(halves[i]);
        }
    }

    /// <summary>The float32 value of the half whose bits are <paramref name="half"/> (see <see cref="Widen"/>).</summary>
    public static float ToSingle(ushort half)
    {
        uint e = (uint)(half >> 10) & 0x1Fu, m = half & 0x3FFu, sign = (uint)(half & 0x8000) << 16;
        return e switch
        {
            0 => BitConverter.UInt32BitsToSingle(sign | BitConverter.SingleToUInt32Bits(m / 16777216f)),   // m · 2^-24, a normal float32
            31 => BitConverter.UInt32BitsToSingle(sign | 0x7F800000u | m << 13),
            _ => BitConverter.UInt32BitsToSingle(sign | (e + 112) << 23 | m << 13),
        };
    }

    // The bits of each half (in the low 16 bits of a lane) as a float32's: the exponent rebiased (moved up by 112), a
    // subnormal as the smallest normal half plus its mantissa less that normal (one exact subtraction), an infinity or NaN
    // given the top exponent and kept as it is (no arithmetic, so a NaN keeps its payload), the sign put back.
    private static Vector<uint> Bits(Vector<uint> half)
    {
        var exponent = half & new Vector<uint>(0x7C00u);
        var subnormal = Vector.Equals(exponent, Vector<uint>.Zero);
        var special = Vector.Equals(exponent, new Vector<uint>(0x7C00u));
        var lowerBound = subnormal & new Vector<uint>(0x3880_0000u);                    // 2^-14, the smallest normal half, as a float32
        var offset = new Vector<uint>(0x3800_0000u) | lowerBound;
        offset = Vector.ConditionalSelect(special, offset << 1, offset);
        var value = ((half << 13) & new Vector<uint>(0x0FFF_E000u)) + offset;
        var magnitude = Vector.ConditionalSelect(subnormal, Vector.AsVectorUInt32(Vector.AsVectorSingle(value) - Vector.AsVectorSingle(lowerBound)), value);
        return magnitude | ((half & new Vector<uint>(0x8000u)) << 16);
    }
}
