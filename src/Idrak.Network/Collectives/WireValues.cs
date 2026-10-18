// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Idrak.Networking.Collectives;

/// <summary>
/// Values as they travel (plan 23's wire format): float32 little-endian, or bfloat16 (the top half of the float32,
/// rounded to nearest even) little-endian. Sums are always made in float32 by the receiver.
/// </summary>
internal static class WireValues
{
    /// <summary>The bytes a value takes on <paramref name="wire"/>.</summary>
    public static int BytesPerValue(WireType wire) => wire == WireType.BFloat16 ? 2 : 4;

    /// <summary>Writes <paramref name="values"/> into <paramref name="bytes"/> as <paramref name="wire"/>; returns the bytes written.</summary>
    public static int Encode(ReadOnlySpan<float> values, Span<byte> bytes, WireType wire)
    {
        if (wire == WireType.Float32)
        {
            int length = values.Length * sizeof(float);
            if (BitConverter.IsLittleEndian)
            {
                MemoryMarshal.AsBytes(values).CopyTo(bytes);
            }
            else
            {
                for (int i = 0; i < values.Length; i++)
                {
                    BinaryPrimitives.WriteSingleLittleEndian(bytes[(i * sizeof(float))..], values[i]);
                }
            }

            return length;
        }

        var halves = bytes[..(values.Length * 2)];
        int k = 0;
        if (BitConverter.IsLittleEndian && Vector256.IsHardwareAccelerated && values.Length >= 2 * Vector256<uint>.Count)
        {
            var target = MemoryMarshal.Cast<byte, ushort>(halves);
            var bits = MemoryMarshal.Cast<float, uint>(values);
            var half = Vector256.Create(0x7FFFu);
            var one = Vector256.Create(1u);
            for (; k <= values.Length - (2 * Vector256<uint>.Count); k += 2 * Vector256<uint>.Count)
            {
                var lo = Vector256.Create(bits[k..]);
                var hi = Vector256.Create(bits[(k + Vector256<uint>.Count)..]);
                lo = (lo + half + ((lo >> 16) & one)) >> 16;
                hi = (hi + half + ((hi >> 16) & one)) >> 16;
                Vector256.Narrow(lo, hi).CopyTo(target[k..]);
            }
        }

        for (; k < values.Length; k++)                                      // the scalar tail
        {
            BinaryPrimitives.WriteUInt16LittleEndian(halves[(k * 2)..], Round(values[k]));
        }

        return halves.Length;
    }

    /// <summary>Reads <paramref name="values"/>.Length values of <paramref name="wire"/> from <paramref name="bytes"/>.</summary>
    public static void Decode(ReadOnlySpan<byte> bytes, Span<float> values, WireType wire)
    {
        if (wire == WireType.Float32)
        {
            if (BitConverter.IsLittleEndian)
            {
                bytes[..(values.Length * sizeof(float))].CopyTo(MemoryMarshal.AsBytes(values));
            }
            else
            {
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes[(i * sizeof(float))..]);
                }
            }

            return;
        }

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = BitConverter.UInt32BitsToSingle((uint)BinaryPrimitives.ReadUInt16LittleEndian(bytes[(i * 2)..]) << 16);
        }
    }

    /// <summary>Rounds each value to bfloat16 in place (what a bfloat16 wire does to a rank's values before they travel).</summary>
    public static void RoundToBFloat16(Span<float> values)
    {
        foreach (ref float v in values)
        {
            v = BitConverter.UInt32BitsToSingle((uint)Round(v) << 16);
        }
    }

    // Round to nearest even at bfloat16's 8 bits of mantissa.
    private static ushort Round(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        return (ushort)((bits + 0x7FFFu + ((bits >> 16) & 1)) >> 16);
    }
}
