// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Runtime.InteropServices;

namespace Idrak.Networking.Collectives;

/// <summary>
/// Values as they travel (plan 23's wire format): float32 little-endian. Sums are made in float32 by the receiver.
/// </summary>
internal static class WireValues
{
    /// <summary>Writes <paramref name="values"/> into <paramref name="bytes"/>; returns the bytes written.</summary>
    public static int Encode(ReadOnlySpan<float> values, Span<byte> bytes)
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

    /// <summary>Reads <paramref name="values"/>.Length values from <paramref name="bytes"/>.</summary>
    public static void Decode(ReadOnlySpan<byte> bytes, Span<float> values)
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
    }
}
