// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;

namespace Idrak.Models.Abstractions;

/// <summary>
/// How one <see cref="WeightFormat"/> stores float values in a weights file (little-endian, whole tensors at a time, so
/// saving and loading make one call per tensor). The formats are part of the file layout (the byte after each shape), so
/// the set is fixed; each format's encoding lives in its own codec instead of a branch per value.
/// </summary>
public abstract class WeightCodec
{
    private static readonly WeightCodec[] Codecs = [new Float32Codec(), new Float16Codec(), new BFloat16Codec()];

    // Only the codecs of the file layout's formats exist.
    private protected WeightCodec()
    {
    }

    /// <summary>The codec of <paramref name="format"/>; a format byte no codec knows is a damaged or newer file.</summary>
    public static WeightCodec For(WeightFormat format) => (uint)format < (uint)Codecs.Length ? Codecs[(int)format]
        : throw new InvalidDataException($"Unknown weight format {(int)format}.");

    /// <summary>Bytes per stored value.</summary>
    public abstract int BytesPerValue { get; }

    /// <summary>Writes <paramref name="values"/> into <paramref name="bytes"/> (<see cref="BytesPerValue"/> each).</summary>
    public abstract void Encode(ReadOnlySpan<float> values, Span<byte> bytes);

    /// <summary>Reads <paramref name="values"/>.Length values from <paramref name="bytes"/>.</summary>
    public abstract void Decode(ReadOnlySpan<byte> bytes, Span<float> values);

    private sealed class Float32Codec : WeightCodec
    {
        public override int BytesPerValue => 4;

        public override void Encode(ReadOnlySpan<float> values, Span<byte> bytes)
        {
            if (BitConverter.IsLittleEndian)
            {
                System.Runtime.InteropServices.MemoryMarshal.AsBytes(values).CopyTo(bytes);   // the stored bytes as they are
                return;
            }

            for (int i = 0; i < values.Length; i++)
            {
                BinaryPrimitives.WriteSingleLittleEndian(bytes[(i * 4)..], values[i]);
            }
        }

        public override void Decode(ReadOnlySpan<byte> bytes, Span<float> values) => StoredFloats.ReadFloat32(bytes, values);
    }

    private sealed class Float16Codec : WeightCodec
    {
        public override int BytesPerValue => 2;

        public override void Encode(ReadOnlySpan<float> values, Span<byte> bytes)
        {
            for (int i = 0; i < values.Length; i++)
            {
                BinaryPrimitives.WriteHalfLittleEndian(bytes[(i * 2)..], (Half)values[i]);
            }
        }

        public override void Decode(ReadOnlySpan<byte> bytes, Span<float> values)
        {
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = (float)BinaryPrimitives.ReadHalfLittleEndian(bytes[(i * 2)..]);
            }
        }
    }

    // Rounds to nearest even; NaN stays NaN.
    private sealed class BFloat16Codec : WeightCodec
    {
        public override int BytesPerValue => 2;

        public override void Encode(ReadOnlySpan<float> values, Span<byte> bytes)
        {
            for (int i = 0; i < values.Length; i++)
            {
                uint bits = BitConverter.SingleToUInt32Bits(values[i]);
                ushort b16 = float.IsNaN(values[i]) ? (ushort)0x7FC0 : (ushort)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
                BinaryPrimitives.WriteUInt16LittleEndian(bytes[(i * 2)..], b16);
            }
        }

        public override void Decode(ReadOnlySpan<byte> bytes, Span<float> values) => StoredFloats.WidenBFloat16(bytes, values);
    }
}

/// <summary>How <c>Idrak.ModuleFiles.Save</c> stores floating-point values in a weights file.</summary>
public enum WeightFormat
{
    /// <summary>32-bit floats: exact (4 bytes per value).</summary>
    Float32,

    /// <summary>IEEE half precision (2 bytes per value): about 3 significant digits, range ±65504.</summary>
    Float16,

    /// <summary>bfloat16 (2 bytes per value): the range of float32 with about 2–3 significant digits.</summary>
    BFloat16,
}
