// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Text;

namespace Idrak.Adad.Types;

/// <summary>A float's bits: <paramref name="Exponent"/> exponent bits and <paramref name="Mantissa"/> mantissa bits, and a sign.</summary>
/// <param name="Exponent">Exponent bits.</param>
/// <param name="Mantissa">Mantissa bits.</param>
public readonly record struct FloatBits(int Exponent, int Mantissa)
{
    /// <summary>The bits a value takes: 1 + <see cref="Exponent"/> + <see cref="Mantissa"/>.</summary>
    public int Bits => 1 + Exponent + Mantissa;

    /// <summary>The exponent's bias: 2^(<see cref="Exponent"/> − 1) − 1.</summary>
    public int Bias => (1 << (Exponent - 1)) - 1;
}

/// <summary>
/// A bit plane of a super-block's codes (<see cref="SuperBlock"/>): fields of <paramref name="Bits"/> bits from byte
/// <paramref name="Offset"/>, in groups of <paramref name="Group"/> values. A group takes Group · Bits / 8 bytes; value w of
/// a group lies in byte w mod (Group · Bits / 8) of it, at bit Bits · ⌊w / (Group · Bits / 8)⌋ (ggml's split order: Q4_K's
/// codes are a plane of 4 bits in groups of 64, Q6_K's high bits a plane of 2 bits in groups of 128).
/// </summary>
/// <param name="Offset">The plane's first byte in the block.</param>
/// <param name="Bits">Bits a field takes (1, 2, 4 or 8).</param>
/// <param name="Group">Values a group holds (a power of two).</param>
public sealed record CodePlane(int Offset, int Bits, int Group)
{
    /// <summary>The byte of value <paramref name="value"/>'s field in the block and the field's lowest bit in it.</summary>
    public (int Byte, int Shift) Place(int value)
    {
        int per = Group * Bits / 8, w = value % Group;
        return (Offset + value / Group * per + w % per, Bits * (w / per));
    }
}

/// <summary>
/// A piece of a sub-block's scale or minimum (<see cref="SuperBlock"/>): <paramref name="Bits"/> bits at bit
/// <paramref name="Shift"/> of byte <paramref name="Offset"/> of the block; a value's pieces are listed from its low bits up.
/// </summary>
/// <param name="Offset">The byte in the block.</param>
/// <param name="Shift">The piece's lowest bit in the byte.</param>
/// <param name="Bits">Bits the piece takes.</param>
public readonly record struct ScalePiece(int Offset, int Shift, int Bits);

/// <summary>
/// How a super-block format reads (ggml's k-quants), as data: each block of <paramref name="Values"/> values takes
/// <paramref name="Bytes"/> bytes; value v of the block is d · (scale_j + <paramref name="SubScaleBias"/>) · (code_v +
/// <paramref name="CodeBias"/>) − dmin · min_j, j = v / <paramref name="SubValues"/> its sub-block, d and dmin floats of
/// <paramref name="Scale"/>'s bits at <paramref name="ScaleOffset"/> and <paramref name="MinOffset"/> (none below 0), the
/// sub-block's scale and minimum assembled from their pieces, the code from its <paramref name="Planes"/>' fields (the first
/// plane's bits lowest). A kernel or decoder reads a type from this description alone, never from its name.
/// </summary>
/// <param name="Values">Values a block holds.</param>
/// <param name="Bytes">Bytes a block takes.</param>
/// <param name="Planes">The code's bit planes, its low bits first.</param>
/// <param name="CodeBias">Added to each code.</param>
/// <param name="Scale">The 16-bit float bits of d and dmin.</param>
/// <param name="ScaleOffset">d's first byte.</param>
/// <param name="MinOffset">dmin's first byte, or −1: no minimums.</param>
/// <param name="SubValues">Values a sub-block's scale covers.</param>
/// <param name="SubScales">Each sub-block's scale, as pieces.</param>
/// <param name="SubScaleBias">Added to each sub-block's scale.</param>
/// <param name="SubScalesSigned">Whether a sub-block's scale is two's complement.</param>
/// <param name="SubMins">Each sub-block's minimum, as pieces, or null.</param>
public sealed record SuperBlock(int Values, int Bytes, IReadOnlyList<CodePlane> Planes, int CodeBias, FloatBits Scale, int ScaleOffset, int MinOffset,
    int SubValues, IReadOnlyList<IReadOnlyList<ScalePiece>> SubScales, int SubScaleBias, bool SubScalesSigned, IReadOnlyList<IReadOnlyList<ScalePiece>>? SubMins)
{
    // Each description's places once: per plane, each value's byte and shift (Place).
    private static readonly ConditionalWeakTable<SuperBlock, (int[] Byte, int[] Shift)[]> Places = new();

    // Each description's code and its FNV-1a hash, once (Code, Hash, Id).
    private static readonly ConditionalWeakTable<SuperBlock, Tuple<string, uint>> Codes = new();

    /// <summary>
    /// The description's canonical text: every field that changes how a value reads, in a fixed order (the same text the
    /// kernels' names carry). Made once a description.
    /// </summary>
    public string Code => CodeAndHash().Item1;

    /// <summary>The 32-bit FNV-1a hash of <see cref="Code"/> (the kernels' names carry it as 8 hex digits).</summary>
    public uint Hash => CodeAndHash().Item2;

    /// <summary>
    /// The description's stable id: FNV-1a 32-bit over <see cref="Code"/>, masked to 0x7FFFFFFF (the key of every choice
    /// measured for it, never an order of first use). Made once a description.
    /// </summary>
    public int Id => (int)(Hash & 0x7FFFFFFF);

    private Tuple<string, uint> CodeAndHash() => Codes.GetValue(this, static s =>
    {
        string text = CanonicalText(s);
        return Tuple.Create(text, Fnv(text));
    });

    private static uint Fnv(string text)
    {
        uint hash = 2166136261;
        foreach (char c in text)
        {
            hash = (hash ^ c) * 16777619;
        }

        return hash;
    }

    private static string CanonicalText(SuperBlock s)
    {
        var text = new StringBuilder(string.Create(CultureInfo.InvariantCulture, $"{s.Values}x{s.Bytes}b{s.CodeBias}f{s.Scale.Exponent}m{s.Scale.Mantissa}s{s.ScaleOffset}n{s.MinOffset}v{s.SubValues}q{s.SubScaleBias}{s.SubScalesSigned}"));
        foreach (var p in s.Planes)
        {
            text.Append(CultureInfo.InvariantCulture, $"|{p.Offset},{p.Bits},{p.Group}");
        }

        foreach (var list in s.SubMins is null ? new[] { s.SubScales } : new[] { s.SubScales, s.SubMins })
        {
            foreach (var pieces in list)
            {
                text.Append(';');
                foreach (var piece in pieces)
                {
                    text.Append(CultureInfo.InvariantCulture, $"{piece.Offset}.{piece.Shift}.{piece.Bits},");
                }
            }
        }

        return text.ToString();
    }

    /// <summary>A sub-block's scale or minimum from its <paramref name="pieces"/> in <paramref name="block"/>.</summary>
    public static int Assemble(IReadOnlyList<ScalePiece> pieces, ReadOnlySpan<byte> block, bool signed)
    {
        int value = 0, at = 0;
        for (int i = 0; i < pieces.Count; i++)                             // indexed: no enumerator a call (a hot path)
        {
            var piece = pieces[i];
            value |= (block[piece.Offset] >> piece.Shift & ((1 << piece.Bits) - 1)) << at;
            at += piece.Bits;
        }

        return signed ? value << (32 - at) >> (32 - at) : value;
    }

    /// <summary>The float of <see cref="Scale"/>'s bits (float16 or bfloat16) at byte <paramref name="offset"/>.</summary>
    public float FloatAt(ReadOnlySpan<byte> block, int offset)
    {
        ushort bits = BinaryPrimitives.ReadUInt16LittleEndian(block[offset..]);
        return Scale.Exponent == 8 ? BitConverter.Int32BitsToSingle(bits << 16) : (float)BitConverter.UInt16BitsToHalf(bits);
    }

    // Whether every 8 consecutive values' fields lie in 8 consecutive bytes at one shift in each plane (a group's bytes 8 or
    // more, values a multiple of 8) and a code fits a byte: then 8 values are read as one 64-bit word a plane.
    private bool Eights => (bool)EightsOf.GetValue(this, static s => s.FitsEights);

    private static readonly ConditionalWeakTable<SuperBlock, object> EightsOf = new();

    private bool FitsEights => Values % 8 == 0 && SubValues % 8 == 0 && Planes.Sum(p => p.Bits) <= 8
        && Planes.All(p => p.Bits is 1 or 2 or 4 or 8 && p.Group * p.Bits / 8 is var per && per >= 8 && per % 8 == 0 && p.Group % 8 == 0);

    // A block's values 8 at a time: each plane's 8 bytes one word, its 8 fields shifted and masked at once, the planes'
    // fields joined per byte, then widened and scaled (each value (code + bias) · scale − min, as the scalar loop rounds it).
    private void DecodeEights(ReadOnlySpan<byte> block, (int[] Byte, int[] Shift)[] places, ReadOnlySpan<int> low, ReadOnlySpan<float> scales, ReadOnlySpan<float> mins, Span<float> into)
    {
        var bias = Vector128.Create(CodeBias);
        for (int v = 0; v < Values; v += 8)
        {
            ulong codes = 0;
            for (int p = 0; p < places.Length; p++)
            {
                ulong word = BinaryPrimitives.ReadUInt64LittleEndian(block[places[p].Byte[v]..]);
                ulong mask = 0x0101010101010101UL * (ulong)((1 << Planes[p].Bits) - 1);
                codes |= (word >> places[p].Shift[v] & mask) << low[p];
            }

            var bytes = Vector128.CreateScalar(codes).AsByte();
            var (shorts, _) = Vector128.Widen(bytes);
            var (first, second) = Vector128.Widen(shorts);
            var scale = Vector128.Create(scales[v / SubValues]);
            var min = Vector128.Create(mins[v / SubValues]);
            (Vector128.ConvertToSingle(first.AsInt32() + bias) * scale - min).CopyTo(into.Slice(v, 4));
            (Vector128.ConvertToSingle(second.AsInt32() + bias) * scale - min).CopyTo(into.Slice(v + 4, 4));
        }
    }

    // Each plane's byte and shift for every value of a block (Place), once a description.
    internal (int[] Byte, int[] Shift)[] PlacesOfValues() => Places.GetValue(this, static s => [.. s.Planes.Select(p => (
        Enumerable.Range(0, s.Values).Select(v => p.Place(v).Byte).ToArray(),
        Enumerable.Range(0, s.Values).Select(v => p.Place(v).Shift).ToArray()))]);

    /// <summary>
    /// Sub-block <paramref name="j"/>'s scale d · (scale_j + <see cref="SubScaleBias"/>) and minimum dmin · min_j (0 without)
    /// of <paramref name="block"/>, each rounded once as <see cref="Decode"/> rounds them.
    /// </summary>
    public (float Scale, float Min) SubScale(ReadOnlySpan<byte> block, int j)
    {
        float d = FloatAt(block, ScaleOffset), dmin = MinOffset >= 0 ? FloatAt(block, MinOffset) : 0f;
        return (d * (Assemble(SubScales[j], block, SubScalesSigned) + SubScaleBias), SubMins is null ? 0f : dmin * Assemble(SubMins[j], block, false));
    }

    /// <summary>Value <paramref name="v"/>'s code of <paramref name="block"/> plus <see cref="CodeBias"/>, from its planes' fields.</summary>
    public int CodeAt(ReadOnlySpan<byte> block, int v)
    {
        var places = PlacesOfValues();
        int code = 0, low = 0;
        for (int p = 0; p < places.Length; p++)
        {
            code |= (block[places[p].Byte[v]] >> places[p].Shift[v] & ((1 << Planes[p].Bits) - 1)) << low;
            low += Planes[p].Bits;
        }

        return code + CodeBias;
    }

    /// <summary>The values of whole blocks of <paramref name="source"/> into <paramref name="destination"/>, by this description alone.</summary>
    public void Decode(ReadOnlySpan<byte> source, Span<float> destination)
    {
        var places = PlacesOfValues();
        int subs = Values / SubValues;
        Span<float> scales = stackalloc float[subs], mins = stackalloc float[subs];
        Span<int> low = stackalloc int[Planes.Count];
        for (int p = 1; p < Planes.Count; p++)
        {
            low[p] = low[p - 1] + Planes[p - 1].Bits;
        }

        for (int b = 0; b < destination.Length / Values; b++)
        {
            var block = source.Slice(b * Bytes, Bytes);
            float d = FloatAt(block, ScaleOffset), dmin = MinOffset >= 0 ? FloatAt(block, MinOffset) : 0f;
            for (int j = 0; j < subs; j++)
            {
                scales[j] = d * (Assemble(SubScales[j], block, SubScalesSigned) + SubScaleBias);
                mins[j] = SubMins is null ? 0f : dmin * Assemble(SubMins[j], block, false);
            }

            var into = destination.Slice(b * Values, Values);
            if (Eights)
            {
                DecodeEights(block, places, low, scales, mins, into);
                continue;
            }

            for (int v = 0; v < Values; v++)
            {
                int code = 0;
                for (int p = 0; p < places.Length; p++)
                {
                    code |= (block[places[p].Byte[v]] >> places[p].Shift[v] & ((1 << Planes[p].Bits) - 1)) << low[p];
                }

                into[v] = scales[v / SubValues] * (code + CodeBias) - mins[v / SubValues];
            }
        }
    }
}
