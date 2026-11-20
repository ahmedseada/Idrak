// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using Idrak.Abstraction.Formats;

namespace Idrak.Abstraction;

/// <summary>Widens stored values to float32: <paramref name="source"/> holds the bytes of <paramref name="destination"/>.Length values (whole blocks for a block format).</summary>
/// <param name="source">The stored bytes.</param>
/// <param name="destination">Where the values go.</param>
public delegate void ElementDecoder(ReadOnlySpan<byte> source, Span<float> destination);

/// <summary>Stores float32 values in a type: <paramref name="destination"/> receives the bytes of <paramref name="source"/>.Length values (whole blocks for a block format).</summary>
/// <param name="source">The values.</param>
/// <param name="destination">Where the stored bytes go.</param>
public delegate void ElementEncoder(ReadOnlySpan<float> source, Span<byte> destination);

/// <summary>
/// The scales of a block format (plan 37, T0): each block's <paramref name="PerBlock"/> scales of type
/// <paramref name="Type"/> (float16, float32, a 6-bit code, an 8-bit exponent), whether it has zero points (a minimum or
/// an offset per block or sub-block), and how they sit in the block, for people.
/// </summary>
/// <param name="Type">The scales' own type.</param>
/// <param name="PerBlock">Scales per block.</param>
/// <param name="ZeroPoints">Whether a block also has zero points (minimums).</param>
/// <param name="Layout">How the scales sit in a block, or null.</param>
public sealed record BlockScales(string Type, int PerBlock, bool ZeroPoints, string? Layout = null);

/// <summary>
/// How a float type's bits read: one sign bit, <paramref name="Exponent"/> exponent bits biased by
/// 2^(Exponent − 1) − 1, then <paramref name="Mantissa"/> mantissa bits, the sign the highest; the largest exponent is an
/// infinity (mantissa 0) or a NaN, the smallest a zero or a subnormal, as IEEE 754's formats are. A kernel that widens
/// values from this description alone needs nothing named after the type. A float whose special values differ (E4M3fn:
/// no infinity) declares none.
/// </summary>
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
/// How a block format whose values are integer codes times one scale per block reads: each block of
/// <paramref name="Values"/> values takes <paramref name="Bytes"/> bytes, its scale a float of <paramref name="Scale"/>'s
/// bits at byte <paramref name="ScaleOffset"/>, its codes <paramref name="CodeBits"/> bits each from byte
/// <paramref name="CodeOffset"/> in value order (the first in the lowest bits), or, 4-bit codes with
/// <paramref name="NibblesSplit"/>, ggml's order (byte i holds value i in its low half and value i + Values / 2 in its
/// high half, as q4_0 stores them), and value i = (code i + <paramref name="CodeBias"/>) · scale, the code read as two's
/// complement when <paramref name="SignedCodes"/>; with a <paramref name="CodeTable"/> (ggml's IQ4_NL: a non-linear grid)
/// the code first stands for its entry of the table, value i = (table[code i] + <paramref name="CodeBias"/>) · scale. With a
/// <paramref name="CodeType"/> each code is one value of that element type (OCP's fp4 E2M1: MXFP4's elements), and with a
/// <paramref name="ScaleType"/> the scale is one value of that type stored at <paramref name="ScaleOffset"/> (OCP's E8M0),
/// so a block of any element type sharing a scale of any type is described the same way: value i = (the code's value +
/// <paramref name="CodeBias"/>) · scale. A kernel that widens values from this description alone needs nothing
/// named after the format. A format whose scales are not one plain float per block (the k-quants' scaled sub-blocks), or
/// that adds a minimum or high bits (q4_1, q5_x), declares none.
/// </summary>
/// <param name="Values">Values a block holds.</param>
/// <param name="Bytes">Bytes a block takes.</param>
/// <param name="CodeBits">Bits a code takes.</param>
/// <param name="SignedCodes">Whether a code is two's complement.</param>
/// <param name="CodeBias">Added to each code before the scale.</param>
/// <param name="Scale">The scale's float bits.</param>
/// <param name="ScaleOffset">The scale's first byte in the block.</param>
/// <param name="CodeOffset">The first code's byte in the block.</param>
/// <param name="NibblesSplit">4-bit codes in ggml's order: byte i holds value i (low half) and value i + Values / 2 (high half).</param>
/// <param name="CodeTable">The integer each code stands for (2^CodeBits entries), or null: the code itself.</param>
/// <param name="CodeType">The element type each code is one value of (its <paramref name="CodeBits"/> bits), or null: an integer code (or its table entry).</param>
/// <param name="ScaleType">The element type the scale is one value of (from <paramref name="ScaleOffset"/>), or null: a float of <paramref name="Scale"/>'s bits.</param>
public sealed record ScaledBlock(int Values, int Bytes, int CodeBits, bool SignedCodes, int CodeBias, FloatBits Scale, int ScaleOffset, int CodeOffset, bool NibblesSplit = false,
    IReadOnlyList<int>? CodeTable = null, ElementType? CodeType = null, ElementType? ScaleType = null);

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
/// How a grid-codebook format reads (ggml's IQ2_XXS, IQ3_XXS): each block of
/// <paramref name="Values"/> values takes <paramref name="Bytes"/> bytes; value v is (scale_j · magnitude) · sign, with
/// j = v / <paramref name="SubValues"/> its sub-block, scale_j = (d · (s_j + <paramref name="SubScaleOffset"/>)) ·
/// <paramref name="SubScaleFactor"/> (d the float of <paramref name="Scale"/>'s bits at <paramref name="ScaleOffset"/>,
/// s_j from its pieces), magnitude = <paramref name="Grid"/>[index_e · <paramref name="EntryValues"/> + v mod
/// <paramref name="EntryValues"/>] for its grid entry e = v / EntryValues (index_e from <paramref name="Entries"/>[e]),
/// and sign −1 where bit (v mod <paramref name="SignValues"/>) of <paramref name="SignTable"/>[sign index] is set, its
/// group's sign index from <paramref name="Signs"/>[v / SignValues].
/// </summary>
/// <param name="Values">Values a block holds.</param>
/// <param name="Bytes">Bytes a block takes.</param>
/// <param name="Scale">The super-scale's float bits.</param>
/// <param name="ScaleOffset">The super-scale's first byte.</param>
/// <param name="SubValues">Values a sub-block's scale covers.</param>
/// <param name="SubScales">Each sub-block's scale, as pieces.</param>
/// <param name="SubScaleOffset">Added to each sub-block's scale.</param>
/// <param name="SubScaleFactor">Times each sub-block's scale.</param>
/// <param name="EntryValues">Values a grid entry holds.</param>
/// <param name="Entries">Each entry's index into the grid, as pieces.</param>
/// <param name="Grid">The grid: each entry's magnitudes.</param>
/// <param name="SignValues">Values a sign mask covers.</param>
/// <param name="Signs">Each group's index into the sign table, as pieces.</param>
/// <param name="SignTable">The sign masks (bit i set: value i of the group negative).</param>
public sealed record GridBlock(int Values, int Bytes, FloatBits Scale, int ScaleOffset, int SubValues, IReadOnlyList<IReadOnlyList<ScalePiece>> SubScales,
    float SubScaleOffset, float SubScaleFactor, int EntryValues, IReadOnlyList<IReadOnlyList<ScalePiece>> Entries, IReadOnlyList<int> Grid,
    int SignValues, IReadOnlyList<IReadOnlyList<ScalePiece>> Signs, IReadOnlyList<int> SignTable)
{
    /// <summary>The float of <see cref="Scale"/>'s bits (float16 or bfloat16) at byte <paramref name="offset"/>.</summary>
    public float FloatAt(ReadOnlySpan<byte> block, int offset)
    {
        ushort bits = System.Buffers.Binary.BinaryPrimitives.ReadUInt16LittleEndian(block[offset..]);
        return Scale.Exponent == 8 ? BitConverter.Int32BitsToSingle(bits << 16) : (float)BitConverter.UInt16BitsToHalf(bits);
    }

    /// <summary>The values of whole blocks of <paramref name="source"/> into <paramref name="destination"/>, by this description alone.</summary>
    public void Decode(ReadOnlySpan<byte> source, Span<float> destination)
    {
        Span<float> scales = stackalloc float[Values / SubValues];
        if (Plan is not null)
        {
            // Each value scale_j · (sign · magnitude): the same bits as (scale_j · magnitude) · sign, a sign flip being exact.
            Span<sbyte> codes = stackalloc sbyte[Values];
            for (int b = 0; b < destination.Length / Values; b++)
            {
                DecodeCodes(source.Slice(b * Bytes, Bytes), codes, scales);
                var into = destination.Slice(b * Values, Values);
                for (int v = 0; v < Values; v++)
                {
                    into[v] = scales[v / SubValues] * codes[v];
                }
            }

            return;
        }

        for (int b = 0; b < destination.Length / Values; b++)
        {
            var block = source.Slice(b * Bytes, Bytes);
            float d = FloatAt(block, ScaleOffset);
            for (int j = 0; j < scales.Length; j++)
            {
                scales[j] = d * (SuperBlock.Assemble(SubScales[j], block, false) + SubScaleOffset) * SubScaleFactor;
            }

            var into = destination.Slice(b * Values, Values);
            for (int e = 0; e < Values / EntryValues; e++)
            {
                int index = SuperBlock.Assemble(Entries[e], block, false) * EntryValues;
                for (int i = 0; i < EntryValues; i++)
                {
                    int v = e * EntryValues + i;
                    int mask = SignTable[SuperBlock.Assemble(Signs[v / SignValues], block, false)];
                    into[v] = scales[v / SubValues] * Grid[index + i] * ((mask >> (v % SignValues) & 1) != 0 ? -1f : 1f);
                }
            }
        }
    }

    // The description's fields as flat arrays, built once (CodePlan): what DecodeCodes reads, or null where the codes are
    // not int8 (a magnitude past ±127, sign groups other than 8 values, entries that do not divide 8).
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<GridBlock, object> Plans = new();

    private CodePlan? Plan => Plans.GetValue(this, static g => (object?)CodePlan.Of(g) ?? g) as CodePlan;

    /// <summary>Whether <see cref="DecodeCodes"/> reads this description: its signed magnitudes within int8, sign groups of 8 values.</summary>
    internal bool HasCodes => Plan is not null;

    /// <summary>
    /// One block's codes (sign · magnitude, as int8: <paramref name="codes"/> holds <see cref="Values"/>) and each sub-block's
    /// scale (<paramref name="scales"/> holds Values / SubValues): value v is scales[v / SubValues] · codes[v].
    /// </summary>
    internal void DecodeCodes(ReadOnlySpan<byte> block, Span<sbyte> codes, Span<float> scales)
    {
        var plan = Plan!;
        float d = FloatAt(block, ScaleOffset);
        for (int j = 0; j < scales.Length; j++)
        {
            scales[j] = d * (plan.Sub.Read(block, j) + SubScaleOffset) * SubScaleFactor;
        }

        var grid = plan.Grid.AsSpan();
        for (int e = 0; e < Values / EntryValues; e++)
        {
            grid.Slice(plan.Entry.Read(block, e) * EntryValues, EntryValues).CopyTo(codes.Slice(e * EntryValues, EntryValues));
        }

        // Each group's sign mask as bytes (0xFF where negative): v = (v ^ m) − m, two groups a pass.
        int groups = Values / 8, g0 = 0;
        for (; g0 + 2 <= groups; g0 += 2)
        {
            var v = Vector128.Create<sbyte>(codes.Slice(g0 * 8, 16));
            var m = Vector128.Create(plan.SignBytes[plan.Sign.Read(block, g0)], plan.SignBytes[plan.Sign.Read(block, g0 + 1)]).AsSByte();
            ((v ^ m) - m).CopyTo(codes.Slice(g0 * 8, 16));
        }

        for (; g0 < groups; g0++)
        {
            int mask = SignTable[plan.Sign.Read(block, g0)];
            for (int i = 0; i < 8; i++)
            {
                if ((mask >> i & 1) != 0)
                {
                    codes[g0 * 8 + i] = (sbyte)-codes[g0 * 8 + i];
                }
            }
        }
    }

    // Fields of pieces flattened: field f's pieces at [Start[f], Start[f + 1]), each its byte, shift, mask and place.
    private sealed class FieldPlan(int[] start, int[] offset, int[] shift, int[] mask, int[] place)
    {
        public static FieldPlan Of(IReadOnlyList<IReadOnlyList<ScalePiece>> fields)
        {
            var start = new int[fields.Count + 1];
            var (offset, shift, mask, place) = (new List<int>(), new List<int>(), new List<int>(), new List<int>());
            for (int f = 0; f < fields.Count; f++)
            {
                int at = 0;
                foreach (var q in fields[f])
                {
                    offset.Add(q.Offset);
                    shift.Add(q.Shift);
                    mask.Add((1 << q.Bits) - 1);
                    place.Add(at);
                    at += q.Bits;
                }

                start[f + 1] = offset.Count;
            }

            return new FieldPlan(start, [.. offset], [.. shift], [.. mask], [.. place]);
        }

        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        public int Read(ReadOnlySpan<byte> block, int field)
        {
            int value = 0;
            for (int p = start[field]; p < start[field + 1]; p++)
            {
                value |= (block[offset[p]] >> shift[p] & mask[p]) << place[p];
            }

            return value;
        }
    }

    private sealed class CodePlan
    {
        public required FieldPlan Sub { get; init; }

        public required FieldPlan Entry { get; init; }

        public required FieldPlan Sign { get; init; }

        public required sbyte[] Grid { get; init; }

        public required ulong[] SignBytes { get; init; }

        public static CodePlan? Of(GridBlock g)
        {
            bool fits = g.SignValues == 8 && g.EntryValues is 1 or 2 or 4 or 8 && g.Values % 16 == 0 && g.Values % g.SubValues == 0
                && g.Grid.All(v => v is >= -127 and <= 127) && g.SignTable.All(v => v is >= 0 and <= 255);
            if (!fits)
            {
                return null;
            }

            var signBytes = new ulong[g.SignTable.Count];
            for (int s = 0; s < signBytes.Length; s++)
            {
                for (int i = 0; i < 8; i++)
                {
                    if ((g.SignTable[s] >> i & 1) != 0)
                    {
                        signBytes[s] |= 0xFFUL << (8 * i);
                    }
                }
            }

            return new CodePlan
            {
                Sub = FieldPlan.Of(g.SubScales), Entry = FieldPlan.Of(g.Entries), Sign = FieldPlan.Of(g.Signs),
                Grid = [.. g.Grid.Select(v => (sbyte)v)], SignBytes = signBytes,
            };
        }
    }
}

/// <summary>
/// How a super-block format reads (ggml's k-quants): each block of <paramref name="Values"/>
/// values takes <paramref name="Bytes"/> bytes; a value's code is the fields of <paramref name="Planes"/> at its place (the
/// first plane the lowest bits); value v of sub-block j = v / <paramref name="SubValues"/> is
/// (d · (scale_j + <paramref name="SubScaleBias"/>)) · (code + <paramref name="CodeBias"/>) − dmin · min_j, d the float of
/// <paramref name="Scale"/>'s bits at <paramref name="ScaleOffset"/>, dmin the same at <paramref name="MinOffset"/> (−1:
/// none), scale_j and min_j assembled from their pieces (<paramref name="SubScales"/>, <paramref name="SubMins"/>), scale_j
/// two's complement over its bits when <paramref name="SubScalesSigned"/>. A kernel that widens values from this
/// description alone needs nothing named after the format.
/// </summary>
/// <param name="Values">Values a block holds.</param>
/// <param name="Bytes">Bytes a block takes.</param>
/// <param name="Planes">The codes' bit planes, the lowest bits first.</param>
/// <param name="CodeBias">Added to each code.</param>
/// <param name="Scale">The super-scale's (and super-minimum's) float bits.</param>
/// <param name="ScaleOffset">The super-scale's first byte.</param>
/// <param name="MinOffset">The super-minimum's first byte, or −1.</param>
/// <param name="SubValues">Values a sub-block's scale covers.</param>
/// <param name="SubScales">Each sub-block's scale, as pieces.</param>
/// <param name="SubScaleBias">Added to each sub-block's scale.</param>
/// <param name="SubScalesSigned">Whether a sub-block's scale is two's complement.</param>
/// <param name="SubMins">Each sub-block's minimum, as pieces, or null.</param>
public sealed record SuperBlock(int Values, int Bytes, IReadOnlyList<CodePlane> Planes, int CodeBias, FloatBits Scale, int ScaleOffset, int MinOffset,
    int SubValues, IReadOnlyList<IReadOnlyList<ScalePiece>> SubScales, int SubScaleBias, bool SubScalesSigned, IReadOnlyList<IReadOnlyList<ScalePiece>>? SubMins)
{
    // Each description's places once: per plane, each value's byte and shift (Place).
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<SuperBlock, (int[] Byte, int[] Shift)[]> Places = new();

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
        ushort bits = System.Buffers.Binary.BinaryPrimitives.ReadUInt16LittleEndian(block[offset..]);
        return Scale.Exponent == 8 ? BitConverter.Int32BitsToSingle(bits << 16) : (float)BitConverter.UInt16BitsToHalf(bits);
    }

    // Whether every 8 consecutive values' fields lie in 8 consecutive bytes at one shift in each plane (a group's bytes 8 or
    // more, values a multiple of 8) and a code fits a byte: then 8 values are read as one 64-bit word a plane.
    private bool Eights => (bool)EightsOf.GetValue(this, static s => s.FitsEights);

    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<SuperBlock, object> EightsOf = new();

    private bool FitsEights => Values % 8 == 0 && SubValues % 8 == 0 && Planes.Sum(p => p.Bits) <= 8
        && Planes.All(p => p.Bits is 1 or 2 or 4 or 8 && p.Group * p.Bits / 8 is var per && per >= 8 && per % 8 == 0 && p.Group % 8 == 0);

    // A block's values 8 at a time: each plane's 8 bytes one word, its 8 fields shifted and masked at once, the planes'
    // fields joined per byte, then widened and scaled (each value (code + bias) · scale − min, as the scalar loop rounds it).
    private void DecodeEights(ReadOnlySpan<byte> block, (int[] Byte, int[] Shift)[] places, ReadOnlySpan<int> low, ReadOnlySpan<float> scales, ReadOnlySpan<float> mins, Span<float> into)
    {
        var bias = System.Runtime.Intrinsics.Vector128.Create(CodeBias);
        for (int v = 0; v < Values; v += 8)
        {
            ulong codes = 0;
            for (int p = 0; p < places.Length; p++)
            {
                ulong word = System.Buffers.Binary.BinaryPrimitives.ReadUInt64LittleEndian(block[places[p].Byte[v]..]);
                ulong mask = 0x0101010101010101UL * (ulong)((1 << Planes[p].Bits) - 1);
                codes |= (word >> places[p].Shift[v] & mask) << low[p];
            }

            var bytes = System.Runtime.Intrinsics.Vector128.CreateScalar(codes).AsByte();
            var (shorts, _) = System.Runtime.Intrinsics.Vector128.Widen(bytes);
            var (first, second) = System.Runtime.Intrinsics.Vector128.Widen(shorts);
            var scale = System.Runtime.Intrinsics.Vector128.Create(scales[v / SubValues]);
            var min = System.Runtime.Intrinsics.Vector128.Create(mins[v / SubValues]);
            (System.Runtime.Intrinsics.Vector128.ConvertToSingle(first.AsInt32() + bias) * scale - min).CopyTo(into.Slice(v, 4));
            (System.Runtime.Intrinsics.Vector128.ConvertToSingle(second.AsInt32() + bias) * scale - min).CopyTo(into.Slice(v + 4, 4));
        }
    }

    // Each plane's byte and shift for every value of a block (Place), once a description.
    private (int[] Byte, int[] Shift)[] PlacesOfValues() => Places.GetValue(this, static s => [.. s.Planes.Select(p => (
        Enumerable.Range(0, s.Values).Select(v => p.Place(v).Byte).ToArray(),
        Enumerable.Range(0, s.Values).Select(v => p.Place(v).Shift).ToArray()))]);

    /// <summary>
    /// One block's codes plus <see cref="CodeBias"/> as int8 (<paramref name="codes"/>: <see cref="Values"/> of them; each
    /// must fit int8, as the k-quants' do), its sub-blocks' scales d · (scale_j + <see cref="SubScaleBias"/>) and minimums
    /// dmin · min_j (0 without): value v = scales[v / SubValues] · codes[v] − mins[v / SubValues], as <see cref="Decode"/> reads it.
    /// </summary>
    public void DecodeCodes(ReadOnlySpan<byte> block, Span<sbyte> codes, Span<float> scales, Span<float> mins)
    {
        var places = PlacesOfValues();
        float d = FloatAt(block, ScaleOffset), dmin = MinOffset >= 0 ? FloatAt(block, MinOffset) : 0f;
        for (int j = 0; j < Values / SubValues; j++)
        {
            scales[j] = d * (Assemble(SubScales[j], block, SubScalesSigned) + SubScaleBias);
            mins[j] = SubMins is null ? 0f : dmin * Assemble(SubMins[j], block, false);
        }

        Span<int> low = stackalloc int[Planes.Count];
        for (int p = 1; p < Planes.Count; p++)
        {
            low[p] = low[p - 1] + Planes[p - 1].Bits;
        }

        if (Eights)
        {
            var bias = Vector128.Create(unchecked((byte)CodeBias));
            for (int v = 0; v < Values; v += 8)
            {
                ulong word = 0;
                for (int p = 0; p < places.Length; p++)
                {
                    ulong plane = System.Buffers.Binary.BinaryPrimitives.ReadUInt64LittleEndian(block[places[p].Byte[v]..]);
                    word |= (plane >> places[p].Shift[v] & 0x0101010101010101UL * (ulong)((1 << Planes[p].Bits) - 1)) << low[p];
                }

                (Vector128.CreateScalar(word).AsByte() + bias).AsSByte().GetLower().CopyTo(codes.Slice(v, 8));
            }

            return;
        }

        for (int v = 0; v < Values; v++)
        {
            int code = 0;
            for (int p = 0; p < places.Length; p++)
            {
                code |= (block[places[p].Byte[v]] >> places[p].Shift[v] & ((1 << Planes[p].Bits) - 1)) << low[p];
            }

            codes[v] = (sbyte)(code + CodeBias);
        }
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

/// <summary>
/// An element type (plan 37, T0): how a tensor's values are stored, as data a registry holds (<see cref="ElementTypes"/>),
/// never a C# type parameter. A per-element type (float32, bfloat16, int32, int4, ...) has a block of one value; a block
/// format (ggml's Q4_K, IQ2_XXS, MXFP4, a block-scaled FP8) stores <see cref="BlockSize"/> values in
/// <see cref="BlockBits"/> bits with its <see cref="Scales"/>, so its bits per value may be fractional. Each entry carries
/// its conversions to and from float32 (<see cref="ToFloat32"/>, <see cref="FromFloat32"/>; on the CPU) and names the
/// operations that read it natively (<see cref="Kernels"/>); any other operation reads it through a conversion the
/// dispatcher makes and counts (<see cref="ElementTypes.Conversions"/>). A new format is registered, never coded into
/// <see cref="Tensor"/>.
/// </summary>
public sealed class ElementType : IKind
{
    /// <summary>32-bit IEEE float: what every operation computes in.</summary>
    public static readonly ElementType Float32 = new()
    {
        Name = "float32", Summary = "32-bit IEEE float, what every operation computes in", BlockBits = 32, IsFloat = true, IsSigned = true, Kernels = ["*"],
        Float = new FloatBits(8, 23),
        ToFloat32 = ElementConversions.Float32ToFloat32, FromFloat32 = ElementConversions.Float32FromFloat32,
    };

    /// <summary>bfloat16: float32's high 16 bits (8 exponent bits, 7 mantissa bits).</summary>
    public static readonly ElementType BFloat16 = new()
    {
        Name = "bfloat16", Summary = "bfloat16: float32's high 16 bits (rounded to nearest, ties to even)", BlockBits = 16, IsFloat = true, IsSigned = true,
        Float = new FloatBits(8, 7),
        Kernels = ["BFloat16MatMul", "GatherBFloat16", "GatherBFloat16Columns", "PackBFloat16"],
        ToFloat32 = ElementConversions.BFloat16ToFloat32, FromFloat32 = ElementConversions.BFloat16FromFloat32,
    };

    /// <summary>IEEE half (5 exponent bits, 10 mantissa bits).</summary>
    public static readonly ElementType Float16 = new()
    {
        Name = "float16", Summary = "16-bit IEEE half (rounded to nearest, ties to even)", BlockBits = 16, IsFloat = true, IsSigned = true, Kernels = [],
        Float = new FloatBits(5, 10),
        ToFloat32 = ElementConversions.Float16ToFloat32, FromFloat32 = ElementConversions.Float16FromFloat32,
    };

    /// <summary>fp8 E4M3 (the "fn" form: no infinity, ±448 at most, one NaN code a sign).</summary>
    public static readonly ElementType Float8E4M3 = new()
    {
        Name = "float8_e4m3", Summary = "8-bit float E4M3fn: 4 exponent bits, 3 mantissa bits, no infinity, ±448 at most (an overflow gives NaN)", BlockBits = 8,
        IsFloat = true, IsSigned = true, Kernels = [],
        ToFloat32 = ElementConversions.E4M3ToFloat32, FromFloat32 = ElementConversions.E4M3FromFloat32,
    };

    /// <summary>fp8 E5M2 (a half's high byte: infinities and NaNs as a half's).</summary>
    public static readonly ElementType Float8E5M2 = new()
    {
        Name = "float8_e5m2", Summary = "8-bit float E5M2: a half's high byte, 5 exponent bits, 2 mantissa bits, with infinities", BlockBits = 8,
        IsFloat = true, IsSigned = true, Kernels = [], Float = new FloatBits(5, 2),
        ToFloat32 = ElementConversions.E5M2ToFloat32, FromFloat32 = ElementConversions.E5M2FromFloat32,
    };

    /// <summary>
    /// 64-bit signed integer: two's complement, little-endian, one a value (ids and indices; as weights, codes of its full
    /// range with one float32 scale an output row, s = max |w| / (2⁶³ − 1), each value code · s).
    /// </summary>
    public static readonly ElementType Int64 = new()
    {
        Name = "int64", Summary = "64-bit signed integer: two's complement, little-endian (ids and indices; as weights, codes of its full range with one float32 scale an output row)",
        BlockBits = 64, IsSigned = true, IsInteger = true, Kernels = [],
        ToFloat32 = ElementConversions.Int64ToFloat32, FromFloat32 = ElementConversions.Int64FromFloat32,
    };

    /// <summary>32-bit signed integer: indices, token ids and offsets, exact past 2²⁴.</summary>
    public static readonly ElementType Int32 = new()
    {
        Name = "int32", Summary = "32-bit signed integer: indices, token ids and offsets, exact past 2^24", BlockBits = 32, IsSigned = true, IsInteger = true,
        Kernels = ["Gather", "GatherBFloat16", "GatherBFloat16Columns", "OneHot", "ScatterAdd", "LogSoftmaxTopK", "SampleRows", "PenalizeRows", "HistoryPush",
            "RouteTopK", "ExpertGatedPair", "ExpertMatMul", "ConvertElements"],
        ToFloat32 = ElementConversions.Int32ToFloat32, FromFloat32 = ElementConversions.Int32FromFloat32,
    };

    /// <summary>8-bit signed integer.</summary>
    public static readonly ElementType Int8 = new()
    {
        Name = "int8", Summary = "8-bit signed integer", BlockBits = 8, IsSigned = true, IsInteger = true, Kernels = [],
        ToFloat32 = ElementConversions.Int8ToFloat32, FromFloat32 = ElementConversions.Int8FromFloat32,
    };

    /// <summary>8-bit unsigned integer.</summary>
    public static readonly ElementType UInt8 = new()
    {
        Name = "uint8", Summary = "8-bit unsigned integer", BlockBits = 8, IsInteger = true, Kernels = [],
        ToFloat32 = ElementConversions.UInt8ToFloat32, FromFloat32 = ElementConversions.UInt8FromFloat32,
    };

    /// <summary>4-bit signed integer, two a byte (the first in the low nibble).</summary>
    public static readonly ElementType Int4 = new()
    {
        Name = "int4", Summary = "4-bit signed integer in [-8, 7], two a byte, the first value in the low nibble", BlockBits = 4, IsSigned = true, IsInteger = true, Kernels = [],
        ToFloat32 = ElementConversions.Int4ToFloat32, FromFloat32 = ElementConversions.Int4FromFloat32,
    };

    /// <summary>
    /// 2-bit signed integer codes (two's complement: −2, −1, 0, 1), four a byte, the first value in the lowest bits, in blocks
    /// of 32 values, each block one float16 scale (its first 2 bytes) and its 8 bytes of codes: value = code · scale (10
    /// bytes a block, 2.5 bits a value). A 2-bit code holds no weight's value without a scale: the scale is part of the type.
    /// </summary>
    public static readonly ElementType Int2 = new()
    {
        Name = "int2", Summary = "2-bit signed integer codes (-2..1, two's complement, four a byte, the first in the lowest bits) in blocks of 32, each with one float16 scale: value = code * scale (10 bytes a block)",
        BlockSize = 32, BlockBits = 80, IsSigned = true, Kernels = [],
        Scales = new BlockScales("float16", 1, false, "one scale d, then 32 2-bit codes, four a byte"),
        Scaled = new ScaledBlock(32, 10, 2, SignedCodes: true, CodeBias: 0, new FloatBits(5, 10), ScaleOffset: 0, CodeOffset: 2),
        ToFloat32 = ElementConversions.Int2ToFloat32, FromFloat32 = ElementConversions.Int2FromFloat32,
    };

    /// <summary>A truth value, one byte each (0 or 1, as PyTorch's bool).</summary>
    public static readonly ElementType Bool = new()
    {
        Name = "bool", Summary = "a truth value, one byte each (0 false, 1 true; any value not 0 reads true)", BlockBits = 8, Kernels = [],
        ToFloat32 = ElementConversions.BoolToFloat32, FromFloat32 = ElementConversions.BoolFromFloat32,
    };

    /// <summary>16-bit signed integer (plan 37, T2): audio's 16-bit samples kept as stored; storage and conversions only.</summary>
    public static readonly ElementType Int16 = new()
    {
        Name = "int16", Summary = "16-bit signed integer (audio's 16-bit samples kept as stored; storage and conversions only)", BlockBits = 16, IsSigned = true, IsInteger = true, Kernels = [],
        ToFloat32 = ElementConversions.Int16ToFloat32, FromFloat32 = ElementConversions.Int16FromFloat32,
    };

    /// <summary>16-bit unsigned integer (plan 37, T2): token ids of vocabularies under 65,536 in datasets; storage and conversions only.</summary>
    public static readonly ElementType UInt16 = new()
    {
        Name = "uint16", Summary = "16-bit unsigned integer (token ids of vocabularies under 65,536, half of int32's memory; storage and conversions only)", BlockBits = 16, IsInteger = true, Kernels = [],
        ToFloat32 = ElementConversions.UInt16ToFloat32, FromFloat32 = ElementConversions.UInt16FromFloat32,
    };

    /// <summary>
    /// 64-bit IEEE 754 float (binary64): one sign bit, 11 exponent bits biased by 1023, 52 mantissa bits, little-endian, one
    /// a value, no scales (as weights, each value stored as it is: a float32 or bfloat16 checkpoint's values exactly).
    /// </summary>
    public static readonly ElementType Float64 = new()
    {
        Name = "float64", Summary = "64-bit IEEE 754 float (binary64): 1 sign, 11 exponent and 52 mantissa bits, little-endian (as weights, each value as it is, no scales)",
        BlockBits = 64, IsFloat = true, IsSigned = true, Kernels = [], Float = new FloatBits(11, 52),
        ToFloat32 = ElementConversions.Float64ToFloat32, FromFloat32 = ElementConversions.Float64FromFloat32,
    };

    /// <summary>32-bit unsigned integer (plan 37, T3): ids and indices past int32's range in half of int64's memory.</summary>
    public static readonly ElementType UInt32 = new()
    {
        Name = "uint32", Summary = "32-bit unsigned integer (ids and indices up to 4,294,967,295)", BlockBits = 32, IsInteger = true, Kernels = [],
        ToFloat32 = ElementConversions.UInt32ToFloat32, FromFloat32 = ElementConversions.UInt32FromFloat32,
    };

    /// <summary>64-bit unsigned integer (plan 37, T3).</summary>
    public static readonly ElementType UInt64 = new()
    {
        Name = "uint64", Summary = "64-bit unsigned integer", BlockBits = 64, IsInteger = true, Kernels = [],
        ToFloat32 = ElementConversions.UInt64ToFloat32, FromFloat32 = ElementConversions.UInt64FromFloat32,
    };

    /// <summary>2-bit unsigned integer, four a byte (the first in the low bits) (plan 37, T3).</summary>
    public static readonly ElementType UInt2 = new()
    {
        Name = "uint2", Summary = "2-bit unsigned integer in [0, 3], four a byte, the first value in the lowest bits", BlockBits = 2, IsInteger = true, Kernels = [],
        ToFloat32 = ElementConversions.UInt2ToFloat32, FromFloat32 = ElementConversions.UInt2FromFloat32,
    };

    /// <summary>
    /// fp4 E2M1 (OCP Microscaling Formats v1.0's element): one sign bit, 2 exponent bits biased by 1, one mantissa bit: ±0,
    /// 0.5, 1, 1.5, 2, 3, 4, 6 (the exponent 0 a subnormal), no infinity or NaN; two a byte, the first value in the low half.
    /// </summary>
    public static readonly ElementType Float4E2M1 = new()
    {
        Name = "float4_e2m1", Summary = "4-bit float E2M1 (OCP): 1 sign, 2 exponent bits (bias 1), 1 mantissa bit: ±0, 0.5, 1, 1.5, 2, 3, 4, 6, no infinity or NaN; two a byte, the first in the low half",
        BlockBits = 4, IsFloat = true, IsSigned = true, Kernels = [],
        ToFloat32 = ElementConversions.E2M1ToFloat32, FromFloat32 = ElementConversions.E2M1FromFloat32,
    };

    /// <summary>
    /// E8M0 (OCP Microscaling Formats v1.0's scale): 8 exponent bits biased by 127, no sign, no mantissa: 2^(e − 127), the
    /// all-ones code NaN; one a byte.
    /// </summary>
    public static readonly ElementType Float8E8M0 = new()
    {
        Name = "float8_e8m0", Summary = "8-bit power of two E8M0 (OCP's microscaling scale): 8 exponent bits biased by 127, no sign, no mantissa: 2^(e - 127); 0xFF is NaN",
        BlockBits = 8, IsFloat = true, Kernels = [],
        ToFloat32 = ElementConversions.E8M0ToFloat32, FromFloat32 = ElementConversions.E8M0FromFloat32,
    };

    /// <summary>
    /// MXFP4 (OCP Microscaling Formats v1.0): blocks of 32 fp4 E2M1 values (<see cref="Float4E2M1"/>) sharing one E8M0 scale
    /// (<see cref="Float8E8M0"/>), value = element · 2^(e − 127); a block as GGUF lays it out: the scale byte, then 16 bytes
    /// holding value i in byte i's low half and value i + 16 in its high half (17 bytes, 4.25 bits a value).
    /// </summary>
    public static readonly ElementType MXFP4 = new()
    {
        Name = "mxfp4", Summary = "MXFP4 (OCP): blocks of 32 fp4 E2M1 values sharing one E8M0 scale, value = element * 2^(e - 127); the scale byte, then value i in byte i's low half and i + 16 in its high half (17 bytes a block)",
        BlockSize = 32, BlockBits = 136, IsFloat = true, IsSigned = true, Kernels = [],
        Scales = new BlockScales("float8_e8m0", 1, false, "one shared power of two (E8M0), then 32 fp4 (E2M1) values"),
        Scaled = new ScaledBlock(32, 17, 4, SignedCodes: false, CodeBias: 0, new FloatBits(8, 0), ScaleOffset: 0, CodeOffset: 1, NibblesSplit: true,
            CodeType: Float4E2M1, ScaleType: Float8E8M0),
        ToFloat32 = ElementConversions.MxFp4ToFloat32, FromFloat32 = ElementConversions.MxFp4FromFloat32,
    };

    /// <inheritdoc />
    public required string Name { get; init; }

    /// <inheritdoc />
    public required string Summary { get; init; }

    /// <summary>Values a block holds: 1 for a per-element type.</summary>
    public int BlockSize
    {
        get;
        init => field = value >= 1 ? value : throw new ArgumentOutOfRangeException(nameof(BlockSize), value, "A block holds at least one value.");
    } = 1;

    /// <summary>The bits a block takes, its scales and zero points included.</summary>
    public required int BlockBits
    {
        get;
        init => field = value >= 1 ? value : throw new ArgumentOutOfRangeException(nameof(BlockBits), value, "A block takes at least one bit.");
    }

    /// <summary>The bits a value takes on average: <see cref="BlockBits"/> / <see cref="BlockSize"/> (fractional for most block formats).</summary>
    public double BitsPerValue => (double)BlockBits / BlockSize;

    /// <summary>Whether values are floating point (false for integers, truth values and integer codes with scales).</summary>
    public bool IsFloat { get; init; }

    /// <summary>Whether values may be negative.</summary>
    public bool IsSigned { get; init; }

    /// <summary>
    /// Whether values are whole numbers of <see cref="BlockBits"/> bits, two's complement when <see cref="IsSigned"/> (int64
    /// down to int2, uint64 down to uint2): ids and indices may be stored in one (<see cref="ElementTypes.ChooseIds"/>).
    /// </summary>
    public bool IsInteger { get; init; }

    /// <summary>The smallest value an integer type holds (0 for an unsigned one).</summary>
    /// <exception cref="InvalidOperationException">Not an integer type.</exception>
    public Int128 MinValue => IsInteger ? IsSigned ? -((Int128)1 << (BlockBits - 1)) : 0 : throw NotInteger();

    /// <summary>The largest value an integer type holds.</summary>
    /// <exception cref="InvalidOperationException">Not an integer type.</exception>
    public Int128 MaxValue => IsInteger ? ((Int128)1 << (IsSigned ? BlockBits - 1 : BlockBits)) - 1 : throw NotInteger();

    /// <summary>Whether an integer type holds every value in [<paramref name="smallest"/>, <paramref name="largest"/>]; false for any other type.</summary>
    public bool Holds(long smallest, long largest) => IsInteger && smallest >= MinValue && largest <= MaxValue;

    private InvalidOperationException NotInteger() => new($"{Name} is not an integer type: it has no range of whole numbers.");

    /// <summary>Whether it is a block format: more than one value a block, or scales.</summary>
    public bool IsBlock => BlockSize > 1 || Scales is not null;

    /// <summary>A block format's scales and zero points, or null (a per-element type, or a format that does not describe them).</summary>
    public BlockScales? Scales { get; init; }

    /// <summary>
    /// How a float type's bits read (<see cref="FloatBits"/>), or null (not a float, or one whose special values are not
    /// IEEE 754's): what a kernel reads to widen it without naming the type.
    /// </summary>
    public FloatBits? Float { get; init; }

    /// <summary>
    /// How a block format of integer codes and one scale a block reads (<see cref="ScaledBlock"/>), or null (not such a
    /// format, or one laid out otherwise): what a kernel reads to widen it without naming the format.
    /// </summary>
    public ScaledBlock? Scaled { get; init; }

    /// <summary>
    /// How a super-block format of planes of codes, scaled sub-blocks and a super-scale reads (<see cref="SuperBlock"/>:
    /// ggml's k-quants), or null: what a kernel reads to widen it without naming the format.
    /// </summary>
    public SuperBlock? Super { get; init; }

    /// <summary>
    /// How a grid-codebook format reads (<see cref="GridBlock"/>: ggml's IQ2_XXS, IQ3_XXS), or null: what a kernel reads
    /// to widen it without naming the format.
    /// </summary>
    public GridBlock? Grid { get; init; }

    /// <summary>
    /// The operations that read or write the type natively, by name (<c>Operation.Name</c>; "*" for every one): any other
    /// reads it through a counted conversion.
    /// </summary>
    public IReadOnlyList<string> Kernels { get; init; } = [];

    /// <summary>Its widening to float32 on the CPU, or null when it has none.</summary>
    public ElementDecoder? ToFloat32 { get; init; }

    /// <summary>Its storing from float32 on the CPU, or null when it has none (a format read only, such as most of ggml's).</summary>
    public ElementEncoder? FromFloat32 { get; init; }

    /// <summary>The bytes <paramref name="values"/> values take: whole blocks, a last partial byte rounded up.</summary>
    /// <exception cref="ArgumentException">A block format given a count that is not whole blocks.</exception>
    public long ByteCount(long values)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(values);
        if (BlockSize > 1 && values % BlockSize != 0)
        {
            throw new ArgumentException($"{Name} stores whole blocks of {BlockSize} values; {values} is not.", nameof(values));
        }

        return checked((values / BlockSize * BlockBits + 7) / 8);
    }

    /// <summary>The 4-byte words a storage of <paramref name="values"/> values of this type takes (what <c>Backend.Allocate</c> gives).</summary>
    public long StorageWords(long values) => (ByteCount(values) + 3) / 4;

    /// <summary>Widens <paramref name="destination"/>.Length stored values from <paramref name="source"/> to float32.</summary>
    /// <exception cref="NotSupportedException">The type has no conversion to float32.</exception>
    public void Decode(ReadOnlySpan<byte> source, Span<float> destination)
    {
        var decode = ToFloat32 ?? throw new NotSupportedException($"The element type {Name} has no conversion to float32.");
        if (source.Length < ByteCount(destination.Length))
        {
            throw new ArgumentException($"{destination.Length} {Name} values take {ByteCount(destination.Length)} bytes; {source.Length} given.", nameof(source));
        }

        decode(source, destination);
    }

    /// <summary>Stores <paramref name="source"/> in this type into <paramref name="destination"/>.</summary>
    /// <exception cref="NotSupportedException">The type has no conversion from float32.</exception>
    public void Encode(ReadOnlySpan<float> source, Span<byte> destination)
    {
        var encode = FromFloat32 ?? throw new NotSupportedException($"The element type {Name} has no conversion from float32.");
        if (destination.Length < ByteCount(source.Length))
        {
            throw new ArgumentException($"{source.Length} {Name} values take {ByteCount(source.Length)} bytes; {destination.Length} given.", nameof(destination));
        }

        encode(source, destination);
    }

    /// <summary>"int32 (32 bits a value)", "q4_k (4.5 bits a value, blocks of 256)".</summary>
    public string Describe() => BlockSize > 1
        ? string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{Name} ({BitsPerValue:0.###} bits a value, blocks of {BlockSize})")
        : string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{Name} ({BitsPerValue:0.###} bits a value)");

    /// <inheritdoc />
    public override string ToString() => Name;
}

/// <summary>Which element type a run computes in on a device and how it was chosen (<see cref="ElementTypes.Choose"/>).</summary>
/// <param name="Type">The type.</param>
/// <param name="Picked">Whether "auto" picked it (else the user selected it).</param>
/// <param name="Source">Where the choice came from when the user made it ("--dtype", "IDRAK_DEFAULT_DTYPE", an option's name), else null.</param>
public sealed record ElementTypeChoice(ElementType Type, bool Picked, string? Source)
{
    /// <summary>The device the run computes on ("cuda:0"), or null when the choice names none.</summary>
    public string? Device { get; init; }

    /// <summary>
    /// What the device reported (plan 37, T2): the types it computes natively and why, "bfloat16, float8_e4m3, float32
    /// (compute capability 8.9: ...)"; null when the choice names no device.
    /// </summary>
    public string? Native { get; init; }

    /// <summary>Whether the device computes <see cref="Type"/> natively (else its operations compute in float32 over it).</summary>
    public bool IsNative { get; init; } = true;

    /// <summary>
    /// What the type is chosen for, as the report names it: "element type" (the values a run computes in,
    /// <see cref="ElementTypes.Choose"/>), "id type" (ids and indices, <see cref="ElementTypes.ChooseIds"/>) or "sum type
    /// for MatMul" (an operation's sums, <see cref="SumTypes.Choose"/>).
    /// </summary>
    public string Role { get; init; } = "element type";

    /// <summary>
    /// What auto measured or found that the pick follows from ("ids from 0 to 151,935 fit uint32", the sum types' times
    /// and errors), said in place of the device's report; null for a pick from what the device reports.
    /// </summary>
    public string? Measured { get; init; }

    /// <summary>
    /// "picked by auto: cuda:0 computes bfloat16, float32 natively (...)", or "selected by the user (IDRAK_DEFAULT_DTYPE)"
    /// (with "; cpu computes it in float32" where the device has no arithmetic of the type).
    /// </summary>
    public string How => Picked
        ? Measured is not null ? $"picked by auto: {Measured}" : Native is null ? "picked by auto" : $"picked by auto: {Device} computes {Native}"
        : IsNative ? $"selected by the user ({Source})" : $"selected by the user ({Source}); {Device} computes it in float32 (it computes {Native})";

    /// <summary>"element type: bfloat16 on cuda:0 (picked by auto: ...)": the line a load's report shows (its <see cref="Role"/> first).</summary>
    public override string ToString() => Device is null ? $"{Role}: {Type.Name} ({How})" : $"{Role}: {Type.Name} on {Device} ({How})";
}

/// <summary>
/// The element types, by name (ignoring case): the per-element types float32, bfloat16, float16, float8_e4m3,
/// float8_e5m2, int64, int32, int8, uint8, int4, bool, int16, uint16, float64, uint32, uint64, uint2, float4_e2m1 and
/// float8_e8m0, and the block types int2 (2-bit codes, 32 a block with a float16 scale) and mxfp4 (OCP's: 32 E2M1 values
/// sharing an E8M0 scale), which Idrak.Abstraction registers, and the block formats the packages register (Idrak adds
/// ggml's, from its GGUF types, but those the library defines itself). The short names
/// bf16, f16, fp16, half, f32, fp32, f64, fp64, double and fp8 find their types too. A name nobody registered is refused
/// with "not registered". The type a run computes in on a device comes from a run's option (<c>--dtype</c>), else
/// <see cref="EnvironmentVariable"/> (IDRAK_DEFAULT_DTYPE), else <see cref="Auto"/> (the default, plan 37 T2), which
/// picks what the device computes natively as its backend reports it (<see cref="Devices.BackendCapabilities.NativeTypes"/>,
/// measured at run time): bfloat16 where native, float16 where only it is, float32 otherwise; float32 selected keeps every
/// result as before T2. The conversions the dispatcher makes for an operation whose kernel takes another type are counted
/// here (<see cref="Conversions"/>) and published to telemetry (<see cref="Diagnostics.ElementsConverted"/>).
/// </summary>
public static class ElementTypes
{
    /// <summary>The variable naming the default element type: "IDRAK_DEFAULT_DTYPE".</summary>
    public const string EnvironmentVariable = "IDRAK_DEFAULT_DTYPE";

    /// <summary>The choice that picks the type per device (the default): what the device computes natively, best first.</summary>
    public const string Auto = "auto";

    private static readonly KindTable<ElementType> Table = new(nameof(ElementTypes), "Element type", "add it with ElementTypes.Register",
        comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "a type decides how stored bytes read: another type cannot stand in for it");

    // The short names people write for the built-ins ("bf16"), resolved before the registry (rule 15: no string made).
    private static readonly Dictionary<string, string> ShortNames = new(StringComparer.OrdinalIgnoreCase)
    {
        ["f32"] = "float32", ["fp32"] = "float32", ["bf16"] = "bfloat16", ["f16"] = "float16", ["fp16"] = "float16", ["half"] = "float16",
        ["f64"] = "float64", ["fp64"] = "float64", ["double"] = "float64", ["fp8"] = "float8_e4m3", ["e4m3"] = "float8_e4m3", ["e5m2"] = "float8_e5m2",
    };

    private static readonly Dictionary<string, string>.AlternateLookup<ReadOnlySpan<char>> ShortLookup = ShortNames.GetAlternateLookup<ReadOnlySpan<char>>();

    private static long s_conversions, s_convertedElements;
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, long> s_byOperation = new(StringComparer.Ordinal);

    static ElementTypes()
    {
        foreach (var type in (ReadOnlySpan<ElementType>)[ElementType.Float32, ElementType.BFloat16, ElementType.Float16, ElementType.Float8E4M3, ElementType.Float8E5M2,
                     ElementType.Int64, ElementType.Int32, ElementType.Int8, ElementType.UInt8, ElementType.Int4, ElementType.Int2, ElementType.Bool,
                     ElementType.Int16, ElementType.UInt16, ElementType.Float64, ElementType.UInt32, ElementType.UInt64, ElementType.UInt2,
                     ElementType.Float4E2M1, ElementType.Float8E8M0, ElementType.MXFP4])
        {
            Table.RegisterDefault(type);
        }

        LibraryDefaults.Ensure(typeof(ElementTypes));                     // the packages' block formats (Idrak: ggml's)
    }

    /// <summary>Registers <paramref name="type"/> under its name (over the library's of that name, which stays behind it).</summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ElementType type)
    {
        ArgumentNullException.ThrowIfNull(type);
        ArgumentException.ThrowIfNullOrWhiteSpace(type.Name);
        Table.Register(type, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's type <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyList<string> Names => Table.Names;

    /// <summary>The registered types, in the order of <see cref="Names"/>.</summary>
    public static IReadOnlyList<ElementType> All => Table.Kinds;

    /// <summary>The type registered as <paramref name="name"/> (or a built-in's short name: bf16, f16, fp32, ...), or null.</summary>
    public static ElementType? Find(ReadOnlySpan<char> name) =>
        Table.Find(name) ?? (ShortLookup.TryGetValue(name.Trim(), out var full) ? Table.Find(full) : null);

    /// <summary>The type registered as <paramref name="name"/> (or a built-in's short name: bf16, f16, fp32, ...).</summary>
    /// <exception cref="NotRegisteredException">None is: the message names it and the registered ones.</exception>
    public static ElementType Get(ReadOnlySpan<char> name) => Find(name) ?? Table.Get(name);

    /// <summary>The library's type <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static ElementType? Default(string name) => Table.Default(name);

    /// <summary>Who registered the type <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered type as "name: summary", one per line.</summary>
    public static string Describe() => Table.Describe();

    /// <summary>The types a run may compute in (plan 37, T2), best first as <see cref="Auto"/> prefers them: bfloat16, float16, float32.</summary>
    public static IReadOnlyList<ElementType> ComputeTypes { get; } = [ElementType.BFloat16, ElementType.Float16, ElementType.Float32];

    /// <summary>The values <see cref="Choose"/> accepts: auto and the <see cref="ComputeTypes"/> (with their short names, bf16, f16, fp32, ...).</summary>
    public static IReadOnlyList<string> Accepted { get; } = [Auto, .. ComputeTypes.Select(t => t.Name)];

    /// <summary>
    /// The type <see cref="Auto"/> picks on a device whose backend reports <paramref name="capabilities"/>: the first of
    /// <see cref="ComputeTypes"/> it computes natively (<see cref="Devices.BackendCapabilities.NativeTypes"/>), float32 when
    /// it computes none of the others.
    /// </summary>
    public static ElementType Pick(Devices.BackendCapabilities capabilities)
    {
        ArgumentNullException.ThrowIfNull(capabilities);
        foreach (var type in ComputeTypes)
        {
            if (IsNative(capabilities, type))
            {
                return type;
            }
        }

        return ElementType.Float32;
    }

    // Whether the device computes `type` natively (float32 always: every backend's kernels compute it).
    private static bool IsNative(Devices.BackendCapabilities capabilities, ElementType type)
    {
        if (ReferenceEquals(type, ElementType.Float32))
        {
            return true;
        }

        foreach (var native in capabilities.NativeTypes)
        {
            if (ReferenceEquals(native, type))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// The type a run computes in on <paramref name="device"/> (the CPU when null): <paramref name="asked"/> (a run's
    /// option, named <paramref name="source"/> in the choice and in a refusal) when given, else
    /// <see cref="EnvironmentVariable"/>, else <see cref="Auto"/>, which picks what the device computes natively
    /// (<see cref="Pick"/>, from what its backend measured at run time). The choice says how it was made and what the
    /// device reported; a type the device has no arithmetic for is stored in it and computed in float32 (said so).
    /// </summary>
    /// <exception cref="ArgumentException">A value other than those <see cref="Accepted"/> lists (the message names them).</exception>
    public static ElementTypeChoice Choose(string? asked = null, string source = "--dtype", Device? device = null)
    {
        device ??= Device.Cpu;
        var capabilities = device.Backend.Capabilities;
        string native = $"{string.Join(", ", capabilities.NativeTypes.Select(t => t.Name))} natively ({capabilities.NativeReport})";
        string? value = asked, from = source;
        if (string.IsNullOrWhiteSpace(value))
        {
            value = Environment.GetEnvironmentVariable(EnvironmentVariable);
            from = EnvironmentVariable;
        }

        if (string.IsNullOrWhiteSpace(value) || value.Trim().Equals(Auto, StringComparison.OrdinalIgnoreCase))
        {
            return new ElementTypeChoice(Pick(capabilities), Picked: true, Source: null) { Device = device.ToString(), Native = native };
        }

        var type = Find(value.AsSpan().Trim());
        if (type is null || !ComputeTypes.Contains(type))
        {
            throw new ArgumentException($"{from} '{value}': the element type a run computes in is one of {string.Join(", ", Accepted)} (bf16, f16 and fp32 for short; auto picks what the device computes natively).");
        }

        return new ElementTypeChoice(type, Picked: false, Source: from) { Device = device.ToString(), Native = native, IsNative = IsNative(capabilities, type) };
    }

    /// <summary>The variable naming the type ids and indices are stored in: "IDRAK_ID_DTYPE".</summary>
    public const string IdEnvironmentVariable = "IDRAK_ID_DTYPE";

    /// <summary>
    /// The types ids and indices may be stored in, as auto tries them: every registered integer type of whole bytes
    /// (<see cref="ElementType.IsInteger"/>: uint8, int8, uint16, int16, uint32, int32, uint64, int64 and any an app
    /// registers), fewest bits first, unsigned before signed.
    /// </summary>
    public static IReadOnlyList<ElementType> IdTypes =>
        [.. All.Where(t => t.IsInteger && t.BlockSize == 1 && t.BlockBits % 8 == 0).OrderBy(t => t.BlockBits).ThenBy(t => t.IsSigned)];

    /// <summary>
    /// The type ids or indices in [<paramref name="smallest"/>, <paramref name="largest"/>] are stored in on
    /// <paramref name="device"/> (the CPU when null), from the data: <paramref name="asked"/> (a run's option, named
    /// <paramref name="source"/>) when given, else <see cref="IdEnvironmentVariable"/>, else <see cref="Auto"/>, which picks
    /// the smallest of <see cref="IdTypes"/> that holds them (a vocabulary of 50,000 tokens: uint16; an expert index:
    /// uint8). A kernel that reads another index type (<see cref="Devices.BackendCapabilities.IndexType"/>) gets them
    /// through the dispatcher's counted conversion.
    /// </summary>
    /// <exception cref="ArgumentException">A value that is no integer type, or a type that does not hold the data (the message names both).</exception>
    public static ElementTypeChoice ChooseIds(long largest, long smallest = 0, string? asked = null, string source = "--id-dtype", Device? device = null)
    {
        if (smallest > largest)
        {
            throw new ArgumentException($"Ids from {smallest} to {largest}: the smallest is past the largest.", nameof(smallest));
        }

        device ??= Device.Cpu;
        var capabilities = device.Backend.Capabilities;
        string range = string.Create(System.Globalization.CultureInfo.InvariantCulture, $"ids from {smallest:N0} to {largest:N0}");
        string native = $"its index kernels read {capabilities.IndexType.Name}";
        string? value = asked, from = source;
        if (string.IsNullOrWhiteSpace(value))
        {
            value = Environment.GetEnvironmentVariable(IdEnvironmentVariable);
            from = IdEnvironmentVariable;
        }

        if (string.IsNullOrWhiteSpace(value) || value.Trim().Equals(Auto, StringComparison.OrdinalIgnoreCase))
        {
            var fits = IdTypes.FirstOrDefault(t => t.Holds(smallest, largest)) ?? throw new ArgumentException($"No registered integer type holds {range}.");
            return new ElementTypeChoice(fits, Picked: true, Source: null) { Device = device.ToString(), Native = native, Role = "id type", Measured = $"{range} fit {fits.Name}" };
        }

        var type = Find(value.AsSpan().Trim());
        if (type is null || !IdTypes.Contains(type))
        {
            throw new ArgumentException($"{from} '{value}': ids are stored in one of {Auto}, {string.Join(", ", IdTypes.Select(t => t.Name))} (auto picks the smallest that holds them).");
        }

        if (!type.Holds(smallest, largest))
        {
            throw new ArgumentException($"{from} '{value}': {type.Name} holds [{type.MinValue}, {type.MaxValue}], not the data's {range}.");
        }

        return new ElementTypeChoice(type, Picked: false, Source: from) { Device = device.ToString(), Native = native, Role = "id type" };
    }

    /// <summary>
    /// The registered type of the C# type <typeparamref name="T"/>: float32 for <see cref="float"/>, float64 for
    /// <see cref="double"/>, float16 for <see cref="Half"/>, int8 to int64 and uint8 to uint64 for the integers, bool for
    /// <see cref="bool"/>; null for a C# type no built-in stores (bfloat16, fp8, int4 and the block formats have none:
    /// their bytes are read with <c>Tensor.ToBytes</c>). Tested on <c>typeof(T)</c>, which the JIT folds (rule 89: no box).
    /// </summary>
    public static ElementType? Of<T>()
        where T : unmanaged =>
        typeof(T) == typeof(float) ? ElementType.Float32
        : typeof(T) == typeof(double) ? ElementType.Float64
        : typeof(T) == typeof(Half) ? ElementType.Float16
        : typeof(T) == typeof(sbyte) ? ElementType.Int8
        : typeof(T) == typeof(byte) ? ElementType.UInt8
        : typeof(T) == typeof(short) ? ElementType.Int16
        : typeof(T) == typeof(ushort) ? ElementType.UInt16
        : typeof(T) == typeof(int) ? ElementType.Int32
        : typeof(T) == typeof(uint) ? ElementType.UInt32
        : typeof(T) == typeof(long) ? ElementType.Int64
        : typeof(T) == typeof(ulong) ? ElementType.UInt64
        : typeof(T) == typeof(bool) ? ElementType.Bool
        : null;

    /// <summary>
    /// Publishes <paramref name="choice"/> for a run on <paramref name="device"/> to telemetry
    /// (<see cref="Diagnostics.ElementTypeChosen"/>, at <see cref="Diagnostics.TelemetryLevel.Devices"/>): a load says which type
    /// it computes in on each device, how it was chosen and what the device reported.
    /// </summary>
    public static void Report(ElementTypeChoice choice, Device device)
    {
        ArgumentNullException.ThrowIfNull(choice);
        ArgumentNullException.ThrowIfNull(device);
        if (Diagnostics.Telemetry.IsEnabled(Diagnostics.TelemetryLevel.Devices))
        {
            Diagnostics.Telemetry.ElementTypeChosen(new Diagnostics.ElementTypeChosen(device.ToString(), choice.Type.Name, choice.How, choice.Native ?? "", choice.Role));
        }
    }

    /// <summary>Conversions the dispatcher made so far for an operation whose kernel takes another type (<see cref="ConvertedElements"/> elements).</summary>
    public static long Conversions => Interlocked.Read(ref s_conversions);

    /// <summary>The elements those conversions moved.</summary>
    public static long ConvertedElements => Interlocked.Read(ref s_convertedElements);

    /// <summary>The conversions so far by operation ("Gather", "RouteTopK", ...).</summary>
    public static IReadOnlyDictionary<string, long> ConversionsByOperation => new Dictionary<string, long>(s_byOperation, StringComparer.Ordinal);

    /// <summary>
    /// Counts one conversion <paramref name="backend"/> made for <paramref name="operation"/> because its kernel takes
    /// another type than its operand's (plan 37, T0; the brief: never widened silently): <paramref name="elements"/>
    /// elements of <paramref name="from"/> converted to <paramref name="to"/>, in <see cref="Conversions"/>,
    /// <see cref="ConversionsByOperation"/> and the telemetry (<see cref="Diagnostics.ElementsConverted"/>), as the
    /// dispatcher counts its own. A backend whose kernel for a type is missing and that runs the operation through another
    /// type (bfloat16 values computed as float32) counts each such conversion here.
    /// </summary>
    public static void CountConversion(Devices.Backend backend, Operations.Operation operation, ElementType from, ElementType to, long elements)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(operation);
        ArgumentNullException.ThrowIfNull(from);
        ArgumentNullException.ThrowIfNull(to);
        ArgumentOutOfRangeException.ThrowIfNegative(elements);
        Converted(backend.Kind, operation.Name, from, to, elements);
    }

    // Counts one conversion for `operation` and publishes it while someone listens at TelemetryLevel.Devices.
    internal static void Converted(string device, string operation, ElementType from, ElementType to, long elements)
    {
        Interlocked.Increment(ref s_conversions);
        Interlocked.Add(ref s_convertedElements, elements);
        s_byOperation.AddOrUpdate(operation, 1, static (_, n) => n + 1);
        if (Diagnostics.Telemetry.IsEnabled(Diagnostics.TelemetryLevel.Devices))
        {
            Diagnostics.Telemetry.ElementsConverted(new Diagnostics.ElementsConverted(device, operation, from.Name, to.Name, elements));
        }
    }
}

// The built-in types' conversions on the CPU: whole vectors where the type is a plain number (a scalar tail for the rest),
// value by value where it is packed or coded.
internal static class ElementConversions
{
    public static void Float32ToFloat32(ReadOnlySpan<byte> source, Span<float> destination) =>
        MemoryMarshal.Cast<byte, float>(source)[..destination.Length].CopyTo(destination);

    public static void Float32FromFloat32(ReadOnlySpan<float> source, Span<byte> destination) =>
        MemoryMarshal.AsBytes(source).CopyTo(destination);

    public static void BFloat16ToFloat32(ReadOnlySpan<byte> source, Span<float> destination) =>
        HalfFloats.WidenBFloat16(MemoryMarshal.Cast<byte, ushort>(source)[..destination.Length], destination);

    public static void BFloat16FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        var halves = MemoryMarshal.Cast<byte, ushort>(destination);
        for (int i = 0; i < source.Length; i++)
        {
            halves[i] = Devices.BFloat16Bits.Round(source[i]);
        }
    }

    public static void Float16ToFloat32(ReadOnlySpan<byte> source, Span<float> destination) =>
        HalfFloats.WidenFloat16(MemoryMarshal.Cast<byte, ushort>(source)[..destination.Length], destination);

    public static void Float16FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        var halves = MemoryMarshal.Cast<byte, ushort>(destination);
        for (int i = 0; i < source.Length; i++)
        {
            halves[i] = BitConverter.HalfToUInt16Bits((Half)source[i]);   // round to nearest, ties to even
        }
    }

    // Every fp8 code's value (a NaN code as the quiet NaN of its sign), as PyTorch and ml_dtypes widen them.
    private static readonly float[] E4M3 = Float8Table(e4m3: true), E5M2 = Float8Table(e4m3: false);

    private static float[] Float8Table(bool e4m3)
    {
        var table = new float[256];
        for (int code = 0; code < 256; code++)
        {
            uint sign = (uint)(code >> 7) << 31;
            float value;
            if (e4m3)
            {
                int e = (code >> 3) & 15, m = code & 7;
                value = e == 15 && m == 7 ? float.NaN
                    : e == 0 ? m * (1f / 512f)
                    : BitConverter.UInt32BitsToSingle((uint)(e - 7 + 127) << 23 | (uint)m << 20);
            }
            else
            {
                value = (float)BitConverter.UInt16BitsToHalf((ushort)((code & 0x7F) << 8));
            }

            table[code] = float.IsNaN(value) ? BitConverter.UInt32BitsToSingle(sign | 0x7FC00000u)
                : BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(value) | sign);
        }

        return table;
    }

    public static void E4M3ToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = E4M3[source[i]];
        }
    }

    public static void E5M2ToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = E5M2[source[i]];
        }
    }

    // E4M3fn: codes 0..126 finite (0x7E = 448), 0x7F NaN; past 448 + half a step (464, ties to the even 448) is NaN.
    public static void E4M3FromFloat32(ReadOnlySpan<float> source, Span<byte> destination) => Float8From(source, destination, E4M3, 0x7E, 0x7F);

    // E5M2: codes 0..123 finite (0x7B = 57344), 0x7C infinity; past 57344 + half a step (61440, ties to the even 0x7C) is infinity.
    public static void E5M2FromFloat32(ReadOnlySpan<float> source, Span<byte> destination) => Float8From(source, destination, E5M2, 0x7B, 0x7C);

    // Round to the nearest code (ties to the even code), the codes' values ascending from 0 to `max`; past the largest the
    // code `overflow` (NaN or infinity), NaN to the NaN code (0x7F for both).
    private static void Float8From(ReadOnlySpan<float> source, Span<byte> destination, float[] table, int max, int overflow)
    {
        float top = table[max], next = max == 0x7E ? 480f : 65536f;
        for (int i = 0; i < source.Length; i++)
        {
            float x = source[i];
            byte sign = (byte)(BitConverter.SingleToUInt32Bits(x) >> 24 & 0x80);
            float a = MathF.Abs(x);
            int code;
            if (float.IsNaN(x))
            {
                code = 0x7F;
            }
            else if (a > top)
            {
                float mid = (top + next) / 2f;
                code = a < mid || a == mid && (max & 1) == 0 ? max : overflow;
            }
            else
            {
                int lo = 0, hi = max;                                       // the largest code whose value is at most a
                while (lo < hi)
                {
                    int m = (lo + hi + 1) >> 1;
                    if (table[m] <= a)
                    {
                        lo = m;
                    }
                    else
                    {
                        hi = m - 1;
                    }
                }

                code = lo;
                if (lo < max)
                {
                    float below = a - table[lo], above = table[lo + 1] - a;
                    if (above < below || above == below && (lo & 1) == 1)
                    {
                        code = lo + 1;
                    }
                }
            }

            destination[i] = (byte)(code | sign);
        }
    }

    public static void Int64ToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        var values = MemoryMarshal.Cast<byte, long>(source);
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = values[i];
        }
    }

    public static void Int64FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        var values = MemoryMarshal.Cast<byte, long>(destination);
        for (int i = 0; i < source.Length; i++)
        {
            values[i] = float.IsNaN(source[i]) ? 0 : (long)Math.Clamp(MathF.Round(source[i]), long.MinValue, long.MaxValue);
        }
    }

    public static void Int32ToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        var values = MemoryMarshal.Cast<byte, int>(source)[..destination.Length];
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            for (; i <= values.Length - Vector<int>.Count; i += Vector<int>.Count)
            {
                Vector.ConvertToSingle(new Vector<int>(values[i..])).CopyTo(destination[i..]);
            }
        }

        for (; i < values.Length; i++)
        {
            destination[i] = values[i];
        }
    }

    // Rounded to nearest (ties to even) and saturated to the type's range; NaN gives 0.
    public static void Int32FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        var values = MemoryMarshal.Cast<byte, int>(destination);
        for (int i = 0; i < source.Length; i++)
        {
            float x = MathF.Round(source[i]);
            values[i] = float.IsNaN(x) ? 0 : x >= 2147483648f ? int.MaxValue : x <= -2147483648f ? int.MinValue : (int)x;
        }
    }

    // Each signed byte to its float32 (exact): whole vectors widened sbyte → short → int → float, then a scalar tail.
    public static void Int8ToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        var values = MemoryMarshal.Cast<byte, sbyte>(source)[..destination.Length];
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int width = Vector<sbyte>.Count, quarter = Vector<int>.Count;
            for (; i <= values.Length - width; i += width)
            {
                Vector.Widen(new Vector<sbyte>(values[i..]), out Vector<short> low, out Vector<short> high);
                Vector.Widen(low, out Vector<int> a, out Vector<int> b);
                Vector.Widen(high, out Vector<int> c, out Vector<int> d);
                Vector.ConvertToSingle(a).CopyTo(destination[i..]);
                Vector.ConvertToSingle(b).CopyTo(destination[(i + quarter)..]);
                Vector.ConvertToSingle(c).CopyTo(destination[(i + 2 * quarter)..]);
                Vector.ConvertToSingle(d).CopyTo(destination[(i + 3 * quarter)..]);
            }
        }

        for (; i < values.Length; i++)
        {
            destination[i] = values[i];
        }
    }

    public static void Int8FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        for (int i = 0; i < source.Length; i++)
        {
            destination[i] = (byte)(sbyte)Saturate(source[i], -128, 127);
        }
    }

    public static void UInt8ToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = source[i];
        }
    }

    public static void UInt8FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        for (int i = 0; i < source.Length; i++)
        {
            destination[i] = (byte)Saturate(source[i], 0, 255);
        }
    }

    public static void Int4ToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        for (int i = 0; i < destination.Length; i++)
        {
            int nibble = source[i >> 1] >> ((i & 1) * 4) & 0xF;
            destination[i] = (nibble ^ 8) - 8;                                  // sign-extend 4 bits
        }
    }

    public static void Int4FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        destination[..(int)((source.Length + 1L) / 2)].Clear();
        for (int i = 0; i < source.Length; i++)
        {
            destination[i >> 1] |= (byte)((Saturate(source[i], -8, 7) & 0xF) << ((i & 1) * 4));
        }
    }

    // int2's blocks, by its description (ScaledToFloat32).
    public static void Int2ToFloat32(ReadOnlySpan<byte> source, Span<float> destination) => ScaledToFloat32(ElementType.Int2.Scaled!, source, destination);

    /// <summary>
    /// Whole blocks of a scaled-block type (<see cref="ScaledBlock"/>) widened by its description alone: each value
    /// (code's value + bias) · scale, the code's value its code type's value, its table entry or the code (two's complement
    /// where signed), the scale its scale type's value or a float of the description's bits, read through a table of every
    /// code byte's values made once a description (<see cref="ScaledPlan"/>).
    /// </summary>
    public static void ScaledToFloat32(ScaledBlock block, ReadOnlySpan<byte> source, Span<float> destination)
    {
        var plan = ScaledPlan.Of(block);
        int per = plan.PerByte, codeBytes = block.Values * block.CodeBits / 8;
        for (int b = 0; b < destination.Length / block.Values; b++)
        {
            var bytes = source.Slice(b * block.Bytes, block.Bytes);
            float scale = plan.Scale(bytes);
            var into = destination.Slice(b * block.Values, block.Values);
            var codes = bytes.Slice(block.CodeOffset, codeBytes);
            for (int i = 0; i < codeBytes; i++)
            {
                var values = plan.Values.AsSpan(codes[i] * per, per);
                for (int j = 0; j < per; j++)
                {
                    into[block.NibblesSplit ? i + j * codeBytes : i * per + j] = values[j] * scale;
                }
            }
        }
    }

    /// <summary>A scaled-block description read once: each code byte's values before the scale (code's value + bias), the scale's reading.</summary>
    internal sealed class ScaledPlan
    {
        private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<ScaledBlock, ScaledPlan> Plans = new();

        private readonly ScaledBlock block;

        private ScaledPlan(ScaledBlock block)
        {
            this.block = block;
            PerByte = 8 / block.CodeBits;
            int mask = (1 << block.CodeBits) - 1;
            Span<float> codeValues = stackalloc float[mask + 1];
            Span<byte> one = stackalloc byte[1];
            for (int code = 0; code <= mask; code++)
            {
                if (block.CodeType is { } type)
                {
                    one[0] = (byte)code;                                    // the code as the type's one value, in its lowest bits
                    type.Decode(one, codeValues.Slice(code, 1));
                }
                else
                {
                    codeValues[code] = block.CodeTable is { } table ? table[code] : block.SignedCodes ? code << (32 - block.CodeBits) >> (32 - block.CodeBits) : code;
                }
            }

            Values = new float[256 * PerByte];
            for (int b = 0; b < 256; b++)
            {
                for (int j = 0; j < PerByte; j++)
                {
                    Values[b * PerByte + j] = codeValues[b >> (j * block.CodeBits) & mask] + block.CodeBias;
                }
            }
        }

        /// <summary>Codes a byte holds (8 / CodeBits).</summary>
        public int PerByte { get; }

        /// <summary>Byte b's codes' values (before the scale) at [b · PerByte, (b + 1) · PerByte), its lowest bits first.</summary>
        public float[] Values { get; }

        /// <summary>The plan of <paramref name="block"/> (codes of 1, 2, 4 or 8 bits).</summary>
        /// <exception cref="NotSupportedException">Codes of another width.</exception>
        public static ScaledPlan Of(ScaledBlock block) => block.CodeBits is 1 or 2 or 4 or 8
            ? Plans.GetValue(block, static b => new ScaledPlan(b))
            : throw new NotSupportedException($"A scaled block's codes of {block.CodeBits} bits are not read by bytes.");

        /// <summary>The block's scale: its scale type's value, else a float of the description's bits (float16, bfloat16 or float32).</summary>
        public float Scale(ReadOnlySpan<byte> bytes)
        {
            var at = bytes[block.ScaleOffset..];
            if (block.ScaleType is { } type)
            {
                Span<float> value = stackalloc float[1];
                type.Decode(at[..(int)type.ByteCount(1)], value);
                return value[0];
            }

            return block.Scale switch
            {
                { Exponent: 5, Mantissa: 10 } => (float)System.Buffers.Binary.BinaryPrimitives.ReadHalfLittleEndian(at),
                { Exponent: 8, Mantissa: 7 } => BitConverter.Int32BitsToSingle(System.Buffers.Binary.BinaryPrimitives.ReadUInt16LittleEndian(at) << 16),
                { Exponent: 8, Mantissa: 23 } => System.Buffers.Binary.BinaryPrimitives.ReadSingleLittleEndian(at),
                var other => throw new NotSupportedException($"A scale of {other.Exponent} exponent and {other.Mantissa} mantissa bits is not read here."),
            };
        }
    }

    /// <summary>
    /// int2's blocks from float32 values: per block of 32, the float16 scale d whose codes clamp(round(x / d), −2, 1) give the
    /// least squared error among candidates (the codes reach twice as far below zero as above, so d's sign puts the value
    /// of largest magnitude on −2 or on 1: d = −xmax / t for t in 1.5, 2, 2.5, 3, and d = xmax / t for t in 1, 1.25, 1.5,
    /// each refined once by least squares over its codes), then the codes rounded again with d as stored. A block of zeros
    /// has d = 0 and every code 0.
    /// </summary>
    public static void Int2FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        ReadOnlySpan<float> below = [1.5f, 2f, 2.5f, 3f], above = [1f, 1.25f, 1.5f];
        for (int b = 0; b < source.Length / 32; b++)
        {
            var x = source.Slice(b * 32, 32);
            var block = destination.Slice(b * 10, 10);
            float xmax = 0f;
            foreach (float v in x)
            {
                if (MathF.Abs(v) > MathF.Abs(xmax))
                {
                    xmax = v;
                }
            }

            float best = 0f;
            double bestError = double.MaxValue;
            if (xmax != 0f && float.IsFinite(xmax))
            {
                foreach (float t in below)
                {
                    Try(x, -xmax / t, ref best, ref bestError);
                }

                foreach (float t in above)
                {
                    Try(x, xmax / t, ref best, ref bestError);
                }
            }

            var d = (Half)best;
            System.Buffers.Binary.BinaryPrimitives.WriteHalfLittleEndian(block, d);
            float stored = (float)d, inverse = stored != 0f ? 1f / stored : 0f;
            for (int i = 0; i < 8; i++)
            {
                int bits = 0;
                for (int j = 0; j < 4; j++)
                {
                    bits |= (Code(x[4 * i + j] * inverse) & 3) << (2 * j);
                }

                block[2 + i] = (byte)bits;
            }
        }

        static int Code(float t) => float.IsNaN(t) ? 0 : (int)Math.Clamp(MathF.Round(t), -2f, 1f);

        // The squared error of the codes for scale d, and of their least-squares scale; the better kept in (best, error).
        static void Try(ReadOnlySpan<float> x, float d, ref float best, ref double error)
        {
            for (int pass = 0; pass < 2 && d != 0f && float.IsFinite(d); pass++)
            {
                double xq = 0, qq = 0, e = 0;
                float inverse = 1f / d;
                foreach (float v in x)
                {
                    int q = Code(v * inverse);
                    xq += (double)v * q;
                    qq += q * q;
                    double r = v - (double)q * d;
                    e += r * r;
                }

                if (e < error)
                {
                    (best, error) = (d, e);
                }

                d = qq > 0 ? (float)(xq / qq) : 0f;
            }
        }
    }

    // fp4 E2M1's 16 codes' values: the sign in bit 3, then 2 exponent bits and one mantissa bit (exponent 0: 0 or 0.5).
    private static readonly float[] E2M1 = [0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f, -0f, -0.5f, -1f, -1.5f, -2f, -3f, -4f, -6f];

    // Two a byte, the first in the low half.
    public static void E2M1ToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = E2M1[source[i >> 1] >> ((i & 1) * 4) & 0xF];
        }
    }

    // Rounded to the nearest value (ties to the even code: mantissa 0), saturated to ±6; NaN gives 0 (E2M1 has none).
    public static void E2M1FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        destination[..(int)((source.Length + 1L) / 2)].Clear();
        for (int i = 0; i < source.Length; i++)
        {
            destination[i >> 1] |= (byte)(E2M1Code(source[i]) << ((i & 1) * 4));
        }
    }

    // The E2M1 code nearest x (ties to the even code), saturated to ±6, NaN as +0.
    private static int E2M1Code(float x)
    {
        if (float.IsNaN(x))
        {
            return 0;
        }

        float a = MathF.Abs(x);
        int code = 7;
        for (int c = 0; c < 7; c++)
        {
            float mid = (E2M1[c] + E2M1[c + 1]) / 2f;
            if (a < mid || a == mid && (c & 1) == 0)
            {
                code = c;
                break;
            }
        }

        return code | (float.IsNegative(x) ? 8 : 0);
    }

    // E8M0: 2^(e - 127); 0 is 2^-127 (a float32 subnormal, exact), 0xFF NaN.
    public static void E8M0ToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        for (int i = 0; i < destination.Length; i++)
        {
            int e = source[i];
            destination[i] = e == 255 ? float.NaN : e == 0 ? BitConverter.UInt32BitsToSingle(0x00400000u) : BitConverter.UInt32BitsToSingle((uint)e << 23);
        }
    }

    // The power of two at or below |x| (floor(log2 |x|)), clamped to [2^-127, 2^127]; NaN and the infinities as NaN (0xFF),
    // 0 and below as the smallest (2^-127: E8M0 has no zero).
    public static void E8M0FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        for (int i = 0; i < source.Length; i++)
        {
            float x = source[i];
            destination[i] = (byte)(!float.IsFinite(x) ? 255 : x <= 0f ? 0 : Math.Clamp(MathF.ILogB(x) + 127, 0, 254));
        }
    }

    // MXFP4's blocks, by their description (ScaledToFloat32: each E2M1 code's value times the E8M0 scale).
    public static void MxFp4ToFloat32(ReadOnlySpan<byte> source, Span<float> destination) => ScaledToFloat32(ElementType.MXFP4.Scaled!, source, destination);

    /// <summary>
    /// MXFP4's blocks from float32 values, as OCP's conversion makes them: per block of 32, the shared exponent
    /// floor(log2(max |v|)) − 2 (E2M1's largest exponent), clamped to E8M0's range (a block of zeros: 2^-127), and each value
    /// over 2^exponent rounded to the nearest E2M1 value (ties to even), saturated to ±6; the scale byte, then value i in
    /// byte i's low half and value i + 16 in its high half.
    /// </summary>
    public static void MxFp4FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        for (int b = 0; b < source.Length / 32; b++)
        {
            var values = source.Slice(b * 32, 32);
            var block = destination.Slice(b * 17, 17);
            float amax = 0f;
            foreach (float v in values)
            {
                if (float.IsFinite(v))
                {
                    amax = MathF.Max(amax, MathF.Abs(v));
                }
            }

            int shared = amax > 0f ? Math.Clamp(MathF.ILogB(amax) - 2, -127, 127) : -127;
            block[0] = (byte)(shared + 127);
            float inverse = MathF.ScaleB(1f, -shared);
            for (int j = 0; j < 16; j++)
            {
                block[1 + j] = (byte)(E2M1Code(values[j] * inverse) | E2M1Code(values[16 + j] * inverse) << 4);
            }
        }
    }

    public static void BoolToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = source[i] != 0 ? 1f : 0f;
        }
    }

    public static void BoolFromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        for (int i = 0; i < source.Length; i++)
        {
            destination[i] = source[i] != 0f && !float.IsNaN(source[i]) ? (byte)1 : (byte)0;
        }
    }

    // Widened a vector at a time (the halves of each 16-bit vector to int32, then to float32), a scalar tail for the rest.
    public static void Int16ToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        var values = MemoryMarshal.Cast<byte, short>(source)[..destination.Length];
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            for (; i <= values.Length - Vector<short>.Count; i += Vector<short>.Count)
            {
                Vector.Widen(new Vector<short>(values[i..]), out var low, out var high);
                Vector.ConvertToSingle(low).CopyTo(destination[i..]);
                Vector.ConvertToSingle(high).CopyTo(destination[(i + Vector<int>.Count)..]);
            }
        }

        for (; i < values.Length; i++)
        {
            destination[i] = values[i];
        }
    }

    public static void Int16FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        var values = MemoryMarshal.Cast<byte, short>(destination);
        for (int i = 0; i < source.Length; i++)
        {
            values[i] = (short)Saturate(source[i], short.MinValue, short.MaxValue);
        }
    }

    public static void UInt16ToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        var values = MemoryMarshal.Cast<byte, ushort>(source)[..destination.Length];
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            for (; i <= values.Length - Vector<ushort>.Count; i += Vector<ushort>.Count)
            {
                Vector.Widen(new Vector<ushort>(values[i..]), out var low, out var high);
                Vector.ConvertToSingle(Vector.AsVectorInt32(low)).CopyTo(destination[i..]);     // at most 65,535: exact as int32
                Vector.ConvertToSingle(Vector.AsVectorInt32(high)).CopyTo(destination[(i + Vector<int>.Count)..]);
            }
        }

        for (; i < values.Length; i++)
        {
            destination[i] = values[i];
        }
    }

    public static void UInt16FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        var values = MemoryMarshal.Cast<byte, ushort>(destination);
        for (int i = 0; i < source.Length; i++)
        {
            values[i] = (ushort)Saturate(source[i], ushort.MinValue, ushort.MaxValue);
        }
    }

    // Rounded to the nearest float32 (exact up to 2^24, as int32's).
    public static void UInt32ToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        var values = MemoryMarshal.Cast<byte, uint>(source);
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = values[i];
        }
    }

    // Rounded to nearest (ties to even) and saturated to [0, 2^32 - 1]; NaN gives 0.
    public static void UInt32FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        var values = MemoryMarshal.Cast<byte, uint>(destination);
        for (int i = 0; i < source.Length; i++)
        {
            float x = MathF.Round(source[i]);
            values[i] = float.IsNaN(x) || x <= 0f ? 0u : x >= 4294967296f ? uint.MaxValue : (uint)x;
        }
    }

    public static void UInt64ToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        var values = MemoryMarshal.Cast<byte, ulong>(source);
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = values[i];
        }
    }

    // Rounded to nearest (ties to even) and saturated to [0, 2^64 - 1]; NaN gives 0.
    public static void UInt64FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        var values = MemoryMarshal.Cast<byte, ulong>(destination);
        for (int i = 0; i < source.Length; i++)
        {
            float x = MathF.Round(source[i]);
            values[i] = float.IsNaN(x) || x <= 0f ? 0ul : x >= 18446744073709551616f ? ulong.MaxValue : (ulong)x;
        }
    }

    public static void UInt2ToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = source[i >> 2] >> ((i & 3) * 2) & 0x3;
        }
    }

    public static void UInt2FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        destination[..(int)((source.Length + 3L) / 4)].Clear();
        for (int i = 0; i < source.Length; i++)
        {
            destination[i >> 2] |= (byte)((Saturate(source[i], 0, 3) & 0x3) << ((i & 3) * 2));
        }
    }

    // Narrowed to the nearest float32 (ties to even; past float32's range an infinity, as a cast does).
    public static void Float64ToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        var values = MemoryMarshal.Cast<byte, double>(source);
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = (float)values[i];
        }
    }

    // Exact: every float32 is a float64.
    public static void Float64FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        var values = MemoryMarshal.Cast<byte, double>(destination);
        for (int i = 0; i < source.Length; i++)
        {
            values[i] = source[i];
        }
    }

    // Rounded to nearest (ties to even), clamped to [low, high]; NaN gives 0.
    private static int Saturate(float x, int low, int high) => float.IsNaN(x) ? 0 : (int)Math.Clamp(MathF.Round(x), low, high);

    /// <summary>
    /// Converts <paramref name="count"/> elements of <paramref name="from"/> in <paramref name="source"/> to
    /// <paramref name="to"/> in <paramref name="destination"/>. Indices between float32, int32 and int64 are exact or
    /// refused (an index that is not a whole number, or past what the other type holds exactly: float32 holds integers up
    /// to 2²⁴); between any two integer types of whole bytes (uint8, uint16, int16, uint32, ...) exact, or refused for a
    /// value past the destination's range; any other pair goes through float32 with the types' own conversions.
    /// </summary>
    /// <exception cref="ArgumentException">A value the destination type cannot hold exactly (integers), naming it.</exception>
    public static void Convert(ElementType from, ReadOnlySpan<byte> source, ElementType to, Span<byte> destination, int count)
    {
        if (ReferenceEquals(from, to))
        {
            int bytes = (int)from.ByteCount(count);
            source[..bytes].CopyTo(destination);
            return;
        }

        if (ReferenceEquals(to, ElementType.Int32) && ReferenceEquals(from, ElementType.Float32))
        {
            FloatsToInt32(MemoryMarshal.Cast<byte, float>(source)[..count], MemoryMarshal.Cast<byte, int>(destination)[..count]);
            return;
        }

        if (ReferenceEquals(to, ElementType.Float32) && ReferenceEquals(from, ElementType.Int32))
        {
            Int32ToFloats(MemoryMarshal.Cast<byte, int>(source)[..count], MemoryMarshal.Cast<byte, float>(destination)[..count]);
            return;
        }

        if (ReferenceEquals(to, ElementType.Int32) && ReferenceEquals(from, ElementType.Int64))
        {
            var longs = MemoryMarshal.Cast<byte, long>(source);
            var ints = MemoryMarshal.Cast<byte, int>(destination);
            for (int i = 0; i < count; i++)
            {
                long v = longs[i];
                ints[i] = v is >= int.MinValue and <= int.MaxValue ? (int)v : throw new ArgumentException($"Index {v} (element {i}) is past what int32 holds: the index kernels read int32.");
            }

            return;
        }

        if (ReferenceEquals(to, ElementType.Int64) && ReferenceEquals(from, ElementType.Int32))
        {
            var ints = MemoryMarshal.Cast<byte, int>(source);
            var longs = MemoryMarshal.Cast<byte, long>(destination);
            for (int i = 0; i < count; i++)
            {
                longs[i] = ints[i];
            }

            return;
        }

        if (ReferenceEquals(to, ElementType.Float32) && ReferenceEquals(from, ElementType.Int64))
        {
            var longs = MemoryMarshal.Cast<byte, long>(source);
            var floats = MemoryMarshal.Cast<byte, float>(destination);
            for (int i = 0; i < count; i++)
            {
                long v = longs[i];
                floats[i] = Math.Abs(v) <= 1L << 24 ? v : throw PastFloat(v, i);
            }

            return;
        }

        if (ReferenceEquals(to, ElementType.Int64) && ReferenceEquals(from, ElementType.Float32))
        {
            var floats = MemoryMarshal.Cast<byte, float>(source);
            var longs = MemoryMarshal.Cast<byte, long>(destination);
            for (int i = 0; i < count; i++)
            {
                float v = floats[i];
                longs[i] = MathF.Truncate(v) == v && MathF.Abs(v) < 9.2e18f ? (long)v : throw NotWhole(v, i);
            }

            return;
        }

        // Between two integer types of whole bytes (ids and indices: uint8, uint16, int32, ...): exact, or refused for a
        // value the destination does not hold.
        if (from.IsInteger && to.IsInteger && from.BlockBits % 8 == 0 && to.BlockBits % 8 == 0)
        {
            for (int i = 0; i < count; i++)
            {
                var v = ReadInteger(from, source, i);
                if (v < to.MinValue || v > to.MaxValue)
                {
                    throw new ArgumentException($"Index {v} (element {i}) is past what {to.Name} holds ([{to.MinValue}, {to.MaxValue}]).");
                }

                WriteInteger(to, destination, i, v);
            }

            return;
        }

        // Any other pair through float32, a chunk at a time (whole blocks).
        int chunk = Math.Max(from.BlockSize, to.BlockSize) * Math.Max(1, 65536 / Math.Max(from.BlockSize, to.BlockSize));
        float[] values = ArrayPool<float>.Shared.Rent(Math.Min(count, chunk));
        try
        {
            for (int at = 0; at < count; at += chunk)
            {
                int n = Math.Min(chunk, count - at);
                var part = values.AsSpan(0, n);
                from.Decode(source[(int)from.ByteCount(at)..], part);
                to.Encode(part, destination[(int)to.ByteCount(at)..]);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(values);
        }
    }

    // Element i of an integer type of whole bytes, little-endian.
    private static Int128 ReadInteger(ElementType type, ReadOnlySpan<byte> source, int i) => (type.BlockBits, type.IsSigned) switch
    {
        (8, true) => (sbyte)source[i],
        (8, false) => source[i],
        (16, true) => System.Buffers.Binary.BinaryPrimitives.ReadInt16LittleEndian(source[(2 * i)..]),
        (16, false) => System.Buffers.Binary.BinaryPrimitives.ReadUInt16LittleEndian(source[(2 * i)..]),
        (32, true) => System.Buffers.Binary.BinaryPrimitives.ReadInt32LittleEndian(source[(4 * i)..]),
        (32, false) => System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(source[(4 * i)..]),
        (64, true) => System.Buffers.Binary.BinaryPrimitives.ReadInt64LittleEndian(source[(8 * i)..]),
        (64, false) => System.Buffers.Binary.BinaryPrimitives.ReadUInt64LittleEndian(source[(8 * i)..]),
        _ => throw new NotSupportedException($"{type.Name}: an integer of {type.BlockBits} bits is not read as whole bytes."),
    };

    // Writes v (held by the type) as element i, little-endian.
    private static void WriteInteger(ElementType type, Span<byte> destination, int i, Int128 v)
    {
        switch (type.BlockBits)
        {
            case 8:
                destination[i] = (byte)v;
                break;
            case 16:
                System.Buffers.Binary.BinaryPrimitives.WriteUInt16LittleEndian(destination[(2 * i)..], (ushort)v);
                break;
            case 32:
                System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(destination[(4 * i)..], (uint)v);
                break;
            case 64:
                System.Buffers.Binary.BinaryPrimitives.WriteUInt64LittleEndian(destination[(8 * i)..], (ulong)v);
                break;
            default:
                throw new NotSupportedException($"{type.Name}: an integer of {type.BlockBits} bits is not written as whole bytes.");
        }
    }

    // float32 indices to int32: whole numbers only (every float32 whole number in int32's range is exact).
    private static void FloatsToInt32(ReadOnlySpan<float> source, Span<int> destination)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            for (; i <= source.Length - Vector<float>.Count; i += Vector<float>.Count)
            {
                var v = new Vector<float>(source[i..]);
                var whole = Vector.ConvertToSingle(Vector.ConvertToInt32(v));
                if (!Vector.EqualsAll(whole, v))
                {
                    break;                                                  // a fraction, NaN or a value out of range: checked one by one
                }

                Vector.ConvertToInt32(v).CopyTo(destination[i..]);
            }
        }

        for (; i < source.Length; i++)
        {
            float v = source[i];
            destination[i] = MathF.Truncate(v) == v && v >= -2147483648f && v < 2147483648f ? (int)v : throw NotWhole(v, i);
        }
    }

    // int32 indices to float32: exact only up to 2²⁴.
    private static void Int32ToFloats(ReadOnlySpan<int> source, Span<float> destination)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            var limit = new Vector<int>(1 << 24);
            for (; i <= source.Length - Vector<int>.Count; i += Vector<int>.Count)
            {
                var v = new Vector<int>(source[i..]);
                if (Vector.GreaterThanAny(Vector.Abs(v), limit) || Vector.LessThanAny(v, -limit))
                {
                    break;                                                  // past 2^24 (or int.MinValue): checked one by one
                }

                Vector.ConvertToSingle(v).CopyTo(destination[i..]);
            }
        }

        for (; i < source.Length; i++)
        {
            int v = source[i];
            destination[i] = v is >= -(1 << 24) and <= 1 << 24 ? v : throw PastFloat(v, i);
        }
    }

    private static ArgumentException NotWhole(float value, int at) =>
        new($"Index {value.ToString(System.Globalization.CultureInfo.InvariantCulture)} (element {at}) is not a whole number an integer index type holds.");

    private static ArgumentException PastFloat(long value, int at) =>
        new($"Index {value} (element {at}) is past 2^24: a kernel that reads indices as float32 cannot read it exactly (that device's integer index kernel is not built yet; run it on a device whose index type is int32, such as the CPU).");
}
