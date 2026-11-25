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
/// <param name="Bits">Bits a field takes (1, 2, 4 or 8; up to 64 in a stream or listed; a digit's number's in a plane of digits).</param>
/// <param name="Group">Values a group holds (a power of two).</param>
/// <param name="Stride">
/// Bytes from a group's first byte to the next group's, where groups do not follow one another (other fields between them,
/// as IQ2_XXS's grid indices lie between its signs and scales); 0: the group's own bytes.
/// </param>
/// <param name="Stream">
/// Whether the fields are a continuous stream of bits from the plane's first byte (field v at bit v · Bits, little-endian),
/// each field read at its bit position, across byte boundaries where it falls so (codes of 3, 5, 6 or 7 bits packed tight);
/// Group and Stride then say nothing.
/// </param>
/// <param name="Places">
/// Each field's place in the block, listed (field v at Places[v]: its byte and its lowest bit there, the field running into
/// the next bytes where its bits pass the byte's; a plane of digits: its byte and its digit), where no rule of Offset, Group,
/// Stride or the stream places them (ggml's IQ1_S high index bits, 3 a field in 16-bit words of 4; IQ2_XS's 9-bit indices,
/// one a 16-bit word); null: by the rule, Offset and the rest saying where.
/// </param>
/// <param name="Radix">
/// 0: each field is Bits bits. 2 or more: each field is a digit of base Radix of its byte read as a fraction, most
/// significant first: digit i of byte b is ⌊Radix · ((b · Radix^i) mod 256) / 256⌋, its byte and i from
/// <paramref name="Places"/> (ggml's TQ1_0: five base-3 digits a byte, b = ⌈n · 256 / 243⌉ of the digits' number n); Bits
/// the bits a digit's number takes (Radix − 1 fits them).
/// </param>
public sealed record CodePlane(int Offset, int Bits, int Group, int Stride = 0, bool Stream = false, IReadOnlyList<FieldPlace>? Places = null, int Radix = 0)
{
    /// <summary>Digits of <see cref="Radix"/> a byte holds as a fraction (the largest n with Radix^n at most 256; 0 for a plane of bits).</summary>
    public int Digits
    {
        get
        {
            int n = 0;
            for (int power = Radix; Radix >= 2 && power <= 256; power *= Radix)
            {
                n++;
            }

            return n;
        }
    }

    /// <summary>
    /// Digit <paramref name="digit"/> (0 the most significant) of <paramref name="value"/> read as a fraction in base
    /// <see cref="Radix"/>: ⌊Radix · ((value · Radix^digit) mod 256) / 256⌋.
    /// </summary>
    public int Digit(byte value, int digit)
    {
        int power = 1;
        for (int i = 0; i < digit; i++)
        {
            power *= Radix;
        }

        return (value * power & 0xFF) * Radix >> 8;
    }

    /// <summary>
    /// The byte of value <paramref name="value"/>'s field in the block and the field's lowest bit in it (a stream's or a
    /// listed field may run into the next bytes; a plane of digits: the digit).
    /// </summary>
    public (int Byte, int Shift) Place(int value)
    {
        if (Places is { } places)
        {
            var place = places[value];
            return (place.Byte, place.Shift);
        }

        if (Stream)
        {
            long bit = 8L * Offset + (long)value * Bits;
            return ((int)(bit >> 3), (int)(bit & 7));
        }

        int per = Group * Bits / 8, w = value % Group;
        return (Offset + value / Group * (Stride > 0 ? Stride : per) + w % per, Bits * (w / per));
    }

    /// <summary>
    /// The plane whose fields are <paramref name="fields"/> (field i one piece, all of one width of 1, 2, 4 or 8 bits), in
    /// groups of a power of two that are a stride apart; null where no such plane places them.
    /// </summary>
    public static CodePlane? Fit(IReadOnlyList<IReadOnlyList<ScalePiece>> fields)
    {
        if (fields.Count == 0 || fields.Any(f => f.Count != 1) || fields[0][0].Bits is not (1 or 2 or 4 or 8) || fields.Any(f => f[0].Bits != fields[0][0].Bits))
        {
            return null;
        }

        int bits = fields[0][0].Bits;
        for (int group = 8 / bits; group <= Math.Max(8 / bits, fields.Count); group *= 2)
        {
            int per = group * bits / 8, stride = fields.Count > group ? fields[group][0].Offset - fields[0][0].Offset : per;
            var plane = new CodePlane(fields[0][0].Offset, bits, group, stride == per ? 0 : stride);
            if (stride >= per && Enumerable.Range(0, fields.Count).All(i => plane.Place(i) == (fields[i][0].Offset, fields[i][0].Shift)))
            {
                return plane;
            }
        }

        return null;
    }
}

/// <summary>
/// Where a listed field of a code plane lies (<see cref="CodePlane.Places"/>): byte <paramref name="Byte"/> of the block,
/// from its bit <paramref name="Shift"/> (a plane of digits: its digit <paramref name="Shift"/>, 0 the most significant).
/// </summary>
/// <param name="Byte">The field's byte in the block.</param>
/// <param name="Shift">The field's lowest bit in the byte (0 to 7), or a digit's place.</param>
public readonly record struct FieldPlace(int Byte, int Shift);

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

/// <summary>What a field of bits is as a number (<see cref="NumberType"/>).</summary>
public enum NumberKind
{
    /// <summary>A whole number: its bits, two's complement when signed.</summary>
    Integer,

    /// <summary>A float: a sign bit when signed (the highest), then the exponent's bits, then the mantissa's.</summary>
    Float,

    /// <summary>A code standing for its entry of a table.</summary>
    Table,
}

/// <summary>Which codes of a float are not finite values (<see cref="NumberType"/>).</summary>
public enum FloatSpecials
{
    /// <summary>IEEE 754's: the largest exponent is an infinity (mantissa 0) or a NaN (any other mantissa).</summary>
    Ieee,

    /// <summary>Only the code with every exponent and mantissa bit set is a NaN; no infinity (fp8 E4M3fn, E8M0).</summary>
    AllOnesNaN,

    /// <summary>None: every code is a finite value (fp4 E2M1).</summary>
    None,
}

/// <summary>
/// How a field of <paramref name="Bits"/> bits reads as a number: a whole number (two's complement when
/// <paramref name="Signed"/>); a float of a sign bit (when <paramref name="Signed"/>, the highest), <paramref name="Exponent"/>
/// exponent bits biased by <paramref name="ExponentBias"/> and <paramref name="Mantissa"/> mantissa bits, the exponent 0 a
/// subnormal (or, without <paramref name="Subnormals"/>, the implicit 1 kept: E8M0's 2^-127), its codes past the finite
/// ones by <paramref name="Specials"/>; or a code standing for its entry of <paramref name="Table"/> (2^Bits entries).
/// A value's code and a block's scales are each one of these (<see cref="TypeDescription"/>).
/// </summary>
/// <param name="Bits">Bits the field takes (1 to 64).</param>
/// <param name="Kind">Integer, float or a table's entry.</param>
/// <param name="Signed">Whether it has a sign (two's complement for an integer, a sign bit for a float).</param>
/// <param name="Exponent">A float's exponent bits.</param>
/// <param name="Mantissa">A float's mantissa bits.</param>
/// <param name="ExponentBias">Subtracted from a float's exponent.</param>
/// <param name="Subnormals">Whether a float's exponent 0 is a subnormal (else 2^(−bias) · (1 + mantissa)).</param>
/// <param name="Specials">A float's codes that are not finite.</param>
/// <param name="Table">Each code's value (2^Bits entries), for a table's codes; a codebook's (<paramref name="Width"/> more than 1) 2^Bits · Width, code c's value i at c · Width + i.</param>
/// <param name="Width">
/// Values one code stands for: 1, a number; more, a codebook (a table's codes only), a code indexing a vector of that many
/// values, one index giving several values (ggml's IQ2_XXS grid: an 8-bit index, 8 magnitudes).
/// </param>
public sealed record NumberType(int Bits, NumberKind Kind, bool Signed, int Exponent = 0, int Mantissa = 0, int ExponentBias = 0, bool Subnormals = true,
    FloatSpecials Specials = FloatSpecials.Ieee, IReadOnlyList<double>? Table = null, int Width = 1)
{
    /// <summary>A whole number of <paramref name="bits"/> bits (two's complement when <paramref name="signed"/>).</summary>
    public static NumberType Integer(int bits, bool signed) => new(bits, NumberKind.Integer, signed);

    /// <summary>A float of a sign bit (when <paramref name="signed"/>), <paramref name="exponent"/> and <paramref name="mantissa"/> bits, the bias 2^(exponent − 1) − 1 unless given.</summary>
    public static NumberType Floating(int exponent, int mantissa, bool signed = true, int? bias = null, bool subnormals = true, FloatSpecials specials = FloatSpecials.Ieee) =>
        new((signed ? 1 : 0) + exponent + mantissa, NumberKind.Float, signed, exponent, mantissa, bias ?? (1 << (exponent - 1)) - 1, subnormals, specials);

    /// <summary>Codes of <paramref name="bits"/> bits, each standing for its entry of <paramref name="table"/>.</summary>
    public static NumberType Entries(int bits, IReadOnlyList<double> table) => new(bits, NumberKind.Table, false, Table: table);

    /// <summary>
    /// A codebook: codes of <paramref name="bits"/> bits, each standing for a vector of <paramref name="width"/> values,
    /// code c's value i <paramref name="table"/>[c · width + i].
    /// </summary>
    public static NumberType Codebook(int bits, int width, IReadOnlyList<double> table) => new(bits, NumberKind.Table, false, Table: table, Width: width);

    /// <summary>Whether the description holds together: the bits add up, a table has an entry a code (Width of them a codebook's).</summary>
    public bool Valid => Bits is >= 1 and <= 64 && Width >= 1 && (Width == 1 || Kind == NumberKind.Table) && Kind switch
    {
        NumberKind.Integer => true,
        NumberKind.Float => Exponent >= 1 && Mantissa >= 0 && (Signed ? 1 : 0) + Exponent + Mantissa == Bits && Exponent <= 15,
        _ => Bits <= 16 && Table is { } t && t.Count == (1 << Bits) * Width,
    };

    /// <summary>Value <paramref name="within"/> (below Width) of the vector the field's <paramref name="bits"/> stand for: a codebook's entry's.</summary>
    public double Value(ulong bits, int within) => Width == 1 ? Value(bits) : Table![(int)(bits & ((1UL << Bits) - 1)) * Width + within];

    /// <summary>The number the field's <paramref name="bits"/> (its lowest Bits bits) stand for (a codebook's: its entry's first value).</summary>
    public double Value(ulong bits)
    {
        bits = Bits == 64 ? bits : bits & ((1UL << Bits) - 1);
        switch (Kind)
        {
            case NumberKind.Integer:
                return Signed && Bits < 64 ? (long)(bits << (64 - Bits)) >> (64 - Bits) : Signed ? (long)bits : (double)bits;
            case NumberKind.Table:
                return Table![(int)bits * Width];
        }

        bool negative = Signed && (bits >> (Bits - 1) & 1) != 0;
        ulong magnitude = Signed ? bits & ((1UL << (Bits - 1)) - 1) : bits;
        long e = (long)(magnitude >> Mantissa), top = (1L << Exponent) - 1;
        ulong m = magnitude & ((1UL << Mantissa) - 1);
        double value;
        if (Specials == FloatSpecials.Ieee && e == top)
        {
            value = m == 0 ? double.PositiveInfinity : double.NaN;
        }
        else if (Specials == FloatSpecials.AllOnesNaN && magnitude == (Signed ? (1UL << (Bits - 1)) - 1 : Bits == 64 ? ulong.MaxValue : (1UL << Bits) - 1))
        {
            value = double.NaN;
        }
        else if (e == 0 && Subnormals)
        {
            value = Math.ScaleB(m, 1 - ExponentBias - Mantissa);
        }
        else
        {
            value = Math.ScaleB((double)((1UL << Mantissa) | m), (int)e - ExponentBias - Mantissa);
        }

        return negative ? -value : value;
    }
}

/// <summary>
/// A block's scale or minimum (<see cref="TypeDescription"/>): a number of <paramref name="Type"/> for each
/// <paramref name="Values"/> values of the block; scale i at bits [8 · <paramref name="Offset"/> + i · Type.Bits, + Type.Bits)
/// of the block (little-endian), or, with <paramref name="Pieces"/>, assembled from Pieces[i] (its low bits first); plus
/// <paramref name="Bias"/>.
/// </summary>
/// <param name="Type">The number each is.</param>
/// <param name="Values">Values one covers (a block's, or a sub-block's).</param>
/// <param name="Offset">The first one's first byte in the block.</param>
/// <param name="Pieces">Each one's bits as pieces, or null: one after another from <paramref name="Offset"/>.</param>
/// <param name="Bias">Added to each one's number.</param>
public sealed record BlockScale(NumberType Type, int Values, int Offset, IReadOnlyList<IReadOnlyList<ScalePiece>>? Pieces = null, double Bias = 0)
{
    /// <summary>The bits of scale <paramref name="index"/> of <paramref name="block"/>.</summary>
    public ulong Bits(ReadOnlySpan<byte> block, int index)
    {
        if (Pieces is { } pieces)
        {
            ulong value = 0;
            int at = 0;
            var own = pieces[index];
            for (int i = 0; i < own.Count; i++)
            {
                var piece = own[i];
                value |= (ulong)(block[piece.Offset] >> piece.Shift & ((1 << piece.Bits) - 1)) << at;
                at += piece.Bits;
            }

            return value;
        }

        return TypeDescription.Field(block, 8L * Offset + (long)index * Type.Bits, Type.Bits);
    }

    /// <summary>The number of scale <paramref name="index"/> of <paramref name="block"/>, its bias added.</summary>
    public double Value(ReadOnlySpan<byte> block, int index) => Type.Value(Bits(block, index)) + Bias;

    /// <summary>
    /// The number scale <paramref name="index"/> of <paramref name="block"/> gives the value <paramref name="within"/> of the
    /// values it covers (a codebook's: its entry's value of that place; a number's: the same for each), its bias added.
    /// </summary>
    public double Value(ReadOnlySpan<byte> block, int index, int within) => Type.Value(Bits(block, index), within % Type.Width) + Bias;
}

/// <summary>
/// How a type's values read from their bits, whatever the type: blocks of <paramref name="Values"/> values in
/// <paramref name="Bytes"/> bytes; value v's code is the fields of <paramref name="Planes"/> at its place in the block (the
/// first plane the lowest bits), read as a number of <paramref name="Code"/>'s kind; value v = (that number +
/// <paramref name="CodeOffset"/>) · the product of the <paramref name="Scales"/> covering v, less the product of the
/// <paramref name="Mins"/> covering v (added where <paramref name="MinsAdded"/>). A per-element type is a block of one value
/// (or of the values a byte holds). A codebook code (<see cref="NumberType.Width"/> more than 1) gives Width values from one
/// index: the planes place each entry's index (entry e = v / Width), and value v takes its entry's value v mod Width; a
/// codebook scale (its Values its Width: a vector of factors, as IQ2_XXS's sign masks) gives each value it covers its own
/// factor. A scale level shared over many blocks (a scale whose Values pass a block's, as a second-level scale of block
/// scales) lives in a group of <paramref name="Group"/> blocks: the group's blocks one after another, then such levels, the
/// group <paramref name="GroupBytes"/> bytes; each level's scale i of a group (its values i · Values of the group on) at its
/// own position, bits [8 · Offset + i · its bits) from the group's first byte (or its pieces', from there). A kernel that
/// reads this alone multiplies any type described so, with no code of its own (the described lanes).
/// </summary>
/// <param name="Values">Values a block holds.</param>
/// <param name="Bytes">Bytes a block takes.</param>
/// <param name="Code">What a value's code is as a number: its bits, integer or float with its bits, or a table.</param>
/// <param name="Planes">Where a code's bits lie in a block, the lowest first (one plane for most types).</param>
/// <param name="CodeOffset">Added to a code's number before the scales.</param>
/// <param name="Scales">Multiplied with a value's number, or null: none.</param>
/// <param name="Mins">Their product taken from (or added to) a value, or null: none.</param>
/// <param name="MinsAdded">Whether the minimums' product is added rather than subtracted.</param>
/// <param name="Group">Blocks a group holds: more than 1 where a scale level is shared over many blocks (stored after them).</param>
/// <param name="GroupBytes">Bytes a group takes, its blocks and its shared levels (0: Group · Bytes).</param>
public sealed record TypeDescription(int Values, int Bytes, NumberType Code, IReadOnlyList<CodePlane> Planes, double CodeOffset = 0,
    IReadOnlyList<BlockScale>? Scales = null, IReadOnlyList<BlockScale>? Mins = null, bool MinsAdded = false, int Group = 1, int GroupBytes = 0)
{
    /// <summary>Values a group of blocks holds (Group · Values): the unit a row is whole of.</summary>
    public int UnitValues => Group * Values;

    /// <summary>Bytes a group takes (GroupBytes, or Group · Bytes).</summary>
    public int UnitBytes => GroupBytes > 0 ? GroupBytes : Group * Bytes;

    /// <summary>Whether <paramref name="scale"/> is a level shared over many blocks (its values pass a block's: stored at the group's own position).</summary>
    public bool Shared(BlockScale scale) => scale.Values > Values;

    /// <summary>Value <paramref name="v"/>'s block's first byte in its group (v within the group).</summary>
    public int BlockStart(int v) => v % UnitValues / Values * Bytes;

    /// <summary>The byte <paramref name="scale"/>'s Offset counts from for value <paramref name="v"/> of a group: its block's first, or the group's (a shared level).</summary>
    public int ScaleBase(BlockScale scale, int v) => Shared(scale) ? 0 : BlockStart(v);

    /// <summary>The index of <paramref name="scale"/>'s number covering value <paramref name="v"/> of a group (in its block, or in the group for a shared level).</summary>
    public int ScaleIndex(BlockScale scale, int v) => (Shared(scale) ? v % UnitValues : v % Values) / scale.Values;

    /// <summary>
    /// Whether the description holds together: the planes' bits make the code's, each plane's fields (an entry's, a
    /// codebook's) within a block, the scales cover whole values (a codebook scale exactly its entry's values).
    /// </summary>
    public bool Valid => Values >= 1 && Bytes >= 1 && Group >= 1 && UnitBytes >= Group * Bytes && Code.Valid && Values % Code.Width == 0 && Planes.Count >= 1
        && Planes.Sum(p => p.Bits) == Code.Bits
        && Planes.All(PlaneHolds)
        && (Scales ?? []).Concat(Mins ?? []).All(s => s.Type.Valid && s.Values >= 1 && (s.Type.Width == 1 || s.Values == s.Type.Width) && (Shared(s)
            ? s.Values % Values == 0 && UnitValues % s.Values == 0 && s.Type.Width == 1
                && (s.Pieces is { } shared ? shared.Count == UnitValues / s.Values : 8L * s.Offset + (long)UnitValues / s.Values * s.Type.Bits <= 8L * UnitBytes)
            : Values % s.Values == 0 && (s.Pieces is { } pieces ? pieces.Count == Values / s.Values : 8L * s.Offset + (long)Values / s.Values * s.Type.Bits <= 8L * Bytes)));

    // Whether plane p's fields hold together: fields of 1, 2, 4 or 8 bits in groups (or whole bytes a value), a stream's or
    // listed fields of up to 64 bits, or listed digits of a radix (each a digit its byte holds, its number within the bits);
    // a listed plane a field an entry; each field within a block.
    private bool PlaneHolds(CodePlane p)
    {
        int fields = Values / Code.Width;
        if (p.Bits < 1 || p.Group < 1 || p.Places is { } listed && listed.Count != fields)
        {
            return false;
        }

        if (p.Radix != 0)
        {
            return p.Radix >= 2 && p.Places is { } digits && !p.Stream && p.Bits <= 8 && p.Radix - 1 < 1 << p.Bits
                && digits.All(f => f.Byte >= 0 && f.Byte < Bytes && f.Shift >= 0 && f.Shift < p.Digits);
        }

        bool shape = p.Places is { } places ? p.Bits <= 64 && places.All(f => f.Byte >= 0 && f.Shift is >= 0 and <= 7)
            : p.Stream ? p.Bits <= 64
            : (p.Bits is 1 or 2 or 4 or 8 || p.Bits % 8 == 0 && p.Group == 1) && p.Group * p.Bits % 8 == 0 && (p.Stride == 0 || p.Stride >= p.Group * p.Bits / 8);
        return shape && Enumerable.Range(0, fields).All(v => p.Place(v).Byte + (p.Place(v).Shift + p.Bits + 7) / 8 <= Bytes);
    }

    /// <summary>Whether a value's code or factors come from a codebook (a code or a scale of more than one value an entry).</summary>
    public bool HasCodebook => Code.Width > 1 || (Scales ?? []).Concat(Mins ?? []).Any(s => s.Type.Width > 1);

    /// <summary>Bits [<paramref name="bit"/>, + <paramref name="bits"/>) of <paramref name="bytes"/>, little-endian (at most 64).</summary>
    public static ulong Field(ReadOnlySpan<byte> bytes, long bit, int bits)
    {
        ulong value = 0;
        for (int i = 0; i < bits; i++)
        {
            long at = bit + i;
            value |= (ulong)(bytes[(int)(at >> 3)] >> (int)(at & 7) & 1) << i;
        }

        return value;
    }

    /// <summary>
    /// Value <paramref name="v"/>'s code bits in <paramref name="block"/> (a group's bytes where Group passes 1, v within the
    /// group), its planes' fields joined (the first the lowest bits; a plane of digits gives its digit): a codebook's, its
    /// entry's index.
    /// </summary>
    public ulong CodeBits(ReadOnlySpan<byte> block, int v)
    {
        block = block[BlockStart(v)..];
        v %= Values;
        ulong code = 0;
        int low = 0, entry = v / Code.Width;
        for (int p = 0; p < Planes.Count; p++)
        {
            var plane = Planes[p];
            var (at, shift) = plane.Place(entry);
            code |= (plane.Radix != 0 ? (ulong)plane.Digit(block[at], shift) : Field(block, 8L * at + shift, plane.Bits)) << low;
            low += plane.Bits;
        }

        return code;
    }

    /// <summary>
    /// Value <paramref name="v"/> of <paramref name="block"/> (a group's bytes where Group passes 1, v within the group), by
    /// the description alone (in float64; the reference a kernel's values are checked against).
    /// </summary>
    public double ValueOf(ReadOnlySpan<byte> block, int v)
    {
        v %= UnitValues;
        double value = Code.Value(CodeBits(block, v), v % Code.Width) + CodeOffset;
        foreach (var scale in Scales ?? [])
        {
            value *= scale.Value(block[ScaleBase(scale, v)..], ScaleIndex(scale, v), v % scale.Values);
        }

        if (Mins is { Count: > 0 } mins)
        {
            double min = 1;
            foreach (var m in mins)
            {
                min *= m.Value(block[ScaleBase(m, v)..], ScaleIndex(m, v), v % m.Values);
            }

            value = MinsAdded ? value + min : value - min;
        }

        return value;
    }

    /// <summary>The values of whole groups of blocks of <paramref name="source"/> into <paramref name="destination"/>, by the description alone.</summary>
    public void Decode(ReadOnlySpan<byte> source, Span<float> destination)
    {
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = (float)ValueOf(source.Slice(i / UnitValues * UnitBytes, UnitBytes), i % UnitValues);
        }
    }

    /// <summary>
    /// The description older entries imply (<see cref="ElementType.Float"/>, the integer bits, <see cref="ElementType.Scaled"/>,
    /// <see cref="ElementType.Super"/>, <see cref="ElementType.Grid"/>: a codebook), or null where they say nothing a value reads from (a truth value).
    /// </summary>
    internal static TypeDescription? Of(ElementType type)
    {
        if (type.Grid is { } grid)
        {
            return OfGrid(grid);
        }

        if (type.Super is { } s)
        {
            int subBits = s.SubScales[0].Sum(p => p.Bits);
            var half = NumberType.Floating(s.Scale.Exponent, s.Scale.Mantissa);
            var scales = new List<BlockScale> { new(half, s.Values, s.ScaleOffset), new(NumberType.Integer(subBits, s.SubScalesSigned), s.SubValues, 0, s.SubScales, s.SubScaleBias) };
            var mins = s.MinOffset >= 0 && s.SubMins is { } subMins
                ? new List<BlockScale> { new(half, s.Values, s.MinOffset), new(NumberType.Integer(subMins[0].Sum(p => p.Bits), false), s.SubValues, 0, subMins) }
                : null;
            return Checked(new TypeDescription(s.Values, s.Bytes, NumberType.Integer(s.Planes.Sum(p => p.Bits), false), s.Planes, s.CodeBias, scales, mins));
        }

        if (type.Scaled is { } b)
        {
            var code = b.CodeType is { } codeType ? codeType.Description?.Code
                : b.CodeTable is { } table ? NumberType.Entries(b.CodeBits, [.. table.Select(e => (double)e)])
                : NumberType.Integer(b.CodeBits, b.SignedCodes);
            var scale = b.ScaleType is { } scaleType ? scaleType.Description?.Code : NumberType.Floating(b.Scale.Exponent, b.Scale.Mantissa);
            if (code is null || scale is null || b.CodeBits is not (1 or 2 or 4 or 8) || b.NibblesSplit && b.CodeBits != 4)
            {
                return null;
            }

            var plane = b.NibblesSplit ? new CodePlane(b.CodeOffset, 4, b.Values) : new CodePlane(b.CodeOffset, b.CodeBits, 8 / b.CodeBits);
            return Checked(new TypeDescription(b.Values, b.Bytes, code with { Bits = b.CodeBits }, [plane], b.CodeBias, [new BlockScale(scale, b.Values, b.ScaleOffset)]));
        }

        if (type.BlockSize != 1 || type.Scales is not null || type.Grid is not null)
        {
            return null;
        }

        var number = type.Float is { } f && f.Bits == type.BlockBits ? NumberType.Floating(f.Exponent, f.Mantissa)
            : type.IsInteger ? NumberType.Integer(type.BlockBits, type.IsSigned) : null;
        if (number is null || type.BlockBits is not (1 or 2 or 4) && type.BlockBits % 8 != 0)
        {
            return null;
        }

        return type.BlockBits >= 8
            ? Checked(new TypeDescription(1, type.BlockBits / 8, number, [new CodePlane(0, type.BlockBits, 1)]))
            : Checked(new TypeDescription(8 / type.BlockBits, 1, number, [new CodePlane(0, type.BlockBits, 8 / type.BlockBits)]));

        static TypeDescription? Checked(TypeDescription d) => d.Valid ? d : null;
    }

    // A grid codebook's description (GridBlock: ggml's IQ2_XXS, IQ3_XXS): the code a codebook of the grid's magnitudes
    // (times the sub-scales' factor, a power of two), each entry's index one plane; the scales the super-scale, each
    // sub-block's scale plus its offset, and each sign group's vector of ±1 (a codebook of the sign masks). Null where the
    // entries' indices do not lie as one plane or the tables do not cover every index.
    private static TypeDescription? OfGrid(GridBlock g)
    {
        if (CodePlane.Fit(g.Entries) is not { } plane || g.Grid.Count != (1 << plane.Bits) * g.EntryValues || g.Signs.Count == 0 || g.SubScales.Count == 0)
        {
            return null;
        }

        int signBits = g.Signs[0].Sum(p => p.Bits), subBits = g.SubScales[0].Sum(p => p.Bits);
        if (g.SignTable.Count != 1 << signBits)
        {
            return null;
        }

        var signs = new double[g.SignTable.Count * g.SignValues];
        for (int s = 0; s < g.SignTable.Count; s++)
        {
            for (int i = 0; i < g.SignValues; i++)
            {
                signs[s * g.SignValues + i] = (g.SignTable[s] >> i & 1) != 0 ? -1 : 1;
            }
        }

        var code = NumberType.Codebook(plane.Bits, g.EntryValues, [.. g.Grid.Select(v => v * (double)g.SubScaleFactor)]);
        var description = new TypeDescription(g.Values, g.Bytes, code, [plane], 0,
        [
            new BlockScale(NumberType.Floating(g.Scale.Exponent, g.Scale.Mantissa), g.Values, g.ScaleOffset),
            new BlockScale(NumberType.Integer(subBits, false), g.SubValues, 0, g.SubScales, g.SubScaleOffset),
            new BlockScale(NumberType.Codebook(signBits, g.SignValues, signs), g.SignValues, 0, g.Signs),
        ]);
        return description.Valid ? description : null;
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
        Description = new TypeDescription(1, 1, NumberType.Floating(4, 3, specials: FloatSpecials.AllOnesNaN), [new CodePlane(0, 8, 1)]),
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
        Description = new TypeDescription(1, 8, NumberType.Integer(64, signed: true), [new CodePlane(0, 64, 1)]),
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
        Description = new TypeDescription(32, 10, NumberType.Integer(2, signed: true), [new CodePlane(2, 2, 4)],
            Scales: [new BlockScale(NumberType.Floating(5, 10), 32, 0)]),
        ToFloat32 = ElementConversions.Int2ToFloat32,
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
        Description = new TypeDescription(1, 8, NumberType.Floating(11, 52), [new CodePlane(0, 64, 1)]),
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
        Description = new TypeDescription(2, 1, NumberType.Floating(2, 1, bias: 1, specials: FloatSpecials.None), [new CodePlane(0, 4, 2)]),
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
        Description = new TypeDescription(1, 1, NumberType.Floating(8, 0, signed: false, bias: 127, subnormals: false, specials: FloatSpecials.AllOnesNaN), [new CodePlane(0, 8, 1)]),
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
        Description = new TypeDescription(32, 17, Float4E2M1.Description!.Code, [new CodePlane(1, 4, 32)], Scales: [new BlockScale(Float8E8M0.Description!.Code, 32, 0)]),
        ToFloat32 = ElementConversions.MxFp4ToFloat32,
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
    /// How the type's values read from their bits (<see cref="TypeDescription"/>): what the CPU's described lane reads to
    /// multiply the type where it is stored, with no code of the type's own. A type that gives none has the one its older
    /// entries imply (<see cref="Float"/>, the integer bits, <see cref="Scaled"/>, <see cref="Super"/>, <see cref="Grid"/>: a
    /// codebook), or null where they say nothing a value reads from (a truth value).
    /// </summary>
    public TypeDescription? Description
    {
        get
        {
            if (!_described)
            {
                field ??= TypeDescription.Of(this);
                _described = true;
            }

            return field;
        }

        init
        {
            field = value is null || value.Valid ? value : throw new ArgumentException($"The description of {Name} does not hold together: {value}.", nameof(Description));
            _described = value is not null;                                 // none given: the one the older entries imply
        }
    }

    private bool _described;

    /// <summary>
    /// The operations that read or write the type natively, by name (<c>Operation.Name</c>; "*" for every one): any other
    /// reads it through a counted conversion.
    /// </summary>
    public IReadOnlyList<string> Kernels { get; init; } = [];

    /// <summary>Its widening to float32 on the CPU, or null when it has none.</summary>
    public ElementDecoder? ToFloat32 { get; init; }

    /// <summary>
    /// Its storing from float32 on the CPU: its own conversion, else the writer of its description (any described type is
    /// written by its <see cref="Description"/> alone: the codes, scales and minimums closest to the values, searched by
    /// measured error), else null (a type that describes nothing it could be written by).
    /// </summary>
    public ElementEncoder? FromFloat32
    {
        get
        {
            if (!_encoderKnown)
            {
                if (field is null && DescribedEncoder.Of(Description) is { } written)
                {
                    field = written.Encode;
                }

                _encoderKnown = true;
            }

            return field;
        }

        init
        {
            field = value;
            _encoderKnown = value is not null;
        }
    }

    private bool _encoderKnown;

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

/// <summary>
/// The writer of any described type (<see cref="TypeDescription"/>): float32 values stored by the type's description alone,
/// with no code of the type's own (<see cref="ElementType.FromFloat32"/> of a type that gives none). Per group of blocks it
/// searches, by measured squared error, for every number the description stores:
/// 1. each unit (the values the finest scale or minimum covers) its own continuous scale, and minimum where there are
///    minimums, with its codes: candidate scales putting the unit's extremes at the codes' extremes and at half steps of the
///    codes' numbers inside and past them, each candidate's codes the nearest and their least-squares scale (and minimum),
///    the best refined by quantizing again with it;
/// 2. the levels of scales from the units' scales: the finest level's numbers and their parent's continuous value by the
///    same search over the finest level's numbers (weighted by how much a unit's error moves with its scale), level by level
///    up to the coarsest, which is rounded to its number type; then each finer level's number the nearest its target over
///    the levels stored above it; the minimums' own levels alike from the units' minimums (a minimum stored in a scale's own
///    bits taking that scale's number);
/// 3. each value's code given the stored levels: its number type's nearest number, or a codebook's nearest entry (a vector
///    factor's entry, as a sign mask, chosen first from the targets' signs, then again given the entries);
/// 4. each level's numbers searched again, the finest first: each chunk's least-squares number given the codes, rounded, and
///    its neighbours, the codes chosen again for each and the error measured, the best kept.
/// Each number type's numbers are read from the type (a table, an integer, a float of any bits up to 16, a bias added), each
/// field written where the description places it (planes, pieces, levels shared over a group of blocks). The same bytes on
/// every CPU: scalar arithmetic in a fixed order.
/// </summary>
internal sealed class DescribedEncoder
{
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<TypeDescription, object> Made = new();

    // A unit's candidate scales put its value of largest magnitude at the codes' extreme moved by these half steps of the
    // extreme's step (outward positive; past the extreme clips the largest values, which the error may favour); with a
    // minimum, also its range over the codes' range moved by these half steps. A codebook's entries cost a nearest-entry
    // search each, and its search tries fewer (BookSteps, BookNeighbours, BookPasses; measured on a real model's weights:
    // the same error within 0.2% at twice the speed).
    private static readonly float[] ScaleSteps = [-2f, -1f, 0f, 1f, 2f, 4f, 6f, 8f];
    private static readonly float[] BookSteps = [-1f, 0f, 1f, 2f];
    private static readonly float[] RangeSteps = [-2f, -1f, 0f, 1f, 2f, 4f];

    // Times a search quantizes again with its best scale; the levels' numbers searched again by neighbours this far either
    // side (a level of at most SmallGrid numbers), in this many passes (a codebook's fewer, as BookSteps).
    private const int Refinements = 4, SmallGrid = 256, Neighbours = 4, Passes = 2, BookNeighbours = 2, BookPasses = 1;

    private readonly int n, size, unit, units, slots, codeWidth, neighbours, passes;
    private readonly Grid? codeGrid;
    private readonly Book? codeBook;
    private readonly FactorLevel? factor;
    private readonly Level[] scales, mins, free, all;
    private readonly bool minsAdded, negativeScale;
    private readonly int minSign;
    private readonly Fields codes;
    private readonly double top, bottom, topStep, bottomStep;

    private DescribedEncoder(TypeDescription d)
    {
        n = d.UnitValues;
        size = d.UnitBytes;
        codeWidth = d.Code.Width;
        slots = n / codeWidth;
        if (codeWidth == 1)
        {
            codeGrid = Grid.Of(d.Code, d.CodeOffset) ?? throw Unwritten(d, "its codes' numbers are not a table, an integer or a float of up to 16 bits");
        }
        else
        {
            codeBook = Book.Of(d.Code, d.CodeOffset) ?? throw Unwritten(d, "its codebook is not a table of up to 16 bits");
        }

        var scaleLevels = new List<Level>();
        foreach (var s in d.Scales ?? [])
        {
            if (s.Type.Width > 1)
            {
                if (factor is not null || s.Values != s.Type.Width)
                {
                    throw Unwritten(d, "it has more than one level of vector factors");
                }

                factor = new FactorLevel(Book.Of(s.Type, s.Bias) ?? throw Unwritten(d, "a vector factor is not a table of up to 16 bits"), s.Values, FieldsOf(d, s, n / s.Values));
                continue;
            }

            scaleLevels.Add(LevelOf(d, s, scale: true));
        }

        var minLevels = new List<Level>();
        var freeLevels = new List<Level>();
        foreach (var m in d.Mins ?? [])
        {
            if (m.Type.Width > 1)
            {
                throw Unwritten(d, "a minimum is a vector of factors");
            }

            if (scaleLevels.Find(s => SameField(s.Source, m)) is { } tied)
            {
                tied.Min = true;                                            // stored in a scale's own bits: that scale's number
                minLevels.Add(tied);
                continue;
            }

            var level = LevelOf(d, m, scale: false);
            minLevels.Add(level);
            freeLevels.Add(level);
        }

        scales = [.. scaleLevels.OrderByDescending(l => l.Values)];
        mins = [.. minLevels.OrderByDescending(l => l.Values)];
        free = [.. freeLevels.OrderByDescending(l => l.Values)];
        all = [.. scales.Concat(free).OrderBy(l => l.Values)];
        for (int i = 0; i < all.Length; i++)
        {
            all[i].Index = i;
        }

        if (!Nested(scales) || !Nested(free))
        {
            throw Unwritten(d, "its levels do not nest (each level's values a whole number of the next finer level's)");
        }

        unit = Math.Min(scales.Length > 0 ? scales[^1].Values : n, mins.Length > 0 ? mins[^1].Values : n);
        if (unit % codeWidth != 0 || factor is { } f && unit % f.Values != 0 || n % unit != 0 || all.Any(l => l.Values % unit != 0))
        {
            throw Unwritten(d, "its codes, factors and levels do not share whole units");
        }

        units = n / unit;
        neighbours = codeBook is null ? Neighbours : BookNeighbours;
        passes = codeBook is null ? Passes : BookPasses;
        minsAdded = d.MinsAdded;
        negativeScale = scales.Length > 0 && scales[^1].Grid.HasNegative;
        if (mins.Length == 0 || mins.Any(l => l.Values == mins[^1].Values && l.Grid.HasNegative && l.Grid.HasPositive))
        {
            minSign = 0;                                                    // a unit's minimum of either sign
        }
        else if (mins.All(l => !(l.Grid.HasNegative && l.Grid.HasPositive)))
        {
            int sign = mins.Aggregate(1, (s, l) => l.Grid.HasNegative ? -s : s);
            minSign = minsAdded ? sign : -sign;                             // the one sign the levels allow
        }
        else
        {
            minSign = 2;                                                    // a sign shared by the group's units
        }

        // Each code slot's planes (the lowest bits first) at its place in its block.
        var start = new int[slots + 1];
        var bits = new List<int>();
        var widths = new List<int>();
        for (int s = 0; s < slots; s++)
        {
            int v = s * codeWidth, block = d.BlockStart(v), entry = v % d.Values / codeWidth;
            foreach (var plane in d.Planes)
            {
                var (at, shift) = plane.Place(entry);
                bits.Add(8 * (block + at) + shift);
                widths.Add(plane.Bits);
            }

            start[s + 1] = bits.Count;
        }

        codes = new Fields(start, [.. bits], [.. widths]);

        // No two fields share a bit (a minimum stored in a scale's bits is that scale's level).
        var owned = new bool[8 * size];
        if (!codes.Claim(owned) || all.Any(l => !l.Fields.Claim(owned)) || factor is { } owner && !owner.Fields.Claim(owned))
        {
            throw Unwritten(d, "two of its fields share bits");
        }

        // The codes' numbers (times a vector factor's): their extremes and the steps beside them.
        double[] numbers = codeGrid?.Values ?? codeBook!.Levels;
        if (factor is { } vector)
        {
            numbers = [.. vector.Book.Levels.SelectMany(p => numbers.Select(q => p * q)).Distinct().Order()];
        }

        top = numbers[^1];
        bottom = numbers[0];
        topStep = numbers.Length > 1 ? numbers[^1] - numbers[^2] : 1;
        bottomStep = numbers.Length > 1 ? numbers[1] - numbers[0] : 1;
    }

    /// <summary>The writer of <paramref name="d"/>, or null where its description is one it does not write (said by <see cref="Why"/>).</summary>
    public static DescribedEncoder? Of(TypeDescription? d) => d is null ? null : Made.GetValue(d, static x => Make(x)) as DescribedEncoder;

    /// <summary>Why <paramref name="d"/> has no writer, or null where it has one.</summary>
    public static string? Why(TypeDescription d) => Made.GetValue(d, static x => Make(x)) as string;

    private static object Make(TypeDescription d)
    {
        if (!d.Valid)
        {
            return "its description does not hold together";
        }

        try
        {
            return new DescribedEncoder(d);
        }
        catch (NotSupportedException e)
        {
            return e.Message;
        }
    }

    private static NotSupportedException Unwritten(TypeDescription d, string why) => new($"A type of {d.Values}-value blocks in {d.Bytes} bytes is not written from its description: {why}.");

    // Whether a minimum is stored in a scale's own bits as the same number.
    private static bool SameField(BlockScale a, BlockScale b) => a.Values == b.Values && a.Offset == b.Offset && a.Bias == b.Bias && SameType(a.Type, b.Type)
        && (a.Pieces is null ? b.Pieces is null : b.Pieces is not null && a.Pieces.Count == b.Pieces.Count && a.Pieces.Zip(b.Pieces).All(p => p.First.SequenceEqual(p.Second)));

    private static bool SameType(NumberType a, NumberType b) => a with { Table = null } == b with { Table = null }
        && (a.Table is null ? b.Table is null : b.Table is not null && a.Table.SequenceEqual(b.Table));

    private static bool Nested(Level[] levels)
    {
        for (int i = 1; i < levels.Length; i++)
        {
            if (levels[i - 1].Values % levels[i].Values != 0)
            {
                return false;
            }
        }

        return true;
    }

    private Level LevelOf(TypeDescription d, BlockScale s, bool scale)
    {
        var grid = Grid.Of(s.Type, s.Bias) ?? throw Unwritten(d, "a scale or minimum is not a table, an integer or a float of up to 16 bits");
        return new Level(s, grid, s.Values, n / s.Values, FieldsOf(d, s, n / s.Values)) { Scale = scale, Min = !scale };
    }

    // Each chunk's bits of a scale or minimum in a group (its pieces', or its run of bits from its offset).
    private static Fields FieldsOf(TypeDescription d, BlockScale s, int chunks)
    {
        var start = new int[chunks + 1];
        var bits = new List<int>();
        var widths = new List<int>();
        for (int j = 0; j < chunks; j++)
        {
            int v = j * s.Values, at = d.ScaleBase(s, v), index = d.ScaleIndex(s, v);
            if (s.Pieces is { } pieces)
            {
                foreach (var piece in pieces[index])
                {
                    bits.Add(8 * (at + piece.Offset) + piece.Shift);
                    widths.Add(piece.Bits);
                }
            }
            else
            {
                bits.Add(8 * (at + s.Offset) + index * s.Type.Bits);
                widths.Add(s.Type.Bits);
            }

            start[j + 1] = bits.Count;
        }

        return new Fields(start, [.. bits], [.. widths]);
    }

    /// <summary>Stores whole groups of <paramref name="source"/> into <paramref name="destination"/>.</summary>
    public void Encode(ReadOnlySpan<float> source, Span<byte> destination) => Encode(source, destination, []);

    /// <summary>
    /// <see cref="Encode(ReadOnlySpan{float}, Span{byte})"/>, and where <paramref name="chosen"/> is not empty, the value it
    /// chose for each (its stored numbers' scale times its code's number plus its minimum, in float32).
    /// </summary>
    internal void Encode(ReadOnlySpan<float> source, Span<byte> destination, Span<float> chosen)
    {
        if (source.Length % n != 0)
        {
            throw new ArgumentException($"The type stores whole groups of {n} values; {source.Length} is not.", nameof(source));
        }

        var w = Work.For(this);
        for (int g = 0; g < source.Length / n; g++)
        {
            EncodeGroup(source.Slice(g * n, n), destination.Slice(g * size, size), w);
            if (!chosen.IsEmpty)
            {
                for (int u = 0; u < units; u++)
                {
                    UnitFactors(w, u, out float a, out float c);
                    for (int v = u * unit; v < (u + 1) * unit; v++)
                    {
                        chosen[g * n + v] = a * w.G[v] + c;
                    }
                }
            }
        }
    }

    private void EncodeGroup(ReadOnlySpan<float> source, Span<byte> group, Work w)
    {
        for (int v = 0; v < n; v++)
        {
            w.X[v] = float.IsFinite(source[v]) ? source[v] : 0f;
        }

        // 1. Each unit's own scale and minimum (a sign shared by the group's units: the units' majority, the rest again).
        for (int u = 0; u < units; u++)
        {
            SolveUnit(w, u, minSign == 2 ? 0 : minSign);
        }

        if (minSign == 2)
        {
            double sum = 0;
            for (int u = 0; u < units; u++)
            {
                sum += w.C[u];
            }

            int sign = sum >= 0 ? 1 : -1;
            for (int u = 0; u < units; u++)
            {
                if (w.C[u] * sign < 0)
                {
                    SolveUnit(w, u, sign);
                }
            }
        }

        // 2. The levels from the units' scales, then the minimums' own levels over what the scales' levels give them.
        if (scales.Length > 0)
        {
            Factor(w, scales, w.A, w.WA);
        }

        if (free.Length > 0)
        {
            for (int u = 0; u < units; u++)
            {
                double tied = 1;
                foreach (var level in mins)
                {
                    if (level.Scale)
                    {
                        tied *= level.Grid.Values[w.Ranks[level.Index][u * unit / level.Values]];
                    }
                }

                double target = minsAdded ? w.C[u] : -w.C[u];
                w.T[u] = tied != 0 ? (float)(target / tied) : 0f;
                w.Wt[u] = (float)(w.WC[u] * tied * tied);
            }

            Factor(w, free, w.T, w.Wt);
        }

        // 3. The codes given the stored levels.
        for (int u = 0; u < units; u++)
        {
            UnitFactors(w, u, out float a, out float c);
            w.Error[u] = QuantizeUnit(w, u, a, c, w.G, w.Codes, w.Factors);
        }

        // 4. Each level's numbers searched again by measured error, the finest first.
        for (int pass = 0; pass < passes; pass++)
        {
            foreach (var level in all)
            {
                for (int j = 0; j < level.Chunks; j++)
                {
                    Search(w, level, j);
                }
            }
        }

        Write(w, group);
    }

    // Unit u's own scale a and minimum c (the value a · g + c) and codes: Solve over its values.
    private void SolveUnit(Work w, int u, int sign)
    {
        var y = w.X.AsSpan(u * unit, unit);
        var g = w.TryG.AsSpan(0, unit);
        var best = w.BestG.AsSpan(0, unit);
        if (scales.Length == 0)
        {
            // No scale: the codes' numbers are the values, less a minimum where there is one.
            float c = 0f;
            for (int r = 0; r < Refinements; r++)
            {
                w.Quantizer.Run(y, 1f, c, g);
                if (mins.Length == 0)
                {
                    break;
                }

                double s = 0;
                for (int i = 0; i < unit; i++)
                {
                    s += y[i] - g[i];
                }

                c = (float)(s / unit);
                if (sign != 0 && c * sign < 0)
                {
                    c = 0f;
                }
            }

            w.A[u] = 1f;
            w.C[u] = c;
            w.WA[u] = unit;
            w.WC[u] = unit;
            return;
        }

        Solve(w.Quantizer, y, w.Ones.AsSpan(0, unit), negativeScale, mins.Length > 0, sign, g, best, out float a, out float m);
        double sgg = 0;
        for (int i = 0; i < unit; i++)
        {
            sgg += (double)best[i] * best[i];
        }

        w.A[u] = a;
        w.C[u] = m;
        w.WA[u] = (float)sgg;
        w.WC[u] = unit;
    }

    // The numbers of `levels` (coarse to fine, each nested in the one before) whose products come closest to the units'
    // targets t (weights wt): the finest level's chunks' targets the weighted means of their units'; bottom up, each parent's
    // continuous number and its children's numbers by Solve over the children's targets and the child level's numbers; the
    // coarsest rounded to its own; top down, each finer level's number nearest its target over the stored levels above it.
    private void Factor(Work w, Level[] levels, ReadOnlySpan<float> t, ReadOnlySpan<float> wt)
    {
        int last = levels.Length - 1;
        var leaf = levels[last];
        int per = leaf.Values / unit;
        var leafTargets = w.LevelTargets[leaf.Index];
        var leafWeights = w.LevelWeights[leaf.Index];
        for (int j = 0; j < leaf.Chunks; j++)
        {
            double sw = 0, swt = 0, st = 0;
            for (int u = j * per; u < (j + 1) * per; u++)
            {
                sw += wt[u];
                swt += (double)wt[u] * t[u];
                st += t[u];
            }

            leafTargets[j] = (float)(sw > 0 ? swt / sw : st / per);
            leafWeights[j] = (float)sw;
        }

        for (int i = last; i >= 1; i--)
        {
            var child = levels[i];
            var parent = levels[i - 1];
            int k = parent.Values / child.Values;
            bool negative = false;
            for (int l = 0; l < i; l++)
            {
                negative |= levels[l].Grid.HasNegative;
            }

            var childTargets = w.LevelTargets[child.Index];
            var childWeights = w.LevelWeights[child.Index];
            var ranks = w.Ranks[child.Index];
            for (int p = 0; p < parent.Chunks; p++)
            {
                var ys = childTargets.AsSpan(p * k, k);
                var ws = childWeights.AsSpan(p * k, k);
                var best = w.BestG.AsSpan(0, k);
                Solve(child.Quantizer, ys, ws, negative, false, 0, w.TryG.AsSpan(0, k), best, out float lambda, out _);
                double weight = 0;
                for (int c = 0; c < k; c++)
                {
                    weight += (double)ws[c] * best[c] * best[c];
                    ranks[p * k + c] = child.Grid.Nearest(best[c]);
                }

                w.LevelTargets[parent.Index][p] = lambda;
                w.LevelWeights[parent.Index][p] = (float)weight;
            }
        }

        var coarsest = levels[0];
        for (int p = 0; p < coarsest.Chunks; p++)
        {
            w.Ranks[coarsest.Index][p] = coarsest.Grid.Nearest(w.LevelTargets[coarsest.Index][p]);
        }

        for (int i = 1; i <= last; i++)
        {
            var level = levels[i];
            var targets = w.LevelTargets[level.Index];
            var ranks = w.Ranks[level.Index];
            for (int j = 0; j < level.Chunks; j++)
            {
                double above = 1;
                for (int l = 0; l < i; l++)
                {
                    above *= levels[l].Grid.Values[w.Ranks[levels[l].Index][j * level.Values / levels[l].Values]];
                }

                if (above != 0)
                {
                    ranks[j] = level.Grid.Nearest(targets[j] / above);
                }
            }
        }
    }

    // The scale a (and, `affine`, the added c) and codes making a · g + c closest to y under weights w, g the numbers q gives
    // for its targets: the targets of candidate scales putting y's largest at the codes' largest number and y's smallest at
    // their smallest (and, `negative`, the other way round), moved by half steps (ScaleSteps); with `affine`, y's range over
    // the codes' range so moved (RangeSteps); each candidate's codes the nearest, their least-squares a and c (Fit), the best
    // refined by quantizing again with it. Returns the weighted squared error; `best` holds the best codes' numbers.
    private static double Solve(Quantizer q, ReadOnlySpan<float> y, ReadOnlySpan<float> w, bool negative, bool affine, int sign,
        Span<float> g, Span<float> best, out float a, out float c)
    {
        float ymax = float.MinValue, ymin = float.MaxValue;
        double sxx = 0;
        for (int i = 0; i < y.Length; i++)
        {
            ymax = MathF.Max(ymax, y[i]);
            ymin = MathF.Min(ymin, y[i]);
            sxx += (double)w[i] * y[i] * y[i];
        }

        var s = new Candidates { Q = q, Y = y, W = w, G = g, Best = best, Negative = negative, Affine = affine, Sign = sign, Sxx = sxx, Error = double.PositiveInfinity };
        s.Try(0f, 0f);                                                       // every target 0
        foreach (float k in q.Steps)
        {
            if (ymax > 0 && q.Top > 0)
            {
                s.Try((float)((q.Top + k * q.TopStep / 2) / ymax), 0f);
            }

            if (ymin < 0 && q.Bottom < 0)
            {
                s.Try((float)((q.Bottom - k * q.BottomStep / 2) / ymin), 0f);
            }

            if (negative && ymax > 0 && q.Bottom < 0)
            {
                s.Try((float)((q.Bottom - k * q.BottomStep / 2) / ymax), 0f);
            }

            if (negative && ymin < 0 && q.Top > 0)
            {
                s.Try((float)((q.Top + k * q.TopStep / 2) / ymin), 0f);
            }
        }

        if (affine && ymax > ymin && q.Top > q.Bottom)
        {
            foreach (float k in RangeSteps)
            {
                double t = (q.Top - q.Bottom + k * (q.TopStep + q.BottomStep) / 4) / (ymax - ymin);
                if (t > 0)
                {
                    s.Try((float)t, (float)(ymin - q.Bottom / t));
                }
            }
        }

        for (int r = 0; r < Refinements && s.A != 0f; r++)
        {
            double before = s.Error;
            s.Try(1f / s.A, s.C);
            if (!(s.Error < before))
            {
                break;
            }
        }

        a = s.A;
        c = s.C;
        return s.Error;
    }

    // A candidate scale's codes and their fit, the best kept (Solve).
    private ref struct Candidates
    {
        public Quantizer Q;
        public ReadOnlySpan<float> Y, W;
        public Span<float> G, Best;
        public bool Negative, Affine;
        public int Sign;
        public double Sxx, Error;
        public float A, C;

        // The targets (y − c0) · t, their codes, the codes' least-squares a and c.
        public void Try(float t, float c0)
        {
            if (!float.IsFinite(t) || !float.IsFinite(c0))
            {
                return;
            }

            Q.Run(Y, t, c0, G);
            double e = Fit(Y, W, G, Negative, Affine, Sign, Sxx, out float a, out float c);
            if (e < Error)
            {
                Error = e;
                A = a;
                C = c;
                G.CopyTo(Best);
            }
        }
    }

    // The least-squares a (and, `affine`, c) of y over the codes' numbers g under weights w (c of `sign` or 0; a not negative
    // unless `negative`), and the weighted squared error of a · g + c.
    private static double Fit(ReadOnlySpan<float> y, ReadOnlySpan<float> w, ReadOnlySpan<float> g, bool negative, bool affine, int sign, double sxx, out float a, out float c)
    {
        double sw = 0, sg = 0, sgg = 0, sy = 0, syg = 0;
        for (int i = 0; i < y.Length; i++)
        {
            double wi = w[i], gi = g[i], yi = y[i];
            sw += wi;
            sg += wi * gi;
            sgg += wi * gi * gi;
            sy += wi * yi;
            syg += wi * yi * gi;
        }

        double aa = sgg > 0 ? syg / sgg : 0, cc = 0;
        if (affine)
        {
            double det = sw * sgg - sg * sg;
            if (det > 1e-9 * sw * sgg)
            {
                aa = (sw * syg - sy * sg) / det;
                cc = (sy - aa * sg) / sw;
            }

            if (sign != 0 && cc * sign < 0)
            {
                cc = 0;
                aa = sgg > 0 ? syg / sgg : 0;
            }
        }

        if (!negative && aa < 0)
        {
            aa = 0;
            cc = affine && sw > 0 && sy * sign >= 0 ? sy / sw : 0;
        }

        a = (float)aa;
        c = (float)cc;
        return sxx - 2 * aa * syg - 2 * cc * sy + aa * aa * sgg + 2 * aa * cc * sg + cc * cc * sw;
    }

    // Unit u's stored scale (the product of its scale levels' numbers) and minimum (∓ the product of its minimum levels').
    private void UnitFactors(Work w, int u, out float a, out float c)
    {
        float scale = 1f, min = 1f;
        foreach (var level in scales)
        {
            scale *= level.Grid.Floats[w.Ranks[level.Index][u * unit / level.Values]];
        }

        foreach (var level in mins)
        {
            min *= level.Grid.Floats[w.Ranks[level.Index][u * unit / level.Values]];
        }

        a = scale;
        c = mins.Length == 0 ? 0f : minsAdded ? min : -min;
    }

    // Unit u's codes for scale a and minimum c, each nearest its target (x − c) / a: numbers into g, codes and factors into
    // their slots (the unit's part of each); the squared error.
    private double QuantizeUnit(Work w, int u, float a, float c, Span<float> g, Span<int> codeSlots, Span<int> factorSlots)
    {
        var y = w.X.AsSpan(u * unit, unit);
        var numbers = g.Slice(u * unit, unit);
        w.Quantizer.Run(y, a != 0f ? 1f / a : 0f, c, numbers);
        w.Quantizer.Codes.AsSpan(0, unit / codeWidth).CopyTo(codeSlots.Slice(u * unit / codeWidth));
        if (factor is { } f)
        {
            w.Quantizer.Factors.AsSpan(0, unit / f.Values).CopyTo(factorSlots.Slice(u * unit / f.Values));
        }

        double error = 0;
        for (int i = 0; i < unit; i++)
        {
            float r = y[i] - (a * numbers[i] + c);
            error += (double)r * r;
        }

        return error;
    }

    // Chunk j of a level searched again: its least-squares number given the codes (each value λ · α + β, λ its number),
    // rounded, and (a grid of few numbers) its neighbours, the codes chosen again for each and the error measured.
    private void Search(Work w, Level level, int j)
    {
        int per = level.Values / unit, u0 = j * per;
        var ranks = w.Ranks[level.Index];
        int r0 = ranks[j];
        double current = 0, saa = 0, sab = 0;
        double sign = minsAdded ? 1 : -1;
        for (int u = u0; u < u0 + per; u++)
        {
            current += w.Error[u];
            double scaleOthers = 1, minOthers = 1, scale = 1, min = 1;
            foreach (var other in scales)
            {
                double value = other.Grid.Values[w.Ranks[other.Index][u * unit / other.Values]];
                scale *= value;
                scaleOthers *= other == level ? 1 : value;
            }

            foreach (var other in mins)
            {
                double value = other.Grid.Values[w.Ranks[other.Index][u * unit / other.Values]];
                min *= value;
                minOthers *= other == level ? 1 : value;
            }

            double c = mins.Length == 0 ? 0 : sign * min;
            for (int v = u * unit; v < (u + 1) * unit; v++)
            {
                double g = w.G[v];
                double alpha = (level.Scale ? scaleOthers * g : 0) + (level.Min ? sign * minOthers : 0);
                double beta = (level.Scale ? 0 : scale * g) + (level.Min ? 0 : c);
                saa += alpha * alpha;
                sab += (w.X[v] - beta) * alpha;
            }
        }

        Span<int> candidates = stackalloc int[1 + 2 * Neighbours];
        int count = Add(candidates, 0, saa > 0 ? level.Grid.Nearest(sab / saa) : r0, r0, level.Grid.Count);
        if (level.Grid.Count <= SmallGrid)
        {
            for (int step = 1; step <= neighbours; step++)
            {
                count = Add(candidates, count, r0 - step, r0, level.Grid.Count);
                count = Add(candidates, count, r0 + step, r0, level.Grid.Count);
            }
        }

        double best = current;
        int bestRank = r0;
        int first = u0 * unit, values = per * unit, slotFirst = first / codeWidth, slotCount = values / codeWidth;
        int factorWidth = factor?.Values ?? 1, factorFirst = factor is null ? 0 : first / factorWidth, factorCount = factor is null ? 0 : values / factorWidth;
        for (int i = 0; i < count; i++)
        {
            ranks[j] = candidates[i];
            double e = 0;
            for (int u = u0; u < u0 + per && e < best; u++)
            {
                UnitFactors(w, u, out float a, out float c);
                w.TryError[u] = QuantizeUnit(w, u, a, c, w.TryG, w.TryCodes, w.TryFactors);
                e += w.TryError[u];
            }

            if (e < best)
            {
                best = e;
                bestRank = candidates[i];
                w.TryG.AsSpan(first, values).CopyTo(w.BestG.AsSpan(first, values));
                w.TryCodes.AsSpan(slotFirst, slotCount).CopyTo(w.BestCodes.AsSpan(slotFirst, slotCount));
                w.TryFactors.AsSpan(factorFirst, factorCount).CopyTo(w.BestFactors.AsSpan(factorFirst, factorCount));
                w.TryError.AsSpan(u0, per).CopyTo(w.BestError.AsSpan(u0, per));
            }
        }

        ranks[j] = bestRank;
        if (bestRank != r0)
        {
            w.BestG.AsSpan(first, values).CopyTo(w.G.AsSpan(first, values));
            w.BestCodes.AsSpan(slotFirst, slotCount).CopyTo(w.Codes.AsSpan(slotFirst, slotCount));
            w.BestFactors.AsSpan(factorFirst, factorCount).CopyTo(w.Factors.AsSpan(factorFirst, factorCount));
            w.BestError.AsSpan(u0, per).CopyTo(w.Error.AsSpan(u0, per));
        }
    }

    // A rank added to a search's candidates: within the grid, not the current one, not listed already.
    private static int Add(Span<int> list, int count, int rank, int current, int ranks)
    {
        if (rank >= 0 && rank < ranks && rank != current && !list[..count].Contains(rank))
        {
            list[count++] = rank;
        }

        return count;
    }

    // The group's bytes: each code slot's planes, each level's chunks, each factor's chunks.
    private void Write(Work w, Span<byte> group)
    {
        group.Clear();
        for (int s = 0; s < slots; s++)
        {
            codes.Write(group, s, codeGrid is { } grid ? grid.Codes[w.Codes[s]] : (ulong)w.Codes[s]);
        }

        foreach (var level in all)
        {
            var ranks = w.Ranks[level.Index];
            for (int j = 0; j < level.Chunks; j++)
            {
                level.Fields.Write(group, j, level.Grid.Codes[ranks[j]]);
            }
        }

        if (factor is { } f)
        {
            for (int c = 0; c < n / f.Values; c++)
            {
                f.Fields.Write(group, c, (ulong)w.Factors[c]);
            }
        }
    }

    // Fields of bits in a group: field f's pieces at [start[f], start[f + 1]), each its first bit and width, the lowest bits
    // of a field's number first.
    private sealed class Fields(int[] start, int[] bits, int[] widths)
    {
        public void Write(Span<byte> group, int field, ulong value)
        {
            for (int p = start[field]; p < start[field + 1]; p++)
            {
                int b = bits[p];
                for (int i = 0; i < widths[p]; i++, b++, value >>= 1)
                {
                    if ((value & 1) != 0)
                    {
                        group[b >> 3] |= (byte)(1 << (b & 7));
                    }
                }
            }
        }

        // Marks the fields' bits as owned; false where a bit is owned already or past the group.
        public bool Claim(bool[] owned)
        {
            for (int p = 0; p < bits.Length; p++)
            {
                for (int b = bits[p]; b < bits[p] + widths[p]; b++)
                {
                    if (b < 0 || b >= owned.Length || owned[b])
                    {
                        return false;
                    }

                    owned[b] = true;
                }
            }

            return true;
        }
    }

    // A level of scales or minimums (a minimum stored in a scale's bits is that scale's level, both Scale and Min).
    private sealed class Level(BlockScale source, Grid grid, int values, int chunks, Fields fields)
    {
        public BlockScale Source { get; } = source;

        public Grid Grid { get; } = grid;

        public GridQuantizer Quantizer { get; } = new(grid);

        public int Values { get; } = values;

        public int Chunks { get; } = chunks;

        public Fields Fields { get; } = fields;

        public int Index { get; set; }

        public bool Scale { get; set; }

        public bool Min { get; set; }
    }

    // A level of vector factors (a codebook scale: a sign mask a group of values, say).
    private sealed class FactorLevel(Book book, int values, Fields fields)
    {
        public Book Book { get; } = book;

        public int Values { get; } = values;

        public Fields Fields { get; } = fields;
    }

    /// <summary>The distinct finite numbers of a number type of up to 16 bits, its bias added, ascending, each with a code giving it.</summary>
    internal sealed class Grid
    {
        private static readonly System.Collections.Concurrent.ConcurrentDictionary<(NumberType Type, double Bias), Grid?> Made = new();

        private readonly int[] ranks;
        private readonly double[] mids;
        private readonly bool uniform, half;
        private readonly double first, step, inverse;

        private Grid(NumberType type, double bias, List<(double Value, ulong Code)> numbers)
        {
            numbers.Sort((p, q) => p.Value != q.Value ? p.Value.CompareTo(q.Value) : p.Code.CompareTo(q.Code));
            var values = new List<double>();
            var codes = new List<ulong>();
            foreach (var (value, code) in numbers)
            {
                if (values.Count == 0 || value != values[^1])
                {
                    values.Add(value);
                    codes.Add(code);
                }
            }

            Values = [.. values];
            Floats = [.. values.Select(v => (float)v)];
            Codes = [.. codes];
            mids = [.. Values.Zip(Values.Skip(1), (p, q) => (p + q) / 2)];
            ranks = new int[1 << type.Bits];
            ranks.AsSpan().Fill(-1);
            foreach (var (value, code) in numbers)
            {
                ranks[code] = Array.BinarySearch(Values, value);
            }

            first = Values[0];
            step = Values.Length > 1 ? Values[1] - Values[0] : 1;
            inverse = 1 / step;
            uniform = Values.Length > 1 && Values.Select((v, i) => v == first + i * step).All(x => x);
            half = bias == 0 && type is { Kind: NumberKind.Float, Bits: 16, Exponent: 5, Mantissa: 10, ExponentBias: 15, Signed: true, Subnormals: true, Specials: FloatSpecials.Ieee };
        }

        public double[] Values { get; }

        public float[] Floats { get; }

        public ulong[] Codes { get; }

        public int Count => Values.Length;

        public bool HasNegative => Values[0] < 0;

        public bool HasPositive => Values[^1] > 0;

        /// <summary>The grid of <paramref name="type"/> plus <paramref name="bias"/>, or null (not one number of up to 16 bits, or no finite number).</summary>
        public static Grid? Of(NumberType type, double bias) => type.Width == 1 && type.Bits <= 16 && type.Valid
            ? Made.GetOrAdd((type, bias), static key =>
            {
                var numbers = new List<(double, ulong)>();
                for (ulong code = 0; code < 1UL << key.Type.Bits; code++)
                {
                    double value = key.Type.Value(code) + key.Bias;
                    if (double.IsFinite(value))
                    {
                        numbers.Add((value, code));
                    }
                }

                return numbers.Count == 0 ? null : new Grid(key.Type, key.Bias, numbers);
            })
            : null;

        /// <summary>The rank of the number nearest <paramref name="x"/> (NaN as 0).</summary>
        public int Nearest(double x)
        {
            if (double.IsNaN(x))
            {
                x = 0;
            }

            if (half)
            {
                var h = (Half)x;
                return Half.IsInfinity(h) ? (x > 0 ? Count - 1 : 0) : ranks[BitConverter.HalfToUInt16Bits(h)];
            }

            if (uniform)
            {
                double q = (x - first) * inverse;
                return q <= 0 ? 0 : q >= Count - 1 ? Count - 1 : (int)Math.Round(q);
            }

            int lo = 0, hi = Count - 1;
            while (lo < hi)
            {
                int m = (lo + hi) >> 1;
                if (x > mids[m])
                {
                    lo = m + 1;
                }
                else
                {
                    hi = m;
                }
            }

            return lo;
        }

        /// <summary>
        /// Each target (y − c0) · t's nearest number into <paramref name="numbers"/> and its rank into <paramref name="ranks"/>
        /// (none where empty): <see cref="Nearest"/> a value, evenly spaced numbers by their step in one loop.
        /// </summary>
        public void Quantize(ReadOnlySpan<float> y, float t, float c0, Span<float> numbers, Span<int> ranks)
        {
            bool keep = !ranks.IsEmpty;
            if (uniform)
            {
                int last = Count - 1;
                var floats = Floats;
                for (int i = 0; i < y.Length; i++)
                {
                    double q = ((y[i] - c0) * t - first) * inverse;
                    int r = !(q > 0) ? 0 : q >= last ? last : (int)Math.Round(q);
                    numbers[i] = floats[r];
                    if (keep)
                    {
                        ranks[i] = r;
                    }
                }

                return;
            }

            for (int i = 0; i < y.Length; i++)
            {
                int r = Nearest((y[i] - c0) * t);
                numbers[i] = Floats[r];
                if (keep)
                {
                    ranks[i] = r;
                }
            }
        }
    }

    /// <summary>
    /// A codebook of up to 16-bit codes (its offset added to each entry's numbers): each code's vector of Width numbers, and
    /// the entry nearest a target vector under weights. Where the patterns of the codebook's numbers (each value one of the
    /// numbers any entry holds) are few, each pattern's nearest entries are listed once, and a target's nearest is the best
    /// of its own pattern's list (the target's values each rounded to the nearest number); otherwise every entry is measured.
    /// </summary>
    internal sealed class Book
    {
        private static readonly System.Collections.Concurrent.ConcurrentDictionary<(NumberType Type, double Offset), Book?> Made = new();

        // Entries listed a pattern, and the most patterns listed.
        private const int Listed = 16, PatternLimit = 1 << 16;

        private readonly float[] levelMids;
        private readonly int listed;
        private readonly long patterns;
        private int[]? lists;

        private Book(NumberType type, double offset)
        {
            Entries = 1 << type.Bits;
            Width = type.Width;
            Table = [.. type.Table!.Take(Entries * Width).Select(v => (float)(v + offset))];
            Levels = [.. Table.Select(v => (double)v).Distinct().Order()];
            levelMids = [.. Levels.Zip(Levels.Skip(1), (p, q) => (float)((p + q) / 2))];
            listed = Math.Min(Entries, Listed);
            double count = Math.Pow(Levels.Length, Width);
            patterns = Entries > listed && count <= PatternLimit ? (long)count : 0;
        }

        public int Entries { get; }

        public int Width { get; }

        /// <summary>Entry e's number i at e · Width + i.</summary>
        public float[] Table { get; }

        /// <summary>The distinct numbers the entries hold, ascending.</summary>
        public double[] Levels { get; }

        /// <summary>The codebook of <paramref name="type"/> plus <paramref name="offset"/>, or null (not a table of up to 16 bits).</summary>
        public static Book? Of(NumberType type, double offset) => type is { Kind: NumberKind.Table, Bits: <= 16, Table: not null } && type.Valid
            ? Made.GetOrAdd((type, offset), static key => new Book(key.Type, key.Offset))
            : null;

        /// <summary>The entry nearest <paramref name="target"/> (Width values) under <paramref name="weight"/>: the least Σ w (t − e)².</summary>
        public int Nearest(ReadOnlySpan<float> target, ReadOnlySpan<float> weight)
        {
            if (patterns == 0)
            {
                return Best(target, weight, null, 0, Entries);
            }

            var list = Volatile.Read(ref lists) ?? List();
            int pattern = 0, scale = 1;
            for (int i = 0; i < Width; i++)
            {
                pattern += Level(target[i]) * scale;
                scale *= Levels.Length;
            }

            return Best(target, weight, list, pattern * listed, listed);
        }

        // The best of `count` entries (from list[at], or entries at..at + count).
        private int Best(ReadOnlySpan<float> target, ReadOnlySpan<float> weight, int[]? list, int at, int count)
        {
            int best = 0;
            float bestDistance = float.PositiveInfinity;
            for (int c = 0; c < count; c++)
            {
                int e = list is null ? at + c : list[at + c];
                var entry = Table.AsSpan(e * Width, Width);
                float distance = 0f;
                for (int i = 0; i < Width && distance < bestDistance; i++)
                {
                    float r = target[i] - entry[i];
                    distance += weight[i] * r * r;
                }

                if (distance < bestDistance)
                {
                    bestDistance = distance;
                    best = e;
                }
            }

            return best;
        }

        // The index of the number nearest x.
        private int Level(float x)
        {
            int lo = 0, hi = Levels.Length - 1;
            while (lo < hi)
            {
                int m = (lo + hi) >> 1;
                if (x > levelMids[m])
                {
                    lo = m + 1;
                }
                else
                {
                    hi = m;
                }
            }

            return lo;
        }

        // Each pattern's nearest entries (by squared distance, then index), made once.
        private int[] List()
        {
            var made = new int[patterns * listed];
            Parallel.For(0, patterns, ComputeResources.ParallelOptions, () => (new float[Width], new float[listed], new int[listed]), (p, _, scratch) =>
            {
                var (point, distances, indices) = scratch;
                long rest = p;
                for (int i = 0; i < Width; i++, rest /= Levels.Length)
                {
                    point[i] = (float)Levels[rest % Levels.Length];
                }

                int held = 0;
                for (int e = 0; e < Entries; e++)
                {
                    float distance = 0f;
                    for (int i = 0; i < Width; i++)
                    {
                        float r = point[i] - Table[e * Width + i];
                        distance += r * r;
                    }

                    if (held == listed && distance >= distances[held - 1])
                    {
                        continue;
                    }

                    int at = held == listed ? held - 1 : held++;
                    while (at > 0 && distances[at - 1] > distance)
                    {
                        distances[at] = distances[at - 1];
                        indices[at] = indices[at - 1];
                        at--;
                    }

                    distances[at] = distance;
                    indices[at] = e;
                }

                indices.AsSpan(0, listed).CopyTo(made.AsSpan((int)(p * listed), listed));
                return scratch;
            }, static _ => { });
            Volatile.Write(ref lists, made);
            return made;
        }
    }

    // What a search quantizes with: targets to the nearest codes' numbers, and the numbers' extremes and steps beside them.
    private abstract class Quantizer
    {
        public double Top { get; protected init; }

        public double Bottom { get; protected init; }

        public double TopStep { get; protected init; }

        public double BottomStep { get; protected init; }

        // The candidates' half steps from the extremes (Solve).
        public float[] Steps { get; protected init; } = ScaleSteps;

        // Each target (y − c0) · t to its nearest codes' number in g.
        public abstract void Run(ReadOnlySpan<float> y, float t, float c0, Span<float> g);
    }

    // A level's numbers (the factoring's children).
    private sealed class GridQuantizer : Quantizer
    {
        private readonly Grid grid;

        public GridQuantizer(Grid grid)
        {
            this.grid = grid;
            var v = grid.Values;
            Top = v[^1];
            Bottom = v[0];
            TopStep = v.Length > 1 ? v[^1] - v[^2] : 1;
            BottomStep = v.Length > 1 ? v[1] - v[0] : 1;
        }

        public override void Run(ReadOnlySpan<float> y, float t, float c0, Span<float> g) => grid.Quantize(y, t, c0, g, []);
    }

    // A unit's codes (a thread's own: it keeps the codes it chose): a grid's nearest numbers, or a codebook's nearest entries
    // (with vector factors: each factor's entry nearest the targets' signs, each code's entry nearest the targets over the
    // factors, then each factor's again given the entries and the entries again, kept where nearer).
    private sealed class CodeQuantizer : Quantizer
    {
        private readonly DescribedEncoder e;
        private readonly int[] savedCodes, savedFactors;
        private readonly float[] targets, saved, target, weight;

        public CodeQuantizer(DescribedEncoder encoder)
        {
            e = encoder;
            Top = e.top;
            Bottom = e.bottom;
            TopStep = e.topStep;
            BottomStep = e.bottomStep;
            Steps = e.codeBook is null ? ScaleSteps : BookSteps;
            Codes = new int[e.unit];
            Factors = new int[e.unit];
            savedCodes = new int[e.unit];
            savedFactors = new int[e.unit];
            targets = new float[e.unit];
            saved = new float[e.unit];
            int width = Math.Max(e.codeWidth, e.factor?.Values ?? 1);
            target = new float[width];
            weight = new float[width];
        }

        // The last run's codes (a value's rank, or an entry's index, a slot each) and factors (a chunk each).
        public int[] Codes { get; }

        public int[] Factors { get; }

        public override void Run(ReadOnlySpan<float> y, float t, float c0, Span<float> g)
        {
            if (e.codeGrid is { } grid)
            {
                grid.Quantize(y, t, c0, g, Codes);
                return;
            }

            var z = targets.AsSpan(0, y.Length);
            for (int i = 0; i < y.Length; i++)
            {
                z[i] = (y[i] - c0) * t;
            }

            var book = e.codeBook!;
            if (e.factor is not { } f)
            {
                Entries(book, null, z, g);
                return;
            }

            var factors = f.Book;
            int fw = factors.Width, width = book.Width;
            for (int c = 0; c < z.Length / fw; c++)
            {
                for (int i = 0; i < fw; i++)
                {
                    float zi = z[c * fw + i];
                    target[i] = zi < 0 ? -1f : 1f;
                    weight[i] = zi * zi;
                }

                Factors[c] = factors.Nearest(target.AsSpan(0, fw), weight.AsSpan(0, fw));
            }

            double first = Entries(book, factors, z, g);
            bool changed = false;
            for (int c = 0; c < z.Length / fw; c++)
            {
                for (int i = 0; i < fw; i++)
                {
                    int v = c * fw + i;
                    float number = book.Table[Codes[v / width] * width + v % width];
                    target[i] = number != 0f ? z[v] / number : 0f;
                    weight[i] = number * number;
                }

                int k = factors.Nearest(target.AsSpan(0, fw), weight.AsSpan(0, fw));
                if (k != Factors[c])
                {
                    if (!changed)
                    {
                        Codes.AsSpan(0, z.Length / width).CopyTo(savedCodes);
                        Factors.AsSpan(0, z.Length / fw).CopyTo(savedFactors);
                        g.CopyTo(saved);
                        changed = true;
                    }

                    Factors[c] = k;
                }
            }

            if (changed && !(Entries(book, factors, z, g) < first))
            {
                savedCodes.AsSpan(0, z.Length / width).CopyTo(Codes);
                savedFactors.AsSpan(0, z.Length / fw).CopyTo(Factors);
                saved.AsSpan(0, z.Length).CopyTo(g);
            }
        }

        // Each entry's code nearest its targets over the factors (none: 1); numbers into g; the squared distance to z.
        private double Entries(Book book, Book? factors, ReadOnlySpan<float> z, Span<float> g)
        {
            int width = book.Width, fw = factors?.Width ?? 1;
            double total = 0;
            for (int s = 0; s < z.Length / width; s++)
            {
                for (int i = 0; i < width; i++)
                {
                    int v = s * width + i;
                    float phi = factors is null ? 1f : factors.Table[Factors[v / fw] * fw + v % fw];
                    target[i] = phi != 0f ? z[v] / phi : 0f;
                    weight[i] = phi * phi;
                }

                int k = book.Nearest(target.AsSpan(0, width), weight.AsSpan(0, width));
                Codes[s] = k;
                for (int i = 0; i < width; i++)
                {
                    int v = s * width + i;
                    float phi = factors is null ? 1f : factors.Table[Factors[v / fw] * fw + v % fw];
                    g[v] = phi * book.Table[k * width + i];
                    float r = z[v] - g[v];
                    total += (double)r * r;
                }
            }

            return total;
        }
    }

    // A thread's working arrays for one encoder.
    private sealed class Work
    {
        [ThreadStatic]
        private static Work? last;

        private Work(DescribedEncoder e)
        {
            Owner = e;
            int n = e.n;
            X = new float[n];
            G = new float[n];
            TryG = new float[n];
            BestG = new float[n];
            Ones = new float[n];
            Ones.AsSpan().Fill(1f);
            Codes = new int[n];
            TryCodes = new int[n];
            BestCodes = new int[n];
            Factors = new int[n];
            TryFactors = new int[n];
            BestFactors = new int[n];
            A = new float[e.units];
            C = new float[e.units];
            WA = new float[e.units];
            WC = new float[e.units];
            T = new float[e.units];
            Wt = new float[e.units];
            Error = new double[e.units];
            TryError = new double[e.units];
            BestError = new double[e.units];
            Ranks = [.. e.all.Select(l => new int[l.Chunks])];
            LevelTargets = [.. e.all.Select(l => new float[l.Chunks])];
            LevelWeights = [.. e.all.Select(l => new float[l.Chunks])];
            Quantizer = new CodeQuantizer(e);
        }

        public DescribedEncoder Owner { get; }

        public float[] X { get; }

        public float[] G { get; }

        public float[] TryG { get; }

        public float[] BestG { get; }

        public float[] Ones { get; }

        public int[] Codes { get; }

        public int[] TryCodes { get; }

        public int[] BestCodes { get; }

        public int[] Factors { get; }

        public int[] TryFactors { get; }

        public int[] BestFactors { get; }

        public float[] A { get; }

        public float[] C { get; }

        public float[] WA { get; }

        public float[] WC { get; }

        public float[] T { get; }

        public float[] Wt { get; }

        public double[] Error { get; }

        public double[] TryError { get; }

        public double[] BestError { get; }

        public int[][] Ranks { get; }

        public float[][] LevelTargets { get; }

        public float[][] LevelWeights { get; }

        public CodeQuantizer Quantizer { get; }

        public static Work For(DescribedEncoder e) => last is { } w && w.Owner == e ? w : last = new Work(e);
    }
}
