// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models;

namespace Idrak;

// The descriptions the old element types held or implied, gathered here as plain values (TypeDescription) and used by
// nothing: each becomes a type definition of its own on the new interface. Each is the description the old entry gave,
// or the one its older entries implied (a float's bits, an integer's bits and sign, ggml's scaled blocks, super-blocks and
// grid codebooks), written out. ggml's are as GGUF lays them out, read as ggml's dequantize_row_* functions read them.
internal static class Descriptions
{
    private static readonly NumberType Float16Number = NumberType.Floating(5, 10);

    // Per-element floats: one value a block (or the values a byte holds).
    internal static readonly TypeDescription Float32 = new(1, 4, NumberType.Floating(8, 23), [new CodePlane(0, 32, 1)]);

    internal static readonly TypeDescription BFloat16 = new(1, 2, NumberType.Floating(8, 7), [new CodePlane(0, 16, 1)]);

    internal static readonly TypeDescription Float16 = new(1, 2, Float16Number, [new CodePlane(0, 16, 1)]);

    internal static readonly TypeDescription Float64 = new(1, 8, NumberType.Floating(11, 52), [new CodePlane(0, 64, 1)]);

    // fp8 E4M3fn: no infinity, ±448 at most, the all-ones code a NaN.
    internal static readonly TypeDescription Float8E4M3 = new(1, 1, NumberType.Floating(4, 3, specials: FloatSpecials.AllOnesNaN), [new CodePlane(0, 8, 1)]);

    // fp8 E5M2: a half's high byte.
    internal static readonly TypeDescription Float8E5M2 = new(1, 1, NumberType.Floating(5, 2), [new CodePlane(0, 8, 1)]);

    // OCP's fp4 E2M1: ±0, 0.5, 1, 1.5, 2, 3, 4, 6, no infinity or NaN; two a byte, the first in the low half.
    internal static readonly TypeDescription Float4E2M1 = new(2, 1, NumberType.Floating(2, 1, bias: 1, specials: FloatSpecials.None), [new CodePlane(0, 4, 2)]);

    // OCP's E8M0 scale: 2^(e − 127), no sign, no mantissa, 0xFF NaN.
    internal static readonly TypeDescription Float8E8M0 = new(1, 1, NumberType.Floating(8, 0, signed: false, bias: 127, subnormals: false, specials: FloatSpecials.AllOnesNaN),
        [new CodePlane(0, 8, 1)]);

    // Per-element integers, two's complement when signed, little-endian.
    internal static readonly TypeDescription Int64 = new(1, 8, NumberType.Integer(64, signed: true), [new CodePlane(0, 64, 1)]);

    internal static readonly TypeDescription UInt64 = new(1, 8, NumberType.Integer(64, signed: false), [new CodePlane(0, 64, 1)]);

    internal static readonly TypeDescription Int32 = new(1, 4, NumberType.Integer(32, signed: true), [new CodePlane(0, 32, 1)]);

    internal static readonly TypeDescription UInt32 = new(1, 4, NumberType.Integer(32, signed: false), [new CodePlane(0, 32, 1)]);

    internal static readonly TypeDescription Int16 = new(1, 2, NumberType.Integer(16, signed: true), [new CodePlane(0, 16, 1)]);

    internal static readonly TypeDescription UInt16 = new(1, 2, NumberType.Integer(16, signed: false), [new CodePlane(0, 16, 1)]);

    internal static readonly TypeDescription Int8 = new(1, 1, NumberType.Integer(8, signed: true), [new CodePlane(0, 8, 1)]);

    internal static readonly TypeDescription UInt8 = new(1, 1, NumberType.Integer(8, signed: false), [new CodePlane(0, 8, 1)]);

    // 4-bit signed integers, two a byte, the first in the low nibble.
    internal static readonly TypeDescription Int4 = new(2, 1, NumberType.Integer(4, signed: true), [new CodePlane(0, 4, 2)]);

    // 2-bit unsigned integers, four a byte, the first in the lowest bits.
    internal static readonly TypeDescription UInt2 = new(4, 1, NumberType.Integer(2, signed: false), [new CodePlane(0, 2, 4)]);

    // 2-bit signed codes in blocks of 32, each block one float16 scale (its first 2 bytes) then its 8 bytes of codes.
    internal static readonly TypeDescription Int2 = new(32, 10, NumberType.Integer(2, signed: true), [new CodePlane(2, 2, 4)],
        Scales: [new BlockScale(Float16Number, 32, 0)]);

    // OCP's MXFP4 as GGUF lays it out: the E8M0 scale byte, then value i in byte i's low half and i + 16 in its high half.
    internal static readonly TypeDescription MXFP4 = new(32, 17, Float4E2M1.Code, [new CodePlane(1, 4, 32)], Scales: [new BlockScale(Float8E8M0.Code, 32, 0)]);

    // ggml's codes times one float16 scale a block (Q4_0's 4-bit halves in ggml's order, value i + 16 in byte i's high half).
    internal static readonly TypeDescription Q8_0 = new(32, 34, NumberType.Integer(8, signed: true), [new CodePlane(2, 8, 1)], 0, [new BlockScale(Float16Number, 32, 0)]);

    internal static readonly TypeDescription Q4_0 = new(32, 18, NumberType.Integer(4, signed: false), [new CodePlane(2, 4, 32)], -8, [new BlockScale(Float16Number, 32, 0)]);

    // IQ4_NL's non-linear grid (ggml's kvalues_iq4nl): the value each 4-bit code stands for, before the block's scale.
    private static readonly double[] Iq4Grid = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

    internal static readonly TypeDescription IQ4_NL = new(32, 18, NumberType.Entries(4, Iq4Grid), [new CodePlane(2, 4, 32)], 0, [new BlockScale(Float16Number, 32, 0)]);

    // Q4_1 and Q5_1: d · q + m; Q5_0 and Q5_1's fifth bits a stream of 1-bit fields, Q5_0's code less 16.
    internal static readonly TypeDescription Q4_1 = new(32, 20, NumberType.Integer(4, false), [new CodePlane(4, 4, 32)], Scales: [new BlockScale(Float16Number, 32, 0)],
        Mins: [new BlockScale(Float16Number, 32, 2)], MinsAdded: true);

    internal static readonly TypeDescription Q5_0 = new(32, 22, NumberType.Integer(5, false), [new CodePlane(6, 4, 32), new CodePlane(2, 1, 1, Stream: true)], -16,
        [new BlockScale(Float16Number, 32, 0)]);

    internal static readonly TypeDescription Q5_1 = new(32, 24, NumberType.Integer(5, false), [new CodePlane(8, 4, 32), new CodePlane(4, 1, 1, Stream: true)],
        Scales: [new BlockScale(Float16Number, 32, 0)], Mins: [new BlockScale(Float16Number, 32, 2)], MinsAdded: true);

    private static ScalePiece[][] Pieces(int count, Func<int, ScalePiece[]> of) => [.. Enumerable.Range(0, count).Select(of)];

    // Q4_K's and Q5_K's twelve bytes from byte 4: sub-blocks 0-3 a 6-bit scale and minimum each; 4-7 their low four bits
    // in bytes 12-15 and their high two in the top bits of bytes 4-11.
    private static ScalePiece[] KScale(int j) => j < 4 ? [new(4 + j, 0, 6)] : [new(8 + j, 0, 4), new(j, 6, 2)];

    private static ScalePiece[] KMin(int j) => j < 4 ? [new(8 + j, 0, 6)] : [new(8 + j, 4, 4), new(4 + j, 6, 2)];

    // Q3_K's twelve bytes from byte 96: sixteen 6-bit scales, the low four bits in the halves of bytes 96-103, the high two
    // in the bit pairs of bytes 104-107.
    private static ScalePiece[] K3Scale(int j) => (j / 4) switch
    {
        0 => [new(96 + j, 0, 4), new(104 + j, 0, 2)],
        1 => [new(96 + j, 0, 4), new(104 + j - 4, 2, 2)],
        2 => [new(96 + j - 8, 4, 4), new(104 + j - 8, 4, 2)],
        _ => [new(96 + j - 8, 4, 4), new(104 + j - 12, 6, 2)],
    };

    // The k-quants: value v of sub-block j (d · (scale_j + bias)) · (code + offset) − dmin · min_j, d and dmin float16, the
    // codes planes of bits (the first the lowest), scale_j and min_j assembled from their pieces.
    internal static readonly TypeDescription Q2_K = new(256, 84, NumberType.Integer(2, false), [new CodePlane(16, 2, 128)], 0,
        [new BlockScale(Float16Number, 256, 80), new BlockScale(NumberType.Integer(4, false), 16, 0, Pieces(16, j => [new(j, 0, 4)]))],
        [new BlockScale(Float16Number, 256, 82), new BlockScale(NumberType.Integer(4, false), 16, 0, Pieces(16, j => [new(j, 4, 4)]))]);

    internal static readonly TypeDescription Q3_K = new(256, 110, NumberType.Integer(3, false), [new CodePlane(32, 2, 128), new CodePlane(0, 1, 256)], -4,
        [new BlockScale(Float16Number, 256, 108), new BlockScale(NumberType.Integer(6, false), 16, 0, Pieces(16, K3Scale), -32)]);

    internal static readonly TypeDescription Q4_K = new(256, 144, NumberType.Integer(4, false), [new CodePlane(16, 4, 64)], 0,
        [new BlockScale(Float16Number, 256, 0), new BlockScale(NumberType.Integer(6, false), 32, 0, Pieces(8, KScale))],
        [new BlockScale(Float16Number, 256, 2), new BlockScale(NumberType.Integer(6, false), 32, 0, Pieces(8, KMin))]);

    internal static readonly TypeDescription Q5_K = new(256, 176, NumberType.Integer(5, false), [new CodePlane(48, 4, 64), new CodePlane(16, 1, 256)], 0,
        [new BlockScale(Float16Number, 256, 0), new BlockScale(NumberType.Integer(6, false), 32, 0, Pieces(8, KScale))],
        [new BlockScale(Float16Number, 256, 2), new BlockScale(NumberType.Integer(6, false), 32, 0, Pieces(8, KMin))]);

    internal static readonly TypeDescription Q6_K = new(256, 210, NumberType.Integer(6, false), [new CodePlane(0, 4, 128), new CodePlane(128, 2, 128)], -32,
        [new BlockScale(Float16Number, 256, 208), new BlockScale(NumberType.Integer(8, true), 16, 0, Pieces(16, j => [new(192 + j, 0, 8)]))]);

    // Bits [first, first + count) of the bytes from `at`, as pieces a byte each.
    private static ScalePiece[] Bits(int at, int first, int count)
    {
        var pieces = new List<ScalePiece>();
        while (count > 0)
        {
            int shift = first % 8, take = Math.Min(count, 8 - shift);
            pieces.Add(new ScalePiece(at + first / 8, shift, take));
            first += take;
            count -= take;
        }

        return [.. pieces];
    }

    // The sign vectors a byte of sign bits gives (bit i set: value i negative); and KSigns' 7-bit indices' masks.
    private static NumberType SignCodebook(int bits, Func<int, int> mask) =>
        NumberType.Codebook(bits, 8, [.. Enumerable.Range(0, 1 << bits).SelectMany(s => Enumerable.Range(0, 8).Select(i => (mask(s) >> i & 1) != 0 ? -1d : 1d))]);

    private static readonly NumberType SignBits = SignCodebook(8, s => s);

    private static readonly NumberType KSignVectors = SignCodebook(7, s => GgufGrids.KSigns[s]);

    // 1 + 2s for an s of `bits` bits.
    private static NumberType Odd(int bits) => NumberType.Entries(bits, [.. Enumerable.Range(0, 1 << bits).Select(s => 1d + 2 * s)]);

    // A grid's entries as a codebook of `index` bits, `width` values an entry, each magnitude times `factor`.
    private static NumberType GridCodebook(int index, int width, sbyte[] grid, double factor) => NumberType.Codebook(index, width, [.. grid.Select(v => v * factor)]);

    // IQ2_XXS and IQ3_XXS: a codebook of the grid's magnitudes (times the sub-scales' factor), each entry's index a byte;
    // the super-scale, each sub-block's 4-bit scale plus 1/2, and each group of 8 values' vector of ±1 (KSigns' masks).
    internal static readonly TypeDescription IQ2_XXS = new(256, 66, GridCodebook(8, 8, GgufGrids.IQ2XXS, 0.25), [new CodePlane(2, 8, 4, 8)], 0,
    [
        new BlockScale(Float16Number, 256, 0),
        new BlockScale(NumberType.Integer(4, false), 32, 0, Pieces(8, j => Bits(2 + 8 * j + 4, 28, 4)), 0.5),
        new BlockScale(KSignVectors, 8, 0, Pieces(32, s => Bits(2 + 8 * (s / 4) + 4, 7 * (s % 4), 7))),
    ]);

    internal static readonly TypeDescription IQ3_XXS = new(256, 98, GridCodebook(8, 4, GgufGrids.IQ3XXS, 0.5), [new CodePlane(2, 8, 1)], 0,
    [
        new BlockScale(Float16Number, 256, 0),
        new BlockScale(NumberType.Integer(4, false), 32, 0, Pieces(8, j => Bits(66 + 4 * j, 28, 4)), 0.5),
        new BlockScale(KSignVectors, 8, 0, Pieces(32, s => Bits(66 + 4 * (s / 4), 7 * (s % 4), 7))),
    ]);

    // IQ4_XS: IQ4_NL's grid entry by a 4-bit code, times d and each sub-block's 6-bit scale less 32 (its low 4 bits a nibble
    // from byte 4, its high 2 from the 16-bit word at 2).
    internal static readonly TypeDescription IQ4_XS = new(256, 136, NumberType.Entries(4, Iq4Grid), [new CodePlane(8, 4, 32)],
        Scales: [new BlockScale(Float16Number, 256, 0), new BlockScale(NumberType.Integer(6, false), 32, 0, Pieces(8, ib => [new(4 + ib / 2, 4 * (ib % 2), 4), new(2 + ib / 4, 2 * (ib % 4), 2)]), -32)]);

    // TQ2_0: 2-bit codes less 1 (−1, 0, 1), value 128c + 32s + m in byte 32c + m at bit 2s; TQ1_0: base-3 digits less 1, five a
    // byte read as a fraction, value 32p + m digit p of byte m, 160 + 16p + m digit p of byte 32 + m, 240 + 4p + m digit p of
    // byte 48 + m; d after them.
    internal static readonly TypeDescription TQ2_0 = new(256, 66, NumberType.Integer(2, false), [new CodePlane(0, 2, 128)], -1, [new BlockScale(Float16Number, 256, 64)]);

    private static FieldPlace[] Fields(int count, Func<int, FieldPlace> of) => [.. Enumerable.Range(0, count).Select(of)];

    private static FieldPlace Tq1Digit(int v) => v < 160 ? new(v % 32, v / 32) : v < 240 ? new(32 + (v - 160) % 16, (v - 160) / 16) : new(48 + (v - 240) % 4, (v - 240) / 4);

    internal static readonly TypeDescription TQ1_0 = new(256, 54, NumberType.Integer(2, false), [new CodePlane(0, 2, 1, Places: Fields(256, Tq1Digit), Radix: 3)], -1,
        [new BlockScale(Float16Number, 256, 52)]);

    // IQ2_XS, IQ2_S and IQ3_S: a grid codebook whose index takes more than a byte (9 bits in a 16-bit word with a 7-bit sign
    // index, KSigns' mask; or a byte plus 2 or 1 high bits in a plane of their own), each sub-block's 4-bit scale (s + 1/2, the
    // grid's quarter folded into its magnitudes; IQ3_S's 1 + 2s, a table), and the signs a vector of factors a group of 8
    // (IQ2_S's and IQ3_S's a byte of sign bits).
    internal static readonly TypeDescription IQ2_XS = new(256, 74, GridCodebook(9, 8, GgufGrids.IQ2XS, 0.25), [new CodePlane(2, 9, 1, Places: Fields(32, e => new(2 + 2 * e, 0)))],
        Scales: [new BlockScale(Float16Number, 256, 0), new BlockScale(NumberType.Integer(4, false), 16, 66, Bias: 0.5), new BlockScale(KSignVectors, 8, 0, Pieces(32, e => [new(3 + 2 * e, 1, 7)]))]);

    internal static readonly TypeDescription IQ2_S = new(256, 82, GridCodebook(10, 8, GgufGrids.IQ2S, 0.25), [new CodePlane(2, 8, 1), new CodePlane(66, 2, 4)],
        Scales: [new BlockScale(Float16Number, 256, 0), new BlockScale(NumberType.Integer(4, false), 16, 74, Bias: 0.5), new BlockScale(SignBits, 8, 34)]);

    internal static readonly TypeDescription IQ3_S = new(256, 110, GridCodebook(9, 4, GgufGrids.IQ3S, 1), [new CodePlane(2, 8, 1), new CodePlane(66, 1, 8)],
        Scales: [new BlockScale(Float16Number, 256, 0), new BlockScale(Odd(4), 32, 106), new BlockScale(SignBits, 8, 74)]);

    // IQ1_S and IQ1_M: a codebook of the grid's values plus the delta (±1/8, its sign the index's top bit): an 8-bit index, its
    // 3 high bits, and the delta's sign (IQ1_S's a 16-bit word's of 4 entries, listed; IQ1_M's a nibble an entry), times each
    // sub-block's 1 + 2s (a table) and d (IQ1_M's float16 the top nibbles of its 4 scale words).
    private static readonly NumberType Iq1Codebook = NumberType.Codebook(12, 8,
        [.. Enumerable.Range(0, 1 << 12).SelectMany(c => Enumerable.Range(0, 8).Select(j => GgufGrids.IQ1S[(c & 2047) * 8 + j] + (c >> 11 != 0 ? -0.125 : 0.125)))]);

    internal static readonly TypeDescription IQ1_S = new(256, 50, Iq1Codebook,
        [new CodePlane(2, 8, 1), new CodePlane(0, 3, 1, Places: Fields(32, i => new(34 + 2 * (i / 4) + 3 * (i % 4) / 8, 3 * (i % 4) % 8))),
            new CodePlane(0, 1, 1, Places: Fields(32, i => new(35 + 2 * (i / 4), 7)))],
        Scales: [new BlockScale(Float16Number, 256, 0), new BlockScale(Odd(3), 32, 0, Pieces(8, ib => [new(35 + 2 * ib, 4, 3)]))]);

    internal static readonly TypeDescription IQ1_M = new(256, 56, Iq1Codebook, [new CodePlane(0, 8, 1), new CodePlane(32, 4, 2)],
        Scales: [new BlockScale(Float16Number, 256, 0, [[new(49, 4, 4), new(51, 4, 4), new(53, 4, 4), new(55, 4, 4)]]),
            new BlockScale(Odd(3), 16, 0, Pieces(16, m => Bits(48 + 2 * (m / 4), 3 * (m % 4), 3)))]);
}
