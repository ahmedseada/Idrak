// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

// ggml's block formats as element types (plan 37, T0): each block type of the GGUF types (GgufTypes) registered in
// ElementTypes under its lower-case name (q4_0, q4_k, iq2_xxs, ...), its bits a value from its block, its scales
// where its layout has a plain description, and its widening to float32 through the GGUF type's own dequantizer (read at
// each call, so an app's type registered over the library's is the one used). Per-element GGUF types (F32, F16, BF16)
// are the built-in types already, and a block type the library defines itself (MXFP4: OCP's, laid out as GGUF lays it)
// is the library's, which reads a GGUF file's blocks of it as they are.
internal static class GgufElementTypes
{
    // The scales of the formats whose layout is a plain per-block (or per-sub-block) scale, as ggml lays them out.
    private static readonly Dictionary<string, BlockScales> Scales = new(StringComparer.OrdinalIgnoreCase)
    {
        ["Q4_0"] = new("float16", 1, false, "one scale d, then 32 4-bit values"),
        ["Q4_1"] = new("float16", 1, true, "a scale d and a minimum m, then 32 4-bit values"),
        ["Q5_0"] = new("float16", 1, false, "one scale d, the 32 fifth bits, then 32 4-bit values"),
        ["Q5_1"] = new("float16", 1, true, "a scale d and a minimum m, the 32 fifth bits, then 32 4-bit values"),
        ["Q8_0"] = new("float16", 1, false, "one scale d, then 32 8-bit values"),
        ["Q2_K"] = new("4-bit codes of float16 super-scales", 16, true, "16 sub-blocks of 16, each a 4-bit scale and minimum, scaled by d and dmin"),
        ["Q3_K"] = new("6-bit codes of a float16 super-scale", 16, false, "16 sub-blocks of 16, each a 6-bit scale, scaled by d"),
        ["Q4_K"] = new("6-bit codes of float16 super-scales", 8, true, "8 sub-blocks of 32, each a 6-bit scale and minimum, scaled by d and dmin"),
        ["Q5_K"] = new("6-bit codes of float16 super-scales", 8, true, "8 sub-blocks of 32, each a 6-bit scale and minimum, scaled by d and dmin"),
        ["Q6_K"] = new("8-bit codes of a float16 super-scale", 16, false, "16 sub-blocks of 16, each an 8-bit scale, scaled by d"),
    };

    // The formats whose blocks are codes times one float16 scale (ScaledBlock): a kernel widens them from this description
    // alone. Q4_0's codes are 4-bit halves in ggml's order (value i + 16 in the high half of byte i), less 8. Q4_1 and Q5_x
    // add a minimum or high bits, the k-quants scale sub-blocks: none of them is described this way yet.
    // IQ4_NL's non-linear grid (ggml's kvalues_iq4nl): the value each 4-bit code stands for, before the block's scale.
    private static readonly int[] Iq4Grid = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

    private static readonly Dictionary<string, ScaledBlock> Described = new(StringComparer.OrdinalIgnoreCase)
    {
        ["Q8_0"] = new(32, 34, 8, SignedCodes: true, CodeBias: 0, new FloatBits(5, 10), ScaleOffset: 0, CodeOffset: 2),
        ["Q4_0"] = new(32, 18, 4, SignedCodes: false, CodeBias: -8, new FloatBits(5, 10), ScaleOffset: 0, CodeOffset: 2, NibblesSplit: true),
        ["IQ4_NL"] = new(32, 18, 4, SignedCodes: false, CodeBias: 0, new FloatBits(5, 10), ScaleOffset: 0, CodeOffset: 2, NibblesSplit: true, CodeTable: Iq4Grid),
    };

    // The k-quants as super-blocks (SuperBlock): their planes of codes, sub-block scales and minimums as pieces, read the way
    // ggml's dequantize_row_q*_K reads them (Gguf.DequantizeBlock); a kernel widens them from this description alone.
    private static readonly FloatBits Half16 = new(5, 10);

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

    private static readonly Dictionary<string, SuperBlock> SuperDescribed = new(StringComparer.OrdinalIgnoreCase)
    {
        ["Q2_K"] = new(256, 84, [new CodePlane(16, 2, 128)], CodeBias: 0, Half16, ScaleOffset: 80, MinOffset: 82, SubValues: 16,
            Pieces(16, j => [new(j, 0, 4)]), SubScaleBias: 0, SubScalesSigned: false, Pieces(16, j => [new(j, 4, 4)])),
        ["Q3_K"] = new(256, 110, [new CodePlane(32, 2, 128), new CodePlane(0, 1, 256)], CodeBias: -4, Half16, ScaleOffset: 108, MinOffset: -1, SubValues: 16,
            Pieces(16, K3Scale), SubScaleBias: -32, SubScalesSigned: false, null),
        ["Q4_K"] = new(256, 144, [new CodePlane(16, 4, 64)], CodeBias: 0, Half16, ScaleOffset: 0, MinOffset: 2, SubValues: 32,
            Pieces(8, KScale), SubScaleBias: 0, SubScalesSigned: false, Pieces(8, KMin)),
        ["Q5_K"] = new(256, 176, [new CodePlane(48, 4, 64), new CodePlane(16, 1, 256)], CodeBias: 0, Half16, ScaleOffset: 0, MinOffset: 2, SubValues: 32,
            Pieces(8, KScale), SubScaleBias: 0, SubScalesSigned: false, Pieces(8, KMin)),
        ["Q6_K"] = new(256, 210, [new CodePlane(0, 4, 128), new CodePlane(128, 2, 128)], CodeBias: -32, Half16, ScaleOffset: 208, MinOffset: -1, SubValues: 16,
            Pieces(16, j => [new(192 + j, 0, 8)]), SubScaleBias: 0, SubScalesSigned: true, null),
    };

    // The grid-codebook formats (GridBlock): IQ2_XXS's 8 × (4 entry bytes, then a word of 4 × 7 sign bits and a 4-bit scale),
    // IQ3_XXS's 64 entry bytes then 8 words of 4 × 7 sign bits and a 4-bit scale; read as ggml's dequantize_row_iq2_xxs and
    // dequantize_row_iq3_xxs read them (GgufFile.LowBit.cs).
    private static ScalePiece[] Bits(int at, int first, int count)
    {
        // Bits [first, first + count) of the bytes from `at`, as pieces a byte each.
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

    private static readonly int[] KSignTable = [.. GgufGrids.KSigns];

    // The formats described as data alone (TypeDescription), read as ggml's dequantize_row_* functions read them
    // (Gguf.DequantizeBlock, GgufFile.LowBit.cs), each value exactly theirs: a code from its planes (an integer, a table's
    // entry, a codebook's index), plus the code offset, times the scales, with the minimum added.
    // - Q4_1 and Q5_1: d · q + m (a float16 scale and minimum a block); Q5_0 and Q5_1's fifth bits a stream of 1-bit
    //   fields (value v's at bit v of the 32-bit word), Q5_0's code less 16.
    // - IQ4_XS: IQ4_NL's grid entry by a 4-bit code, times d and each sub-block's 6-bit scale less 32 (its low 4 bits a
    //   nibble from byte 4, its high 2 from the 16-bit word at 2).
    // - TQ2_0: 2-bit codes less 1 (−1, 0, 1), value 128c + 32s + m in byte 32c + m at bit 2s; TQ1_0: base-3 digits less 1,
    //   five a byte read as a fraction (a plane of digits: CodePlane.Radix), value 32p + m digit p of byte m, 160 + 16p + m
    //   digit p of byte 32 + m, 240 + 4p + m digit p of byte 48 + m; d after them.
    // - IQ2_XS, IQ2_S and IQ3_S: a grid codebook whose index takes more than a byte (9 bits in a 16-bit word with a 7-bit
    //   sign index, KSigns' mask; or a byte plus 2 or 1 high bits in a plane of their own), each sub-block's 4-bit scale (s +
    //   1/2, the grid's quarter folded into its magnitudes; IQ3_S's 1 + 2s, a table), and the signs a vector of factors a
    //   group of 8 (IQ2_S's and IQ3_S's a byte of sign bits).
    // - IQ1_S and IQ1_M: a codebook of the grid's values plus the delta (±1/8, its sign the index's top bit): an 8-bit
    //   index, its 3 high bits, and the delta's sign (IQ1_S's a 16-bit word's of 4 entries, listed; IQ1_M's a nibble an
    //   entry), times each sub-block's 1 + 2s (a table) and d (IQ1_M's float16 the top nibbles of its 4 scale words).
    private static readonly NumberType Float16Number = NumberType.Floating(5, 10);

    // The sign vectors a byte of sign bits gives (bit i set: value i negative); and KSigns' 7-bit indices' masks.
    private static readonly NumberType SignBits = SignCodebook(8, s => s);

    private static readonly NumberType KSignVectors = SignCodebook(7, s => KSignTable[s]);

    private static NumberType SignCodebook(int bits, Func<int, int> mask) =>
        NumberType.Codebook(bits, 8, [.. Enumerable.Range(0, 1 << bits).SelectMany(s => Enumerable.Range(0, 8).Select(i => (mask(s) >> i & 1) != 0 ? -1d : 1d))]);

    // 1 + 2s for an s of `bits` bits.
    private static NumberType Odd(int bits) => NumberType.Entries(bits, [.. Enumerable.Range(0, 1 << bits).Select(s => 1d + 2 * s)]);

    // A grid's entries as a codebook of `index` bits, `width` values an entry, each magnitude times `factor`.
    private static NumberType GridCodebook(int index, int width, sbyte[] grid, double factor) => NumberType.Codebook(index, width, [.. grid.Select(v => v * factor)]);

    // IQ1_S's and IQ1_M's codebook: index c's low 11 bits an IQ1 grid entry, its bit 11 the delta's sign (set: −1/8).
    private static readonly NumberType Iq1Codebook = NumberType.Codebook(12, 8,
        [.. Enumerable.Range(0, 1 << 12).SelectMany(c => Enumerable.Range(0, 8).Select(j => GgufGrids.IQ1S[(c & 2047) * 8 + j] + (c >> 11 != 0 ? -0.125 : 0.125)))]);

    private static FieldPlace[] Fields(int count, Func<int, FieldPlace> of) => [.. Enumerable.Range(0, count).Select(of)];

    // TQ1_0's digits: value v's byte and digit (above).
    private static FieldPlace Tq1Digit(int v) => v < 160 ? new(v % 32, v / 32) : v < 240 ? new(32 + (v - 160) % 16, (v - 160) / 16) : new(48 + (v - 240) % 4, (v - 240) / 4);

    private static readonly Dictionary<string, TypeDescription> Descriptions = new(StringComparer.OrdinalIgnoreCase)
    {
        ["Q4_1"] = new(32, 20, NumberType.Integer(4, false), [new CodePlane(4, 4, 32)], Scales: [new BlockScale(Float16Number, 32, 0)], Mins: [new BlockScale(Float16Number, 32, 2)],
            MinsAdded: true),
        ["Q5_0"] = new(32, 22, NumberType.Integer(5, false), [new CodePlane(6, 4, 32), new CodePlane(2, 1, 1, Stream: true)], -16, [new BlockScale(Float16Number, 32, 0)]),
        ["Q5_1"] = new(32, 24, NumberType.Integer(5, false), [new CodePlane(8, 4, 32), new CodePlane(4, 1, 1, Stream: true)], Scales: [new BlockScale(Float16Number, 32, 0)],
            Mins: [new BlockScale(Float16Number, 32, 2)], MinsAdded: true),
        ["IQ4_XS"] = new(256, 136, NumberType.Entries(4, [.. Iq4Grid.Select(v => (double)v)]), [new CodePlane(8, 4, 32)],
            Scales: [new BlockScale(Float16Number, 256, 0), new BlockScale(NumberType.Integer(6, false), 32, 0, Pieces(8, ib => [new(4 + ib / 2, 4 * (ib % 2), 4), new(2 + ib / 4, 2 * (ib % 4), 2)]), -32)]),
        ["TQ2_0"] = new(256, 66, NumberType.Integer(2, false), [new CodePlane(0, 2, 128)], -1, [new BlockScale(Float16Number, 256, 64)]),
        ["TQ1_0"] = new(256, 54, NumberType.Integer(2, false), [new CodePlane(0, 2, 1, Places: Fields(256, Tq1Digit), Radix: 3)], -1, [new BlockScale(Float16Number, 256, 52)]),
        ["IQ2_XS"] = new(256, 74, GridCodebook(9, 8, GgufGrids.IQ2XS, 0.25), [new CodePlane(2, 9, 1, Places: Fields(32, e => new(2 + 2 * e, 0)))],
            Scales: [new BlockScale(Float16Number, 256, 0), new BlockScale(NumberType.Integer(4, false), 16, 66, Bias: 0.5), new BlockScale(KSignVectors, 8, 0, Pieces(32, e => [new(3 + 2 * e, 1, 7)]))]),
        ["IQ2_S"] = new(256, 82, GridCodebook(10, 8, GgufGrids.IQ2S, 0.25), [new CodePlane(2, 8, 1), new CodePlane(66, 2, 4)],
            Scales: [new BlockScale(Float16Number, 256, 0), new BlockScale(NumberType.Integer(4, false), 16, 74, Bias: 0.5), new BlockScale(SignBits, 8, 34)]),
        ["IQ3_S"] = new(256, 110, GridCodebook(9, 4, GgufGrids.IQ3S, 1), [new CodePlane(2, 8, 1), new CodePlane(66, 1, 8)],
            Scales: [new BlockScale(Float16Number, 256, 0), new BlockScale(Odd(4), 32, 106), new BlockScale(SignBits, 8, 74)]),
        ["IQ1_S"] = new(256, 50, Iq1Codebook,
            [new CodePlane(2, 8, 1), new CodePlane(0, 3, 1, Places: Fields(32, i => new(34 + 2 * (i / 4) + 3 * (i % 4) / 8, 3 * (i % 4) % 8))),
                new CodePlane(0, 1, 1, Places: Fields(32, i => new(35 + 2 * (i / 4), 7)))],
            Scales: [new BlockScale(Float16Number, 256, 0), new BlockScale(Odd(3), 32, 0, Pieces(8, ib => [new(35 + 2 * ib, 4, 3)]))]),
        ["IQ1_M"] = new(256, 56, Iq1Codebook, [new CodePlane(0, 8, 1), new CodePlane(32, 4, 2)],
            Scales: [new BlockScale(Float16Number, 256, 0, [[new(49, 4, 4), new(51, 4, 4), new(53, 4, 4), new(55, 4, 4)]]),
                new BlockScale(Odd(3), 16, 0, Pieces(16, m => Bits(48 + 2 * (m / 4), 3 * (m % 4), 3)))]),
    };

    private static readonly Dictionary<string, GridBlock> GridDescribed = new(StringComparer.OrdinalIgnoreCase)
    {
        ["IQ2_XXS"] = new(256, 66, Half16, ScaleOffset: 0, SubValues: 32, Pieces(8, j => Bits(2 + 8 * j + 4, 28, 4)), SubScaleOffset: 0.5f, SubScaleFactor: 0.25f,
            EntryValues: 8, Pieces(32, e => [new(2 + 8 * (e / 4) + e % 4, 0, 8)]), [.. GgufGrids.IQ2XXS],
            SignValues: 8, Pieces(32, s => Bits(2 + 8 * (s / 4) + 4, 7 * (s % 4), 7)), KSignTable),
        ["IQ3_XXS"] = new(256, 98, Half16, ScaleOffset: 0, SubValues: 32, Pieces(8, j => Bits(66 + 4 * j, 28, 4)), SubScaleOffset: 0.5f, SubScaleFactor: 0.5f,
            EntryValues: 4, Pieces(64, e => [new(2 + e, 0, 8)]), [.. GgufGrids.IQ3XXS],
            SignValues: 8, Pieces(32, s => Bits(66 + 4 * (s / 4), 7 * (s % 4), 7)), KSignTable),
    };

    public static void RegisterAll()
    {
        foreach (int id in GgufTypes.Ids)
        {
            if (GgufTypes.Find(id) is not { BlockValues: > 1 } type)
            {
                continue;                                                   // F32, F16, BF16: the built-in types
            }

            if (ElementTypes.Default(type.Name.ToLowerInvariant()) is not null)
            {
                continue;                                                   // MXFP4: the library's (OCP's, in GGUF's block layout)
            }

            int typeId = id;
            ElementTypes.Register(new ElementType
            {
                Name = type.Name.ToLowerInvariant(),
                Summary = string.Create(CultureInfo.InvariantCulture, $"ggml {type.Name}: blocks of {type.BlockValues} values in {type.BlockBytes} bytes (GGUF type {id})"),
                BlockSize = type.BlockValues,
                BlockBits = type.BlockBytes * 8,
                IsFloat = type.Name.StartsWith("MXFP", StringComparison.OrdinalIgnoreCase),
                IsSigned = true,
                Scales = Scales.GetValueOrDefault(type.Name),
                Scaled = Described.GetValueOrDefault(type.Name),
                Super = SuperDescribed.GetValueOrDefault(type.Name),
                Grid = GridDescribed.GetValueOrDefault(type.Name),
                Description = Descriptions.GetValueOrDefault(type.Name),
                ToFloat32 = (source, destination) => GgufTypes.Get(typeId).Dequantize(source, destination),
            });
        }
    }
}
