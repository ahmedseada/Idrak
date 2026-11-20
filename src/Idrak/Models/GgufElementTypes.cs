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

    // IQ4_NL from float32 (a simple encoder, not ggml's searching quantize_row_iq4_nl_ref): per block of 32, d = the value
    // of largest |x| over the grid's first entry (−127, float16), each code the grid entry nearest x / d.
    private static void EncodeIQ4_NL(ReadOnlySpan<float> source, Span<byte> destination)
    {
        for (int b = 0; b < source.Length / 32; b++)
        {
            var values = source.Slice(b * 32, 32);
            var block = destination.Slice(b * 18, 18);
            float amax = 0f, max = 0f;
            foreach (float v in values)
            {
                if (amax < MathF.Abs(v))
                {
                    amax = MathF.Abs(v);
                    max = v;
                }
            }

            var dh = (Half)(max / Iq4Grid[0]);
            System.Buffers.Binary.BinaryPrimitives.WriteHalfLittleEndian(block, dh);
            float d = (float)dh, id = d != 0f ? 1f / d : 0f;
            for (int j = 0; j < 16; j++)
            {
                block[2 + j] = (byte)(Nearest(values[j] * id) | Nearest(values[16 + j] * id) << 4);
            }
        }

        static int Nearest(float t)
        {
            int best = 0;
            for (int i = 1; i < Iq4Grid.Length; i++)
            {
                if (MathF.Abs(Iq4Grid[i] - t) < MathF.Abs(Iq4Grid[best] - t))
                {
                    best = i;
                }
            }

            return best;
        }
    }

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

    private static readonly Dictionary<string, GridBlock> GridDescribed = new(StringComparer.OrdinalIgnoreCase)
    {
        ["IQ2_XXS"] = new(256, 66, Half16, ScaleOffset: 0, SubValues: 32, Pieces(8, j => Bits(2 + 8 * j + 4, 28, 4)), SubScaleOffset: 0.5f, SubScaleFactor: 0.25f,
            EntryValues: 8, Pieces(32, e => [new(2 + 8 * (e / 4) + e % 4, 0, 8)]), [.. GgufGrids.IQ2XXS],
            SignValues: 8, Pieces(32, s => Bits(2 + 8 * (s / 4) + 4, 7 * (s % 4), 7)), KSignTable),
        ["IQ3_XXS"] = new(256, 98, Half16, ScaleOffset: 0, SubValues: 32, Pieces(8, j => Bits(66 + 4 * j, 28, 4)), SubScaleOffset: 0.5f, SubScaleFactor: 0.5f,
            EntryValues: 4, Pieces(64, e => [new(2 + e, 0, 8)]), [.. GgufGrids.IQ3XXS],
            SignValues: 8, Pieces(32, s => Bits(66 + 4 * (s / 4), 7 * (s % 4), 7)), KSignTable),
    };

    // A grid format from float32 (a simple encoder, not ggml's searching quantizers): each sub-block's scale from its largest
    // |x| over the grid's largest magnitude, d the largest over (15 + offset) · factor (float16), each sub-block's 4-bit
    // scale rounded; each group's signs (made even, as the sign table holds them, by flipping the smallest |x|'s), and each
    // entry the grid's nearest to the magnitudes over the scale (each magnitude to its nearest level, the pattern of levels
    // to its nearest entry by a table made once).
    private static void EncodeGrid(GridBlock g, ReadOnlySpan<float> source, Span<byte> destination)
    {
        var levels = g.Grid.Distinct().Order().ToArray();
        var nearest = NearestEntries(g, levels);
        int subs = g.Values / g.SubValues;
        Span<float> scale = stackalloc float[subs];
        Span<int> s = stackalloc int[subs];
        for (int b = 0; b < source.Length / g.Values; b++)
        {
            var x = source.Slice(b * g.Values, g.Values);
            var block = destination.Slice(b * g.Bytes, g.Bytes);
            block.Clear();
            float top = levels[^1], largest = 0f;
            for (int j = 0; j < subs; j++)
            {
                float amax = 0f;
                foreach (float v in x.Slice(j * g.SubValues, g.SubValues))
                {
                    amax = MathF.Max(amax, MathF.Abs(v));
                }

                scale[j] = amax / top;
                largest = MathF.Max(largest, scale[j]);
            }

            var dh = (Half)(largest / ((15 + g.SubScaleOffset) * g.SubScaleFactor));
            System.Buffers.Binary.BinaryPrimitives.WriteHalfLittleEndian(block[g.ScaleOffset..], dh);
            float d = (float)dh;
            for (int j = 0; j < subs; j++)
            {
                s[j] = d > 0f ? Math.Clamp((int)MathF.Round(scale[j] / (d * g.SubScaleFactor) - g.SubScaleOffset), 0, 15) : 0;
                Put(block, g.SubScales[j], s[j]);
            }

            for (int group = 0; group < g.Values / g.SignValues; group++)
            {
                int mask = 0, smallest = 0;
                var values = x.Slice(group * g.SignValues, g.SignValues);
                for (int i = 0; i < g.SignValues; i++)
                {
                    if (values[i] < 0)
                    {
                        mask |= 1 << i;
                    }

                    if (MathF.Abs(values[i]) < MathF.Abs(values[smallest]))
                    {
                        smallest = i;
                    }
                }

                if (System.Numerics.BitOperations.PopCount((uint)mask) % 2 != 0)
                {
                    mask ^= 1 << smallest;
                }

                Put(block, g.Signs[group], Array.IndexOf(KSignTable, mask) is var at and >= 0 ? at : 0);
            }

            for (int e = 0; e < g.Values / g.EntryValues; e++)
            {
                float step = d * (s[e * g.EntryValues / g.SubValues] + g.SubScaleOffset) * g.SubScaleFactor;
                int pattern = 0;
                for (int i = 0; i < g.EntryValues; i++)
                {
                    float t = step > 0f ? MathF.Abs(x[e * g.EntryValues + i]) / step : 0f;
                    int level = 0;
                    for (int l = 1; l < levels.Length; l++)
                    {
                        if (MathF.Abs(levels[l] - t) < MathF.Abs(levels[level] - t))
                        {
                            level = l;
                        }
                    }

                    pattern = pattern * levels.Length + level;
                }

                Put(block, g.Entries[e], nearest[pattern]);
            }
        }

        static void Put(Span<byte> block, IReadOnlyList<ScalePiece> pieces, int value)
        {
            for (int i = 0; i < pieces.Count; i++)
            {
                var piece = pieces[i];
                block[piece.Offset] |= (byte)((value & ((1 << piece.Bits) - 1)) << piece.Shift);
                value >>= piece.Bits;
            }
        }
    }

    // Each pattern of levels (one a value of an entry) to the grid's nearest entry, made once a description.
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<GridBlock, int[]> NearestTables = new();

    private static int[] NearestEntries(GridBlock g, int[] levels) => NearestTables.GetOrAdd(g, _ =>
    {
        int patterns = (int)Math.Pow(levels.Length, g.EntryValues), entries = g.Grid.Count / g.EntryValues;
        var table = new int[patterns];
        Span<int> wanted = stackalloc int[g.EntryValues];
        for (int p = 0; p < patterns; p++)
        {
            for (int i = g.EntryValues - 1, rest = p; i >= 0; i--, rest /= levels.Length)
            {
                wanted[i] = levels[rest % levels.Length];
            }

            int best = 0;
            long bestDistance = long.MaxValue;
            for (int e = 0; e < entries; e++)
            {
                long distance = 0;
                for (int i = 0; i < g.EntryValues; i++)
                {
                    long diff = g.Grid[e * g.EntryValues + i] - wanted[i];
                    distance += diff * diff;
                }

                if (distance < bestDistance)
                {
                    (best, bestDistance) = (e, distance);
                }
            }

            table[p] = best;
        }

        return table;
    });

    // A code into its planes' fields of a block (the inverse of SuperBlock.Decode's read).
    private static void PutCode(SuperBlock s, Span<byte> block, int v, int code)
    {
        // An indexed loop: a foreach over the list would allocate an enumerator a value.
        int low = 0;
        var planes = s.Planes;
        for (int i = 0; i < planes.Count; i++)
        {
            var plane = planes[i];
            var (at, shift) = plane.Place(v);
            block[at] |= (byte)((code >> low & ((1 << plane.Bits) - 1)) << shift);
            low += plane.Bits;
        }
    }

    // Q4_K from float32 (a simple encoder, not ggml's searching quantize_row_q4_K_ref): each sub-block of 32 its range
    // [min(x, 0), max(x, 0)], scale (hi − lo) / 15 and minimum −lo; d and dmin the largest of each over 63 (float16); the
    // 6-bit scales and minimums rounded from them, each code round((x + m) / s) within 0..15.
    private static void EncodeQ4_K(ReadOnlySpan<float> source, Span<byte> destination)
    {
        var s = SuperDescribed["Q4_K"];
        Span<float> scale = stackalloc float[8], min = stackalloc float[8];
        Span<int> sc = stackalloc int[8], mn = stackalloc int[8];
        for (int b = 0; b < source.Length / 256; b++)
        {
            var x = source.Slice(b * 256, 256);
            var block = destination.Slice(b * 144, 144);
            block.Clear();
            for (int j = 0; j < 8; j++)
            {
                float lo = 0f, hi = 0f;
                foreach (float v in x.Slice(j * 32, 32))
                {
                    lo = MathF.Min(lo, v);
                    hi = MathF.Max(hi, v);
                }

                scale[j] = (hi - lo) / 15f;
                min[j] = -lo;
            }

            float maxScale = 0f, maxMin = 0f;
            for (int j = 0; j < 8; j++)
            {
                maxScale = MathF.Max(maxScale, scale[j]);
                maxMin = MathF.Max(maxMin, min[j]);
            }

            var dh = (Half)(maxScale / 63f);
            var mh = (Half)(maxMin / 63f);
            System.Buffers.Binary.BinaryPrimitives.WriteHalfLittleEndian(block, dh);
            System.Buffers.Binary.BinaryPrimitives.WriteHalfLittleEndian(block[2..], mh);
            float d = (float)dh, dmin = (float)mh;
            for (int j = 0; j < 8; j++)
            {
                sc[j] = d > 0f ? Math.Clamp((int)MathF.Round(scale[j] / d), 0, 63) : 0;
                mn[j] = dmin > 0f ? Math.Clamp((int)MathF.Round(min[j] / dmin), 0, 63) : 0;
            }

            for (int j = 0; j < 4; j++)
            {
                block[4 + j] = (byte)(sc[j] | (sc[j + 4] >> 4) << 6);
                block[8 + j] = (byte)(mn[j] | (mn[j + 4] >> 4) << 6);
                block[12 + j] = (byte)(sc[j + 4] & 0xF | (mn[j + 4] & 0xF) << 4);
            }

            for (int v = 0; v < 256; v++)
            {
                float step = d * sc[v / 32], offset = dmin * mn[v / 32];
                int code = step > 0f ? Math.Clamp((int)MathF.Round((x[v] + offset) / step), 0, 15) : 0;
                PutCode(s, block, v, code);
            }
        }
    }

    // Q6_K from float32 (a simple encoder, not ggml's searching quantize_row_q6_K_ref): each sub-block of 16 its scale
    // max |x| / 31; d the largest over 127 (float16); the int8 scales rounded from it, each code round(x / s) + 32 within 0..63.
    private static void EncodeQ6_K(ReadOnlySpan<float> source, Span<byte> destination)
    {
        var s = SuperDescribed["Q6_K"];
        Span<float> scale = stackalloc float[16];
        Span<int> sc = stackalloc int[16];
        for (int b = 0; b < source.Length / 256; b++)
        {
            var x = source.Slice(b * 256, 256);
            var block = destination.Slice(b * 210, 210);
            block.Clear();
            float maxScale = 0f;
            for (int j = 0; j < 16; j++)
            {
                float amax = 0f;
                foreach (float v in x.Slice(j * 16, 16))
                {
                    amax = MathF.Max(amax, MathF.Abs(v));
                }

                scale[j] = amax / 31f;
                maxScale = MathF.Max(maxScale, scale[j]);
            }

            var dh = (Half)(maxScale / 127f);
            System.Buffers.Binary.BinaryPrimitives.WriteHalfLittleEndian(block[208..], dh);
            float d = (float)dh;
            for (int j = 0; j < 16; j++)
            {
                sc[j] = d > 0f ? Math.Clamp((int)MathF.Round(scale[j] / d), -128, 127) : 0;
                block[192 + j] = (byte)(sbyte)sc[j];
            }

            for (int v = 0; v < 256; v++)
            {
                float step = d * sc[v / 16];
                int code = step != 0f ? Math.Clamp((int)MathF.Round(x[v] / step), -32, 31) + 32 : 32;
                PutCode(s, block, v, code);
            }
        }
    }

    // The formats stored from float32 values here, as ggml's reference quantizers store them (quantize_row_*_ref), and the
    // k-quants by simple encoders of their own (above).
    private static readonly Dictionary<string, ElementEncoder> Encoders = new(StringComparer.OrdinalIgnoreCase)
    {
        ["Q8_0"] = EncodeQ8_0,
        ["Q4_0"] = EncodeQ4_0,
        ["IQ4_NL"] = EncodeIQ4_NL,
        ["Q4_K"] = EncodeQ4_K,
        ["Q6_K"] = EncodeQ6_K,
        ["IQ2_XXS"] = (source, destination) => EncodeGrid(GridDescribed["IQ2_XXS"], source, destination),
        ["IQ3_XXS"] = (source, destination) => EncodeGrid(GridDescribed["IQ3_XXS"], source, destination),
    };

    // ggml's quantize_row_q4_0_ref: per block of 32, the value of largest |x| (the first such) over -8 is d, stored as
    // float16; each code min(15, trunc(x / d + 8.5)) with the reciprocal of the unrounded d, value j and j + 16 sharing
    // byte j (low and high half); a block of zeros has d = 0 and every code 8.
    private static void EncodeQ4_0(ReadOnlySpan<float> source, Span<byte> destination)
    {
        for (int b = 0; b < source.Length / 32; b++)
        {
            var values = source.Slice(b * 32, 32);
            var block = destination.Slice(b * 18, 18);
            float amax = 0f, max = 0f;
            foreach (float v in values)
            {
                if (amax < MathF.Abs(v))
                {
                    amax = MathF.Abs(v);
                    max = v;
                }
            }

            float d = max / -8f, id = d != 0f ? 1f / d : 0f;
            System.Buffers.Binary.BinaryPrimitives.WriteHalfLittleEndian(block, (Half)d);
            for (int j = 0; j < 16; j++)
            {
                int low = Math.Min(15, (int)(sbyte)(values[j] * id + 8.5f)), high = Math.Min(15, (int)(sbyte)(values[16 + j] * id + 8.5f));
                block[2 + j] = (byte)(low | high << 4);
            }
        }
    }

    // ggml's quantize_row_q8_0_ref: per block of 32, d = max |x| / 127 stored as float16 (to nearest, ties to even), and
    // each code round(x / d) (half away from zero) with the reciprocal of the unrounded d; a block of zeros has d = 0.
    private static void EncodeQ8_0(ReadOnlySpan<float> source, Span<byte> destination)
    {
        for (int b = 0; b < source.Length / 32; b++)
        {
            var values = source.Slice(b * 32, 32);
            var block = destination.Slice(b * 34, 34);
            float amax = 0f;
            foreach (float v in values)
            {
                amax = MathF.Max(amax, MathF.Abs(v));
            }

            float d = amax / 127f, id = d != 0f ? 1f / d : 0f;
            System.Buffers.Binary.BinaryPrimitives.WriteHalfLittleEndian(block, (Half)d);
            for (int j = 0; j < 32; j++)
            {
                block[2 + j] = (byte)(sbyte)MathF.Round(values[j] * id, MidpointRounding.AwayFromZero);
            }
        }
    }

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
                ToFloat32 = (source, destination) => GgufTypes.Get(typeId).Dequantize(source, destination),
                FromFloat32 = Encoders.GetValueOrDefault(type.Name),
            });
        }
    }
}
