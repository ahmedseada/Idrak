// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;

namespace Idrak.Adad.Types;

/// <summary>
/// The super-block types Adad describes, as data (their descriptions read the way ggml's dequantize_row_q*_K reads them),
/// and their encoders from float32. A type is found here by its name; every kernel and decoder reads it through its
/// description alone.
/// </summary>
public static class SuperBlocks
{
    private static readonly FloatBits Half16 = new(5, 10);

    private static ScalePiece[][] Pieces(int count, Func<int, ScalePiece[]> of) => [.. Enumerable.Range(0, count).Select(of)];

    // Q4_K's twelve bytes from byte 4: sub-blocks 0-3 a 6-bit scale and minimum each; 4-7 their low four bits in bytes
    // 12-15 and their high two in the top bits of bytes 4-11.
    private static ScalePiece[] KScale(int j) => j < 4 ? [new(4 + j, 0, 6)] : [new(8 + j, 0, 4), new(j, 6, 2)];

    private static ScalePiece[] KMin(int j) => j < 4 ? [new(8 + j, 0, 6)] : [new(8 + j, 4, 4), new(4 + j, 6, 2)];

    /// <summary>ggml's Q4_K: 256 values in 144 bytes; d, dmin (float16), 8 sub-blocks of 32 with 6-bit scales and minimums, 4-bit codes in groups of 64.</summary>
    public static readonly SuperBlock Q4K = new(256, 144, [new CodePlane(16, 4, 64)], CodeBias: 0, Half16, ScaleOffset: 0, MinOffset: 2, SubValues: 32,
        Pieces(8, KScale), SubScaleBias: 0, SubScalesSigned: false, Pieces(8, KMin));

    /// <summary>The descriptions by name (a type is looked up by name; kernels never are).</summary>
    public static IReadOnlyDictionary<string, SuperBlock> ByName { get; } = new Dictionary<string, SuperBlock>(StringComparer.OrdinalIgnoreCase) { ["q4_k"] = Q4K };

    // Where a super-block's codes go (the inverse of SuperBlock.Decode's read): each plane's byte and shift for every value
    // (CodePlane.Place), its lowest bit in the code and its mask, made once a description, so a code is placed with no
    // division and no call through the planes' list a value.
    private sealed class CodePlaces
    {
        private readonly int[][] _byte;
        private readonly int[][] _shift;
        private readonly int[] _low;
        private readonly int[] _mask;

        public CodePlaces(SuperBlock s)
        {
            int planes = s.Planes.Count;
            (_byte, _shift, _low, _mask) = (new int[planes][], new int[planes][], new int[planes], new int[planes]);
            for (int p = 0, low = 0; p < planes; low += s.Planes[p].Bits, p++)
            {
                var plane = s.Planes[p];
                _byte[p] = [.. Enumerable.Range(0, s.Values).Select(v => plane.Place(v).Byte)];
                _shift[p] = [.. Enumerable.Range(0, s.Values).Select(v => plane.Place(v).Shift)];
                (_low[p], _mask[p]) = (low, (1 << plane.Bits) - 1);
            }
        }

        // The block's codes (one a value) into its planes' fields, the block cleared before.
        public void Put(Span<byte> block, ReadOnlySpan<int> codes)
        {
            for (int p = 0; p < _byte.Length; p++)
            {
                int[] at = _byte[p], shift = _shift[p];
                int low = _low[p], mask = _mask[p];
                for (int v = 0; v < at.Length; v++)
                {
                    block[at[v]] |= (byte)((codes[v] >> low & mask) << shift[v]);
                }
            }
        }
    }

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<SuperBlock, CodePlaces> PlacesMade = new();

    private static CodePlaces PlacesOf(SuperBlock s) => PlacesMade.GetOrAdd(s, static s => new CodePlaces(s));

    /// <summary>
    /// Q4_K from float32 (a simple encoder, not ggml's searching quantize_row_q4_K_ref): each sub-block of 32 its range
    /// [min(x, 0), max(x, 0)], scale (hi − lo) / 15 and minimum −lo; d and dmin the largest of each over 63 (float16); the
    /// 6-bit scales and minimums rounded from them, each code round((x + m) / s) within 0..15. The codes are computed one at a
    /// time and placed by a table made once (the bytes the first encoder gave, faster).
    /// </summary>
    public static void EncodeQ4K(ReadOnlySpan<float> source, Span<byte> destination)
    {
        var places = PlacesOf(Q4K);
        Span<float> scale = stackalloc float[8], min = stackalloc float[8];
        Span<int> sc = stackalloc int[8], mn = stackalloc int[8], codes = stackalloc int[256];
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
            BinaryPrimitives.WriteHalfLittleEndian(block, dh);
            BinaryPrimitives.WriteHalfLittleEndian(block[2..], mh);
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

            for (int j = 0; j < 8; j++)
            {
                float step = d * sc[j], offset = dmin * mn[j];
                var to = codes.Slice(j * 32, 32);
                if (step > 0f)
                {
                    var from = x.Slice(j * 32, 32);
                    for (int i = 0; i < 32; i++)
                    {
                        to[i] = Math.Clamp((int)MathF.Round((from[i] + offset) / step), 0, 15);
                    }
                }
                else
                {
                    to.Clear();
                }
            }

            places.Put(block, codes);
        }
    }
}
