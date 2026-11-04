// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;

namespace Idrak.Models;

// The low-bit ggml types (plan 36, C14): IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, TQ1_0, TQ2_0 and MXFP4,
// one block at a time as ggml's dequantize_row_* functions and gguf-py's dequantize_blocks give them: every product in
// their float32 order (a scale's own products first, then the grid value, then its sign), so the values are theirs bit
// for bit. The grids come from GgufGrids; the blocks are split across threads by DequantizeBuiltIn.
public sealed partial class GgufFile
{
    // MXFP4's E2M1 values doubled (ggml's kvalues_mxfp4); its block scale is ggml_e8m0_to_fp32_half.
    private static readonly sbyte[] Mxfp4Values = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

    // ±1 by a sign bit: multiplied, as ggml does, so a scale's NaN keeps its bits.
    private static float Sign(int bits, int j) => ((bits >> j) & 1) != 0 ? -1f : 1f;

    private static void DequantizeLowBit(int type, ReadOnlySpan<byte> x, Span<float> y)
    {
        switch (type)
        {
            case 16:                                                // IQ2_XXS: d, then 8 × (4 grid bytes, 4 × 7 sign bits + a 4-bit scale)
            {
                float d = Half(x);
                var grid = GgufGrids.IQ2XXS;
                for (int ib = 0; ib < 8; ib++)
                {
                    var q = x.Slice(2 + 8 * ib, 8);
                    uint signs = BinaryPrimitives.ReadUInt32LittleEndian(q[4..]);
                    float db = d * (0.5f + (signs >> 28)) * 0.25f;
                    for (int l = 0; l < 4; l++)
                    {
                        int g = q[l] * 8, s = GgufGrids.KSigns[(signs >> (7 * l)) & 127];
                        var o = y.Slice(32 * ib + 8 * l, 8);
                        for (int j = 0; j < 8; j++)
                        {
                            o[j] = db * grid[g + j] * Sign(s, j);
                        }
                    }
                }

                return;
            }

            case 17:                                                // IQ2_XS: d, 32 × (9-bit grid index, 7 sign bits), 16 4-bit scales
            {
                float d = Half(x);
                var grid = GgufGrids.IQ2XS;
                var scales = x.Slice(66, 8);
                for (int k = 0; k < 16; k++)
                {
                    float db = d * (0.5f + ((scales[k / 2] >> (4 * (k % 2))) & 0xF)) * 0.25f;
                    for (int t = 0; t < 2; t++)
                    {
                        int q = BinaryPrimitives.ReadUInt16LittleEndian(x[(2 + 2 * (2 * k + t))..]);
                        int g = (q & 511) * 8, s = GgufGrids.KSigns[q >> 9];
                        var o = y.Slice(16 * k + 8 * t, 8);
                        for (int j = 0; j < 8; j++)
                        {
                            o[j] = db * grid[g + j] * Sign(s, j);
                        }
                    }
                }

                return;
            }

            case 22:                                                // IQ2_S: d, qs[32], signs[32], qh[8] (2 high index bits each), scales[8]
            {
                float d = Half(x);
                var grid = GgufGrids.IQ2S;
                ReadOnlySpan<byte> qs = x.Slice(2, 32), signs = x.Slice(34, 32), qh = x.Slice(66, 8), scales = x.Slice(74, 8);
                for (int k = 0; k < 16; k++)
                {
                    float db = d * (0.5f + ((scales[k / 2] >> (4 * (k % 2))) & 0xF)) * 0.25f;
                    for (int t = 0; t < 2; t++)
                    {
                        int i = 2 * k + t;
                        int g = (qs[i] | (((qh[i / 4] >> (2 * (i % 4))) & 3) << 8)) * 8, s = signs[i];
                        var o = y.Slice(8 * i, 8);
                        for (int j = 0; j < 8; j++)
                        {
                            o[j] = db * grid[g + j] * Sign(s, j);
                        }
                    }
                }

                return;
            }

            case 18:                                                // IQ3_XXS: d, 64 grid bytes, 8 × (4 × 7 sign bits + a 4-bit scale)
            {
                float d = Half(x);
                var grid = GgufGrids.IQ3XXS;
                var qs = x.Slice(2, 64);
                for (int ib = 0; ib < 8; ib++)
                {
                    uint aux = BinaryPrimitives.ReadUInt32LittleEndian(x[(66 + 4 * ib)..]);
                    float db = d * (0.5f + (aux >> 28)) * 0.5f;
                    for (int l = 0; l < 4; l++)
                    {
                        int s = GgufGrids.KSigns[(aux >> (7 * l)) & 127];
                        int g0 = qs[8 * ib + 2 * l] * 4, g1 = qs[8 * ib + 2 * l + 1] * 4;
                        var o = y.Slice(32 * ib + 8 * l, 8);
                        for (int j = 0; j < 4; j++)
                        {
                            o[j] = db * grid[g0 + j] * Sign(s, j);
                            o[j + 4] = db * grid[g1 + j] * Sign(s, j + 4);
                        }
                    }
                }

                return;
            }

            case 21:                                                // IQ3_S: d, qs[64], qh[8] (a high index bit each), signs[32], scales[4]
            {
                float d = Half(x);
                var grid = GgufGrids.IQ3S;
                ReadOnlySpan<byte> qs = x.Slice(2, 64), qh = x.Slice(66, 8), signs = x.Slice(74, 32), scales = x.Slice(106, 4);
                for (int ib = 0; ib < 8; ib++)
                {
                    float db = d * (1 + 2 * ((scales[ib / 2] >> (4 * (ib % 2))) & 0xF));
                    for (int e = 8 * ib; e < 8 * ib + 8; e++)               // grid entries of 4 values
                    {
                        int g = (qs[e] | (((qh[e / 8] >> (e % 8)) & 1) << 8)) * 4;
                        for (int j = 0; j < 4; j++)
                        {
                            int v = 4 * e + j;
                            y[v] = db * grid[g + j] * Sign(signs[v / 8], v % 8);
                        }
                    }
                }

                return;
            }

            case 19:                                                // IQ1_S: d, qs[32], qh[8 × 16 bits: 4 × 3 index bits, a 3-bit scale, the delta's sign]
            {
                float d = Half(x);
                var grid = GgufGrids.IQ1S;
                var qs = x.Slice(2, 32);
                for (int ib = 0; ib < 8; ib++)
                {
                    int h = BinaryPrimitives.ReadUInt16LittleEndian(x[(34 + 2 * ib)..]);
                    float dl = d * (2 * ((h >> 12) & 7) + 1);
                    float delta = (h & 0x8000) == 0 ? 0.125f : -0.125f;
                    for (int l = 0; l < 4; l++)
                    {
                        int i = 4 * ib + l;
                        int g = (qs[i] | (((h >> (3 * l)) & 7) << 8)) * 8;
                        var o = y.Slice(8 * i, 8);
                        for (int j = 0; j < 8; j++)
                        {
                            o[j] = dl * (grid[g + j] + delta);
                        }
                    }
                }

                return;
            }

            case 29:                                                // IQ1_M: qs[32], qh[16] (3 index bits + the delta's sign a nibble), scales[4 × 16 bits]
            {
                var grid = GgufGrids.IQ1S;
                ReadOnlySpan<byte> qs = x[..32], qh = x.Slice(32, 16);
                Span<int> sc = stackalloc int[4];
                for (int w = 0; w < 4; w++)
                {
                    sc[w] = BinaryPrimitives.ReadUInt16LittleEndian(x[(48 + 2 * w)..]);
                }

                // The block's float16 scale is the top nibbles of the four scale words.
                ushort bits = (ushort)(((sc[0] & 0xF000) >> 12) | ((sc[1] & 0xF000) >> 8) | ((sc[2] & 0xF000) >> 4) | (sc[3] & 0xF000));
                float d = (float)BitConverter.UInt16BitsToHalf(bits);
                for (int i = 0; i < 32; i++)
                {
                    int m = i / 2;                                          // a 3-bit scale per 16 values
                    float dl = d * (2 * ((sc[m / 4] >> (3 * (m % 4))) & 7) + 1);
                    int e = qh[i / 2] >> (4 * (i % 2));
                    int g = (qs[i] | ((e & 7) << 8)) * 8;
                    float delta = (e & 8) == 0 ? 0.125f : -0.125f;
                    var o = y.Slice(8 * i, 8);
                    for (int j = 0; j < 8; j++)
                    {
                        o[j] = dl * (grid[g + j] + delta);
                    }
                }

                return;
            }

            case 34:                                                // TQ1_0: qs[48] and qh[4] of base-3 digits (5 and 4 a byte), d
            {
                float d = Half(x[52..]);
                ReadOnlySpan<byte> powers = [1, 3, 9, 27, 81];
                for (int p = 0; p < 5; p++)
                {
                    for (int m = 0; m < 32; m++)
                    {
                        y[32 * p + m] = d * Trit((byte)(x[m] * powers[p]));
                    }

                    for (int m = 0; m < 16; m++)
                    {
                        y[160 + 16 * p + m] = d * Trit((byte)(x[32 + m] * powers[p]));
                    }
                }

                for (int p = 0; p < 4; p++)
                {
                    for (int m = 0; m < 4; m++)
                    {
                        y[240 + 4 * p + m] = d * Trit((byte)(x[48 + m] * powers[p]));
                    }
                }

                return;
            }

            case 35:                                                // TQ2_0: qs[64] of 2-bit values (-1, 0, 1), d
            {
                float d = Half(x[64..]);
                for (int c = 0; c < 2; c++)
                {
                    for (int s = 0; s < 4; s++)
                    {
                        for (int m = 0; m < 32; m++)
                        {
                            y[128 * c + 32 * s + m] = d * (((x[32 * c + m] >> (2 * s)) & 3) - 1);
                        }
                    }
                }

                return;
            }

            case 39:                                                // MXFP4: an E8M0 scale, 16 bytes of E2M1 nibbles
            {
                int e = x[0];
                float d = BitConverter.UInt32BitsToSingle(e < 2 ? 0x00200000u << e : (uint)(e - 1) << 23);
                for (int m = 0; m < 16; m++)
                {
                    y[m] = d * Mxfp4Values[x[1 + m] & 0xF];
                    y[16 + m] = d * Mxfp4Values[x[1 + m] >> 4];
                }

                return;
            }

            default:
                throw new NotSupportedException($"ggml type {type} has no built-in dequantizer.");
        }
    }

    // A TQ1_0 digit: the top of q · 3 (q the byte the digit was scaled into), less 1.
    private static int Trit(byte q) => ((q * 3) >> 8) - 1;
}
