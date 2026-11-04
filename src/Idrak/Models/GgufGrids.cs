// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Models;

/// <summary>
/// The lookup grids of ggml's IQ types (GgufGrids.g.cs holds them as gguf-py packs them), decoded once on first use into
/// one value per byte: entry e's values at [e · width, (e + 1) · width). The IQ dequantizers read a block's values from
/// these as ggml reads its iq*_grid tables.
/// </summary>
internal static partial class GgufGrids
{
    private static sbyte[]? s_iq1s, s_iq2xxs, s_iq2xs, s_iq2s, s_iq3xxs, s_iq3s;

    /// <summary>IQ1_S and IQ1_M: 2,048 entries of 8 values in {-1, 0, 1}.</summary>
    public static sbyte[] IQ1S => s_iq1s ??= Decode(IQ1SHex, IQ1SMap, 2048, 8);

    /// <summary>IQ2_XXS: 256 entries of 8 values in {8, 25, 43}.</summary>
    public static sbyte[] IQ2XXS => s_iq2xxs ??= Decode(IQ2XXSHex, IQ2XXSMap, 256, 8);

    /// <summary>IQ2_XS: 512 entries of 8 values in {8, 25, 43}.</summary>
    public static sbyte[] IQ2XS => s_iq2xs ??= Decode(IQ2XSHex, IQ2XSMap, 512, 8);

    /// <summary>IQ2_S: 1,024 entries of 8 values in {8, 25, 43}.</summary>
    public static sbyte[] IQ2S => s_iq2s ??= Decode(IQ2SHex, IQ2SMap, 1024, 8);

    /// <summary>IQ3_XXS: 256 entries of 4 values in {4, 12, …, 62}.</summary>
    public static sbyte[] IQ3XXS => s_iq3xxs ??= Decode(IQ3XXSHex, IQ3XXSMap, 256, 4);

    /// <summary>IQ3_S: 512 entries of 4 values in {1, 3, …, 15}.</summary>
    public static sbyte[] IQ3S => s_iq3s ??= Decode(IQ3SHex, IQ3SMap, 512, 4);

    // gguf-py's init_grid: two hexadecimal digits a byte (high digit first); each byte holds 8 / bits indices of `bits`
    // bits (bits = ⌈log2 |map|⌉), the first in the low bits, each `8 / (8 / bits)` bits apart; an index picks a map value.
    private static sbyte[] Decode(string hex, sbyte[] map, int entries, int width)
    {
        int bits = map.Length <= 2 ? 1 : map.Length <= 4 ? 2 : map.Length <= 8 ? 3 : 4;
        int perByte = 8 / bits, step = 8 / perByte, mask = (1 << bits) - 1;
        var values = new sbyte[entries * width];
        if (hex.Length * perByte / 2 != values.Length)
        {
            throw new InvalidOperationException($"A grid of {hex.Length / 2} bytes does not hold {entries} entries of {width} values.");
        }

        int at = 0;
        for (int i = 0; i < hex.Length; i += 2)
        {
            int b = (Digit(hex[i]) << 4) | Digit(hex[i + 1]);
            for (int k = 0; k < perByte; k++)
            {
                values[at++] = map[(b >> (k * step)) & mask];
            }
        }

        return values;
    }

    private static int Digit(char c) => c <= '9' ? c - '0' : (c | 0x20) - 'a' + 10;
}
