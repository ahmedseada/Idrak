// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Models;

/// <summary>
/// The lookup grids of ggml's IQ types (GgufGrids.g.cs holds them as gguf-py packs them), decoded once on first use into
/// one value per byte: entry e's values at [e · width, (e + 1) · width), and the number types of ggml's grids the IQ types build
/// their descriptions from (their codebooks, sign vectors and odd sub-scales).
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

    /// <summary>IQ4_NL's and IQ4_XS's non-linear grid (ggml's kvalues_iq4nl): the number each 4-bit code stands for, before the scales.</summary>
    public static NumberType Iq4Numbers { get; } = NumberType.Entries(4, [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]);

    /// <summary>The sign vectors a byte of sign bits gives (bit i set: value i negative), 8 values an entry.</summary>
    public static NumberType SignBits { get; } = SignCodebook(8, s => s);

    /// <summary>The sign vectors of ggml's 7-bit sign indices (KSigns' masks: seven signs and their parity), 8 values an entry.</summary>
    public static NumberType KSignVectors => s_ksigns ??= SignCodebook(7, s => KSigns[s]);

    private static NumberType? s_ksigns;

    /// <summary>IQ1_S's and IQ1_M's codebook: a 12-bit index, the grid's 8 values of its low 11 bits plus the delta (±1/8, its sign the top bit).</summary>
    public static NumberType Iq1Codebook => s_iq1 ??= NumberType.Codebook(12, 8,
        [.. Enumerable.Range(0, 1 << 12).SelectMany(c => Enumerable.Range(0, 8).Select(j => IQ1S[(c & 2047) * 8 + j] + (c >> 11 != 0 ? -0.125 : 0.125)))]);

    private static NumberType? s_iq1;

    /// <summary>A grid's entries as a codebook of <paramref name="index"/>-bit indices, <paramref name="width"/> values an entry, each magnitude times <paramref name="factor"/>.</summary>
    public static NumberType Codebook(int index, int width, sbyte[] grid, double factor) => NumberType.Codebook(index, width, [.. grid.Select(v => v * factor)]);

    /// <summary>1 + 2s for an s of <paramref name="bits"/> bits (ggml's odd sub-scales).</summary>
    public static NumberType Odd(int bits) => NumberType.Entries(bits, [.. Enumerable.Range(0, 1 << bits).Select(s => 1d + 2 * s)]);

    // The sign vectors of `bits`-bit indices, mask(s) bit i set: value i negative.
    private static NumberType SignCodebook(int bits, Func<int, int> mask) =>
        NumberType.Codebook(bits, 8, [.. Enumerable.Range(0, 1 << bits).SelectMany(s => Enumerable.Range(0, 8).Select(i => (mask(s) >> i & 1) != 0 ? -1d : 1d))]);

    private static int Digit(char c) => c <= '9' ? c - '0' : (c | 0x20) - 'a' + 10;
}
