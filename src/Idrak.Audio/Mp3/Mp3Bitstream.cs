// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using static Idrak.Audio.Mp3EncoderTables;

namespace Idrak.Audio;

/// <summary>Writes bits most significant first into a span (zeros pad the last byte).</summary>
internal ref struct Mp3BitWriter(Span<byte> buffer)
{
    private readonly Span<byte> _buffer = buffer;
    private ulong _cache;
    private int _cached, _at, _written;

    /// <summary>The bits written so far.</summary>
    public readonly int Bits => _written;

    /// <summary>Writes the low <paramref name="count"/> (0 to 32) bits of <paramref name="value"/>.</summary>
    public void Write(uint value, int count)
    {
        if (count == 0)
        {
            return;
        }

        _cache = (_cache << count) | (value & (uint)((1UL << count) - 1));
        _cached += count;
        _written += count;
        while (_cached >= 8)
        {
            _cached -= 8;
            _buffer[_at++] = (byte)(_cache >> _cached);
        }
    }

    /// <summary>Writes the bits still held, padded to a byte with zeros.</summary>
    public void Flush()
    {
        if (_cached > 0)
        {
            _buffer[_at++] = (byte)(_cache << (8 - _cached));
            _cached = 0;
        }
    }

    /// <summary>Appends <paramref name="count"/> bits of <paramref name="source"/> (most significant first) to a zeroed <paramref name="target"/> from bit <paramref name="bit"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Append(Span<byte> target, long bit, ReadOnlySpan<byte> source, int count)
    {
        int bytes = (count + 7) >> 3, at = (int)(bit >> 3), shift = (int)(bit & 7);
        if (shift == 0)
        {
            source[..bytes].CopyTo(target[at..]);
            if ((count & 7) != 0)
            {
                target[at + bytes - 1] &= (byte)(0xFF << (8 - (count & 7)));
            }

            return;
        }

        for (int i = 0; i < bytes; i++)
        {
            int b = source[i];
            if (i == bytes - 1 && (count & 7) != 0)
            {
                b &= 0xFF << (8 - (count & 7));
            }

            target[at + i] |= (byte)(b >> shift);
            if (((count - 8 * i) + shift) > 8)
            {
                target[at + i + 1] |= (byte)(b << (8 - shift));
            }
        }
    }
}

/// <summary>
/// The Layer III Huffman tables for encoding (11172-3 Annex B, from <see cref="Mp3EncoderTables"/>): per held table the
/// code and its length by x * 16 + y (x and y clamped to 15 for the escape tables), and the bits a pair costs with its
/// sign bits.
/// </summary>
internal static class Mp3Huffman
{
    /// <summary>The held tables (table numbers 1, 2, 3, 5-13, 15, 16, 24).</summary>
    public const int Held = 15;

    /// <summary>The held table of an escape group: 16 for tables 16-23, 24 for 24-31 (indices into the held tables).</summary>
    public const int Escape16 = 13, Escape24 = 14;

    // Built by a field initializer, not a static constructor: the class stays beforefieldinit, so loops compiled
    // optimized at once read these arrays without a class-initialization check.
    private static readonly Built Tables = Build();

    /// <summary>The held table index of a table number (-1 for 0, 4 and 14).</summary>
    public static readonly sbyte[] HeldIndex = Tables.HeldIndex;

    /// <summary>The values a held table holds per side (2 to 16).</summary>
    public static readonly int[] XLen = Tables.XLen;

    /// <summary>Codes by held table * 256 + x * 16 + y.</summary>
    public static readonly uint[] Codes = Tables.Codes;

    /// <summary>The codes' lengths, by the same index.</summary>
    public static readonly byte[] CodeLengths = Tables.CodeLengths;

    /// <summary>A pair's bits: its code and a sign bit per value that is not 0 (escape tables: not their linbits).</summary>
    public static readonly byte[] PairBits = Tables.PairBits;

    /// <summary>The linbits of table numbers 0-31.</summary>
    public static readonly byte[] Linbits = Mp3Tables.Linbits;

    private sealed record Built(sbyte[] HeldIndex, int[] XLen, uint[] Codes, byte[] CodeLengths, byte[] PairBits);

    private static Built Build()
    {
        var held = new sbyte[32];
        held.AsSpan().Fill(-1);
        var xlen = new int[Held];
        var codes = new uint[Held * 256];
        var lengths = new byte[Held * 256];
        var pairBits = new byte[Held * 256];
        int[] sizes = [2, 3, 3, 4, 4, 6, 6, 6, 8, 8, 8, 16, 16, 16, 16];
        for (int h = 0; h < Held; h++)
        {
            int table = HuffmanTables[h], n = sizes[h];
            xlen[h] = n;
            if (table == 16 || table == 24)
            {
                for (int t = table; t < table + 8; t++)
                {
                    held[t] = (sbyte)h;
                }
            }
            else
            {
                held[table] = (sbyte)h;
            }

            for (int x = 0; x < n; x++)
            {
                for (int y = 0; y < n; y++)
                {
                    int at = HuffmanOffsets[h] + x * n + y, i = h * 256 + x * 16 + y;
                    codes[i] = (uint)HuffmanCodes[at];
                    lengths[i] = HuffmanLengths[at];
                    pairBits[i] = (byte)(HuffmanLengths[at] + (x != 0 ? 1 : 0) + (y != 0 ? 1 : 0));
                }
            }
        }

        return new Built(held, xlen, codes, lengths, pairBits);
    }
}
