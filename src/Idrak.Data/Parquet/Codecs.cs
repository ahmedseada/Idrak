// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.IO.Compression;
using Idrak.Data.Abstractions;

namespace Idrak.Data.Parquet;

/// <summary>The built-in decompressors of Parquet pages (Snappy, Gzip, Brotli and LZ4 raw), registered in <see cref="ParquetCodecs"/>.</summary>
internal static class Codecs
{
    private delegate byte[] Decompressor(ReadOnlySpan<byte> input, int uncompressedSize);

    // Called by LibraryRegistrations (once, before ParquetCodecs is first used).
    public static void RegisterAll()
    {
        ParquetCodecs.Register(new BuiltIn(0, "Uncompressed", (input, _) => input.ToArray()));
        ParquetCodecs.Register(new BuiltIn(1, "Snappy", Snappy));
        ParquetCodecs.Register(new BuiltIn(2, "Gzip", Gzip));
        ParquetCodecs.Register(new BuiltIn(4, "Brotli", Brotli));
        ParquetCodecs.Register(new BuiltIn(7, "Lz4Raw", Lz4Block));
    }

    public static byte[] Gzip(ReadOnlySpan<byte> input, int uncompressedSize)
    {
        using var gzip = new GZipStream(new MemoryStream(input.ToArray()), CompressionMode.Decompress);
        var output = new byte[uncompressedSize];
        gzip.ReadExactly(output);
        return output;
    }

    public static byte[] Brotli(ReadOnlySpan<byte> input, int uncompressedSize)
    {
        var brotli = new byte[uncompressedSize];
        if (!BrotliDecoder.TryDecompress(input, brotli, out int written) || written != uncompressedSize)
        {
            throw new InvalidDataException("A Brotli-compressed Parquet page did not decompress.");
        }

        return brotli;
    }

    // Snappy's raw block format: a varint length, then literals and back-references.
    public static byte[] Snappy(ReadOnlySpan<byte> input, int expected)
    {
        int pos = 0;
        uint length = 0;
        for (int shift = 0; ; shift += 7)
        {
            byte b = input[pos++];
            length |= (uint)(b & 0x7F) << shift;
            if ((b & 0x80) == 0)
            {
                break;
            }
        }

        var output = new byte[length];
        int outPos = 0;
        while (pos < input.Length)
        {
            byte tag = input[pos++];
            int kind = tag & 3;
            if (kind == 0)
            {
                int literal = tag >> 2;
                if (literal >= 60)
                {
                    int bytes = literal - 59;
                    literal = 0;
                    for (int i = 0; i < bytes; i++)
                    {
                        literal |= input[pos++] << (8 * i);
                    }
                }

                literal++;
                input.Slice(pos, literal).CopyTo(output.AsSpan(outPos));
                pos += literal;
                outPos += literal;
                continue;
            }

            int count, offset;
            if (kind == 1)
            {
                count = 4 + ((tag >> 2) & 7);
                offset = ((tag >> 5) << 8) | input[pos++];
            }
            else if (kind == 2)
            {
                count = (tag >> 2) + 1;
                offset = BinaryPrimitives.ReadUInt16LittleEndian(input[pos..]);
                pos += 2;
            }
            else
            {
                count = (tag >> 2) + 1;
                offset = BinaryPrimitives.ReadInt32LittleEndian(input[pos..]);
                pos += 4;
            }

            if (offset <= 0 || offset > outPos)
            {
                throw new InvalidDataException("Corrupt Snappy data in a Parquet page.");
            }

            for (int i = 0; i < count; i++)                         // byte by byte: copies may overlap
            {
                output[outPos + i] = output[outPos - offset + i];
            }

            outPos += count;
        }

        if (outPos != length || length != expected)
        {
            throw new InvalidDataException("A Snappy-compressed Parquet page has the wrong size.");
        }

        return output;
    }

    // LZ4's block format (sequences of literals and matches).
    public static byte[] Lz4Block(ReadOnlySpan<byte> input, int expected)
    {
        var output = new byte[expected];
        int pos = 0, outPos = 0;
        while (pos < input.Length)
        {
            byte token = input[pos++];
            int literal = token >> 4;
            if (literal == 15)
            {
                byte b;
                do
                {
                    b = input[pos++];
                    literal += b;
                }
                while (b == 255);
            }

            input.Slice(pos, literal).CopyTo(output.AsSpan(outPos));
            pos += literal;
            outPos += literal;
            if (pos >= input.Length)
            {
                break;                                              // the last sequence has no match
            }

            int offset = input[pos] | (input[pos + 1] << 8);
            pos += 2;
            int match = token & 15;
            if (match == 15)
            {
                byte b;
                do
                {
                    b = input[pos++];
                    match += b;
                }
                while (b == 255);
            }

            match += 4;
            if (offset <= 0 || offset > outPos)
            {
                throw new InvalidDataException("Corrupt LZ4 data in a Parquet page.");
            }

            for (int i = 0; i < match; i++)
            {
                output[outPos + i] = output[outPos - offset + i];
            }

            outPos += match;
        }

        if (outPos != expected)
        {
            throw new InvalidDataException("An LZ4-compressed Parquet page has the wrong size.");
        }

        return output;
    }

    private sealed class BuiltIn(int id, string name, Decompressor decompress) : IParquetCodec
    {
        public int Id => id;

        public string Name => name;

        public byte[] Decompress(ReadOnlySpan<byte> input, int uncompressedSize) => decompress(input, uncompressedSize);
    }
}
