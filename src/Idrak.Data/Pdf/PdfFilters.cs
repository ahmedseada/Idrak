// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Binary;
using System.IO.Compression;
using System.Runtime.InteropServices;

namespace Idrak.Data.Pdf;

/// <summary>One filter of a stream: its name and its parameters (resolved), or none.</summary>
internal readonly record struct PdfFilter(PdfName Name, PdfDictionary? Parameters);

/// <summary>
/// The standard filters (ISO 32000-2, 7.4): <c>FlateDecode</c> (zlib through <see cref="ZLibStream"/>, raw deflate when
/// the data has no zlib header), <c>LZWDecode</c> (with or without early change), <c>ASCIIHexDecode</c>,
/// <c>ASCII85Decode</c>, <c>RunLengthDecode</c>, the PNG and TIFF predictors, and the identity crypt filter; chained in
/// order. The image encodings (<c>DCTDecode</c>, <c>JPXDecode</c>, <c>CCITTFaxDecode</c>, <c>JBIG2Decode</c>) are not
/// decoded: their data is handed over as it is, for the image codecs. Every output is bounded: passing the limit is the
/// <see cref="PdfLimit"/> error the caller words.
/// </summary>
internal static class PdfFilters
{
    private const int Chunk = 64 * 1024;

    /// <summary>The encoding <paramref name="filter"/> names when it is an image encoding (DocumentImage's names), else null.</summary>
    public static string? ImageEncoding(PdfName filter) =>
        filter.Is("DCTDecode"u8) || filter.Is("DCT"u8) ? DocumentImage.Jpeg
        : filter.Is("JPXDecode"u8) ? DocumentImage.Jpeg2000
        : filter.Is("CCITTFaxDecode"u8) || filter.Is("CCF"u8) ? DocumentImage.Ccitt
        : filter.Is("JBIG2Decode"u8) ? DocumentImage.Jbig2
        : null;

    /// <summary>
    /// <paramref name="stored"/> through <paramref name="filters"/>, stopping before an image encoding. The result owns
    /// the buffers it uses (dispose it); <paramref name="window"/>, the stored bytes' buffer, passes to the result when
    /// no filter changes them, and is disposed here otherwise.
    /// </summary>
    /// <exception cref="InvalidDataException">
    /// An output passed <paramref name="limit"/> (a <see cref="PdfLimit"/> error saying <paramref name="overLimit"/>), a filter is
    /// unknown, or its data is damaged before any of it decoded.
    /// </exception>
    public static PdfDecoded Decode(ReadOnlyMemory<byte> stored, PdfWindow? window, ReadOnlySpan<PdfFilter> filters, long limit, string overLimit)
    {
        var input = stored;
        PdfBuffer? current = null;
        string? encoding = null;
        bool damaged = false;
        try
        {
            foreach (var filter in filters)
            {
                if (ImageEncoding(filter.Name) is { } image)
                {
                    encoding = image;
                    break;
                }

                if (filter.Name.Is("Crypt"u8))
                {
                    if (filter.Parameters?["Name"u8] is { Kind: PdfKind.Name } name && !name.IsName("Identity"u8))
                    {
                        throw new InvalidDataException($"the crypt filter /{name.AsName} in a file that is not encrypted");
                    }

                    continue;   // the identity filter changes nothing
                }

                var output = new PdfBuffer((int)Math.Clamp(input.Length * 2L, 256, int.MaxValue), limit, overLimit);
                try
                {
                    DecodeOne(filter, input, output, ref damaged);
                }
                catch
                {
                    output.Dispose();
                    throw;
                }

                current?.Dispose();
                current = output;
                input = output.Memory;
            }

            if (current is null)
            {
                var kept = window;
                window = null;
                return new PdfDecoded(stored, null, kept, encoding, damaged);
            }

            var result = new PdfDecoded(current.Memory, current, null, encoding, damaged);
            current = null;
            return result;
        }
        finally
        {
            current?.Dispose();
            window?.Dispose();
        }
    }

    private static void DecodeOne(PdfFilter filter, ReadOnlyMemory<byte> input, PdfBuffer output, ref bool damaged)
    {
        var name = filter.Name;
        if (name.Is("FlateDecode"u8) || name.Is("Fl"u8))
        {
            Flate(input, output, ref damaged);
            Predict(output, filter.Parameters, ref damaged);
        }
        else if (name.Is("LZWDecode"u8) || name.Is("LZW"u8))
        {
            long early = 1;
            if (filter.Parameters?["EarlyChange"u8] is { } value && value.TryInteger(out long e))
            {
                early = e == 0 ? 0 : 1;
            }

            Lzw(input.Span, output, (int)early, ref damaged);
            Predict(output, filter.Parameters, ref damaged);
        }
        else if (name.Is("ASCIIHexDecode"u8) || name.Is("AHx"u8))
        {
            AsciiHex(input.Span, output);
        }
        else if (name.Is("ASCII85Decode"u8) || name.Is("A85"u8))
        {
            Ascii85(input.Span, output, ref damaged);
        }
        else if (name.Is("RunLengthDecode"u8) || name.Is("RL"u8))
        {
            RunLength(input.Span, output, ref damaged);
        }
        else
        {
            throw new InvalidDataException($"the filter /{name} is not one a PDF reader knows");
        }
    }

    // Flate: zlib when the data starts with a zlib header, raw deflate otherwise (or when the header lies). Data that is
    // damaged part of the way keeps what decoded before it, as readers do; damaged from the start is refused.
    private static void Flate(ReadOnlyMemory<byte> input, PdfBuffer output, ref bool damaged)
    {
        var span = input.Span;
        if (span.IsEmpty)
        {
            return;
        }

        bool zlib = span.Length >= 2 && (span[0] & 0x0F) == 8 && span[0] >> 4 <= 7 && ((span[0] << 8) | span[1]) % 31 == 0;
        if (zlib)
        {
            try
            {
                Inflate(input, output, zlib: true);
                return;
            }
            catch (InvalidDataException e) when (!PdfLimit.Is(e))
            {
                if (output.Length > 0)
                {
                    damaged = true;
                    return;
                }

                output.Clear();
            }
        }

        try
        {
            Inflate(input, output, zlib: false);
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            if (output.Length == 0)
            {
                throw new InvalidDataException($"the FlateDecode data is damaged ({e.Message})", e);
            }

            damaged = true;
        }
    }

    private static void Inflate(ReadOnlyMemory<byte> input, PdfBuffer output, bool zlib)
    {
        var segment = MemoryMarshal.TryGetArray(input, out var array) ? array : new ArraySegment<byte>(input.ToArray());
        using var source = new MemoryStream(segment.Array!, segment.Offset, segment.Count, writable: false);
        using Stream inflater = zlib ? new ZLibStream(source, CompressionMode.Decompress) : new DeflateStream(source, CompressionMode.Decompress);
        Span<byte> probe = stackalloc byte[1];
        while (true)
        {
            if (output.Full)
            {
                // At the limit: one more byte means the stream passes it.
                if (inflater.Read(probe) > 0)
                {
                    throw output.OverLimit();
                }

                return;
            }

            int read = inflater.Read(output.GetSpan(Chunk));
            if (read == 0)
            {
                return;
            }

            output.Advance(read);
        }
    }

    // LZW (7.4.4): codes of 9 to 12 bits, 256 clears the table, 257 ends; with early change (the default) the code length
    // grows one code early. Each entry is its prefix's code, its last byte, its first byte and its length, so an entry is
    // written back to front without a stack.
    private static void Lzw(ReadOnlySpan<byte> input, PdfBuffer output, int earlyChange, ref bool damaged)
    {
        const int Size = 4096;
        int[] table = ArrayPool<int>.Shared.Rent(Size * 2);   // prefix code, length
        byte[] bytes = ArrayPool<byte>.Shared.Rent(Size * 2); // last byte, first byte
        try
        {
            for (int i = 0; i < 256; i++)
            {
                table[i * 2] = -1;
                table[i * 2 + 1] = 1;
                bytes[i * 2] = bytes[i * 2 + 1] = (byte)i;
            }

            int codeLength = 9, next = 258, previous = -1, position = 0, bits = 0;
            ulong buffer = 0;
            while (true)
            {
                while (bits < codeLength && position < input.Length)
                {
                    buffer = (buffer << 8) | input[position++];
                    bits += 8;
                }

                if (bits < codeLength)
                {
                    break;   // the data ends without the end code
                }

                int code = (int)((buffer >> (bits - codeLength)) & ((1UL << codeLength) - 1));
                bits -= codeLength;
                if (code == 256)
                {
                    (codeLength, next, previous) = (9, 258, -1);
                    continue;
                }

                if (code == 257)
                {
                    break;
                }

                if (previous < 0)
                {
                    if (code > 255)
                    {
                        damaged = true;
                        break;
                    }

                    WriteEntry(table, bytes, code, output);
                    previous = code;
                    continue;
                }

                if (code < next)
                {
                    if (next < Size)
                    {
                        AddEntry(table, bytes, next++, previous, bytes[code * 2 + 1]);
                    }

                    WriteEntry(table, bytes, code, output);
                }
                else if (code == next && next < Size)
                {
                    AddEntry(table, bytes, next++, previous, bytes[previous * 2 + 1]);
                    WriteEntry(table, bytes, code, output);
                }
                else
                {
                    damaged = true;
                    break;
                }

                previous = code;
                if (next + earlyChange >= 1 << codeLength && codeLength < 12)
                {
                    codeLength++;
                }
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(table);
            ArrayPool<byte>.Shared.Return(bytes);
        }
    }

    private static void AddEntry(int[] table, byte[] bytes, int code, int prefix, byte last)
    {
        table[code * 2] = prefix;
        table[code * 2 + 1] = table[prefix * 2 + 1] + 1;
        bytes[code * 2] = last;
        bytes[code * 2 + 1] = bytes[prefix * 2 + 1];
    }

    private static void WriteEntry(int[] table, byte[] bytes, int code, PdfBuffer output)
    {
        int length = table[code * 2 + 1];
        var span = output.Reserve(length);
        for (int i = length - 1; i >= 0; i--)
        {
            span[i] = bytes[code * 2];
            code = table[code * 2];
        }

        output.Advance(length);
    }

    // ASCIIHex: pairs of hex digits, whitespace skipped, '>' ends, a last odd digit followed by 0.
    private static void AsciiHex(ReadOnlySpan<byte> input, PdfBuffer output)
    {
        var writer = new Writer(output);
        int half = -1;
        foreach (byte c in input)
        {
            if (PdfParser.IsWhitespace(c))
            {
                continue;
            }

            if (c == (byte)'>')
            {
                break;
            }

            if (!char.IsAsciiHexDigit((char)c))
            {
                throw new InvalidDataException($"'{(char)c}' in ASCIIHexDecode data");
            }

            if (half < 0)
            {
                half = PdfParser.HexValue(c);
            }
            else
            {
                writer.Put((byte)(half << 4 | PdfParser.HexValue(c)));
                half = -1;
            }
        }

        if (half >= 0)
        {
            writer.Put((byte)(half << 4));
        }

        writer.Flush();
    }

    // ASCII85: groups of five characters '!' to 'u' for four bytes, 'z' for four zeros, "~>" ends; a last group of n
    // characters gives n - 1 bytes.
    private static void Ascii85(ReadOnlySpan<byte> input, PdfBuffer output, ref bool damaged)
    {
        var writer = new Writer(output);
        ulong group = 0;
        int count = 0;
        foreach (byte c in input)
        {
            if (PdfParser.IsWhitespace(c))
            {
                continue;
            }

            if (c == (byte)'~')
            {
                break;
            }

            if (c == (byte)'z' && count == 0)
            {
                writer.Put(0);
                writer.Put(0);
                writer.Put(0);
                writer.Put(0);
                continue;
            }

            if (c is < (byte)'!' or > (byte)'u')
            {
                throw new InvalidDataException($"'{(char)c}' in ASCII85Decode data");
            }

            group = group * 85 + (ulong)(c - '!');
            if (++count == 5)
            {
                if (group > uint.MaxValue)
                {
                    throw new InvalidDataException("an ASCII85Decode group past four bytes");
                }

                writer.Put((byte)(group >> 24));
                writer.Put((byte)(group >> 16));
                writer.Put((byte)(group >> 8));
                writer.Put((byte)group);
                (group, count) = (0, 0);
            }
        }

        if (count == 1)
        {
            damaged = true;   // one character alone holds no byte
        }
        else if (count > 1)
        {
            for (int k = count; k < 5; k++)
            {
                group = group * 85 + 84;
            }

            for (int k = 0; k < count - 1; k++)
            {
                writer.Put((byte)(group >> (24 - 8 * k)));
            }
        }

        writer.Flush();
    }

    // RunLength: a length byte n; 0-127 copies the next n + 1 bytes, 129-255 repeats the next byte 257 - n times, 128 ends.
    private static void RunLength(ReadOnlySpan<byte> input, PdfBuffer output, ref bool damaged)
    {
        int i = 0;
        while (i < input.Length)
        {
            int n = input[i++];
            if (n == 128)
            {
                return;
            }

            if (n < 128)
            {
                int count = Math.Min(n + 1, input.Length - i);
                damaged |= count < n + 1;
                output.Write(input.Slice(i, count));
                i += count;
            }
            else
            {
                if (i >= input.Length)
                {
                    damaged = true;
                    return;
                }

                output.Reserve(257 - n).Fill(input[i++]);
                output.Advance(257 - n);
            }
        }
    }

    // The predictor of a Flate or LZW stream (7.4.4.4), in place: TIFF 2 (each sample less the one a pixel before) or PNG
    // (10-15: each row's own filter byte, then None, Sub, Up, Average or Paeth).
    private static void Predict(PdfBuffer output, PdfDictionary? parameters, ref bool damaged)
    {
        if (parameters is null || !parameters["Predictor"u8].TryInteger(out long predictor) || predictor <= 1)
        {
            return;
        }

        long colors = Parameter(parameters, "Colors"u8, 1), bpc = Parameter(parameters, "BitsPerComponent"u8, 8), columns = Parameter(parameters, "Columns"u8, 1);
        if (colors is < 1 or > 32 || bpc is not (1 or 2 or 4 or 8 or 16) || columns < 1 || columns * colors * bpc > int.MaxValue / 2)
        {
            throw new InvalidDataException($"a predictor with {colors} colours of {bpc} bits in {columns} columns");
        }

        int bytesPerPixel = (int)Math.Max(1, (colors * bpc + 7) / 8), rowLength = (int)((columns * colors * bpc + 7) / 8);
        if (predictor == 2)
        {
            Tiff(output.Written, rowLength, (int)colors, (int)bpc, (int)columns);
        }
        else if (predictor >= 10)
        {
            output.Truncate(Png(output.Written, rowLength, bytesPerPixel, ref damaged));
        }
        else
        {
            throw new InvalidDataException($"the predictor {predictor}");
        }
    }

    private static long Parameter(PdfDictionary parameters, ReadOnlySpan<byte> key, long otherwise) =>
        parameters[key].TryInteger(out long value) ? value : otherwise;

    // PNG rows in place: row r's filter byte at r * (rowLength + 1), its bytes after it; decoded row r goes to
    // r * rowLength, never past what is still to be read. A last row cut short keeps what it has.
    private static int Png(Span<byte> data, int rowLength, int bytesPerPixel, ref bool damaged)
    {
        int stride = rowLength + 1, written = 0;
        for (int start = 0; start < data.Length; start += stride)
        {
            int type = data[start];
            int count = Math.Min(rowLength, data.Length - start - 1);
            int row = written;
            bool hasUp = row >= rowLength;
            if (type > 4)
            {
                damaged = true;
                type = 0;
            }

            for (int i = 0; i < count; i++)
            {
                int x = data[start + 1 + i];
                int a = i >= bytesPerPixel ? data[row + i - bytesPerPixel] : 0;
                int b = hasUp ? data[row - rowLength + i] : 0;
                int c = hasUp && i >= bytesPerPixel ? data[row - rowLength + i - bytesPerPixel] : 0;
                data[row + i] = type switch
                {
                    1 => (byte)(x + a),
                    2 => (byte)(x + b),
                    3 => (byte)(x + ((a + b) >> 1)),
                    4 => (byte)(x + Paeth(a, b, c)),
                    _ => (byte)x,
                };
            }

            written += Math.Max(0, count);
            damaged |= count < rowLength;
        }

        return written;
    }

    private static int Paeth(int a, int b, int c)
    {
        int p = a + b - c, pa = Math.Abs(p - a), pb = Math.Abs(p - b), pc = Math.Abs(p - c);
        return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
    }

    // TIFF predictor 2 in place, row by row: each sample plus the same component of the pixel before it, modulo its bits.
    private static void Tiff(Span<byte> data, int rowLength, int colors, int bpc, int columns)
    {
        int samples = columns * colors;
        for (int start = 0; start + rowLength <= data.Length; start += rowLength)
        {
            var row = data.Slice(start, rowLength);
            if (bpc == 8)
            {
                for (int i = colors; i < samples; i++)
                {
                    row[i] = (byte)(row[i] + row[i - colors]);
                }
            }
            else if (bpc == 16)
            {
                for (int i = colors; i < samples; i++)
                {
                    ushort value = (ushort)(BinaryPrimitives.ReadUInt16BigEndian(row[(2 * i)..]) + BinaryPrimitives.ReadUInt16BigEndian(row[(2 * (i - colors))..]));
                    BinaryPrimitives.WriteUInt16BigEndian(row[(2 * i)..], value);
                }
            }
            else
            {
                int mask = (1 << bpc) - 1;
                for (int i = colors; i < samples; i++)
                {
                    int value = (Sample(row, i, bpc) + Sample(row, i - colors, bpc)) & mask;
                    int bit = i * bpc, shift = 8 - bpc - (bit & 7);
                    row[bit >> 3] = (byte)((row[bit >> 3] & ~(mask << shift)) | (value << shift));
                }
            }
        }
    }

    private static int Sample(ReadOnlySpan<byte> row, int index, int bpc)
    {
        int bit = index * bpc;
        return (row[bit >> 3] >> (8 - bpc - (bit & 7))) & ((1 << bpc) - 1);
    }

    // Byte-at-a-time writes into a pooled buffer, a span at a time.
    private ref struct Writer(PdfBuffer output)
    {
        private Span<byte> _span;
        private int _count;

        public void Put(byte value)
        {
            if (_count == _span.Length)
            {
                Flush();
                _span = output.GetSpan(Chunk);
            }

            _span[_count++] = value;
        }

        public void Flush()
        {
            if (_count > 0)
            {
                output.Advance(_count);
            }

            _span = default;
            _count = 0;
        }
    }
}
