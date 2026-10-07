// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.IO.Compression;
using System.Numerics;
using Idrak.Data.Abstractions;

namespace Idrak.Data;

// The built-in image codecs of ImageCodecs: PNG, BMP and Netpbm, decoded without dependencies.

/// <summary>
/// PNG (ISO/IEC 15948): grey, grey with alpha, RGB, RGBA and palette images at every allowed bit depth (1, 2, 4, 8 and
/// 16), interlaced (Adam7) or not. The image data is inflated with .NET's <see cref="ZLibStream"/>; ancillary chunks
/// (transparency, gamma, colour profiles) are ignored, and alpha is dropped.
/// </summary>
internal sealed class PngCodec : IImageCodec
{
    private static ReadOnlySpan<byte> Signature => [0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A];

    // Adam7: the start and step of each pass, in x and in y.
    private static readonly (int X0, int Y0, int Dx, int Dy)[] Adam7 =
        [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)];

    public string Name => "png";

    public IReadOnlyCollection<string> Extensions { get; } = [".png"];

    public ImageInfo? ReadInfo(ReadOnlySpan<byte> header)
    {
        if (header.Length < 26 || !header[..8].SequenceEqual(Signature))
        {
            return null;
        }

        int w = BinaryPrimitives.ReadInt32BigEndian(header[16..]), h = BinaryPrimitives.ReadInt32BigEndian(header[20..]);
        return new ImageInfo(w, h, header[25] is 0 or 4 ? 1 : 3, Name);
    }

    public ImageData Decode(ReadOnlySpan<byte> file)
    {
        int position = 8, w = 0, h = 0, depth = 0, color = -1, interlace = 0;
        byte[]? palette = null;
        using var compressed = new MemoryStream();
        while (position + 8 <= file.Length)
        {
            int length = BinaryPrimitives.ReadInt32BigEndian(file[position..]);
            var type = file.Slice(position + 4, 4);
            if (length < 0 || position + 12L + length > file.Length)
            {
                throw new InvalidDataException("a PNG chunk runs past the end of the file.");
            }

            var data = file.Slice(position + 8, length);
            if (type.SequenceEqual("IHDR"u8))
            {
                w = BinaryPrimitives.ReadInt32BigEndian(data);
                h = BinaryPrimitives.ReadInt32BigEndian(data[4..]);
                depth = data[8];
                color = data[9];
                interlace = data[12];
            }
            else if (type.SequenceEqual("PLTE"u8))
            {
                palette = data.ToArray();
            }
            else if (type.SequenceEqual("IDAT"u8))
            {
                compressed.Write(data);
            }
            else if (type.SequenceEqual("IEND"u8))
            {
                break;
            }

            position += 12 + length;                                                    // length, type, data, CRC
        }

        int samples = color switch { 0 => 1, 2 => 3, 3 => 1, 4 => 2, 6 => 4, _ => 0 };
        bool depthAllowed = color switch
        {
            0 => depth is 1 or 2 or 4 or 8 or 16,
            3 => depth is 1 or 2 or 4 or 8,
            2 or 4 or 6 => depth is 8 or 16,
            _ => false,
        };
        if (samples == 0 || !depthAllowed || w <= 0 || h <= 0 || interlace > 1)
        {
            throw new InvalidDataException($"PNG colour type {color} at {depth} bits, {w} x {h}, interlace {interlace} is not a valid image.");
        }

        if (color == 3 && palette is null)
        {
            throw new InvalidDataException("a PNG palette image without a palette.");
        }

        var passes = interlace == 1 ? Adam7 : [(0, 0, 1, 1)];
        int bitsPerPixel = samples * depth, bpp = Math.Max(1, bitsPerPixel / 8);
        long total = 0;
        foreach (var (x0, y0, dx, dy) in passes)
        {
            long pw = (w - x0 + dx - 1) / dx, ph = (h - y0 + dy - 1) / dy;
            total += pw == 0 ? 0 : ph * (1 + (pw * bitsPerPixel + 7) / 8);
        }

        var raw = new byte[checked((int)total)];
        compressed.Position = 0;
        using (var inflate = new ZLibStream(compressed, CompressionMode.Decompress))
        {
            inflate.ReadExactly(raw);
        }

        int channels = color is 2 or 3 or 6 ? 3 : 1;
        var pixels = new float[checked(channels * h * w)];
        float max = (1 << depth) - 1;
        int offset = 0;
        foreach (var (x0, y0, dx, dy) in passes)
        {
            int pw = (w - x0 + dx - 1) / dx, ph = (h - y0 + dy - 1) / dy;
            if (pw == 0 || ph == 0)
            {
                continue;
            }

            int stride = (pw * bitsPerPixel + 7) / 8;
            Span<byte> previous = new byte[stride];
            for (int py = 0; py < ph; py++)
            {
                var line = raw.AsSpan(offset + 1, stride);
                Unfilter(raw[offset], line, previous, bpp);
                offset += 1 + stride;
                int y = y0 + py * dy;
                for (int px = 0; px < pw; px++)
                {
                    int at = y * w + x0 + px * dx;
                    if (color == 3)
                    {
                        int index = SampleAt(line, px, 0, 1, depth);
                        for (int ch = 0; ch < 3; ch++)
                        {
                            pixels[ch * h * w + at] = palette![Math.Min(index * 3 + ch, palette.Length - 1)] / 255f;
                        }
                    }
                    else
                    {
                        for (int ch = 0; ch < channels; ch++)
                        {
                            pixels[ch * h * w + at] = SampleAt(line, px, ch, samples, depth) / max;
                        }
                    }
                }

                previous = line;
            }
        }

        return new ImageData(pixels, channels, h, w);
    }

    // Sample s of pixel x in an unfiltered line: 16 bits big-endian, a byte, or 1, 2 or 4 bits from the high bits down.
    private static int SampleAt(ReadOnlySpan<byte> line, int x, int s, int samples, int depth)
    {
        int index = x * samples + s;
        return depth switch
        {
            16 => line[index * 2] << 8 | line[index * 2 + 1],
            8 => line[index],
            _ => line[index * depth >> 3] >> (8 - depth - (index * depth & 7)) & ((1 << depth) - 1),
        };
    }

    // The PNG filters (none, sub, up, average, Paeth) undone in place; bpp is the bytes per complete pixel (at least 1).
    private static void Unfilter(byte filter, Span<byte> line, ReadOnlySpan<byte> previous, int bpp)
    {
        if (filter > 4)
        {
            throw new InvalidDataException($"PNG filter {filter} does not exist.");
        }

        for (int i = 0; i < line.Length; i++)
        {
            int left = i >= bpp ? line[i - bpp] : 0, up = previous[i], upLeft = i >= bpp ? previous[i - bpp] : 0;
            int predicted = filter switch
            {
                1 => left,
                2 => up,
                3 => (left + up) / 2,
                4 => Paeth(left, up, upLeft),
                _ => 0,
            };
            line[i] = (byte)(line[i] + predicted);
        }
    }

    private static int Paeth(int a, int b, int c)
    {
        int p = a + b - c, pa = Math.Abs(p - a), pb = Math.Abs(p - b), pc = Math.Abs(p - c);
        return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
    }
}

/// <summary>
/// Windows bitmaps: 1, 4 and 8 bits with a palette (grey when every palette entry is grey), 16 bits (5-5-5 or bit
/// fields), 24 bits, and 32 bits (with or without bit fields), bottom-up or top-down. Run-length compressed bitmaps are
/// not read.
/// </summary>
internal sealed class BmpCodec : IImageCodec
{
    public string Name => "bmp";

    public IReadOnlyCollection<string> Extensions { get; } = [".bmp", ".dib"];

    public ImageInfo? ReadInfo(ReadOnlySpan<byte> header)
    {
        if (header.Length < 26 || header[0] != 'B' || header[1] != 'M')
        {
            return null;
        }

        var (w, h, bits, _) = Size(header);
        bool grey = bits <= 8 && header.Length >= 54 && IsGrey(header, bits);
        return new ImageInfo(w, Math.Abs(h), grey ? 1 : 3, Name);
    }

    public ImageData Decode(ReadOnlySpan<byte> file)
    {
        if (file.Length < 54)
        {
            throw new InvalidDataException("a BMP file shorter than its headers.");
        }

        int offset = BinaryPrimitives.ReadInt32LittleEndian(file[10..]);
        var (w, rawHeight, bits, compression) = Size(file);
        int h = Math.Abs(rawHeight);
        if (w <= 0 || h == 0 || bits is not (1 or 4 or 8 or 16 or 24 or 32))
        {
            throw new InvalidDataException($"a BMP of {w} x {h} at {bits} bits is not read.");
        }

        if (compression is not (0 or 3 or 6) || compression != 0 && bits is not (16 or 32))
        {
            throw new InvalidDataException($"BMP compression {compression} (run-length or embedded) is not read; save it uncompressed.");
        }

        var colors = bits <= 8 ? Palette(file, bits) : null;
        bool grey = colors is not null && colors.All(c => c.R == c.G && c.G == c.B);
        int channels = grey ? 1 : 3;
        uint[] masks = compression is 3 or 6
            ? [BinaryPrimitives.ReadUInt32LittleEndian(file[54..]), BinaryPrimitives.ReadUInt32LittleEndian(file[58..]), BinaryPrimitives.ReadUInt32LittleEndian(file[62..])]
            : bits == 16 ? [0x7C00, 0x03E0, 0x001F] : [0xFF0000, 0xFF00, 0xFF];
        int stride = (w * bits + 31) / 32 * 4;
        if (offset < 0 || offset + (long)stride * h > file.Length)
        {
            throw new InvalidDataException("the BMP pixels run past the end of the file.");
        }

        var pixels = new float[checked(channels * h * w)];
        for (int y = 0; y < h; y++)
        {
            var row = file.Slice(offset + (rawHeight > 0 ? h - 1 - y : y) * stride, stride);     // bottom-up unless the height is negative
            for (int x = 0; x < w; x++)
            {
                int at = y * w + x;
                if (colors is not null)
                {
                    int index = bits == 8 ? row[x] : row[x * bits >> 3] >> (8 - bits - (x * bits & 7)) & ((1 << bits) - 1);
                    var (r, g, b) = index < colors.Length ? colors[index] : ((byte)0, (byte)0, (byte)0);
                    if (grey)
                    {
                        pixels[at] = r / 255f;
                        continue;
                    }

                    pixels[at] = r / 255f;
                    pixels[h * w + at] = g / 255f;
                    pixels[2 * h * w + at] = b / 255f;
                }
                else if (bits == 24 || bits == 32 && compression == 0)
                {
                    int p = x * (bits / 8);
                    pixels[at] = row[p + 2] / 255f;                                     // stored as blue, green, red
                    pixels[h * w + at] = row[p + 1] / 255f;
                    pixels[2 * h * w + at] = row[p] / 255f;
                }
                else
                {
                    uint value = bits == 16 ? BinaryPrimitives.ReadUInt16LittleEndian(row[(x * 2)..]) : BinaryPrimitives.ReadUInt32LittleEndian(row[(x * 4)..]);
                    for (int ch = 0; ch < 3; ch++)
                    {
                        pixels[ch * h * w + at] = Field(value, masks[ch]);
                    }
                }
            }
        }

        return new ImageData(pixels, channels, h, w);
    }

    // The width, height (negative for top-down), bits per pixel and compression, from the core (12-byte) or a later info header.
    private static (int Width, int Height, int Bits, int Compression) Size(ReadOnlySpan<byte> b)
    {
        int dib = BinaryPrimitives.ReadInt32LittleEndian(b[14..]);
        return dib == 12
            ? (BinaryPrimitives.ReadInt16LittleEndian(b[18..]), BinaryPrimitives.ReadInt16LittleEndian(b[20..]), BinaryPrimitives.ReadInt16LittleEndian(b[24..]), 0)
            : (BinaryPrimitives.ReadInt32LittleEndian(b[18..]), BinaryPrimitives.ReadInt32LittleEndian(b[22..]), BinaryPrimitives.ReadInt16LittleEndian(b[28..]),
                b.Length >= 34 ? BinaryPrimitives.ReadInt32LittleEndian(b[30..]) : 0);
    }

    // The palette after the info header: 3-byte entries after a core header, 4-byte (blue, green, red, unused) otherwise.
    private static (byte R, byte G, byte B)[] Palette(ReadOnlySpan<byte> b, int bits)
    {
        int dib = BinaryPrimitives.ReadInt32LittleEndian(b[14..]);
        int entry = dib == 12 ? 3 : 4;
        int used = dib >= 40 ? BinaryPrimitives.ReadInt32LittleEndian(b[46..]) : 0;
        int count = used > 0 && used <= 1 << bits ? used : 1 << bits;
        int start = 14 + dib;
        count = Math.Max(0, Math.Min(count, (b.Length - start) / entry));
        var colors = new (byte, byte, byte)[count];
        for (int i = 0; i < count; i++)
        {
            int at = start + i * entry;
            colors[i] = (b[at + 2], b[at + 1], b[at]);
        }

        return colors;
    }

    private static bool IsGrey(ReadOnlySpan<byte> header, int bits)
    {
        var colors = Palette(header, bits);
        return colors.Length > 0 && colors.All(c => c.R == c.G && c.G == c.B);
    }

    private static float Field(uint value, uint mask)
    {
        if (mask == 0)
        {
            return 0f;
        }

        int shift = BitOperations.TrailingZeroCount(mask);
        return ((value & mask) >> shift) / (float)(mask >> shift);
    }
}

/// <summary>Netpbm grey and colour images: PGM (P2 text, P5 binary) and PPM (P3 text, P6 binary), 8 or 16 bits.</summary>
internal sealed class NetpbmCodec : IImageCodec
{
    public string Name => "netpbm";

    public IReadOnlyCollection<string> Extensions { get; } = [".pgm", ".ppm", ".pnm"];

    public ImageInfo? ReadInfo(ReadOnlySpan<byte> header)
    {
        if (header.Length < 2 || header[0] != 'P' || header[1] is not ((byte)'2' or (byte)'3' or (byte)'5' or (byte)'6'))
        {
            return null;
        }

        int position = 2;
        int w = Number(header, ref position), h = Number(header, ref position);
        return w > 0 && h > 0 ? new ImageInfo(w, h, Colour(header) ? 3 : 1, Name) : null;
    }

    public ImageData Decode(ReadOnlySpan<byte> file)
    {
        int position = 2;
        int w = Number(file, ref position), h = Number(file, ref position), max = Number(file, ref position);
        if (w <= 0 || h <= 0 || max is <= 0 or > 65535)
        {
            throw new InvalidDataException($"a Netpbm header of {w} x {h} with maximum {max}.");
        }

        int c = Colour(file) ? 3 : 1;
        bool binary = file[1] is (byte)'5' or (byte)'6';
        var pixels = new float[checked(c * h * w)];
        position++;                                                                     // the single whitespace after the maximum
        int wide = max > 255 ? 2 : 1;
        if (binary && position + (long)h * w * c * wide > file.Length)
        {
            throw new InvalidDataException("the Netpbm pixels run past the end of the file.");
        }

        for (int i = 0; i < h * w; i++)
        {
            for (int ch = 0; ch < c; ch++)
            {
                int value;
                if (binary)
                {
                    int at = position + (i * c + ch) * wide;
                    value = wide == 1 ? file[at] : file[at] << 8 | file[at + 1];
                }
                else if ((value = Number(file, ref position)) < 0)
                {
                    throw new InvalidDataException("a Netpbm text image with fewer values than pixels.");
                }

                pixels[ch * h * w + i] = value / (float)max;
            }
        }

        return new ImageData(pixels, c, h, w);
    }

    private static bool Colour(ReadOnlySpan<byte> b) => b[1] is (byte)'3' or (byte)'6';

    // The next decimal number, after whitespace and # comments; -1 when there is none.
    private static int Number(ReadOnlySpan<byte> b, ref int position)
    {
        while (position < b.Length && (char.IsWhiteSpace((char)b[position]) || b[position] == '#'))
        {
            if (b[position] == '#')
            {
                while (position < b.Length && b[position] != '\n')
                {
                    position++;
                }
            }

            position++;
        }

        int value = 0, start = position;
        while (position < b.Length && b[position] is >= (byte)'0' and <= (byte)'9' && value < 100_000_000)
        {
            value = value * 10 + (b[position++] - '0');
        }

        return position > start ? value : -1;
    }
}
