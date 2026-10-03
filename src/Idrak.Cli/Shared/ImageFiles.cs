// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;

namespace Idrak.Cli.Shared;

/// <summary>
/// Image files for every command that reads them (<c>idrak suggest</c> and the data it prepares, <c>idrak train</c> and
/// <c>idrak predict</c> on image folders): the size and channels of PNG, JPEG, BMP and Netpbm (PGM, PPM) files from
/// their headers, and the pixels of PNG (8-bit grey, grey with alpha, RGB, RGBA and palette images, not interlaced),
/// BMP (24 and 32 bits, uncompressed) and Netpbm files. The library has no image decoder; JPEG pixels are not read here
/// (a gap noted in the plan), so a folder of JPEG files is profiled but not trained.
/// </summary>
internal static class ImageFiles
{
    /// <summary>The extensions read as images.</summary>
    public static readonly string[] Extensions = [".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".ppm", ".pnm"];

    /// <summary>Whether <paramref name="path"/> has an image extension.</summary>
    public static bool IsImage(string path) => Extensions.Contains(Path.GetExtension(path), StringComparer.OrdinalIgnoreCase);

    /// <summary>The extensions whose pixels are decoded (training and prediction read these).</summary>
    public static readonly string[] DecodedExtensions = [".png", ".bmp", ".pgm", ".ppm", ".pnm"];

    /// <summary>Whether <paramref name="path"/> has the extension of an image whose pixels are decoded.</summary>
    public static bool IsDecoded(string path) => DecodedExtensions.Contains(Path.GetExtension(path), StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// The image as [channels, height, width] values in [0, 1] with <paramref name="channels"/> channels and the given
    /// size (<see cref="Fit"/>); a file this reader cannot decode is an error naming it.
    /// </summary>
    public static float[] Load(string path, int channels, int height, int width) =>
        Decode(path) is var (pixels, c, h, w)
            ? Fit(pixels, c, h, w, channels, height, width)
            : throw new InvalidDataException($"{path}: not an image this tool decodes (PNG 8-bit not interlaced, BMP 24/32-bit uncompressed, PGM/PPM).");

    /// <summary>What an image header says.</summary>
    public readonly record struct ImageInfo(int Width, int Height, int Channels, string Format, bool Decodable);

    /// <summary>The size and channels of an image, or null when the file is not one of the formats above.</summary>
    public static ImageInfo? ReadHeader(string path)
    {
        byte[] head = new byte[64 * 1024];
        int length;
        using (var stream = File.OpenRead(path))
        {
            length = stream.ReadAtLeast(head, head.Length, throwOnEndOfStream: false);
        }

        var b = head.AsSpan(0, length);
        if (b.Length >= 26 && b[..8].SequenceEqual(PngSignature))
        {
            int w = BinaryPrimitives.ReadInt32BigEndian(b[16..]), h = BinaryPrimitives.ReadInt32BigEndian(b[20..]);
            int depth = b[24], color = b[25];
            int channels = color switch { 0 => 1, 2 => 3, 3 => 3, 4 => 2, 6 => 4, _ => 0 };
            bool interlaced = b.Length > 28 && b[28] != 0;
            return channels == 0 ? null : new ImageInfo(w, h, channels, "png", depth == 8 && !interlaced);
        }

        if (b.Length >= 4 && b[0] == 0xFF && b[1] == 0xD8)
        {
            return Jpeg(path);
        }

        if (b.Length >= 30 && b[0] == 'B' && b[1] == 'M')
        {
            int w = BinaryPrimitives.ReadInt32LittleEndian(b[18..]), h = Math.Abs(BinaryPrimitives.ReadInt32LittleEndian(b[22..]));
            int bits = BinaryPrimitives.ReadInt16LittleEndian(b[28..]);
            int compression = b.Length >= 34 ? BinaryPrimitives.ReadInt32LittleEndian(b[30..]) : 0;
            return new ImageInfo(w, h, bits == 32 ? 4 : bits == 8 ? 1 : 3, "bmp", (bits is 24 or 32) && compression is 0 or 3);
        }

        if (b.Length >= 2 && b[0] == 'P' && b[1] is (byte)'2' or (byte)'3' or (byte)'5' or (byte)'6')
        {
            int position = 2;
            int w = NetpbmNumber(b, ref position), h = NetpbmNumber(b, ref position);
            return w > 0 && h > 0 ? new ImageInfo(w, h, b[1] is (byte)'3' or (byte)'6' ? 3 : 1, b[1] is (byte)'3' or (byte)'6' ? "ppm" : "pgm", true) : null;
        }

        return null;
    }

    /// <summary>
    /// The pixels of an image as [channels, height, width] values in [0, 1] (alpha dropped), or null when this reader
    /// cannot decode the file (JPEG, PNG with 16-bit or packed samples or interlacing, compressed BMP).
    /// </summary>
    public static (float[] Pixels, int Channels, int Height, int Width)? Decode(string path)
    {
        var bytes = File.ReadAllBytes(path);
        if (bytes.Length >= 8 && bytes.AsSpan(0, 8).SequenceEqual(PngSignature))
        {
            return DecodePng(bytes);
        }

        if (bytes.Length >= 30 && bytes[0] == 'B' && bytes[1] == 'M')
        {
            return DecodeBmp(bytes);
        }

        if (bytes.Length >= 2 && bytes[0] == 'P' && bytes[1] is (byte)'2' or (byte)'3' or (byte)'5' or (byte)'6')
        {
            return DecodeNetpbm(bytes);
        }

        return null;
    }

    /// <summary>
    /// An image as <paramref name="channels"/> × <paramref name="height"/> × <paramref name="width"/> values: channels
    /// converted (gray repeated to colour, colour averaged to gray) and the size changed by bilinear sampling.
    /// </summary>
    public static float[] Fit(float[] pixels, int c, int h, int w, int channels, int height, int width)
    {
        var result = new float[channels * height * width];
        for (int oc = 0; oc < channels; oc++)
        {
            for (int y = 0; y < height; y++)
            {
                float sy = height == 1 ? 0 : y * (h - 1) / (float)(height - 1);
                int y0 = (int)sy, y1 = Math.Min(y0 + 1, h - 1);
                float fy = sy - y0;
                for (int x = 0; x < width; x++)
                {
                    float sx = width == 1 ? 0 : x * (w - 1) / (float)(width - 1);
                    int x0 = (int)sx, x1 = Math.Min(x0 + 1, w - 1);
                    float fx = sx - x0;
                    float Sample(int yy, int xx)
                    {
                        if (channels == c || c == 1)
                        {
                            return pixels[(c == 1 ? 0 : oc) * h * w + yy * w + xx];
                        }

                        float sum = 0;                                                  // colour to gray: the mean of the channels
                        for (int ic = 0; ic < c; ic++)
                        {
                            sum += pixels[ic * h * w + yy * w + xx];
                        }

                        return sum / c;
                    }

                    float top = Sample(y0, x0) * (1 - fx) + Sample(y0, x1) * fx, bottom = Sample(y1, x0) * (1 - fx) + Sample(y1, x1) * fx;
                    result[oc * height * width + y * width + x] = top * (1 - fy) + bottom * fy;
                }
            }
        }

        return result;
    }

    private static ReadOnlySpan<byte> PngSignature => [0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A];

    // The frame header (SOF0..SOF15 but DHT, JPG and DAC) holds the size and the number of components.
    private static ImageInfo? Jpeg(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);
        stream.Position = 2;
        while (stream.Position + 4 <= stream.Length)
        {
            if (reader.ReadByte() != 0xFF)
            {
                continue;
            }

            int marker = reader.ReadByte();
            if (marker is 0xD8 or 0x01 or >= 0xD0 and <= 0xD7 or 0xFF)
            {
                continue;
            }

            int size = (reader.ReadByte() << 8) | reader.ReadByte();
            if (marker is >= 0xC0 and <= 0xCF and not 0xC4 and not 0xC8 and not 0xCC)
            {
                reader.ReadByte();                                                      // sample precision
                int h = (reader.ReadByte() << 8) | reader.ReadByte(), w = (reader.ReadByte() << 8) | reader.ReadByte();
                int components = reader.ReadByte();
                return new ImageInfo(w, h, components, "jpeg", false);
            }

            stream.Position += size - 2;
        }

        return null;
    }

    private static int NetpbmNumber(ReadOnlySpan<byte> b, ref int position)
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
        while (position < b.Length && b[position] is >= (byte)'0' and <= (byte)'9')
        {
            value = value * 10 + (b[position++] - '0');
        }

        return position > start ? value : -1;
    }

    private static (float[], int, int, int)? DecodeNetpbm(byte[] bytes)
    {
        int position = 2;
        int w = NetpbmNumber(bytes, ref position), h = NetpbmNumber(bytes, ref position), max = NetpbmNumber(bytes, ref position);
        if (w <= 0 || h <= 0 || max <= 0)
        {
            return null;
        }

        int c = bytes[1] is (byte)'3' or (byte)'6' ? 3 : 1;
        bool binary = bytes[1] is (byte)'5' or (byte)'6';
        var pixels = new float[c * h * w];
        position++;                                                                     // the single whitespace after the maximum
        int wide = max > 255 ? 2 : 1;
        for (int i = 0; i < h * w; i++)
        {
            for (int ch = 0; ch < c; ch++)
            {
                int value;
                if (binary)
                {
                    int at = position + (i * c + ch) * wide;
                    if (at + wide > bytes.Length)
                    {
                        return null;
                    }

                    value = wide == 1 ? bytes[at] : (bytes[at] << 8) | bytes[at + 1];
                }
                else
                {
                    value = NetpbmNumber(bytes, ref position);
                }

                pixels[ch * h * w + i] = value / (float)max;
            }
        }

        return (pixels, c, h, w);
    }

    private static (float[], int, int, int)? DecodeBmp(byte[] bytes)
    {
        int offset = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(10));
        int w = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(18)), rawHeight = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(22));
        int bits = BinaryPrimitives.ReadInt16LittleEndian(bytes.AsSpan(28)), compression = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(30));
        if (bits is not (24 or 32) || compression is not (0 or 3))
        {
            return null;
        }

        int h = Math.Abs(rawHeight), bytesPerPixel = bits / 8, stride = (w * bytesPerPixel + 3) & ~3;
        var pixels = new float[3 * h * w];
        for (int y = 0; y < h; y++)
        {
            int row = offset + (rawHeight > 0 ? h - 1 - y : y) * stride;            // bottom-up unless the height is negative
            for (int x = 0; x < w; x++)
            {
                int at = row + x * bytesPerPixel;
                if (at + 2 >= bytes.Length)
                {
                    return null;
                }

                pixels[0 * h * w + y * w + x] = bytes[at + 2] / 255f;                  // stored as blue, green, red
                pixels[1 * h * w + y * w + x] = bytes[at + 1] / 255f;
                pixels[2 * h * w + y * w + x] = bytes[at] / 255f;
            }
        }

        return (pixels, 3, h, w);
    }

    private static (float[], int, int, int)? DecodePng(byte[] bytes)
    {
        int position = 8, w = 0, h = 0, color = 0, depth = 0, interlace = 0;
        byte[]? palette = null;
        using var compressed = new MemoryStream();
        while (position + 8 <= bytes.Length)
        {
            int length = BinaryPrimitives.ReadInt32BigEndian(bytes.AsSpan(position));
            string type = Encoding.ASCII.GetString(bytes, position + 4, 4);
            var data = bytes.AsSpan(position + 8, Math.Min(length, bytes.Length - position - 8));
            switch (type)
            {
                case "IHDR":
                    w = BinaryPrimitives.ReadInt32BigEndian(data);
                    h = BinaryPrimitives.ReadInt32BigEndian(data[4..]);
                    depth = data[8];
                    color = data[9];
                    interlace = data[12];
                    break;
                case "PLTE":
                    palette = data.ToArray();
                    break;
                case "IDAT":
                    compressed.Write(data);
                    break;
            }

            if (type == "IEND")
            {
                break;
            }

            position += 12 + length;                                                    // length, type, data, CRC
        }

        int samples = color switch { 0 => 1, 2 => 3, 3 => 1, 4 => 2, 6 => 4, _ => 0 };
        if (depth != 8 || interlace != 0 || samples == 0 || w <= 0 || h <= 0 || color == 3 && palette is null)
        {
            return null;
        }

        int stride = w * samples;
        var raw = new byte[h * stride];
        var previous = new byte[stride];
        compressed.Position = 0;
        using (var inflate = new ZLibStream(compressed, CompressionMode.Decompress))
        {
            var line = new byte[stride + 1];
            for (int y = 0; y < h; y++)
            {
                inflate.ReadExactly(line);
                Unfilter(line[0], line.AsSpan(1), previous, samples);
                line.AsSpan(1).CopyTo(raw.AsSpan(y * stride));
                line.AsSpan(1).CopyTo(previous);
            }
        }

        int channels = color is 2 or 3 or 6 ? 3 : 1;
        var pixels = new float[channels * h * w];
        for (int i = 0; i < h * w; i++)
        {
            for (int ch = 0; ch < channels; ch++)
            {
                byte value = color == 3 ? palette![Math.Min(raw[i] * 3 + ch, palette.Length - 1)] : raw[i * samples + ch];
                pixels[ch * h * w + i] = value / 255f;
            }
        }

        return (pixels, channels, h, w);
    }

    // The PNG filters (none, sub, up, average, Paeth) undone in place, with bytes per pixel = samples at 8 bits.
    private static void Unfilter(byte filter, Span<byte> line, ReadOnlySpan<byte> previous, int bpp)
    {
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
