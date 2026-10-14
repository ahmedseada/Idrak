// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Numerics;
using System.Text;

namespace Idrak.Imaging;

// The library's image encoders (ImageEncoders): 8-bit levels from values in [0, 1], rounded.
internal static class ImageLevels
{
    public static byte Level(float value) => (byte)Math.Clamp((int)MathF.Round(value * 255f), 0, 255);

    /// <summary>Each byte's value as the decoders give it, k x (1/255) in float32 (ToUnit's values), for 8-bit images.</summary>
    public static float[] UnitLevels { get; } = [.. Enumerable.Range(0, 256).Select(k => k * (1f / 255f))];

    /// <summary>Each byte's value as k / 255 in float32 (the PNG decoder's 8-bit values), for 8-bit images.</summary>
    public static float[] DividedLevels { get; } = [.. Enumerable.Range(0, 256).Select(k => k / 255f)];

    // Bytes as values in [0, 1]: each byte times 1/255 in float32, as the decoders give them; a vector loop (bytes widened
    // to whole numbers, which convert exactly) and a scalar tail.
    public static void ToUnit(ReadOnlySpan<byte> bytes, Span<float> values)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated && bytes.Length >= Vector<byte>.Count)
        {
            var scale = new Vector<float>(1f / 255f);
            int step = Vector<byte>.Count, quarter = Vector<int>.Count;
            for (; i <= bytes.Length - step; i += step)
            {
                Vector.Widen(new Vector<byte>(bytes[i..]), out var low, out var high);
                Vector.Widen(low, out var a, out var b);
                Vector.Widen(high, out var c, out var d);
                (Vector.ConvertToSingle(Vector.AsVectorInt32(a)) * scale).CopyTo(values[i..]);
                (Vector.ConvertToSingle(Vector.AsVectorInt32(b)) * scale).CopyTo(values[(i + quarter)..]);
                (Vector.ConvertToSingle(Vector.AsVectorInt32(c)) * scale).CopyTo(values[(i + 2 * quarter)..]);
                (Vector.ConvertToSingle(Vector.AsVectorInt32(d)) * scale).CopyTo(values[(i + 3 * quarter)..]);
            }
        }

        for (; i < bytes.Length; i++)
        {
            values[i] = bytes[i] * (1f / 255f);
        }
    }

    // The pixels row by row, channels interleaved (grey: one byte a pixel, colour: three).
    public static byte[] Interleaved(ImageData image)
    {
        if (image.Channels is not (1 or 3))
        {
            throw new NotSupportedException($"An image of {image.Channels} channels: the encoders write grey (1) or colour (3).");
        }

        int plane = image.Width * image.Height, channels = image.Channels;
        var bytes = GC.AllocateUninitializedArray<byte>(plane * channels);  // every byte written below
        if (image.TryGetPlanes(out var planes, out _))
        {
            // An 8-bit image: its bytes (Level of their values gives them back), interleaved; nothing made in floats.
            if (channels == 1)
            {
                planes[0].Span.CopyTo(bytes);
                return bytes;
            }

            HostParallel.For(plane, 1 << 15, (first, last) =>
            {
                for (int c = 0; c < channels; c++)
                {
                    var source = planes[c].Span;
                    for (int i = first; i < last; i++)
                    {
                        bytes[i * channels + c] = source[i];
                    }
                }
            });
            return bytes;
        }

        var pixels = image.Pixels;
        HostParallel.For(plane, 1 << 15, (first, last) =>
        {
            for (int c = 0; c < channels; c++)
            {
                for (int i = first; i < last; i++)
                {
                    bytes[i * channels + c] = Level(pixels[c * plane + i]);
                }
            }
        });

        return bytes;
    }
}

/// <summary>
/// "png": 8-bit grey (colour type 0) or RGB (type 2), no interlacing, every row unfiltered, deflated with .NET's zlib at
/// its fastest level. Measured on a scan-like 1,024 x 1,448 grey page: unfiltered rows at the fastest level take 13 ms
/// and 910 KiB, at the optimal level 44 ms and 577 KiB; the "Up" filter turns paper grain into larger differences
/// (19 ms, 1,138 KiB) and pays only on flat images, which are small either way. Speed first: the fastest level.
/// </summary>
internal sealed class PngEncoder : IImageEncoder
{
    // CRC-32 (ISO 3309) tables for eight bytes at a time (slicing-by-8): table k gives a byte's effect k bytes on.
    private static readonly uint[] CrcTables = MakeCrcTables();

    public string Name => "png";

    public IReadOnlyCollection<string> Extensions { get; } = [".png"];

    public byte[] Encode(ImageData image)
    {
        ArgumentNullException.ThrowIfNull(image);
        var pixels = ImageLevels.Interleaved(image);
        int row = image.Width * image.Channels, height = image.Height;
        var raw = GC.AllocateUninitializedArray<byte>((row + 1) * height);  // a filter byte (0: none) before each row, then the row
        HostParallel.For(height, 64, (first, last) =>
        {
            for (int y = first; y < last; y++)
            {
                raw[y * (row + 1)] = 0;                                  // filter None
                pixels.AsSpan(y * row, row).CopyTo(raw.AsSpan(y * (row + 1) + 1));
            }
        });

        using var file = new MemoryStream(raw.Length / 4 + 64);
        file.Write([0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        Span<byte> header = stackalloc byte[13];
        BinaryPrimitives.WriteInt32BigEndian(header, image.Width);
        BinaryPrimitives.WriteInt32BigEndian(header[4..], height);
        header[8] = 8;                                                       // bit depth
        header[9] = (byte)(image.Channels == 1 ? 0 : 2);                     // grey or RGB
        header[10..].Clear();                                                // deflate, adaptive filters, no interlacing
        Chunk(file, "IHDR"u8, header);
        using (var compressed = new MemoryStream(raw.Length / 4 + 64))
        {
            using (var zlib = new ZLibStream(compressed, CompressionLevel.Fastest, leaveOpen: true))
            {
                zlib.Write(raw);
            }

            Chunk(file, "IDAT"u8, compressed.GetBuffer().AsSpan(0, (int)compressed.Length));
        }

        Chunk(file, "IEND"u8, []);
        return file.ToArray();
    }

    private static void Chunk(Stream file, ReadOnlySpan<byte> type, ReadOnlySpan<byte> data)
    {
        Span<byte> number = stackalloc byte[4];
        BinaryPrimitives.WriteInt32BigEndian(number, data.Length);
        file.Write(number);
        file.Write(type);
        file.Write(data);
        uint crc = Crc(Crc(0xFFFFFFFFu, type), data) ^ 0xFFFFFFFFu;
        BinaryPrimitives.WriteUInt32BigEndian(number, crc);
        file.Write(number);
    }

    // CRC-32 as PNG's chunks carry it: eight bytes a step through the sliced tables, then a byte at a time.
    internal static uint Crc(uint crc, ReadOnlySpan<byte> bytes)
    {
        var t = CrcTables;
        int i = 0;
        for (; i + 8 <= bytes.Length; i += 8)
        {
            uint low = crc ^ BinaryPrimitives.ReadUInt32LittleEndian(bytes[i..]);
            uint high = BinaryPrimitives.ReadUInt32LittleEndian(bytes[(i + 4)..]);
            crc = t[7 * 256 + (low & 0xFF)] ^ t[6 * 256 + ((low >> 8) & 0xFF)] ^ t[5 * 256 + ((low >> 16) & 0xFF)] ^ t[4 * 256 + (low >> 24)]
                  ^ t[3 * 256 + (high & 0xFF)] ^ t[2 * 256 + ((high >> 8) & 0xFF)] ^ t[256 + ((high >> 16) & 0xFF)] ^ t[high >> 24];
        }

        for (; i < bytes.Length; i++)
        {
            crc = t[(crc ^ bytes[i]) & 0xFF] ^ (crc >> 8);
        }

        return crc;
    }

    private static uint[] MakeCrcTables()
    {
        var tables = new uint[8 * 256];
        for (uint n = 0; n < 256; n++)
        {
            uint c = n;
            for (int k = 0; k < 8; k++)
            {
                c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            }

            tables[n] = c;
        }

        for (int k = 1; k < 8; k++)
        {
            for (int n = 0; n < 256; n++)
            {
                uint previous = tables[(k - 1) * 256 + n];
                tables[k * 256 + n] = tables[previous & 0xFF] ^ (previous >> 8);
            }
        }

        return tables;
    }
}

/// <summary>"netpbm": binary PGM (P5) for a grey image, PPM (P6) for a colour one, 255 levels.</summary>
internal sealed class NetpbmEncoder : IImageEncoder
{
    public string Name => "netpbm";

    public IReadOnlyCollection<string> Extensions { get; } = [".pgm", ".ppm", ".pnm"];

    public byte[] Encode(ImageData image)
    {
        ArgumentNullException.ThrowIfNull(image);
        var pixels = ImageLevels.Interleaved(image);
        var header = Encoding.ASCII.GetBytes(string.Create(CultureInfo.InvariantCulture, $"{(image.Channels == 1 ? "P5" : "P6")}\n{image.Width} {image.Height}\n255\n"));
        return [.. header, .. pixels];
    }
}
