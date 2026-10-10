// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
using Idrak.Data.Abstractions;

namespace Idrak.Data;

// The library's image encoders (ImageEncoders): 8-bit levels from values in [0, 1], rounded.
internal static class ImageLevels
{
    public static byte Level(float value) => (byte)Math.Clamp((int)MathF.Round(value * 255f), 0, 255);

    // The pixels row by row, channels interleaved (grey: one byte a pixel, colour: three).
    public static byte[] Interleaved(ImageData image)
    {
        if (image.Channels is not (1 or 3))
        {
            throw new NotSupportedException($"An image of {image.Channels} channels: the encoders write grey (1) or colour (3).");
        }

        int plane = image.Width * image.Height, channels = image.Channels;
        var bytes = new byte[plane * channels];
        var pixels = image.Pixels;
        for (int c = 0; c < channels; c++)
        {
            for (int i = 0; i < plane; i++)
            {
                bytes[i * channels + c] = Level(pixels[c * plane + i]);
            }
        }

        return bytes;
    }
}

/// <summary>"png": 8-bit grey (colour type 0) or RGB (type 2), no interlacing, every row unfiltered, deflated with .NET's zlib.</summary>
internal sealed class PngEncoder : IImageEncoder
{
    private static readonly uint[] CrcTable = MakeCrcTable();

    public string Name => "png";

    public IReadOnlyCollection<string> Extensions { get; } = [".png"];

    public byte[] Encode(ImageData image)
    {
        ArgumentNullException.ThrowIfNull(image);
        var pixels = ImageLevels.Interleaved(image);
        int row = image.Width * image.Channels;
        var raw = new byte[(row + 1) * image.Height];                       // a filter byte (0: none) before each row
        for (int y = 0; y < image.Height; y++)
        {
            pixels.AsSpan(y * row, row).CopyTo(raw.AsSpan(y * (row + 1) + 1));
        }

        using var compressed = new MemoryStream();
        using (var zlib = new ZLibStream(compressed, CompressionLevel.Optimal, leaveOpen: true))
        {
            zlib.Write(raw);
        }

        using var file = new MemoryStream();
        file.Write([0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        var header = new byte[13];
        BinaryPrimitives.WriteInt32BigEndian(header, image.Width);
        BinaryPrimitives.WriteInt32BigEndian(header.AsSpan(4), image.Height);
        header[8] = 8;                                                       // bit depth
        header[9] = (byte)(image.Channels == 1 ? 0 : 2);                     // grey or RGB
        Chunk(file, "IHDR", header);
        Chunk(file, "IDAT", compressed.ToArray());
        Chunk(file, "IEND", []);
        return file.ToArray();
    }

    private static void Chunk(Stream file, string type, ReadOnlySpan<byte> data)
    {
        Span<byte> number = stackalloc byte[4];
        BinaryPrimitives.WriteInt32BigEndian(number, data.Length);
        file.Write(number);
        var name = Encoding.ASCII.GetBytes(type);
        file.Write(name);
        file.Write(data);
        uint crc = Crc(Crc(0xFFFFFFFFu, name), data) ^ 0xFFFFFFFFu;
        BinaryPrimitives.WriteUInt32BigEndian(number, crc);
        file.Write(number);
    }

    // CRC-32 (ISO 3309), as PNG's chunks carry it.
    private static uint Crc(uint crc, ReadOnlySpan<byte> bytes)
    {
        foreach (byte b in bytes)
        {
            crc = CrcTable[(crc ^ b) & 0xFF] ^ (crc >> 8);
        }

        return crc;
    }

    private static uint[] MakeCrcTable()
    {
        var table = new uint[256];
        for (uint n = 0; n < 256; n++)
        {
            uint c = n;
            for (int k = 0; k < 8; k++)
            {
                c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            }

            table[n] = c;
        }

        return table;
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
        var header = Encoding.ASCII.GetBytes($"{(image.Channels == 1 ? "P5" : "P6")}\n{image.Width} {image.Height}\n255\n");
        return [.. header, .. pixels];
    }
}
