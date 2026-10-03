// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;

namespace Idrak.Cli.Shared;

/// <summary>
/// Image files for <c>idrak train</c> and <c>idrak predict</c> on image folders: PNG (8-bit grey, grey with alpha, RGB,
/// RGBA and palette images, not interlaced) and Netpbm (.pgm, .ppm, .pnm; text or binary, up to 8 bits), decoded into
/// [channels, height, width] values in [0, 1] and fitted to a network's input (channels converted, size by nearest
/// neighbour). The library has no image decoding; this covers the common lossless formats without a dependency.
/// </summary>
internal static class Images
{
    /// <summary>The extensions read.</summary>
    public static readonly string[] Extensions = [".png", ".pgm", ".ppm", ".pnm"];

    /// <summary>Whether <paramref name="path"/> is an image file this reads.</summary>
    public static bool IsImage(string path) => Extensions.Contains(Path.GetExtension(path).ToLowerInvariant());

    /// <summary>The image as [channels, height, width] floats in [0, 1], with <paramref name="channels"/> channels (1 or 3) and the given size.</summary>
    public static float[] Load(string path, int channels, int height, int width)
    {
        var (w, h, c, pixels) = Decode(path);
        var result = new float[channels * height * width];
        for (int y = 0; y < height; y++)
        {
            int sy = y * h / height;
            for (int x = 0; x < width; x++)
            {
                int sx = x * w / width, source = (sy * w + sx) * c;
                float grey = c >= 3 ? (pixels[source] + pixels[source + 1] + pixels[source + 2]) / 3f : pixels[source];
                for (int k = 0; k < channels; k++)
                {
                    float value = channels == 1 ? grey : c >= 3 ? pixels[source + Math.Min(k, 2)] : grey;
                    result[(k * height + y) * width + x] = value / 255f;
                }
            }
        }

        return result;
    }

    /// <summary>Width, height, channels (1, 2, 3 or 4) and the 8-bit samples, row by row.</summary>
    public static (int Width, int Height, int Channels, byte[] Pixels) Decode(string path)
    {
        byte[] data = File.ReadAllBytes(path);
        return data.Length > 8 && data[0] == 0x89 && data[1] == (byte)'P' && data[2] == (byte)'N' && data[3] == (byte)'G' ? Png(data, path)
            : data.Length > 2 && data[0] == (byte)'P' && data[1] is (byte)'2' or (byte)'3' or (byte)'5' or (byte)'6' ? Netpbm(data, path)
            : throw new InvalidDataException($"{path}: not a PNG or Netpbm (.pgm, .ppm) image.");
    }

    private static (int, int, int, byte[]) Png(byte[] data, string path)
    {
        int width = 0, height = 0, depth = 0, type = 0, interlace = 0;
        byte[]? palette = null;
        using var compressed = new MemoryStream();
        for (int at = 8; at + 8 <= data.Length;)
        {
            int length = BinaryPrimitives.ReadInt32BigEndian(data.AsSpan(at));
            string chunk = Encoding.ASCII.GetString(data, at + 4, 4);
            var body = data.AsSpan(at + 8, length);
            switch (chunk)
            {
                case "IHDR":
                    (width, height, depth, type, interlace) = (BinaryPrimitives.ReadInt32BigEndian(body), BinaryPrimitives.ReadInt32BigEndian(body[4..]), body[8], body[9], body[12]);
                    break;
                case "PLTE":
                    palette = body.ToArray();
                    break;
                case "IDAT":
                    compressed.Write(body);
                    break;
            }

            if (chunk == "IEND")
            {
                break;
            }

            at += 12 + length;
        }

        if (depth != 8 || interlace != 0 || type is not (0 or 2 or 3 or 4 or 6))
        {
            throw new InvalidDataException($"{path}: PNG with {depth}-bit samples, colour type {type}{(interlace != 0 ? ", interlaced" : "")} is not read (8-bit, not interlaced PNGs are).");
        }

        int channels = type switch { 0 => 1, 2 => 3, 3 => 1, 4 => 2, _ => 4 };
        int stride = width * channels;
        var raw = new byte[height * stride];
        compressed.Position = 0;
        using (var inflate = new ZLibStream(compressed, CompressionMode.Decompress))
        {
            var previous = new byte[stride];
            var line = new byte[stride];
            for (int y = 0; y < height; y++)
            {
                int filter = inflate.ReadByte();
                inflate.ReadExactly(line);
                for (int i = 0; i < stride; i++)
                {
                    int left = i >= channels ? line[i - channels] : 0, up = previous[i], corner = i >= channels ? previous[i - channels] : 0;
                    line[i] = (byte)(line[i] + filter switch
                    {
                        1 => left,
                        2 => up,
                        3 => (left + up) / 2,
                        4 => Paeth(left, up, corner),
                        _ => 0,
                    });
                }

                line.CopyTo(raw, y * stride);
                (previous, line) = (line, previous);
            }
        }

        if (type != 3)
        {
            return (width, height, channels, raw);
        }

        // Palette images: each index becomes its RGB colour.
        if (palette is null)
        {
            throw new InvalidDataException($"{path}: palette PNG without a palette.");
        }

        var rgb = new byte[width * height * 3];
        for (int i = 0; i < raw.Length; i++)
        {
            Array.Copy(palette, raw[i] * 3, rgb, i * 3, 3);
        }

        return (width, height, 3, rgb);
    }

    private static int Paeth(int a, int b, int c)
    {
        int p = a + b - c, pa = Math.Abs(p - a), pb = Math.Abs(p - b), pc = Math.Abs(p - c);
        return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
    }

    private static (int, int, int, byte[]) Netpbm(byte[] data, string path)
    {
        char kind = (char)data[1];
        int at = 2;
        int Number()
        {
            while (at < data.Length && (char.IsWhiteSpace((char)data[at]) || data[at] == '#'))
            {
                if (data[at] == '#')
                {
                    while (at < data.Length && data[at] != '\n')
                    {
                        at++;
                    }
                }
                else
                {
                    at++;
                }
            }

            int value = 0, start = at;
            while (at < data.Length && data[at] is >= (byte)'0' and <= (byte)'9')
            {
                value = value * 10 + (data[at++] - '0');
            }

            return at > start ? value : throw new InvalidDataException($"{path}: malformed Netpbm header.");
        }

        int width = Number(), height = Number(), max = Number();
        int channels = kind is '3' or '6' ? 3 : 1;
        var pixels = new byte[width * height * channels];
        if (kind is '5' or '6')
        {
            if (max > 255)
            {
                throw new InvalidDataException($"{path}: 16-bit Netpbm images are not read.");
            }

            at++;                                                                   // the single whitespace after the header
            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = (byte)(data[at + i] * 255 / Math.Max(1, max));
            }
        }
        else
        {
            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = (byte)(Number() * 255 / Math.Max(1, max));
            }
        }

        return (width, height, channels, pixels);
    }
}
