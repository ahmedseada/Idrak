// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Text;
using Idrak.Data.Abstractions;

namespace Idrak.Data;

/// <summary>
/// A chat message's image (<see cref="ChatImage"/>: the encoded bytes as a client sent them) read the way transformers'
/// <c>load_image</c> reads a file: decoded by the registered codecs (<see cref="ImageCodecs"/>) and turned upright by its
/// EXIF orientation (<c>ImageOps.exif_transpose</c>). The codecs and <see cref="ImagePreprocessor"/> apply no orientation
/// themselves; the command line's images and the server's uploads and data URLs come through here.
/// </summary>
public static class ChatImageDecoder
{
    /// <summary>The image's pixels: decoded by the codec that reads its bytes, then turned by its EXIF orientation.</summary>
    /// <exception cref="InvalidDataException">No registered codec reads it, or its data is damaged.</exception>
    public static ImageData Decode(ChatImage image)
    {
        ArgumentNullException.ThrowIfNull(image);
        return Decode(image.Data.Span);
    }

    /// <summary>An image file held in memory, decoded and turned upright (see <see cref="Decode(ChatImage)"/>).</summary>
    /// <exception cref="InvalidDataException">No registered codec reads it, or its data is damaged.</exception>
    public static ImageData Decode(ReadOnlySpan<byte> file) => Orient(ImageCodecs.Decode(file), Orientation(file));

    /// <summary>
    /// The name of the registered codec that reads the image's bytes (for example "png" or "jpeg"), or null when none
    /// does (a GIF or WebP file, or bytes that are no image): a check of the header, before anything is decoded.
    /// </summary>
    public static string? FormatOf(ChatImage image)
    {
        ArgumentNullException.ThrowIfNull(image);
        var header = image.Data.Span[..Math.Min(image.Data.Length, ImageCodecs.HeaderBytes)];
        foreach (string name in ImageCodecs.Names)
        {
            if (ImageCodecs.Get(name).ReadInfo(header) is not null)
            {
                return name;
            }
        }

        return null;
    }

    /// <summary>
    /// The image upright and grey, as an 8-bit binary PGM: each pixel Pillow's <c>convert("L")</c> of its 8-bit red, green
    /// and blue ((19595 R + 38470 G + 7471 B + 32768) &gt;&gt; 16; a grey image keeps its values). A preprocessor reads it
    /// exactly as it reads the original with <see cref="ImagePreprocessor.Grayscale"/> on, so a server can turn one
    /// request's images grey without changing the model's preprocessing.
    /// </summary>
    /// <exception cref="InvalidDataException">No registered codec reads it, or its data is damaged.</exception>
    public static ChatImage Grayscale(ChatImage image)
    {
        var pixels = Decode(image);
        var grey = ImagePreprocessor.ToBytes(pixels, grayscale: true)[0];
        byte[] header = Encoding.ASCII.GetBytes($"P5\n{pixels.Width} {pixels.Height}\n255\n");
        var file = new byte[header.Length + grey.Length];
        header.CopyTo(file, 0);
        grey.CopyTo(file, header.Length);
        return ChatImage.FromBytes(file, "image/x-portable-graymap");
    }

    /// <summary>
    /// The image grey, as Pillow's <c>convert("L")</c> of its 8-bit red, green and blue ((19595 R + 38470 G + 7471 B +
    /// 32768) &gt;&gt; 16; a grey image keeps its values), one channel: the library's <c>grayscale</c> image transform.
    /// </summary>
    public static ImageData Grayscale(ImageData image)
    {
        ArgumentNullException.ThrowIfNull(image);
        return image.Channels == 1 ? image : PillowImageOps.FromBytes(ImagePreprocessor.ToBytes(image, grayscale: true), image.Height, image.Width);
    }

    /// <summary>
    /// The EXIF orientation of an image file (1 to 8; 1 when it has none): tag 0x0112 of the first IFD, read from a JPEG's
    /// APP1 "Exif" segment or a PNG's eXIf chunk.
    /// </summary>
    public static int Orientation(ReadOnlySpan<byte> file)
    {
        try
        {
            if (file.Length > 4 && file[0] == 0xFF && file[1] == 0xD8)
            {
                int at = 2;
                while (at + 4 <= file.Length && file[at] == 0xFF)
                {
                    int marker = file[at + 1];
                    if (marker is 0xD8 or 0x01 or >= 0xD0 and <= 0xD7 or 0xFF)
                    {
                        at += marker == 0xFF ? 1 : 2;                               // fill bytes and markers without a length
                        continue;
                    }

                    if (marker is 0xDA or 0xD9)
                    {
                        break;                                                      // the scan: no EXIF after it
                    }

                    int length = BinaryPrimitives.ReadUInt16BigEndian(file[(at + 2)..]);
                    var segment = file.Slice(at + 4, Math.Min(length - 2, file.Length - at - 4));
                    if (marker == 0xE1 && segment.StartsWith("Exif\0\0"u8))
                    {
                        return Tiff(segment[6..]);
                    }

                    at += 2 + length;
                }
            }
            else if (file.Length > 8 && file[..8].SequenceEqual((ReadOnlySpan<byte>)[0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A]))
            {
                int at = 8;
                while (at + 8 <= file.Length)
                {
                    int length = (int)BinaryPrimitives.ReadUInt32BigEndian(file[at..]);
                    var type = file.Slice(at + 4, 4);
                    if (length < 0 || at + 8 + length > file.Length || type.SequenceEqual("IEND"u8))
                    {
                        break;
                    }

                    if (type.SequenceEqual("eXIf"u8))
                    {
                        var data = file.Slice(at + 8, length);
                        return Tiff(data.StartsWith("Exif\0\0"u8) ? data[6..] : data);
                    }

                    at += 12 + length;
                }
            }
        }
        catch (ArgumentOutOfRangeException)
        {
            // A damaged segment: no orientation.
        }

        return 1;
    }

    // Tag 0x0112 of a TIFF structure's first IFD (a SHORT, or a LONG as some writers store it); 1 when absent or out of range.
    private static int Tiff(ReadOnlySpan<byte> tiff)
    {
        if (tiff.Length < 8)
        {
            return 1;
        }

        bool little = tiff[0] == 'I' && tiff[1] == 'I';
        if (!little && !(tiff[0] == 'M' && tiff[1] == 'M'))
        {
            return 1;
        }

        long ifd = U32(tiff, 4, little);
        if (ifd < 8 || ifd + 2 > tiff.Length)
        {
            return 1;
        }

        int entries = U16(tiff, (int)ifd, little);
        for (int i = 0; i < entries; i++)
        {
            int entry = (int)ifd + 2 + 12 * i;
            if (entry + 12 > tiff.Length)
            {
                break;
            }

            if (U16(tiff, entry, little) != 0x0112)
            {
                continue;
            }

            long value = U16(tiff, entry + 2, little) switch { 3 => U16(tiff, entry + 8, little), 4 => U32(tiff, entry + 8, little), _ => 1 };
            return value is >= 1 and <= 8 ? (int)value : 1;
        }

        return 1;
    }

    private static int U16(ReadOnlySpan<byte> data, int at, bool little) =>
        little ? BinaryPrimitives.ReadUInt16LittleEndian(data[at..]) : BinaryPrimitives.ReadUInt16BigEndian(data[at..]);

    private static long U32(ReadOnlySpan<byte> data, int at, bool little) =>
        little ? BinaryPrimitives.ReadUInt32LittleEndian(data[at..]) : BinaryPrimitives.ReadUInt32BigEndian(data[at..]);

    /// <summary>
    /// The image turned upright for EXIF orientation <paramref name="orientation"/>, as Pillow's <c>exif_transpose</c>:
    /// 2 mirrored, 3 turned half a turn, 4 flipped, 5 transposed, 6 turned a quarter clockwise, 7 transversed, 8 turned a
    /// quarter anticlockwise; 1 and anything else unchanged.
    /// </summary>
    public static ImageData Orient(ImageData image, int orientation)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (orientation is < 2 or > 8)
        {
            return image;
        }

        int c = image.Channels, h = image.Height, w = image.Width;
        bool swap = orientation >= 5;
        int oh = swap ? w : h, ow = swap ? h : w;
        var source = image.Pixels;
        var pixels = new float[source.Length];
        for (int y = 0; y < oh; y++)
        {
            for (int x = 0; x < ow; x++)
            {
                // The source pixel (row, column) shown at (y, x).
                var (sy, sx) = orientation switch
                {
                    2 => (y, w - 1 - x),
                    3 => (h - 1 - y, w - 1 - x),
                    4 => (h - 1 - y, x),
                    5 => (x, y),
                    6 => (h - 1 - x, y),
                    7 => (h - 1 - x, w - 1 - y),
                    _ => (x, w - 1 - y),
                };
                for (int k = 0; k < c; k++)
                {
                    pixels[(k * oh + y) * ow + x] = source[(k * h + sy) * w + sx];
                }
            }
        }

        return new ImageData(pixels, c, oh, ow);
    }
}
