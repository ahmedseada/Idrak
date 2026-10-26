// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;

namespace Idrak.Imaging;

/// <summary>
/// A chat message's image (<see cref="ChatImage"/>: the encoded bytes as a client sent them) read the way transformers'
/// <c>load_image</c> reads a file: decoded by the registered codecs (<see cref="ImageCodecs"/>) and turned upright by its
/// EXIF orientation (<c>ImageOps.exif_transpose</c>; an 8-bit image stays 8-bit). The codecs and <see cref="ImagePreprocessor"/> apply no orientation
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
    /// The width and height of an image file held in memory as <see cref="Decode(ReadOnlySpan{byte})"/> gives it (turned
    /// upright: swapped for the EXIF orientations 5 to 8), read from its header without decoding it; null when no
    /// registered codec knows its format or its header gives no size.
    /// </summary>
    public static (int Width, int Height)? Size(ReadOnlySpan<byte> file) =>
        ImageCodecs.ReadInfo(file) is { Width: > 0, Height: > 0 } info
            ? Orientation(file) >= 5 ? (info.Height, info.Width) : (info.Width, info.Height)
            : null;

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
    /// quarter anticlockwise; 1 and anything else unchanged. An image held as 8-bit planes (what the JPEG and 8-bit PNG
    /// decoders give) stays 8-bit: its planes are turned, so a phone's rotated photo is not widened to floats here (a
    /// quarter of the memory); a float image's values are moved the same way. The turns walk the image in tiles, so the
    /// side read across its rows is read a cache line at a time (rule 72), the tiles' rows spread over the cores.
    /// </summary>
    public static ImageData Orient(ImageData image, int orientation)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (orientation is < 2 or > 8)
        {
            return image;
        }

        int c = image.Channels, h = image.Height, w = image.Width, size = h * w;
        bool swap = orientation >= 5;
        int oh = swap ? w : h, ow = swap ? h : w;
        if (image.TryGetPlanes(out var planes, out var levels))
        {
            var turned = new byte[c][];
            for (int k = 0; k < c; k++)
            {
                turned[k] = GC.AllocateUninitializedArray<byte>(size);          // every byte written by the turn
                Turn(planes[k], turned[k], h, w, orientation);
            }

            float[] table = MemoryMarshal.TryGetArray(levels, out var held) && held.Offset == 0 && held.Count == held.Array!.Length ? held.Array : levels.ToArray();
            return new ImageData(turned, oh, ow, table);
        }

        var source = image.Pixels;
        var pixels = GC.AllocateUninitializedArray<float>(source.Length);     // every value written by the turn
        for (int k = 0; k < c; k++)
        {
            Turn(source.AsMemory(k * size, size), pixels.AsMemory(k * size, size), h, w, orientation);
        }

        return new ImageData(pixels, c, oh, ow);
    }

    // The side of a tile of the quarter turns (5 to 8): 64 values a row of it, 64 rows read across.
    private const int TurnTile = 64;

    // One plane (h x w) turned into `to` (w x h for 5 to 8): each output pixel (y, x) takes the source pixel exif_transpose
    // puts there. Mirrors and flips (2 to 4) copy whole rows; the quarter turns go tile by tile.
    private static void Turn<T>(ReadOnlyMemory<T> from, Memory<T> to, int h, int w, int orientation)
    {
        if (orientation <= 4)
        {
            HostParallel.For(h, 64, (first, last) =>
            {
                var source = from.Span;
                var target = to.Span;
                for (int y = first; y < last; y++)
                {
                    int sy = orientation == 2 ? y : h - 1 - y;
                    var row = target.Slice(y * w, w);
                    source.Slice(sy * w, w).CopyTo(row);
                    if (orientation != 4)
                    {
                        row.Reverse();                                          // 2 and 3: the columns reversed
                    }
                }
            });
            return;
        }

        int oh = w, ow = h;
        int tiles = (oh + TurnTile - 1) / TurnTile;
        HostParallel.For(tiles, 1, (firstTile, lastTile) =>
        {
            var source = from.Span;
            var target = to.Span;
            for (int ty = firstTile * TurnTile; ty < Math.Min(oh, lastTile * TurnTile); ty += TurnTile)
            {
                int yEnd = Math.Min(oh, ty + TurnTile);
                for (int tx = 0; tx < ow; tx += TurnTile)
                {
                    int xEnd = Math.Min(ow, tx + TurnTile);
                    for (int y = ty; y < yEnd; y++)
                    {
                        // Output (y, x) reads source column sx = y (5, 6) or w - 1 - y (7, 8), and row x (5, 8) or h - 1 - x (6, 7).
                        int sx = orientation is 5 or 6 ? y : w - 1 - y;
                        var row = target.Slice(y * ow, ow);
                        if (orientation is 5 or 8)
                        {
                            for (int x = tx; x < xEnd; x++)
                            {
                                row[x] = source[x * w + sx];
                            }
                        }
                        else
                        {
                            for (int x = tx; x < xEnd; x++)
                            {
                                row[x] = source[(h - 1 - x) * w + sx];
                            }
                        }
                    }
                }
            }
        });
    }
}
