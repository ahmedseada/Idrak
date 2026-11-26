// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Data;

/// <summary>What an image file's header says.</summary>
/// <param name="Width">Pixels per row.</param>
/// <param name="Height">Rows.</param>
/// <param name="Channels">Channels once decoded: 1 for grey, 3 for colour (alpha is not counted).</param>
/// <param name="Format">The codec's name, e.g. "png".</param>
public readonly record struct ImageInfo(int Width, int Height, int Channels, string Format);

/// <summary>
/// Reads one image format; register it with <c>Register</c>. A codec decides from a file's first
/// bytes whether the file is in its format, so the extension only decides which files a folder source picks up.
/// </summary>
public interface IImageCodec
{
    /// <summary>The format's name, e.g. "png" (names ignore case).</summary>
    string Name { get; }

    /// <summary>The file extensions of the format, with the dot, e.g. ".png".</summary>
    IReadOnlyCollection<string> Extensions { get; }

    /// <summary>
    /// The size and channels from the start of a file (its first 64 KiB, or all of a smaller file), or null when the
    /// bytes are not this format. A file of the format whose size lies further in (a JPEG with long EXIF and ICC
    /// segments before its frame header) answers a width and height of 0; <c>ReadInfo</c>
    /// (and its in-memory sibling) then asks
    /// again with the whole file.
    /// </summary>
    ImageInfo? ReadInfo(ReadOnlySpan<byte> header);

    /// <summary>
    /// Decodes a whole file: a damaged file is an <see cref="InvalidDataException"/>, a valid file in a variant the codec
    /// does not read (a CMYK JPEG, say) an <see cref="InvalidDataException"/> or a <see cref="NotSupportedException"/>.
    /// </summary>
    ImageData Decode(ReadOnlySpan<byte> file);
}

/// <summary>
/// The image formats that <c>ImageFolderSource</c> and the command-line tool read, by name (ignoring case). Built in,
/// without dependencies: "png" (every bit depth and colour type, interlaced or not, inflated with the zlib in .NET),
/// "jpeg" (baseline and progressive, grey, YCbCr, RGB, CMYK and YCCK, any whole sampling ratio, decoded to Pillow's
/// RGB; not 12-bit or arithmetic-coded), "bmp" (1, 4, 8, 16, 24 and 32 bits, uncompressed or with bit fields) and
/// "netpbm" (PGM and PPM, text or binary). Alpha is dropped.
/// </summary>
public static class ImageCodecs
{
    /// <summary>The bytes of a file's start that <see cref="IImageCodec.ReadInfo"/> receives.</summary>
    public const int HeaderBytes = 64 * 1024;
}
