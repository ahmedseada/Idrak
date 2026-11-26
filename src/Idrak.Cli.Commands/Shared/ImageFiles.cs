// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Data;
using Idrak.Data.Abstractions;

namespace Idrak.Cli.Shared;

/// <summary>
/// Image files for every command that reads them (<c>idrak suggest</c> and the data it prepares, <c>idrak train</c> and
/// <c>idrak predict</c> on image folders). The pixels are decoded by the library's codecs, through
/// <see cref="ImageFolderSource"/>; picking a codec by a file's extension or header is not on the new design yet, so this
/// class knows no codec: it reads the size and channels of a JPEG file from its header.
/// </summary>
internal static class ImageFiles
{
    private static readonly string[] HeaderOnly = [".jpg", ".jpeg"];

    /// <summary>The extensions read as images: JPEG's (picking a codec by its extension is not on the new design yet).</summary>
    public static IReadOnlyCollection<string> Extensions => HeaderOnly;

    /// <summary>Whether <paramref name="path"/> has an image extension.</summary>
    public static bool IsImage(string path) => Extensions.Contains(Path.GetExtension(path), StringComparer.OrdinalIgnoreCase);

    /// <summary>The extensions whose pixels are decoded (training and prediction read these): none, no codec being picked by its extension.</summary>
    public static IReadOnlyCollection<string> DecodedExtensions => [];

    /// <summary>Whether <paramref name="path"/> has the extension of an image whose pixels are decoded: picking its codec by its extension is not on the new design yet.</summary>
    public static bool IsDecoded(string path) => NotOnTheNewDesign.Yet<bool>($"Picking the image codec of {path}");

    /// <summary>What an image header says.</summary>
    public readonly record struct ImageInfo(int Width, int Height, int Channels, string Format, bool Decodable);

    /// <summary>The size and channels of an image, or null when the file is not one of the formats above.</summary>
    public static ImageInfo? ReadHeader(string path) => Jpeg(path);

    // The frame header (SOF0..SOF15 but DHT, JPG and DAC) holds the size and the number of components.
    private static ImageInfo? Jpeg(string path)
    {
        using var stream = File.OpenRead(path);
        if (stream.Length < 4 || stream.ReadByte() != 0xFF || stream.ReadByte() != 0xD8)
        {
            return null;
        }

        using var reader = new BinaryReader(stream);
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
}
