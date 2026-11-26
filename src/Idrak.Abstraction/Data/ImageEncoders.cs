// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Data;

/// <summary>What an image is encoded with beyond its pixels: the settings a format may take (a lossy format's quality).</summary>
public sealed record ImageEncodeOptions
{
    private readonly int? _quality;

    /// <summary>
    /// The quality of a lossy format, 1 (smallest) to 100 (best), or null for the encoder's own default; a format without
    /// loss ignores it.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Not from 1 to 100.</exception>
    public int? Quality
    {
        get => _quality;
        init => _quality = value is null or (>= 1 and <= 100) ? value : throw new ArgumentOutOfRangeException(nameof(Quality), value, "An image's quality is from 1 to 100.");
    }
}

/// <summary>
/// Writes one image format. The reading side of a format is its
/// <see cref="IImageCodec"/>.
/// </summary>
public interface IImageEncoder
{
    /// <summary>The format's name, e.g. "png" (names ignore case).</summary>
    string Name { get; }

    /// <summary>The file extensions it writes, with the dot, e.g. ".png"; <c>Save</c> picks an
    /// encoder by them.</summary>
    IReadOnlyCollection<string> Extensions { get; }

    /// <summary>
    /// The file's bytes for <paramref name="image"/>: values in [0, 1] become the format's 8-bit levels (rounded), grey
    /// (1 channel) or colour (3 channels).
    /// </summary>
    byte[] Encode(ImageData image);

    /// <summary>
    /// The file's bytes for <paramref name="image"/> written with <paramref name="options"/> (a lossy format's quality);
    /// a format with nothing to choose writes what <see cref="Encode(ImageData)"/> writes.
    /// </summary>
    byte[] Encode(ImageData image, ImageEncodeOptions options) => Encode(image);
}

/// <summary>
/// The image formats files are written in. The library's (Idrak.Imaging): "png" (8-bit grey or RGB,
/// deflated with the zlib in .NET), "jpeg" (baseline, the quality from <see cref="ImageEncodeOptions.Quality"/>, 75
/// unless given) and "netpbm" (binary PGM for grey, PPM for colour). A segmentation mask, for example, is an 8-bit grey
/// image whose level is the class (<c>idrak predict --out</c> writes them). Picking one by its name is not on the new
/// design yet.
/// </summary>
public static class ImageEncoders
{
}
