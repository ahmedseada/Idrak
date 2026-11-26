// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Imaging;

/// <summary>
/// "jpeg": <see cref="JpegEncoder"/> through <see cref="ImageEncoders"/>, so a package that does not reference Idrak.Imaging
/// (the PDF writer's JPEG pages) writes JPEG by name. Baseline, grey as one component and colour as YCbCr 4:2:0, at
/// <see cref="ImageEncodeOptions.Quality"/> (Pillow's default, 75, unless given); the same bytes as
/// <see cref="JpegEncoder.Encode(ImageData, int, bool, JpegSubsampling)"/> at that quality.
/// </summary>
public sealed class JpegImageEncoder : IImageEncoder
{
    /// <summary>The quality written when none is given (Pillow's default).</summary>
    public const int DefaultQuality = 75;

    /// <summary>"jpeg".</summary>
    public string Name => "jpeg";

    /// <summary>".jpg" and ".jpeg".</summary>
    public IReadOnlyCollection<string> Extensions { get; } = [".jpg", ".jpeg"];

    /// <inheritdoc/>
    public byte[] Encode(ImageData image) => JpegEncoder.Encode(image, DefaultQuality);

    /// <inheritdoc/>
    public byte[] Encode(ImageData image, ImageEncodeOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return JpegEncoder.Encode(image, options.Quality ?? DefaultQuality);
    }
}
