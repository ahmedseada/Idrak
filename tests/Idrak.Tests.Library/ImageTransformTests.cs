// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Security.Cryptography;
using System.Text.Json.Nodes;
using Idrak.Data;
using Idrak.Data.Abstractions;
using Idrak.Imaging;
using Idrak.Text;

// Plan 11, image transforms: the JPEG encoder against libjpeg-turbo 3.1 and Pillow 12.3's files
// (tools/vlm/make_image_transforms.py writes tests/Idrak.Tests/data/image-transforms).
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] ImageTransformGroup =
    [
        ("image transforms: the JPEG encoder writes its first version's files byte for byte (grey and colour; floats, 8-bit images, planes and 16-bit-like floats; sizes 1 x 1 to 1,001 x 701 not a multiple of 8; qualities 1 to 100; 4:2:0, 4:2:2, 4:4:4; standard and optimized tables), and a warm page encode allocates little beyond its file", JpegEncoderUnchanged),
        ("image transforms: the jpeg round trip gives its file's pixels without making the file (grey and colour; 8-bit and floats; sizes not a multiple of 8; qualities 1 to 100; 4:2:0, 4:2:2, 4:4:4; pages at the scan profiles' qualities), and a warm page round trip allocates its pixels and little else", JpegRoundTripWithoutFile),
    ];

    internal static string TransformFixtures => Path.Combine(DataFolder(), "image-transforms");

    // The 8-bit planes of an image, concatenated (Pillow's bytes in planar order).
    internal static byte[] PlanarBytes(ImageData image)
    {
        var bytes = new byte[image.Pixels.Length];
        for (int i = 0; i < bytes.Length; i++)
        {
            bytes[i] = (byte)Math.Clamp((int)MathF.Round(image.Pixels[i] * 255f), 0, 255);
        }

        return bytes;
    }
}
