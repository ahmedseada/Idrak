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
        ("image transforms: the JPEG encoder's files decode to the pixels of Pillow's at the same quality (grey, 4:2:0, 4:2:2, 4:4:4, odd sizes, qualities 10 to 100), and so does the jpeg round trip, which makes no file", JpegEncoderMatchesPillow),
        ("image transforms: the JPEG encoder writes its first version's files byte for byte (grey and colour; floats, 8-bit images, planes and 16-bit-like floats; sizes 1 x 1 to 1,001 x 701 not a multiple of 8; qualities 1 to 100; 4:2:0, 4:2:2, 4:4:4; standard and optimized tables), and a warm page encode allocates little beyond its file", JpegEncoderUnchanged),
        ("image transforms: the jpeg round trip gives its file's pixels without making the file (grey and colour; 8-bit and floats; sizes not a multiple of 8; qualities 1 to 100; 4:2:0, 4:2:2, 4:4:4; pages at the scan profiles' qualities), and a warm page round trip allocates its pixels and little else", JpegRoundTripWithoutFile),
    ];

    internal static string TransformFixtures => Path.Combine(DataFolder(), "image-transforms");

    internal static JsonObject TransformManifest() => JsonNode.Parse(File.ReadAllText(Path.Combine(TransformFixtures, "manifest.json")))!.AsObject();

    // An input of the manifest, decoded as a chat image is (codecs, EXIF upright).
    internal static ImageData TransformInput(string name)
    {
        string path = Path.Combine(DataFolder(), name.Split('#')[0].Replace('/', Path.DirectorySeparatorChar));
        var image = ChatImageDecoder.Decode(File.ReadAllBytes(path));
        return name.EndsWith("#L", StringComparison.Ordinal) ? ChatImageDecoder.Grayscale(image) : image;
    }

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

    internal static void JpegEncoderMatchesPillow(Device device)
    {
        _ = device;
        var cases = TransformManifest()["jpeg"]!.AsArray();
        Check(cases.Count >= 40, $"{cases.Count} JPEG cases");
        var failures = new List<string>();
        var roundTrips = new List<string>();
        int identical = 0;
        foreach (var node in cases)
        {
            var entry = node!.AsObject();
            string input = (string)entry["input"]!, sub = (string)entry["subsampling"]!;
            int quality = (int)entry["quality"]!;
            var image = TransformInput(input);
            var subsampling = sub switch { "4:4:4" => JpegSubsampling.Full444, "4:2:2" => JpegSubsampling.Half422, _ => JpegSubsampling.Half420 };
            var file = JpegEncoder.Encode(image, quality, optimize: true, subsampling);
            identical += Convert.ToHexStringLower(SHA256.HashData(file)) == (string)entry["sha256"]! ? 1 : 0;

            // The round trip (jpeg=Q) makes no file: its pixels are still those of Pillow's file decoded.
            if (Convert.ToHexStringLower(SHA256.HashData(PlanarBytes(PillowImageOps.JpegRoundTrip(image, quality, subsampling)))) != (string)entry["decoded_sha256"]!)
            {
                roundTrips.Add($"{input} q{quality} {sub}");
            }
            var decoded = new JpegCodec().Decode(file);
            var bytes = PlanarBytes(decoded);
            if (Convert.ToHexStringLower(SHA256.HashData(bytes)) == (string)entry["decoded_sha256"]!)
            {
                continue;
            }

            string worst = "";
            if (entry["file"] is { } pillow)
            {
                var expected = PlanarBytes(new JpegCodec().Decode(File.ReadAllBytes(Path.Combine(TransformFixtures, (string)pillow!))));
                int most = 0;
                for (int i = 0; i < Math.Min(bytes.Length, expected.Length); i++)
                {
                    most = Math.Max(most, Math.Abs(bytes[i] - expected[i]));
                }

                worst = $" (largest difference {most})";
            }

            failures.Add($"{input} q{quality} {sub}: decoded pixels differ from Pillow's file{worst}");
        }

        Check(failures.Count == 0, $"{failures.Count} of {cases.Count}: {string.Join("; ", failures.Take(12))}");
        Check(roundTrips.Count == 0, $"{roundTrips.Count} of {cases.Count} round trips differ from Pillow's file decoded: {string.Join("; ", roundTrips.Take(12))}");

        // Optimized Huffman tables change the size, never the pixels; the files are libjpeg-turbo's, byte for byte.
        Check(identical == cases.Count, $"{identical} of {cases.Count} files are byte for byte Pillow's");
        var sample = TransformInput("vlm/image.png");
        var plain = new JpegCodec().Decode(JpegEncoder.Encode(sample, 90));
        var optimized = new JpegCodec().Decode(JpegEncoder.Encode(sample, 90, optimize: true));
        Check(PlanarBytes(plain).SequenceEqual(PlanarBytes(optimized)), "optimize changed the pixels");
        Check(JpegEncoder.Encode(sample, 90, optimize: true).Length < JpegEncoder.Encode(sample, 90).Length, "optimized tables are not smaller");
    }
}
