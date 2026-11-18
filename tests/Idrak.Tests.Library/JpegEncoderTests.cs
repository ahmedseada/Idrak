// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Security.Cryptography;
using Idrak.Imaging;

// The JPEG encoder's files byte for byte as its first version wrote them (recorded from 0e13af7, before its buffers were
// pooled, with `--bench-jpeg cases`: data/image-transforms/jpeg-encoder.sha256), and what one encode allocates. The images
// are made here from integers, the same bytes on every machine.
internal abstract partial class Tests
{
    internal static string JpegEncoderHashes => Path.Combine(TransformFixtures, "jpeg-encoder.sha256");

    // The levels of an 8-bit image as the codecs make them (byte k stands for k / 255).
    private static readonly float[] JpegTestLevels = [.. Enumerable.Range(0, 256).Select(k => k / 255f)];

    /// <summary>
    /// The byte-identity cases, in a fixed order: small sizes (none but 8 x 8 a multiple of 8, so every MCU edge pads),
    /// grey with standard and optimized tables at two qualities each, colour at 4:2:0, 4:2:2 and 4:4:4 with both kinds of
    /// tables, the input given as floats, as an 8-bit image, as planes, or as floats that are not whole 255ths (16-bit
    /// sources); then a grey page and a colour image large enough for many MCU rows.
    /// </summary>
    internal static IEnumerable<(string Name, Func<byte[]> Encode)> JpegEncoderCases()
    {
        (int Width, int Height)[] sizes =
        [
            (1, 1), (2, 3), (7, 5), (8, 8), (9, 17), (16, 15), (17, 33), (31, 29), (33, 7), (47, 63), (63, 47), (100, 75), (129, 97), (250, 131),
        ];
        int[] qualities = [1, 10, 25, 50, 75, 90, 95, 100];
        string[] inputs = ["floats", "8-bit", "planes", "wide floats"];
        (JpegSubsampling Mode, string Name)[] samplings = [(JpegSubsampling.Half420, "4:2:0"), (JpegSubsampling.Half422, "4:2:2"), (JpegSubsampling.Full444, "4:4:4")];
        int index = 0;
        for (int s = 0; s < sizes.Length; s++)
        {
            var (w, h) = sizes[s];
            foreach (bool optimize in (bool[])[false, true])
            {
                for (int k = 0; k < 2; k++)
                {
                    int quality = qualities[(s + 3 * k + (optimize ? 5 : 0)) % qualities.Length];
                    string input = inputs[index % inputs.Length];
                    var sub = samplings[index % samplings.Length].Mode;           // ignored for grey: proved so
                    yield return Case(w, h, 1, quality, optimize, sub, "", input, s);
                    index++;
                }

                foreach (var (mode, name) in samplings)
                {
                    int quality = qualities[(s + index) % qualities.Length];
                    yield return Case(w, h, 3, quality, optimize, mode, name, inputs[index % inputs.Length], s);
                    index++;
                }
            }
        }

        yield return Case(827, 1169, 1, 75, false, JpegSubsampling.Half420, "", "8-bit", 101);
        yield return Case(827, 1169, 1, 75, true, JpegSubsampling.Half420, "", "floats", 101);
        foreach (var (mode, name) in samplings)
        {
            yield return Case(1001, 701, 3, 90, false, mode, name, "8-bit", 102);
        }

        yield return Case(1001, 701, 3, 85, true, JpegSubsampling.Half420, "4:2:0", "planes", 102);
        yield return Case(1001, 701, 3, 60, true, JpegSubsampling.Half422, "4:2:2", "wide floats", 102);

        static (string, Func<byte[]>) Case(int w, int h, int channels, int quality, bool optimize, JpegSubsampling sub, string sampling, string input, int seed)
        {
            string name = $"{(channels == 1 ? "grey" : "colour")} {w}x{h} q{quality}{(sampling.Length > 0 ? " " + sampling : "")} {(optimize ? "optimized" : "standard")} {input}";
            return (name, () =>
            {
                var planes = JpegTestPlanes(w, h, channels, seed);
                return input switch
                {
                    "planes" => JpegEncoder.Encode(planes, h, w, quality, optimize, sub),
                    "8-bit" => JpegEncoder.Encode(new ImageData(planes, h, w, JpegTestLevels), quality, optimize, sub),
                    "floats" => JpegEncoder.Encode(new ImageData(JpegTestFloats(planes, wide: false), channels, h, w), quality, optimize, sub),
                    _ => JpegEncoder.Encode(new ImageData(JpegTestFloats(planes, wide: true), channels, h, w), quality, optimize, sub),
                };
            });
        }
    }

    /// <summary>
    /// A generic test image from integers only: four areas, a smooth gradient across and down (long runs of zero
    /// coefficients), stripes and a checker (large coefficients of both signs), noise (every coefficient), and a flat field
    /// with a few dots (blocks that end early); each channel shifted from the others so colour has chroma.
    /// </summary>
    internal static byte[][] JpegTestPlanes(int width, int height, int channels, int seed)
    {
        var planes = new byte[channels][];
        for (int c = 0; c < channels; c++)
        {
            var plane = planes[c] = new byte[width * height];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    uint n = (uint)(x * 73856093) ^ (uint)(y * 19349663) ^ (uint)((c + 1) * 83492791) ^ (uint)seed * 2654435761u;
                    n ^= n >> 13;
                    n *= 0x5bd1e995;
                    n ^= n >> 15;
                    int value = (2 * y / height * 2 + 2 * x / width) switch
                    {
                        0 => (x * 255 / Math.Max(1, width - 1) * (3 - c) + y * 255 / Math.Max(1, height - 1) * (c + 1)) / 4,
                        1 => ((x / (1 + c) + y / 3) & 1) == 0 ? 16 + 40 * c : 255 - 30 * c,
                        2 => (int)(n & 255),
                        _ => n % 61 == 0 ? 255 - 100 * c : 128 + 30 * c - 30,
                    };
                    plane[y * width + x] = (byte)value;
                }
            }
        }

        return planes;
    }

    // The planes as floats: whole 255ths (as an 8-bit source widens), or values between them (16-bit-like sources, which
    // the encoder brings to 8 bits by their high byte, as Pillow reads 16-bit files).
    private static float[] JpegTestFloats(byte[][] planes, bool wide)
    {
        int size = planes[0].Length;
        var pixels = new float[planes.Length * size];
        for (int c = 0; c < planes.Length; c++)
        {
            for (int i = 0; i < size; i++)
            {
                int b = planes[c][i];
                pixels[c * size + i] = wide ? Math.Clamp(b * 257 + (i * 7 + c * 13) % 255 - 127, 0, 65535) / 65535f : b / 255f;
            }
        }

        return pixels;
    }

    internal static void JpegEncoderUnchanged(Device device)
    {
        _ = device;
        var expected = new List<(string Hash, string Name)>();
        foreach (string line in File.ReadLines(JpegEncoderHashes))
        {
            if (line.Length > 66 && line[64] == ' ' && line[65] == ' ')
            {
                expected.Add((line[..64], line[66..]));
            }
        }

        var cases = JpegEncoderCases().ToList();
        Check(cases.Count == expected.Count && cases.Count >= 140, $"{cases.Count} cases, {expected.Count} recorded hashes");
        var failures = new List<string>();
        for (int i = 0; i < cases.Count; i++)
        {
            var (name, encode) = cases[i];
            if (name != expected[i].Name)
            {
                failures.Add($"case {i} is '{name}', recorded '{expected[i].Name}'");
                continue;
            }

            byte[] file = encode();
            if (Convert.ToHexStringLower(SHA256.HashData(file)) != expected[i].Hash)
            {
                failures.Add($"{name}: {file.Length} bytes, not the recorded file");
            }
        }

        Check(failures.Count == 0, $"{failures.Count} of {cases.Count} files differ from the encoder's first version: {string.Join("; ", failures.Take(12))}");

        // One warm encode of a page allocates its file and little else (the first version: about 6 MB for this page, a
        // 64-entry array per block, the planes copied, padded sample planes, a scan list and a growing stream).
        var planes = JpegTestPlanes(827, 1169, 1, 101);
        var eightBit = new ImageData(planes, 1169, 827, JpegTestLevels);
        var floats = new ImageData(JpegTestFloats(planes, wide: false), 1, 1169, 827);
        foreach (var (what, run) in new (string, Func<byte[]>)[]
                 {
                     ("8-bit grey page, standard tables", () => JpegEncoder.Encode(eightBit, 75)),
                     ("float grey page, standard tables", () => JpegEncoder.Encode(floats, 75)),
                     ("8-bit grey page, optimized tables", () => JpegEncoder.Encode(eightBit, 75, optimize: true)),
                 })
        {
            for (int i = 0; i < 3; i++)
            {
                run();                                                                   // warm: the pooled buffers
            }

            long before = GC.GetTotalAllocatedBytes(precise: true);
            byte[] file = run();
            long allocated = GC.GetTotalAllocatedBytes(precise: true) - before;
            Check(allocated < file.Length + (256 << 10), $"{what}: an encode allocated {allocated:N0} bytes for a file of {file.Length:N0}");
            Console.WriteLine($"    {what}: {allocated:N0} bytes allocated for a file of {file.Length:N0}");
        }
    }
}
