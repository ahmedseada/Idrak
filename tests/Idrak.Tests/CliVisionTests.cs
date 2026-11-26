// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Imaging;
using Idrak.Layers;
using Idrak.Models;

// Plan 13, step 7: the command line for image models: the PNG and Netpbm encoders and codecs read back exactly, explain
// and kernels.
internal abstract partial class DesktopTests : Tests
{
    internal static readonly (string Name, Action<Device> Run)[] CliVisionGroup =
    [
        ("cli vision: the PNG and Netpbm encoders write PNG and PGM that their codecs read back exactly; explain knows every step; kernels lists the vision operations", CliVisionBasics),
    ];

    private static void CliVisionBasics(Device device)
    {
        _ = device;
        string folder = TempFolder("cli-vision-basics");
        try
        {
            // PNG (grey and colour) and binary PGM/PPM: the codecs read back the same 8-bit levels.
            var random = new Random(4);
            var png = new PngEncoder();
            var pngCodec = new PngCodec();
            foreach (var (channels, name, encoder, codec) in new (int, string, IImageEncoder, IImageCodec)[]
                     {
                         (1, "grey.png", png, pngCodec), (3, "colour.png", png, pngCodec), (1, "grey.pgm", new NetpbmEncoder(), new NetpbmCodec()),
                         (3, "colour.ppm", new NetpbmEncoder(), new NetpbmCodec()),
                     })
            {
                var pixels = Enumerable.Range(0, channels * 5 * 7).Select(_ => random.Next(256) / 255f).ToArray();
                string path = Path.Combine(folder, name);
                File.WriteAllBytes(path, encoder.Encode(new ImageData(pixels, channels, 5, 7)));
                var read = codec.Decode(File.ReadAllBytes(path));
                Check(read.Channels == channels && read.Height == 5 && read.Width == 7 && read.Pixels.Zip(pixels).All(p => MathF.Abs(p.First - p.Second) < 1e-6f),
                    $"{name}: read back as written");
            }

            // Larger pages through the encoder's parallel rows and the sliced CRC (a gradient with noise and random bytes, odd
            // widths, one row, grey and colour), read back exactly.
            foreach (var (channels, height, width) in new[] { (1, 37, 101), (3, 29, 67), (1, 1, 33), (3, 64, 64) })
            {
                var pixels = new float[channels * height * width];
                for (int i = 0; i < pixels.Length; i++)
                {
                    int x = i % width, y = i / width % height;
                    pixels[i] = (i % 3 == 0 ? random.Next(256) : (x * 3 + y * 2 + random.Next(4)) % 256) / 255f;
                }

                var read = pngCodec.Decode(png.Encode(new ImageData(pixels, channels, height, width)));
                Check(read.Channels == channels && read.Height == height && read.Width == width && read.Pixels.Zip(pixels).All(p => MathF.Abs(p.First - p.Second) < 1e-6f),
                    $"a {width} x {height} PNG of {channels} channels: read back as written");
            }

            Check(Idrak.Imaging.PngEncoder.Crc(0xFFFFFFFFu, "123456789"u8) == ~0xCBF43926u, "CRC-32's check value (the sliced tables and the byte tail)");

            // explain: the normalize step is known (no step built just to count it).
            string network = Path.Combine(folder, "normalized.json");
            File.WriteAllText(network, Network.Image(3, 8, 8).Normalize([0.5f, 0.5f, 0.5f], [0.25f, 0.25f, 0.25f]).Conv2d(4, 3, padding: 1).ReLU().GlobalAveragePool2d().Linear(2)
                .ToJson().ToJsonString());
            var explained = TrainCliJson("explain", network);
            Check(explained["unknownSteps"]!.AsArray().Count == 0, $"explain knows every step: {explained["unknownSteps"]}");

            // kernels: the convolution, resampling, CTC and detection loss operations of plan 13 are listed.
            var kernels = TrainCliJson("kernels", "-d", "cpu");
            var names = kernels["operations"]!.AsArray().Select(o => (string)o!["name"]!).ToHashSet(StringComparer.Ordinal);
            string[] expected = ["Convolution", "ConvolutionBackwardInput", "ConvolutionBackwardWeight", "AvgPool", "AvgPoolBackward", "Interpolate2d", "ResizeNormalize",
                "AdaptiveAvgPool", "AdaptiveMaxPool", "AdaptiveMaxPoolBackward", "CtcLoss", "CtcLossBackward", "BoxIouLoss", "SigmoidFocalLoss"];
            Check(expected.All(names.Contains), $"kernels lists {string.Join(", ", expected.Where(n => !names.Contains(n)))}");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }
}
