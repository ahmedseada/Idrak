// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using Idrak;
using Idrak.Data;
using Idrak.Data.Abstractions;

// The CPU side of reading a page (plan 17, lane C): a scan-sized page (1,654 x 2,339, an A4 page at 200 dpi, made here from
// a seeded generator: a light background with grain and dark strokes) through every stage a read runs: the JPEG and PNG
// decoders, the size from the header, the model card's pipeline and each transform on its own, the preprocessor's resize
// to the encoder's 896 x 896, and the PNG of the page as the model sees it. Each stage: the first call (cold) apart from
// the median of five (warm), the bytes allocated on every thread, and the SHA-256 of its output (an optimization must
// leave it as it is: rule 71).
internal static partial class Tests
{
    internal static int BenchImage()
    {
        const int Width = 1654, Height = 2339;
        var page = ScanPage(Width, Height, new Random(17));
        byte[] jpeg = JpegEncoder.Encode(page, quality: 90);
        var grey = ImageTransformPipeline.Parse("grayscale").Apply(page);
        byte[] png = ImageEncoders.Get("png").Encode(grey);
        var card = ImageTransformPipeline.Parse("grayscale,max_width=1024,contrast=1.5");
        var seen = card.Apply(ChatImageDecoder.Decode(jpeg));
        var preprocessor = ImagePreprocessor.Parse("""
            {"do_resize": true, "size": {"height": 896, "width": 896}, "resample": 2, "do_rescale": true, "rescale_factor": 0.00392156862745098,
             "do_normalize": true, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]}
            """);
        var decoded = ChatImageDecoder.Decode(jpeg);
        Console.WriteLine($"Image pipeline on the CPU ({Environment.ProcessorCount} threads): a {Width} x {Height} colour page, JPEG {jpeg.Length / 1024} KiB "
                          + $"(quality 90, 4:2:0), grey PNG {png.Length / 1024} KiB; after the card's pipeline {seen.Width} x {seen.Height}");
        Console.WriteLine($"{"stage",-44} {"cold ms",9} {"warm ms",9} {"MB warm",9}  output SHA-256");

        Stage("jpeg decode (colour page)", () => ChatImageDecoder.Decode(jpeg).Pixels);
        Stage("png decode (grey page)", () => ChatImageDecoder.Decode(png).Pixels);
        foreach (var (filter, name) in new[] { ((byte)2, "Up"), ((byte)4, "Paeth") })
        {
            byte[] filtered = FilteredPng(grey, filter);
            Stage($"png decode (grey page, {name} rows)", () => ChatImageDecoder.Decode(filtered).Pixels);
        }

        Stage("size from the header (jpeg)", () => ChatImageDecoder.Size(jpeg) is var (w, h) ? $"{w} x {h}" : "none");
        Stage("card: grayscale,max_width=1024,contrast=1.5", () => card.Apply(decoded).Pixels);
        foreach (string step in new[] { "grayscale", "max_width=1024", "contrast=1.5", "brightness=1.1", "sharpness=1.5", "autocontrast=1" })
        {
            var pipeline = ImageTransformPipeline.Parse(step);
            Stage($"{step} (colour page)", () => pipeline.Apply(decoded).Pixels);
        }

        foreach (string step in new[] { "equalize", "gamma=0.8", "blur=0.5", "unsharp=2,percent=150,threshold=3", "median=3", "min_filter=3", "max_filter=3",
                     "binarize", "invert", "pad=32,fill=255", "scale=1.5", "jpeg=95" })
        {
            var pipeline = ImageTransformPipeline.Parse(step);
            Stage($"{step} (grey 1024)", () => pipeline.Apply(seen).Pixels);
        }

        Stage("jpeg encode (grey 1024, quality 95, optimized)", () => JpegEncoder.Encode(seen, quality: 95, optimize: true));
        byte[] seenJpeg = JpegEncoder.Encode(seen, quality: 95, optimize: true);
        Stage("jpeg decode (grey 1024, quality 95)", () => ChatImageDecoder.Decode(seenJpeg).Pixels);
        Stage("preprocess to 896 (grey 1024 page)", () => preprocessor.Pixels(seen));
        Stage("preprocess to 896 (colour page)", () => preprocessor.Pixels(decoded));
        Stage("png encode (the page as the model sees it)", () => ImageEncoders.Get("png").Encode(seen));
        return 0;
    }

    // Times one stage: a cold call, then the median of five, and the bytes allocated by those five (every thread). Only
    // the stage is timed; its output's hash (pixels, file bytes or text) is taken after each call and must not change.
    private static void Stage(string name, Func<object> run)
    {
        var clock = Stopwatch.StartNew();
        object result = run();
        double cold = clock.Elapsed.TotalMilliseconds;
        string output = Describe(result);
        var times = new double[5];
        long allocated = 0;
        for (int i = 0; i < times.Length; i++)
        {
            long before = GC.GetTotalAllocatedBytes(precise: true);
            long start = Stopwatch.GetTimestamp();
            result = run();
            times[i] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            allocated += GC.GetTotalAllocatedBytes(precise: true) - before;
            string again = Describe(result);
            if (again != output)
            {
                throw new InvalidOperationException($"{name}: run {i + 2} gave {again}, the first {output}");
            }
        }

        Array.Sort(times);
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"{name,-44} {cold,9:F1} {times[2],9:F1} {allocated / (double)times.Length / (1 << 20),9:F1}  {output}"));

        static string Describe(object value) => value switch
        {
            float[] pixels => Hash(pixels),
            byte[] bytes => Hash(bytes),
            _ => value.ToString() ?? "",
        };
    }

    private static string Hash(float[] values) => Hash(MemoryMarshal.AsBytes(values.AsSpan()));

    private static string Hash(ReadOnlySpan<byte> bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes))[..16];

    // The grey page as a PNG whose rows all use one filter (other encoders filter their rows; ours writes them as they are),
    // so the decoder's filters are timed: the same pixels as the unfiltered file.
    private static byte[] FilteredPng(ImageData grey, byte filter)
    {
        int w = grey.Width, h = grey.Height;
        var bytes = new byte[w * h];
        for (int i = 0; i < bytes.Length; i++)
        {
            bytes[i] = (byte)MathF.Round(grey.Pixels[i] * 255f);
        }

        var rows = new byte[h * (w + 1)];
        for (int y = 0; y < h; y++)
        {
            rows[y * (w + 1)] = filter;
            for (int x = 0; x < w; x++)
            {
                int a = x > 0 ? bytes[y * w + x - 1] : 0, b = y > 0 ? bytes[(y - 1) * w + x] : 0, c = x > 0 && y > 0 ? bytes[(y - 1) * w + x - 1] : 0;
                int p = a + b - c, pa = Math.Abs(p - a), pb = Math.Abs(p - b), pc = Math.Abs(p - c);
                int predictor = filter == 2 ? b : pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
                rows[y * (w + 1) + 1 + x] = (byte)(bytes[y * w + x] - predictor);
            }
        }

        var data = new MemoryStream();
        using (var zlib = new System.IO.Compression.ZLibStream(data, System.IO.Compression.CompressionLevel.Fastest, leaveOpen: true))
        {
            zlib.Write(rows);
        }

        var file = new MemoryStream();
        file.Write([0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        var header = new byte[13];
        System.Buffers.Binary.BinaryPrimitives.WriteInt32BigEndian(header, w);
        System.Buffers.Binary.BinaryPrimitives.WriteInt32BigEndian(header.AsSpan(4), h);
        header[8] = 8;                                                    // 8 bits, grey, no interlace
        Chunk("IHDR"u8, header);
        Chunk("IDAT"u8, data.ToArray());
        Chunk("IEND"u8, []);
        return file.ToArray();

        void Chunk(ReadOnlySpan<byte> type, ReadOnlySpan<byte> body)
        {
            Span<byte> length = stackalloc byte[4];
            System.Buffers.Binary.BinaryPrimitives.WriteInt32BigEndian(length, body.Length);
            file.Write(length);
            file.Write(type);
            file.Write(body);
            file.Write([0, 0, 0, 0]);                                     // the CRC (the decoder does not read it)
        }
    }

    // A light page with grain and dark strokes of text-like size, in colour (a slightly warm paper), from the seed.
    private static ImageData ScanPage(int width, int height, Random random)
    {
        var pixels = new float[3 * width * height];
        int plane = width * height;
        for (int i = 0; i < plane; i++)
        {
            float paper = 0.90f + (float)random.NextDouble() * 0.06f;
            pixels[i] = MathF.Round(paper * 255f) / 255f;
            pixels[plane + i] = MathF.Round((paper - 0.02f) * 255f) / 255f;
            pixels[2 * plane + i] = MathF.Round((paper - 0.06f) * 255f) / 255f;
        }

        for (int line = 120; line < height - 120; line += 46)
        {
            for (int x = 140; x < width - 140;)
            {
                int stroke = random.Next(3, 22), tall = random.Next(14, 30);
                float ink = MathF.Round((0.05f + (float)random.NextDouble() * 0.2f) * 255f) / 255f;
                for (int y = line; y < Math.Min(height, line + tall); y++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        pixels.AsSpan(c * plane + y * width + x, Math.Min(stroke, width - x)).Fill(ink);
                    }
                }

                x += stroke + random.Next(2, 14);
            }
        }

        return new ImageData(pixels, 3, height, width);
    }
}
