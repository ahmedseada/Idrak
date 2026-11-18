// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using Idrak.Imaging;

// The JPEG encoder (plan 28, D6: every page of a scanned PDF goes through it): a grey page of 827 x 1,169 (A4 at 100 dpi,
// the seeded scan of --bench-image), colour images, given as 8-bit images (as the decoders make them) and as floats. Each
// stage: its first call (the first stage's is the process's first encode, its methods compiled) apart from the median of
// nine warm calls, the bytes allocated a call on every thread, and the file's size and SHA-256 (rule 71: an optimization
// leaves it as it is). `--bench-jpeg cases` prints instead the SHA-256 and name of every case of the encoder's
// byte-identity test, the lines of data/image-transforms/jpeg-encoder.sha256.
internal abstract partial class Tests
{
    internal static int BenchJpeg(string? section)
    {
        if (section == "cases")
        {
            foreach (var (name, encode) in JpegEncoderCases())
            {
                Console.WriteLine($"{Convert.ToHexStringLower(SHA256.HashData(encode()))}  {name}");
            }

            return 0;
        }

        if (section is not null)
        {
            Console.Error.WriteLine($"--bench-jpeg takes cases or nothing, not '{section}'.");
            return 2;
        }

        var scan = ScanPage(827, 1169, new Random(17));
        var greyBytes = new byte[827 * 1169];
        for (int i = 0; i < greyBytes.Length; i++)
        {
            greyBytes[i] = (byte)MathF.Round(scan.Pixels[i] * 255f);
        }

        var greyPage = new ImageData([greyBytes], 1169, 827, JpegTestLevels);
        var greyFloats = new ImageData([.. greyBytes.Select(b => b / 255f)], 1, 1169, 827);
        var pattern = new ImageData(JpegTestPlanes(1600, 1200, 3, 7), 1200, 1600, JpegTestLevels);
        var colourPage = ScanPage(1654, 2339, new Random(17));
        Console.WriteLine($"JPEG encoder on the CPU ({Environment.ProcessorCount} threads, {ComputeResources.MaxCpuThreads} used)");
        Console.WriteLine($"{"stage",-52} {"cold ms",8} {"warm ms",8} {"KiB a call",11} {"file KiB",9}  file SHA-256");
        JpegStage("grey page 827 x 1169, q75, 8-bit", () => JpegEncoder.Encode(greyPage, 75));
        JpegStage("grey page 827 x 1169, q75, floats", () => JpegEncoder.Encode(greyFloats, 75));
        JpegStage("grey page 827 x 1169, q75 optimized, 8-bit", () => JpegEncoder.Encode(greyPage, 75, optimize: true));
        JpegStage("colour pattern 1600 x 1200, q90 4:2:0, 8-bit", () => JpegEncoder.Encode(pattern, 90));
        JpegStage("colour pattern 1600 x 1200, q90 4:2:0 optimized", () => JpegEncoder.Encode(pattern, 90, optimize: true));
        JpegStage("colour pattern 1600 x 1200, q90 4:4:4, 8-bit", () => JpegEncoder.Encode(pattern, 90, subsampling: JpegSubsampling.Full444));
        JpegStage("colour page 1654 x 2339, q90 4:2:0, floats", () => JpegEncoder.Encode(colourPage, 90));
        return 0;
    }

    // One stage: the first call, then the median of nine and the bytes those nine allocated (every thread), each file
    // the first's.
    private static void JpegStage(string name, Func<byte[]> run)
    {
        long start = Stopwatch.GetTimestamp();
        byte[] file = run();
        double cold = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        string hash = Hash(file);
        var times = new double[9];
        long allocated = 0;
        for (int i = 0; i < times.Length; i++)
        {
            long before = GC.GetTotalAllocatedBytes(precise: true);
            start = Stopwatch.GetTimestamp();
            byte[] again = run();
            times[i] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            allocated += GC.GetTotalAllocatedBytes(precise: true) - before;
            if (Hash(again) != hash)
            {
                throw new InvalidOperationException($"{name}: call {i + 2} wrote another file than the first");
            }
        }

        Array.Sort(times);
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"{name,-52} {cold,8:F1} {times[times.Length / 2],8:F1} {allocated / (double)times.Length / 1024,11:F1} {file.Length / 1024.0,9:F1}  {hash}"));
    }
}
