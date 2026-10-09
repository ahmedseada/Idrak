// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Data.Abstractions;

namespace Idrak.Data;

/// <summary>Chroma subsampling of a colour JPEG.</summary>
public enum JpegSubsampling
{
    /// <summary>4:2:0: chroma halved across and down (libjpeg's and Pillow's default).</summary>
    Half420,

    /// <summary>4:2:2: chroma halved across.</summary>
    Half422,

    /// <summary>4:4:4: chroma at full resolution.</summary>
    Full444,
}

/// <summary>
/// Writes baseline JPEG files as Pillow's <c>Image.save(..., "JPEG", quality=Q)</c> does through libjpeg-turbo: a JFIF
/// header, the standard (Annex K) quantization tables scaled for the quality (jcparam.c, baseline: values within 1 to
/// 255), grey images as one component and colour ones as YCbCr (jccolor.c's fixed-point tables) with chroma subsampled
/// 4:2:0 unless asked (jcsample.c's averaging with its alternating bias; edges replicated as jcprepct.c pads them), the
/// "islow" integer forward DCT of jfdctint.c and libjpeg-turbo's quantization by reciprocals (jcdctmgr.c), dummy blocks
/// at the MCU edges as jccoefct.c makes them. Huffman tables are the standard ones, or, with <c>optimize</c>
/// (Pillow's <c>optimize=True</c>), built from the image's own statistics by jchuff.c's <c>jpeg_gen_optimal_table</c>:
/// they change the size of the file, not its pixels. The coefficients are the reference encoder's, so any decoder gives
/// the same pixels for this file as for Pillow's.
/// </summary>
public static class JpegEncoder
{
    private static readonly int[] StdLuminance =
    [
        16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
    ];

    private static readonly int[] StdChrominance =
    [
        17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    ];

    // Zigzag order: the natural (row-major) index of the k-th coefficient written.
    private static readonly int[] Zigzag =
    [
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
    ];

    // The standard Huffman tables (jcparam.c's std_huff_tables): code-length counts 1..16, then the symbols.
    private static readonly byte[] DcLuminanceBits = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
    private static readonly byte[] DcChrominanceBits = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
    private static readonly byte[] DcValues = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    private static readonly byte[] AcLuminanceBits = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d];
    private static readonly byte[] AcChrominanceBits = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];

    private static readonly byte[] AcLuminanceValues =
    [
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
        0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
        0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
        0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
        0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
    ];

    private static readonly byte[] AcChrominanceValues =
    [
        0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
        0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0, 0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
        0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
        0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
        0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
        0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
        0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
    ];

    /// <summary>
    /// The image (grey: one channel; colour: three) as a JPEG file at <paramref name="quality"/> (1 to 100; Pillow's
    /// default is 75), its pixels first brought to 8 bits as Pillow holds them.
    /// </summary>
    public static byte[] Encode(ImageData image, int quality = 75, bool optimize = false, JpegSubsampling subsampling = JpegSubsampling.Half420)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (image.Channels is not 1 and not 3)
        {
            throw new ArgumentException($"A JPEG holds a grey or an RGB image, not {image.Channels} channels.", nameof(image));
        }

        return Encode(ImagePreprocessor.ToBytes(image, grayscale: false), image.Height, image.Width, quality, optimize, subsampling);
    }

    /// <summary>8-bit planes (one grey, or red, green and blue) of <paramref name="height"/> x <paramref name="width"/> as a JPEG file.</summary>
    public static byte[] Encode(byte[][] planes, int height, int width, int quality = 75, bool optimize = false, JpegSubsampling subsampling = JpegSubsampling.Half420)
    {
        ArgumentNullException.ThrowIfNull(planes);
        ArgumentOutOfRangeException.ThrowIfLessThan(quality, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(quality, 100);
        if (width is < 1 or > 65535 || height is < 1 or > 65535)
        {
            throw new ArgumentException($"A JPEG is 1 to 65,535 pixels a side, not {width} x {height}.");
        }

        if (planes.Length is not 1 and not 3 || planes.Any(p => p.Length != width * height))
        {
            throw new ArgumentException($"Give one or three planes of {width} x {height} bytes.", nameof(planes));
        }

        bool colour = planes.Length == 3;
        int scale = quality < 50 ? 5000 / quality : 200 - quality * 2;                     // jpeg_quality_scaling
        int[][] tables = colour ? [Scaled(StdLuminance, scale), Scaled(StdChrominance, scale)] : [Scaled(StdLuminance, scale)];

        // Components: (plane, horizontal and vertical sampling, table).
        int hs = colour && subsampling != JpegSubsampling.Full444 ? 2 : 1, vs = colour && subsampling == JpegSubsampling.Half420 ? 2 : 1;
        var components = new List<Component>();
        if (colour)
        {
            var (y, cb, cr) = ToYCbCr(planes, width * height);
            components.Add(new Component(1, hs, vs, 0, y));
            components.Add(new Component(2, 1, 1, 1, cb));
            components.Add(new Component(3, 1, 1, 1, cr));
        }
        else
        {
            components.Add(new Component(1, 1, 1, 0, planes[0]));
        }

        int maxH = components.Max(c => c.H), maxV = components.Max(c => c.V);
        int mcusAcross = DivUp(width, 8 * maxH), mcusDown = DivUp(height, 8 * maxV);
        bool interleaved = components.Count > 1;
        foreach (var c in components)
        {
            c.Prepare(width, height, maxH, maxV, interleaved ? mcusAcross * c.H : -1, interleaved ? mcusDown * c.V : -1);
            c.Coefficients = Transform(c, tables[c.Table]);
        }

        // Blocks in scan order: interleaved MCUs (each component's H x V blocks, dummies past its edge), or one component's blocks.
        var sequence = new List<(int Component, short[] Block)>();
        if (interleaved)
        {
            for (int my = 0; my < mcusDown; my++)
            {
                for (int mx = 0; mx < mcusAcross; mx++)
                {
                    for (int ci = 0; ci < components.Count; ci++)
                    {
                        var c = components[ci];
                        for (int by = 0; by < c.V; by++)
                        {
                            for (int bx = 0; bx < c.H; bx++)
                            {
                                sequence.Add((ci, c.Coefficients[(my * c.V + by) * c.PaddedBlocksAcross + mx * c.H + bx]));
                            }
                        }
                    }
                }
            }
        }
        else
        {
            sequence.AddRange(components[0].Coefficients.Select(b => (0, b)));
        }

        int tableCount = colour ? 2 : 1;
        var dc = new HuffmanTable[tableCount];
        var ac = new HuffmanTable[tableCount];
        if (optimize)
        {
            var dcFrequency = new long[tableCount][];
            var acFrequency = new long[tableCount][];
            for (int t = 0; t < tableCount; t++)
            {
                dcFrequency[t] = new long[257];
                acFrequency[t] = new long[257];
            }

            var last = new int[components.Count];
            foreach (var (ci, block) in sequence)
            {
                int t = components[ci].Table;
                int diff = block[0] - last[ci];
                last[ci] = block[0];
                dcFrequency[t][Bits(diff)]++;
                int run = 0;
                for (int k = 1; k < 64; k++)
                {
                    int v = block[Zigzag[k]];
                    if (v == 0)
                    {
                        run++;
                        continue;
                    }

                    while (run > 15)
                    {
                        acFrequency[t][0xF0]++;
                        run -= 16;
                    }

                    acFrequency[t][(run << 4) + Bits(v)]++;
                    run = 0;
                }

                if (run > 0)
                {
                    acFrequency[t][0]++;
                }
            }

            for (int t = 0; t < tableCount; t++)
            {
                dc[t] = OptimalTable(dcFrequency[t]);
                ac[t] = OptimalTable(acFrequency[t]);
            }
        }
        else
        {
            dc[0] = new HuffmanTable(DcLuminanceBits, DcValues);
            ac[0] = new HuffmanTable(AcLuminanceBits, AcLuminanceValues);
            if (colour)
            {
                dc[1] = new HuffmanTable(DcChrominanceBits, DcValues);
                ac[1] = new HuffmanTable(AcChrominanceBits, AcChrominanceValues);
            }
        }

        var output = new MemoryStream();
        output.Write([0xFF, 0xD8]);
        output.Write([0xFF, 0xE0, 0, 16, (byte)'J', (byte)'F', (byte)'I', (byte)'F', 0, 1, 1, 0, 0, 1, 0, 1, 0, 0]);   // JFIF 1.01, aspect 1:1
        for (int t = 0; t < tables.Length; t++)
        {
            output.Write([0xFF, 0xDB, 0, 67, (byte)t]);
            for (int k = 0; k < 64; k++)
            {
                output.WriteByte((byte)tables[t][Zigzag[k]]);
            }
        }

        int sofLength = 8 + 3 * components.Count;
        output.Write([0xFF, 0xC0, (byte)(sofLength >> 8), (byte)sofLength, 8, (byte)(height >> 8), (byte)height, (byte)(width >> 8), (byte)width, (byte)components.Count]);
        foreach (var c in components)
        {
            output.Write([(byte)c.Id, (byte)(c.H << 4 | c.V), (byte)c.Table]);
        }

        for (int t = 0; t < tableCount; t++)
        {
            WriteTable(output, dc[t], (byte)t);
            WriteTable(output, ac[t], (byte)(0x10 | t));
        }

        int sosLength = 6 + 2 * components.Count;
        output.Write([0xFF, 0xDA, (byte)(sosLength >> 8), (byte)sosLength, (byte)components.Count]);
        foreach (var c in components)
        {
            output.Write([(byte)c.Id, (byte)(c.Table << 4 | c.Table)]);
        }

        output.Write([0, 63, 0]);
        var writer = new BitWriter(output);
        var previous = new int[components.Count];
        foreach (var (ci, block) in sequence)
        {
            int t = components[ci].Table;
            int diff = block[0] - previous[ci];
            previous[ci] = block[0];
            int size = Bits(diff);
            writer.Write(dc[t].Codes[size], dc[t].Lengths[size]);
            if (size > 0)
            {
                writer.Write(diff < 0 ? diff - 1 : diff, size);
            }

            int run = 0;
            for (int k = 1; k < 64; k++)
            {
                int v = block[Zigzag[k]];
                if (v == 0)
                {
                    run++;
                    continue;
                }

                while (run > 15)
                {
                    writer.Write(ac[t].Codes[0xF0], ac[t].Lengths[0xF0]);
                    run -= 16;
                }

                int bits = Bits(v), symbol = (run << 4) + bits;
                writer.Write(ac[t].Codes[symbol], ac[t].Lengths[symbol]);
                writer.Write(v < 0 ? v - 1 : v, bits);
                run = 0;
            }

            if (run > 0)
            {
                writer.Write(ac[t].Codes[0], ac[t].Lengths[0]);
            }
        }

        writer.Flush();
        output.Write([0xFF, 0xD9]);
        return output.ToArray();
    }

    private static int DivUp(int a, int b) => (a + b - 1) / b;

    // The bits of a coefficient's magnitude (its category): 0 for 0.
    private static int Bits(int value)
    {
        int magnitude = Math.Abs(value), bits = 0;
        while (magnitude > 0)
        {
            bits++;
            magnitude >>= 1;
        }

        return bits;
    }

    // jpeg_add_quant_table with force_baseline: (base * scale + 50) / 100 within 1 to 255.
    private static int[] Scaled(int[] table, int scale) => [.. table.Select(q => Math.Clamp((q * scale + 50) / 100, 1, 255))];

    // jccolor.c rgb_ycc_convert: 16-bit fixed point, Y rounded by ONE_HALF, Cb and Cr by ONE_HALF - 1 with the 128 offset.
    private static (byte[] Y, byte[] Cb, byte[] Cr) ToYCbCr(byte[][] rgb, int size)
    {
        const int Scale = 16, Half = 1 << (Scale - 1), Offset = 128 << Scale;
        static int Fix(double x) => (int)(x * (1L << Scale) + 0.5);
        int ry = Fix(0.29900), gy = Fix(0.58700), by = Fix(0.11400), rcb = -Fix(0.16874), gcb = -Fix(0.33126), half = Fix(0.5), gcr = -Fix(0.41869), bcr = -Fix(0.08131);
        var y = new byte[size];
        var cb = new byte[size];
        var cr = new byte[size];
        byte[] r = rgb[0], g = rgb[1], b = rgb[2];
        for (int i = 0; i < size; i++)
        {
            int R = r[i], G = g[i], B = b[i];
            y[i] = (byte)((ry * R + gy * G + (by * B + Half)) >> Scale);
            cb[i] = (byte)((rcb * R + gcb * G + (half * B + Offset + Half - 1)) >> Scale);
            cr[i] = (byte)((half * R + Offset + Half - 1 + gcr * G + bcr * B) >> Scale);
        }

        return (y, cb, cr);
    }

    private sealed class Component(int id, int h, int v, int table, byte[] full)
    {
        public int Id { get; } = id;

        public int H { get; } = h;

        public int V { get; } = v;

        public int Table { get; } = table;

        // The component's samples, padded as libjpeg pads them, PaddedBlocksAcross * 8 wide and PaddedBlocksDown * 8 high.
        public byte[] Samples { get; private set; } = [];

        // Blocks with real samples (width_in_blocks, height_in_blocks), and the blocks the scan holds (MCU-padded).
        public int BlocksAcross { get; private set; }

        public int BlocksDown { get; private set; }

        public int PaddedBlocksAcross { get; private set; }

        public int PaddedBlocksDown { get; private set; }

        public short[][] Coefficients { get; set; } = [];

        // Downsample (jcsample.c) from the full-size plane, padding edges by repetition (jcprepct.c, jcsample.c).
        public void Prepare(int width, int height, int maxH, int maxV, int paddedAcross, int paddedDown)
        {
            BlocksAcross = DivUp(width * H, maxH * 8);
            BlocksDown = DivUp(height * V, maxV * 8);
            PaddedBlocksAcross = paddedAcross > 0 ? paddedAcross : BlocksAcross;
            PaddedBlocksDown = paddedDown > 0 ? paddedDown : BlocksDown;
            int fx = maxH / H, fy = maxV / V;
            int outW = BlocksAcross * 8;                                                         // output_cols
            int outRows = DivUp(height, fy);                                                     // rows the image's row groups give
            int samplesW = PaddedBlocksAcross * 8, samplesH = Math.Max(PaddedBlocksDown * 8, BlocksDown * 8);
            var samples = new byte[samplesW * samplesH];
            var input = full;

            byte In(int row, int x) => input[Math.Min(row, height - 1) * width + Math.Min(x, width - 1)];

            for (int oy = 0; oy < outRows; oy++)
            {
                var line = samples.AsSpan(oy * samplesW, samplesW);
                if (fx == 1 && fy == 1)
                {
                    for (int x = 0; x < outW; x++)
                    {
                        line[x] = In(oy, x);
                    }
                }
                else if (fx == 2 && fy == 1)
                {
                    int bias = 0;                                                                // h2v1_downsample: 0, 1, 0, 1, ...
                    for (int x = 0; x < outW; x++)
                    {
                        line[x] = (byte)((In(oy, 2 * x) + In(oy, 2 * x + 1) + bias) >> 1);
                        bias ^= 1;
                    }
                }
                else if (fx == 2 && fy == 2)
                {
                    int bias = 1;                                                                // h2v2_downsample: 1, 2, 1, 2, ...
                    for (int x = 0; x < outW; x++)
                    {
                        line[x] = (byte)((In(2 * oy, 2 * x) + In(2 * oy, 2 * x + 1) + In(2 * oy + 1, 2 * x) + In(2 * oy + 1, 2 * x + 1) + bias) >> 2);
                        bias ^= 3;
                    }
                }
                else
                {
                    throw new NotSupportedException($"Sampling {fx} x {fy} is not supported.");
                }

                for (int x = outW; x < samplesW; x++)
                {
                    line[x] = line[outW - 1];
                }
            }

            for (int oy = outRows; oy < samplesH; oy++)
            {
                samples.AsSpan((outRows - 1) * samplesW, samplesW).CopyTo(samples.AsSpan(oy * samplesW, samplesW));   // expand_bottom_edge
            }

            Samples = samples;
        }
    }

    // The forward DCT and quantization of every block the scan holds: real blocks from the samples, dummy blocks (past the
    // component's width or height in an MCU) all zero but for a DC equal to the block before (right edge) or the last
    // block of the row above (bottom edge), as jccoefct.c makes them.
    private static short[][] Transform(Component c, int[] table)
    {
        var divisors = new (int Reciprocal, int Correction, int Shift)[64];
        for (int i = 0; i < 64; i++)
        {
            divisors[i] = Reciprocal(table[i] << 3);
        }

        int across = c.PaddedBlocksAcross, down = c.PaddedBlocksDown, stride = across * 8;
        var blocks = new short[across * down][];
        Abstraction.Devices.HostParallel.For(Math.Min(down, c.BlocksDown), 1, (first, last) =>
        {
            var workspace = new int[64];
            for (int by = first; by < last; by++)
            {
                for (int bx = 0; bx < c.BlocksAcross; bx++)
                {
                    for (int y = 0; y < 8; y++)
                    {
                        for (int x = 0; x < 8; x++)
                        {
                            workspace[y * 8 + x] = c.Samples[(by * 8 + y) * stride + bx * 8 + x] - 128;
                        }
                    }

                    ForwardDct(workspace);
                    var block = new short[64];
                    for (int i = 0; i < 64; i++)
                    {
                        int temp = workspace[i];
                        var (recip, corr, shift) = divisors[i];
                        if (temp < 0)
                        {
                            temp = -(int)((uint)(-temp + corr) * (uint)recip >> (shift + 16));
                        }
                        else
                        {
                            temp = (int)((uint)(temp + corr) * (uint)recip >> (shift + 16));
                        }

                        block[i] = (short)temp;
                    }

                    blocks[by * across + bx] = block;
                }
            }
        });

        for (int by = 0; by < down; by++)
        {
            for (int bx = 0; bx < across; bx++)
            {
                if (blocks[by * across + bx] is not null)
                {
                    continue;
                }

                var dummy = new short[64];
                if (by < c.BlocksDown)
                {
                    dummy[0] = blocks[by * across + bx - 1][0];                                  // right edge: the block before
                }
                else
                {
                    // Bottom edge: the DC of the last block of the MCU row above in this MCU (jccoefct.c's blkn - 1).
                    int mcuX = bx / c.H * c.H;
                    int rowAbove = by - 1;
                    dummy[0] = blocks[rowAbove * across + mcuX + c.H - 1][0];
                }

                blocks[by * across + bx] = dummy;
            }
        }

        return blocks;
    }

    // jcdctmgr.c compute_reciprocal for a 16-bit DCTELEM (libjpeg-turbo with SIMD, as Pillow's wheels build it).
    private static (int Reciprocal, int Correction, int Shift) Reciprocal(int divisor)
    {
        if (divisor == 1)
        {
            return (1, 0, -16);
        }

        int b = 31 - System.Numerics.BitOperations.LeadingZeroCount((uint)divisor);        // flss(divisor) - 1
        int r = 16 + b;
        long fq = (1L << r) / divisor, fr = (1L << r) % divisor;
        int c = divisor / 2;
        if (fr == 0)
        {
            fq >>= 1;
            r--;
        }
        else if (fr <= divisor / 2)
        {
            c++;
        }
        else
        {
            fq++;
        }

        return ((int)(fq & 0xFFFF), c, r - 16);
    }

    // jfdctint.c jpeg_fdct_islow: outputs scaled up by 8.
    private static void ForwardDct(int[] data)
    {
        const int ConstBits = 13, Pass1Bits = 2;
        const int F0298 = 2446, F0390 = 3196, F0541 = 4433, F0765 = 6270, F0899 = 7373, F1175 = 9633, F1501 = 12299,
            F1847 = 15137, F1961 = 16069, F2053 = 16819, F2562 = 20995, F3072 = 25172;
        static int Descale(int x, int n) => (x + (1 << (n - 1))) >> n;

        for (int row = 0; row < 8; row++)
        {
            int o = row * 8;
            int tmp0 = data[o] + data[o + 7], tmp7 = data[o] - data[o + 7];
            int tmp1 = data[o + 1] + data[o + 6], tmp6 = data[o + 1] - data[o + 6];
            int tmp2 = data[o + 2] + data[o + 5], tmp5 = data[o + 2] - data[o + 5];
            int tmp3 = data[o + 3] + data[o + 4], tmp4 = data[o + 3] - data[o + 4];
            int tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3, tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2;
            data[o] = (tmp10 + tmp11) << Pass1Bits;
            data[o + 4] = (tmp10 - tmp11) << Pass1Bits;
            int z1 = (tmp12 + tmp13) * F0541;
            data[o + 2] = Descale(z1 + tmp13 * F0765, ConstBits - Pass1Bits);
            data[o + 6] = Descale(z1 + tmp12 * -F1847, ConstBits - Pass1Bits);
            z1 = tmp4 + tmp7;
            int z2 = tmp5 + tmp6, z3 = tmp4 + tmp6, z4 = tmp5 + tmp7;
            int z5 = (z3 + z4) * F1175;
            tmp4 *= F0298;
            tmp5 *= F2053;
            tmp6 *= F3072;
            tmp7 *= F1501;
            z1 *= -F0899;
            z2 *= -F2562;
            z3 *= -F1961;
            z4 *= -F0390;
            z3 += z5;
            z4 += z5;
            data[o + 7] = Descale(tmp4 + z1 + z3, ConstBits - Pass1Bits);
            data[o + 5] = Descale(tmp5 + z2 + z4, ConstBits - Pass1Bits);
            data[o + 3] = Descale(tmp6 + z2 + z3, ConstBits - Pass1Bits);
            data[o + 1] = Descale(tmp7 + z1 + z4, ConstBits - Pass1Bits);
        }

        for (int col = 0; col < 8; col++)
        {
            int tmp0 = data[col] + data[col + 56], tmp7 = data[col] - data[col + 56];
            int tmp1 = data[col + 8] + data[col + 48], tmp6 = data[col + 8] - data[col + 48];
            int tmp2 = data[col + 16] + data[col + 40], tmp5 = data[col + 16] - data[col + 40];
            int tmp3 = data[col + 24] + data[col + 32], tmp4 = data[col + 24] - data[col + 32];
            int tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3, tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2;
            data[col] = Descale(tmp10 + tmp11, Pass1Bits);
            data[col + 32] = Descale(tmp10 - tmp11, Pass1Bits);
            int z1 = (tmp12 + tmp13) * F0541;
            data[col + 16] = Descale(z1 + tmp13 * F0765, ConstBits + Pass1Bits);
            data[col + 48] = Descale(z1 + tmp12 * -F1847, ConstBits + Pass1Bits);
            z1 = tmp4 + tmp7;
            int z2 = tmp5 + tmp6, z3 = tmp4 + tmp6, z4 = tmp5 + tmp7;
            int z5 = (z3 + z4) * F1175;
            tmp4 *= F0298;
            tmp5 *= F2053;
            tmp6 *= F3072;
            tmp7 *= F1501;
            z1 *= -F0899;
            z2 *= -F2562;
            z3 *= -F1961;
            z4 *= -F0390;
            z3 += z5;
            z4 += z5;
            data[col + 56] = Descale(tmp4 + z1 + z3, ConstBits + Pass1Bits);
            data[col + 40] = Descale(tmp5 + z2 + z4, ConstBits + Pass1Bits);
            data[col + 24] = Descale(tmp6 + z2 + z3, ConstBits + Pass1Bits);
            data[col + 8] = Descale(tmp7 + z1 + z4, ConstBits + Pass1Bits);
        }
    }

    private sealed class HuffmanTable
    {
        public HuffmanTable(byte[] bits, byte[] values)
        {
            Bits = bits;
            Values = values;
            int code = 0, k = 0;
            for (int length = 1; length <= 16; length++)
            {
                for (int i = 0; i < bits[length - 1]; i++)
                {
                    Codes[values[k]] = code++;
                    Lengths[values[k]] = length;
                    k++;
                }

                code <<= 1;
            }
        }

        public byte[] Bits { get; }

        public byte[] Values { get; }

        public int[] Codes { get; } = new int[256];

        public int[] Lengths { get; } = new int[256];
    }

    // jchuff.c jpeg_gen_optimal_table: Huffman code lengths from the symbol counts (a reserved symbol 256 keeps any code
    // from being all ones), lengths over 16 folded as Annex K.3 does.
    private static HuffmanTable OptimalTable(long[] counts)
    {
        const int MaxLength = 32;
        var freq = (long[])counts.Clone();
        var bits = new int[MaxLength + 1];
        var codeSize = new int[257];
        var others = Enumerable.Repeat(-1, 257).ToArray();
        freq[256] = 1;
        while (true)
        {
            int c1 = -1, c2 = -1;
            long v = 1000000000L;
            for (int i = 0; i <= 256; i++)
            {
                if (freq[i] != 0 && freq[i] <= v)
                {
                    v = freq[i];
                    c1 = i;
                }
            }

            v = 1000000000L;
            for (int i = 0; i <= 256; i++)
            {
                if (freq[i] != 0 && freq[i] <= v && i != c1)
                {
                    v = freq[i];
                    c2 = i;
                }
            }

            if (c2 < 0)
            {
                break;
            }

            freq[c1] += freq[c2];
            freq[c2] = 0;
            codeSize[c1]++;
            while (others[c1] >= 0)
            {
                c1 = others[c1];
                codeSize[c1]++;
            }

            others[c1] = c2;
            codeSize[c2]++;
            while (others[c2] >= 0)
            {
                c2 = others[c2];
                codeSize[c2]++;
            }
        }

        for (int i = 0; i <= 256; i++)
        {
            if (codeSize[i] != 0)
            {
                bits[codeSize[i]]++;
            }
        }

        for (int i = MaxLength; i > 16; i--)
        {
            while (bits[i] > 0)
            {
                int j = i - 2;
                while (bits[j] == 0)
                {
                    j--;
                }

                bits[i] -= 2;
                bits[i - 1]++;
                bits[j + 1] += 2;
                bits[j]--;
            }
        }

        int last = 16;
        while (bits[last] == 0)
        {
            last--;
        }

        bits[last]--;                                                                            // drop the reserved symbol
        var values = new List<byte>();
        for (int length = 1; length <= MaxLength; length++)
        {
            for (int symbol = 0; symbol <= 255; symbol++)
            {
                if (codeSize[symbol] == length)
                {
                    values.Add((byte)symbol);
                }
            }
        }

        return new HuffmanTable([.. bits.Skip(1).Take(16).Select(b => (byte)b)], [.. values]);
    }

    private static void WriteTable(Stream output, HuffmanTable table, byte classAndId)
    {
        int length = 2 + 1 + 16 + table.Values.Length;
        output.Write([0xFF, 0xC4, (byte)(length >> 8), (byte)length, classAndId]);
        output.Write(table.Bits);
        output.Write(table.Values);
    }

    private sealed class BitWriter(Stream output)
    {
        private int _buffer, _count;

        public void Write(int value, int length)
        {
            for (int i = length - 1; i >= 0; i--)
            {
                _buffer = _buffer << 1 | (value >> i & 1);
                if (++_count == 8)
                {
                    Emit((byte)_buffer);
                    _buffer = 0;
                    _count = 0;
                }
            }
        }

        public void Flush()
        {
            while (_count != 0)
            {
                Write(1, 1);                                                                     // pad with ones
            }
        }

        private void Emit(byte value)
        {
            output.WriteByte(value);
            if (value == 0xFF)
            {
                output.WriteByte(0);
            }
        }
    }
}
