// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Binary;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace Idrak.Imaging;

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
/// <para>
/// One encode allocates its file and little else: an 8-bit image's planes are read in place and floats brought to bytes
/// a row at a time; the colour conversion, downsampling, DCT and quantization run an MCU row at a time on all cores
/// (<see cref="ComputeResources.MaxCpuThreads"/>) into pooled coefficient buffers, a few MCU rows at a time with the
/// standard tables (each written out before the next are made) and the whole image with optimized ones (their
/// statistics need every block first); the entropy-coded bytes go to one pooled buffer, copied once into the file.
/// </para>
/// <para>
/// A round trip (<see cref="PillowImageOps.JpegRoundTrip"/>, the <c>jpeg=Q</c> transform) needs the pixels, not the file:
/// <see cref="RoundTrip"/> writes the same quantized coefficients straight into the library's decoder's arrays and takes
/// its pixels, with nothing entropy coded, written or read back.
/// </para>
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

    // The standard tables, made once: [0] luminance, [1] chrominance.
    private static readonly HuffmanTable[] StandardDc = [new(DcLuminanceBits, DcValues), new(DcChrominanceBits, DcValues)];
    private static readonly HuffmanTable[] StandardAc = [new(AcLuminanceBits, AcLuminanceValues), new(AcChrominanceBits, AcChrominanceValues)];

    // jccolor.c rgb_ycc_convert's factors: 16-bit fixed point, Y rounded by ONE_HALF, Cb and Cr by ONE_HALF - 1 with the
    // 128 offset.
    private const int ColourBits = 16, ColourHalf = 1 << (ColourBits - 1), ColourOffset = 128 << ColourBits;
    private static readonly int RedY = Fix(0.29900), GreenY = Fix(0.58700), BlueY = Fix(0.11400), RedCb = -Fix(0.16874), GreenCb = -Fix(0.33126),
        HalfFactor = Fix(0.5), GreenCr = -Fix(0.41869), BlueCr = -Fix(0.08131);

    // Zigzag order: the natural (row-major) index of the k-th coefficient written. Blocks are kept in this order.
    private static ReadOnlySpan<byte> Zigzag =>
    [
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
    ];

    // Natural (row-major) order: a round trip's blocks, kept as the decoder keeps them.
    private static ReadOnlySpan<byte> RowMajor =>
    [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    ];

    // The symbols of a table class: 256, and the reserved one jpeg_gen_optimal_table adds; a count array's length.
    private const int Symbols = 257;

    /// <summary>
    /// The image (grey: one channel; colour: three) as a JPEG file at <paramref name="quality"/> (1 to 100; Pillow's
    /// default is 75), its pixels first brought to 8 bits as Pillow holds them.
    /// </summary>
    public static byte[] Encode(ImageData image, int quality = 75, bool optimize = false, JpegSubsampling subsampling = JpegSubsampling.Half420)
    {
        var source = SourceOf(image, quality);
        return Write(source, image.Channels == 3, quality, optimize, subsampling);
    }

    /// <summary>
    /// The pixels the library's JPEG decoder gives for the file <see cref="Encode(ImageData, int, bool, JpegSubsampling)"/>
    /// writes at <paramref name="quality"/> (with standard or optimized tables alike: Huffman tables change the file, never
    /// its pixels), without the file: every MCU row's quantized coefficients go straight into the decoder's coefficient
    /// arrays, in its order, as its scan would leave them, and its pixel stage (inverse DCT, upsampling, colour conversion)
    /// makes the image. Nothing is entropy coded, written or read back, and no file-sized buffer is held.
    /// </summary>
    internal static ImageData RoundTrip(ImageData image, int quality, JpegSubsampling subsampling)
    {
        var source = SourceOf(image, quality);
        bool colour = image.Channels == 3;
        int scale = Scale(quality), tableCount = colour ? 2 : 1;
        var (hs, vs) = Sampling(colour, subsampling);
        var quantizers = ArrayPool<Quantizer>.Shared.Rent(64 * tableCount);
        JpegDecoder? decoder = null;
        try
        {
            // The divisors in natural order, the order the decoder keeps a block's coefficients in.
            for (int t = 0; t < tableCount; t++)
            {
                for (int n = 0; n < 64; n++)
                {
                    quantizers[t * 64 + n] = Divisor(Quantized(t, n, scale) << 3);
                }
            }

            // The decoder's frame from the frame header and quantization tables the file would hold.
            var job = new Job(source, colour, hs, vs, quantizers);
            Span<byte> frame = stackalloc byte[6 + 3 * 3];
            Span<byte> tables = stackalloc byte[2 * 65];
            int frameLength = FrameBody(job, frame);
            for (int t = 0; t < tableCount; t++)
            {
                TableBody(t, scale, tables.Slice(65 * t, 65));
            }

            decoder = JpegDecoder.ForFrame(frame[..frameLength], tables[..(65 * tableCount)]);
            var targets = new short[job.Components.Length][];
            for (int i = 0; i < targets.Length; i++)
            {
                targets[i] = decoder.Coefficients(i);
            }

            // Every MCU row on all cores, each into its own blocks of the decoder's arrays.
            job.Targets = targets;
            HostParallel.For(job.McusDown, 1, [MethodImpl(MethodImplOptions.AggressiveOptimization)] (from, to) => TransformRows(job, from, to));
            return JpegCodec.ToImage(decoder.Pixels());
        }
        finally
        {
            decoder?.ReturnCoefficients();
            ArrayPool<Quantizer>.Shared.Return(quantizers);
        }
    }

    // The image to encode, checked (grey or RGB; a quality and size a JPEG takes). An 8-bit image is read in place (its
    // planes are the bytes ImagePreprocessor.ToBytes would copy); floats are brought to bytes a row at a time, rounded as
    // ToBytes rounds them.
    private static Source SourceOf(ImageData image, int quality)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (image.Channels is not 1 and not 3)
        {
            throw new ArgumentException($"A JPEG holds a grey or an RGB image, not {image.Channels} channels.", nameof(image));
        }

        CheckSettings(quality, image.Height, image.Width);
        return image.TryGetPlanes(out var planes, out _)
            ? new Source(planes, null, image.Height, image.Width)
            : new Source(null, image.Pixels, image.Height, image.Width);
    }

    /// <summary>8-bit planes (one grey, or red, green and blue) of <paramref name="height"/> x <paramref name="width"/> as a JPEG file.</summary>
    public static byte[] Encode(byte[][] planes, int height, int width, int quality = 75, bool optimize = false, JpegSubsampling subsampling = JpegSubsampling.Half420)
    {
        ArgumentNullException.ThrowIfNull(planes);
        CheckSettings(quality, height, width);
        bool given = planes.Length is 1 or 3;
        var held = new ReadOnlyMemory<byte>[planes.Length];
        for (int i = 0; i < planes.Length && given; i++)
        {
            given = planes[i] is { } plane && plane.Length == (long)width * height;
            held[i] = planes[i];
        }

        if (!given)
        {
            throw new ArgumentException($"Give one or three planes of {width} x {height} bytes.", nameof(planes));
        }

        return Write(new Source(held, null, height, width), planes.Length == 3, quality, optimize, subsampling);
    }

    private static void CheckSettings(int quality, int height, int width)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(quality, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(quality, 100);
        if (width is < 1 or > 65535 || height is < 1 or > 65535)
        {
            throw new ArgumentException($"A JPEG is 1 to 65,535 pixels a side, not {width} x {height}.");
        }
    }

    // jpeg_quality_scaling: the percentage the standard quantization tables are scaled by.
    private static int Scale(int quality) => quality < 50 ? 5000 / quality : 200 - quality * 2;

    // Luma's sampling factors (chroma's are 1 x 1): 2 x 2 for 4:2:0, 2 x 1 for 4:2:2, 1 x 1 for 4:4:4 and for grey.
    private static (int H, int V) Sampling(bool colour, JpegSubsampling subsampling) =>
        (colour && subsampling != JpegSubsampling.Full444 ? 2 : 1, colour && subsampling == JpegSubsampling.Half420 ? 2 : 1);

    private static byte[] Write(Source source, bool colour, int quality, bool optimize, JpegSubsampling subsampling)
    {
        int scale = Scale(quality), tableCount = colour ? 2 : 1;
        var (hs, vs) = Sampling(colour, subsampling);
        var quantizers = ArrayPool<Quantizer>.Shared.Rent(64 * tableCount);
        short[]?[] buffers = [];
        long[]? counts = null;
        var writer = default(BitWriter);
        try
        {
            for (int t = 0; t < tableCount; t++)
            {
                for (int k = 0; k < 64; k++)
                {
                    quantizers[t * 64 + k] = Divisor(Quantized(t, Zigzag[k], scale) << 3);
                }
            }

            // A chunk of MCU rows: enough for every core (HostParallel splits a count into four parts a thread), made and
            // written before the next with the standard tables; every chunk kept until the optimized tables are known.
            var job = new Job(source, colour, hs, vs, quantizers);
            int chunkRows = Math.Min(job.McusDown, 4 * Math.Max(1, ComputeResources.MaxCpuThreads));
            int chunkCount = DivUp(job.McusDown, chunkRows), chunkLength = checked(chunkRows * job.RowStride);
            buffers = new short[optimize ? chunkCount : 1][];
            writer = new BitWriter(ArrayPool<byte>.Shared.Rent((int)Math.Clamp((long)source.Width * source.Height * (colour ? 3 : 1) / 4, 1 << 12, 1 << 24)));
            Action<int, int> transform = [MethodImpl(MethodImplOptions.AggressiveOptimization)] (from, to) => TransformRows(job, from, to);

            // The chunk's coefficients into its buffer; the MCU rows it held.
            int Transform(int chunk, short[] buffer)
            {
                job.Buffer = buffer;
                job.ChunkStart = chunk * chunkRows;
                int rows = Math.Min(chunkRows, job.McusDown - job.ChunkStart);
                HostParallel.For(rows, 1, transform);
                return rows;
            }

            Span<int> last = stackalloc int[3];
            HuffmanTable[] dc = StandardDc, ac = StandardAc;
            if (optimize)
            {
                counts = ArrayPool<long>.Shared.Rent(2 * Symbols * tableCount);
                counts.AsSpan(0, 2 * Symbols * tableCount).Clear();
                for (int chunk = 0; chunk < chunkCount; chunk++)
                {
                    short[] buffer = ArrayPool<short>.Shared.Rent(chunkLength);
                    buffers[chunk] = buffer;
                    Count(job, buffer, Transform(chunk, buffer), counts, last);
                }

                (dc, ac) = (new HuffmanTable[tableCount], new HuffmanTable[tableCount]);
                for (int t = 0; t < tableCount; t++)
                {
                    dc[t] = OptimalTable(counts.AsSpan(2 * Symbols * t, Symbols));
                    ac[t] = OptimalTable(counts.AsSpan(2 * Symbols * t + Symbols, Symbols));
                }

                last.Clear();
            }

            WriteHeaders(ref writer, job, scale, dc, ac);
            for (int chunk = 0; chunk < chunkCount; chunk++)
            {
                if (optimize)
                {
                    Scan(ref writer, job, buffers[chunk]!, Math.Min(chunkRows, job.McusDown - chunk * chunkRows), dc, ac, last);
                }
                else
                {
                    short[] buffer = buffers[0] ?? (buffers[0] = ArrayPool<short>.Shared.Rent(chunkLength));
                    Scan(ref writer, job, buffer, Transform(chunk, buffer), dc, ac, last);
                }
            }

            writer.Flush();
            writer.Write([0xFF, 0xD9]);
            return writer.ToFile();
        }
        finally
        {
            foreach (var buffer in buffers)
            {
                if (buffer is not null)
                {
                    ArrayPool<short>.Shared.Return(buffer);
                }
            }

            if (counts is not null)
            {
                ArrayPool<long>.Shared.Return(counts);
            }

            ArrayPool<Quantizer>.Shared.Return(quantizers);
            if (writer.Buffer is { } output)
            {
                ArrayPool<byte>.Shared.Return(output);
            }
        }
    }

    private static int DivUp(int a, int b) => (a + b - 1) / b;

    private static int Fix(double x) => (int)(x * (1L << ColourBits) + 0.5);

    // jpeg_add_quant_table with force_baseline: (base * scale + 50) / 100 within 1 to 255, for table t's natural index.
    private static int Quantized(int table, int index, int scale) => Math.Clamp(((table == 0 ? StdLuminance : StdChrominance)[index] * scale + 50) / 100, 1, 255);

    // The bits of a coefficient's magnitude (its category): 0 for 0.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Category(int value) => 32 - BitOperations.LeadingZeroCount((uint)(value < 0 ? -value : value));

    // jcdctmgr.c compute_reciprocal for a 16-bit DCTELEM (libjpeg-turbo with SIMD, as Pillow's wheels build it); the
    // shift kept with its 16 added, as the quantization shifts.
    private static Quantizer Divisor(int divisor)
    {
        if (divisor == 1)
        {
            return new Quantizer(1, 0, 0);
        }

        int b = 31 - BitOperations.LeadingZeroCount((uint)divisor);                          // flss(divisor) - 1
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

        return new Quantizer((int)(fq & 0xFFFF), c, r);
    }

    // The MCU rows [first, last) of the job's chunk: each one's blocks transformed and quantized into each component's
    // part of the row (Job.Blocks: the chunk's buffer, or the decoder's arrays for a round trip), with the scratch they
    // need.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void TransformRows(Job job, int first, int last)
    {
        Span<int> workspace = stackalloc int[64];
        Span<int> upper = stackalloc int[8];
        Span<int> lower = stackalloc int[8];
        byte[]? scratch = job.ScratchLength > 0 ? ArrayPool<byte>.Shared.Rent(job.ScratchLength) : null;
        try
        {
            for (int local = first; local < last; local++)
            {
                int m = job.ChunkStart + local;
                if (job.Colour)
                {
                    ColourRow(job, m, job.Blocks(0, m), job.Blocks(1, m), job.Blocks(2, m), scratch, workspace, upper, lower);
                }
                else
                {
                    GreyRow(job, m, job.Blocks(0, m), scratch, workspace, upper);
                }
            }
        }
        finally
        {
            if (scratch is not null)
            {
                ArrayPool<byte>.Shared.Return(scratch);
            }
        }
    }

    // A grey MCU row (eight rows of the image, one block each MCU) into its blocks, read from the plane, or from its
    // floats brought to bytes; rows past the bottom repeat the last, columns past the right edge the last (jcprepct.c's
    // padding).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GreyRow(Job job, int m, Span<short> blocks, byte[]? scratch, Span<int> workspace, Span<int> starts)
    {
        var order = Order(job);
        int width = job.Width, height = job.Height, top = m * 8;
        ReadOnlySpan<byte> plane;
        if (job.Source.Floats)
        {
            var rows = scratch.AsSpan(0, 8 * width);
            int real = Math.Min(8, height - top);
            for (int r = 0; r < real; r++)
            {
                job.Source.Convert(0, top + r, rows.Slice(r * width, width));
            }

            plane = rows;
            for (int y = 0; y < 8; y++)
            {
                starts[y] = Math.Min(y, real - 1) * width;
            }
        }
        else
        {
            plane = job.Source.Plane(0);
            for (int y = 0; y < 8; y++)
            {
                starts[y] = Math.Min(top + y, height - 1) * width;
            }
        }

        ReadOnlySpan<Quantizer> divisors = job.Quantizers.AsSpan(0, 64);
        for (int bx = 0; bx < job.McusAcross; bx++)
        {
            int left = bx * 8;
            if (left + 8 <= width)
            {
                for (int y = 0; y < 8; y++)
                {
                    var source = plane.Slice(starts[y] + left, 8);
                    for (int x = 0; x < 8; x++)
                    {
                        workspace[y * 8 + x] = source[x] - 128;
                    }
                }
            }
            else
            {
                for (int y = 0; y < 8; y++)
                {
                    for (int x = 0; x < 8; x++)
                    {
                        workspace[y * 8 + x] = plane[starts[y] + Math.Min(left + x, width - 1)] - 128;
                    }
                }
            }

            ForwardDct(workspace);
            Quantize(workspace, blocks.Slice(bx * 64, 64), divisors, order);
        }
    }

    // A colour MCU row (8 x maxV rows of the image) into each component's blocks: its real rows to YCbCr (jccolor.c) in a
    // stripe as wide as the MCUs, the columns past the right edge repeating the last; luma's blocks read from it (rows
    // past the bottom repeating the last), chroma's downsampled from it as jcsample.c does (h2v1: bias 0, 1, 0, 1, ...;
    // h2v2: 1, 2, 1, 2, ...; the downsampled rows past the image's repeating its last, jcsample.c's expand_bottom_edge);
    // then luma's dummy blocks.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ColourRow(Job job, int m, Span<short> lumaBlocks, Span<short> cbBlocks, Span<short> crBlocks, byte[]? scratch, Span<int> workspace,
        Span<int> upper, Span<int> lower)
    {
        var order = Order(job);
        int width = job.Width, height = job.Height, rows = 8 * job.MaxV, stripe = job.StripeWidth, top = m * rows;
        int real = Math.Min(rows, height - top);
        var lumaPlane = scratch.AsSpan(0, rows * stripe);
        var cbPlane = scratch.AsSpan(rows * stripe, rows * stripe);
        var crPlane = scratch.AsSpan(2 * rows * stripe, rows * stripe);
        for (int r = 0; r < real; r++)
        {
            ReadOnlySpan<byte> red, green, blue;
            if (job.Source.Floats)
            {
                var converted = scratch.AsSpan(3 * rows * stripe, 3 * width);
                job.Source.Convert(0, top + r, converted[..width]);
                job.Source.Convert(1, top + r, converted.Slice(width, width));
                job.Source.Convert(2, top + r, converted.Slice(2 * width, width));
                red = converted[..width];
                green = converted.Slice(width, width);
                blue = converted.Slice(2 * width, width);
            }
            else
            {
                int offset = (top + r) * width;
                red = job.Source.Plane(0).Slice(offset, width);
                green = job.Source.Plane(1).Slice(offset, width);
                blue = job.Source.Plane(2).Slice(offset, width);
            }

            ToYCbCr(red, green, blue, lumaPlane.Slice(r * stripe, stripe), cbPlane.Slice(r * stripe, stripe), crPlane.Slice(r * stripe, stripe));
        }

        // Luma: the real blocks of each of the MCU row's block rows.
        var luma = job.Components[0];
        ReadOnlySpan<Quantizer> divisors = job.Quantizers.AsSpan(0, 64);
        for (int by = 0; by < luma.V; by++)
        {
            int blockRow = m * luma.V + by;
            if (blockRow >= luma.BlocksDown)
            {
                continue;
            }

            for (int y = 0; y < 8; y++)
            {
                upper[y] = (Math.Min(blockRow * 8 + y, height - 1) - top) * stripe;
            }

            for (int bx = 0; bx < luma.BlocksAcross; bx++)
            {
                for (int y = 0; y < 8; y++)
                {
                    var source = lumaPlane.Slice(upper[y] + bx * 8, 8);
                    for (int x = 0; x < 8; x++)
                    {
                        workspace[y * 8 + x] = source[x] - 128;
                    }
                }

                ForwardDct(workspace);
                Quantize(workspace, lumaBlocks.Slice((by * luma.PaddedAcross + bx) << 6, 64), divisors, order);
            }
        }

        // Chroma: one block row, every block real; each sample from fx x fy samples of the stripe.
        int fx = job.MaxH, fy = job.MaxV, outRows = DivUp(height, fy);
        for (int y = 0; y < 8; y++)
        {
            int downsampled = Math.Min(m * 8 + y, outRows - 1);
            upper[y] = (Math.Min(fy * downsampled, height - 1) - top) * stripe;
            lower[y] = (Math.Min(fy * downsampled + fy - 1, height - 1) - top) * stripe;
        }

        divisors = job.Quantizers.AsSpan(64, 64);
        for (int ci = 1; ci < 3; ci++)
        {
            var plane = ci == 1 ? cbPlane : crPlane;
            var blocks = ci == 1 ? cbBlocks : crBlocks;
            for (int bx = 0; bx < job.McusAcross; bx++)
            {
                for (int y = 0; y < 8; y++)
                {
                    if (fx == 1)
                    {
                        var source = plane.Slice(upper[y] + bx * 8, 8);
                        for (int x = 0; x < 8; x++)
                        {
                            workspace[y * 8 + x] = source[x] - 128;
                        }
                    }
                    else if (fy == 1)
                    {
                        var source = plane.Slice(upper[y] + bx * 16, 16);
                        for (int x = 0; x < 8; x++)
                        {
                            workspace[y * 8 + x] = ((source[2 * x] + source[2 * x + 1] + (x & 1)) >> 1) - 128;
                        }
                    }
                    else
                    {
                        var above = plane.Slice(upper[y] + bx * 16, 16);
                        var below = plane.Slice(lower[y] + bx * 16, 16);
                        for (int x = 0; x < 8; x++)
                        {
                            workspace[y * 8 + x] = ((above[2 * x] + above[2 * x + 1] + below[2 * x] + below[2 * x + 1] + 1 + (x & 1)) >> 2) - 128;
                        }
                    }
                }

                ForwardDct(workspace);
                Quantize(workspace, blocks.Slice(bx << 6, 64), divisors, order);
            }
        }

        // Luma's dummy blocks (past its width or height in an MCU), all zero but for a DC equal to the block before (right
        // edge) or the last block of the MCU's block row above (bottom edge), as jccoefct.c makes them; in row order, so
        // each one's source is already there. The DC is coefficient 0 in either order.
        for (int by = 0; by < luma.V; by++)
        {
            bool below = m * luma.V + by >= luma.BlocksDown;
            for (int bx = below ? 0 : luma.BlocksAcross; bx < luma.PaddedAcross; bx++)
            {
                int source = below ? (by - 1) * luma.PaddedAcross + bx / luma.H * luma.H + luma.H - 1 : by * luma.PaddedAcross + bx - 1;
                short dc = lumaBlocks[source << 6];
                var block = lumaBlocks.Slice((by * luma.PaddedAcross + bx) << 6, 64);
                block.Clear();
                block[0] = dc;
            }
        }
    }

    // The order a job's blocks are written in: zigzag for the file's scan, natural for a round trip (the decoder's).
    private static ReadOnlySpan<byte> Order(Job job) => job.Targets is null ? Zigzag : RowMajor;

    // jccolor.c rgb_ycc_convert for one row, then the columns past it repeating its last (the stripe's width).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ToYCbCr(ReadOnlySpan<byte> red, ReadOnlySpan<byte> green, ReadOnlySpan<byte> blue, Span<byte> y, Span<byte> cb, Span<byte> cr)
    {
        int width = red.Length;
        green = green[..width];
        blue = blue[..width];
        for (int x = 0; x < width; x++)
        {
            int r = red[x], g = green[x], b = blue[x];
            y[x] = (byte)((RedY * r + GreenY * g + (BlueY * b + ColourHalf)) >> ColourBits);
            cb[x] = (byte)((RedCb * r + GreenCb * g + (HalfFactor * b + ColourOffset + ColourHalf - 1)) >> ColourBits);
            cr[x] = (byte)((HalfFactor * r + ColourOffset + ColourHalf - 1 + GreenCr * g + BlueCr * b) >> ColourBits);
        }

        y[width..].Fill(y[width - 1]);
        cb[width..].Fill(cb[width - 1]);
        cr[width..].Fill(cr[width - 1]);
    }

    // libjpeg-turbo's quantization by reciprocals (jcdctmgr.c quantize), written in `order` (the block's coefficient k is
    // the DCT's order[k]: zigzag for the file, natural for a round trip), the divisors given in the same order.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Quantize(ReadOnlySpan<int> workspace, Span<short> block, ReadOnlySpan<Quantizer> divisors, ReadOnlySpan<byte> order)
    {
        order = order[..64];
        divisors = divisors[..64];
        block = block[..64];
        for (int k = 0; k < 64; k++)
        {
            int temp = workspace[order[k]];
            var (recip, corr, shift) = divisors[k];
            if (temp < 0)
            {
                temp = -(int)((uint)(-temp + corr) * (uint)recip >> shift);
            }
            else
            {
                temp = (int)((uint)(temp + corr) * (uint)recip >> shift);
            }

            block[k] = (short)temp;
        }
    }

    // jfdctint.c jpeg_fdct_islow: outputs scaled up by 8.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ForwardDct(Span<int> data)
    {
        const int ConstBits = 13, Pass1Bits = 2;
        const int F0298 = 2446, F0390 = 3196, F0541 = 4433, F0765 = 6270, F0899 = 7373, F1175 = 9633, F1501 = 12299,
            F1847 = 15137, F1961 = 16069, F2053 = 16819, F2562 = 20995, F3072 = 25172;
        static int Descale(int x, int n) => (x + (1 << (n - 1))) >> n;

        data = data[..64];
        for (int row = 0; row < 8; row++)
        {
            var d = data.Slice(row * 8, 8);
            int tmp0 = d[0] + d[7], tmp7 = d[0] - d[7];
            int tmp1 = d[1] + d[6], tmp6 = d[1] - d[6];
            int tmp2 = d[2] + d[5], tmp5 = d[2] - d[5];
            int tmp3 = d[3] + d[4], tmp4 = d[3] - d[4];
            int tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3, tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2;
            d[0] = (tmp10 + tmp11) << Pass1Bits;
            d[4] = (tmp10 - tmp11) << Pass1Bits;
            int z1 = (tmp12 + tmp13) * F0541;
            d[2] = Descale(z1 + tmp13 * F0765, ConstBits - Pass1Bits);
            d[6] = Descale(z1 + tmp12 * -F1847, ConstBits - Pass1Bits);
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
            d[7] = Descale(tmp4 + z1 + z3, ConstBits - Pass1Bits);
            d[5] = Descale(tmp5 + z2 + z4, ConstBits - Pass1Bits);
            d[3] = Descale(tmp6 + z2 + z3, ConstBits - Pass1Bits);
            d[1] = Descale(tmp7 + z1 + z4, ConstBits - Pass1Bits);
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

    // The nonzero AC coefficients of a block in zigzag order: bit k for coefficient k (1 to 63), without branches.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong NonZero(ReadOnlySpan<short> block)
    {
        block = block[..64];
        ulong mask = 0;
        for (int k = 1; k < 64; k++)
        {
            int v = block[k];
            mask |= (ulong)((uint)(v | -v) >> 31) << k;
        }

        return mask;
    }

    // The MCU rows of a chunk in scan order (each MCU: each component's H x V blocks, row by row), as the scan holds them.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Scan(ref BitWriter writer, Job job, short[] buffer, int rows, HuffmanTable[] dc, HuffmanTable[] ac, Span<int> last)
    {
        var components = job.Components;
        for (int r = 0; r < rows; r++)
        {
            ReadOnlySpan<short> row = buffer.AsSpan(r * job.RowStride, job.RowStride);
            for (int mx = 0; mx < job.McusAcross; mx++)
            {
                for (int ci = 0; ci < components.Length; ci++)
                {
                    var c = components[ci];
                    HuffmanTable dcTable = dc[c.Table], acTable = ac[c.Table];
                    for (int by = 0; by < c.V; by++)
                    {
                        for (int bx = 0; bx < c.H; bx++)
                        {
                            EncodeBlock(ref writer, row.Slice(c.Offset + ((by * c.PaddedAcross + mx * c.H + bx) << 6), 64), ref last[ci], dcTable, acTable);
                        }
                    }
                }
            }
        }
    }

    // A block's DC difference and its AC runs: each code and its value's bits written together.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void EncodeBlock(ref BitWriter writer, ReadOnlySpan<short> block, ref int last, HuffmanTable dc, HuffmanTable ac)
    {
        int diff = block[0] - last;
        last = block[0];
        int size = Category(diff);
        writer.Put(((uint)dc.Codes[size] << size) | ((uint)(diff < 0 ? diff - 1 : diff) & ((1u << size) - 1)), dc.Lengths[size] + size);
        ulong nonzero = NonZero(block);
        int previous = 0;
        while (nonzero != 0)
        {
            int k = BitOperations.TrailingZeroCount(nonzero);
            nonzero &= nonzero - 1;
            int run = k - previous - 1;
            previous = k;
            for (; run > 15; run -= 16)
            {
                writer.Put((uint)ac.Codes[0xF0], ac.Lengths[0xF0]);
            }

            int v = block[k], bits = Category(v), symbol = (run << 4) + bits;
            writer.Put(((uint)ac.Codes[symbol] << bits) | ((uint)(v < 0 ? v - 1 : v) & ((1u << bits) - 1)), ac.Lengths[symbol] + bits);
        }

        if (previous != 63)
        {
            writer.Put((uint)ac.Codes[0], ac.Lengths[0]);                                   // end of block
        }
    }

    // The symbols the scan would write (jchuff.c's htest_one_block), counted per table: [t] DC then [t] AC.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Count(Job job, short[] buffer, int rows, Span<long> counts, Span<int> last)
    {
        var components = job.Components;
        for (int r = 0; r < rows; r++)
        {
            ReadOnlySpan<short> row = buffer.AsSpan(r * job.RowStride, job.RowStride);
            for (int mx = 0; mx < job.McusAcross; mx++)
            {
                for (int ci = 0; ci < components.Length; ci++)
                {
                    var c = components[ci];
                    var dc = counts.Slice(2 * Symbols * c.Table, Symbols);
                    var ac = counts.Slice(2 * Symbols * c.Table + Symbols, Symbols);
                    for (int by = 0; by < c.V; by++)
                    {
                        for (int bx = 0; bx < c.H; bx++)
                        {
                            var block = row.Slice(c.Offset + ((by * c.PaddedAcross + mx * c.H + bx) << 6), 64);
                            int diff = block[0] - last[ci];
                            last[ci] = block[0];
                            dc[Category(diff)]++;
                            ulong nonzero = NonZero(block);
                            int previous = 0;
                            while (nonzero != 0)
                            {
                                int k = BitOperations.TrailingZeroCount(nonzero);
                                nonzero &= nonzero - 1;
                                int run = k - previous - 1;
                                previous = k;
                                for (; run > 15; run -= 16)
                                {
                                    ac[0xF0]++;
                                }

                                ac[(run << 4) + Category(block[k])]++;
                            }

                            if (previous != 63)
                            {
                                ac[0]++;
                            }
                        }
                    }
                }
            }
        }
    }

    private static void WriteHeaders(ref BitWriter writer, Job job, int scale, HuffmanTable[] dc, HuffmanTable[] ac)
    {
        int tableCount = job.Colour ? 2 : 1, count = job.Components.Length;
        Span<byte> body = stackalloc byte[65];
        writer.Write([0xFF, 0xD8]);
        writer.Write([0xFF, 0xE0, 0, 16, (byte)'J', (byte)'F', (byte)'I', (byte)'F', 0, 1, 1, 0, 0, 1, 0, 1, 0, 0]);   // JFIF 1.01, aspect 1:1
        for (int t = 0; t < tableCount; t++)
        {
            TableBody(t, scale, body);
            writer.Write([0xFF, 0xDB, 0, 67]);
            writer.Write(body);
        }

        int frameLength = FrameBody(job, body);
        writer.Write([0xFF, 0xC0, (byte)((frameLength + 2) >> 8), (byte)(frameLength + 2)]);
        writer.Write(body[..frameLength]);
        for (int t = 0; t < tableCount; t++)
        {
            WriteTable(ref writer, dc[t], (byte)t);
            WriteTable(ref writer, ac[t], (byte)(0x10 | t));
        }

        int sosLength = 6 + 2 * count;
        writer.Write([0xFF, 0xDA, (byte)(sosLength >> 8), (byte)sosLength, (byte)count]);
        foreach (var c in job.Components)
        {
            writer.Write([(byte)c.Id, (byte)(c.Table << 4 | c.Table)]);
        }

        writer.Write([0, 63, 0]);
    }

    // The frame header after its length (SOF0's body, which the decoder's frame is also made from): 8-bit samples, the
    // height and width, then each component's id, sampling (H << 4 | V) and quantization table; its length.
    private static int FrameBody(Job job, Span<byte> body)
    {
        var components = job.Components;
        body[0] = 8;
        BinaryPrimitives.WriteUInt16BigEndian(body[1..], (ushort)job.Height);
        BinaryPrimitives.WriteUInt16BigEndian(body[3..], (ushort)job.Width);
        body[5] = (byte)components.Length;
        for (int i = 0; i < components.Length; i++)
        {
            var c = components[i];
            body[6 + 3 * i] = (byte)c.Id;
            body[7 + 3 * i] = (byte)(c.H << 4 | c.V);
            body[8 + 3 * i] = (byte)c.Table;
        }

        return 6 + 3 * components.Length;
    }

    // Quantization table `table` as a DQT segment holds it after its length: its id (8-bit values), then its 64 values in
    // zigzag order.
    private static void TableBody(int table, int scale, Span<byte> body)
    {
        body = body[..65];
        body[0] = (byte)table;
        for (int k = 0; k < 64; k++)
        {
            body[1 + k] = (byte)Quantized(table, Zigzag[k], scale);
        }
    }

    private static void WriteTable(ref BitWriter writer, HuffmanTable table, byte classAndId)
    {
        int length = 2 + 1 + 16 + table.Values.Length;
        writer.Write([0xFF, 0xC4, (byte)(length >> 8), (byte)length, classAndId]);
        writer.Write(table.Bits);
        writer.Write(table.Values);
    }

    // jchuff.c jpeg_gen_optimal_table: Huffman code lengths from the symbol counts (a reserved symbol 256 keeps any code
    // from being all ones), lengths over 16 folded as Annex K.3 does.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static HuffmanTable OptimalTable(ReadOnlySpan<long> counts)
    {
        const int MaxLength = 32;
        long[] freq = ArrayPool<long>.Shared.Rent(Symbols);
        int[] work = ArrayPool<int>.Shared.Rent(MaxLength + 1 + 2 * Symbols);
        try
        {
            counts[..Symbols].CopyTo(freq);
            var bits = work.AsSpan(0, MaxLength + 1);
            var codeSize = work.AsSpan(MaxLength + 1, Symbols);
            var others = work.AsSpan(MaxLength + 1 + Symbols, Symbols);
            bits.Clear();
            codeSize.Clear();
            others.Fill(-1);
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

            int lastLength = 16;
            while (bits[lastLength] == 0)
            {
                lastLength--;
            }

            bits[lastLength]--;                                                                  // drop the reserved symbol
            Span<byte> lengths = stackalloc byte[16];
            for (int length = 1; length <= 16; length++)
            {
                lengths[length - 1] = (byte)bits[length];
            }

            Span<byte> values = stackalloc byte[256];
            int n = 0;
            for (int length = 1; length <= MaxLength; length++)
            {
                for (int symbol = 0; symbol <= 255; symbol++)
                {
                    if (codeSize[symbol] == length)
                    {
                        values[n++] = (byte)symbol;
                    }
                }
            }

            return new HuffmanTable(lengths, values[..n]);
        }
        finally
        {
            ArrayPool<long>.Shared.Return(freq);
            ArrayPool<int>.Shared.Return(work);
        }
    }

    // jccoefct.c's quantization divisor of one coefficient: the reciprocal, its rounding correction and the shift.
    private readonly record struct Quantizer(int Reciprocal, int Correction, int Shift);

    // A component of the frame: its sampling, table, the blocks with real samples (width_in_blocks, height_in_blocks),
    // the blocks a row of MCUs holds across, and where its blocks start in an MCU row's coefficients.
    private readonly record struct Component(int Id, int H, int V, int Table, int BlocksAcross, int BlocksDown, int PaddedAcross, int Offset);

    // The image to encode: 8-bit planes read in place, or float pixels ([channel, row, column] in [0, 1]) brought to bytes.
    private readonly struct Source(ReadOnlyMemory<byte>[]? planes, float[]? pixels, int height, int width)
    {
        public int Height { get; } = height;

        public int Width { get; } = width;

        public bool Floats => planes is null;

        public ReadOnlySpan<byte> Plane(int channel) => planes![channel].Span;

        // A row of a channel's floats as bytes (ImagePreprocessor.ToBytes' rounding); compiled optimized at once, as a
        // process's first float pages otherwise ran its unoptimized loop (130 ms against 54 for a colour A4 page at 200 dpi).
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Convert(int channel, int row, Span<byte> destination)
        {
            var values = pixels.AsSpan((channel * Height + row) * Width, Width);
            destination = destination[..values.Length];
            for (int x = 0; x < values.Length; x++)
            {
                destination[x] = (byte)ImagePreprocessor.Byte(values[x]);
            }
        }
    }

    // One encode's frame (its components and MCUs), its quantizers, and the chunk of MCU rows being transformed.
    private sealed class Job
    {
        public Job(Source source, bool colour, int hs, int vs, Quantizer[] quantizers)
        {
            Source = source;
            Colour = colour;
            Quantizers = quantizers;
            Width = source.Width;
            Height = source.Height;
            MaxH = hs;
            MaxV = vs;
            McusAcross = DivUp(Width, 8 * hs);
            McusDown = DivUp(Height, 8 * vs);
            if (colour)
            {
                int chroma = vs * McusAcross * hs * 64;
                Components =
                [
                    new Component(1, hs, vs, 0, DivUp(Width, 8), DivUp(Height, 8), McusAcross * hs, 0),
                    new Component(2, 1, 1, 1, McusAcross, McusDown, McusAcross, chroma),
                    new Component(3, 1, 1, 1, McusAcross, McusDown, McusAcross, chroma + McusAcross * 64),
                ];
                RowStride = chroma + 2 * McusAcross * 64;
                StripeWidth = McusAcross * 8 * hs;
                ScratchLength = 3 * 8 * vs * StripeWidth + (source.Floats ? 3 * Width : 0);
            }
            else
            {
                Components = [new Component(1, 1, 1, 0, McusAcross, McusDown, McusAcross, 0)];
                RowStride = McusAcross * 64;
                ScratchLength = source.Floats ? 8 * Width : 0;
            }
        }

        public Source Source { get; }

        public bool Colour { get; }

        public Quantizer[] Quantizers { get; }

        public int Width { get; }

        public int Height { get; }

        public int MaxH { get; }

        public int MaxV { get; }

        public int McusAcross { get; }

        public int McusDown { get; }

        public Component[] Components { get; }

        // Coefficients of one MCU row (every component's blocks, zigzag order); a colour stripe's width; a worker's scratch.
        public int RowStride { get; }

        public int StripeWidth { get; }

        public int ScratchLength { get; }

        // The chunk being transformed: its buffer (RowStride a row) and its first MCU row.
        public short[] Buffer { get; set; } = [];

        public int ChunkStart { get; set; }

        // A round trip's destination instead of the chunk: each component's coefficient array (the decoder's: its blocks
        // row by row, PaddedAcross a block row, each block in natural order); null when the blocks are entropy coded.
        public short[][]? Targets { get; set; }

        // Component ci's blocks of MCU row m (its V block rows of PaddedAcross blocks), in its array for a round trip, else
        // in the chunk's buffer.
        public Span<short> Blocks(int ci, int m)
        {
            var c = Components[ci];
            int length = (c.V * c.PaddedAcross) << 6;
            return Targets is { } targets
                ? targets[ci].AsSpan(m * length, length)
                : Buffer.AsSpan((m - ChunkStart) * RowStride + c.Offset, length);
        }
    }

    private sealed class HuffmanTable
    {
        public HuffmanTable(ReadOnlySpan<byte> bits, ReadOnlySpan<byte> values)
        {
            Bits = bits.ToArray();
            Values = values.ToArray();
            int code = 0, k = 0;
            for (int length = 1; length <= 16; length++)
            {
                for (int i = 0; i < bits[length - 1]; i++)
                {
                    Codes[values[k]] = code++;
                    Lengths[values[k]] = (byte)length;
                    k++;
                }

                code <<= 1;
            }
        }

        public byte[] Bits { get; }

        public byte[] Values { get; }

        public int[] Codes { get; } = new int[256];

        public byte[] Lengths { get; } = new byte[256];
    }

    // The file being written, in a pooled buffer that grows by doubling: header bytes as they are, the scan's bits
    // gathered in a 64-bit word and written 32 at a time, a 0 stuffed after every 0xFF the scan writes.
    private struct BitWriter(byte[] buffer)
    {
        private ulong _bits;
        private int _count;

        public byte[] Buffer { get; private set; } = buffer;

        public int Position { get; private set; }

        // The low `length` bits of value (at most 32), first bit first.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Put(uint value, int length)
        {
            _bits = (_bits << length) | value;
            _count += length;
            if (_count >= 32)
            {
                _count -= 32;
                Emit((uint)(_bits >> _count));
            }
        }

        // Pads the last byte with ones and writes what is left.
        public void Flush()
        {
            int pad = (8 - (_count & 7)) & 7;
            _bits = (_bits << pad) | ((1UL << pad) - 1);
            _count += pad;
            while (_count > 0)
            {
                _count -= 8;
                Stuffed((byte)(_bits >> _count));
            }
        }

        public void Write(ReadOnlySpan<byte> bytes)
        {
            Reserve(bytes.Length);
            bytes.CopyTo(Buffer.AsSpan(Position));
            Position += bytes.Length;
        }

        public readonly byte[] ToFile()
        {
            var file = GC.AllocateUninitializedArray<byte>(Position);
            Buffer.AsSpan(0, Position).CopyTo(file);
            return file;
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private void Emit(uint word)
        {
            Reserve(8);
            uint inverted = ~word;
            if (((inverted - 0x01010101u) & ~inverted & 0x80808080u) == 0)
            {
                BinaryPrimitives.WriteUInt32BigEndian(Buffer.AsSpan(Position), word);                // no 0xFF byte
                Position += 4;
                return;
            }

            for (int shift = 24; shift >= 0; shift -= 8)
            {
                Stuffed((byte)(word >> shift));
            }
        }

        private void Stuffed(byte value)
        {
            Reserve(2);
            Buffer[Position++] = value;
            if (value == 0xFF)
            {
                Buffer[Position++] = 0;
            }
        }

        private void Reserve(int bytes)
        {
            if (Position + bytes <= Buffer.Length)
            {
                return;
            }

            var larger = ArrayPool<byte>.Shared.Rent(Math.Max(Position + bytes, 2 * Buffer.Length));
            Buffer.AsSpan(0, Position).CopyTo(larger);
            ArrayPool<byte>.Shared.Return(Buffer);
            Buffer = larger;
        }
    }
}
