// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using Idrak.Abstraction.Devices;

namespace Idrak.Data.Pdf;

/// <summary>What <see cref="DocumentImageEncoding.Auto"/> measures of an image.</summary>
/// <param name="Levels">The most distinct 8-bit levels any channel uses.</param>
/// <param name="SmallSteps">The share of neighbouring pixel pairs (each pixel against its right and its lower neighbour, the largest channel difference) differing by 1 to 16 levels; NaN when not measured.</param>
/// <param name="LargeSteps">The share differing by more than 64 levels; NaN when not measured.</param>
internal readonly record struct PdfImageMeasures(int Levels, double SmallSteps, double LargeSteps);

/// <summary>
/// An image page's image as an image XObject: its values in [0, 1] as 8-bit levels (rounded), grey or RGB, stored
/// <see cref="DocumentImageEncoding.Flate"/> (exact: the PNG predictor, each row's filter chosen by the smallest sum of
/// its residuals as libpng chooses, kept where it lowers the rows' residuals by a fifth, then Flate at the options' level)
/// or <see cref="DocumentImageEncoding.Jpeg"/> (<c>DCTDecode</c>, written by <c>ImageEncoders</c>' "jpeg" at
/// <see cref="DocumentWriteOptions.JpegQuality"/>); <see cref="DocumentImageEncoding.Auto"/> measures the image
/// (<see cref="DocumentWriteOptions.LineArtLevels"/>, then <see cref="DocumentWriteOptions.SmoothSteps"/>). The image is
/// measured a band of rows at a time (its levels made for the band, never copied whole), and a Flate image holds one copy
/// besides itself: the stream the deflater takes in one piece (filtered band by band); every buffer is rented from the
/// pool and returned before the call ends; rows are worked in parallel, each one's result its own, so any number of
/// threads gives the same bytes.
/// </summary>
internal static class PdfWriterImages
{
    /// <summary>The name of the image encoder JPEG images are written with.</summary>
    public const string JpegEncoder = "jpeg";

    // The image encoders were registered (ImageEncoders) and looked up by name: not on the new design yet. The one JPEG
    // images are written with, "jpeg" (Idrak.Imaging's JpegImageEncoder), is reached directly; another name stops.
    private static readonly Idrak.Imaging.JpegImageEncoder Jpeg = new();

    private static IImageEncoder EncoderNamed(string name) =>
        name.Equals(JpegEncoder, StringComparison.OrdinalIgnoreCase) ? Jpeg : throw new NotSupportedException($"The image encoder '{name}' by name is not on the new design yet.");

    // The predictor is kept when its rows' residuals sum to less than this share of the plain rows' (a row's best of five
    // filters is a little smaller than its plain sum even on noise, by chance: a fifth smaller is a predictor that pays).
    private const double PredictorPays = 0.8;

    /// <summary>
    /// Writes <paramref name="image"/> as image XObject <paramref name="number"/>; <paramref name="requested"/>
    /// <see cref="DocumentImageEncoding.Auto"/> takes the options' encoding (measured when that is Auto too); a
    /// <paramref name="softMask"/> above 0 names the grey image object it shows through. The encoding written.
    /// </summary>
    /// <exception cref="ArgumentException">The image is neither grey nor RGB.</exception>
    public static DocumentImageEncoding Write(PdfFileWriter file, int number, ImageData image, DocumentImageEncoding requested, DocumentWriteOptions options, int softMask = 0)
    {
        if (image.Channels is not (1 or 3))
        {
            throw new ArgumentException($"A document's image is grey (1 channel) or RGB (3), not {image.Channels} channels.", nameof(image));
        }

        var encoding = requested == DocumentImageEncoding.Auto ? options.ImageEncoding : requested;
        int width = image.Width, height = image.Height, channels = image.Channels;
        if (encoding == DocumentImageEncoding.Jpeg)
        {
            WriteJpeg(file, number, image, options, EncoderNamed(JpegEncoder), softMask);
            return encoding;
        }

        if (encoding == DocumentImageEncoding.Auto)
        {
            var measures = Measure(image, options.LineArtLevels);
            bool photographic = measures.Levels > options.LineArtLevels && measures.SmallSteps >= options.SmoothSteps;
            if (photographic)
            {
                WriteJpeg(file, number, image, options, EncoderNamed(JpegEncoder), softMask);
                return DocumentImageEncoding.Jpeg;
            }
        }

        WriteFlate(file, number, image, softMask);
        return DocumentImageEncoding.Flate;
    }

    /// <summary>
    /// Writes <paramref name="image"/> (grey: dark where it paints) as image mask XObject <paramref name="number"/>: one bit
    /// a pixel, rows padded to whole bytes, 0 where it paints (a level below a half), deflated.
    /// </summary>
    public static void WriteStencil(PdfFileWriter file, int number, ImageData image)
    {
        int width = image.Width, height = image.Height, row = (width + 7) / 8;
        var values = image.Pixels;
        byte[] bits = ArrayPool<byte>.Shared.Rent(checked(row * height));
        try
        {
            bits.AsSpan(0, row * height).Clear();
            for (int y = 0; y < height; y++)
            {
                var line = bits.AsSpan(y * row, row);
                for (int x = 0; x < width; x++)
                {
                    if (values[y * width + x] >= 0.5f)
                    {
                        line[x >> 3] |= (byte)(0x80 >> (x & 7));
                    }
                }
            }

            file.BeginStream(number).Raw("/Type/XObject/Subtype/Image/Width "u8).Integer(width).Raw("/Height "u8).Integer(height).Raw("/ImageMask true/BitsPerComponent 1"u8);
            file.EndStream(bits.AsSpan(0, row * height), deflate: true);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(bits);
        }
    }

    // The rows an image is worked in at a time: four parts a thread of the 64 KiB grains rows are spread in (at least a row).
    private static int BandRows(int row) => Math.Max(1, 4 * ComputeResources.MaxCpuThreads * Math.Max(1, (1 << 16) / Math.Max(1, row)));

    /// <summary>
    /// <see cref="Measure(byte[], int, int, int, int)"/> of <paramref name="image"/>'s levels, made a band of rows at a time
    /// (the image is not copied whole): the same measures.
    /// </summary>
    public static PdfImageMeasures Measure(ImageData image, int lineArtLevels)
    {
        int width = image.Width, height = image.Height, channels = image.Channels, row = width * channels, band = BandRows(row);
        byte[] rows = ArrayPool<byte>.Shared.Rent(checked((band + 1) * row));
        try
        {
            Span<bool> seen = stackalloc bool[3 * 256];
            seen.Clear();
            for (int y0 = 0; y0 < height; y0 += band)
            {
                int n = Math.Min(band, height - y0), length = n * row;
                Levels(image, y0, n, rows, 0);
                for (int c = 0; c < channels; c++)
                {
                    var used = seen.Slice(c * 256, 256);
                    for (int i = c; i < length; i += channels)
                    {
                        used[rows[i]] = true;
                    }
                }
            }

            int levels = 0;
            for (int c = 0; c < channels; c++)
            {
                levels = Math.Max(levels, seen.Slice(c * 256, 256).Count(true));
            }

            if (levels <= lineArtLevels)
            {
                return new PdfImageMeasures(levels, double.NaN, double.NaN);
            }

            // The steps: each band with the row below it, each pixel against its right and its lower neighbour.
            long small = 0, large = 0;
            var gate = new Lock();
            for (int y0 = 0; y0 < height; y0 += band)
            {
                int n = Math.Min(band, height - y0), withBelow = Math.Min(n + 1, height - y0);
                Levels(image, y0, withBelow, rows, 0);
                HostParallel.For(n, Math.Max(1, (1 << 16) / Math.Max(1, row)), (first, last) =>
                {
                    long s = 0, l = 0;
                    for (int y = first; y < last; y++)
                    {
                        var here = rows.AsSpan(y * row, row);
                        var below = y + 1 < withBelow ? rows.AsSpan((y + 1) * row, row) : default;
                        for (int x = 0; x < width; x++)
                        {
                            int at = x * channels;
                            if (x + 1 < width)
                            {
                                Count(Step(here, at, here, at + channels, channels), ref s, ref l);
                            }

                            if (!below.IsEmpty)
                            {
                                Count(Step(here, at, below, at, channels), ref s, ref l);
                            }
                        }
                    }

                    lock (gate)
                    {
                        small += s;
                        large += l;
                    }
                });
            }

            double pairs = (double)(width - 1) * height + (double)width * (height - 1);
            return pairs > 0 ? new PdfImageMeasures(levels, small / pairs, large / pairs) : new PdfImageMeasures(levels, 0, 0);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rows);
        }
    }

    /// <summary>
    /// The measures of 8-bit <paramref name="pixels"/> (rows of interleaved channels, <paramref name="width"/> x
    /// <paramref name="height"/>): the most levels a channel uses and, when that passes <paramref name="lineArtLevels"/>
    /// (else NaN), the shares of small and large steps between neighbours.
    /// </summary>
    public static PdfImageMeasures Measure(byte[] pixels, int width, int height, int channels, int lineArtLevels)
    {
        int length = width * height * channels;
        Span<bool> seen = stackalloc bool[3 * 256];
        seen.Clear();
        int levels = 0;
        for (int c = 0; c < channels; c++)
        {
            var used = seen.Slice(c * 256, 256);
            for (int i = c; i < length; i += channels)
            {
                used[pixels[i]] = true;
            }

            levels = Math.Max(levels, used.Count(true));
        }

        if (levels <= lineArtLevels)
        {
            return new PdfImageMeasures(levels, double.NaN, double.NaN);
        }

        long small = 0, large = 0;
        var gate = new Lock();
        int row = width * channels;
        HostParallel.For(height, Math.Max(1, (1 << 16) / Math.Max(1, row)), (first, last) =>
        {
            long s = 0, l = 0;
            for (int y = first; y < last; y++)
            {
                var here = pixels.AsSpan(y * row, row);
                var below = y + 1 < height ? pixels.AsSpan((y + 1) * row, row) : default;
                for (int x = 0; x < width; x++)
                {
                    int at = x * channels;
                    if (x + 1 < width)
                    {
                        Count(Step(here, at, here, at + channels, channels), ref s, ref l);
                    }

                    if (!below.IsEmpty)
                    {
                        Count(Step(here, at, below, at, channels), ref s, ref l);
                    }
                }
            }

            lock (gate)
            {
                small += s;
                large += l;
            }
        });

        double pairs = (double)(width - 1) * height + (double)width * (height - 1);
        return pairs > 0 ? new PdfImageMeasures(levels, small / pairs, large / pairs) : new PdfImageMeasures(levels, 0, 0);
    }

    // The largest channel difference between two pixels.
    private static int Step(ReadOnlySpan<byte> a, int i, ReadOnlySpan<byte> b, int j, int channels)
    {
        int d = Math.Abs(a[i] - b[j]);
        for (int c = 1; c < channels; c++)
        {
            d = Math.Max(d, Math.Abs(a[i + c] - b[j + c]));
        }

        return d;
    }

    private static void Count(int step, ref long small, ref long large)
    {
        if (step is >= 1 and <= 16)
        {
            small++;
        }
        else if (step > 64)
        {
            large++;
        }
    }

    /// <summary>The image's levels into <paramref name="destination"/>, rows of interleaved channels: each value rounded to a byte, or an 8-bit image's own bytes.</summary>
    public static void Levels(ImageData image, byte[] destination) => Levels(image, 0, image.Height, destination, 0);

    /// <summary>
    /// Rows [<paramref name="firstRow"/>, + <paramref name="rows"/>) of the image's levels into <paramref name="destination"/>
    /// from <paramref name="at"/>, as <see cref="Levels(ImageData, byte[])"/> makes them.
    /// </summary>
    public static void Levels(ImageData image, int firstRow, int rows, byte[] destination, int at)
    {
        int channels = image.Channels, plane = image.Width * image.Height, start = firstRow * image.Width, count = rows * image.Width;
        if (image.TryGetPlanes(out var planes, out var table))
        {
            var map = new byte[256];
            bool identity = true;
            var values = table.Span;
            for (int k = 0; k < 256; k++)
            {
                map[k] = Level(values[k]);
                identity &= map[k] == k;
            }

            if (channels == 1 && identity)
            {
                planes[0].Span.Slice(start, count).CopyTo(destination.AsSpan(at));
                return;
            }

            HostParallel.For(count, 1 << 15, (first, last) =>
            {
                for (int c = 0; c < channels; c++)
                {
                    var source = planes[c].Span;
                    for (int i = first; i < last; i++)
                    {
                        destination[at + i * channels + c] = map[source[start + i]];
                    }
                }
            });
            return;
        }

        var pixels = image.Pixels;
        HostParallel.For(count, 1 << 15, (first, last) =>
        {
            for (int c = 0; c < channels; c++)
            {
                int offset = c * plane + start;
                for (int i = first; i < last; i++)
                {
                    destination[at + i * channels + c] = Level(pixels[offset + i]);
                }
            }
        });
    }

    // A value in [0, 1] as an 8-bit level, rounded (the image encoders' rule).
    private static byte Level(float value) => (byte)Math.Clamp((int)MathF.Round(value * 255f), 0, 255);

    private static void WriteJpeg(PdfFileWriter file, int number, ImageData image, DocumentWriteOptions options, IImageEncoder encoder, int softMask)
    {
        byte[] jpeg = encoder.Encode(image, new ImageEncodeOptions { Quality = options.JpegQuality });
        Header(file, number, image.Width, image.Height, image.Channels, softMask).Raw("/Filter/DCTDecode"u8);
        file.EndStream(jpeg, deflate: false);
    }

    // The image's dictionary as far as its filters: its size, colour space, bits and (above 0) the soft mask it shows through.
    private static PdfSyntax Header(PdfFileWriter file, int number, int width, int height, int channels, int softMask)
    {
        var s = file.BeginStream(number).Raw("/Type/XObject/Subtype/Image/Width "u8).Integer(width).Raw("/Height "u8).Integer(height)
            .Raw(channels == 1 ? "/ColorSpace/DeviceGray"u8 : "/ColorSpace/DeviceRGB"u8).Raw("/BitsPerComponent 8"u8);
        return softMask > 0 ? s.Raw("/SMask "u8).Reference(softMask) : s;
    }

    private static void WriteFlate(PdfFileWriter file, int number, ImageData image, int softMask)
    {
        int width = image.Width, height = image.Height, channels = image.Channels, row = width * channels;
        if (file.Level == 0)
        {
            // Stored: no filter, no predictor; the file takes a stored stream whole.
            byte[] levels = ArrayPool<byte>.Shared.Rent(checked(row * height));
            try
            {
                Levels(image, levels);
                Header(file, number, width, height, channels, softMask);
                file.EndStream(levels.AsSpan(0, row * height), deflate: false);
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(levels);
            }

            return;
        }

        // First each row's filter (the PNG predictor's five, the one whose residuals sum smallest, as libpng chooses) and
        // whether the predictor pays (the chosen rows' sums under PredictorPays of the plain rows'), the levels made a band
        // of rows at a time; then the stream the deflater takes in one piece, as it always has (its bytes depend on the
        // pieces it is given): the filtered rows (a filter byte, then the row), made band by band, or the plain levels.
        int band = BandRows(row);
        byte[] rows = ArrayPool<byte>.Shared.Rent(checked((band + 1) * row));
        byte[] filters = ArrayPool<byte>.Shared.Rent(height);
        byte[]? stream = null;
        try
        {
            long plainTotal = 0, chosenTotal = 0;
            var gate = new Lock();
            for (int y0 = 0; y0 < height; y0 += band)
            {
                int n = BandWithAbove(image, y0, band, rows);
                HostParallel.For(n, Math.Max(1, (1 << 16) / Math.Max(1, row)), (first, last) =>
                {
                    long plain = 0, chosen = 0;
                    for (int k = first; k < last; k++)
                    {
                        var here = rows.AsSpan((k + 1) * row, row);
                        var above = y0 + k > 0 ? rows.AsSpan(k * row, row) : default;
                        var (none, best, filter) = Choose(here, above, channels);
                        filters[y0 + k] = (byte)filter;
                        plain += none;
                        chosen += best;
                    }

                    lock (gate)
                    {
                        plainTotal += plain;
                        chosenTotal += chosen;
                    }
                });
            }

            if (chosenTotal >= PredictorPays * plainTotal)
            {
                stream = ArrayPool<byte>.Shared.Rent(checked(row * height));
                Levels(image, stream);
                Header(file, number, width, height, channels, softMask);
                file.EndStream(stream.AsSpan(0, row * height), deflate: true);
                return;
            }

            int filtered = checked((row + 1) * height);
            byte[] filteredRows = stream = ArrayPool<byte>.Shared.Rent(filtered);
            for (int y0 = 0; y0 < height; y0 += band)
            {
                int n = BandWithAbove(image, y0, band, rows);
                HostParallel.For(n, Math.Max(1, (1 << 16) / Math.Max(1, row)), (first, last) =>
                {
                    for (int k = first; k < last; k++)
                    {
                        var here = rows.AsSpan((k + 1) * row, row);
                        var above = y0 + k > 0 ? rows.AsSpan(k * row, row) : default;
                        var target = filteredRows.AsSpan((y0 + k) * (row + 1), row + 1);
                        target[0] = filters[y0 + k];
                        Residuals(here, above, channels, filters[y0 + k], target[1..]);
                    }
                });
            }

            Header(file, number, width, height, channels, softMask).Raw("/DecodeParms<</Predictor 15/Colors "u8).Integer(channels)
                .Raw("/BitsPerComponent 8/Columns "u8).Integer(width).Raw(">>"u8);
            file.EndStream(filteredRows.AsSpan(0, filtered), deflate: true);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rows);
            ArrayPool<byte>.Shared.Return(filters);
            if (stream is not null)
            {
                ArrayPool<byte>.Shared.Return(stream);
            }
        }
    }

    // Rows [y0, y0 + n) of the image's levels into `rows` from its second row, the row above them (when there is one) in
    // its first; n, the band's rows.
    private static int BandWithAbove(ImageData image, int y0, int band, byte[] rows)
    {
        int n = Math.Min(band, image.Height - y0);
        if (y0 > 0)
        {
            Levels(image, y0 - 1, n + 1, rows, 0);
        }
        else
        {
            Levels(image, 0, n, rows, image.Width * image.Channels);
        }

        return n;
    }

    // A row's filter: the one of five whose residuals, read as signed bytes, sum smallest (the first
    // of equal sums); with the plain row's sum and the chosen one's.
    private static (long None, long Best, int Filter) Choose(ReadOnlySpan<byte> here, ReadOnlySpan<byte> above, int bpp)
    {
        long none = 0, best = long.MaxValue;
        int filter = 0;
        for (int f = 0; f < 5; f++)
        {
            long sum = Residuals(here, above, bpp, f, default);
            if (f == 0)
            {
                none = sum;
            }

            if (sum < best)
            {
                (best, filter) = (sum, f);
            }
        }

        return (none, best, filter);
    }

    // A row through PNG filter `filter` (0 None, 1 Sub, 2 Up, 3 Average, 4 Paeth): the residuals written to `output` when
    // it is not empty; their sum as signed bytes (libpng's measure of how well a filter packs).
    private static long Residuals(ReadOnlySpan<byte> row, ReadOnlySpan<byte> above, int bpp, int filter, Span<byte> output)
    {
        long sum = 0;
        bool write = !output.IsEmpty, hasAbove = !above.IsEmpty;
        for (int i = 0; i < row.Length; i++)
        {
            int a = i >= bpp ? row[i - bpp] : 0, b = hasAbove ? above[i] : 0, c = i >= bpp && hasAbove ? above[i - bpp] : 0;
            int predicted = filter switch
            {
                0 => 0,
                1 => a,
                2 => b,
                3 => (a + b) >> 1,
                _ => Paeth(a, b, c),
            };

            byte residual = (byte)(row[i] - predicted);
            if (write)
            {
                output[i] = residual;
            }

            int signed = (sbyte)residual;                                      // -128 to 127: its size as an int (Math.Abs of sbyte -128 overflows)
            sum += signed < 0 ? -signed : signed;
        }

        return sum;
    }

    private static int Paeth(int a, int b, int c)
    {
        int p = a + b - c, pa = Math.Abs(p - a), pb = Math.Abs(p - b), pc = Math.Abs(p - c);
        return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
    }
}
