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
/// or <see cref="DocumentImageEncoding.Jpeg"/> (<c>DCTDecode</c>, written by <see cref="ImageEncoders"/>' "jpeg" at
/// <see cref="DocumentWriteOptions.JpegQuality"/>); <see cref="DocumentImageEncoding.Auto"/> measures the image
/// (<see cref="DocumentWriteOptions.LineArtLevels"/>, then <see cref="DocumentWriteOptions.SmoothSteps"/>). The image's
/// buffers are rented from the pool and returned before the call ends; rows are worked in parallel, each one's result
/// its own, so any number of threads gives the same bytes.
/// </summary>
internal static class PdfWriterImages
{
    /// <summary>The name of the image encoder JPEG images are written with.</summary>
    public const string JpegEncoder = "jpeg";

    // The predictor is kept when its rows' residuals sum to less than this share of the plain rows' (a row's best of five
    // filters is a little smaller than its plain sum even on noise, by chance: a fifth smaller is a predictor that pays).
    private const double PredictorPays = 0.8;

    /// <summary>
    /// Writes <paramref name="image"/> as image XObject <paramref name="number"/>; <paramref name="requested"/>
    /// <see cref="DocumentImageEncoding.Auto"/> takes the options' encoding (measured when that is Auto too). The encoding written.
    /// </summary>
    /// <exception cref="ArgumentException">The image is neither grey nor RGB.</exception>
    /// <exception cref="NotRegisteredException">JPEG was asked and no image encoder "jpeg" is registered.</exception>
    public static DocumentImageEncoding Write(PdfFileWriter file, int number, ImageData image, DocumentImageEncoding requested, DocumentWriteOptions options)
    {
        if (image.Channels is not (1 or 3))
        {
            throw new ArgumentException($"A document's image is grey (1 channel) or RGB (3), not {image.Channels} channels.", nameof(image));
        }

        var encoding = requested == DocumentImageEncoding.Auto ? options.ImageEncoding : requested;
        int width = image.Width, height = image.Height, channels = image.Channels;
        if (encoding == DocumentImageEncoding.Jpeg)
        {
            WriteJpeg(file, number, image, options, ImageEncoders.Get(JpegEncoder));
            return encoding;
        }

        int length = width * height * channels;
        byte[] levels = ArrayPool<byte>.Shared.Rent(length);
        try
        {
            Levels(image, levels);
            if (encoding == DocumentImageEncoding.Auto)
            {
                var measures = Measure(levels, width, height, channels, options.LineArtLevels);
                bool photographic = measures.Levels > options.LineArtLevels && measures.SmallSteps >= options.SmoothSteps;
                if (photographic && ImageEncoders.Names.Contains(JpegEncoder, StringComparer.OrdinalIgnoreCase))
                {
                    WriteJpeg(file, number, image, options, ImageEncoders.Get(JpegEncoder));
                    return DocumentImageEncoding.Jpeg;
                }
            }

            WriteFlate(file, number, levels, width, height, channels);
            return DocumentImageEncoding.Flate;
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(levels);
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
    public static void Levels(ImageData image, byte[] destination)
    {
        int channels = image.Channels, plane = image.Width * image.Height;
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
                planes[0].Span.CopyTo(destination);
                return;
            }

            HostParallel.For(plane, 1 << 15, (first, last) =>
            {
                for (int c = 0; c < channels; c++)
                {
                    var source = planes[c].Span;
                    for (int i = first; i < last; i++)
                    {
                        destination[i * channels + c] = map[source[i]];
                    }
                }
            });
            return;
        }

        var pixels = image.Pixels;
        HostParallel.For(plane, 1 << 15, (first, last) =>
        {
            for (int c = 0; c < channels; c++)
            {
                int offset = c * plane;
                for (int i = first; i < last; i++)
                {
                    destination[i * channels + c] = Level(pixels[offset + i]);
                }
            }
        });
    }

    // A value in [0, 1] as an 8-bit level, rounded (the image encoders' rule).
    private static byte Level(float value) => (byte)Math.Clamp((int)MathF.Round(value * 255f), 0, 255);

    private static void WriteJpeg(PdfFileWriter file, int number, ImageData image, DocumentWriteOptions options, IImageEncoder encoder)
    {
        byte[] jpeg = encoder.Encode(image, new ImageEncodeOptions { Quality = options.JpegQuality });
        Header(file, number, image.Width, image.Height, image.Channels).Raw("/Filter/DCTDecode"u8);
        file.EndStream(jpeg, deflate: false);
    }

    private static PdfSyntax Header(PdfFileWriter file, int number, int width, int height, int channels) =>
        file.BeginStream(number).Raw("/Type/XObject/Subtype/Image/Width "u8).Integer(width).Raw("/Height "u8).Integer(height)
            .Raw(channels == 1 ? "/ColorSpace/DeviceGray"u8 : "/ColorSpace/DeviceRGB"u8).Raw("/BitsPerComponent 8"u8);

    private static void WriteFlate(PdfFileWriter file, int number, byte[] levels, int width, int height, int channels)
    {
        int row = width * channels, length = row * height;
        if (file.Level == 0)
        {
            Header(file, number, width, height, channels);
            file.EndStream(levels.AsSpan(0, length), deflate: false);           // stored: no filter, no predictor
            return;
        }

        int filtered = (row + 1) * height;
        byte[] rows = ArrayPool<byte>.Shared.Rent(filtered);
        try
        {
            if (Predict(levels, rows, row, height, channels))
            {
                Header(file, number, width, height, channels).Raw("/DecodeParms<</Predictor 15/Colors "u8).Integer(channels)
                    .Raw("/BitsPerComponent 8/Columns "u8).Integer(width).Raw(">>"u8);
                file.EndStream(rows.AsSpan(0, filtered), deflate: true);
                return;
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rows);
        }

        Header(file, number, width, height, channels);
        file.EndStream(levels.AsSpan(0, length), deflate: true);
    }

    /// <summary>
    /// The PNG predictor's rows (a filter byte, then the row filtered) of <paramref name="pixels"/> into
    /// <paramref name="output"/>, each row's filter the one whose residuals, read as signed bytes, sum smallest (None,
    /// Sub, Up, Average, Paeth); whether it pays: the chosen rows' sums under <see cref="PredictorPays"/> of the plain rows'.
    /// </summary>
    public static bool Predict(byte[] pixels, byte[] output, int row, int height, int bpp)
    {
        long plainTotal = 0, chosenTotal = 0;
        var gate = new Lock();
        HostParallel.For(height, Math.Max(1, (1 << 16) / Math.Max(1, row)), (first, last) =>
        {
            long plain = 0, chosen = 0;
            for (int y = first; y < last; y++)
            {
                var here = pixels.AsSpan(y * row, row);
                var above = y > 0 ? pixels.AsSpan((y - 1) * row, row) : default;
                var target = output.AsSpan(y * (row + 1), row + 1);
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

                target[0] = (byte)filter;
                Residuals(here, above, bpp, filter, target[1..]);
                plain += none;
                chosen += best;
            }

            lock (gate)
            {
                plainTotal += plain;
                chosenTotal += chosen;
            }
        });

        return chosenTotal < PredictorPays * plainTotal;
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
