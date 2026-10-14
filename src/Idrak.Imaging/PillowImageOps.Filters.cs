// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;

namespace Idrak.Imaging;

// Pillow's filters, scaling, padding and point tables, byte for byte (Pillow 12.3's C in libImaging: BoxBlur.c,
// UnsharpMask.c, RankFilter.c, Filter.c's border; ImageOps.equalize and expand in Python). What helps a model read a
// poor scan: larger text, specks removed, thin strokes thickened, faint ink darkened, a clean black and white page.
public static partial class PillowImageOps
{
    /// <summary>
    /// The image resized by <paramref name="factor"/> (<c>image.resize((int(w * f + 0.5), int(h * f + 0.5)))</c>, each side at
    /// least 1): above 1 enlarges small text before the model's own resize.
    /// </summary>
    public static ImageData Scale(ImageData image, double factor, ImageResampling resampling = ImageResampling.Lanczos)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(factor);
        return Resize(image, Math.Max(1, (int)(image.Width * factor + 0.5)), Math.Max(1, (int)(image.Height * factor + 0.5)), resampling);
    }

    /// <summary><c>ImageOps.expand(image, border, fill)</c>: a margin of <paramref name="border"/> pixels of grey level <paramref name="fill"/> on every side (every channel).</summary>
    public static ImageData Pad(ImageData image, int border, byte fill = 255)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentOutOfRangeException.ThrowIfNegative(border);
        if (border == 0)
        {
            return image;
        }

        var planes = Bytes(image);
        int h = image.Height, w = image.Width, oh = h + 2 * border, ow = w + 2 * border;
        var result = new byte[planes.Length][];
        for (int c = 0; c < planes.Length; c++)
        {
            var output = new byte[oh * ow];
            output.AsSpan().Fill(fill);
            for (int y = 0; y < h; y++)
            {
                planes[c].AsSpan(y * w, w).CopyTo(output.AsSpan((y + border) * ow + border, w));
            }

            result[c] = output;
        }

        return FromBytes(result, oh, ow);
    }

    /// <summary>
    /// <c>image.filter(ImageFilter.RankFilter(size, rank))</c> on every channel: each pixel becomes the
    /// <paramref name="rank"/>-th smallest of its <paramref name="size"/> x <paramref name="size"/> neighbourhood, the
    /// border's pixels repeated outwards. <see cref="Median"/>, <see cref="MinFilter"/> and <see cref="MaxFilter"/> are its
    /// usual ranks.
    /// </summary>
    public static ImageData RankFilter(ImageData image, int size, int rank)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (size < 1 || (size & 1) == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(size), size, "A rank filter's size is odd and positive.");
        }

        ArgumentOutOfRangeException.ThrowIfNegative(rank);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(rank, size * size);
        if (size == 1)
        {
            return image;
        }

        var planes = Bytes(image);
        int h = image.Height, w = image.Width, margin = size / 2;
        var result = new byte[planes.Length][];
        for (int c = 0; c < planes.Length; c++)
        {
            var plane = planes[c];
            if (size == 3)
            {
                result[c] = RankOfNine(plane, h, w, rank);
                continue;
            }

            var output = new byte[h * w];
            // Rows are independent: spread over the cores. A window's values are counted (256 bins), then the rank found.
            Parallel.For(0, h, y =>
            {
                Span<int> counts = stackalloc int[256];
                for (int x = 0; x < w; x++)
                {
                    counts.Clear();
                    for (int dy = -margin; dy <= margin; dy++)
                    {
                        int row = Math.Clamp(y + dy, 0, h - 1) * w;
                        for (int dx = -margin; dx <= margin; dx++)
                        {
                            counts[plane[row + Math.Clamp(x + dx, 0, w - 1)]]++;
                        }
                    }

                    int seen = 0, value = 0;
                    while ((seen += counts[value]) <= rank)
                    {
                        value++;
                    }

                    output[y * w + x] = (byte)value;
                }
            });
            result[c] = output;
        }

        return FromBytes(result, h, w);
    }

    // Knuth's 25 compare-exchanges that sort nine values (checked on every 0-1 input, so on every input): a 3 x 3 window
    // sorted without branches, its rank then read. The rank of a window is a value, whatever the order, so this equals
    // counting.
    private static ReadOnlySpan<byte> NineSort =>
    [
        0, 1, 3, 4, 6, 7, 1, 2, 4, 5, 7, 8, 0, 1, 3, 4, 6, 7, 0, 3, 3, 6, 0, 3, 1, 4,
        4, 7, 1, 4, 2, 5, 5, 8, 2, 5, 1, 3, 5, 7, 2, 6, 4, 6, 2, 4, 2, 3, 5, 6,
    ];

    // RankFilter of size 3: rows in parallel; inside a row, the windows of a vector of pixels at once (the nine shifted rows'
    // vectors through the network), the first and last columns (the border repeated) and the tail one at a time.
    private static byte[] RankOfNine(byte[] plane, int h, int w, int rank)
    {
        var output = GC.AllocateUninitializedArray<byte>(h * w);
        HostParallel.For(h, Math.Max(1, (1 << 15) / w), (first, last) =>
        {
            Span<Vector<byte>> lanes = stackalloc Vector<byte>[9];
            Span<byte> nine = stackalloc byte[9];
            int n = Vector<byte>.Count;
            for (int y = first; y < last; y++)
            {
                int up = Math.Max(y - 1, 0) * w, mid = y * w, down = Math.Min(y + 1, h - 1) * w;
                RankOne(plane, output, nine, up, mid, down, w, 0, rank);
                int x = 1;
                if (Vector.IsHardwareAccelerated)
                {
                    for (; x + n < w; x += n)                                   // reads columns x - 1 to x + n
                    {
                        lanes[0] = new Vector<byte>(plane.AsSpan(up + x - 1));
                        lanes[1] = new Vector<byte>(plane.AsSpan(up + x));
                        lanes[2] = new Vector<byte>(plane.AsSpan(up + x + 1));
                        lanes[3] = new Vector<byte>(plane.AsSpan(mid + x - 1));
                        lanes[4] = new Vector<byte>(plane.AsSpan(mid + x));
                        lanes[5] = new Vector<byte>(plane.AsSpan(mid + x + 1));
                        lanes[6] = new Vector<byte>(plane.AsSpan(down + x - 1));
                        lanes[7] = new Vector<byte>(plane.AsSpan(down + x));
                        lanes[8] = new Vector<byte>(plane.AsSpan(down + x + 1));
                        var net = NineSort;
                        for (int i = 0; i < net.Length; i += 2)
                        {
                            var a = lanes[net[i]];
                            var b = lanes[net[i + 1]];
                            lanes[net[i]] = Vector.Min(a, b);
                            lanes[net[i + 1]] = Vector.Max(a, b);
                        }

                        lanes[rank].CopyTo(output.AsSpan(mid + x));
                    }
                }

                for (; x < w; x++)
                {
                    RankOne(plane, output, nine, up, mid, down, w, x, rank);
                }
            }
        });
        return output;
    }

    // One 3 x 3 window of RankOfNine (rows up, mid and down; columns clamped to the image) through the same network.
    private static void RankOne(byte[] plane, byte[] output, Span<byte> nine, int up, int mid, int down, int w, int x, int rank)
    {
        int left = Math.Max(x - 1, 0), right = Math.Min(x + 1, w - 1);
        nine[0] = plane[up + left];
        nine[1] = plane[up + x];
        nine[2] = plane[up + right];
        nine[3] = plane[mid + left];
        nine[4] = plane[mid + x];
        nine[5] = plane[mid + right];
        nine[6] = plane[down + left];
        nine[7] = plane[down + x];
        nine[8] = plane[down + right];
        var net = NineSort;
        for (int i = 0; i < net.Length; i += 2)
        {
            byte a = nine[net[i]], b = nine[net[i + 1]];
            nine[net[i]] = Math.Min(a, b);
            nine[net[i + 1]] = Math.Max(a, b);
        }

        output[mid + x] = nine[rank];
    }

    /// <summary><c>ImageFilter.MedianFilter(size)</c>: each pixel the median of its neighbourhood (specks and salt-and-pepper noise removed).</summary>
    public static ImageData Median(ImageData image, int size = 3) => RankFilter(image, size, size * size / 2);

    /// <summary><c>ImageFilter.MinFilter(size)</c>: each pixel the darkest of its neighbourhood (dark strokes on a light page grow thicker).</summary>
    public static ImageData MinFilter(ImageData image, int size = 3) => RankFilter(image, size, 0);

    /// <summary><c>ImageFilter.MaxFilter(size)</c>: each pixel the lightest of its neighbourhood (dark strokes grow thinner, dark specks go).</summary>
    public static ImageData MaxFilter(ImageData image, int size = 3) => RankFilter(image, size, size * size - 1);

    /// <summary>
    /// <c>ImageOps.equalize(image)</c>: each channel's histogram spread evenly over 0 to 255 (a channel of one value, or
    /// whose step is 0, unchanged).
    /// </summary>
    public static ImageData Equalize(ImageData image)
    {
        ArgumentNullException.ThrowIfNull(image);
        var planes = Bytes(image);
        int size = image.Height * image.Width;
        var lut = new byte[256];
        foreach (var plane in planes)
        {
            var histogram = new long[256];
            foreach (byte v in plane.AsSpan(0, size))
            {
                histogram[v]++;
            }

            long total = 0, last = 0;
            int used = 0;
            for (int i = 0; i < 256; i++)
            {
                if (histogram[i] != 0)
                {
                    total += histogram[i];
                    last = histogram[i];
                    used++;
                }
            }

            long step = used <= 1 ? 0 : (total - last) / 255;
            if (step == 0)
            {
                continue;                                                          // lut = range(256)
            }

            long n = step / 2;
            for (int i = 0; i < 256; i++)
            {
                lut[i] = (byte)Math.Min(255, n / step);                       // point() clips the table to a byte
                n += histogram[i];
            }

            Map(plane, size, lut);
        }

        return FromBytes(planes, image.Height, image.Width);
    }

    /// <summary>
    /// Gamma by a point table: each byte v becomes <c>int((v / 255.0) ** (1.0 / gamma) * 255 + 0.5)</c> on every channel.
    /// Below 1 darkens the mid-tones (faint, thin ink stands out); above 1 lightens them.
    /// </summary>
    public static ImageData Gamma(ImageData image, double gamma)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(gamma);
        var lut = new byte[256];
        for (int i = 0; i < 256; i++)
        {
            lut[i] = (byte)(int)(Math.Pow(i / 255.0, 1.0 / gamma) * 255 + 0.5);
        }

        var planes = Bytes(image);
        int size = image.Height * image.Width;
        foreach (var plane in planes)
        {
            Map(plane, size, lut);
        }

        return FromBytes(planes, image.Height, image.Width);
    }

    /// <summary>
    /// A black and white page: the image's grey (<c>convert("L")</c>) made 255 where it is at least
    /// <paramref name="threshold"/> and 0 below; without one, the threshold is Otsu's (the grey level that best separates
    /// ink from paper by the histogram, plus one). One channel.
    /// </summary>
    public static ImageData Binarize(ImageData image, int? threshold = null)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (threshold is < 0 or > 256)
        {
            throw new ArgumentOutOfRangeException(nameof(threshold), threshold, "A threshold is a grey level from 0 to 256.");
        }

        int size = image.Height * image.Width;
        var grey = image.Channels == 1 ? Bytes(image)[0] : ImagePreprocessor.ToBytes(image, grayscale: true)[0];   // one pass from the floats
        int t = threshold ?? Otsu(grey.AsSpan(0, size)) + 1;
        var lut = new byte[256];
        for (int i = 0; i < 256; i++)
        {
            lut[i] = (byte)(i >= t ? 255 : 0);
        }

        Map(grey, size, lut);
        return FromBytes([grey], image.Height, image.Width);
    }

    /// <summary><c>ImageFilter.GaussianBlur(radius)</c>: Pillow's three box blurs that approximate a Gaussian of that radius (in pixels; 0 unchanged).</summary>
    public static ImageData GaussianBlur(ImageData image, double radius)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentOutOfRangeException.ThrowIfNegative(radius);
        if (radius == 0)
        {
            return image;
        }

        var planes = Bytes(image);
        int size = image.Height * image.Width;
        float box = GaussianBoxRadius((float)radius, 3);
        for (int c = 0; c < planes.Length; c++)
        {
            planes[c] = BoxBlur(planes[c].AsSpan(0, size).ToArray(), image.Height, image.Width, box, 3);
        }

        return FromBytes(planes, image.Height, image.Width);
    }

    /// <summary>
    /// <c>ImageFilter.UnsharpMask(radius, percent, threshold)</c>: where a pixel differs from its Gaussian blur by more than
    /// <paramref name="threshold"/>, the difference times <paramref name="percent"/> / 100 is added (edges of strokes
    /// sharpened, flat paper left alone).
    /// </summary>
    public static ImageData UnsharpMask(ImageData image, double radius = 2, int percent = 150, int threshold = 3)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentOutOfRangeException.ThrowIfNegative(radius);
        var planes = Bytes(image);
        int size = image.Height * image.Width;
        float box = GaussianBoxRadius((float)radius, 3);
        var result = new byte[planes.Length][];
        for (int c = 0; c < planes.Length; c++)
        {
            var input = planes[c];
            var blurred = BoxBlur(input.AsSpan(0, size).ToArray(), image.Height, image.Width, box, 3);
            HostParallel.For(size, 1 << 15, (first, last) =>
            {
                for (int i = first; i < last; i++)
                {
                    int diff = input[i] - blurred[i];
                    blurred[i] = Math.Abs(diff) > threshold ? (byte)Math.Clamp(input[i] + diff * percent / 100, 0, 255) : input[i];
                }
            });

            result[c] = blurred;
        }

        return FromBytes(result, image.Height, image.Width);
    }

    // ---------------------------------------------------------------- helpers

    // Every byte of plane[..length] through a 256-entry table, in place, in parallel chunks (each byte on its own: the
    // result does not depend on the split).
    internal static void Map(byte[] plane, int length, byte[] lut)
    {
        HostParallel.For(length, 1 << 15, (first, last) =>
        {
            for (int i = first; i < last; i++)
            {
                plane[i] = lut[plane[i]];
            }
        });
    }

    // Otsu's threshold: the level t that maximizes the between-class variance of [0, t] and (t, 255], in double as the
    // reference computes it (the first such level on a tie).
    private static int Otsu(ReadOnlySpan<byte> grey)
    {
        Span<long> histogram = stackalloc long[256];
        histogram.Clear();
        foreach (byte v in grey)
        {
            histogram[v]++;
        }

        long total = grey.Length, sumAll = 0;
        for (int i = 0; i < 256; i++)
        {
            sumAll += i * histogram[i];
        }

        long weightBack = 0, sumBack = 0;
        double best = -1;
        int level = 0;
        for (int t = 0; t < 256; t++)
        {
            weightBack += histogram[t];
            if (weightBack == 0)
            {
                continue;
            }

            long weightFore = total - weightBack;
            if (weightFore == 0)
            {
                break;
            }

            sumBack += t * histogram[t];
            double meanBack = sumBack / (double)weightBack, meanFore = (sumAll - sumBack) / (double)weightFore;
            double difference = meanBack - meanFore;
            double between = (double)weightBack * weightFore * (difference * difference);
            if (between > best)
            {
                (best, level) = (between, t);
            }
        }

        return level;
    }

    // BoxBlur.c's _gaussian_blur_radius: the box radius whose `passes` box blurs approximate a Gaussian (float32, as C).
    private static float GaussianBoxRadius(float radius, int passes)
    {
        float sigma2 = radius * radius / passes;
        float length = (float)Math.Sqrt(12.0 * sigma2 + 1.0);
        float whole = (float)Math.Floor((length - 1.0) / 2.0);
        float fraction = (2 * whole + 1) * (whole * (whole + 1) - 3 * sigma2);
        fraction /= 6 * (sigma2 - (whole + 1) * (whole + 1));
        return whole + fraction;
    }

    // ImagingBoxBlur on one 8-bit plane: `passes` horizontal blurs, then `passes` vertical ones (columns as rows: the plane
    // transposed in tiles, its rows blurred, transposed back, so each column is read in order, not a row's stride apart).
    private static byte[] BoxBlur(byte[] plane, int height, int width, float radius, int passes)
    {
        if (radius == 0)
        {
            return plane;
        }

        RowsBoxBlur(plane, height, width, radius, passes);
        var columns = Transposed(plane, height, width);
        RowsBoxBlur(columns, width, height, radius, passes);
        return Transposed(columns, width, height);
    }

    // `passes` box blurs of each row of a [rows, width] plane, in place, rows in parallel.
    private static void RowsBoxBlur(byte[] plane, int rows, int width, float radius, int passes) =>
        Parallel.For(0, rows, () => new byte[width], (y, _, line) =>
        {
            var row = plane.AsSpan(y * width, width);
            for (int p = 0; p < passes; p++)
            {
                LineBoxBlur(row, line, radius);
                line.CopyTo(row);
            }

            return line;
        }, _ => { });

    // A [rows, columns] plane as [columns, rows], in tiles that stay in the cache, tile rows in parallel.
    private static byte[] Transposed(byte[] plane, int rows, int columns)
    {
        const int Tile = 64;
        var result = GC.AllocateUninitializedArray<byte>(rows * columns);
        HostParallel.For((rows + Tile - 1) / Tile, 1, (first, last) =>
        {
            for (int r0 = first * Tile; r0 < Math.Min(rows, last * Tile); r0 += Tile)
            {
                int r1 = Math.Min(rows, r0 + Tile);
                for (int c0 = 0; c0 < columns; c0 += Tile)
                {
                    int c1 = Math.Min(columns, c0 + Tile);
                    for (int r = r0; r < r1; r++)
                    {
                        for (int c = c0; c < c1; c++)
                        {
                            result[c * rows + r] = plane[r * columns + c];
                        }
                    }
                }
            }
        });
        return result;
    }

    // ImagingHorizontalBoxBlur's line (ImagingLineBoxBlur8): unsigned 32-bit sums, the fractional edge weights in 8.24
    // fixed point, edges repeated.
    private static void LineBoxBlur(ReadOnlySpan<byte> input, Span<byte> output, float floatRadius)
    {
        int size = input.Length, lastx = size - 1;
        int radius = (int)floatRadius;
        uint ww = (uint)(16777216f / (floatRadius * 2 + 1));
        uint fw = (16777216u - (uint)(radius * 2 + 1) * ww) / 2;
        int edgeA = Math.Min(radius + 1, size), edgeB = Math.Max(size - radius - 1, 0);
        unchecked
        {
            uint acc = (uint)(input[0] * (radius + 1));
            for (int x = 0; x < edgeA - 1; x++)
            {
                acc += input[x];
            }

            acc += (uint)(input[lastx] * (radius - edgeA + 1));
            void Step(ref uint sum, Span<byte> into, ReadOnlySpan<byte> line, int x, int subtract, int add, int left, int right)
            {
                sum += (uint)(line[add] - line[subtract]);
                uint bulk = sum * ww + (uint)(line[left] + line[right]) * fw;
                into[x] = (byte)((bulk + (1u << 23)) >> 24);
            }

            if (edgeA <= edgeB)
            {
                for (int x = 0; x < edgeA; x++)
                {
                    Step(ref acc, output, input, x, 0, x + radius, 0, x + radius + 1);
                }

                for (int x = edgeA; x < edgeB; x++)
                {
                    Step(ref acc, output, input, x, x - radius - 1, x + radius, x - radius - 1, x + radius + 1);
                }

                for (int x = edgeB; x <= lastx; x++)
                {
                    Step(ref acc, output, input, x, x - radius - 1, lastx, x - radius - 1, lastx);
                }
            }
            else
            {
                for (int x = 0; x < edgeB; x++)
                {
                    Step(ref acc, output, input, x, 0, x + radius, 0, x + radius + 1);
                }

                for (int x = edgeB; x < edgeA; x++)
                {
                    Step(ref acc, output, input, x, 0, lastx, 0, lastx);
                }

                for (int x = edgeA; x <= lastx; x++)
                {
                    Step(ref acc, output, input, x, x - radius - 1, lastx, x - radius - 1, lastx);
                }
            }
        }
    }
}
