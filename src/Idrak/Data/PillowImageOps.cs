// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Data.Abstractions;

namespace Idrak.Data;

/// <summary>
/// Pillow's image operations on 8-bit pixels, byte for byte: what a model card's Python preprocessing does to a scan
/// (<c>convert("L")</c>, <c>resize</c>, <c>ImageEnhance</c>, <c>ImageOps.autocontrast</c>, a JPEG round trip), so an
/// image prepared here is the image prepared there. Each takes and gives decoded pixels (<see cref="ImageData"/>, values
/// in [0, 1]), read as the 8-bit bytes Pillow holds (<see cref="ImagePreprocessor"/>'s rounding) and given back as
/// byte / 255. Grey images stay one channel and colour ones three. They are the library's image transforms
/// (<see cref="ImageTransforms"/>: grayscale, max_width, max_height, contrast, brightness, sharpness, autocontrast, invert, jpeg).
/// <para>
/// Floating-point steps (the enhancers' blend, the smoothing filter) copy Pillow's C: float32 arithmetic in its order,
/// truncated to a byte; Pillow built for x86-64 (its wheels) does not fuse multiply-adds there, and neither does .NET.
/// A Pillow compiled to fuse them (some ARM builds) can differ by 1 at a few pixels.
/// </para>
/// </summary>
public static class PillowImageOps
{
    /// <summary>
    /// The image shrunk, keeping its aspect ratio, so it is at most <paramref name="maxWidth"/> wide and
    /// <paramref name="maxHeight"/> high (0: no limit), as the usual Python does: when wider than the limit, the width
    /// becomes the limit and the height <c>int(height * (limit / float(width)))</c> (truncated, at least 1); then the same
    /// for the height. An image within its limits is returned as it is. The resize is Pillow's <c>Image.resize</c>.
    /// </summary>
    public static ImageData ScaleDown(ImageData image, int maxWidth, int maxHeight, ImageResampling resampling = ImageResampling.Lanczos)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentOutOfRangeException.ThrowIfNegative(maxWidth);
        ArgumentOutOfRangeException.ThrowIfNegative(maxHeight);
        int w = image.Width, h = image.Height;
        if (maxWidth > 0 && w > maxWidth)
        {
            double ratio = maxWidth / (double)w;
            (w, h) = (maxWidth, Math.Max(1, (int)(h * ratio)));
        }

        if (maxHeight > 0 && h > maxHeight)
        {
            double ratio = maxHeight / (double)h;
            (w, h) = (Math.Max(1, (int)(w * ratio)), maxHeight);
        }

        return Resize(image, w, h, resampling);
    }

    /// <summary>The image resized to <paramref name="width"/> x <paramref name="height"/> as Pillow's <c>Image.resize</c> (the same image when the size does not change).</summary>
    public static ImageData Resize(ImageData image, int width, int height, ImageResampling resampling = ImageResampling.Lanczos)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        if (width == image.Width && height == image.Height)
        {
            return image;
        }

        var planes = Bytes(image);
        for (int c = 0; c < planes.Length; c++)
        {
            planes[c] = ImagePreprocessor.PillowResize(planes[c], image.Height, image.Width, height, width, resampling);
        }

        return FromBytes(planes, height, width);
    }

    /// <summary>
    /// <c>ImageEnhance.Contrast(image).enhance(factor)</c>: the mean of the image's grey (<c>convert("L")</c>) histogram,
    /// rounded as <c>int(mean + 0.5)</c>, is a flat grey image, which <c>Image.blend</c> moves away from (factor above 1)
    /// or towards (below 1).
    /// </summary>
    public static ImageData Contrast(ImageData image, double factor)
    {
        ArgumentNullException.ThrowIfNull(image);
        var planes = Bytes(image);
        var grey = planes.Length == 3 ? ImagePreprocessor.ToBytes(image, grayscale: true)[0] : planes[0];
        long sum = 0;
        foreach (byte v in grey)
        {
            sum += v;
        }

        int mean = (int)((double)sum / grey.Length + 0.5);
        return Blend(planes, _ => mean, (float)factor, image.Height, image.Width);
    }

    /// <summary><c>ImageEnhance.Brightness(image).enhance(factor)</c>: a blend with black.</summary>
    public static ImageData Brightness(ImageData image, double factor)
    {
        ArgumentNullException.ThrowIfNull(image);
        return Blend(Bytes(image), _ => 0, (float)factor, image.Height, image.Width);
    }

    /// <summary>
    /// <c>ImageEnhance.Sharpness(image).enhance(factor)</c>: a blend with the image smoothed by Pillow's 3 x 3
    /// <c>ImageFilter.SMOOTH</c> (weights 1 around 5, over 13, in float32; the border pixels unchanged).
    /// </summary>
    public static ImageData Sharpness(ImageData image, double factor)
    {
        ArgumentNullException.ThrowIfNull(image);
        var planes = Bytes(image);
        int h = image.Height, w = image.Width;
        var smooth = planes.Select(p => Smooth(p, h, w)).ToArray();
        float alpha = (float)factor;
        return Blend(planes, null, alpha, h, w, smooth);
    }

    /// <summary>
    /// <c>ImageOps.autocontrast(image, cutoff, ignore, preserve_tone=preserveTone)</c>: each channel's histogram (or the
    /// grey one's, keeping the tone) less <paramref name="cutoffLow"/> and <paramref name="cutoffHigh"/> percent at its
    /// ends, stretched so its darkest value becomes 0 and its lightest 255.
    /// </summary>
    public static ImageData Autocontrast(ImageData image, double cutoffLow = 0, double cutoffHigh = 0, int? ignore = null, bool preserveTone = false)
    {
        ArgumentNullException.ThrowIfNull(image);
        var planes = Bytes(image);
        var histograms = preserveTone
            ? [Histogram(planes.Length == 3 ? ImagePreprocessor.ToBytes(image, grayscale: true)[0] : planes[0])]
            : planes.Select(Histogram).ToArray();
        var tables = histograms.Select(hist => Stretch(hist, cutoffLow, cutoffHigh, ignore)).ToArray();
        var result = new byte[planes.Length][];
        for (int c = 0; c < planes.Length; c++)
        {
            var table = tables[Math.Min(c, tables.Length - 1)];
            result[c] = [.. planes[c].Select(v => table[v])];
        }

        return FromBytes(result, image.Height, image.Width);

        static long[] Histogram(byte[] plane)
        {
            var hist = new long[256];
            foreach (byte v in plane)
            {
                hist[v]++;
            }

            return hist;
        }
    }

    /// <summary><c>ImageOps.invert(image)</c>: every byte of every channel v becomes 255 - v (a dark page on light text turns light on dark, and back).</summary>
    public static ImageData Invert(ImageData image)
    {
        ArgumentNullException.ThrowIfNull(image);
        var planes = Bytes(image);
        foreach (var plane in planes)
        {
            var span = plane.AsSpan(0, image.Height * image.Width);
            int i = 0;
            if (System.Numerics.Vector.IsHardwareAccelerated)
            {
                var bytes = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, System.Numerics.Vector<byte>>(span);
                for (int v = 0; v < bytes.Length; v++)
                {
                    bytes[v] = ~bytes[v];                                      // 255 - v for a byte
                }

                i = bytes.Length * System.Numerics.Vector<byte>.Count;
            }

            for (; i < span.Length; i++)
            {
                span[i] = (byte)~span[i];
            }
        }

        return FromBytes(planes, image.Height, image.Width);
    }

    /// <summary>
    /// The image saved as a JPEG at <paramref name="quality"/> (Pillow's <c>save(..., "JPEG", quality=Q,
    /// optimize=True)</c>: <see cref="JpegEncoder"/>), then decoded (<see cref="ImageCodecs"/>' JPEG, as Pillow decodes):
    /// what a server that receives the file as a base64 data URL reads.
    /// </summary>
    public static ImageData JpegRoundTrip(ImageData image, int quality, JpegSubsampling subsampling = JpegSubsampling.Half420) =>
        ImageCodecs.Get("jpeg").Decode(JpegEncoder.Encode(image, quality, optimize: true, subsampling));

    // ---------------------------------------------------------------- helpers

    internal static byte[][] Bytes(ImageData image) => ImagePreprocessor.ToBytes(image, grayscale: false);

    internal static ImageData FromBytes(byte[][] planes, int height, int width)
    {
        int size = height * width;
        var pixels = new float[planes.Length * size];
        for (int c = 0; c < planes.Length; c++)
        {
            ImageLevels.ToUnit(planes[c].AsSpan(0, size), pixels.AsSpan(c * size, size));
        }

        return new ImageData(pixels, planes.Length, height, width);
    }

    // Blend.c ImagingBlend(degenerate, image, alpha): in1 + alpha * (in2 - in1) in float32, truncated to a byte (clipped
    // when extrapolating); alpha 0 gives the degenerate image and 1 the image.
    private static ImageData Blend(byte[][] planes, Func<int, int>? flat, float alpha, int h, int w, byte[][]? degenerate = null)
    {
        var result = new byte[planes.Length][];
        for (int c = 0; c < planes.Length; c++)
        {
            var image = planes[c];
            var output = result[c] = new byte[image.Length];
            byte[]? other = degenerate?[c];
            int constant = flat?.Invoke(c) ?? 0;
            if (alpha == 0f)
            {
                for (int i = 0; i < output.Length; i++)
                {
                    output[i] = other is null ? (byte)constant : other[i];
                }

                continue;
            }

            if (alpha == 1f)
            {
                image.CopyTo(output, 0);
                continue;
            }

            bool interpolate = alpha >= 0 && alpha <= 1.0;
            for (int i = 0; i < output.Length; i++)
            {
                int in1 = other is null ? constant : other[i], in2 = image[i];
                float temp = in1 + alpha * (in2 - in1);
                output[i] = interpolate ? (byte)(int)temp : temp <= 0f ? (byte)0 : temp >= 255f ? (byte)255 : (byte)(int)temp;
            }
        }

        return FromBytes(result, h, w);
    }

    // Filter.c ImagingFilter3x3 with SMOOTH's kernel (each weight divided by 13 in float32), offset 0 plus 0.5 for rounding.
    private static byte[] Smooth(byte[] plane, int h, int w)
    {
        var output = (byte[])plane.Clone();
        if (h < 3 || w < 3)
        {
            return output;                                                                       // ImagingFilter copies an image smaller than its kernel
        }

        float edge = 1f / 13f, centre = 5f / 13f;
        for (int y = 1; y < h - 1; y++)
        {
            int up = (y - 1) * w, row = y * w, down = (y + 1) * w;
            for (int x = 1; x < w - 1; x++)
            {
                float ss = 0.5f;
                ss += plane[down + x - 1] * edge + plane[down + x] * edge + plane[down + x + 1] * edge;
                ss += plane[row + x - 1] * edge + plane[row + x] * centre + plane[row + x + 1] * edge;
                ss += plane[up + x - 1] * edge + plane[up + x] * edge + plane[up + x + 1] * edge;
                ss = ss < 0f ? 0f : ss;
                ss = ss > 255f ? 255f : ss;
                output[row + x] = (byte)(int)ss;
            }
        }

        return output;
    }

    // ImageOps.autocontrast's lookup table for one histogram (Python's float arithmetic, floor division as CPython's).
    private static byte[] Stretch(long[] histogram, double cutoffLow, double cutoffHigh, int? ignore)
    {
        var h = (long[])histogram.Clone();
        if (ignore is int skip and >= 0 and <= 255)
        {
            h[skip] = 0;
        }

        if (cutoffLow != 0 || cutoffHigh != 0)
        {
            long n = h.Sum();
            long cut = (long)FloorDivide(n * cutoffLow, 100);
            for (int lo = 0; lo < 256; lo++)
            {
                if (cut > h[lo])
                {
                    cut -= h[lo];
                    h[lo] = 0;
                }
                else
                {
                    h[lo] -= cut;
                    cut = 0;
                }

                if (cut <= 0)
                {
                    break;
                }
            }

            cut = (long)FloorDivide(n * cutoffHigh, 100);
            for (int hi = 255; hi >= 0; hi--)
            {
                if (cut > h[hi])
                {
                    cut -= h[hi];
                    h[hi] = 0;
                }
                else
                {
                    h[hi] -= cut;
                    cut = 0;
                }

                if (cut <= 0)
                {
                    break;
                }
            }
        }

        int low = 0, high = 255;
        while (low < 255 && h[low] == 0)
        {
            low++;
        }

        while (high > 0 && h[high] == 0)
        {
            high--;
        }

        var table = new byte[256];
        if (high <= low)
        {
            for (int i = 0; i < 256; i++)
            {
                table[i] = (byte)i;
            }

            return table;
        }

        double scale = 255.0 / (high - low), offset = -low * scale;
        for (int i = 0; i < 256; i++)
        {
            int v = (int)(i * scale + offset);
            table[i] = (byte)Math.Clamp(v, 0, 255);
        }

        return table;

        // CPython's float_floor_div (Objects/floatobject.c, _float_div_mod).
        static double FloorDivide(double vx, double wx)
        {
            double mod = vx % wx;                                                                // C's fmod
            double div = (vx - mod) / wx;
            if (mod != 0 && (wx < 0) != (mod < 0))
            {
                div -= 1.0;
            }

            if (div == 0)
            {
                return Math.CopySign(0.0, vx / wx);
            }

            double floor = Math.Floor(div);
            if (div - floor > 0.5)
            {
                floor += 1.0;
            }

            return floor;
        }
    }
}
