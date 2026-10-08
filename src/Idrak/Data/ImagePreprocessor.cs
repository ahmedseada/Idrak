// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json;
using Idrak.Abstraction.Devices;
using Idrak.Data.Abstractions;

namespace Idrak.Data;

/// <summary>
/// Pillow's resampling filters, numbered as Pillow numbers them (and as the "resample" of a Hugging Face
/// <c>preprocessor_config.json</c> gives them). Nearest (0) is not offered.
/// </summary>
public enum ImageResampling
{
    /// <summary>Lanczos (a windowed sinc over 3 pixels), Pillow's 1.</summary>
    Lanczos = 1,

    /// <summary>Bilinear (a triangle over 1 pixel), Pillow's 2; Gemma 3's choice.</summary>
    Bilinear = 2,

    /// <summary>Bicubic (Keys' cubic with a = -0.5, over 2 pixels), Pillow's 3; SigLIP's and CLIP's choice.</summary>
    Bicubic = 3,

    /// <summary>Box (the mean of the pixels covered), Pillow's 4.</summary>
    Box = 4,

    /// <summary>Hamming (a Hamming-windowed sinc over 1 pixel), Pillow's 5.</summary>
    Hamming = 5,
}

/// <summary>
/// The image steps a vision-language model's <c>preprocessor_config.json</c> asks for (Gemma 3's and SigLIP's image
/// processors): optionally to grey, to RGB, resized, rescaled and normalized, giving the [3, height, width] float values
/// transformers calls <c>pixel_values</c> (channels first).
/// <para>
/// The resize is Pillow's <c>Image.resize</c> (its <c>ImagingResample</c>), which transformers' PIL image processors
/// call: two passes (across, then down) on 8-bit pixels, each output pixel a weighted sum over a support widened by the
/// shrink factor (so shrinking antialiases: a 2,000-pixel scan shrunk to 896 sees every source pixel), weights rounded
/// to 22 fractional bits and every pass rounded to 8 bits, as Pillow does; the files of the tests give Pillow's bytes
/// exactly. Rescaling and normalizing then follow transformers' arithmetic: the byte times the factor in double
/// precision, rounded to float, then (value - mean) / std in float.
/// </para>
/// <para>
/// Grayscale (not a transformers step: some fine-tunes, such as OCR models trained on grey scans, ask for it) converts
/// the 8-bit RGB pixels as Pillow's <c>convert("L")</c> does (L = (19595 R + 38470 G + 7471 B + 32768) >> 16, the
/// ITU-R 601 weights in 16-bit fixed point) before the resize, and repeats the grey to three channels.
/// </para>
/// </summary>
public sealed class ImagePreprocessor
{
    /// <summary>Whether images are resized to <see cref="Height"/> x <see cref="Width"/> ("do_resize").</summary>
    public bool Resize { get; init; } = true;

    /// <summary>The height images are resized to ("size": {"height"}).</summary>
    public int Height { get; init; }

    /// <summary>The width images are resized to ("size": {"width"}).</summary>
    public int Width { get; init; }

    /// <summary>The resize filter ("resample"); bilinear unless set.</summary>
    public ImageResampling Resampling { get; init; } = ImageResampling.Bilinear;

    /// <summary>Whether the 8-bit values are multiplied by <see cref="RescaleFactor"/> ("do_rescale").</summary>
    public bool Rescale { get; init; } = true;

    /// <summary>The factor ("rescale_factor"), 1 / 255 unless set.</summary>
    public double RescaleFactor { get; init; } = 1 / 255.0;

    /// <summary>Whether (value - mean) / std is taken per channel ("do_normalize").</summary>
    public bool Normalize { get; init; } = true;

    /// <summary>The mean per channel ("image_mean"), or one for every channel; 0.5 unless set.</summary>
    public IReadOnlyList<float> Mean { get; init; } = [0.5f, 0.5f, 0.5f];

    /// <summary>The standard deviation per channel ("image_std"), or one for every channel; 0.5 unless set.</summary>
    public IReadOnlyList<float> Std { get; init; } = [0.5f, 0.5f, 0.5f];

    /// <summary>Whether grey images become three channels ("do_convert_rgb"); without it a grey image stays one channel.</summary>
    public bool ConvertRgb { get; init; } = true;

    /// <summary>Whether colour images are turned to grey first (Pillow's "L"), then repeated to three channels.</summary>
    public bool Grayscale { get; init; }

    /// <summary>
    /// Reads a <c>preprocessor_config.json</c> (the file, or the model folder holding it). Keys it does not name keep the
    /// defaults above; <c>do_pan_and_scan</c> true, a <c>size</c> without height and width (shortest edge) and a
    /// center crop are not supported yet.
    /// </summary>
    public static ImagePreprocessor FromConfig(string path, bool grayscale = false)
    {
        string file = Directory.Exists(path) ? Path.Combine(path, "preprocessor_config.json") : path;
        return Parse(File.ReadAllText(file), grayscale);
    }

    /// <summary>The steps of a <c>preprocessor_config.json</c>'s text (see <see cref="FromConfig"/>).</summary>
    /// <exception cref="NotSupportedException">Pan and scan, a size by shortest edge, a center crop or nearest resampling.</exception>
    public static ImagePreprocessor Parse(string json, bool grayscale = false)
    {
        using var document = JsonDocument.Parse(json);
        var root = document.RootElement;
        if (root.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException("A preprocessor config is a JSON object.");
        }

        if (Flag(root, "do_pan_and_scan", false))
        {
            throw new NotSupportedException("do_pan_and_scan (Gemma 3's crops of tall or wide images) is not supported yet; set it to false or null.");
        }

        if (Flag(root, "do_center_crop", false))
        {
            throw new NotSupportedException("do_center_crop is not supported; the processors of Gemma 3 and SigLIP do not crop.");
        }

        bool resize = Flag(root, "do_resize", true);
        int height = 0, width = 0;
        if (root.TryGetProperty("size", out var size) && size.ValueKind == JsonValueKind.Object)
        {
            if (size.TryGetProperty("height", out var h) && size.TryGetProperty("width", out var w))
            {
                height = h.GetInt32();
                width = w.GetInt32();
            }
            else if (resize)
            {
                throw new NotSupportedException($"A size of {size.GetRawText()} (by shortest or longest edge) is not supported; give height and width.");
            }
        }

        if (resize && (height <= 0 || width <= 0))
        {
            throw new InvalidDataException("do_resize without a size of height and width.");
        }

        int resample = root.TryGetProperty("resample", out var r) && r.ValueKind == JsonValueKind.Number ? r.GetInt32() : (int)ImageResampling.Bilinear;
        if (!Enum.IsDefined((ImageResampling)resample))
        {
            throw new NotSupportedException($"resample {resample} is not supported (1 Lanczos, 2 bilinear, 3 bicubic, 4 box, 5 Hamming).");
        }

        return new ImagePreprocessor
        {
            Resize = resize,
            Height = height,
            Width = width,
            Resampling = (ImageResampling)resample,
            Rescale = Flag(root, "do_rescale", true),
            RescaleFactor = root.TryGetProperty("rescale_factor", out var f) && f.ValueKind == JsonValueKind.Number ? f.GetDouble() : 1 / 255.0,
            Normalize = Flag(root, "do_normalize", true),
            Mean = Values(root, "image_mean"),
            Std = Values(root, "image_std"),
            ConvertRgb = Flag(root, "do_convert_rgb", true),
            Grayscale = grayscale,
        };

        static bool Flag(JsonElement root, string name, bool otherwise) =>
            root.TryGetProperty(name, out var value) ? value.ValueKind switch
            {
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                _ => otherwise,                                                          // null: the processor's default
            } : otherwise;

        static float[] Values(JsonElement root, string name) =>
            !root.TryGetProperty(name, out var value) || value.ValueKind == JsonValueKind.Null ? [0.5f, 0.5f, 0.5f]
            : value.ValueKind == JsonValueKind.Number ? [value.GetSingle()]
            : [.. value.EnumerateArray().Select(v => v.GetSingle())];
    }

    /// <summary>The pixel values of <paramref name="image"/> as [channels, height, width] floats (channels is 3 unless a grey image is kept grey).</summary>
    public float[] Pixels(ImageData image) => Pixels(image, out _, out _, out _);

    /// <summary>The pixel values of an image file (any registered codec), as <see cref="Pixels(ImageData)"/>.</summary>
    public float[] Pixels(string path) => Pixels(ImageCodecs.Decode(path));

    /// <summary>The pixel values of <paramref name="image"/> as a [channels, height, width] tensor on <paramref name="device"/>.</summary>
    public Tensor Process(ImageData image, Device? device = null)
    {
        var values = Pixels(image, out int channels, out int height, out int width);
        return Tensor.From(values, [channels, height, width], device);
    }

    /// <summary>The pixel values of an image file (any registered codec) as a tensor, as <see cref="Process(ImageData, Device?)"/>.</summary>
    public Tensor Process(string path, Device? device = null) => Process(ImageCodecs.Decode(path), device);

    private float[] Pixels(ImageData image, out int channels, out int height, out int width)
    {
        ArgumentNullException.ThrowIfNull(image);
        int h = image.Height, w = image.Width;
        var planes = ToBytes(image, Grayscale);
        if (Resize && (Height != h || Width != w))
        {
            if (Height <= 0 || Width <= 0)
            {
                throw new InvalidOperationException($"Resize to {Height} x {Width}: give a positive height and width.");
            }

            for (int c = 0; c < planes.Length; c++)
            {
                planes[c] = PillowResize(planes[c], h, w, Height, Width, Resampling);
            }

            (h, w) = (Height, Width);
        }

        channels = planes.Length == 1 && (ConvertRgb || Grayscale) ? 3 : planes.Length;
        height = h;
        width = w;
        if (Normalize && (Mean.Count is not 1 && Mean.Count != channels || Std.Count is not 1 && Std.Count != channels))
        {
            throw new InvalidOperationException($"{Mean.Count} means and {Std.Count} standard deviations for {channels} channels.");
        }

        var result = new float[checked(channels * h * w)];
        var table = new float[256];
        for (int c = 0; c < channels; c++)
        {
            for (int v = 0; v < 256; v++)
            {
                float value = Rescale ? (float)(v * RescaleFactor) : v;                 // transformers: float64 product, then float32
                if (Normalize)
                {
                    float mean = Mean[Mean.Count == 1 ? 0 : c], std = Std[Std.Count == 1 ? 0 : c];
                    value = (value - mean) / std;
                }

                table[v] = value;
            }

            var plane = planes[Math.Min(c, planes.Length - 1)];
            var target = result.AsSpan(c * h * w, h * w);
            for (int i = 0; i < plane.Length; i++)
            {
                target[i] = table[plane[i]];
            }
        }

        return result;
    }

    // The image as 8-bit planes (values rounded from [0, 1]), turned to one grey plane as Pillow's "L" when asked.
    private static byte[][] ToBytes(ImageData image, bool grayscale)
    {
        int size = image.Height * image.Width, c = image.Channels;
        var pixels = image.Pixels;
        if (grayscale && c == 3)
        {
            var grey = new byte[size];
            HostParallel.For(size, 1 << 14, (first, last) =>
            {
                for (int i = first; i < last; i++)
                {
                    int red = Byte(pixels[i]), green = Byte(pixels[size + i]), blue = Byte(pixels[2 * size + i]);
                    grey[i] = (byte)((red * 19595 + green * 38470 + blue * 7471 + 0x8000) >> 16);
                }
            });
            return [grey];
        }

        var planes = new byte[c][];
        for (int ch = 0; ch < c; ch++)
        {
            var plane = planes[ch] = new byte[size];
            int offset = ch * size;
            HostParallel.For(size, 1 << 14, (first, last) =>
            {
                for (int i = first; i < last; i++)
                {
                    plane[i] = (byte)Byte(pixels[offset + i]);
                }
            });
        }

        return planes;

        static int Byte(float v) => v <= 0 ? 0 : v >= 1 ? 255 : (int)MathF.Round(v * 255f);
    }

    // ---------------------------------------------------------------- Pillow's resize (Resample.c)

    private const int PrecisionBits = 32 - 8 - 2;

    /// <summary>
    /// One 8-bit plane resized as Pillow's <c>ImagingResample</c> resizes an 8-bit band: the horizontal pass first (only
    /// over the rows the vertical pass reads), then the vertical, each skipped when its size does not change.
    /// </summary>
    internal static byte[] PillowResize(byte[] source, int height, int width, int outHeight, int outWidth, ImageResampling filter)
    {
        if (outHeight == height && outWidth == width)
        {
            return (byte[])source.Clone();
        }

        var (xBounds, xWeights, xTaps) = Coefficients(width, outWidth, filter);
        var (yBounds, yWeights, yTaps) = Coefficients(height, outHeight, filter);
        byte[] current = source;
        int rows = height;
        if (outWidth != width)
        {
            int first = yBounds[0], last = yBounds[2 * (outHeight - 1)] + yBounds[2 * (outHeight - 1) + 1];
            for (int i = 0; i < outHeight; i++)
            {
                yBounds[2 * i] -= first;
            }

            rows = last - first;
            var across = new byte[checked(rows * outWidth)];
            HostParallel.For(rows, 8, (r0, r1) =>
            {
                for (int y = r0; y < r1; y++)
                {
                    var input = source.AsSpan((first + y) * width, width);
                    var output = across.AsSpan(y * outWidth, outWidth);
                    for (int x = 0; x < outWidth; x++)
                    {
                        int start = xBounds[2 * x], count = xBounds[2 * x + 1];
                        var k = xWeights.AsSpan(x * xTaps, count);
                        int sum = 1 << (PrecisionBits - 1);
                        for (int i = 0; i < k.Length; i++)
                        {
                            sum += input[start + i] * k[i];
                        }

                        output[x] = Clip8(sum);
                    }
                }
            });
            current = across;
        }

        if (outHeight == height)
        {
            return current;
        }

        var result = new byte[checked(outHeight * outWidth)];
        var plane = current;
        HostParallel.For(outHeight, 4, (r0, r1) =>
        {
            var sums = new int[outWidth];
            for (int y = r0; y < r1; y++)
            {
                int start = yBounds[2 * y], count = yBounds[2 * y + 1];
                var k = yWeights.AsSpan(y * yTaps, count);
                Array.Fill(sums, 1 << (PrecisionBits - 1));
                for (int i = 0; i < k.Length; i++)
                {
                    var input = plane.AsSpan((start + i) * outWidth, outWidth);
                    int weight = k[i];
                    for (int x = 0; x < outWidth; x++)
                    {
                        sums[x] += input[x] * weight;                                   // integer sums: the order does not matter
                    }
                }

                var output = result.AsSpan(y * outWidth, outWidth);
                for (int x = 0; x < outWidth; x++)
                {
                    output[x] = Clip8(sums[x]);
                }
            }
        });
        return result;
    }

    private static byte Clip8(int value) => value >= 1 << PrecisionBits << 8 ? (byte)255 : value <= 0 ? (byte)0 : (byte)(value >> PrecisionBits);

    // Resample.c precompute_coeffs and normalize_coeffs_8bpc: for each output pixel, the first input pixel and the count
    // it reads (bounds, in pairs) and its weights (`taps` per pixel), as 22-bit fixed point rounded away from zero.
    private static (int[] Bounds, int[] Weights, int Taps) Coefficients(int inSize, int outSize, ImageResampling filter)
    {
        double support = filter switch
        {
            ImageResampling.Box => 0.5,
            ImageResampling.Bilinear or ImageResampling.Hamming => 1.0,
            ImageResampling.Bicubic => 2.0,
            ImageResampling.Lanczos => 3.0,
            _ => throw new NotSupportedException($"Resampling {filter} is not supported."),
        };
        double scale = (double)inSize / outSize, filterScale = Math.Max(scale, 1.0);
        support *= filterScale;
        int taps = (int)Math.Ceiling(support) * 2 + 1;
        var bounds = new int[outSize * 2];
        var weights = new int[outSize * taps];
        var k = new double[taps];
        double ss = 1.0 / filterScale;
        for (int xx = 0; xx < outSize; xx++)
        {
            double center = (xx + 0.5) * scale, sum = 0.0;
            int xmin = Math.Max(0, (int)(center - support + 0.5));
            int xmax = Math.Min(inSize, (int)(center + support + 0.5)) - xmin;
            for (int x = 0; x < xmax; x++)
            {
                double w = Filter(filter, ((x + xmin) - center + 0.5) * ss);
                k[x] = w;
                sum += w;
            }

            for (int x = 0; x < xmax; x++)
            {
                double w = sum != 0.0 ? k[x] / sum : k[x];
                weights[xx * taps + x] = w < 0 ? (int)(-0.5 + w * (1 << PrecisionBits)) : (int)(0.5 + w * (1 << PrecisionBits));
            }

            bounds[2 * xx] = xmin;
            bounds[2 * xx + 1] = Math.Max(0, xmax);
        }

        return (bounds, weights, taps);
    }

    private static double Filter(ImageResampling filter, double x)
    {
        switch (filter)
        {
            case ImageResampling.Box:
                return x > -0.5 && x <= 0.5 ? 1.0 : 0.0;
            case ImageResampling.Bilinear:
                x = Math.Abs(x);
                return x < 1.0 ? 1.0 - x : 0.0;
            case ImageResampling.Hamming:
                x = Math.Abs(x);
                if (x == 0.0)
                {
                    return 1.0;
                }

                if (x >= 1.0)
                {
                    return 0.0;
                }

                x *= Math.PI;
                return Math.Sin(x) / x * (0.54 + 0.46 * Math.Cos(x));
            case ImageResampling.Bicubic:
            {
                const double a = -0.5;
                x = Math.Abs(x);
                return x < 1.0 ? ((a + 2.0) * x - (a + 3.0)) * x * x + 1
                    : x < 2.0 ? (((x - 5) * x + 8) * x - 4) * a
                    : 0.0;
            }

            default:
                return x >= -3.0 && x < 3.0 ? Sinc(x) * Sinc(x / 3) : 0.0;
        }

        static double Sinc(double x)
        {
            if (x == 0.0)
            {
                return 1.0;
            }

            x *= Math.PI;
            return Math.Sin(x) / x;
        }
    }
}
