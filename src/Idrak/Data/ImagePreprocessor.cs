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
/// The image steps a vision-language model's <c>preprocessor_config.json</c> asks for (the PIL image processors of
/// transformers: SigLIP's, Gemma 3's, CLIP's, LLaVA's): optionally to grey, to RGB, resized (to a size, or by its shortest
/// edge), center-cropped, rescaled and normalized, giving the [3, height, width] float values transformers calls
/// <c>pixel_values</c> (channels first). A building block: each vision family chooses its own steps and the defaults of
/// its own processor class (<see cref="Parse(string, bool, ImagePreprocessor?)"/>); the defaults of this class are
/// transformers' base processor's (bilinear, mean and std 0.5).
/// <para>
/// The resize is Pillow's <c>Image.resize</c> (its <c>ImagingResample</c>), which transformers' PIL image processors
/// call: two passes (across, then down) on 8-bit pixels, each output pixel a weighted sum over a support widened by the
/// shrink factor (so shrinking antialiases: a 2,000-pixel scan shrunk to 896 sees every source pixel), weights rounded
/// to 22 fractional bits and every pass rounded to 8 bits, as Pillow does; the files of the tests give Pillow's bytes
/// exactly. Rescaling and normalizing then follow transformers' arithmetic: the byte times the factor in double
/// precision, rounded to float, then (value - mean) / std in float.
/// </para>
/// <para>
/// Colour is the default path, and every format of the built-in codecs reaches it as the RGB bytes transformers'
/// <c>convert_to_rgb</c> (Pillow's <c>convert("RGB")</c>) gives: alpha is dropped, not composited (the colours under
/// clear pixels stay), a palette's tRNS is ignored, grey is repeated to three channels, CMYK and YCCK JPEGs are
/// converted as Pillow converts them, 16-bit PNG and PPM channels become bytes as Pillow makes them. The one known
/// difference: a 16-bit grey PNG (or a grey PGM deeper than 8 bits) is taken by its high byte here, where Pillow's
/// conversion clips its 16-bit values at 255 (an almost white image). Neither the ICC profile nor the EXIF orientation
/// is applied (transformers' image processors do not; its <c>load_image</c> rotates by the orientation).
/// </para>
/// <para>
/// Grayscale (not a transformers step: some fine-tunes, such as OCR models trained on grey scans, ask for it) converts
/// the 8-bit RGB pixels as Pillow's <c>convert("L")</c> does (L = (19595 R + 38470 G + 7471 B + 32768) >> 16, the
/// ITU-R 601 weights in 16-bit fixed point) before the resize, and repeats the grey to three channels.
/// </para>
/// </summary>
public sealed class ImagePreprocessor
{
    /// <summary>Whether images are resized to <see cref="Height"/> x <see cref="Width"/>, or by <see cref="ShortestEdge"/> ("do_resize").</summary>
    public bool Resize { get; init; } = true;

    /// <summary>
    /// Resize so the shorter side has this length and the longer keeps the aspect ratio, truncated as transformers'
    /// <c>get_resize_output_image_size</c> does ("size": {"shortest_edge"}); 0 (the default): resize to
    /// <see cref="Height"/> x <see cref="Width"/>.
    /// </summary>
    public int ShortestEdge { get; init; }

    /// <summary>
    /// Whether the resized image is cut to <see cref="CropHeight"/> x <see cref="CropWidth"/> around its center
    /// ("do_center_crop"; transformers' <c>center_crop</c>: offsets rounded down, zeros around an image smaller than the crop).
    /// </summary>
    public bool CenterCrop { get; init; }

    /// <summary>The crop's height ("crop_size": {"height"}).</summary>
    public int CropHeight { get; init; }

    /// <summary>The crop's width ("crop_size": {"width"}).</summary>
    public int CropWidth { get; init; }

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
    /// Reads a <c>preprocessor_config.json</c> (the file, or the model folder holding it). Keys it does not name (or names
    /// as null) take <paramref name="defaults"/>' values: the family's processor class's own defaults (null: this class's);
    /// a size by longest edge and nearest resampling are not supported. Keys that are not steps of this class (a family's
    /// own, such as the views or tiles some processors make of an image) are left to the family that reads them.
    /// </summary>
    public static ImagePreprocessor FromConfig(string path, bool grayscale = false, ImagePreprocessor? defaults = null)
    {
        string file = Directory.Exists(path) ? Path.Combine(path, "preprocessor_config.json") : path;
        return Parse(File.ReadAllText(file), grayscale, defaults);
    }

    /// <summary>The steps of a <c>preprocessor_config.json</c>'s text (see <see cref="FromConfig"/>).</summary>
    /// <exception cref="NotSupportedException">A size by longest edge, or nearest resampling.</exception>
    public static ImagePreprocessor Parse(string json, bool grayscale = false, ImagePreprocessor? defaults = null)
    {
        var d = defaults ?? new ImagePreprocessor();
        using var document = JsonDocument.Parse(json);
        var root = document.RootElement;
        if (root.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException("A preprocessor config is a JSON object.");
        }

        bool resize = Flag(root, "do_resize", d.Resize);
        int height = d.Height, width = d.Width, shortest = d.ShortestEdge;
        if (root.TryGetProperty("size", out var size) && size.ValueKind is JsonValueKind.Object or JsonValueKind.Number)
        {
            if (size.ValueKind == JsonValueKind.Number)
            {
                (height, width, shortest) = (size.GetInt32(), size.GetInt32(), 0);       // an old processor's one number: a square
            }
            else if (size.TryGetProperty("height", out var h) && size.TryGetProperty("width", out var w))
            {
                (height, width, shortest) = (h.GetInt32(), w.GetInt32(), 0);
            }
            else if (size.TryGetProperty("shortest_edge", out var edge) && !size.TryGetProperty("longest_edge", out _))
            {
                (height, width, shortest) = (0, 0, edge.GetInt32());
            }
            else if (resize)
            {
                throw new NotSupportedException($"A size of {size.GetRawText()} is not supported; give height and width, or shortest_edge alone.");
            }
        }

        if (resize && shortest <= 0 && (height <= 0 || width <= 0))
        {
            throw new InvalidDataException("do_resize without a size (height and width, or shortest_edge).");
        }

        bool crop = Flag(root, "do_center_crop", d.CenterCrop);
        int cropHeight = d.CropHeight, cropWidth = d.CropWidth;
        if (root.TryGetProperty("crop_size", out var cropSize))
        {
            if (cropSize.ValueKind == JsonValueKind.Number)
            {
                cropHeight = cropWidth = cropSize.GetInt32();
            }
            else if (cropSize.ValueKind == JsonValueKind.Object && cropSize.TryGetProperty("height", out var ch) && cropSize.TryGetProperty("width", out var cw))
            {
                (cropHeight, cropWidth) = (ch.GetInt32(), cw.GetInt32());
            }
        }

        if (crop && (cropHeight <= 0 || cropWidth <= 0))
        {
            throw new InvalidDataException("do_center_crop without a crop_size of height and width.");
        }

        int resample = root.TryGetProperty("resample", out var r) && r.ValueKind == JsonValueKind.Number ? r.GetInt32() : (int)d.Resampling;
        if (!Enum.IsDefined((ImageResampling)resample))
        {
            throw new NotSupportedException($"resample {resample} is not supported (1 Lanczos, 2 bilinear, 3 bicubic, 4 box, 5 Hamming).");
        }

        return new ImagePreprocessor
        {
            Resize = resize,
            Height = height,
            Width = width,
            ShortestEdge = shortest,
            CenterCrop = crop,
            CropHeight = cropHeight,
            CropWidth = cropWidth,
            Resampling = (ImageResampling)resample,
            Rescale = Flag(root, "do_rescale", d.Rescale),
            RescaleFactor = root.TryGetProperty("rescale_factor", out var f) && f.ValueKind == JsonValueKind.Number ? f.GetDouble() : d.RescaleFactor,
            Normalize = Flag(root, "do_normalize", d.Normalize),
            Mean = Values(root, "image_mean") ?? d.Mean,
            Std = Values(root, "image_std") ?? d.Std,
            ConvertRgb = Flag(root, "do_convert_rgb", d.ConvertRgb),
            Grayscale = grayscale,
        };

        static bool Flag(JsonElement root, string name, bool otherwise) =>
            root.TryGetProperty(name, out var value) ? value.ValueKind switch
            {
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                _ => otherwise,                                                          // null: the processor's default
            } : otherwise;

        static float[]? Values(JsonElement root, string name) =>
            !root.TryGetProperty(name, out var value) || value.ValueKind == JsonValueKind.Null ? null
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
        var (toHeight, toWidth) = ResizedSize(h, w);
        if (Resize && (toHeight != h || toWidth != w))
        {
            for (int c = 0; c < planes.Length; c++)
            {
                planes[c] = PillowResize(planes[c], h, w, toHeight, toWidth, Resampling);
            }

            (h, w) = (toHeight, toWidth);
        }

        if (CenterCrop)
        {
            for (int c = 0; c < planes.Length; c++)
            {
                planes[c] = Crop(planes[c], h, w, CropHeight, CropWidth);
            }

            (h, w) = (CropHeight, CropWidth);
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

    /// <summary>The [height, width] of the pixel values of an image of <paramref name="height"/> x <paramref name="width"/> (after the resize and the crop).</summary>
    public (int Height, int Width) OutputSize(int height, int width)
    {
        var (h, w) = ResizedSize(height, width);
        return CenterCrop ? (CropHeight, CropWidth) : (h, w);
    }

    // The size the resize gives: Height x Width, or by the shortest edge as transformers' get_resize_output_image_size
    // (default_to_square false): the short side becomes ShortestEdge, the long one int(ShortestEdge * long / short).
    private (int Height, int Width) ResizedSize(int h, int w)
    {
        if (!Resize)
        {
            return (h, w);
        }

        if (ShortestEdge > 0)
        {
            int shortSide = Math.Min(w, h), longSide = Math.Max(w, h);
            int newLong = (int)(ShortestEdge * (double)longSide / shortSide);
            return w <= h ? (newLong, ShortestEdge) : (ShortestEdge, newLong);
        }

        if (Height <= 0 || Width <= 0)
        {
            throw new InvalidOperationException($"Resize to {Height} x {Width}: give a positive height and width (or a shortest edge).");
        }

        return (Height, Width);
    }

    // transformers' center_crop on one plane: top = (h - ch) // 2 and left = (w - cw) // 2 (rounded down); a plane smaller
    // than the crop is first put on zeros, ceil((new - old) / 2) from the top and left.
    private static byte[] Crop(byte[] plane, int h, int w, int ch, int cw)
    {
        int top = FloorHalf(h - ch), left = FloorHalf(w - cw);
        if (top >= 0 && left >= 0 && top + ch <= h && left + cw <= w)
        {
            var cut = new byte[ch * cw];
            for (int y = 0; y < ch; y++)
            {
                Array.Copy(plane, (top + y) * w + left, cut, y * cw, cw);
            }

            return cut;
        }

        int nh = Math.Max(ch, h), nw = Math.Max(cw, w), topPad = (nh - h + 1) / 2, leftPad = (nw - w + 1) / 2;
        var padded = new byte[nh * nw];
        for (int y = 0; y < h; y++)
        {
            Array.Copy(plane, y * w, padded, (y + topPad) * nw + leftPad, w);
        }

        int y0 = Math.Max(0, top + topPad), y1 = Math.Min(nh, top + topPad + ch), x0 = Math.Max(0, left + leftPad), x1 = Math.Min(nw, left + leftPad + cw);
        if (y1 - y0 != ch || x1 - x0 != cw)
        {
            throw new InvalidOperationException($"A center crop of {ch} x {cw} from {h} x {w} does not fit.");
        }

        var result = new byte[ch * cw];
        for (int y = 0; y < ch; y++)
        {
            Array.Copy(padded, (y0 + y) * nw + x0, result, y * cw, cw);
        }

        return result;

        static int FloorHalf(int v) => (int)Math.Floor(v / 2.0);
    }

    // The image as 8-bit planes (values rounded from [0, 1]), turned to one grey plane as Pillow's "L" when asked.
    internal static byte[][] ToBytes(ImageData image, bool grayscale)
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

        // The byte Pillow gives: a value that is a whole number of 255ths (8-bit sources, and those the codecs widen to 8
        // bits as Pillow does) is that byte; any other (16-bit sources) is its high byte, as Pillow reads 16-bit PNG and
        // PPM. The high byte of a 16-bit k * 257 is k, so both agree where both apply.
        static int Byte(float v)
        {
            if (v <= 0)
            {
                return 0;
            }

            if (v >= 1)
            {
                return 255;
            }

            float scaled = v * 255f;
            float nearest = MathF.Round(scaled);
            return MathF.Abs(scaled - nearest) < 1e-3f ? (int)nearest : (int)MathF.Round(v * 65535f) >> 8;
        }
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
