// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Binary;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
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
    /// Whether <see cref="Process(ImageData, Device?)"/> resizes, rescales and normalizes on a GPU in one device operation
    /// (<see cref="Backend.ResizeNormalize"/>) instead of on the host (default true). The device runs Pillow's 8-bit passes
    /// in integers with the same coefficients and the same per-channel table of the 256 byte values, so its values are the
    /// host path's bit for bit; a center crop, the CPU and <see cref="Pixels(ImageData)"/> take the host path.
    /// </summary>
    public bool ResizeOnDevice { get; init; } = true;

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
        var target = device ?? Device.Default;
        if (ResizeOnDevice && !CenterCrop && target.Type != DeviceType.Cpu)
        {
            return ProcessOnDevice(image, target);
        }

        var values = Pixels(image, out int channels, out int height, out int width);
        return Tensor.From(values, [channels, height, width], device);
    }

    /// <summary>
    /// The pixel values of several images (the views of a page, pages read together) as one [images, channels, height,
    /// width] tensor on <paramref name="device"/>, as <see cref="Process(ImageData, Device?)"/> makes each, so the same
    /// values bit for bit: on a GPU (<see cref="ResizeOnDevice"/>, no center crop) each image's 8-bit planes are uploaded
    /// (a byte a value, not four) and resized, rescaled and normalized there, one device operation for the images of a
    /// size; on the CPU, and otherwise, the host path writes them into one pooled buffer, uploaded once.
    /// </summary>
    /// <exception cref="ArgumentException">No images, or images whose values differ in shape.</exception>
    public Tensor ProcessBatch(IReadOnlyList<ImageData> images, Device? device = null)
    {
        ArgumentNullException.ThrowIfNull(images);
        if (images.Count == 0)
        {
            throw new ArgumentException("No images to process.", nameof(images));
        }

        var target = device ?? Device.Default;
        if (ResizeOnDevice && !CenterCrop && target.Type != DeviceType.Cpu)
        {
            return ProcessBatchOnDevice(images, target);
        }

        var (first, channels, height, width) = Planes(images[0]);
        int each = checked(channels * height * width);
        float[] values = ArrayPool<float>.Shared.Rent(checked(images.Count * each));   // written, copied into the tensor, given back
        try
        {
            Fill(first, channels, height * width, values.AsSpan(0, each));
            for (int i = 1; i < images.Count; i++)
            {
                var (planes, c, h, w) = Planes(images[i]);
                if (c != channels || h != height || w != width)
                {
                    throw new ArgumentException($"Image {i}'s values are {c} x {h} x {w}, the first's {channels} x {height} x {width}.", nameof(images));
                }

                Fill(planes, c, h * w, values.AsSpan(i * each, each));
            }

            return Tensor.From(values.AsSpan(0, images.Count * each), [images.Count, channels, height, width], target);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(values);
        }
    }

    /// <summary>
    /// <see cref="Process(ImageData, Device?)"/> through the device operation (Backend.ResizeNormalize, 8-bit passes) on any
    /// device, the CPU included (tests compare it with the host path); no center crop.
    /// </summary>
    internal Tensor ProcessOnDevice(ImageData image, Device device)
    {
        ArgumentNullException.ThrowIfNull(image);
        using var batch = ProcessBatchOnDevice([image], device);
        return batch.Reshape(batch.Shape[1], batch.Shape[2], batch.Shape[3]);
    }

    /// <summary>
    /// <see cref="ProcessBatch"/> through the device operation on any device, the CPU included (tests compare it with the
    /// host path): the images of one input size in one ResizeNormalize over all their planes (plane p is channel p % C),
    /// into the batch or, for the sizes after the first, into a buffer copied into their places.
    /// </summary>
    internal Tensor ProcessBatchOnDevice(IReadOnlyList<ImageData> images, Device device)
    {
        ArgumentNullException.ThrowIfNull(images);
        if (CenterCrop)
        {
            throw new InvalidOperationException("A center crop runs on the host (Pixels).");
        }

        var prepared = new (byte[][] Planes, int Channels, int Height, int Width)[images.Count];
        for (int i = 0; i < images.Count; i++)
        {
            ArgumentNullException.ThrowIfNull(images[i], nameof(images));
            var planes = ToBytes(images[i], Grayscale);
            prepared[i] = (planes, planes.Length == 1 && (ConvertRgb || Grayscale) ? 3 : planes.Length, images[i].Height, images[i].Width);
        }

        int channels = prepared[0].Channels;
        var (toHeight, toWidth) = ResizedSize(prepared[0].Height, prepared[0].Width);
        for (int i = 1; i < prepared.Length; i++)
        {
            if (prepared[i].Channels != channels || ResizedSize(prepared[i].Height, prepared[i].Width) != (toHeight, toWidth))
            {
                throw new ArgumentException($"Image {i}'s values are {prepared[i].Channels} channels of {ResizedSize(prepared[i].Height, prepared[i].Width)}, "
                    + $"the first's {channels} of {(toHeight, toWidth)}.", nameof(images));
            }
        }

        int outImage = channels * toHeight * toWidth;
        var y = Tensor.Empty([images.Count, channels, toHeight, toWidth], device);
        try
        {
            using var values = Tensor.From(Table(channels), [256 * channels], device);
            var done = new bool[prepared.Length];
            for (int i = 0; i < prepared.Length; i++)
            {
                if (done[i])
                {
                    continue;
                }

                // The images of this input size, in order.
                var (h, w) = (prepared[i].Height, prepared[i].Width);
                var group = new List<int>();
                for (int j = i; j < prepared.Length; j++)
                {
                    if (!done[j] && prepared[j].Height == h && prepared[j].Width == w)
                    {
                        group.Add(j);
                        done[j] = true;
                    }
                }

                bool inPlace = group.Count == prepared.Length;               // one size: straight into the batch
                using var x = Tensor.From(PackedBytes(prepared, group, channels, h * w), [Math.Max(1, (group.Count * channels * h * w + 3) / 4)], device);
                var (coefficients, xTaps, yTaps) = ResizeCoefficients(h, w, toHeight, toWidth);
                using var taps = Tensor.From(coefficients, [coefficients.Length], device);
                using var output = inPlace ? null : Tensor.Empty([group.Count * outImage], device);
                device.Backend.ResizeNormalize(x.Storage, taps.Storage, values.Storage, (output ?? y).Storage, group.Count * channels, channels, h, w, toHeight, toWidth,
                    xTaps, yTaps, bytes: true);
                for (int g = 0; output is not null && g < group.Count; g++)
                {
                    device.Backend.Copy2D(output.Storage, g * outImage, 0, y.Storage, group[g] * outImage, 0, 1, outImage, false);
                }
            }

            return y;
        }
        catch
        {
            y.Dispose();
            throw;
        }
    }

    // The 8-bit values of the images `group` names (each output channel's plane, a grey plane repeated), one after another,
    // four to a float word with the first in the low byte (the device operation's layout).
    private static float[] PackedBytes((byte[][] Planes, int Channels, int Height, int Width)[] images, List<int> group, int channels, int area)
    {
        int count = group.Count * channels * area;
        var words = new float[Math.Max(1, (count + 3) / 4)];
        if (BitConverter.IsLittleEndian)
        {
            var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(words.AsSpan());
            for (int g = 0; g < group.Count; g++)
            {
                var planes = images[group[g]].Planes;
                for (int c = 0; c < channels; c++)
                {
                    planes[Math.Min(c, planes.Length - 1)].AsSpan(0, area).CopyTo(bytes.Slice((g * channels + c) * area, area));
                }
            }

            return words;
        }

        // A big-endian host: the same bytes placed word by word.
        var packed = new uint[words.Length];
        for (int g = 0; g < group.Count; g++)
        {
            var planes = images[group[g]].Planes;
            for (int c = 0; c < channels; c++)
            {
                var plane = planes[Math.Min(c, planes.Length - 1)];
                for (int i = 0, at = (g * channels + c) * area; i < area; i++, at++)
                {
                    packed[at >> 2] |= (uint)plane[i] << (8 * (at & 3));
                }
            }
        }

        for (int i = 0; i < packed.Length; i++)
        {
            words[i] = BitConverter.UInt32BitsToSingle(packed[i]);
        }

        return words;
    }

    // Per output column (then row): its first input pixel, its tap count and its weights, as int bits; none for a size kept
    // (one zero when both are kept). Written straight into the array the device reads.
    private (float[] Coefficients, int XTaps, int YTaps) ResizeCoefficients(int h, int w, int toHeight, int toWidth)
    {
        var columns = Resize && w != toWidth ? Coefficients(w, toWidth, Resampling) : default;
        var rows = Resize && h != toHeight ? Coefficients(h, toHeight, Resampling) : default;
        int xTaps = columns.Weights is null ? 0 : columns.Taps, yTaps = rows.Weights is null ? 0 : rows.Taps;
        int length = (xTaps == 0 ? 0 : toWidth * (2 + xTaps)) + (yTaps == 0 ? 0 : toHeight * (2 + yTaps));
        var result = new float[Math.Max(1, length)];
        int at = 0;
        void Write((int[] Bounds, int[] Weights, int Taps) pass, int outSize)
        {
            for (int o = 0; o < outSize; o++)
            {
                result[at++] = BitConverter.Int32BitsToSingle(pass.Bounds[2 * o]);
                result[at++] = BitConverter.Int32BitsToSingle(pass.Bounds[2 * o + 1]);
                for (int t = 0; t < pass.Taps; t++)
                {
                    result[at++] = BitConverter.Int32BitsToSingle(pass.Weights[o * pass.Taps + t]);
                }
            }
        }

        if (xTaps > 0)
        {
            Write(columns, toWidth);
        }

        if (yTaps > 0)
        {
            Write(rows, toHeight);
        }

        return (result, xTaps, yTaps);
    }

    // Each channel's value of each byte (256 per channel): rescaled as transformers does (the byte times the factor in
    // double precision, rounded to float) and normalized ((value - mean) / std in float).
    private float[] Table(int channels)
    {
        if (Normalize && (Mean.Count is not 1 && Mean.Count != channels || Std.Count is not 1 && Std.Count != channels))
        {
            throw new InvalidOperationException($"{Mean.Count} means and {Std.Count} standard deviations for {channels} channels.");
        }

        var table = new float[256 * channels];
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

                table[256 * c + v] = value;
            }
        }

        return table;
    }

    /// <summary>The pixel values of an image file (any registered codec) as a tensor, as <see cref="Process(ImageData, Device?)"/>.</summary>
    public Tensor Process(string path, Device? device = null) => Process(ImageCodecs.Decode(path), device);

    /// <summary>
    /// The pixel values of <paramref name="image"/> as <see cref="Pixels(ImageData)"/> gives them, written into
    /// <paramref name="destination"/> ([channels, height, width] from its start) instead of a new array: an encoder fills a
    /// pooled buffer of every view and uploads it, with nothing left on the managed heap. Returns the count written.
    /// </summary>
    /// <exception cref="ArgumentException"><paramref name="destination"/> is shorter than the image's pixel values.</exception>
    public int Pixels(ImageData image, Span<float> destination)
    {
        var (planes, channels, h, w) = Planes(image);
        int count = checked(channels * h * w);
        if (destination.Length < count)
        {
            throw new ArgumentException($"The image's {channels} x {h} x {w} pixel values do not fit {destination.Length} floats.", nameof(destination));
        }

        Fill(planes, channels, h * w, destination);
        return count;
    }

    private float[] Pixels(ImageData image, out int channels, out int height, out int width)
    {
        var (planes, c, h, w) = Planes(image);
        (channels, height, width) = (c, h, w);
        var result = new float[checked(c * h * w)];
        Fill(planes, c, h * w, result);
        return result;
    }

    // The image's 8-bit planes after the resize and the crop, with the channels its values take (a grey plane read as
    // three when the model takes RGB).
    private (byte[][] Planes, int Channels, int Height, int Width) Planes(ImageData image)
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

        return (planes, planes.Length == 1 && (ConvertRgb || Grayscale) ? 3 : planes.Length, h, w);
    }

    // Each channel's plane through its rescale-and-normalize table into `destination` ([channels, size]).
    private void Fill(byte[][] planes, int channels, int size, Span<float> destination)
    {
        var table = Table(channels);
        for (int c = 0; c < channels; c++)
        {
            var plane = planes[Math.Min(c, planes.Length - 1)];
            var target = destination.Slice(c * size, size);
            var values = table.AsSpan(256 * c, 256);
            for (int i = 0; i < plane.Length; i++)
            {
                target[i] = values[plane[i]];
            }
        }
    }

    /// <summary>The [height, width] of the pixel values of an image of <paramref name="height"/> x <paramref name="width"/> (after the resize and the crop).</summary>
    public (int Height, int Width) OutputSize(int height, int width)
    {
        var (h, w) = ResizedSize(height, width);
        return CenterCrop ? (CropHeight, CropWidth) : (h, w);
    }

    /// <summary>
    /// The [height, width] the resize gives an image of <paramref name="h"/> x <paramref name="w"/>, before any crop:
    /// <see cref="Height"/> x <see cref="Width"/>, or by the shortest edge as transformers' <c>get_resize_output_image_size</c>
    /// (default_to_square false: the short side becomes <see cref="ShortestEdge"/>, the long one int(ShortestEdge · long / short)),
    /// or the image's own size without <see cref="Resize"/>. With <see cref="OutputSize"/> it maps the network's pixels back
    /// to the image's (a detector's boxes).
    /// </summary>
    public (int Height, int Width) ResizedSize(int h, int w)
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
        if (image.TryGetPlanes(out var held, out _))
        {
            // An 8-bit image: its planes are the bytes the floats would round to (copied: steps may change them in place).
            if (grayscale && c == 3)
            {
                var (red, green, blue) = (held[0], held[1], held[2]);
                var grey = GC.AllocateUninitializedArray<byte>(size);
                HostParallel.For(size, 1 << 15, (first, last) =>
                {
                    ReadOnlySpan<byte> r = red.Span, g = green.Span, b = blue.Span;
                    for (int i = first; i < last; i++)
                    {
                        grey[i] = (byte)((r[i] * 19595 + g[i] * 38470 + b[i] * 7471 + 0x8000) >> 16);
                    }
                });
                return [grey];
            }

            return [.. held.Select(p => p.ToArray())];
        }

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
        var (yBounds, yWeights, yTaps) = Coefficients(height, outHeight, filter);           // shared: read, never written
        byte[] current = source;
        int rows = height, first = 0;                                                       // the rows the vertical pass reads start at `first`
        if (outWidth != width)
        {
            first = yBounds[0];
            int last = yBounds[2 * (outHeight - 1)] + yBounds[2 * (outHeight - 1) + 1];
            rows = last - first;
            var across = new byte[checked(rows * outWidth)];

            // Strips of eight rows through vectors where they are accelerated; the rows short of a strip (and every row
            // without vectors) one at a time.
            int stripped = Vector128.IsHardwareAccelerated ? rows / StripRows * StripRows : 0;
            HostParallel.For(stripped / StripRows, 1, (s0, s1) =>
            {
                int[] interleaved = ArrayPool<int>.Shared.Rent(width * StripRows);
                byte[] columns = ArrayPool<byte>.Shared.Rent((outWidth + 3) / 4 * 4 * StripRows);
                try
                {
                    for (int s = s0; s < s1; s++)
                    {
                        HorizontalStrip(source.AsSpan((first + s * StripRows) * width, StripRows * width), width,
                            across.AsSpan(s * StripRows * outWidth, StripRows * outWidth), outWidth, xBounds, xWeights, xTaps, interleaved, columns);
                    }
                }
                finally
                {
                    ArrayPool<int>.Shared.Return(interleaved);
                    ArrayPool<byte>.Shared.Return(columns);
                }
            });
            HostParallel.For(rows - stripped, 8, (r0, r1) =>
            {
                for (int y = stripped + r0; y < stripped + r1; y++)
                {
                    HorizontalRow(source.AsSpan((first + y) * width, width), across.AsSpan(y * outWidth, outWidth), xBounds, xWeights, xTaps);
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
            for (int y = r0; y < r1; y++)
            {
                int start = yBounds[2 * y] - first, count = yBounds[2 * y + 1];
                VerticalRow(plane, outWidth, start, yWeights.AsSpan(y * yTaps, count), result.AsSpan(y * outWidth, outWidth));
            }
        });
        return result;
    }

    // The rows a strip of the horizontal pass holds: one 8 x 8 byte transpose wide, and one Vector256<int> (or two
    // Vector128<int>) of sums per output column.
    private const int StripRows = 8;

    // One row of the horizontal pass, pixel by pixel (Resample.c's loop): the rows short of a strip, or every row when
    // vectors are not accelerated.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void HorizontalRow(ReadOnlySpan<byte> input, Span<byte> output, int[] bounds, int[] weights, int taps)
    {
        for (int x = 0; x < output.Length; x++)
        {
            int start = bounds[2 * x], count = bounds[2 * x + 1];
            var k = weights.AsSpan(x * taps, count);
            int sum = 1 << (PrecisionBits - 1);
            for (int i = 0; i < k.Length; i++)
            {
                sum += input[start + i] * k[i];
            }

            output[x] = Clip8(sum);
        }
    }

    // Eight rows of the horizontal pass at once, each vector lane its own row, so a tap is one load and one multiply for
    // eight pixels and no lane is summed across: the rows are interleaved once (`interleaved[x * 8 + r]` is row r's byte
    // x, eight columns at a time through a byte transpose), each output column's eight sums are clipped and narrowed to
    // bytes (`columns[x * 8 + r]`), and those are transposed back to the eight output rows. Each lane holds the same sum
    // as HorizontalRow's (integer sums are order-free), so the bytes are the same.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void HorizontalStrip(ReadOnlySpan<byte> rows, int width, Span<byte> output, int outWidth, int[] bounds, int[] weights, int taps,
        int[] interleaved, byte[] columns)
    {
        const int r = StripRows;
        int x = 0;
        for (; x + r <= width; x += r)
        {
            ulong a0 = Read(rows, x), a1 = Read(rows, width + x), a2 = Read(rows, 2 * width + x), a3 = Read(rows, 3 * width + x),
                a4 = Read(rows, 4 * width + x), a5 = Read(rows, 5 * width + x), a6 = Read(rows, 6 * width + x), a7 = Read(rows, 7 * width + x);
            Transpose8(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7);
            var to = interleaved.AsSpan(x * r, r * r);
            Widen(a0, to);
            Widen(a1, to[r..]);
            Widen(a2, to[(2 * r)..]);
            Widen(a3, to[(3 * r)..]);
            Widen(a4, to[(4 * r)..]);
            Widen(a5, to[(5 * r)..]);
            Widen(a6, to[(6 * r)..]);
            Widen(a7, to[(7 * r)..]);
        }

        for (; x < width; x++)
        {
            for (int row = 0; row < r; row++)
            {
                interleaved[x * r + row] = rows[row * width + x];
            }
        }

        // Every column's taps lie within the row (Resample.c's bounds), so the unchecked loads below stay in `interleaved`.
        ref int samples = ref MemoryMarshal.GetArrayDataReference(interleaved);
        ref int k0 = ref MemoryMarshal.GetArrayDataReference(weights);
        if (Vector256.IsHardwareAccelerated)
        {
            var half = Vector256.Create(1 << (PrecisionBits - 1));
            for (x = 0; x < outWidth; x += 4)
            {
                var c0 = Column256(ref samples, ref k0, bounds, taps, x, outWidth, half);
                var c1 = Column256(ref samples, ref k0, bounds, taps, x + 1, outWidth, half);
                var c2 = Column256(ref samples, ref k0, bounds, taps, x + 2, outWidth, half);
                var c3 = Column256(ref samples, ref k0, bounds, taps, x + 3, outWidth, half);
                Vector256.Narrow(Vector256.Narrow(c0, c1), Vector256.Narrow(c2, c3)).CopyTo(columns.AsSpan(x * r));
            }
        }
        else
        {
            var half = Vector128.Create(1 << (PrecisionBits - 1));
            for (x = 0; x < outWidth; x += 2)
            {
                var (l0, h0) = Column128(ref samples, ref k0, bounds, taps, x, outWidth, half);
                var (l1, h1) = Column128(ref samples, ref k0, bounds, taps, x + 1, outWidth, half);
                Vector128.Narrow(Vector128.Narrow(l0, h0), Vector128.Narrow(l1, h1)).CopyTo(columns.AsSpan(x * r));
            }
        }

        for (x = 0; x + r <= outWidth; x += r)
        {
            var from = columns.AsSpan(x * r, r * r);
            ulong a0 = Read(from, 0), a1 = Read(from, r), a2 = Read(from, 2 * r), a3 = Read(from, 3 * r),
                a4 = Read(from, 4 * r), a5 = Read(from, 5 * r), a6 = Read(from, 6 * r), a7 = Read(from, 7 * r);
            Transpose8(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7);
            BinaryPrimitives.WriteUInt64LittleEndian(output[x..], a0);
            BinaryPrimitives.WriteUInt64LittleEndian(output[(outWidth + x)..], a1);
            BinaryPrimitives.WriteUInt64LittleEndian(output[(2 * outWidth + x)..], a2);
            BinaryPrimitives.WriteUInt64LittleEndian(output[(3 * outWidth + x)..], a3);
            BinaryPrimitives.WriteUInt64LittleEndian(output[(4 * outWidth + x)..], a4);
            BinaryPrimitives.WriteUInt64LittleEndian(output[(5 * outWidth + x)..], a5);
            BinaryPrimitives.WriteUInt64LittleEndian(output[(6 * outWidth + x)..], a6);
            BinaryPrimitives.WriteUInt64LittleEndian(output[(7 * outWidth + x)..], a7);
        }

        for (; x < outWidth; x++)
        {
            for (int row = 0; row < r; row++)
            {
                output[row * outWidth + x] = columns[x * r + row];
            }
        }

        static ulong Read(ReadOnlySpan<byte> bytes, int at) => BinaryPrimitives.ReadUInt64LittleEndian(bytes[at..]);

        // Eight bytes (rows 0 to 7 of one column) to eight ints.
        static void Widen(ulong column, Span<int> to)
        {
            var (low, high) = Vector128.Widen(Vector128.WidenLower(Vector128.CreateScalar(column).AsByte()));
            low.AsInt32().CopyTo(to);
            high.AsInt32().CopyTo(to[4..]);
        }
    }

    // One output column of a strip: its eight row sums, clipped (Clip8 on every lane); zeros past the last column. Never
    // tier-0 code (a loop keeps the JIT from inlining it on its own; unoptimized, the first strips ran several times slower).
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static Vector256<uint> Column256(ref int samples, ref int k0, int[] bounds, int taps, int x, int outWidth, Vector256<int> half)
    {
        if (x >= outWidth)
        {
            return Vector256<uint>.Zero;
        }

        int start = bounds[2 * x], count = bounds[2 * x + 1];
        ref int at = ref Unsafe.Add(ref samples, start * StripRows);
        ref int k = ref Unsafe.Add(ref k0, x * taps);
        var sums = half;
        for (int i = 0; i < count; i++)
        {
            sums += Vector256.LoadUnsafe(ref Unsafe.Add(ref at, i * StripRows)) * Vector256.Create(Unsafe.Add(ref k, i));
        }

        return Vector256.Min(Vector256.Max(Vector256.ShiftRightArithmetic(sums, PrecisionBits), Vector256<int>.Zero), Vector256.Create(255)).AsUInt32();
    }

    // Column256 as two halves of four rows, where only 128-bit vectors are accelerated.
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static (Vector128<uint> Low, Vector128<uint> High) Column128(ref int samples, ref int k0, int[] bounds, int taps, int x, int outWidth,
        Vector128<int> half)
    {
        if (x >= outWidth)
        {
            return (Vector128<uint>.Zero, Vector128<uint>.Zero);
        }

        int start = bounds[2 * x], count = bounds[2 * x + 1];
        ref int at = ref Unsafe.Add(ref samples, start * StripRows);
        ref int k = ref Unsafe.Add(ref k0, x * taps);
        Vector128<int> low = half, high = half;
        for (int i = 0; i < count; i++)
        {
            var w = Vector128.Create(Unsafe.Add(ref k, i));
            low += Vector128.LoadUnsafe(ref Unsafe.Add(ref at, i * StripRows)) * w;
            high += Vector128.LoadUnsafe(ref Unsafe.Add(ref at, i * StripRows + 4)) * w;
        }

        var top = Vector128.Create(255);
        return (Vector128.Min(Vector128.Max(Vector128.ShiftRightArithmetic(low, PrecisionBits), Vector128<int>.Zero), top).AsUInt32(),
                Vector128.Min(Vector128.Max(Vector128.ShiftRightArithmetic(high, PrecisionBits), Vector128<int>.Zero), top).AsUInt32());
    }

    // Eight rows of eight bytes (each read little-endian, so byte j is column j on every platform) transposed in place:
    // afterwards a[j] holds column j, its byte r from row r. Three rounds of swaps: bytes, pairs, quads.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Transpose8(ref ulong a0, ref ulong a1, ref ulong a2, ref ulong a3, ref ulong a4, ref ulong a5, ref ulong a6, ref ulong a7)
    {
        Swap(ref a0, ref a1, 8, 0x00FF00FF00FF00FF);
        Swap(ref a2, ref a3, 8, 0x00FF00FF00FF00FF);
        Swap(ref a4, ref a5, 8, 0x00FF00FF00FF00FF);
        Swap(ref a6, ref a7, 8, 0x00FF00FF00FF00FF);
        Swap(ref a0, ref a2, 16, 0x0000FFFF0000FFFF);
        Swap(ref a1, ref a3, 16, 0x0000FFFF0000FFFF);
        Swap(ref a4, ref a6, 16, 0x0000FFFF0000FFFF);
        Swap(ref a5, ref a7, 16, 0x0000FFFF0000FFFF);
        Swap(ref a0, ref a4, 32, 0x00000000FFFFFFFF);
        Swap(ref a1, ref a5, 32, 0x00000000FFFFFFFF);
        Swap(ref a2, ref a6, 32, 0x00000000FFFFFFFF);
        Swap(ref a3, ref a7, 32, 0x00000000FFFFFFFF);

        // The high parts of `low` trade places with the low parts of `high`.
        static void Swap(ref ulong low, ref ulong high, int shift, ulong mask)
        {
            ulong t = ((low >> shift) ^ high) & mask;
            high ^= t;
            low ^= t << shift;
        }
    }

    // One output row of the vertical pass: output[x] = Clip8(sum over taps of row (start + i)'s byte x times weight i). A
    // vector of columns at a time (bytes widened to 32-bit lanes) with every tap added into sums held in registers and the
    // clip fused into the store, then a scalar tail. Integer sums wrap alike in either, so the order does not matter.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void VerticalRow(byte[] plane, int width, int start, ReadOnlySpan<int> weights, Span<byte> output)
    {
        int x = 0, n = Vector<byte>.Count;
        if (Vector.IsHardwareAccelerated && width >= n)
        {
            var half = new Vector<int>(1 << (PrecisionBits - 1));
            var top = new Vector<int>(255);
            for (; x <= width - n; x += n)
            {
                Vector<int> a = half, b = half, c = half, d = half;
                for (int i = 0; i < weights.Length; i++)
                {
                    var w = new Vector<int>(weights[i]);
                    Vector.Widen(new Vector<byte>(plane.AsSpan((start + i) * width + x, n)), out var low, out var high);
                    Vector.Widen(low, out var la, out var lb);
                    Vector.Widen(high, out var ha, out var hb);
                    a += Vector.AsVectorInt32(la) * w;
                    b += Vector.AsVectorInt32(lb) * w;
                    c += Vector.AsVectorInt32(ha) * w;
                    d += Vector.AsVectorInt32(hb) * w;
                }

                // Clip8 on every lane: value >> 22 held to [0, 255] (at or above 1 << 30 it is 255, at or below 0 it is 0).
                var low16 = Vector.Narrow(Clip(a, top), Clip(b, top));
                var high16 = Vector.Narrow(Clip(c, top), Clip(d, top));
                Vector.Narrow(low16, high16).CopyTo(output[x..]);
            }
        }

        for (; x < width; x++)
        {
            int sum = 1 << (PrecisionBits - 1);
            for (int i = 0; i < weights.Length; i++)
            {
                sum += plane[(start + i) * width + x] * weights[i];
            }

            output[x] = Clip8(sum);
        }

        static Vector<uint> Clip(Vector<int> sums, Vector<int> top) =>
            Vector.AsVectorUInt32(Vector.Min(Vector.Max(Vector.ShiftRightArithmetic(sums, PrecisionBits), Vector<int>.Zero), top));
    }

    private static byte Clip8(int value) => value >= 1 << PrecisionBits << 8 ? (byte)255 : value <= 0 ? (byte)0 : (byte)(value >> PrecisionBits);

    // The coefficients of the sizes resized last: a resize asks for them per plane and per call (a sine per tap for Lanczos
    // and Hamming), and a page's planes, a batch's pages and a pipeline's repeated steps ask for the same sizes again. A few
    // entries, replaced in turn; an entry is never changed once made, so readers on other threads need no lock.
    private sealed record ResizeTaps(int InSize, int OutSize, ImageResampling Filter, int[] Bounds, int[] Weights, int Taps);

    private static readonly ResizeTaps?[] TapCache = new ResizeTaps?[16];
    private static int tapCacheNext = -1;

    // Resample.c's coefficients for one direction (see ComputeCoefficients), from the cache when the sizes were seen. The
    // arrays are shared: callers read them and never write to them.
    private static (int[] Bounds, int[] Weights, int Taps) Coefficients(int inSize, int outSize, ImageResampling filter)
    {
        for (int i = 0; i < TapCache.Length; i++)
        {
            var entry = Volatile.Read(ref TapCache[i]);
            if (entry is not null && entry.InSize == inSize && entry.OutSize == outSize && entry.Filter == filter)
            {
                return (entry.Bounds, entry.Weights, entry.Taps);
            }
        }

        var (bounds, weights, taps) = ComputeCoefficients(inSize, outSize, filter);
        int slot = (int)((uint)Interlocked.Increment(ref tapCacheNext) % (uint)TapCache.Length);
        Volatile.Write(ref TapCache[slot], new ResizeTaps(inSize, outSize, filter, bounds, weights, taps));
        return (bounds, weights, taps);
    }

    // Resample.c precompute_coeffs and normalize_coeffs_8bpc: for each output pixel, the first input pixel and the count
    // it reads (bounds, in pairs) and its weights (`taps` per pixel), as 22-bit fixed point rounded away from zero.
    private static (int[] Bounds, int[] Weights, int Taps) ComputeCoefficients(int inSize, int outSize, ImageResampling filter)
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
