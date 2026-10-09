// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Vision;

// The objects kept after the pixels moved, shared by the geometric augmentations.
internal static class MovedObjects
{
    /// <summary>
    /// The sample on <paramref name="image"/> with each object's moved box clipped to it, kept when at least
    /// <paramref name="minVisibility"/> of the moved box's area is left (something, at 0) and, with masks, its mask is not
    /// empty; with <paramref name="boxesFromMasks"/> a kept object's box is its mask's (a rotated box's corners enclose
    /// more than the object).
    /// </summary>
    public static AnnotatedImage Keep(ImageData image, IReadOnlyList<BoundingBox> moved, IReadOnlyList<int> labels, byte[][]? masks, int[]? classes,
        double minVisibility, bool boxesFromMasks = false)
    {
        int w = image.Width, h = image.Height;
        var kept = new List<int>();
        var boxes = new List<BoundingBox>();
        for (int i = 0; i < moved.Count; i++)
        {
            var clipped = moved[i].Clip(w, h);
            if (!(clipped.Width > 0 && clipped.Height > 0) || clipped.Area < minVisibility * moved[i].Area)
            {
                continue;
            }

            if (masks is not null)
            {
                var bounds = ObjectMask.PixelBounds(masks[i], w, h);
                if (bounds.Width <= 0)
                {
                    continue;
                }

                if (boxesFromMasks)
                {
                    clipped = bounds;
                }
            }

            kept.Add(i);
            boxes.Add(clipped);
        }

        return new AnnotatedImage(image, [.. boxes], [.. kept.Select(i => labels[i])], masks is null ? null : [.. kept.Select(i => masks[i])], classes);
    }

    public static double Uniform(Random random, (double Low, double High) range) => range.Low + random.NextDouble() * (range.High - range.Low);

    // A draw from the standard normal distribution (Box-Muller).
    public static double Normal(Random random) => Math.Sqrt(-2 * Math.Log(1 - random.NextDouble())) * Math.Cos(2 * Math.PI * random.NextDouble());

    // A draw from Gamma(shape, 1) (Marsaglia and Tsang 2000; shape below 1 by the boost u^(1/shape)).
    public static double Gamma(Random random, double shape)
    {
        if (shape < 1)
        {
            return Gamma(random, shape + 1) * Math.Pow(1 - random.NextDouble(), 1 / shape);
        }

        double d = shape - 1.0 / 3, c = 1 / Math.Sqrt(9 * d);
        while (true)
        {
            double x = Normal(random), v = 1 + c * x;
            if (v <= 0)
            {
                continue;
            }

            v = v * v * v;
            double u = 1 - random.NextDouble();
            if (Math.Log(u) < 0.5 * x * x + d - d * v + d * Math.Log(v))
            {
                return d * v;
            }
        }
    }

    // A draw from Beta(a, b).
    public static double Beta(Random random, double a, double b)
    {
        double x = Gamma(random, a), y = Gamma(random, b);
        return x / (x + y);
    }
}

/// <summary>
/// Resizes samples to <see cref="Width"/> x <see cref="Height"/> ("resize" in <see cref="Augmentations"/>, options
/// <c>width</c>, <c>height</c>, <c>keep_ratio</c>, <c>fill</c>), stretching them or, with <see cref="KeepRatio"/>,
/// scaling them to fit and padding the rest evenly (a letterbox). Deterministic.
/// </summary>
/// <param name="width">The output width.</param>
/// <param name="height">The output height.</param>
/// <param name="keepRatio">Keep the aspect ratio and pad (with <paramref name="fill"/>; masks with 0, pixel classes with 0).</param>
/// <param name="fill">The value of padded pixels.</param>
public sealed class SampleResize(int width, int height, bool keepRatio = false, float fill = 0.5f) : IAugmentation
{
    /// <summary>The output width.</summary>
    public int Width { get; } = width > 0 ? width : throw new ArgumentOutOfRangeException(nameof(width));

    /// <summary>The output height.</summary>
    public int Height { get; } = height > 0 ? height : throw new ArgumentOutOfRangeException(nameof(height));

    /// <summary>Whether the aspect ratio is kept (a letterbox).</summary>
    public bool KeepRatio { get; } = keepRatio;

    /// <summary>The value of padded pixels.</summary>
    public float Fill { get; } = fill;

    /// <inheritdoc />
    public string Name => "resize";

    /// <inheritdoc />
    public AnnotatedImage Apply(AnnotatedImage sample, AugmentationContext context)
    {
        ArgumentNullException.ThrowIfNull(sample);
        int w = sample.Width, h = sample.Height;
        if (w == Width && h == Height)
        {
            return sample;
        }

        // The source region that maps onto the output: the image itself, or (letterboxed) the image with its padding.
        double sx = (double)Width / w, sy = (double)Height / h;
        if (KeepRatio)
        {
            sx = sy = Math.Min(sx, sy);
        }

        double regionWidth = Width / sx, regionHeight = Height / sy;
        double x0 = -(regionWidth - w) / 2, y0 = -(regionHeight - h) / 2;
        return SampleRegion.Resampled(sample, x0, y0, regionWidth, regionHeight, Width, Height, Fill, 0);
    }
}

// A sample's region [x0, x0 + width) x [y0, y0 + height) resampled to an output size, objects moved with it.
internal static class SampleRegion
{
    public static AnnotatedImage Resampled(AnnotatedImage sample, double x0, double y0, double width, double height, int outWidth, int outHeight, float fill, double minVisibility)
    {
        int w = sample.Width, h = sample.Height;
        var pixels = ImageWarp.Resample(sample.Image, x0, y0, width, height, outWidth, outHeight, sample.Image.Channels, fill);
        double sx = outWidth / width, sy = outHeight / height;
        var moved = sample.Boxes.Select(b => new BoundingBox((float)((b.X - x0) * sx), (float)((b.Y - y0) * sy), (float)(b.Width * sx), (float)(b.Height * sy))).ToList();
        byte[][]? masks = sample.Masks?.Select(m => ImageWarp.ResampleNearest(m, w, h, x0, y0, width, height, outWidth, outHeight, (byte)0)).ToArray();
        int[]? classes = sample.PixelClasses is { } map ? ImageWarp.ResampleNearest(map, w, h, x0, y0, width, height, outWidth, outHeight, 0) : null;
        return MovedObjects.Keep(new ImageData(pixels, sample.Image.Channels, outHeight, outWidth), moved, sample.Labels, masks, classes, minVisibility);
    }
}

/// <summary>
/// Crops a random part of each sample and resizes it to the output size ("resized-crop" in <see cref="Augmentations"/>,
/// options <c>width</c>, <c>height</c>, <c>scale</c>, <c>ratio</c>, <c>min_visibility</c>), choosing the part as
/// torchvision's <c>RandomResizedCrop</c>: an area share in <see cref="Scale"/> and an aspect ratio in
/// <see cref="Ratio"/> (log-uniform), ten tries, then a centre crop. Boxes are cut to the crop; those with less than
/// <see cref="MinVisibility"/> of their area left are dropped.
/// </summary>
/// <param name="width">The output width (0: the sample's).</param>
/// <param name="height">The output height (0: the sample's).</param>
/// <param name="scale">The crop's share of the image's area.</param>
/// <param name="ratio">The crop's width over height.</param>
/// <param name="minVisibility">The share of a box's area that must stay in the crop for the box to be kept (0: any part).</param>
public sealed class RandomResizedCrop(int width = 0, int height = 0, (double Low, double High)? scale = null, (double Low, double High)? ratio = null, double minVisibility = 0)
    : IAugmentation
{
    /// <summary>The output width (0: the sample's).</summary>
    public int Width { get; } = width >= 0 ? width : throw new ArgumentOutOfRangeException(nameof(width));

    /// <summary>The output height (0: the sample's).</summary>
    public int Height { get; } = height >= 0 ? height : throw new ArgumentOutOfRangeException(nameof(height));

    /// <summary>The crop's share of the image's area.</summary>
    public (double Low, double High) Scale { get; } = scale is { Low: > 0, High: <= 1 } s && s.Low <= s.High ? s : scale is null ? (0.08, 1.0)
        : throw new ArgumentOutOfRangeException(nameof(scale), "A share of the area in (0, 1].");

    /// <summary>The crop's width over height.</summary>
    public (double Low, double High) Ratio { get; } = ratio is { Low: > 0 } r && r.Low <= r.High ? r : ratio is null ? (3.0 / 4, 4.0 / 3)
        : throw new ArgumentOutOfRangeException(nameof(ratio), "A positive range of width over height.");

    /// <summary>The share of a box's area that must stay in the crop.</summary>
    public double MinVisibility { get; } = minVisibility is >= 0 and <= 1 ? minVisibility : throw new ArgumentOutOfRangeException(nameof(minVisibility));

    /// <inheritdoc />
    public string Name => "resized-crop";

    /// <inheritdoc />
    public AnnotatedImage Apply(AnnotatedImage sample, AugmentationContext context)
    {
        ArgumentNullException.ThrowIfNull(sample);
        ArgumentNullException.ThrowIfNull(context);
        var (x, y, w, h) = Choose(sample.Width, sample.Height, context.Random);
        return SampleRegion.Resampled(sample, x, y, w, h, Width > 0 ? Width : sample.Width, Height > 0 ? Height : sample.Height, 0f, MinVisibility);
    }

    // torchvision's RandomResizedCrop.get_params: (left, top, width, height).
    private (int X, int Y, int W, int H) Choose(int width, int height, Random random)
    {
        double area = (double)width * height;
        var logRatio = (Math.Log(Ratio.Low), Math.Log(Ratio.High));
        for (int attempt = 0; attempt < 10; attempt++)
        {
            double target = area * MovedObjects.Uniform(random, Scale), aspect = Math.Exp(MovedObjects.Uniform(random, logRatio));
            int w = (int)Math.Round(Math.Sqrt(target * aspect)), h = (int)Math.Round(Math.Sqrt(target / aspect));
            if (w > 0 && w <= width && h > 0 && h <= height)
            {
                int top = random.Next(0, height - h + 1), left = random.Next(0, width - w + 1);
                return (left, top, w, h);
            }
        }

        double inRatio = (double)width / height;
        var (cw, ch) = inRatio < Ratio.Low ? (width, (int)Math.Round(width / Ratio.Low)) : inRatio > Ratio.High ? ((int)Math.Round(height * Ratio.High), height) : (width, height);
        return ((width - cw) / 2, (height - ch) / 2, cw, ch);
    }
}

/// <summary>
/// Moves each sample by a random affine map about its centre ("affine" in <see cref="Augmentations"/>, options
/// <c>degrees</c>, <c>translate</c> (or <c>translate_x</c> and <c>translate_y</c>), <c>scale</c>, <c>shear</c>, <c>fill</c>, <c>p</c>, <c>min_visibility</c>; "rotation"
/// is the same with <c>degrees</c> only): a rotation in <see cref="Degrees"/> (counter-clockwise for positive angles), a
/// shift up to <see cref="Translate"/> of the size (whole pixels), a scale in <see cref="Scale"/> and a horizontal shear
/// in <see cref="Shear"/> degrees, as torchvision's <c>RandomAffine</c> draws them. The image is sampled bilinearly,
/// masks and pixel classes by the nearest pixel; a box becomes the box of its mask when the sample has masks, else the
/// box around its moved corners.
/// </summary>
/// <param name="degrees">The rotation's range in degrees.</param>
/// <param name="translate">The largest shift as a share of the width and of the height.</param>
/// <param name="scale">The scale's range.</param>
/// <param name="shear">The horizontal shear's range in degrees.</param>
/// <param name="fill">The value of pixels from outside the image.</param>
/// <param name="probability">The chance the map is applied.</param>
/// <param name="minVisibility">The share of a moved box's area that must stay in the image for the box to be kept (0: any part).</param>
public sealed class RandomAffine((double Low, double High) degrees, (double X, double Y) translate = default, (double Low, double High)? scale = null,
    (double Low, double High) shear = default, float fill = 0f, double probability = 1, double minVisibility = 0) : IAugmentation
{
    /// <summary>The rotation's range in degrees.</summary>
    public (double Low, double High) Degrees { get; } = degrees.Low <= degrees.High ? degrees : throw new ArgumentOutOfRangeException(nameof(degrees));

    /// <summary>The largest shift as a share of the width and of the height.</summary>
    public (double X, double Y) Translate { get; } = translate is { X: >= 0 and <= 1, Y: >= 0 and <= 1 } ? translate : throw new ArgumentOutOfRangeException(nameof(translate));

    /// <summary>The scale's range.</summary>
    public (double Low, double High) Scale { get; } = scale is { Low: > 0 } s && s.Low <= s.High ? s : scale is null ? (1, 1) : throw new ArgumentOutOfRangeException(nameof(scale));

    /// <summary>The horizontal shear's range in degrees.</summary>
    public (double Low, double High) Shear { get; } = shear.Low <= shear.High && shear.Low > -90 && shear.High < 90 ? shear : throw new ArgumentOutOfRangeException(nameof(shear));

    /// <summary>The value of pixels from outside the image.</summary>
    public float Fill { get; } = fill;

    /// <summary>The chance the map is applied.</summary>
    public double Probability { get; } = probability is >= 0 and <= 1 ? probability : throw new ArgumentOutOfRangeException(nameof(probability));

    /// <summary>The share of a moved box's area that must stay in the image.</summary>
    public double MinVisibility { get; } = minVisibility is >= 0 and <= 1 ? minVisibility : throw new ArgumentOutOfRangeException(nameof(minVisibility));

    /// <inheritdoc />
    public string Name { get; private init; } = "affine";

    /// <summary>A rotation only, by an angle in [-<paramref name="degrees"/>, <paramref name="degrees"/>] ("rotation" in <see cref="Augmentations"/>).</summary>
    public static RandomAffine Rotation(double degrees, float fill = 0f, double probability = 1, double minVisibility = 0) =>
        new((-Math.Abs(degrees), Math.Abs(degrees)), fill: fill, probability: probability, minVisibility: minVisibility) { Name = "rotation" };

    /// <inheritdoc />
    public AnnotatedImage Apply(AnnotatedImage sample, AugmentationContext context)
    {
        ArgumentNullException.ThrowIfNull(sample);
        ArgumentNullException.ThrowIfNull(context);
        var random = context.Random;
        if (random.NextDouble() >= Probability)
        {
            return sample;
        }

        int w = sample.Width, h = sample.Height;
        double angle = MovedObjects.Uniform(random, Degrees) * Math.PI / 180;
        double tx = Math.Round(MovedObjects.Uniform(random, (-Translate.X * w, Translate.X * w)));
        double ty = Math.Round(MovedObjects.Uniform(random, (-Translate.Y * h, Translate.Y * h)));
        double s = MovedObjects.Uniform(random, Scale), shear = MovedObjects.Uniform(random, Shear) * Math.PI / 180;
        var forward = Map(w, h, angle, tx, ty, s, shear);
        var pixels = ImageWarp.Affine(sample.Image, forward, w, h, Fill);
        var moved = sample.Boxes.Select(b => ImageWarp.Transform(b, forward)).ToList();
        byte[][]? masks = sample.Masks?.Select(m => ImageWarp.AffineNearest(m, w, h, forward, w, h, (byte)0)).ToArray();
        int[]? classes = sample.PixelClasses is { } map ? ImageWarp.AffineNearest(map, w, h, forward, w, h, 0) : null;
        return MovedObjects.Keep(new ImageData(pixels, sample.Image.Channels, h, w), moved, sample.Labels, masks, classes, MinVisibility, boxesFromMasks: true);
    }

    // Source to output, about the centre c: x' = c + t + R(angle) · S(scale) · Shear · (x - c), with R counter-clockwise on
    // the screen (y down) and the shear x += -tan(shear) · y, as torchvision's affine matrix.
    private static double[] Map(int w, int h, double angle, double tx, double ty, double scale, double shear)
    {
        double cos = Math.Cos(angle), sin = Math.Sin(angle), k = -Math.Tan(shear);
        // R · S · Shear: [[cos, sin], [-sin, cos]] · s · [[1, k], [0, 1]].
        double a = cos * scale, b = (cos * k + sin) * scale, d = -sin * scale, e = (-sin * k + cos) * scale;
        double cx = w / 2.0, cy = h / 2.0;
        return [a, b, cx + tx - a * cx - b * cy, d, e, cy + ty - d * cx - e * cy];
    }
}

/// <summary>
/// Changes the colours of each sample ("color-jitter" in <see cref="Augmentations"/>, options <c>brightness</c>,
/// <c>contrast</c>, <c>saturation</c>, <c>hue</c>, <c>p</c>), as torchvision's <c>ColorJitter</c>: factors drawn in
/// [1 - x, 1 + x] (the hue's shift in [-x, x] of a turn) and applied in a random order. Boxes and masks do not change;
/// saturation and hue do nothing to grey images.
/// </summary>
/// <param name="brightness">How much the brightness may change (0: not).</param>
/// <param name="contrast">How much the contrast may change.</param>
/// <param name="saturation">How much the saturation may change.</param>
/// <param name="hue">How far the hue may turn, at most 0.5.</param>
/// <param name="probability">The chance the colours change.</param>
public sealed class ColorJitter(double brightness = 0, double contrast = 0, double saturation = 0, double hue = 0, double probability = 1) : IAugmentation
{
    /// <summary>How much the brightness may change.</summary>
    public double Brightness { get; } = brightness >= 0 ? brightness : throw new ArgumentOutOfRangeException(nameof(brightness));

    /// <summary>How much the contrast may change.</summary>
    public double Contrast { get; } = contrast >= 0 ? contrast : throw new ArgumentOutOfRangeException(nameof(contrast));

    /// <summary>How much the saturation may change.</summary>
    public double Saturation { get; } = saturation >= 0 ? saturation : throw new ArgumentOutOfRangeException(nameof(saturation));

    /// <summary>How far the hue may turn.</summary>
    public double Hue { get; } = hue is >= 0 and <= 0.5 ? hue : throw new ArgumentOutOfRangeException(nameof(hue));

    /// <summary>The chance the colours change.</summary>
    public double Probability { get; } = probability is >= 0 and <= 1 ? probability : throw new ArgumentOutOfRangeException(nameof(probability));

    /// <inheritdoc />
    public string Name => "color-jitter";

    /// <inheritdoc />
    public AnnotatedImage Apply(AnnotatedImage sample, AugmentationContext context)
    {
        ArgumentNullException.ThrowIfNull(sample);
        ArgumentNullException.ThrowIfNull(context);
        var random = context.Random;
        if (random.NextDouble() >= Probability)
        {
            return sample;
        }

        int[] order = [0, 1, 2, 3];
        random.Shuffle(order);
        double b = MovedObjects.Uniform(random, (Math.Max(0, 1 - Brightness), 1 + Brightness));
        double c = MovedObjects.Uniform(random, (Math.Max(0, 1 - Contrast), 1 + Contrast));
        double s = MovedObjects.Uniform(random, (Math.Max(0, 1 - Saturation), 1 + Saturation));
        double hShift = MovedObjects.Uniform(random, (-Hue, Hue));
        var image = sample.Image;
        int plane = image.Width * image.Height, channels = image.Channels;
        var pixels = (float[])image.Pixels.Clone();
        foreach (int step in order)
        {
            switch (step)
            {
                case 0 when Brightness > 0:
                    Blend(pixels, (float)b, 0f);
                    break;
                case 1 when Contrast > 0:
                {
                    double mean = 0;
                    for (int i = 0; i < plane; i++)
                    {
                        mean += Grey(pixels, i, plane, channels);
                    }

                    Blend(pixels, (float)c, (float)(mean / plane));
                    break;
                }

                case 2 when Saturation > 0 && channels == 3:
                {
                    // Each pixel towards (or away from) its own grey.
                    float f = (float)s, g = 1 - f;
                    for (int i = 0; i < plane; i++)
                    {
                        float grey = g * Grey(pixels, i, plane, channels);
                        pixels[i] = Math.Clamp(f * pixels[i] + grey, 0f, 1f);
                        pixels[plane + i] = Math.Clamp(f * pixels[plane + i] + grey, 0f, 1f);
                        pixels[2 * plane + i] = Math.Clamp(f * pixels[2 * plane + i] + grey, 0f, 1f);
                    }

                    break;
                }

                case 3 when Hue > 0 && channels == 3:
                    TurnHue(pixels, plane, (float)hShift);
                    break;
            }
        }

        return sample.WithImage(new ImageData(pixels, channels, image.Height, image.Width));
    }

    // value = clamp(factor · value + (1 - factor) · other), as torchvision's _blend.
    private static void Blend(float[] pixels, float factor, float other)
    {
        float offset = (1 - factor) * other;
        for (int i = 0; i < pixels.Length; i++)
        {
            pixels[i] = Math.Clamp(factor * pixels[i] + offset, 0f, 1f);
        }
    }

    // ITU-R 601 luma, as torchvision's rgb_to_grayscale.
    private static float Grey(float[] pixels, int i, int plane, int channels) =>
        channels == 3 ? 0.299f * pixels[i] + 0.587f * pixels[plane + i] + 0.114f * pixels[2 * plane + i] : pixels[i];

    // The hue turned by `shift` of a turn (RGB to HSV and back, as torchvision's adjust_hue), in single precision.
    private static void TurnHue(float[] pixels, int plane, float shift)
    {
        for (int i = 0; i < plane; i++)
        {
            float r = pixels[i], g = pixels[plane + i], b = pixels[2 * plane + i];
            float max = MathF.Max(r, MathF.Max(g, b)), min = MathF.Min(r, MathF.Min(g, b)), delta = max - min;
            if (delta <= 0)
            {
                continue;                                           // grey: no hue to turn
            }

            float h = max == r ? (g - b) / delta : max == g ? 2 + (b - r) / delta : 4 + (r - g) / delta;
            h = h / 6 + shift;
            h = (h - MathF.Floor(h)) * 6;
            float sat = delta / max, v = max;
            int sector = Math.Min(5, (int)h);
            float f = h - sector, p = v * (1 - sat), q = v * (1 - sat * f), t = v * (1 - sat * (1 - f));
            (r, g, b) = sector switch
            {
                0 => (v, t, p),
                1 => (q, v, p),
                2 => (p, v, t),
                3 => (p, q, v),
                4 => (t, p, v),
                _ => (v, p, q),
            };
            (pixels[i], pixels[plane + i], pixels[2 * plane + i]) = (r, g, b);
        }
    }
}

/// <summary>
/// Erases random rectangles of each sample ("cutout" in <see cref="Augmentations"/>, options <c>p</c>, <c>scale</c>,
/// <c>ratio</c>, <c>value</c>, <c>count</c>), as torchvision's <c>RandomErasing</c> chooses them (an area share in
/// <see cref="Scale"/>, an aspect ratio in <see cref="Ratio"/>, ten tries). The objects stay as they are: an occluded
/// object is still there.
/// </summary>
/// <param name="probability">The chance each rectangle is erased.</param>
/// <param name="scale">The rectangle's share of the image's area.</param>
/// <param name="ratio">The rectangle's height over width.</param>
/// <param name="value">The erased pixels' value.</param>
/// <param name="count">How many rectangles are tried.</param>
public sealed class Cutout(double probability = 0.5, (double Low, double High)? scale = null, (double Low, double High)? ratio = null, float value = 0f, int count = 1)
    : IAugmentation
{
    /// <summary>The chance each rectangle is erased.</summary>
    public double Probability { get; } = probability is >= 0 and <= 1 ? probability : throw new ArgumentOutOfRangeException(nameof(probability));

    /// <summary>The rectangle's share of the image's area.</summary>
    public (double Low, double High) Scale { get; } = scale is { Low: >= 0, High: <= 1 } s && s.Low <= s.High ? s : scale is null ? (0.02, 0.33)
        : throw new ArgumentOutOfRangeException(nameof(scale));

    /// <summary>The rectangle's height over width.</summary>
    public (double Low, double High) Ratio { get; } = ratio is { Low: > 0 } r && r.Low <= r.High ? r : ratio is null ? (0.3, 3.3)
        : throw new ArgumentOutOfRangeException(nameof(ratio));

    /// <summary>The erased pixels' value.</summary>
    public float Value { get; } = value;

    /// <summary>How many rectangles are tried.</summary>
    public int Count { get; } = count > 0 ? count : throw new ArgumentOutOfRangeException(nameof(count));

    /// <inheritdoc />
    public string Name => "cutout";

    /// <inheritdoc />
    public AnnotatedImage Apply(AnnotatedImage sample, AugmentationContext context)
    {
        ArgumentNullException.ThrowIfNull(sample);
        ArgumentNullException.ThrowIfNull(context);
        var random = context.Random;
        var image = sample.Image;
        int w = image.Width, h = image.Height, plane = w * h;
        float[]? pixels = null;
        var logRatio = (Math.Log(Ratio.Low), Math.Log(Ratio.High));
        for (int n = 0; n < Count; n++)
        {
            if (random.NextDouble() >= Probability)
            {
                continue;
            }

            for (int attempt = 0; attempt < 10; attempt++)
            {
                double area = plane * MovedObjects.Uniform(random, Scale), aspect = Math.Exp(MovedObjects.Uniform(random, logRatio));
                int eh = (int)Math.Round(Math.Sqrt(area * aspect)), ew = (int)Math.Round(Math.Sqrt(area / aspect));
                if (!(eh < h && ew < w))
                {
                    continue;
                }

                int top = random.Next(0, h - eh + 1), left = random.Next(0, w - ew + 1);
                pixels ??= (float[])image.Pixels.Clone();
                for (int c = 0; c < image.Channels; c++)
                {
                    for (int y = top; y < top + eh; y++)
                    {
                        pixels.AsSpan(c * plane + y * w + left, ew).Fill(Value);
                    }
                }

                break;
            }
        }

        return pixels is null ? sample : sample.WithImage(new ImageData(pixels, image.Channels, h, w));
    }
}

/// <summary>
/// Joins four samples in a 2 x 2 mosaic around a random centre ("mosaic" in <see cref="Augmentations"/>, options
/// <c>width</c>, <c>height</c>, <c>fill</c>, <c>p</c>, <c>min_visibility</c>): the sample and three drawn from the data
/// set, in a random order, each scaled to fit the output and set with a corner at the centre, cut by its quarter. Boxes,
/// masks and pixel classes follow their image; uncovered pixels take <see cref="Fill"/>.
/// </summary>
/// <param name="width">The output width (0: the sample's).</param>
/// <param name="height">The output height (0: the sample's).</param>
/// <param name="fill">The value of uncovered pixels.</param>
/// <param name="probability">The chance a mosaic is made.</param>
/// <param name="minVisibility">The share of a box's area that must stay in its quarter for the box to be kept (0: any part).</param>
public sealed class Mosaic(int width = 0, int height = 0, float fill = 0.5f, double probability = 1, double minVisibility = 0) : IAugmentation
{
    /// <summary>The output width (0: the sample's).</summary>
    public int Width { get; } = width >= 0 ? width : throw new ArgumentOutOfRangeException(nameof(width));

    /// <summary>The output height (0: the sample's).</summary>
    public int Height { get; } = height >= 0 ? height : throw new ArgumentOutOfRangeException(nameof(height));

    /// <summary>The value of uncovered pixels.</summary>
    public float Fill { get; } = fill;

    /// <summary>The chance a mosaic is made.</summary>
    public double Probability { get; } = probability is >= 0 and <= 1 ? probability : throw new ArgumentOutOfRangeException(nameof(probability));

    /// <summary>The share of a box's area that must stay in its quarter.</summary>
    public double MinVisibility { get; } = minVisibility is >= 0 and <= 1 ? minVisibility : throw new ArgumentOutOfRangeException(nameof(minVisibility));

    /// <inheritdoc />
    public string Name => "mosaic";

    /// <inheritdoc />
    public AnnotatedImage Apply(AnnotatedImage sample, AugmentationContext context)
    {
        ArgumentNullException.ThrowIfNull(sample);
        ArgumentNullException.ThrowIfNull(context);
        var random = context.Random;
        if (random.NextDouble() >= Probability)
        {
            return sample;
        }

        int ow = Width > 0 ? Width : sample.Width, oh = Height > 0 ? Height : sample.Height, channels = sample.Image.Channels, plane = ow * oh;
        var tiles = new[] { sample, context.Draw(), context.Draw(), context.Draw() };
        random.Shuffle(tiles);
        int xc = (int)Math.Round(MovedObjects.Uniform(random, (0.25 * ow, 0.75 * ow))), yc = (int)Math.Round(MovedObjects.Uniform(random, (0.25 * oh, 0.75 * oh)));
        var pixels = new float[channels * plane];
        Array.Fill(pixels, Fill);
        bool anyMasks = tiles.Any(t => t.Masks is not null), anyClasses = tiles.Any(t => t.PixelClasses is not null);
        int[]? classes = anyClasses ? new int[plane] : null;
        var boxes = new List<BoundingBox>();
        var labels = new List<int>();
        var masks = new List<byte[]>();
        for (int q = 0; q < 4; q++)
        {
            var tile = tiles[q];
            double r = Math.Min((double)ow / tile.Width, (double)oh / tile.Height);
            double sw = tile.Width * r, sh = tile.Height * r;
            // The quarter, and where the scaled tile's top-left corner goes so that a corner of it touches the centre.
            var (qx1, qy1, qx2, qy2) = q switch { 0 => (0, 0, xc, yc), 1 => (xc, 0, ow, yc), 2 => (0, yc, xc, oh), _ => (xc, yc, ow, oh) };
            double ox = q is 0 or 2 ? xc - sw : xc, oy = q is 0 or 1 ? yc - sh : yc;
            int rw = qx2 - qx1, rh = qy2 - qy1;
            if (rw <= 0 || rh <= 0)
            {
                continue;
            }

            // The quarter's pixels come from the tile's region [(qx1 - ox) / r, ...) of rw / r x rh / r source pixels.
            double sx0 = (qx1 - ox) / r, sy0 = (qy1 - oy) / r, swidth = rw / r, sheight = rh / r;
            var part = ImageWarp.Resample(tile.Image, sx0, sy0, swidth, sheight, rw, rh, channels, Fill);
            for (int c = 0; c < channels; c++)
            {
                for (int y = 0; y < rh; y++)
                {
                    part.AsSpan((c * rh + y) * rw, rw).CopyTo(pixels.AsSpan(c * plane + (qy1 + y) * ow + qx1, rw));
                }
            }

            if (classes is not null && tile.PixelClasses is { } map)
            {
                var cut = ImageWarp.ResampleNearest(map, tile.Width, tile.Height, sx0, sy0, swidth, sheight, rw, rh, 0);
                for (int y = 0; y < rh; y++)
                {
                    cut.AsSpan(y * rw, rw).CopyTo(classes.AsSpan((qy1 + y) * ow + qx1, rw));
                }
            }

            for (int i = 0; i < tile.Count; i++)
            {
                var b = tile.Boxes[i];
                var moved = new BoundingBox((float)(b.X * r + ox), (float)(b.Y * r + oy), (float)(b.Width * r), (float)(b.Height * r));
                var x1 = Math.Max(moved.X, qx1);
                var y1 = Math.Max(moved.Y, qy1);
                var clipped = BoundingBox.FromCorners(x1, y1, Math.Max(x1, Math.Min(moved.Right, qx2)), Math.Max(y1, Math.Min(moved.Bottom, qy2)));
                if (!(clipped.Width > 0 && clipped.Height > 0) || clipped.Area < MinVisibility * moved.Area)
                {
                    continue;
                }

                byte[]? mask = null;
                if (anyMasks)
                {
                    mask = new byte[plane];
                    var source = tile.Masks?[i] ?? AnnotatedImage.Fill(b, tile.Width, tile.Height);
                    var cut = ImageWarp.ResampleNearest(source, tile.Width, tile.Height, sx0, sy0, swidth, sheight, rw, rh, (byte)0);
                    for (int y = 0; y < rh; y++)
                    {
                        cut.AsSpan(y * rw, rw).CopyTo(mask.AsSpan((qy1 + y) * ow + qx1, rw));
                    }

                    if (ObjectMask.PixelBounds(mask, ow, oh).Width <= 0)
                    {
                        continue;
                    }
                }

                boxes.Add(clipped);
                labels.Add(tile.Labels[i]);
                if (mask is not null)
                {
                    masks.Add(mask);
                }
            }
        }

        return new AnnotatedImage(new ImageData(pixels, channels, oh, ow), [.. boxes], [.. labels], anyMasks ? [.. masks] : null, classes);
    }
}

/// <summary>
/// Mixes each sample with one drawn from the data set ("mixup" in <see cref="Augmentations"/>, options <c>alpha</c>,
/// <c>p</c>; Zhang et al. 2018): pixels λ·a + (1 - λ)·b with λ ~ Beta(α, α) (the other sample stretched to this one's size),
/// the objects of both kept. Pixel classes are the more weighted sample's.
/// </summary>
/// <param name="alpha">The Beta distribution's parameter (32 keeps λ near 0.5; 0.2 near 0 or 1).</param>
/// <param name="probability">The chance a mix is made.</param>
public sealed class MixUp(double alpha = 32, double probability = 1) : IAugmentation
{
    /// <summary>The Beta distribution's parameter.</summary>
    public double Alpha { get; } = alpha > 0 ? alpha : throw new ArgumentOutOfRangeException(nameof(alpha));

    /// <summary>The chance a mix is made.</summary>
    public double Probability { get; } = probability is >= 0 and <= 1 ? probability : throw new ArgumentOutOfRangeException(nameof(probability));

    /// <inheritdoc />
    public string Name => "mixup";

    /// <inheritdoc />
    public AnnotatedImage Apply(AnnotatedImage sample, AugmentationContext context)
    {
        ArgumentNullException.ThrowIfNull(sample);
        ArgumentNullException.ThrowIfNull(context);
        var random = context.Random;
        if (random.NextDouble() >= Probability)
        {
            return sample;
        }

        var other = new SampleResize(sample.Width, sample.Height).Apply(context.Draw(), context);
        double lambda = MovedObjects.Beta(random, Alpha, Alpha);
        var image = sample.Image;
        var mine = image.Pixels;
        var theirs = other.Image.Channels == image.Channels ? other.Image.Pixels : other.Image.Resize(image.Channels, image.Height, image.Width);
        var pixels = new float[mine.Length];
        for (int i = 0; i < pixels.Length; i++)
        {
            pixels[i] = (float)(lambda * mine[i] + (1 - lambda) * theirs[i]);
        }

        bool anyMasks = sample.Masks is not null || other.Masks is not null;
        byte[][]? masks = anyMasks ? [.. Masks(sample), .. Masks(other)] : null;
        int[]? classes = lambda >= 0.5 ? sample.PixelClasses ?? other.PixelClasses : other.PixelClasses ?? sample.PixelClasses;
        return new AnnotatedImage(new ImageData(pixels, image.Channels, image.Height, image.Width), [.. sample.Boxes, .. other.Boxes], [.. sample.Labels, .. other.Labels],
            masks, classes);

        static IEnumerable<byte[]> Masks(AnnotatedImage s) => s.Masks ?? s.Boxes.Select(b => AnnotatedImage.Fill(b, s.Width, s.Height));
    }
}
