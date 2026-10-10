// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Data;

/// <summary>
/// The library's image transforms (<see cref="ImageTransforms"/>), Pillow's operations byte for byte
/// (<see cref="PillowImageOps"/>), registered by <c>LibraryRegistrations</c> when the registry is first used.
/// </summary>
internal static class LibraryImageTransforms
{
    internal static void RegisterDefaults()
    {
        foreach (var transform in (IImageTransform[])
        [
            new Transform("grayscale", "Pillow's convert(\"L\"): one grey channel (no value)", [], s => s.ThrowIfValue(), (i, _) => ChatImageDecoder.Grayscale(i))
            {
                Label = "Grayscale",
            },
            new Transform("max_width", "max_width=N: shrink a wider image to N pixels wide, keeping its aspect ratio (the height int(h * N / w)); "
                                       + "resample=lanczos (default), bicubic, bilinear, box or hamming",
                ["resample"], s => CheckSize(s), (i, s) => PillowImageOps.ScaleDown(i, CheckSize(s), 0, Resampling(s)))
            {
                Label = "Shrink to a width", Parameters = [Pixels("Width, pixels", 16, 8192, 8), ResampleInput],
            },
            new Transform("max_height", "max_height=N: shrink a taller image to N pixels high, keeping its aspect ratio (the width int(w * N / h)); "
                                        + "resample as max_width",
                ["resample"], s => CheckSize(s), (i, s) => PillowImageOps.ScaleDown(i, 0, CheckSize(s), Resampling(s)))
            {
                Label = "Shrink to a height", Parameters = [Pixels("Height, pixels", 16, 8192, 8), ResampleInput],
            },
            new Transform("scale", "scale=F: resize by a factor (int(w * F + 0.5) x int(h * F + 0.5)): above 1 enlarges small text; resample as max_width",
                ["resample"], s => ScaleFactor(s), (i, s) => PillowImageOps.Scale(i, ScaleFactor(s), Resampling(s)))
            {
                Label = "Scale", Parameters = [Number("Factor", 0.1, 8, 0.05, "above 1 enlarges small characters"), ResampleInput],
            },
            new Transform("pad", "pad=N: ImageOps.expand: a margin of N pixels on every side; fill=V its grey level (255, white, unless given)",
                ["fill"], s => Pad(s, null), (i, s) => Pad(s, i))
            {
                Label = "Pad", Parameters = [Pixels("Margin, pixels", 0, 1024, 1), Byte("fill", "Margin grey level (255 white)")],
            },
            new Transform("contrast", "contrast=F: ImageEnhance.Contrast(image).enhance(F) (1 unchanged, above 1 more contrast)", [],
                s => Factor(s), (i, s) => PillowImageOps.Contrast(i, Factor(s)))
            {
                Label = "Contrast", Parameters = [Number("Factor", 0, 5, 0.05, "1 unchanged, above 1 more contrast")],
            },
            new Transform("brightness", "brightness=F: ImageEnhance.Brightness(image).enhance(F) (1 unchanged)", [],
                s => Factor(s), (i, s) => PillowImageOps.Brightness(i, Factor(s)))
            {
                Label = "Brightness", Parameters = [Number("Factor", 0, 5, 0.05, "1 unchanged, below 1 darker")],
            },
            new Transform("sharpness", "sharpness=F: ImageEnhance.Sharpness(image).enhance(F) (1 unchanged, 2 sharper, 0 smoothed)", [],
                s => Factor(s), (i, s) => PillowImageOps.Sharpness(i, Factor(s)))
            {
                Label = "Sharpness", Parameters = [Number("Factor", 0, 5, 0.05, "1 unchanged, 2 sharper, 0 smoothed")],
            },
            new Transform("autocontrast", "autocontrast[=CUTOFF]: ImageOps.autocontrast (CUTOFF percent cut at each end, 0 unless given); "
                                          + "ignore=V, preserve_tone=true",
                ["ignore", "preserve_tone"], s => Autocontrast(s, null), (i, s) => Autocontrast(s, i))
            {
                Label = "Auto contrast",
                Parameters =
                [
                    Number("Cutoff, percent", 0, 49, 0.5, "the darkest and lightest share ignored at each end") with { Optional = true },
                    Byte("ignore", "A grey level to ignore (the background)") with { Optional = true },
                    new ImageTransformParameter("preserve_tone", "Keep the tone", ImageTransformInput.Choice) { Choices = ["false", "true"], Optional = true },
                ],
            },
            new Transform("equalize", "ImageOps.equalize: each channel's histogram spread evenly (faded scans; no value)", [],
                s => s.ThrowIfValue(), (i, _) => PillowImageOps.Equalize(i))
            {
                Label = "Equalize",
            },
            new Transform("gamma", "gamma=G: each byte v to int((v / 255) ** (1 / G) * 255 + 0.5): below 1 darkens faint ink, above 1 lightens", [],
                s => Positive(s, 0.05, 10), (i, s) => PillowImageOps.Gamma(i, Positive(s, 0.05, 10)))
            {
                Label = "Gamma", Parameters = [Number("Gamma", 0.05, 10, 0.05, "below 1 darkens faint, thin ink")],
            },
            new Transform("blur", "blur=R: ImageFilter.GaussianBlur(R): smooths grain and noise (R pixels)", [],
                s => Positive(s, 0, 100), (i, s) => PillowImageOps.GaussianBlur(i, Positive(s, 0, 100)))
            {
                Label = "Blur", Parameters = [Number("Radius, pixels", 0, 100, 0.1, "smooths grain; small values only")],
            },
            new Transform("unsharp", "unsharp=R: ImageFilter.UnsharpMask(R, percent, threshold): sharpens stroke edges; percent=P (150), threshold=T (3)",
                ["percent", "threshold"], s => Unsharp(s, null), (i, s) => Unsharp(s, i))
            {
                Label = "Unsharp mask",
                Parameters =
                [
                    Number("Radius, pixels", 0, 100, 0.1, "the width of the edges sharpened"),
                    new ImageTransformParameter("percent", "Strength, percent", ImageTransformInput.Integer) { Min = 0, Max = 1000, Step = 10 },
                    Byte("threshold", "Least difference sharpened (flat paper left alone)"),
                ],
            },
            new Transform("median", "median=N: ImageFilter.MedianFilter(N) (odd N): removes specks and salt-and-pepper noise",
                [], s => RankSize(s), (i, s) => PillowImageOps.Median(i, RankSize(s)))
            {
                Label = "Remove specks (median)", Parameters = [Window],
            },
            new Transform("min_filter", "min_filter=N: ImageFilter.MinFilter(N) (odd N): each pixel the darkest around it: dark strokes thicken",
                [], s => RankSize(s), (i, s) => PillowImageOps.MinFilter(i, RankSize(s)))
            {
                Label = "Thicken dark strokes (min filter)", Parameters = [Window],
            },
            new Transform("max_filter", "max_filter=N: ImageFilter.MaxFilter(N) (odd N): each pixel the lightest around it: dark strokes thin, dark specks go",
                [], s => RankSize(s), (i, s) => PillowImageOps.MaxFilter(i, RankSize(s)))
            {
                Label = "Thin dark strokes (max filter)", Parameters = [Window],
            },
            new Transform("binarize", "binarize[=T]: grey, then white from T and black below (T by Otsu's method when not given)", [],
                s => Threshold(s), (i, s) => PillowImageOps.Binarize(i, Threshold(s)))
            {
                Label = "Black and white",
                Parameters = [Byte("", "Threshold (empty: Otsu's, from the page)") with { Optional = true }],
            },
            new Transform("invert", "ImageOps.invert: every channel's byte v becomes 255 - v (light text on a dark page to dark on light; no value)", [],
                s => s.ThrowIfValue(), (i, _) => PillowImageOps.Invert(i))
            {
                Label = "Invert",
            },
            new Transform("jpeg", "jpeg=Q: save as JPEG at quality Q (1 to 100; optimize=True, 4:2:0 for colour) and decode again, as a base64 upload of a Pillow-saved file",
                ["subsampling"], s => Jpeg(s, null), (i, s) => Jpeg(s, i))
            {
                Label = "JPEG round trip",
                Parameters =
                [
                    new ImageTransformParameter("", "Quality", ImageTransformInput.Integer) { Min = 1, Max = 100, Step = 1 },
                    new ImageTransformParameter("subsampling", "Chroma subsampling", ImageTransformInput.Choice) { Choices = ["4:2:0", "4:2:2", "4:4:4"], Optional = true },
                ],
            },
        ])
        {
            ImageTransforms.Register(transform);
        }
    }

    private static int CheckSize(ImageTransformStep step)
    {
        int n = step.Integer();
        if (n < 1)
        {
            throw new ArgumentException($"{step.Name} is a positive number of pixels, not {n}.");
        }

        Resampling(step);
        return n;
    }

    private static ImageResampling Resampling(ImageTransformStep step) => step.Option("resample", "lanczos")!.ToLowerInvariant() switch
    {
        "lanczos" or "antialias" or "1" => ImageResampling.Lanczos,
        "bilinear" or "linear" or "2" => ImageResampling.Bilinear,
        "bicubic" or "cubic" or "3" => ImageResampling.Bicubic,
        "box" or "4" => ImageResampling.Box,
        "hamming" or "5" => ImageResampling.Hamming,
        var other => throw new ArgumentException($"{step.Name}: resample is lanczos, bicubic, bilinear, box or hamming, not '{other}'."),
    };

    private static double Factor(ImageTransformStep step)
    {
        double f = step.Number();
        if (f < 0)
        {
            throw new ArgumentException($"{step.Name} is a factor of 0 or more, not {f.ToString(CultureInfo.InvariantCulture)}.");
        }

        return f;
    }

    private static ImageData Autocontrast(ImageTransformStep step, ImageData? image)
    {
        double cutoff = step.Number(0);
        if (cutoff is < 0 or >= 50)
        {
            throw new ArgumentException($"autocontrast's cutoff is a percentage from 0 to under 50, not {cutoff.ToString(CultureInfo.InvariantCulture)}.");
        }

        int? ignore = step.Option("ignore") is { } text
            ? int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) && v is >= 0 and <= 255 ? v
                : throw new ArgumentException($"autocontrast's ignore is a byte value 0 to 255, not '{text}'.")
            : null;
        bool tone = step.Option("preserve_tone", "false")!.ToLowerInvariant() switch
        {
            "true" or "1" or "yes" or "on" => true,
            "false" or "0" or "no" or "off" => false,
            var other => throw new ArgumentException($"autocontrast's preserve_tone is true or false, not '{other}'."),
        };
        return image is null ? null! : PillowImageOps.Autocontrast(image, cutoff, cutoff, ignore, tone);
    }

    private static ImageData Jpeg(ImageTransformStep step, ImageData? image)
    {
        int quality = step.Integer();
        if (quality is < 1 or > 100)
        {
            throw new ArgumentException($"jpeg's quality is 1 to 100, not {quality}.");
        }

        var subsampling = step.Option("subsampling", "4:2:0") switch
        {
            "4:2:0" or "420" or "2" => JpegSubsampling.Half420,
            "4:2:2" or "422" or "1" => JpegSubsampling.Half422,
            "4:4:4" or "444" or "0" => JpegSubsampling.Full444,
            var other => throw new ArgumentException($"jpeg's subsampling is 4:2:0, 4:2:2 or 4:4:4, not '{other}'."),
        };
        return image is null ? null! : PillowImageOps.JpegRoundTrip(image, quality, subsampling);
    }

    private static readonly ImageTransformParameter ResampleInput = new("resample", "Resampling", ImageTransformInput.Choice)
    {
        Choices = ["lanczos", "bicubic", "bilinear", "box", "hamming"], Optional = true,
    };

    private static readonly ImageTransformParameter Window = new("", "Window, pixels (odd)", ImageTransformInput.Choice) { Choices = ["3", "5", "7", "9"] };

    private static ImageTransformParameter Pixels(string label, double min, double max, double step) =>
        new("", label, ImageTransformInput.Integer) { Min = min, Max = max, Step = step };

    private static ImageTransformParameter Number(string label, double min, double max, double step, string help) =>
        new("", label, ImageTransformInput.Number) { Min = min, Max = max, Step = step, Help = help };

    private static ImageTransformParameter Byte(string key, string label) =>
        new(key, label, ImageTransformInput.Integer) { Min = 0, Max = 255, Step = 1 };

    private static double ScaleFactor(ImageTransformStep step)
    {
        Resampling(step);
        return Positive(step, 0.01, 16);
    }

    private static double Positive(ImageTransformStep step, double min, double max)
    {
        double v = step.Number();
        if (!(v >= min && v <= max) || (min == 0 && v < 0))
        {
            throw new ArgumentException($"{step.Name} is from {min.ToString(CultureInfo.InvariantCulture)} to {max.ToString(CultureInfo.InvariantCulture)}, not {v.ToString(CultureInfo.InvariantCulture)}.");
        }

        return v;
    }

    private static ImageData Pad(ImageTransformStep step, ImageData? image)
    {
        int border = step.Integer();
        if (border is < 0 or > 4096)
        {
            throw new ArgumentException($"pad is 0 to 4096 pixels, not {border}.");
        }

        int fill = OptionByte(step, "fill", 255);
        return image is null ? null! : PillowImageOps.Pad(image, border, (byte)fill);
    }

    private static int OptionByte(ImageTransformStep step, string key, int otherwise) =>
        step.Option(key) is not { } text ? otherwise
        : int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) && v is >= 0 and <= 255 ? v
        : throw new ArgumentException($"{step.Name}'s {key} is 0 to 255, not '{text}'.");

    private static ImageData Unsharp(ImageTransformStep step, ImageData? image)
    {
        double radius = step.Value is null or "" ? 2 : Positive(step, 0, 100);
        int percent = step.Option("percent") is not { } text ? 150
            : int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int p) && p is >= 0 and <= 10000 ? p
            : throw new ArgumentException($"unsharp's percent is a whole number from 0 to 10000, not '{text}'.");
        int threshold = OptionByte(step, "threshold", 3);
        return image is null ? null! : PillowImageOps.UnsharpMask(image, radius, percent, threshold);
    }

    private static int RankSize(ImageTransformStep step)
    {
        int n = step.Integer(3);
        if (n is < 1 or > 31 || n % 2 == 0)
        {
            throw new ArgumentException($"{step.Name} is an odd window size from 1 to 31, not {n}.");
        }

        return n;
    }

    private static int? Threshold(ImageTransformStep step)
    {
        if (step.Value is null or "")
        {
            return null;
        }

        int t = step.Integer();
        return t is >= 0 and <= 256 ? t : throw new ArgumentException($"binarize's threshold is a grey level from 0 to 256, not {t}.");
    }

    private sealed class Transform(string name, string summary, string[] keys, Action<ImageTransformStep> check, Func<ImageData, ImageTransformStep, ImageData> apply)
        : IImageTransform
    {
        public string Name => name;

        public string Label { get => field ?? Name; init; }

        public IReadOnlyList<ImageTransformParameter> Parameters { get; init; } = [];

        public string Summary => summary;

        public IReadOnlyCollection<string> Keys => keys;

        public void Check(ImageTransformStep step) => check(step);

        public ImageData Apply(ImageData image, ImageTransformStep step) => apply(image, step);
    }
}
