// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Imaging;

/// <summary>
/// The library's image transforms (<see cref="ImageTransforms"/>), Pillow's operations byte for byte
/// (<see cref="PillowImageOps"/>), registered by <c>LibraryRegistrations</c> when the registry is first used.
/// </summary>
internal static class LibraryImageTransforms
{
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

}
