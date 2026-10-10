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
            new Transform("grayscale", "Pillow's convert(\"L\"): one grey channel (no value)", [], s => s.ThrowIfValue(), (i, _) => ChatImageDecoder.Grayscale(i)),
            new Transform("max_width", "max_width=N: shrink a wider image to N pixels wide, keeping its aspect ratio (the height int(h * N / w)); "
                                       + "resample=lanczos (default), bicubic, bilinear, box or hamming",
                ["resample"], s => CheckSize(s), (i, s) => PillowImageOps.ScaleDown(i, CheckSize(s), 0, Resampling(s))),
            new Transform("max_height", "max_height=N: shrink a taller image to N pixels high, keeping its aspect ratio (the width int(w * N / h)); "
                                        + "resample as max_width",
                ["resample"], s => CheckSize(s), (i, s) => PillowImageOps.ScaleDown(i, 0, CheckSize(s), Resampling(s))),
            new Transform("contrast", "contrast=F: ImageEnhance.Contrast(image).enhance(F) (1 unchanged, above 1 more contrast)", [],
                s => Factor(s), (i, s) => PillowImageOps.Contrast(i, Factor(s))),
            new Transform("brightness", "brightness=F: ImageEnhance.Brightness(image).enhance(F) (1 unchanged)", [],
                s => Factor(s), (i, s) => PillowImageOps.Brightness(i, Factor(s))),
            new Transform("sharpness", "sharpness=F: ImageEnhance.Sharpness(image).enhance(F) (1 unchanged, 2 sharper, 0 smoothed)", [],
                s => Factor(s), (i, s) => PillowImageOps.Sharpness(i, Factor(s))),
            new Transform("autocontrast", "autocontrast[=CUTOFF]: ImageOps.autocontrast (CUTOFF percent cut at each end, 0 unless given); "
                                          + "ignore=V, preserve_tone=true",
                ["ignore", "preserve_tone"], s => Autocontrast(s, null), (i, s) => Autocontrast(s, i)),
            new Transform("invert", "ImageOps.invert: every channel's byte v becomes 255 - v (light text on a dark page to dark on light; no value)", [],
                s => s.ThrowIfValue(), (i, _) => PillowImageOps.Invert(i)),
            new Transform("jpeg", "jpeg=Q: save as JPEG at quality Q (1 to 100; optimize=True, 4:2:0 for colour) and decode again, as a base64 upload of a Pillow-saved file",
                ["subsampling"], s => Jpeg(s, null), (i, s) => Jpeg(s, i)),
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

    private sealed class Transform(string name, string summary, string[] keys, Action<ImageTransformStep> check, Func<ImageData, ImageTransformStep, ImageData> apply)
        : IImageTransform
    {
        public string Name => name;

        public string Summary => summary;

        public IReadOnlyCollection<string> Keys => keys;

        public void Check(ImageTransformStep step) => check(step);

        public ImageData Apply(ImageData image, ImageTransformStep step) => apply(image, step);
    }
}
