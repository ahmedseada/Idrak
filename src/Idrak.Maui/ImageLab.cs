// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Imaging;
using Idrak.Maui.Abstractions;

namespace Idrak.Maui;

/// <summary>
/// The Image lab's settings: the pipeline last applied, the scan profile and its seed, how a result is saved (the format
/// and the JPEG quality) and how big the pictures on the page are. A null value is the library's default (the PNG
/// encoder, <see cref="JpegEncoder"/>'s quality, <see cref="PillowImageOps.ScaleDown"/>'s resampling) or, for the
/// pictures, the screen's size measured at run time. Kept between runs by <see cref="ImageSettings"/>.
/// </summary>
public sealed class ImageLabOptions
{
    /// <summary>The pipeline of the page's box (<see cref="ImageTransformPipeline"/> text); null: none.</summary>
    public string? Pipeline { get; set; }

    /// <summary>The scan profile picked (a <see cref="ScanProfiles"/> name, or profiles joined with '+'); null: none.</summary>
    public string? Profile { get; set; }

    /// <summary>The seed a profile draws its steps from (the same seed, the same steps); null: a new one each time, shown.</summary>
    public int? Seed { get; set; }

    /// <summary>The format a result is saved and shared in: "png", "jpeg" or another <see cref="ImageEncoders"/> name; null: png.</summary>
    public string? SaveFormat { get; set; }

    /// <summary>The JPEG quality, 1 to 100 (<see cref="JpegEncoder.Encode(ImageData, int, bool, JpegSubsampling)"/>); null: the library's.</summary>
    public int? JpegQuality { get; set; }

    /// <summary>The longest side, in pixels, of a picture on the page (shown only: the image itself keeps its size); null: the screen's, measured.</summary>
    public int? PreviewPixels { get; set; }

    /// <summary>How a picture is shrunk for the page (an <see cref="ImageResampling"/> name); null: the library's (<see cref="PillowImageOps.ScaleDown"/>'s).</summary>
    public string? PreviewResampling { get; set; }

    /// <summary>A copy.</summary>
    public ImageLabOptions Clone() => (ImageLabOptions)MemberwiseClone();

    /// <summary>The save format in effect ("png" when none is set).</summary>
    public string Format => SaveFormat ?? ImageSettings.Png;

    /// <summary>The pipeline of <see cref="Pipeline"/>, checked against the registered transforms (empty when none).</summary>
    /// <exception cref="ArgumentException">An unknown transform or option, or a value the transform refuses (the message names the registered ones).</exception>
    public ImageTransformPipeline ToPipeline() => ImageTransformPipeline.Parse(Pipeline);

    /// <summary>
    /// <paramref name="image"/> as a file of <see cref="Format"/>: the library's JPEG encoder at <see cref="JpegQuality"/>
    /// (the library's quality when none), else the registered encoder of that name.
    /// </summary>
    public byte[] Encode(ImageData image)
    {
        ArgumentNullException.ThrowIfNull(image);
        return Format == ImageSettings.Jpeg
            ? JpegQuality is { } quality ? JpegEncoder.Encode(image, quality) : JpegEncoder.Encode(image)
            : ImageEncoders.Get(Format).Encode(image);
    }

    /// <summary>The file extension of <see cref="Format"/> (".png", ".jpg").</summary>
    public string Extension => Format == ImageSettings.Jpeg ? ".jpg" : ImageEncoders.Get(Format).Extensions.FirstOrDefault() ?? "." + Format;

    /// <summary>The media type of <see cref="Format"/> for the share sheet.</summary>
    public string MediaType => Format switch
    {
        ImageSettings.Jpeg => "image/jpeg",
        ImageSettings.Png => "image/png",
        _ => "application/octet-stream",
    };

    /// <summary>The resampling of the page's pictures (the library's when none is set).</summary>
    public ImageResampling? Resampling => PreviewResampling is { } name ? Enum.Parse<ImageResampling>(name, ignoreCase: true) : null;
}

/// <summary>
/// The Image lab's settings kept between runs, each under <see cref="Prefix"/> and its key in the app's
/// <see cref="ISettingsStore"/> (MAUI's Preferences): the page's pipeline, and every <see cref="Definitions"/> value.
/// </summary>
public static class ImageSettings
{
    /// <summary>What every key starts with.</summary>
    public const string Prefix = IdrakMauiSettings.Prefix + "image.";

    /// <summary>The PNG format's name (the library's encoder; the default).</summary>
    public const string Png = "png";

    /// <summary>The JPEG format's name (the library's <see cref="JpegEncoder"/>).</summary>
    public const string Jpeg = "jpeg";

    private const string PipelineKey = Prefix + "pipeline";

    /// <summary>The settings edited on the page, in its order.</summary>
    public static IReadOnlyList<LabSetting<ImageLabOptions>> Definitions { get; } = Define();

    /// <summary>The formats a result is saved in: png, jpeg, then the other registered encoders.</summary>
    public static IReadOnlyList<string> Formats => [Png, Jpeg, .. ImageEncoders.Names.Where(n => !n.Equals(Png, StringComparison.OrdinalIgnoreCase) && !n.Equals(Jpeg, StringComparison.OrdinalIgnoreCase))];

    /// <summary>
    /// The options saved in <paramref name="store"/>, over <paramref name="defaults"/> (a copy; the library's defaults when
    /// null); a saved value that no longer reads is skipped and named.
    /// </summary>
    public static ImageLabOptions Load(ISettingsStore store, out IReadOnlyList<string> skipped, ImageLabOptions? defaults = null)
    {
        ArgumentNullException.ThrowIfNull(store);
        var options = LabSettings.Load(store, Prefix, Definitions, defaults?.Clone() ?? new ImageLabOptions(), out skipped);
        if (store.Get(PipelineKey) is { } pipeline)
        {
            try
            {
                options.Pipeline = LabSettings.Pipeline(pipeline);
            }
            catch (FormatException e)
            {
                skipped = [.. skipped, $"Pipeline: {e.Message}"];
            }
        }

        return options;
    }

    /// <summary>Keeps every value of <paramref name="options"/> in <paramref name="store"/> (an empty one removed: the default).</summary>
    public static void Save(ISettingsStore store, ImageLabOptions options)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentNullException.ThrowIfNull(options);
        store.Set(PipelineKey, LabSettings.Empty(options.Pipeline));
        LabSettings.Save(store, Prefix, Definitions, options);
    }

    private static IReadOnlyList<LabSetting<ImageLabOptions>> Define() =>
    [
        new("profile", "Scan profile", "The scan profile Apply profile draws from (ScanProfiles; several joined with '+', such as damaged+marked).", LabSettingKind.Choice, "none",
            o => o.Profile ?? "", (o, t) => o.Profile = Profile(t), () => ["none", .. ScanProfiles.Names]),
        new("seed", "Profile seed", "The seed a scan profile draws its steps from: the same seed, the same steps; empty: a new one each time (shown).",
            LabSettingKind.Number, "a new one each time", o => LabSettings.Show(o.Seed), (o, t) => o.Seed = LabSettings.Whole(t, 0)),
        new("save_format", "Save as", "The format a result is saved and shared in (png: the library's PNG encoder; jpeg: its JPEG encoder; or another registered encoder).",
            LabSettingKind.Choice, Png, o => o.SaveFormat ?? "", (o, t) => o.SaveFormat = Format(t), () => Formats),
        new("jpeg_quality", "JPEG quality", "1 to 100, for a result saved as JPEG (JpegEncoder.Encode's quality); empty: the library's.",
            LabSettingKind.Number, "the library's", o => LabSettings.Show(o.JpegQuality), (o, t) => o.JpegQuality = LabSettings.Whole(t, 1, 100)),
        new("preview_px", "Picture size on the page (px)", "The longest side a picture is shrunk to for the page only (the image keeps its size); empty: the screen's, measured.",
            LabSettingKind.Number, "the screen's", o => LabSettings.Show(o.PreviewPixels), (o, t) => o.PreviewPixels = LabSettings.Whole(t, 16)),
        new("preview_resample", "Picture shrinking", "How a picture is shrunk for the page (ImageResampling); empty: the library's (PillowImageOps.ScaleDown's).", LabSettingKind.Choice,
            "default", o => o.PreviewResampling ?? "", (o, t) => o.PreviewResampling = LabSettings.Registered(t, Resamplings, "a resampling"), () => ["default", .. Resamplings]),
    ];

    // The resampling filters by name, lower case.
    internal static IReadOnlyList<string> Resamplings { get; } = [.. Enum.GetNames<ImageResampling>().Select(n => n.ToLowerInvariant())];

    private static string? Profile(string text)
    {
        if (text.Length == 0 || text.Equals("none", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        try
        {
            _ = ScanProfiles.Resolve(text);
            return text;
        }
        catch (Exception e) when (e is NotSupportedException or ArgumentException)
        {
            throw new FormatException(e.Message);
        }
    }

    private static string? Format(string text) =>
        text.Length == 0 || text.Equals(Png, StringComparison.OrdinalIgnoreCase) ? null
        : text.Equals(Jpeg, StringComparison.OrdinalIgnoreCase) || text.Equals("jpg", StringComparison.OrdinalIgnoreCase) ? Jpeg
        : ImageEncoders.Names.FirstOrDefault(n => n.Equals(text, StringComparison.OrdinalIgnoreCase))
          ?? throw new FormatException($"'{text}' is not a format a result is saved in ({string.Join(", ", Formats)}).");
}
