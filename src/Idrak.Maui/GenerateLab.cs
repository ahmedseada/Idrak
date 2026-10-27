// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Text;

namespace Idrak.Maui;

/// <summary>
/// The Generate lab's settings: what <c>idrak data synth</c> takes (the pages a text makes, the share in grey, the page
/// settings, the parts file, the seed, the image format and quality, and for a data set the prompt, the answer key, the zip
/// and the workers), and the lab's own: where fonts are found and kept, how many it picks for a text when none is chosen,
/// the pictures' sizes on the page, where data sets and saved pictures go. A null value is the library's default
/// (<see cref="SyntheticPageOptions"/>'s) or, for the folders, the platform's and the app's own. Kept between runs by
/// <see cref="GenerateSettings"/>.
/// </summary>
public sealed class GenerateLabOptions
{
    /// <summary>The phone's own fonts folder; null: the platform's, as its <see cref="ISystemFonts"/> lists them (Android's system fonts), none where it has none.</summary>
    public string? SystemFonts { get; set; }

    /// <summary>Where the fonts the user adds are copied; null: <c>fonts</c> in the app's storage.</summary>
    public string? FontsFolder { get; set; }

    /// <summary>How many fonts are picked for a text when none is chosen (those that have its characters, the user's first); null: 3.</summary>
    public int? AutoFonts { get; set; }

    /// <summary>The largest font file, in MiB, read when fonts are picked for a text (a larger one is chosen by hand only); null: 8.</summary>
    public double? AutoFontMiB { get; set; }

    /// <summary>The pages made at once (<c>-n, --count</c>, <see cref="SyntheticPageOptions.Variants"/>); null: the library's.</summary>
    public int? Count { get; set; }

    /// <summary>The seed (<c>--seed</c>, <see cref="SyntheticPageOptions.Seed"/>): the same seed and text give the same pages; null: 0.</summary>
    public int? Seed { get; set; }

    /// <summary>The share of pages in grey (<c>--grey</c>, <see cref="SyntheticPageOptions.Grey"/>); null: the library's.</summary>
    public double? Grey { get; set; }

    /// <summary>The page settings (<c>--page</c>, <see cref="SyntheticPages.Style"/>: KEY=VALUE,...); null: the library's style.</summary>
    public string? Page { get; set; }

    /// <summary>A parts file (<c>--parts</c>, <see cref="SyntheticPageParts.Read"/>): headers, subjects, signatures, footers; null: the text alone.</summary>
    public string? Parts { get; set; }

    /// <summary>The pages' image format (<c>--image-format</c>: jpeg or a registered encoder); null: the library's (jpeg).</summary>
    public string? ImageFormat { get; set; }

    /// <summary>The JPEG quality (<c>--quality</c>, 1 to 100); null: the library's.</summary>
    public int? Quality { get; set; }

    /// <summary>A data set's question (<c>--prompt</c>, <see cref="SyntheticPageOptions.Prompt"/>); null: the library's.</summary>
    public string? Prompt { get; set; }

    /// <summary>A data set's answer written as JSON under this key (<c>--answer-key</c>); null: the text itself.</summary>
    public string? AnswerKey { get; set; }

    /// <summary>A data set's images in a zip (<c>--zip</c>); null: a folder.</summary>
    public bool? Zip { get; set; }

    /// <summary>Pages made at once for a data set (<c>--workers</c>); null: one per processor.</summary>
    public int? Workers { get; set; }

    /// <summary>Where data sets are written; null: <c>datasets</c> in the app's storage.</summary>
    public string? DatasetFolder { get; set; }

    /// <summary>The album saved pictures go to, under the phone's Pictures; null: "Idrak".</summary>
    public string? Album { get; set; }

    /// <summary>The longest side, in pixels, of the large preview (shown only; the page keeps its size); null: the screen's, measured.</summary>
    public int? PreviewPixels { get; set; }

    /// <summary>The longest side, in pixels, of a page's tile in the grid; null: 320.</summary>
    public int? ThumbnailPixels { get; set; }

    /// <summary>A copy.</summary>
    public GenerateLabOptions Clone() => (GenerateLabOptions)MemberwiseClone();

    /// <summary>The phone's fonts folders in effect: the one set, else those <paramref name="platform"/> lists (none when it is null).</summary>
    public IReadOnlyList<string> SystemFontFolders(ISystemFonts? platform) => SystemFonts is { } folder ? [folder] : platform?.Folders ?? [];

    /// <summary>The added fonts' folder in effect under <paramref name="appStorage"/>.</summary>
    public string AddedFontsFolder(string appStorage) => FontsFolder ?? Path.Combine(appStorage, "fonts");

    /// <summary>The data sets' folder in effect under <paramref name="appStorage"/>.</summary>
    public string DatasetsFolder(string appStorage) => DatasetFolder ?? Path.Combine(appStorage, "datasets");

    /// <summary>The fonts picked for a text when none is chosen.</summary>
    public int AutoFontCount => AutoFonts ?? 3;

    /// <summary>The largest font file read when picking, in bytes.</summary>
    public long AutoFontBytes => (long)((AutoFontMiB ?? 8) * 1024 * 1024);

    /// <summary>The tiles' side in pixels.</summary>
    public int Thumbnail => ThumbnailPixels ?? 320;

    /// <summary>The album in effect.</summary>
    public string AlbumName => Album ?? "Idrak";
}

/// <summary>
/// The Generate lab's settings kept between runs, each under <see cref="Prefix"/> and its key in the app's
/// <see cref="ISettingsStore"/>: every <see cref="Definitions"/> value, and the page's own (the text, the fonts chosen, the
/// layouts, the scan look and whether every look is drawn).
/// </summary>
public static class GenerateSettings
{
    /// <summary>What every key starts with.</summary>
    public const string Prefix = IdrakMauiSettings.Prefix + "generate.";

    internal const string TextKey = Prefix + "text", FontsKey = Prefix + "fonts", LayoutsKey = Prefix + "layouts", ScanKey = Prefix + "scan", EveryLookKey = Prefix + "every_look";

    /// <summary>The settings edited on the page, in its order.</summary>
    public static IReadOnlyList<LabSetting<GenerateLabOptions>> Definitions { get; } = Define();

    /// <summary>The image formats a page is written in: jpeg, then the registered encoders.</summary>
    public static IReadOnlyList<string> Formats => ["jpeg", .. ImageEncoders.Names.Where(n => !n.Equals("jpeg", StringComparison.OrdinalIgnoreCase))];

    /// <summary>The options saved in <paramref name="store"/> over <paramref name="defaults"/> (a copy; the library's when null); a value that no longer reads is skipped and named.</summary>
    public static GenerateLabOptions Load(ISettingsStore store, out IReadOnlyList<string> skipped, GenerateLabOptions? defaults = null)
    {
        ArgumentNullException.ThrowIfNull(store);
        return LabSettings.Load(store, Prefix, Definitions, defaults?.Clone() ?? new GenerateLabOptions(), out skipped);
    }

    /// <summary>Keeps every value of <paramref name="options"/> in <paramref name="store"/> (an empty one removed: the default).</summary>
    public static void Save(ISettingsStore store, GenerateLabOptions options)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentNullException.ThrowIfNull(options);
        LabSettings.Save(store, Prefix, Definitions, options);
    }

    // The library's defaults, shown in the empty fields.
    private static SyntheticPageOptions Library => field ??= new SyntheticPageOptions { Style = new PageStyle { Fonts = [] } };

    private static IReadOnlyList<LabSetting<GenerateLabOptions>> Define() =>
    [
        new("count", "Pages at once", "The pages Generate makes from the text, each its own variant (idrak data synth -n, SyntheticPageOptions.Variants).", LabSettingKind.Number,
            LabSettings.Show(Library.Variants), o => LabSettings.Show(o.Count), (o, t) => o.Count = LabSettings.Whole(t, 1, 256)),
        new("grey", "Share in grey", "The share of pages drawn in grey, 0 to 1 (--grey, SyntheticPageOptions.Grey).", LabSettingKind.Number,
            LabSettings.Show(Library.Grey), o => LabSettings.Show(o.Grey), (o, t) => o.Grey = LabSettings.Real(t, atLeast: 0, atMost: 1)),
        new("page", "Page settings", $"KEY=VALUE,... over the page's style (--page, SyntheticPages.Style): {string.Join(", ", SyntheticPages.StyleKeys)}; ranges as MIN..MAX.",
            LabSettingKind.Text, "the library's style", o => o.Page ?? "", (o, t) => o.Page = PageSettings(t)),
        new("parts", "Parts file", "A JSON file of header, subject, signature and footer parts around the text (--parts, SyntheticPageParts); empty: the text alone.",
            LabSettingKind.Text, "none", o => o.Parts ?? "", (o, t) => o.Parts = PartsFile(t)),
        new("image_format", "Image format", "The format a page is saved and a data set's images written in (--image-format): jpeg, or a registered encoder.", LabSettingKind.Choice,
            Library.ImageFormat, o => o.ImageFormat ?? "", (o, t) => o.ImageFormat = Format(t), () => Formats),
        new("quality", "JPEG quality", "1 to 100 (--quality).", LabSettingKind.Number, LabSettings.Show(Library.Quality), o => LabSettings.Show(o.Quality),
            (o, t) => o.Quality = LabSettings.Whole(t, 1, 100)),
        new("prompt", "Data set: question", "The question of a data set's records (--prompt).", LabSettingKind.Text, Library.Prompt, o => o.Prompt ?? "",
            (o, t) => o.Prompt = LabSettings.Empty(t)),
        new("answer_key", "Data set: answer key", "Write a data set's answers as JSON under this key (--answer-key); empty: the text itself.", LabSettingKind.Text, "the text itself",
            o => o.AnswerKey ?? "", (o, t) => o.AnswerKey = LabSettings.Empty(t)),
        new("zip", "Data set: images in a zip", "on: a data set's images in a .zip beside its records (--zip); off or empty: a folder.", LabSettingKind.Choice, "default",
            o => LabSettings.Show(o.Zip), (o, t) => o.Zip = LabSettings.Switch(t), () => ["default", "on", "off"]),
        new("workers", "Data set: pages at once", "The pages a data set makes at once (--workers); empty: one per processor.", LabSettingKind.Number, "one per processor",
            o => LabSettings.Show(o.Workers), (o, t) => o.Workers = LabSettings.Whole(t, 1, 1024)),
        new("dataset_folder", "Data sets folder", "Where data sets are written (records, images, manifest); empty: datasets in the app's storage.", LabSettingKind.Text,
            "the app's storage", o => o.DatasetFolder ?? "", (o, t) => o.DatasetFolder = FullPath(t)),
        new("system_fonts", "Phone's fonts folder", "Where the phone's own fonts are listed from; empty: the platform's (Android's system fonts).", LabSettingKind.Text,
            "the platform's", o => o.SystemFonts ?? "", (o, t) => o.SystemFonts = FullPath(t)),
        new("fonts_folder", "Added fonts folder", "Where the fonts you add are copied; empty: fonts in the app's storage.", LabSettingKind.Text, "the app's storage",
            o => o.FontsFolder ?? "", (o, t) => o.FontsFolder = FullPath(t)),
        new("auto_fonts", "Fonts picked for a text", "When no font is chosen, Generate picks this many that have the text's characters (yours first).", LabSettingKind.Number, "3",
            o => LabSettings.Show(o.AutoFonts), (o, t) => o.AutoFonts = LabSettings.Whole(t, 1, 64)),
        new("auto_font_mb", "Largest font read when picking (MiB)", "A larger font file is used only when chosen by hand (reading it costs its size in memory).", LabSettingKind.Number,
            "8", o => LabSettings.Show(o.AutoFontMiB), (o, t) => o.AutoFontMiB = LabSettings.Real(t, above: 0)),
        new("album", "Pictures album", "The album under the phone's Pictures a page is saved to.", LabSettingKind.Text, "Idrak", o => o.Album ?? "", (o, t) => o.Album = LabSettings.Empty(t)),
        new("preview_px", "Preview size (px)", "The longest side of the large preview (the page keeps its size); empty: the screen's, measured.", LabSettingKind.Number,
            "the screen's", o => LabSettings.Show(o.PreviewPixels), (o, t) => o.PreviewPixels = LabSettings.Whole(t, 16)),
        new("thumb_px", "Tile size (px)", "The longest side of a page's tile in the grid.", LabSettingKind.Number, "320", o => LabSettings.Show(o.ThumbnailPixels),
            (o, t) => o.ThumbnailPixels = LabSettings.Whole(t, 16)),
    ];

    private static string? PageSettings(string text)
    {
        if (text.Length == 0)
        {
            return null;
        }

        try
        {
            _ = SyntheticPages.Style(new PageStyle { Fonts = [] }, text);
            return text;
        }
        catch (ArgumentException e)
        {
            throw new FormatException(e.Message);
        }
    }

    private static string? PartsFile(string text) =>
        text.Length == 0 ? null
        : !Path.IsPathFullyQualified(text) ? throw new FormatException($"'{text}' is not a full path.")
        : File.Exists(text) ? text
        : throw new FormatException($"The parts file {text} is not there.");

    private static string? FullPath(string text) =>
        text.Length == 0 ? null : Path.IsPathFullyQualified(text) ? text : throw new FormatException($"'{text}' is not a full path.");

    private static string? Format(string text) =>
        text.Length == 0 || text.Equals("jpeg", StringComparison.OrdinalIgnoreCase) || text.Equals("jpg", StringComparison.OrdinalIgnoreCase) ? null
        : ImageEncoders.Names.FirstOrDefault(n => n.Equals(text, StringComparison.OrdinalIgnoreCase))
          ?? throw new FormatException($"'{text}' is not an image format ({string.Join(", ", Formats)}).");
}
