// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Text;
using Idrak.Text.Generation;

namespace Idrak.Maui;

/// <summary>
/// The OCR lab's settings: the reader picked, the instruction and the system text, the preprocessing pipeline, the
/// answer's length and its repetition stop, the weights' and the attention cache's formats, the context and the size of
/// the picture on the page. A null value is the library's own default (<see cref="ImageReadRequest"/>,
/// <see cref="ImageReaderOptions"/>, <see cref="GenerationOptions"/>, the reading pages' instruction
/// <see cref="SyntheticPageOptions.Prompt"/>), or for the picture the screen's size measured at run time. Kept between runs
/// by <see cref="OcrSettings"/>.
/// </summary>
public sealed class OcrLabOptions
{
    /// <summary>The reader chosen, as the model store names it ("owner/name@revision"); null: none yet.</summary>
    public string? Model { get; set; }

    /// <summary>The instruction sent after the page (<see cref="ImageReadRequest.Prompt"/>); null: the library's (<see cref="OcrSettings.LibraryPrompt"/>).</summary>
    public string? Prompt { get; set; }

    /// <summary>A system turn before the page (<see cref="ImageReadRequest.System"/>); null: none.</summary>
    public string? System { get; set; }

    /// <summary>The preprocessing pipeline run on the page before the family's own processor (<see cref="ImageTransformPipeline"/> text); null: none.</summary>
    public string? Pipeline { get; set; }

    /// <summary>The most tokens the answer takes (<see cref="ImageReadRequest.MaxTokens"/>); null: the library's.</summary>
    public int? MaxTokens { get; set; }

    /// <summary>Ends an answer that repeats one run of tokens this many times in a row (<see cref="ImageReadRequest.RepeatStop"/>, 0 off); null: the library's.</summary>
    public int? RepeatStop { get; set; }

    /// <summary>The decoder's projections' format once loaded: "bfloat16", "int8", "int4" or a registered packed format; null: the library's (float32).</summary>
    public string? Weights { get; set; }

    /// <summary>The attention cache's format (<see cref="ImageReaderOptions.CacheFormat"/>); null: the library's.</summary>
    public KeyValueFormat? CacheFormat { get; set; }

    /// <summary>The longest prompt and answer in tokens, the page's image tokens included (<see cref="ImageReaderOptions.ContextLength"/>); null: the library's generation context (<see cref="GenerationOptions.NumCtx"/>).</summary>
    public int? Context { get; set; }

    /// <summary>The longest side, in pixels, of the page's picture on the screen (shown only); null: the screen's, measured.</summary>
    public int? PreviewPixels { get; set; }

    /// <summary>A copy.</summary>
    public OcrLabOptions Clone() => (OcrLabOptions)MemberwiseClone();

    /// <summary>The pipeline of <see cref="Pipeline"/>, checked against the registered transforms (empty when none).</summary>
    /// <exception cref="ArgumentException">An unknown transform or option, or a value the transform refuses.</exception>
    public ImageTransformPipeline ToPipeline() => ImageTransformPipeline.Parse(Pipeline);

    /// <summary>The read of <paramref name="page"/>: the instruction, the system text, the pipeline, the length and the stop over the library's defaults.</summary>
    public ImageReadRequest ToRequest(ChatImage page)
    {
        ArgumentNullException.ThrowIfNull(page);
        var library = new ImageReadRequest(page, OcrSettings.LibraryPrompt);
        return library with
        {
            Prompt = string.IsNullOrWhiteSpace(Prompt) ? library.Prompt : Prompt,
            System = string.IsNullOrWhiteSpace(System) ? library.System : System,
            Transforms = ToPipeline(),
            MaxTokens = MaxTokens ?? library.MaxTokens,
            RepeatStop = RepeatStop ?? library.RepeatStop,
        };
    }

    /// <summary>What a model at <paramref name="path"/> is loaded as with these settings on <paramref name="device"/>.</summary>
    public VisionLoadRequest ToLoad(string model, string path, string device) =>
        new(model, path, device, Weights, CacheFormat ?? new ImageReaderOptions().CacheFormat, Context ?? new GenerationOptions().NumCtx);
}

/// <summary>
/// The OCR lab's settings kept between runs, each under <see cref="Prefix"/> and its key in the app's
/// <see cref="ISettingsStore"/> (MAUI's Preferences): the reader the page chose, and every <see cref="Definitions"/> value.
/// </summary>
public static class OcrSettings
{
    /// <summary>What every key starts with.</summary>
    public const string Prefix = IdrakMauiSettings.Prefix + "ocr.";

    private const string ModelKey = Prefix + "model";

    /// <summary>The library's instruction for reading a page: the one its synthetic reading pages ask (<see cref="SyntheticPageOptions.Prompt"/>).</summary>
    public static string LibraryPrompt { get; } = new SyntheticPageOptions { Style = null! }.Prompt;   // the default read alone: no page is made, no style read

    /// <summary>The settings edited on the page, in its order.</summary>
    public static IReadOnlyList<LabSetting<OcrLabOptions>> Definitions { get; } = Define();

    /// <summary>
    /// The options saved in <paramref name="store"/>, over <paramref name="defaults"/> (a copy; the library's defaults when
    /// null); a saved value that no longer reads is skipped and named.
    /// </summary>
    public static OcrLabOptions Load(ISettingsStore store, out IReadOnlyList<string> skipped, OcrLabOptions? defaults = null)
    {
        ArgumentNullException.ThrowIfNull(store);
        var options = LabSettings.Load(store, Prefix, Definitions, defaults?.Clone() ?? new OcrLabOptions(), out skipped);
        if (store.Get(ModelKey) is { } model)
        {
            options.Model = LabSettings.Empty(model);
        }

        return options;
    }

    /// <summary>Keeps every value of <paramref name="options"/> in <paramref name="store"/> (an empty one removed: the default).</summary>
    public static void Save(ISettingsStore store, OcrLabOptions options)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentNullException.ThrowIfNull(options);
        store.Set(ModelKey, LabSettings.Empty(options.Model));
        LabSettings.Save(store, Prefix, Definitions, options);
    }

    private static IReadOnlyList<LabSetting<OcrLabOptions>> Define()
    {
        var request = new ImageReadRequest(ChatImage.FromBytes([0]), LibraryPrompt);
        var reader = new ImageReaderOptions();
        string[] caches = [.. Enum.GetNames<KeyValueFormat>().Select(n => n.ToLowerInvariant())];
        return
        [
            new("prompt", "Instruction", "Sent after the page in the user turn (ImageReadRequest.Prompt): what to read and how to answer.",
                LabSettingKind.Text, LibraryPrompt, o => o.Prompt ?? "", (o, t) => o.Prompt = t.Length == 0 ? null : t),
            new("system", "System text", "A system turn before the page (ImageReadRequest.System); empty: none.",
                LabSettingKind.Text, "none", o => o.System ?? "", (o, t) => o.System = t.Length == 0 ? null : t),
            new("pipeline", "Preprocessing", "Image transforms run on the page before the model's own processor, such as grayscale,max_width=1024,contrast=1.5 (ImageTransformPipeline).",
                LabSettingKind.Text, "none", o => o.Pipeline ?? "", (o, t) => o.Pipeline = LabSettings.Pipeline(t)),
            new("max_tokens", "Most answer tokens", "The most tokens the answer takes (ImageReadRequest.MaxTokens).",
                LabSettingKind.Number, LabSettings.Show(request.MaxTokens), o => LabSettings.Show(o.MaxTokens), (o, t) => o.MaxTokens = LabSettings.Whole(t, 1)),
            new("repeat_stop", "Repetition stop", "Ends an answer that repeats one run of tokens this many times in a row; 0: off (ImageReadRequest.RepeatStop).",
                LabSettingKind.Number, LabSettings.Show(request.RepeatStop), o => LabSettings.Show(o.RepeatStop), (o, t) => o.RepeatStop = LabSettings.Whole(t, 0)),
            new("weights", "Weights", "The decoder's projections once loaded (bfloat16, int8, int4 or a registered packed format); empty: the library's float32. Taken at the next load.",
                LabSettingKind.Choice, "default (float32)", o => o.Weights ?? "", (o, t) => o.Weights = LabSettings.Registered(t, Weights(), "a weights format"), () => ["default", .. Weights()]),
            new("cache_format", "Attention cache", "The attention cache's format (ImageReaderOptions.CacheFormat). Taken at the next load.", LabSettingKind.Choice,
                $"default ({reader.CacheFormat.ToString().ToLowerInvariant()})", o => o.CacheFormat?.ToString().ToLowerInvariant() ?? "",
                (o, t) => o.CacheFormat = LabSettings.Registered(t, caches, "a cache format") is { } c ? Enum.Parse<KeyValueFormat>(c, ignoreCase: true) : null, () => ["default", .. caches]),
            new("context", "Context (tokens)", "The longest prompt and answer, the page's image tokens included; it sizes the attention cache the model is measured with. Taken at the next load.",
                LabSettingKind.Number, LabSettings.Show(new GenerationOptions().NumCtx), o => LabSettings.Show(o.Context), (o, t) => o.Context = LabSettings.Whole(t, 16)),
            new("preview_px", "Picture size on the page (px)", "The longest side the page's picture is shrunk to on the screen only (the model reads the page itself); empty: the screen's, measured.",
                LabSettingKind.Number, "the screen's", o => LabSettings.Show(o.PreviewPixels), (o, t) => o.PreviewPixels = LabSettings.Whole(t, 16)),
        ];
    }

    // The weights formats a load takes: the library's three and every registered packed format.
    private static IReadOnlyList<string> Weights() =>
        [.. new[] { "bfloat16", "int8", "int4" }.Concat(PackedWeightFormats.Names).Distinct(StringComparer.OrdinalIgnoreCase)];
}
