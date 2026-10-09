// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json;

namespace Idrak.Gemma3Vision;

/// <summary>
/// Gemma 3's pan and scan: a tall or wide image is read as the whole image (squeezed to the encoder's square, as always)
/// followed by up to <see cref="MaxCrops"/> crops of it, cut from the original before any resize and each resized to the
/// square on its own, so a page's lines reach the encoder at a higher resolution. Each crop is one more block of
/// <see cref="Gemma3ImageTokens.TokensPerImage"/> soft tokens, with Gemma3Processor's text around them ("Here is the
/// original image ... and here are some crops to help you see better ..."). transformers' <c>do_pan_and_scan</c>
/// (<c>Gemma3ImageProcessor.pan_and_scan</c>, the same in 4.5x and 5.x), with Gemma3Processor's defaults: off, 256-pixel
/// crops at least, 4 crops at most, an aspect ratio of 1.2 at least.
/// </summary>
/// <remarks>
/// As vision options (<see cref="VisionOptions"/>, transformers' keyword names): <see cref="EnableKey"/> (or
/// <see cref="ShortEnableKey"/>), <see cref="MinCropSizeKey"/>, <see cref="MaxCropsKey"/>, <see cref="MinRatioKey"/>. The
/// model folder's <c>preprocessor_config.json</c> gives the defaults when it sets them (null: Gemma3Processor's): that is
/// what Gemma3ImageProcessor alone does with them; transformers' Gemma3Processor passes its own defaults to the image
/// processor on every call, so through it only call arguments turn pan and scan on (checked with 4.57.6 and 5.19.0).
/// </remarks>
public sealed record Gemma3PanAndScan
{
    /// <summary>The option that turns it on: <c>do_pan_and_scan</c>.</summary>
    public const string EnableKey = "do_pan_and_scan";

    /// <summary>A shorter name for <see cref="EnableKey"/>: <c>pan_and_scan</c>.</summary>
    public const string ShortEnableKey = "pan_and_scan";

    /// <summary>The smallest crop side in pixels: <c>pan_and_scan_min_crop_size</c>.</summary>
    public const string MinCropSizeKey = "pan_and_scan_min_crop_size";

    /// <summary>The most crops: <c>pan_and_scan_max_num_crops</c>.</summary>
    public const string MaxCropsKey = "pan_and_scan_max_num_crops";

    /// <summary>The aspect ratio (long side over short) from which it crops: <c>pan_and_scan_min_ratio_to_activate</c>.</summary>
    public const string MinRatioKey = "pan_and_scan_min_ratio_to_activate";

    /// <summary>The vision options it reads.</summary>
    public static IReadOnlyList<string> Keys { get; } = [EnableKey, ShortEnableKey, MinCropSizeKey, MaxCropsKey, MinRatioKey];

    /// <summary>Gemma3Processor's defaults: off, 256, 4, 1.2.</summary>
    public static Gemma3PanAndScan Default { get; } = new();

    /// <summary>Whether images are cropped (<c>do_pan_and_scan</c>); off unless asked.</summary>
    public bool Enabled { get; init; }

    /// <summary>A crop's shorter side is at least this many pixels, or the image is not cropped (default 256).</summary>
    public int MinCropSize { get; init; } = 256;

    /// <summary>The most crops of one image (default 4); at least 2 are made when the image is cropped, unless this is 1.</summary>
    public int MaxCrops { get; init; } = 4;

    /// <summary>The image is cropped only when its long side is at least this many times its short side (default 1.2).</summary>
    public double MinRatio { get; init; } = 1.2;

    /// <summary>
    /// The settings a <c>preprocessor_config.json</c> gives over Gemma3Processor's defaults (absent or null keys keep
    /// them).
    /// </summary>
    /// <exception cref="InvalidDataException">A value is of the wrong kind or out of range.</exception>
    public static Gemma3PanAndScan FromConfig(string json)
    {
        using var document = JsonDocument.Parse(json);
        var root = document.RootElement;
        if (root.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException("A preprocessor config is a JSON object.");
        }

        var result = Default;
        try
        {
            result = result with
            {
                Enabled = Value(root, EnableKey) is { } on ? on.ValueKind switch
                {
                    JsonValueKind.True => true,
                    JsonValueKind.False => false,
                    _ => throw new InvalidDataException($"{EnableKey} is true, false or null, not {on.GetRawText()}."),
                } : result.Enabled,
                MinCropSize = Value(root, MinCropSizeKey) is { } size ? WholeNumber(size, MinCropSizeKey) : result.MinCropSize,
                MaxCrops = Value(root, MaxCropsKey) is { } crops ? WholeNumber(crops, MaxCropsKey) : result.MaxCrops,
                MinRatio = Value(root, MinRatioKey) is { } ratio && ratio.ValueKind == JsonValueKind.Number ? ratio.GetDouble()
                    : Value(root, MinRatioKey) is { } other ? throw new InvalidDataException($"{MinRatioKey} is a number, not {other.GetRawText()}.") : result.MinRatio,
            };
            return result.Checked();
        }
        catch (ArgumentException ex)
        {
            throw new InvalidDataException($"preprocessor_config.json: {ex.Message}");
        }

        static JsonElement? Value(JsonElement root, string name) =>
            root.TryGetProperty(name, out var value) && value.ValueKind != JsonValueKind.Null ? value : null;

        static int WholeNumber(JsonElement value, string name) =>
            value.ValueKind == JsonValueKind.Number && value.TryGetDouble(out double d) && d == Math.Floor(d) && d is >= int.MinValue and <= int.MaxValue ? (int)d
                : throw new InvalidDataException($"{name} is a whole number, not {value.GetRawText()}.");
    }

    /// <summary>
    /// These settings with <paramref name="options"/>' keys over them; every other key is refused, naming the keys
    /// Gemma 3 takes.
    /// </summary>
    /// <exception cref="ArgumentException">An unknown key, a value of the wrong kind or out of range, or both names of the switch.</exception>
    public Gemma3PanAndScan With(VisionOptions? options)
    {
        if (options is null || options.Count == 0)
        {
            return this;
        }

        options.ThrowIfUnknown(Gemma3VisionFamily.Architecture, Keys);
        if (options.ContainsKey(EnableKey) && options.ContainsKey(ShortEnableKey) && options.Flag(EnableKey, false) != options.Flag(ShortEnableKey, false))
        {
            throw new ArgumentException($"The vision options {EnableKey} and {ShortEnableKey} (the same switch) disagree.");
        }

        return (this with
        {
            Enabled = options.Flag(EnableKey, options.Flag(ShortEnableKey, Enabled)),
            MinCropSize = options.Integer(MinCropSizeKey, MinCropSize),
            MaxCrops = options.Integer(MaxCropsKey, MaxCrops),
            MinRatio = options.Number(MinRatioKey, MinRatio),
        }).Checked();
    }

    /// <summary>
    /// The crops of an image of <paramref name="height"/> x <paramref name="width"/> pixels, in order (row by row, left
    /// to right), as rectangles of the original: none when off, when the image is not elongated enough, or when a crop
    /// would be smaller than <see cref="MinCropSize"/>. transformers' rule exactly: for a landscape or square image,
    /// round(width / height) crops across (halves up), no more than width / min crop size, at least 2, at most
    /// <see cref="MaxCrops"/>; for a portrait one the same down; each crop ⌈side / crops⌉ long, the last one cut at the
    /// image's edge.
    /// </summary>
    public IReadOnlyList<(int Top, int Left, int Height, int Width)> Crops(int height, int width)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        if (!Enabled)
        {
            return [];
        }

        int across, down;
        if (width >= height)
        {
            if ((double)width / height < MinRatio)
            {
                return [];
            }

            across = (int)Math.Floor((double)width / height + 0.5);                        // half rounds up
            across = Math.Min((int)Math.Floor((double)width / MinCropSize), across);
            across = Math.Min(MaxCrops, Math.Max(2, across));
            down = 1;
        }
        else
        {
            if ((double)height / width < MinRatio)
            {
                return [];
            }

            down = (int)Math.Floor((double)height / width + 0.5);
            down = Math.Min((int)Math.Floor((double)height / MinCropSize), down);
            down = Math.Min(MaxCrops, Math.Max(2, down));
            across = 1;
        }

        int cropWidth = (width + across - 1) / across, cropHeight = (height + down - 1) / down;
        if (Math.Min(cropWidth, cropHeight) < MinCropSize)
        {
            return [];
        }

        var crops = new List<(int, int, int, int)>(across * down);
        for (int r = 0; r < down; r++)
        {
            for (int c = 0; c < across; c++)
            {
                int top = r * cropHeight, left = c * cropWidth;
                crops.Add((top, left, Math.Min(cropHeight, height - top), Math.Min(cropWidth, width - left)));
            }
        }

        return crops;
    }

    /// <summary>"pan and scan on (crops of 256+ pixels, at most 4, from a ratio of 1.2)" or "pan and scan off".</summary>
    public override string ToString() =>
        Enabled ? $"pan and scan on (crops of {MinCropSize}+ pixels, at most {MaxCrops}, from a ratio of {MinRatio.ToString(System.Globalization.CultureInfo.InvariantCulture)})" : "pan and scan off";

    private Gemma3PanAndScan Checked() =>
        MinCropSize < 1 ? throw new ArgumentException($"{MinCropSizeKey} is a positive number of pixels, not {MinCropSize}.")
        : MaxCrops < 1 ? throw new ArgumentException($"{MaxCropsKey} is at least 1, not {MaxCrops}.")
        : !double.IsFinite(MinRatio) || MinRatio < 0 ? throw new ArgumentException($"{MinRatioKey} is a number of 0 or more, not {MinRatio}.")
        : this;
}
