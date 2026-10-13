// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Imaging;

namespace Idrak.QwenVision;

/// <summary>
/// Qwen-VL's image processor (Qwen2VLImageProcessor): a page resized to a height and width that are multiples of
/// patch × merge within the pixel budget (<see cref="SmartResize"/>: <see cref="MinPixels"/> ≤ h × w ≤
/// <see cref="MaxPixels"/>, the aspect ratio kept), with Pillow's bicubic filter on the decoded bytes, rescaled and
/// normalized with the processor's mean and std (the library's <see cref="ImagePreprocessor"/>, Pillow's bytes exactly),
/// then cut into patches in the tower's order: merge block by merge block, row-major, each patch's values channel by
/// channel, its frame repeated to fill a temporal patch, row by row.
/// </summary>
public sealed class QwenImageProcessor
{
    /// <summary>The vision option and preprocessor_config.json key of the smallest pixel count.</summary>
    public const string MinPixelsKey = "min_pixels";

    /// <summary>The vision option and preprocessor_config.json key of the largest pixel count.</summary>
    public const string MaxPixelsKey = "max_pixels";

    private static readonly float[] ClipMean = [0.48145466f, 0.4578275f, 0.40821073f];
    private static readonly float[] ClipStd = [0.26862954f, 0.26130258f, 0.27577711f];

    /// <summary>A processor for one configuration and pixel budget.</summary>
    /// <exception cref="ArgumentException">The budget is not positive, or its least exceeds its most.</exception>
    public QwenImageProcessor(QwenVisionConfig config, int minPixels, int maxPixels, IReadOnlyList<float>? mean = null, IReadOnlyList<float>? std = null, bool grayscale = false)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (minPixels <= 0 || maxPixels < minPixels)
        {
            throw new ArgumentException($"The pixel budget {minPixels}..{maxPixels} is not a positive range ({MinPixelsKey} ≤ {MaxPixelsKey}).");
        }

        (Config, MinPixels, MaxPixels, Mean, Std, Grayscale) = (config, minPixels, maxPixels, mean ?? ClipMean, std ?? ClipStd, grayscale);
    }

    /// <summary>The tower's configuration (patch, merge and temporal sizes, channels).</summary>
    public QwenVisionConfig Config { get; }

    /// <summary>The smallest pixel count a resized image keeps.</summary>
    public int MinPixels { get; }

    /// <summary>The largest pixel count a resized image may have: what an app lowers for memory and speed.</summary>
    public int MaxPixels { get; }

    /// <summary>The per-channel mean subtracted after rescaling.</summary>
    public IReadOnlyList<float> Mean { get; }

    /// <summary>The per-channel std divided by.</summary>
    public IReadOnlyList<float> Std { get; }

    /// <summary>Whether images turn grayscale first (three equal channels).</summary>
    public bool Grayscale { get; }

    /// <summary>
    /// The processor of a model folder: its <c>preprocessor_config.json</c> (<c>min_pixels</c> and <c>max_pixels</c>, or
    /// transformers 5's <c>size.shortest_edge</c> and <c>size.longest_edge</c>; <c>image_mean</c>, <c>image_std</c>), absent
    /// keys and an absent file taking Qwen2VLImageProcessor's defaults (56 × 56 to 28 × 28 × 1280 pixels, CLIP's mean and std).
    /// </summary>
    public static QwenImageProcessor FromFolder(string? folder, QwenVisionConfig config, bool grayscale = false)
    {
        int min = 56 * 56, max = 28 * 28 * 1280;
        IReadOnlyList<float>? mean = null, std = null;
        string? path = folder is null ? null : Path.Combine(folder, "preprocessor_config.json");
        if (path is not null && File.Exists(path))
        {
            var json = JsonNode.Parse(File.ReadAllText(path))?.AsObject() ?? [];
            min = (int?)json[MinPixelsKey] ?? (int?)json["size"]?["shortest_edge"] ?? min;
            max = (int?)json[MaxPixelsKey] ?? (int?)json["size"]?["longest_edge"] ?? max;
            mean = json["image_mean"] is JsonArray m ? [.. m.Select(v => (float)v!)] : null;
            std = json["image_std"] is JsonArray s ? [.. s.Select(v => (float)v!)] : null;
            int patch = (int?)json["patch_size"] ?? config.PatchSize, merge = (int?)json["merge_size"] ?? config.MergeSize;
            int temporal = (int?)json["temporal_patch_size"] ?? config.TemporalPatchSize;
            if (patch != config.PatchSize || merge != config.MergeSize || temporal != config.TemporalPatchSize)
            {
                throw new InvalidDataException($"{path}: patch {patch}, merge {merge}, temporal {temporal} differ from vision_config's "
                    + $"{config.PatchSize}, {config.MergeSize}, {config.TemporalPatchSize}.");
            }
        }

        return new QwenImageProcessor(config, min, max, mean, std, grayscale);
    }

    /// <summary>
    /// This processor with a request's options over it: <c>min_pixels</c> and <c>max_pixels</c> (whole numbers); any
    /// other key is refused, naming those it takes.
    /// </summary>
    /// <exception cref="ArgumentException">An unknown key, or a value that is not a whole number or out of range.</exception>
    public QwenImageProcessor With(VisionOptions? options)
    {
        if (options is null || options.Count == 0)
        {
            return this;
        }

        int min = MinPixels, max = MaxPixels;
        foreach (var (key, value) in options)
        {
            if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int pixels) || pixels <= 0)
            {
                throw new ArgumentException($"Vision option {key}={value}: a positive whole number of pixels.", nameof(options));
            }

            switch (key)
            {
                case MinPixelsKey:
                    min = pixels;
                    break;
                case MaxPixelsKey:
                    max = pixels;
                    break;
                default:
                    throw new ArgumentException($"Vision option '{key}' is not one Qwen-VL takes ({MinPixelsKey}, {MaxPixelsKey}).", nameof(options));
            }
        }

        return min == MinPixels && max == MaxPixels ? this : new QwenImageProcessor(Config, Math.Min(min, max), max, Mean, Std, Grayscale);
    }

    /// <summary>
    /// The size an image of <paramref name="height"/> × <paramref name="width"/> is resized to (transformers'
    /// <c>smart_resize</c>): each side rounded to a multiple of patch × merge (halves to even, as Python rounds), then
    /// shrunk or grown by one factor into the budget, rounding down or up to the multiple.
    /// </summary>
    /// <exception cref="ArgumentException">One side is more than 200 times the other.</exception>
    public (int Height, int Width) SmartResize(int height, int width)
    {
        int factor = Config.PatchSize * Config.MergeSize;
        if (height <= 0 || width <= 0 || Math.Max(height, width) / (double)Math.Min(height, width) > 200)
        {
            throw new ArgumentException($"An image of {width} x {height}: its sides must be positive and within 200 times each other.");
        }

        int h = (int)Math.Round(height / (double)factor, MidpointRounding.ToEven) * factor;
        int w = (int)Math.Round(width / (double)factor, MidpointRounding.ToEven) * factor;
        if ((long)h * w > MaxPixels)
        {
            double beta = Math.Sqrt(height * (double)width / MaxPixels);
            h = Math.Max(factor, (int)Math.Floor(height / beta / factor) * factor);
            w = Math.Max(factor, (int)Math.Floor(width / beta / factor) * factor);
        }
        else if ((long)h * w < MinPixels)
        {
            double beta = Math.Sqrt(MinPixels / (height * (double)width));
            h = (int)Math.Ceiling(height * beta / factor) * factor;
            w = (int)Math.Ceiling(width * beta / factor) * factor;
        }

        return (h, w);
    }

    /// <summary>The patch grid (one frame) an image becomes, before merging.</summary>
    public QwenImageGrid Grid(ImageData image)
    {
        ArgumentNullException.ThrowIfNull(image);
        var (h, w) = SmartResize(image.Height, image.Width);
        return new QwenImageGrid(1, h / Config.PatchSize, w / Config.PatchSize);
    }

    /// <summary>
    /// The pixel patches of one image, [rows · columns of its grid, <see cref="QwenVisionConfig.PatchValues"/>], into
    /// <paramref name="destination"/> (at least that long), and its grid.
    /// </summary>
    public QwenImageGrid Patches(ImageData image, Span<float> destination)
    {
        var grid = Grid(image);
        int p = Config.PatchSize, m = Config.MergeSize, t = Config.TemporalPatchSize, c = Config.Channels;
        int height = grid.Rows * p, width = grid.Columns * p, values = Config.PatchValues;
        if (destination.Length < grid.Patches * values)
        {
            throw new ArgumentException($"The destination holds {destination.Length} values; the patches need {grid.Patches * values}.", nameof(destination));
        }

        var resize = new ImagePreprocessor
        {
            Height = height, Width = width, Resize = true, Resampling = ImageResampling.Bicubic, ConvertRgb = true, Rescale = true, RescaleFactor = 1.0 / 255,
            Normalize = true, Mean = Mean, Std = Std, Grayscale = Grayscale,
        };
        int planeSize = c * height * width;
        float[] pixels = ArrayPool<float>.Shared.Rent(planeSize);                     // [channels, height, width], given back below
        try
        {
            int written = resize.Pixels(image, pixels.AsSpan(0, planeSize));
            if (written != planeSize)
            {
                throw new InvalidOperationException($"The preprocessor wrote {written} values for {c} x {height} x {width}.");
            }

            // Merge block (by, bx) row-major, then the block's patches (my, mx), then channel, frame, patch row, patch column.
            int at = 0;
            for (int by = 0; by < grid.Rows / m; by++)
            {
                for (int bx = 0; bx < grid.Columns / m; bx++)
                {
                    for (int my = 0; my < m; my++)
                    {
                        for (int mx = 0; mx < m; mx++)
                        {
                            int top = (by * m + my) * p, left = (bx * m + mx) * p;
                            for (int ch = 0; ch < c; ch++)
                            {
                                var plane = pixels.AsSpan(ch * height * width, height * width);
                                int frame = at;
                                for (int y = 0; y < p; y++)
                                {
                                    plane.Slice((top + y) * width + left, p).CopyTo(destination.Slice(at, p));
                                    at += p;
                                }

                                for (int f = 1; f < t; f++)
                                {
                                    destination.Slice(frame, p * p).CopyTo(destination.Slice(at, p * p));
                                    at += p * p;
                                }
                            }
                        }
                    }
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(pixels);
        }

        return grid;
    }

    /// <inheritdoc />
    public override string ToString() => $"QwenImageProcessor(patch {Config.PatchSize}, merge {Config.MergeSize}, {MinPixels}..{MaxPixels} pixels{(Grayscale ? ", grayscale" : "")})";
}

/// <summary>The patch grid of one image before merging: frames, rows and columns of patches (transformers' <c>image_grid_thw</c>).</summary>
/// <param name="Frames">Temporal patches (1 for a still image).</param>
/// <param name="Rows">Patch rows (the resized height / patch size).</param>
/// <param name="Columns">Patch columns (the resized width / patch size).</param>
public sealed record QwenImageGrid(int Frames, int Rows, int Columns)
{
    /// <summary>Patches in the grid: the tower's sequence length for this image.</summary>
    public int Patches => Frames * Rows * Columns;

    /// <summary>The tokens the image becomes after merging <paramref name="mergeSize"/>² patches each: the decoder's image length.</summary>
    public int Tokens(int mergeSize) => Frames * (Rows / mergeSize) * (Columns / mergeSize);

    /// <summary>The token layout the decoder sees: the merged grid [rows / merge, columns / merge].</summary>
    public ImageTokenLayout Layout(int mergeSize) => new(Tokens(mergeSize)) { Grid = [Rows / mergeSize, Columns / mergeSize] };
}

// A store showing some tensors of another under other names (the vision part's names, without the checkpoint's prefix).
internal sealed class RenamedTensorStore(ITensorStore inner, Dictionary<string, string> names) : ITensorStore
{
    public IEnumerable<string> Names => names.Keys;

    public bool Contains(string name) => names.ContainsKey(name);

    public int[] ShapeOf(string name) => inner.ShapeOf(Stored(name));

    public float[] Read(string name) => inner.Read(Stored(name));

    public float[] ReadTransposed(string name) => inner.ReadTransposed(Stored(name));

    public WeightFormat? FormatOf(string name) => inner.FormatOf(Stored(name));

    public bool ReadBFloat16(string name, bool transposed, Span<ushort> destination, int stride) => inner.ReadBFloat16(Stored(name), transposed, destination, stride);

    public void Dispose() => inner.Dispose();

    private string Stored(string name) => names.TryGetValue(name, out var stored) ? stored : throw new KeyNotFoundException($"No vision tensor '{name}'.");
}
