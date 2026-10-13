// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.QwenVision;

/// <summary>
/// Qwen-VL's image processor (Qwen2VLImageProcessor), the family's own preprocessing: the smart resize of a page to a
/// height and width that are multiples of patch × merge, within the pixel budget (<c>min_pixels</c> ≤ h × w ≤
/// <c>max_pixels</c>, keeping the aspect ratio), rescale and normalize with the processor's mean and std, then the
/// pixels cut into patches ordered for the tower: [grid_t × grid_h × grid_w, channels × temporal × patch × patch],
/// with the grid (t, h, w) the position rule and the prompt format count from. A still image fills one temporal patch
/// (its pixels repeated). Pillow's bytes exactly, like the library's transforms; on bytes until the normalize
/// (plan 17, lane C). Built in plan 16, step 5; the fixture of step 0 gives the pixels and the grid to match.
/// </summary>
public sealed class QwenImageProcessor
{
    /// <summary>A processor for one configuration, with the pixel budget the model folder's <c>preprocessor_config.json</c> sets or the processor's defaults.</summary>
    public QwenImageProcessor(QwenVisionConfig config, int minPixels, int maxPixels) => (Config, MinPixels, MaxPixels) = (config, minPixels, maxPixels);

    /// <summary>The tower's configuration (patch, merge and temporal sizes).</summary>
    public QwenVisionConfig Config { get; }

    /// <summary>The smallest pixel count a resized image keeps (<c>min_pixels</c>).</summary>
    public int MinPixels { get; }

    /// <summary>The largest pixel count a resized image may have (<c>max_pixels</c>); the app lowers it for memory.</summary>
    public int MaxPixels { get; }

    /// <summary>This processor with a request's options over it (<c>min_pixels</c>, <c>max_pixels</c>); a value out of range is refused.</summary>
    public QwenImageProcessor With(VisionOptions? options) => options is null ? this : throw QwenVisionPlugin.NotBuilt("vision options over the processor", 5);

    /// <summary>The size an image is resized to: multiples of patch × merge within the budget, the aspect ratio kept (transformers' smart_resize).</summary>
    public (int Height, int Width) SmartResize(int height, int width) => throw QwenVisionPlugin.NotBuilt("the smart resize", 5);

    /// <summary>The grid (t, h, w) of patches an image becomes, before merging.</summary>
    public QwenImageGrid Grid(ImageData image) => throw QwenVisionPlugin.NotBuilt("the patch grid", 5);

    /// <summary>The pixel patches of one image, [t × h × w, channels × temporal × patch × patch], and its grid.</summary>
    public (Tensor Patches, QwenImageGrid Grid) Patches(ImageData image, Device device) => throw QwenVisionPlugin.NotBuilt("the pixel patches", 5);

    /// <inheritdoc />
    public override string ToString() => $"QwenImageProcessor(patch {Config.PatchSize}, merge {Config.MergeSize}, {MinPixels}..{MaxPixels} pixels)";
}

/// <summary>The patch grid of one image before merging: frames, rows and columns of patches (transformers' <c>image_grid_thw</c>).</summary>
/// <param name="Frames">Temporal patches (1 for a still image).</param>
/// <param name="Rows">Patch rows (the resized height / patch size).</param>
/// <param name="Columns">Patch columns (the resized width / patch size).</param>
public sealed record QwenImageGrid(int Frames, int Rows, int Columns)
{
    /// <summary>Patches in the grid, the tower's sequence length for this image.</summary>
    public int Patches => Frames * Rows * Columns;

    /// <summary>The tokens the image becomes after merging <paramref name="mergeSize"/>² patches each: the decoder's image length.</summary>
    public int Tokens(int mergeSize) => Frames * (Rows / mergeSize) * (Columns / mergeSize);

    /// <summary>The token layout the decoder sees: the merged grid [rows / merge, columns / merge] of <see cref="Tokens"/> tokens.</summary>
    public ImageTokenLayout Layout(int mergeSize) => new(Tokens(mergeSize)) { Grid = [Rows / mergeSize, Columns / mergeSize] };
}
