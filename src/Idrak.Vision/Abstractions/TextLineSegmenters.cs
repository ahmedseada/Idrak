// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Vision.Abstractions;

/// <summary>A text line found on a page: its place (in the deskewed page's pixels) and its pixels (grey, as on the page).</summary>
/// <param name="Index">1 for the top line, then down the page.</param>
/// <param name="X">The line's left edge (padding included).</param>
/// <param name="Y">The line's top edge (padding included).</param>
/// <param name="Width">Its width in pixels.</param>
/// <param name="Height">Its height in pixels.</param>
/// <param name="Image">Its pixels: one grey channel, cut from the deskewed page.</param>
public sealed record PageLine(int Index, int X, int Y, int Width, int Height, ImageData Image);

/// <summary>The lines of a page, top to bottom, and the skew undone before they were cut.</summary>
/// <param name="Lines">The lines, top to bottom.</param>
/// <param name="Skew">The skew undone, in degrees (0 when none was).</param>
public sealed record PageLines(IReadOnlyList<PageLine> Lines, double Skew);

/// <summary>
/// How lines are found (<see cref="TextLineSegmenters"/>). Every size is relative to the page or the line except
/// <see cref="MinPadding"/>; each segmenter reads those it has (the library's "projection" reads them all), and one of
/// its own reads <see cref="Parameters"/>.
/// </summary>
public sealed record LineSegmentationOptions
{
    /// <summary>The largest skew corrected, in degrees either way (0: no deskew). Default 3.</summary>
    public double MaxSkew { get; init; } = 3;

    /// <summary>The step between the skews tried, in degrees. Default 0.2.</summary>
    public double SkewStep { get; init; } = 0.2;

    /// <summary>
    /// How much sharper (the sum of squared row counts) a skewed profile must be than the straight one to be taken, as a
    /// factor. Default 1.02.
    /// </summary>
    public double MinSkewGain { get; init; } = 1.02;

    /// <summary>The most ink pixels the skew is measured on (every n-th ink pixel when the page has more). Default 200,000.</summary>
    public int MaxSkewSamples { get; init; } = 200_000;

    /// <summary>Padding around a line, as a share of its height. Default 0.2.</summary>
    public double Padding { get; init; } = 0.2;

    /// <summary>The least padding around a line, in pixels. Default 2.</summary>
    public int MinPadding { get; init; } = 2;

    /// <summary>A row has ink when at least this share of the page's width is ink (and at least one pixel). Default 0.001.</summary>
    public double RowInk { get; init; } = 0.001;

    /// <summary>A band with less ink than this share of a typical line's is a speck and left out. Default 0.05.</summary>
    public double SpeckShare { get; init; } = 0.05;

    /// <summary>
    /// A band shorter than this share of a typical line's height (dots, marks, a broken stroke) joins its nearer neighbour
    /// when the gap is under a typical line's height. Default 0.5.
    /// </summary>
    public double MarkShare { get; init; } = 0.5;

    /// <summary>Settings of a segmenter of one's own, by name (empty for the library's).</summary>
    public IReadOnlyDictionary<string, double> Parameters { get; init; } = new Dictionary<string, double>();
}

/// <summary>
/// Cuts a page image into its text lines: each line's place and pixels, top to bottom, with the page's skew undone. The
/// library's is "projection" (<see cref="ProjectionLineSegmenter"/>); register another with
/// <c>TextLineSegmenters.Register</c> (a learned layout model, a column-aware one), and readers and apps use it by
/// name.
/// </summary>
public interface ITextLineSegmenter
{
    /// <summary>Its name in <see cref="TextLineSegmenters"/>.</summary>
    string Name { get; }

    /// <summary>The lines of <paramref name="page"/>, top to bottom, and the skew undone (degrees); no lines for a blank page.</summary>
    PageLines Find(ImageData page, LineSegmentationOptions options);
}

/// <summary>
/// The text line segmenters, by name (ignoring case). The library's: <see cref="Projection"/> (the page binarized, its skew
/// measured by the sharpest row profile and undone, rows with ink grouped into bands, marks joined to their line, each band
/// cut out with padding). Register another with <c>Register</c>; one under the library's name takes its place, which
/// stays behind it (see <c>SetPolicy</c>: an app's segmenter may fall back on the library's per page, or run beside
/// it and be compared).
/// </summary>
public static class TextLineSegmenters
{
    /// <summary>"projection": lines by horizontal projection of the binarized, deskewed page.</summary>
    public const string Projection = "projection";

    /// <summary>
    /// The lines of <paramref name="page"/>, cut by the library's projection segmenter (<see cref="ProjectionLineSegmenter"/>,
    /// reached directly: the segmenters were registered and picked by name, which is not on the new design yet).
    /// </summary>
    public static PageLines Segment(ImageData page, LineSegmentationOptions? options = null) =>
        new ProjectionLineSegmenter().Find(page, options ?? new LineSegmentationOptions());
}
