// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Data;

/// <summary>
/// A glyph's ink as coverage: how much of each pixel is ink, from 0 (paper) to 1 (full ink), <see cref="Height"/> rows
/// of <see cref="Width"/>, row 0 at the top. It is what <see cref="IOutlineRasterizer.Fill(GlyphOutline, float, float, float, Span{float}, int, int)"/> writes and an
/// <see cref="IOutlineTracer"/> reads back, and what an <see cref="IGlyphSource"/> cuts out of a page for each glyph.
/// The values are checked once on the way in, in the same pass that finds the box around the ink
/// (<see cref="InkBox"/>); the coverage is not changed after, so it is safe to read from several threads.
/// </summary>
public sealed class GlyphCoverage
{
    private readonly float[] _values;

    /// <summary>The coverage <paramref name="values"/> (row by row, used directly, not copied: the caller does not change them after).</summary>
    /// <param name="values"><paramref name="height"/> rows of <paramref name="width"/> values, each from 0 to 1.</param>
    /// <param name="width">Pixels per row (more than 0).</param>
    /// <param name="height">Rows (more than 0).</param>
    /// <exception cref="ArgumentException">Not <paramref name="width"/> x <paramref name="height"/> values, or a value not from 0 to 1 (the message names its pixel).</exception>
    public GlyphCoverage(float[] values, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(values);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        if (values.Length != (long)width * height)
        {
            throw new ArgumentException($"{values.Length} values are not {width} x {height}.", nameof(values));
        }

        // One pass in memory order: every value checked, the ink's first and last column of each row kept.
        int left = width, top = -1, right = 0, bottom = 0;
        for (int y = 0; y < height; y++)
        {
            var row = values.AsSpan(y * width, width);
            int first = -1, last = -1;
            for (int x = 0; x < row.Length; x++)
            {
                float v = row[x];
                if (!(v >= 0f && v <= 1f))
                {
                    throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"Coverage is from 0 to 1; pixel ({x}, {y}) holds {v}."), nameof(values));
                }

                if (v > 0f)
                {
                    first = first < 0 ? x : first;
                    last = x;
                }
            }

            if (first >= 0)
            {
                left = Math.Min(left, first);
                right = Math.Max(right, last + 1);
                top = top < 0 ? y : top;
                bottom = y + 1;
            }
        }

        _values = values;
        Width = width;
        Height = height;
        InkBox = top < 0 ? default : new PixelBox(left, top, right - left, bottom - top);
    }

    /// <summary>Coverage of <paramref name="width"/> x <paramref name="height"/> pixels with no ink.</summary>
    public static GlyphCoverage Blank(int width, int height)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        if ((long)width * height > Array.MaxLength)
        {
            throw new ArgumentException($"{width} x {height} pixels are more than an array holds.", nameof(width));
        }

        return new GlyphCoverage(new float[width * height], width, height);
    }

    /// <summary>Pixels per row.</summary>
    public int Width { get; }

    /// <summary>Rows.</summary>
    public int Height { get; }

    /// <summary>The values, row by row, each from 0 (paper) to 1 (full ink).</summary>
    public ReadOnlyMemory<float> Values => _values;

    /// <summary>Row <paramref name="y"/> (0 at the top).</summary>
    public ReadOnlySpan<float> Row(int y)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(y);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(y, Height);
        return _values.AsSpan(y * Width, Width);
    }

    /// <summary>The ink at pixel (<paramref name="x"/>, <paramref name="y"/>).</summary>
    public float this[int x, int y] => (uint)x < (uint)Width && (uint)y < (uint)Height
        ? _values[y * Width + x]
        : throw new ArgumentOutOfRangeException(nameof(x), $"Pixel ({x}, {y}) lies outside the coverage's {Width} x {Height}.");

    /// <summary>The smallest box around every pixel with any ink; all zero when there is none.</summary>
    public PixelBox InkBox { get; }

    /// <summary>Whether no pixel holds any ink.</summary>
    public bool IsBlank => InkBox.Width == 0;

    /// <summary>A copy of the part <paramref name="box"/> (inside the coverage).</summary>
    /// <exception cref="ArgumentOutOfRangeException">The box is empty or reaches outside the coverage.</exception>
    public GlyphCoverage Crop(PixelBox box)
    {
        if (box.Width <= 0 || box.Height <= 0 || box.X < 0 || box.Y < 0 || box.Right > Width || box.Bottom > Height)
        {
            throw new ArgumentOutOfRangeException(nameof(box), $"{box} is empty or reaches outside the coverage's {Width} x {Height}.");
        }

        var part = new float[box.Width * box.Height];
        for (int y = 0; y < box.Height; y++)
        {
            _values.AsSpan((box.Y + y) * Width + box.X, box.Width).CopyTo(part.AsSpan(y * box.Width, box.Width));
        }

        return new GlyphCoverage(part, box.Width, box.Height);
    }
}

/// <summary>
/// Where coverage sits against font units, as <see cref="IOutlineRasterizer.Fill(GlyphOutline, float, float, float, Span{float}, int, int)"/> takes it: <see cref="Scale"/> pixels
/// per font unit, and the glyph's origin (on its baseline) at pixel (<see cref="X"/>, <see cref="Y"/>); y grows downwards
/// in pixels and upwards in font units. A tracer gives the outline that, filled with the same placement, covers the same
/// pixels.
/// </summary>
/// <param name="Scale">Pixels per font unit (more than 0): a glyph written <c>s</c> pixels per em in a face of <c>u</c> units per em is <c>s / u</c>.</param>
/// <param name="X">The column of the glyph's origin, in pixels from the coverage's left edge.</param>
/// <param name="Y">The row of the glyph's baseline, in pixels from the coverage's top edge.</param>
public readonly record struct OutlinePlacement(float Scale, float X, float Y)
{
    /// <summary>Whether the scale is more than 0 and every value finite.</summary>
    public bool IsValid => Scale > 0f && float.IsFinite(Scale) && float.IsFinite(X) && float.IsFinite(Y);

    /// <summary>The pixel point (<paramref name="x"/>, <paramref name="y"/>) in font units.</summary>
    public (float X, float Y) ToUnits(float x, float y) => ((x - X) / Scale, (Y - y) / Scale);

    /// <summary>The font-unit point (<paramref name="x"/>, <paramref name="y"/>) in pixels.</summary>
    public (float X, float Y) ToPixels(float x, float y) => (X + x * Scale, Y - y * Scale);
}

/// <summary>
/// An outline tracer's settings, read once from its options (<see cref="IOutlineTracer.Read"/>) and used for every glyph
/// traced with them: the options as given, and whatever a tracer parses them into (a tracer's own settings derive from
/// this class).
/// </summary>
public class OutlineTracerSettings
{
    /// <summary>Settings of <paramref name="options"/> (none when null).</summary>
    public OutlineTracerSettings(IReadOnlyDictionary<string, string>? options = null) => Options = options ?? PageOptions.None;

    /// <summary>No options: a tracer's defaults.</summary>
    public static OutlineTracerSettings None { get; } = new();

    /// <summary>The options as given, by key.</summary>
    public IReadOnlyDictionary<string, string> Options { get; }
}

/// <summary>
/// The stage from a glyph's ink to its outline: coverage (<see cref="GlyphCoverage"/>) in, a <see cref="GlyphOutline"/>
/// in font units (y up) out, which a face made from glyphs holds and a font writer writes. The outline has one winding:
/// no point lies inside two contours that turn the same way, the outer contours turn one way and their holes the other,
/// so the nonzero and even-odd rules fill it alike; holes are kept. Filled again with the same placement, it covers the
/// coverage's ink to within <see cref="Tolerance"/>. Its settings (thresholds, smoothing, corner angle, simplification)
/// are read once (<see cref="Read"/>) and declare that tolerance. The same coverage, placement and settings give the
/// same outline, from any thread.
/// </summary>
public interface IOutlineTracer
{
    /// <summary>The tracer's settings from <paramref name="options"/> (its own keys, <c>Keys</c>; defaults for those not given).</summary>
    /// <exception cref="ArgumentException">A key it does not take (the message names its keys), or a bad
    /// value.</exception>
    OutlineTracerSettings Read(IReadOnlyDictionary<string, string> options);

    /// <summary>
    /// The most, in pixels, the edge of an outline traced with <paramref name="settings"/> strays from the coverage's edge
    /// (where the ink is half): 0 or more, finite.
    /// </summary>
    float Tolerance(OutlineTracerSettings settings);

    /// <summary>
    /// The outline of <paramref name="coverage"/>'s ink, in font units placed by <paramref name="placement"/>, with one
    /// winding and its holes kept; <see cref="GlyphOutline.Empty"/> for coverage with no ink.
    /// </summary>
    /// <exception cref="ArgumentException">The placement is not valid (<see cref="OutlinePlacement.IsValid"/>).</exception>
    GlyphOutline Trace(GlyphCoverage coverage, OutlinePlacement placement, OutlineTracerSettings settings);
}

/// <summary>
/// The name of the library's outline tracer (Idrak.Imaging): "contour" (<see cref="Contour"/>: the ink's edge at a
/// level, Otsu's unless set, traced in one pass, simplified into lines and quadratic curves with its corners kept).
/// Picking one by its name is not on the new design yet.
/// </summary>
public static class OutlineTracers
{
    /// <summary>The library's tracer (Idrak.Imaging): the ink's edge at a level, simplified into lines and quadratic curves, its corners kept.</summary>
    public const string Contour = "contour";
}
