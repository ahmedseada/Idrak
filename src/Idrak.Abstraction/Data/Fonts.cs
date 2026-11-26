// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Data;

/// <summary>A font's vertical metrics, in font units (y up: the ascender is above the baseline, the descender below, negative).</summary>
/// <param name="UnitsPerEm">Font units per em.</param>
/// <param name="Ascender">Top of the tallest letters above the baseline.</param>
/// <param name="Descender">Bottom of the descenders (negative: below the baseline).</param>
/// <param name="LineGap">Extra space between lines.</param>
public readonly record struct FontMetrics(int UnitsPerEm, int Ascender, int Descender, int LineGap)
{
    /// <summary>The distance from one baseline to the next: ascender, minus descender, plus line gap.</summary>
    public int LineHeight => Ascender - Descender + LineGap;
}

/// <summary>What a step of an outline does; each takes its points from <see cref="GlyphOutline.Points"/> in order.</summary>
public enum OutlineVerb : byte
{
    /// <summary>Starts a contour at one point.</summary>
    Move,

    /// <summary>A straight line to one point.</summary>
    Line,

    /// <summary>A quadratic curve: a control point, then the end point.</summary>
    Quadratic,

    /// <summary>A cubic curve: two control points, then the end point.</summary>
    Cubic,

    /// <summary>Closes the contour (a line back to its start); no point.</summary>
    Close,
}

/// <summary>
/// A glyph's outline in font units (y up): contours of lines and curves, filled by the nonzero winding rule.
/// <see cref="Points"/> holds x, y pairs: one for a move or a line, two for a quadratic curve, three for a cubic.
/// </summary>
public sealed class GlyphOutline
{
    /// <summary>An outline with no contour (a space).</summary>
    public static readonly GlyphOutline Empty = new([], []);

    /// <summary>An outline of <paramref name="verbs"/> and their points (x, y pairs; used directly, not copied).</summary>
    /// <exception cref="ArgumentException">The points are not the verbs' count.</exception>
    public GlyphOutline(OutlineVerb[] verbs, float[] points)
    {
        ArgumentNullException.ThrowIfNull(verbs);
        ArgumentNullException.ThrowIfNull(points);
        int needed = 0;
        foreach (var verb in verbs)
        {
            needed += verb switch { OutlineVerb.Move or OutlineVerb.Line => 2, OutlineVerb.Quadratic => 4, OutlineVerb.Cubic => 6, _ => 0 };
        }

        if (needed != points.Length)
        {
            throw new ArgumentException($"The verbs take {needed / 2} points; {points.Length / 2.0} are given.", nameof(points));
        }

        Verbs = verbs;
        Points = points;
    }

    /// <summary>The steps of the contours.</summary>
    public ReadOnlyMemory<OutlineVerb> Verbs { get; }

    /// <summary>The steps' points, x then y.</summary>
    public ReadOnlyMemory<float> Points { get; }

    /// <summary>Whether there is nothing to fill.</summary>
    public bool IsEmpty => Verbs.Length == 0;

    /// <summary>The smallest box around the points (control points included), or all zero when empty.</summary>
    public (float MinX, float MinY, float MaxX, float MaxY) Bounds()
    {
        var p = Points.Span;
        if (p.Length == 0)
        {
            return default;
        }

        float minX = p[0], maxX = p[0], minY = p[1], maxY = p[1];
        for (int i = 2; i < p.Length; i += 2)
        {
            minX = MathF.Min(minX, p[i]);
            maxX = MathF.Max(maxX, p[i]);
            minY = MathF.Min(minY, p[i + 1]);
            maxY = MathF.Max(maxY, p[i + 1]);
        }

        return (minX, minY, maxX, maxY);
    }
}

/// <summary>
/// One font of a font file: its glyphs by character, their advances and outlines, and its raw tables (the shapers read the
/// layout tables, GSUB, GPOS and GDEF, from <see cref="Table"/>). Safe to use from several threads at once.
/// </summary>
public interface IFontFace
{
    /// <summary>The family name ("DejaVu Sans"), or the file's name when the font has none.</summary>
    string Family { get; }

    /// <summary>The style name ("Bold"), or "Regular".</summary>
    string Style { get; }

    /// <summary>The name of the <see cref="IFontFormat"/> that read it.</summary>
    string Format { get; }

    /// <summary>The vertical metrics.</summary>
    FontMetrics Metrics { get; }

    /// <summary>The number of glyphs (glyph ids are 0 to this minus one; 0 is the missing-character glyph).</summary>
    int GlyphCount { get; }

    /// <summary>The glyph of a Unicode code point, or 0 when the font has none.</summary>
    int Glyph(int codePoint);

    /// <summary>A glyph's advance width, in font units.</summary>
    int Advance(int glyph);

    /// <summary>A glyph's outline (<see cref="GlyphOutline.Empty"/> for a glyph with none).</summary>
    GlyphOutline Outline(int glyph);

    /// <summary>The bytes of the table <paramref name="tag"/> ("GSUB"), or empty when the font has none.</summary>
    ReadOnlyMemory<byte> Table(string tag);
}

/// <summary>Reads one kind of font file into its faces; register it with <c>Register</c>.</summary>
public interface IFontFormat
{
    /// <summary>The format's name, e.g. "opentype" (names ignore case).</summary>
    string Name { get; }

    /// <summary>The file extensions of the format, with the dot, e.g. ".ttf".</summary>
    IReadOnlyCollection<string> Extensions { get; }

    /// <summary>Whether a file starting with <paramref name="header"/> (its first bytes) is in this format.</summary>
    bool Accepts(ReadOnlySpan<byte> header);

    /// <summary>
    /// The faces of a whole file: one, or each of a collection. A damaged file is an <see cref="InvalidDataException"/>,
    /// a valid file in a variant the format does not read a <see cref="NotSupportedException"/>.
    /// </summary>
    /// <param name="file">The file's bytes (kept by the faces, not copied).</param>
    /// <param name="name">A name for messages and for a face whose font has no family name (the file's name).</param>
    IReadOnlyList<IFontFace> Read(ReadOnlyMemory<byte> file, string name);
}

/// <summary>
/// The font file formats, by name (ignoring case). Built in (Idrak.Imaging): "opentype" (.ttf, .otf, .ttc: TrueType and
/// CFF outlines, single fonts and collections), and "cff" and "type1" (bare CFF and Type 1 programs, as documents embed
/// them: read by their first bytes, no file extension).
/// </summary>
public static class FontFormats
{
    /// <summary>The bytes of a file's start that <see cref="IFontFormat.Accepts"/> receives.</summary>
    public const int HeaderBytes = 16;
}

/// <summary>How a rasterizer finds an outline's inside (<see cref="IOutlineRasterizer.Fill(GlyphOutline, float, float, float, Span{float}, int, int, OutlineFill)"/>).</summary>
public enum OutlineFill : byte
{
    /// <summary>
    /// The rasterizer's own glyph drawing, the default: a glyph's contours, which never wind against each other, filled by
    /// the nonzero rule over the outline's whole box (what every glyph and page has always been drawn with).
    /// </summary>
    Glyph,

    /// <summary>
    /// The nonzero winding rule for contours that overlap in any direction: a point is inside when the contours wind around
    /// it a nonzero number of times, either way. Only the target's rows and columns are worked (an outline may reach far
    /// past them: a band of a page).
    /// </summary>
    NonZero,

    /// <summary>The even-odd rule: a point is inside when a ray from it crosses the contours an odd number of times. Only the target is worked, as <see cref="NonZero"/>.</summary>
    EvenOdd,
}

/// <summary>
/// Fills outlines into coverage: how much of each pixel the shape covers, 0 to 1. Register one with <c>Register</c>.
/// </summary>
public interface IOutlineRasterizer
{
    /// <summary>The rasterizer's name (names ignore case).</summary>
    string Name { get; }

    /// <summary>
    /// Adds <paramref name="outline"/> to <paramref name="coverage"/> (<paramref name="height"/> rows of
    /// <paramref name="width"/>, row 0 at the top): the outline is scaled by <paramref name="scale"/> pixels per font
    /// unit, flipped so y grows downwards, and its origin placed at (<paramref name="x"/>, <paramref name="y"/>) pixels
    /// (the baseline at y). A pixel's coverage combines with what is there as ink over ink: 1 - (1 - a)(1 - b).
    /// </summary>
    void Fill(GlyphOutline outline, float scale, float x, float y, Span<float> coverage, int width, int height);

    /// <summary>
    /// <see cref="Fill(GlyphOutline, float, float, float, Span{float}, int, int)"/> by <paramref name="rule"/>:
    /// <see cref="OutlineFill.Glyph"/> is that fill itself; <see cref="OutlineFill.NonZero"/> and
    /// <see cref="OutlineFill.EvenOdd"/> find the inside of contours that overlap in any direction, working only the
    /// target's pixels. A rasterizer that does not implement this overload fills the glyph rule alone.
    /// </summary>
    /// <exception cref="NotSupportedException">The rasterizer has no such rule (the message names it).</exception>
    void Fill(GlyphOutline outline, float scale, float x, float y, Span<float> coverage, int width, int height, OutlineFill rule)
    {
        if (rule != OutlineFill.Glyph)
        {
            throw new NotSupportedException($"The outline rasterizer '{Name}' fills glyphs by its own rule only, not by the {rule} rule; register one that implements it with OutlineRasterizers.Register.");
        }

        Fill(outline, scale, x, y, coverage, width, height);
    }

    /// <summary>
    /// <see cref="Fill(GlyphOutline, float, float, float, Span{float}, int, int, OutlineFill)"/> with each curve flattened
    /// into lines that stray at most <paramref name="flatness"/> pixels from it (a page drawer's
    /// <see cref="DocumentDrawOptions.FillFlatness"/>). A rasterizer that does not implement this overload flattens by its
    /// own measure.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="flatness"/> is not more than 0 and finite.</exception>
    /// <exception cref="NotSupportedException">The rasterizer has no such rule (the message names it).</exception>
    void Fill(GlyphOutline outline, float scale, float x, float y, Span<float> coverage, int width, int height, OutlineFill rule, float flatness)
    {
        if (!(flatness > 0f) || !float.IsFinite(flatness))
        {
            throw new ArgumentOutOfRangeException(nameof(flatness), flatness, "A curve's flatness is more than 0 pixels and finite.");
        }

        Fill(outline, scale, x, y, coverage, width, height, rule);
    }
}

/// <summary>
/// The outline rasterizers, by name. Built in (Idrak.Imaging): "coverage" (exact area coverage per pixel, no hinting; the
/// glyph rule by default, and the nonzero and even-odd rules for contours that overlap, <see cref="OutlineFill"/>).
/// </summary>
public static class OutlineRasterizers
{
    /// <summary>The library's rasterizer.</summary>
    public const string Coverage = "coverage";
}
