// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Vision;

/// <summary>A rectangle of pixels on a page: its left column, top row, width and height.</summary>
public readonly record struct PixelBox(int X, int Y, int Width, int Height)
{
    /// <summary>The first column right of the box.</summary>
    public int Right => X + Width;

    /// <summary>The first row below the box.</summary>
    public int Bottom => Y + Height;
}

/// <summary>One character found on a page, and whether a space comes before it on its line (in page order, left to right).</summary>
public sealed record GlyphRegion(PixelBox Box, bool SpaceBefore);

/// <summary>A text line: its box and its characters from left to right.</summary>
public sealed record TextLineRegion(PixelBox Box, IReadOnlyList<GlyphRegion> Glyphs);

/// <summary>
/// The text lines and characters <see cref="PageSegmenter"/> found on a page, with the page's ink (ink high, paper 0)
/// that <see cref="GlyphFrame.Extract(PageLayout, PixelBox, Span{float}, int, int)"/> frames characters from.
/// </summary>
public sealed class PageLayout
{
    internal PageLayout(int width, int height, float[] ink, float threshold, bool darkOnLight, IReadOnlyList<TextLineRegion> lines)
    {
        Width = width;
        Height = height;
        Ink = ink;
        Threshold = threshold;
        DarkOnLight = darkOnLight;
        Lines = lines;
    }

    /// <summary>The page's width in pixels.</summary>
    public int Width { get; }

    /// <summary>The page's height in pixels.</summary>
    public int Height { get; }

    /// <summary>The ink level that separates ink from paper (Otsu's, unless one was given).</summary>
    public float Threshold { get; }

    /// <summary>Whether the page had dark ink on light paper (its ink was inverted to be high).</summary>
    public bool DarkOnLight { get; }

    /// <summary>The text lines from top to bottom.</summary>
    public IReadOnlyList<TextLineRegion> Lines { get; }

    /// <summary>The number of characters on every line.</summary>
    public int GlyphCount => Lines.Sum(l => l.Glyphs.Count);

    /// <summary>The page's ink, row by row: high where there is ink, 0 on paper.</summary>
    internal float[] Ink { get; }
}
