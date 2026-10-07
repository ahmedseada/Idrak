// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Vision.Abstractions;

/// <summary>A rectangle of whole pixels: its left column, top row, width and height.</summary>
public readonly record struct PixelBox(int X, int Y, int Width, int Height)
{
    /// <summary>The first column right of the box.</summary>
    public int Right => X + Width;

    /// <summary>The first row below the box.</summary>
    public int Bottom => Y + Height;

    /// <summary>The number of pixels in the box.</summary>
    public int Area => Width * Height;

    /// <summary>The smallest box holding both boxes.</summary>
    public PixelBox Union(PixelBox other)
    {
        int x = Math.Min(X, other.X), y = Math.Min(Y, other.Y);
        return new PixelBox(x, y, Math.Max(Right, other.Right) - x, Math.Max(Bottom, other.Bottom) - y);
    }

    /// <summary>The box in sub-pixel coordinates.</summary>
    public BoundingBox ToBoundingBox() => new(X, Y, Width, Height);
}

/// <summary>A rectangle in sub-pixel image coordinates (x to the right, y down): left, top, width and height.</summary>
public readonly record struct BoundingBox(float X, float Y, float Width, float Height)
{
    /// <summary>The box from its corners (any order).</summary>
    public static BoundingBox FromCorners(float x1, float y1, float x2, float y2) =>
        new(Math.Min(x1, x2), Math.Min(y1, y2), Math.Abs(x2 - x1), Math.Abs(y2 - y1));

    /// <summary>The box from its centre and size (how most detection networks predict boxes).</summary>
    public static BoundingBox FromCenter(float centerX, float centerY, float width, float height) =>
        new(centerX - width / 2, centerY - height / 2, width, height);

    /// <summary>The right edge.</summary>
    public float Right => X + Width;

    /// <summary>The bottom edge.</summary>
    public float Bottom => Y + Height;

    /// <summary>The centre's x.</summary>
    public float CenterX => X + Width / 2;

    /// <summary>The centre's y.</summary>
    public float CenterY => Y + Height / 2;

    /// <summary>Width times height (0 for an empty or inverted box).</summary>
    public float Area => Math.Max(Width, 0) * Math.Max(Height, 0);

    /// <summary>The area both boxes cover.</summary>
    public float IntersectionArea(BoundingBox other) =>
        Math.Max(0, Math.Min(Right, other.Right) - Math.Max(X, other.X)) * Math.Max(0, Math.Min(Bottom, other.Bottom) - Math.Max(Y, other.Y));

    /// <summary>Intersection over union: 1 for the same box, 0 for boxes that do not touch.</summary>
    public float IntersectionOverUnion(BoundingBox other)
    {
        float intersection = IntersectionArea(other), union = Area + other.Area - intersection;
        return union > 0 ? intersection / union : 0;
    }

    /// <summary>The box scaled by <paramref name="sx"/> and <paramref name="sy"/> (from a model's input size to an image's size).</summary>
    public BoundingBox Scale(float sx, float sy) => new(X * sx, Y * sy, Width * sx, Height * sy);

    /// <summary>The box clipped to an image of <paramref name="width"/> x <paramref name="height"/>.</summary>
    public BoundingBox Clip(float width, float height)
    {
        float x1 = Math.Clamp(X, 0, width), y1 = Math.Clamp(Y, 0, height), x2 = Math.Clamp(Right, 0, width), y2 = Math.Clamp(Bottom, 0, height);
        return new BoundingBox(x1, y1, x2 - x1, y2 - y1);
    }

    /// <summary>The whole pixels the box covers (rounded outwards).</summary>
    public PixelBox ToPixelBox()
    {
        int x1 = (int)MathF.Floor(X), y1 = (int)MathF.Floor(Y), x2 = (int)MathF.Ceiling(Right), y2 = (int)MathF.Ceiling(Bottom);
        return new PixelBox(x1, y1, x2 - x1, y2 - y1);
    }
}

/// <summary>An object found in an image: its box, its class (index and name, when known) and the detector's score.</summary>
public sealed record Detection(BoundingBox Box, int Class, float Score, string? Label = null);

/// <summary>Finds objects in images. Implement it over any detection network; Idrak.Vision's <c>ModelDetector</c> is one.</summary>
public interface IObjectDetector
{
    /// <summary>The objects in <paramref name="image"/>, in its pixel coordinates.</summary>
    IReadOnlyList<Detection> Detect(Abstraction.Data.ImageData image);
}

/// <summary>
/// Proposes the regions of an image to classify. <c>ComponentProposer</c> takes its connected regions;
/// implement it for other layouts (text lines split into characters, a grid of form fields, a detector's boxes).
/// </summary>
public interface IRegionProposer
{
    /// <summary>The regions of <paramref name="image"/>, in the order they should be reported.</summary>
    IReadOnlyList<PixelBox> Propose(ForegroundImage image);
}
