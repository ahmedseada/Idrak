// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Data;

/// <summary>
/// How a part's colour is composed with the colour under it (the separable and non-separable blend modes of the common
/// document formats): <see cref="Normal"/> paints over; the others mix the two colours by their rule before the part's
/// opacity is applied.
/// </summary>
public enum DocumentBlend : byte
{
    /// <summary>The part's colour over what is under it.</summary>
    Normal,

    /// <summary>The two colours multiplied (never lighter).</summary>
    Multiply,

    /// <summary>The colours' complements multiplied, then complemented (never darker).</summary>
    Screen,

    /// <summary>Multiply or screen, by the colour under it.</summary>
    Overlay,

    /// <summary>The darker of the two, per channel.</summary>
    Darken,

    /// <summary>The lighter of the two, per channel.</summary>
    Lighten,

    /// <summary>The colour under it brightened to reflect the part's.</summary>
    ColorDodge,

    /// <summary>The colour under it darkened to reflect the part's.</summary>
    ColorBurn,

    /// <summary>Multiply or screen, by the part's colour.</summary>
    HardLight,

    /// <summary>A softer <see cref="HardLight"/>: darkened or lightened by the part's colour.</summary>
    SoftLight,

    /// <summary>The difference of the two, per channel.</summary>
    Difference,

    /// <summary>A softer <see cref="Difference"/>.</summary>
    Exclusion,

    /// <summary>The part's hue with the saturation and luminosity under it.</summary>
    Hue,

    /// <summary>The part's saturation with the hue and luminosity under it.</summary>
    Saturation,

    /// <summary>The part's hue and saturation with the luminosity under it.</summary>
    Color,

    /// <summary>The part's luminosity with the hue and saturation under it.</summary>
    Luminosity,
}

/// <summary>The kind of a <see cref="DocumentShading"/>.</summary>
public enum DocumentShadingKind : byte
{
    /// <summary>A colour for every point of a rectangle of the shading's space (its <see cref="DocumentShading.Colour"/> of x and y).</summary>
    Function,

    /// <summary>A colour along an axis from one point to another, constant across it.</summary>
    Axial,

    /// <summary>A colour from one circle to another (the circles between them blend).</summary>
    Radial,

    /// <summary>Triangles with a colour at each corner, blended between them (free-form and lattice meshes, and patches cut into triangles).</summary>
    Mesh,
}

/// <summary>
/// A smooth shading: a colour that changes over the page (an axis, circles, a function of the place, or a mesh of
/// triangles). Painted on its own over its clip (the page's <see cref="DocumentPage.Shadings"/>), or as the paint of a
/// path's fill or stroke (<see cref="DocumentPattern"/>). Safe to use from several threads.
/// </summary>
public sealed class DocumentShading
{
    /// <summary>A shading of <paramref name="kind"/>.</summary>
    /// <param name="kind">Its kind.</param>
    /// <param name="colour">
    /// The colour at a place (thread-safe): axial and radial take (t, 0), t from <see cref="DomainStart"/> to
    /// <see cref="DomainEnd"/>; a function-based shading (x, y) in its space. Null for a mesh (its corners give the colours).
    /// </param>
    /// <exception cref="ArgumentException">A kind other than a mesh with no colour.</exception>
    public DocumentShading(DocumentShadingKind kind, Func<float, float, DocumentColour>? colour)
    {
        if (colour is null && kind != DocumentShadingKind.Mesh)
        {
            throw new ArgumentException($"A {kind} shading needs its colour.", nameof(colour));
        }

        (Kind, Colour) = (kind, colour);
    }

    /// <summary>The kind.</summary>
    public DocumentShadingKind Kind { get; }

    /// <summary>The colour at a place (see the constructor), or null for a mesh.</summary>
    public Func<float, float, DocumentColour>? Colour { get; }

    /// <summary>
    /// In the shading's space: an axial shading's axis (x0, y0, x1, y1); a radial one's circles (x0, y0, r0, x1, y1, r1); a
    /// function-based one's rectangle (x0, x1, y0, y1). Empty for a mesh.
    /// </summary>
    public ReadOnlyMemory<float> Coords { get; init; }

    /// <summary>The t of an axial or radial shading's start (0).</summary>
    public float DomainStart { get; init; }

    /// <summary>The t of its end (1).</summary>
    public float DomainEnd { get; init; } = 1f;

    /// <summary>Whether the colour goes on before the start (past the first point or circle).</summary>
    public bool ExtendStart { get; init; }

    /// <summary>Whether the colour goes on past the end.</summary>
    public bool ExtendEnd { get; init; }

    /// <summary>The map from the shading's space onto the page (points, top-left origin, y down).</summary>
    public DocumentMatrix Matrix { get; init; } = DocumentMatrix.Identity;

    /// <summary>A mesh's triangles on the page: x, y (points), then red, green and blue (0 to 1) for each corner, three corners a triangle.</summary>
    public ReadOnlyMemory<float> Triangles { get; init; }

    /// <summary>The colour outside the shading's extent when it fills a path (null: nothing is painted there).</summary>
    public DocumentColour? Background { get; init; }

    /// <summary>The shading's place in its page's content when it is painted on its own (text runs, images, paths and shadings share one count).</summary>
    public int Order { get; init; }

    /// <summary>The clip it is painted under: an index into its page's <see cref="DocumentPage.Clips"/>, or -1 for the whole page.</summary>
    public int Clip { get; init; } = -1;

    /// <summary>Its opacity, 0 to 1 (1).</summary>
    public float Alpha { get; init; } = 1f;

    /// <summary>How it is composed with what is under it (<see cref="DocumentBlend.Normal"/>).</summary>
    public DocumentBlend Blend { get; init; }

    /// <summary>The transparency group it is drawn in: an index into its page's <see cref="DocumentPage.Groups"/>, or -1 for none.</summary>
    public int Group { get; init; } = -1;

    /// <summary>The soft mask it is drawn through: an index into its page's <see cref="DocumentPage.SoftMasks"/>, or -1 for none.</summary>
    public int SoftMask { get; init; } = -1;
}

/// <summary>
/// The paint of a path's fill or stroke when it is not a colour: a shading over the page, or a tiling (a cell of parts
/// repeated across the page at fixed steps).
/// </summary>
public sealed class DocumentPattern
{
    /// <summary>A shading pattern: the path shows <paramref name="shading"/> (its <see cref="DocumentShading.Matrix"/> places it on the page).</summary>
    public DocumentPattern(DocumentShading shading)
    {
        ArgumentNullException.ThrowIfNull(shading);
        Shading = shading;
    }

    /// <summary>
    /// A tiling pattern: <paramref name="cell"/>'s parts (in the pattern's own space, y up) drawn within
    /// <paramref name="bounds"/>, repeated every <paramref name="xStep"/> across and <paramref name="yStep"/> up, and placed
    /// on the page by <paramref name="matrix"/>.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">A step is zero or not finite.</exception>
    public DocumentPattern(DocumentPage cell, DocumentBox bounds, float xStep, float yStep, DocumentMatrix matrix)
    {
        ArgumentNullException.ThrowIfNull(cell);
        if (!(MathF.Abs(xStep) > 0f && float.IsFinite(xStep)) || !(MathF.Abs(yStep) > 0f && float.IsFinite(yStep)))
        {
            throw new ArgumentOutOfRangeException(nameof(xStep), $"A tiling's steps are {xStep} and {yStep}: they are finite and not zero.");
        }

        (Cell, Bounds, XStep, YStep, Matrix) = (cell, bounds, xStep, yStep, matrix);
    }

    /// <summary>The shading of a shading pattern, or null.</summary>
    public DocumentShading? Shading { get; }

    /// <summary>A tiling's cell: its parts in the pattern's space (y up), or null for a shading pattern.</summary>
    public DocumentPage? Cell { get; }

    /// <summary>The box of the pattern's space the cell is drawn within (x and y its lower-left corner).</summary>
    public DocumentBox Bounds { get; }

    /// <summary>The distance between cells across, in the pattern's space.</summary>
    public float XStep { get; }

    /// <summary>The distance between cells up, in the pattern's space.</summary>
    public float YStep { get; }

    /// <summary>The map from the pattern's space onto the page (points, top-left origin, y down).</summary>
    public DocumentMatrix Matrix { get; } = DocumentMatrix.Identity;
}

/// <summary>
/// A transparency group: parts drawn together and then composed with what is under them as one, with the group's own
/// opacity, blend mode and soft mask. Each part names its innermost group; a group names the group it sits in.
/// </summary>
public sealed class DocumentGroup
{
    /// <summary>Whether the group is drawn on its own (transparent) backdrop rather than on what is under it.</summary>
    public bool Isolated { get; init; }

    /// <summary>Whether each of its parts covers the ones before it within the group (composed with the group's backdrop, not with them).</summary>
    public bool Knockout { get; init; }

    /// <summary>The group it sits in: an index into the page's <see cref="DocumentPage.Groups"/>, or -1.</summary>
    public int Parent { get; init; } = -1;

    /// <summary>The group's opacity when it is composed (1).</summary>
    public float Alpha { get; init; } = 1f;

    /// <summary>How the group is composed with what is under it (<see cref="DocumentBlend.Normal"/>).</summary>
    public DocumentBlend Blend { get; init; }

    /// <summary>The soft mask the group is composed through: an index into the page's <see cref="DocumentPage.SoftMasks"/>, or -1.</summary>
    public int SoftMask { get; init; } = -1;
}

/// <summary>
/// A soft mask: how much of each part drawn through it shows, from parts of its own drawn on a backdrop (their
/// luminosity, or their opacity).
/// </summary>
public sealed class DocumentSoftMask
{
    /// <summary>A mask made of <paramref name="content"/>'s parts (on the page, as its parts are placed).</summary>
    /// <param name="content">The parts the mask is drawn from (their places on the page).</param>
    /// <param name="luminosity">True: the drawing's luminosity on <see cref="Backdrop"/> is the mask; false: its opacity.</param>
    public DocumentSoftMask(DocumentPage content, bool luminosity)
    {
        ArgumentNullException.ThrowIfNull(content);
        (Content, Luminosity) = (content, luminosity);
    }

    /// <summary>The parts the mask is drawn from.</summary>
    public DocumentPage Content { get; }

    /// <summary>Whether the mask is the luminosity of its drawing (true) or its opacity (false).</summary>
    public bool Luminosity { get; }

    /// <summary>The colour the content is drawn on, for a luminosity mask (black: nothing shows where the content is not).</summary>
    public DocumentColour Backdrop { get; init; } = DocumentColour.Black;
}
