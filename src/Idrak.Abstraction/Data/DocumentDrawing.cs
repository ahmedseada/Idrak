// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Data;

/// <summary>The pixels a page is drawn into.</summary>
public enum DocumentDrawColour : byte
{
    /// <summary>One grey channel.</summary>
    Grey,

    /// <summary>Red, green and blue.</summary>
    Rgb,
}

/// <summary>
/// How a page is drawn to pixels: the resolution, the colour, the parts drawn and the pieces the drawer draws with (all by
/// name, so each can be replaced). <see cref="DpiVariable"/> sets the resolution and <see cref="FontsVariable"/> the folder of
/// fonts for text whose font a document does not embed. The drawer's numbers are settings too (its bands, a shading's
/// steps, the largest tile, how a face found for a font is fitted, how closely curves are flattened), each with its default.
/// </summary>
public sealed partial record DocumentDrawOptions
{
    /// <summary>The variable that sets the resolution in dots per inch (<see cref="Dpi"/>; 150 unless set): "IDRAK_DOCUMENT_DPI".</summary>
    public const string DpiVariable = "IDRAK_DOCUMENT_DPI";

    /// <summary>The variable that names the folder of fonts (<see cref="Fonts"/>): "IDRAK_DOCUMENT_FONTS".</summary>
    public const string FontsVariable = "IDRAK_DOCUMENT_FONTS";

    private readonly float _pixelsPerPoint = (DocumentSettings.Float(DpiVariable, 1f, 9600f) ?? 150f) / 72f;

    /// <summary>Pixels per point (1/72 inch): <see cref="Dpi"/> / 72.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Not positive and finite.</exception>
    public float PixelsPerPoint
    {
        get => _pixelsPerPoint;
        init => _pixelsPerPoint = value > 0f && float.IsFinite(value) ? value : throw new ArgumentOutOfRangeException(nameof(PixelsPerPoint), value, "Pixels per point are more than 0.");
    }

    /// <summary>The resolution in dots per inch (150 unless <see cref="DpiVariable"/> sets it): <see cref="PixelsPerPoint"/> × 72.</summary>
    public float Dpi
    {
        get => _pixelsPerPoint * 72f;
        init => PixelsPerPoint = value / 72f;
    }

    /// <summary>Grey or RGB pixels (<see cref="DocumentDrawColour.Rgb"/>).</summary>
    public DocumentDrawColour Colour { get; init; } = DocumentDrawColour.Rgb;

    /// <summary>The colour under the page's content (white).</summary>
    public DocumentColour Background { get; init; } = DocumentColour.White;

    /// <summary>
    /// A folder of fonts for runs whose font the document does not embed, matched by name and then by the characters they
    /// cover (<see cref="FontsVariable"/>; null: such runs are left out and counted in the drawing's issues).
    /// </summary>
    public string? Fonts { get; init; } = DocumentSettings.Text(FontsVariable);

    /// <summary>The parts drawn: any of text, images, paths and annotations (text, images and paths).</summary>
    public DocumentParts Parts { get; init; } = DocumentParts.Text | DocumentParts.Images | DocumentParts.Paths;

    /// <summary>Whether invisible text (a scan's text layer) is drawn as if it were painted (false).</summary>
    public bool Hidden { get; init; }

    /// <summary>The drawer that draws the page: a <see cref="DocumentPageDrawers"/> name (<see cref="DocumentPageDrawers.Coverage"/>).</summary>
    public string Drawer { get; init; } = DocumentPageDrawers.Coverage;

    /// <summary>The rasterizer outlines are filled with: an <see cref="OutlineRasterizers"/> name (<see cref="OutlineRasterizers.Coverage"/>).</summary>
    public string Rasterizer { get; init; } = OutlineRasterizers.Coverage;

    /// <summary>The pen strokes are turned into outlines with: a <see cref="Pens"/> name (<see cref="Pens.Ideal"/>).</summary>
    public string Pen { get; init; } = Pens.Ideal;
}

/// <summary>A page drawn to pixels, and what could not be drawn (counted: a font that is missing, a part the drawer does not draw).</summary>
/// <param name="Image">The pixels.</param>
/// <param name="Issues">What was left out, and why.</param>
public sealed record DocumentDrawing(ImageData Image, IReadOnlyList<DocumentIssue> Issues);

/// <summary>
/// Draws any <see cref="DocumentPage"/> (its paths, glyph runs and images, in content order, under their clips) to
/// pixels, whatever format read it. Safe to use from
/// several threads at once.
/// </summary>
public interface IDocumentPageDrawer
{
    /// <summary>
    /// <paramref name="page"/> drawn to an image of its size times <see cref="DocumentDrawOptions.PixelsPerPoint"/>
    /// (<see cref="DocumentPageDrawers.PixelSize"/>), on the options' background, with the options' parts. A run whose font
    /// is not there is left out and counted in the issues, never thrown.
    /// </summary>
    DocumentDrawing Draw(DocumentPage page, DocumentDrawOptions options);
}

/// <summary>
/// The page drawers' helpers and the library's drawer's name (Idrak.Imaging): <see cref="Coverage"/>, the one
/// <see cref="IDocument.DrawPage"/> reaches directly; picking another by its name is not on the new design yet.
/// </summary>
public static class DocumentPageDrawers
{
    /// <summary>"coverage": the library's drawer's name (exact area coverage through <see cref="OutlineRasterizers"/>).</summary>
    public const string Coverage = "coverage";

    /// <summary>The image size a page is drawn at: its width and height times the pixels per point, rounded, at least one pixel each.</summary>
    public static (int Width, int Height) PixelSize(DocumentPage page, DocumentDrawOptions options)
    {
        ArgumentNullException.ThrowIfNull(page);
        ArgumentNullException.ThrowIfNull(options);
        return (Math.Max(1, (int)MathF.Round(page.Width * options.PixelsPerPoint)), Math.Max(1, (int)MathF.Round(page.Height * options.PixelsPerPoint)));
    }

    /// <summary>The parts a page is read with to be drawn: the options' parts, and the glyphs when text is drawn.</summary>
    public static DocumentParts PartsFor(DocumentDrawOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return (options.Parts & DocumentParts.Text) != 0 ? options.Parts | DocumentParts.Glyphs : options.Parts;
    }
}
