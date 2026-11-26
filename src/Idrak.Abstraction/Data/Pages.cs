// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using Idrak.Abstraction.Text;

namespace Idrak.Abstraction.Data;

/// <summary>
/// What a page says, by part: the header's three places, a title, a subject line, the body's paragraphs, a signature
/// block and a footer. "Start" and "end" follow the text's direction (start is the right side of an Arabic page). Empty
/// parts are not drawn.
/// </summary>
public sealed record PageContent
{
    /// <summary>Header lines at the start side (an Arabic letter's state and ministry lines, at its right).</summary>
    public IReadOnlyList<string> HeaderStart { get; init; } = [];

    /// <summary>Header lines in the middle (a basmala, an emblem's caption).</summary>
    public IReadOnlyList<string> HeaderCenter { get; init; } = [];

    /// <summary>Header lines at the end side (number, date, attachments).</summary>
    public IReadOnlyList<string> HeaderEnd { get; init; } = [];

    /// <summary>A title over the body, centred; null for none.</summary>
    public string? Title { get; init; }

    /// <summary>A subject line; null for none.</summary>
    public string? Subject { get; init; }

    /// <summary>The body's paragraphs (an empty one is a section break, which the "shapes" layout reads with its <c>sections=blank</c> setting).</summary>
    public IReadOnlyList<string> Body { get; init; } = [];

    /// <summary>The signature block's lines (a name and a title), at the end side.</summary>
    public IReadOnlyList<string> Signature { get; init; } = [];

    /// <summary>Footer lines, under a rule at the bottom.</summary>
    public IReadOnlyList<string> Footer { get; init; } = [];

    /// <summary>The text's language, a BCP 47 tag ("ar"), or null.</summary>
    public string? Language { get; init; }

    /// <summary>Whether the page reads right to left; null to follow its text.</summary>
    public bool? RightToLeft { get; init; }

    /// <summary>The page's text in reading order, one part a line, as a page drawn from it reads (header start, centre, end; title; subject; body; signature; footer).</summary>
    public string ReadingOrder() => string.Join("\n", HeaderStart.Concat(HeaderCenter).Concat(HeaderEnd)
        .Concat(Title is null ? [] : [Title]).Concat(Subject is null ? [] : [Subject]).Concat(Body).Concat(Signature).Concat(Footer).Where(s => s.Length > 0));
}

/// <summary>
/// How a page is drawn: its fonts (one is drawn per page, others back up characters it lacks), its size, and the ranges a
/// layout draws its sizes and spacing from (each page its own values, from its seed).
/// </summary>
public sealed record PageStyle
{
    /// <summary>The fonts a page's text is set in (one drawn per page; per part for some layouts).</summary>
    public required IReadOnlyList<IFontFace> Fonts { get; init; }

    /// <summary>Fonts for characters a page's font lacks (Latin, digits), tried in order.</summary>
    public IReadOnlyList<IFontFace> Fallbacks { get; init; } = [];

    /// <summary>The page's width in pixels (1240: A4 at 150 dots per inch).</summary>
    public int Width { get; init; } = 1240;

    /// <summary>The page's least height in pixels (1754: A4 at 150 dots per inch); a longer text makes a taller page.</summary>
    public int Height { get; init; } = 1754;

    /// <summary>The body's text size range, in pixels per em.</summary>
    public (float Min, float Max) BodySize { get; init; } = (26, 36);

    /// <summary>The line spacing range (multiples of the font's line height).</summary>
    public (float Min, float Max) LineSpacing { get; init; } = (1.0f, 1.5f);

    /// <summary>The side margins, as a share of the width.</summary>
    public (float Min, float Max) Margin { get; init; } = (0.05f, 0.10f);

    /// <summary>How often the body is justified (0 to 1); else aligned at its start.</summary>
    public float Justify { get; init; } = 0.5f;

    /// <summary>How often a letter's header is set in another of the fonts than its body (0 to 1).</summary>
    public float OtherHeaderFont { get; init; } = 0.4f;

    /// <summary>How often a letter has a rule under its header (0 to 1).</summary>
    public float Rule { get; init; } = 0.7f;

    /// <summary>
    /// How often a letter with no centre header lines has an emblem there (0 to 1): one of the shape profiles its
    /// <c>emblem-shapes</c> option names (<see cref="Options"/>; the library's emblems by default).
    /// </summary>
    public float Emblem { get; init; } = 0.5f;

    /// <summary>
    /// How often a letter with a signature has a stamp over it (0 to 1): one of the shape profiles its <c>stamp-shapes</c>
    /// option names (<see cref="Options"/>; the library's stamps by default), slightly turned, its ink uneven.
    /// </summary>
    public float Stamp { get; init; } = 0.3f;

    /// <summary>How often a colour letter's signature (the pen-drawn one over the name, or the name itself when none is drawn) is in blue ink (0 to 1).</summary>
    public float BlueSignature { get; init; } = 0.4f;

    /// <summary>The ink's darkness range (0 paper, 1 black).</summary>
    public (float Min, float Max) Ink { get; init; } = (0.8f, 1.0f);

    /// <summary>Colour pages: tinted paper, blue signatures and stamps (else grey).</summary>
    public bool Color { get; init; } = true;

    /// <summary>
    /// How often a page is cut to its drawn parts (0 to 1; 0, the default: the whole page): the box around every part,
    /// narrowed to the ink inside it, with <see cref="CropPadding"/> of paper around it, so the text fills the image (for
    /// a reader learning from scratch, the text and not the paper). The text is unchanged: every drawn part is inside.
    /// </summary>
    public float Crop { get; init; }

    /// <summary>The paper kept around a cut page's parts, in pixels (a range: each page its own).</summary>
    public (float Min, float Max) CropPadding { get; init; } = (8, 24);

    /// <summary>
    /// Settings a layout reads by key (keys ignore case), values as text: the "shapes" layout's <c>shapes</c> (a shape
    /// profile's name or text), <c>pen</c>, <c>bind</c>, <c>blue-pen</c>, <c>cell-align</c>, <c>stroke</c>,
    /// <c>least-size</c> and the chosen
    /// pen's and text binding's own keys; the "letter" layout's <c>emblem-shapes</c>, <c>stamp-shapes</c> and
    /// <c>signature-shapes</c> (shape profiles between '|', one drawn per page, <c>Choices</c>),
    /// <c>emblem-pen</c>, <c>stamp-pen</c> (ideal) and <c>signature-pen</c> (ballpoint) with the pens' own keys,
    /// <c>stamp-turn</c> (degrees, a range; -12..12), <c>stamp-ink</c> (how uneven a stamp's ink is, 0 to 1, a range;
    /// 0.2..0.65), <c>stamp-black</c> (the chance a colour page's stamp is black; 0.2) and <c>signature</c> (the chance
    /// a
    /// signature is drawn over the signature block's name; 1); every built-in layout's <c>pose</c> (a glyph pose,
    /// <see cref="GlyphPoses"/>; none by default) and that pose's own keys. A layout reads its own keys and leaves the
    /// others; none by default.
    /// </summary>
    public IReadOnlyDictionary<string, string> Options { get; init; } = PageOptions.None;

    /// <summary></summary>
    public ITextHabits? Habits { get; init; }

    /// <summary>Every habit's chance (0 to 1), over each habit's own; null, the default: each its own (0 unless its data gives one).</summary>
    public double? HabitChance { get; init; }

    /// <summary>A habit's chance by its name (0 to 1), over <see cref="HabitChance"/>.</summary>
    public IReadOnlyDictionary<string, double> HabitChances { get; init; } = new Dictionary<string, double>();

    /// <summary>What a page's label says when habits change its text: what is drawn (the default) or the source's spelling.</summary>
    public PageLabel Label { get; init; }

    /// <summary>
    /// Whether a layout also keeps the page as what it was drawn from (<see cref="RenderedPage.Vectors"/>: its text and
    /// shapes as vectors, for a document writer: a vector page, or a hidden text layer over the page's image); false, the
    /// default: the image alone, every page drawn as before. A layout that keeps nothing gives none.
    /// </summary>
    public bool Vectors { get; init; }
}

/// <summary>
/// A drawn page kept as what it was drawn from (a layout gives it when <see cref="PageStyle.Vectors"/> asks): its paper,
/// rules, shapes and text, before any scan look, in the page's pixels (from its top-left corner, y down), written through
/// any document writer's page (<see cref="IDocumentPageBuilder"/>): whole, as vector content (text as glyphs in their fonts
/// with the text they stand for, shapes as paths), or its own text alone over an image of the page (a hidden layer that a
/// reader finds and selects where the text shows).
/// </summary>
public interface IPageVectors
{
    /// <summary>The page's width in pixels (its image's, before any cut).</summary>
    int Width { get; }

    /// <summary>The page's height in pixels.</summary>
    int Height { get; }

    /// <summary>
    /// Writes the page as drawn: its paper, then what was drawn in its ink (rules, filled areas, shapes as ideal strokes,
    /// the page's text and its ornaments' texts, each glyph where and as turned as it was drawn), then what was drawn in its
    /// second ink (blue signatures and stamps), with pixel (<paramref name="x"/>, <paramref name="y"/>) at the page
    /// builder's top-left corner and <paramref name="scale"/> points a pixel. A pen's wobble and pressure and a stamp's
    /// uneven ink are written as the ideal stroke and the full ink they vary.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="scale"/> is not finite and above 0.</exception>
    void Write(IDocumentPageBuilder page, float scale, float x, float y);

    /// <summary>
    /// Writes the page's own text alone (the text its label reads, none of its ornaments'), each run on the glyphs it was
    /// drawn with, in reading order, painted as <paramref name="mode"/> says (<see cref="DocumentTextMode.Hidden"/>: a text
    /// layer over an image of the page); placed as <see cref="Write"/> places the page.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="scale"/> is not finite and above 0.</exception>
    void WriteText(IDocumentPageBuilder page, float scale, float x, float y, DocumentTextMode mode);
}

/// <summary>What a drawn page's label says when the text was varied for drawing (handwriting habits).</summary>
public enum PageLabel
{
    /// <summary>Exactly what is drawn: the label matches the image.</summary>
    Drawn,

    /// <summary>The source's spelling, while the image shows the varied text.</summary>
    Source,
}

/// <summary>A drawn part of a page: what it is, its text and its box (pixels).</summary>
/// <param name="Kind">"header", "title", "subject", "body", "signature", "footer", "emblem", "stamp", "scribble" (a pen-drawn signature, not text) or "rule"; "shape" (a drawn shape's box) or "region" (a region a shape made, with its text).</param>
/// <param name="Text">Its text in reading order (empty for a drawing).</param>
/// <param name="X">Its left edge.</param>
/// <param name="Y">Its top edge.</param>
/// <param name="Width">Its width.</param>
/// <param name="Height">Its height.</param>
public readonly record struct PageBlock(string Kind, string Text, float X, float Y, float Width, float Height);

/// <summary>A page drawn by a layout: its image, its text in reading order (what an OCR of it should read), and its parts.</summary>
/// <param name="Image">The page, red, green and blue (or grey) in [0, 1].</param>
/// <param name="Text">The drawn text in reading order, one part a line (paragraphs whole, not broken as drawn).</param>
/// <param name="Blocks">The drawn parts in reading order.</param>
/// <param name="Description">What was chosen (font, sizes, alignment), for a data set's manifest.</param>
public sealed record RenderedPage(ImageData Image, string Text, IReadOnlyList<PageBlock> Blocks, string Description)
{
    /// <summary>The page as what it was drawn from, when its style asked (<see cref="PageStyle.Vectors"/>) and the layout keeps it; else null.</summary>
    public IPageVectors? Vectors { get; init; }
}

/// <summary>Draws a page's content in one arrangement (a letter, a plain page).</summary>
public interface IPageLayout
{
    /// <summary>The layout's name ("letter").</summary>
    string Name { get; }

    /// <summary>One line on what it draws.</summary>
    string Summary { get; }

    /// <summary>Draws <paramref name="content"/> in <paramref name="style"/>, the choices within the style's ranges made from <paramref name="seed"/> (the same seed, the same page).</summary>
    RenderedPage Render(PageContent content, PageStyle style, int seed);
}

/// <summary>
/// The page layouts. The library's (Idrak.Imaging): "letter" (an official letter: a header of three places with an
/// optional emblem, a rule, subject, body, a signature block with a pen-drawn signature, an optional stamp, footer),
/// "plain" (header lines, title, body, footer, no ornament) and "shapes" (shapes drawn by a pen, <see cref="Shapes"/>
/// and <see cref="Pens"/>, the body's lines set in the regions they make, <see cref="TextBindings"/>; it reads <see cref="PageStyle.Options"/>).
/// Picking one by its name is not on the new design yet.
/// </summary>
public static class PageLayouts
{
}
