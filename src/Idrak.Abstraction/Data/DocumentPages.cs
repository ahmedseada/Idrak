// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Abstraction.Data;

/// <summary>
/// A box on a page, in points (1/72 inch) from the page's top-left corner: x to the right, y down. A format whose text has
/// no place on a page (a word processor's paragraph) gives <see cref="Empty"/>.
/// </summary>
/// <param name="X">The left edge.</param>
/// <param name="Y">The top edge.</param>
/// <param name="Width">The width.</param>
/// <param name="Height">The height.</param>
public readonly record struct DocumentBox(float X, float Y, float Width, float Height)
{
    /// <summary>No box: no place on a page.</summary>
    public static DocumentBox Empty => default;

    /// <summary>Whether the box is <see cref="Empty"/> (no width and no height).</summary>
    public bool IsEmpty => !(Width > 0) && !(Height > 0);

    /// <summary>The right edge.</summary>
    public float Right => X + Width;

    /// <summary>The bottom edge.</summary>
    public float Bottom => Y + Height;

    /// <summary>The smallest box holding this one and <paramref name="other"/>; an empty box adds nothing.</summary>
    public DocumentBox Union(DocumentBox other)
    {
        if (other.IsEmpty)
        {
            return this;
        }

        if (IsEmpty)
        {
            return other;
        }

        float left = MathF.Min(X, other.X), top = MathF.Min(Y, other.Y);
        return new DocumentBox(left, top, MathF.Max(Right, other.Right) - left, MathF.Max(Bottom, other.Bottom) - top);
    }

    /// <summary>Whether this box lies inside <paramref name="outer"/>, give or take <paramref name="tolerance"/> points on each side.</summary>
    public bool Within(DocumentBox outer, float tolerance = 0f) =>
        X >= outer.X - tolerance && Y >= outer.Y - tolerance && Right <= outer.Right + tolerance && Bottom <= outer.Bottom + tolerance;

    /// <inheritdoc />
    public override string ToString() => string.Create(CultureInfo.InvariantCulture, $"({X:G6}, {Y:G6}, {Width:G6} x {Height:G6})");
}

/// <summary>
/// An affine map of the plane: (x, y) goes to (A·x + C·y + E, B·x + D·y + F). A page's parts use it towards page points
/// from the top-left corner (y down): an image's matrix maps the image's unit square (0 to 1 each way, from its top-left
/// corner) onto the page, so an upright image in a box is <see cref="FromBox"/> of that box.
/// </summary>
/// <param name="A">How x scales into x.</param>
/// <param name="B">How x adds into y.</param>
/// <param name="C">How y adds into x.</param>
/// <param name="D">How y scales into y.</param>
/// <param name="E">The shift along x.</param>
/// <param name="F">The shift along y.</param>
public readonly record struct DocumentMatrix(float A, float B, float C, float D, float E, float F)
{
    /// <summary>The map that changes nothing.</summary>
    public static DocumentMatrix Identity => new(1f, 0f, 0f, 1f, 0f, 0f);

    /// <summary>The map of the unit square onto <paramref name="box"/>, upright.</summary>
    public static DocumentMatrix FromBox(DocumentBox box) => new(box.Width, 0f, 0f, box.Height, box.X, box.Y);

    /// <summary>Where (<paramref name="x"/>, <paramref name="y"/>) goes.</summary>
    public (float X, float Y) Transform(float x, float y) => (A * x + C * y + E, B * x + D * y + F);

    /// <summary>This map followed by <paramref name="next"/>: a point goes through this one first.</summary>
    public DocumentMatrix Then(DocumentMatrix next) => new(
        next.A * A + next.C * B, next.B * A + next.D * B,
        next.A * C + next.C * D, next.B * C + next.D * D,
        next.A * E + next.C * F + next.E, next.B * E + next.D * F + next.F);

    /// <summary>The smallest box holding the unit square's image (where an image placed by this matrix lands).</summary>
    public DocumentBox Bounds()
    {
        var (x0, y0) = Transform(0f, 0f);
        var (x1, y1) = Transform(1f, 0f);
        var (x2, y2) = Transform(0f, 1f);
        var (x3, y3) = Transform(1f, 1f);
        float left = MathF.Min(MathF.Min(x0, x1), MathF.Min(x2, x3)), top = MathF.Min(MathF.Min(y0, y1), MathF.Min(y2, y3));
        float right = MathF.Max(MathF.Max(x0, x1), MathF.Max(x2, x3)), bottom = MathF.Max(MathF.Max(y0, y1), MathF.Max(y2, y3));
        return new DocumentBox(left, top, right - left, bottom - top);
    }
}

/// <summary>The colour space a <see cref="DocumentColour"/> was given in.</summary>
public enum DocumentColourSpace : byte
{
    /// <summary>One level, 0 black to 1 white.</summary>
    Grey,

    /// <summary>Red, green and blue, each 0 to 1.</summary>
    Rgb,

    /// <summary>Cyan, magenta, yellow and black ink, each 0 (none) to 1 (full).</summary>
    Cmyk,
}

/// <summary>
/// A colour as a document gives it (grey, RGB or CMYK, each component 0 to 1), with its RGB for drawing and its opacity.
/// The default value is opaque black.
/// </summary>
public readonly record struct DocumentColour
{
    // The opacity is kept as its complement, so the default value is opaque.
    private readonly float _c0, _c1, _c2, _c3, _transparency;

    private DocumentColour(DocumentColourSpace space, float c0, float c1, float c2, float c3, float alpha)
    {
        Space = space;
        (_c0, _c1, _c2, _c3) = (Unit(c0, "component"), Unit(c1, "component"), Unit(c2, "component"), Unit(c3, "component"));
        _transparency = 1f - Unit(alpha, "alpha");
    }

    /// <summary>Opaque black.</summary>
    public static DocumentColour Black => default;

    /// <summary>Opaque white.</summary>
    public static DocumentColour White => Grey(1f);

    /// <summary>A grey level, 0 black to 1 white.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The level is not in [0, 1].</exception>
    public static DocumentColour Grey(float level) => new(DocumentColourSpace.Grey, level, 0f, 0f, 0f, 1f);

    /// <summary>Red, green and blue, each 0 to 1.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A component is not in [0, 1].</exception>
    public static DocumentColour Rgb(float red, float green, float blue) => new(DocumentColourSpace.Rgb, red, green, blue, 0f, 1f);

    /// <summary>Cyan, magenta, yellow and black ink, each 0 to 1.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A component is not in [0, 1].</exception>
    public static DocumentColour Cmyk(float cyan, float magenta, float yellow, float black) => new(DocumentColourSpace.Cmyk, cyan, magenta, yellow, black, 1f);

    /// <summary>The space the colour was given in.</summary>
    public DocumentColourSpace Space { get; }

    /// <summary>How many components the space has: 1, 3 or 4.</summary>
    public int Count => Space switch { DocumentColourSpace.Grey => 1, DocumentColourSpace.Rgb => 3, _ => 4 };

    /// <summary>Component <paramref name="index"/> as given (0 to <see cref="Count"/> - 1).</summary>
    /// <exception cref="ArgumentOutOfRangeException">The index is not one of the space's components.</exception>
    public float this[int index] => (uint)index < (uint)Count
        ? index switch { 0 => _c0, 1 => _c1, 2 => _c2, _ => _c3 }
        : throw new ArgumentOutOfRangeException(nameof(index), index, $"A {Space} colour has {Count} components.");

    /// <summary>The opacity, 0 (transparent) to 1 (opaque, the default).</summary>
    public float Alpha => 1f - _transparency;

    /// <summary>The red of its RGB (a CMYK colour as (1 - c)(1 - k)).</summary>
    public float Red => Space switch { DocumentColourSpace.Grey => _c0, DocumentColourSpace.Rgb => _c0, _ => (1f - _c0) * (1f - _c3) };

    /// <summary>The green of its RGB.</summary>
    public float Green => Space switch { DocumentColourSpace.Grey => _c0, DocumentColourSpace.Rgb => _c1, _ => (1f - _c1) * (1f - _c3) };

    /// <summary>The blue of its RGB.</summary>
    public float Blue => Space switch { DocumentColourSpace.Grey => _c0, DocumentColourSpace.Rgb => _c2, _ => (1f - _c2) * (1f - _c3) };

    /// <summary>Its grey level for drawing in grey: the luma of its RGB (0.299 R + 0.587 G + 0.114 B), or the level itself.</summary>
    public float Luma => Space == DocumentColourSpace.Grey ? _c0 : 0.299f * Red + 0.587f * Green + 0.114f * Blue;

    /// <summary>The same colour with opacity <paramref name="alpha"/>.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The opacity is not in [0, 1].</exception>
    public DocumentColour WithAlpha(float alpha) => new(Space, _c0, _c1, _c2, _c3, alpha);

    /// <inheritdoc />
    public override string ToString()
    {
        string components = Space switch
        {
            DocumentColourSpace.Grey => string.Create(CultureInfo.InvariantCulture, $"grey {_c0:G6}"),
            DocumentColourSpace.Rgb => string.Create(CultureInfo.InvariantCulture, $"rgb {_c0:G6} {_c1:G6} {_c2:G6}"),
            _ => string.Create(CultureInfo.InvariantCulture, $"cmyk {_c0:G6} {_c1:G6} {_c2:G6} {_c3:G6}"),
        };
        return _transparency > 0f ? string.Create(CultureInfo.InvariantCulture, $"{components}, alpha {Alpha:G6}") : components;
    }

    private static float Unit(float value, string what) => value is >= 0f and <= 1f
        ? value
        : throw new ArgumentOutOfRangeException(what, value, $"A colour's {what} is from 0 to 1.");
}

/// <summary>How a path's inside is found when it is filled (or used as a clip).</summary>
public enum DocumentFill : byte
{
    /// <summary>Not filled: a path that is only stroked.</summary>
    None,

    /// <summary>The nonzero winding rule: a point is inside when the contours wind around it a nonzero number of times.</summary>
    NonZero,

    /// <summary>The even-odd rule: a point is inside when a ray from it crosses the contours an odd number of times.</summary>
    EvenOdd,
}

/// <summary>The shape a stroke's open ends take.</summary>
public enum DocumentLineCap : byte
{
    /// <summary>Square at the end point.</summary>
    Butt,

    /// <summary>A half circle around the end point.</summary>
    Round,

    /// <summary>Square, half the width past the end point.</summary>
    Square,
}

/// <summary>The shape a stroke takes where two segments meet.</summary>
public enum DocumentLineJoin : byte
{
    /// <summary>The outer edges extended until they meet (cut off past <see cref="DocumentPath.MiterLimit"/>).</summary>
    Miter,

    /// <summary>A circular arc.</summary>
    Round,

    /// <summary>The outer corners joined by a straight line.</summary>
    Bevel,
}

/// <summary>
/// A page's vector part: contours of lines and curves (<see cref="OutlineVerb"/>, as glyph outlines take them) in page
/// points from the top-left corner (y down), with how it is filled and stroked. A document format reads its drawings as
/// these and a writer writes them, so a written page reads back exactly. A clip is a path too (its <see cref="Fill"/> the
/// rule that finds its inside).
/// </summary>
public sealed class DocumentPath
{
    /// <summary>A path of <paramref name="verbs"/> and their points (x, y pairs; used directly, not copied).</summary>
    /// <exception cref="ArgumentException">The points are not what the verbs take, or a point is not finite.</exception>
    public DocumentPath(OutlineVerb[] verbs, float[] points)
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

        foreach (float value in points)
        {
            if (!float.IsFinite(value))
            {
                throw new ArgumentException("A path's points are finite numbers.", nameof(points));
            }
        }

        Verbs = verbs;
        Points = points;
    }

    /// <summary>A closed rectangle: from the box's top-left corner, clockwise on the page.</summary>
    public static DocumentPath Rectangle(DocumentBox box) => new(
        [OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close],
        [box.X, box.Y, box.Right, box.Y, box.Right, box.Bottom, box.X, box.Bottom]);

    /// <summary>The steps of the contours.</summary>
    public ReadOnlyMemory<OutlineVerb> Verbs { get; }

    /// <summary>The steps' points, x then y, in page points.</summary>
    public ReadOnlyMemory<float> Points { get; }

    /// <summary>How the path is filled (<see cref="DocumentFill.NonZero"/> unless set; <see cref="DocumentFill.None"/> for a path only stroked).</summary>
    public DocumentFill Fill { get; init; } = DocumentFill.NonZero;

    /// <summary>The stroke's width in points, or null when the path is not stroked (0: the thinnest line the device draws).</summary>
    public float? Stroke { get; init; }

    /// <summary>The stroke's ends.</summary>
    public DocumentLineCap Cap { get; init; }

    /// <summary>The stroke's corners.</summary>
    public DocumentLineJoin Join { get; init; }

    /// <summary>The longest a mitred corner may reach, as a multiple of the stroke's width, before it is bevelled (10).</summary>
    public float MiterLimit { get; init; } = 10f;

    /// <summary>The stroke's dashes: lengths in points, on and off in turn (empty: a solid stroke).</summary>
    public ReadOnlyMemory<float> Dash { get; init; }

    /// <summary>How far into the dash pattern the stroke starts, in points.</summary>
    public float DashPhase { get; init; }

    /// <summary>The fill's colour (black).</summary>
    public DocumentColour FillColour { get; init; }

    /// <summary>The stroke's colour (black).</summary>
    public DocumentColour StrokeColour { get; init; }

    /// <summary>The clip the path was drawn under: an index into its page's <see cref="DocumentPage.Clips"/>, or -1 for none.</summary>
    public int Clip { get; init; } = -1;

    /// <summary>The path's place in its page's content: text runs, images and paths share one count, so a drawer draws them in this order.</summary>
    public int Order { get; init; }

    /// <summary>The pattern the path is filled with (a tiling or a shading), or null: <see cref="FillColour"/>.</summary>
    public DocumentPattern? FillPattern { get; init; }

    /// <summary>The pattern its stroke is painted with, or null: <see cref="StrokeColour"/>.</summary>
    public DocumentPattern? StrokePattern { get; init; }

    /// <summary>How it is composed with what is under it (<see cref="DocumentBlend.Normal"/>).</summary>
    public DocumentBlend Blend { get; init; }

    /// <summary>The transparency group it is drawn in: an index into its page's <see cref="DocumentPage.Groups"/>, or -1 for none.</summary>
    public int Group { get; init; } = -1;

    /// <summary>The soft mask it is drawn through: an index into its page's <see cref="DocumentPage.SoftMasks"/>, or -1 for none.</summary>
    public int SoftMask { get; init; } = -1;

    /// <summary>
    /// For a clip made of text (runs shown to clip, as PDF's render modes 4 to 7): the <see cref="DocumentTextRun.Order"/>
    /// of its first run, the clip being the glyphs of the <see cref="ClipRunCount"/> runs from it (no verbs of its own); -1
    /// for a clip that is the path itself.
    /// </summary>
    public int ClipRun { get; init; } = -1;

    /// <summary>How many runs, in content order from <see cref="ClipRun"/>, make a clip of text (0 for a path).</summary>
    public int ClipRunCount { get; init; }

    /// <summary>Whether the path has no contour.</summary>
    public bool IsEmpty => Verbs.Length == 0;

    /// <summary>The smallest box around the points (control points included); empty for a path with none.</summary>
    public DocumentBox Bounds()
    {
        var p = Points.Span;
        if (p.Length == 0)
        {
            return DocumentBox.Empty;
        }

        float minX = p[0], maxX = p[0], minY = p[1], maxY = p[1];
        for (int i = 2; i < p.Length; i += 2)
        {
            minX = MathF.Min(minX, p[i]);
            maxX = MathF.Max(maxX, p[i]);
            minY = MathF.Min(minY, p[i + 1]);
            maxY = MathF.Max(maxY, p[i + 1]);
        }

        return new DocumentBox(minX, minY, maxX - minX, maxY - minY);
    }
}

/// <summary>How a run of text is painted.</summary>
public enum DocumentTextMode : byte
{
    /// <summary>The glyphs filled (ordinary text).</summary>
    Fill,

    /// <summary>The glyphs' outlines stroked.</summary>
    Stroke,

    /// <summary>Filled, then stroked.</summary>
    FillStroke,

    /// <summary>Not painted: text that is there to be found and copied, such as the text layer over a scanned page.</summary>
    Hidden,
}

/// <summary>One glyph of a run, as placed on the page.</summary>
/// <param name="Glyph">The glyph's id in its run's font (<see cref="DocumentTextRun.Face"/>).</param>
/// <param name="X">Its origin on the baseline: points from the page's left edge.</param>
/// <param name="Y">Its origin on the baseline: points from the page's top edge.</param>
/// <param name="Advance">How far it moves the next glyph along the line, in points.</param>
public readonly record struct DocumentGlyph(int Glyph, float X, float Y, float Advance);

/// <summary>
/// A run of text on a page: what it says, where it stands and how it is drawn. A format gives each piece of text it shows
/// as a run (a reader of a fixed-layout file: each string of glyphs it places; a word processor: each span of one style,
/// with no box).
/// </summary>
/// <param name="Text">
/// The characters. In the order the glyphs stand on the page from left to right, unless <see cref="Logical"/> says they
/// are in the order they are read in.
/// </param>
/// <param name="Box">Where the run stands (empty when the format gives no places).</param>
/// <param name="FontSize">The size of its font, in points.</param>
public sealed record DocumentTextRun(string Text, DocumentBox Box, float FontSize)
{
    /// <summary>The font's name as the document gives it, or null.</summary>
    public string? Font { get; init; }

    /// <summary>How the run is painted (<see cref="DocumentTextMode.Fill"/>).</summary>
    public DocumentTextMode Mode { get; init; }

    /// <summary>Whether the run is not painted (invisible text, such as a scan's text layer): <see cref="Mode"/> is <see cref="DocumentTextMode.Hidden"/>.</summary>
    public bool Hidden => Mode == DocumentTextMode.Hidden;

    /// <summary>The run's place in its page's content (text runs, images and paths share one count).</summary>
    public int Order { get; init; }

    /// <summary>The y of the run's baseline, points from the page's top edge; null when the format gives none (the box's bottom is used).</summary>
    public float? Baseline { get; init; }

    /// <summary>
    /// Whether <see cref="Text"/> is in logical order (the order it is read in: a span whose text the document states, or a
    /// format whose text has no layout) rather than in the order the glyphs stand from left to right (the default). The two
    /// differ only for right-to-left text.
    /// </summary>
    public bool Logical { get; init; }

    /// <summary>The colour it is painted in (black); its opacity the fill's.</summary>
    public DocumentColour Colour { get; init; }

    /// <summary>The colour its outlines are stroked in, for <see cref="DocumentTextMode.Stroke"/> and <see cref="DocumentTextMode.FillStroke"/> (black).</summary>
    public DocumentColour StrokeColour { get; init; }

    /// <summary>The width its outlines are stroked with, in points (1).</summary>
    public float StrokeWidth { get; init; } = 1f;

    /// <summary>
    /// How each glyph is drawn: its outline at one unit an em (font units over the face's units per em, y up) through this
    /// map onto the page (points, y down), from the glyph's origin (<see cref="DocumentGlyph.X"/>, <see cref="DocumentGlyph.Y"/>;
    /// the map's own shift is not used); null: upright at <see cref="FontSize"/>, (size, 0, 0, -size). A turned, slanted
    /// or stretched run gives its map.
    /// </summary>
    public DocumentMatrix? GlyphMatrix { get; init; }

    /// <summary>The glyphs, when they were asked for (<see cref="DocumentParts.Glyphs"/>); empty otherwise.</summary>
    public ReadOnlyMemory<DocumentGlyph> Glyphs { get; init; }

    /// <summary>The run's font as a face to draw with (read from the program the document embeds), or null when it embeds none or glyphs were not asked for.</summary>
    public IFontFace? Face { get; init; }

    /// <summary>The clip the run was drawn under: an index into its page's <see cref="DocumentPage.Clips"/>, or -1 for none.</summary>
    public int Clip { get; init; } = -1;

    /// <summary>How it is composed with what is under it (<see cref="DocumentBlend.Normal"/>).</summary>
    public DocumentBlend Blend { get; init; }

    /// <summary>The transparency group it is drawn in: an index into its page's <see cref="DocumentPage.Groups"/>, or -1 for none.</summary>
    public int Group { get; init; } = -1;

    /// <summary>The soft mask it is drawn through: an index into its page's <see cref="DocumentPage.SoftMasks"/>, or -1 for none.</summary>
    public int SoftMask { get; init; } = -1;
}

/// <summary>
/// An image on a page: where it is placed, its size in pixels and its encoding as the document stores it; its bytes and its
/// pixels are read when asked. Safe to use from several threads.
/// </summary>
public sealed class DocumentImage
{
    /// <summary>Pixels stored as they are (the format decodes them itself).</summary>
    public const string Raw = "raw";

    /// <summary>JPEG (decoded through <see cref="ImageCodecs"/>' "jpeg").</summary>
    public const string Jpeg = "jpeg";

    /// <summary>JPEG 2000.</summary>
    public const string Jpeg2000 = "jpeg2000";

    /// <summary>CCITT fax (group 3 or 4).</summary>
    public const string Ccitt = "ccitt";

    /// <summary>JBIG2.</summary>
    public const string Jbig2 = "jbig2";

    private readonly Func<ReadOnlyMemory<byte>> _bytes;
    private readonly Func<ImageData>? _decode;
    private readonly Func<ImageData?>? _mask;

    /// <summary>An image of <paramref name="pixelWidth"/> x <paramref name="pixelHeight"/> pixels, stored in <paramref name="encoding"/>.</summary>
    /// <param name="encoding">The encoding's name (<see cref="Raw"/>, <see cref="Jpeg"/>, ... or another, as the document names it).</param>
    /// <param name="pixelWidth">Its width in pixels.</param>
    /// <param name="pixelHeight">Its height in pixels.</param>
    /// <param name="bytes">Reads its stored bytes (called on each <see cref="Bytes"/>; thread-safe).</param>
    /// <param name="decode">
    /// Decodes it when the format does so itself (raw pixels, a colour space to convert); null to decode the stored bytes
    /// with the image codec named as the encoding.
    /// </param>
    /// <param name="mask">Reads its mask as grey (see <see cref="Mask"/>), or null when it has none; null when the format reads none.</param>
    public DocumentImage(string encoding, int pixelWidth, int pixelHeight, Func<ReadOnlyMemory<byte>> bytes, Func<ImageData>? decode = null, Func<ImageData?>? mask = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(encoding);
        ArgumentOutOfRangeException.ThrowIfNegative(pixelWidth);
        ArgumentOutOfRangeException.ThrowIfNegative(pixelHeight);
        ArgumentNullException.ThrowIfNull(bytes);
        (Encoding, PixelWidth, PixelHeight, _bytes, _decode, _mask) = (encoding, pixelWidth, pixelHeight, bytes, decode, mask);
    }

    /// <summary>The encoding's name as the document stores it (<see cref="Raw"/>, <see cref="Jpeg"/>, <see cref="Jpeg2000"/>, <see cref="Ccitt"/>, <see cref="Jbig2"/>, ...).</summary>
    public string Encoding { get; }

    /// <summary>The width in pixels.</summary>
    public int PixelWidth { get; }

    /// <summary>The height in pixels.</summary>
    public int PixelHeight { get; }

    /// <summary>The box it covers on the page (for a turned or skewed image, the box around it).</summary>
    public DocumentBox Box { get; init; }

    /// <summary>How it is placed: its unit square (from its top-left corner) onto the page; null for upright in <see cref="Box"/>.</summary>
    public DocumentMatrix? Matrix { get; init; }

    /// <summary>The image's place in its page's content (text runs, images and paths share one count).</summary>
    public int Order { get; init; }

    /// <summary>The clip the image was drawn under: an index into its page's <see cref="DocumentPage.Clips"/>, or -1 for none.</summary>
    public int Clip { get; init; } = -1;

    /// <summary>Its opacity, 0 (transparent) to 1 (opaque, the default).</summary>
    public float Alpha { get; init; } = 1f;

    /// <summary>For an image that is a stencil (an image mask): the colour it paints where its <see cref="Mask"/> covers; null for an image of its own colours.</summary>
    public DocumentColour? Stencil { get; init; }

    /// <summary>How it is composed with what is under it (<see cref="DocumentBlend.Normal"/>).</summary>
    public DocumentBlend Blend { get; init; }

    /// <summary>The transparency group it is drawn in: an index into its page's <see cref="DocumentPage.Groups"/>, or -1 for none.</summary>
    public int Group { get; init; } = -1;

    /// <summary>The soft mask it is drawn through: an index into its page's <see cref="DocumentPage.SoftMasks"/>, or -1 for none.</summary>
    public int SoftMask { get; init; } = -1;

    /// <summary>The placement in use: <see cref="Matrix"/>, or the map onto <see cref="Box"/>.</summary>
    public DocumentMatrix Placement => Matrix ?? DocumentMatrix.FromBox(Box);

    /// <summary>The stored bytes, read now.</summary>
    public ReadOnlyMemory<byte> Bytes() => _bytes();

    /// <summary>The pixels: decoded by the format, or by the image codec its encoding names.</summary>
    /// <exception cref="NotSupportedException">No registered image codec reads the encoding (the message names it).</exception>
    /// <exception cref="InvalidDataException">The stored bytes are damaged.</exception>
    public ImageData Decode()
    {
        if (_decode is not null)
        {
            return _decode();
        }

        if (string.Equals(Encoding, Raw, StringComparison.OrdinalIgnoreCase))
        {
            throw new NotSupportedException($"A {Raw} image is decoded by its document's format, and this one gives no decoder.");
        }

        IImageCodec codec;
        try
        {
            codec = ImageCodecs.Get(Encoding);
        }
        catch (NotRegisteredException e)
        {
            throw new NotRegisteredException(e.Registry, e.Name, e.Registered,
                $"No image codec reads the encoding '{Encoding}' of this document's image (registered: {string.Join(", ", e.Registered)}); register one with ImageCodecs.Register.");
        }

        return codec.Decode(_bytes().Span);
    }

    /// <summary>
    /// The image's mask (a stencil, a soft mask or a colour key) as grey at its own size (spread over the image's place
    /// when the sizes differ): how much of the image shows at each pixel, 255 fully and 0 not at all (a soft mask's levels;
    /// a stencil 255 where it paints); null when it has none.
    /// </summary>
    public ImageData? Mask() => _mask?.Invoke();
}

/// <summary>A file a document carries inside it, read when asked.</summary>
public sealed class DocumentAttachment
{
    private readonly Func<Stream> _read;

    /// <summary>A file named <paramref name="name"/> that <paramref name="read"/> opens.</summary>
    /// <param name="name">Its file name.</param>
    /// <param name="read">Opens its bytes (a new stream on each call, the caller's to dispose; thread-safe).</param>
    public DocumentAttachment(string name, Func<Stream> read)
    {
        ArgumentNullException.ThrowIfNull(name);
        ArgumentNullException.ThrowIfNull(read);
        (Name, _read) = (name, read);
    }

    /// <summary>Its file name.</summary>
    public string Name { get; }

    /// <summary>Its media type ("text/plain"), or null when the document gives none.</summary>
    public string? MediaType { get; init; }

    /// <summary>Its size in bytes, or -1 when the document does not say.</summary>
    public long Size { get; init; } = -1;

    /// <summary>What the document says about it, or null.</summary>
    public string? Description { get; init; }

    /// <summary>Opens its bytes: a new stream, the caller's to dispose.</summary>
    public Stream Read() => _read();
}

/// <summary>A line of a page's text, as the layout found it.</summary>
/// <param name="Text">The line's text in logical order (the order it is read in).</param>
/// <param name="Box">The box around its runs (empty when they have none).</param>
/// <param name="RightToLeft">Whether the line runs right to left.</param>
/// <param name="Runs">Its runs, as indices into the page's <see cref="DocumentPage.Runs"/>, in the order the line meets them in its direction.</param>
public sealed record DocumentLine(string Text, DocumentBox Box, bool RightToLeft, IReadOnlyList<int> Runs);

/// <summary>A page's text as a layout arranges it: its lines, and the whole text (the lines joined by line feeds, for the library's layouts).</summary>
/// <param name="Lines">The lines, in reading order.</param>
/// <param name="Text">The page's text.</param>
public sealed record DocumentText(IReadOnlyList<DocumentLine> Lines, string Text)
{
    /// <summary>No text.</summary>
    public static DocumentText Empty { get; } = new([], "");
}

/// <summary>
/// Something on a page that could not be read or drawn as it is, counted, so an app knows what to do with the page (send
/// it to OCR, supply a font).
/// </summary>
/// <param name="Kind">What happened: <see cref="UnmappedGlyphs"/>, <see cref="ImageEncoding"/>, <see cref="DamagedStream"/>, <see cref="MissingFont"/>, <see cref="NotDrawn"/>, <see cref="NotSupported"/> or a format's own.</param>
/// <param name="Detail">What and where, for a person.</param>
/// <param name="Count">How many times.</param>
public readonly record struct DocumentIssue(string Kind, string Detail, int Count = 1)
{
    /// <summary>Glyphs with no code point (their text is U+FFFD).</summary>
    public const string UnmappedGlyphs = "unmapped-glyphs";

    /// <summary>An image in an encoding no registered image codec reads.</summary>
    public const string ImageEncoding = "image-encoding";

    /// <summary>A part of the page whose data is damaged (skipped).</summary>
    public const string DamagedStream = "damaged-stream";

    /// <summary>Text whose font is not embedded and not found (left out of a drawing).</summary>
    public const string MissingFont = "missing-font";

    /// <summary>A part a drawer does not draw (named in the detail).</summary>
    public const string NotDrawn = "not-drawn";

    /// <summary>A feature of the document the format does not read (named in the detail).</summary>
    public const string NotSupported = "not-supported";
}

/// <summary>
/// One page as read: its size, its text (runs, and the lines and text a layout made of them), images, vector paths, clips
/// and annotations, each read only when asked for (<see cref="DocumentParts"/>), and what could not be read.
/// </summary>
public sealed class DocumentPage
{
    /// <summary>Page <paramref name="index"/> (from 0), <paramref name="width"/> x <paramref name="height"/> points as shown.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The index is negative, or a size is negative or not finite.</exception>
    public DocumentPage(int index, float width, float height)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        if (!(width >= 0f && float.IsFinite(width)) || !(height >= 0f && float.IsFinite(height)))
        {
            throw new ArgumentOutOfRangeException(nameof(width), $"A page is {width} x {height} points: sizes are finite and not negative.");
        }

        (Index, Width, Height) = (index, width, height);
    }

    /// <summary>The page's index, from 0.</summary>
    public int Index { get; }

    /// <summary>Its width in points, as shown (after its rotation); 0 for a format without pages of a fixed size.</summary>
    public float Width { get; }

    /// <summary>Its height in points, as shown.</summary>
    public float Height { get; }

    /// <summary>The page's box: (0, 0, <see cref="Width"/>, <see cref="Height"/>).</summary>
    public DocumentBox Box => new(0f, 0f, Width, Height);

    /// <summary>How the page is turned when shown, clockwise in degrees: 0, 90, 180 or 270 (sizes and boxes are as shown).</summary>
    public int Rotation { get; init; }

    /// <summary>The text runs, in content order (empty unless <see cref="DocumentParts.Text"/> was asked).</summary>
    public IReadOnlyList<DocumentTextRun> Runs { get; init; } = [];

    /// <summary>The lines the layout made of the runs (<see cref="DocumentOptions.Layout"/>), in reading order.</summary>
    public IReadOnlyList<DocumentLine> Lines { get; init; } = [];

    /// <summary>The page's text as the layout gives it.</summary>
    public string Text { get; init; } = "";

    /// <summary>The images, in content order (empty unless <see cref="DocumentParts.Images"/> was asked).</summary>
    public IReadOnlyList<DocumentImage> Images { get; init; } = [];

    /// <summary>The vector paths, in content order (empty unless <see cref="DocumentParts.Paths"/> was asked).</summary>
    public IReadOnlyList<DocumentPath> Paths { get; init; } = [];

    /// <summary>The clips the page's parts were drawn under, each a path whose <see cref="DocumentPath.Clip"/> is the clip it was made under (they intersect).</summary>
    public IReadOnlyList<DocumentPath> Clips { get; init; } = [];

    /// <summary>The shadings painted on their own (filling their clip), in content order (empty unless <see cref="DocumentParts.Paths"/> was asked).</summary>
    public IReadOnlyList<DocumentShading> Shadings { get; init; } = [];

    /// <summary>The transparency groups the page's parts are drawn in (each part names its innermost by <c>Group</c>).</summary>
    public IReadOnlyList<DocumentGroup> Groups { get; init; } = [];

    /// <summary>The soft masks the page's parts are drawn through (each part names its own by <c>SoftMask</c>).</summary>
    public IReadOnlyList<DocumentSoftMask> SoftMasks { get; init; } = [];

    /// <summary>The annotations (empty unless <see cref="DocumentParts.Annotations"/> was asked).</summary>
    public IReadOnlyList<DocumentAnnotation> Annotations { get; init; } = [];

    /// <summary>What could not be read, counted (glyphs with no code point, an image encoding no codec reads, a damaged part).</summary>
    public IReadOnlyList<DocumentIssue> Issues { get; init; } = [];
}
