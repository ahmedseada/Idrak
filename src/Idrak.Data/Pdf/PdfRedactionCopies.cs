// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Data.Pdf;

/// <summary>
/// Where a content paints with one tiling pattern: the space the pattern is placed in (its /Matrix applies after it: the
/// page's, or a form's), and each paint's box on the page, whether it reached an area (removed, clipped or cut) and whether
/// the paint covers its box (a path's; text's glyphs cover only part of theirs).
/// </summary>
internal sealed class PdfPatternUses(PdfMatrix space)
{
    /// <summary>The space the pattern's /Matrix maps its own onto.</summary>
    public PdfMatrix Space { get; } = space;

    /// <summary>Each paint: its box on the page (two corners, page points), whether it reached an area, whether it covers its box.</summary>
    public List<(double X0, double Y0, double X1, double Y1, bool Touched, bool Covers)> Uses { get; } = [];

    /// <summary>Whether any paint reached an area.</summary>
    public bool Touched { get; private set; }

    /// <summary>One more paint.</summary>
    public void Add(double x0, double y0, double x1, double y1, bool touched, bool covers)
    {
        Uses.Add((x0, y0, x1, y1, touched, covers));
        Touched |= touched;
    }
}

/// <summary>
/// A Type 3 font as a redacted content shows it (ISO 32000-2, 9.6.4): how many glyphs of each code stay drawn as they were
/// and how many left (removed with the text, or drawn from a redacted copy), and the copies: each glyph whose drawing
/// reached an area, redacted where it stands (its images' pixels, paths and text under the area), is a glyph of a font of
/// the content's own (<see cref="Own"/>, codes 0, 1, ... with the widths and text of the glyphs they stand for), so the
/// font's other glyphs of that code stay whole. Written, a glyph no longer drawn here leaves the content's copy of the font
/// with the resources only it used (unless a form takes the content's resources as its own), so what it drew is not in
/// the file for this page.
/// </summary>
internal sealed class PdfType3Redaction(string name, PdfName key, PdfDictionary font, PdfDictionary charProcs, PdfDictionary? resources, bool ownResources,
    PdfMatrix matrix, PdfName?[] glyphs, PdfNames names)
{
    /// <summary>The font's name in the content's resources.</summary>
    public string Name { get; } = name;

    /// <summary>That name as the resources write it.</summary>
    public PdfName Key { get; } = key;

    /// <summary>The font's dictionary.</summary>
    public PdfDictionary Font { get; } = font;

    /// <summary>Its glyph descriptions by name.</summary>
    public PdfDictionary CharProcs { get; } = charProcs;

    /// <summary>The resources its glyphs draw with: its own, else the content's.</summary>
    public PdfDictionary? Resources { get; } = resources;

    /// <summary>Whether the font has resources of its own.</summary>
    public bool OwnResources { get; } = ownResources;

    /// <summary>Its glyph space to text space (/FontMatrix).</summary>
    public PdfMatrix Matrix { get; } = matrix;

    /// <summary>
    /// Each glyph's drawing measured in its glyph space (by name, as it is first needed): the box around what it draws, null
    /// when it draws nothing, a box of NaN when it may draw anywhere (a shading, or a description that cannot be read). Its
    /// font's declared boxes are not trusted: a glyph may draw past them.
    /// </summary>
    public Dictionary<string, double[]?> Boxes { get; } = new(StringComparer.Ordinal);

    /// <summary>Each code's glyph name (its encoding's /Differences), or null.</summary>
    public PdfName?[] Glyphs { get; } = glyphs;

    /// <summary>Names for the copies' images and forms, in the resources they draw with.</summary>
    public PdfNames Names { get; } = names;

    /// <summary>Each code's glyphs drawn as they were.</summary>
    public int[] Kept { get; } = new int[256];

    /// <summary>Each code's glyphs removed or drawn from a copy.</summary>
    public int[] Gone { get; } = new int[256];

    /// <summary>The copies, by their code in <see cref="Own"/>: the glyph's name there, its width (glyph space), its text and its description redacted.</summary>
    public List<(string Glyph, double Width, string Text, PdfRedactedContent Content)> Redacted { get; } = [];

    /// <summary>The copies' font's name in the content's resources, once there is a copy.</summary>
    public string? Own { get; set; }

    /// <summary>The glyphs (by name) no longer drawn here: left out of the content's copy of the font.</summary>
    public HashSet<string> Dropped { get; } = new(StringComparer.Ordinal);

    /// <summary>The resource names the glyphs left in the copy use (null: every one kept).</summary>
    public HashSet<string>? Used { get; set; }

    /// <summary>The resource names the copies use (null: every one kept).</summary>
    public HashSet<string>? CopiesUse { get; set; }

    /// <summary>
    /// Writes the content's copy of the font (when a glyph left it) under the font's own name, and the copies' font (when
    /// there are copies) under <see cref="Own"/>, into <paramref name="written"/>'s fonts.
    /// </summary>
    public void Write(PdfFileWriter file, PdfEditCopier copier, PdfEditCopier.Source source, Func<PdfRedactedImage, int> images, Func<int, PdfValue> link, PdfRedactedObjects written)
    {
        if (Dropped.Count > 0)
        {
            int number = file.Allocate();
            copier.WriteObject(number, s =>
            {
                s.Raw("<<"u8);
                for (int i = 0; i < Font.Count; i++)
                {
                    var key = Font.KeyAt(i);
                    if (key.Is("CharProcs"u8))
                    {
                        s.Raw("/CharProcs<<"u8);
                        for (int k = 0; k < CharProcs.Count; k++)
                        {
                            if (!Dropped.Contains(CharProcs.KeyAt(k).ToString()))
                            {
                                copier.Entry(s, source, CharProcs.KeyAt(k).Bytes, CharProcs.ValueAt(k));
                            }
                        }

                        s.Raw(">>"u8);
                    }
                    else if (key.Is("Resources"u8) && Used is not null && OwnResources && Resources is not null)
                    {
                        copier.Entry(s, source, key.Bytes, PdfValue.Of(Filtered(source.File, Resources, Used)));
                    }
                    else
                    {
                        copier.Entry(s, source, key.Bytes, Font.ValueAt(i));
                    }
                }

                s.Raw(">>"u8);
            });
            written.Fonts.Add((Name, number));
        }

        if (Redacted.Count == 0)
        {
            return;
        }

        // The copies' font: each copy's description (its own images and forms written first), its text for the reader.
        var made = new PdfRedactedObjects();
        var procs = new int[Redacted.Count];
        for (int k = 0; k < Redacted.Count; k++)
        {
            var inner = Redacted[k].Content;
            var innerWritten = PdfRedactor.Write(inner, file, copier, source, images, link);
            made.XObjects.AddRange(innerWritten.XObjects);
            made.Patterns.AddRange(innerWritten.Patterns);
            made.States.AddRange(innerWritten.States);
            made.Fonts.AddRange(innerWritten.Fonts);
            procs[k] = file.Allocate();
            file.BeginStream(procs[k]);
            file.EndStream(inner.Content, deflate: true);
        }

        int toUnicode = file.Allocate();
        WriteToUnicode(file, toUnicode);
        var copied = PdfRedactor.Resources(source.File, Resources is null ? null : Filtered(source.File, Resources, CopiesUse), new PdfRedactedContent(), made, link);
        int own = file.Allocate();
        copier.WriteObject(own, s =>
        {
            s.Raw("<</Type/Font/Subtype/Type3"u8);
            if (Font["FontBBox"u8].IsNull)
            {
                s.Raw("/FontBBox[0 0 0 0]"u8);
            }
            else
            {
                copier.Entry(s, source, "FontBBox"u8, Font["FontBBox"u8]);
            }

            if (Font["FontMatrix"u8].IsNull)
            {
                s.Raw("/FontMatrix[0.001 0 0 0.001 0 0]"u8);
            }
            else
            {
                copier.Entry(s, source, "FontMatrix"u8, Font["FontMatrix"u8]);
            }

            s.Raw("/CharProcs<<"u8);
            for (int k = 0; k < Redacted.Count; k++)
            {
                s.Name(Redacted[k].Glyph).Space().Reference(procs[k]);
            }

            s.Raw(">>/Encoding<</Type/Encoding/Differences[0"u8);
            foreach (var copy in Redacted)
            {
                s.Name(copy.Glyph);
            }

            s.Raw("]>>/FirstChar 0/LastChar "u8).Integer(Redacted.Count - 1).Raw("/Widths["u8);
            for (int k = 0; k < Redacted.Count; k++)
            {
                if (k > 0)
                {
                    s.Space();
                }

                s.Number(Redacted[k].Width, 4);
            }

            s.Byte((byte)']');
            copier.Entry(s, source, "Resources"u8, PdfValue.Of(copied));
            s.Raw("/ToUnicode "u8).Reference(toUnicode).Raw(">>"u8);
        });
        written.Fonts.Add((Own!, own));
    }

    // The copies' text (one byte a code, to UTF-16BE), a hundred to a block.
    private void WriteToUnicode(PdfFileWriter file, int number)
    {
        using var cmap = new PdfSyntax(1024);
        cmap.Raw("/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n/CIDSystemInfo <</Registry (Adobe) /Ordering (UCS) /Supplement 0>> def\n"u8)
            .Raw("/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n1 begincodespacerange\n<00> <FF>\nendcodespacerange\n"u8);
        var mapped = Enumerable.Range(0, Redacted.Count).Where(k => Redacted[k].Text.Length > 0).ToList();
        for (int start = 0; start < mapped.Count; start += 100)
        {
            int count = Math.Min(100, mapped.Count - start);
            cmap.Integer(count).Raw(" beginbfchar\n"u8);
            for (int k = start; k < start + count; k++)
            {
                int code = mapped[k];
                cmap.Byte((byte)'<').Byte("0123456789ABCDEF"u8[code >> 4]).Byte("0123456789ABCDEF"u8[code & 15]).Raw("> <"u8);
                foreach (char c in Redacted[code].Text)
                {
                    cmap.HexPair(c);
                }

                cmap.Raw(">\n"u8);
            }

            cmap.Raw("endbfchar\n"u8);
        }

        cmap.Raw("endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n"u8);
        file.BeginStream(number);
        file.EndStream(cmap.Span, deflate: true);
    }

    /// <summary>
    /// <paramref name="resources"/> with only the names <paramref name="used"/> holds in its categories of named objects
    /// (XObjects, patterns, shadings, graphics states, fonts, colour spaces, property lists); the rest as it is. As it is
    /// when <paramref name="used"/> is null.
    /// </summary>
    public static PdfDictionary Filtered(PdfFile file, PdfDictionary resources, HashSet<string>? used)
    {
        if (used is null)
        {
            return resources;
        }

        var keys = new PdfName[resources.Count];
        var values = new PdfValue[resources.Count];
        for (int i = 0; i < resources.Count; i++)
        {
            var key = resources.KeyAt(i);
            (keys[i], values[i]) = (key, resources.ValueAt(i));
            bool named = key.Is("XObject"u8) || key.Is("Pattern"u8) || key.Is("Shading"u8) || key.Is("ExtGState"u8) || key.Is("Font"u8) || key.Is("ColorSpace"u8)
                         || key.Is("Properties"u8);
            if (named && file.Dictionary(values[i]) is { } category)
            {
                var kept = Enumerable.Range(0, category.Count).Where(k => used.Contains(category.KeyAt(k).ToString())).ToList();
                values[i] = PdfValue.Of(new PdfDictionary([.. kept.Select(category.KeyAt)], [.. kept.Select(category.ValueAt)]));
            }
        }

        return new PdfDictionary(keys, values);
    }
}

/// <summary>
/// The content interpreter's redacting mode, for what a content draws from contents of its own: a form's, a soft mask's
/// group, a tiling pattern's cell and a Type 3 glyph's description, each redacted as it is placed where it is drawn.
/// </summary>
internal sealed partial class PdfContentReader
{
    // Past this many cells of a pattern a paint reaches, a cell clear of every area is taken to show somewhere.
    private const int MostCells = 1 << 16;

    // Past this many areas as a pattern's cell sees them (one for each of its copies an area reaches), the whole cell is redacted.
    private const int MostShifted = 4096;

    /// <summary>
    /// What a content leaves to its end (called once it is read, before its rewriter finishes): each tiling pattern a paint
    /// reached an area with, redacted where a copy of its cell lies under an area (<see cref="RedactPattern"/>); each Type 3
    /// font shown with glyphs gone or copied, finished (<see cref="FinishType3"/>).
    /// </summary>
    public void FinishRedaction(PdfRewriter w, PdfDictionary? resources)
    {
        foreach (var (key, uses) in w.Patterns)
        {
            if (uses.Touched)
            {
                RedactPattern(w, key, uses, resources);
            }
        }

        foreach (var (_, font) in w.Type3)
        {
            if (font is not null)
            {
                FinishType3(w, font);
            }
        }
    }

    // A content of its own (a form's, a pattern's cell, a soft mask's group, a Type 3 glyph's) redacted where `placed` puts
    // it (`box` clipping it when given) with `areas`: its copy, not wrapped in a saved state when `bare` (a glyph's, whose
    // first operator is its d0 or d1); new names from `names` when its resources are shared with other copies.
    private PdfRedactedContent RedactNested(PdfRewriter w, PdfStream stream, PdfDictionary? resources, PdfMatrix placed, double[]? box, DocumentBox[] areas, DocumentColour[] fills,
        bool bare, PdfNames? names = null)
    {
        if (w.Depth + 1 > _file.MaxDepth)
        {
            throw PdfLimit.Error(string.Create(CultureInfo.InvariantCulture,
                $"forms, patterns, soft masks and glyphs redacted within each other {w.Depth + 1} levels deep, deeper than the limit {nameof(DocumentOptions.MaxDepth)} (DocumentOptions.MaxDepth, {_file.MaxDepth})"));
        }

        var inner = new PdfRedactedContent();
        using (var reader = new PdfContentReader(_file, _page, _issues, _includeHidden))
        using (var rewriter = new PdfRewriter(areas, fills, w.Width, w.Height, resources, inner, w.Depth + 1, names))
        {
            reader.ReadImages();
            reader.ReadVector();
            reader.Rewriting(rewriter);
            using var decoded = _file.Decode(stream);
            reader.ReadContent(decoded.Span, resources, placed, placed, box, 0);
            reader.FinishRedaction(rewriter, resources);
            inner.Content = rewriter.Finish([], wrap: !bare);
        }

        return inner;
    }

    // ---- soft masks --------------------------------------------------------------------------------------------------

    // A graphics state just set (`raw`, `name`, the CTM `ctm` when it was set) whose soft mask's group lies over an area:
    // set from a copy whose group is redacted where the CTM places it; false when it has no such mask, the group lies clear
    // of the areas, or nothing in it was under one.
    private bool RedactSoftMask(PdfRewriter w, ReadOnlySpan<byte> raw, string name, PdfDictionary? resources, PdfMatrix ctm)
    {
        Span<byte> buffer = stackalloc byte[128];
        var bytes = NameBytes(raw, buffer).ToArray();
        PdfDictionary? state, mask;
        PdfStream? group;
        try
        {
            state = _file.Dictionary(Lookup(resources, "ExtGState"u8, bytes));
            mask = state is null ? null : _file.Dictionary(state["SMask"u8]);
            group = mask is null ? null : _file.Resolve(mask["G"u8]).AsStream;
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            return false;                                                      // the reader counted it already
        }

        if (group is null)
        {
            return false;
        }

        var placed = PdfShadings.Matrix(_file, group.Dictionary["Matrix"u8]).Then(ctm);
        var box = PdfFunction.Numbers(_file, group.Dictionary["BBox"u8]);
        if (box is { Length: 4 })
        {
            var bounds = PdfAnnotations.Bounds(placed, box[0], box[1], box[2], box[3]);
            if (!w.Areas.Any(a => PdfRedactor.Overlap(a, bounds.X, bounds.Y, bounds.Right, bounds.Bottom)))
            {
                return false;
            }
        }

        var groupResources = _file.Dictionary(group.Dictionary["Resources"u8]) ?? resources;
        PdfRedactedContent inner;
        try
        {
            inner = RedactNested(w, group, groupResources, placed, box, w.Areas, w.AreaFills, bare: false);
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            _issues.Add(DocumentIssue.DamagedStream, $"page {_page}: the soft mask of /{name} cannot be read, so it is not redacted: {e.Message}");
            return false;
        }

        if (!inner.Changed)
        {
            return false;
        }

        string own = w.Name("RdGs", "ExtGState"u8);
        w.Result.States.Add((own, state!, mask!, group, groupResources, inner));
        w.Result.Add(inner);
        w.ReplacedState(name);
        w.Write().Byte((byte)'/').Raw(System.Text.Encoding.ASCII.GetBytes(own)).Raw(" gs\n"u8);
        return true;
    }

    // ---- tiling patterns ---------------------------------------------------------------------------------------------

    // The tiling patterns a paint uses (its fill, its stroke), told where it painted, whether it reached an area and
    // whether it covers its box (a path) or only part of it (text).
    private void UsePatterns(PdfRewriter w, bool fill, bool stroke, double x0, double y0, double x1, double y1, bool touched, bool covers)
    {
        if (fill && _state.FillPattern is { Cell: not null } filled && PatternKey(filled) is { } a)
        {
            w.UsePattern(a, _state.BaseCtm, x0, y0, x1, y1, touched, covers);
        }

        if (stroke && _state.StrokePattern is { Cell: not null } stroked && PatternKey(stroked) is { } b)
        {
            w.UsePattern(b, _state.BaseCtm, x0, y0, x1, y1, touched, covers);
        }
    }

    // A pattern read by this content, by the key it was read under (its name's bytes as Latin-1).
    private string? PatternKey(DocumentPattern pattern)
    {
        foreach (var (key, value) in _patterns)
        {
            if (ReferenceEquals(value, pattern))
            {
                int cut = key.IndexOf('\u0001', StringComparison.Ordinal);
                return cut < 0 ? key : key[..cut];
            }
        }

        return null;
    }

    // A tiling pattern a paint reached an area with: its cell redacted where a copy of it lies under an area within a paint
    // and no copy shows it clear of the areas (each copy's parts moved back by its offset, as the first cell sees them; the
    // parts a path's box shows clear taken away, text's boxes being only partly painted). Nothing changes when a whole
    // cell lies within a path's box and clear of every area: every part of the cell is then shown whole elsewhere. A
    // pattern turned on the page (its cells not upright boxes) is redacted wherever a copy lies under an area.
    private void RedactPattern(PdfRewriter w, string key, PdfPatternUses uses, PdfDictionary? resources)
    {
        byte[] raw = System.Text.Encoding.Latin1.GetBytes(key);
        PdfStream? stream;
        try
        {
            stream = _file.Resolve(Lookup(resources, "Pattern"u8, raw)).AsStream;
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            return;                                                            // the reader counted it already
        }

        var dictionary = stream?.Dictionary;
        if (dictionary is null || !_file.Resolve(dictionary["PatternType"u8]).TryInteger(out long type) || type != 1)
        {
            return;
        }

        double[] box = PdfFunction.Numbers(_file, dictionary["BBox"u8]) is { Length: 4 } b ? b : [0, 0, 1, 1];
        double xStep = _file.Resolve(dictionary["XStep"u8]).TryNumber(out double x) && x != 0 ? x : box[2] - box[0];
        double yStep = _file.Resolve(dictionary["YStep"u8]).TryNumber(out double y) && y != 0 ? y : box[3] - box[1];
        var m = PdfShadings.Matrix(_file, dictionary["Matrix"u8]).Then(uses.Space);
        double det = m.A * m.D - m.B * m.C;
        if (xStep == 0 || yStep == 0 || !double.IsFinite(xStep) || !double.IsFinite(yStep) || det == 0 || !double.IsFinite(det))
        {
            return;
        }

        bool upright = (Math.Abs(m.B) < 1e-9 && Math.Abs(m.C) < 1e-9) || (Math.Abs(m.A) < 1e-9 && Math.Abs(m.D) < 1e-9);
        var inverse = PdfEditPages.Inverse(m);
        double left = Math.Min(box[0], box[2]), right = Math.Max(box[0], box[2]), bottom = Math.Min(box[1], box[3]), top = Math.Max(box[1], box[3]);
        var under = new List<(PdfBox Box, DocumentColour Fill)>();
        var shown = new List<PdfBox>();
        long budget = MostCells;
        foreach (var use in uses.Uses)
        {
            var (k0, k1) = Cells(inverse, use, left, right, xStep, across: true);
            var (l0, l1) = Cells(inverse, use, bottom, top, yStep, across: false);
            for (long k = k0; k <= k1; k++)
            {
                for (long l = l0; l <= l1; l++)
                {
                    if (--budget < 0)
                    {
                        return;                                                // more cells than are worth counting: one shows clear
                    }

                    // This copy of the cell, and the part of it the paint's box holds (one that only touches it shows nothing).
                    var cell = PdfAnnotations.Bounds(m, left + k * xStep, bottom + l * yStep, right + k * xStep, top + l * yStep);
                    var part = new PdfBox(Math.Max(cell.X, use.X0), Math.Max(cell.Y, use.Y0), Math.Min(cell.Right, use.X1), Math.Min(cell.Bottom, use.Y1));
                    if (part.Empty)
                    {
                        continue;
                    }

                    bool clear = !w.Areas.Any(a => PdfRedactor.Overlap(a, cell.X, cell.Y, cell.Right, cell.Bottom));
                    if (clear && use.Covers && cell.X >= use.X0 - 1e-3 && cell.Right <= use.X1 + 1e-3 && cell.Y >= use.Y0 - 1e-3 && cell.Bottom <= use.Y1 + 1e-3)
                    {
                        return;                                                // a whole cell shown clear: nothing of it is only under an area
                    }

                    // Moved back by this copy's offset on the page: what lies under an area (a paint that reached one),
                    // and what a path shows clear of them.
                    double dx = m.A * k * xStep + m.C * l * yStep, dy = m.B * k * xStep + m.D * l * yStep;
                    if (use.Touched)
                    {
                        for (int a = 0; a < w.Areas.Length; a++)
                        {
                            var area = w.Areas[a];
                            var hidden = part.Cut(new PdfBox(area.X, area.Y, area.Right, area.Bottom));
                            if (!hidden.Empty)
                            {
                                under.Add((hidden.Moved(-dx, -dy), w.AreaFills[a]));
                            }
                        }
                    }

                    if (use.Covers && upright && shown.Count <= MostShifted)
                    {
                        List<PdfBox> pieces = [part];
                        foreach (var area in w.Areas)
                        {
                            pieces = PdfBox.Less(pieces, new PdfBox(area.X, area.Y, area.Right, area.Bottom));
                        }

                        shown.AddRange(pieces.Select(p => p.Moved(-dx, -dy)));
                    }
                }
            }
        }

        // What only an area hides: each part under one, less what a copy shows clear (all of it when there is too much to
        // take away, or the pattern is turned).
        var shifted = new List<DocumentBox>();
        var fills = new List<DocumentColour>();
        foreach (var (hidden, fill) in under)
        {
            List<PdfBox> pieces = [hidden];
            if (upright && shown.Count <= MostShifted)
            {
                foreach (var clear in shown)
                {
                    pieces = PdfBox.Less(pieces, clear);
                    if (pieces.Count == 0 || pieces.Count > MostShifted)
                    {
                        break;
                    }
                }

                if (pieces.Count > MostShifted)
                {
                    pieces = [hidden];
                }
            }

            foreach (var piece in pieces)
            {
                shifted.Add(new DocumentBox((float)piece.X0, (float)piece.Y0, (float)(piece.X1 - piece.X0), (float)(piece.Y1 - piece.Y0)));
                fills.Add(fill);
            }
        }

        if (shifted.Count == 0)
        {
            return;
        }

        if (shifted.Count > MostShifted)
        {
            var whole = PdfAnnotations.Bounds(m, left, bottom, right, top);
            (shifted, fills) = ([new DocumentBox(whole.X - 1, whole.Y - 1, whole.Width + 2, whole.Height + 2)], [fills[0]]);
        }

        var cellResources = _file.Dictionary(dictionary["Resources"u8]) ?? resources;
        PdfRedactedContent inner;
        try
        {
            inner = RedactNested(w, stream!, cellResources, m, box, [.. shifted], [.. fills], bare: false);
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            _issues.Add(DocumentIssue.DamagedStream, $"page {_page}: the pattern /{PdfText.Utf8OrLatin1(raw)} cannot be read, so it is not redacted: {e.Message}");
            return;
        }

        if (inner.Changed)
        {
            w.Result.Patterns.Add((PdfText.Utf8OrLatin1(raw), stream!, cellResources, inner));
            w.Result.Add(inner);
        }
    }

    // The cells along one axis of the pattern's space (each from `first` to `last`, plus a step a cell) that a paint's box
    // reaches, through `inverse` (the page to the pattern's space).
    private static (long From, long To) Cells(PdfMatrix inverse, (double X0, double Y0, double X1, double Y1, bool Touched, bool Covers) use, double first, double last, double step,
        bool across)
    {
        double low = double.MaxValue, high = double.MinValue;
        foreach (var (x, y) in ((double, double)[])[(use.X0, use.Y0), (use.X1, use.Y0), (use.X1, use.Y1), (use.X0, use.Y1)])
        {
            var (px, py) = PdfShadings.Apply(inverse, x, y);
            double v = across ? px : py;
            (low, high) = (Math.Min(low, v), Math.Max(high, v));
        }

        double a = (low - last) / step, b = (high - first) / step;
        return ((long)Math.Ceiling(Math.Clamp(Math.Min(a, b), -1e6, 1e6)), (long)Math.Floor(Math.Clamp(Math.Max(a, b), -1e6, 1e6)));
    }

    // ---- Type 3 glyphs -----------------------------------------------------------------------------------------------

    // The font text is shown in now, when it is a Type 3 font: how this content shows it (read on its first show).
    private PdfType3Redaction? Type3(PdfRewriter w, PdfDictionary? resources)
    {
        if (_state.FontLabel is not { } label || _state.Font is null)
        {
            return null;
        }

        if (w.Type3.TryGetValue(label, out var known))
        {
            return known;
        }

        PdfType3Redaction? made = null;
        try
        {
            var category = resources is null ? null : _file.Dictionary(resources["Font"u8]);
            int at = -1;
            for (int i = 0; i < (category?.Count ?? 0) && at < 0; i++)
            {
                at = string.Equals(category!.KeyAt(i).ToString(), label, StringComparison.Ordinal) ? i : -1;
            }

            if (at >= 0 && _file.Dictionary(category!.ValueAt(at)) is { } font && _file.Resolve(font["Subtype"u8]).IsName("Type3"u8)
                && _file.Dictionary(font["CharProcs"u8]) is { } procs)
            {
                var matrix = PdfShadings.Matrix(_file, font["FontMatrix"u8]);
                if (matrix == PdfMatrix.Identity)
                {
                    matrix = new PdfMatrix(0.001, 0, 0, 0.001, 0, 0);          // as the face reads a font that gives none
                }

                var glyphs = new PdfName?[256];
                if (_file.Dictionary(font["Encoding"u8]) is { } encoding && _file.Resolve(encoding["Differences"u8]).AsArray is { } differences)
                {
                    int code = -1;
                    foreach (var item in differences.Items)
                    {
                        var value = _file.Resolve(item);
                        if (value.TryInteger(out long start))
                        {
                            code = (int)Math.Clamp(start, -1, 256);
                        }
                        else if (value.AsName is { } glyph && code is >= 0 and < 256)
                        {
                            glyphs[code++] = glyph;
                        }
                    }
                }

                var own = _file.Dictionary(font["Resources"u8]);
                made = new PdfType3Redaction(label, category.KeyAt(at), font, procs, own ?? resources, own is not null, matrix, glyphs, new PdfNames(own ?? resources));
            }
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            made = null;                                                       // the reader counted it already
        }

        w.Type3[label] = made;
        return made;
    }

    // A Type 3 glyph kept with the text: when its drawing reaches an area and its description, redacted where the glyph
    // stands, changed, the copy's code in the copies' font; else -1 (drawn as it was).
    private int Type3Glyph(PdfRewriter w, PdfType3Redaction type3, in Glyph glyph)
    {
        int code = (int)(glyph.Code & 0xFF);
        if (type3.Redacted.Count >= 256 || type3.Glyphs[code] is not { } glyphName || !Type3Reaches(w, type3, glyph))
        {
            return -1;
        }

        PdfStream? proc;
        try
        {
            proc = _file.Resolve(type3.CharProcs[glyphName.Bytes]).AsStream;
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            return -1;
        }

        if (proc is null)
        {
            return -1;
        }

        PdfRedactedContent inner;
        try
        {
            inner = RedactNested(w, proc, type3.Resources, Type3Matrix(type3, glyph), null, w.Areas, w.AreaFills, bare: true, type3.Names);
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            _issues.Add(DocumentIssue.DamagedStream, $"page {_page}: the Type 3 glyph /{glyphName} of /{type3.Name} cannot be read, so it is not redacted: {e.Message}");
            return -1;
        }

        if (!inner.Changed)
        {
            return -1;
        }

        double width = 0;
        if (_file.Resolve(type3.Font["Widths"u8]).AsArray is { } widths && _file.Resolve(type3.Font["FirstChar"u8]).TryInteger(out long firstChar)
            && code - firstChar is >= 0 and var at && at < widths.Count && _file.Resolve(widths[(int)at]).TryNumber(out double given))
        {
            width = given;
        }

        int copy = type3.Redacted.Count;
        type3.Own ??= w.Name("RdFt", "Font"u8);
        string text = glyph.Unmapped ? "" : new string(_chars, glyph.TextStart, glyph.TextLength);
        type3.Redacted.Add((string.Create(CultureInfo.InvariantCulture, $"RdGl{copy}"), width, text, inner));
        w.Result.Add(inner);
        return copy;
    }

    // Whether a Type 3 glyph's drawing (measured: Type3Box) reaches an area where the glyph stands.
    private bool Type3Reaches(PdfRewriter w, PdfType3Redaction type3, in Glyph glyph)
    {
        if (type3.Glyphs[(int)(glyph.Code & 0xFF)] is not { } name || Type3Box(type3, name) is not { } box)
        {
            return false;                                                      // it draws nothing
        }

        if (double.IsNaN(box[0]))
        {
            return true;
        }

        var bounds = PdfAnnotations.Bounds(Type3Matrix(type3, glyph), box[0], box[1], box[2], box[3]);
        return w.Areas.Any(a => PdfRedactor.Overlap(a, bounds.X, bounds.Y, bounds.Right, bounds.Bottom));
    }

    // The box a Type 3 glyph's description draws in, in its glyph space, read once a content: around its images, paths
    // (their strokes' half widths), text and form parts; null when it draws nothing; NaN when it may draw anywhere (a
    // shading over its clip, or a description that cannot be read).
    private double[]? Type3Box(PdfType3Redaction type3, PdfName glyph)
    {
        string key = glyph.ToString();
        if (type3.Boxes.TryGetValue(key, out var known))
        {
            return known;
        }

        double[]? box = null;
        try
        {
            if (_file.Resolve(type3.CharProcs[glyph.Bytes]).AsStream is { } proc)
            {
                double x0 = double.MaxValue, y0 = double.MaxValue, x1 = double.MinValue, y1 = double.MinValue;
                void Add(double ax, double ay, double bx, double by) => (x0, y0, x1, y1) = (Math.Min(x0, ax), Math.Min(y0, ay), Math.Max(x1, bx), Math.Max(y1, by));
                using var reader = new PdfContentReader(_file, _page, new PdfIssues(), _includeHidden);
                reader.ReadImages();
                reader.ReadVector();
                using (var decoded = _file.Decode(proc))
                {
                    reader.ReadContent(decoded.Span, type3.Resources, PdfMatrix.Identity, PdfMatrix.Identity, null, 0);
                }

                foreach (var image in reader.Images)
                {
                    Add(image.Box.X, image.Box.Y, image.Box.Right, image.Box.Bottom);
                }

                foreach (var path in reader.Paths)
                {
                    float half = (path.Stroke ?? 0f) / 2f;
                    var points = path.Points.Span;
                    for (int i = 0; i + 1 < points.Length; i += 2)
                    {
                        Add(points[i] - half, points[i + 1] - half, points[i] + half, points[i + 1] + half);
                    }
                }

                foreach (var run in reader.TextRuns(glyphs: false, out _, out _))
                {
                    Add(run.Box.X, run.Box.Y, run.Box.Right, run.Box.Bottom);
                }

                box = reader.Shadings.Count > 0 ? [double.NaN, double.NaN, double.NaN, double.NaN] : x0 <= x1 ? [x0, y0, x1, y1] : null;
            }
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            box = [double.NaN, double.NaN, double.NaN, double.NaN];
        }

        type3.Boxes[key] = box;
        return box;
    }

    // A Type 3 glyph's space on the page: its font's matrix, then the text space at the glyph's origin.
    private PdfMatrix Type3Matrix(PdfType3Redaction type3, in Glyph glyph)
    {
        var s = _segments[glyph.Segment];
        return type3.Matrix.Then(new PdfMatrix(s.A, s.B, s.C, s.D, glyph.X, glyph.Y));
    }

    // A Type 3 font once its content is read: each glyph no longer drawn here (every one of its glyphs gone) leaves the
    // content's copy of it, unless a form takes the content's resources; the resource names what stays uses are found so
    // the copies hold only those. Nothing when no glyph left and none was copied.
    private void FinishType3(PdfRewriter w, PdfType3Redaction font)
    {
        if (!w.Inherits)
        {
            var kept = new HashSet<string>(StringComparer.Ordinal);
            for (int c = 0; c < 256; c++)
            {
                if (font.Kept[c] > 0 && font.Glyphs[c] is { } name)
                {
                    kept.Add(name.ToString());
                }
            }

            for (int c = 0; c < 256; c++)
            {
                if (font.Gone[c] > 0 && font.Glyphs[c] is { } name && !kept.Contains(name.ToString()))
                {
                    font.Dropped.Add(name.ToString());
                }
            }
        }

        if (font.Dropped.Count == 0 && font.Redacted.Count == 0)
        {
            return;
        }

        if (font.Dropped.Count > 0 && font.OwnResources)
        {
            var used = new HashSet<string>(StringComparer.Ordinal);
            for (int k = 0; k < font.CharProcs.Count && used is not null; k++)
            {
                if (font.Dropped.Contains(font.CharProcs.KeyAt(k).ToString()))
                {
                    continue;
                }

                try
                {
                    if (_file.Resolve(font.CharProcs.ValueAt(k)).AsStream is { } proc)
                    {
                        using var decoded = _file.Decode(proc);
                        NamesIn(decoded.Span, used);
                    }
                }
                catch (InvalidDataException e) when (!PdfLimit.Is(e))
                {
                    used = null;                                               // a glyph that cannot be read: every resource kept for it
                }
            }

            font.Used = used;
        }

        if (font.Redacted.Count > 0)
        {
            var used = new HashSet<string>(StringComparer.Ordinal);
            foreach (var copy in font.Redacted)
            {
                NamesIn(copy.Content.Content, used);
            }

            font.CopiesUse = used;
        }

        w.Result.Fonts.Add(font);
    }

    // Every name a content's operators take (an inline image's entries among them; its data passed over to its end).
    private static void NamesIn(ReadOnlySpan<byte> content, HashSet<string> names)
    {
        var lexer = new PdfContentLexer(content);
        Span<byte> buffer = stackalloc byte[128];
        while (true)
        {
            var token = lexer.Next(out int start, out int length, out _);
            if (token == PdfToken.End)
            {
                return;
            }

            if (token == PdfToken.Name)
            {
                names.Add(PdfText.Utf8OrLatin1(NameBytes(content.Slice(start, length), buffer)));
            }
            else if (token == PdfToken.Keyword && content.Slice(start, length).SequenceEqual("ID"u8))
            {
                // The data: up to an "EI" between whitespace (or a delimiter, or the end).
                int at = lexer.Position + 1;
                while (at < content.Length)
                {
                    int found = content[at..].IndexOf("EI"u8);
                    if (found < 0)
                    {
                        return;
                    }

                    int p = at + found;
                    if (PdfParser.IsWhitespace(content[p - 1]) && (p + 2 == content.Length || PdfParser.IsWhitespace(content[p + 2]) || PdfParser.IsDelimiter(content[p + 2])))
                    {
                        lexer.Position = p + 2;
                        break;
                    }

                    at = p + 1;
                }

                if (at >= content.Length)
                {
                    return;
                }
            }
        }
    }
}

/// <summary>A box of the page (two corners, page points), for the parts of a pattern's copies a redaction weighs.</summary>
internal readonly record struct PdfBox(double X0, double Y0, double X1, double Y1)
{
    /// <summary>Whether it holds no area (a millionth of a point either way at most).</summary>
    public bool Empty => X1 - X0 <= 1e-6 || Y1 - Y0 <= 1e-6;

    /// <summary>The part it shares with <paramref name="other"/>.</summary>
    public PdfBox Cut(PdfBox other) => new(Math.Max(X0, other.X0), Math.Max(Y0, other.Y0), Math.Min(X1, other.X1), Math.Min(Y1, other.Y1));

    /// <summary>It moved by (<paramref name="dx"/>, <paramref name="dy"/>).</summary>
    public PdfBox Moved(double dx, double dy) => new(X0 + dx, Y0 + dy, X1 + dx, Y1 + dy);

    /// <summary><paramref name="boxes"/> less <paramref name="cut"/>: each box that meets it split into the up to four boxes around it.</summary>
    public static List<PdfBox> Less(List<PdfBox> boxes, PdfBox cut)
    {
        var left = new List<PdfBox>(boxes.Count);
        foreach (var b in boxes)
        {
            if (b.Cut(cut).Empty)
            {
                left.Add(b);
                continue;
            }

            Add(left, new PdfBox(b.X0, b.Y0, b.X1, cut.Y0));                               // above it
            Add(left, new PdfBox(b.X0, cut.Y1, b.X1, b.Y1));                               // below it
            Add(left, new PdfBox(b.X0, Math.Max(b.Y0, cut.Y0), cut.X0, Math.Min(b.Y1, cut.Y1)));   // to its left
            Add(left, new PdfBox(cut.X1, Math.Max(b.Y0, cut.Y0), b.X1, Math.Min(b.Y1, cut.Y1)));   // to its right
        }

        return left;
    }

    private static void Add(List<PdfBox> boxes, PdfBox box)
    {
        if (!box.Empty)
        {
            boxes.Add(box);
        }
    }
}
