// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.Data.Pdf;

/// <summary>
/// The font a field's text is written in: a simple font the document holds (or the writer adds: a standard font), written
/// with its own one-byte codes and widths and named in the appearance's resources as the field's default appearance names
/// it; or faces the text is set in through <c>DocumentTextSetters</c> (every script: the shapers and the reading
/// order), each embedded through the writer's font path (D6: a composite font of the glyphs used, its ToUnicode).
/// </summary>
internal sealed class PdfFieldFont
{
    private readonly Dictionary<int, byte>? _codes;
    private readonly double[]? _widths;

    private PdfFieldFont(byte[]? resource, Dictionary<int, byte>? codes, double[]? widths, IReadOnlyList<IFontFace>? faces, double ascent, double descent, double capHeight)
    {
        (Resource, _codes, _widths, Faces) = (resource, codes, widths, faces);
        (Ascent, Descent, CapHeight) = (ascent, descent, capHeight);
    }

    /// <summary>A simple font's name in the appearance's <c>/Font</c> resources (its default appearance's), or null for faces.</summary>
    public byte[]? Resource { get; }

    /// <summary>A simple font's object in the document edited (its form's resources' entry), or null.</summary>
    public PdfValue Source { get; private init; }

    /// <summary>A simple font's object number in the document written (a standard font the writer adds), or 0.</summary>
    public int Number { get; private init; }

    /// <summary>The faces text is set in (the first, then the others for what it lacks), or null for a simple font.</summary>
    public IReadOnlyList<IFontFace>? Faces { get; }

    /// <summary>How far its glyphs reach above the baseline, at size 1.</summary>
    public double Ascent { get; }

    /// <summary>How far below (negative), at size 1.</summary>
    public double Descent { get; }

    /// <summary>The height of its capitals at size 1 (a single line is centred by it).</summary>
    public double CapHeight { get; }

    /// <summary>
    /// A simple font named <paramref name="resource"/>: each character's code from <paramref name="font"/>'s own (the
    /// first code giving it; one whose glyph its embedded program lacks left out), widths and extents.
    /// </summary>
    public static PdfFieldFont Simple(byte[] resource, PdfFont font, PdfValue source)
    {
        var codes = new Dictionary<int, byte>();
        var widths = new double[256];
        Span<char> text = stackalloc char[8];
        bool embedded = font.Face is not null;
        for (int code = 255; code >= 0; code--)
        {
            widths[code] = font.Width((uint)code, 1);
            int n = font.Text((uint)code, 1, text);
            int codePoint = n == 1 && !char.IsSurrogate(text[0]) ? text[0] : n == 2 && char.IsSurrogatePair(text[0], text[1]) ? char.ConvertToUtf32(text[0], text[1]) : -1;
            if (codePoint > 0 && (!embedded || font.GlyphId((uint)code, 1) > 0))
            {
                codes[codePoint] = (byte)code;                                // the lowest code giving the character wins (walked down)
            }
        }

        return new PdfFieldFont(resource, codes, widths, null, font.Ascent, font.Descent, font.Ascent) { Source = source };
    }

    /// <summary>A standard font (its index in the standard 14) written in the WinAnsi encoding as object <paramref name="number"/>, named <paramref name="resource"/>.</summary>
    public static PdfFieldFont Standard(byte[] resource, int standard, int number)
    {
        var codes = new Dictionary<int, byte>();
        var widths = new double[256];
        Span<char> text = stackalloc char[PdfFontTables.MaxGlyphText];
        for (int code = 255; code >= 0; code--)
        {
            var glyph = PdfFontTables.EncodingName(PdfFontTables.WinAnsiEncoding, code);
            int n = glyph.IsEmpty ? 0 : PdfFontTables.GlyphText(glyph, false, text);
            if (n == 1 && !char.IsSurrogate(text[0]))
            {
                int width = PdfFontTables.StandardWidth(standard, text[0]);
                if (width >= 0)
                {
                    codes[text[0]] = (byte)code;
                    widths[code] = width / 1000.0;
                }
            }
        }

        var (ascent, descent) = PdfFontTables.StandardExtents(standard);
        return new PdfFieldFont(resource, codes, widths, null, ascent / 1000.0, descent / 1000.0, ascent / 1000.0) { Number = number };
    }

    /// <summary>Faces text is set in: their first one's extents (its capitals' height from its OS/2 table when it states it).</summary>
    public static PdfFieldFont OfFaces(IReadOnlyList<IFontFace> faces)
    {
        var face = faces[0];
        double unit = Math.Max(1, face.Metrics.UnitsPerEm);
        double ascent = face.Metrics.Ascender / unit, descent = face.Metrics.Descender / unit, cap = ascent;
        var os2 = face.Table("OS/2").Span;
        if (os2.Length >= 90 && System.Buffers.Binary.BinaryPrimitives.ReadUInt16BigEndian(os2) >= 2)
        {
            short stated = System.Buffers.Binary.BinaryPrimitives.ReadInt16BigEndian(os2[88..]);
            if (stated > 0)
            {
                cap = stated / unit;
            }
        }

        return new PdfFieldFont(null, null, null, faces, ascent, descent, cap);
    }

    /// <summary>Whether it is a simple font (codes and widths of its own).</summary>
    public bool IsSimple => _codes is not null;

    /// <summary>Whether a simple font has a code for every character of <paramref name="text"/> (line ends aside: they break lines, not shown).</summary>
    public bool Encodes(string text)
    {
        if (_codes is null)
        {
            return false;
        }

        foreach (var rune in text.EnumerateRunes())
        {
            if (rune.Value is not ('\n' or '\r') && !_codes.ContainsKey(rune.Value))
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>A simple font's codes for <paramref name="text"/> (every character encoded, <see cref="Encodes"/>).</summary>
    public byte[] Encode(string text)
    {
        var bytes = new List<byte>(text.Length);
        foreach (var rune in text.EnumerateRunes())
        {
            bytes.Add(_codes![rune.Value]);
        }

        return [.. bytes];
    }

    /// <summary>A simple font's width of <paramref name="text"/> at size 1.</summary>
    public double Width(string text)
    {
        double width = 0;
        foreach (var rune in text.EnumerateRunes())
        {
            if (_codes!.TryGetValue(rune.Value, out byte code))
            {
                width += _widths![code];
            }
        }

        return width;
    }
}

/// <summary>A field's look to make: its kind, values and choices, how its text is shown, its box's colours and its mark.</summary>
internal sealed record PdfFieldLook(DocumentFieldKind Kind, DocumentFieldFlags Flags)
{
    public IReadOnlyList<string> Values { get; init; } = [];

    public IReadOnlyList<DocumentFieldOption> Options { get; init; } = [];

    public DocumentFieldAlignment Alignment { get; init; }

    public int? MaxLength { get; init; }

    /// <summary>The text's size in points; 0 fits it to the box.</summary>
    public float FontSize { get; init; }

    public DocumentColour TextColour { get; init; }

    public DocumentColour? Background { get; init; }

    public DocumentColour? Border { get; init; }

    /// <summary>The border's width (drawn only with a border colour, but a beveled or inset box's shading).</summary>
    public float BorderWidth { get; init; } = 1f;

    /// <summary>How the border is drawn.</summary>
    public DocumentFieldBorderStyle BorderStyle { get; init; }

    /// <summary>A dashed border's dashes and gaps (none: 3 on, 3 off).</summary>
    public IReadOnlyList<float> Dashes { get; init; } = [];

    /// <summary>The value as rich text (runs each in its own style), or null for plain text.</summary>
    public IReadOnlyList<DocumentFieldRun>? Rich { get; init; }

    /// <summary>Whether the background and border are drawn (false: an existing appearance's own are kept around the text).</summary>
    public bool Framed { get; init; } = true;

    public DocumentFieldMark Mark { get; init; }

    public string? Caption { get; init; }

    /// <summary>A list's first choice shown (<c>/TI</c>).</summary>
    public int TopIndex { get; init; }

    /// <summary>The text a text field or a drop-down list shows (its value; a password's characters as asterisks), or "".</summary>
    public string Shown()
    {
        string value = Values.Count > 0 ? Values[0] : "";
        if (Kind == DocumentFieldKind.Combo)
        {
            foreach (var option in Options)
            {
                if (option.Export == value)
                {
                    return option.Text;
                }
            }

            return value;
        }

        if ((Flags & DocumentFieldFlags.Password) != 0)
        {
            return new string('*', new StringInfo(value).LengthInTextElements);
        }

        if ((Flags & DocumentFieldFlags.Multiline) == 0)
        {
            value = value.Replace("\r\n", " ", StringComparison.Ordinal).Replace('\n', ' ').Replace('\r', ' ');
        }

        return value;
    }
}

/// <summary>
/// The parts of an existing appearance a new value's text goes into (ISO 32000-2, 12.7.4.3): its content before and after
/// its <c>/Tx BMC ... EMC</c> section (the producer's own box, border and drawing kept), and its resources.
/// </summary>
/// <param name="Prefix">Its content before the section.</param>
/// <param name="Suffix">Its content after the section.</param>
/// <param name="Resources">Its resources (null for none).</param>
/// <param name="File">The file it is of (its resources' entries resolved through it).</param>
internal sealed record PdfAppearanceBase(byte[] Prefix, byte[] Suffix, PdfDictionary? Resources, PdfFile File)
{
    /// <summary>
    /// The base of <paramref name="stream"/> when its box is the widget's as turned by <paramref name="rotation"/>
    /// (<c>[0 0 width height]</c>; its matrix none, or one that only turns it so) and its content holds a <c>/Tx</c> marked
    /// section; else null (a new appearance replaces it whole).
    /// </summary>
    public static PdfAppearanceBase? Of(PdfFile file, PdfStream stream, float width, float height, int rotation = 0)
    {
        var box = PdfFunction.Numbers(file, stream.Dictionary["BBox"u8]);
        var matrix = PdfFunction.Numbers(file, stream.Dictionary["Matrix"u8]);
        var turn = PdfAppearance.Turn(rotation, width, height);
        bool turned = turn is { } t
            ? matrix is { Length: 6 } m && m[0] == t.A && m[1] == t.B && m[2] == t.C && m[3] == t.D
            : matrix is null || (matrix.Length == 6 && matrix[0] == 1 && matrix[1] == 0 && matrix[2] == 0 && matrix[3] == 1 && matrix[4] == 0 && matrix[5] == 0);
        if (box is not { Length: 4 } || Math.Abs(Math.Min(box[0], box[2])) > 0.01 || Math.Abs(Math.Min(box[1], box[3])) > 0.01
            || Math.Abs(Math.Abs(box[2] - box[0]) - width) > 0.01 || Math.Abs(Math.Abs(box[3] - box[1]) - height) > 0.01 || !turned)
        {
            return null;
        }

        using var decoded = file.Decode(stream);
        var data = decoded.Span;
        var lexer = new PdfContentLexer(data);
        int start = -1, depth = 0, nameStart = -1;
        bool tx = false;
        while (true)
        {
            var token = lexer.Next(out int at, out int length, out _);
            if (token == PdfToken.End)
            {
                return null;                                                   // no section, or one never closed
            }

            if (token == PdfToken.Name)
            {
                (tx, nameStart) = (data.Slice(at, length).SequenceEqual("Tx"u8), at - 1);
                continue;
            }

            if (token != PdfToken.Keyword)
            {
                tx = false;
                continue;
            }

            var word = data.Slice(at, length);
            if (start < 0)
            {
                if (tx && word.SequenceEqual("BMC"u8))
                {
                    (start, depth) = (nameStart, 1);
                }
            }
            else if (word.SequenceEqual("BMC"u8) || word.SequenceEqual("BDC"u8))
            {
                depth++;
            }
            else if (word.SequenceEqual("EMC"u8) && --depth == 0)
            {
                return new PdfAppearanceBase(data[..start].ToArray(), data[(at + length)..].ToArray(), file.Dictionary(stream.Dictionary["Resources"u8]), file);
            }

            tx = false;
        }
    }
}

/// <summary>
/// A widget's appearance made by <see cref="PdfFieldAppearances"/>: a form XObject of its box's size (its width and height
/// swapped when it is turned a quarter), its content, and the resources it names (the embedded faces' fonts, simple fonts,
/// opacities); or, over an existing appearance's <see cref="PdfAppearanceBase"/>, the text section that replaces the
/// existing one. A turned one maps its box back onto the widget's by its <c>/Matrix</c> (12.5.5), as <c>/MK /R</c> says.
/// </summary>
internal sealed class PdfAppearance(float width, float height, byte[] content, IReadOnlyList<PdfWriterFont> fonts, IReadOnlyList<(float Fill, float Stroke)> alphas,
    IReadOnlyList<PdfFieldFont> simple)
{
    public float Width { get; } = width;

    public float Height { get; } = height;

    public byte[] Content { get; } = content;

    /// <summary>The faces' fonts it names (<c>/F</c> and each one's index).</summary>
    public IReadOnlyList<PdfWriterFont> Fonts { get; } = fonts;

    /// <summary>The opacities it sets (<c>/GS1</c>, ...).</summary>
    public IReadOnlyList<(float Fill, float Stroke)> Alphas { get; } = alphas;

    /// <summary>The simple fonts it names, each by its <see cref="PdfFieldFont.Resource"/> (none, one, or a rich text's several).</summary>
    public IReadOnlyList<PdfFieldFont> Simple { get; } = simple;

    /// <summary>The existing appearance its content is the text section of, or null when it stands alone.</summary>
    public PdfAppearanceBase? Base { get; init; }

    /// <summary>How far its content is turned counter-clockwise (0, 90, 180 or 270): its box is the widget's turned, its matrix turns it back.</summary>
    public int Rotation { get; set; }

    /// <summary>The same appearance over <paramref name="kept"/> (its text section in place of the existing one's).</summary>
    public PdfAppearance Over(PdfAppearanceBase kept) => new(Width, Height, Content, Fonts, Alphas, Simple) { Base = kept, Rotation = Rotation };

    /// <summary>The matrix that turns a box of <paramref name="width"/> x <paramref name="height"/> turned by <paramref name="rotation"/> back onto its widget (from the origin), or null for none.</summary>
    public static (float A, float B, float C, float D, float E, float F)? Turn(int rotation, float width, float height) => rotation switch
    {
        90 => (0, 1, -1, 0, height, 0),
        180 => (-1, 0, 0, -1, width, height),
        270 => (0, -1, 1, 0, 0, width),
        _ => null,
    };

    /// <summary>
    /// Writes it as object <paramref name="number"/> (a form XObject, its content deflated), each simple font's resource
    /// value written by <paramref name="simpleFont"/> and a base's resources' values by <paramref name="value"/>; what its
    /// dictionary refers to is written first (a copier's references), and an encrypting file gets the dictionary written
    /// again with its strings encrypted.
    /// </summary>
    public void Write(PdfFileWriter file, int number, Action<PdfSyntax, PdfFieldFont> simpleFont, Action<PdfSyntax, PdfValue>? value = null)
    {
        var own = Base?.Resources ?? PdfDictionary.Empty;
        // A category of resources: the base's own entries (but those this one names), then this one's.
        void Category(PdfSyntax s, ReadOnlySpan<byte> category, Action<PdfSyntax> mine, Func<PdfName, bool> taken)
        {
            s.Name(category).Raw("<<"u8);
            if (value is not null && Base is { } b && b.File.Dictionary(own[category]) is { } dictionary)
            {
                for (int i = 0; i < dictionary.Count; i++)
                {
                    if (!taken(dictionary.KeyAt(i)))
                    {
                        s.Name(dictionary.KeyAt(i).Bytes).Space();
                        value(s, dictionary.ValueAt(i));
                    }
                }
            }

            mine(s);
            s.Raw(">>"u8);
        }

        void Entries(PdfSyntax s)
        {
            s.Raw("/Type/XObject/Subtype/Form/BBox[0 0 "u8).Number(Width).Space().Number(Height).Raw("]"u8);
            if (Turn(Rotation, Width, Height) is { } m)
            {
                s.Raw("/Matrix["u8).Number(m.A).Space().Number(m.B).Space().Number(m.C).Space().Number(m.D).Space().Number(m.E).Space().Number(m.F).Raw("]"u8);
            }

            s.Raw("/Resources<<"u8);
            bool fonts = Fonts.Count > 0 || Simple.Count > 0 || !own["Font"u8].IsNull;
            if (fonts)
            {
                Category(s, "Font"u8, t =>
                {
                    foreach (var font in Fonts)
                    {
                        t.Raw("/F"u8).Integer(font.Index).Space().Reference(font.Number);
                    }

                    foreach (var simple in Simple)
                    {
                        t.Name(simple.Resource!).Space();                      // a reference follows: "N G R"
                        simpleFont(t, simple);
                    }
                }, key => Simple.Any(f => key.Is(f.Resource)) || Fonts.Any(f => key.Is(Encoding.ASCII.GetBytes("F" + f.Index.ToString(CultureInfo.InvariantCulture)))));
            }

            if (Alphas.Count > 0 || !own["ExtGState"u8].IsNull)
            {
                Category(s, "ExtGState"u8, t =>
                {
                    for (int i = 0; i < Alphas.Count; i++)
                    {
                        t.Raw("/GS"u8).Integer(i + 1).Raw("<</ca "u8).Number(Alphas[i].Fill).Raw("/CA "u8).Number(Alphas[i].Stroke).Raw(">>"u8);
                    }
                }, key => Enumerable.Range(1, Alphas.Count).Any(i => key.Is(Encoding.ASCII.GetBytes("GS" + i.ToString(CultureInfo.InvariantCulture)))));
            }

            // The base's other resources, as they are.
            for (int i = 0; value is not null && i < own.Count; i++)
            {
                if (!own.KeyAt(i).Is("Font"u8) && !own.KeyAt(i).Is("ExtGState"u8))
                {
                    s.Name(own.KeyAt(i).Bytes).Space();
                    value(s, own.ValueAt(i));
                }
            }

            s.Raw(">>"u8);
        }

        using var scratch = new PdfSyntax(256);
        Entries(scratch);                                                      // what the resources refer to written first
        var target = file.BeginStream(number);
        if (target.Encrypts)
        {
            Entries(target);
        }
        else
        {
            target.Raw(scratch.Span);
        }

        if (Base is { } b)
        {
            var whole = new byte[b.Prefix.Length + Content.Length + b.Suffix.Length];
            b.Prefix.CopyTo(whole, 0);
            Content.CopyTo(whole, b.Prefix.Length);
            b.Suffix.CopyTo(whole, b.Prefix.Length + Content.Length);
            file.EndStream(whole, deflate: true);
            return;
        }

        file.EndStream(Content, deflate: true);
    }
}

/// <summary>
/// Makes form fields' appearances (ISO 32000-2, 12.7.4.3): a text box's value (one line centred on its capitals, several
/// lines from the top wrapped to the box, comb cells, a password's asterisks), a list's choices with the chosen rows
/// highlighted, a drop-down list's choice, a push button's caption, a check box's or radio button's mark (drawn as a
/// path: no font needed) on and off, a signature field's box; each on its background and inside its border. Text is in
/// the field's simple font when that has every character, else set in faces through the text setter (the shapers and the
/// reading order: every script) and embedded. A size of 0 fits the text to the box (<see cref="DocumentFormOptions"/>).
/// </summary>
internal static class PdfFieldAppearances
{
    // The step a fitted size of several lines shrinks by.
    private const float SizeStep = 0.5f;

    /// <summary>A text box's or a drop-down list's appearance, <paramref name="width"/> x <paramref name="height"/> points.</summary>
    public static PdfAppearance Text(PdfFieldLook look, PdfFieldFont font, float width, float height, Func<IFontFace, PdfWriterFont> faces, DocumentFormOptions options)
    {
        var page = new PdfPageBuilder(faces, NoImages, 0, width, height);
        try
        {
            string shown = look.Shown();
            float inset = Inset(look, options);
            float innerWidth = Math.Max(0f, width - 2 * inset), innerHeight = Math.Max(0f, height - 2 * inset);
            bool multiline = look.Kind == DocumentFieldKind.Text && (look.Flags & DocumentFieldFlags.Multiline) != 0;
            bool comb = look.Kind == DocumentFieldKind.Text && (look.Flags & DocumentFieldFlags.Comb) != 0 && look.MaxLength is > 0 && !multiline
                        && (look.Flags & DocumentFieldFlags.Password) == 0;
            var placed = new List<Placed>();
            if (shown.Length > 0)
            {
                if (comb)
                {
                    Comb(placed, shown, look, font, width, height, inset, options);
                }
                else if (multiline)
                {
                    Lines(placed, shown, look, font, width, inset, innerWidth, innerHeight, options);
                }
                else
                {
                    Line(placed, shown, look, font, width, height, inset, innerWidth, innerHeight, options);
                }
            }

            return Finish(page, placed, look, font, width, height, Inset(look, 0f), text: true);
        }
        finally
        {
            page.Release();
        }
    }

    /// <summary>
    /// A rich text box's appearance, <paramref name="width"/> x <paramref name="height"/> points: the look's runs, each in the
    /// font <paramref name="runFont"/> chooses for its text, its own size (the field's unless it sets one; a field's size
    /// of 0 fits the plain text as a plain value fits) and colour, underlined or struck through; one line centred on the
    /// tallest capitals, or several wrapped from the top at spaces (each paragraph on a line of its own), aligned.
    /// </summary>
    public static PdfAppearance Rich(PdfFieldLook look, PdfFieldFont font, Func<DocumentFieldRun, string, PdfFieldFont> runFont, float width, float height,
        Func<IFontFace, PdfWriterFont> faces, DocumentFormOptions options)
    {
        var page = new PdfPageBuilder(faces, NoImages, 0, width, height);
        try
        {
            float inset = Inset(look, options);
            float innerWidth = Math.Max(0f, width - 2 * inset), innerHeight = Math.Max(0f, height - 2 * inset);
            bool multiline = (look.Flags & DocumentFieldFlags.Multiline) != 0;
            var pieces = new List<Piece>();
            foreach (var run in look.Rich ?? [])
            {
                string text = multiline ? run.Text : OneLine(run.Text);
                if (text.Length > 0)
                {
                    pieces.Add(new Piece(run, text, runFont(run, text)));
                }
            }

            var placed = new List<Placed>();
            if (pieces.Count > 0)
            {
                float size = look.FontSize;
                List<List<Segment>> lines;
                if (multiline)
                {
                    if (size > 0)
                    {
                        lines = RichLines(pieces, size, innerWidth, options);
                    }
                    else
                    {
                        size = Math.Max(options.MaxFontSize, options.MinFontSize);
                        while (true)
                        {
                            lines = RichLines(pieces, size, innerWidth, options);
                            if (RichHeight(lines, options) <= innerHeight || size - SizeStep < options.MinFontSize)
                            {
                                break;
                            }

                            size -= SizeStep;
                        }
                    }
                }
                else
                {
                    if (size <= 0)
                    {
                        string plain = OneLine(PdfRichText.Plain(look.Rich!));
                        double extent = Math.Max(1e-3, font.Ascent - font.Descent);
                        double tall = innerHeight / extent, wide = Measure(plain, font, 1f, options) is > 0 and var w ? innerWidth / w : double.MaxValue;
                        size = (float)Math.Max(options.MinFontSize, Math.Min(tall, wide));
                    }

                    lines = RichLines(pieces, size, float.PositiveInfinity, options);
                }

                float baseline = 0;
                for (int k = 0; k < lines.Count; k++)
                {
                    var line = lines[k];
                    if (multiline)
                    {
                        baseline = k == 0 ? inset + (float)Max(line, s => s.Piece.Font.Ascent * s.Size) : baseline + (float)Max(line, s => s.Size) * options.LineSpacing;
                    }
                    else
                    {
                        baseline = height - (height - (float)Max(line, s => s.Piece.Font.CapHeight * s.Size)) / 2;
                    }

                    float lineWidth = 0;
                    foreach (var segment in line)
                    {
                        lineWidth += segment.Set.Width;
                    }

                    float x = look.Alignment switch
                    {
                        DocumentFieldAlignment.Centre => inset + (innerWidth - lineWidth) / 2,
                        DocumentFieldAlignment.Right => inset + innerWidth - lineWidth,
                        _ => inset,
                    };
                    foreach (var segment in line)
                    {
                        var run = segment.Piece.Run;
                        placed.Add(new Placed(segment.Set, segment.Size, x, baseline)
                        {
                            Font = segment.Piece.Font, Colour = run.Colour, Underline = run.Underline, StrikeOut = run.StrikeOut,
                        });
                        x += segment.Set.Width;
                    }
                }
            }

            return Finish(page, placed, look, font, width, height, Inset(look, 0f), text: true);
        }
        finally
        {
            page.Release();
        }
    }

    /// <summary>A list's appearance: its choices from its first shown, the chosen rows highlighted.</summary>
    public static PdfAppearance List(PdfFieldLook look, PdfFieldFont font, float width, float height, Func<IFontFace, PdfWriterFont> faces, DocumentFormOptions options)
    {
        var page = new PdfPageBuilder(faces, NoImages, 0, width, height);
        try
        {
            float inset = Inset(look, options), edge = Inset(look, 0f);
            float innerWidth = Math.Max(0f, width - 2 * inset);
            float size = look.FontSize;
            if (size <= 0)
            {
                double widest = look.Options.Count == 0 ? 0 : look.Options.Max(o => Measure(o.Text, font, 1f, options));
                size = Math.Max(options.MinFontSize, widest > 0 ? (float)Math.Min(options.MaxFontSize, innerWidth / widest) : options.MaxFontSize);
            }

            float row = size * options.LineSpacing;
            var placed = new List<Placed>();
            float top = edge;
            for (int k = Math.Clamp(look.TopIndex, 0, Math.Max(0, look.Options.Count - 1)), r = 0; k < look.Options.Count && top + r * row < height; k++, r++)
            {
                var option = look.Options[k];
                float rowTop = top + r * row;
                if (look.Values.Contains(option.Export) || look.Values.Contains(option.Text))
                {
                    page.Path(Filled(Corners(new DocumentBox(edge, rowTop, Math.Max(0f, width - 2 * edge), row)), options.Highlight));
                }

                float baseline = rowTop + (float)(row - (font.Ascent - font.Descent) * size) / 2 + (float)(font.Ascent * size);
                Place(placed, option.Text, font, size, inset, baseline, innerWidth, DocumentFieldAlignment.Left, options);
            }

            return Finish(page, placed, look, font, width, height, edge, text: true);
        }
        finally
        {
            page.Release();
        }
    }

    /// <summary>A push button's appearance: its caption centred on its background.</summary>
    public static PdfAppearance Button(PdfFieldLook look, PdfFieldFont font, float width, float height, Func<IFontFace, PdfWriterFont> faces, DocumentFormOptions options)
    {
        var page = new PdfPageBuilder(faces, NoImages, 0, width, height);
        try
        {
            float inset = Inset(look, options);
            var placed = new List<Placed>();
            if (look.Caption is { Length: > 0 } caption)
            {
                Line(placed, caption, look with { Alignment = DocumentFieldAlignment.Centre }, font, width, height, inset, Math.Max(0f, width - 2 * inset),
                    Math.Max(0f, height - 2 * inset), options);
            }

            return Finish(page, placed, look, font, width, height, Inset(look, 0f), text: false);
        }
        finally
        {
            page.Release();
        }
    }

    /// <summary>A check box's or radio button's appearance: its mark (drawn as a path) when <paramref name="on"/>, its box alone when off.</summary>
    public static PdfAppearance Check(PdfFieldLook look, bool on, float width, float height, Func<IFontFace, PdfWriterFont> faces, DocumentFormOptions options)
    {
        var page = new PdfPageBuilder(faces, NoImages, 0, width, height);
        try
        {
            if (on)
            {
                float edge = Inset(look, 0f);
                float side = Math.Max(0f, Math.Min(width, height) - 2 * edge) * 0.7f;
                if (side > 0)
                {
                    page.Path(Filled(Mark(look.Mark, width / 2, height / 2, side), look.TextColour));
                }
            }

            return Finish(page, [], look, null, width, height, 0f, text: false);
        }
        finally
        {
            page.Release();
        }
    }

    /// <summary>A signature field's (or any field's) box alone: its background and border.</summary>
    public static PdfAppearance Box(PdfFieldLook look, float width, float height, Func<IFontFace, PdfWriterFont> faces, DocumentFormOptions options)
    {
        var page = new PdfPageBuilder(faces, NoImages, 0, width, height);
        try
        {
            return Finish(page, [], look, null, width, height, 0f, text: false);
        }
        finally
        {
            page.Release();
        }
    }

    // A widget's appearance has no images: a field draws text and paths alone.
    private static int NoImages(ImageData image, DocumentImageEncoding encoding) =>
        throw new InvalidOperationException("A form field's appearance holds no images.");

    // The distance from the box's edge to its text: the border (when drawn: a beveled or inset box's shading always, twice
    // the border's width) and the padding.
    private static float Inset(PdfFieldLook look, DocumentFormOptions options) => Inset(look, options.Padding);

    private static float Inset(PdfFieldLook look, float padding) =>
        (look.BorderStyle is DocumentFieldBorderStyle.Beveled or DocumentFieldBorderStyle.Inset ? 2 * look.BorderWidth : look.Border is not null ? look.BorderWidth : 0f) + padding;

    // The background and the border, inside the box, as content of the form's own space (none when an existing
    // appearance's are kept).
    private static void Frame(PdfSyntax content, PdfFieldLook look, float width, float height)
    {
        if (!look.Framed)
        {
            return;
        }

        if (look.Background is { } background)
        {
            content.Raw("q "u8);
            Colour(content, background);
            content.Raw("0 0 "u8).Number(width).Space().Number(height).Raw(" re f Q\n"u8);
        }

        float w = look.BorderWidth;
        if (look.BorderStyle is DocumentFieldBorderStyle.Beveled or DocumentFieldBorderStyle.Inset && w > 0)
        {
            Bevel(content, look, width, height);
            return;
        }

        if (look.Border is { } border && w > 0)
        {
            float half = w / 2;
            content.Raw("q "u8).Number(w).Raw(" w "u8);
            if (look.BorderStyle == DocumentFieldBorderStyle.Dashed)
            {
                content.Byte((byte)'[');
                if (look.Dashes.Count == 0)
                {
                    content.Byte((byte)'3');
                }

                for (int i = 0; i < look.Dashes.Count; i++)
                {
                    (i > 0 ? content.Space() : content).Number(look.Dashes[i]);
                }

                content.Raw("] 0 d "u8);
            }

            Colour(content, border, stroke: true);
            if (look.BorderStyle == DocumentFieldBorderStyle.Underline)
            {
                content.Raw("0 "u8).Number(half).Raw(" m "u8).Number(width).Space().Number(half).Raw(" l S Q\n"u8);   // the bottom edge alone
                return;
            }

            content.Number(half).Space().Number(half).Space().Number(Math.Max(0f, width - w)).Space().Number(Math.Max(0f, height - w))
                .Raw(" re S Q\n"u8);
        }
    }

    // A beveled (raised) or inset (sunken) box (ISO 32000-2, 12.5.4): the shading twice the border's width inside the box's
    // edge, lighter along the top and left and darker along the bottom and right for a raised one (white and half grey),
    // the other way for a sunken one (half grey and three-quarter grey); the border's colour around it when it has one.
    private static void Bevel(PdfSyntax content, PdfFieldLook look, float width, float height)
    {
        float half = look.BorderWidth, full = 2 * look.BorderWidth;
        bool raised = look.BorderStyle == DocumentFieldBorderStyle.Beveled;
        content.Raw("q "u8);
        Colour(content, DocumentColour.Grey(raised ? 1f : 0.5f));
        content.Number(half).Space().Number(half).Raw(" m "u8).Number(half).Space().Number(height - half).Raw(" l "u8).Number(width - half).Space().Number(height - half)
            .Raw(" l "u8).Number(width - full).Space().Number(height - full).Raw(" l "u8).Number(full).Space().Number(height - full).Raw(" l "u8).Number(full).Space()
            .Number(full).Raw(" l f\n"u8);
        Colour(content, DocumentColour.Grey(raised ? 0.5f : 0.75f));
        content.Number(width - half).Space().Number(height - half).Raw(" m "u8).Number(width - half).Space().Number(half).Raw(" l "u8).Number(half).Space().Number(half)
            .Raw(" l "u8).Number(full).Space().Number(full).Raw(" l "u8).Number(width - full).Space().Number(full).Raw(" l "u8).Number(width - full).Space()
            .Number(height - full).Raw(" l f\n"u8);
        if (look.Border is { } border)
        {
            Colour(content, border);
            content.Raw("0 0 "u8).Number(width).Space().Number(height).Raw(" re "u8).Number(half).Space().Number(half).Space().Number(Math.Max(0f, width - 2 * half)).Space()
                .Number(Math.Max(0f, height - 2 * half)).Raw(" re f*\n"u8);
        }

        content.Raw("Q\n"u8);
    }

    // A rectangle's contour, from its top-left corner (as DocumentPath.Rectangle).
    private static (OutlineVerb[] Verbs, float[] Points) Corners(DocumentBox box) =>
        ([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close],
            [box.X, box.Y, box.Right, box.Y, box.Right, box.Bottom, box.X, box.Bottom]);

    // A shape filled in a colour (the nonzero rule).
    private static DocumentPath Filled((OutlineVerb[] Verbs, float[] Points) shape, DocumentColour colour) =>
        new(shape.Verbs, shape.Points) { Fill = DocumentFill.NonZero, FillColour = colour };

    // One line: fitted to the box's height and width when its size is 0, centred on its capitals.
    private static void Line(List<Placed> placed, string text, PdfFieldLook look, PdfFieldFont font, float width, float height, float inset, float innerWidth,
        float innerHeight, DocumentFormOptions options)
    {
        float size = look.FontSize;
        if (size <= 0)
        {
            double extent = Math.Max(1e-3, font.Ascent - font.Descent);
            double tall = innerHeight / extent, wide = Measure(text, font, 1f, options) is > 0 and var w ? innerWidth / w : double.MaxValue;
            size = (float)Math.Max(options.MinFontSize, Math.Min(tall, wide));
        }

        float baseline = height - (float)(height - font.CapHeight * size) / 2;
        Place(placed, text, font, size, inset, baseline, innerWidth, look.Alignment, options);
    }

    // Several lines: wrapped to the box's inner width, from the top; a size of 0 shrinks from the largest until they fit.
    private static void Lines(List<Placed> placed, string text, PdfFieldLook look, PdfFieldFont font, float width, float inset, float innerWidth, float innerHeight,
        DocumentFormOptions options)
    {
        float size = look.FontSize;
        List<SetLine> lines;
        if (size > 0)
        {
            lines = Wrap(text, font, size, innerWidth, options);
        }
        else
        {
            size = Math.Max(options.MaxFontSize, options.MinFontSize);
            while (true)
            {
                lines = Wrap(text, font, size, innerWidth, options);
                double used = (lines.Count - 1) * size * options.LineSpacing + (font.Ascent - font.Descent) * size;
                if (used <= innerHeight || size - SizeStep < options.MinFontSize)
                {
                    break;
                }

                size -= SizeStep;
            }
        }

        float baseline = inset + (float)(font.Ascent * size);
        foreach (var line in lines)
        {
            placed.Add(Position(line, size, inset, baseline, innerWidth, look.Alignment));
            baseline += size * options.LineSpacing;
        }
    }

    // A comb: each character centred in its cell of the box's width split MaxLength ways.
    private static void Comb(List<Placed> placed, string text, PdfFieldLook look, PdfFieldFont font, float width, float height, float inset, DocumentFormOptions options)
    {
        int cells = look.MaxLength!.Value;
        float cell = width / cells;
        float size = look.FontSize;
        if (size <= 0)
        {
            double extent = Math.Max(1e-3, font.Ascent - font.Descent);
            size = (float)Math.Max(options.MinFontSize, Math.Min(Math.Max(0f, height - 2 * inset) / extent, cell * 0.9 / Math.Max(1e-3, font.Ascent - font.Descent)));
        }

        float baseline = height - (float)(height - font.CapHeight * size) / 2;
        var elements = StringInfo.GetTextElementEnumerator(text);
        for (int k = 0; k < cells && elements.MoveNext(); k++)
        {
            string element = elements.GetTextElement();
            var line = Set(element, font, size, options);
            placed.Add(Position(line, size, k * cell, baseline, cell, DocumentFieldAlignment.Centre));
        }
    }

    // A line placed by its alignment within [left, left + width).
    private static void Place(List<Placed> placed, string text, PdfFieldFont font, float size, float left, float baseline, float width, DocumentFieldAlignment alignment,
        DocumentFormOptions options) =>
        placed.Add(Position(Set(text, font, size, options), size, left, baseline, width, alignment));

    private static Placed Position(SetLine line, float size, float left, float baseline, float width, DocumentFieldAlignment alignment)
    {
        float x = alignment switch
        {
            DocumentFieldAlignment.Centre => left + (width - line.Width) / 2,
            DocumentFieldAlignment.Right => left + width - line.Width,
            _ => left,
        };
        return new Placed(line, size, x, baseline);
    }

    // The width of a text set on one line at `size`.
    private static double Measure(string text, PdfFieldFont font, float size, DocumentFormOptions options) =>
        font.IsSimple ? font.Width(text) * size : Set(text, font, size, options).Width;

    // A text set on one line: a simple font's codes and width, or the faces' runs through the text setter.
    private static SetLine Set(string text, PdfFieldFont font, float size, DocumentFormOptions options)
    {
        if (font.IsSimple)
        {
            return new SetLine(text, (float)(font.Width(text) * size), null);
        }

        var lines = SetterNamed(options.TextSetter).Set(text, new DocumentTextSetting(font.Faces!, size));
        return lines.Count == 0 ? new SetLine(text, 0f, []) : new SetLine(lines[0].Text, lines[0].Width, lines[0].Runs);
    }

    // The text setters were registered (DocumentTextSetters) and looked up by name: not on the new design yet.
    private static IDocumentTextSetter SetterNamed(string name) =>
        throw new NotSupportedException($"The text setter '{name}' by name is not on the new design yet.");

    // A text's lines wrapped to `width`: a simple font's words broken greedily at spaces (inside a word too long for a line);
    // faces' through the text setter.
    private static List<SetLine> Wrap(string text, PdfFieldFont font, float size, float width, DocumentFormOptions options)
    {
        var lines = new List<SetLine>();
        if (!font.IsSimple)
        {
            var setting = new DocumentTextSetting(font.Faces!, size) { Width = Math.Max(width, 1e-3f) };
            foreach (var line in SetterNamed(options.TextSetter).Set(text, setting))
            {
                lines.Add(new SetLine(line.Text, line.Width, line.Runs));
            }

            return lines;
        }

        foreach (string raw in text.Split('\n'))
        {
            string paragraph = raw.TrimEnd('\r');
            if (paragraph.Length == 0)
            {
                lines.Add(new SetLine("", 0f, null));
                continue;
            }

            int start = 0;
            while (start < paragraph.Length)
            {
                // As many characters as fit (at least one), then back to the last space among them when the line goes on.
                int end = start, lastSpace = -1;
                double used = 0;
                while (end < paragraph.Length)
                {
                    int length = char.IsSurrogatePair(paragraph, end) ? 2 : 1;
                    double advance = font.Width(paragraph.Substring(end, length)) * size;
                    if (paragraph[end] != ' ' && used + advance > width && end > start)
                    {
                        break;
                    }

                    used += advance;
                    end += length;
                    if (paragraph[end - 1] == ' ')
                    {
                        lastSpace = end;
                    }
                }

                if (end < paragraph.Length && lastSpace > start)
                {
                    end = lastSpace;
                }

                string piece = paragraph[start..end].TrimEnd(' ');
                lines.Add(new SetLine(piece, (float)(font.Width(piece) * size), null));
                start = end;
                while (start < paragraph.Length && paragraph[start] == ' ')
                {
                    start++;
                }
            }
        }

        return lines;
    }

    // The appearance's content: the page builder's (background, border, the faces' glyph runs under the inner clip), then
    // a simple font's text (a rich text's in each run's font and colour), in the form's own space, inside /Tx BMC ... EMC
    // for a text field; a rich run's underline or strike-out last.
    private static PdfAppearance Finish(PdfPageBuilder page, List<Placed> placed, PdfFieldLook look, PdfFieldFont? font, float width, float height, float edge, bool text)
    {
        var clip = new DocumentBox(edge, edge, Math.Max(0f, width - 2 * edge), Math.Max(0f, height - 2 * edge));
        bool anyFaces = false, anySimple = false, decorated = false;
        foreach (var p in placed)
        {
            var own = p.Font ?? font;
            anyFaces |= own is { IsSimple: false };
            anySimple |= own is { IsSimple: true };
            decorated |= p.Underline || p.StrikeOut;
        }

        if (anyFaces)
        {
            page.Save();
            page.Clip(DocumentPath.Rectangle(clip));
            foreach (var p in placed)
            {
                if ((p.Font ?? font) is not { IsSimple: false })
                {
                    continue;
                }

                foreach (var run in p.Line.Runs ?? [])
                {
                    var glyphs = run.Glyphs.ToArray();
                    for (int i = 0; i < glyphs.Length; i++)
                    {
                        glyphs[i] = glyphs[i] with { X = glyphs[i].X + p.X, Y = glyphs[i].Y + p.Baseline };
                    }

                    page.Glyphs(run.Face, run.Size, glyphs, run.Text, DocumentTextMode.Fill, p.Colour ?? look.TextColour);
                }
            }

            page.Restore();
        }

        using var content = new PdfSyntax(1024);
        Frame(content, look, width, height);
        if (text)
        {
            content.Raw("/Tx BMC\n"u8);                                        // the section a value's text goes in (12.7.4.3)
        }

        if (page.Close() is { } drawn)
        {
            content.Raw(drawn.Span);
        }

        var simple = new List<PdfFieldFont>(1);
        if (anySimple)
        {
            content.Raw("q "u8).Number(clip.X).Space().Number(clip.Y).Space().Number(clip.Width).Space().Number(clip.Height).Raw(" re W n\nBT\n"u8);
            var colour = look.TextColour;
            Colour(content, colour);
            float size = -1;
            PdfFieldFont? current = null;
            foreach (var p in placed)
            {
                if ((p.Font ?? font) is not { IsSimple: true } own)
                {
                    continue;
                }

                if ((p.Colour ?? look.TextColour) is var wanted && wanted != colour)
                {
                    Colour(content, wanted);
                    colour = wanted;
                }

                if (p.Size != size || !ReferenceEquals(own, current))
                {
                    content.Name(own.Resource!).Space().Number(p.Size, 3).Raw(" Tf\n"u8);
                    (size, current) = (p.Size, own);
                    if (!simple.Contains(own))
                    {
                        simple.Add(own);
                    }
                }

                content.Raw("1 0 0 1 "u8).Number(p.X, 3).Space().Number(height - p.Baseline, 3).Raw(" Tm "u8).HexString(own.Encode(p.Line.Text)).Raw(" Tj\n"u8);
            }

            content.Raw("ET\nQ\n"u8);
        }

        if (decorated)
        {
            // A rich run's lines: under the baseline, or through the middle of its lower case, a sixteenth of its size thick.
            content.Raw("q "u8).Number(clip.X).Space().Number(clip.Y).Space().Number(clip.Width).Space().Number(clip.Height).Raw(" re W n\n"u8);
            foreach (var p in placed)
            {
                float thick = Math.Max(0.5f, p.Size / 16);
                foreach (var (on, below) in new[] { (p.Underline, p.Size * 0.12f), (p.StrikeOut, -p.Size * 0.28f) })
                {
                    if (on && p.Line.Width > 0)
                    {
                        Colour(content, p.Colour ?? look.TextColour);
                        content.Number(p.X, 3).Space().Number(height - p.Baseline - below - thick / 2, 3).Space().Number(p.Line.Width, 3).Space().Number(thick, 3).Raw(" re f\n"u8);
                    }
                }
            }

            content.Raw("Q\n"u8);
        }

        if (text)
        {
            content.Raw("EMC\n"u8);
        }

        return new PdfAppearance(width, height, content.Span.ToArray(), [.. page.Fonts], [.. page.Alphas], simple);
    }

    // A fill (or stroke) colour in its space (g, rg or k; G, RG or K).
    private static void Colour(PdfSyntax content, DocumentColour colour, bool stroke = false)
    {
        for (int c = 0; c < colour.Count; c++)
        {
            content.Number(colour[c]).Space();
        }

        content.Raw(colour.Space switch
        {
            DocumentColourSpace.Grey => stroke ? "G\n"u8 : "g\n"u8,
            DocumentColourSpace.Rgb => stroke ? "RG\n"u8 : "rg\n"u8,
            _ => stroke ? "K\n"u8 : "k\n"u8,
        });
    }

    // A mark of `side` points centred on (cx, cy), in the page builder's frame (y down).
    private static (OutlineVerb[] Verbs, float[] Points) Mark(DocumentFieldMark mark, float cx, float cy, float side)
    {
        float r = side / 2;
        (float X, float Y) P(double ux, double uy) => ((float)(cx - r + ux * side), (float)(cy - r + uy * side));
        switch (mark)
        {
            case DocumentFieldMark.Circle:
            {
                const float k = 0.5523f;                                         // a quarter circle's control distance
                float c = r * k;
                return (
                    [OutlineVerb.Move, OutlineVerb.Cubic, OutlineVerb.Cubic, OutlineVerb.Cubic, OutlineVerb.Cubic, OutlineVerb.Close],
                    [cx + r, cy, cx + r, cy + c, cx + c, cy + r, cx, cy + r, cx - c, cy + r, cx - r, cy + c, cx - r, cy, cx - r, cy - c, cx - c, cy - r, cx, cy - r,
                        cx + c, cy - r, cx + r, cy - c, cx + r, cy]);
            }

            case DocumentFieldMark.Square:
                return Corners(new DocumentBox(cx - r * 0.8f, cy - r * 0.8f, side * 0.8f, side * 0.8f));
            case DocumentFieldMark.Diamond:
                return Polygon([P(0.5, 0), P(1, 0.5), P(0.5, 1), P(0, 0.5)]);
            case DocumentFieldMark.Cross:
            {
                const double t = 0.12;                                           // half the bars' width, in the mark's unit
                var verbs = new[] { OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close, OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line,
                    OutlineVerb.Line, OutlineVerb.Close };
                var a = new[] { P(t, 0), P(1, 1 - t), P(1 - t, 1), P(0, t) };
                var b = new[] { P(1 - t, 0), P(1, t), P(t, 1), P(0, 1 - t) };
                return (verbs, [.. a.Concat(b).SelectMany(p => new[] { p.X, p.Y })]);
            }

            case DocumentFieldMark.Star:
            {
                var points = new (float X, float Y)[10];
                for (int i = 0; i < 10; i++)
                {
                    double angle = -Math.PI / 2 + i * Math.PI / 5, radius = i % 2 == 0 ? 0.5 : 0.19;
                    points[i] = P(0.5 + radius * Math.Cos(angle), 0.5 + radius * Math.Sin(angle));
                }

                return Polygon(points);
            }

            default:
                return Polygon([P(0.05, 0.55), P(0.2, 0.4), P(0.4, 0.6), P(0.8, 0.12), P(0.95, 0.27), P(0.4, 0.88)]);
        }
    }

    private static (OutlineVerb[] Verbs, float[] Points) Polygon((float X, float Y)[] points)
    {
        var verbs = new OutlineVerb[points.Length + 1];
        verbs[0] = OutlineVerb.Move;
        for (int i = 1; i < points.Length; i++)
        {
            verbs[i] = OutlineVerb.Line;
        }

        verbs[^1] = OutlineVerb.Close;
        return (verbs, [.. points.SelectMany(p => new[] { p.X, p.Y })]);
    }

    // Text on one line: line ends as spaces.
    private static string OneLine(string text) => text.Replace("\r\n", " ", StringComparison.Ordinal).Replace('\n', ' ').Replace('\r', ' ');

    // A rich text's lines: its runs' words (each with the spaces after it) laid in turn, a line broken before a word that
    // would pass `width` (a word wider than a line kept whole) and at each line feed; a run's words on a line set together.
    private static List<List<Segment>> RichLines(List<Piece> pieces, float size, float width, DocumentFormOptions options)
    {
        var lines = new List<List<(Piece Piece, StringBuilder Text, float Size)>> { new() };
        double used = 0;
        foreach (var piece in pieces)
        {
            float own = piece.Run.Size ?? size;
            string text = piece.Text;
            int i = 0;
            while (i < text.Length)
            {
                if (text[i] is '\n' or '\r')
                {
                    i += text[i] == '\r' && i + 1 < text.Length && text[i + 1] == '\n' ? 2 : 1;
                    lines.Add([]);
                    used = 0;
                    continue;
                }

                int j = i;
                while (j < text.Length && text[j] is not (' ' or '\n' or '\r'))
                {
                    j++;
                }

                int word = j;
                while (j < text.Length && text[j] == ' ')
                {
                    j++;
                }

                double shown = Measure(text[i..word], piece.Font, own, options), advance = word == j ? shown : Measure(text[i..j], piece.Font, own, options);
                if (used + shown > width && lines[^1].Count > 0)
                {
                    lines.Add([]);
                    used = 0;
                }

                var line = lines[^1];
                if (line.Count > 0 && ReferenceEquals(line[^1].Piece, piece))
                {
                    line[^1].Text.Append(text, i, j - i);
                }
                else
                {
                    line.Add((piece, new StringBuilder(text, i, j - i, j - i), own));
                }

                used += advance;
                i = j;
            }
        }

        var set = new List<List<Segment>>(lines.Count);
        foreach (var line in lines)
        {
            var segments = new List<Segment>(line.Count);
            for (int k = 0; k < line.Count; k++)
            {
                string text = line[k].Text.ToString();
                if (k == line.Count - 1)
                {
                    text = text.TrimEnd(' ');                                  // a line's last spaces neither shown nor aligned
                }

                segments.Add(new Segment(line[k].Piece, line[k].Size, Set(text, line[k].Piece.Font, line[k].Size, options)));
            }

            set.Add(segments);
        }

        return set;
    }

    // The height rich lines take: their first line's extent and the spacing to each next one.
    private static double RichHeight(List<List<Segment>> lines, DocumentFormOptions options)
    {
        double height = lines.Count > 0 ? Max(lines[0], s => (s.Piece.Font.Ascent - s.Piece.Font.Descent) * s.Size) : 0;
        for (int k = 1; k < lines.Count; k++)
        {
            height += Max(lines[k], s => s.Size) * options.LineSpacing;
        }

        return height;
    }

    // The most a line's segments give (0 for an empty line).
    private static double Max(List<Segment> line, Func<Segment, double> of)
    {
        double most = 0;
        foreach (var segment in line)
        {
            most = Math.Max(most, of(segment));
        }

        return most;
    }

    // A line as set: its text, width and, for faces, its runs (null for a simple font's codes).
    private sealed record SetLine(string Text, float Width, IReadOnlyList<DocumentSetRun>? Runs);

    // A line placed: its pen's start and baseline in the box (points from its top-left corner, y down); a rich run's own
    // font, colour and decoration (null: the field's font and colour).
    private sealed record Placed(SetLine Line, float Size, float X, float Baseline)
    {
        public PdfFieldFont? Font { get; init; }

        public DocumentColour? Colour { get; init; }

        public bool Underline { get; init; }

        public bool StrikeOut { get; init; }
    }

    // A rich text's run with the text it shows (one line's: line ends as spaces) and the font chosen for it.
    private sealed record Piece(DocumentFieldRun Run, string Text, PdfFieldFont Font);

    // A part of a rich line: a run's words on it, at the run's size, set.
    private sealed record Segment(Piece Piece, float Size, SetLine Set);
}
