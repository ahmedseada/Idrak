// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using System.Xml;

namespace Idrak.Data.Pdf;

/// <summary>
/// A text field's rich text (ISO 32000-2, 12.7.4.3, "Rich text strings"): its <c>/RV</c> read as runs and runs written as
/// one. The value is XHTML of <c>body</c>, <c>p</c>, <c>span</c>, <c>b</c>, <c>i</c> and <c>br</c> with the CSS subset the
/// standard allows in their <c>style</c>: <c>font</c>, <c>font-family</c>, <c>font-size</c>, <c>font-style</c>,
/// <c>font-weight</c>, <c>color</c> and <c>text-decoration</c> (alignment and stretch are the field's: not read), over the
/// field's default style string (<c>/DS</c>). It is read as XML with no DTD and nothing outside fetched; a value that is not
/// well-formed gives no runs (the plain <c>/V</c> stays the field's value). Each paragraph after the first starts with a line
/// feed.
/// </summary>
internal static class PdfRichText
{
    // The XHTML a writer gives /RV: its namespaces and the version of the rich text format it follows.
    private const string Head = "<?xml version=\"1.0\"?><body xmlns=\"http://www.w3.org/1999/xhtml\" xmlns:xfa=\"http://www.xfa.org/schema/xfa-data/1.0/\" xfa:APIVersion=\"Idrak\" xfa:spec=\"2.0.2\"><p>";

    /// <summary>The runs of a field's rich text value (a text string or a stream) over its default style (<c>/DS</c>); none when it cannot be read.</summary>
    public static IReadOnlyList<DocumentFieldRun> Read(PdfFile file, PdfValue value, PdfValue style)
    {
        string? xml = TextOf(file, value);
        if (string.IsNullOrEmpty(xml))
        {
            return [];
        }

        var initial = Style.Parse(TextOf(file, style), default);
        try
        {
            return Parse(xml, initial, file.MaxStreamBytes);
        }
        catch (XmlException)
        {
            return [];
        }
    }

    /// <summary>The plain text of <paramref name="runs"/> (their characters, in order).</summary>
    public static string Plain(IReadOnlyList<DocumentFieldRun> runs)
    {
        if (runs.Count == 1)
        {
            return runs[0].Text;
        }

        var text = new StringBuilder();
        foreach (var run in runs)
        {
            text.Append(run.Text);
        }

        return text.ToString();
    }

    /// <summary>
    /// <paramref name="runs"/> as a rich text value: one <c>p</c> a paragraph (a line feed starts the next), each run a
    /// <c>span</c> with the style it sets (a run that sets none as plain text).
    /// </summary>
    public static string Write(IReadOnlyList<DocumentFieldRun> runs)
    {
        var xml = new StringBuilder(Head);
        foreach (var run in runs)
        {
            string css = Css(run);
            int start = 0;
            while (start <= run.Text.Length)
            {
                int end = run.Text.IndexOf('\n', start);
                int stop = end < 0 ? run.Text.Length : end;
                if (stop > start)
                {
                    if (css.Length > 0)
                    {
                        xml.Append("<span style=\"").Append(css).Append("\">");
                    }

                    Escape(xml, run.Text.AsSpan(start, stop - start));
                    if (css.Length > 0)
                    {
                        xml.Append("</span>");
                    }
                }

                if (end < 0)
                {
                    break;
                }

                xml.Append("</p><p>");
                start = end + 1;
            }
        }

        return xml.Append("</p></body>").ToString();
    }

    // A text string's text, or a stream's (its bytes as UTF-8, else Latin-1); null for anything else.
    private static string? TextOf(PdfFile file, PdfValue value)
    {
        var resolved = PdfForms.SafeResolve(file, value);
        if (resolved.AsString is { } s)
        {
            return PdfText.Decode(s.Bytes);
        }

        if (resolved.AsStream is { } stream)
        {
            using var decoded = file.Decode(stream);
            return PdfText.Utf8OrLatin1(decoded.Span);
        }

        return null;
    }

    // The runs of an XHTML body: text in the style of the elements around it, a run for each change of style.
    private static List<DocumentFieldRun> Parse(string xml, Style initial, long maxCharacters)
    {
        if (xml.StartsWith("<?xml", StringComparison.Ordinal) && xml.IndexOf("?>", StringComparison.Ordinal) is >= 0 and var declared)
        {
            xml = xml[(declared + 2)..];                                       // its declaration says how bytes were encoded: these are characters
        }

        var settings = new XmlReaderSettings
        {
            DtdProcessing = DtdProcessing.Prohibit, XmlResolver = null, IgnoreComments = true, IgnoreProcessingInstructions = true,
            ConformanceLevel = ConformanceLevel.Fragment, MaxCharactersInDocument = Math.Max(1, maxCharacters), MaxCharactersFromEntities = 1024,
        };
        using var reader = XmlReader.Create(new StringReader(xml), settings);
        var runs = new List<DocumentFieldRun>();
        var styles = new Stack<Style>();
        var style = initial;
        int paragraphs = 0, inParagraph = 0;
        var text = new StringBuilder();
        Style? textStyle = null;

        // Text in a style: joined to the run before when its style is the same.
        void Emit(string chars, Style with)
        {
            if (chars.Length == 0)
            {
                return;
            }

            if (textStyle is { } current && !current.Equals(with))
            {
                runs.Add(current.Run(text.ToString()));
                text.Clear();
            }

            textStyle = with;
            text.Append(chars);
        }

        while (reader.Read())
        {
            switch (reader.NodeType)
            {
                case XmlNodeType.Element:
                {
                    string name = reader.LocalName;
                    var inner = style;
                    if (name == "b")
                    {
                        inner.Bold = true;
                    }
                    else if (name == "i")
                    {
                        inner.Italic = true;
                    }

                    if (reader.GetAttribute("style") is { Length: > 0 } css)
                    {
                        inner = Style.Parse(css, inner);
                    }

                    if (name == "p")
                    {
                        if (paragraphs++ > 0)
                        {
                            Emit("\n", inner);
                        }

                        inParagraph++;
                    }
                    else if (name == "br")
                    {
                        Emit("\n", inner);
                    }

                    if (!reader.IsEmptyElement)
                    {
                        styles.Push(style);
                        style = inner;
                    }
                    else if (name == "p")
                    {
                        inParagraph--;
                    }

                    break;
                }

                case XmlNodeType.EndElement:
                    if (reader.LocalName == "p")
                    {
                        inParagraph--;
                    }

                    if (styles.Count > 0)
                    {
                        style = styles.Pop();
                    }

                    break;
                case XmlNodeType.Text or XmlNodeType.CDATA or XmlNodeType.SignificantWhitespace:
                    Emit(reader.Value, style);
                    break;
                case XmlNodeType.Whitespace:
                    if (inParagraph > 0)
                    {
                        Emit(reader.Value, style);                             // between spans of a paragraph: a space it shows
                    }

                    break;
            }
        }

        if (textStyle is { } last)
        {
            runs.Add(last.Run(text.ToString()));
        }

        return runs;
    }

    // A run's style as CSS declarations (only what it sets).
    private static string Css(DocumentFieldRun run)
    {
        var css = new StringBuilder();
        if (run.Family is { Length: > 0 } family)
        {
            css.Append("font-family:'").Append(family.Replace("'", "", StringComparison.Ordinal).Replace("\"", "", StringComparison.Ordinal)).Append("';");
        }

        if (run.Size is { } size)
        {
            css.Append(CultureInfo.InvariantCulture, $"font-size:{size:0.###}pt;");
        }

        if (run.Bold)
        {
            css.Append("font-weight:bold;");
        }

        if (run.Italic)
        {
            css.Append("font-style:italic;");
        }

        if (run.Colour is { } colour)
        {
            var (r, g, b) = Rgb(colour);
            css.Append(CultureInfo.InvariantCulture, $"color:#{r:x2}{g:x2}{b:x2};");
        }

        if (run.Underline || run.StrikeOut)
        {
            css.Append("text-decoration:").Append(run.Underline ? "underline" : "").Append(run.Underline && run.StrikeOut ? " " : "").Append(run.StrikeOut ? "line-through" : "").Append(';');
        }

        if (css.Length > 0)
        {
            css.Length--;                                                      // the last ';'
        }

        return css.ToString();
    }

    // A colour as 8-bit RGB (grey and CMYK converted plainly).
    private static (int R, int G, int B) Rgb(DocumentColour colour)
    {
        static int Level(double v) => (int)Math.Round(Math.Clamp(v, 0, 1) * 255);
        return colour.Space switch
        {
            DocumentColourSpace.Grey => (Level(colour[0]), Level(colour[0]), Level(colour[0])),
            DocumentColourSpace.Rgb => (Level(colour[0]), Level(colour[1]), Level(colour[2])),
            _ => (Level((1 - colour[0]) * (1 - colour[3])), Level((1 - colour[1]) * (1 - colour[3])), Level((1 - colour[2]) * (1 - colour[3]))),
        };
    }

    private static void Escape(StringBuilder xml, ReadOnlySpan<char> text)
    {
        foreach (char c in text)
        {
            _ = c switch
            {
                '&' => xml.Append("&amp;"),
                '<' => xml.Append("&lt;"),
                '>' => xml.Append("&gt;"),
                '"' => xml.Append("&quot;"),
                '\r' => xml.Append("&#13;"),
                _ => xml.Append(c),
            };
        }
    }

    // A style in effect: what the elements around a text set of the CSS subset.
    private struct Style : IEquatable<Style>
    {
        public bool Bold, Italic, Underline, StrikeOut;
        public float? Size;
        public DocumentColour? Colour;
        public string? Family;

        public readonly DocumentFieldRun Run(string text) =>
            new(text) { Bold = Bold, Italic = Italic, Underline = Underline, StrikeOut = StrikeOut, Size = Size, Colour = Colour, Family = Family };

        public readonly bool Equals(Style other) =>
            Bold == other.Bold && Italic == other.Italic && Underline == other.Underline && StrikeOut == other.StrikeOut && Size == other.Size
            && Nullable.Equals(Colour, other.Colour) && string.Equals(Family, other.Family, StringComparison.Ordinal);

        public override readonly bool Equals(object? obj) => obj is Style other && Equals(other);

        public override readonly int GetHashCode() => HashCode.Combine(Bold, Italic, Underline, StrikeOut, Size, Colour, Family);

        // `css`'s declarations over `style` (those not of the subset, or not understood, left out).
        public static Style Parse(string? css, Style style)
        {
            if (string.IsNullOrWhiteSpace(css))
            {
                return style;
            }

            foreach (string declaration in css.Split(';'))
            {
                int colon = declaration.IndexOf(':');
                if (colon <= 0)
                {
                    continue;
                }

                string property = declaration[..colon].Trim().ToLowerInvariant(), value = declaration[(colon + 1)..].Trim();
                switch (property)
                {
                    case "font-size":
                        style.Size = SizeOf(value, style.Size) ?? style.Size;
                        break;
                    case "font-weight":
                        style.Bold = WeightOf(value) ?? style.Bold;
                        break;
                    case "font-style":
                        style.Italic = SlantOf(value) ?? style.Italic;
                        break;
                    case "font-family":
                        style.Family = FamilyOf(value) ?? style.Family;
                        break;
                    case "color":
                        style.Colour = ColourOf(value) ?? style.Colour;
                        break;
                    case "text-decoration":
                        string decoration = value.ToLowerInvariant();
                        (style.Underline, style.StrikeOut) = decoration.Contains("none", StringComparison.Ordinal)
                            ? (false, false)
                            : (decoration.Contains("underline", StringComparison.Ordinal), decoration.Contains("line-through", StringComparison.Ordinal));
                        break;
                    case "font":
                        style = Shorthand(value, style);
                        break;
                }
            }

            return style;
        }

        // `font: [style] [weight] size[/line-height] family, ...`.
        private static Style Shorthand(string value, Style style)
        {
            var words = value.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            for (int i = 0; i < words.Length; i++)
            {
                string word = words[i];
                if (SlantOf(word) is { } italic && SizeOf(word, null) is null)
                {
                    style.Italic = italic;
                    continue;
                }

                if (WeightOf(word) is { } bold && SizeOf(word, null) is null)
                {
                    style.Bold = bold;
                    continue;
                }

                int slash = word.IndexOf('/');
                if (SizeOf(slash > 0 ? word[..slash] : word, style.Size) is { } size)
                {
                    style.Size = size;
                    if (i + 1 < words.Length && FamilyOf(string.Join(' ', words[(i + 1)..])) is { } family)
                    {
                        style.Family = family;
                    }

                    break;
                }
            }

            return style;
        }

        // A size in points: pt, px (one point a pixel, as viewers read it), or a bare number; em and % of the size in effect.
        private static float? SizeOf(string value, float? current)
        {
            var v = value.AsSpan().Trim();
            float scale = 1;
            bool relative = false;
            if (v.EndsWith("pt", StringComparison.OrdinalIgnoreCase) || v.EndsWith("px", StringComparison.OrdinalIgnoreCase))
            {
                v = v[..^2];
            }
            else if (v.EndsWith("em", StringComparison.OrdinalIgnoreCase))
            {
                v = v[..^2];
                relative = true;
            }
            else if (v.Length > 0 && v[^1] == '%')
            {
                v = v[..^1];
                relative = true;
                scale = 0.01f;
            }

            if (!float.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out float n) || !(n > 0) || !float.IsFinite(n) || n > 10000)
            {
                return null;
            }

            return relative ? current is { } c ? c * n * scale : null : n;
        }

        private static bool? WeightOf(string value) => value.ToLowerInvariant() switch
        {
            "bold" or "bolder" => true,
            "normal" or "lighter" => false,
            var number when int.TryParse(number, NumberStyles.None, CultureInfo.InvariantCulture, out int w) => w >= 600,
            _ => null,
        };

        private static bool? SlantOf(string value) => value.ToLowerInvariant() switch
        {
            "italic" or "oblique" => true,
            "normal" => false,
            _ => null,
        };

        // The first family of a list, its quotes taken off.
        private static string? FamilyOf(string value)
        {
            string first = value.Split(',')[0].Trim().Trim('\'', '"').Trim();
            return first.Length > 0 ? first : null;
        }

        // #rgb, #rrggbb, rgb(r, g, b) (0-255 or percentages), or one of the sixteen basic colour keywords.
        private static DocumentColour? ColourOf(string value)
        {
            string v = value.Trim().ToLowerInvariant();
            if (v.StartsWith('#'))
            {
                var hex = v.AsSpan(1);
                if (hex.Length == 3)
                {
                    Span<char> wide = [hex[0], hex[0], hex[1], hex[1], hex[2], hex[2]];
                    return HexOf(wide);
                }

                return hex.Length == 6 ? HexOf(hex) : null;
            }

            if (v.StartsWith("rgb(", StringComparison.Ordinal) && v.EndsWith(')'))
            {
                var parts = v[4..^1].Split(',', StringSplitOptions.TrimEntries);
                if (parts.Length != 3)
                {
                    return null;
                }

                Span<float> c = stackalloc float[3];
                for (int i = 0; i < 3; i++)
                {
                    bool percent = parts[i].EndsWith('%');
                    if (!float.TryParse(percent ? parts[i][..^1] : parts[i], NumberStyles.Float, CultureInfo.InvariantCulture, out float n) || !float.IsFinite(n))
                    {
                        return null;
                    }

                    c[i] = Math.Clamp(percent ? n / 100f : n / 255f, 0f, 1f);
                }

                return DocumentColour.Rgb(c[0], c[1], c[2]);
            }

            return v switch
            {
                "black" => DocumentColour.Rgb(0, 0, 0),
                "white" => DocumentColour.Rgb(1, 1, 1),
                "red" => DocumentColour.Rgb(1, 0, 0),
                "lime" => DocumentColour.Rgb(0, 1, 0),
                "blue" => DocumentColour.Rgb(0, 0, 1),
                "yellow" => DocumentColour.Rgb(1, 1, 0),
                "aqua" => DocumentColour.Rgb(0, 1, 1),
                "fuchsia" => DocumentColour.Rgb(1, 0, 1),
                "silver" => DocumentColour.Rgb(0.75f, 0.75f, 0.75f),
                "gray" or "grey" => DocumentColour.Rgb(128 / 255f, 128 / 255f, 128 / 255f),
                "maroon" => DocumentColour.Rgb(128 / 255f, 0, 0),
                "olive" => DocumentColour.Rgb(128 / 255f, 128 / 255f, 0),
                "green" => DocumentColour.Rgb(0, 128 / 255f, 0),
                "purple" => DocumentColour.Rgb(128 / 255f, 0, 128 / 255f),
                "teal" => DocumentColour.Rgb(0, 128 / 255f, 128 / 255f),
                "navy" => DocumentColour.Rgb(0, 0, 128 / 255f),
                _ => null,
            };
        }

        private static DocumentColour? HexOf(ReadOnlySpan<char> hex)
        {
            if (!int.TryParse(hex, NumberStyles.HexNumber, CultureInfo.InvariantCulture, out int rgb))
            {
                return null;
            }

            return DocumentColour.Rgb(((rgb >> 16) & 0xFF) / 255f, ((rgb >> 8) & 0xFF) / 255f, (rgb & 0xFF) / 255f);
        }
    }
}

/// <summary>
/// The font a run of rich text is shown in: the field's own when the run asks nothing of it (no family, not bold, not
/// italic); else the standard font of its family in its style (Helvetica, Times and Courier, by their names or the CSS
/// generic families) when that has a code for every character; else faces of that family and style that show it
/// (<see cref="PdfFieldFonts"/>), else any that do; else the field's own. Nothing here names a script.
/// </summary>
internal static class PdfRichFonts
{
    /// <exception cref="InvalidOperationException">No font shows the run's text (the message names the options).</exception>
    public static PdfFieldFont Choose(DocumentFieldRun run, string text, PdfFieldFont own, string? family, PdfFieldFonts fonts, Func<string, int, PdfFieldFont> standard,
        string field, DocumentFormOptions options)
    {
        if (run.Family is null && !run.Bold && !run.Italic && (!own.IsSimple || own.Encodes(text)))
        {
            return own;
        }

        if (Standard(family, run.Bold, run.Italic) is >= 0 and var index)
        {
            string resource = Resource(index);
            if (own is { IsSimple: true, Resource: { } named } && named.AsSpan().SequenceEqual(Encoding.ASCII.GetBytes(resource)) && own.Encodes(text))
            {
                return own;                                                    // the field's own font goes by that name: one font a name
            }

            if (PdfFieldFont.Standard(Encoding.ASCII.GetBytes(resource), index, 0).Encodes(text))
            {
                return standard(resource, index);                              // its object only when used
            }
        }

        string wanted = (family ?? "") + (run.Bold ? "Bold" : "") + (run.Italic ? "Italic" : "");
        if (fonts.Faces(text, wanted.Length > 0 ? wanted : null, null) is { } faces)
        {
            return PdfFieldFont.OfFaces(faces);
        }

        if (!own.IsSimple || own.Encodes(text))
        {
            return own;
        }

        throw new InvalidOperationException(string.Create(CultureInfo.InvariantCulture,
            $"Field '{field}': no font shows the rich text \"{text}\" (U+{PdfFieldFonts.FirstMissing(text, c => own.Encodes(char.ConvertFromUtf32(c))):X4} first): neither its font, a standard font of '{family}', nor a face of DocumentWriteOptions.Forms.Faces or of the fonts folder (DocumentWriteOptions.Forms.Fonts, {DocumentDrawOptions.FontsVariable}{(options.Fonts is null ? ": not set" : $": {options.Fonts}")}) has its glyphs."));
    }

    /// <summary>The standard font of <paramref name="family"/> in a style (its index in the standard 14), or -1 when the family is none of theirs.</summary>
    public static int Standard(string? family, bool bold, bool italic)
    {
        if (string.IsNullOrEmpty(family))
        {
            return -1;
        }

        var plain = new StringBuilder(family.Length);
        foreach (char c in family.Length > 7 && family[6] == '+' ? family[7..] : family)
        {
            if (char.IsLetter(c))
            {
                plain.Append(char.ToLowerInvariant(c));
            }
        }

        string name = plain.ToString();
        int style = (bold ? 1 : 0) + (italic ? 2 : 0);
        string[]? variants = name.StartsWith("helv", StringComparison.Ordinal) || name.StartsWith("arial", StringComparison.Ordinal) || name == "sansserif"
            ? ["Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique"]
            : name.StartsWith("times", StringComparison.Ordinal) || name == "serif"
                ? ["Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic"]
                : name.StartsWith("cour", StringComparison.Ordinal) || name == "monospace"
                    ? ["Courier", "Courier-Bold", "Courier-Oblique", "Courier-BoldOblique"]
                    : null;
        return variants is null ? -1 : PdfFontTables.StandardFontIndex(variants[style]);
    }

    /// <summary>The family a field's own font is of: a simple font's resource name ("Helv"), or its first face's family.</summary>
    public static string? FamilyOf(PdfFieldFont font) => font.IsSimple ? Encoding.ASCII.GetString(font.Resource!) : font.Faces is [var face, ..] ? face.Family : null;

    /// <summary>The resource name a standard font is given in a form ("Helv" for Helvetica, else its name without dashes).</summary>
    public static string Resource(int standard)
    {
        string name = PdfFontTables.StandardFonts[standard];
        return name == "Helvetica" ? "Helv" : name.Replace("-", "", StringComparison.Ordinal);
    }
}
