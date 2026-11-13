// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Collections.Frozen;
using System.Text;

namespace Idrak.Imaging;

/// <summary>
/// "opentype": applies a <see cref="FontMap"/> to a face as OpenType data: the map's characters answer through the
/// face's own glyphs (found by the font's code points, glyph ids or post table names), the letters' positional forms and
/// the ligatures become GSUB single and ligature substitutions under their features, the marks get no width, GDEF's mark
/// class and GPOS mark-to-base and mark-to-mark anchors measured from the font's ink (the mark's ink centre over the
/// base's ink top or bottom around its middle, with the map's gap), and what the font lacks of the library's drawn glyphs
/// (Latin digits, combining marks) is drawn in at the font's measured size. The shapers then read the face as any font.
/// Everything is resolved once per face; the tables and drawn glyphs are made on first use.
/// </summary>
public sealed class OpenTypeFontMapper : IFontMapper
{
    /// <inheritdoc/>
    public string Name => FontMappers.OpenType;

    /// <inheritdoc/>
    public string Summary => "a map's forms, ligatures and marks as OpenType GSUB, GDEF and GPOS tables, mark anchors measured from the font's ink, missing digits and marks drawn in";

    /// <inheritdoc/>
    public IFontFace? Map(IFontFace face, FontMap map, bool detect, FontMapDrawing drawing)
    {
        ArgumentNullException.ThrowIfNull(face);
        ArgumentNullException.ThrowIfNull(map);
        ArgumentNullException.ThrowIfNull(drawing);
        if (detect)
        {
            if (face.GlyphCount < map.DetectMinGlyphs)
            {
                return null;
            }

            foreach (string table in map.DetectWithoutTables)
            {
                if (!face.Table(table).IsEmpty)
                {
                    return null;
                }
            }

            int own = 0;
            foreach (int letter in map.Letters.Keys)
            {
                own += face.Glyph(letter) != 0 ? 1 : 0;
            }

            if (2 * own >= map.Letters.Count)
            {
                return null;                                                    // the font answers the letters itself
            }
        }

        var resolver = new GlyphResolver(face, map);
        var forms = new Dictionary<int, int[]>(map.Letters.Count);
        int found = 0;
        foreach (var (code, letter) in map.Letters)
        {
            var glyphs = new int[FontMap.Forms.Count];
            for (int f = 0; f < glyphs.Length; f++)
            {
                glyphs[f] = resolver.Form(letter, FontMap.Forms[f]);
            }

            if (glyphs[0] != 0)
            {
                forms[code] = glyphs;
                found++;
            }
        }

        if (found == 0 || (detect && found < map.DetectShare * map.Letters.Count))
        {
            return null;
        }

        return new MappedFontFace(face, map, resolver, forms, drawing);
    }
}

/// <summary>Finds a map's glyph references in a face: by code point, glyph id or post table name, with the map's fallbacks.</summary>
internal sealed class GlyphResolver(IFontFace face, FontMap map)
{
    private const int MaxFallbackDepth = 4;
    private Dictionary<string, int>? _names;

    /// <summary>The glyph of the first reference found, passing over one with no ink when a later one has ink; 0 for none.</summary>
    public int Resolve(IReadOnlyList<FontGlyphRef> refs) => Resolve(refs, 0);

    /// <summary>A letter's glyph in <paramref name="form"/>: given outright, else its names after the form's prefixes; 0 for none.</summary>
    public int Form(FontMapLetter letter, string form)
    {
        if (letter.Forms.TryGetValue(form, out var given))
        {
            return Resolve(given);
        }

        if (letter.Names.Count == 0 || !map.Prefixes.TryGetValue(form, out var prefixes))
        {
            return 0;
        }

        var candidates = new List<FontGlyphRef>(letter.Names.Count * prefixes.Count);
        foreach (string name in letter.Names)
        {
            foreach (string prefix in prefixes)
            {
                candidates.Add(new FontGlyphRef(FontGlyphRefKind.Name, 0, prefix + name));
            }
        }

        return Resolve(candidates);
    }

    private int Resolve(IReadOnlyList<FontGlyphRef> refs, int depth)
    {
        int first = 0;
        foreach (var r in refs)
        {
            int glyph = One(r, depth);
            if (glyph == 0)
            {
                continue;
            }

            if (!face.Outline(glyph).IsEmpty)
            {
                return glyph;
            }

            first = first == 0 ? glyph : first;
        }

        return first;
    }

    private int One(FontGlyphRef r, int depth)
    {
        int glyph = r.Kind switch
        {
            FontGlyphRefKind.Code => face.Glyph(r.Value),
            FontGlyphRefKind.Id => (uint)r.Value < (uint)face.GlyphCount ? r.Value : 0,
            _ => (_names ??= PostNames.Read(face.Table("post").Span, face.GlyphCount)).GetValueOrDefault(r.Text),
        };
        if (glyph == 0 && depth < MaxFallbackDepth && map.Fallbacks.TryGetValue(r.Text, out var fallback))
        {
            glyph = Resolve(fallback, depth + 1);
        }

        return glyph;
    }
}

/// <summary>Glyph names from a font's post table (format 2.0: the standard Macintosh names and the font's own; format 1.0: the standard order).</summary>
internal static class PostNames
{
    // The 258 standard Macintosh glyph names (the OpenType post table's list).
    private static readonly string[] Standard =
        (".notdef .null nonmarkingreturn space exclam quotedbl numbersign dollar percent ampersand quotesingle parenleft parenright asterisk plus comma "
         + "hyphen period slash zero one two three four five six seven eight nine colon semicolon less equal greater question at A B C D E F G H I J K L M "
         + "N O P Q R S T U V W X Y Z bracketleft backslash bracketright asciicircum underscore grave a b c d e f g h i j k l m n o p q r s t u v w x y z "
         + "braceleft bar braceright asciitilde Adieresis Aring Ccedilla Eacute Ntilde Odieresis Udieresis aacute agrave acircumflex adieresis atilde "
         + "aring ccedilla eacute egrave ecircumflex edieresis iacute igrave icircumflex idieresis ntilde oacute ograve ocircumflex odieresis otilde "
         + "uacute ugrave ucircumflex udieresis dagger degree cent sterling section bullet paragraph germandbls registered copyright trademark acute "
         + "dieresis notequal AE Oslash infinity plusminus lessequal greaterequal yen mu partialdiff summation product pi integral ordfeminine "
         + "ordmasculine Omega ae oslash questiondown exclamdown logicalnot radical florin approxequal Delta guillemotleft guillemotright ellipsis "
         + "nonbreakingspace Agrave Atilde Otilde OE oe endash emdash quotedblleft quotedblright quoteleft quoteright divide lozenge ydieresis "
         + "Ydieresis fraction currency guilsinglleft guilsinglright fi fl daggerdbl periodcentered quotesinglbase quotedblbase perthousand "
         + "Acircumflex Ecircumflex Aacute Edieresis Egrave Iacute Icircumflex Idieresis Igrave Oacute Ocircumflex apple Ograve Uacute Ucircumflex "
         + "Ugrave dotlessi circumflex tilde macron breve dotaccent ring cedilla hungarumlaut ogonek caron Lslash lslash Scaron scaron Zcaron zcaron "
         + "brokenbar Eth eth Yacute yacute Thorn thorn minus multiply onesuperior twosuperior threesuperior onehalf onequarter threequarters franc "
         + "Gbreve gbreve Idotaccent Scedilla scedilla Cacute cacute Ccaron ccaron dcroat").Split(' ');

    /// <summary>Each name's first glyph; empty when the table has no names.</summary>
    public static Dictionary<string, int> Read(ReadOnlySpan<byte> post, int glyphCount)
    {
        var names = new Dictionary<string, int>(StringComparer.Ordinal);
        if (post.Length < 32)
        {
            return names;
        }

        uint version = BinaryPrimitives.ReadUInt32BigEndian(post);
        if (version == 0x00010000)
        {
            for (int g = 0; g < Math.Min(glyphCount, Standard.Length); g++)
            {
                names.TryAdd(Standard[g], g);
            }

            return names;
        }

        if (version != 0x00020000 || post.Length < 34)
        {
            return names;
        }

        int count = Math.Min(BinaryPrimitives.ReadUInt16BigEndian(post[32..]), glyphCount);
        int strings = 34 + 2 * count;
        if (strings > post.Length)
        {
            return names;
        }

        // The font's own names, in order, after the index array.
        var own = new List<string>();
        for (int at = strings; at < post.Length;)
        {
            int length = post[at];
            if (at + 1 + length > post.Length)
            {
                break;
            }

            own.Add(Encoding.ASCII.GetString(post.Slice(at + 1, length)));
            at += 1 + length;
        }

        for (int g = 0; g < count; g++)
        {
            int index = BinaryPrimitives.ReadUInt16BigEndian(post[(34 + 2 * g)..]);
            string? name = index < Standard.Length ? Standard[index] : index - Standard.Length < own.Count ? own[index - Standard.Length] : null;
            if (name is not null)
            {
                names.TryAdd(name, g);
            }
        }

        return names;
    }
}

/// <summary>
/// A face a <see cref="FontMap"/> made usable: the map's characters answer through the face's glyphs (and glyphs drawn
/// in after its own), the marks have no width, and GSUB, GDEF and GPOS are made from the map; every other answer and
/// table is the face's. Thread-safe: the tables and drawn glyphs are made once, on first use.
/// </summary>
internal sealed class MappedFontFace : IFontFace
{
    private readonly IFontFace _inner;
    private readonly FontMap _map;
    private readonly FontMapDrawing _drawing;
    private readonly FrozenDictionary<int, int> _cmap;
    private readonly byte[] _classes;                                         // GDEF class of each glyph: 1 base, 2 ligature, 3 mark
    private readonly Dictionary<int, FontMarkPlace> _places;                // placed marks by glyph
    private readonly List<(string Feature, int From, int To)> _singles;
    private readonly List<(string Feature, int[] Components, int Glyph, bool Mark)> _ligatures;
    private readonly List<int> _bases;
    private readonly List<int> _ownMarks;                                   // the font's own marks with ink (they size drawn marks)
    private readonly List<(DrawnSet Set, int Index, int Glyph)> _drawn;     // drawn glyphs: their set, their index in it, their glyph id
    private readonly int[] _letters;                                         // the letters' plain glyphs (they size drawn glyphs of "letters")
    private readonly List<(ComposedForm Recipe, int Body, int Glyph)> _composed;   // composed glyphs: their recipe, the font's glyph they start from, their id
    private readonly Lazy<(GlyphOutline Outline, int Advance)[]> _drawnGlyphs;     // the drawn and composed glyphs, after the font's own
    private readonly Lazy<byte[]> _gsub, _gdef, _gpos;

    public MappedFontFace(IFontFace inner, FontMap map, GlyphResolver resolver, Dictionary<int, int[]> forms, FontMapDrawing drawing)
    {
        _inner = inner;
        _map = map;
        _drawing = drawing;
        var cmap = new Dictionary<int, int>();
        _places = new();
        _singles = [];
        _ligatures = [];
        _ownMarks = [];
        _drawn = [];
        _composed = [];
        var ligatureGlyphs = new HashSet<int>();

        foreach (var (code, glyphs) in forms)
        {
            cmap[code] = glyphs[0];
            for (int f = 0; f < glyphs.Length; f++)
            {
                if (glyphs[f] != 0 && glyphs[f] != glyphs[0])
                {
                    _singles.Add((FontMap.Forms[f], glyphs[0], glyphs[f]));
                }
            }
        }

        _letters = [.. forms.Values.Select(g => g[0]).Distinct().Order()];

        foreach (var (code, mark) in map.Marks)
        {
            int glyph = resolver.Resolve(mark.Glyph);
            if (glyph != 0 && cmap.TryAdd(code, glyph))
            {
                if (mark.Place != FontMarkPlace.None)
                {
                    _places.TryAdd(glyph, mark.Place);
                }

                if (!inner.Outline(glyph).IsEmpty && !_ownMarks.Contains(glyph))
                {
                    _ownMarks.Add(glyph);
                }
            }
        }

        foreach (var (code, refs) in map.Characters)
        {
            int glyph = resolver.Resolve(refs);
            if (glyph != 0)
            {
                cmap.TryAdd(code, glyph);
            }
        }

        int FormGlyph(FontMapComponent c) => c.Form is null ? cmap.GetValueOrDefault(c.CodePoint)
            : forms.TryGetValue(c.CodePoint, out var g) ? g[Index(c.Form)] : 0;

        foreach (var (code, target) in map.Aliases)
        {
            int glyph = FormGlyph(target);
            if (glyph != 0)
            {
                cmap.TryAdd(code, glyph);
            }
        }

        // Drawn glyphs: what the font lacks of each set (a base set only when the font answers none of it, so a font's
        // own digits are never mixed with drawn ones), as new glyphs after its own.
        int next = inner.GlyphCount;
        var drawnCodes = new HashSet<int>();
        foreach (var set in DrawnGlyphData.Library.Sets)
        {
            if (set.Mark ? !drawing.Marks : !drawing.Digits)
            {
                continue;
            }

            if (!set.Mark && set.Glyphs.Any(g => cmap.ContainsKey(g.CodePoint) || inner.Glyph(g.CodePoint) != 0))
            {
                continue;
            }

            for (int i = 0; i < set.Glyphs.Length; i++)
            {
                var g = set.Glyphs[i];
                int own = cmap.TryGetValue(g.CodePoint, out int mapped) ? mapped : inner.Glyph(g.CodePoint);
                bool missing = own == 0 || (set.Mark && drawing.ReplaceBlankMarks && inner.Outline(own).IsEmpty);
                if (!missing || !drawnCodes.Add(g.CodePoint))
                {
                    continue;
                }

                _drawn.Add((set, i, next));
                cmap[g.CodePoint] = next;
                if (set.Mark && g.Place != FontMarkPlace.None)
                {
                    _places[next] = g.Place;
                }

                next++;
            }
        }

        // Letters the font lacks, composed from its own glyphs in every form the recipe's letter has (only when its plain
        // form can be made): new glyphs after the drawn ones, with the single substitutions of their forms.
        if (drawing.Letters)
        {
            foreach (var letter in DrawnGlyphData.Library.Composed)
            {
                if (cmap.ContainsKey(letter.CodePoint) || inner.Glyph(letter.CodePoint) != 0)
                {
                    continue;
                }

                int Body(int f) => letter.Of(FontMap.Forms[f]) is { } r && forms.TryGetValue(r.From, out var g) ? g[Index(r.FromForm ?? FontMap.Forms[f])] : 0;
                if (Body(0) == 0)
                {
                    continue;
                }

                var glyphs = new int[FontMap.Forms.Count];
                for (int f = 0; f < glyphs.Length; f++)
                {
                    if (Body(f) is int body and not 0)
                    {
                        _composed.Add((letter.Of(FontMap.Forms[f])!, body, next));
                        glyphs[f] = next++;
                    }
                }

                forms[letter.CodePoint] = glyphs;
                cmap[letter.CodePoint] = glyphs[0];
                for (int f = 1; f < glyphs.Length; f++)
                {
                    if (glyphs[f] != 0)
                    {
                        _singles.Add((FontMap.Forms[f], glyphs[0], glyphs[f]));
                    }
                }
            }
        }

        foreach (var lig in map.Ligatures)
        {
            // A ligature made for some font families only: where fonts sharing a layout draw the code differently.
            if (lig.Fonts.Count > 0 && !lig.Fonts.Contains(inner.Family, StringComparer.OrdinalIgnoreCase))
            {
                continue;
            }

            var components = lig.Components.Select(FormGlyph).ToArray();
            int glyph = resolver.Resolve(lig.Glyph);
            // A ligature of glyphs one of which is drawn is left out: the drawn glyphs are placed by anchors instead.
            if (glyph == 0 || components.Any(c => c == 0 || c >= inner.GlyphCount) || lig.Components.Any(c => drawnCodes.Contains(c.CodePoint)))
            {
                continue;
            }

            bool mark = lig.Place is not null;
            _ligatures.Add((lig.Feature, components, glyph, mark));
            if (mark && lig.Place != FontMarkPlace.None)
            {
                _places.TryAdd(glyph, lig.Place!.Value);
            }

            if (!mark)
            {
                ligatureGlyphs.Add(glyph);
            }
        }

        _cmap = cmap.ToFrozenDictionary();
        GlyphCount = next;
        _classes = new byte[next];
        var marks = new HashSet<int>(_places.Keys);
        foreach (var (code, _) in map.Marks)
        {
            if (cmap.TryGetValue(code, out int g))
            {
                marks.Add(g);
            }
        }

        foreach (var lig in _ligatures.Where(l => l.Mark))
        {
            marks.Add(lig.Glyph);
        }

        foreach (var (_, _, glyph) in _drawn.Where(d => d.Set.Mark))
        {
            marks.Add(glyph);
        }

        for (int g = 1; g < next; g++)
        {
            _classes[g] = marks.Contains(g) ? (byte)3 : ligatureGlyphs.Contains(g) ? (byte)2 : (byte)1;
        }

        // The glyphs marks are placed on: the letters in every form, the ligatures and the other characters' glyphs.
        var bases = new SortedSet<int>(forms.Values.SelectMany(g => g).Where(g => g != 0));
        bases.UnionWith(ligatureGlyphs);
        bases.UnionWith(cmap.Values.Where(g => _classes[g] != 3));
        bases.Remove(0);
        _bases = [.. bases];

        _drawnGlyphs = new(DrawGlyphs);
        _gsub = new(() => FontMapTables.Substitutions(_map.Scripts, _singles, _ligatures));
        _gdef = new(() => FontMapTables.Definitions(_classes));
        _gpos = new(Positions);
    }

    public string Family => _inner.Family;

    public string Style => _inner.Style;

    public string Format => _inner.Format;

    public FontMetrics Metrics => _inner.Metrics;

    public int GlyphCount { get; }

    /// <summary>The glyphs drawn in, by character.</summary>
    public IEnumerable<int> DrawnCharacters => _drawn.Select(d => d.Set.Glyphs[d.Index].CodePoint);

    public int Glyph(int codePoint) => _cmap.TryGetValue(codePoint, out int glyph) ? glyph : _inner.Glyph(codePoint);

    public int Advance(int glyph)
    {
        if (glyph >= _inner.GlyphCount)
        {
            return (uint)glyph < (uint)GlyphCount ? _drawnGlyphs.Value[glyph - _inner.GlyphCount].Advance : 0;
        }

        return (uint)glyph < (uint)_classes.Length && _classes[glyph] == 3 ? 0 : _inner.Advance(glyph);
    }

    public GlyphOutline Outline(int glyph) => glyph >= _inner.GlyphCount
        ? (uint)glyph < (uint)GlyphCount ? _drawnGlyphs.Value[glyph - _inner.GlyphCount].Outline : GlyphOutline.Empty
        : _inner.Outline(glyph);

    public ReadOnlyMemory<byte> Table(string tag) => tag switch
    {
        "GSUB" => _gsub.Value,
        "GDEF" => _gdef.Value,
        "GPOS" => _places.Count > 0 ? _gpos.Value : _inner.Table(tag),
        _ => _inner.Table(tag),
    };

    private static int Index(string form)
    {
        for (int i = 0; i < FontMap.Forms.Count; i++)
        {
            if (FontMap.Forms[i] == form)
            {
                return i;
            }
        }

        return 0;
    }

    // Each drawn set sized from the font's own glyphs of its kind, drawn once, then the composed letters; the glyphs in
    // glyph id order after the font's own.
    private (GlyphOutline, int)[] DrawGlyphs()
    {
        var result = new (GlyphOutline, int)[GlyphCount - _inner.GlyphCount];
        if (result.Length == 0)
        {
            return result;
        }

        int upem = _inner.Metrics.UnitsPerEm;
        var digits = Size(_map.Measure.Select(c => _cmap.TryGetValue(c, out int g) ? g : _inner.Glyph(c)).Where(g => g != 0 && g < _inner.GlyphCount))
            ?? Size(_letters) ?? new DrawnSize(upem * 0.33, upem * 0.6, upem * 0.07, upem * 0.03);
        var letters = Size(_letters) ?? digits;
        var ownMarks = _ownMarks.Select(g => InkMeasure.Of(_inner.Outline(g), upem, _map.AnchorWindow)).OfType<InkStats>().ToList();
        DrawnSize MarkSize(string measure)
        {
            var basis = measure == "letters" ? letters : digits;
            double side = ownMarks.Count > 0 ? InkMeasure.Median([.. ownMarks.Select(i => Math.Max(i.Width, i.Height))]) : _drawing.MarkShare * basis.Height;
            double pen = ownMarks.Count > 0 ? InkMeasure.Median([.. ownMarks.Select(i => i.Thickness)]) : _drawing.MarkPenShare * basis.Pen;
            side = _drawing.MarkSize ?? side;
            pen = _drawing.MarkPen ?? pen;
            return new DrawnSize(Math.Max(1, side - pen), Math.Max(1, side - pen), pen, 0);
        }

        foreach (var set in _drawn.Select(d => d.Set).Distinct())
        {
            DrawnSize size;
            if (set.Mark)
            {
                size = MarkSize(set.Measure);
            }
            else
            {
                var basis = set.Measure == "letters" ? letters : digits;
                size = basis with
                {
                    Width = _drawing.DigitWidth ?? basis.Width,
                    Height = _drawing.DigitHeight ?? basis.Height,
                    Pen = _drawing.DigitPen ?? basis.Pen,
                };
                size = size with { Width = Math.Max(1, size.Width - size.Pen) };
            }

            var drawn = StrokeDrawing.Draw(set, size, _drawing, DrawnGlyphData.Library.DiscSides);
            for (int k = 0; k < _drawn.Count; k++)
            {
                if (_drawn[k].Set == set)
                {
                    result[_drawn[k].Glyph - _inner.GlyphCount] = drawn[_drawn[k].Index];
                }
            }
        }

        // Composed letters: the font's glyph, its dots as the recipe says, its strokes drawn at the marks' size.
        foreach (var (recipe, body, glyph) in _composed)
        {
            GlyphOutline? strokes = null;
            if (recipe.Strokes.Length > 0)
            {
                var stroke = new DrawnSet("composed", Mark: true, "marks", [new DrawnGlyph(0, recipe.Place, recipe.Strokes)]);
                strokes = StrokeDrawing.Draw(stroke, MarkSize("marks"), _drawing, DrawnGlyphData.Library.DiscSides)[0].Outline;
            }

            result[glyph - _inner.GlyphCount] = (Composition.Compose(_inner.Outline(body), recipe, strokes, upem, _map.AnchorGap, _map.AnchorWindow), _inner.Advance(body));
        }

        return result;
    }

    // The typical size of a font's glyphs: of those at least half as tall as the tallest, the median ink height, width
    // and side bearing; the median stroke thickness of them all. Null when none has ink.
    private DrawnSize? Size(IEnumerable<int> glyphs)
    {
        int upem = _inner.Metrics.UnitsPerEm;
        var measured = glyphs.Distinct().Select(g => (Glyph: g, Ink: InkMeasure.Of(_inner.Outline(g), upem, _map.AnchorWindow))).Where(m => m.Ink is not null)
            .Select(m => (m.Glyph, Ink: m.Ink!.Value)).ToList();
        if (measured.Count == 0)
        {
            return null;
        }

        double tallest = measured.Max(m => m.Ink.Height);
        var tall = measured.Where(m => m.Ink.Height >= tallest / 2).ToList();
        return new DrawnSize(
            InkMeasure.Median([.. tall.Select(m => m.Ink.Width)]),
            InkMeasure.Median([.. tall.Select(m => m.Ink.Height)]),
            InkMeasure.Median([.. tall.Select(m => m.Ink.Thickness)]),
            Math.Max(0, InkMeasure.Median([.. tall.Select(m => (m.Ink.MinX + _inner.Advance(m.Glyph) - m.Ink.MaxX) / 2)])));
    }

    // Mark-to-base and mark-to-mark anchors from the ink: a mark's at its ink's centre, bottom for one above and top for
    // one below; a base's over (or under) its ink around its middle, the map's gap away; a mark under one of its side
    // stacks on it the same way.
    private byte[] Positions()
    {
        int upem = _inner.Metrics.UnitsPerEm;
        InkStats? Ink(int glyph) => InkMeasure.Of(Outline(glyph), upem, _map.AnchorWindow);
        int gap = _map.AnchorGap;
        var classes = new List<FontMarkPlace>();
        if (_places.ContainsValue(FontMarkPlace.Above))
        {
            classes.Add(FontMarkPlace.Above);
        }

        if (_places.ContainsValue(FontMarkPlace.Below))
        {
            classes.Add(FontMarkPlace.Below);
        }

        var marks = new List<(int Glyph, int Class, (int X, int Y) Anchor)>();
        var stacks = new List<(int Glyph, (int X, int Y)?[] Anchors)>();
        foreach (var (glyph, place) in _places.OrderBy(p => p.Key))
        {
            var ink = Ink(glyph);
            int cls = classes.IndexOf(place);
            if (ink is not { } i)
            {
                marks.Add((glyph, cls, (0, 0)));
                continue;
            }

            int cx = (int)Math.Round(i.CenterX);
            marks.Add((glyph, cls, (cx, (int)Math.Round(place == FontMarkPlace.Above ? i.MinY : i.MaxY))));
            var anchors = new (int, int)?[classes.Count];
            anchors[cls] = (cx, (int)Math.Round(place == FontMarkPlace.Above ? i.MaxY + gap : i.MinY - gap));
            stacks.Add((glyph, anchors));
        }

        var bases = new List<(int Glyph, (int X, int Y)?[] Anchors)>();
        foreach (int glyph in _bases)
        {
            if (Ink(glyph) is not { } i)
            {
                continue;
            }

            var anchors = new (int, int)?[classes.Count];
            int cx = (int)Math.Round(i.CenterX);
            for (int c = 0; c < classes.Count; c++)
            {
                anchors[c] = (cx, (int)Math.Round(classes[c] == FontMarkPlace.Above ? i.Top + gap : i.Bottom - gap));
            }

            bases.Add((glyph, anchors));
        }

        return FontMapTables.Positions(_map.Scripts, classes.Count, marks, bases, stacks);
    }
}

/// <summary>Writes the OpenType layout tables a mapped face carries (big-endian, offsets from each table's start).</summary>
internal static class FontMapTables
{
    /// <summary>GSUB: a single substitution lookup per form feature, a ligature lookup per feature (marks' ligatures apart).</summary>
    public static byte[] Substitutions(IReadOnlyList<string> scripts, List<(string Feature, int From, int To)> singles,
        List<(string Feature, int[] Components, int Glyph, bool Mark)> ligatures)
    {
        var lookups = new List<(string Feature, int Type, int Flags, Action<TableWriter, int> Write)>();
        foreach (var group in ligatures.Where(l => l.Mark).GroupBy(l => l.Feature, StringComparer.Ordinal))
        {
            var list = group.ToList();
            lookups.Add((group.Key, 4, 0, (w, at) => Ligatures(w, at, list)));
        }

        foreach (string form in FontMap.Forms)
        {
            var pairs = singles.Where(s => s.Feature == form).GroupBy(s => s.From).Select(g => (From: g.Key, To: g.First().To)).OrderBy(p => p.From).ToList();
            if (pairs.Count > 0)
            {
                lookups.Add((form, 1, 0, (w, at) => Singles(w, at, pairs)));
            }
        }

        foreach (var group in ligatures.Where(l => !l.Mark).GroupBy(l => l.Feature, StringComparer.Ordinal))
        {
            var list = group.ToList();
            lookups.Add((group.Key, 4, 8, (w, at) => Ligatures(w, at, list)));   // marks between the components are passed over
        }

        return Layout(scripts, lookups);
    }

    /// <summary>GDEF 1.0 with the glyph classes only.</summary>
    public static byte[] Definitions(byte[] classes)
    {
        var w = new TableWriter();
        w.U32(0x00010000);
        int classDef = w.Reserve();
        w.U16(0);
        w.U16(0);
        w.U16(0);
        w.Offset(classDef, 0);
        var ranges = new List<(int Start, int End, int Class)>();
        for (int g = 0; g < classes.Length; g++)
        {
            if (classes[g] == 0)
            {
                continue;
            }

            if (ranges.Count > 0 && ranges[^1].End == g - 1 && ranges[^1].Class == classes[g])
            {
                ranges[^1] = (ranges[^1].Start, g, classes[g]);
            }
            else
            {
                ranges.Add((g, g, classes[g]));
            }
        }

        w.U16(2);
        w.U16(ranges.Count);
        foreach (var (start, end, cls) in ranges)
        {
            w.U16(start);
            w.U16(end);
            w.U16(cls);
        }

        return w.ToArray();
    }

    /// <summary>GPOS: mark-to-base ("mark") and mark-to-mark ("mkmk"), one lookup each.</summary>
    public static byte[] Positions(IReadOnlyList<string> scripts, int classCount, List<(int Glyph, int Class, (int X, int Y) Anchor)> marks,
        List<(int Glyph, (int X, int Y)?[] Anchors)> bases, List<(int Glyph, (int X, int Y)?[] Anchors)> stacks)
    {
        var lookups = new List<(string Feature, int Type, int Flags, Action<TableWriter, int> Write)>
        {
            ("mark", 4, 0, (w, at) => MarkAttachment(w, at, classCount, marks, bases)),
        };
        if (stacks.Count > 0)
        {
            lookups.Add(("mkmk", 6, 0, (w, at) => MarkAttachment(w, at, classCount, marks, stacks)));
        }

        return Layout(scripts, lookups);
    }

    // The script, feature and lookup lists, every script's default language system listing every feature.
    private static byte[] Layout(IReadOnlyList<string> scripts, List<(string Feature, int Type, int Flags, Action<TableWriter, int> Write)> lookups)
    {
        var features = lookups.Select((l, i) => (l.Feature, Index: i)).GroupBy(l => l.Feature, StringComparer.Ordinal)
            .OrderBy(g => g.Key, StringComparer.Ordinal).Select(g => (Tag: g.Key, Lookups: g.Select(x => x.Index).ToList())).ToList();
        var w = new TableWriter();
        w.U32(0x00010000);
        int scriptList = w.Reserve(), featureList = w.Reserve(), lookupList = w.Reserve();

        int sl = w.Position;
        w.Offset(scriptList, 0);
        var sorted = scripts.Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal).ToList();
        w.U16(sorted.Count);
        var slots = new int[sorted.Count];
        for (int i = 0; i < sorted.Count; i++)
        {
            w.Tag(sorted[i]);
            slots[i] = w.Reserve();
        }

        foreach (int slot in slots)
        {
            w.Offset(slot, sl);
            w.U16(4);                                                           // the default language system right after
            w.U16(0);
            w.U16(0);
            w.U16(0xFFFF);
            w.U16(features.Count);
            for (int f = 0; f < features.Count; f++)
            {
                w.U16(f);
            }
        }

        int fl = w.Position;
        w.Offset(featureList, 0);
        w.U16(features.Count);
        slots = new int[features.Count];
        for (int i = 0; i < features.Count; i++)
        {
            w.Tag(features[i].Tag);
            slots[i] = w.Reserve();
        }

        for (int i = 0; i < features.Count; i++)
        {
            w.Offset(slots[i], fl);
            w.U16(0);
            w.U16(features[i].Lookups.Count);
            foreach (int l in features[i].Lookups)
            {
                w.U16(l);
            }
        }

        int ll = w.Position;
        w.Offset(lookupList, 0);
        w.U16(lookups.Count);
        slots = new int[lookups.Count];
        for (int i = 0; i < lookups.Count; i++)
        {
            slots[i] = w.Reserve();
        }

        for (int i = 0; i < lookups.Count; i++)
        {
            int lt = w.Position;
            w.Offset(slots[i], ll);
            w.U16(lookups[i].Type);
            w.U16(lookups[i].Flags);
            w.U16(1);
            int sub = w.Reserve();
            int st = w.Position;
            w.Offset(sub, lt);
            lookups[i].Write(w, st);
        }

        return w.ToArray();
    }

    private static void Coverage(TableWriter w, IReadOnlyList<int> sortedGlyphs)
    {
        w.U16(1);
        w.U16(sortedGlyphs.Count);
        foreach (int g in sortedGlyphs)
        {
            w.U16(g);
        }
    }

    private static void Singles(TableWriter w, int st, List<(int From, int To)> pairs)
    {
        w.U16(2);
        int coverage = w.Reserve();
        w.U16(pairs.Count);
        foreach (var (_, to) in pairs)
        {
            w.U16(to);
        }

        w.Offset(coverage, st);
        Coverage(w, [.. pairs.Select(p => p.From)]);
    }

    private static void Ligatures(TableWriter w, int st, List<(string Feature, int[] Components, int Glyph, bool Mark)> ligatures)
    {
        var sets = ligatures.GroupBy(l => l.Components[0]).OrderBy(g => g.Key)
            .Select(g => (First: g.Key, Ligatures: g.DistinctBy(l => string.Join(",", l.Components)).OrderByDescending(l => l.Components.Length).ToList())).ToList();
        w.U16(1);
        int coverage = w.Reserve();
        w.U16(sets.Count);
        var slots = sets.Select(_ => w.Reserve()).ToArray();
        for (int s = 0; s < sets.Count; s++)
        {
            int set = w.Position;
            w.Offset(slots[s], st);
            w.U16(sets[s].Ligatures.Count);
            var ligatureSlots = sets[s].Ligatures.Select(_ => w.Reserve()).ToArray();
            for (int l = 0; l < ligatureSlots.Length; l++)
            {
                w.Offset(ligatureSlots[l], set);
                var lig = sets[s].Ligatures[l];
                w.U16(lig.Glyph);
                w.U16(lig.Components.Length);
                for (int c = 1; c < lig.Components.Length; c++)
                {
                    w.U16(lig.Components[c]);
                }
            }
        }

        w.Offset(coverage, st);
        Coverage(w, [.. sets.Select(s => s.First)]);
    }

    // MarkBasePos and MarkMarkPos share a layout: marks (with their class and anchor), then targets (an anchor per class).
    private static void MarkAttachment(TableWriter w, int st, int classCount, List<(int Glyph, int Class, (int X, int Y) Anchor)> marks,
        List<(int Glyph, (int X, int Y)?[] Anchors)> targets)
    {
        w.U16(1);
        int markCoverage = w.Reserve(), targetCoverage = w.Reserve();
        w.U16(classCount);
        int markArray = w.Reserve(), targetArray = w.Reserve();
        w.Offset(markCoverage, st);
        Coverage(w, [.. marks.Select(m => m.Glyph)]);
        w.Offset(targetCoverage, st);
        Coverage(w, [.. targets.Select(t => t.Glyph)]);

        int ma = w.Position;
        w.Offset(markArray, st);
        w.U16(marks.Count);
        var slots = new int[marks.Count];
        for (int i = 0; i < marks.Count; i++)
        {
            w.U16(marks[i].Class);
            slots[i] = w.Reserve();
        }

        for (int i = 0; i < marks.Count; i++)
        {
            w.Offset(slots[i], ma);
            Anchor(w, marks[i].Anchor);
        }

        int ta = w.Position;
        w.Offset(targetArray, st);
        w.U16(targets.Count);
        var anchorSlots = new List<(int Slot, (int X, int Y) Anchor)>();
        foreach (var (_, anchors) in targets)
        {
            foreach (var anchor in anchors)
            {
                int slot = w.Reserve();
                if (anchor is { } a)
                {
                    anchorSlots.Add((slot, a));
                }
            }
        }

        foreach (var (slot, anchor) in anchorSlots)
        {
            w.Offset(slot, ta);
            Anchor(w, anchor);
        }
    }

    private static void Anchor(TableWriter w, (int X, int Y) anchor)
    {
        w.U16(1);
        w.I16(anchor.X);
        w.I16(anchor.Y);
    }
}

/// <summary>A growing big-endian buffer with 16-bit offset slots filled in later.</summary>
internal sealed class TableWriter
{
    private byte[] _bytes = new byte[1024];

    public int Position { get; private set; }

    public void U16(int value)
    {
        Ensure(2);
        BinaryPrimitives.WriteUInt16BigEndian(_bytes.AsSpan(Position), checked((ushort)value));
        Position += 2;
    }

    public void I16(int value)
    {
        Ensure(2);
        BinaryPrimitives.WriteInt16BigEndian(_bytes.AsSpan(Position), checked((short)value));
        Position += 2;
    }

    public void U32(uint value)
    {
        Ensure(4);
        BinaryPrimitives.WriteUInt32BigEndian(_bytes.AsSpan(Position), value);
        Position += 4;
    }

    public void Tag(string tag)
    {
        Ensure(4);
        for (int i = 0; i < 4; i++)
        {
            _bytes[Position + i] = (byte)tag[i];
        }

        Position += 4;
    }

    /// <summary>A 16-bit slot for an offset written later; returns where it is.</summary>
    public int Reserve()
    {
        int at = Position;
        U16(0);
        return at;
    }

    /// <summary>Fills the slot at <paramref name="slot"/> with the distance from <paramref name="start"/> to here.</summary>
    /// <exception cref="InvalidOperationException">The distance does not fit 16 bits (a map too large for one subtable).</exception>
    public void Offset(int slot, int start)
    {
        int distance = Position - start;
        if (distance is < 0 or > 0xFFFF)
        {
            throw new InvalidOperationException($"A font map's layout table needs an offset of {distance} bytes, past the 65,535 one subtable allows.");
        }

        BinaryPrimitives.WriteUInt16BigEndian(_bytes.AsSpan(slot), (ushort)distance);
    }

    public byte[] ToArray() => _bytes.AsSpan(0, Position).ToArray();

    private void Ensure(int more)
    {
        if (Position + more > _bytes.Length)
        {
            Array.Resize(ref _bytes, Math.Max(_bytes.Length * 2, Position + more));
        }
    }
}
