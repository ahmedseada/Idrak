// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text.Json;

namespace Idrak.Abstraction.Data;

/// <summary>How a <see cref="FontGlyphRef"/> names a glyph of a font.</summary>
public enum FontGlyphRefKind
{
    /// <summary>By the code point the font's own character map gives it ("U+F241").</summary>
    Code,

    /// <summary>By its glyph id ("#12").</summary>
    Id,

    /// <summary>By its name in the font's post table ("inbaa").</summary>
    Name,
}

/// <summary>
/// A glyph of a font as a <see cref="FontMap"/> names it: "U+F241" (the code point the font's own character map gives
/// it), "#12" (its glyph id) or any other text (its name in the font's post table).
/// </summary>
/// <param name="Kind">How the glyph is named.</param>
/// <param name="Value">The code point or the glyph id (0 for a name).</param>
/// <param name="Text">The reference as written (a fallback is found by it).</param>
public readonly record struct FontGlyphRef(FontGlyphRefKind Kind, int Value, string Text)
{
    /// <summary>The name, for <see cref="FontGlyphRefKind.Name"/>.</summary>
    public string Name => Text;

    /// <summary>Reads a reference; null when <paramref name="text"/> is empty or a malformed code or id.</summary>
    public static FontGlyphRef? Parse(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (text.Length == 0)
        {
            return null;
        }

        if (text.StartsWith("U+", StringComparison.Ordinal))
        {
            return FontMap.TryCodePoint(text, out int code) ? new FontGlyphRef(FontGlyphRefKind.Code, code, text) : null;
        }

        if (text[0] == '#')
        {
            return int.TryParse(text.AsSpan(1), NumberStyles.None, CultureInfo.InvariantCulture, out int id) ? new FontGlyphRef(FontGlyphRefKind.Id, id, text) : null;
        }

        return new FontGlyphRef(FontGlyphRefKind.Name, 0, text);
    }

    /// <inheritdoc/>
    public override string ToString() => Text;
}

/// <summary>Where a mark sits on the letter before it: above its ink, below it, or where the font drew it.</summary>
public enum FontMarkPlace
{
    /// <summary>Not placed: drawn where the font draws it, with no width.</summary>
    None,

    /// <summary>On the top of its base's ink.</summary>
    Above,

    /// <summary>Under the bottom of its base's ink.</summary>
    Below,
}

/// <summary>A character in one of its positional forms, as a <see cref="FontMap"/> writes it: "U+0644.init", or "U+0644" (its plain glyph).</summary>
/// <param name="CodePoint">The character.</param>
/// <param name="Form">"isol", "init", "medi" or "fina"; null for the character's plain glyph.</param>
public readonly record struct FontMapComponent(int CodePoint, string? Form)
{
    /// <inheritdoc/>
    public override string ToString() => Form is null ? $"U+{CodePoint:X4}" : $"U+{CodePoint:X4}.{Form}";
}

/// <summary>A letter of a <see cref="FontMap"/>: its glyph in each positional form, named outright or found by names.</summary>
/// <param name="Forms">The glyph of each form given outright ("isol", "init", "medi", "fina": alternatives, the first found wins).</param>
/// <param name="Names">The letter's names: the glyph of a form not given outright is a form prefix (<see cref="FontMap.Prefixes"/>) before one of them.</param>
public sealed record FontMapLetter(IReadOnlyDictionary<string, IReadOnlyList<FontGlyphRef>> Forms, IReadOnlyList<string> Names);

/// <summary>A ligature of a <see cref="FontMap"/>: glyphs in a row that one glyph replaces under an OpenType feature.</summary>
/// <param name="Feature">The feature that makes it ("rlig", "liga", "ccmp").</param>
/// <param name="Components">The characters it replaces, each in its form.</param>
/// <param name="Glyph">The ligature's glyph (alternatives, the first found wins).</param>
/// <param name="Place">Where the ligature sits when it is a mark (a stacked mark: <see cref="FontMarkPlace.Above"/>); null for a base.</param>
public sealed record FontMapLigature(string Feature, IReadOnlyList<FontMapComponent> Components, IReadOnlyList<FontGlyphRef> Glyph, FontMarkPlace? Place)
{
    /// <summary>The font families (as their name table gives them, ignoring case) the ligature is made for; empty for every font the map fits.</summary>
    public IReadOnlyList<string> Fonts { get; init; } = [];
}

/// <summary>A mark of a <see cref="FontMap"/>: its glyph, which gets no width and the mark class, and where it sits on its base.</summary>
/// <param name="Glyph">The mark's glyph (alternatives).</param>
/// <param name="Place">Where it is placed.</param>
public sealed record FontMapMark(IReadOnlyList<FontGlyphRef> Glyph, FontMarkPlace Place);

/// <summary>
/// A mark glyph's alternate over short bases (a font's second, lower set of marks, drawn for letters that do not reach the
/// tall letters' height): a mark placed above whose glyph is <see cref="Glyph"/> becomes <see cref="Short"/> when it follows
/// a short base, which the mapper measures per font (<see cref="FontMapDrawing.ShortMarks"/>).
/// </summary>
/// <param name="Glyph">The mark's glyph, a mark's or a stacked mark ligature's (alternatives, the first found wins).</param>
/// <param name="Short">Its alternate over short bases (alternatives).</param>
public sealed record FontMapShortMark(IReadOnlyList<FontGlyphRef> Glyph, IReadOnlyList<FontGlyphRef> Short)
{
    /// <summary>The font families (as their name table gives them, ignoring case) the alternate is used in; empty for every font the map fits.</summary>
    public IReadOnlyList<string> Fonts { get; init; } = [];
}

/// <summary>
/// A font map: rules, as data, that make a "legacy" font usable directly, without converting the file: one whose letters
/// sit under other code points (the Private Use Area) or are found by their glyph names, with no Unicode character map for
/// them and no shaping tables. A <see cref="IFontMapper"/> applies it to an <see cref="IFontFace"/> and gives back a
/// face that answers the Unicode code points and carries the substitution, class and positioning tables the shapers read,
/// made from the rules; <see cref="FontMaps"/> holds the maps fonts are matched against. The rules know no language: the
/// letters, their forms and ligatures are data. A map is read from JSON (<see cref="Parse"/>), an object of these keys:
/// <list type="table">
/// <item><term>name</term><description>the map's name (required).</description></item>
/// <item><term>summary</term><description>one line for listings.</description></item>
/// <item><term>scripts</term><description>the OpenType script tags the made tables list (["arab", "DFLT"]; required).</description></item>
/// <item><term>detect</term><description>when a font is taken to fit: {"share": the part of the letters whose glyph must be found (0.9),
/// "minGlyphs": the fewest glyphs it has (0), "withoutTables": tables it must not have (["GSUB"])}.</description></item>
/// <item><term>prefixes</term><description>for letters given by names: each form's name prefixes ({"isol": [""], "init": ["in"], "medi": ["mid"], "fina": ["fn"]}).</description></item>
/// <item><term>letters</term><description>the letters with positional forms, by code point ("U+0628"): an object of forms to glyphs
/// ({"isol": "U+F24C", "init": "U+F249", ...}), or a list of the letter's names (["baa"]) its forms are found by (required).</description></item>
/// <item><term>characters</term><description>other characters to glyphs ({"U+0660": "U+F230"}).</description></item>
/// <item><term>aliases</term><description>characters that take another's form ({"U+FE8E": "U+0627.fina"}).</description></item>
/// <item><term>ligatures</term><description>[{"feature": "rlig", "components": ["U+0644.init", "U+0627.fina"], "glyph": "U+F2DC", "place": "above" (for a stacked mark),
/// "fonts": ["Family"] (only for these font families, where fonts sharing a layout draw a code differently)}]. Ligatures under
/// <see cref="PairFeature"/> are a hand's letter pairs, under <see cref="SecondStyleFeature"/> its second style of some pairs,
/// under <see cref="WordFeature"/> words it draws whole: a page turns each on at its chance (<see cref="FontMapDrawing.Pairs"/>,
/// <see cref="FontMapDrawing.SecondStyle"/>, <see cref="FontMapDrawing.Words"/>).</description></item>
/// <item><term>marks</term><description>{"U+064E": {"glyph": "U+F2F2", "place": "above" | "below" | "none"}}: no width, the mark class, placed on their base's ink.</description></item>
/// <item><term>shortMarks</term><description>[{"glyph": "U+F2F2", "short": "U+F2E4", "fonts": ["Family"]}]: a mark placed above (a
/// mark's or a stacked mark ligature's glyph) and its alternate over short bases, used where the font draws both with ink
/// (and only in these font families when "fonts" is given); short bases are measured per font (<see cref="FontMapDrawing.ShortMarks"/>).</description></item>
/// <item><term>fallbacks</term><description>a glyph to use when one is missing, by the missing one as written ({"U+F264": "U+F263"}).</description></item>
/// <item><term>anchors</term><description>how marks are placed: {"gap": units between ink and mark (60), "window": half the width, in units, where a base's ink is measured (130)}.</description></item>
/// <item><term>measure</term><description>characters whose glyphs size the drawn glyphs of kind "base" (the font's own digits): their ink height, stroke and width.</description></item>
/// </list>
/// A glyph is "U+F241" (the font's own code point), "#12" (a glyph id) or a name, or a list of these, the first found
/// winning (one without ink passed over for a later one with ink). Unknown keys and bad values are refused, naming the
/// file, the key and what is valid.
/// </summary>
public sealed class FontMap
{
    /// <summary>
    /// The feature a map's letter-pair ligatures are given under ("PAIR", a private tag no font has of its own): off unless a
    /// page turns it on, at the chance <see cref="FontMapDrawing.Pairs"/>, so a hand joins those pairs on some pages and not
    /// on others.
    /// </summary>
    public const string PairFeature = "PAIR";

    /// <summary>
    /// The feature a map's second-style letter pairs are given under ("PAI2", private): a hand's other way of writing some
    /// pairs, used instead of the first style on the pages that ask for it, at the chance <see cref="FontMapDrawing.SecondStyle"/>.
    /// Its ligatures come before the first style's in the map, so where both are on the second style wins.
    /// </summary>
    public const string SecondStyleFeature = "PAI2";

    /// <summary>
    /// The feature a map's whole-word ligatures are given under ("WORD", private): one glyph for a word the font draws whole,
    /// on the pages that ask for it, at the chance <see cref="FontMapDrawing.Words"/>.
    /// </summary>
    public const string WordFeature = "WORD";

    /// <summary>The positional forms a letter may have (OpenType feature tags).</summary>
    public static readonly IReadOnlyList<string> Forms = ["isol", "init", "medi", "fina"];

    private static readonly string[] TopKeys = ["name", "summary", "scripts", "detect", "prefixes", "letters", "characters", "aliases", "ligatures", "marks", "shortMarks", "fallbacks", "anchors", "measure"];
    private static readonly string[] DetectKeys = ["share", "minGlyphs", "withoutTables"];
    private static readonly string[] LigatureKeys = ["feature", "components", "glyph", "place", "fonts"];
    private static readonly string[] MarkKeys = ["glyph", "place"];
    private static readonly string[] ShortMarkKeys = ["glyph", "short", "fonts"];
    private static readonly string[] AnchorKeys = ["gap", "window"];
    private static readonly string[] Places = ["above", "below", "none"];

    private FontMap(string source) => Source = source;

    /// <summary>The map's name.</summary>
    public string Name { get; private set; } = "";

    /// <summary>One line for listings.</summary>
    public string Summary { get; private set; } = "";

    /// <summary>Where the map was read from (a file's path, or a resource's name).</summary>
    public string Source { get; }

    /// <summary>The OpenType script tags of the tables made from it.</summary>
    public IReadOnlyList<string> Scripts { get; private set; } = [];

    /// <summary>The part of the letters (0 to 1) whose glyph a font must have to be taken to fit, when matched.</summary>
    public double DetectShare { get; private set; } = 0.9;

    /// <summary>The fewest glyphs a font must have to be taken to fit, when matched.</summary>
    public int DetectMinGlyphs { get; private set; }

    /// <summary>Tables a font must not have to be taken to fit, when matched (a font with its own GSUB is not legacy).</summary>
    public IReadOnlyList<string> DetectWithoutTables { get; private set; } = ["GSUB"];

    /// <summary>The name prefixes of each form, for letters given by names.</summary>
    public IReadOnlyDictionary<string, IReadOnlyList<string>> Prefixes { get; private set; } = new Dictionary<string, IReadOnlyList<string>>();

    /// <summary>The letters with positional forms, by code point.</summary>
    public IReadOnlyDictionary<int, FontMapLetter> Letters { get; private set; } = new Dictionary<int, FontMapLetter>();

    /// <summary>Other characters and their glyphs.</summary>
    public IReadOnlyDictionary<int, IReadOnlyList<FontGlyphRef>> Characters { get; private set; } = new Dictionary<int, IReadOnlyList<FontGlyphRef>>();

    /// <summary>Characters that answer another character's form.</summary>
    public IReadOnlyDictionary<int, FontMapComponent> Aliases { get; private set; } = new Dictionary<int, FontMapComponent>();

    /// <summary>The ligatures, in order.</summary>
    public IReadOnlyList<FontMapLigature> Ligatures { get; private set; } = [];

    /// <summary>The marks, by code point.</summary>
    public IReadOnlyDictionary<int, FontMapMark> Marks { get; private set; } = new Dictionary<int, FontMapMark>();

    /// <summary>The marks' alternates over short bases, in order.</summary>
    public IReadOnlyList<FontMapShortMark> ShortMarks { get; private set; } = [];

    /// <summary>The glyph used when one is missing, by the missing one as written.</summary>
    public IReadOnlyDictionary<string, IReadOnlyList<FontGlyphRef>> Fallbacks { get; private set; } = new Dictionary<string, IReadOnlyList<FontGlyphRef>>();

    /// <summary>Font units between a base's ink and the mark placed on it.</summary>
    public int AnchorGap { get; private set; } = 60;

    /// <summary>Half the width, in font units around a base's middle, where its ink top and bottom are measured.</summary>
    public int AnchorWindow { get; private set; } = 130;

    /// <summary>The characters whose glyphs are measured for the font's digit height and pen width, in order of preference.</summary>
    public IReadOnlyList<int> Measure { get; private set; } = [];

    /// <inheritdoc/>
    public override string ToString() => $"{Name} ({Source})";

    /// <summary>Reads the map file at <paramref name="path"/> (JSON).</summary>
    /// <exception cref="InvalidDataException">The file is not a valid map: the message names it, the key and what is valid.</exception>
    public static FontMap Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        return Parse(File.ReadAllBytes(path), path);
    }

    /// <summary>Reads a map from its JSON bytes (UTF-8; comments allowed); <paramref name="source"/> names it in messages.</summary>
    /// <exception cref="InvalidDataException">The bytes are not a valid map: the message names the source, the key and what is valid.</exception>
    public static FontMap Parse(ReadOnlyMemory<byte> utf8, string source)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(source);
        JsonDocument document;
        try
        {
            document = JsonDocument.Parse(utf8, new JsonDocumentOptions { CommentHandling = JsonCommentHandling.Skip, AllowTrailingCommas = true });
        }
        catch (JsonException e)
        {
            throw new InvalidDataException($"{source}: not valid JSON ({e.Message}).", e);
        }

        using (document)
        {
            var map = new FontMap(source);
            map.Read(document.RootElement);
            return map;
        }
    }

    /// <summary>Reads "U+XXXX" (4 to 6 hexadecimal digits, at most U+10FFFF).</summary>
    public static bool TryCodePoint(string text, out int codePoint)
    {
        ArgumentNullException.ThrowIfNull(text);
        codePoint = 0;
        return text.Length is >= 6 and <= 8 && text.StartsWith("U+", StringComparison.Ordinal)
            && int.TryParse(text.AsSpan(2), NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture, out codePoint) && codePoint is >= 0 and <= 0x10FFFF;
    }

    private InvalidDataException Bad(string key, string what) => new($"{Source}: '{key}' {what}.");

    private void Keys(JsonElement element, string where, string[] valid)
    {
        if (element.ValueKind != JsonValueKind.Object)
        {
            throw Bad(where, $"must be an object ({string.Join(", ", valid)})");
        }

        foreach (var property in element.EnumerateObject())
        {
            if (Array.IndexOf(valid, property.Name) < 0)
            {
                throw Bad(where.Length == 0 ? property.Name : $"{where}.{property.Name}", $"is not a key of a font map{(where.Length == 0 ? "" : "'s " + where)} (valid: {string.Join(", ", valid)})");
            }
        }
    }

    private int Code(string text, string key) =>
        TryCodePoint(text, out int code) ? code : throw Bad(key, $"needs a character as U+XXXX (4 to 6 hexadecimal digits), not '{text}'");

    private IReadOnlyList<FontGlyphRef> Glyphs(JsonElement element, string key)
    {
        var refs = new List<FontGlyphRef>();
        void Add(JsonElement e)
        {
            if (e.ValueKind != JsonValueKind.String || FontGlyphRef.Parse(e.GetString()!) is not { } r)
            {
                throw Bad(key, $"needs a glyph: \"U+F241\" (the font's own code point), \"#12\" (a glyph id) or a glyph name, or a list of these; not {e.GetRawText()}");
            }

            refs.Add(r);
        }

        if (element.ValueKind == JsonValueKind.Array)
        {
            foreach (var e in element.EnumerateArray())
            {
                Add(e);
            }
        }
        else
        {
            Add(element);
        }

        return refs.Count > 0 ? refs : throw Bad(key, "needs at least one glyph");
    }

    private FontMapComponent Component(string text, string key)
    {
        int dot = text.IndexOf('.', StringComparison.Ordinal);
        string? form = dot < 0 ? null : text[(dot + 1)..];
        if (form is not null && !Forms.Contains(form))
        {
            throw Bad(key, $"names the form '{form}' (valid: {string.Join(", ", Forms)})");
        }

        return new FontMapComponent(Code(dot < 0 ? text : text[..dot], key), form);
    }

    private FontMarkPlace Place(JsonElement element, string key) => element.ValueKind == JsonValueKind.String ? element.GetString() switch
    {
        "above" => FontMarkPlace.Above,
        "below" => FontMarkPlace.Below,
        "none" => FontMarkPlace.None,
        _ => throw Bad(key, $"is '{element.GetString()}' (valid: {string.Join(", ", Places)})"),
    } : throw Bad(key, $"must be one of {string.Join(", ", Places)}");

    private string Text(JsonElement element, string key) =>
        element.ValueKind == JsonValueKind.String && element.GetString() is { Length: > 0 } s ? s : throw Bad(key, "must be a text");

    private int Whole(JsonElement element, string key, int min) =>
        element.ValueKind == JsonValueKind.Number && element.TryGetInt32(out int v) && v >= min ? v : throw Bad(key, $"must be a whole number of at least {min}");

    private void Read(JsonElement root)
    {
        Keys(root, "", TopKeys);
        Name = root.TryGetProperty("name", out var name) ? Text(name, "name") : throw Bad("name", "is missing (every map has a name)");
        Summary = root.TryGetProperty("summary", out var summary) ? Text(summary, "summary") : Name;
        if (!root.TryGetProperty("scripts", out var scripts) || scripts.ValueKind != JsonValueKind.Array)
        {
            throw Bad("scripts", "is missing: a list of OpenType script tags of four characters ([\"arab\", \"DFLT\"])");
        }

        var tags = new List<string>();
        foreach (var s in scripts.EnumerateArray())
        {
            string tag = s.ValueKind == JsonValueKind.String ? s.GetString()! : "";
            tags.Add(tag.Length == 4 && tag.All(c => c is >= ' ' and <= '~') ? tag : throw Bad("scripts", $"holds '{s.GetRawText()}': each is an OpenType script tag of four characters"));
        }

        Scripts = tags.Count > 0 ? tags : throw Bad("scripts", "is empty");

        if (root.TryGetProperty("detect", out var detect))
        {
            Keys(detect, "detect", DetectKeys);
            if (detect.TryGetProperty("share", out var share))
            {
                DetectShare = share.ValueKind == JsonValueKind.Number && share.GetDouble() is >= 0 and <= 1 ? share.GetDouble() : throw Bad("detect.share", "must be a number from 0 to 1");
            }

            if (detect.TryGetProperty("minGlyphs", out var min))
            {
                DetectMinGlyphs = Whole(min, "detect.minGlyphs", 0);
            }

            if (detect.TryGetProperty("withoutTables", out var without))
            {
                DetectWithoutTables = without.ValueKind == JsonValueKind.Array
                    ? [.. without.EnumerateArray().Select(t => t.ValueKind == JsonValueKind.String && t.GetString()!.Length == 4 ? t.GetString()! : throw Bad("detect.withoutTables", "holds table tags of four characters"))]
                    : throw Bad("detect.withoutTables", "must be a list of table tags ([\"GSUB\"])");
            }
        }

        var prefixes = new Dictionary<string, IReadOnlyList<string>>(StringComparer.Ordinal);
        if (root.TryGetProperty("prefixes", out var prefix))
        {
            Keys(prefix, "prefixes", [.. Forms]);
            foreach (var p in prefix.EnumerateObject())
            {
                prefixes[p.Name] = p.Value.ValueKind == JsonValueKind.Array
                    ? [.. p.Value.EnumerateArray().Select(e => e.ValueKind == JsonValueKind.String ? e.GetString()! : throw Bad($"prefixes.{p.Name}", "holds texts"))]
                    : throw Bad($"prefixes.{p.Name}", "must be a list of name prefixes ([\"in\"]; [\"\"] for none)");
            }
        }

        Prefixes = prefixes;

        if (!root.TryGetProperty("letters", out var letters) || letters.ValueKind != JsonValueKind.Object)
        {
            throw Bad("letters", "is missing: an object of letters by code point (\"U+0628\") to their forms or names");
        }

        var letterMap = new Dictionary<int, FontMapLetter>();
        foreach (var l in letters.EnumerateObject())
        {
            string key = $"letters.{l.Name}";
            int code = Code(l.Name, key);
            if (l.Value.ValueKind == JsonValueKind.Array)
            {
                if (!prefixes.ContainsKey("isol"))
                {
                    throw Bad(key, "is given by names, which need 'prefixes' with at least \"isol\"");
                }

                var names = l.Value.EnumerateArray().Select(e => e.ValueKind == JsonValueKind.String && e.GetString()!.Length > 0 ? e.GetString()! : throw Bad(key, "holds the letter's glyph names")).ToList();
                letterMap[code] = new FontMapLetter(new Dictionary<string, IReadOnlyList<FontGlyphRef>>(), names.Count > 0 ? names : throw Bad(key, "needs at least one name"));
            }
            else
            {
                Keys(l.Value, key, [.. Forms]);
                var forms = new Dictionary<string, IReadOnlyList<FontGlyphRef>>(StringComparer.Ordinal);
                foreach (var f in l.Value.EnumerateObject())
                {
                    forms[f.Name] = Glyphs(f.Value, $"{key}.{f.Name}");
                }

                letterMap[code] = forms.ContainsKey("isol") ? new FontMapLetter(forms, []) : throw Bad(key, "has no \"isol\" form (the letter's plain glyph)");
            }
        }

        Letters = letterMap.Count > 0 ? letterMap : throw Bad("letters", "is empty");

        var characters = new Dictionary<int, IReadOnlyList<FontGlyphRef>>();
        if (root.TryGetProperty("characters", out var chars) && chars.ValueKind != JsonValueKind.Object)
        {
            throw Bad("characters", "must be an object of characters by code point (\"U+0660\") to their glyphs");
        }

        var marks = new Dictionary<int, FontMapMark>();
        if (root.TryGetProperty("marks", out var markList))
        {
            if (markList.ValueKind != JsonValueKind.Object)
            {
                throw Bad("marks", "must be an object of marks by code point");
            }

            foreach (var m in markList.EnumerateObject())
            {
                string key = $"marks.{m.Name}";
                Keys(m.Value, key, MarkKeys);
                var glyph = m.Value.TryGetProperty("glyph", out var g) ? Glyphs(g, $"{key}.glyph") : throw Bad($"{key}.glyph", "is missing");
                var place = m.Value.TryGetProperty("place", out var p) ? Place(p, $"{key}.place") : FontMarkPlace.None;
                marks[Code(m.Name, key)] = new FontMapMark(glyph, place);
            }
        }

        Marks = marks;

        if (root.TryGetProperty("characters", out chars))
        {
            foreach (var c in chars.EnumerateObject())
            {
                characters[Code(c.Name, $"characters.{c.Name}")] = Glyphs(c.Value, $"characters.{c.Name}");
            }
        }

        Characters = characters;

        bool Known(int code) => letterMap.ContainsKey(code) || characters.ContainsKey(code) || marks.ContainsKey(code);
        FontMapComponent Checked(FontMapComponent c, string key) =>
            (c.Form is null ? Known(c.CodePoint) : letterMap.ContainsKey(c.CodePoint)) ? c
                : throw Bad(key, c.Form is null ? $"names {c}, which is no letter, character or mark of the map" : $"names {c}, which is no letter of the map");

        var aliases = new Dictionary<int, FontMapComponent>();
        if (root.TryGetProperty("aliases", out var aliasList))
        {
            if (aliasList.ValueKind != JsonValueKind.Object)
            {
                throw Bad("aliases", "must be an object of characters to a letter's form (\"U+0627.fina\")");
            }

            foreach (var a in aliasList.EnumerateObject())
            {
                string key = $"aliases.{a.Name}";
                aliases[Code(a.Name, key)] = Checked(Component(Text(a.Value, key), key), key);
            }
        }

        Aliases = aliases;

        var ligatures = new List<FontMapLigature>();
        if (root.TryGetProperty("ligatures", out var ligs))
        {
            if (ligs.ValueKind != JsonValueKind.Array)
            {
                throw Bad("ligatures", "must be a list of {feature, components, glyph}");
            }

            int index = 0;
            foreach (var l in ligs.EnumerateArray())
            {
                string key = $"ligatures[{index++}]";
                Keys(l, key, LigatureKeys);
                string feature = l.TryGetProperty("feature", out var f) ? Text(f, $"{key}.feature") : throw Bad($"{key}.feature", "is missing");
                if (feature.Length != 4)
                {
                    throw Bad($"{key}.feature", $"is '{feature}': an OpenType feature tag of four characters (rlig, liga, ccmp)");
                }

                if (!l.TryGetProperty("components", out var comps) || comps.ValueKind != JsonValueKind.Array || comps.GetArrayLength() < 2)
                {
                    throw Bad($"{key}.components", "must list at least two characters (\"U+0644.init\")");
                }

                var components = comps.EnumerateArray().Select(e => Checked(Component(Text(e, $"{key}.components"), $"{key}.components"), $"{key}.components")).ToList();
                var glyph = l.TryGetProperty("glyph", out var g) ? Glyphs(g, $"{key}.glyph") : throw Bad($"{key}.glyph", "is missing");
                FontMarkPlace? place = l.TryGetProperty("place", out var p) ? Place(p, $"{key}.place") : null;
                IReadOnlyList<string> fonts = !l.TryGetProperty("fonts", out var families) ? []
                    : families.ValueKind == JsonValueKind.Array && families.GetArrayLength() > 0
                        ? [.. families.EnumerateArray().Select(e => Text(e, $"{key}.fonts"))]
                        : throw Bad($"{key}.fonts", "must be a list of font family names ([\"Family\"])");
                ligatures.Add(new FontMapLigature(feature, components, glyph, place) { Fonts = fonts });
            }
        }

        Ligatures = ligatures;

        var shortMarks = new List<FontMapShortMark>();
        if (root.TryGetProperty("shortMarks", out var shorts))
        {
            if (shorts.ValueKind != JsonValueKind.Array)
            {
                throw Bad("shortMarks", "must be a list of {glyph, short}");
            }

            int index = 0;
            foreach (var m in shorts.EnumerateArray())
            {
                string key = $"shortMarks[{index++}]";
                Keys(m, key, ShortMarkKeys);
                var glyph = m.TryGetProperty("glyph", out var g) ? Glyphs(g, $"{key}.glyph") : throw Bad($"{key}.glyph", "is missing");
                var low = m.TryGetProperty("short", out var l) ? Glyphs(l, $"{key}.short") : throw Bad($"{key}.short", "is missing");
                IReadOnlyList<string> fonts = !m.TryGetProperty("fonts", out var families) ? []
                    : families.ValueKind == JsonValueKind.Array && families.GetArrayLength() > 0
                        ? [.. families.EnumerateArray().Select(e => Text(e, $"{key}.fonts"))]
                        : throw Bad($"{key}.fonts", "must be a list of font family names ([\"Family\"])");
                shortMarks.Add(new FontMapShortMark(glyph, low) { Fonts = fonts });
            }
        }

        ShortMarks = shortMarks;

        var fallbacks = new Dictionary<string, IReadOnlyList<FontGlyphRef>>(StringComparer.Ordinal);
        if (root.TryGetProperty("fallbacks", out var fall))
        {
            if (fall.ValueKind != JsonValueKind.Object)
            {
                throw Bad("fallbacks", "must be an object of a missing glyph (as written) to the glyph used instead");
            }

            foreach (var f in fall.EnumerateObject())
            {
                fallbacks[f.Name] = Glyphs(f.Value, $"fallbacks.{f.Name}");
            }
        }

        Fallbacks = fallbacks;

        if (root.TryGetProperty("anchors", out var anchors))
        {
            Keys(anchors, "anchors", AnchorKeys);
            if (anchors.TryGetProperty("gap", out var gap))
            {
                AnchorGap = Whole(gap, "anchors.gap", 0);
            }

            if (anchors.TryGetProperty("window", out var window))
            {
                AnchorWindow = Whole(window, "anchors.window", 1);
            }
        }

        if (root.TryGetProperty("measure", out var measure))
        {
            Measure = measure.ValueKind == JsonValueKind.Array
                ? [.. measure.EnumerateArray().Select(e => Code(Text(e, "measure"), "measure"))]
                : throw Bad("measure", "must be a list of characters (\"U+0661\")");
        }
    }
}

/// <summary>
/// How glyphs a mapped font lacks are drawn into it (the library's drawn glyphs, data shipped with Idrak.Imaging): the
/// Latin and Arabic-Indic digits (legacy fonts write numbers in their own digits only) and combining marks, as pen strokes
/// thickened into round-ended strokes, and letters composed from the font's own glyphs. A font keeps every glyph it has: only a character the mapped font does not answer (or answers
/// with a mark that has no ink, under <see cref="ReplaceBlankMarks"/>) is drawn. Each size left null is measured from the
/// font itself: the digits' height, pen and width from its own digits (else its letters), the marks' size and pen from its
/// own marks (else a share of the digits' or letters' height and pen).
/// </summary>
public sealed record FontMapDrawing
{
    /// <summary>The variable that turns the drawn digits off ("0", "off", "false" or "no"): "IDRAK_FONT_MAP_DIGITS".</summary>
    public const string DigitsVariable = "IDRAK_FONT_MAP_DIGITS";

    /// <summary>The variable that turns the drawn marks off: "IDRAK_FONT_MAP_MARKS".</summary>
    public const string MarksVariable = "IDRAK_FONT_MAP_MARKS";

    /// <summary>The variable that turns the composed letters off: "IDRAK_FONT_MAP_LETTERS".</summary>
    public const string LettersVariable = "IDRAK_FONT_MAP_LETTERS";

    /// <summary>The variable that sets the chance a page joins letter pairs (<see cref="Pairs"/>; 0 to 1, or off): "IDRAK_FONT_MAP_PAIRS".</summary>
    public const string PairsVariable = "IDRAK_FONT_MAP_PAIRS";

    /// <summary>
    /// The chance (0 to 1) a page drawn in a mapped font writes the font's letter-pair ligatures (<see cref="FontMap.PairFeature"/>):
    /// drawn per page from the page's seed, so a hand joins those pairs on some pages and writes the letters apart on
    /// others. 0.5 by default: half the pages each way, so both shapes are seen as often (0 turns it off, 1 joins them
    /// always); <see cref="PairsVariable"/> sets it ("0", "off", "false" or "no" turn it off).
    /// </summary>
    public double Pairs { get; init; } = Chance(Environment.GetEnvironmentVariable(PairsVariable), 0.5);

    /// <summary>The variable that sets the chance a page writes the second style of letter pairs (<see cref="SecondStyle"/>): "IDRAK_FONT_MAP_STYLE2".</summary>
    public const string SecondStyleVariable = "IDRAK_FONT_MAP_STYLE2";

    /// <summary>
    /// The chance (0 to 1) a page drawn in a mapped font writes the font's second style of letter pairs
    /// (<see cref="FontMap.SecondStyleFeature"/>) instead of the first: drawn per page from its seed, apart from
    /// <see cref="Pairs"/>. 0.5 by default, as for the pairs: both styles seen as often (0 turns it off);
    /// <see cref="SecondStyleVariable"/> sets it.
    /// </summary>
    public double SecondStyle { get; init; } = Chance(Environment.GetEnvironmentVariable(SecondStyleVariable), 0.5);

    /// <summary>The variable that sets the chance a page writes whole-word ligatures (<see cref="Words"/>): "IDRAK_FONT_MAP_WORDS".</summary>
    public const string WordsVariable = "IDRAK_FONT_MAP_WORDS";

    /// <summary>
    /// The chance (0 to 1) a page drawn in a mapped font writes the font's whole-word ligatures (<see cref="FontMap.WordFeature"/>)
    /// where its text holds those words: drawn per page from its seed. 0.5 by default, so a word is seen whole and spelled out
    /// as often (0 turns it off); <see cref="WordsVariable"/> sets it.
    /// </summary>
    public double Words { get; init; } = Chance(Environment.GetEnvironmentVariable(WordsVariable), 0.5);

    /// <summary>The variable that sets which bases are short for marks (<see cref="ShortMarks"/>; auto, off, or a share from 0 to 1): "IDRAK_FONT_MAP_SHORT_MARKS".</summary>
    public const string ShortMarksVariable = "IDRAK_FONT_MAP_SHORT_MARKS";

    /// <summary>
    /// Which bases take a map's marks' alternates over short bases (<see cref="FontMap.ShortMarks"/>): a base whose ink
    /// top (around its middle, where marks sit) is at most a height measured per font. Null (auto, the default): the median
    /// ink bottom of the font's own short alternates less the map's anchor gap, the height the font's maker drew them to
    /// clear; a share from 0 to 1: that share of the font's tall letters' height (the 90th percentile of its bases' ink
    /// tops); 0 turns the alternates off, every mark as before. <see cref="ShortMarksVariable"/> sets it ("auto", a share,
    /// or "0", "off", "false" or "no").
    /// </summary>
    public double? ShortMarks { get; init; } = Share(Environment.GetEnvironmentVariable(ShortMarksVariable));

    /// <summary>Whether the drawn glyphs of kind "base" (the Latin and Arabic-Indic digits) are added (on unless <see cref="DigitsVariable"/> turns them off).</summary>
    public bool Digits { get; init; } = !ComputeResources.SwitchedOff(Environment.GetEnvironmentVariable(DigitsVariable));

    /// <summary>Whether the drawn glyphs of kind "mark" are added (on unless <see cref="MarksVariable"/> turns them off).</summary>
    public bool Marks { get; init; } = !ComputeResources.SwitchedOff(Environment.GetEnvironmentVariable(MarksVariable));

    /// <summary>
    /// Whether letters the font lacks are composed from its own glyphs (another letter's form with its dots replaced, or a
    /// stroke added), in every form (on unless <see cref="LettersVariable"/> turns it off).
    /// </summary>
    public bool Letters { get; init; } = !ComputeResources.SwitchedOff(Environment.GetEnvironmentVariable(LettersVariable));

    /// <summary>Whether a mark the font maps to a glyph with no ink is drawn as if missing (on).</summary>
    public bool ReplaceBlankMarks { get; init; } = true;

    /// <summary>The slant: x moves by this times y (0.15).</summary>
    public double Slant { get; init; } = 0.15;

    /// <summary>The hand's wobble: each point moves by up to this share of the box, either way (0.012).</summary>
    public double Wobble { get; init; } = 0.012;

    /// <summary>Each stroke's pen is the pen times a number drawn from this to <see cref="PenScaleHigh"/> (0.94).</summary>
    public double PenScaleLow { get; init; } = 0.94;

    /// <summary>The top of each stroke's pen scale (1.06).</summary>
    public double PenScaleHigh { get; init; } = 1.06;

    /// <summary>The digits' pen width in font units; null: measured.</summary>
    public double? DigitPen { get; init; }

    /// <summary>The digits' height in font units; null: measured.</summary>
    public double? DigitHeight { get; init; }

    /// <summary>The digits' ink width in font units; null: measured.</summary>
    public double? DigitWidth { get; init; }

    /// <summary>The marks' size (their box's side) in font units; null: measured.</summary>
    public double? MarkSize { get; init; }

    /// <summary>The marks' pen width in font units; null: measured.</summary>
    public double? MarkPen { get; init; }

    /// <summary>A font with no marks of its own: the marks' size as this share of the digits' (or letters') height (0.3).</summary>
    public double MarkShare { get; init; } = 0.3;

    /// <summary>A font with no marks of its own: the marks' pen as this share of the digits' (or letters') pen (0.6).</summary>
    public double MarkPenShare { get; init; } = 0.6;

    /// <summary>The seed of the hand's wobble: the same seed draws the same outlines on every machine (7).</summary>
    public int Seed { get; init; } = 7;

    // A share from a variable: 0 for a switch turned off, the number when it is one from 0 to 1, else null (measured).
    private static double? Share(string? text) =>
        ComputeResources.SwitchedOff(text) ? 0
        : double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double v) && v is >= 0 and <= 1 ? v
        : null;

    // A chance from a variable: off for a switch turned off, the number when it is one from 0 to 1, else the default.
    private static double Chance(string? text, double otherwise) =>
        ComputeResources.SwitchedOff(text) ? 0
        : double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double v) && v is >= 0 and <= 1 ? v
        : otherwise;
}

/// <summary>
/// Applies a <see cref="FontMap"/> to a font: gives back a face that answers the map's Unicode code points with the
/// font's glyphs and carries the layout tables (GSUB, GDEF, GPOS) made from the map, which the text shapers read as they
/// read any font's. Registered in <see cref="FontMappers"/> under its <see cref="IKind.Name"/>; thread-safe.
/// </summary>
public interface IFontMapper : IKind
{
    /// <summary>
    /// The face <paramref name="map"/> makes of <paramref name="face"/>, or null when the map does not fit it. With
    /// <paramref name="detect"/>, the map fits only a font that passes its detection (<see cref="FontMap.DetectShare"/>,
    /// <see cref="FontMap.DetectMinGlyphs"/>, <see cref="FontMap.DetectWithoutTables"/>) and does not answer most of the
    /// map's letters itself; without it (a map given for the font), it fits when any of its letters is found.
    /// <paramref name="drawing"/> says whether and how the glyphs the font lacks are drawn into it.
    /// </summary>
    IFontFace? Map(IFontFace face, FontMap map, bool detect, FontMapDrawing drawing);
}

/// <summary>
/// The font mappers, by name (ignoring case). Built in (Idrak.Imaging): "opentype" (the map's substitutions, classes and
/// mark positions as OpenType tables, with mark anchors measured from the font's ink). Register another with
/// <see cref="Register"/>; one under a library name shadows the library's, guarded by its <see cref="SetPolicy">policy</see>.
/// </summary>
public static class FontMappers
{
    /// <summary>The library's mapper.</summary>
    public const string OpenType = "opentype";

    private static readonly KindTable<IFontMapper> Table = new(nameof(FontMappers), "Font mapper", "add it with FontMappers.Register, or reference Idrak.Imaging",
        (slot, app, library) => new GuardedMapper(slot, app, library), StringComparer.OrdinalIgnoreCase);

    static FontMappers() => LibraryDefaults.Ensure(typeof(FontMappers));

    /// <summary>Registers <paramref name="mapper"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IFontMapper mapper) => Table.Register(mapper, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's mapper <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Names;

    /// <summary>The mapper registered as <paramref name="name"/>, or null.</summary>
    public static IFontMapper? Find(ReadOnlySpan<char> name) => Table.Find(name);

    /// <summary>The mapper registered as <paramref name="name"/>; an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names the registered ones.</exception>
    public static IFontMapper Get(ReadOnlySpan<char> name) => Table.Get(name);

    /// <summary>The library's mapper <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IFontMapper? Default(string name) => Table.Default(name);

    /// <summary>Who registered the mapper <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>What happens when the app's mapper <paramref name="name"/> fails.</summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    /// <summary>Each registered mapper as "name: summary", one per line.</summary>
    public static string Describe() => Table.Describe();

    // An app's mapper over the library's: a failed map is made by the library's, or both are compared on a sample.
    private sealed class GuardedMapper(Slot slot, IFontMapper app, IFontMapper library) : IFontMapper
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public IFontFace? Map(IFontFace face, FontMap map, bool detect, FontMapDrawing drawing)
        {
            if (slot.Policy == SlotPolicy.Shadow)
            {
                var run = slot.Shadow();
                var answer = library.Map(face, map, detect, drawing);
                if (run is not null)
                {
                    run.Answered();
                    try
                    {
                        run.Done(Comparisons.Exact(Answers(answer, map), Answers(app.Map(face, map, detect, drawing), map)));
                    }
                    catch (Exception e) when (e is not OperationCanceledException)
                    {
                        run.Failed(e);
                    }
                }

                return answer;
            }

            try
            {
                return app.Map(face, map, detect, drawing);
            }
            catch (Exception e) when (slot.Failed(e))
            {
                return library.Map(face, map, detect, drawing);
            }
        }

        private static string Answers(IFontFace? face, FontMap map) =>
            face is null ? "none" : string.Join(" ", map.Letters.Keys.Order().Select(cp => face.Glyph(cp).ToString(CultureInfo.InvariantCulture)));
    }
}

/// <summary>
/// The font maps fonts are matched against as they are read (<see cref="FontFormats.Read(string)"/>), by name (ignoring
/// case). The library ships its maps as data with Idrak.Imaging (legacy layouts shared by many fonts); an app adds its
/// own with <see cref="Register"/> (tried before the library's; one of a library name replaces it). A font file with a
/// map beside it (FONT.ttf and FONT<see cref="SidecarSuffix"/>) is mapped by that map. Nothing is fixed: the library's
/// maps (<see cref="Builtin"/>), the maps beside fonts (<see cref="Sidecars"/>), the drawn glyphs (<see cref="Drawing"/>)
/// and the mapper (<see cref="Mapper"/>) are settings. A font no map fits is given back as it was read: the same face.
/// </summary>
public static class FontMaps
{
    /// <summary>The variable that turns the library's maps off ("0", "off", "false" or "no"): "IDRAK_FONT_MAPS".</summary>
    public const string BuiltinVariable = "IDRAK_FONT_MAPS";

    /// <summary>The variable that turns the maps beside fonts off: "IDRAK_FONT_MAP_SIDECARS".</summary>
    public const string SidecarsVariable = "IDRAK_FONT_MAP_SIDECARS";

    /// <summary>What a map beside a font file is named after the font's name without its extension: FONT.ttf, FONT.map.json.</summary>
    public const string SidecarSuffix = ".map.json";

    private static readonly SlotTable<string, FontMap> Registry = new(nameof(FontMaps), guard: null, StringComparer.OrdinalIgnoreCase, newestFirst: true,
        unguarded: "font maps are data: an app's map is tried before the library's and replaces one of its name");

    private static readonly ConditionalWeakTable<IFontFace, FontMap> Mapped = [];

    static FontMaps() => LibraryDefaults.Ensure(typeof(FontMaps));

    /// <summary>Whether the library's maps are matched against fonts (on unless <see cref="BuiltinVariable"/> turns them off).</summary>
    public static bool Builtin { get; set; } = !ComputeResources.SwitchedOff(Environment.GetEnvironmentVariable(BuiltinVariable));

    /// <summary>Whether a map beside a font file maps it (on unless <see cref="SidecarsVariable"/> turns it off).</summary>
    public static bool Sidecars { get; set; } = !ComputeResources.SwitchedOff(Environment.GetEnvironmentVariable(SidecarsVariable));

    /// <summary>How the glyphs a mapped font lacks are drawn into it (<see cref="FontMapDrawing"/>).</summary>
    public static FontMapDrawing Drawing { get; set; } = new();

    /// <summary>The mapper that applies maps (<see cref="FontMappers.OpenType"/>).</summary>
    public static string Mapper { get; set; } = FontMappers.OpenType;

    /// <summary>Registers <paramref name="map"/> under its name: tried before the library's, replacing one of its name.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(FontMap map)
    {
        ArgumentNullException.ThrowIfNull(map);
        Registry.Register(map.Name, map, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's map <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered names, in the order they are tried.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The map registered as <paramref name="name"/>, or null.</summary>
    public static FontMap? Find(string name) => Registry.Find(name);

    /// <summary>The map registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotRegisteredException">None is.</exception>
    public static FontMap Get(string name) => Registry.Find(name)
        ?? throw new NotRegisteredException(nameof(FontMaps), name, Names, $"No font map '{name}' is registered ({string.Join(", ", Names)}); add it with FontMaps.Register(FontMap.Read(path)).");

    /// <summary>The library's map <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static FontMap? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the map <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>Each registered map as "name: summary", one per line.</summary>
    public static string Describe() => string.Join("\n", Registry.Values.Select(m => $"{m.Name}: {m.Summary}"));

    /// <summary>The map that made <paramref name="face"/>, or null for a face no map made.</summary>
    public static FontMap? MappedBy(IFontFace face)
    {
        ArgumentNullException.ThrowIfNull(face);
        return Mapped.TryGetValue(face, out var map) ? map : null;
    }

    /// <summary>The map file that belongs beside the font file <paramref name="fontPath"/>: FONT.ttf, FONT.map.json.</summary>
    public static string SidecarOf(string fontPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(fontPath);
        return Path.Combine(Path.GetDirectoryName(fontPath) ?? "", Path.GetFileNameWithoutExtension(fontPath) + SidecarSuffix);
    }

    /// <summary>
    /// <paramref name="face"/> made usable by the first registered map that fits it (the app's first, then the library's
    /// while <see cref="Builtin"/>); the face itself when none fits.
    /// </summary>
    public static IFontFace Apply(IFontFace face)
    {
        ArgumentNullException.ThrowIfNull(face);
        var maps = Registry.Values;
        if (maps.Count == 0)
        {
            return face;
        }

        IFontMapper? mapper = null;
        var drawing = Drawing;
        foreach (var map in maps)
        {
            if (!Builtin && Registry.Origin(map.Name) == Overrides.Library)
            {
                continue;
            }

            mapper ??= FontMappers.Get(Mapper);
            if (mapper.Map(face, map, detect: true, drawing) is { } mapped)
            {
                Mapped.AddOrUpdate(mapped, map);
                return mapped;
            }
        }

        return face;
    }

    /// <summary>Each of <paramref name="faces"/> through <see cref="Apply(IFontFace)"/>; the same list when no face changed.</summary>
    public static IReadOnlyList<IFontFace> Apply(IReadOnlyList<IFontFace> faces)
    {
        ArgumentNullException.ThrowIfNull(faces);
        IFontFace[]? changed = null;
        for (int i = 0; i < faces.Count; i++)
        {
            var face = Apply(faces[i]);
            if (!ReferenceEquals(face, faces[i]))
            {
                changed ??= [.. faces];
                changed[i] = face;
            }
        }

        return changed ?? faces;
    }

    /// <summary><paramref name="face"/> made usable by <paramref name="map"/>, given for it (no detection).</summary>
    /// <exception cref="InvalidDataException">None of the map's letters is in the font.</exception>
    public static IFontFace Apply(IFontFace face, FontMap map)
    {
        ArgumentNullException.ThrowIfNull(face);
        ArgumentNullException.ThrowIfNull(map);
        var mapped = FontMappers.Get(Mapper).Map(face, map, detect: false, Drawing)
            ?? throw new InvalidDataException($"{map.Source}: none of the map's {map.Letters.Count} letters is in the font {face.Family} {face.Style}.");
        Mapped.AddOrUpdate(mapped, map);
        return mapped;
    }

    /// <summary>
    /// The faces read from the font file <paramref name="path"/>, mapped: by the map beside it when there is one (and
    /// <see cref="Sidecars"/>), else as <see cref="Apply(IReadOnlyList{IFontFace})"/> matches them.
    /// </summary>
    /// <exception cref="InvalidDataException">The map beside the font is not valid, or fits none of its faces.</exception>
    public static IReadOnlyList<IFontFace> ForFile(string path, IReadOnlyList<IFontFace> faces)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(faces);
        string sidecar = SidecarOf(path);
        if (!Sidecars || !File.Exists(sidecar))
        {
            return Apply(faces);
        }

        var map = FontMap.Read(sidecar);
        return [.. faces.Select(f => Apply(f, map))];
    }
}
