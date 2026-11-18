// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Collections.Frozen;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Text;

namespace Idrak.Imaging;

/// <summary>
/// A font face made from glyphs (a person's handwriting traced from images, plan 38): characters, their outlines and
/// advances, the vertical metrics and the names the caller gave, made by a <see cref="MadeFontFaceBuilder"/> and not
/// changed after, so it is safe to read from several threads at once. It is used like any face read from a file (drawn,
/// shaped through <see cref="TextShapers"/>, mapped by a font map) and written to a font file through
/// <see cref="FontWriters"/>. Its only table is <c>name</c>, made from the names given (nothing else is made up); it has no
/// layout tables of its own (the forms and alternates a sheet labelled are given their GSUB, GDEF and GPOS by a font map).
/// </summary>
public sealed class MadeFontFace : IFontFace
{
    /// <summary>What <see cref="Format"/> says: the face was made, not read by a font format.</summary>
    public const string FormatName = "made";

    private readonly GlyphOutline[] _outlines;
    private readonly int[] _advances;
    private readonly FrozenDictionary<int, int> _glyphs;
    private readonly byte[] _nameTable;

    internal MadeFontFace(FontMetrics metrics, IReadOnlyDictionary<FontNameKind, string> names, GlyphOutline[] outlines, int[] advances, Dictionary<int, int> glyphs, byte[] nameTable)
    {
        Metrics = metrics;
        Names = names;
        _outlines = outlines;
        _advances = advances;
        _glyphs = glyphs.ToFrozenDictionary();
        int[] codes = [.. glyphs.Keys];
        Array.Sort(codes);
        Characters = Array.AsReadOnly(codes);
        _nameTable = nameTable;
        Family = names[FontNameKind.Family];
        Style = names.TryGetValue(FontNameKind.Style, out string? style) ? style : "Regular";
    }

    /// <summary>The family name given.</summary>
    public string Family { get; }

    /// <summary>The style name given, or "Regular" when none was.</summary>
    public string Style { get; }

    /// <summary><see cref="FormatName"/>: no font format read the face.</summary>
    public string Format => FormatName;

    /// <summary>The vertical metrics given.</summary>
    public FontMetrics Metrics { get; }

    /// <summary>The number of glyphs: the missing-character glyph (0) and every glyph added.</summary>
    public int GlyphCount => _advances.Length;

    /// <summary>The names given, by what they hold, in the order of their kinds.</summary>
    public IReadOnlyDictionary<FontNameKind, string> Names { get; }

    /// <summary>The characters the face answers (Unicode code points), in increasing order.</summary>
    public IReadOnlyList<int> Characters { get; }

    /// <summary>The glyph of <paramref name="codePoint"/>, or 0 when the face has none.</summary>
    public int Glyph(int codePoint) => _glyphs.TryGetValue(codePoint, out int glyph) ? glyph : 0;

    /// <summary>The advance width of <paramref name="glyph"/>, in font units (0 for a glyph the face does not have).</summary>
    public int Advance(int glyph) => (uint)glyph < (uint)_advances.Length ? _advances[glyph] : 0;

    /// <summary>The outline of <paramref name="glyph"/> (<see cref="GlyphOutline.Empty"/> for one with none, or one the face does not have).</summary>
    public GlyphOutline Outline(int glyph) => (uint)glyph < (uint)_outlines.Length ? _outlines[glyph] : GlyphOutline.Empty;

    /// <summary>The <c>name</c> table made from the names given (Windows, Unicode, English); empty for every other tag.</summary>
    public ReadOnlyMemory<byte> Table(string tag) => tag == "name" ? _nameTable : ReadOnlyMemory<byte>.Empty;

    /// <summary>"Family Style (n glyphs, m characters, made)".</summary>
    public override string ToString() => $"{Family} {Style} ({GlyphCount} glyphs, {Characters.Count} characters, made)";
}

/// <summary>
/// Makes a <see cref="MadeFontFace"/> from glyphs: each a character's (or, with no character, a form's or an alternate's,
/// which a font map reaches by its id), with its outline in font units (y up) and its advance; the vertical metrics and
/// the names the caller gives (a family at least; nothing is made up). Glyph 0, the missing-character glyph, is given with
/// <see cref="Missing"/>, else it is empty with an advance of half an em. Every input is checked as it is given, and a
/// bad one refused naming what is wrong; the outlines are copied, so the caller's arrays stay the caller's. Not safe to
/// use from several threads at once; <see cref="Build"/> may be called again after more glyphs are added (each face keeps
/// what it was built with).
/// </summary>
public sealed class MadeFontFaceBuilder
{
    /// <summary>The most glyphs a face holds (font files number them in 16 bits).</summary>
    public const int MostGlyphs = 65_535;

    /// <summary>The farthest a point may lie from the origin, in font units, on either axis (font files hold 16-bit coordinates).</summary>
    public const int MostCoordinate = 32_767;

    /// <summary>The largest advance, in font units (font files hold it in 16 bits).</summary>
    public const int MostAdvance = 65_535;

    private readonly List<GlyphOutline> _outlines = [GlyphOutline.Empty];
    private readonly List<int> _advances;
    private readonly Dictionary<int, int> _glyphs = [];
    private readonly SortedDictionary<FontNameKind, string> _names = [];

    /// <summary>A builder of a face of the family <paramref name="family"/> with <paramref name="metrics"/>.</summary>
    /// <param name="family">The family name (the name the caller gives the hand).</param>
    /// <param name="metrics">
    /// The vertical metrics: units per em from 16 to 16,384, the ascender at or above the baseline and the descender at or
    /// below it (not both on it), a line gap of 0 or more, each within ±32,767.
    /// </param>
    /// <exception cref="ArgumentException">No family name, or metrics a font cannot hold (the message says which).</exception>
    public MadeFontFaceBuilder(string family, FontMetrics metrics)
    {
        if (string.IsNullOrWhiteSpace(family))
        {
            throw new ArgumentException("A made face needs a family name: the one the caller gives the hand (nothing is made up).", nameof(family));
        }

        if (metrics.UnitsPerEm is < 16 or > 16_384)
        {
            throw new ArgumentException($"Units per em is from 16 to 16,384 (what a font file holds), not {metrics.UnitsPerEm}.", nameof(metrics));
        }

        if (metrics.Descender > 0 || metrics.Ascender < 0 || metrics.Ascender <= metrics.Descender)
        {
            throw new ArgumentException($"The ascender is at or above the baseline and the descender at or below it, not both on it: ascender {metrics.Ascender}, descender {metrics.Descender}.",
                nameof(metrics));
        }

        if (metrics.LineGap < 0 || metrics.Ascender > MostCoordinate || metrics.Descender < -MostCoordinate || metrics.LineGap > MostCoordinate)
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"The line gap is 0 or more and every metric within ±{MostCoordinate:N0} units (what a font file holds): {metrics}."),
                nameof(metrics));
        }

        Metrics = metrics;
        _advances = [metrics.UnitsPerEm / 2];
        _names[FontNameKind.Family] = family;
    }

    /// <summary>The vertical metrics given.</summary>
    public FontMetrics Metrics { get; }

    /// <summary>The glyphs so far: the missing-character glyph (0) and every glyph added.</summary>
    public int GlyphCount => _advances.Count;

    /// <summary>The characters given so far.</summary>
    public int CharacterCount => _glyphs.Count;

    /// <summary>Sets the name <paramref name="kind"/> (the family given to the constructor included) to <paramref name="value"/>.</summary>
    /// <exception cref="ArgumentException">A kind of name a font does not hold, or a value with no text.</exception>
    public MadeFontFaceBuilder Name(FontNameKind kind, string value)
    {
        if (!Enum.IsDefined(kind))
        {
            throw new ArgumentException($"{(int)kind} is not a kind of font name ({string.Join(", ", Enum.GetNames<FontNameKind>())}).", nameof(kind));
        }

        if (string.IsNullOrWhiteSpace(value))
        {
            throw new ArgumentException($"The name {kind} needs text: the one the caller gives (nothing is made up).", nameof(value));
        }

        _names[kind] = value;
        return this;
    }

    /// <summary>Sets glyph 0, the missing-character glyph, to <paramref name="outline"/> and <paramref name="advance"/>.</summary>
    /// <exception cref="ArgumentException">An outline or advance a font cannot hold (see <see cref="Add(GlyphOutline, int)"/>).</exception>
    public MadeFontFaceBuilder Missing(GlyphOutline outline, int advance)
    {
        var copy = Checked(outline, "The missing glyph");
        CheckAdvance(advance, "The missing glyph");
        _outlines[0] = copy;
        _advances[0] = advance;
        return this;
    }

    /// <summary>
    /// Adds a glyph with no character (a joining form, an alternate fill, a ligature), which a font map reaches by its id;
    /// returns its id.
    /// </summary>
    /// <param name="outline">Its outline, in font units (y up): every contour starts with a move, every point finite and within ±<see cref="MostCoordinate"/>.</param>
    /// <param name="advance">Its advance width, from 0 to <see cref="MostAdvance"/> font units.</param>
    /// <exception cref="ArgumentException">An outline or advance a font cannot hold, or the face is full (<see cref="MostGlyphs"/>).</exception>
    public int Add(GlyphOutline outline, int advance)
    {
        if (_advances.Count >= MostGlyphs)
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"A face holds at most {MostGlyphs:N0} glyphs (font files number them in 16 bits)."), nameof(outline));
        }

        string what = $"Glyph {_advances.Count}";
        var copy = Checked(outline, what);
        CheckAdvance(advance, what);
        _outlines.Add(copy);
        _advances.Add(advance);
        return _advances.Count - 1;
    }

    /// <summary>Adds the glyph of the character <paramref name="codePoint"/>; returns its id.</summary>
    /// <param name="codePoint">The character (a Unicode scalar value), not given before.</param>
    /// <param name="outline">Its outline (see <see cref="Add(GlyphOutline, int)"/>).</param>
    /// <param name="advance">Its advance width, from 0 to <see cref="MostAdvance"/> font units.</param>
    /// <exception cref="ArgumentException">Not a character, a character given before, or an outline or advance a font cannot hold.</exception>
    public int Add(int codePoint, GlyphOutline outline, int advance)
    {
        CheckCharacter(codePoint);
        int glyph = Add(outline, advance);
        _glyphs[codePoint] = glyph;
        return glyph;
    }

    /// <summary>Gives the character <paramref name="codePoint"/> the glyph <paramref name="glyph"/>, added before (several characters may share a glyph).</summary>
    /// <exception cref="ArgumentException">Not a character, a character given before, or a glyph not added yet (or the missing glyph, 0).</exception>
    public MadeFontFaceBuilder Map(int codePoint, int glyph)
    {
        CheckCharacter(codePoint);
        if (glyph <= 0 || glyph >= _advances.Count)
        {
            throw new ArgumentException(glyph == 0
                ? $"{Code(codePoint)} given glyph 0, the missing-character glyph: leave the character out instead."
                : $"Glyph {glyph} is not added yet (the glyphs are 1 to {_advances.Count - 1}).", nameof(glyph));
        }

        _glyphs[codePoint] = glyph;
        return this;
    }

    /// <summary>The face of what is given so far; the builder can go on (each face keeps what it was built with).</summary>
    /// <exception cref="ArgumentException">The names take more than a name table holds (65,535 bytes).</exception>
    public MadeFontFace Build() =>
        new(Metrics, new ReadOnlyDictionary<FontNameKind, string>(new SortedDictionary<FontNameKind, string>(_names)), [.. _outlines], [.. _advances],
            new Dictionary<int, int>(_glyphs), NameTable(_names));

    // A name table of format 0 holding the names given, each a Windows Unicode (UTF-16, big-endian) English record, sorted
    // by name id as the format asks.
    private static byte[] NameTable(SortedDictionary<FontNameKind, string> names)
    {
        var strings = names.Select(n => (Id: (int)n.Key, Bytes: Encoding.BigEndianUnicode.GetBytes(n.Value))).OrderBy(n => n.Id).ToList();
        int header = 6 + 12 * strings.Count, storage = strings.Sum(s => s.Bytes.Length);
        if (header + storage > ushort.MaxValue)
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"The names take {header + storage:N0} bytes; a name table holds at most {ushort.MaxValue:N0}."), nameof(names));
        }

        var table = new byte[header + storage];
        var span = table.AsSpan();
        BinaryPrimitives.WriteUInt16BigEndian(span, 0);                                 // format 0
        BinaryPrimitives.WriteUInt16BigEndian(span[2..], (ushort)strings.Count);
        BinaryPrimitives.WriteUInt16BigEndian(span[4..], (ushort)header);              // where the strings start
        int offset = 0;
        for (int i = 0; i < strings.Count; i++)
        {
            var record = span.Slice(6 + 12 * i, 12);
            BinaryPrimitives.WriteUInt16BigEndian(record, 3);                           // Windows
            BinaryPrimitives.WriteUInt16BigEndian(record[2..], 1);                      // Unicode
            BinaryPrimitives.WriteUInt16BigEndian(record[4..], 0x0409);                 // English
            BinaryPrimitives.WriteUInt16BigEndian(record[6..], (ushort)strings[i].Id);
            BinaryPrimitives.WriteUInt16BigEndian(record[8..], (ushort)strings[i].Bytes.Length);
            BinaryPrimitives.WriteUInt16BigEndian(record[10..], (ushort)offset);
            strings[i].Bytes.CopyTo(span[(header + offset)..]);
            offset += strings[i].Bytes.Length;
        }

        return table;
    }

    private void CheckCharacter(int codePoint)
    {
        if (!Rune.IsValid(codePoint))
        {
            throw new ArgumentException($"{Code(codePoint)} is not a character (a surrogate, negative, or past U+10FFFF).", nameof(codePoint));
        }

        if (_glyphs.TryGetValue(codePoint, out int had))
        {
            throw new ArgumentException($"{Code(codePoint)} is given twice: glyph {had} has it already.", nameof(codePoint));
        }
    }

    private static void CheckAdvance(int advance, string what)
    {
        if (advance is < 0 or > MostAdvance)
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"{what}'s advance is from 0 to {MostAdvance:N0} font units, not {advance}."), nameof(advance));
        }
    }

    // A copy of the outline, its contours checked: each starts with a move, every step a known one, every point finite and
    // within what a font file holds.
    private static GlyphOutline Checked(GlyphOutline outline, string what)
    {
        ArgumentNullException.ThrowIfNull(outline);
        if (outline.IsEmpty)
        {
            return GlyphOutline.Empty;
        }

        var verbs = outline.Verbs.Span;
        bool open = false;
        for (int k = 0; k < verbs.Length; k++)
        {
            var verb = verbs[k];
            if (verb is not (OutlineVerb.Move or OutlineVerb.Line or OutlineVerb.Quadratic or OutlineVerb.Cubic or OutlineVerb.Close))
            {
                throw new ArgumentException($"{what}'s outline has step {k + 1} of no kind there is ({(int)verb}).", nameof(outline));
            }

            if (verb == OutlineVerb.Move)
            {
                open = true;
            }
            else if (!open)
            {
                throw new ArgumentException($"{what}'s outline draws step {k + 1} ({verb}) before a move starts its contour.", nameof(outline));
            }
            else if (verb == OutlineVerb.Close)
            {
                open = false;
            }
        }

        var points = outline.Points.Span;
        for (int i = 0; i < points.Length; i += 2)
        {
            float x = points[i], y = points[i + 1];
            if (!(MathF.Abs(x) <= MostCoordinate && MathF.Abs(y) <= MostCoordinate))
            {
                throw new ArgumentException(string.Create(CultureInfo.InvariantCulture,
                    $"{what}'s outline has the point ({x}, {y}): a font file holds finite coordinates within ±{MostCoordinate:N0} units."), nameof(outline));
            }
        }

        return new GlyphOutline(verbs.ToArray(), points.ToArray());
    }

    private static string Code(int codePoint) => codePoint is >= 0 and <= 0x10FFFF
        ? string.Create(CultureInfo.InvariantCulture, $"U+{codePoint:X4}")
        : codePoint.ToString(CultureInfo.InvariantCulture);
}
