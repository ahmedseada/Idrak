// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Data;

/// <summary>
/// A font maker's settings, read once from its options (<see cref="IFontMaker.Read"/>) and used for every font made with
/// them: the options as given, and whatever a maker parses them into (a maker's own settings derive from this class).
/// </summary>
public class FontMakerSettings
{
    /// <summary>Settings of <paramref name="options"/> (none when null).</summary>
    public FontMakerSettings(IReadOnlyDictionary<string, string>? options = null) => Options = options ?? PageOptions.None;

    /// <summary>No options: a maker's defaults.</summary>
    public static FontMakerSettings None { get; } = new();

    /// <summary>The options as given, by key.</summary>
    public IReadOnlyDictionary<string, string> Options { get; }
}

/// <summary>
/// What a font maker is asked: the glyphs a glyph source found (each labelled with its text and joining form, its ink,
/// box, baseline and size), the names the caller gives the hand (a family at least; nothing is made up), the tracer that
/// turns each glyph's ink into its outline, and the maker's settings.
/// </summary>
/// <param name="Glyphs">The glyphs found (<see cref="IGlyphSource.Find"/>), and what their source left out.</param>
/// <param name="Family">The family name the caller gives the hand.</param>
/// <param name="Settings">The maker's settings (<see cref="IFontMaker.Read"/>).</param>
public sealed record FontMakerRequest(FoundGlyphs Glyphs, string Family, FontMakerSettings Settings)
{
    /// <summary>The glyphs found, and what their source left out.</summary>
    public FoundGlyphs Glyphs { get; init; } = Glyphs ?? throw new ArgumentNullException(nameof(Glyphs));

    /// <summary>The family name the caller gives the hand.</summary>
    public string Family { get; init; } = !string.IsNullOrWhiteSpace(Family) ? Family
        : throw new ArgumentException("A made font needs a family name: the one the caller gives the hand (nothing is made up).", nameof(Family));

    /// <summary>The maker's settings.</summary>
    public FontMakerSettings Settings { get; init; } = Settings ?? throw new ArgumentNullException(nameof(Settings));

    /// <summary>The style name the caller gives, or null ("Regular").</summary>
    public string? Style { get; init; }

    /// <summary>More name strings the caller gives (a designer, a licence), or null.</summary>
    public IReadOnlyDictionary<FontNameKind, string>? Names { get; init; }

    /// <summary>The outline tracer each glyph's ink is traced with, by name (<see cref="OutlineTracers"/>; <see cref="OutlineTracers.Contour"/> unless set).</summary>
    public string Tracer { get; init; } = OutlineTracers.Contour;

    /// <summary>The tracer's options (its own keys), or null for its defaults.</summary>
    public IReadOnlyDictionary<string, string>? TracerOptions { get; init; }
}

/// <summary>A glyph of a made font and the labelled glyph it was made from: its id, its text and form, and which fill of them it is.</summary>
/// <param name="Glyph">Its glyph id in the made face.</param>
/// <param name="Text">The character (or the letters of a ligature) it was labelled with.</param>
/// <param name="Form">The joining form it was labelled with, or null for none.</param>
/// <param name="Fill">Which fill of that text and form it is: 0 the first (the one shaping gives), 1 and on its alternates, in the order they were found.</param>
public readonly record struct MadeGlyph(int Glyph, string Text, string? Form, int Fill);

/// <summary>
/// The metrics a font maker measured from the glyphs' ink, in font units of <see cref="UnitsPerEm"/> (y up from the
/// baseline), with how each was measured (<see cref="Measured"/>).
/// </summary>
/// <param name="UnitsPerEm">The face's units per em.</param>
/// <param name="Ascender">The height above the baseline the line takes.</param>
/// <param name="Descender">The depth below it (0 or less).</param>
/// <param name="LineGap">The gap added between lines (0 or more).</param>
/// <param name="XHeight">The short glyphs' typical height.</param>
/// <param name="CapHeight">The tall glyphs' typical height.</param>
/// <param name="Pen">The pen's typical width (the strokes' median thickness).</param>
/// <param name="SideBearing">The space left on each side of a glyph that does not join its neighbours.</param>
/// <param name="Space">The space character's advance.</param>
/// <param name="Measured">How the metrics were measured, in words (the measure stated).</param>
public sealed record MadeFontMetrics(int UnitsPerEm, int Ascender, int Descender, int LineGap, int XHeight, int CapHeight, int Pen, int SideBearing, int Space, string Measured)
{
    /// <summary>"units per em, ascender, descender, ...".</summary>
    public override string ToString() => string.Create(CultureInfo.InvariantCulture,
        $"{UnitsPerEm} units per em: ascender {Ascender}, descender {Descender}, line gap {LineGap}, x-height {XHeight}, cap height {CapHeight}, pen {Pen}, side bearing {SideBearing}, space {Space}");
}

/// <summary>
/// A font a font maker made: the face (its outlines, advances, characters and metrics, and the layout tables that shape
/// its forms, ligatures, marks and alternates), used like any face and written through <see cref="FontWriters"/>; the
/// font map it made as data and applied (<c>Apply</c>); the metrics it measured;
/// each
/// glyph made from a labelled glyph; and what it left out, with why (its source's left-out characters among them).
/// </summary>
/// <param name="Face">The made face.</param>
/// <param name="Map">The font map it made and applied.</param>
/// <param name="Metrics">The metrics it measured.</param>
/// <param name="Glyphs">Each glyph made from a labelled glyph, in the order they were found.</param>
/// <param name="LeftOut">The characters and fills not made into the font, and why.</param>
public sealed record MadeFont(IFontFace Face, FontMap Map, MadeFontMetrics Metrics, IReadOnlyList<MadeGlyph> Glyphs, IReadOnlyList<GlyphLeftOut> LeftOut)
{
    /// <summary>The made face.</summary>
    public IFontFace Face { get; init; } = Face ?? throw new ArgumentNullException(nameof(Face));

    /// <summary>The font map it made and applied.</summary>
    public FontMap Map { get; init; } = Map ?? throw new ArgumentNullException(nameof(Map));

    /// <summary>The metrics it measured.</summary>
    public MadeFontMetrics Metrics { get; init; } = Metrics ?? throw new ArgumentNullException(nameof(Metrics));

    /// <summary>Each glyph made from a labelled glyph, in the order they were found.</summary>
    public IReadOnlyList<MadeGlyph> Glyphs { get; init; } = Glyphs ?? throw new ArgumentNullException(nameof(Glyphs));

    /// <summary>The characters and fills not made into the font, and why.</summary>
    public IReadOnlyList<GlyphLeftOut> LeftOut { get; init; } = LeftOut ?? throw new ArgumentNullException(nameof(LeftOut));
}

/// <summary>
/// The stage from labelled glyphs to a font: each glyph's ink traced into its outline (an <see cref="IOutlineTracer"/>), the face's metrics measured from the ink and the baselines and sizes the glyphs were
/// written at, each glyph's advance and side bearings from its ink, and the forms, ligatures, marks and alternates the
/// glyphs were labelled with made into the layout tables that shape them (through a font map, as data). A character's
/// several fills are kept, the first as its glyph and the others as its alternates (<see cref="MadeGlyph.Fill"/>), or
/// reported left out with why; every glyph given is made or reported. Which characters and forms there are is data (the
/// labels): nothing here names a script. Its settings are read once (<see cref="Read"/>); the same glyphs and settings
/// make the same face, from any thread. The face it makes is written by a font writer and read back by a font format
/// like any font.
/// </summary>
public interface IFontMaker
{
    /// <summary>The maker's settings from <paramref name="options"/> (its own keys, <c>Keys</c>; defaults for those not given).</summary>
    /// <exception cref="ArgumentException">A key it does not take (the message names its keys), or a bad
    /// value.</exception>
    FontMakerSettings Read(IReadOnlyDictionary<string, string> options);

    /// <summary>The font made from <paramref name="request"/>'s glyphs.</summary>
    /// <exception cref="ArgumentException">No glyph to make a font from, or names a font cannot hold (the message says which).</exception>
    MadeFont Make(FontMakerRequest request);
}

/// <summary>
/// The name of the library's font maker (Idrak.Imaging): "traced" (<see cref="Traced"/>: each glyph
/// traced, the metrics and advances measured from the ink, the forms, ligatures, marks and alternates made into a font
/// map's layout tables, the digits and marks the glyphs lack drawn). Picking one by its name is not on the new design
/// yet.
/// </summary>
public static class FontMakers
{
    /// <summary>The library's maker (Idrak.Imaging): glyphs traced, measured and shaped through a font map.</summary>
    public const string Traced = "traced";
}
