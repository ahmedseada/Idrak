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
/// font map it made as data and applied (<see cref="FontMaps.Apply(IFontFace, FontMap)"/>); the metrics it measured; each
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
/// The stage from labelled glyphs to a font: each glyph's ink traced into its outline (an <see cref="IOutlineTracer"/>),
/// the face's metrics measured from the ink and the baselines and sizes the glyphs were written at, each glyph's advance
/// and side bearings from its ink, and the forms, ligatures, marks and alternates the glyphs were labelled with made into
/// the layout tables that shape them (through a font map, as data). A character's several fills are kept, the first as
/// its glyph and the others as its alternates (<see cref="MadeGlyph.Fill"/>), or reported left out with why; every glyph
/// given is made or reported. Which characters and forms there are is data (the labels): nothing here names a script.
/// Its settings are read once (<see cref="Read"/>); the same glyphs and settings make the same face, from any thread. The
/// face it makes is written through <see cref="FontWriters"/> and read back through <see cref="FontFormats"/> like any
/// font. Register one with <see cref="FontMakers.Register"/>.
/// </summary>
public interface IFontMaker : IConfigurableKind
{
    /// <summary>The maker's settings from <paramref name="options"/> (its own keys, <see cref="IConfigurableKind.Keys"/>; defaults for those not given).</summary>
    /// <exception cref="ArgumentException">A key it does not take (the message names its keys), or a bad value.</exception>
    FontMakerSettings Read(IReadOnlyDictionary<string, string> options);

    /// <summary>The font made from <paramref name="request"/>'s glyphs.</summary>
    /// <exception cref="ArgumentException">No glyph to make a font from, or names a font cannot hold (the message says which).</exception>
    MadeFont Make(FontMakerRequest request);
}

/// <summary>
/// The font makers, by name (ignoring case). Built in (Idrak.Imaging): "traced" (<see cref="Traced"/>: each glyph traced,
/// the metrics and advances measured from the ink, the forms, ligatures, marks and alternates made into a font map's
/// layout tables, the digits and marks the glyphs lack drawn). A package registers its own with <see cref="Register"/>,
/// and one registered under a library name shadows the library's, which <see cref="Unregister"/> brings back, guarded by
/// its <see cref="SetPolicy">policy</see>.
/// </summary>
public static class FontMakers
{
    /// <summary>The library's maker (Idrak.Imaging): glyphs traced, measured and shaped through a font map.</summary>
    public const string Traced = "traced";

    private static readonly KindTable<IFontMaker> Table = new(nameof(FontMakers), "Font maker",
        "add it with FontMakers.Register, or reference a package that registers it (Idrak.Imaging registers traced)",
        (slot, app, library) => new GuardedMaker(slot, app, library), StringComparer.OrdinalIgnoreCase);

    static FontMakers() => LibraryDefaults.Ensure(typeof(FontMakers));

    /// <summary>Registers <paramref name="maker"/> under its <see cref="IKind.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IFontMaker maker) => Table.Register(maker, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's maker <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Names;

    /// <summary>The maker registered as <paramref name="name"/>, or null.</summary>
    public static IFontMaker? Find(ReadOnlySpan<char> name) => Table.Find(name);

    /// <summary>The maker registered as <paramref name="name"/>; an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    /// <exception cref="NotRegisteredException">None is: the message names the registered ones and what to do.</exception>
    public static IFontMaker Get(ReadOnlySpan<char> name) => Table.Get(name);

    /// <summary>The library's maker <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IFontMaker? Default(string name) => Table.Default(name);

    /// <summary>Who registered the maker <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>
    /// What happens when the app's maker <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set;
    /// <see cref="SlotPolicy.FallBack"/> makes the font with the library's; <see cref="SlotPolicy.Shadow"/> lets the
    /// library's make it and compares the app's (its metrics and each labelled glyph's advance) on a sample of requests).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    /// <summary>Each registered maker as "name: summary", one per line.</summary>
    public static string Describe() => Table.Describe();

    /// <summary>The keys of every registered maker, each once, in registration order.</summary>
    public static IReadOnlyList<string> AllKeys() => [.. Table.Kinds.SelectMany(m => m.Keys).Distinct(StringComparer.OrdinalIgnoreCase)];

    // The app's settings and the library's, read from the same options (the library's null when it refuses them).
    private sealed class GuardedSettings(FontMakerSettings app, FontMakerSettings? library, IReadOnlyDictionary<string, string> options) : FontMakerSettings(options)
    {
        public FontMakerSettings App => app;

        public FontMakerSettings? Library => library;
    }

    // An app's maker over the library's: a failed font is made by the library's, or both are compared on a sample.
    private sealed class GuardedMaker(Slot slot, IFontMaker app, IFontMaker library) : IFontMaker
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public IReadOnlySet<string> Keys => app.Keys;

        public FontMakerSettings Read(IReadOnlyDictionary<string, string> options)
        {
            var own = app.Read(options);
            FontMakerSettings? theirs;
            try
            {
                theirs = library.Read(PageOptions.Only(options, library.Keys));
            }
            catch (ArgumentException)
            {
                theirs = null;
            }

            return new GuardedSettings(own, theirs, options);
        }

        public MadeFont Make(FontMakerRequest request)
        {
            ArgumentNullException.ThrowIfNull(request);
            var (own, theirs) = request.Settings is GuardedSettings g ? (g.App, g.Library) : (request.Settings, request.Settings);
            var mine = request with { Settings = own };
            if (theirs is null)
            {
                return app.Make(mine);
            }

            var forLibrary = request with { Settings = theirs };
            return slot.Call(() => app.Make(mine), () => library.Make(forLibrary), (a, b) => Comparisons.Exact(Described(a), Described(b)));
        }

        // The metrics, then each labelled glyph's text, form, fill and advance (ids may differ between makers).
        private static string Described(MadeFont made) =>
            $"{made.Metrics.UnitsPerEm} {made.Face.Metrics} | "
            + string.Join("; ", made.Glyphs.Select(g => string.Create(CultureInfo.InvariantCulture, $"{g.Text}/{g.Form}/{g.Fill} {made.Face.Advance(g.Glyph)}")));
    }
}
