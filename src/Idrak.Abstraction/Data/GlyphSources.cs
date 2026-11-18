// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Data;

/// <summary>
/// One glyph a glyph source found in an image: what it is (<see cref="Text"/>, in its joining <see cref="Form"/>), its ink
/// (<see cref="Ink"/>, coverage of its <see cref="Box"/>) and where it sat: its box in the image, the cell or ink piece it
/// was cut from (<see cref="Region"/>), the baseline it was written on and the size it was written at. Which characters
/// and forms there are is data (the image's known text, a sheet's character list): nothing here names a script.
/// </summary>
public sealed record LabelledGlyph
{
    /// <summary>A glyph found in an image.</summary>
    /// <param name="text">The character it is (a whole character: a letter with its marks, or the letters of a ligature).</param>
    /// <param name="form">Its joining form as data (for example "isol", "init", "medi" or "fina"), or null for none.</param>
    /// <param name="ink">Its ink: coverage of <paramref name="box"/>, the same size.</param>
    /// <param name="box">Its box in the image, in pixels.</param>
    /// <param name="region">The cell or ink piece it was cut from, in pixels.</param>
    /// <param name="baseline">The row of the baseline it sat on, in pixels from the image's top (y down).</param>
    /// <param name="size">The size it was written at, in pixels per em (more than 0).</param>
    /// <exception cref="ArgumentException">No text, a form that is empty or holds spaces, ink not its box's size, or a baseline or size that is not a finite number (a size not more than 0).</exception>
    public LabelledGlyph(string text, string? form, GlyphCoverage ink, PixelBox box, PixelBox region, float baseline, float size)
    {
        ArgumentException.ThrowIfNullOrEmpty(text);
        ArgumentNullException.ThrowIfNull(ink);
        if (form is not null && !IsName(form))
        {
            throw new ArgumentException($"A joining form is a name (such as isol, init, medi or fina) or none (null), not '{form}'.", nameof(form));
        }

        if (ink.Width != box.Width || ink.Height != box.Height)
        {
            throw new ArgumentException($"The ink of '{text}' is {ink.Width} x {ink.Height} pixels; its box {box} is {box.Width} x {box.Height}.", nameof(ink));
        }

        if (!float.IsFinite(baseline))
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"The baseline of '{text}' is a row of the image, not {baseline}."), nameof(baseline));
        }

        if (!(size > 0f && float.IsFinite(size)))
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"The size of '{text}' is pixels per em, more than 0, not {size}."), nameof(size));
        }

        (Text, Form, Ink, Box, Region, Baseline, Size) = (text, form, ink, box, region, baseline, size);
    }

    /// <summary>The character it is (a whole character: a letter with its marks, or the letters of a ligature).</summary>
    public string Text { get; }

    /// <summary>Its joining form as data ("isol", "init", "medi", "fina" or any a sheet names), or null for none.</summary>
    public string? Form { get; }

    /// <summary>Its ink: coverage of <see cref="Box"/>.</summary>
    public GlyphCoverage Ink { get; }

    /// <summary>Its box in the image, in pixels.</summary>
    public PixelBox Box { get; }

    /// <summary>The cell or ink piece it was cut from, in pixels: its box lies inside.</summary>
    public PixelBox Region { get; }

    /// <summary>The row of the baseline it sat on, in pixels from the image's top (y down).</summary>
    public float Baseline { get; }

    /// <summary>The size it was written at, in pixels per em.</summary>
    public float Size { get; }

    /// <summary>
    /// Where <see cref="Ink"/> sits against the font units of a face of <paramref name="unitsPerEm"/> units per em: a
    /// tracer's placement (<see cref="IOutlineTracer.Trace"/>) with the ink's left edge at x = 0 and the baseline at y = 0.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="unitsPerEm"/> is not more than 0.</exception>
    public OutlinePlacement Placement(int unitsPerEm)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(unitsPerEm);
        return new OutlinePlacement(Size / unitsPerEm, 0f, Baseline - Box.Y);
    }

    /// <summary>"'text' (form) at box in region, baseline, size".</summary>
    public override string ToString() =>
        string.Create(CultureInfo.InvariantCulture, $"'{Text}'{(Form is null ? "" : $" ({Form})")} at {Box} in {Region}, baseline {Baseline}, {Size} px per em");

    // A form's name: not empty, no spaces or control characters.
    private static bool IsName(string form)
    {
        foreach (char c in form)
        {
            if (char.IsWhiteSpace(c) || char.IsControl(c))
            {
                return false;
            }
        }

        return form.Length > 0;
    }
}

/// <summary>A character of the image's known text (or a sheet's list) that a glyph source gave no glyph for, and why.</summary>
/// <param name="Text">The character.</param>
/// <param name="Form">The joining form it was asked in, or null.</param>
/// <param name="Reason">Why it was left out ("no ink in its cell", "two ink pieces for one character").</param>
public readonly record struct GlyphLeftOut(string Text, string? Form, string Reason);

/// <summary>What a glyph source found in an image: its glyphs, and the characters it was asked for and left out, with why.</summary>
/// <param name="Glyphs">The glyphs found, each labelled once.</param>
/// <param name="LeftOut">The characters of the known text (or a sheet's list) it gave no glyph for.</param>
public sealed record FoundGlyphs(IReadOnlyList<LabelledGlyph> Glyphs, IReadOnlyList<GlyphLeftOut> LeftOut)
{
    /// <summary>Nothing found and nothing left out.</summary>
    public static FoundGlyphs None { get; } = new([], []);

    /// <summary>The glyphs found, each labelled once.</summary>
    public IReadOnlyList<LabelledGlyph> Glyphs { get; init; } = Glyphs ?? throw new ArgumentNullException(nameof(Glyphs));

    /// <summary>The characters of the known text (or a sheet's list) it gave no glyph for.</summary>
    public IReadOnlyList<GlyphLeftOut> LeftOut { get; init; } = LeftOut ?? throw new ArgumentNullException(nameof(LeftOut));
}

/// <summary>
/// A glyph source's settings, read once from its options (<see cref="IGlyphSource.Read"/>) and used for every image read
/// with them: the options as given, and whatever a source parses them into (a source's own settings derive from this
/// class).
/// </summary>
public class GlyphSourceSettings
{
    /// <summary>Settings of <paramref name="options"/> (none when null).</summary>
    public GlyphSourceSettings(IReadOnlyDictionary<string, string>? options = null) => Options = options ?? PageOptions.None;

    /// <summary>No options: a source's defaults.</summary>
    public static GlyphSourceSettings None { get; } = new();

    /// <summary>The options as given, by key.</summary>
    public IReadOnlyDictionary<string, string> Options { get; }
}

/// <summary>What a glyph source is asked: one image, its known text where it has one, and the settings read from its options.</summary>
/// <param name="Image">The image (a scanned or photographed page, a filled sheet, a crop of a table).</param>
/// <param name="Text">The image's known text (its transcription), or null when the source knows its labels otherwise (a sheet's own list).</param>
/// <param name="Settings">The source's settings (<see cref="IGlyphSource.Read"/>).</param>
public sealed record GlyphSourceRequest(ImageData Image, string? Text, GlyphSourceSettings Settings)
{
    /// <summary>The image (a scanned or photographed page, a filled sheet, a crop of a table).</summary>
    public ImageData Image { get; init; } = Image ?? throw new ArgumentNullException(nameof(Image));

    /// <summary>The source's settings (<see cref="IGlyphSource.Read"/>).</summary>
    public GlyphSourceSettings Settings { get; init; } = Settings ?? throw new ArgumentNullException(nameof(Settings));
}

/// <summary>
/// The stage from an image to labelled glyphs: it finds each written character's ink in the image (a sheet's cells, the
/// ink pieces of a table's cells, the letters of a word) and labels it with the character it is, in its joining form, from
/// the image's known text or the source's own list; it gives the ink as coverage with its box, the region it was cut from,
/// its baseline and size (<see cref="LabelledGlyph"/>), and reports each character it was asked for and gave no glyph
/// for (<see cref="GlyphLeftOut"/>). Every glyph is labelled once and lies inside its region. Its settings (ink level,
/// specks, the character list) are read once (<see cref="Read"/>). The same image, text and settings give the same glyphs,
/// from any thread. Register one with <see cref="GlyphSources.Register"/>.
/// </summary>
public interface IGlyphSource : IConfigurableKind
{
    /// <summary>The source's settings from <paramref name="options"/> (its own keys, <see cref="IConfigurableKind.Keys"/>; defaults for those not given).</summary>
    /// <exception cref="ArgumentException">A key it does not take (the message names its keys), or a bad value.</exception>
    GlyphSourceSettings Read(IReadOnlyDictionary<string, string> options);

    /// <summary>The glyphs found in <paramref name="request"/>'s image, and the characters it was asked for and left out.</summary>
    FoundGlyphs Find(GlyphSourceRequest request);
}

/// <summary>
/// The glyph sources, by name (ignoring case). None is built in yet; a package registers its own with
/// <see cref="Register"/>, and one registered under a library name shadows the library's, which <see cref="Unregister"/>
/// brings back, guarded by its <see cref="SetPolicy">policy</see>.
/// </summary>
public static class GlyphSources
{
    private static readonly KindTable<IGlyphSource> Table = new(nameof(GlyphSources), "Glyph source", "add it with GlyphSources.Register, or reference a package that registers it",
        (slot, app, library) => new GuardedSource(slot, app, library), StringComparer.OrdinalIgnoreCase);

    static GlyphSources() => LibraryDefaults.Ensure(typeof(GlyphSources));

    /// <summary>Registers <paramref name="source"/> under its <see cref="IKind.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IGlyphSource source) => Table.Register(source, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's source <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Names;

    /// <summary>The source registered as <paramref name="name"/>, or null.</summary>
    public static IGlyphSource? Find(ReadOnlySpan<char> name) => Table.Find(name);

    /// <summary>The source registered as <paramref name="name"/>; an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    /// <exception cref="NotRegisteredException">None is: the message names the registered ones and what to do.</exception>
    public static IGlyphSource Get(ReadOnlySpan<char> name) => Table.Get(name);

    /// <summary>The library's source <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IGlyphSource? Default(string name) => Table.Default(name);

    /// <summary>Who registered the source <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>
    /// What happens when the app's source <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set;
    /// <see cref="SlotPolicy.FallBack"/> reads the image with the library's; <see cref="SlotPolicy.Shadow"/> lets the
    /// library's read it and compares the app's glyphs (their labels, forms and boxes) on a sample of images).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    /// <summary>Each registered source as "name: summary", one per line.</summary>
    public static string Describe() => Table.Describe();

    /// <summary>The keys of every registered source, each once, in registration order.</summary>
    public static IReadOnlyList<string> AllKeys() => [.. Table.Kinds.SelectMany(s => s.Keys).Distinct(StringComparer.OrdinalIgnoreCase)];

    // The app's settings and the library's, read from the same options (the library's null when it refuses them).
    private sealed class GuardedSettings(GlyphSourceSettings app, GlyphSourceSettings? library, IReadOnlyDictionary<string, string> options) : GlyphSourceSettings(options)
    {
        public GlyphSourceSettings App => app;

        public GlyphSourceSettings? Library => library;
    }

    // An app's source over the library's: a failed read is made by the library's, or both are compared on a sample.
    private sealed class GuardedSource(Slot slot, IGlyphSource app, IGlyphSource library) : IGlyphSource
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public IReadOnlySet<string> Keys => app.Keys;

        public GlyphSourceSettings Read(IReadOnlyDictionary<string, string> options)
        {
            var own = app.Read(options);
            GlyphSourceSettings? theirs;
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

        public FoundGlyphs Find(GlyphSourceRequest request)
        {
            ArgumentNullException.ThrowIfNull(request);
            var (own, theirs) = request.Settings is GuardedSettings g ? (g.App, g.Library) : (request.Settings, request.Settings);
            var mine = request with { Settings = own };
            if (theirs is null)
            {
                return app.Find(mine);
            }

            var forLibrary = request with { Settings = theirs };
            return slot.Call(() => app.Find(mine), () => library.Find(forLibrary), (a, b) => Comparisons.Exact(Labels(a), Labels(b)));
        }

        // The glyphs as their labels, forms and boxes, then the characters left out.
        private static string Labels(FoundGlyphs found) =>
            string.Join("; ", found.Glyphs.Select(g => $"{g.Text}/{g.Form} {g.Box}")) + " | " + string.Join("; ", found.LeftOut.Select(l => $"{l.Text}/{l.Form}"));
    }
}
