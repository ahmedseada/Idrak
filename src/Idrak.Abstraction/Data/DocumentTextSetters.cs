// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;

namespace Idrak.Abstraction.Data;

/// <summary>How <see cref="IDocumentTextSetter.Set"/> sets text: in which faces, at what size, how wide a line may be, in which direction.</summary>
/// <param name="Faces">The faces: the first one's glyphs, and for characters it lacks, the first of the others that has them.</param>
/// <param name="Size">The size, in points (an em).</param>
public sealed record DocumentTextSetting(IReadOnlyList<IFontFace> Faces, float Size)
{
    private readonly float _width = float.PositiveInfinity;

    /// <summary>
    /// The widest a line may be, in points: a paragraph wider than it breaks at spaces onto more lines (inside a word too
    /// long for a line); <see cref="float.PositiveInfinity"/> (the default) sets each paragraph on one line.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Not above 0.</exception>
    public float Width
    {
        get => _width;
        init => _width = value > 0f ? value : throw new ArgumentOutOfRangeException(nameof(Width), value, "A line's width is above 0 (infinity for no limit).");
    }

    /// <summary>The paragraphs' direction: right to left (true), left to right (false), or each one's own by its first strong character (null, Unicode's rule).</summary>
    public bool? RightToLeft { get; init; }
}

/// <summary>A run of glyphs of one face on a set line, and the characters it stands for.</summary>
/// <param name="Face">The face its glyphs are of.</param>
/// <param name="Size">Its size, in points.</param>
/// <param name="Glyphs">The glyphs left to right as drawn, each placed from the line's start (x 0) on its baseline (y 0, y down), in points.</param>
/// <param name="Text">The characters its glyphs stand for, in logical order (what a reader reads back).</param>
public sealed record DocumentSetRun(IFontFace Face, float Size, ReadOnlyMemory<DocumentGlyph> Glyphs, string Text);

/// <summary>One line of set text.</summary>
/// <param name="Text">The line's text in logical order (trailing spaces left out).</param>
/// <param name="Width">How far its glyphs advance, in points (trailing spaces left out).</param>
/// <param name="RightToLeft">Whether its paragraph reads right to left.</param>
/// <param name="Runs">Its runs, left to right as drawn.</param>
public sealed record DocumentSetLine(string Text, float Width, bool RightToLeft, IReadOnlyList<DocumentSetRun> Runs);

/// <summary>
/// Sets text into lines of positioned glyphs, for a writer that places text itself (a form field's appearance): each
/// paragraph through the reading order, shaped by its script's shaper in the first face that has its characters. Register
/// one with <see cref="DocumentTextSetters.Register"/>. Safe to use from several threads at once.
/// </summary>
public interface IDocumentTextSetter : IKind
{
    /// <summary>
    /// <paramref name="text"/>'s lines, top to bottom: each paragraph (a line feed ends one) on one line, or on several
    /// where it passes <see cref="DocumentTextSetting.Width"/>; an empty paragraph an empty line.
    /// </summary>
    /// <exception cref="ArgumentException">No face is given, or the size is not above 0.</exception>
    IReadOnlyList<DocumentSetLine> Set(string text, DocumentTextSetting setting);
}

/// <summary>
/// The text setters, by name (ignoring case). Built in (Idrak.Imaging): <see cref="Shaped"/>. Register another with
/// <see cref="Register"/>; one registered under a library name takes its place, which stays behind it (see
/// <see cref="SetPolicy"/>).
/// </summary>
public static class DocumentTextSetters
{
    /// <summary>
    /// "shaped": the library's setter: each paragraph through the Unicode bidirectional algorithm, cut into runs of one
    /// script and direction, each shaped by <see cref="TextShapers"/> with the first face that has its characters; lines
    /// broken at spaces (between CJK characters, and inside a word too long for a line); each line's runs in visual order.
    /// </summary>
    public const string Shaped = "shaped";

    private static readonly KindTable<IDocumentTextSetter> Table = new(nameof(DocumentTextSetters), "Document text setter",
        "add it with DocumentTextSetters.Register, or reference a package that registers one (DocumentFormOptions.TextSetter names it)",
        (slot, app, library) => new GuardedSetter(slot, app, library), StringComparer.OrdinalIgnoreCase);

    static DocumentTextSetters() => LibraryDefaults.Ensure(typeof(DocumentTextSetters));

    /// <summary>Registers <paramref name="setter"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IDocumentTextSetter setter) => Table.Register(setter, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's setter <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Names;

    /// <summary>The setter registered as <paramref name="name"/>, or null.</summary>
    public static IDocumentTextSetter? Find(ReadOnlySpan<char> name) => Table.Find(name);

    /// <summary>The setter registered as <paramref name="name"/>; an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    /// <exception cref="NotRegisteredException">None is: the message names the registered ones and what to do.</exception>
    public static IDocumentTextSetter Get(ReadOnlySpan<char> name) => Table.Get(name);

    /// <summary>The library's setter <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IDocumentTextSetter? Default(string name) => Table.Default(name);

    /// <summary>Who registered the setter <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>
    /// What happens when the app's setter <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set;
    /// <see cref="SlotPolicy.FallBack"/> sets the text with the library's), or whether it only runs beside the library's
    /// (<see cref="SlotPolicy.Shadow"/>: the library's lines are used, the two compared: text, glyphs and widths).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    /// <summary>Each registered setter as "name: summary", one per line.</summary>
    public static string Describe() => Table.Describe();

    // Two settings of the same text against each other: each line's text, direction and width (to a thousandth of a
    // point), and its runs' glyph ids and places.
    internal static string? CompareLines(IReadOnlyList<DocumentSetLine> expected, IReadOnlyList<DocumentSetLine> actual) =>
        Comparisons.Difference(Describe(expected), Describe(actual));

    private static string Describe(IReadOnlyList<DocumentSetLine> lines)
    {
        var text = new StringBuilder();
        foreach (var line in lines)
        {
            text.Append(line.Text).Append(line.RightToLeft ? " rtl " : " ltr ").Append(line.Width.ToString("F3", CultureInfo.InvariantCulture)).Append(':');
            foreach (var run in line.Runs)
            {
                foreach (var g in run.Glyphs.Span)
                {
                    text.Append(' ').Append(g.Glyph.ToString(CultureInfo.InvariantCulture)).Append('@').Append(g.X.ToString("F3", CultureInfo.InvariantCulture));
                }
            }

            text.Append('\n');
        }

        return text.ToString();
    }

    // An app's setter over the library's: each call falls back on its own, or is compared with the library's lines.
    private sealed class GuardedSetter(Slot slot, IDocumentTextSetter app, IDocumentTextSetter library) : IDocumentTextSetter
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public IReadOnlyList<DocumentSetLine> Set(string text, DocumentTextSetting setting) =>
            slot.Call(() => app.Set(text, setting), () => library.Set(text, setting), (a, b) => CompareLines(a, b));
    }
}
