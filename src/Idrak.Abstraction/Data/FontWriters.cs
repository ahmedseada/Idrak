// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Data;

/// <summary>A font's name strings, by what they hold (the font file formats number them this way).</summary>
public enum FontNameKind
{
    /// <summary>The copyright notice.</summary>
    Copyright = 0,

    /// <summary>The family ("DejaVu Sans").</summary>
    Family = 1,

    /// <summary>The style within the family ("Bold").</summary>
    Style = 2,

    /// <summary>A unique identifier of the font.</summary>
    UniqueId = 3,

    /// <summary>The full name ("DejaVu Sans Bold").</summary>
    FullName = 4,

    /// <summary>The version ("Version 2.37").</summary>
    Version = 5,

    /// <summary>The name without spaces programs refer to it by.</summary>
    PostScriptName = 6,

    /// <summary>The trademark notice.</summary>
    Trademark = 7,

    /// <summary>Who made the font.</summary>
    Manufacturer = 8,

    /// <summary>Who designed it.</summary>
    Designer = 9,

    /// <summary>A description.</summary>
    Description = 10,

    /// <summary>The maker's web address.</summary>
    VendorUrl = 11,

    /// <summary>The designer's web address.</summary>
    DesignerUrl = 12,

    /// <summary>The licence.</summary>
    License = 13,

    /// <summary>The licence's web address.</summary>
    LicenseUrl = 14,

    /// <summary>The family a large family groups its styles under.</summary>
    TypographicFamily = 16,

    /// <summary>The style within that family.</summary>
    TypographicStyle = 17,

    /// <summary>A sample of text in the font.</summary>
    SampleText = 19,
}

/// <summary>
/// How a face is written as a font file: whole or a subset, with or without its layout tables, which name strings, and how
/// curves are converted when the format takes another kind. <see cref="CurveToleranceVariable"/> sets the tolerance.
/// </summary>
public sealed record FontWriteOptions
{
    /// <summary>The variable that sets <see cref="CurveTolerance"/> (font units, more than 0): "IDRAK_FONT_CURVE_TOLERANCE".</summary>
    public const string CurveToleranceVariable = "IDRAK_FONT_CURVE_TOLERANCE";

    private readonly float _curveTolerance = DocumentSettings.Float(CurveToleranceVariable, 1e-3f, 1000f) ?? 1f;

    /// <summary>
    /// A subset: these glyph ids, the parts of composite glyphs among them and, with <see cref="KeepLayout"/>, every glyph
    /// the layout tables reach from them; null writes every glyph. The ids stay the same: the glyphs left out are written
    /// empty (no outline, their advance kept), so text already placed by glyph id draws the same.
    /// </summary>
    public IReadOnlyList<int>? Glyphs { get; init; }

    /// <summary>
    /// Whether the layout tables (substitutions, positioning, glyph classes) are written (true): whole, or for a subset with
    /// every glyph they reach; false drops them (what an embedded subset of a document needs).
    /// </summary>
    public bool KeepLayout { get; init; } = true;

    /// <summary>
    /// Name strings written over the source face's: the face's own name records are kept (its copyright and licence
    /// included), these replace theirs or add to them; nothing else is made up.
    /// </summary>
    public IReadOnlyDictionary<FontNameKind, string>? Names { get; init; }

    /// <summary>
    /// How far a curve converted to another kind (cubic to quadratic for a format of quadratic outlines) may stray from the
    /// original, in font units (1 unless <see cref="CurveToleranceVariable"/> sets it).
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Not positive and finite.</exception>
    public float CurveTolerance
    {
        get => _curveTolerance;
        init => _curveTolerance = value > 0f && float.IsFinite(value) ? value : throw new ArgumentOutOfRangeException(nameof(CurveTolerance), value, $"A curve tolerance is more than 0 font units ({CurveToleranceVariable}).");
    }

    /// <summary>Whether the hinting instructions are kept (false: dropped; they are kept only when the face is written whole).</summary>
    public bool KeepHinting { get; init; }
}

/// <summary>
/// Writes a face as one font file format; register it with <c>Register</c>. The reading side of a
/// format is its <see cref="IFontFormat"/> in <see cref="FontFormats"/>: what a writer writes reads back there with the
/// same outlines, advances, characters and shaping.
/// </summary>
public interface IFontWriter
{
    /// <summary>The format's name (names ignore case).</summary>
    string Name { get; }

    /// <summary>The file extensions it writes, with the dot; <c>For</c> picks a writer by
    /// them.</summary>
    IReadOnlyCollection<string> Extensions { get; }

    /// <summary>
    /// Writes <paramref name="face"/> to <paramref name="destination"/> as <paramref name="options"/> say (the stream stays
    /// the caller's; it may not seek, so a writer that must go back keeps the file until it is whole).
    /// </summary>
    /// <exception cref="NotSupportedException">The face holds what the format cannot (the message says what).</exception>
    void Write(IFontFace face, Stream destination, FontWriteOptions options);
}

/// <summary>
/// The font file writers, by name (ignoring case), beside <see cref="FontFormats"/>. Built in (Idrak.Imaging):
/// "truetype" (.ttf, whole fonts and subsets, the format Idrak works from).
/// </summary>
public static class FontWriters
{
}
