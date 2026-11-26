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
/// paragraph through the reading order, shaped by its script's shaper in the first face that has its characters.
/// Safe to use from several threads at once.
/// </summary>
public interface IDocumentTextSetter
{
    /// <summary>
    /// <paramref name="text"/>'s lines, top to bottom: each paragraph (a line feed ends one) on one line, or on several
    /// where it passes <see cref="DocumentTextSetting.Width"/>; an empty paragraph an empty line.
    /// </summary>
    /// <exception cref="ArgumentException">No face is given, or the size is not above 0.</exception>
    IReadOnlyList<DocumentSetLine> Set(string text, DocumentTextSetting setting);
}

/// <summary>
/// The name of the library's text setter (Idrak.Imaging): <see cref="Shaped"/>. Picking one by its name is not on the
/// new design yet.
/// </summary>
public static class DocumentTextSetters
{
    /// <summary>
    /// "shaped": the library's setter: each paragraph through the Unicode bidirectional algorithm, cut into runs of one
    /// script and direction, each shaped by <see cref="TextShapers"/> with the first face that has its characters; lines
    /// broken at spaces (between CJK characters, and inside a word too long for a line); each line's runs in visual order.
    /// </summary>
    public const string Shaped = "shaped";
}
