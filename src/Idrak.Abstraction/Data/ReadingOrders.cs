// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Data;

/// <summary>
/// The order a line's characters are met in along its columns, and back. A text is stored in logical order (the order it
/// is typed and read in); on the page a line is drawn in visual order, where a right-to-left line holds left-to-right runs
/// (numbers, Latin words) drawn the other way. A recognizer that reads a line's columns in its reading direction (a
/// right-to-left line flipped first) meets the characters in column order: <see cref="ToColumns"/> gives that order for a
/// transcription (the labels a line recognizer trains on) and <see cref="FromColumns"/> turns a reading back to logical
/// order. Characters are kept as they are (no mirrored glyphs).
/// </summary>
public interface IReadingOrder
{
    /// <summary>Its name in <see cref="ReadingOrders"/>.</summary>
    string Name { get; }

    /// <summary>
    /// <paramref name="logical"/> in the order its line's columns show it, read in the line's direction (right to left
    /// when <paramref name="rightToLeft"/>).
    /// </summary>
    string ToColumns(string logical, bool rightToLeft);

    /// <summary>A text in column order (as <see cref="ToColumns"/> gives it) back in logical order.</summary>
    string FromColumns(string columns, bool rightToLeft);

    /// <summary>
    /// The direction <paramref name="text"/> runs in, by its first strong character (Unicode's rules P2 and P3 of UAX #9:
    /// characters inside isolates skipped, the search ending at a paragraph separator): true when it is right-to-left
    /// (bidirectional class R or AL), false when left-to-right (L), null when the text has none (only digits, spaces and
    /// punctuation). A document's <c>lines</c> layout gives each line its direction this way.
    /// </summary>
    bool? RightToLeft(string text);
}

/// <summary>
/// The reading orders, by name (ignoring case). Built in (Idrak.Imaging): <see cref="Bidi"/> (the Unicode bidirectional
/// algorithm, UAX #9: levels resolved for a paragraph of the line's direction, reordered by rules L1 and L2) and <see cref="Runs"/> (in a right-to-left line, each run of digits and left-to-right letters, with the spaces and separators
/// inside it, reversed; its own inverse).
/// </summary>
public static class ReadingOrders
{
    /// <summary>"bidi": the Unicode bidirectional algorithm (UAX #9).</summary>
    public const string Bidi = "bidi";

    /// <summary>"runs": left-to-right runs reversed in a right-to-left line (right-to-left runs in a left-to-right one).</summary>
    public const string Runs = "runs";
}
