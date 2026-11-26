// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Data;

/// <summary>
/// One glyph of shaped text, in font units: which glyph, the text position it came from, how far it moves the pen and
/// how far it sits from the pen (y up).
/// </summary>
/// <param name="Glyph">The glyph id in the font.</param>
/// <param name="Cluster">The index (UTF-16) in the text of the first character the glyph stands for; glyphs of one cluster draw one unit of text.</param>
/// <param name="XAdvance">How far the pen moves after it.</param>
/// <param name="YAdvance">The same, vertically (0 for horizontal text).</param>
/// <param name="XOffset">Where it is drawn, from the pen.</param>
/// <param name="YOffset">The same, vertically (a mark above a letter is drawn higher).</param>
public readonly record struct ShapedGlyph(int Glyph, int Cluster, int XAdvance, int YAdvance, int XOffset, int YOffset);

/// <summary>
/// A run of text one shaper turns into glyphs: one script, one direction.
/// </summary>
/// <param name="Start">Its first index (UTF-16) in the text.</param>
/// <param name="Length">Its length (UTF-16).</param>
/// <param name="Script">Its script, an ISO 15924 code ("Arab", "Latn", "Hebr", "Hani"; "Zyyy" for text of no script, such as digits and punctuation alone).</param>
/// <param name="RightToLeft">Whether it reads right to left.</param>
/// <param name="Language">Its language, a BCP 47 tag ("ar", "fa", "ur"), or null: the font's default rules for the script.</param>
public readonly record struct TextRun(int Start, int Length, string Script, bool RightToLeft, string? Language = null)
{
    /// <summary>How characters share clusters (<see cref="TextClusters.Graphemes"/> unless set).</summary>
    public TextClusters Clusters { get; init; }

    /// <summary>
    /// The seed of the features given the value "random" ("ALTS=random", <see cref="ITextShaper.Shape"/>): each glyph their
    /// alternate substitutions cover takes itself or one of its alternates, picked from this seed, the glyph and its
    /// cluster, so the same seed and text give the same picks (0 unless set).
    /// </summary>
    public int Seed { get; init; }
}

/// <summary>How a shaper groups characters into clusters (HarfBuzz's monotone cluster levels).</summary>
public enum TextClusters
{
    /// <summary>A letter and its marks are one cluster from the start (HarfBuzz's default, MONOTONE_GRAPHEMES).</summary>
    Graphemes,

    /// <summary>Each character is its own cluster until shaping joins them (MONOTONE_CHARACTERS).</summary>
    Characters,
}

/// <summary>
/// Turns a run of text into positioned glyphs of a font, for the scripts it shapes: the characters mapped to glyphs,
/// substituted and positioned by the font's layout tables as the
/// script's rules say. The glyphs come back in the order they are drawn, left to right (a right-to-left run reversed).
/// </summary>
public interface ITextShaper
{
    /// <summary>The shaper's name ("general", "arabic").</summary>
    string Name { get; }

    /// <summary>
    /// Shapes <paramref name="run"/> of <paramref name="text"/> with <paramref name="font"/>. The text around the run is
    /// context (an Arabic letter joins across a run's edge). <paramref name="features"/> turn OpenType features on
    /// ("+smcp") or off ("-liga") over the script's defaults, or give a value as HarfBuzz reads it ("salt=2": an alternate
    /// substitution's second alternate), or "random" ("ALTS=random": each glyph an alternate substitution covers takes
    /// itself or one of its alternates, picked from <see cref="TextRun.Seed"/>); null for the defaults.
    /// </summary>
    ShapedGlyph[] Shape(IFontFace font, string text, TextRun run, IReadOnlyCollection<string>? features = null);
}

/// <summary>
/// The text shapers, keyed by script (ISO 15924 code). A script with no shaper of its own is shaped by the one for
/// <see cref="AnyScript"/>. The library's (Idrak.Imaging): "general" for <see cref="AnyScript"/> (the
/// character map, then the font's GSUB and GPOS features for the script: right for scripts that neither join nor
/// reorder, such as Latin, Greek, Cyrillic, Armenian, Georgian, Hebrew and the CJK scripts) and "arabic" for Arab,
/// Syrc, Nkoo, Mong, Mand, Mani, Phlp, Adlm, Rohg, Sogd and Ougr (cursive joining). Picking one by its script is not on
/// the new design yet.
/// </summary>
public static class TextShapers
{
    /// <summary>The key of the shaper for every script without its own.</summary>
    public const string AnyScript = "*";
}
