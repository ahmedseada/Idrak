// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Vision;

/// <summary>One of a character's likeliest classes and its probability.</summary>
public readonly record struct CharacterCandidate(string Text, float Probability);

/// <summary>
/// A character read from a page: the text chosen for it (after the line's script and the word's context), the
/// probability the model gave that text, its box, and the model's likeliest classes before any context. When context
/// changed the model's choice (a 1 among letters read as I), <see cref="Confidence"/> is the model's probability for the
/// new text, which can be low; <see cref="Candidates"/> shows what the model itself preferred.
/// </summary>
public sealed record RecognizedCharacter(string Text, float Confidence, PixelBox Box, IReadOnlyList<CharacterCandidate> Candidates);

/// <summary>A text line read from a page: its text in reading order, its script, and its characters in page order (left to right).</summary>
public sealed record RecognizedLine(string Text, WritingScript Script, PixelBox Box, IReadOnlyList<RecognizedCharacter> Characters)
{
    /// <summary>Whether the line runs right to left (its script does).</summary>
    public bool RightToLeft => Script.RightToLeft;
}

/// <summary>The text read from a page, line by line from top to bottom.</summary>
public sealed record RecognizedPage(int Width, int Height, IReadOnlyList<RecognizedLine> Lines)
{
    /// <summary>The page's text: its lines in reading order, one per line.</summary>
    public string Text => string.Join('\n', Lines.Select(l => l.Text));
}
