// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Vision;

/// <summary>
/// The order characters take on a page (left to right) against the order they are read in. A right-to-left line runs
/// its words from right to left and each word's letters from right to left, but a number keeps its digits left to
/// right: the core of Unicode's bidirectional rules for a line of one script. The mapping is its own inverse.
/// </summary>
internal static class TextOrder
{
    /// <summary>A line's words between page order (left to right, each spelled left to right) and reading order.</summary>
    public static string Reorder(IReadOnlyList<string> words, bool rightToLeft)
    {
        if (!rightToLeft)
        {
            return string.Join(' ', words);
        }

        var reordered = new string[words.Count];
        for (int i = 0; i < words.Count; i++)
        {
            string word = words[words.Count - 1 - i];
            reordered[i] = IsNumber(word) ? word : string.Create(word.Length, word, (span, w) =>
            {
                w.AsSpan().CopyTo(span);
                span.Reverse();
            });
        }

        return string.Join(' ', reordered);
    }

    /// <summary>Whether a word is a number (digits only, in any script).</summary>
    public static bool IsNumber(string word) => word.Length > 0 && word.All(char.IsDigit);
}
