// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using Idrak.Abstraction.Generation;

namespace Idrak.Text.Generation;

/// <summary>
/// The writing systems of a text's letters, by Unicode ranges only (no language detection): for the coding agent's check
/// that a reply's prose is written in a script its request uses (<see cref="AgentOptions.LanguageNudges"/>).
/// </summary>
internal static class ReplyScripts
{
    // Fewer letters of prose than this are not judged (a file name, a short "Done.").
    internal const int MinLetters = 20;

    // The scripts, by name; each range below names one by its index. Letters outside every range are not counted.
    private static readonly string[] Names =
    [
        "Latin", "Greek", "Cyrillic", "Armenian", "Hebrew", "Arabic", "Syriac", "Thaana", "Devanagari", "Bengali",
        "Gurmukhi", "Gujarati", "Oriya", "Tamil", "Telugu", "Kannada", "Malayalam", "Sinhala", "Thai", "Lao",
        "Tibetan", "Myanmar", "Georgian", "Hangul", "Ethiopic", "Khmer", "Hiragana", "Katakana", "Bopomofo", "Han",
    ];

    // Sorted, non-overlapping [start, end] code point ranges and the script of each.
    private static readonly int[] Starts =
    [
        0x0041, 0x00C0, 0x0370, 0x0400, 0x0530, 0x0590, 0x0600, 0x0700, 0x0750, 0x0780, 0x08A0, 0x0900, 0x0980, 0x0A00,
        0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00, 0x0D80, 0x0E00, 0x0E80, 0x0F00, 0x1000, 0x10A0, 0x1100, 0x1200,
        0x1780, 0x1E00, 0x1F00, 0x2C60, 0x3040, 0x30A0, 0x3100, 0x3130, 0x31F0, 0x3400, 0x4E00, 0xA720, 0xAC00, 0xF900,
        0xFB50, 0xFE70, 0xFF21, 0xFF41, 0xFF66, 0x20000,
    ];

    private static readonly int[] Ends =
    [
        0x007A, 0x024F, 0x03FF, 0x052F, 0x058F, 0x05FF, 0x06FF, 0x074F, 0x077F, 0x07BF, 0x08FF, 0x097F, 0x09FF, 0x0A7F,
        0x0AFF, 0x0B7F, 0x0BFF, 0x0C7F, 0x0CFF, 0x0D7F, 0x0DFF, 0x0E7F, 0x0EFF, 0x0FFF, 0x109F, 0x10FF, 0x11FF, 0x139F,
        0x17FF, 0x1EFF, 0x1FFF, 0x2C7F, 0x309F, 0x30FF, 0x312F, 0x318F, 0x31FF, 0x4DBF, 0x9FFF, 0xA7FF, 0xD7AF, 0xFAFF,
        0xFDFF, 0xFEFF, 0xFF3A, 0xFF5A, 0xFF9F, 0x2FA1F,
    ];

    private static readonly byte[] Scripts =
    [
        0, 0, 1, 2, 3, 4, 5, 6, 5, 7, 5, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 0, 1, 0, 26, 27, 28, 23, 27, 29, 29, 0, 23, 29,
        5, 5, 0, 0, 27, 29,
    ];

    /// <summary>The number of scripts counted (<see cref="Count"/>'s span length).</summary>
    internal static int ScriptCount => Names.Length;

    /// <summary>A script's name by its index.</summary>
    internal static string Name(int script) => Names[script];

    /// <summary>
    /// Counts the letters of <paramref name="text"/> per script into <paramref name="counts"/> (<see cref="ScriptCount"/>
    /// long) and returns how many letters were counted.
    /// </summary>
    internal static int Count(ReadOnlySpan<char> text, Span<int> counts)
    {
        int total = 0;
        foreach (var rune in text.EnumerateRunes())
        {
            if (rune.Value < 0x41 || !Rune.IsLetter(rune))
            {
                continue;
            }

            int script = Script(rune.Value);
            if (script >= 0)
            {
                counts[script]++;
                total++;
            }
        }

        return total;
    }

    // The script of a code point, or -1.
    private static int Script(int code)
    {
        int lo = 0, hi = Starts.Length - 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >>> 1;
            if (code < Starts[mid])
            {
                hi = mid - 1;
            }
            else if (code > Ends[mid])
            {
                lo = mid + 1;
            }
            else
            {
                return Scripts[mid];
            }
        }

        return -1;
    }

    /// <summary>
    /// The script most of <paramref name="reply"/>'s prose is written in, when the request uses none of it: more than half
    /// of the prose's letters (at least <see cref="MinLetters"/>) are in it and <paramref name="request"/> has letters but
    /// none in it. Prose is the reply without its fenced blocks, inline code, the template's call markers and what they hold,
    /// and JSON objects (calls written as text). Null otherwise.
    /// </summary>
    internal static string? Foreign(string reply, string request, ToolCallContext context)
    {
        Span<int> replyCounts = stackalloc int[Names.Length];
        Span<int> requestCounts = stackalloc int[Names.Length];
        int letters = 0;
        foreach (var range in Prose(reply, context))
        {
            letters += Count(reply.AsSpan(range), replyCounts);
        }

        if (letters < MinLetters || Count(request, requestCounts) == 0)
        {
            return null;
        }

        int top = 0;
        for (int i = 1; i < replyCounts.Length; i++)
        {
            top = replyCounts[i] > replyCounts[top] ? i : top;
        }

        return replyCounts[top] * 2 > letters && requestCounts[top] == 0 ? Names[top] : null;
    }

    // The ranges of `text` that are prose: outside fences, inline code, call markers and JSON objects.
    private static List<Range> Prose(string text, ToolCallContext context)
    {
        var ranges = new List<Range>();
        var format = context.Template.ToolCalls;
        int start = 0;
        for (int i = 0; i < text.Length;)
        {
            int skip = -1;
            if (string.CompareOrdinal(text, i, "```", 0, 3) == 0)
            {
                int close = text.IndexOf("```", i + 3, StringComparison.Ordinal);
                skip = close < 0 ? text.Length : close + 3;
            }
            else if (text[i] == '`')
            {
                int close = text.IndexOf('`', i + 1);
                skip = close < 0 ? i + 1 : close + 1;
            }
            else if (format.Open.Length > 0 && string.CompareOrdinal(text, i, format.Open, 0, format.Open.Length) == 0)
            {
                int close = format.Close.Length > 0 ? text.IndexOf(format.Close, i + format.Open.Length, StringComparison.Ordinal) : -1;
                skip = close < 0 ? text.Length : close + format.Close.Length;
            }
            else if (text[i] == '{' && i + 1 < text.Length && text.AsSpan(i + 1).TrimStart() is [ '"', .. ])
            {
                int end = ToolCallText.JsonValueEnd(text, i);
                skip = end < 0 ? text.Length : end;
            }

            if (skip < 0)
            {
                i++;
                continue;
            }

            if (i > start)
            {
                ranges.Add(start..i);
            }

            start = i = skip;
        }

        if (start < text.Length)
        {
            ranges.Add(start..text.Length);
        }

        return ranges;
    }
}
