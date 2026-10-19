// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Text;
using Idrak.Abstraction.Text;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>
/// SentencePiece's built-in normalization rules ("nfkc", "nmt_nfkc"), built as sentencepiece 0.2.2's <c>builder.cc</c>
/// builds them (<c>BuildNFKCMap</c>, <c>MergeNmtMap</c>, <c>RemoveRedundantMap</c>), with the library's Unicode
/// normalization (<see cref="UnicodeNormalization"/>) where sentencepiece calls ICU's, then compiled into a precompiled
/// charsmap (<see cref="PrecompiledCharsMap.Build"/>). sentencepiece ships the maps its ICU made; these are the same maps
/// for the Unicode version of the library's tables, so they differ only at code points sentencepiece's version lacks.
/// Each rule is built once a process and kept (a few hundred thousand keys: every Hangul syllable spelled with jamo of
/// each width, for instance).
/// </summary>
internal static class SentencePieceRules
{
    private const int LastCodePoint = 0x10FFFF;

    private static readonly ConcurrentDictionary<string, Lazy<Dictionary<string, string>>> Maps = new(StringComparer.Ordinal);
    private static readonly ConcurrentDictionary<string, Lazy<byte[]>> Charsmaps = new(StringComparer.Ordinal);

    /// <summary>
    /// The rule's map (a key's code points → what they become; an empty value removes them), as sentencepiece's builder
    /// makes it for <paramref name="name"/> (<see cref="TokenizerTrainers.Nfkc"/> or <see cref="TokenizerTrainers.NmtNfkc"/>).
    /// </summary>
    public static IReadOnlyDictionary<string, string> Map(string name) => Maps.GetOrAdd(name, n => new Lazy<Dictionary<string, string>>(() => BuildMap(n))).Value;

    /// <summary>The rule's map built anew (<see cref="Map"/> keeps the one built first).</summary>
    public static Dictionary<string, string> BuildMap(string name) => name switch
    {
        TokenizerTrainers.Nfkc => BuildNfkc(),
        TokenizerTrainers.NmtNfkc => MergeNmt(BuildNfkc()),
        _ => throw new ArgumentException($"No built-in normalization rule '{name}' (rules: {TokenizerTrainers.Nfkc}, {TokenizerTrainers.NmtNfkc}).", nameof(name)),
    };

    /// <summary>
    /// The precompiled charsmap of the rule <paramref name="name"/> with the Arabic <paramref name="removal"/> folded in:
    /// the marks removed, a key that holds one dropped, and the marks removed from what the rule writes. Empty for the
    /// identity rule without removals.
    /// </summary>
    public static byte[] Charsmap(string name, ArabicMarks removal)
    {
        if (name == TokenizerTrainers.Identity)
        {
            return removal.Any ? PrecompiledCharsMap.Build([.. removal.CodePoints.Select(c => (char.ConvertFromUtf32(c), ""))]) : [];
        }

        return Charsmaps.GetOrAdd($"{name}/{removal.Key}", _ => new Lazy<byte[]>(() =>
        {
            var rules = new List<(string Key, string Value)>();
            foreach (var (key, value) in Map(name))
            {
                if (removal.Any && key.AsSpan().ContainsAny(removal.Characters))
                {
                    continue;
                }

                string replaced = removal.Any ? removal.Remove(value) : value;
                if (replaced != key)
                {
                    rules.Add((key, replaced));
                }
            }

            foreach (int c in removal.CodePoints)
            {
                rules.Add((char.ConvertFromUtf32(c), ""));
            }

            rules.Sort((a, b) => string.CompareOrdinal(a.Key, b.Key));      // the same bytes from every run
            return PrecompiledCharsMap.Build(rules);
        })).Value;
    }

    // builder.cc's BuildMapInternal with NFKC and NFKD: every code point's NFKC where it changes; then every sequence that
    // NFKD-decomposes to the same characters as some code point's NFKD, spelled with any character whose NFKD is each of
    // them, mapped to that NFKD's NFKC where it changes; then RemoveRedundantMap.
    private static Dictionary<string, string> BuildNfkc()
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        var others = new Dictionary<int, List<int>>();                     // a character → the others whose NFKD it is
        var decomposed = new HashSet<string>(StringComparer.Ordinal);       // NFKDs of two or more code points
        Span<char> one = stackalloc char[2];
        for (int cp = 1; cp <= LastCodePoint; cp++)
        {
            if (!IsUnicodeChar(cp))
            {
                continue;
            }

            var text = one[..new Rune(cp).EncodeToUtf16(one)];
            if (!UnicodeNormalization.IsNormalized(text, NormalizationForm.FormKC))
            {
                string nfkc = UnicodeNormalization.Normalize(text.ToString(), NormalizationForm.FormKC);
                if (!nfkc.AsSpan().SequenceEqual(text))
                {
                    map[text.ToString()] = nfkc;
                }
            }

            if (!UnicodeNormalization.IsNormalized(text, NormalizationForm.FormKD))
            {
                string nfkd = UnicodeNormalization.Normalize(text.ToString(), NormalizationForm.FormKD);
                if (CodePoints(nfkd) == 1)
                {
                    int target = char.ConvertToUtf32(nfkd, 0);
                    if (target != cp)
                    {
                        (others.TryGetValue(target, out var list) ? list : others[target] = []).Add(cp);
                    }
                }
                else
                {
                    decomposed.Add(nfkd);
                }
            }
        }

        var key = new StringBuilder();
        foreach (string nfkd in decomposed.Order(StringComparer.Ordinal))
        {
            string nfkc = UnicodeNormalization.Normalize(nfkd, NormalizationForm.FormKC);
            if (nfkc == nfkd)
            {
                continue;                                                    // its code point's own entry covers it
            }

            // Every spelling: each character of the NFKD, or any character whose NFKD it is (an odometer over them).
            var choices = new List<int[]>();
            for (int i = 0; i < nfkd.Length; i += char.IsSurrogatePair(nfkd, i) ? 2 : 1)
            {
                int c = char.ConvertToUtf32(nfkd, i);
                choices.Add(others.TryGetValue(c, out var list) ? [c, .. list] : [c]);
            }

            var at = new int[choices.Count];
            while (true)
            {
                key.Clear();
                for (int i = 0; i < at.Length; i++)
                {
                    AppendCodePoint(key, choices[i][at[i]]);
                }

                string spelled = key.ToString();
                if (spelled != nfkc)
                {
                    map[spelled] = nfkc;
                }

                int digit = at.Length - 1;
                while (digit >= 0 && ++at[digit] == choices[digit].Length)
                {
                    at[digit--] = 0;
                }

                if (digit < 0)
                {
                    break;
                }
            }
        }

        return RemoveRedundant(map);
    }

    // builder.cc's MergeNmtMap: other whitespace characters become a space, ASCII control characters go, and the
    // full-width tilde stays (Japanese uses it apart from the tilde); then RemoveRedundantMap.
    private static Dictionary<string, string> MergeNmt(Dictionary<string, string> nfkc)
    {
        var map = new Dictionary<string, string>(nfkc, StringComparer.Ordinal);
        foreach (int c in (int[])[0x0009, 0x000A, 0x000C, 0x000D, 0x1680, 0x200B, 0x200E, 0x200F, 0x2028, 0x2029, 0x2581, 0xFEFF, 0xFFFD, 0x200C])
        {
            map[char.ConvertFromUtf32(c)] = " ";
        }

        foreach (int c in (int[])[.. Enumerable.Range(0x01, 8), 0x0B, .. Enumerable.Range(0x0E, 0x20 - 0x0E), 0x7F, 0x8F, 0x9F])
        {
            map[char.ConvertFromUtf32(c)] = "";
        }

        map.Remove("～");
        return RemoveRedundant(map);
    }

    // builder.cc's RemoveRedundantMap: the one-character keys stay; a longer key stays only when the shorter keys kept so
    // far do not already normalize it to its value.
    private static Dictionary<string, string> RemoveRedundant(Dictionary<string, string> map)
    {
        var byLength = map.GroupBy(p => CodePoints(p.Key)).OrderBy(g => g.Key).ToList();
        var kept = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var group in byLength.Where(g => g.Key == 1))
        {
            foreach (var (k, v) in group)
            {
                kept[k] = v;
            }
        }

        var lookup = kept.GetAlternateLookup<ReadOnlySpan<char>>();
        var output = new StringBuilder();
        foreach (var group in byLength.Where(g => g.Key >= 2))
        {
            foreach (var (k, v) in group)
            {
                Normalize(lookup, k, group.Key - 1, output);
                if (!output.Equals(v.AsSpan()))
                {
                    kept[k] = v;
                }
            }
        }

        return kept;
    }

    // builder.cc's Normalize: the longest key of at most maxLength code points at each place, else one code point as it is.
    private static void Normalize(Dictionary<string, string>.AlternateLookup<ReadOnlySpan<char>> map, string text, int maxLength, StringBuilder output)
    {
        output.Clear();
        int i = 0;
        while (i < text.Length)
        {
            int end = i, found = -1;
            string? value = null;
            for (int n = 0; n < maxLength && end < text.Length; n++)
            {
                end += char.IsSurrogatePair(text, end) ? 2 : 1;
                if (map.TryGetValue(text.AsSpan(i, end - i), out string? hit))
                {
                    (found, value) = (end, hit);
                }
            }

            if (found < 0)
            {
                int width = char.IsSurrogatePair(text, i) ? 2 : 1;
                output.Append(text, i, width);
                i += width;
            }
            else
            {
                output.Append(value);
                i = found;
            }
        }
    }

    // ICU's U_IS_UNICODE_CHAR: not a surrogate, not a noncharacter.
    private static bool IsUnicodeChar(int c) =>
        c < 0xD800 || (c > 0xDFFF && c <= LastCodePoint && (c & 0xFFFE) != 0xFFFE && !(c >= 0xFDD0 && c <= 0xFDEF));

    private static int CodePoints(string text)
    {
        int count = text.Length;
        foreach (char c in text)
        {
            count -= char.IsLowSurrogate(c) ? 1 : 0;
        }

        return count;
    }

    private static void AppendCodePoint(StringBuilder builder, int c)
    {
        if (c < 0x10000)
        {
            builder.Append((char)c);
        }
        else
        {
            builder.Append(char.ConvertFromUtf32(c));
        }
    }
}
