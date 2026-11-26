// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Data;
using Idrak.Text;
using Idrak.Text.Abstractions;

// Plan 18: random text in a language for drawing training pages when there is no text to draw (idrak data text).
internal abstract partial class DesktopTests : Tests
{
    internal static readonly (string Name, Action<Device> Run)[] RandomTextGroup =
    [
        ("random text: pages, lines or words as asked, each line its number of words within the range, the same seed the same text and another seed other text", RandomTextUnits),
        ("random text: the Arabic source writes every letter, its own digits at their share, marks only when asked and never after a letter that takes none", RandomTextArabic),
        ("random text: any language from a pools file or a word list; English built in; bad pools are refused", RandomTextLanguages),
    ];

    private static readonly string ArabicLetters = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي";

    private static void RandomTextUnits(Device device)
    {
        var arabic = PooledTextSource.Embedded("random-text.ar.json");
        var options = new RandomTextOptions { Count = 40, PageLines = (3, 5), LineWords = (4, 7), Seed = 2 };
        var pages = RandomText.Documents(arabic, options).ToList();
        Check(pages.Count == 40 && pages.All(p => p.Split('\n').Length is >= 3 and <= 5), $"{pages.Count} pages of 3 to 5 lines");
        Check(pages.SelectMany(p => p.Split('\n')).All(l => l.Split(' ').Length is >= 4 and <= 7 && !l.Contains("  ", StringComparison.Ordinal) && l == l.Trim()),
            "every line 4 to 7 words, single spaces");
        Check(RandomText.Documents(arabic, options).SequenceEqual(pages) && !RandomText.Documents(arabic, options with { Seed = 3 }).SequenceEqual(pages), "seeded");

        var lines = RandomText.Documents(arabic, options with { Unit = RandomTextUnit.Lines, Count = 25 }).ToList();
        Check(lines.Count == 25 && lines.All(l => !l.Contains('\n') && l.Split(' ').Length is >= 4 and <= 7), "25 single lines");
        var words = RandomText.Documents(arabic, options with { Unit = RandomTextUnit.Words, Count = 30 }).ToList();
        Check(words.Count == 30 && words.All(w => w.Length > 0 && !w.Contains(' ')), $"30 single words: {string.Join(" | ", words.Take(5))}");

        foreach (var bad in new[] { options with { Count = 0 }, options with { PageLines = (5, 3) }, options with { LineWords = (0, 2) }, options with { NativeDigits = 2 } })
        {
            Throws<ArgumentException>(() => RandomText.Documents(arabic, bad), bad.ToString());
        }
    }

    private static void RandomTextArabic(Device device)
    {
        var arabic = PooledTextSource.Embedded("random-text.ar.json");
        string text = string.Join("\n", RandomText.Documents(arabic, new RandomTextOptions { Count = 150, Seed = 5 }));
        var missing = ArabicLetters.Where(c => !text.Contains(c)).ToList();
        Check(missing.Count == 0, $"every letter written; missing: {string.Join(" ", missing)}");
        int native = text.Count(c => c is >= '٠' and <= '٩'), latin = text.Count(char.IsAsciiDigit);
        Check(native > 0 && latin > 0 && native < latin, $"Arabic-Indic digits at about 0.3 of the numbers ({native} against {latin} Latin digits)");
        Check(!text.Any(c => c is >= 'ً' and <= 'ْ'), "no marks unless asked");

        string none = string.Join("\n", RandomText.Documents(arabic, new RandomTextOptions { Count = 30, NativeDigits = 0 }));
        Check(!none.Any(c => c is >= '٠' and <= '٩'), "no Arabic-Indic digits at share 0");
        string marked = string.Join("\n", RandomText.Documents(arabic, new RandomTextOptions { Count = 30, Diacritics = 1 }));
        bool afterBare = false;
        for (int i = 1; i < marked.Length; i++)
        {
            afterBare |= marked[i] is >= 'ً' and <= 'ْ' && "اآىءة".Contains(marked[i - 1]);
        }

        Check(marked.Any(c => c is >= 'ً' and <= 'ْ') && !afterBare, "marks when asked, never after alef, alef maqsura, hamza or ta marbuta");
    }

    private static void RandomTextLanguages(Device device)
    {
        string english = string.Join(" ", RandomText.Documents(PooledTextSource.Embedded("random-text.en.json"), new RandomTextOptions { Count = 20 }));
        Check(english.Any(char.IsAsciiLetter) && !english.Any(c => c is >= '؀' and <= 'ۿ'), "English is English");

        var french = PooledTextSource.FromJson(JsonNode.Parse("""
            {"name": "fr", "pools": {"nom": ["rapport", "réunion", "projet"], "verbe": ["publie", "prépare"]},
             "patterns": [["=le", "nom", "verbe"], ["#date"]], "punctuation": [","], "years": [2000, 2010]}
            """)!.AsObject());
        var lines = RandomText.Documents(french, new RandomTextOptions { Unit = RandomTextUnit.Lines, Count = 20, LineWords = (3, 6) }).ToList();
        Check(lines.All(l => l.Split(' ').All(w => w.TrimEnd(',', '.') is "le" or "rapport" or "réunion" or "projet" or "publie" or "prépare" || w.Contains('/'))),
            $"French from its pools: {lines[0]}");

        var listed = PooledTextSource.FromWords("de", ["Gericht Urteil", "", "Klage"]);
        var german = RandomText.Documents(listed, new RandomTextOptions { Unit = RandomTextUnit.Lines, Count = 10 }).SelectMany(l => l.Split(' ')).ToHashSet();
        Check(german.SetEquals(["Gericht", "Urteil", "Klage"]), $"a word list's words: {string.Join(",", german)}");

        foreach (var (json, why) in new[]
        {
            ("""{"pools": {"a": ["x"]}, "patterns": [["b"]]}""", "a pattern naming no pool"),
            ("""{"pools": {"a": []}}""", "an empty pool"),
            ("""{"pools": {"a": ["x"]}, "colour": 1}""", "an unknown key"),
            ("""{"pools": {"a": ["x"]}, "digits": "012"}""", "digits not ten"),
            ("""{"pools": {"a": ["x"]}, "years": [2030, 2000]}""", "years backwards"),
            ("""{"patterns": [["a"]]}""", "no pools"),
        })
        {
            Throws<InvalidDataException>(() => PooledTextSource.FromJson(JsonNode.Parse(json)!.AsObject()), why);
        }

        Throws<InvalidDataException>(() => PooledTextSource.FromWords("x", ["", " "]), "a word list with no words");
    }
}
