// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Data;
using Idrak.Nlp;
using Idrak.Nlp.Abstractions;

// Plan 18: random text in a language for drawing training pages when there is no text to draw (idrak data text).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] RandomTextGroup =
    [
        ("random text: pages, lines or words as asked, each line its number of words within the range, the same seed the same text and another seed other text", RandomTextUnits),
        ("random text: the Arabic source writes every letter, its own digits at their share, marks only when asked and never after a letter that takes none", RandomTextArabic),
        ("random text: any language from a pools file or a word list, registered under its tag; English built in; bad pools are refused", RandomTextLanguages),
        ("random text: the written file is the plain text synthetic pages read, a document a page; idrak data text writes it by -n, --unit and -l, refusing what it cannot do", RandomTextFile),
    ];

    private static readonly string ArabicLetters = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي";

    private static void RandomTextUnits(Device device)
    {
        var arabic = RandomTextSources.Get("ar");
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
        var arabic = RandomTextSources.Get("ar");
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
        Check(RandomTextSources.Names.Contains("ar") && RandomTextSources.Names.Contains("en"), $"built in: {string.Join(", ", RandomTextSources.Names)}");
        string english = string.Join(" ", RandomText.Documents(RandomTextSources.Get("en"), new RandomTextOptions { Count = 20 }));
        Check(english.Any(char.IsAsciiLetter) && !english.Any(c => c is >= '؀' and <= 'ۿ'), "English is English");
        Throws<NotSupportedException>(() => RandomTextSources.Get("xx"), "an unknown language");

        var french = PooledTextSource.FromJson(JsonNode.Parse("""
            {"name": "fr", "pools": {"nom": ["tribunal", "décision", "créance"], "verbe": ["rend", "ordonne"]},
             "patterns": [["=le", "nom", "verbe"], ["#date"]], "punctuation": [","], "years": [2000, 2010]}
            """)!.AsObject());
        RandomTextSources.Register(french);
        try
        {
            var lines = RandomText.Documents(RandomTextSources.Get("fr"), new RandomTextOptions { Unit = RandomTextUnit.Lines, Count = 20, LineWords = (3, 6) }).ToList();
            Check(lines.All(l => l.Split(' ').All(w => w.TrimEnd(',', '.') is "le" or "tribunal" or "décision" or "créance" or "rend" or "ordonne" || w.Contains('/'))),
                $"French from its pools: {lines[0]}");
            Check(RandomTextSources.Get("ar").Name == "ar", "and Arabic as it was");
        }
        finally
        {
            RandomTextSources.Unregister("fr");
        }

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

    private static void RandomTextFile(Device device)
    {
        string folder = SyntheticFolder("random-text");
        try
        {
            string path = Path.Combine(folder, "pages.txt");
            var result = RandomText.Write(path, RandomTextSources.Get("ar"), new RandomTextOptions { Count = 6, PageLines = (2, 3), LineWords = (3, 5), Seed = 1 });
            var records = SyntheticRecordFormats.Detect(path).Read(path, "Read it.", "full_text_extract").ToList();
            Check(SyntheticRecordFormats.Detect(path).Name == "text" && records.Count == 6 && result.Documents == 6, $"{records.Count} documents read back");
            Check(!File.ReadAllText(path).Contains('\r') && records[0].Answer.Contains("full_text_extract", StringComparison.Ordinal), "'\\n' lines, answers under the key");
            var page = SyntheticPages.Make(records[0].Answer, new SyntheticPageOptions { Style = new PageStyle { Fonts = [SubsetFace()], Width = 400, Height = 300, BodySize = (10, 12) }, Scans = ["none"] });
            Check(page.Text == string.Join("\n", RandomText.Documents(RandomTextSources.Get("ar"), new RandomTextOptions { Count = 1, PageLines = (2, 3), LineWords = (3, 5), Seed = 1 }).Single().Split('\n')),
                "the first document drawn as a page, its text the answer");

            var (code, output, error) = SynthCli(["data", "text", "-n", "4", "--unit", "lines", "-l", "en", "-o", Path.Combine(folder, "en.txt"), "--seed", "3", "-j"]);
            var json = JsonNode.Parse(output)!;
            Check(code == 0 && (int)json["documents"]! == 4 && (string)json["language"]! == "en" && (string)json["unit"]! == "lines" && File.ReadAllText(Path.Combine(folder, "en.txt")).Split("\n---\n").Length == 4,
                $"exit {code}: {output} {error}");
            var arabic = SynthCli(["data", "text", "-n", "3", "-o", Path.Combine(folder, "ar.txt"), "-j"]);
            Check(arabic.Code == 0 && (string)JsonNode.Parse(arabic.Out)!["language"]! == "ar" && (string)JsonNode.Parse(arabic.Out)!["unit"]! == "pages", "Arabic pages unless asked");
            File.WriteAllText(Path.Combine(folder, "words.txt"), "Gericht Urteil\nKlage", new UTF8Encoding(false));
            var listed = SynthCli(["data", "text", "-n", "5", "--unit", "words", "-l", "de", "--words", Path.Combine(folder, "words.txt"), "-o", Path.Combine(folder, "de.txt"), "-j"]);
            Check(listed.Code == 0 && File.ReadAllText(Path.Combine(folder, "de.txt")).Split("\n---\n").All(w => w.Trim() is "Gericht" or "Urteil" or "Klage"), $"a word list: {listed.Err}");

            foreach (var (args, why) in new (string[], string)[]
            {
                (["data", "text", "-n", "3"], "no -o"),
                (["data", "text", "-n", "3", "--unit", "chapters", "-o", Path.Combine(folder, "x.txt")], "an unknown unit"),
                (["data", "text", "-n", "3", "-l", "xx", "-o", Path.Combine(folder, "x.txt")], "a language with no source"),
                (["data", "text", "-n", "3", "--line-words", "9..2", "-o", Path.Combine(folder, "x.txt")], "a backwards range"),
            })
            {
                var bad = SynthCli(args);
                Check(bad.Code == 2 && bad.Err.Length > 0, $"{why}: exit {bad.Code}, {bad.Err.Trim()}");
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }
}
