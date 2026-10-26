// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.IO.Compression;
using System.Text;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Idrak.Abstraction.Text;
using Idrak.Models;

// Plan 24, "What is left" item 1: Unicode normalization owned by the library (UnicodeNormalization, tables made by
// tools/unicode/make_normalization_tables.py), so NFC, NFD, NFKC and NFKD work alike on every platform and under
// invariant globalization (the CLI), where .NET's own normalization returns the text unchanged.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] UnicodeNormalizationGroup =
    [
        ("tokenizers: unicode normalization: every line of the Unicode character database's NormalizationTest.txt holds in the four forms (its five columns), every other code point is left as it is, and IsNormalized, TryNormalize and GetNormalizedLength agree with Normalize", UnicodeNormalizationConformance),
        ("tokenizers: unicode normalization: equal to the platform's normalization (this test process's ICU or Windows) on every code point and on the corpora in the four forms, except where the platform, on an older Unicode, leaves a newer character as it is", UnicodeNormalizationPlatform),
        ("tokenizers: unicode normalization: nothing allocated once warm (IsNormalized, TryNormalize, a normalized string returned as the same instance), a lone surrogate refused as .NET refuses it, a bad form refused, and a destination too small reported", UnicodeNormalizationContract),
        ("tokenizers: unicode normalization: nothing in src/ calls the platform's normalization (string.Normalize, IsNormalized, TryNormalize, GetNormalizedLength)", UnicodeNormalizationNoPlatformCalls),
        ("tokenizers: unicode normalization: the idrak tool's own process (invariant globalization) strips accents and decomposes Hangul through BERT's normalizer, giving the ids this process gives", UnicodeNormalizationCliProcess),
        ("tokenizers: unicode normalization: speed against the platform's normalization on the corpora (printed)", UnicodeNormalizationSpeed),
    ];

    private static readonly NormalizationForm[] Forms = [NormalizationForm.FormC, NormalizationForm.FormD, NormalizationForm.FormKC, NormalizationForm.FormKD];

    private static string ShowCodePoints(string text) => string.Join(" ", text.EnumerateRunes().Select(r => r.Value.ToString("X4", CultureInfo.InvariantCulture)));

    private static void UnicodeNormalizationConformance(Device device)
    {
        string path = TestData("unicode/normalization-test.txt.gz");
        using var reader = new StreamReader(new GZipStream(File.OpenRead(path), CompressionMode.Decompress), Encoding.UTF8);
        Check(reader.ReadLine()!.Contains(UnicodeNormalization.UnicodeVersion, StringComparison.Ordinal),
            $"the conformance file is for Unicode {UnicodeNormalization.UnicodeVersion}, as the tables are");
        var partOne = new HashSet<int>();
        string part = "";
        int lines = 0;
        var failures = new List<string>();
        for (string? line; (line = reader.ReadLine()) is not null;)
        {
            if (line.StartsWith('#'))
            {
                continue;
            }

            if (line.StartsWith('@'))
            {
                part = line;
                continue;
            }

            string[] c = [.. line.Split(';').Take(5).Select(f => string.Concat(f.Split(' ', StringSplitOptions.RemoveEmptyEntries)
                .Select(h => char.ConvertFromUtf32(int.Parse(h, NumberStyles.HexNumber, CultureInfo.InvariantCulture)))))];
            lines++;
            if (part == "@Part1" && char.ConvertToUtf32(c[0], 0) is int single && c[0].Length == char.ConvertFromUtf32(single).Length)
            {
                partOne.Add(single);
            }

            // NFC: c2 = NFC(c1, c2, c3), c4 = NFC(c4, c5); NFD: c3 = NFD(c1, c2, c3), c5 = NFD(c4, c5);
            // NFKC: c4 = NFKC(c1..c5); NFKD: c5 = NFKD(c1..c5).
            void Expect(string expected, NormalizationForm form, params int[] columns)
            {
                foreach (int column in columns)
                {
                    string actual = UnicodeNormalization.Normalize(c[column], form);
                    if (actual != expected && failures.Count < 10)
                    {
                        failures.Add($"{form}({ShowCodePoints(c[column])}) = {ShowCodePoints(actual)}, not {ShowCodePoints(expected)}");
                    }

                    ExpectAgreement(c[column], form, actual, failures);
                }
            }

            Expect(c[1], NormalizationForm.FormC, 0, 1, 2);
            Expect(c[3], NormalizationForm.FormC, 3, 4);
            Expect(c[2], NormalizationForm.FormD, 0, 1, 2);
            Expect(c[4], NormalizationForm.FormD, 3, 4);
            Expect(c[3], NormalizationForm.FormKC, 0, 1, 2, 3, 4);
            Expect(c[4], NormalizationForm.FormKD, 0, 1, 2, 3, 4);
        }

        Check(lines > 19000 && partOne.Count > 10000, $"the conformance file was read ({lines} lines, {partOne.Count} single code points)");
        int others = 0;
        for (int cp = 0; cp <= 0x10FFFF; cp++)
        {
            if (cp is >= 0xD800 and <= 0xDFFF || partOne.Contains(cp))
            {
                continue;
            }

            string one = char.ConvertFromUtf32(cp);
            foreach (var form in Forms)
            {
                string actual = UnicodeNormalization.Normalize(one, form);
                if (!ReferenceEquals(actual, one) && failures.Count < 10)
                {
                    failures.Add($"{form}({ShowCodePoints(one)}) = {ShowCodePoints(actual)}: a code point Part 1 does not list changed");
                }

                if (!UnicodeNormalization.IsNormalized(one, form) && failures.Count < 10)
                {
                    failures.Add($"{form}: {ShowCodePoints(one)} not reported normalized");
                }
            }

            others++;
        }

        Console.WriteLine($"    Unicode {UnicodeNormalization.UnicodeVersion}: {lines} conformance lines, {partOne.Count} code points of Part 1, {others} others left as they are");
        Check(failures.Count == 0, $"normalization differs from NormalizationTest.txt:\n{string.Join("\n", failures)}");
    }

    // IsNormalized, TryNormalize (with room and without) and GetNormalizedLength agree with Normalize's result.
    private static void ExpectAgreement(string text, NormalizationForm form, string normalized, List<string> failures)
    {
        bool isNormalized = UnicodeNormalization.IsNormalized(text, form);
        int length = UnicodeNormalization.GetNormalizedLength(text, form);
        var room = new char[normalized.Length + 2];
        bool fits = UnicodeNormalization.TryNormalize(text, room, out int written, form);
        bool tooSmall = normalized.Length > 0 && UnicodeNormalization.TryNormalize(text, room.AsSpan(0, normalized.Length - 1), out int none, form);
        if ((isNormalized != (normalized == text) || length != normalized.Length || !fits || !room.AsSpan(0, written).SequenceEqual(normalized) || tooSmall)
            && failures.Count < 10)
        {
            failures.Add($"{form}({ShowCodePoints(text)}): IsNormalized {isNormalized}, length {length}, TryNormalize {fits} {ShowCodePoints(new string(room, 0, written))}, too small {tooSmall}; Normalize gave {ShowCodePoints(normalized)}");
        }
    }

    private static void UnicodeNormalizationPlatform(Device device)
    {
        var tolerated = new int[4];
        var failures = new List<string>();
        void Compare(string text, int form)
        {
            string platform;
            try
            {
                platform = text.Normalize(Forms[form]);
            }
            catch (ArgumentException)
            {
                return;                                                            // the platform refuses a character it does not know
            }

            string ours = UnicodeNormalization.Normalize(text, Forms[form]);
            if (ours == platform)
            {
                return;
            }

            if (platform == text || (Canonical(ours) == Canonical(platform) && ours.EnumerateRunes().Any(r => NewToPlatform(r.Value))))
            {
                tolerated[form]++;                                                 // a newer character the platform does not know
            }
            else if (failures.Count < 10)
            {
                failures.Add($"{Forms[form]}({ShowCodePoints(text)}): ours {ShowCodePoints(ours)}, the platform's {ShowCodePoints(platform)}");
            }
        }

        for (int cp = 0; cp <= 0x10FFFF; cp++)
        {
            if (cp is >= 0xD800 and <= 0xDFFF)
            {
                continue;
            }

            string one = char.ConvertFromUtf32(cp);
            for (int form = 0; form < 4; form++)
            {
                Compare(one, form);
            }
        }

        var texts = PipelineTexts().ToList();
        Span<int> parts = stackalloc int[18];
        // Every composition pair, decomposed and composed: a starter with each mark after it, and Hangul jamo.
        for (int cp = 0; cp <= 0x10FFFF; cp++)
        {
            if (cp is >= 0xD800 and <= 0xDFFF)
            {
                continue;
            }

            int count = UnicodeNormalization.Decompose(cp, parts, compatibility: false);
            if (count > 1)
            {
                var builder = new StringBuilder();
                foreach (int p in parts[..count])
                {
                    builder.Append(char.ConvertFromUtf32(p));
                }

                texts.Add(builder.ToString());
                texts.Add(builder.ToString() + "\u0323\u0301");
            }
        }

        foreach (string text in texts)
        {
            for (int form = 0; form < 4; form++)
            {
                Compare(text, form);
            }
        }

        Console.WriteLine($"    the platform's normalization left newer characters as they are: NFC {tolerated[0]}, NFD {tolerated[1]}, NFKC {tolerated[2]}, NFKD {tolerated[3]} ({texts.Count} texts and every code point compared)");
        Check(failures.Count == 0, $"the library's normalization differs from the platform's:\n{string.Join("\n", failures)}");
    }

    // Canonically equivalent texts have one NFD (by the library's tables); a code point the platform does not know it
    // leaves as it is where the library decomposes it.
    private static string Canonical(string text) => UnicodeNormalization.Normalize(text, NormalizationForm.FormD);

    private static bool NewToPlatform(int codePoint)
    {
        string one = char.ConvertFromUtf32(codePoint);
        return one.Normalize(NormalizationForm.FormKD) == one && UnicodeNormalization.Normalize(one, NormalizationForm.FormKD) != one;
    }

    private static void UnicodeNormalizationContract(Device device)
    {
        string[] texts = ["plain ASCII text", "مَرْحَبًا بالعالم", "ﬁne Ｆｕｌｌ ｗｉｄｔｈ cafe\u0301", "\u1100\u1161\u11A8 \uAC01", "e\u0323\u0302 \u1E0B\u0323", "😀 \u0645\u0651\u064E"];
        var room = new char[256];
        for (int warm = 0; warm < 50; warm++)
        {
            foreach (string text in texts)
            {
                foreach (var form in Forms)
                {
                    UnicodeNormalization.IsNormalized(text, form);
                    UnicodeNormalization.TryNormalize(text, room, out _, form);
                    UnicodeNormalization.GetNormalizedLength(text, form);
                }
            }
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int run = 0; run < 100; run++)
        {
            foreach (string text in texts)
            {
                foreach (var form in Forms)
                {
                    UnicodeNormalization.IsNormalized(text, form);
                    UnicodeNormalization.TryNormalize(text, room, out _, form);
                    UnicodeNormalization.GetNormalizedLength(text, form);
                }
            }
        }

        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Check(allocated == 0, $"a warm IsNormalized, TryNormalize and GetNormalizedLength allocated {allocated} bytes");

        string normalized = "caf\u00E9 \u0645\u064E\u0651";
        Check(ReferenceEquals(UnicodeNormalization.Normalize(normalized), normalized), "a normalized string comes back as the same instance");
        Check(UnicodeNormalization.Normalize("cafe\u0301") == "caf\u00E9" && UnicodeNormalization.Normalize("caf\u00E9", NormalizationForm.FormD) == "cafe\u0301",
            "NFC composes and NFD decomposes é");
        Check(UnicodeNormalization.Normalize("\u0645\u0651\u064E") == "\u0645\u064E\u0651", "Arabic shadda before fatha is put in canonical order");
        Check(UnicodeNormalization.Normalize("\uFB01 \u2460 \uFF21", NormalizationForm.FormKC) == "fi 1 A", "NFKC folds a ligature, a circled digit and a full-width letter");

        foreach (string bad in new[] { "a\uD800b", "a\uDC00", "\uD83D" })
        {
            foreach (var form in Forms)
            {
                bool refused = false;
                try
                {
                    UnicodeNormalization.Normalize(bad + "\u0301", form);
                }
                catch (ArgumentException)
                {
                    refused = true;
                }

                Check(refused, $"{form}: a lone surrogate is refused");
            }
        }

        bool badForm = false;
        try
        {
            UnicodeNormalization.Normalize("x", (NormalizationForm)3);
        }
        catch (ArgumentException e)
        {
            badForm = e.Message.Contains("C, D, KC and KD", StringComparison.Ordinal);
        }

        Check(badForm, "a form other than C, D, KC and KD is refused naming the four");
        Check(!UnicodeNormalization.TryNormalize("\uFDFA", room.AsSpan(0, 5), out int written, NormalizationForm.FormKD) && written == 0
            && UnicodeNormalization.GetNormalizedLength("\uFDFA", NormalizationForm.FormKD) == 18, "a destination too small is reported, with the length it needs (U+FDFA: 18)");
        Check(UnicodeNormalization.MaxDecompositionLength(true) == 18 && UnicodeNormalization.MaxDecompositionLength(false) == 4, "the longest decompositions are 18 and 4");
        Check(UnicodeNormalization.CombiningClass(0x0301) == 230 && UnicodeNormalization.CombiningClass('a') == 0 && UnicodeNormalization.CombiningClass(0x0651) == 33,
            "combining classes are read from the tables");
        Check(UnicodeNormalization.Compose('e', 0x0301) == 0xE9 && UnicodeNormalization.Compose(0x1100, 0x1161) == 0xAC00 && UnicodeNormalization.Compose(0xAC00, 0x11A8) == 0xAC01
            && UnicodeNormalization.Compose(0x0915, 0x093C) == -1, "Compose gives primary composites (Hangul by the algorithm), not excluded ones");
    }

    private static void UnicodeNormalizationNoPlatformCalls(Device device)
    {
        var platform = new Regex(@"\.(?:Normalize|IsNormalized)\(\s*(?:\)|NormalizationForm\.|\w*[Ff]orm\w*\s*\))|(?<!UnicodeNormalization)\.(?:TryNormalize|GetNormalizedLength)\(", RegexOptions.CultureInvariant);
        string src = RepositoryPath("src");
        var found = new List<string>();
        int files = 0;
        foreach (string file in Directory.EnumerateFiles(src, "*.cs", SearchOption.AllDirectories))
        {
            if (file.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.Ordinal)
                || file.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}", StringComparison.Ordinal))
            {
                continue;
            }

            files++;
            int number = 0;
            foreach (string line in File.ReadLines(file))
            {
                number++;
                string code = line.TrimStart();
                if (!code.StartsWith("//", StringComparison.Ordinal) && platform.IsMatch(code))
                {
                    found.Add($"{Path.GetRelativePath(src, file)}:{number}: {code}");
                }
            }
        }

        Check(files > 100, $"the scan read src/ ({files} files)");
        Check(found.Count == 0, $"platform normalization calls in src/ (use UnicodeNormalization):\n{string.Join("\n", found)}");
    }

    private static void UnicodeNormalizationCliProcess(Device device)
    {
        string bin = AppContext.BaseDirectory;
        string dll = Path.Combine(bin, "Idrak.Cli.dll"), config = Path.Combine(bin, "Idrak.Cli.runtimeconfig.json");
        Check(File.Exists(dll) && File.Exists(config), $"the idrak tool is built beside the tests ({dll}, {config})");
        var options = JsonNode.Parse(File.ReadAllText(config))!["runtimeOptions"]?["configProperties"];
        Check((bool?)options?["System.Globalization.Invariant"] == true, "the idrak tool runs with invariant globalization (as it ships), so this test checks what .NET's own normalization would miss");

        string folder = TestData("tokenizers/wordpiece-uncased");
        const string text = "Caf\u00E9 na\u00EFve R\u00C9SUM\u00C9 \u00DCn\u00EFc\u00F6d\u00E9 e\u0301t\u00E9 \uD55C\uAD6D\uC5B4 \uFB01";
        // Standard input given and closed: the tool adds piped input to its text, and an inherited pipe nobody closes (a
        // runner started from a script) would hold it.
        var info = new ProcessStartInfo("dotnet")
        {
            RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false, StandardOutputEncoding = Encoding.UTF8,
        };
        foreach (string arg in new[] { dll, "tokenize", folder, text, "--json", "--offline" })
        {
            info.ArgumentList.Add(arg);
        }

        info.Environment["DOTNET_CLI_TELEMETRY_OPTOUT"] = "1";
        info.Environment["DOTNET_NOLOGO"] = "1";
        using var process = Process.Start(info)!;
        process.StandardInput.Close();
        var error = process.StandardError.ReadToEndAsync();
        string output = process.StandardOutput.ReadToEnd();
        process.WaitForExit();
        Check(process.ExitCode == 0, $"idrak tokenize ran: {output}{error.Result}");
        var json = JsonNode.Parse(output)!;
        int[] cli = [.. json["ids"]!.AsArray().Select(i => (int)i!)];
        string[] tokens = [.. json["tokens"]!.AsArray().Select(t => (string)t!)];

        var tokenizer = PretrainedTokenizers.Load(folder);
        int[] here = [.. tokenizer.Encode(text)];
        Check(cli.SequenceEqual(here), $"the tool's ids [{string.Join(", ", cli)}] equal this process's [{string.Join(", ", here)}]");
        Check(!tokens.Any(t => t.Any(c => c is >= '\u00C0' and <= '\u024F' || c == '\u0301')), $"accents are stripped in the tool: {string.Join(" ", tokens)}");
        string joined = string.Concat(tokens.Select(t => t.StartsWith("##", StringComparison.Ordinal) ? t[2..] : " " + t));
        Check(joined.Contains(" cafe ", StringComparison.Ordinal) && joined.Contains(" resume ", StringComparison.Ordinal) && joined.Contains(" unicode ", StringComparison.Ordinal),
            $"café, RÉSUMÉ and Ünïcödé are lowercased without their accents: {joined}");
        Check(joined.Contains('\u1100'), $"Hangul syllables are decomposed into jamo, as BERT's NFD does: {joined}");
    }

    private static void UnicodeNormalizationSpeed(Device device)
    {
        string[] texts = [.. PipelineTexts().Where(t => t.Length > 0)];
        string all = string.Join("\n", texts);
        foreach (var form in Forms)
        {
            double ours = TimeNormalization(() => UnicodeNormalization.Normalize(all, form));
            double platform = TimeNormalization(() => all.Normalize(form));
            double oursLines = TimeNormalization(() =>
            {
                foreach (string text in texts)
                {
                    UnicodeNormalization.Normalize(text, form);
                }
            });
            double platformLines = TimeNormalization(() =>
            {
                foreach (string text in texts)
                {
                    text.Normalize(form);
                }
            });
            Console.WriteLine($"    {form}: whole text ({all.Length:N0} chars) {ours:F3} ms against the platform's {platform:F3} ms; line by line ({texts.Length} lines) {oursLines:F3} ms against {platformLines:F3} ms");
        }
    }

    // The best of several timed runs, after a warm-up, in milliseconds.
    private static double TimeNormalization(Action action)
    {
        for (int i = 0; i < 30; i++)
        {
            action();
        }

        double best = double.MaxValue;
        for (int run = 0; run < 15; run++)
        {
            long start = Stopwatch.GetTimestamp();
            action();
            best = Math.Min(best, Stopwatch.GetElapsedTime(start).TotalMilliseconds);
        }

        return best;
    }
}
