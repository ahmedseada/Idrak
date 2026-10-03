// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Idrak;
using Idrak.Cli;
using Idrak.Cli.Shared;

// Arabic messages in the idrak tool (--lang ar) and the renderer that shows them in terminals without bidirectional
// support: Arabic shaping, the Unicode Bidirectional Algorithm on known lines (expected visual order worked out from the
// standard), lines checked against an independent implementation (GNU FriBidi, tests/Idrak.Tests/data/arabic), the
// column-keeping line renderer and its writer, terminal detection, the message catalogs (every message and summary in
// both languages) and the tool end to end (help, doctor and devices in Arabic, JSON never translated).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] CliArabicGroup =
    [
        ("cli arabic: shaping to presentation forms (isolated, initial, medial, final, lam-alef)", CliArabicShaping),
        ("cli arabic: bidirectional algorithm on known lines (Arabic, English, numbers, paths, brackets, marks)", CliArabicBidi),
        ("cli arabic: visual order matches an independent implementation on 369 lines", CliArabicVectors),
        ("cli arabic: line renderer keeps columns, indentation and colours; the writer holds right-to-left lines", CliArabicRenderer),
        ("cli arabic: terminal detection and --lang-render", CliArabicDetection),
        ("cli arabic: every message, summary and group title has both languages", CliArabicCatalog),
        ("cli arabic: help, doctor and devices in Arabic; JSON, CSV and English output unchanged", CliArabicCommands),
    ];

    private static (int Exit, string Out, string Err) ArabicCli(params string[] args)
    {
        var output = new StringWriter();
        var error = new StringWriter();
        int exit = StandardInput.With(new StringReader(""), () =>
            CommandLine.Run([.. args, "-C", Path.Combine(Path.GetTempPath(), "idrak-cli-arabic-no-config.json")], output, error));
        return (exit, output.ToString(), error.ToString());
    }

    // help takes no config option (a word after "help" names a page), so it runs as typed.
    private static (int Exit, string Out, string Err) HelpCli(params string[] args)
    {
        var output = new StringWriter();
        var error = new StringWriter();
        int exit = CommandLine.Run(args, output, error);
        return (exit, output.ToString(), error.ToString());
    }

    private static string Codes(string text) => string.Join(' ', text.Select(c => ((int)c).ToString("X4")));

    // The visual order of a paragraph, as a string.
    private static string Visual(string text, int level = -1)
    {
        var (points, _, _) = VisualText.Reorder(text, level);
        var builder = new StringBuilder();
        foreach (int p in points)
        {
            VisualText.Append(builder, p);
        }

        return builder.ToString();
    }

    private static string Reverse(string text) => new([.. text.Reverse()]);

    private static void CliArabicShaping(Device device)
    {
        _ = device;
        void Expect(string text, string shaped) =>
            Check(ArabicShaping.Shape(text) == shaped, $"shape({text}): {Codes(ArabicShaping.Shape(text))}, expected {Codes(shaped)}");

        // مرحبا: meem initial (reh joins it), reh final (reh never joins the next), hah initial, beh medial, alef final.
        Expect("مرحبا", "\uFEE3\uFEAE\uFEA3\uFE92\uFE8E");
        // One letter alone: isolated; beh between two dual-joining letters: medial.
        Expect("ب", "\uFE8F");
        Expect("ببب", "\uFE91\uFE92\uFE90");
        // Lam-alef: the isolated ligature alone, the final one after a joining letter; the ligature takes one place.
        Expect("لا", "\uFEFB");
        Expect("سلام", "\uFEB3\uFEFC\uFEE1");
        Expect("لأ لإ لآ", "\uFEF7 \uFEF9 \uFEF5");
        // Hamza never joins and keeps its own code point; a vowel mark is transparent (beh still joins across it).
        Expect("ءب", "\u0621\uFE8F");
        Expect("بَب", "\uFE91\u064E\uFE90");
        // Tatweel causes joining on both sides; text without Arabic is returned as it is.
        Expect("بـ", "\uFE91\u0640");
        Expect("idrak 0.1.7", "idrak 0.1.7");
        Check(ArabicShaping.JoiningType(0x0627) == 'R' && ArabicShaping.JoiningType(0x0628) == 'D' && ArabicShaping.JoiningType(0x064E) == 'T'
              && ArabicShaping.JoiningType('a') == 'U' && ArabicShaping.JoiningType(0x0640) == 'C', "joining types");
        Check(Bidi.ClassOf('a') == BidiClass.L && Bidi.ClassOf(0x0628) == BidiClass.AL && Bidi.ClassOf(0x05D0) == BidiClass.R && Bidi.ClassOf('1') == BidiClass.EN
              && Bidi.ClassOf(0x0661) == BidiClass.AN && Bidi.ClassOf(',') == BidiClass.CS && Bidi.ClassOf('+') == BidiClass.ES && Bidi.ClassOf('%') == BidiClass.ET
              && Bidi.ClassOf(0x064E) == BidiClass.NSM && Bidi.ClassOf(' ') == BidiClass.WS && Bidi.ClassOf('(') == BidiClass.ON && Bidi.ClassOf(0x200D) == BidiClass.BN
              && Bidi.ClassOf(0x2067) == BidiClass.RLI && Bidi.ClassOf(0x1E900) == BidiClass.R, "bidirectional classes");
        Check(Bidi.Mirror('(') == ')' && Bidi.Mirror('<') == '>' && Bidi.Mirror('«') == '»' && Bidi.Mirror('a') == 'a', "mirrored glyphs");
    }

    private static void CliArabicBidi(Device device)
    {
        _ = device;
        void Expect(string text, string visual, int level = -1) =>
            Check(Visual(text, level) == visual, $"visual({text}): {Codes(Visual(text, level))}, expected {Codes(visual)}");
        string Shaped(string word) => Reverse(ArabicShaping.Shape(word));

        // Left-to-right text alone is unchanged, in either paragraph direction (one run at level 0, or 2 inside 1).
        Expect("idrak doctor", "idrak doctor");
        Expect("idrak doctor", "idrak doctor", 1);
        // Arabic alone: shaped, then reversed (one run at level 1).
        Expect("مرحبا", "\uFE8E\uFE92\uFEA3\uFEAE\uFEE3");
        Expect("لا يوجد", Shaped("يوجد") + " " + "\uFEFB");
        // An English word inside Arabic (paragraph right to left from its first letter): the words go right to left, the
        // English one keeps its letters' order.
        Expect("الجهاز cpu جاهز", Shaped("جاهز") + " cpu " + Shaped("الجهاز"));
        // Numbers keep their order: European digits (W7 does not make them L after Arabic, so they are a level 2 run).
        Expect("عدد 123 ملف", Shaped("ملف") + " 123 " + Shaped("عدد"));
        Expect("النسخة 0.1.7", "0.1.7 " + Shaped("النسخة"));
        // A path (neutral separators between left-to-right letters resolve left to right, N1) stays whole.
        Expect("افتح C:\\models\\qwen.gguf الآن", Shaped("الآن") + " C:\\models\\qwen.gguf " + Shaped("افتح"));
        // Brackets around Arabic in a right-to-left run are reversed with it and mirrored (L4): "(" stays "(" on the left.
        Expect("(تجربة)", "(" + Shaped("تجربة") + ")");
        // Arabic in a left-to-right paragraph: only the Arabic run is reversed.
        Expect("idrak يعمل", "idrak " + Shaped("يعمل"));
        Expect("Error: لا يوجد", "Error: " + Shaped("يوجد") + " \uFEFB");
        // Trailing punctuation of a right-to-left sentence ends it on the left (eos is R).
        Expect("كل شيء يعمل.", "." + Shaped("يعمل") + " " + Shaped("شيء") + " " + Shaped("كل"));
        // A vowel mark follows its letter after reordering (L3), so the terminal puts it over the right letter.
        Expect("بَ", "\uFE8F\u064E");
        // Direction marks steer the algorithm and are dropped from the output; a left-to-right mark keeps "--device" whole.
        Expect("الخيار \u200E--device", "--device " + Shaped("الخيار"));
        Expect("الخيار --device", "device-- " + Shaped("الخيار"));
        // Arabic-Indic digits are Arabic numbers (AN): their order is kept too.
        Expect("رقم ١٢٣", "١٢٣ " + Shaped("رقم"));
        Check(Bidi.ParagraphLevel([BidiClass.WS, BidiClass.AL, BidiClass.L]) == 1 && Bidi.ParagraphLevel([BidiClass.EN, BidiClass.L]) == 0
              && Bidi.ParagraphLevel([BidiClass.RLI, BidiClass.L, BidiClass.PDI, BidiClass.R]) == 1 && Bidi.ParagraphLevel([BidiClass.ON]) == 0,
            "paragraph levels (P2, P3): first strong letter, isolates skipped, none means left to right");
        // Levels of a known sequence: R ON L EN in a right-to-left paragraph gives 1 1 2 2, and L2 reverses to 3 2 1 0.
        var levels = Bidi.Levels([BidiClass.R, BidiClass.WS, BidiClass.L, BidiClass.EN], ReadOnlySpan<int>.Empty, 1, out _);
        Check(levels.SequenceEqual(new sbyte[] { 1, 1, 2, 2 }) && Bidi.VisualOrder(levels).SequenceEqual([2, 3, 1, 0]), $"levels {string.Join(' ', levels)}");
        // Embedding controls are removed (X9): level -1, left out of the order.
        levels = Bidi.Levels([BidiClass.L, BidiClass.RLE, BidiClass.L, BidiClass.PDF], ReadOnlySpan<int>.Empty, 0, out _);
        Check(levels.SequenceEqual(new sbyte[] { 0, -1, 2, -1 }) && Bidi.VisualOrder(levels).SequenceEqual([0, 2]), $"levels with RLE {string.Join(' ', levels)}");
    }

    private static void CliArabicVectors(Device device)
    {
        _ = device;
        string path = Path.Combine(RepositoryRoot(), "tests", "Idrak.Tests", "data", "arabic", "visual-order.jsonl");
        int count = 0;
        var failed = new List<string>();
        foreach (string line in File.ReadLines(path))
        {
            var row = JsonNode.Parse(line)!;
            string text = (string)row["text"]!, direction = (string)row["dir"]!, expected = (string)row["visual"]!;
            string visual = Visual(text, direction switch { "ltr" => 0, "rtl" => 1, _ => -1 });
            if (visual != expected)
            {
                failed.Add($"[{direction}] {text}: {Codes(visual)}, expected {Codes(expected)}");
            }

            count++;
        }

        Check(count == 369, $"{count} vectors read");
        Check(failed.Count == 0, $"{failed.Count} lines differ from the reference, first: {failed.FirstOrDefault()}");
    }

    private static void CliArabicRenderer(Device device)
    {
        _ = device;
        string Shaped(string word) => Reverse(ArabicShaping.Shape(word));
        // Lines without right-to-left text come back unchanged (escape sequences too).
        Check(VisualText.Line("  -d, --device NAME   cpu, cuda:0") == "  -d, --device NAME   cpu, cuda:0", "an English line changed");
        Check(VisualText.Line("\u001b[32mok\u001b[0m  cache") == "\u001b[32mok\u001b[0m  cache", "colours on an English line changed");
        // Columns (two or more spaces) are paragraphs of their own and keep their start; indentation stays on the left.
        string line = VisualText.Line("  doctor (doc)    ما يعمل");
        Check(line == "  doctor (doc)    " + Shaped("يعمل") + " " + Shaped("ما"), $"help line: {line}");
        // A column that got shorter (lam-alef: two letters, one cell) is padded so the next column starts where it did.
        line = VisualText.Line("لا  x");
        Check(line == "\uFEFB   x", $"padded column: {Codes(line)}");
        // A table header in Arabic: each cell reversed in place, the cells left to right at their columns.
        line = VisualText.Line("الجهاز    الذاكرة");
        Check(line == Shaped("الجهاز") + "    " + Shaped("الذاكرة"), $"table header: {line}");
        // Colours stay on the characters they coloured.
        line = VisualText.Line("\u001b[32mسليم\u001b[0m  cpu");
        Check(line == "\u001b[0m\u001b[32m" + Shaped("سليم") + "\u001b[0m  cpu\u001b[0m", $"coloured line: {line.Replace("\u001b", "ESC")}");
        // Right alignment: one right-to-left paragraph ends at the given column.
        line = VisualText.Line("كل شيء يعمل.", 20);
        Check(line.Length == 20 && line.TrimStart() == Visual("كل شيء يعمل."), $"right-aligned: '{line}'");

        // The writer: English goes straight through (a prompt shows at once); a right-to-left line is held until its line
        // break, also across flushes (a streamed answer), and rendered as one paragraph.
        var inner = new StringWriter();
        var writer = new VisualWriter(inner);
        writer.Write("> ");
        Check(inner.ToString() == "> ", "the prompt was held");
        writer.Write("مر");
        writer.Flush();
        writer.Write("حبا");
        Check(inner.ToString() == "> ", "a right-to-left line was written before its line break");
        writer.WriteLine();
        Check(inner.ToString() == "> " + Visual("مرحبا") + "\n", $"streamed line: {Codes(inner.ToString())}");
        // A question renders its held line before waiting for the answer.
        inner.GetStringBuilder().Clear();
        writer.Write("هل تتابع؟ [y/N] ");
        writer.RenderPending();
        Check(inner.ToString().Contains(Shaped("تتابع"), StringComparison.Ordinal), "a question was not shown");
        // A line wider than the terminal is broken at spaces before it is reordered; rows after the first hang under it.
        inner = new StringWriter();
        writer = new VisualWriter(inner, width: 16);
        writer.WriteLine("ab  واحد اثنان ثلاثة");
        string[] rows = inner.ToString().TrimEnd('\n').Split('\n');
        Check(rows.Length == 2 && rows[0] == "ab  " + Shaped("اثنان") + " " + Shaped("واحد") && rows[1] == "    " + Shaped("ثلاثة"),
            $"wrapped rows: {string.Join(" | ", rows)}");
    }

    private static void CliArabicDetection(Device device)
    {
        _ = device;
        bool? Reorders(params (string Name, string Value)[] variables)
        {
            var map = variables.ToDictionary(v => v.Name, v => v.Value);
            return VisualText.TerminalReordersItself(name => map.GetValueOrDefault(name));
        }

        Check(Reorders(("VTE_VERSION", "7600")) == true && Reorders(("VTE_VERSION", "5200")) == false, "VTE from 0.58");
        Check(Reorders(("KONSOLE_VERSION", "230400")) == true && Reorders(("TERM", "mlterm")) == true && Reorders(("TERM_PROGRAM", "mintty")) == true
              && Reorders(("TERM_PROGRAM", "Apple_Terminal")) == true, "terminals that reorder text themselves");
        Check(Reorders(("TERM_PROGRAM", "vscode")) == false && Reorders(("WT_SESSION", "1b2c")) == false && Reorders(("TERM_PROGRAM", "iTerm.app")) == false,
            "terminals that do not");
        Check(Reorders(("TERM", "xterm-256color")) is null && Reorders() is null, "an unknown terminal says nothing");
        string? None(string name) => null;
        Check(VisualText.Resolve(TextRendering.Auto, terminal: true, None) == TextRendering.Visual, "auto on an unknown terminal: visual");
        Check(VisualText.Resolve(TextRendering.Auto, terminal: false, None) == TextRendering.Logical, "auto into a file or pipe: logical");
        Check(VisualText.Resolve(TextRendering.Auto, terminal: true, n => n == "KONSOLE_VERSION" ? "1" : null) == TextRendering.Logical, "auto on Konsole: logical");
        Check(VisualText.Resolve(TextRendering.Visual, terminal: false, None) == TextRendering.Visual
              && VisualText.Resolve(TextRendering.Logical, terminal: true, None) == TextRendering.Logical, "an explicit choice wins");
        Check(VisualText.Parse("visual-right", "--lang-render") == (TextRendering.Visual, true), "visual-right");
        Check(Messages.ParseLanguage("AR", "--lang") == "ar" && Messages.ParseLanguage("ar-EG", "--lang") == "ar" && Messages.ParseLanguage("english", "--lang") == "en",
            "language names");
    }

    private static void CliArabicCatalog(Device device)
    {
        _ = device;
        // Every message written at a call (Messages.T("...")) in the tool's sources.
        string root = Path.Combine(RepositoryRoot(), "src", "Idrak.Cli");
        var used = new HashSet<string>(StringComparer.Ordinal);
        foreach (string file in Directory.EnumerateFiles(root, "*.cs", SearchOption.AllDirectories).Where(f => !f.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}")))
        {
            foreach (Match m in Regex.Matches(File.ReadAllText(file), "Messages\\.T\\(\\s*\"((?:[^\"\\\\]|\\\\.)*)\""))
            {
                used.Add(Regex.Unescape(m.Groups[1].Value));
            }
        }

        Check(used.Count > 100, $"only {used.Count} messages found: the scan is broken");
        // Messages that reach Messages.T as values: command summaries, group titles and the two help blocks.
        var values = CommandTable.All.Select(c => c.Summary).Concat(CommandTable.Groups.Select(g => g.Title))
            .Append(Help.CommonOptions).Append(Idrak.Cli.Commands.Run.ChatSession.CommandHelp).Append("Other");
        used.UnionWith(values);
        var missing = used.Where(k => !Messages.Arabic.ContainsKey(k)).ToList();
        Check(missing.Count == 0, $"{missing.Count} messages have no Arabic text, first: {missing.FirstOrDefault()}");
        var unused = Messages.Arabic.Keys.Where(k => !used.Contains(k)).ToList();
        Check(unused.Count == 0, $"{unused.Count} Arabic entries are not used, first: {unused.FirstOrDefault()}");
        foreach (var (english, arabic) in Messages.Arabic)
        {
            // The same placeholders in both languages; Arabic text, without vowel marks, and as many lines.
            static string Placeholders(string text) => string.Join(',', Regex.Matches(text, @"\{(\d+)(:[^}]*)?\}").Select(m => m.Value).Order(StringComparer.Ordinal));
            Check(Placeholders(english) == Placeholders(arabic), $"placeholders differ: '{english}' / '{arabic}'");
            Check(arabic.Any(c => c is >= '\u0621' and <= '\u064A'), $"no Arabic letters in the translation of '{english}'");
            Check(!arabic.Any(c => c is >= '\u064B' and <= '\u065F'), $"vowel marks in the translation of '{english}'");
            Check(english.Count(c => c == '\n') == arabic.Count(c => c == '\n'), $"line count differs for '{english}'");
            Check(!arabic.Any(c => Bidi.ClassOf(c) is BidiClass.LRE or BidiClass.RLE or BidiClass.LRO or BidiClass.RLO or BidiClass.PDF),
                $"embedding controls in the translation of '{english}'");
        }

        // Every Arabic letter of the catalog has its presentation forms (so the renderer shapes all of it).
        var letters = Messages.Arabic.Values.SelectMany(v => v).Where(c => c is >= '\u0621' and <= '\u064A' and not '\u0621' and not '\u0640').Distinct().ToList();
        Check(letters.All(c => ArabicShaping.Shape(c.ToString())[0] >= '\uFE70'), "a letter of the catalog has no presentation form");
    }

    private static void CliArabicCommands(Device device)
    {
        _ = device;
        static bool HasArabic(string text) => text.Any(c => c is >= '\u0600' and <= '\u06FF' or >= '\uFB50' and <= '\uFEFC');
        static bool HasBaseLetters(string text) => text.Any(c => c is >= '\u0622' and <= '\u064A');

        // English by default: no Arabic anywhere, help as it was.
        var (exit, output, _) = HelpCli("help");
        Check(exit == 0 && !HasArabic(output) && output.StartsWith("idrak: Idrak's command-line tool", StringComparison.Ordinal), "English help changed");
        // Arabic, logical (the output is not a terminal): the title, groups and summaries in Arabic, the names in Latin.
        (exit, output, _) = HelpCli("help", "--lang", "ar");
        Check(exit == 0 && output.StartsWith("idrak: أداة سطر الأوامر", StringComparison.Ordinal), $"Arabic help title: {output.Split('\n')[0]}");
        Check(output.Contains("\nالإعداد والفحص:\n", StringComparison.Ordinal) && output.Contains("  doctor (doc) ", StringComparison.Ordinal)
              && output.Contains("خيارات عامة:", StringComparison.Ordinal), "Arabic help groups, names and common options");
        Check(output.Split('\n').All(l => l.Length <= Help.Width), "an Arabic help line is wider than the help width");
        // Visual: shaped (presentation forms, no base letters left) and reordered.
        (exit, output, _) = HelpCli("help", "--lang=ar", "--lang-render", "visual");
        Check(exit == 0 && HasArabic(output) && !HasBaseLetters(output) && !output.Contains('\u200E'), "visual help still has base letters or marks");
        Check(output.StartsWith("idrak: " + Visual("أداة سطر الأوامر لمكتبة إدراك"), StringComparison.Ordinal), $"visual help title: {output.Split('\n')[0]}");
        // A command's help: the summary and labels in Arabic, the body noted as English.
        (exit, output, _) = HelpCli("help", "doctor", "--lang", "ar");
        Check(exit == 0 && output.Contains("الاستخدام: \u200Eidrak doctor", StringComparison.Ordinal) && output.Contains("(بقية هذه المساعدة بالإنجليزية.)", StringComparison.Ordinal),
            "Arabic command help");
        // The environment: IDRAK_LANG; the config: "lang"; --lang wins over both.
        string config = Path.Combine(Path.GetTempPath(), $"idrak-cli-arabic-{Guid.NewGuid():N}.json");
        File.WriteAllText(config, "{\"lang\": \"ar\", \"lang-render\": \"visual\"}");
        var configured = new StringWriter();
        CommandLine.Run(["help", "-C", config], configured, new StringWriter());
        Check(HasArabic(configured.ToString()) && !HasBaseLetters(configured.ToString()), "the config's lang and lang-render");
        configured = new StringWriter();
        CommandLine.Run(["help", "-C", config, "--lang", "en"], configured, new StringWriter());
        Check(!HasArabic(configured.ToString()), "--lang en over the config");
        File.Delete(config);
        WithEnvironment(new Dictionary<string, string?> { ["IDRAK_LANG"] = "ar" }, () =>
        {
            var (_, text, _) = HelpCli("help");
            Check(HasArabic(text), "IDRAK_LANG=ar");
        });
        // Usage errors in Arabic; bad values are usage errors.
        var (code, _, error) = ArabicCli("doctor", "--lang", "ar", "--nonsense");
        Check(code == 2 && error.Contains("خيار غير معروف", StringComparison.Ordinal) && error.Contains("الاستخدام: \u200Eidrak doctor", StringComparison.Ordinal), $"Arabic usage error: {error}");
        Check(HelpCli("help", "--lang", "fr").Exit == 2 && HelpCli("help", "--lang", "ar", "--lang-render", "upside-down").Exit == 2, "bad --lang or --lang-render accepted");
        (code, _, error) = ArabicCli("nonsense-command", "--lang", "ar");
        Check(code == 2 && error.Contains("أمر غير معروف", StringComparison.Ordinal), "Arabic unknown command");

        // doctor and devices: text in Arabic; JSON and CSV never translated.
        (exit, output, _) = ArabicCli("doctor", "--lang", "ar");
        Check(exit is 0 or 1 && output.Contains("الإعدادات", StringComparison.Ordinal) && (output.Contains("كل شيء يعمل.", StringComparison.Ordinal)
              || output.Contains("كل ما يلزم يعمل", StringComparison.Ordinal) || output.Contains("فشل من الفحوص", StringComparison.Ordinal)), "Arabic doctor");
        (exit, output, _) = ArabicCli("doctor", "--lang", "ar", "-j");
        var json = JsonNode.Parse(output)!;
        Check(!HasArabic(output) && json["checks"]!.AsArray().Any(c => (string?)c!["name"] == "cache"), "doctor JSON was translated");
        (exit, output, _) = ArabicCli("devices", "--lang", "ar", "--lang-render", "visual");
        Check(exit == 0 && output.Contains(Visual("الذاكرة"), StringComparison.Ordinal) && output.Contains("cpu", StringComparison.Ordinal), "Arabic devices table");
        // The table's columns line up: every row's second column starts where the header's does.
        var starts = output.Split('\n').Where(l => l.Length > 0).Select(l => Regex.Match(l, @"^\S+\s{2,}").Length).Distinct().ToList();
        Check(starts.Count == 1 && starts[0] > 0, $"the Arabic devices table's second column starts at {string.Join(", ", starts)}");
        (_, output, _) = ArabicCli("devices", "--lang", "ar", "--format", "csv");
        Check(!HasArabic(output) && output.StartsWith("Device,Backend", StringComparison.Ordinal), "devices CSV was translated");
        (_, output, _) = ArabicCli("devices", "--lang", "ar", "--json");
        Check(!HasArabic(output) && JsonNode.Parse(output)!["devices"] is JsonArray, "devices JSON was translated");
        // The help topic.
        (exit, output, _) = HelpCli("help", "arabic");
        Check(exit == 0 && output.Contains("--lang-render", StringComparison.Ordinal) && output.Contains("visual-right", StringComparison.Ordinal), "idrak help arabic");
    }
}
