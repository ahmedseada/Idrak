// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Models;
using Idrak.Models.Abstractions;

// Plan 24, phase L1: the span pipeline (normalizers and pre-tokenizers on spans, with each character's origin).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] TokenizerPipelineGroup =
    [
        ("tokenizers: pipeline: every built-in normalizer and pre-tokenizer on spans gives the pieces its string form gives (every fixture's tokenizer.json and hand-made specs, the suite's texts and both corpora, at the start and not)", TokenizerPipelineEqualsStrings),
        ("tokenizers: pipeline: each piece stands for its range of the original text (splits in order, a metaspace's replacement for its space, a prefix for the first character, NFKC and lowercase per character, a range inside a longer text)", TokenizerPipelineOrigins),
        ("tokenizers: pipeline: an app's string-only normalizer and pre-tokenizer run in the span pipeline with their results, Strip, WhitespaceSplit and Punctuation load in a BPE tokenizer.json, and the post-processor table lists the built-ins and registers, guards and unregisters an app's", TokenizerPipelineAppComponents),
        ("tokenizers: pipeline: a warm run allocates nothing on text normalization leaves as it is (NFKC, Replace, Lowercase, Strip, Metaspace, Split and ByteLevel on Latin, Arabic and CJK text); text NFKC changes allocates nothing either (the library's own normalization)", TokenizerPipelineAllocations),
        ("tokenizers: pipeline: the BERT one pass (BertNormalizer and BertPreTokenizer) writes the normalized text, every character's range and the pieces the steps write, for every BertNormalizer setting and none, on the pipeline texts and hand-made ones; text it leaves to the steps is theirs; a warm pass allocates nothing", TokenizerPipelineBertOnePass),
    ];

    // Characters each BertNormalizer step treats specially, alone and in words: controls and format characters, whitespace
    // of every kind (a compatibility space NFD changes), CJK and compatibility ideographs, accents precomposed and
    // combining, Arabic marks in and out of canonical order, a spacing mark, Hangul, U+0130, a surrogate pair, a lone one.
    private static readonly string[] BertTexts =
    [
        "Hello, World! It's 3.14 -- \"quoted\" (a+b)=c; x^2 ~ y|z `code` $5 <tag> [x] {y} #1 @me 100%",
        "tab\there\nline\r\nend \u000b\u000c vertical \u0085next \u00a0nbsp \u2000quad\u2003em \u3000ideographic \u2028\u2029",
        "ctrl\u0001x\u001fy zero\u200bwidth \u00adsoft\ufeffbom \u0000nul \ufffdrep \ue000private \u001c\u001d\u001e",
        "中文字符和标点。日本語のテキスト 한국어 텍스트 \uf900\uf901 豈 \U0002F800 \u2f80",
        "Café naïve résumé Ελληνικά ΣΊΣΥΦΟΣ İstanbul ǅ ß ﬁ ＦＵＬＬ ①",
        "cafe\u0301 nai\u0308ve a\u0300\u0301 e\u0327\u0301 o\u031b\u0323",
        "اَلْعِلْمُ نُورٌ، كَتَبَ الطَّالِبُ؛ ب\u0651\u064e ب\u064e\u0651 ب\u0670 ١٢٣ ۱۲۳ ـــ",
        "देवनागरी क्षत्रिय की ि ा ঔ ক্ষ \u0903\u093e",
        "emoji 😀 👍🏽 ok 𝒳 𐐀𐐨 end",
        "lone \ud800 high and \udc00 low",
        "", " ", "a", ".", "  hello  world  ", "x\u0301", "\u0301x", "\u064e", "İ", "ı I i",
    ];

    private static void TokenizerPipelineBertOnePass(Device device)
    {
        _ = device;
        string[] texts = [.. PipelineTexts(), .. BertTexts];
        var specs = new List<JsonObject> { new() { ["normalizer"] = null, ["pre_tokenizer"] = new JsonObject { ["type"] = "BertPreTokenizer" } } };
        for (int setting = 0; setting < 16; setting++)
        {
            specs.Add(new JsonObject
            {
                ["normalizer"] = new JsonObject
                {
                    ["type"] = "BertNormalizer", ["clean_text"] = (setting & 1) != 0, ["handle_chinese_chars"] = (setting & 2) != 0,
                    ["strip_accents"] = (setting & 4) != 0, ["lowercase"] = (setting & 8) != 0,
                },
                ["pre_tokenizer"] = new JsonObject { ["type"] = "BertPreTokenizer" },
            });
        }

        specs.Add(new JsonObject
        {
            ["normalizer"] = new JsonObject { ["type"] = "BertNormalizer", ["lowercase"] = true, ["strip_accents"] = null },
            ["pre_tokenizer"] = new JsonObject { ["type"] = "BertPreTokenizer" },
        });
        using var fused = new PreTokenizedText();
        using var steps = new PreTokenizedText();
        foreach (var spec in specs)
        {
            var onePass = BertComponents.OnePass.Of(spec);
            Check(onePass is not null, $"{spec.ToJsonString()}: no one pass");
            var pipeline = new TokenizerPipeline(spec, onePass: false);
            int taken = 0;
            foreach (string text in texts)
            {
                foreach (var (start, length) in new[] { (0, text.Length), (Math.Min(2, text.Length), Math.Max(0, text.Length - 3)) })
                {
                    if (!onePass!.TryRun(text, start, length, fused))
                    {
                        continue;
                    }

                    taken++;
                    pipeline.Run(text, start, length, true, steps);
                    string what = $"{spec["normalizer"]?.ToJsonString() ?? "no normalizer"}: '{Escaped(text[start..(start + length)])}'";
                    Check(fused.Text.Span.SequenceEqual(steps.Text.Span), $"{what}: one pass '{Escaped(fused.Text.ToString())}', the steps '{Escaped(steps.Text.ToString())}'");
                    for (int i = 0; i < steps.Text.Length; i++)
                    {
                        Check(fused.Text.OriginOf(i) == steps.Text.OriginOf(i), $"{what}: character {i} stands for {fused.Text.OriginOf(i)}, the steps' for {steps.Text.OriginOf(i)}");
                    }

                    Check(fused.Pieces.SequenceEqual(steps.Pieces), $"{what}: pieces [{string.Join(" | ", fused.PieceStrings())}], the steps' [{string.Join(" | ", steps.PieceStrings())}]");
                    Check(fused.Text.Original == steps.Text.Original && fused.Text.OriginalStart == steps.Text.OriginalStart && fused.Text.OriginalEnd == steps.Text.OriginalEnd,
                        $"{what}: the original range differs");
                }
            }

            Console.WriteLine($"    {spec["normalizer"]?.ToJsonString() ?? "no normalizer"}: the one pass took {taken} of {texts.Length * 2} texts");
            Check(taken * 2 >= texts.Length, $"{spec.ToJsonString()}: the one pass took only {taken} of {texts.Length * 2} texts");
        }

        // The corpus through the default uncased BERT normalizer: most lines take the one pass, and a warm pass allocates nothing.
        var uncased = BertComponents.OnePass.Of(specs[^1])!;
        var lines = TokenizerCorpusLines();
        int lineCount = lines.Count(line => uncased.TryRun(line, 0, line.Length, fused));
        Console.WriteLine($"    uncased BERT pipeline: the one pass takes {lineCount} of {lines.Length} corpus lines");
        Check(lineCount * 2 > lines.Length, $"the one pass took {lineCount} of {lines.Length} corpus lines");
        var bert = new TokenizerPipeline(specs[^1]);
        foreach (string line in lines)
        {
            bert.Run(line, 0, line.Length, true, fused);
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        foreach (string line in lines)
        {
            bert.Run(line, 0, line.Length, true, fused);
        }

        Check(GC.GetAllocatedBytesForCurrentThread() == before, $"a warm pass allocated {GC.GetAllocatedBytesForCurrentThread() - before} bytes");
    }

    // The texts every pipeline check runs on: the suite's, both corpora.
    private static string[] PipelineTexts()
    {
        var baseline = JsonNode.Parse(File.ReadAllText(TestData("tokenizers/bpe-baseline.json")))!;
        return [.. baseline["texts"]!.AsArray().Select(t => (string)t!), .. File.ReadAllLines(PretrainData("corpus.txt")),
            .. File.ReadAllText(TestData("tokenizers/corpus.txt")).Split('\n')];
    }

    // Hand-made specs for the types and settings the fixtures do not use.
    private static readonly string[] HandNormalizers =
    [
        """{"type": "NFKC"}""", """{"type": "NFD"}""", """{"type": "NFKD"}""", """{"type": "NFC"}""", """{"type": "Lowercase"}""",
        """{"type": "Strip", "strip_left": true, "strip_right": true}""", """{"type": "Strip", "strip_left": false, "strip_right": true}""",
        """{"type": "Prepend", "prepend": "▁"}""", """{"type": "Replace", "pattern": {"String": " "}, "content": "▁"}""",
        """{"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "}""", """{"type": "Replace", "pattern": {"Regex": "a*"}, "content": "-"}""",
        """{"type": "Replace", "pattern": {"Regex": "(\\w)(\\w)"}, "content": "$2$1"}""", """{"type": "Replace", "pattern": {"String": "ab"}, "content": ""}""",
        """{"type": "Sequence", "normalizers": [{"type": "NFKD"}, {"type": "Replace", "pattern": {"Regex": "\\p{Mn}"}, "content": ""}, {"type": "Lowercase"}, {"type": "Strip", "strip_left": true, "strip_right": true}]}""",
    ];

    private static readonly string[] HandPreTokenizers =
    [
        """{"type": "Whitespace"}""", """{"type": "WhitespaceSplit"}""", """{"type": "Punctuation"}""", """{"type": "Punctuation", "behavior": "Removed"}""",
        """{"type": "Digits", "individual_digits": true}""", """{"type": "Digits", "individual_digits": false}""",
        """{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": true}""", """{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "never", "split": false}""",
        """{"type": "Metaspace", "replacement": "▁", "add_prefix_space": true}""",
        """{"type": "ByteLevel", "add_prefix_space": true, "use_regex": true}""", """{"type": "ByteLevel", "add_prefix_space": true, "use_regex": false}""",
        """{"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}""",
        """{"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithNext", "invert": false}""",
        """{"type": "Split", "pattern": {"Regex": "\\p{L}+"}, "behavior": "MergedWithPrevious", "invert": false}""",
        """{"type": "Split", "pattern": {"Regex": "\\s+"}, "behavior": "Removed", "invert": true}""",
        """{"type": "Sequence", "pretokenizers": [{"type": "WhitespaceSplit"}, {"type": "Punctuation"}, {"type": "Digits", "individual_digits": true}]}""",
        """{"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": "\\p{N}{1,3}"}, "behavior": "Isolated", "invert": false}, {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true}]}""",
    ];

    private static void TokenizerPipelineEqualsStrings(Device device)
    {
        var specs = new List<(string Source, JsonObject? Normalizer, JsonObject? PreTokenizer)>();
        foreach (string file in Directory.EnumerateFiles(TestData(""), "tokenizer.json", SearchOption.AllDirectories).Order(StringComparer.Ordinal))
        {
            var json = JsonNode.Parse(File.ReadAllText(file))!.AsObject();
            specs.Add((file, json["normalizer"] as JsonObject, json["pre_tokenizer"] as JsonObject));
        }

        specs.AddRange(HandNormalizers.Select(n => ("hand normalizer", (JsonObject?)JsonNode.Parse(n)!.AsObject(), (JsonObject?)null)));
        specs.AddRange(HandPreTokenizers.Select(p => ("hand pre-tokenizer", (JsonObject?)null, (JsonObject?)JsonNode.Parse(p)!.AsObject())));
        string[] texts = PipelineTexts();
        int ran = 0, skipped = 0;
        using var output = new PreTokenizedText();
        foreach (var (source, normalizer, preTokenizer) in specs)
        {
            Func<string, string>? normalize;
            Func<List<string>, bool, List<string>>? preTokenize;
            TokenizerPipeline pipeline;
            try
            {
                normalize = normalizer is null ? null : BpeTokenizer.NormalizerStep(normalizer);
                preTokenize = preTokenizer is null ? null : BpeTokenizer.PreTokenizerStage(preTokenizer);
                pipeline = new TokenizerPipeline(new JsonObject { ["normalizer"] = normalizer?.DeepClone(), ["pre_tokenizer"] = preTokenizer?.DeepClone() });
            }
            catch (NotSupportedException)
            {
                skipped++;                                                         // a type later phases build (Precompiled, Bert*)
                continue;
            }

            ran++;
            foreach (string text in texts)
            {
                foreach (bool atStart in new[] { true, false })
                {
                    List<string>? expected = null;
                    Exception? refused = null;
                    try
                    {
                        string normalized = normalize is null ? text : normalize(text);
                        expected = preTokenize is null ? [normalized] : preTokenize([normalized], atStart);
                    }
                    catch (ArgumentException e)
                    {
                        refused = e;                                               // a lone surrogate under Unicode normalization
                    }

                    try
                    {
                        pipeline.Run(text, 0, text.Length, atStart, output);
                        Check(refused is null, $"{source}: the span pipeline accepted what the string form refused ({refused?.Message}) on '{text}'");
                    }
                    catch (ArgumentException) when (refused is not null)
                    {
                        continue;
                    }

                    var pieces = output.PieceStrings();
                    Check(pieces.SequenceEqual(expected!), $"{source} ({normalizer?.ToJsonString()} {preTokenizer?.ToJsonString()}), at start {atStart}: "
                        + $"'{text}' gives [{string.Join(" | ", pieces)}], the string form [{string.Join(" | ", expected!)}]");
                    for (int i = 0; i < output.Pieces.Count; i++)
                    {
                        var (start, end) = output.OriginOf(i);
                        Check(start >= 0 && start <= end && end <= text.Length, $"{source}: piece {i} of '{text}' stands for ({start}, {end}), outside the text");
                    }
                }
            }
        }

        Check(ran >= 25, $"{ran} pipelines compared ({skipped} with types of later phases)");
    }

    private static void TokenizerPipelineOrigins(Device device)
    {
        using var output = new PreTokenizedText();
        List<(string Piece, int Start, int End)> Run(string normalizer, string preTokenizer, string text, int start = 0, int? length = null, bool atStart = true)
        {
            var json = new JsonObject
            {
                ["normalizer"] = normalizer.Length == 0 ? null : JsonNode.Parse(normalizer),
                ["pre_tokenizer"] = preTokenizer.Length == 0 ? null : JsonNode.Parse(preTokenizer),
            };
            new TokenizerPipeline(json).Run(text, start, length ?? text.Length - start, atStart, output);
            return [.. Enumerable.Range(0, output.Pieces.Count).Select(i => (new string(output.Piece(i)), output.OriginOf(i).Start, output.OriginOf(i).End))];
        }

        var words = Run("", """{"type": "WhitespaceSplit"}""", "  hello   big world ");
        Check(words.SequenceEqual([("hello", 2, 7), ("big", 10, 13), ("world", 14, 19)]), $"splits keep their ranges: {string.Join(", ", words)}");

        var meta = Run("""{"type": "Lowercase"}""", """{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true}""", "Hello World");
        Check(meta.SequenceEqual([("▁hello", 0, 5), ("▁world", 5, 11)]), $"a prefix stands for the first character, a replaced space for the space: {string.Join(", ", meta)}");

        // "ﬁ" (one character) becomes "fi" under NFKC: both stand for it; "e" and a combining acute composed stand for the
        // "e" alone, as tokenizers aligns them.
        var nfkc = Run("""{"type": "NFKC"}""", """{"type": "Whitespace"}""", "\uFB01ne cafe\u0301");
        Check(nfkc.SequenceEqual([("fine", 0, 3), ("caf\u00E9", 4, 8)]), $"NFKC aligns as tokenizers does: {string.Join(", ", nfkc)}");

        // Each normalized character's range, against tokenizers 0.23.2 (its unicode_normalization_alignments): a source
        // character's first decomposed character stands for the next source character, the others for the one before,
        // a composition passes over what it absorbed, all by position after canonical reordering.
        foreach (var (form, text, expected) in new (string, string, (int, int)[])[]
                 {
                     ("NFKC", "\uFB01ne cafe\u0301", [(0, 1), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]),
                     ("NFC", "e\u0301\u0301x", [(0, 1), (2, 3), (3, 4)]),
                     ("NFC", "\u0344a", [(0, 1), (0, 1), (1, 2)]),
                     ("NFC", "a\u0301\u0328b", [(0, 1), (2, 3), (3, 4)]),
                     ("NFKC", "\u1100\u1161\u11A8", [(0, 1)]),
                     ("NFD", "\uFB01ne cafe\u0301", [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)]),
                     ("NFD", "a\u0301\u0328b", [(0, 1), (1, 2), (2, 3), (3, 4)]),
                     ("NFD", "\uAC01", [(0, 1), (0, 1), (0, 1)]),
                     ("NFKD", "\uFB01ne", [(0, 1), (0, 1), (1, 2), (2, 3)]),
                     ("NFKD", "\u0344a", [(0, 1), (0, 1), (1, 2)]),
                 })
        {
            using var normalized = new NormalizedText();
            normalized.Reset(text);
            SpanComponents.BuiltInNormalizer(new JsonObject { ["type"] = form })(normalized);
            var origins = Enumerable.Range(0, normalized.Span.Length).Select(i => normalized.OriginOf(i)).ToArray();
            Check(origins.SequenceEqual(expected), $"{form} of {string.Join(" ", text.Select(c => $"U+{(int)c:X4}"))}: [{string.Join(", ", origins)}], tokenizers [{string.Join(", ", expected)}]");
        }

        // Replace's content stands for the occurrence's last character, as tokenizers' NormalizedString.replace aligns it.
        foreach (var (pattern, text, expected) in new (string, string, (int, int)[])[]
                 {
                     ("""{"Regex": " {2,}"}""", "a   b", [(0, 1), (3, 4), (4, 5)]),
                     ("""{"String": "ab"}""", "xaby", [(0, 1), (2, 3), (2, 3), (3, 4)]),
                 })
        {
            using var normalized = new NormalizedText();
            normalized.Reset(text);
            SpanComponents.BuiltInNormalizer(JsonNode.Parse($$"""{"type": "Replace", "pattern": {{pattern}}, "content": "{{(pattern.Contains("Regex", StringComparison.Ordinal) ? "▁" : "--")}}"}""")!.AsObject())(normalized);
            var origins = Enumerable.Range(0, normalized.Span.Length).Select(i => normalized.OriginOf(i)).ToArray();
            Check(origins.SequenceEqual(expected), $"Replace {pattern} in '{text}': [{string.Join(", ", origins)}], tokenizers [{string.Join(", ", expected)}]");
        }

        var range = Run("", """{"type": "Whitespace"}""", "xx alpha beta yy", start: 3, length: 10);
        Check(range.SequenceEqual([("alpha", 3, 8), ("beta", 9, 13)]), $"a range inside a longer text gives offsets in the whole text: {string.Join(", ", range)}");

        var bytes = Run("", """{"type": "ByteLevel", "add_prefix_space": true, "use_regex": true}""", "é!");
        Check(bytes.SequenceEqual([("ĠÃ©", 0, 1), ("!", 1, 2)]), $"byte characters stand for the character they are bytes of: {string.Join(", ", bytes)}");

        var stripped = Run("""{"type": "Strip", "strip_left": true, "strip_right": true}""", "", "  a b  ");
        Check(stripped.SequenceEqual([("a b", 2, 5)]), $"Strip keeps the inner range: {string.Join(", ", stripped)}");

        // Inside one text the ranges of consecutive split pieces never go back.
        var arabic = Run("""{"type": "NFKC"}""", """{"type": "Sequence", "pretokenizers": [{"type": "WhitespaceSplit"}, {"type": "Punctuation"}]}""", "مَرْحَبًا، يا عالم!");
        for (int i = 1; i < arabic.Count; i++)
        {
            Check(arabic[i].Start >= arabic[i - 1].End, $"pieces in order: {string.Join(", ", arabic)}");
        }
    }

    private static void TokenizerPipelineAppComponents(Device device)
    {
        using var output = new PreTokenizedText();
        try
        {
            TokenizerComponents.RegisterNormalizer("TestUpper", spec => new UpperNormalizer());
            TokenizerComponents.RegisterPreTokenizer("TestPairs", spec => new PairsPreTokenizer());
            TokenizerComponents.RegisterPreTokenizer("TestEcho", spec => new EchoPreTokenizer());
            var pipeline = new TokenizerPipeline(JsonNode.Parse("""
                {"normalizer": {"type": "TestUpper"}, "pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "WhitespaceSplit"}, {"type": "TestPairs"}]}}
                """)!.AsObject());
            pipeline.Run("abc de", 0, 6, true, output);
            Check(output.PieceStrings().SequenceEqual(["AB", "C", "DE"]), $"the app's components run: {string.Join(" | ", output.PieceStrings())}");
            Check(Enumerable.Range(0, 3).All(i => output.OriginOf(i) == (0, 6)), "a string-only normalizer's change stands for the whole changed text (coarse)");

            new TokenizerPipeline(JsonNode.Parse("""{"pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "WhitespaceSplit"}, {"type": "TestPairs"}]}}""")!.AsObject())
                .Run("abc de", 0, 6, true, output);
            Check(output.OriginOf(0) == (0, 2) && output.OriginOf(1) == (2, 3) && output.OriginOf(2) == (4, 6), "a string-only split keeps the ranges it was found at");

            new TokenizerPipeline(JsonNode.Parse("""{"pre_tokenizer": {"type": "TestEcho"}}""")!.AsObject()).Run("one two", 0, 7, true, output);
            Check(output.PieceStrings().SequenceEqual(["<one two>"]) && output.OriginOf(0) == (0, 7), "a rewritten piece stands for the piece it came from");
        }
        finally
        {
            TokenizerComponents.UnregisterNormalizer("TestUpper");
            TokenizerComponents.UnregisterPreTokenizer("TestPairs");
            TokenizerComponents.UnregisterPreTokenizer("TestEcho");
        }

        // The new built-in types load in a BPE tokenizer.json and split as the string pipeline does.
        var json = JsonNode.Parse(File.ReadAllText(PretrainData("tokenizer/tokenizer.json")))!.AsObject();
        json["normalizer"] = JsonNode.Parse("""{"type": "Strip", "strip_left": true, "strip_right": true}""");
        json["pre_tokenizer"] = JsonNode.Parse("""{"type": "Sequence", "pretokenizers": [{"type": "WhitespaceSplit"}, {"type": "Punctuation"}, {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}]}""");
        var tokenizer = BpeTokenizer.FromJson(json);
        var ids = tokenizer.Encode("  Hello, world!  ");
        var pieces = new[] { "Hello", ",", "world", "!" }.SelectMany(p => BpeTokenizer.FromJson(json).Encode(p)).ToArray();
        Check(ids.SequenceEqual(pieces), "Strip, WhitespaceSplit and Punctuation run in a BPE tokenizer");

        // The post-processor table: the built-in types (T2), an app's type registered, guarded and removed.
        Check(new[] { "TemplateProcessing", "BertProcessing", "RobertaProcessing", "ByteLevel", "Sequence" }.All(TokenizerComponents.PostProcessorTypes.Contains),
            $"the built-in post-processors (registered: {string.Join(", ", TokenizerComponents.PostProcessorTypes)})");
        TokenizerComponents.RegisterPostProcessor("TestWrap", spec => new WrapPostProcessor((int)spec["id"]!));
        try
        {
            var processor = TokenizerComponents.FindPostProcessor("TestWrap")!(JsonNode.Parse("""{"type": "TestWrap", "id": 9}""")!.AsObject());
            var single = TokenizerEncoding.Of([5, 6], [(0, 1), (1, 2)]);
            var wrapped = processor.Process(single, null, addSpecialTokens: true);
            Check(processor.AddedTokens(pair: false) == 2 && wrapped.Ids.SequenceEqual([9, 5, 6, 9]) && wrapped.SpecialTokensMask.SequenceEqual([1, 0, 0, 1]),
                "the app's post-processor runs");
            Check(TokenizerComponents.PostProcessorOrigin("TestWrap") is { } origin && origin != Overrides.Library, "it is the app's");
        }
        finally
        {
            TokenizerComponents.UnregisterPostProcessor("TestWrap");
        }

        Check(!TokenizerComponents.PostProcessorTypes.Contains("TestWrap"), "unregistered");
    }

    private sealed class UpperNormalizer : ITokenizerNormalizer
    {
        public string Normalize(string text) => text.ToUpperInvariant();
    }

    // Cuts each piece into pairs of characters.
    private sealed class PairsPreTokenizer : IPreTokenizer
    {
        public IReadOnlyList<string> PreTokenize(IReadOnlyList<string> pieces, bool atStart) =>
            [.. pieces.SelectMany(p => Enumerable.Range(0, (p.Length + 1) / 2).Select(i => p.Substring(i * 2, Math.Min(2, p.Length - i * 2))))];
    }

    // Rewrites each piece.
    private sealed class EchoPreTokenizer : IPreTokenizer
    {
        public IReadOnlyList<string> PreTokenize(IReadOnlyList<string> pieces, bool atStart) => [.. pieces.Select(p => $"<{p}>")];
    }

    private sealed class WrapPostProcessor(int id) : ITokenizerPostProcessor
    {
        public int AddedTokens(bool pair) => pair ? 3 : 2;

        public TokenizerEncoding Process(TokenizerEncoding encoding, TokenizerEncoding? pair, bool addSpecialTokens) => new()
        {
            Ids = [id, .. encoding.Ids, id], TypeIds = [0, .. encoding.TypeIds, 0], Offsets = [(0, 0), .. encoding.Offsets, (0, 0)],
            AttentionMask = [1, .. encoding.AttentionMask, 1], SpecialTokensMask = [1, .. encoding.SpecialTokensMask, 1],
            SequenceIds = [-1, .. encoding.SequenceIds, -1],
        };
    }

    private static void TokenizerPipelineAllocations(Device device)
    {
        string[] texts = ["Hello, World!  This is a test of 12345 things.", "مرحبا بالعالم، كيف حالك؟", "中文字符和标点。日本語のテキスト", "café 😀 naïve"];
        string[] changing = ["مَرْحَبًا بالعالم", "ﬁne Ｆｕｌｌ ｗｉｄｔｈ cafe\u0301"];
        string[] pipelines =
        [
            """{"normalizer": {"type": "Sequence", "normalizers": [{"type": "NFKC"}, {"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "}, {"type": "Lowercase"}, {"type": "Strip", "strip_left": true, "strip_right": true}]}, "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true}}""",
            """{"normalizer": null, "pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": "\\p{N}{1,3}"}, "behavior": "Isolated", "invert": false}, {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true}]}}""",
            """{"normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}, "pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "WhitespaceSplit"}, {"type": "Punctuation"}, {"type": "Digits", "individual_digits": true}]}}""",
        ];
        using var output = new PreTokenizedText();
        foreach (string spec in pipelines)
        {
            var pipeline = new TokenizerPipeline(JsonNode.Parse(spec)!.AsObject());
            for (int warm = 0; warm < 50; warm++)
            {
                foreach (string text in texts)
                {
                    pipeline.Run(text, 0, text.Length, true, output);
                }
            }

            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int run = 0; run < 100; run++)
            {
                foreach (string text in texts)
                {
                    pipeline.Run(text, 0, text.Length, true, output);
                }
            }

            long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            Check(allocated == 0, $"a warm run allocated {allocated} bytes over 400 texts: {spec}");
        }

        // Text NFKC changes: the library's own normalization (UnicodeNormalization) allocates nothing either.
        var nfkc = new TokenizerPipeline(JsonNode.Parse(pipelines[0])!.AsObject());
        for (int warm = 0; warm < 50; warm++)
        {
            foreach (string text in changing)
            {
                nfkc.Run(text, 0, text.Length, true, output);
            }
        }

        long start = GC.GetAllocatedBytesForCurrentThread();
        for (int run = 0; run < 100; run++)
        {
            foreach (string text in changing)
            {
                nfkc.Run(text, 0, text.Length, true, output);
            }
        }

        double perText = (GC.GetAllocatedBytesForCurrentThread() - start) / 200.0;
        Console.WriteLine($"    text NFKC changes: {perText:F0} bytes a text");
        Check(perText == 0, $"text NFKC changes allocated {perText:F0} bytes a text; the library's normalization allocates nothing");
    }
}
