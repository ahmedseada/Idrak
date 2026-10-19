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
        ("tokenizers: pipeline: a warm run allocates nothing on text normalization leaves as it is (NFKC, Replace, Lowercase, Strip, Metaspace, Split and ByteLevel on Latin, Arabic and CJK text); text NFKC changes costs only the platform's normalization calls", TokenizerPipelineAllocations),
    ];

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

        // Text NFKC changes: each changed cluster and the whole text take one call into the platform's normalization, which
        // allocates (72 bytes a call on Linux with ICU); nothing else does.
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
        Console.WriteLine($"    text NFKC changes: {perText:F0} bytes a text (the platform's normalization calls)");
        Check(perText <= 2048, $"text NFKC changes allocated {perText:F0} bytes a text, more than its normalization calls cost");
    }
}
