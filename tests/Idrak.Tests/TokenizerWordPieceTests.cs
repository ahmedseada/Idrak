// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Models;
using Idrak.Models.Abstractions;

// Plan 24, phase T2: WordPiece (BERT) against Hugging Face tokenizers.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] TokenizerWordPieceGroup =
    [
        ("tokenizers: wordpiece: both fixtures' tokenizer.json encode every corpus line to tokenizers' ids, its post-processor adds the same special tokens, type ids and sequence ids (single texts and pairs), and decoding gives its text with and without special tokens", TokenizersWordPieceReference),
        ("tokenizers: wordpiece: vocab.txt with tokenizer_config.json reads to the tokenizer BertConverter writes: the same ids and special tokens as tokenizer.json on every line", TokenizersWordPieceVocabTxt),
        ("tokenizers: wordpiece: a word longer than max_input_chars_per_word, or with no split, is [UNK]; MaxInputCharsPerWord and IDRAK_TOKENIZER_MAX_WORD_CHARS change the limit and bad values are refused", TokenizersWordPieceUnknown),
        ("tokenizers: wordpiece: lowercase is Rust's (İ becomes i and a combining dot) in Lowercase and BertNormalizer, string and span forms alike", TokenizersLowercaseDottedI),
        ("tokenizers: wordpiece: the pretrained tokenizer suite passes, a saved tokenizer loads back to the same ids, and a warm encode allocates nothing", TokenizersWordPieceSuite),
    ];

    private static string WordPieceFixture(string name) => TestData($"tokenizers/{name}");

    private static void TokenizersWordPieceReference(Device device)
    {
        foreach (string name in new[] { "wordpiece-uncased", "wordpiece-cased" })
        {
            // tokenizers' own decode, which never takes spaces out (the folder's tokenizer_config.json asks transformers to:
            // tokenizers: the decoded text's spaces).
            var tokenizer = PretrainedTokenizers.Load(WordPieceFixture(name), new TokenizerOptions { CleanUpSpaces = false });
            Check(tokenizer.Model == TokenizerModels.WordPiece && tokenizer is WordPieceTokenizer, $"{name}: the WordPiece model");
            var post = ((WordPieceTokenizer)tokenizer).PostProcessor ?? throw new Exception($"{name}: no post-processor");
            var reference = JsonNode.Parse(File.ReadAllText(Path.Combine(WordPieceFixture(name), "expected-hf.json")))!;
            int lines = 0;
            foreach (var line in reference["lines"]!.AsArray())
            {
                string text = (string)line!["text"]!;
                int[] expected = [.. line["ids"]!.AsArray().Select(i => (int)i!)];
                int[] mask = [.. line["special_tokens_mask"]!.AsArray().Select(i => (int)i!)];
                int[] plain = [.. expected.Where((_, i) => mask[i] == 0)];
                int[] ids = [.. tokenizer.Encode(text)];
                Check(ids.SequenceEqual(plain), $"{name}, line {lines}: '{text}' gives [{string.Join(", ", ids)}], tokenizers [{string.Join(", ", plain)}]");
                CheckProcessed(name, $"line {lines}", post.Process(TokenizerEncoding.Of(ids, new (int, int)[ids.Length]), null, addSpecialTokens: true), line);
                // tokenizers' "decoded" skips special tokens ([UNK] is one): the ids without them.
                string decoded = tokenizer.Decode(ids.Where(id => !tokenizer.IsSpecial(id)).ToArray());
                Check(decoded == (string)line["decoded"]!, $"{name}, line {lines}: decodes to '{decoded}', tokenizers '{line["decoded"]}'");
                Check(tokenizer.Decode(expected) == (string)line["decoded_with_special"]!,
                    $"{name}, line {lines}: with special tokens decodes to '{tokenizer.Decode(expected)}', tokenizers '{line["decoded_with_special"]}'");
                lines++;
            }

            int pairs = 0;
            foreach (var pair in reference["pairs"]!.AsArray())
            {
                int[] a = [.. tokenizer.Encode((string)pair!["text"]!)], b = [.. tokenizer.Encode((string)pair["pair"]!)];
                var joined = post.Process(TokenizerEncoding.Of(a, new (int, int)[a.Length]), TokenizerEncoding.Of(b, new (int, int)[b.Length], sequence: 1), addSpecialTokens: true);
                CheckProcessed(name, $"pair {pairs}", joined, pair);
                Check(tokenizer.Decode(joined.Ids) == (string)pair["decoded_with_special"]!, $"{name}, pair {pairs}: decodes to '{tokenizer.Decode(joined.Ids)}'");
                pairs++;
            }

            Check(lines == 102 && pairs > 0, $"{name}: {lines} lines and {pairs} pairs checked");
            Check(post.AddedTokens(false) == 2 && post.AddedTokens(true) == 3, $"{name}: [CLS] … [SEP] adds 2 tokens, a pair 3");
        }
    }

    // The post-processed ids, type ids, special-tokens mask and sequence ids against a tokenizers reference entry.
    private static void CheckProcessed(string fixture, string what, TokenizerEncoding encoding, JsonNode reference)
    {
        foreach (var (field, actual) in new[] { ("ids", encoding.Ids), ("type_ids", encoding.TypeIds), ("special_tokens_mask", encoding.SpecialTokensMask), ("sequence_ids", encoding.SequenceIds) })
        {
            int[] expected = [.. reference[field]!.AsArray().Select(i => (int)i!)];
            Check(actual.SequenceEqual(expected), $"{fixture}, {what}: {field} [{string.Join(", ", actual)}], tokenizers [{string.Join(", ", expected)}]");
        }
    }

    private static void TokenizersWordPieceVocabTxt(Device device)
    {
        foreach (string name in new[] { "wordpiece-uncased", "wordpiece-cased" })
        {
            string folder = WordPieceFixture(name);
            Check(PretrainedTokenizers.Find(folder) is { } found && found.Reader.Name == TokenizerFiles.TokenizerJson, $"{name}: tokenizer.json is read before vocab.txt");
            var json = PretrainedTokenizers.Load(folder);
            var vocab = PretrainedTokenizers.Load(folder, new TokenizerOptions { Format = TokenizerFiles.VocabTxt });
            Check(vocab is WordPieceTokenizer && vocab.VocabularySize == json.VocabularySize, $"{name}: vocab.txt reads to a WordPiece tokenizer of {json.VocabularySize} tokens");
            var post = ((WordPieceTokenizer)vocab).PostProcessor!;
            var reference = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "expected-hf.json")))!;
            int line = 0;
            foreach (var entry in reference["lines"]!.AsArray())
            {
                string text = (string)entry!["text"]!;
                int[] ids = [.. vocab.Encode(text)];
                Check(ids.SequenceEqual(json.Encode(text)), $"{name}, line {line}: vocab.txt gives other ids than tokenizer.json for '{text}'");
                CheckProcessed(name + " (vocab.txt)", $"line {line}", post.Process(TokenizerEncoding.Of(ids, new (int, int)[ids.Length]), null, true), entry);
                line++;
            }

            Check(vocab.IsSpecial(vocab.IdOf("[CLS]")!.Value) && vocab.IsSpecial(vocab.IdOf("[MASK]")!.Value), $"{name}: the special tokens are special");
            Check(vocab.Encode("[CLS] hello [MASK]") is var marked && marked[0] == vocab.IdOf("[CLS]") && marked[^1] == vocab.IdOf("[MASK]"),
                $"{name}: special tokens in the text are matched first");
        }
    }

    private static void TokenizersWordPieceUnknown(Device device)
    {
        string folder = WordPieceFixture("wordpiece-uncased");
        var tokenizer = PretrainedTokenizers.Load(folder);
        int unknown = tokenizer.IdOf("[UNK]")!.Value;
        Check(tokenizer.Encode(new string('a', 101)).SequenceEqual([unknown]), "a word of 101 characters is [UNK]");
        Check(tokenizer.Encode(new string('a', 100)).All(id => id != unknown), "a word of 100 characters is split");
        Check(tokenizer.Encode("\U00013000").SequenceEqual([unknown]), "a character the vocabulary lacks makes its word [UNK]");
        var short3 = PretrainedTokenizers.Load(folder, new TokenizerOptions { MaxInputCharsPerWord = 3 });
        Check(short3.Encode("the hello")[^1] == unknown && !short3.Encode("the").Contains(unknown), "MaxInputCharsPerWord 3: 'hello' is [UNK], 'the' is not");
        Expect<ArgumentOutOfRangeException>(() => PretrainedTokenizers.Load(folder, new TokenizerOptions { MaxInputCharsPerWord = 0 }), "MaxInputCharsPerWord", "a limit of 0");
        string? previous = Environment.GetEnvironmentVariable(TokenizerOptions.MaxWordCharsVariable);
        try
        {
            Environment.SetEnvironmentVariable(TokenizerOptions.MaxWordCharsVariable, "4");
            Check(PretrainedTokenizers.Load(folder).Encode("hello").SequenceEqual([unknown]), "IDRAK_TOKENIZER_MAX_WORD_CHARS 4: 'hello' is [UNK]");
            Environment.SetEnvironmentVariable(TokenizerOptions.MaxWordCharsVariable, "none");
            Expect<InvalidOperationException>(() => PretrainedTokenizers.Load(folder), "IDRAK_TOKENIZER_MAX_WORD_CHARS is 'none'", "a variable that is not a number");
        }
        finally
        {
            Environment.SetEnvironmentVariable(TokenizerOptions.MaxWordCharsVariable, previous);
        }
    }

    private static void TokenizersLowercaseDottedI(Device device)
    {
        Check(BpeTokenizer.NormalizerStep(JsonNode.Parse("""{"type": "Lowercase"}""")!.AsObject())("İSTANBUL ΣΑΣ") == "i\u0307stanbul σασ", "Lowercase (string)");
        using var output = new PreTokenizedText();
        new TokenizerPipeline(JsonNode.Parse("""{"normalizer": {"type": "Lowercase"}}""")!.AsObject()).Run("İSTANBUL", 0, 8, true, output);
        Check(new string(output.Piece(0)) == "i\u0307stanbul" && output.Text.OriginOf(0) == (0, 1) && output.Text.OriginOf(1) == (0, 1),
            "Lowercase (span): both characters of İ's lowercase stand for it");
        var bert = JsonNode.Parse("""{"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true, "strip_accents": false, "lowercase": true}""")!.AsObject();
        Check(BpeTokenizer.NormalizerStep(bert)("İ\u200Bx\u0085y\u2028z中") == "i\u0307xy z 中 ", "BertNormalizer: İ lowercased, format and control characters dropped, whitespace a space, spaces around 中");
    }

    private static void TokenizersWordPieceSuite(Device device)
    {
        string folder = WordPieceFixture("wordpiece-uncased");
        var tokenizer = PretrainedTokenizers.Load(folder);
        Conformance.CheckPretrained(tokenizer).ThrowIfFailed();
        Conformance.CheckPretrained(PretrainedTokenizers.Load(WordPieceFixture("wordpiece-cased"))).ThrowIfFailed();

        var temporary = Directory.CreateTempSubdirectory("idrak-wordpiece-");
        try
        {
            tokenizer.Save(temporary.FullName);
            var loaded = PretrainedTokenizers.Load(temporary.FullName);
            string corpus = File.ReadAllText(TestData("tokenizers/corpus.txt"));
            Check(loaded.Model == TokenizerModels.WordPiece && loaded.Encode(corpus).SequenceEqual(tokenizer.Encode(corpus)), "a saved tokenizer loads back to the same ids");
        }
        finally
        {
            temporary.Delete(recursive: true);
        }

        // Text BertNormalizer leaves as it is apart from case and spacing (no accents to strip through the platform's
        // normalization): a warm encode into a reused list allocates nothing.
        string[] texts = ["Hello, World! This is a test of 12345 things.", "مرحبا بالعالم، كيف حالك؟", "中文字符和标点。", "THE QUICK BROWN FOX"];
        var ids = new List<int>(4096);
        for (int warm = 0; warm < 50; warm++)
        {
            foreach (string text in texts)
            {
                ids.Clear();
                tokenizer.EncodeRange(text, 0, text.Length, ids);
            }
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int run = 0; run < 100; run++)
        {
            foreach (string text in texts)
            {
                ids.Clear();
                tokenizer.EncodeRange(text, 0, text.Length, ids);
            }
        }

        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Check(allocated == 0, $"a warm WordPiece encode allocated {allocated} bytes over 400 texts");
    }
}
