// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Models;
using Idrak.Models.Abstractions;

// Plan 24, phase T3: detailed encodings (offsets, type ids, masks, pairs, truncation, padding, batches) against tokenizers.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] TokenizerEncodingGroup =
    [
        ("tokenizers: encoding: BPE (byte level, Metaspace, Qwen's split, Gemma's byte fallback, Whisper, ByteLevel and RobertaProcessing trimming) gives tokenizers' ids, tokens, offsets, type ids, special-tokens mask and sequence ids on every corpus line", TokenizerEncodingBpe),
        ("tokenizers: encoding: WordPiece (uncased, cased), SentencePiece-style BPE and Unigram (with and without byte fallback) give expected-hf.json's offsets, ids, type ids, masks and sequence ids for single texts and pairs, and Decode skips special tokens when asked", TokenizerEncodingReferences),
        ("tokenizers: encoding: the detailed encode gives Encode's ids for every tokenizer.json in the test data, on the suite's texts and the corpus", TokenizerEncodingSameIds),
        ("tokenizers: encoding: truncation (longest_first, only_first, only_second) and padding (a batch to its longest, to a fixed length) as tokenizers does them", TokenizerEncodingTruncationPadding),
        ("tokenizers: encoding: a batch equals its texts one by one, on one worker and on every core (Workers, IDRAK_TOKENIZER_WORKERS); a bad worker count is refused", TokenizerEncodingBatch),
    ];

    private static JsonNode EncodingReferences() => JsonNode.Parse(File.ReadAllText(TestData("tokenizers/encodings.json")))!;

    private static int[] Ints(JsonNode node, string field) => [.. node[field]!.AsArray().Select(i => (int)i!)];

    // Every field of an encoding against a tokenizers reference entry; offsets as [start, end] pairs.
    private static void CheckEncoding(string where, TokenizerEncoding actual, JsonNode expected, bool tokens = false)
    {
        foreach (var (field, values) in new[] { ("ids", actual.Ids), ("type_ids", actual.TypeIds), ("special_tokens_mask", actual.SpecialTokensMask), ("sequence_ids", actual.SequenceIds) })
        {
            int[] wanted = Ints(expected, field);
            Check(values.SequenceEqual(wanted), $"{where}: {field} [{string.Join(", ", values)}], tokenizers [{string.Join(", ", wanted)}]");
        }

        if (expected["attention_mask"] is JsonArray)
        {
            Check(actual.AttentionMask.SequenceEqual(Ints(expected, "attention_mask")), $"{where}: attention_mask");
        }

        (int, int)[] offsets = [.. expected["offsets"]!.AsArray().Select(o => ((int)o![0]!, (int)o[1]!))];
        int first = Enumerable.Range(0, Math.Min(offsets.Length, actual.Count)).FirstOrDefault(i => actual.Offsets[i] != offsets[i], -1);
        Check(actual.Offsets.Length == offsets.Length && first < 0,
            $"{where}: offsets differ at token {first} ('{(first >= 0 && first < actual.Tokens.Length ? actual.Tokens[first] : "")}'): {(first >= 0 ? actual.Offsets[first] : default)}, tokenizers {(first >= 0 ? offsets[first] : default)}");
        if (tokens && expected["tokens"] is JsonArray names)
        {
            Check(actual.Tokens.SequenceEqual(names.Select(n => (string)n!)), $"{where}: tokens");
        }
    }

    private static void TokenizerEncodingBpe(Device device)
    {
        var references = EncodingReferences()["byte_level"]!.AsObject();
        Check(references.Count >= 7, "seven BPE references");
        foreach (var (path, lines) in references)
        {
            var tokenizer = PretrainedTokenizers.Load(Path.GetDirectoryName(TestData(path))!);
            int line = 0, refused = 0;
            foreach (var entry in lines!.AsArray())
            {
                // A tokenizer without an unknown token: tokenizers drops a character its vocabulary lacks, Idrak refuses it
                // (by design, as Encode does since before plan 24); such lines are counted, not compared.
                TokenizerEncoding encoding;
                try
                {
                    encoding = tokenizer.Encode((string)entry!["text"]!, null, new TokenizerEncodeOptions());
                }
                catch (InvalidDataException e) when (e.Message.Contains("has no unknown token", StringComparison.Ordinal))
                {
                    refused++;
                    line++;
                    continue;
                }

                CheckEncoding($"{path}, line {line}", encoding, entry, tokens: true);
                line++;
            }

            if (refused > 0)
            {
                Console.WriteLine($"    {path}: {refused} of {line} lines refused (a character without an unknown token; tokenizers drops it)");
            }

            Check(refused < line, $"{path}: all {line} lines refused, none compared");
        }
    }

    private static void TokenizerEncodingReferences(Device device)
    {
        foreach (string name in new[] { "wordpiece-uncased", "wordpiece-cased", "bpe", "unigram", "unigram-bytes" })
        {
            string folder = TestData($"tokenizers/{name}");
            var tokenizer = PretrainedTokenizers.Load(Path.Combine(folder, "tokenizer.json"));
            var reference = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "expected-hf.json")))!;
            int line = 0;
            foreach (var entry in reference["lines"]!.AsArray())
            {
                string text = (string)entry!["text"]!;
                var encoding = tokenizer.Encode(text, null, new TokenizerEncodeOptions());
                CheckEncoding($"{name}, line {line}", encoding, entry);
                if (entry["decoded"] is JsonValue decoded)
                {
                    Check(tokenizer.Decode(encoding.Ids, skipSpecialTokens: true) == (string)decoded!, $"{name}, line {line}: Decode skipping special tokens");
                }

                line++;
            }

            int pairs = 0;
            foreach (var entry in reference["pairs"]?.AsArray() ?? [])
            {
                CheckEncoding($"{name}, pair {pairs}", tokenizer.Encode((string)entry!["text"]!, (string)entry["pair"]!, new TokenizerEncodeOptions()), entry);
                pairs++;
            }
        }
    }

    private static void TokenizerEncodingSameIds(Device device)
    {
        string[] texts = PipelineTexts();
        int checkedFiles = 0;
        foreach (string file in Directory.EnumerateFiles(TestData(""), "tokenizer.json", SearchOption.AllDirectories).Order(StringComparer.Ordinal))
        {
            IPretrainedTokenizer tokenizer;
            try
            {
                tokenizer = PretrainedTokenizers.Load(file);
            }
            catch (NotSupportedException)
            {
                continue;                                                      // a model type nobody registered
            }

            foreach (string text in texts)
            {
                int[] ids;
                try
                {
                    ids = [.. tokenizer.Encode(text)];
                }
                catch (InvalidDataException)
                {
                    continue;                                                  // a character the vocabulary refuses (no unknown token)
                }

                var detailed = tokenizer.Encode(text, null, new TokenizerEncodeOptions { AddSpecialTokens = false });
                Check(detailed.Ids.SequenceEqual(ids), $"{file}: the detailed encode of '{text}' gives other ids than Encode");
                Check(detailed.Offsets.All(o => o.Start >= 0 && o.Start <= o.End && o.End <= text.Length), $"{file}: offsets of '{text}' outside it");
            }

            checkedFiles++;
        }

        Check(checkedFiles >= 20, $"{checkedFiles} tokenizer.json files checked");
    }

    private static void TokenizerEncodingTruncationPadding(Device device)
    {
        var references = EncodingReferences();
        var tokenizer = PretrainedTokenizers.Load(TestData("tokenizers/wordpiece-uncased"));
        int cases = 0;
        foreach (var entry in references["truncation"]!.AsArray())
        {
            var options = new TokenizerEncodeOptions { MaxLength = (int)entry!["max_length"]!, Truncation = (string)entry["strategy"]! };
            CheckEncoding($"truncation {entry["strategy"]} ({cases})", tokenizer.Encode((string)entry["text"]!, (string)entry["pair"]!, options), entry);
            cases++;
        }

        Check(cases >= 6, $"{cases} truncation cases");
        foreach (var (padding, options) in new[]
                 {
                     ("longest", new TokenizerEncodeOptions { Padding = "longest", PadId = 0 }),
                     ("max_length", new TokenizerEncodeOptions { Padding = "max_length", MaxLength = 24, PadId = 0 }),
                 })
        {
            var batch = references["padding"]![padding]!.AsArray();
            var encoded = tokenizer.EncodeBatch([.. batch.Select(e => (string)e!["text"]!)], null, options);
            for (int i = 0; i < batch.Count; i++)
            {
                CheckEncoding($"padding {padding}, text {i}", encoded[i], batch[i]!, tokens: true);
            }
        }

        Expect<ArgumentException>(() => tokenizer.Encode("a", null, new TokenizerEncodeOptions { Padding = "everywhere" }), "Padding 'everywhere'", "an unknown padding");
        Expect<ArgumentException>(() => tokenizer.Encode("a", null, new TokenizerEncodeOptions { Truncation = "middle", MaxLength = 4 }), "Truncation 'middle'", "an unknown truncation");
    }

    private static void TokenizerEncodingBatch(Device device)
    {
        string[] texts = [.. File.ReadAllText(TestData("tokenizers/corpus.txt")).Split('\n').Where(t => t.Length > 0)];
        string folder = TestData("tokenizers/wordpiece-uncased");
        var options = new TokenizerEncodeOptions();
        foreach (int workers in new[] { 1, 0 })
        {
            var tokenizer = PretrainedTokenizers.Load(folder, new TokenizerOptions { Workers = workers });
            var batch = tokenizer.EncodeBatch(texts, null, options);
            for (int i = 0; i < texts.Length; i++)
            {
                var single = tokenizer.Encode(texts[i], null, options);
                Check(batch[i].Ids.SequenceEqual(single.Ids) && batch[i].Offsets.SequenceEqual(single.Offsets), $"workers {workers}: text {i} in a batch equals it alone");
            }
        }

        var bpe = PretrainedTokenizers.Load(PretrainData("tokenizer"), new TokenizerOptions { Workers = 0 });
        var bpeBatch = bpe.EncodeBatch(texts, [.. Enumerable.Reverse(texts)], options);
        for (int i = 0; i < texts.Length; i++)
        {
            Check(bpeBatch[i].Ids.SequenceEqual(bpe.Encode(texts[i], texts[^(i + 1)], options).Ids), $"BPE pairs: pair {i} in a batch equals it alone");
        }

        Expect<ArgumentOutOfRangeException>(() => PretrainedTokenizers.Load(folder, new TokenizerOptions { Workers = -1 }), "Workers", "a negative worker count");
        string? previous = Environment.GetEnvironmentVariable(TokenizerOptions.WorkersVariable);
        try
        {
            Environment.SetEnvironmentVariable(TokenizerOptions.WorkersVariable, "many");
            Expect<InvalidOperationException>(() => PretrainedTokenizers.Load(folder), "IDRAK_TOKENIZER_WORKERS is 'many'", "a variable that is not a number");
        }
        finally
        {
            Environment.SetEnvironmentVariable(TokenizerOptions.WorkersVariable, previous);
        }
    }
}
