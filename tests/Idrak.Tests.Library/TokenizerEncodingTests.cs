// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Models;
using Idrak.Models.Abstractions;

// Plan 24, phase T3: detailed encodings (offsets, type ids, masks, pairs, truncation, padding, batches) against tokenizers.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] TokenizerEncodingGroup =
    [
        ("tokenizers: encoding: the detailed encode gives Encode's ids for every tokenizer.json in the test data, on the suite's texts and the corpus; with Offsets false (ids only, through the plain encode) it gives the same ids, type ids and masks with special tokens, truncation from either end and padding", TokenizerEncodingSameIds),
    ];

    internal static int[] Ints(JsonNode node, string field) => [.. node[field]!.AsArray().Select(i => (int)i!)];

    internal static readonly TokenizerEncodeOptions[] LeanCases =
    [
        new() { AddSpecialTokens = false }, new() { AddSpecialTokens = true },
        new() { AddSpecialTokens = true, MaxLength = 7, Truncation = "longest_first", TruncationDirection = "right", PadId = 0, Padding = "max_length" },
        new() { AddSpecialTokens = false, MaxLength = 5, Truncation = "longest_first", TruncationDirection = "left" },
    ];

    internal static void TokenizerEncodingSameIds(Device device)
    {
        string[] texts = PipelineTexts();
        int checkedFiles = 0, lean = 0;
        foreach (string file in Directory.EnumerateFiles(TestData(""), "tokenizer.json", SearchOption.AllDirectories).Order(StringComparer.Ordinal))
        {
            IPretrainedTokenizer tokenizer;
            try
            {
                tokenizer = PretrainedTokenizers.Load(file);
            }
            catch (NotSupportedException)
            {
                continue;                                                      // a model type not found
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
                if (lean++ % 5 != 0)
                {
                    continue;
                }

                // Offsets false: the same ids, type ids and masks through the plain encode, with special tokens, truncated
                // from either end and padded; offsets (0, 0) and no tokens.
                foreach (var options in LeanCases)
                {
                    var full = tokenizer.Encode(text, null, options);
                    var plain = tokenizer.Encode(text, null, options with { Offsets = false });
                    Check(plain.Ids.SequenceEqual(full.Ids) && plain.TypeIds.SequenceEqual(full.TypeIds) && plain.AttentionMask.SequenceEqual(full.AttentionMask)
                        && plain.SpecialTokensMask.SequenceEqual(full.SpecialTokensMask) && plain.SequenceIds.SequenceEqual(full.SequenceIds),
                        $"{file}: Offsets false on '{text}' ({options}) gives other ids, type ids or masks than the detailed encode");
                    Check(plain.Offsets.Length == plain.Count && plain.Offsets.All(o => o == (0, 0)) && (plain.Tokens.Length == 0 || plain.SpecialTokensMask.Zip(plain.AttentionMask).All(p => p.First == 1 || p.Second == 0)),
                        $"{file}: Offsets false leaves offsets (0, 0) and no tokens (but where every token is special or padding, as for an empty text)");
                }
            }

            checkedFiles++;
        }

        Check(checkedFiles >= 20, $"{checkedFiles} tokenizer.json files checked");
    }
}
