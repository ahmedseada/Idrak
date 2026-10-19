// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Models;

// Plan 24, phase T0 (wave 1): the SentencePiece model reader, the protobuf reader under it, and the double-array trie,
// against the fixtures tools/tokenizers/make_fixtures.py writes (data/tokenizers/README.md).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] TokenizerReaderGroup =
    [
        ("tokenizers: sentencepiece: each fixture's tokenizer.model reads to the ModelProto the official protobuf reads: every piece, its score bit for bit and its type, every trainer and normalizer field, the charsmap's size and SHA-256", SentencePieceReadsFixtures),
        ("tokenizers: sentencepiece: the protobuf reader reads varints to 64 bits, negative int32s, fixed32 and fixed64, strings and nested messages, skips unknown fields of every wire type (groups too), and refuses overlong varints, field 0, bad wire types and unmatched groups naming the byte", SentencePieceProtoReader),
        ("tokenizers: sentencepiece: a truncated model, a corrupt varint, a corrupt nested field and invalid UTF-8 fail with InvalidDataException naming the file and the byte offset", SentencePieceCorruptFiles),
        ("tokenizers: trie: each fixture's precompiled charsmap reads to the entries an independent Python reader decodes (count, SHA-256) and common-prefix search at every byte of every corpus line equals a brute-force scan", TrieReadsCharsmaps),
        ("tokenizers: trie: a trie built from seeded random keys (one past 2^21 units, so offsets take the far form) and from each fixture's pieces answers exact and common-prefix searches as a brute-force scan; bad keys are refused", TrieBuiltFromKeys),
        ("tokenizers: trie: common-prefix and exact search allocate nothing once warm, on a read and on a built trie", TrieSearchAllocatesNothing),
    ];

    private static readonly string[] SentencePieceFixtures = ["unigram", "unigram-bytes", "bpe"];

    private static string TokenizerData(string relative) => TestData($"tokenizers/{relative}");

    private static JsonObject TokenizerJson(string relative) => JsonNode.Parse(File.ReadAllText(TokenizerData(relative)))!.AsObject();

    // The corpus lines as the fixture script splits them: at LF, the final LF ending the last line (CR kept).
    private static string[] TokenizerCorpusLines()
    {
        string text = File.ReadAllText(TokenizerData("corpus.txt"), Encoding.UTF8);
        Check(text.EndsWith('\n'), "corpus.txt must end with a line feed");
        return text[..^1].Split('\n');
    }

    private static string PieceTypeName(SentencePieceType type) => type switch
    {
        SentencePieceType.Normal => "NORMAL",
        SentencePieceType.Unknown => "UNKNOWN",
        SentencePieceType.Control => "CONTROL",
        SentencePieceType.UserDefined => "USER_DEFINED",
        SentencePieceType.Unused => "UNUSED",
        SentencePieceType.Byte => "BYTE",
        _ => $"?{(int)type}",
    };

    // Every TrainerSpec field by its proto name, as the C# reader has it.
    private static (string Name, object Value)[] TrainerFields(SentencePieceTrainerSpec s) =>
    [
        ("input", s.Input), ("input_format", s.InputFormat), ("model_prefix", s.ModelPrefix),
        ("model_type", s.ModelType.ToString().ToUpperInvariant()), ("vocab_size", s.VocabSize), ("accept_language", s.AcceptLanguage),
        ("self_test_sample_size", s.SelfTestSampleSize), ("enable_differential_privacy", s.EnableDifferentialPrivacy),
        ("differential_privacy_noise_level", s.DifferentialPrivacyNoiseLevel),
        ("differential_privacy_clipping_threshold", s.DifferentialPrivacyClippingThreshold),
        ("character_coverage", s.CharacterCoverage), ("input_sentence_size", s.InputSentenceSize),
        ("shuffle_input_sentence", s.ShuffleInputSentence), ("mining_sentence_size", s.MiningSentenceSize),
        ("training_sentence_size", s.TrainingSentenceSize), ("seed_sentencepiece_size", s.SeedSentencepieceSize),
        ("shrinking_factor", s.ShrinkingFactor), ("max_sentence_length", s.MaxSentenceLength), ("num_threads", s.NumThreads),
        ("num_sub_iterations", s.NumSubIterations), ("max_sentencepiece_length", s.MaxSentencepieceLength),
        ("split_by_unicode_script", s.SplitByUnicodeScript), ("split_by_number", s.SplitByNumber),
        ("split_by_whitespace", s.SplitByWhitespace), ("treat_whitespace_as_suffix", s.TreatWhitespaceAsSuffix),
        ("allow_whitespace_only_pieces", s.AllowWhitespaceOnlyPieces), ("split_digits", s.SplitDigits),
        ("pretokenization_delimiter", s.PretokenizationDelimiter), ("control_symbols", s.ControlSymbols),
        ("user_defined_symbols", s.UserDefinedSymbols), ("required_chars", s.RequiredChars), ("byte_fallback", s.ByteFallback),
        ("vocabulary_output_piece_score", s.VocabularyOutputPieceScore), ("hard_vocab_limit", s.HardVocabLimit),
        ("use_all_vocab", s.UseAllVocab), ("unk_id", s.UnkId), ("bos_id", s.BosId), ("eos_id", s.EosId), ("pad_id", s.PadId),
        ("unk_piece", s.UnkPiece), ("bos_piece", s.BosPiece), ("eos_piece", s.EosPiece), ("pad_piece", s.PadPiece),
        ("unk_surface", s.UnkSurface), ("train_extremely_large_corpus", s.TrainExtremelyLargeCorpus),
    ];

    private static (string Name, object Value)[] NormalizerFields(SentencePieceNormalizerSpec s) =>
    [
        ("name", s.Name), ("precompiled_charsmap", s.PrecompiledCharsmap), ("add_dummy_prefix", s.AddDummyPrefix),
        ("remove_extra_whitespaces", s.RemoveExtraWhitespaces), ("escape_whitespaces", s.EscapeWhitespaces),
        ("normalization_rule_tsv", s.NormalizationRuleTsv),
    ];

    // Compares the C# fields with the Python dump of the same message: each field the dump has is one the reader has, and
    // its value is the same (floats bit for bit through the *_bits entries, bytes by size and SHA-256).
    private static void CheckSpec(string what, (string Name, object Value)[] fields, JsonObject reference)
    {
        var names = fields.Select(f => f.Name).ToHashSet(StringComparer.Ordinal);
        foreach (var key in reference.Select(p => p.Key))
        {
            Check(key == "present" || key.EndsWith("_bits", StringComparison.Ordinal) || names.Contains(key),
                $"{what}: the proto has a field '{key}' the reader does not map");
        }

        foreach (var (name, value) in fields)
        {
            var node = reference[name];
            Check(node is not null, $"{what}: the reference has no field '{name}'");
            bool same = value switch
            {
                float f => BitConverter.SingleToUInt32Bits(f) == (uint)reference[name + "_bits"]!,
                byte[] b => b.Length == (int)node!["bytes"]! && Convert.ToHexStringLower(SHA256.HashData(b)) == (string)node["sha256"]!,
                string[] a => a.SequenceEqual(node!.AsArray().Select(x => (string)x!)),
                string text => text == (string)node!,
                bool flag => flag == (bool)node!,
                int number => number == (long)node!,
                ulong number => number == (ulong)node!,
                _ => false,
            };
            Check(same, $"{what}.{name}: the reader has {Show(value)}, the official protobuf {node!.ToJsonString()}");
        }

        static string Show(object value) => value switch
        {
            string[] a => "[" + string.Join(", ", a) + "]",
            byte[] b => $"{b.Length} bytes",
            float f => f.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
            _ => Convert.ToString(value, System.Globalization.CultureInfo.InvariantCulture) ?? "null",
        };
    }

    private static void SentencePieceReadsFixtures(Device device)
    {
        foreach (var name in SentencePieceFixtures)
        {
            var model = SentencePieceModel.Load(TokenizerData($"{name}/tokenizer.model"));
            var reference = TokenizerJson($"{name}/model.json");
            var pieces = reference["pieces"]!.AsArray();
            Check(model.Count == pieces.Count, $"{name}: {model.Count} pieces, the reference {pieces.Count}");
            for (int i = 0; i < pieces.Count; i++)
            {
                var piece = pieces[i]!;
                Check(model.Pieces[i] == (string)piece["piece"]!, $"{name}: piece {i} is '{model.Pieces[i]}', the reference '{(string)piece["piece"]!}'");
                Check(BitConverter.SingleToUInt32Bits(model.Scores[i]) == (uint)piece["score_bits"]!,
                    $"{name}: piece {i}'s score is {model.Scores[i]:R}, the reference {(double)piece["score"]!:R} (bits {(uint)piece["score_bits"]!:X8})");
                Check(PieceTypeName(model.Types[i]) == (string)piece["type"]!, $"{name}: piece {i}'s type is {model.Types[i]}, the reference {(string)piece["type"]!}");
            }

            CheckSpec($"{name} trainer_spec", TrainerFields(model.TrainerSpec), reference["trainer_spec"]!.AsObject());
            CheckSpec($"{name} normalizer_spec", NormalizerFields(model.NormalizerSpec), reference["normalizer_spec"]!.AsObject());
            if (reference["denormalizer_spec"] is JsonObject denormalizer)
            {
                Check(model.DenormalizerSpec is not null, $"{name}: the file has a denormalizer_spec the reader did not read");
                CheckSpec($"{name} denormalizer_spec", NormalizerFields(model.DenormalizerSpec!), denormalizer);
            }
            else
            {
                Check(model.DenormalizerSpec is null, $"{name}: the reader found a denormalizer_spec the file does not have");
            }

            var file = reference["file"]!;
            Check(new FileInfo(TokenizerData($"{name}/tokenizer.model")).Length == (long)file["bytes"]!, $"{name}: tokenizer.model is not the file model.json describes (regenerate the fixtures)");
        }

        // The fixtures between them cover the piece types and settings the reader maps.
        var all = SentencePieceFixtures.Select(n => SentencePieceModel.Load(TokenizerData($"{n}/tokenizer.model"))).ToArray();
        foreach (var type in Enum.GetValues<SentencePieceType>().Where(t => t != SentencePieceType.Unused))
        {
            Check(all.Any(m => m.Types.Contains(type)), $"no fixture has a {type} piece");
        }

        Check(all.Any(m => m.TrainerSpec.ModelType == SentencePieceAlgorithm.Bpe) && all.Any(m => m.TrainerSpec.ModelType == SentencePieceAlgorithm.Unigram), "the fixtures must hold a BPE and a unigram model");
        Check(all.Any(m => m.TrainerSpec.PadId >= 0) && all.Any(m => m.TrainerSpec.BosId < 0) && all.Any(m => m.NormalizerSpec.PrecompiledCharsmap.Length == 0), "the fixtures must cover a pad id, no bos id and an empty charsmap");
    }

    private static void SentencePieceProtoReader(Device device)
    {
        var data = new List<byte> { 0x08, 0x96, 0x01 };                               // 1: varint 150
        data.AddRange([0x10, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01]);  // 2: ulong.MaxValue (ten bytes)
        data.AddRange([0x18, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01]);  // 3: int32 -1, sign-extended
        data.Add(0x25);                                                                 // 4: fixed32 1.5f
        var f32 = new byte[4];
        BinaryPrimitives.WriteSingleLittleEndian(f32, 1.5f);
        data.AddRange(f32);
        data.Add(0x29);                                                                 // 5: fixed64 -2.25
        var f64 = new byte[8];
        BinaryPrimitives.WriteDoubleLittleEndian(f64, -2.25);
        data.AddRange(f64);
        data.AddRange([0x32, 0x04, 0x61, 0xD8, 0xA8, 0x63]);                             // 6: "aبc"
        data.AddRange([0x3A, 0x05, 0x08, 0x07, 0x12, 0x01, 0x7A]);                      // 7: message { 1: 7, 2: "z" }
        data.AddRange([0x43, 0x08, 0x01, 0x4B, 0x15, 0, 0, 0, 0, 0x4C, 0x44]);          // 8: a group holding a group (unknown)
        data.AddRange([0xF8, 0xFF, 0xFF, 0xFF, 0x0F, 0x05]);                            // 536870911: varint 5 (unknown)
        data.AddRange([0x59, 1, 2, 3, 4, 5, 6, 7, 8]);                                   // 11: fixed64 (unknown)
        byte[] bytes = [.. data];

        var reader = new ProtoReader(bytes);
        Field(ref reader, 1, ProtoWireType.Varint);
        Check(reader.Varint() == 150, "varint 150");
        Field(ref reader, 2, ProtoWireType.Varint);
        Check(reader.Varint() == ulong.MaxValue, "a ten-byte varint is 64 bits");
        Field(ref reader, 3, ProtoWireType.Varint);
        Check(reader.Int32() == -1, "an int32 -1 written sign-extended reads back as -1");
        Field(ref reader, 4, ProtoWireType.Fixed32);
        Check(reader.Float() == 1.5f, "fixed32 float");
        Field(ref reader, 5, ProtoWireType.Fixed64);
        Check(reader.Double() == -2.25, "fixed64 double");
        Field(ref reader, 6, ProtoWireType.LengthDelimited);
        Check(reader.String() == "aبc", "a UTF-8 string");
        Field(ref reader, 7, ProtoWireType.LengthDelimited);
        int messageStart = reader.Offset;
        var inner = reader.Message();
        Check(inner.Offset == messageStart + 1, $"a nested reader counts from the outer data (offset {inner.Offset}, expected {messageStart + 1})");
        Field(ref inner, 1, ProtoWireType.Varint);
        Check(inner.Int32() == 7, "nested varint");
        Field(ref inner, 2, ProtoWireType.LengthDelimited);
        Check(inner.String() == "z" && inner.End && !inner.Next(out _, out _), "nested string, then the nested message ends");
        int skipped = 0;
        while (reader.Next(out int field, out var wire))
        {
            reader.Skip(field, wire);
            skipped++;
        }

        Check(skipped == 3 && reader.End && reader.Offset == bytes.Length, $"three unknown fields (a group with a nested group, the largest field number, fixed64) skipped to the end ({skipped}, offset {reader.Offset} of {bytes.Length})");

        // Errors name the byte where the bad item starts.
        ProtoFails([0x00], "field number 0", 0);
        ProtoFails([0x0F], "wire type 7", 0);
        ProtoFails([0x80, 0x80, 0x80, 0x80, 0x80, 0x01], "field number 4294967296", 0);
        ProtoFails([0x0C], "an end-group (field 1) with no group open", 0);
        ProtoFails([0x0B, 0x14], "the group of field 1 is ended by field 2", 1);
        ProtoFails([0x0B, 0x08, 0x01], "the group of field 1 has no end", 0);
        ProtoFails([0x08, 0x80], "truncated varint", 1);
        ProtoFails([0x08, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01], "varint longer than ten bytes", 1);
        ProtoFails([0x08, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x02], "varint above 64 bits", 1);
        ProtoFails([0x25, 0x00, 0x00], "truncated fixed32", 1);
        ProtoFails([0x29, 0x00], "truncated fixed64", 1);
        ProtoFails([0x32, 0x05, 0x61], "a length-delimited field of 5 bytes runs past the end", 1);
        ProtoFails([0x0A, 0x02, 0x08, 0x80], "truncated varint", 3, nested: true);
        ProtoFails([0x0A, 0x02, 0x01, 0xFF], "invalid UTF-8", 1, text: true);

        static void Field(ref ProtoReader reader, int field, ProtoWireType wire)
        {
            Check(reader.Next(out int f, out var w), $"field {field} expected, the data ended");
            Check(f == field && w == wire, $"field {f} ({w}), expected field {field} ({wire})");
        }
    }

    // Reads every field of `data` (into nested messages with `nested`, strings with `text`): it must fail with `what` at
    // `offset`.
    private static void ProtoFails(byte[] data, string what, int offset, bool nested = false, bool text = false)
    {
        void ReadAll()
        {
            var reader = new ProtoReader(data);
            while (reader.Next(out int field, out var wire))
            {
                if (nested && wire == ProtoWireType.LengthDelimited)
                {
                    var inner = reader.Message();
                    while (inner.Next(out int f, out var w))
                    {
                        inner.Skip(f, w);
                    }
                }
                else if (text && wire == ProtoWireType.LengthDelimited)
                {
                    reader.String();
                }
                else
                {
                    reader.Skip(field, wire);
                }
            }
        }

        Expect<InvalidDataException>(ReadAll, what, $"{Convert.ToHexString(data)}: '{what}'");
        Expect<InvalidDataException>(ReadAll, $" at byte {offset}.", $"{Convert.ToHexString(data)}: the offset of '{what}'");
    }

    private static void SentencePieceCorruptFiles(Device device)
    {
        byte[] model = File.ReadAllBytes(TokenizerData("unigram-bytes/tokenizer.model"));

        // The top-level fields: where each starts, where its length varint starts, and its payload.
        var fields = new List<(int Field, int Start, int LengthStart, int PayloadStart, int PayloadLength)>();
        var walk = new ProtoReader(model);
        while (true)
        {
            int start = walk.Offset;
            if (!walk.Next(out int field, out var wire))
            {
                break;
            }

            Check(wire == ProtoWireType.LengthDelimited, $"top-level field {field} of the fixture is not length-delimited");
            int lengthStart = walk.Offset;
            int length = walk.LengthDelimited().Length;
            fields.Add((field, start, lengthStart, walk.Offset - length, length));
        }

        var pieces = fields.Where(f => f.Field == 1).ToArray();
        var trainer = fields.Single(f => f.Field == 2);
        var normalizer = fields.Single(f => f.Field == 3);
        foreach (var (what, f) in new[] { ("the first piece", pieces[0]), ("the last piece", pieces[^1]), ("trainer_spec", trainer), ("normalizer_spec", normalizer) })
        {
            int cut = f.PayloadStart + (f.PayloadLength / 2);
            Expect<InvalidDataException>(() => SentencePieceModel.Read(model.AsSpan(0, cut), "cut.model"),
                $"SentencePiece model 'cut.model': Protocol Buffers data: a length-delimited field of {f.PayloadLength} bytes runs past the end of its data (byte {cut}",
                $"a file cut inside {what}");
            Expect<InvalidDataException>(() => SentencePieceModel.Read(model.AsSpan(0, cut), "cut.model"), $" at byte {f.LengthStart}.", $"the offset of a file cut inside {what}");
        }

        Check(normalizer.PayloadStart - normalizer.LengthStart > 1, "normalizer_spec's length is one byte; the test needs a longer one");
        Expect<InvalidDataException>(() => SentencePieceModel.Read(model.AsSpan(0, normalizer.LengthStart + 1), "cut.model"),
            $"truncated varint (the data ends after 1 of its bytes) at byte {normalizer.LengthStart}.", "a file cut inside a length varint");

        byte[] Corrupt(int at, params byte[] with) => [.. model.AsSpan(0, at), .. with, .. model.AsSpan(at + 1)];
        Expect<InvalidDataException>(() => SentencePieceModel.Read(Corrupt(1, [.. Enumerable.Repeat((byte)0xFF, 10), 0x01]), "bad.model"),
            "SentencePiece model 'bad.model': Protocol Buffers data: varint longer than ten bytes at byte 1.", "an eleven-byte varint");
        Expect<InvalidDataException>(() => SentencePieceModel.Read(Corrupt(1, [.. Enumerable.Repeat((byte)0xFF, 9), 0x02]), "bad.model"),
            "varint above 64 bits at byte 1.", "a varint above 64 bits");
        Expect<InvalidDataException>(() => SentencePieceModel.Read(Corrupt(trainer.PayloadStart, 0x07), "bad.model"),
            $"field number 0 (valid: 1 to 536870911) at byte {trainer.PayloadStart}.", "a corrupt field header inside trainer_spec (the offset counts from the file's start)");
        // The first piece's string: its header at the payload's start, its length after it, its first byte after that.
        Expect<InvalidDataException>(() => SentencePieceModel.Read(Corrupt(pieces[0].PayloadStart + 2, 0xFF), "bad.model"),
            $"invalid UTF-8 in a string field at byte {pieces[0].PayloadStart + 1}.", "a piece that is not UTF-8");
        Expect<InvalidDataException>(() => SentencePieceModel.Read(Corrupt(0, 0x0F), "bad.model"), "wire type 7 (valid: 0 to 5) in field 1 at byte 0.", "a bad wire type");
        Check(SentencePieceModel.Read(model.AsSpan(0, pieces[1].Start), "short.model").Count == 1, "a file cut between top-level fields is a valid, shorter model (protobuf has no end marker)");
        Check(SentencePieceModel.Read([], "empty.model") is { Count: 0, TrainerSpec.VocabSize: 8000, TrainerSpec.PadId: -1, NormalizerSpec.AddDummyPrefix: true },
            "an empty file is a model with no pieces and the proto's defaults");
    }

    private static void TrieReadsCharsmaps(Device device)
    {
        var lines = TokenizerCorpusLines().Select(Encoding.UTF8.GetBytes).ToArray();
        Span<TrieMatch> matches = stackalloc TrieMatch[64];
        int read = 0;
        foreach (var name in SentencePieceFixtures)
        {
            var model = SentencePieceModel.Load(TokenizerData($"{name}/tokenizer.model"));
            var reference = TokenizerJson($"{name}/model.json")["charsmap"];
            if (reference is null)
            {
                Check(model.NormalizerSpec.PrecompiledCharsmap.Length == 0, $"{name}: a charsmap the reference does not have");
                continue;
            }

            byte[] charsmap = model.NormalizerSpec.PrecompiledCharsmap;
            var trie = DoubleArrayTrie.ReadCharsmap(charsmap, out var normalizedSpan);
            byte[] normalized = normalizedSpan.ToArray();
            Check(trie.Units == (int)reference["units"]! && normalized.Length == (int)reference["normalized_bytes"]!,
                $"{name}: {trie.Units} units and {normalized.Length} normalized bytes, the reference {reference["units"]} and {reference["normalized_bytes"]}");

            var entries = trie.Entries();
            var listing = new StringBuilder();
            var brute = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var (key, value) in entries)
            {
                Check(value < normalized.Length, $"{name}: an entry's value {value} is past the normalized strings ({normalized.Length} bytes)");
                int end = Array.IndexOf(normalized, (byte)0, value);
                Check(end >= 0, $"{name}: the normalized string at {value} has no 0 byte after it");
                listing.Append(Convert.ToHexStringLower(key)).Append('\t').Append(Convert.ToHexStringLower(normalized.AsSpan(value, end - value))).Append('\n');
                brute.Add(Encoding.Latin1.GetString(key), value);
            }

            Check(entries.Count == (int)reference["entries"]!, $"{name}: {entries.Count} entries, the Python reader {reference["entries"]}");
            Check(Convert.ToHexStringLower(SHA256.HashData(Encoding.ASCII.GetBytes(listing.ToString()))) == (string)reference["entries_sha256"]!,
                $"{name}: the entries (key, normalized string) differ from the Python reader's");
            int longest = (int)reference["longest_key"]!;

            // Every byte position of every line (inside characters too): the trie's matches, shortest first, equal the
            // keys of the decoded entries that are prefixes there.
            var expected = new List<TrieMatch>();
            int searched = 0, found = 0;
            foreach (var line in lines)
            {
                for (int start = 0; start < line.Length; start++)
                {
                    var rest = line.AsSpan(start);
                    expected.Clear();
                    for (int length = 1; length <= Math.Min(longest, rest.Length); length++)
                    {
                        if (brute.TryGetValue(Encoding.Latin1.GetString(rest[..length]), out int value))
                        {
                            expected.Add(new TrieMatch(value, length));
                        }
                    }

                    int count = trie.CommonPrefixSearch(rest, matches);
                    Check(count == expected.Count && matches[..Math.Min(count, matches.Length)].SequenceEqual(expected.ToArray()),
                        $"{name}: at byte {start} of '{Encoding.UTF8.GetString(line)}' the trie found {count} matches, a scan {expected.Count}");
                    int exact = trie.ExactMatch(rest[..Math.Min(longest, rest.Length)]);
                    Check(exact == (brute.TryGetValue(Encoding.Latin1.GetString(rest[..Math.Min(longest, rest.Length)]), out int e) ? e : -1), $"{name}: exact match at byte {start}");
                    searched++;
                    found += count;
                }
            }

            Check(found > 0 && searched > 1000, $"{name}: {searched} searches found {found} matches; the corpus must reach the charsmap");
            read++;
        }

        Check(read >= 2, "at least two fixtures must have a charsmap (nmt_nfkc and a user rule)");
        Expect<InvalidDataException>(() => DoubleArrayTrie.ReadCharsmap([1, 2], out _), "its 4-byte trie size at byte 0 is cut off", "a charsmap shorter than its header");
        Expect<InvalidDataException>(() => DoubleArrayTrie.ReadCharsmap([8, 0, 0, 0, 1, 2, 3, 4], out _), "the trie size at byte 0 is 8", "a trie size past the end");
        Expect<InvalidDataException>(() => DoubleArrayTrie.ReadCharsmap([6, 0, 0, 0, 1, 2, 3, 4, 5, 6], out _), "the trie size at byte 0 is 6", "a trie size that is not whole units");
    }

    private static void TrieBuiltFromKeys(Device device)
    {
        var random = new Random(2026);
        byte[] alphabet = [1, 2, 3, 0x41, 0x42, 0xC3, 0xD8, 0xE2, 0xFF];
        byte[] RandomBytes(int length, bool withZero) =>
            [.. Enumerable.Range(0, length).Select(_ => withZero && random.Next(20) == 0 ? (byte)0 : alphabet[random.Next(alphabet.Length)])];

        var set = new Dictionary<string, int>(StringComparer.Ordinal);
        while (set.Count < 5000)
        {
            set.TryAdd(Encoding.Latin1.GetString(RandomBytes(1 + random.Next(10), false)), set.Count == 17 ? int.MaxValue : random.Next());
        }

        var sorted = set.OrderBy(p => p.Key, StringComparer.Ordinal).ToArray();      // Latin-1 keeps the byte order
        var keys = sorted.Select(p => Encoding.Latin1.GetBytes(p.Key)).ToArray();
        var values = sorted.Select(p => p.Value).ToArray();
        var trie = DoubleArrayTrie.Build(keys, values);
        for (int i = 0; i < keys.Length; i++)
        {
            Check(trie.ExactMatch(keys[i]) == values[i], $"key {i} ({Convert.ToHexString(keys[i])}) gives {trie.ExactMatch(keys[i])}, expected {values[i]}");
        }

        Span<TrieMatch> matches = stackalloc TrieMatch[16];
        var expected = new List<TrieMatch>();
        for (int probe = 0; probe < 20000; probe++)
        {
            byte[] text = probe % 4 == 0 ? [.. keys[random.Next(keys.Length)], .. RandomBytes(random.Next(4), true)] : RandomBytes(random.Next(15), true);
            expected.Clear();
            for (int length = 1; length <= text.Length; length++)
            {
                if (set.TryGetValue(Encoding.Latin1.GetString(text, 0, length), out int value))
                {
                    expected.Add(new TrieMatch(value, length));
                }
            }

            int count = trie.CommonPrefixSearch(text, matches);
            Check(count == expected.Count && matches[..count].SequenceEqual(expected.ToArray()), $"probe {Convert.ToHexString(text)}: {count} matches, a scan {expected.Count}");
            int exact = set.TryGetValue(Encoding.Latin1.GetString(text), out int v) ? v : -1;
            Check(trie.ExactMatch(text) == exact, $"probe {Convert.ToHexString(text)}: exact {trie.ExactMatch(text)}, a scan {exact}");
        }

        // A span shorter than the matches gets the first ones and the full count.
        byte[] longKey = keys[0];
        int all = 0;
        foreach (var key in keys)
        {
            int n = trie.CommonPrefixSearch(key, matches);
            if (n > all)
            {
                (longKey, all) = (key, n);
            }
        }

        trie.CommonPrefixSearch(longKey, matches);
        Span<TrieMatch> two = stackalloc TrieMatch[2];
        Check(all > 2 && trie.CommonPrefixSearch(longKey, two) == all && two.SequenceEqual(matches[..2]), "a short span gets the shortest matches and the whole count");

        // Far offsets: a trie past 2^21 units stores offsets as multiples of 256.
        var big = new HashSet<string>(StringComparer.Ordinal);
        var bigRandom = new Random(7);
        while (big.Count < 400_000)
        {
            var bytes = new byte[8];
            for (int i = 0; i < bytes.Length; i++)
            {
                bytes[i] = (byte)(1 + bigRandom.Next(255));
            }

            big.Add(Encoding.Latin1.GetString(bytes));
        }

        var bigKeys = big.Order(StringComparer.Ordinal).Select(Encoding.Latin1.GetBytes).ToArray();
        var bigTrie = DoubleArrayTrie.Build(bigKeys, [.. Enumerable.Range(0, bigKeys.Length)]);
        Check(bigTrie.Units > 1 << 21, $"the large trie has {bigTrie.Units} units; it must pass 2^21 to need far offsets");
        for (int i = 0; i < bigKeys.Length; i++)
        {
            Check(bigTrie.ExactMatch(bigKeys[i]) == i, $"large trie: key {i} gives {bigTrie.ExactMatch(bigKeys[i])}");
            Check(bigTrie.ExactMatch(bigKeys[i].AsSpan(0, 7)) == -1, $"large trie: a prefix of key {i} is not a key");
        }

        // Each fixture's pieces (strings, unsorted, as the vocabulary holds them) map back to their ids.
        foreach (var name in SentencePieceFixtures)
        {
            var model = SentencePieceModel.Load(TokenizerData($"{name}/tokenizer.model"));
            var pieceTrie = DoubleArrayTrie.Build(model.Pieces, [.. Enumerable.Range(0, model.Count)]);
            for (int id = 0; id < model.Count; id++)
            {
                Check(pieceTrie.ExactMatch(Encoding.UTF8.GetBytes(model.Pieces[id])) == id, $"{name}: piece {id} '{model.Pieces[id]}' gives {pieceTrie.ExactMatch(Encoding.UTF8.GetBytes(model.Pieces[id]))}");
            }
        }

        var empty = DoubleArrayTrie.Build(Array.Empty<byte[]>(), []);
        for (int b = 0; b < 256; b++)
        {
            Check(empty.CommonPrefixSearch([(byte)b, 1], matches) == 0 && empty.ExactMatch([(byte)b]) == -1, $"the empty trie matched byte {b}");
        }

        Expect<ArgumentException>(() => DoubleArrayTrie.Build([[2], [1]], [0, 1]), "Key 1 sorts before key 0", "unsorted keys");
        Expect<ArgumentException>(() => DoubleArrayTrie.Build([[1], [1]], [0, 1]), "Key 1 repeats key 0", "a repeated key");
        Expect<ArgumentException>(() => DoubleArrayTrie.Build([[]], [0]), "Key 0 is empty", "an empty key");
        Expect<ArgumentException>(() => DoubleArrayTrie.Build([[1, 0, 2]], [0]), "Key 0 has a 0 byte at 1", "a key with a 0 byte");
        Expect<ArgumentException>(() => DoubleArrayTrie.Build([[1]], [-1]), "Value 0 is -1", "a negative value");
        Expect<ArgumentException>(() => DoubleArrayTrie.Build(["b", "a", "b"], [0, 1, 2]), "Key 2 ('b') repeats key 0", "a repeated string");
    }

    private static void TrieSearchAllocatesNothing(Device device)
    {
        var lines = TokenizerCorpusLines().Select(Encoding.UTF8.GetBytes).ToArray();
        var model = SentencePieceModel.Load(TokenizerData("unigram/tokenizer.model"));
        var charsmap = DoubleArrayTrie.ReadCharsmap(model.NormalizerSpec.PrecompiledCharsmap, out _);
        var pieces = DoubleArrayTrie.Build(model.Pieces, [.. Enumerable.Range(0, model.Count)]);
        Span<TrieMatch> matches = stackalloc TrieMatch[32];

        long Pass(Span<TrieMatch> into)
        {
            long total = 0;
            foreach (var line in lines)
            {
                for (int start = 0; start < line.Length; start++)
                {
                    var rest = new ReadOnlySpan<byte>(line, start, line.Length - start);
                    total += charsmap.CommonPrefixSearch(rest, into) + pieces.CommonPrefixSearch(rest, into);
                    total += charsmap.ExactMatch(rest[..Math.Min(4, rest.Length)]) + pieces.ExactMatch(rest[..Math.Min(6, rest.Length)]);
                }
            }

            return total;
        }

        for (int warm = 0; warm < 40; warm++)
        {
            Pass(matches);
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        long result = Pass(matches);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Check(allocated == 0, $"one warm pass of searches over the corpus allocated {allocated} bytes (result {result})");
    }
}
