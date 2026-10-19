// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Models;
using Idrak.Models.Abstractions;

// Plan 19, phase A2: tokenizers learned from text (the library's "bpe" trainer against Hugging Face tokenizers).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] TokenizerTrainingGroup =
    [
        ("tokenizer training: bpe learns the vocabulary and merges Hugging Face tokenizers learns from the same file, in the same order, and encodes as it does (byte level)", TokenizerTrainingByteLevel),
        ("tokenizer training: bpe merging until no pair is left (minimum frequency 1) gives Hugging Face's merges through every tie", TokenizerTrainingFull),
        ("tokenizer training: bpe with the metaspace pre-tokenizer learns Hugging Face's vocabulary and merges and encodes as it does", TokenizerTrainingMetaspace),
        ("tokenizer training: a trained tokenizer saved as tokenizer.json loads back to the same ids, passes the tokenizer contract, and a loaded file saves to the same vocabulary and merges", TokenizerTrainingSave),
        ("tokenizer training: a file with Windows line ends (CRLF) trains the tokenizer its LF copy trains (byte level and metaspace), LineEnds \"keep\" learns the \\r as Hugging Face does, and another value is refused", TokenizerTrainingLineEnds),
        ("tokenizer training: the registry names bpe, unigram and wordpiece, refuses an unregistered trainer, bad options and missing files, and one worker counts the same words as many", TokenizerTrainingRegistry),
        ("tokenizer training: unigram learns SentencePiece's vocabulary from the same text and options (the special and byte pieces at the same ids, the learned pieces overlapping above the measured threshold), with and without byte fallback, split digits and the Arabic removals", TokenizerTrainingUnigramReference),
        ("tokenizer training: unigram's tokenizer saved as tokenizer.json and tokenizer.model loads back from both to its ids on every line, and with byte fallback decodes every line back to its text", TokenizerTrainingUnigramFiles),
        ("tokenizer training: wordpiece by frequency learns the vocabulary of Hugging Face's WordPieceTrainer algorithm with its ties fixed, exactly, and overlaps Hugging Face's own run above the measured threshold", TokenizerTrainingWordPieceReference),
        ("tokenizer training: wordpiece (frequency and likelihood) loads back from its tokenizer.json to its ids and spells every word of every line with its pieces (no unknown token)", TokenizerTrainingWordPieceFiles),
        ("tokenizer training: one worker learns the same unigram and wordpiece (both scorings) as many, bit for bit", TokenizerTrainingWorkers),
        ("tokenizer training: the Arabic removals (tatweel, diacritics) are written into the unigram and wordpiece tokenizers as standard normalizers, so text with and without them encodes alike, also from tokenizer.model", TokenizerTrainingArabic),
        ("tokenizer training: unigram and wordpiece refuse bad options naming them, and unigram reports words, seeds, EM and pruning", TokenizerTrainingNewOptions),
    ];

    // Measured 2026-10-19 (plans/24-tokenizers.md, T4 as built): the share of SentencePiece's learned pieces the unigram
    // trainer learns, and of Hugging Face's WordPiece run the frequency trainer learns, each with some room below.
    private const double UnigramOverlapFloor = 0.98;
    private const double WordPieceOverlapFloor = 0.95;

    // The text the trainers' references were made from (tools/tokenizers/make_training_fixtures.py).
    private static string[] TrainingCorpus => [TestData("tokenizers/corpus.txt"), PretrainData("corpus.txt")];

    private static JsonObject TrainingReference(string name) => JsonNode.Parse(File.ReadAllText(TestData($"tokenizer-training/{name}")))!.AsObject();

    // Every line of the training text, as the trainers read them (LineEnds "lf").
    private static List<string> TrainingLines() =>
        [.. TrainingCorpus.SelectMany(f => File.ReadAllText(f).Split('\n').SkipLast(1).Select(l => l.EndsWith('\r') ? l[..^1] : l))];

    private static string[] Vocabulary(IPretrainedTokenizer tokenizer) => [.. Enumerable.Range(0, tokenizer.VocabularySize).Select(tokenizer.TokenOf)];

    private static TokenizerTrainingOptions UnigramOptions(JsonNode o) => new()
    {
        VocabularySize = (int)o["vocab_size"]!,
        SpecialTokens = [.. o["special_tokens"]!.AsArray().Select(t => (string)t!)],
        ByteFallback = (bool?)o["byte_fallback"] ?? false,
        SplitDigits = (bool?)o["split_digits"] ?? false,
        MaxPieceLength = (int?)o["max_sentencepiece_length"] ?? 16,
        ShrinkingFactor = (double?)o["shrinking_factor"] ?? 0.75,
        EmIterations = (int?)o["num_sub_iterations"] ?? 2,
        CharacterCoverage = (double)o["character_coverage"]!,
        Arabic = [.. o["arabic"]!.AsArray().Select(a => (string)a!)],
    };

    private static TokenizerTrainingOptions WordPieceOptions(JsonNode o) => new()
    {
        VocabularySize = (int)o["vocab_size"]!,
        MinFrequency = (int)o["min_frequency"]!,
        SpecialTokens = [.. o["special_tokens"]!.AsArray().Select(t => (string)t!)],
        Lowercase = (bool)o["lowercase"]!,
    };

    private static string PretrainData(string relative) => TestData($"pretrain/{relative}");

    private static TokenizerTrainingOptions ReferenceTrainingOptions(string folder)
    {
        var trainer = JsonNode.Parse(File.ReadAllText(PretrainData($"{folder}/train.json")))!["trainer"]!;
        return new TokenizerTrainingOptions
        {
            VocabularySize = (int)trainer["vocab_size"]!,
            MinFrequency = (int)trainer["min_frequency"]!,
            SpecialTokens = [.. trainer["special_tokens"]!.AsArray().Select(t => (string)t!)],
            PreTokenizer = (string)trainer["pre_tokenizer"]!,
            AddPrefixSpace = (bool)trainer["add_prefix_space"]!,
        };
    }

    private static void CheckTrainedAgainstReference(IPretrainedTokenizer trained, string folder)
    {
        var reference = JsonNode.Parse(File.ReadAllText(PretrainData($"{folder}/train.json")))!.AsObject();
        var json = trained.ToJson();
        var model = json["model"]!.AsObject();
        var vocabulary = model["vocab"]!.AsObject().Select(p => (p.Key, (int)p.Value!)).ToList();
        var expectedVocabulary = reference["vocab"]!.AsArray().Select(e => ((string)e![0]!, (int)e[1]!)).ToList();
        int firstVocabulary = Enumerable.Range(0, Math.Min(vocabulary.Count, expectedVocabulary.Count)).FirstOrDefault(i => vocabulary[i] != expectedVocabulary[i], -1);
        Check(vocabulary.Count == expectedVocabulary.Count && firstVocabulary < 0,
            $"{folder}: vocabulary of {vocabulary.Count} (expected {expectedVocabulary.Count}); first difference at {firstVocabulary}"
            + (firstVocabulary >= 0 ? $": {vocabulary[firstVocabulary]} against {expectedVocabulary[firstVocabulary]}" : ""));

        var merges = model["merges"]!.AsArray().Select(m => ((string)m![0]!, (string)m[1]!)).ToList();
        var expectedMerges = reference["merges"]!.AsArray().Select(m => ((string)m![0]!, (string)m[1]!)).ToList();
        int firstMerge = Enumerable.Range(0, Math.Min(merges.Count, expectedMerges.Count)).FirstOrDefault(i => merges[i] != expectedMerges[i], -1);
        Check(merges.Count == expectedMerges.Count && firstMerge < 0,
            $"{folder}: {merges.Count} merges (expected {expectedMerges.Count}); first difference at {firstMerge}"
            + (firstMerge >= 0 ? $": {merges[firstMerge]} against {expectedMerges[firstMerge]}" : ""));

        foreach (var encoding in reference["encodings"]!.AsArray())
        {
            string text = (string)encoding!["text"]!;
            var expected = encoding["ids"]!.AsArray().Select(i => (int)i!).ToArray();
            var ids = trained.Encode(text).ToArray();
            Check(ids.SequenceEqual(expected), $"{folder}: '{text}' encodes to [{string.Join(", ", ids)}], Hugging Face to [{string.Join(", ", expected)}]");
            Check(trained.Decode(ids) == text || folder == "metaspace", $"{folder}: '{text}' decodes back");
        }

        var file = BpeTokenizer.Load(PretrainData($"{folder}/tokenizer.json"));
        foreach (var line in File.ReadLines(PretrainData("corpus.txt")).Take(200))
        {
            Check(trained.Encode(line).SequenceEqual(file.Encode(line)), $"{folder}: the trained tokenizer and Hugging Face's file encode '{line}' alike");
        }
    }

    private static void TokenizerTrainingByteLevel(Device device)
    {
        if (!FirstRun(nameof(TokenizerTrainingByteLevel)))
        {
            return;
        }

        var trained = TokenizerTrainers.Get("bpe").Train([PretrainData("corpus.txt")], ReferenceTrainingOptions("tokenizer"));
        CheckTrainedAgainstReference(trained, "tokenizer");
    }

    private static void TokenizerTrainingFull(Device device)
    {
        if (!FirstRun(nameof(TokenizerTrainingFull)))
        {
            return;
        }

        var trained = TokenizerTrainers.Get("bpe").Train([PretrainData("corpus.txt")], ReferenceTrainingOptions("tokenizer-full"));
        CheckTrainedAgainstReference(trained, "tokenizer-full");
    }

    private static void TokenizerTrainingMetaspace(Device device)
    {
        if (!FirstRun(nameof(TokenizerTrainingMetaspace)))
        {
            return;
        }

        var trained = TokenizerTrainers.Get("bpe").Train([PretrainData("corpus.txt")], ReferenceTrainingOptions("metaspace"));
        CheckTrainedAgainstReference(trained, "metaspace");
    }

    private static void TokenizerTrainingSave(Device device)
    {
        if (!FirstRun(nameof(TokenizerTrainingSave)))
        {
            return;
        }

        var trained = TokenizerTrainers.Get("bpe").Train([PretrainData("corpus.txt")], ReferenceTrainingOptions("tokenizer"));
        string folder = TempFolder("tokenizer-training");
        try
        {
            trained.Save(folder);
            var loaded = BpeTokenizer.Load(folder);
            foreach (var line in File.ReadLines(PretrainData("corpus.txt")).Take(100).Append("<|im_start|>user\nرقم 12<|im_end|>\n"))
            {
                Check(loaded.Encode(line).SequenceEqual(trained.Encode(line)), $"saved and loaded: '{line}' encodes alike");
            }

            Check(loaded.IsSpecial(loaded.IdOf("<|im_end|>")!.Value) && loaded.VocabularySize == trained.VocabularySize, "special tokens and the vocabulary survive the file");
            Conformance.Check(loaded, trained).ThrowIfFailed();

            // A file read from Hugging Face writes back the same model vocabulary and merges (added tokens outside the
            // model's vocabulary stay outside it).
            var original = JsonNode.Parse(File.ReadAllText(PretrainData("tokenizer/tokenizer.json")))!["model"]!;
            var rewritten = BpeTokenizer.Load(PretrainData("tokenizer/tokenizer.json")).ToJson()["model"]!;
            Check(JsonNode.DeepEquals(original["vocab"], rewritten["vocab"]) && JsonNode.DeepEquals(original["merges"], rewritten["merges"]),
                "a loaded tokenizer.json saves to the same vocabulary and merges");
            var gemma = BpeTokenizer.Load(TestData("vlm/tiny-gemma3"));
            var gemmaModel = JsonNode.Parse(File.ReadAllText(TestData("vlm/tiny-gemma3/tokenizer.json")))!["model"]!;
            Check(JsonNode.DeepEquals(gemmaModel["vocab"], gemma.ToJson()["model"]!["vocab"]), "a SentencePiece-style file keeps its vocabulary");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void TokenizerTrainingRegistry(Device device)
    {
        if (!FirstRun(nameof(TokenizerTrainingRegistry)))
        {
            return;
        }

        foreach (string name in new[] { "bpe", "unigram", "wordpiece" })
        {
            Check(TokenizerTrainers.Names.Contains(name) && TokenizerTrainers.Origin(name) == Idrak.Abstraction.Overrides.Library, $"{name} is the library's");
            Check(TokenizerTrainers.Describe().Contains($"{name}:", StringComparison.Ordinal), TokenizerTrainers.Describe());
        }
        string refused = "";
        try
        {
            TokenizerTrainers.Get("nope");
        }
        catch (NotSupportedException e)
        {
            refused = e.Message;
        }

        Check(refused.Contains("is not registered", StringComparison.Ordinal) && refused.Contains("TokenizerTrainers.Register", StringComparison.Ordinal), $"an unregistered trainer: {refused}");
        var bpe = TokenizerTrainers.Get("bpe");
        string corpus = PretrainData("corpus.txt");
        Throws<ArgumentException>(() => bpe.Train([corpus], new TokenizerTrainingOptions { VocabularySize = 0 }), "vocabulary 0");
        Throws<ArgumentException>(() => bpe.Train([corpus], new TokenizerTrainingOptions { PreTokenizer = "whitespace" }), "an unknown pre-tokenizer");
        Throws<ArgumentException>(() => bpe.Train([], new TokenizerTrainingOptions()), "no files");
        Throws<FileNotFoundException>(() => bpe.Train([corpus + ".missing"], new TokenizerTrainingOptions()), "a missing file");

        var options = ReferenceTrainingOptions("tokenizer");
        var one = bpe.Train([corpus], options with { Workers = 1 }).ToJson()["model"]!;
        var many = bpe.Train([corpus], options with { Workers = 4 }).ToJson()["model"]!;
        Check(JsonNode.DeepEquals(one, many), "one worker and four learn the same tokenizer");

        var stages = new System.Collections.Concurrent.ConcurrentQueue<TokenizerTrainingProgress>();
        bpe.Train([corpus], options with { VocabularySize = 300 }, new SynchronousProgress<TokenizerTrainingProgress>(stages.Enqueue));
        Check(stages.Any(p => p.Stage == "words" && p.Done == p.Total) && stages.Any(p => p.Stage == "merges"), "progress for both stages");
    }

    private static void TokenizerTrainingLineEnds(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                              // host work: once
        }

        string folder = Path.Combine(Path.GetTempPath(), $"idrak-line-ends-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        try
        {
            string[] lines = ["the quick brown fox", "jumps over the lazy dog", "a quick fox, a lazy dog", "مرحبا بكم في إدراك", "the end"];
            string lf = Path.Combine(folder, "lf.txt"), crlf = Path.Combine(folder, "crlf.txt");
            File.WriteAllText(lf, string.Join("\n", lines) + "\n");
            File.WriteAllText(crlf, string.Join("\r\n", lines) + "\r\n");
            var trainer = TokenizerTrainers.Get(TokenizerTrainers.Bpe);
            static string[] Vocabulary(IPretrainedTokenizer t) => [.. Enumerable.Range(0, t.VocabularySize).Select(t.TokenOf)];
            foreach (string pre in new[] { TokenizerTrainers.ByteLevel, TokenizerTrainers.Metaspace })
            {
                var options = new TokenizerTrainingOptions { VocabularySize = 300, MinFrequency = 1, PreTokenizer = pre, Workers = 1 };
                var fromLf = Vocabulary(trainer.Train([lf], options));
                var fromCrlf = Vocabulary(trainer.Train([crlf], options));
                Check(fromLf.SequenceEqual(fromCrlf), $"{pre}: the CRLF file gives the LF file's {fromLf.Length} tokens in order (got {fromCrlf.Length})");
                var kept = Vocabulary(trainer.Train([crlf], options with { LineEnds = "keep" }));
                Check(!kept.SequenceEqual(fromLf), $"{pre}: LineEnds \"keep\" learns the \\r of the CRLF file");
            }

            ThrowsNaming<ArgumentException>(() => trainer.Train([lf], new TokenizerTrainingOptions { LineEnds = "cr" }), "an unknown LineEnds", "LineEnds", "'cr'");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void TokenizerTrainingUnigramReference(Device device)
    {
        if (!FirstRun(nameof(TokenizerTrainingUnigramReference)))
        {
            return;
        }

        foreach (var (name, entry) in TrainingReference("unigram.json"))
        {
            var options = UnigramOptions(entry!["options"]!);
            var watch = Stopwatch.StartNew();
            var trained = TokenizerTrainers.Get(TokenizerTrainers.Unigram).Train(TrainingCorpus, options);
            double seconds = watch.Elapsed.TotalSeconds;
            var pieces = Vocabulary(trained);
            var expected = entry["pieces"]!.AsArray().Select(p => (Piece: (string)p![0]!, Type: (int)p[2]!)).ToList();
            var meta = expected.TakeWhile(p => p.Type != 1).Select(p => p.Piece).ToList();
            Check(pieces.Take(meta.Count).SequenceEqual(meta), $"{name}: the special and byte pieces at SentencePiece's ids: [{string.Join(", ", pieces.Take(Math.Min(8, meta.Count)))}...] against [{string.Join(", ", meta.Take(8))}...]");
            var reference = expected.Where(p => p.Type == 1).Select(p => p.Piece).ToList();
            var learned = pieces.Skip(meta.Count).ToHashSet(StringComparer.Ordinal);
            double overlap = reference.Count(learned.Contains) / (double)reference.Count;
            var scores = trained.ToJson()["model"]!["vocab"]!.AsArray().Select(p => (double)p![1]!).ToArray();
            var expectedScores = entry["pieces"]!.AsArray().Select(p => (double)p![2 - 1]!).ToArray();
            int sameId = Enumerable.Range(0, Math.Min(pieces.Length, expected.Count)).Count(i => pieces[i] == expected[i].Piece);
            double scoreDifference = Enumerable.Range(0, Math.Min(pieces.Length, expected.Count)).Where(i => pieces[i] == expected[i].Piece)
                .Select(i => Math.Abs(scores[i] - expectedScores[i])).DefaultIfEmpty(0).Max();
            Console.WriteLine($"    {name}: {pieces.Length} pieces (sentencepiece {expected.Count}), {overlap:P2} of sentencepiece's learned pieces, {sameId} at the same id (scores apart by {scoreDifference:G3} at most), {seconds:F2} s");
            Check(pieces.Length == expected.Count, $"{name}: {pieces.Length} pieces, sentencepiece learned {expected.Count}");
            Check(overlap >= UnigramOverlapFloor, $"{name}: {overlap:P2} of sentencepiece's learned pieces, below the measured floor {UnigramOverlapFloor:P0}; missing: {string.Join(" ", reference.Where(p => !learned.Contains(p)).Take(30))}");
        }
    }

    private static void TokenizerTrainingUnigramFiles(Device device)
    {
        if (!FirstRun(nameof(TokenizerTrainingUnigramFiles)))
        {
            return;
        }

        string folder = TempFolder("unigram-training");
        try
        {
            var trained = TokenizerTrainers.Get(TokenizerTrainers.Unigram).Train(TrainingCorpus,
                new TokenizerTrainingOptions { VocabularySize = 1200, SpecialTokens = ["<unk>", "<s>", "</s>", "<|im_end|>"], ByteFallback = true });
            Check(trained.Model == TokenizerModels.Unigram, $"the trained model is {trained.Model}");
            var companions = trained as ITokenizerCompanionFiles;
            Check(companions is not null && companions.CompanionFiles.SequenceEqual(["tokenizer.model"]), "the trained tokenizer writes tokenizer.model beside tokenizer.json");
            trained.Save(folder);
            companions!.SaveCompanion("tokenizer.model", Path.Combine(folder, "tokenizer.model"));
            ThrowsNaming<ArgumentException>(() => companions.SaveCompanion("vocab.txt", Path.Combine(folder, "vocab.txt")), "an unknown companion file", "vocab.txt");
            var json = PretrainedTokenizers.Load(folder);
            var model = PretrainedTokenizers.Load(folder, new TokenizerOptions { Format = TokenizerFiles.SentencePiece });
            Check(json.Model == TokenizerModels.Unigram && model.Model == TokenizerModels.Unigram && Vocabulary(json).SequenceEqual(Vocabulary(trained)) && Vocabulary(model).SequenceEqual(Vocabulary(trained)),
                "tokenizer.json and tokenizer.model read back to the trained vocabulary");
            var failures = new List<string>();
            foreach (string line in TrainingLines().Append("<|im_end|> 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 ☃ \u0007 مرحبا"))
            {
                var ids = trained.Encode(line).ToArray();
                if (!json.Encode(line).SequenceEqual(ids) || !model.Encode(line).SequenceEqual(ids))
                {
                    failures.Add($"'{line}': trained [{string.Join(", ", ids)}], tokenizer.json [{string.Join(", ", json.Encode(line))}], tokenizer.model [{string.Join(", ", model.Encode(line))}]");
                }

                string text = System.Text.RegularExpressions.Regex.Replace(line, " {2,}", " ").Trim(' ');
                // Not held: the unknown piece decodes as " ⁇ " (as sentencepiece writes it), and text glued to a special token
                // after it is a segment of its own, so it decodes with the dummy prefix's space (T1's added tokens).
                bool special = line.Contains("<unk>", StringComparison.Ordinal) || System.Text.RegularExpressions.Regex.IsMatch(line, @"(?:<s>|</s>|<\|im_end\|>)(?=\S)");
                if (!special && trained.Decode(ids) != text)
                {
                    failures.Add($"'{line}' decodes to '{trained.Decode(ids)}'");
                }
            }

            Check(failures.Count == 0, $"{failures.Count} lines differ: {string.Join("\n", failures.Take(5))}");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void TokenizerTrainingWordPieceReference(Device device)
    {
        if (!FirstRun(nameof(TokenizerTrainingWordPieceReference)))
        {
            return;
        }

        var reference = TrainingReference("wordpiece.json");
        var trained = TokenizerTrainers.Get(TokenizerTrainers.WordPiece).Train(TrainingCorpus, WordPieceOptions(reference["options"]!));
        Check(trained.Model == TokenizerModels.WordPiece, $"the trained model is {trained.Model}");
        var vocabulary = Vocabulary(trained);
        var expected = reference["fixed"]!.AsArray().Select(t => (string)t!).ToArray();
        int first = Enumerable.Range(0, Math.Min(vocabulary.Length, expected.Length)).FirstOrDefault(i => vocabulary[i] != expected[i], -1);
        Check(vocabulary.Length == expected.Length && first < 0,
            $"{vocabulary.Length} tokens (the algorithm with fixed ties: {expected.Length}); first difference at {first}" + (first >= 0 ? $": '{vocabulary[first]}' against '{expected[first]}'" : ""));
        var huggingFace = reference["huggingface"]!.AsArray().Select(t => (string)t!).ToList();
        var learned = vocabulary.ToHashSet(StringComparer.Ordinal);
        double overlap = huggingFace.Count(learned.Contains) / (double)huggingFace.Count;
        Console.WriteLine($"    wordpiece: {vocabulary.Length} tokens, {overlap:P2} of Hugging Face's run ({huggingFace.Count} tokens)");
        Check(overlap >= WordPieceOverlapFloor, $"{overlap:P2} of Hugging Face's WordPiece run, below the measured floor {WordPieceOverlapFloor:P0}");
    }

    private static void TokenizerTrainingWordPieceFiles(Device device)
    {
        if (!FirstRun(nameof(TokenizerTrainingWordPieceFiles)))
        {
            return;
        }

        string folder = TempFolder("wordpiece-training");
        try
        {
            foreach (string scoring in new[] { TokenizerTrainers.Frequency, TokenizerTrainers.Likelihood })
            {
                var options = WordPieceOptions(TrainingReference("wordpiece.json")["options"]!) with { Scoring = scoring };
                var trained = TokenizerTrainers.Get(TokenizerTrainers.WordPiece).Train(TrainingCorpus, options);
                string file = Path.Combine(folder, $"{scoring}.json");
                trained.Save(file);
                var loaded = PretrainedTokenizers.Load(file);
                Check(Vocabulary(loaded).SequenceEqual(Vocabulary(trained)), $"{scoring}: the saved tokenizer.json reads back to the vocabulary");
                var pipeline = new TokenizerPipeline(trained.ToJson());
                using var pieces = new PreTokenizedText();
                var failures = new List<string>();
                foreach (string line in TrainingLines().Where(l => l.Length > 0 && !options.SpecialTokens.Any(t => l.Contains(t, StringComparison.OrdinalIgnoreCase))))
                {
                    var ids = trained.Encode(line).ToArray();
                    if (!loaded.Encode(line).SequenceEqual(ids))
                    {
                        failures.Add($"'{line}': the loaded tokenizer encodes otherwise");
                    }

                    pipeline.Run(line, 0, line.Length, atStart: true, pieces);
                    var words = Enumerable.Range(0, pieces.Pieces.Count).Select(i => pieces.Piece(i).ToString())
                        .Select(w => w.EnumerateRunes().Count() > 100 ? "[UNK]" : w).ToList();
                    var spelled = new List<string>();
                    foreach (int id in ids)
                    {
                        string token = trained.TokenOf(id);
                        if (token.StartsWith("##", StringComparison.Ordinal) && spelled.Count > 0)
                        {
                            spelled[^1] += token[2..];
                        }
                        else
                        {
                            spelled.Add(token);
                        }
                    }

                    if (!spelled.SequenceEqual(words))
                    {
                        failures.Add($"'{line}': pieces spell [{string.Join(" ", spelled)}], the words are [{string.Join(" ", words)}]");
                    }
                }

                Check(failures.Count == 0, $"{scoring}: {failures.Count} lines: {string.Join("\n", failures.Take(5))}");
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void TokenizerTrainingWorkers(Device device)
    {
        if (!FirstRun(nameof(TokenizerTrainingWorkers)))
        {
            return;
        }

        var unigram = new TokenizerTrainingOptions { VocabularySize = 900, SpecialTokens = ["<unk>", "<s>", "</s>"], SeedSize = 5000 };
        var wordpiece = WordPieceOptions(TrainingReference("wordpiece.json")["options"]!);
        foreach (var (name, options) in new[]
                 {
                     (TokenizerTrainers.Unigram, unigram), (TokenizerTrainers.WordPiece, wordpiece),
                     (TokenizerTrainers.WordPiece, wordpiece with { Scoring = TokenizerTrainers.Likelihood }),
                 })
        {
            var trainer = TokenizerTrainers.Get(name);
            var one = trainer.Train(TrainingCorpus, options with { Workers = 1 }).ToJson()["model"]!;
            var many = trainer.Train(TrainingCorpus, options with { Workers = 4 }).ToJson()["model"]!;
            Check(JsonNode.DeepEquals(one, many), $"{name} ({options.Scoring}): one worker and four learn the same model");
        }

        var frequency = Vocabulary(TokenizerTrainers.Get(TokenizerTrainers.WordPiece).Train(TrainingCorpus, wordpiece));
        var likelihood = Vocabulary(TokenizerTrainers.Get(TokenizerTrainers.WordPiece).Train(TrainingCorpus, wordpiece with { Scoring = TokenizerTrainers.Likelihood }));
        Check(!frequency.SequenceEqual(likelihood), "likelihood scoring learns another vocabulary than frequency");
    }

    private static void TokenizerTrainingArabic(Device device)
    {
        if (!FirstRun(nameof(TokenizerTrainingArabic)))
        {
            return;
        }

        string folder = TempFolder("arabic-training");
        try
        {
            string[] arabic = [TokenizerTrainers.Tatweel, TokenizerTrainers.Diacritics];
            var unigram = TokenizerTrainers.Get(TokenizerTrainers.Unigram).Train(TrainingCorpus,
                new TokenizerTrainingOptions { VocabularySize = 900, SpecialTokens = ["<unk>", "<s>", "</s>"], Arabic = arabic });
            var wordpiece = TokenizerTrainers.Get(TokenizerTrainers.WordPiece).Train(TrainingCorpus,
                WordPieceOptions(TrainingReference("wordpiece.json")["options"]!) with { Arabic = arabic });
            unigram.Save(folder);
            ((ITokenizerCompanionFiles)unigram).SaveCompanion("tokenizer.model", Path.Combine(folder, "tokenizer.model"));
            var fromModel = PretrainedTokenizers.Load(folder, new TokenizerOptions { Format = TokenizerFiles.SentencePiece });
            (string Marked, string Plain)[] pairs = [("كتـــابٌ جَمِيلٌ", "كتاب جميل"), ("الْعَرَبِيَّةُ لُغَـةٌ", "العربية لغة"), ("قُرْآنٌ ٰ", "قرآن")];
            foreach (var (name, tokenizer) in new[] { ("unigram", unigram), ("unigram from tokenizer.model", fromModel), ("wordpiece", wordpiece) })
            {
                foreach (var (marked, plain) in pairs)
                {
                    Check(tokenizer.Encode(marked).SequenceEqual(tokenizer.Encode(plain)),
                        $"{name}: '{marked}' encodes as '{plain}': [{string.Join(", ", tokenizer.Encode(marked))}] against [{string.Join(", ", tokenizer.Encode(plain))}]");
                }

                Check(!Vocabulary(tokenizer).Any(t => t.Contains('\u0640') || t.Contains('\u064E')), $"{name}: no learned token holds tatweel or a fatha");
            }

            string unigramNormalizer = unigram.ToJson()["normalizer"]!.ToJsonString();
            string wordpieceNormalizer = wordpiece.ToJson()["normalizer"]!.ToJsonString();
            Check(unigramNormalizer.Contains("\"Precompiled\"", StringComparison.Ordinal), $"unigram: the removals are a precompiled map, as tokenizer.model holds them: {unigramNormalizer[..Math.Min(200, unigramNormalizer.Length)]}");
            Check(wordpieceNormalizer.Contains("\"Replace\"", StringComparison.Ordinal) && wordpieceNormalizer.Contains("BertNormalizer", StringComparison.Ordinal),
                $"wordpiece: the removals are Replace normalizers before BERT's: {wordpieceNormalizer}");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void TokenizerTrainingNewOptions(Device device)
    {
        if (!FirstRun(nameof(TokenizerTrainingNewOptions)))
        {
            return;
        }

        string corpus = PretrainData("corpus.txt");
        var unigram = TokenizerTrainers.Get(TokenizerTrainers.Unigram);
        var wordpiece = TokenizerTrainers.Get(TokenizerTrainers.WordPiece);
        var good = new TokenizerTrainingOptions { VocabularySize = 400 };
        foreach (var (options, setting) in new (TokenizerTrainingOptions, string)[]
                 {
                     (good with { CharacterCoverage = 0 }, "CharacterCoverage"), (good with { CharacterCoverage = 1.5 }, "CharacterCoverage"),
                     (good with { ShrinkingFactor = 1 }, "ShrinkingFactor"), (good with { EmIterations = 0 }, "EmIterations"),
                     (good with { MaxPieceLength = 0 }, "MaxPieceLength"), (good with { SeedSize = 0 }, "SeedSize"),
                     (good with { Scoring = "best" }, "Scoring"), (good with { Arabic = ["kashida"] }, "Arabic"), (good with { UnknownToken = "" }, "UnknownToken"),
                     (good with { VocabularySize = 0 }, "VocabularySize"), (good with { LineEnds = "cr" }, "LineEnds"),
                 })
        {
            ThrowsNaming<ArgumentException>(() => unigram.Train([corpus], options), $"unigram, {setting}", setting);
            ThrowsNaming<ArgumentException>(() => wordpiece.Train([corpus], options), $"wordpiece, {setting}", setting);
        }

        ThrowsNaming<ArgumentException>(() => unigram.Train([corpus], good with { VocabularySize = 40 }), "unigram, a vocabulary smaller than the alphabet", "VocabularySize", "CharacterCoverage");
        ThrowsNaming<ArgumentException>(() => TokenizerTrainers.Get(TokenizerTrainers.Bpe).Train([corpus], good with { Arabic = ["tatweel"] }), "bpe, Arabic", "Arabic");
        Throws<FileNotFoundException>(() => unigram.Train([corpus + ".missing"], good), "unigram, a missing file");
        Throws<ArgumentException>(() => wordpiece.Train([], good), "wordpiece, no files");

        var stages = new System.Collections.Concurrent.ConcurrentQueue<TokenizerTrainingProgress>();
        unigram.Train([corpus], good with { SpecialTokens = ["<unk>"] }, new SynchronousProgress<TokenizerTrainingProgress>(stages.Enqueue));
        foreach (string stage in new[] { "words", "seeds", "em", "prune" })
        {
            Check(stages.Any(p => p.Stage == stage), $"unigram reports the stage {stage}");
        }

        Check(stages.Any(p => p.Stage == "em" && p.Done == p.Total && p.Total > 0), "the EM stage ends at its total");
    }
}
