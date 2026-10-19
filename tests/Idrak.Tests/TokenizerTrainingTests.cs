// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

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
        ("tokenizer training: the registry names bpe, refuses an unregistered trainer, bad options and missing files, and one worker counts the same words as many", TokenizerTrainingRegistry),
    ];

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

        Check(TokenizerTrainers.Names.Contains("bpe") && TokenizerTrainers.Origin("bpe") == Idrak.Abstraction.Overrides.Library, "bpe is the library's");
        Check(TokenizerTrainers.Describe().Contains("bpe:", StringComparison.Ordinal), TokenizerTrainers.Describe());
        string refused = "";
        try
        {
            TokenizerTrainers.Get("wordpiece");
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
}
