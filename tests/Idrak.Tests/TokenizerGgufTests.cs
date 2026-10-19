// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models;
using Idrak.Models.Abstractions;

// Plan 24, phase T3: GGUF tokenizers beyond gpt2 (llama, t5, bert), through the GgufTokenizers registry.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] TokenizerGgufGroup =
    [
        ("tokenizers: gguf: a llama (SentencePiece BPE) GGUF tokenizer writes the tokenizer.json its tokenizer.model reads to, byte for byte, and encodes and decodes every corpus line as it", TokenizersGgufLlama),
        ("tokenizers: gguf: a t5 (SentencePiece Unigram) GGUF tokenizer with its precompiled character map writes the tokenizer.json its tokenizer.model reads to, byte for byte", TokenizersGgufT5),
        ("tokenizers: gguf: a bert GGUF tokenizer in llama.cpp's form (▁ word starts, bare continuations) gives tokenizer.json's ids and special tokens on every corpus line", TokenizersGgufBert),
        ("tokenizers: gguf: a tokenizer nobody registered fails naming it and GgufTokenizers.Register; an app's converter is used, then unregistered", TokenizersGgufRegistry),
    ];

    // A GGUF holding only tokenizer metadata, prepared as GgufModel prepares a model's tokenizer; the folder it wrote.
    private static string PrepareGgufTokenizer(string root, string name, List<KeyValuePair<string, object>> metadata, List<string>? notes = null)
    {
        string path = Path.Combine(root, name + ".gguf");
        WriteGguf(path, [KeyValuePair.Create<string, object>("general.architecture", "llama"), .. metadata], []);
        string folder = Directory.CreateDirectory(Path.Combine(root, name)).FullName;
        using var file = GgufFile.Open(path);
        GgufModel.WriteTokenizer(file, folder, file.Get<string[]>("tokenizer.ggml.tokens", []), notes ?? []);
        return folder;
    }

    // The metadata llama.cpp's converter writes for a SentencePiece model.
    private static List<KeyValuePair<string, object>> SentencePieceGguf(string fixture, string model)
    {
        var spm = SentencePieceModel.Load(TestData($"tokenizers/{fixture}/tokenizer.model"));
        var metadata = new List<KeyValuePair<string, object>>
        {
            new("tokenizer.ggml.model", model),
            new("tokenizer.ggml.tokens", spm.Pieces.ToArray()),
            new("tokenizer.ggml.scores", spm.Scores.ToArray()),
            new("tokenizer.ggml.token_type", spm.Types.Select(t => (long)t).ToArray()),
            new("tokenizer.ggml.add_space_prefix", spm.NormalizerSpec.AddDummyPrefix),
            new("tokenizer.ggml.remove_extra_whitespaces", spm.NormalizerSpec.RemoveExtraWhitespaces),
        };
        foreach (var (key, id) in new[] { ("unknown", spm.TrainerSpec.UnkId), ("bos", spm.TrainerSpec.BosId), ("eos", spm.TrainerSpec.EosId), ("padding", spm.TrainerSpec.PadId) })
        {
            if (id >= 0)
            {
                metadata.Add(new($"tokenizer.ggml.{key}_token_id", (long)id));
            }
        }

        if (spm.NormalizerSpec.PrecompiledCharsmap.Length > 0)
        {
            metadata.Add(new("tokenizer.ggml.precompiled_charsmap", spm.NormalizerSpec.PrecompiledCharsmap));
        }

        return metadata;
    }

    // The tokenizer.json the tokenizer.model path writes, for comparing bytes.
    private static byte[] SentencePieceJson(string fixture, string root)
    {
        string path = Path.Combine(root, fixture + "-from-model.json");
        TokenizerJsonReader.Write(PretrainedTokenizers.Read(TestData($"tokenizers/{fixture}/tokenizer.model")), path);
        return File.ReadAllBytes(path);
    }

    private static void TokenizersGgufLlama(Device device)
    {
        var root = Directory.CreateTempSubdirectory("idrak-gguf-tokenizer-");
        try
        {
            string folder = PrepareGgufTokenizer(root.FullName, "llama", SentencePieceGguf("bpe", "llama"));
            Check(File.ReadAllBytes(Path.Combine(folder, "tokenizer.json")).AsSpan().SequenceEqual(SentencePieceJson("bpe", root.FullName)),
                "the GGUF's tokenizer.json equals the tokenizer.model's, byte for byte");
            var gguf = PretrainedTokenizers.Load(folder);
            var model = PretrainedTokenizers.Load(TestData("tokenizers/bpe/tokenizer.model"));
            Check(gguf.Model == TokenizerModels.Bpe, "a llama GGUF tokenizer is a BPE one");
            int lines = 0;
            foreach (string text in File.ReadAllText(TestData("tokenizers/corpus.txt")).Split('\n'))
            {
                var ids = gguf.Encode(text);
                Check(ids.SequenceEqual(model.Encode(text)) && gguf.Decode(ids) == model.Decode(ids), $"line {lines}: the GGUF encodes '{text}' as the tokenizer.model does");
                lines++;
            }
        }
        finally
        {
            root.Delete(recursive: true);
        }
    }

    private static void TokenizersGgufT5(Device device)
    {
        var root = Directory.CreateTempSubdirectory("idrak-gguf-tokenizer-");
        try
        {
            foreach (string fixture in new[] { "unigram", "unigram-bytes" })
            {
                string folder = PrepareGgufTokenizer(root.FullName, fixture, SentencePieceGguf(fixture, "t5"));
                Check(File.ReadAllBytes(Path.Combine(folder, "tokenizer.json")).AsSpan().SequenceEqual(SentencePieceJson(fixture, root.FullName)),
                    $"{fixture}: the GGUF's tokenizer.json equals the tokenizer.model's, byte for byte");
                Check(PretrainedTokenizers.Read(folder).ModelType == "Unigram", $"{fixture}: a t5 GGUF tokenizer is a Unigram one");
            }
        }
        finally
        {
            root.Delete(recursive: true);
        }
    }

    private static void TokenizersGgufBert(Device device)
    {
        var root = Directory.CreateTempSubdirectory("idrak-gguf-tokenizer-");
        try
        {
            // llama.cpp's convert_hf_to_gguf writes BERT pieces "phantom space" style.
            string fixture = TestData("tokenizers/wordpiece-uncased");
            string[] vocabulary = File.ReadAllText(Path.Combine(fixture, "vocab.txt")).Split('\n')[..^1];
            string[] phantom = [.. vocabulary.Select(t => t.StartsWith('[') && t.EndsWith(']') ? t : t.StartsWith("##", StringComparison.Ordinal) ? t[2..] : "▁" + t)];
            long Id(string token) => Array.IndexOf(vocabulary, token);
            var notes = new List<string>();
            string folder = PrepareGgufTokenizer(root.FullName, "bert", [
                new("tokenizer.ggml.model", "bert"), new("tokenizer.ggml.tokens", phantom),
                new("tokenizer.ggml.unknown_token_id", Id("[UNK]")), new("tokenizer.ggml.seperator_token_id", Id("[SEP]")),
                new("tokenizer.ggml.cls_token_id", Id("[CLS]")), new("tokenizer.ggml.padding_token_id", Id("[PAD]")), new("tokenizer.ggml.mask_token_id", Id("[MASK]")),
            ], notes);
            Check(notes.Any(n => n.Contains("lowercases", StringComparison.Ordinal)), "the note says the GGUF tokenizer lowercases");
            var gguf = PretrainedTokenizers.Load(folder);
            var json = PretrainedTokenizers.Load(fixture);
            Check(gguf.Model == TokenizerModels.WordPiece && gguf.VocabularySize == json.VocabularySize, "a bert GGUF tokenizer is a WordPiece one of the same size");
            int lines = 0;
            foreach (string text in File.ReadAllText(TestData("tokenizers/corpus.txt")).Split('\n'))
            {
                Check(gguf.Encode(text).SequenceEqual(json.Encode(text)), $"line {lines}: the GGUF encodes '{text}' as tokenizer.json does");
                lines++;
            }

            Check(gguf.IsSpecial(gguf.IdOf("[CLS]")!.Value) && ((WordPieceTokenizer)gguf).PostProcessor is not null, "special tokens and [CLS] … [SEP]");
        }
        finally
        {
            root.Delete(recursive: true);
        }
    }

    private static void TokenizersGgufRegistry(Device device)
    {
        Check(new[] { "gpt2", "llama", "t5", "bert" }.All(GgufTokenizers.Names.Contains), $"the built-in GGUF tokenizers (registered: {string.Join(", ", GgufTokenizers.Names)})");
        var root = Directory.CreateTempSubdirectory("idrak-gguf-tokenizer-");
        try
        {
            List<KeyValuePair<string, object>> metadata = [new("tokenizer.ggml.model", "imaginary"), new("tokenizer.ggml.tokens", new[] { "a", "b" })];
            Expect<NotSupportedException>(() => PrepareGgufTokenizer(root.FullName, "unknown", metadata), "GGUF tokenizer 'imaginary' is not registered", "an unregistered tokenizer");
            Expect<NotSupportedException>(() => PrepareGgufTokenizer(root.FullName, "unknown2", metadata), "add it with GgufTokenizers.Register", "the hint");
            GgufTokenizers.Register("imaginary", (file, tokens, notes) => GgufTokenizers.Default("bert")!(file, ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "▁a"], notes));
            try
            {
                string folder = PrepareGgufTokenizer(root.FullName, "app", metadata);
                Check(PretrainedTokenizers.Load(folder).Encode("a").SequenceEqual([5]), "the app's converter is used");
            }
            finally
            {
                GgufTokenizers.Unregister("imaginary");
            }

            Check(!GgufTokenizers.Names.Contains("imaginary"), "unregistered");
        }
        finally
        {
            root.Delete(recursive: true);
        }
    }
}
