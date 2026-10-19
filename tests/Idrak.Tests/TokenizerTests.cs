// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Models;
using Idrak.Models.Abstractions;

// Plan 24, phase L0: the tokenizer model registry, the file readers and the one loader.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] TokenizerGroup =
    [
        ("tokenizers: every BPE fixture loads through PretrainedTokenizers.Load (and BpeTokenizer.Load) to the ids, refusals and text BpeTokenizer.Load gave before the registry (tokenizers/bpe-baseline.json)", TokenizersBpeBaseline),
        ("tokenizers: the model registry names BPE, builds it from a definition, refuses an unknown model type naming the registered ones and TokenizerModels.Register, and an app's model type is built and unregistered", TokenizersModelRegistry),
        ("tokenizers: the readers find tokenizer.json in a folder and as a file, a list vocabulary is read as pieces and scores, Format and IDRAK_TOKENIZER_FORMAT pick the reader, and a folder without a tokenizer names the readers", TokenizersFileReaders),
        ("tokenizers: an app's model overriding BPE is guarded: under FallBack one that fails to build or to encode (a text, a range, a batch, padded as the built-in pads it) is answered by the built-in, under Throw its failure reaches the caller", TokenizersGuardedModel),
        ("tokenizers: the piece cache is a setting (PieceCacheLimit, IDRAK_TOKENIZER_CACHE; 0 caches nothing and gives the same ids), a bad value is refused naming it, and PretrainedOptions carries the settings", TokenizersPieceCache),
        ("tokenizers: the built-in BPE passes the pretrained tokenizer suite (ranges, tokens by name, tokenizer.json naming the model) and a saved tokenizer loads back to the same ids", TokenizersPretrainedSuite),
        ("tokenizers: bpe: EncodeRange into a reused list allocates 0 bytes once warm on every corpus line (SentencePiece-style from tokenizer.json and tokenizer.model, byte level, with the piece cache and without it)", TokenizersBpeAllocatesNothing),
    ];

    // Plan 24's rule: after warm-up, encoding into a caller's list allocates nothing, cache hit or miss (a miss merges in the
    // caller's list; long pieces, which are never cached, merge in the thread's scratch).
    private static void TokenizersBpeAllocatesNothing(Device device)
    {
        _ = device;
        var lines = TokenizerCorpusLines();
        var ids = new List<int>(4096);
        var sources = new (string Name, string Path, TokenizerOptions Options)[]
        {
            ("bpe from tokenizer.json", TokenizerData("bpe"), new() { Format = TokenizerFiles.TokenizerJson }),
            ("bpe from tokenizer.model", TokenizerData("bpe"), new() { Format = TokenizerFiles.SentencePiece }),
            ("byte-level-roberta", TokenizerData("byte-level-roberta"), new()),
            ("byte-level-trim", TokenizerData("byte-level-trim"), new()),
            ("pretrain/tokenizer", PretrainData("tokenizer"), new()),
        };
        foreach (var (name, path, options) in sources)
        {
            foreach (int cache in new[] { TokenizerOptions.DefaultPieceCacheLimit, 0 })
            {
                var tokenizer = PretrainedTokenizers.Load(path, options with { PieceCacheLimit = cache });
                Check(tokenizer.Model == TokenizerModels.Bpe, $"{name}: model {tokenizer.Model}");

                // The lines the vocabulary takes (a tokenizer without an unknown token refuses a character it lacks).
                var taken = lines.Where(line =>
                {
                    try
                    {
                        ids.Clear();
                        tokenizer.EncodeRange(line, 0, line.Length, ids);
                        return true;
                    }
                    catch (InvalidDataException)
                    {
                        return false;
                    }
                }).ToArray();
                Check(taken.Length > lines.Length / 2, $"{name}: only {taken.Length} of {lines.Length} corpus lines encode");
                for (int warm = 0; warm < 3; warm++)
                {
                    foreach (var line in taken)
                    {
                        ids.Clear();
                        tokenizer.EncodeRange(line, 0, line.Length, ids);
                    }
                }

                long before = GC.GetAllocatedBytesForCurrentThread();
                for (int run = 0; run < 3; run++)
                {
                    foreach (var line in taken)
                    {
                        ids.Clear();
                        tokenizer.EncodeRange(line, 0, line.Length, ids);
                    }
                }

                long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
                if (allocated != 0)
                {
                    // Name the lines that allocate.
                    var culprits = new List<string>();
                    foreach (var (line, i) in taken.Select((l, i) => (l, i)))
                    {
                        long start = GC.GetAllocatedBytesForCurrentThread();
                        ids.Clear();
                        tokenizer.EncodeRange(line, 0, line.Length, ids);
                        if (GC.GetAllocatedBytesForCurrentThread() - start is var bytes and > 0)
                        {
                            culprits.Add($"line {i} ({bytes} bytes, {line.Length} characters) '{Escaped(line[..Math.Min(40, line.Length)])}'");
                        }
                    }

                    Check(false, $"{name}, cache {cache}: a warm pass allocated {allocated} bytes over {taken.Length * 3} lines:\n  {string.Join("\n  ", culprits.Take(8))}");
                }
            }
        }
    }

    private static void TokenizersBpeBaseline(Device device)
    {
        var baseline = JsonNode.Parse(File.ReadAllText(TestData("tokenizers/bpe-baseline.json")))!.AsObject();
        string[] texts = [.. baseline["texts"]!.AsArray().Select(t => (string)t!),
            .. File.ReadAllLines(TestData((string)baseline["corpus"]!))];
        var fixtures = baseline["fixtures"]!.AsObject();
        Check(fixtures.Count > 0, "the baseline lists fixtures");
        foreach (var (fixture, expected) in fixtures)
        {
            string path = TestData(fixture);
            var loaded = PretrainedTokenizers.Load(Path.GetDirectoryName(path)!);
            Check(loaded is BpeTokenizer && loaded.Model == TokenizerModels.Bpe, $"{fixture}: the loader builds the BPE model, not {loaded.GetType().Name}");
            foreach (var tokenizer in new IPretrainedTokenizer[] { loaded, BpeTokenizer.Load(path) })
            {
                var (ids, decoded) = BaselineHashes(tokenizer, texts);
                Check(ids == (string)expected!["ids"]!, $"{fixture}: the ids (or refusals) changed from the baseline");
                Check(decoded == (string)expected["decoded"]!, $"{fixture}: the decoded texts changed from the baseline");
            }
        }
    }

    // As tools wrote bpe-baseline.json: SHA-256 of each text's id count and ids (int32 little-endian), or of "!", the
    // refusal's type and message and a NUL; and of each decoded text (UTF-8) followed by a NUL.
    private static (string Ids, string Decoded) BaselineHashes(IPretrainedTokenizer tokenizer, string[] texts)
    {
        using var ids = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        using var decoded = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var four = new byte[4];
        foreach (string text in texts)
        {
            IReadOnlyList<int> encoded;
            try
            {
                encoded = tokenizer.Encode(text);
            }
            catch (Exception e) when (e is InvalidDataException or NotSupportedException)
            {
                ids.AppendData(Encoding.UTF8.GetBytes("!" + e.GetType().Name + ": " + e.Message + "\u0000"));
                continue;
            }

            BitConverter.TryWriteBytes(four, encoded.Count);
            ids.AppendData(four);
            foreach (int id in encoded)
            {
                BitConverter.TryWriteBytes(four, id);
                ids.AppendData(four);
            }

            decoded.AppendData(Encoding.UTF8.GetBytes(tokenizer.Decode(encoded) + "\u0000"));
        }

        return (Convert.ToHexStringLower(ids.GetHashAndReset()), Convert.ToHexStringLower(decoded.GetHashAndReset()));
    }

    private static void TokenizersModelRegistry(Device device)
    {
        Check(TokenizerModels.Names.Contains(TokenizerModels.Bpe) && TokenizerModels.Origin(TokenizerModels.Bpe) == Overrides.Library,
            $"the library registers BPE (registered: {string.Join(", ", TokenizerModels.Names)})");
        string folder = PretrainData("tokenizer");
        var definition = PretrainedTokenizers.Read(folder);
        Check(definition.ModelType == TokenizerModels.Bpe && definition.Vocabulary is { Count: > 0 } && definition.Merges is { Length: > 0 } && definition.Pieces is null,
            "a BPE tokenizer.json reads to its vocabulary object and merges");
        Check(definition.Json["model"]?["vocab"] is null && definition.Json["model"]?["merges"] is null, "the definition's JSON leaves the big tables out");
        var built = TokenizerModels.Create(definition);
        Check(built.Encode("hello world").SequenceEqual(BpeTokenizer.Load(folder).Encode("hello world")), "the registry's BPE encodes as BpeTokenizer.Load's");

        var unknown = new TokenizerDefinition { Source = "test", Json = new JsonObject { ["model"] = new JsonObject { ["type"] = "Imaginary" } } };
        Expect<NotSupportedException>(() => TokenizerModels.Create(unknown), "Tokenizer model 'Imaginary' is not registered (registered: ", "an unknown model type");
        Expect<NotSupportedException>(() => TokenizerModels.Create(unknown), "add it with TokenizerModels.Register", "the hint to register one");
        Expect<InvalidDataException>(() => TokenizerModels.Create(new TokenizerDefinition { Source = "untyped", Json = [] }), "names no model type", "a definition without a model type");
        Expect<NotSupportedException>(() => BpeTokenizer.FromJson(new JsonObject { ["model"] = new JsonObject { ["type"] = "Imaginary", ["vocab"] = new JsonObject() } }),
            "PretrainedTokenizers.Load", "BpeTokenizer pointing to the loader for other model types");

        // An app's model type: built for files naming it, gone when unregistered.
        TokenizerModels.Register("Imaginary", (d, o) => BpeTokenizer.Load(folder));
        try
        {
            Check(TokenizerModels.Origin("Imaginary") is { } origin && origin != Overrides.Library, "the app's model type is the app's");
            Check(TokenizerModels.Create(unknown).Encode("hello world").Count > 0, "the app's model type is built");
        }
        finally
        {
            TokenizerModels.Unregister("Imaginary");
        }

        Check(!TokenizerModels.Names.Contains("Imaginary") && TokenizerModels.Find("Imaginary") is null, "unregistering leaves the registry as found");
        Check(!TokenizerModels.Unregister(TokenizerModels.Bpe), "the built-in BPE cannot be unregistered");
    }

    private static void TokenizersFileReaders(Device device)
    {
        Check(TokenizerFiles.Names.FirstOrDefault() == TokenizerFiles.TokenizerJson, $"tokenizer.json is the first reader (readers: {string.Join(", ", TokenizerFiles.Names)})");
        string folder = PretrainData("tokenizer");
        string file = Path.Combine(folder, "tokenizer.json");
        Check(PretrainedTokenizers.Find(folder) is { } inFolder && inFolder.File == file && inFolder.Reader.Name == TokenizerFiles.TokenizerJson, "a folder's tokenizer.json is found");
        Check(PretrainedTokenizers.Find(file) is { } asFile && asFile.File == file, "a tokenizer.json given as a file is found");

        var temporary = Directory.CreateTempSubdirectory("idrak-tokenizers-");
        string? previous = Environment.GetEnvironmentVariable(TokenizerOptions.FormatVariable);
        try
        {
            // A Unigram tokenizer.json: the list vocabulary becomes pieces and scores.
            string unigram = Path.Combine(temporary.FullName, "unigram.json");
            File.WriteAllText(unigram, """
                {"added_tokens": [], "normalizer": null, "pre_tokenizer": null, "post_processor": null, "decoder": null,
                 "model": {"type": "Unigram", "unk_id": 0, "vocab": [["<unk>", 0.0], ["▁a", -1.5], ["b", -2.25]], "byte_fallback": false}}
                """);
            var definition = PretrainedTokenizers.Read(unigram);
            Check(definition.ModelType == "Unigram" && definition.Vocabulary is null && definition.Pieces is { Count: 3 } pieces
                  && pieces[1] == ("▁a", -1.5) && pieces[2] == ("b", -2.25), "a list vocabulary reads to pieces and scores in id order");
            Check((int?)definition.Json["model"]?["unk_id"] == 0, "the model's other fields stay in its JSON");

            string empty = Directory.CreateDirectory(Path.Combine(temporary.FullName, "empty")).FullName;
            Check(PretrainedTokenizers.TryLoad(empty) is null && !PretrainedTokenizers.Exists(empty), "a folder without a tokenizer has none");
            Expect<FileNotFoundException>(() => PretrainedTokenizers.Load(empty), "No tokenizer file at", "a folder without a tokenizer");
            Expect<FileNotFoundException>(() => PretrainedTokenizers.Load(empty), "readers: tokenizer.json", "the readers it looked with");

            Check(PretrainedTokenizers.Find(folder, new TokenizerOptions { Format = "TOKENIZER.JSON" }) is not null, "Format names a reader, ignoring case");
            Expect<NotSupportedException>(() => PretrainedTokenizers.Find(folder, new TokenizerOptions { Format = "vocab.bin" }),
                "Tokenizer file reader 'vocab.bin' is not registered", "an unknown Format");
            Environment.SetEnvironmentVariable(TokenizerOptions.FormatVariable, "vocab.bin");
            Expect<NotSupportedException>(() => PretrainedTokenizers.Find(folder), "'vocab.bin' is not registered", "IDRAK_TOKENIZER_FORMAT when no Format is set");
            Check(PretrainedTokenizers.Find(folder, new TokenizerOptions { Format = TokenizerFiles.TokenizerJson }) is not null, "Format wins over the variable");
        }
        finally
        {
            Environment.SetEnvironmentVariable(TokenizerOptions.FormatVariable, previous);
            temporary.Delete(recursive: true);
        }
    }

    private static void TokenizersGuardedModel(Device device)
    {
        string folder = PretrainData("tokenizer");
        var expected = BpeTokenizer.Load(folder).Encode("guarded models answer");
        try
        {
            TokenizerModels.Register(TokenizerModels.Bpe, (d, o) => throw new InvalidOperationException("the app's model cannot be built"));
            Expect<InvalidOperationException>(() => PretrainedTokenizers.Load(folder), "cannot be built", "under Throw, the app's failure");
            TokenizerModels.SetPolicy(TokenizerModels.Bpe, SlotPolicy.FallBack);
            Check(PretrainedTokenizers.Load(folder).Encode("guarded models answer").SequenceEqual(expected), "under FallBack, the built-in builds the tokenizer");

            TokenizerModels.Register(TokenizerModels.Bpe, (d, o) => new FailingEncoder(BpeTokenizer.FromDefinition(d, o)));
            var tokenizer = PretrainedTokenizers.Load(folder);
            Check(tokenizer.Encode("guarded models answer").SequenceEqual(expected), "under FallBack, the built-in answers an encode that throws");
            var ranged = new List<int>();
            tokenizer.EncodeRange("guarded models answer", 0, 21, ranged);
            Check(ranged.SequenceEqual(expected), "and a range that throws");
            var batch = tokenizer.EncodeBatch(["guarded", "guarded models answer"], null,
                new TokenizerEncodeOptions { AddSpecialTokens = false, Padding = "longest", PadId = 0 });
            Check(batch.Count == 2 && batch[1].Ids.SequenceEqual(expected) && batch[0].Count == batch[1].Count && batch[0].AttentionMask.Sum() < batch[0].Count,
                "and a batch that throws, through the built-in's own batch encode (padded to its longest)");
        }
        finally
        {
            TokenizerModels.SetPolicy(TokenizerModels.Bpe, SlotPolicy.Throw);
            TokenizerModels.Unregister(TokenizerModels.Bpe);
        }

        Check(TokenizerModels.Origin(TokenizerModels.Bpe) == Overrides.Library, "the built-in BPE is back");
    }

    // A model whose encodes fail (its tokens, decodes and files are the BPE model's).
    private sealed class FailingEncoder(IPretrainedTokenizer inner) : IPretrainedTokenizer
    {
        public string Model => inner.Model;

        public int VocabularySize => inner.VocabularySize;

        public IReadOnlyList<int> Encode(string text) => throw new InvalidOperationException("the app's encode fails");

        public void EncodeRange(string text, int start, int length, List<int> ids) => throw new InvalidOperationException("the app's range fails");

        public string Decode(IEnumerable<int> ids) => inner.Decode(ids);

        public TokenizerEncoding Encode(string text, string? pair, TokenizerEncodeOptions options) => throw new InvalidOperationException("the app's encode fails");

        public IReadOnlyList<TokenizerEncoding> EncodeBatch(IReadOnlyList<string> texts, IReadOnlyList<string>? pairs, TokenizerEncodeOptions options) =>
            throw new InvalidOperationException("the app's encode fails");

        public string TokenOf(int id) => inner.TokenOf(id);

        public int? IdOf(string token) => inner.IdOf(token);

        public bool IsSpecial(int id) => inner.IsSpecial(id);

        public void PadVocabulary(int size) => inner.PadVocabulary(size);

        public JsonObject ToJson() => inner.ToJson();

        public void WriteTo(Stream stream) => inner.WriteTo(stream);

        public void Save(string path) => inner.Save(path);
    }

    private static void TokenizersPieceCache(Device device)
    {
        string folder = PretrainData("tokenizer");
        string text = string.Join(' ', File.ReadLines(PretrainData("corpus.txt")).Take(40));
        var expected = PretrainedTokenizers.Load(folder).Encode(text);
        if (Environment.GetEnvironmentVariable(TokenizerOptions.PieceCacheVariable) is null)
        {
            Check(new TokenizerOptions().ResolvedPieceCacheLimit() == TokenizerOptions.DefaultPieceCacheLimit, "100,000 pieces by default");
        }

        var uncached = PretrainedTokenizers.Load(folder, new TokenizerOptions { PieceCacheLimit = 0 });
        Check(uncached.Encode(text).SequenceEqual(expected) && uncached.Encode(text).SequenceEqual(expected), "no cache gives the same ids, twice");
        Expect<ArgumentOutOfRangeException>(() => new TokenizerOptions { PieceCacheLimit = -1 }.ResolvedPieceCacheLimit(), "PieceCacheLimit", "a negative cache");

        string? previous = Environment.GetEnvironmentVariable(TokenizerOptions.PieceCacheVariable);
        try
        {
            Environment.SetEnvironmentVariable(TokenizerOptions.PieceCacheVariable, "12");
            Check(new TokenizerOptions().ResolvedPieceCacheLimit() == 12 && new TokenizerOptions { PieceCacheLimit = 5 }.ResolvedPieceCacheLimit() == 5,
                "the variable sets the default, the setting wins over it");
            Check(PretrainedTokenizers.Load(folder).Encode(text).SequenceEqual(expected), "a small cache gives the same ids");
            Environment.SetEnvironmentVariable(TokenizerOptions.PieceCacheVariable, "lots");
            Expect<InvalidOperationException>(() => PretrainedTokenizers.Load(folder), "IDRAK_TOKENIZER_CACHE is 'lots'", "a variable that is not a number");
        }
        finally
        {
            Environment.SetEnvironmentVariable(TokenizerOptions.PieceCacheVariable, previous);
        }

        Check(new PretrainedOptions { Tokenizer = new TokenizerOptions { PieceCacheLimit = 7 } }.Tokenizer!.PieceCacheLimit == 7, "PretrainedOptions carries the tokenizer's settings");
    }

    private static void TokenizersPretrainedSuite(Device device)
    {
        string folder = PretrainData("tokenizer");
        var tokenizer = PretrainedTokenizers.Load(folder);
        Conformance.CheckPretrained(tokenizer, BpeTokenizer.Load(folder)).ThrowIfFailed();
        Conformance.CheckPretrained(PretrainedTokenizers.Load(Path.GetDirectoryName(TestData("gguf/tiny-qwen3-q8-hf/tokenizer.json"))!)).ThrowIfFailed();

        var temporary = Directory.CreateTempSubdirectory("idrak-tokenizers-");
        try
        {
            tokenizer.Save(temporary.FullName);
            var loaded = PretrainedTokenizers.Load(temporary.FullName);
            string text = File.ReadAllText(PretrainData("corpus.txt"));
            Check(loaded.Model == tokenizer.Model && loaded.Encode(text).SequenceEqual(tokenizer.Encode(text)), "a saved tokenizer loads back to the same ids");
        }
        finally
        {
            temporary.Delete(recursive: true);
        }
    }
}
