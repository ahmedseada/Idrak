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
        ("tokenizers: the decoded text's spaces: a folder whose tokenizer_config.json sets clean_up_tokenization_spaces takes out the space before punctuation and English contractions as transformers does (each replacement in its order); CleanUpSpaces and IDRAK_TOKENIZER_CLEANUP override it, a bad value refused naming the variable; without the setting nothing is taken out", TokenizersCleanUpSpaces),
        ("tokenizers: a tokenizer.json that names no model type (written before tokenizers did: t5-small's, many BERT folders') is read as Hugging Face tokenizers reads it, from what its model holds (merges BPE, [piece, score] pairs Unigram, WordPiece's settings WordPiece, any other vocabulary WordLevel), to the ids of the same file naming it", TokenizersUntypedModel),
        ("tokenizers: an app's model overriding BPE is guarded: under FallBack one that fails to build or to encode (a text, a range, a batch, padded as the built-in pads it) is answered by the built-in, under Throw its failure reaches the caller", TokenizersGuardedModel),
        ("tokenizers: the piece cache is a setting (PieceCacheLimit, IDRAK_TOKENIZER_CACHE; 0 caches nothing and gives the same ids), a bad value is refused naming it, and PretrainedOptions carries the settings", TokenizersPieceCache),
        ("tokenizers: the built-in BPE passes the pretrained tokenizer suite (ranges, tokens by name, tokenizer.json naming the model) and a saved tokenizer loads back to the same ids", TokenizersPretrainedSuite),
        ("tokenizers: bpe: EncodeRange into a reused list allocates 0 bytes once warm on every corpus line (SentencePiece-style from tokenizer.json and tokenizer.model, byte level, with the piece cache and without it)", TokenizersBpeAllocatesNothing),
        ("tokenizers: a batch on 4 workers allocates per text what 1 worker allocates, detailed and ids only (each thread matches a tokenizer's patterns through its own copy): BPE with a Split pre-tokenizer and added tokens, WordPiece uncased and cased, Unigram", TokenizersBatchAllocatesAsOneWorker),
        ("tokenizers: bpe: a byte-level tokenizer's two piece caches (raw text for ids, alphabet pieces with their ranges for the detailed encode) share one PieceCacheLimit, and give the uncached ids and offsets", TokenizersByteLevelCacheBudget),
    ];

    // A Regex keeps one matcher, and a thread that finds it taken by another makes a new one per match: a tokenizer shared by
    // a batch's workers allocated 70-2,100 bytes a text more on 4 workers than on 1 (measured) until each thread matched
    // through its own copy (TokenizerRegex). The tolerance, 32 bytes a text, covers what a batch on 4 workers allocates
    // itself (Parallel.For's loop state and tasks, a few KB over the batch's texts) and a pool thread that joins a later
    // batch making its scratch once; the cost it guards against is ten to sixty times that.
    private static void TokenizersBatchAllocatesAsOneWorker(Device device)
    {
        _ = device;
        var lines = TokenizerCorpusLines().Where(line => line.Length > 0).ToArray();
        string[] texts = [.. Enumerable.Repeat(lines, 8).SelectMany(l => l)];
        const double Tolerance = 32;

        // byte-level-roberta's vocabulary cut by a Split stage (GPT-2's rule) before ByteLevel, as most byte-level
        // tokenizer.json files cut text today.
        var json = JsonNode.Parse(File.ReadAllText(Path.Combine(TokenizerData("byte-level-roberta"), "tokenizer.json")))!.AsObject();
        json["pre_tokenizer"] = new JsonObject
        {
            ["type"] = "Sequence",
            ["pretokenizers"] = new JsonArray(
                new JsonObject { ["type"] = "Split", ["pattern"] = new JsonObject { ["Regex"] = BpeTokenizer.Gpt2Pattern }, ["behavior"] = "Isolated", ["invert"] = false },
                new JsonObject { ["type"] = "ByteLevel", ["add_prefix_space"] = false, ["trim_offsets"] = true, ["use_regex"] = false }),
        };
        string folder = TempFolder();
        try
        {
            File.WriteAllText(Path.Combine(folder, "tokenizer.json"), json.ToJsonString());
            var sources = new (string Name, string Path, TokenizerOptions Options)[]
            {
                ("bpe with a Split pre-tokenizer", folder, new() { Format = TokenizerFiles.TokenizerJson }),
                ("wordpiece-uncased", TokenizerData("wordpiece-uncased"), new() { Format = TokenizerFiles.TokenizerJson }),
                ("wordpiece-cased", TokenizerData("wordpiece-cased"), new() { Format = TokenizerFiles.TokenizerJson }),
                ("unigram", TokenizerData("unigram"), new() { Format = TokenizerFiles.TokenizerJson }),
            };
            foreach (var (name, path, options) in sources)
            {
                foreach (var (label, encode) in new[] { ("detailed", new TokenizerEncodeOptions()), ("ids only", new TokenizerEncodeOptions { Offsets = false }) })
                {
                    double one = BatchBytesPerText(PretrainedTokenizers.Load(path, options with { Workers = 1 }), texts, encode);
                    double four = BatchBytesPerText(PretrainedTokenizers.Load(path, options with { Workers = 4 }), texts, encode);
                    Console.WriteLine($"    {name}, {label}: {one:F1} bytes a text on 1 worker, {four:F1} on 4");
                    Check(four - one <= Tolerance, $"{name}, {label}: a batch on 4 workers allocated {four:F1} bytes a text, on 1 worker {one:F1}: "
                        + $"{four - one:F1} more (at most {Tolerance} is the batch's own); threads share a matcher (a Regex not matched through TokenizerRegex.Local?)");
                }
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    // The fewest bytes a warm batch allocated (on every thread, rule 63), per text.
    private static double BatchBytesPerText(IPretrainedTokenizer tokenizer, string[] texts, TokenizerEncodeOptions options)
    {
        for (int warm = 0; warm < 3; warm++)
        {
            tokenizer.EncodeBatch(texts, null, options);
        }

        double fewest = double.MaxValue;
        for (int run = 0; run < 5; run++)
        {
            long before = GC.GetTotalAllocatedBytes(precise: true);
            tokenizer.EncodeBatch(texts, null, options);
            fewest = Math.Min(fewest, (GC.GetTotalAllocatedBytes(precise: true) - before) / (double)texts.Length);
        }

        return fewest;
    }

    // A byte-level tokenizer keys its ids cache by raw text and the detailed encode's by alphabet pieces (the keys would
    // clash in one): both count against one budget, so together they hold at most PieceCacheLimit pieces.
    private static void TokenizersByteLevelCacheBudget(Device device)
    {
        _ = device;
        string path = TokenizerData("byte-level-roberta");
        var lines = TokenizerCorpusLines();
        var detailed = new TokenizerEncodeOptions();
        var reference = PretrainedTokenizers.Load(path, new TokenizerOptions { PieceCacheLimit = 0 });
        var expected = lines.Select(line => (Ids: reference.Encode(line), Detailed: reference.Encode(line, null, detailed))).ToArray();

        void Pass(BpeTokenizer tokenizer, string what)
        {
            var ids = new List<int>();
            for (int i = 0; i < lines.Length; i++)
            {
                ids.Clear();
                tokenizer.EncodeRange(lines[i], 0, lines[i].Length, ids);
                var encoding = tokenizer.Encode(lines[i], null, detailed);
                Check(ids.SequenceEqual(expected[i].Ids) && encoding.Ids.SequenceEqual(expected[i].Detailed.Ids)
                    && encoding.Offsets.SequenceEqual(expected[i].Detailed.Offsets) && encoding.Tokens.SequenceEqual(expected[i].Detailed.Tokens),
                    $"{what}: line {i} '{Escaped(lines[i][..Math.Min(40, lines[i].Length)])}' encodes otherwise than without a cache");
            }
        }

        // The ids pass alone, then both: the detailed encode adds its own pieces to the same count.
        var whole = (BpeTokenizer)PretrainedTokenizers.Load(path);
        var idsOnly = new List<int>();
        foreach (string line in lines)
        {
            idsOnly.Clear();
            whole.EncodeRange(line, 0, line.Length, idsOnly);
        }

        int fast = whole.CachedPieces;
        Pass(whole, "the default cache");
        Check(fast > 0 && whole.CachedPieces > fast, $"the ids cache held {fast} pieces, both {whole.CachedPieces}: the detailed encode's pieces count too");

        // A limit the ids pass fills: the detailed encode adds nothing more, and both still give the uncached results.
        var limited = (BpeTokenizer)PretrainedTokenizers.Load(path, new TokenizerOptions { PieceCacheLimit = fast });
        Pass(limited, $"a cache of {fast} pieces");
        Check(limited.CachedPieces == fast, $"two caches under a limit of {fast} pieces hold {limited.CachedPieces}");
        var small = (BpeTokenizer)PretrainedTokenizers.Load(path, new TokenizerOptions { PieceCacheLimit = 10 });
        Pass(small, "a cache of 10 pieces");
        Check(small.CachedPieces == 10, $"two caches under a limit of 10 pieces hold {small.CachedPieces}");
    }

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
                    var culprits = new List<(int Index, string Text)>();
                    foreach (var (line, i) in taken.Select((l, i) => (l, i)))
                    {
                        long start = GC.GetAllocatedBytesForCurrentThread();
                        ids.Clear();
                        tokenizer.EncodeRange(line, 0, line.Length, ids);
                        if (GC.GetAllocatedBytesForCurrentThread() - start is var bytes and > 0)
                        {
                            culprits.Add((i, $"line {i} ({bytes} bytes, {line.Length} characters) '{Escaped(line[..Math.Min(40, line.Length)])}'"));
                        }
                    }

                    string first = culprits.Count > 0 ? taken[culprits[0].Index] : taken[0];
                    string types = AllocatedTypes(() =>
                    {
                        ids.Clear();
                        tokenizer.EncodeRange(first, 0, first.Length, ids);
                    });
                    Check(false, $"{name}, cache {cache}: a warm pass allocated {allocated} bytes over {taken.Length * 3} lines:\n  {string.Join("\n  ", culprits.Take(8).Select(c => c.Text))}\n  line {(culprits.Count > 0 ? culprits[0].Index : 0)} again: {types}\n  {BpeStageBytes(tokenizer, first)}");
                }
            }
        }
    }

    // Each step of a BPE encode on its own, in bytes a call (the steps EncodeRange takes for a piece the cache holds), and
    // the methods the JIT compiles in a second without calls (others compiling hold back tiering), so a failure names the step.
    private static string BpeStageBytes(IPretrainedTokenizer tokenizer, string line)
    {
        if (tokenizer is not BpeTokenizer bpe)
        {
            return $"the tokenizer is a {tokenizer.GetType().Name}, not BpeTokenizer: no steps measured";
        }

        static double PerCall(Action call)
        {
            for (int i = 0; i < 2000; i++)
            {
                call();
            }

            long start = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 2000; i++)
            {
                call();
            }

            return (GC.GetAllocatedBytesForCurrentThread() - start) / 2000.0;
        }

        T Field<T>(string field) => (T)typeof(BpeTokenizer).GetField(field, System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.GetValue(bpe)!;
        var steps = new List<string>();
        if (Field<TokenizerRegex?>("_addedPattern") is { } added)
        {
            steps.Add($"added-token matches {PerCall(() => { foreach (var match in added.Local.EnumerateMatches(line)) { _ = match; } }):F1}");
        }

        char[] buffer = new char[line.Length + 16];
        foreach (var (prepend, from, to) in Field<List<(string? Prepend, char From, char To)>?>("_normalizerSteps") ?? [])
        {
            if (prepend is null)
            {
                steps.Add($"Replace('{from}', '{to}') {PerCall(() => { line.CopyTo(buffer); buffer.AsSpan(0, line.Length).Replace(from, to); }):F1}");
            }
        }

        var cache = Field<PieceCache>("_cache");
        string piece = ((Field<List<(string? Prepend, char From, char To)>?>("_normalizerSteps") ?? []).FirstOrDefault(s => s.Prepend is not null).Prepend ?? "")
            + line.Replace(' ', '\u2581');
        bool held = cache.TryGet(piece, out _);
        steps.Add($"piece cache lookup ({(held ? "held" : "not held")}) {PerCall(() => cache.TryGet(piece, out _)):F1}");
        var ids = new List<int>(4096);
        int[] some = [1, 2, 3, 4, 5, 6, 7, 8];
        steps.Add($"ids.AddRange(span) {PerCall(() => { ids.Clear(); ids.AddRange(some.AsSpan()); }):F1}");
        steps.Add($"the whole EncodeRange {PerCall(() => { ids.Clear(); bpe.EncodeRange(line, 0, line.Length, ids); }):F1}");
        long compiled = System.Runtime.JitInfo.GetCompiledMethodCount();
        Thread.Sleep(1000);
        long quiet = System.Runtime.JitInfo.GetCompiledMethodCount() - compiled;
        return $"bytes a call by step: {string.Join(", ", steps)}; the JIT compiled {quiet} methods in a second without calls"
            + $" (tiered compilation {(AppContext.TryGetSwitch("System.Runtime.TieredCompilation", out bool tiered) ? tiered.ToString() : "default")})";
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

    private static void TokenizersCleanUpSpaces(Device device)
    {
        _ = device;
        Check(DecodedSpaces.CleanUp("it 's here , is n't it ? yes ! I 'm sure we 've seen they 're ' fine' .")
              == "it's here, isn't it? yes! I'm sure we've seen they're'fine'.", "transformers' replacements (\" ' \" too, which joins the quote to both words)");
        Check(DecodedSpaces.CleanUp("a  .") == "a .", "each replacement once over the text, as str.replace: two spaces keep one");

        string folder = TokenizerData("wordpiece-uncased");                     // its tokenizer_config.json sets the clean-up
        Check(PretrainedTokenizers.Read(folder).CleanUpTokenizationSpaces == true, "the folder's tokenizer_config.json asks for it");
        var asked = PretrainedTokenizers.Load(folder);
        var raw = PretrainedTokenizers.Load(folder, new TokenizerOptions { CleanUpSpaces = false });
        int[] ids = [.. raw.Encode("it's here, isn't it?")];
        string rawText = raw.Decode(ids), cleaned = asked.Decode(ids);
        Check(cleaned == DecodedSpaces.CleanUp(rawText) && cleaned != rawText,
            $"the folder's tokenizer takes the spaces out ('{cleaned}'), CleanUpSpaces false keeps them ('{rawText}')");
        Check(PretrainedTokenizers.Load(Path.Combine(folder, "tokenizer.json")).Decode(ids) == cleaned, "the tokenizer.json given as a file reads its folder's tokenizer_config.json too");
        Check(PretrainedTokenizers.Read(TokenizerData("unigram")).CleanUpTokenizationSpaces is null, "a folder whose tokenizer_config.json says nothing of it leaves it off");

        string? previous = Environment.GetEnvironmentVariable(TokenizerOptions.CleanUpVariable);
        try
        {
            Environment.SetEnvironmentVariable(TokenizerOptions.CleanUpVariable, "0");
            Check(PretrainedTokenizers.Load(folder).Decode(ids) == rawText, "IDRAK_TOKENIZER_CLEANUP=0 keeps the spaces over the folder's setting");
            Check(PretrainedTokenizers.Load(folder, new TokenizerOptions { CleanUpSpaces = true }).Decode(ids) == cleaned, "the option wins over the variable");
            Environment.SetEnvironmentVariable(TokenizerOptions.CleanUpVariable, "some");
            Expect<InvalidOperationException>(() => PretrainedTokenizers.Load(folder), "IDRAK_TOKENIZER_CLEANUP is 'some'", "a variable that is not 1 or 0");
        }
        finally
        {
            Environment.SetEnvironmentVariable(TokenizerOptions.CleanUpVariable, previous);
        }
    }

    private static void TokenizersUntypedModel(Device device)
    {
        _ = device;
        var temporary = Directory.CreateTempSubdirectory("idrak-tokenizers-untyped-");
        try
        {
            string[] texts = [.. File.ReadAllText(TestData("tokenizers/corpus.txt")).Split('\n').Where(t => t.Length > 0)];
            foreach (var (name, type) in new[] { ("byte-level-roberta", TokenizerModels.Bpe), ("wordpiece-uncased", TokenizerModels.WordPiece), ("unigram", TokenizerModels.Unigram) })
            {
                string typed = Path.Combine(TokenizerData(name), "tokenizer.json");
                var json = JsonNode.Parse(File.ReadAllText(typed))!.AsObject();
                Check((string?)json["model"]?["type"] == type, $"{name}: the fixture names {type}");
                json["model"]!.AsObject().Remove("type");
                string untyped = Path.Combine(temporary.FullName, $"{name}.json");
                File.WriteAllText(untyped, json.ToJsonString());
                Check(PretrainedTokenizers.Read(untyped).ModelType == type, $"{name}: without model.type it is read as {type}");
                var expected = PretrainedTokenizers.Load(typed);
                var actual = PretrainedTokenizers.Load(untyped);
                foreach (string text in texts)
                {
                    Check(actual.Encode(text).SequenceEqual(expected.Encode(text)), $"{name}: '{text}' encodes as with model.type");
                }
            }

            string wordLevel = Path.Combine(temporary.FullName, "word-level.json");
            File.WriteAllText(wordLevel, """
                {"added_tokens": [], "normalizer": null, "pre_tokenizer": null, "post_processor": null, "decoder": null,
                 "model": {"vocab": {"[UNK]": 0, "a": 1}, "unk_token": "[UNK]"}}
                """);
            Check(PretrainedTokenizers.Read(wordLevel).ModelType == "WordLevel", "a plain vocabulary without WordPiece's settings is read as WordLevel");
            Expect<NotSupportedException>(() => PretrainedTokenizers.Load(wordLevel), "'WordLevel' is not registered", "a WordLevel model, which the library does not build");
        }
        finally
        {
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
