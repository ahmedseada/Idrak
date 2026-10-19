// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Idrak.Cli.Commands.Data;
using Idrak.Generation;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Nlp;
using Idrak.Nlp.Retrieval;

// --bench-text [text] [tokenizers] [trainers]: the library's text paths, CPU only, no model or download needed (default:
// every section).
//   text        the tokenizer, chat template, streamed output parsing, chunking and keyword search, timed on the README.
//   tokenizers  each fixture tokenizer (data/tokenizers: Unigram from tokenizer.json and tokenizer.model, WordPiece from
//               tokenizer.json and vocab.txt, SentencePiece-style BPE from tokenizer.model and tokenizer.json) on the fixture
//               corpus, repeated as tools/tokenizers/make_fixtures.py --bench repeats it: the load's time (cold and warm),
//               bytes allocated and heap held (after the load, and after the corpus has filled the piece cache); then
//               tokens a second, cold (the first pass of a freshly loaded tokenizer) and warm (the best later pass) apart,
//               and the bytes allocated per text once warm, for the ids into a reused list (EncodeRange; the target is 0),
//               the detailed encode (Encode(text, null, options)) and EncodeBatch, detailed and ids-only (Offsets = false),
//               on one worker and on all (TokenizerOptions.Workers).
//   trainers    TokenizerTrainers "bpe", "unigram" and "wordpiece" on the training text: time, text a second, bytes
//               allocated, the managed heap's peak above the start (sampled every millisecond), full collections, the
//               process's peak working set so far, and the vocabulary learned, on one worker and on all
//               (TokenizerTrainingOptions.Workers); each trainer's first run, on the fixture text alone, is reported apart
//               as its cold run.
// Settings (environment variables; IDRAK_TOKENIZER_CACHE and the library's other tokenizer variables apply as usual):
//   IDRAK_BENCH_TEXT_CHARACTERS   the corpus' non-blank lines are repeated whole until they hold this many code points
//                                 (default 200000, as --bench)
//   IDRAK_BENCH_TEXT_SECONDS      the least time each row's warm passes take (default 1, as --bench's --seconds)
//   IDRAK_BENCH_TEXT_LOADS        the loads timed for a tokenizer's warm load time (default 5)
//   IDRAK_BENCH_TEXT_WORKERS      the workers of the "all workers" rows, encoding and training (default 0: every core)
//   IDRAK_BENCH_TEXT_TOKENIZERS   more tokenizers to time after the fixtures (folders or files, separated by the path
//                                 separator), read as PretrainedTokenizers.Load reads them (IDRAK_TOKENIZER_FORMAT picks the
//                                 reader of a folder with several), for example a real model's folder
//   IDRAK_BENCH_TEXT_REFERENCE    a file holding the JSON make_fixtures.py --bench prints: its rows are printed under each
//                                 fixture's, so the references stand beside Idrak's
//   IDRAK_BENCH_TEXT_TRAIN_FILES  the trainers' text files (separated by the path separator); default the fixture corpus
//                                 and data/pretrain/corpus.txt repeated to IDRAK_BENCH_TEXT_TRAIN_MB in a temporary file
//   IDRAK_BENCH_TEXT_TRAIN_MB     the default training text's size in MB (default 16: a few seconds a run)
//   IDRAK_BENCH_TEXT_TRAIN_VOCAB  the vocabulary size each trainer learns (default 3000)
//   IDRAK_BENCH_TEXT_TRAINERS     the trainers timed, comma-separated (default bpe,unigram,wordpiece; a plug-in's name works)
internal static partial class Tests
{
    private static readonly string[] BenchTextSections = ["text", "tokenizers", "trainers"];

    internal static int BenchText(string[] sections)
    {
        foreach (string section in sections)
        {
            if (!BenchTextSections.Contains(section, StringComparer.OrdinalIgnoreCase))
            {
                Console.Error.WriteLine($"--bench-text: unknown section '{section}' (sections: {string.Join(", ", BenchTextSections)})");
                return 2;
            }
        }

        bool Runs(string section) => sections.Length == 0 || sections.Contains(section, StringComparer.OrdinalIgnoreCase);
        if (Runs("text"))
        {
            BenchTextPaths();
        }

        if (Runs("tokenizers"))
        {
            BenchTokenizers();
        }

        if (Runs("trainers"))
        {
            BenchTrainers();
        }

        return 0;
    }

    // The text paths, on the README.
    private static void BenchTextPaths()
    {
        string root = AppContext.BaseDirectory;
        while (!File.Exists(Path.Combine(root, "README.md")) && Directory.GetParent(root) is { } parent)
        {
            root = parent.FullName;
        }

        string text = File.ReadAllText(Path.Combine(root, "README.md")).ReplaceLineEndings("\n");   // the same input on every OS
        Console.WriteLine($"text paths on the README ({text.Length:N0} characters), .NET {Environment.Version}, {Environment.ProcessorCount} threads");

        // Measures `run` (warmed up first): the best of five rounds, each long enough to time.
        static (double Microseconds, long Bytes) Measure(Action run)
        {
            run();
            int repeats = 1;
            var watch = Stopwatch.StartNew();
            run();
            while (watch.ElapsedMilliseconds < 50 && repeats < 1 << 20)
            {
                repeats *= 2;
                watch.Restart();
                for (int i = 0; i < repeats; i++)
                {
                    run();
                }
            }

            double best = double.MaxValue;
            long bytes = long.MaxValue;
            for (int round = 0; round < 5; round++)
            {
                long allocated = GC.GetAllocatedBytesForCurrentThread();
                watch.Restart();
                for (int i = 0; i < repeats; i++)
                {
                    run();
                }

                best = Math.Min(best, watch.Elapsed.TotalMilliseconds * 1000 / repeats);
                bytes = Math.Min(bytes, (GC.GetAllocatedBytesForCurrentThread() - allocated) / repeats);
            }

            return (best, bytes);
        }

        static void Row(string name, (double Microseconds, long Bytes) result, string? rate = null) =>
            Console.WriteLine($"  {name,-52} {result.Microseconds,12:F1} us {result.Bytes / 1024.0,12:F1} KiB allocated  {rate}");

        var tokenizer = TrainedTokenizer(text, merges: 3000);
        IReadOnlyList<int> ids = tokenizer.Encode(text);
        Console.WriteLine($"tokenizer: byte-level BPE, {tokenizer.VocabularySize:N0} tokens trained on the text; {ids.Count:N0} tokens");
        var encode = Measure(() => tokenizer.Encode(text));
        Row("encode the text", encode, $"{text.Length / encode.Microseconds:F1} MB/s");
        int[] idArray = [.. ids];
        var decode = Measure(() => tokenizer.Decode(idArray));
        Row("decode the text", decode, $"{text.Length / decode.Microseconds:F1} MB/s");
        var single = Measure(() =>
        {
            for (int i = 0; i < 1000; i++)
            {
                tokenizer.Decode(idArray.AsSpan(i, 1));
            }
        });
        Row("decode 1,000 single tokens (streaming)", single);
        var shortText = Measure(() => tokenizer.Encode("What is the capital of France? Answer in one word."));
        Row("encode a 50-character prompt", shortText);

        var template = new JinjaChatTemplate(Qwen3Template, ["<|im_end|>"]);
        var paragraphs = text.Split("\n\n", StringSplitOptions.RemoveEmptyEntries);
        var messages = Enumerable.Range(0, 20).Select(i => new ChatMessage(i % 2 == 0 ? "user" : "assistant", paragraphs[i % paragraphs.Length])).ToList();
        Row("chat template: render 20 messages", Measure(() => template.Render(messages, [], think: false)));

        string reply = "<think>" + paragraphs[3] + "</think>" + paragraphs[5] + "<tool_call>{\"name\": \"search\", \"arguments\": {\"query\": \"idrak\"}}</tool_call>";
        var pieces = tokenizer.Encode(reply).Select(id => tokenizer.Decode([id])).ToArray();
        Row($"parse streamed reply ({pieces.Length} pieces)", Measure(() =>
        {
            var parser = new ChatOutputParser(template);
            foreach (var piece in pieces)
            {
                parser.Feed(piece);
            }

            parser.Finish();
        }));

        var documents = paragraphs.Select((p, i) => new Document($"d{i}", p)).ToList();
        Row("chunk the text (words, 64 with 16 overlap)", Measure(() => Chunker.Split(documents, ChunkUnit.Words, 64, 16)));
        Row("chunk the text (sentences, 4 with 1 overlap)", Measure(() => Chunker.Split(documents, ChunkUnit.Sentences, 4, 1)));
        Row("BM25: index the paragraphs", Measure(() => new Bm25Index(paragraphs)));
        var index = new Bm25Index(paragraphs);
        Row("BM25: 100 searches", Measure(() =>
        {
            for (int i = 0; i < 100; i++)
            {
                index.Search("offload weights to system memory while training on the GPU", 5);
            }
        }));
    }

    // A bench setting: the variable's text, or the fallback when it is unset or empty.
    private static string BenchTextSetting(string name, string fallback) =>
        Environment.GetEnvironmentVariable(name) is { Length: > 0 } value ? value : fallback;

    // A bench setting that is a number of at least `least`; anything else is refused naming the variable.
    private static double BenchTextNumber(string name, double fallback, double least)
    {
        if (Environment.GetEnvironmentVariable(name) is not { Length: > 0 } text)
        {
            return fallback;
        }

        return double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double value) && double.IsFinite(value) && value >= least
            ? value
            : throw new InvalidOperationException($"{name} is '{text}': give a number, {least.ToString(CultureInfo.InvariantCulture)} or more.");
    }

    // A list setting (paths separated by the path separator), empty when unset.
    private static string[] BenchTextFiles(string name) =>
        BenchTextSetting(name, "").Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

    // A full collection, so a heap reading after it counts what is alive.
    private static void BenchTextSettle()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
    }

    private static string BenchTextMegabytes(long bytes) => string.Create(CultureInfo.InvariantCulture, $"{bytes / 1048576.0:F2} MB");

    // The fixture corpus as tools/tokenizers/make_fixtures.py --bench takes it: the lines that are not blank (blank as
    // Python's str.strip sees it, so the \x1c-\x1f separators count as blank too), repeated whole until they hold
    // `codePoints` code points (Python's len), so both sides encode the same texts. Shared with "tokenizers: unigram: speed".
    private static List<string> TokenizerBenchTexts(int codePoints)
    {
        static bool Blank(string line) => line.All(c => char.IsWhiteSpace(c) || c is >= '\u001c' and <= '\u001f');
        static int CodePoints(string text) => text.Length - text.Count(char.IsLowSurrogate);
        var lines = TokenizerCorpusLines().Where(line => !Blank(line)).ToArray();
        long perCopy = lines.Sum(CodePoints);
        var texts = new List<string>();
        for (long total = 0; total < codePoints; total += perCopy)
        {
            texts.AddRange(lines);
        }

        return texts;
    }

    // One encoding row: `pass` encodes every text and returns the tokens it made. Cold is the first pass, warm the best of
    // the passes that follow for at least `seconds`; bytes are the fewest a warm pass allocated, per text (the calling
    // thread's, or every thread's when the pass spreads over threads, rule 63).
    private static (long Tokens, double Cold, double Warm, double Bytes) BenchTextRate(Func<long> pass, int texts, double seconds, bool threads)
    {
        long Allocated() => threads ? GC.GetTotalAllocatedBytes(precise: true) : GC.GetAllocatedBytesForCurrentThread();
        long start = Stopwatch.GetTimestamp();
        long tokens = pass();
        double cold = tokens / Stopwatch.GetElapsedTime(start).TotalSeconds, warm = 0, bytes = double.MaxValue;
        long spent = Stopwatch.GetTimestamp();
        do
        {
            long allocated = Allocated();
            start = Stopwatch.GetTimestamp();
            tokens = pass();
            double elapsed = Stopwatch.GetElapsedTime(start).TotalSeconds;
            bytes = Math.Min(bytes, (double)(Allocated() - allocated) / texts);
            warm = Math.Max(warm, tokens / elapsed);
        }
        while (Stopwatch.GetElapsedTime(spent).TotalSeconds < seconds);
        return (tokens, cold, warm, bytes);
    }

    private static void BenchTextRateRow(string name, long tokens, double cold, double warm, double? bytes) =>
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"    {name,-50} {tokens,9:N0} tokens {cold / 1e6,7:F2}M cold {warm / 1e6,7:F2}M warm{(bytes is { } b ? $" {b,10:N0} B a text" : "")}"));

    // Every fixture tokenizer (and the ones IDRAK_BENCH_TEXT_TOKENIZERS names) on the fixture corpus.
    private static void BenchTokenizers()
    {
        int codePoints = (int)BenchTextNumber("IDRAK_BENCH_TEXT_CHARACTERS", 200_000, 1);
        double seconds = BenchTextNumber("IDRAK_BENCH_TEXT_SECONDS", 1, 0);
        int loads = (int)BenchTextNumber("IDRAK_BENCH_TEXT_LOADS", 5, 1);
        int workers = (int)BenchTextNumber("IDRAK_BENCH_TEXT_WORKERS", 0, 0);
        string allWorkers = workers == 0 ? $"all {Environment.ProcessorCount} workers" : $"{workers} workers";
        var reference = BenchTextSetting("IDRAK_BENCH_TEXT_REFERENCE", "") is { Length: > 0 } file ? JsonNode.Parse(File.ReadAllText(file))!.AsObject() : null;

        var texts = TokenizerBenchTexts(codePoints);
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"tokenizers on the fixture corpus: {texts.Count:N0} texts, {texts.Sum(t => t.Length - t.Count(char.IsLowSurrogate)):N0} code points ({texts.Sum(t => t.Length):N0} UTF-16 characters), "
            + $"{Environment.ProcessorCount} threads; tokens a second (M: million), cold: the first pass of a freshly loaded tokenizer, warm: the best pass in {seconds} s"));
        if (reference is not null)
        {
            Console.WriteLine($"  references: {reference["texts"]} texts, {reference["characters"]} code points, {reference["threads"]} threads; "
                + string.Join(", ", (reference["versions"] as JsonObject ?? new JsonObject()).Where(v => v.Key is "tokenizers" or "sentencepiece").Select(v => $"{v.Key} {v.Value}")));
        }

        var fixtures = new List<(string Fixture, string Source, string Path, TokenizerOptions Options)>
        {
            ("unigram", "tokenizer.json", TokenizerData("unigram"), new() { Format = TokenizerFiles.TokenizerJson }),
            ("unigram", "tokenizer.model", TokenizerData("unigram"), new() { Format = TokenizerFiles.SentencePiece }),
            ("unigram-bytes", "tokenizer.json", TokenizerData("unigram-bytes"), new() { Format = TokenizerFiles.TokenizerJson }),
            ("unigram-bytes", "tokenizer.model", TokenizerData("unigram-bytes"), new() { Format = TokenizerFiles.SentencePiece }),
            ("wordpiece-uncased", "tokenizer.json", TokenizerData("wordpiece-uncased"), new() { Format = TokenizerFiles.TokenizerJson }),
            ("wordpiece-uncased", "vocab.txt", TokenizerData("wordpiece-uncased"), new() { Format = TokenizerFiles.VocabTxt }),
            ("wordpiece-cased", "tokenizer.json", TokenizerData("wordpiece-cased"), new() { Format = TokenizerFiles.TokenizerJson }),
            ("bpe", "tokenizer.model", TokenizerData("bpe"), new() { Format = TokenizerFiles.SentencePiece }),
            ("bpe", "tokenizer.json", TokenizerData("bpe"), new() { Format = TokenizerFiles.TokenizerJson }),
        };
        foreach (string path in BenchTextFiles("IDRAK_BENCH_TEXT_TOKENIZERS"))
        {
            fixtures.Add((Path.GetFileName(Path.TrimEndingDirectorySeparator(path)), "as found", path, new()));
        }

        var detailed = new TokenizerEncodeOptions();
        var idsOnly = new TokenizerEncodeOptions { Offsets = false };
        for (int f = 0; f < fixtures.Count; f++)
        {
            var (fixture, source, path, options) = fixtures[f];
            IPretrainedTokenizer Load(int threads) => PretrainedTokenizers.Load(path, options with { Workers = threads });

            // The load: the first (cold), the best of `loads` more (warm), then one more between full collections for the
            // bytes it allocates and the heap it holds.
            long start = Stopwatch.GetTimestamp();
            var first = Load(1);
            double coldLoad = Stopwatch.GetElapsedTime(start).TotalMilliseconds, warmLoad = double.MaxValue;
            for (int i = 0; i < loads; i++)
            {
                start = Stopwatch.GetTimestamp();
                Load(1);
                warmLoad = Math.Min(warmLoad, Stopwatch.GetElapsedTime(start).TotalMilliseconds);
            }

            BenchTextSettle();
            long heap = GC.GetTotalMemory(forceFullCollection: true), allocated = GC.GetTotalAllocatedBytes(precise: true);
            var tokenizer = Load(1);
            long loadBytes = GC.GetTotalAllocatedBytes(precise: true) - allocated;
            long held = GC.GetTotalMemory(forceFullCollection: true) - heap;

            // The ids into a reused list, on the tokenizer just loaded (its piece cache empty, so the first pass is cold).
            var ids = new List<int>(4096);
            var plain = BenchTextRate(() =>
            {
                long tokens = 0;
                foreach (var text in texts)
                {
                    ids.Clear();
                    tokenizer.EncodeRange(text, 0, text.Length, ids);
                    tokens += ids.Count;
                }

                return tokens;
            }, texts.Count, seconds, threads: false);
            long heldAfter = GC.GetTotalMemory(forceFullCollection: true) - heap;
            GC.KeepAlive(tokenizer);
            GC.KeepAlive(first);

            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"  {fixture} from {source}: {tokenizer.Model}, {tokenizer.VocabularySize:N0} tokens; loads in {coldLoad:F1} ms cold, {warmLoad:F1} ms warm, "
                + $"allocating {BenchTextMegabytes(loadBytes)}; holds {BenchTextMegabytes(held)} loaded, {BenchTextMegabytes(heldAfter)} after the corpus (piece cache filled)"));
            BenchTextRateRow("ids, one thread (EncodeRange into a reused list)", plain.Tokens, plain.Cold, plain.Warm, plain.Bytes);

            var single = Load(1);
            var one = BenchTextRate(() =>
            {
                long tokens = 0;
                foreach (var text in texts)
                {
                    tokens += single.Encode(text, null, detailed).Count;
                }

                return tokens;
            }, texts.Count, seconds, threads: false);
            BenchTextRateRow("detailed, one thread (Encode(text, null, options))", one.Tokens, one.Cold, one.Warm, one.Bytes);

            foreach (var (label, encodeOptions) in new[] { ("detailed", detailed), ("ids only", idsOnly) })
            {
                foreach (int threads in new[] { 1, workers })
                {
                    var batched = Load(threads);
                    var batch = BenchTextRate(() => batched.EncodeBatch(texts, null, encodeOptions).Sum(e => (long)e.Count), texts.Count, seconds, threads: true);
                    BenchTextRateRow($"EncodeBatch {label}, {(threads == 1 ? "1 worker" : allWorkers)}", batch.Tokens, batch.Cold, batch.Warm, batch.Bytes);
                }
            }

            // The references' rows for this fixture, under its last tokenizer.
            if ((f + 1 == fixtures.Count || fixtures[f + 1].Fixture != fixture) && reference?["tokenizers"]?[fixture] is JsonObject rows)
            {
                Console.WriteLine($"  {fixture}: the references (make_fixtures.py --bench)");
                foreach (var (name, row) in rows.OrderBy(r => r.Key, StringComparer.Ordinal))
                {
                    BenchTextRateRow(name, (long)row!["tokens"]!, (double)row["cold"]!, (double)row["warm"]!, null);
                }
            }
        }
    }

    // TokenizerTrainers "bpe", "unigram" and "wordpiece" (or IDRAK_BENCH_TEXT_TRAINERS) on the training text.
    private static void BenchTrainers()
    {
        double megabytes = BenchTextNumber("IDRAK_BENCH_TEXT_TRAIN_MB", 16, 0);
        int vocabulary = (int)BenchTextNumber("IDRAK_BENCH_TEXT_TRAIN_VOCAB", 3000, 1);
        int workers = (int)BenchTextNumber("IDRAK_BENCH_TEXT_WORKERS", 0, 0);
        string[] names = BenchTextSetting("IDRAK_BENCH_TEXT_TRAINERS", $"{TokenizerTrainers.Bpe},{TokenizerTrainers.Unigram},{TokenizerTrainers.WordPiece}")
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        // The text: the files named, or the fixture texts repeated whole into a temporary file to the size asked.
        string[] files = BenchTextFiles("IDRAK_BENCH_TEXT_TRAIN_FILES");
        string? temporary = null;
        string described;
        if (files.Length == 0)
        {
            temporary = Path.Combine(Path.GetTempPath(), $"idrak-bench-text-{Environment.ProcessId}");
            Directory.CreateDirectory(temporary);
            string corpus = Path.Combine(temporary, "corpus.txt");
            byte[][] parts = [.. TrainingCorpus.Select(File.ReadAllBytes)];
            using (var stream = File.Create(corpus))
            {
                for (long written = 0; written < megabytes * 1048576;)
                {
                    foreach (var part in parts)
                    {
                        stream.Write(part);
                        written += part.Length;
                        if (part.Length > 0 && part[^1] != (byte)'\n')
                        {
                            stream.WriteByte((byte)'\n');
                            written++;
                        }
                    }
                }
            }

            files = [corpus];
            described = "the fixture corpus and data/pretrain/corpus.txt, repeated";
        }
        else
        {
            described = string.Join(", ", files.Select(Path.GetFileName));
        }

        try
        {
            long textBytes = files.Sum(file => new FileInfo(file).Length);
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"trainers on {BenchTextMegabytes(textBytes)} of text ({described}), vocabulary {vocabulary:N0}, {Environment.ProcessorCount} threads; "
                + $"heap peak: the managed heap's largest size above the start, sampled every millisecond; process peak: the working set's peak so far in this process"));
            foreach (string name in names)
            {
                var trainer = TokenizerTrainers.Get(name);
                var options = new TokenizerTrainingOptions
                {
                    VocabularySize = vocabulary,
                    SpecialTokens = name.Equals(TokenizerTrainers.WordPiece, StringComparison.OrdinalIgnoreCase) ? DataTokenizerCommand.WordPieceSpecials : DataTokenizerCommand.DefaultSpecials,
                };

                // The cold run: the first in this process, on the fixture text alone (its methods compiled as it goes).
                long start = Stopwatch.GetTimestamp();
                var cold = trainer.Train(TrainingCorpus, options with { Workers = workers });
                Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                    $"  {name,-10} cold, the fixture text ({BenchTextMegabytes(TrainingCorpus.Sum(file => new FileInfo(file).Length))}), {(workers == 0 ? "all" : workers.ToString(CultureInfo.InvariantCulture))} workers: "
                    + $"{Stopwatch.GetElapsedTime(start).TotalSeconds:F2} s, vocabulary {cold.VocabularySize:N0}"));
                foreach (int threads in new[] { 1, workers })
                {
                    BenchTextSettle();
                    long heap = GC.GetTotalMemory(forceFullCollection: true), allocated = GC.GetTotalAllocatedBytes(precise: true);
                    int fullCollections = GC.CollectionCount(2);
                    long peak = heap;
                    using var done = new ManualResetEventSlim();
                    var sampler = new Thread(() =>
                    {
                        do
                        {
                            peak = Math.Max(peak, GC.GetTotalMemory(forceFullCollection: false));
                        }
                        while (!done.Wait(1));
                    }) { IsBackground = true, Name = "bench heap sampler" };
                    sampler.Start();
                    start = Stopwatch.GetTimestamp();
                    var trained = trainer.Train(files, options with { Workers = threads });
                    double elapsed = Stopwatch.GetElapsedTime(start).TotalSeconds;
                    done.Set();
                    sampler.Join();
                    long bytes = GC.GetTotalAllocatedBytes(precise: true) - allocated;
                    using var process = Process.GetCurrentProcess();
                    Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                        $"  {name,-10} {(threads == 1 ? "1 worker" : workers == 0 ? $"all {Environment.ProcessorCount} workers" : $"{threads} workers"),-15} {elapsed,7:F2} s {textBytes / 1048576.0 / elapsed,7:F2} MB/s"
                        + $"  allocated {BenchTextMegabytes(bytes),10}  heap peak {BenchTextMegabytes(peak - heap),10}  gen2 {GC.CollectionCount(2) - fullCollections,3}"
                        + $"  process peak {BenchTextMegabytes(process.PeakWorkingSet64),10}  vocabulary {trained.VocabularySize:N0}"));
                }
            }
        }
        finally
        {
            if (temporary is not null)
            {
                Directory.Delete(temporary, recursive: true);
            }
        }
    }

    // A GPT-2-style byte-level BPE trained on `text` (word counts, most frequent pair merged first), so the timings use a
    // vocabulary shaped like a real one.
    private static BpeTokenizer TrainedTokenizer(string text, int merges)
    {
        const string Pattern = @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
        var printable = Enumerable.Range('!', '~' - '!' + 1).Concat(Enumerable.Range('¡', '¬' - '¡' + 1)).Concat(Enumerable.Range('®', 'ÿ' - '®' + 1)).ToHashSet();
        var byteMap = new string[256];
        int extra = 0;
        for (int b = 0; b < 256; b++)
        {
            byteMap[b] = ((char)(printable.Contains(b) ? b : 256 + extra++)).ToString();
        }

        var words = new Dictionary<string, int>();
        foreach (Match match in Regex.Matches(text, Pattern))
        {
            string word = string.Concat(Encoding.UTF8.GetBytes(match.Value).Select(b => byteMap[b]));
            words[word] = words.GetValueOrDefault(word) + 1;
        }

        var split = words.ToDictionary(w => w.Key, w => w.Key.Select(c => c.ToString()).ToList());
        var vocab = new JsonObject();
        foreach (var token in byteMap)
        {
            vocab[token] = vocab.Count;
        }

        var mergeList = new JsonArray();
        for (int m = 0; m < merges; m++)
        {
            var pairs = new Dictionary<(string, string), int>();
            foreach (var (word, parts) in split)
            {
                for (int i = 0; i + 1 < parts.Count; i++)
                {
                    pairs[(parts[i], parts[i + 1])] = pairs.GetValueOrDefault((parts[i], parts[i + 1])) + words[word];
                }
            }

            if (pairs.Count == 0)
            {
                break;
            }

            var best = pairs.MaxBy(p => p.Value).Key;
            string merged = best.Item1 + best.Item2;
            if (!vocab.ContainsKey(merged))
            {
                vocab[merged] = vocab.Count;
            }

            mergeList.Add($"{best.Item1} {best.Item2}");
            foreach (var parts in split.Values)
            {
                for (int i = 0; i + 1 < parts.Count; i++)
                {
                    if (parts[i] == best.Item1 && parts[i + 1] == best.Item2)
                    {
                        parts[i] = merged;
                        parts.RemoveAt(i + 1);
                    }
                }
            }
        }

        return BpeTokenizer.FromJson(new JsonObject
        {
            ["added_tokens"] = new JsonArray(new JsonObject { ["id"] = vocab.Count, ["content"] = "<|im_end|>", ["special"] = true }),
            ["pre_tokenizer"] = new JsonObject
            {
                ["type"] = "Sequence",
                ["pretokenizers"] = new JsonArray(
                    new JsonObject { ["type"] = "Split", ["pattern"] = new JsonObject { ["Regex"] = Pattern }, ["behavior"] = "Isolated", ["invert"] = false },
                    new JsonObject { ["type"] = "ByteLevel", ["add_prefix_space"] = false, ["use_regex"] = false }),
            },
            ["model"] = new JsonObject { ["type"] = "BPE", ["vocab"] = vocab, ["merges"] = mergeList },
            ["decoder"] = new JsonObject { ["type"] = "ByteLevel" },
        });
    }
}
