// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Abstraction;
using Idrak.Abstraction.Generation;
using Idrak.Abstraction.Testing;
using Idrak.Models;
using Idrak.Models.Abstractions;

// Plan 24, phase T1: the Unigram model, against the fixtures' references (data/tokenizers/README.md): tokenizers for each
// tokenizer.json, sentencepiece for each tokenizer.model.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] TokenizerUnigramGroup =
    [
        ("tokenizers: unigram: the library registers the Unigram model; each fixture's tokenizer.json encodes every corpus line to tokenizers' ids (without the post-processor's special tokens) and decodes to its text, and tokenizers' ids with the special tokens decode to its text with them (the known differences named)", UnigramJsonMatchesTokenizers),
        ("tokenizers: unigram: each fixture's tokenizer.model encodes every corpus line to sentencepiece's ids and decodes to its text, scores summed in float as sentencepiece sums them, so the 300-a tie agrees too (the known differences named)", UnigramModelMatchesSentencePiece),
        ("tokenizers: unigram: byte fallback round-trips arbitrary characters (outside the vocabulary, through their <0xNN> pieces) from tokenizer.json and tokenizer.model; without it a run of unknown characters is one unknown id", UnigramByteFallbackRoundTrips),
        ("tokenizers: unigram: sampling is off by default; with SamplingAlpha it is repeatable with a seed (whole lattice and n-best), varies, decodes to the best segmentation's text and bypasses the cache; SamplingBest 1 gives the best ids; IDRAK_TOKENIZER_ALPHA, _NBEST and _SEED set it and a bad value is refused naming it", UnigramSampling),
        ("tokenizers: unigram: both fixtures from tokenizer.json and tokenizer.model pass the pretrained tokenizer suite, and a saved tokenizer loads back to the same ids, pieces and scores bit for bit (the sentencepiece mode kept)", UnigramSuiteAndSave),
        ("tokenizers: unigram: EncodeRange into a reused list allocates 0 bytes once warm on every corpus line (both fixtures, both paths, with the piece cache and without it)", UnigramAllocatesNothing),
        ("tokenizers: unigram: speed: tokens a second on the corpus as the references' --bench measures it (one thread and all threads, cold and warm)", UnigramSpeed),
    ];

    private static readonly string[] UnigramFixtures = ["unigram", "unigram-bytes"];

    private static IPretrainedTokenizer UnigramFromJson(string name, TokenizerOptions? options = null) =>
        PretrainedTokenizers.Load(TokenizerData(name), options ?? new TokenizerOptions());

    private static IPretrainedTokenizer UnigramFromModel(string name, TokenizerOptions? options = null) =>
        PretrainedTokenizers.Load(TokenizerData(name), (options ?? new TokenizerOptions()) with { Format = TokenizerFiles.SentencePiece });

    // The fixture's known differences: corpus line → its row in known-differences.json.
    private static Dictionary<int, JsonObject> KnownDifferences(string name)
    {
        var lines = TokenizerCorpusLines();
        var known = TokenizerJson("known-differences.json")["differences"]!.AsArray().Select(d => d!.AsObject())
            .Where(d => (string)d["tokenizer"]! == name).ToDictionary(d => (int)d["line"]!);
        foreach (var (line, row) in known)
        {
            Check(lines[line] == (string)row["text"]!, $"known-differences.json's {name} line {line} is not corpus line {line}");
        }

        return known;
    }

    private static int[] Ints(JsonNode? array) => [.. array!.AsArray().Select(x => (int)x!)];

    // Compares one path of every fixture with its reference; lines that differ and are known are named, the rest fail.
    private static void UnigramAgainst(string path, Func<string, IPretrainedTokenizer> load,
        Func<string, JsonObject, (int[] Ids, string Decoded, (int[] Ids, string Decoded)? Extra)> reference, string referenceFile)
    {
        var lines = TokenizerCorpusLines();
        var failures = new List<string>();
        var notes = new List<string>();
        foreach (var name in UnigramFixtures)
        {
            var tokenizer = load(name);
            Check(tokenizer.Model == TokenizerModels.Unigram, $"{name} {path}: model {tokenizer.Model}");
            var rows = TokenizerJson($"{name}/{referenceFile}")["lines"]!.AsArray();
            Check(rows.Count == lines.Length, $"{name}: {rows.Count} reference lines for {lines.Length} corpus lines");
            var known = KnownDifferences(name);
            int matched = 0, knownAgreeing = 0, idsMatched = 0;
            for (int i = 0; i < lines.Length; i++)
            {
                var row = rows[i]!.AsObject();
                Check((string)row["text"]! == lines[i], $"{name}: {referenceFile} line {i} is not corpus line {i}");
                var (want, wanted, extra) = reference(name, row);
                var ids = tokenizer.Encode(lines[i]);
                string decoded = tokenizer.Decode(ids);
                idsMatched += ids.SequenceEqual(want) ? 1 : 0;
                bool same = ids.SequenceEqual(want) && decoded == wanted;
                string? extraDifference = null;
                if (extra is { } more && tokenizer.Decode(more.Ids) is var full && full != more.Decoded)
                {
                    same = false;
                    extraDifference = $"; the reference's ids with special tokens decode to '{Escaped(full)}', reference '{Escaped(more.Decoded)}'";
                }

                if (same)
                {
                    matched++;
                    knownAgreeing += known.ContainsKey(i) ? 1 : 0;
                    continue;
                }

                string line = $"{name} line {i} '{Escaped(lines[i])}' from {path}: ids [{string.Join(",", ids)}] decoded '{Escaped(decoded)}'; "
                    + $"reference [{string.Join(",", want)}] '{Escaped(wanted)}'{extraDifference}";
                (known.ContainsKey(i) ? notes : failures).Add(line);
            }

            Console.WriteLine($"    {name} from {path}: {matched} of {lines.Length} lines equal the reference in ids and text, {idsMatched} in ids ({known.Count} known differences, {knownAgreeing} of them agree here)");
        }

        foreach (var note in notes)
        {
            Console.WriteLine($"    known difference: {note}");
        }

        Check(failures.Count == 0, $"{failures.Count} lines differ from their reference:\n  {string.Join("\n  ", failures.Take(8))}");
    }

    private static void UnigramJsonMatchesTokenizers(Device device)
    {
        _ = device;
        Check(TokenizerModels.Names.Contains(TokenizerModels.Unigram) && TokenizerModels.Origin(TokenizerModels.Unigram) == Overrides.Library,
            $"the library registers Unigram (registered: {string.Join(", ", TokenizerModels.Names)})");
        UnigramAgainst("tokenizer.json", name => UnigramFromJson(name), (_, row) =>
        {
            // Encode adds no special tokens: tokenizers' ids without the post-processor's (sequence id -1).
            int[] all = Ints(row["ids"]);
            int[] sequence = Ints(row["sequence_ids"]);
            int[] ids = [.. all.Where((_, k) => sequence[k] != -1)];
            return (ids, (string)row["decoded"]!, (all, (string)row["decoded_with_special"]!));
        }, "expected-hf.json");
    }

    private static void UnigramModelMatchesSentencePiece(Device device)
    {
        _ = device;
        UnigramAgainst("tokenizer.model", name => UnigramFromModel(name), (_, row) => (Ints(row["ids"]), (string)row["decoded"]!, null), "expected-spm.json");
    }

    private static void UnigramByteFallbackRoundTrips(Device device)
    {
        _ = device;
        var random = new Random(2026);
        var texts = new List<string>();
        for (int t = 0; t < 200; t++)
        {
            var text = new StringBuilder();
            int length = random.Next(1, 40);
            for (int i = 0; i < length; i++)
            {
                text.Append(char.ConvertFromUtf32(random.Next(8) switch
                {
                    0 => NotA(random.Next(0x21, 0x7F)),                                // the user rule maps AA
                    1 => random.Next(0xA1, 0x250),
                    2 => random.Next(0x370, 0x400),
                    3 => random.Next(0x4E00, 0xA000),
                    4 => random.Next(0x1F300, 0x1FB00),
                    5 => random.Next(0xE000, 0xF900),
                    6 => random.Next(0x20000, 0x2A6E0),
                    _ => random.Next(0x900, 0xE00),
                }));
            }

            texts.Add(text.ToString());
        }

        foreach (var (path, tokenizer) in new[] { ("tokenizer.json", UnigramFromJson("unigram-bytes")), ("tokenizer.model", UnigramFromModel("unigram-bytes")) })
        {
            int bytes = 0;
            var failures = new List<string>();
            foreach (var text in texts)
            {
                var ids = tokenizer.Encode(text);
                bytes += ids.Count(id => tokenizer.TokenOf(id) is { Length: 6 } token && token.StartsWith("<0x", StringComparison.Ordinal));
                Check(!ids.Contains(tokenizer.IdOf("<unk>")!.Value), $"{path}: '{Escaped(text)}' encodes an unknown id with byte fallback on");
                if (tokenizer.Decode(ids) is var decoded && decoded != text)
                {
                    failures.Add($"'{Escaped(text)}' decodes to '{Escaped(decoded)}'");
                }
            }

            Check(failures.Count == 0, $"unigram-bytes from {path}: {failures.Count} of {texts.Count} texts do not round-trip:\n  {string.Join("\n  ", failures.Take(5))}");
            Check(bytes > 1000, $"unigram-bytes from {path}: only {bytes} byte pieces over the texts (they must exercise byte fallback)");
        }

        // Without byte fallback: two unknown characters in a row are one unknown id, as both references fuse them.
        var plain = UnigramFromJson("unigram");
        int unknown = plain.IdOf("<unk>")!.Value;
        var fused = plain.Encode("x\U0002A6D0\U0002A6D1y");
        Check(fused.Count(id => id == unknown) == 1, $"two unknown characters give [{string.Join(",", fused)}]: one unknown id expected");
    }

    private static int NotA(int c) => c == 'A' ? 'B' : c;

    private static void UnigramSampling(Device device)
    {
        _ = device;
        string[] variables = [TokenizerOptions.SamplingAlphaVariable, TokenizerOptions.SamplingBestVariable, TokenizerOptions.SamplingSeedVariable];
        var saved = variables.Select(Environment.GetEnvironmentVariable).ToArray();
        try
        {
            foreach (var variable in variables)
            {
                Environment.SetEnvironmentVariable(variable, null);
            }

            // The unigram fixture (sentencepiece samples 48 of these 60 lines other than the best with alpha 0.1; the
            // unigram-bytes model's scores are so far apart that it samples 2).
            var lines = TokenizerCorpusLines().Where(l => l.Length > 8).Take(60).ToArray();
            var options = new TokenizerOptions();
            Check(options.ResolvedSamplingAlpha() is null && options.ResolvedSamplingBest() == TokenizerOptions.DefaultSamplingBest && options.ResolvedSamplingSeed() is null,
                "sampling is off unless set");

            foreach (var (path, load) in new (string, Func<TokenizerOptions, IPretrainedTokenizer>)[]
                     { ("tokenizer.json", o => UnigramFromJson("unigram", o)), ("tokenizer.model", o => UnigramFromModel("unigram", o)) })
            {
                var best = load(new TokenizerOptions());
                var bestIds = lines.Select(l => best.Encode(l).ToArray()).ToArray();
                Check(lines.Select(l => best.Encode(l)).Zip(bestIds).All(p => p.First.SequenceEqual(p.Second)), $"{path}: unsampled encodes repeat");
                Check(UnigramEncodeAll(load(new TokenizerOptions { SamplingAlpha = 0.1, SamplingBest = 1 }), lines).Zip(bestIds).All(p => p.First.SequenceEqual(p.Second)),
                    $"{path}: SamplingBest 1 gives the best ids");

                foreach (int nbest in new[] { -1, 0, 8 })
                {
                    var settings = new TokenizerOptions { SamplingAlpha = 0.1, SamplingBest = nbest, SamplingSeed = 42 };
                    var first = UnigramEncodeAll(load(settings), lines);
                    var again = UnigramEncodeAll(load(settings), lines);
                    Check(first.Zip(again).All(p => p.First.SequenceEqual(p.Second)), $"{path}, n-best {nbest}: the same seed gives the same samples");
                    var other = UnigramEncodeAll(load(settings with { SamplingSeed = 43 }), lines);
                    int changed = first.Zip(bestIds).Count(p => !p.First.SequenceEqual(p.Second));
                    Check(changed > lines.Length / 4, $"{path}, n-best {nbest}: only {changed} of {lines.Length} lines sampled other than the best segmentation");
                    Check(first.Zip(other).Any(p => !p.First.SequenceEqual(p.Second)), $"{path}, n-best {nbest}: another seed gives other samples");

                    // Every sample is a segmentation of the same normalized text, so it decodes as the best one does; and
                    // the same tokenizer samples anew on each encode (no cache).
                    var sampler = load(settings);
                    for (int i = 0; i < lines.Length; i++)
                    {
                        Check(best.Decode(first[i]) == best.Decode(bestIds[i]), $"{path}, n-best {nbest}: line {i}'s sample decodes to '{Escaped(best.Decode(first[i]))}', not '{Escaped(best.Decode(bestIds[i]))}'");
                    }

                    var draws = Enumerable.Range(0, 20).Select(_ => string.Join(",", sampler.Encode(lines[0]))).Distinct().Count();
                    Check(draws > 1, $"{path}, n-best {nbest}: 20 encodes of one line gave {draws} segmentation (a sampled encode must bypass the cache)");
                }
            }

            Environment.SetEnvironmentVariable(TokenizerOptions.SamplingAlphaVariable, "0.2");
            Environment.SetEnvironmentVariable(TokenizerOptions.SamplingBestVariable, "4");
            Environment.SetEnvironmentVariable(TokenizerOptions.SamplingSeedVariable, "11");
            var fromVariables = new TokenizerOptions();
            Check(fromVariables.ResolvedSamplingAlpha() == 0.2 && fromVariables.ResolvedSamplingBest() == 4 && fromVariables.ResolvedSamplingSeed() == 11,
                "the variables set the sampling");
            Check(new TokenizerOptions { SamplingAlpha = 0.5, SamplingBest = 2, SamplingSeed = 3 } is var set
                  && set.ResolvedSamplingAlpha() == 0.5 && set.ResolvedSamplingBest() == 2 && set.ResolvedSamplingSeed() == 3, "the settings win over the variables");
            var byVariables = UnigramEncodeAll(UnigramFromJson("unigram"), lines);
            var bySettings = UnigramEncodeAll(UnigramFromJson("unigram", new TokenizerOptions { SamplingAlpha = 0.2, SamplingBest = 4, SamplingSeed = 11 }), lines);
            Check(byVariables.Zip(bySettings).All(p => p.First.SequenceEqual(p.Second)), "a tokenizer loaded under the variables samples as one given the same settings");

            Environment.SetEnvironmentVariable(TokenizerOptions.SamplingAlphaVariable, "lots");
            Expect<InvalidOperationException>(() => UnigramFromJson("unigram"), "IDRAK_TOKENIZER_ALPHA is 'lots'", "an alpha that is not a number");
            Environment.SetEnvironmentVariable(TokenizerOptions.SamplingAlphaVariable, "0.1");
            Environment.SetEnvironmentVariable(TokenizerOptions.SamplingBestVariable, "many");
            Expect<InvalidOperationException>(() => UnigramFromJson("unigram"), "IDRAK_TOKENIZER_NBEST is 'many'", "an n-best that is not a number");
            Environment.SetEnvironmentVariable(TokenizerOptions.SamplingBestVariable, null);
            Environment.SetEnvironmentVariable(TokenizerOptions.SamplingSeedVariable, "x");
            Expect<InvalidOperationException>(() => UnigramFromJson("unigram"), "IDRAK_TOKENIZER_SEED is 'x'", "a seed that is not a number");
            Expect<ArgumentOutOfRangeException>(() => new TokenizerOptions { SamplingAlpha = -1 }.ResolvedSamplingAlpha(), "SamplingAlpha", "a negative alpha");
        }
        finally
        {
            for (int i = 0; i < variables.Length; i++)
            {
                Environment.SetEnvironmentVariable(variables[i], saved[i]);
            }
        }
    }

    private static int[][] UnigramEncodeAll(IPretrainedTokenizer tokenizer, string[] lines) => [.. lines.Select(l => tokenizer.Encode(l).ToArray())];

    private static void UnigramSuiteAndSave(Device device)
    {
        _ = device;
        var lines = TokenizerCorpusLines();
        foreach (var name in UnigramFixtures)
        {
            foreach (var (path, tokenizer) in new[] { ("tokenizer.json", UnigramFromJson(name)), ("tokenizer.model", UnigramFromModel(name)) })
            {
                // The suite's "round trip is stable" cannot hold where text decodes to the unknown piece's text: an
                // added token (<unk>) or the surface " ⁇ " re-encodes with the pre-tokenizer's ▁ after it, in both
                // references. Those cases are named; every other failure fails.
                var report = Conformance.CheckPretrained(tokenizer);
                string unknownText = tokenizer.Decode([tokenizer.IdOf("<unk>")!.Value]).Trim();
                var failures = report.Failures.Where(f => !(f.Check == "round trip is stable" && f.Message.Contains(unknownText, StringComparison.Ordinal))).ToList();
                Check(failures.Count == 0, $"{name} from {path}: the pretrained tokenizer suite:\n  {string.Join("\n  ", failures)}");
                foreach (var failure in report.Failures.Except(failures))
                {
                    Console.WriteLine($"    {name} from {path}, the unknown piece's text '{unknownText}' does not re-encode to itself: {failure}");
                }

                var temporary = Directory.CreateTempSubdirectory("idrak-unigram-");
                try
                {
                    tokenizer.Save(temporary.FullName);
                    var loaded = PretrainedTokenizers.Load(temporary.FullName);
                    Check(loaded.Model == TokenizerModels.Unigram, $"{name} from {path}: the saved file loads as {loaded.Model}");
                    var saved = PretrainedTokenizers.Read(temporary.FullName);
                    var original = path == "tokenizer.json" ? TokenizerJsonReader.ReadFile(TokenizerData($"{name}/tokenizer.json")) : SentencePieceDefinition(name);
                    Check(saved.Pieces!.Count == original.Pieces!.Count
                          && saved.Pieces.Zip(original.Pieces).All(p => p.First.Piece == p.Second.Piece
                              && BitConverter.DoubleToInt64Bits(p.First.Score) == BitConverter.DoubleToInt64Bits(p.Second.Score)),
                        $"{name} from {path}: the saved pieces and scores are not the read ones bit for bit");
                    Check((string?)saved.Json["model"]?[UnigramTokenizer.ModeField] == (string?)original.Json["model"]?[UnigramTokenizer.ModeField],
                        $"{name} from {path}: the saved model's mode is {saved.Json["model"]?[UnigramTokenizer.ModeField]}");
                    Check(tokenizer.ToJson()["model"]!["vocab"]!.AsArray().Count == original.Pieces!.Count, $"{name} from {path}: ToJson writes the vocabulary");
                    for (int i = 0; i < lines.Length; i++)
                    {
                        Check(loaded.Encode(lines[i]).SequenceEqual(tokenizer.Encode(lines[i])), $"{name} from {path}: line {i} encodes otherwise after Save and Load");
                    }
                }
                finally
                {
                    temporary.Delete(recursive: true);
                }
            }
        }
    }

    private static void UnigramAllocatesNothing(Device device)
    {
        _ = device;
        var lines = TokenizerCorpusLines();
        var ids = new List<int>(4096);
        foreach (var name in UnigramFixtures)
        {
            foreach (int cache in new[] { TokenizerOptions.DefaultPieceCacheLimit, 0 })
            {
                var options = new TokenizerOptions { PieceCacheLimit = cache };
                foreach (var (path, tokenizer) in new[] { ("tokenizer.json", UnigramFromJson(name, options)), ("tokenizer.model", UnigramFromModel(name, options)) })
                {
                    for (int warm = 0; warm < 3; warm++)
                    {
                        foreach (var line in lines)
                        {
                            ids.Clear();
                            tokenizer.EncodeRange(line, 0, line.Length, ids);
                        }
                    }

                    long before = GC.GetAllocatedBytesForCurrentThread();
                    for (int run = 0; run < 3; run++)
                    {
                        foreach (var line in lines)
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
                        foreach (var (line, i) in lines.Select((l, i) => (l, i)))
                        {
                            long start = GC.GetAllocatedBytesForCurrentThread();
                            ids.Clear();
                            tokenizer.EncodeRange(line, 0, line.Length, ids);
                            if (GC.GetAllocatedBytesForCurrentThread() - start is var bytes and > 0)
                            {
                                culprits.Add($"line {i} ({bytes} bytes) '{Escaped(line[..Math.Min(40, line.Length)])}'");
                            }
                        }

                        Check(false, $"{name} from {path}, cache {cache}: a warm pass allocated {allocated} bytes over {lines.Length * 3} lines:\n  {string.Join("\n  ", culprits.Take(8))}");
                    }
                }
            }
        }
    }

    private static void UnigramSpeed(Device device)
    {
        _ = device;

        // As the fixture script's --bench (and --bench-text): the corpus' non-blank lines, repeated to 200,000 code points.
        var texts = TokenizerBenchTexts(200_000);

        foreach (var name in UnigramFixtures)
        {
            // With the piece cache, and without it (every piece through the lattice, as sentencepiece, which caches nothing).
            var noCache = new TokenizerOptions { PieceCacheLimit = 0 };
            foreach (var (path, load) in new (string, Func<IPretrainedTokenizer>)[]
                     {
                         ("tokenizer.json", () => UnigramFromJson(name)), ("tokenizer.model", () => UnigramFromModel(name)),
                         ("tokenizer.json, no cache", () => UnigramFromJson(name, noCache)), ("tokenizer.model, no cache", () => UnigramFromModel(name, noCache)),
                     })
            {
                var tokenizer = load();
                var ids = new List<int>(4096);
                long Run()
                {
                    long tokens = 0;
                    foreach (var text in texts)
                    {
                        ids.Clear();
                        tokenizer.EncodeRange(text, 0, text.Length, ids);
                        tokens += ids.Count;
                    }

                    return tokens;
                }

                var watch = Stopwatch.StartNew();
                long count = Run();
                double cold = count / watch.Elapsed.TotalSeconds, warm = 0;
                var spent = Stopwatch.StartNew();
                while (spent.ElapsedMilliseconds < 1000)
                {
                    watch.Restart();
                    count = Run();
                    warm = Math.Max(warm, count / watch.Elapsed.TotalSeconds);
                }

                // All threads: the texts split over every core (a fresh tokenizer, so the first pass is cold too).
                var parallel = load();
                long RunAll()
                {
                    long total = 0;
                    System.Threading.Tasks.Parallel.ForEach(System.Collections.Concurrent.Partitioner.Create(0, texts.Count, 64), () => (new List<int>(4096), 0L), (range, _, local) =>
                    {
                        for (int i = range.Item1; i < range.Item2; i++)
                        {
                            local.Item1.Clear();
                            parallel.EncodeRange(texts[i], 0, texts[i].Length, local.Item1);
                            local.Item2 += local.Item1.Count;
                        }

                        return local;
                    }, local => Interlocked.Add(ref total, local.Item2));
                    return total;
                }

                watch.Restart();
                long parallelCount = RunAll();
                double parallelCold = parallelCount / watch.Elapsed.TotalSeconds, parallelWarm = 0;
                spent.Restart();
                while (spent.ElapsedMilliseconds < 1000)
                {
                    watch.Restart();
                    parallelCount = RunAll();
                    parallelWarm = Math.Max(parallelWarm, parallelCount / watch.Elapsed.TotalSeconds);
                }

                Check(count > 0 && parallelCount == count, $"{name} from {path}: {count} tokens on one thread, {parallelCount} on all");
                if (path.EndsWith("no cache", StringComparison.Ordinal))
                {
                    // How much of a pass is the normalizer and pre-tokenizer alone (the same definition's pipeline).
                    var definition = path.StartsWith("tokenizer.json", StringComparison.Ordinal)
                        ? TokenizerJsonReader.ReadFile(TokenizerData($"{name}/tokenizer.json")) : SentencePieceDefinition(name);
                    var pipeline = new TokenizerPipeline(definition.Json);
                    using var pieces = new PreTokenizedText();
                    double fastest = double.MaxValue;
                    for (int round = 0; round < 5; round++)
                    {
                        watch.Restart();
                        foreach (var text in texts)
                        {
                            pipeline.Run(text, 0, text.Length, true, pieces);
                        }

                        fastest = Math.Min(fastest, watch.Elapsed.TotalSeconds);
                    }

                    Console.WriteLine($"    {name} from {path}: the pipeline alone takes {fastest * 1000:F1} ms a pass, {100 * fastest * warm / count:F0}% of a warm pass");
                }

                Console.WriteLine($"    {name} from {path}: {count:N0} tokens of {texts.Sum(t => t.Length):N0} characters; one thread {cold:N0} cold, {warm:N0} warm; "
                    + $"{Environment.ProcessorCount} threads {parallelCold:N0} cold, {parallelWarm:N0} warm (tokens a second)");
            }
        }
    }
}
