// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Formats;
using Idrak.Abstraction.Testing;
using Idrak.Cli;
using Idrak.Cli.Shared;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Layers.Abstractions;
using Idrak.Models;
using Idrak.Audio.Plugins.OpenVoice;
using Idrak.Text.Abstractions;
using Idrak.Text.Retrieval;
using Idrak.Text.Retrieval.Abstractions;

// Plan 26, phase P2: the generic components moved into the library (G1-G7), each with what it replaced proven unchanged:
// the renaming store (the vision and voice plug-ins' copies), the timings probe (the chat sample's recorder), whole code
// points in CharTokenizer (GPT training's rune tokenizer), the progress reporters (the tool's line, the OCR samples'),
// the keyword indexes (the re-ranker's BM25), the network recipes (the GPT samples' network) and finite JSON numbers.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] GenericComponentGroup =
    [
        ("generic components: timings probe: time per operation by name and shape and per layer type and depth in a run of its own, the device waited for meanwhile and restored after, nothing subscribed outside the run", GenericTimingsProbe),
        ("generic components: char tokenizer: the default reads UTF-16 units with the ids it always gave on BMP text (and its file as before); whole code points read a surrogate pair as one id, a lone half as U+FFFD, the unknown character may be any code point, files carry the setting; GPT training's rune tokenizer gives the same ids; the tokenizer suite passes", GenericCharTokenizer),
        ("generic components: progress: the library's console, plain, json and quiet reporters, in order; console redraws in place as the tool's line did, plain writes lines at intervals, json writes records, quiet nothing; the choice by option, IDRAK_PROGRESS (refused when unknown, in the tool too) or the output; an unknown reporter refused", GenericProgress),
        ("generic components: keyword index: Bm25Index passes the keyword-index suite, with its settings too; the re-ranker's BM25 ranked the matching passages the same", GenericKeywordIndex),
        ("generic components: network architectures: gpt with the GPT samples' sizes and random is their network weight for weight", GenericNetworkRecipes),
        ("generic components: json numbers: NaN and infinity written as null by the converters (nullable forms too) and the writer helpers, null read back as NaN, finite numbers written as the serializer writes them, and the profile's json writer writing a NaN value as null", GenericJsonNumbers),
    ];

    // ------------------------------------------------------------------ G1: the renaming tensor store

    // A store that counts its disposals, over another.
    private sealed class CountingStore(ITensorStore inner) : ITensorStore
    {
        public int Disposed { get; private set; }

        public IEnumerable<string> Names => inner.Names;

        public bool Contains(string name) => inner.Contains(name);

        public int[] ShapeOf(string name) => inner.ShapeOf(name);

        public float[] Read(string name) => inner.Read(name);


        public void Dispose() => Disposed++;
    }

    // ------------------------------------------------------------------ G2: the timings probe

    private static void GenericTimingsProbe(Device device)
    {
        var probe = ProfileProbes.Named("timings");
        Check(probe.Intrusive && probe.Supports(device) && ProfileProbes.Describe().Contains("timings", StringComparison.Ordinal),
            "the timings probe is the library's, intrusive, on every device");
        bool synchronize = Telemetry.SynchronizeForTiming;
        var levels = Telemetry.ActiveLevels;
        bool waited = false;
        using var network = new Sequential(new Linear(8, 16, device: device, random: new Random(1)), new ReLU(), new Linear(16, 4, device: device, random: new Random(2)));
        var report = Profiler.Run("timed", device, run =>
        {
            waited |= Telemetry.SynchronizeForTiming && Telemetry.IsEnabled(TelemetryLevel.Operations) && Telemetry.IsEnabled(TelemetryLevel.Layers);
            using var scope = new TensorScope();
            var a = Tensor.From([.. Enumerable.Range(0, 32 * 32).Select(i => (i % 7) / 7f)], [32, 32], device);
            _ = a.MatMul(a).Softmax();
            _ = network.Predict(Tensor.From([.. Enumerable.Range(0, 3 * 8).Select(i => i / 24f)], [3, 8], device));
        }, new ProfileOptions { Cold = false, WarmRuns = 1, Probes = ["timings"] });
        Check(Telemetry.SynchronizeForTiming == synchronize && Telemetry.ActiveLevels == levels, "the device waiting and the subscription end with the run");
        Check(waited, "the probe's run waits for the device and listens to operations and layers");
        var pass = report.Runs.Single(r => r.Probe == "timings");
        Check(report.Runs.Where(r => r.Kind == ProfileRunKind.Warm).All(r => r.Tables.Count == 0 && r.Value("operation calls") is null), "the timed run is not measured by the probe");
        Check(pass.Value("operation calls") >= 2 && pass.Value("operation time") >= 0 && pass.Value("operation shapes") >= 2 && pass.Value("layer calls") >= 3,
            $"values: {string.Join(", ", pass.Values.Select(v => $"{v.Name} {v.Value}"))}");
        var operations = pass.Tables.Single(t => t.Name == "operation timings");
        var layers = pass.Tables.Single(t => t.Name == "layer timings");
        Check(operations.Columns.SequenceEqual(["operation", "shape", "calls", "ms", "share"]) && layers.Columns.SequenceEqual(["layer", "depth", "calls", "ms"]), "the tables' columns");
        Check(operations.Rows.Any(r => r[0] == "matmul" && r[1] == Tensor.FormatShape([32, 32])), $"matmul by its shape: {string.Join("; ", operations.Rows.Select(r => string.Join(" ", r)))}");
        Check(operations.Rows.Select(r => (r[0], r[1])).Distinct().Count() == operations.Rows.Count, "one row per operation and shape");
        Check(layers.Rows.Any(r => r[1] == "0") && layers.Rows.Any(r => r[1] == "1") && layers.Rows.Sum(r => int.Parse(r[2], System.Globalization.CultureInfo.InvariantCulture)) == (int)pass.Value("layer calls")!,
            $"layers by type and depth: {string.Join("; ", layers.Rows.Select(r => string.Join(" ", r)))}");
        var ms = operations.Rows.Select(r => double.Parse(r[3], System.Globalization.CultureInfo.InvariantCulture)).ToList();
        Check(ms.Zip(ms.Skip(1)).All(p => p.First >= p.Second), "the slowest first");
    }

    // ------------------------------------------------------------------ G3: whole code points

    private static void GenericCharTokenizer(Device device)
    {
        _ = device;
        const string Alphabet = "abc الن.";
        string text = "a b.ناxzéc";                                    // x, z and e-acute are outside the alphabet

        // The default: one id per UTF-16 unit, as always: position in the alphabet, unknown characters dropped or replaced.
        int[] expected = [.. text.Where(c => Alphabet.Contains(c)).Select(c => Alphabet.IndexOf(c))];
        var plain = new CharTokenizer(Alphabet);
        Check(plain.Encode(text).SequenceEqual(expected) && plain.Options == new CharTokenizerOptions() && plain.VocabularySize == Alphabet.Length, "default ids on BMP text");
        var points = new CharTokenizer(Alphabet, new CharTokenizerOptions { CodePoints = true });
        Check(points.Encode(text).SequenceEqual(expected) && points.VocabularySize == Alphabet.Length && points.Decode(expected) == plain.Decode(expected),
            "code points read BMP text as UTF-16 units do");
        var replacing = new CharTokenizer(Alphabet, ' ');
        Check(replacing.Encode(text).SequenceEqual(text.Select(c => Alphabet.Contains(c) ? Alphabet.IndexOf(c) : 3)) && replacing.UnknownCharacter == ' '
              && new CharTokenizer(Alphabet, 'q').UnknownCharacter is null, "an unknown character in the alphabet replaces, outside it is ignored (as before)");
        var saved = new MemoryStream();
        replacing.Save(saved);
        var json = JsonNode.Parse(saved.ToArray())!.AsObject();
        Check(json.Count == 3 && (string?)json["type"] == "char" && (string?)json["vocabulary"] == Alphabet && (string?)json["unknown"] == " ", $"the default file as before: {json}");

        // Whole code points: a pair is one id, a lone half reads as U+FFFD, the unknown character may be beyond the BMP.
        const string Emoji = "😀", Rocket = "🚀";
        var wide = new CharTokenizer(["a", Emoji, "b", "�", Rocket], new CharTokenizerOptions { CodePoints = true, UnknownCharacter = Rocket });
        Check(wide.VocabularySize == 5 && wide.Vocabulary == "a" + Emoji + "b�" + Rocket, "five code points, one string");
        Check(wide.Encode("a" + Emoji + "b").SequenceEqual([0, 1, 2]) && wide.Encode("x\uD83Db\uDE00").SequenceEqual([4, 3, 2, 3]), "pairs whole, lone halves as U+FFFD, unknown as the rocket");
        Check(wide.Decode([1, 0, 4, 9]) == Emoji + "a" + Rocket + "�" && wide.UnknownCharacter is null && wide.Options.UnknownCharacter == Rocket, "decoded with pairs whole");
        Check(wide.Covers("ab" + Emoji) && !wide.Covers("c"), "covers by code point");
        var halves = new CharTokenizer("a" + Emoji);
        Check(halves.Encode(Emoji).Count == 2 && halves.VocabularySize == 3, "UTF-16 units split a pair (the default, unchanged)");
        var written = new MemoryStream();
        wide.Save(written);
        var wideJson = JsonNode.Parse(written.ToArray())!.AsObject();
        Check((bool?)wideJson["code_points"] == true && (string?)wideJson["unknown"] == Rocket && (string?)wideJson["vocabulary"] == wide.Vocabulary,
            $"the setting and a wide unknown are in the file: {wideJson}");
        ThrowsNaming<ArgumentException>(() => _ = new CharTokenizer(["a", "a"], new CharTokenizerOptions { CodePoints = true }), "a repeated code point", ["twice"]);
        ThrowsNaming<ArgumentException>(() => _ = new CharTokenizer(["ab"], new CharTokenizerOptions { CodePoints = true }), "two characters as one", ["not one code point"]);
        ThrowsNaming<ArgumentException>(() => _ = new CharTokenizer("ab", new CharTokenizerOptions { CodePoints = true, UnknownCharacter = "c" }), "an unknown character outside", ["not one code point"]);

        // GPT training's rune tokenizer (its vocabulary file read as a list of code points; unknown: the space, else id 0).
        var random = new Random(5);
        string[] symbols = [" ", "a", "ب", Emoji, "\n", "é", Rocket];
        var tokenizer = new CharTokenizer(symbols, new CharTokenizerOptions { CodePoints = true, UnknownCharacter = symbols.Contains(" ") ? " " : symbols[0] });
        var index = symbols.Select((s, i) => (s, i)).ToDictionary(p => p.s, p => p.i);
        for (int round = 0; round < 50; round++)
        {
            string sample = string.Concat(Enumerable.Range(0, random.Next(0, 30)).Select(_ => random.Next(4) == 0 ? "z" : random.Next(9) == 0 ? "\uD83D" : symbols[random.Next(symbols.Length)]));
            int[] runes = [.. sample.EnumerateRunes().Select(r => index.TryGetValue(r.ToString(), out int id) ? id : index[" "])];
            Check(tokenizer.Encode(sample).SequenceEqual(runes), $"the rune tokenizer's ids for '{sample}'");
            Check(tokenizer.Decode(runes) == string.Concat(runes.Select(i => symbols[i])), "and its text");
        }

        Conformance.Check(wide, options: new ContractCheckOptions { RandomCases = 10 }).ThrowIfFailed();
        Conformance.Check(points, plain, new ContractCheckOptions { RandomCases = 10 }).ThrowIfFailed();
    }

    // ------------------------------------------------------------------ G4: progress

    private static void GenericProgress(Device device)
    {
        _ = device;
        Check(ProgressReporters.Names.SequenceEqual([ProgressReporters.ConsoleReporter, ProgressReporters.PlainReporter, ProgressReporters.JsonReporter, ProgressReporters.QuietReporter]),
            "the library's four reporters, in order");

        // console: redrawn in place, as the tool's line was.
        var screen = new StringWriter();
        var console = ProgressReporters.Start("work", 10, ProgressUnit.Items, new ProgressOptions { Reporter = "console", Output = screen });
        console.Report(5);
        Check(screen.ToString() == "\rwork   50%  5 / 10", $"console: '{screen}'");
        console.Erase();
        console.Detail = () => "loss 0.5";
        console.Report(6);
        console.Finish();
        Check(screen.ToString().StartsWith("\rwork   50%  5 / 10\r" + new string(' ', 18) + "\r\rwork   60%  6 / 10  ", StringComparison.Ordinal)
              && screen.ToString().Contains("  loss 0.5", StringComparison.Ordinal) && screen.ToString().EndsWith('\n'), $"erased, redrawn with the detail, finished: '{screen}'");

        // plain: a line, then none until the interval passes; json: records.
        var lines = new StringWriter();
        using (var plain = ProgressReporters.Start("copy", 1 << 20, ProgressUnit.Bytes, new ProgressOptions { Reporter = "plain", Output = lines, Interval = TimeSpan.FromHours(1) }))
        {
            plain.Report(512 << 10);
            plain.Report(600 << 10);
            plain.NextStage("verify", null, ProgressUnit.Items, "copied");
            plain.Finish("done");
        }

        Check(lines.ToString().Split('\n') is [var first, "copied", "done", ""] && first.StartsWith("copy   50%  512 KB / 1 MB", StringComparison.Ordinal), $"plain: '{lines}'");
        var records = new StringWriter();
        using (var json = ProgressReporters.Start("load", 4, ProgressUnit.Items, new ProgressOptions { Reporter = "json", Output = records, Interval = TimeSpan.FromHours(1) }))
        {
            json.Report(1);
            json.Report(2);
            json.Finish("loaded");
        }

        var parsed = records.ToString().Split('\n', StringSplitOptions.RemoveEmptyEntries).Select(l => JsonNode.Parse(l)!.AsObject()).ToList();
        Check(parsed.Count == 2 && (string?)parsed[0]["label"] == "load" && (long?)parsed[0]["done"] == 1 && (long?)parsed[0]["total"] == 4 && (string?)parsed[0]["unit"] == "items"
              && (bool?)parsed[1]["end"] == true && (string?)parsed[1]["message"] == "loaded" && (long?)parsed[1]["done"] == 2, $"json: '{records}'");
        var quiet = new StringWriter();
        using (var none = ProgressReporters.Start("x", 3, ProgressUnit.Items, new ProgressOptions { Reporter = "quiet", Output = quiet }))
        {
            none.Report(2);
            none.Show();
            Check(!none.Enabled && none.Done == 2, "quiet counts");
        }

        Check(quiet.ToString().Length == 0, "quiet writes nothing");

        // The choice: an option, then IDRAK_PROGRESS, then the output (another writer than the console's: quiet).
        WithEnvironment(new Dictionary<string, string?> { [ProgressReporters.EnvironmentVariable] = null }, () =>
        {
            Check(ProgressReporters.Choose(new ProgressOptions { Output = new StringWriter() }) == "quiet" && ProgressReporters.FromEnvironment() is null, "another writer: quiet");
            Check(ProgressReporters.Choose(new ProgressOptions { Reporter = "json", Output = new StringWriter() }) == "json", "the option wins");
            Check(ProgressReporters.Start("s", options: new ProgressOptions { Output = new StringWriter() }) is { Enabled: false }, "Start makes the chosen one");
        });
        WithEnvironment(new Dictionary<string, string?> { [ProgressReporters.EnvironmentVariable] = " plain " }, () =>
        {
            Check(ProgressReporters.Choose(new ProgressOptions { Output = new StringWriter() }) == "plain" && ProgressReporters.Choose(new ProgressOptions { Reporter = "quiet" }) == "quiet",
                "IDRAK_PROGRESS chooses unless the option does");
            var command = CommandTable.All.First(c => c.Name == "version");
            using var context = new CommandContext(command, [], [], [], new StringWriter(), new StringWriter());
            Check(new ProgressLine(context, "work", 10).Enabled && !new ProgressLine(context, "work", 10, enabled: false).Enabled, "the tool takes IDRAK_PROGRESS where it would draw nothing");
        });
        WithEnvironment(new Dictionary<string, string?> { [ProgressReporters.EnvironmentVariable] = "fancy" }, () =>
        {
            ThrowsNaming<ArgumentException>(() => ProgressReporters.Choose(), "an unknown reporter", ["IDRAK_PROGRESS=fancy", "console, plain, json, quiet"]);
            var command = CommandTable.All.First(c => c.Name == "version");
            using var context = new CommandContext(command, [], [], [], new StringWriter(), new StringWriter());
            ThrowsNaming<UsageException>(() => _ = new ProgressLine(context, "work", 10), "the tool refuses it", ["IDRAK_PROGRESS=fancy"]);
        });
        Check(EnvironmentVariables.All.Any(v => v.Name == ProgressReporters.EnvironmentVariable), "IDRAK_PROGRESS is in the tool's table");

        // An unknown name refused.
        ThrowsNaming<NotSupportedException>(() => ProgressReporters.Start("x", options: new ProgressOptions { Reporter = "fancy", Output = new StringWriter() }), "an unknown reporter",
            ["'fancy'", "console, plain, json, quiet"]);
    }

    // ------------------------------------------------------------------ G5: keyword indexes

    // The re-ranker sample's BM25 (before it took the library's): every text scored by a full scan.
    private static int[] OldBm25(IReadOnlyList<string> texts, string query, int k, double k1 = 1.2, double b = 0.75)
    {
        var documents = texts.Select(d => WordTokenizer.Split(d).ToArray()).ToList();
        double average = documents.Average(d => d.Length);
        var idf = documents.SelectMany(d => d.Distinct()).GroupBy(w => w).ToDictionary(g => g.Key, g => Math.Log(1 + (documents.Count - g.Count() + 0.5) / (g.Count() + 0.5)));
        var terms = WordTokenizer.Split(query).Where(idf.ContainsKey).Distinct().ToArray();
        var scores = documents.Select(doc => terms.Sum(term => doc.Count(w => w == term) is int tf and > 0 ? idf[term] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc.Length / average)) : 0)).ToArray();
        return [.. Enumerable.Range(0, scores.Length).Where(d => scores[d] > 0).OrderByDescending(d => scores[d]).ThenBy(d => d).Take(k)];
    }

    private static void GenericKeywordIndex(Device device)
    {
        _ = device;
        static KeywordIndexUnderTest Of(Func<IEnumerable<string>, IKeywordIndex> build) => new(texts => build(texts).Search);
        Conformance.CheckKeywordIndex(Of(texts => new Bm25Index(texts))).ThrowIfFailed();
        Conformance.CheckKeywordIndex(Of(texts => new Bm25Index(texts, 1.7, 0.3))).ThrowIfFailed();

        // The re-ranker's BM25 ranked the passages sharing a word with the query as the library does.
        var random = new Random(11);
        string[] words = ["how", "many", "people", "live", "in", "north", "river", "town", "the", "market", "has", "about", "residents"];
        string[] passages = [.. Enumerable.Range(0, 60).Select(_ => string.Join(' ', Enumerable.Range(0, random.Next(1, 14)).Select(_ => words[random.Next(words.Length)])))];
        var library = new Bm25Index(passages);
        for (int q = 0; q < 30; q++)
        {
            string query = string.Join(' ', Enumerable.Range(0, random.Next(1, 6)).Select(_ => words[random.Next(words.Length)]));
            Check(library.Search(query, 20).Select(h => h.Id).SequenceEqual(OldBm25(passages, query, 20)), $"'{query}': the re-ranker's ranking");
        }
    }

    // ------------------------------------------------------------------ G6: network architectures

    private static void GenericNetworkRecipes(Device device)
    {
        // gpt with the GPT samples' sizes and random is the network they built themselves, weight for weight.
        const int Vocabulary = 23, Context = 16, Dim = 24, Heads = 4, Layers = 2;
        var random = new Random(2);
        using var own = new Sequential { new Embedding(Vocabulary, Dim, device, random), new PositionalEncoding(Context, Dim, device) };
        for (int layer = 0; layer < Layers; layer++)
        {
            own.Add(new TransformerEncoderLayer(Dim, Heads, ffDim: 4 * Dim, dropout: 0.1f, causal: true, device: device, random: random));
        }

        own.Add(new LayerNorm(Dim, device: device));
        own.Add(new Linear(Dim, Vocabulary, device: device, random: random));
        using var recipe = Architectures.Gpt(Vocabulary, Context, Dim, Heads, Layers, 4 * Dim, 0.1f).OnDevice(device).WithRandom(new Random(2)).Named("char-gpt").Build();
        var a = own.Parameters().ToList();
        var b = recipe.Parameters().ToList();
        Check(a.Count == b.Count && own.ParameterCount == recipe.ParameterCount && recipe.Name == "char-gpt" && recipe.Count == own.Count, $"{b.Count} parameters, {a.Count} expected");
        for (int i = 0; i < a.Count; i++)
        {
            Check(a[i].Shape.SequenceEqual(b[i].Shape) && a[i].ToArray().SequenceEqual(b[i].ToArray()), $"parameter {i} is the samples' own");
        }

        using var ids = Tensor.From([.. Enumerable.Range(0, 2 * Context).Select(i => (float)(i * 7 % Vocabulary))], [2, Context], device);
        AssertClose(own.Predict(ids).ToArray(), recipe.Predict(ids).ToArray(), 0f, "the same predictions");
    }

    // ------------------------------------------------------------------ G7: finite JSON numbers

    private sealed record FiniteSample(float A, double B, float C, double D, float? E, double? F);

    private static void GenericJsonNumbers(Device device)
    {
        var options = FiniteJsonNumbers.Add(FiniteJsonNumbers.Add(new JsonSerializerOptions()));
        Check(options.Converters.Count == 2 && options.Converters.Any(c => c is FiniteSingleJsonConverter) && options.Converters.Any(c => c is FiniteDoubleJsonConverter), "added once");
        string json = JsonSerializer.Serialize(new FiniteSample(float.NaN, double.PositiveInfinity, 1.5f, 2.25, float.NegativeInfinity, double.NaN), options);
        Check(json == """{"A":null,"B":null,"C":1.5,"D":2.25,"E":null,"F":null}""", json);
        var finite = new FiniteSample(0.1f, 1e-300, -3.25e7f, Math.PI, 2f, null);
        Check(JsonSerializer.Serialize(finite, options) == JsonSerializer.Serialize(finite), "finite numbers as the serializer writes them");
        var back = JsonSerializer.Deserialize<FiniteSample>("""{"A":null,"B":null,"C":1.5,"D":2.25,"E":null,"F":3}""", options)!;
        Check(float.IsNaN(back.A) && double.IsNaN(back.B) && back.C == 1.5f && back.E is null && back.F == 3, "null read back as NaN (and as null where the type allows it)");
        var buffer = new MemoryStream();
        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            FiniteJsonNumbers.WriteNumber(writer, "x", double.NaN);
            FiniteJsonNumbers.WriteNumber(writer, "y", 4.5);
            writer.WriteStartArray("z");
            FiniteJsonNumbers.WriteNumberValue(writer, double.NegativeInfinity);
            FiniteJsonNumbers.WriteNumberValue(writer, 1);
            writer.WriteEndArray();
            writer.WriteEndObject();
        }

        Check(System.Text.Encoding.UTF8.GetString(buffer.ToArray()) == """{"x":null,"y":4.5,"z":[null,1]}""", "the writer helpers");

        // The profile's json writer: a NaN value as null, the rest as before.
        var report = Profiler.Run("nan", device, run =>
        {
            run.Record("loss", double.NaN);
            run.Record("items", 3, "rows");
        }, new ProfileOptions { Cold = false, WarmRuns = 1, Probes = [] });
        var written = new StringWriter();
        report.Write(written, "json");
        var parsed = JsonNode.Parse(written.ToString())!;
        var values = parsed["runs"]![0]!["values"]!.AsArray();
        Check(values.Single(v => (string?)v!["name"] == "loss")!["value"] is null && (double?)values.Single(v => (string?)v!["name"] == "items")!["value"] == 3
              && parsed["summary"]!.AsArray().Single(s => (string?)s!["name"] == "loss")!["value"] is null, written.ToString());
    }
}
