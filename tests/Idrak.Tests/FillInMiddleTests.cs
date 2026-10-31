// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Text.Plugins.DeepSeek;
using Idrak.Text.Plugins.Gemma;
using Idrak.Text.Plugins.Mistral;
using Idrak.Text.Plugins.Qwen;

// Craft's plan, phase A0: fill-in-the-middle as a contract (IFillInMiddleFormat in FillInMiddleFormats, the library's two
// layouts, no family of its own), the families' formats in Idrak.Text.Plugins, and the editor's pattern measured: one
// request per pause, each cancelled by the next, the prefix growing a few characters at a time (FillInMiddleBench; its
// numbers with --bench-fim).
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] FillInMiddleGroup =
    [
        ("fill-in-middle: contract: the library registers no format; a name nobody registered is refused as not registered naming FillInMiddleFormats.Register and Idrak.Text.Plugins; a tokenizer without a registered format's tokens detects none (For refuses it); an app's layout registers, is detected, asked first, and unregisters; the two layouts write prefix-suffix-middle and suffix-prefix-middle (with and without a middle token), Begin, repository files after a separator or a marker, and fail naming a missing token", d => { if (d == Device.Cpu) FillInMiddleContract(); }),
        ("fill-in-middle: suite: FillInMiddleFormatSuite passes both layouts and catches a format whose ids are not its text's and one that rewrites the prefix", d => { if (d == Device.Cpu) FillInMiddleSuiteKit(); }),
        ("fill-in-middle: families: qwen, deepseek, gemma and mistral are Idrak.Text.Plugins', each detected from a tokenizer with its family's tokens (and only that family's), laid out as the family documents it with its stops, passing the suite; unregistering the family takes its format away; CodeLlama registers none", d => { if (d == Device.Cpu) FillInMiddleFamilies(); }),
        ("fill-in-middle: bench: a thousand requests of the editor's pattern on a tiny decoder: KeepCache reuses at least the shared prompt prefix on every request (prefill tokens processed below the prompt), a completed request's text is a fresh generator's, a request cancelled at a chunk samples no token after it, one cancelled before its prompt's pass keeps the cache, StreamAsync's reader stopping leaves the cache held, and device and managed memory do not grow from the tenth request to the last", FillInMiddleBenchChecks),
    ];

    // A family's tokenizer for the tests: the tiny Qwen3 fixture's byte-level BPE with the family's tokens added (special).
    internal static BpeTokenizer FamilyTokenizer(params string[] tokens)
    {
        var json = JsonNode.Parse(File.ReadAllText(TestData("gguf/tiny-qwen3-q8-hf/tokenizer.json")))!.AsObject();
        var added = json["added_tokens"]!.AsArray();
        int next = added.Max(t => (int)t!["id"]!) + 1;
        foreach (string token in tokens)
        {
            added.Add(new JsonObject { ["id"] = next++, ["content"] = token, ["special"] = true });
        }

        return BpeTokenizer.FromJson(json);
    }

    internal static readonly string[] QwenFimTokens = ["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>", "<|endoftext|>"];

    internal static readonly string[] DeepSeekFimTokens = ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜fim▁begin｜>", "<｜fim▁hole｜>", "<｜fim▁end｜>", "<|EOT|>"];

    internal static readonly string[] GemmaFimTokens = ["<bos>", "<eos>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|file_separator|>"];

    internal static readonly string[] MistralFimTokens = ["<s>", "</s>", "[PREFIX]", "[SUFFIX]", "[MIDDLE]"];

    private static void FillInMiddleContract()
    {
        string library = Overrides.Library;
        Check(FillInMiddleFormats.Names.All(n => FillInMiddleFormats.Origin(n) != library && FillInMiddleFormats.Default(n) is null),
            $"the library registers no fill-in-the-middle format: {string.Join(", ", FillInMiddleFormats.Names.Select(n => $"{n}={FillInMiddleFormats.Origin(n)}"))}");
        ThrowsNaming<NotRegisteredException>(() => FillInMiddleFormats.Get("no-such-format"), "an unregistered format",
            "No fill-in-the-middle format 'no-such-format' is registered", "FillInMiddleFormats.Register", "Idrak.Text.Plugins");
        var plain = FamilyTokenizer();
        Check(FillInMiddleFormats.Detect(plain) is null, $"a tokenizer with no FIM tokens detects none, not {FillInMiddleFormats.Detect(plain)}");
        ThrowsNaming<NotRegisteredException>(() => FillInMiddleFormats.For(plain), "For without a format", "No fill-in-the-middle format registered for this tokenizer", "FillInMiddleFormats.Register");

        // An app's layout: registered under its name, detected first (newest first), its Origin the registering assembly.
        var tokenizer = FamilyTokenizer("<PRE>", "<SUF>", "<MID>", "<SEP>", "<BEGIN>", "<END>");
        var psm = FillInMiddleLayout.PrefixSuffixMiddle("test-psm", "a test layout",
            new FillInMiddleTokens("<PRE>", "<SUF>", "<MID>") { Begin = "<BEGIN>", FileSeparator = "<SEP>", Stop = ["<END>", "<NOT-THERE>"] });
        FillInMiddleFormats.Register(psm);
        try
        {
            Check(FillInMiddleFormats.Detect(tokenizer) == "test-psm" && FillInMiddleFormats.Names.First() == "test-psm" && ReferenceEquals(FillInMiddleFormats.For(tokenizer), psm)
                  && FillInMiddleFormats.Origin("test-psm") == typeof(DesktopTests).Assembly.GetName().Name,
                $"the app's layout is detected first ({FillInMiddleFormats.Detect(tokenizer)}; {string.Join(", ", FillInMiddleFormats.Names)}; origin {FillInMiddleFormats.Origin("test-psm")})");
            var prompt = FillInMiddleFormats.Create("test-psm", new FillInMiddleContext(tokenizer, "def f():\n    ", "\n    return x")
            {
                Snippets = [new("a.py", "A = 1")], Path = "b.py",
            });
            const string Expected = "<BEGIN><SEP>a.py\nA = 1\n<SEP>b.py\n<PRE>def f():\n    <SUF>\n    return x<MID>";
            Check(prompt.Text == Expected && prompt.Snippets == 1, $"prefix-suffix-middle: '{prompt.Text}' ({prompt.Snippets} snippets)");
            Check(prompt.Text[..prompt.PrefixEnd].EndsWith("def f():\n    ", StringComparison.Ordinal) && prompt.Ids.SequenceEqual(tokenizer.Encode(prompt.Text)),
                $"PrefixEnd {prompt.PrefixEnd} ends the prefix and the ids are the text's");
            Check(prompt.Stop.SequenceEqual(["<END>"]) && prompt.StopIds.SequenceEqual([tokenizer.IdOf("<END>")!.Value]), $"stops: only the tokenizer's ({string.Join(", ", prompt.Stop)})");
            var options = FillInMiddleFormats.Options(prompt, new GenerationOptions { Stop = ["\n\n", "<END>"] });
            Check(options.Stop.SequenceEqual(["\n\n", "<END>"]), $"Options adds the stops once: {string.Join(", ", options.Stop)}");
        }
        finally
        {
            Check(FillInMiddleFormats.Unregister("test-psm") && !FillInMiddleFormats.Names.Contains("test-psm") && FillInMiddleFormats.Detect(tokenizer) is null,
                "Unregister takes the app's layout back");
        }

        // Suffix first, with a middle token (the FIM paper's SPM) and without one (Mistral's), and a marker for snippets.
        var spm = FillInMiddleLayout.SuffixPrefixMiddle("test-spm", "suffix first", new FillInMiddleTokens("<PRE>", "<SUF>", "<MID>") { SnippetMarker = "# " });
        var spmPrompt = spm.Create(new FillInMiddleContext(tokenizer, "x = ", "\nprint(x)") { Snippets = [new("a.py", "A = 1")], Path = "b.py" });
        Check(spmPrompt.Text == "<PRE><SUF>\nprint(x)<MID># a.py\nA = 1\n# b.py\nx = " && spmPrompt.PrefixEnd == spmPrompt.Text.Length && spmPrompt.Snippets == 1,
            $"suffix-prefix-middle with a middle token: '{spmPrompt.Text}'");
        var bare = FillInMiddleLayout.SuffixPrefixMiddle("test-bare", "no middle", new FillInMiddleTokens("<PRE>", "<SUF>", null));
        var barePrompt = bare.Create(new FillInMiddleContext(tokenizer, "x = ", "\n") { Snippets = [new("a.py", "A = 1")] });
        Check(barePrompt.Text == "<SUF>\n<PRE>x = " && barePrompt.Snippets == 0, $"suffix-prefix without a middle token or a way to write snippets: '{barePrompt.Text}' ({barePrompt.Snippets} snippets)");
        ThrowsNaming<ArgumentException>(() => FillInMiddleLayout.PrefixSuffixMiddle("x", "y", new FillInMiddleTokens("<PRE>", "<SUF>", null)), "PSM without a middle", "needs a middle token");
        ThrowsNaming<NotSupportedException>(() => psm.Create(new FillInMiddleContext(FamilyTokenizer("<PRE>"), "a", "b")), "a tokenizer missing tokens",
            "'test-psm' needs the tokens", "no <SUF>, <MID>, <SEP>", "FillInMiddleFormats.Detect");
        Check(!psm.Detect(FamilyTokenizer("<PRE>", "<SUF>", "<MID>")), "a layout with a file separator is not detected without it");
    }

    private static void FillInMiddleSuiteKit()
    {
        var tokenizer = FamilyTokenizer("<PRE>", "<SUF>", "<MID>", "<SEP>", "<REPO>", "<END>");
        var psm = FillInMiddleLayout.PrefixSuffixMiddle("kit-psm", "kit", new FillInMiddleTokens("<PRE>", "<SUF>", "<MID>") { Repository = "<REPO>", FileSeparator = "<SEP>", Stop = ["<END>"] });
        var spm = FillInMiddleLayout.SuffixPrefixMiddle("kit-spm", "kit", new FillInMiddleTokens("<PRE>", "<SUF>", null) { SnippetMarker = "#", Stop = ["<END>"] });
        var options = new ContractCheckOptions { RandomCases = 30 };
        Conformance.CheckFillInMiddleFormat("kit-psm", tokenizer, psm, psm, options).ThrowIfFailed();
        Conformance.CheckFillInMiddleFormat("kit-spm", tokenizer, spm, options: options).ThrowIfFailed();
        Conformance.CheckFillInMiddleFormat("kit-spm", tokenizer, spm, options: options with { Large = true, RandomCases = 4 }).ThrowIfFailed();

        var wrongIds = Conformance.CheckFillInMiddleFormat("kit-psm", tokenizer, new Rewritten(psm, ids: true));
        Check(!wrongIds.Passed && wrongIds.Failures.Any(f => f.Check == "the ids are the text's"), wrongIds.ToString());
        var moved = Conformance.CheckFillInMiddleFormat("kit-psm", tokenizer, new Rewritten(psm, ids: false), reference: psm);
        Check(!moved.Passed && moved.Failures.Any(f => f.Check == "a longer prefix keeps the text up to the old prefix's end") && moved.Failures.Any(f => f.Check == "agrees with the reference"),
            moved.ToString());
    }

    // A broken format: its ids miss the last one, or it writes the prefix's length as the file's path (so a longer prefix moves the text before it).
    private sealed class Rewritten(IFillInMiddleFormat inner, bool ids) : IFillInMiddleFormat
    {
        public string Name => inner.Name;

        public string Summary => "broken";

        public IReadOnlySet<string> Keys => OptionKeys.None;

        public bool Detect(IPretrainedTokenizer tokenizer) => inner.Detect(tokenizer);

        public IFillInMiddlePrompt Create(FillInMiddleContext request)
        {
            if (ids)
            {
                var prompt = inner.Create(request);
                return new FillInMiddlePrompt(prompt.Text, [.. prompt.Ids.SkipLast(1)], prompt.Stop, prompt.StopIds, prompt.PrefixEnd, prompt.Snippets);
            }

            return inner.Create(request with { Path = request.Prefix.Length.ToString(CultureInfo.InvariantCulture) });
        }
    }

    private static void FillInMiddleFamilies()
    {
        string package = "Idrak.Text.Plugins";
        var families = new (string Name, string[] Tokens, string Expected, string[] Stops)[]
        {
            (QwenPlugin.FillInMiddleFormat, QwenFimTokens, "<|fim_prefix|>def f():\n    <|fim_suffix|>\n    return x<|fim_middle|>",
                ["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>", "<|endoftext|>", "<|im_end|>"]),
            (DeepSeekPlugin.FillInMiddleFormat, DeepSeekFimTokens, "<｜begin▁of▁sentence｜><｜fim▁begin｜>def f():\n    <｜fim▁hole｜>\n    return x<｜fim▁end｜>",
                ["<｜end▁of▁sentence｜>", "<|EOT|>"]),
            (GemmaPlugin.FillInMiddleFormat, GemmaFimTokens, "<bos><|fim_prefix|>def f():\n    <|fim_suffix|>\n    return x<|fim_middle|>",
                ["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|file_separator|>", "<eos>"]),
            (MistralPlugin.FillInMiddleFormat, MistralFimTokens, "<s>[SUFFIX]\n    return x[PREFIX]def f():\n    ", ["</s>"]),
        };
        foreach (var (name, tokens, expected, stops) in families)
        {
            var tokenizer = FamilyTokenizer(tokens);
            Check(FillInMiddleFormats.Origin(name) == package && FillInMiddleFormats.Default(name) is null, $"{name}: registered by {package} ({FillInMiddleFormats.Origin(name)})");
            Check(FillInMiddleFormats.Detect(tokenizer) == name, $"{name}: its tokenizer detects '{FillInMiddleFormats.Detect(tokenizer)}'");
            var prompt = FillInMiddleFormats.For(tokenizer).Create(new FillInMiddleContext(tokenizer, "def f():\n    ", "\n    return x"));
            Check(prompt.Text == expected, $"{name}: '{prompt.Text}', the family's layout '{expected}'");
            Check(prompt.Stop.SequenceEqual(stops), $"{name}: stops {string.Join(", ", prompt.Stop)}, the family's {string.Join(", ", stops)}");
            Conformance.CheckFillInMiddleFormat(name, tokenizer, options: new ContractCheckOptions { RandomCases = 12 }).ThrowIfFailed();
        }

        // Repository files as each family writes them.
        var qwen = FamilyTokenizer(QwenFimTokens);
        var repository = new FillInMiddleContext(qwen, "x = ", "") { Repository = "demo", Snippets = [new("util.py", "A = 1")], Path = "main.py" };
        string qwenText = FillInMiddleFormats.Create(QwenPlugin.FillInMiddleFormat, repository).Text;
        Check(qwenText == "<|repo_name|>demo\n<|file_sep|>util.py\nA = 1\n<|file_sep|>main.py\n<|fim_prefix|>x = <|fim_suffix|><|fim_middle|>", $"qwen repository level: '{qwenText}'");
        var deepSeek = FamilyTokenizer(DeepSeekFimTokens);
        string deepSeekText = FillInMiddleFormats.Create(DeepSeekPlugin.FillInMiddleFormat, repository with { Tokenizer = deepSeek }).Text;
        Check(deepSeekText == "<｜begin▁of▁sentence｜><｜fim▁begin｜>#util.py\nA = 1\n#main.py\nx = <｜fim▁hole｜><｜fim▁end｜>", $"deepseek repository files: '{deepSeekText}'");
        var mistral = FamilyTokenizer(MistralFimTokens);
        Check(FillInMiddleFormats.Create(MistralPlugin.FillInMiddleFormat, repository with { Tokenizer = mistral }).Snippets == 0, "mistral writes no repository files, and says so");

        // A Qwen tokenizer is not CodeGemma's (no <|file_separator|>) and the other way round, though both have <|fim_prefix|>.
        Check(FillInMiddleFormats.Detect(FamilyTokenizer(GemmaFimTokens)) == GemmaPlugin.FillInMiddleFormat && !GemmaPlugin.FillInMiddle.Detect(qwen) && !QwenPlugin.FillInMiddle.Detect(FamilyTokenizer(GemmaFimTokens)),
            "Qwen's and CodeGemma's tokens tell them apart");
        Check(!FillInMiddleFormats.Names.Any(n => n.Contains("llama", StringComparison.OrdinalIgnoreCase)), "CodeLlama registers no format (its infilling is ids-only)");

        QwenPlugin.Unregister();
        try
        {
            Check(!FillInMiddleFormats.Names.Contains(QwenPlugin.FillInMiddleFormat) && FillInMiddleFormats.Detect(qwen) is null, "unregistering Qwen takes its format away");
            ThrowsNaming<NotRegisteredException>(() => FillInMiddleFormats.Get(QwenPlugin.FillInMiddleFormat), "qwen unregistered", "Idrak.Text.Plugins");
        }
        finally
        {
            QwenPlugin.Register();
        }

        Check(FillInMiddleFormats.Detect(qwen) == QwenPlugin.FillInMiddleFormat, "registering Qwen again brings it back");
    }

    private static void FillInMiddleBenchChecks(Device device)
    {
        var tokenizer = FamilyTokenizer(QwenFimTokens);
        var format = FillInMiddleFormats.Get(QwenPlugin.FillInMiddleFormat);
        using var model = TinyFimDecoder(tokenizer, device);
        var result = FillInMiddleBench(model, tokenizer, format, device, new FillInMiddleBenchOptions { Requests = 1000, Context = 256, Predict = 24, Verify = 6 });
        Console.WriteLine(result.Report());
        Check(result.ReuseShortfalls.Count == 0, $"KeepCache reuses the shared prompt prefix on every request: {string.Join("; ", result.ReuseShortfalls.Take(5))}");
        Check(result.GrowthProcessed.Count > 100 && result.GrowthProcessed.Average() < result.GrowthPrompt.Average(),
            $"a growing prefix processes fewer tokens than its prompt: {result.GrowthProcessed.Average():F1} of {result.GrowthPrompt.Average():F1} on {result.GrowthProcessed.Count} requests");
        Check(result.Mismatches.Count == 0, $"a request with the kept cache gives a fresh generator's text: {string.Join("; ", result.Mismatches)}");
        Check(result.CancelledAtChunk > 50 && result.MaxTokensAfterCancel == 0,
            $"a request cancelled at a chunk samples no token after it ({result.CancelledAtChunk} cancelled, at most {result.MaxTokensAfterCancel} after)");
        Check(result.CancelledBeforePass > 20 && result.PassCancelKeptCache == result.CancelledBeforePass,
            $"a request cancelled before its prompt's pass keeps the cache ({result.PassCancelKeptCache} of {result.CancelledBeforePass})");
        Check(result.AsyncStopsHeld, "StreamAsync's reader stopping leaves the generator holding its cache, its producer finished");
        Check(result.DeviceGrowth <= 64 * 1024, $"device memory in use does not grow from request 10 to {result.Requests}: {result.DeviceGrowth:N0} bytes");
        Check(result.ManagedGrowth <= 2 * 1024 * 1024, $"the managed heap does not grow from request 10 to {result.Requests}: {result.ManagedGrowth:N0} bytes");
    }

    // A tiny random-weight Qwen3-like decoder over the tokenizer's vocabulary (q/k norms, grouped keys, RoPE).
    internal static Sequential TinyFimDecoder(IPretrainedTokenizer tokenizer, Device device, int context = 256)
    {
        var spec = new DecoderSpec
        {
            Vocabulary = tokenizer.VocabularySize, Dim = 32, Layers = 2, Heads = 4, KvHeads = 2, HeadDim = 8, FfDim = 64, MaxPositions = context,
            QkNorm = true, Rope = new RopeSettings(10000f, null, false, null),
        };
        return spec.Build(null, new DecoderBuildOptions { Device = device, Seed = 3 });
    }

    /// <summary>How <see cref="FillInMiddleBench"/> runs (each from an IDRAK_BENCH_FIM_* variable in --bench-fim).</summary>
    internal sealed record FillInMiddleBenchOptions
    {
        public int Requests { get; init; } = 1000;      // IDRAK_BENCH_FIM_REQUESTS
        public int Context { get; init; } = 256;        // IDRAK_BENCH_FIM_CONTEXT: the window (GenerationOptions.NumCtx)
        public int Predict { get; init; } = 24;         // IDRAK_BENCH_FIM_PREDICT: tokens a request may sample
        public int Seed { get; init; } = 11;            // IDRAK_BENCH_FIM_SEED: the keystrokes' and the sampling's seed
        public int Verify { get; init; }                // IDRAK_BENCH_FIM_VERIFY: completed requests compared with a fresh generator
        public int SuffixCharacters { get; init; } = 160;
    }

    internal sealed class FillInMiddleBenchResult
    {
        public int Requests;
        public string Format = "";
        public TimeSpan ColdTotal, ColdPrompt;
        public int ColdPromptTokens;
        public readonly List<double> WarmPromptMs = [], WarmTokenMs = [], WarmTotalMs = [];
        public readonly List<int> GrowthProcessed = [], GrowthPrompt = [], GrowthCached = [];
        public readonly List<string> ReuseShortfalls = [], Mismatches = [];
        public int CancelledAtChunk, MaxTokensAfterCancel, TokensAfterCancel, CancelledBeforePass, PassCancelKeptCache, Completed, Trims;
        public readonly List<double> CrossThreadCancelMs = [];
        public bool AsyncStopsHeld;
        public long DeviceAt10, DeviceAtEnd, DevicePeak, ManagedAt10, ManagedAtEnd;

        public long DeviceGrowth => DeviceAtEnd - DeviceAt10;

        public long ManagedGrowth => ManagedAtEnd - ManagedAt10;

        private static double Median(List<double> values) => values.Count == 0 ? 0 : values.Order().ElementAt(values.Count / 2);

        public string Report()
        {
            var c = CultureInfo.InvariantCulture;
            double tokenMs = Median(WarmTokenMs);
            return string.Create(c, $"""
                fill-in-the-middle bench ({Format}): {Requests} requests, {Completed} completed, {CancelledAtChunk} cancelled at a chunk, {CancelledBeforePass} cancelled before the prompt's pass, {Trims} prefix trims
                  cold (first request): prompt {ColdPromptTokens} tokens in {ColdPrompt.TotalMilliseconds:F1} ms, request {ColdTotal.TotalMilliseconds:F1} ms
                  warm (median from request 10): prompt pass {Median(WarmPromptMs):F2} ms, a token {tokenMs:F3} ms, a request {Median(WarmTotalMs):F2} ms
                  reuse (prefix grown): prompt {(GrowthPrompt.Count == 0 ? 0 : GrowthPrompt.Average()):F1} tokens, read from the kept cache {(GrowthCached.Count == 0 ? 0 : GrowthCached.Average()):F1}, processed {(GrowthProcessed.Count == 0 ? 0 : GrowthProcessed.Average()):F1} ({(GrowthPrompt.Count == 0 ? 0 : 100.0 * GrowthCached.Sum() / Math.Max(1, GrowthPrompt.Sum())):F1}% reused) over {GrowthPrompt.Count} requests
                  cancel at a chunk: {TokensAfterCancel} tokens sampled after {CancelledAtChunk} cancels (at most {MaxTokensAfterCancel}); from another thread: median {Median(CrossThreadCancelMs):F2} ms, worst {(CrossThreadCancelMs.Count == 0 ? 0 : CrossThreadCancelMs.Max()):F2} ms to the stream's end ({(tokenMs > 0 ? Median(CrossThreadCancelMs) / tokenMs : 0):F2} tokens' time)
                  cancelled before the pass: {PassCancelKeptCache} of {CancelledBeforePass} kept the cache; StreamAsync stopped with the cache held: {AsyncStopsHeld}
                  memory: device in use {DeviceAt10:N0} B at request 10, {DeviceAtEnd:N0} B at {Requests} ({DeviceGrowth:+#,0;-#,0;0} B), peak {DevicePeak:N0} B; managed {ManagedAt10:N0} B, {ManagedAtEnd:N0} B ({ManagedGrowth:+#,0;-#,0;0} B)
                """);
        }
    }

    // The source the bench types from: code-like lines, typed a few characters per request after a fixed head.
    private const string FimSource = """
        def load(path):
            with open(path) as f:
                rows = [line.split(",") for line in f]
            return [r for r in rows if r]

        class Table:
            def __init__(self, rows):
                self.rows = rows
                self.index = {r[0]: r for r in rows}

            def find(self, key):
                return self.index.get(key)

        """;

    /// <summary>
    /// The editor's pattern on <paramref name="model"/>: per request the prefix grows by 1 to 4 characters typed from a code
    /// source (trimmed by half at a line start when the prompt outgrows the window), the suffix stays, and the request is
    /// cancelled by the next keystroke after 1 or 2 chunks, before its prompt's pass (one in ten), or left to finish (one in
    /// five). Measures and checks what <see cref="FillInMiddleBenchResult"/> holds.
    /// </summary>
    internal static FillInMiddleBenchResult FillInMiddleBench(Sequential model, IPretrainedTokenizer tokenizer, IFillInMiddleFormat format, Device device,
        FillInMiddleBenchOptions settings)
    {
        var result = new FillInMiddleBenchResult { Requests = settings.Requests, Format = format.Name };
        var random = new Random(settings.Seed);
        var generator = new TextGenerator(model, tokenizer, settings.Context);
        var options = new GenerationOptions { Seed = settings.Seed, NumPredict = settings.Predict, ChunkSize = 1, NumCtx = settings.Context, Temperature = 0.7f, TopK = 20 };
        string source = FimSource.ReplaceLineEndings("\n");
        string suffix = string.Concat(Enumerable.Repeat(source, 4))[..settings.SuffixCharacters];
        string head = "import os\n\n";
        string typed = "";
        int at = 0, verified = 0;
        IReadOnlyList<int> cachedModel = [];                       // what the kept cache holds at least (ids), as the bench follows it
        int budget = settings.Context - settings.Predict - 2;

        // Types `count` characters at the caret; a prompt past the budget loses the first half of what was typed (at a line
        // start), as an editor's context budget would cut it, so a reply never outgrows the window.
        IFillInMiddlePrompt Type(int count)
        {
            for (int k = 0; k < count; k++)
            {
                typed += source[at];
                at = (at + 1) % source.Length;
            }

            var made = format.Create(new FillInMiddleContext(tokenizer, head + typed, suffix) { Path = "table.py" });
            while (made.Ids.Count > budget && typed.Length > 0)
            {
                int cut = typed.IndexOf('\n', typed.Length / 2);
                typed = cut < 0 ? typed[(typed.Length / 2)..] : typed[(cut + 1)..];
                made = format.Create(new FillInMiddleContext(tokenizer, head + typed, suffix) { Path = "table.py" });
                result.Trims++;
            }

            return made;
        }

        for (int request = 0; request < settings.Requests; request++)
        {
            var prompt = Type(random.Next(1, 5));

            var ids = prompt.Ids;
            int shared = 0;
            while (shared < cachedModel.Count && shared < ids.Count && cachedModel[shared] == ids[shared])
            {
                shared++;
            }

            int expectedCached = Math.Min(shared, ids.Count - 1);
            var requestOptions = FillInMiddleFormats.Options(prompt, options);
            int kind = request < 10 ? 2 : random.Next(10);          // 0: cancelled before the pass; 1-2: completed; else cancelled after 1-2 chunks
            if (kind == 0)
            {
                using var cancelled = new CancellationTokenSource();
                cancelled.Cancel();
                bool threw = false;
                try
                {
                    foreach (var _ in generator.Stream(prompt.Text, requestOptions, cancelled.Token))
                    {
                    }
                }
                catch (OperationCanceledException)
                {
                    threw = true;
                }

                result.CancelledBeforePass++;
                if (threw && generator.HoldsCache)
                {
                    result.PassCancelKeptCache++;
                }

                cachedModel = expectedCached > 0 ? [.. ids.Take(expectedCached)] : [];
                continue;
            }

            int chunksBeforeCancel = kind <= 2 ? int.MaxValue : random.Next(1, 3);
            using var stop = new CancellationTokenSource();
            int chunks = 0, generatedAtCancel = -1;
            long start = Stopwatch.GetTimestamp();
            GenerationChunk? last = null;
            var text = new System.Text.StringBuilder();
            foreach (var chunk in generator.Stream(prompt.Text, requestOptions, stop.Token))
            {
                last = chunk;
                text.Append(chunk.Text);
                if (!chunk.Done && ++chunks == chunksBeforeCancel)
                {
                    generatedAtCancel = chunk.Generated;
                    stop.Cancel();                                  // the next keystroke
                }
            }

            var elapsed = Stopwatch.GetElapsedTime(start);
            var stats = last!.Stats!;
            if (stats.CachedTokens < expectedCached && result.ReuseShortfalls.Count < 20)
            {
                result.ReuseShortfalls.Add($"request {request}: {stats.CachedTokens} cached of {ids.Count}, the shared prefix {expectedCached}");
            }

            if (generatedAtCancel >= 0)
            {
                int after = stats.GeneratedTokens - generatedAtCancel;
                result.CancelledAtChunk++;
                result.TokensAfterCancel += after;
                result.MaxTokensAfterCancel = Math.Max(result.MaxTokensAfterCancel, after);
            }
            else
            {
                result.Completed++;
                if (verified < settings.Verify && request >= 10)
                {
                    verified++;
                    var fresh = new TextGenerator(model, tokenizer, settings.Context) { KeepCache = false };
                    string expected = fresh.Generate(prompt.Text, requestOptions).Text;
                    if (expected != text.ToString())
                    {
                        result.Mismatches.Add($"request {request}: '{text}' with the kept cache, '{expected}' fresh");
                    }
                }
            }

            if (request == 0)
            {
                (result.ColdTotal, result.ColdPrompt, result.ColdPromptTokens) = (elapsed, stats.PromptDuration, stats.PromptTokens);
            }
            else if (request >= 10)
            {
                result.WarmPromptMs.Add(stats.PromptDuration.TotalMilliseconds);
                result.WarmTotalMs.Add(elapsed.TotalMilliseconds);
                if (stats.GeneratedTokens > 1)
                {
                    result.WarmTokenMs.Add(stats.GenerationDuration.TotalMilliseconds / (stats.GeneratedTokens - 1));
                }

                if (shared > 0 && cachedModel.Count > 0)
                {
                    result.GrowthPrompt.Add(stats.PromptTokens);
                    result.GrowthCached.Add(stats.CachedTokens);
                    result.GrowthProcessed.Add(stats.PromptTokens - stats.CachedTokens);
                }
            }

            cachedModel = ids;                                       // the cache holds at least the whole prompt now
            if (request == 9)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
                result.DeviceAt10 = device.Backend.GetMemoryUsage().InUse;
                result.ManagedAt10 = GC.GetTotalMemory(forceFullCollection: true);
                device.Backend.ResetPeakMemoryUsage();
            }
        }

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        result.DeviceAtEnd = device.Backend.GetMemoryUsage().InUse;
        result.DevicePeak = device.Backend.GetMemoryUsage().Peak;
        result.ManagedAtEnd = GC.GetTotalMemory(forceFullCollection: true);

        // From another thread, as an editor's keystroke comes: the stream runs on a worker, the request is cancelled after
        // its first chunk, timed to the stream's end.
        for (int i = 0; i < 20; i++)
        {
            var prompt = Type(1);
            using var stop = new CancellationTokenSource();
            using var first = new ManualResetEventSlim();
            var worker = Task.Run(() =>
            {
                foreach (var chunk in generator.Stream(prompt.Text, options, stop.Token))
                {
                    first.Set();
                }

                first.Set();
            });
            first.Wait();
            long cancelAt = Stopwatch.GetTimestamp();
            stop.Cancel();
            worker.Wait();
            result.CrossThreadCancelMs.Add(Stopwatch.GetElapsedTime(cancelAt).TotalMilliseconds);
        }

        // StreamAsync: the reader stops at the first chunk (its token cancelled); the producer has finished and the cache is held.
        result.AsyncStopsHeld = Task.Run(async () =>
        {
            var prompt = Type(1);
            using var stop = new CancellationTokenSource();
            try
            {
                await foreach (var _ in generator.StreamAsync(prompt.Text, options with { Stop = [] }, stop.Token).ConfigureAwait(false))
                {
                    await stop.CancelAsync().ConfigureAwait(false);
                }
            }
            catch (OperationCanceledException)
            {
            }

            return generator.HoldsCache;
        }).GetAwaiter().GetResult();
        generator.ReleaseCache();
        return result;
    }

    // --bench-fim: the editor's pattern timed (FillInMiddleBench), on the tiny random decoder with each family's tokens, or
    // on a real model (IDRAK_BENCH_FIM_MODEL, a model folder: its tokenizer's format is detected).
    // Settings: IDRAK_BENCH_FIM_MODEL, IDRAK_BENCH_FIM_FORMATS (default qwen,mistral: a prefix-first and a suffix-first layout),
    // IDRAK_BENCH_FIM_REQUESTS (1000), IDRAK_BENCH_FIM_CONTEXT (256; a real model's: 2048 or its window),
    // IDRAK_BENCH_FIM_PREDICT (24), IDRAK_BENCH_FIM_SEED (11), IDRAK_BENCH_FIM_VERIFY (6); IDRAK_DEVICES (default the CPU).
    internal static int BenchFillInMiddle()
    {
        static int Setting(string name, int fallback) =>
            int.TryParse(Environment.GetEnvironmentVariable(name), NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) && value > 0 ? value : fallback;

        List<Device> devices = Environment.GetEnvironmentVariable("IDRAK_DEVICES") is { Length: > 0 } chosen
            ? [.. chosen.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(Device.Parse)]
            : [Device.Cpu];
        var settings = new FillInMiddleBenchOptions
        {
            Requests = Setting("IDRAK_BENCH_FIM_REQUESTS", 1000), Predict = Setting("IDRAK_BENCH_FIM_PREDICT", 24), Seed = Setting("IDRAK_BENCH_FIM_SEED", 11),
            Verify = Setting("IDRAK_BENCH_FIM_VERIFY", 6),
        };
        Idrak.Text.Plugins.TextPlugins.RegisterAll();
        foreach (var device in devices)
        {
            if (Environment.GetEnvironmentVariable("IDRAK_BENCH_FIM_MODEL") is { Length: > 0 } folder)
            {
                using var loaded = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
                var tokenizer = loaded.Tokenizer as IPretrainedTokenizer ?? throw new NotSupportedException($"{folder}: the model's tokenizer has no special tokens to look up.");
                var format = FillInMiddleFormats.For(tokenizer);
                var real = settings with { Context = Setting("IDRAK_BENCH_FIM_CONTEXT", Math.Min(2048, loaded.MaxPositions)), SuffixCharacters = 400 };
                Console.WriteLine($"{device}: {folder}, format {format.Name}");
                Console.WriteLine(FillInMiddleBench(loaded.Network, tokenizer, format, device, real).Report());
                continue;
            }

            foreach (string name in (Environment.GetEnvironmentVariable("IDRAK_BENCH_FIM_FORMATS") ?? "qwen,mistral").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            {
                string[] tokens = name switch
                {
                    "qwen" => QwenFimTokens, "deepseek" => DeepSeekFimTokens, "gemma" => GemmaFimTokens, "mistral" => MistralFimTokens,
                    _ => throw new NotSupportedException($"IDRAK_BENCH_FIM_FORMATS: no test tokenizer for '{name}' (qwen, deepseek, gemma, mistral; a real model: IDRAK_BENCH_FIM_MODEL)"),
                };
                var tokenizer = FamilyTokenizer(tokens);
                var tiny = settings with { Context = Setting("IDRAK_BENCH_FIM_CONTEXT", 256) };
                using var model = TinyFimDecoder(tokenizer, device, tiny.Context);
                Console.WriteLine($"{device}: tiny random decoder (2 layers, width 32, vocabulary {tokenizer.VocabularySize})");
                Console.WriteLine(FillInMiddleBench(model, tokenizer, FillInMiddleFormats.Get(name), device, tiny).Report());
            }
        }

        return 0;
    }
}
