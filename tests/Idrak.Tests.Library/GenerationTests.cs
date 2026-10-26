// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Datasets.Abstractions;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models.Abstractions;
using Idrak.Text.Generation;

internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] Generation =
    [
        ("generation: parser splits thinking, content and tool calls from any chunking", ParserSplitsOutput),
        ("generation: word tokenizer splits, maps unknown words and decodes piecewise", WordTokenizerBehaviour),
        ("generation: ChatML template renders system, tools, turns and think modes", TemplateRendering),
        ("generation: keep-alive durations and model host expiry", KeepAliveAndHost),
        ("generation: num_predict, stop sequences, cache/graph/recompute agree; without a slide a reply ends where the window does", GeneratorBehaviour),
        ("generation: a batch of prompts of different lengths keeps a bfloat16 cache (IRowStartsLayout) and gives each prompt its text alone, greedy, through prompts and steps; float32 and bfloat16 caches give the same texts with recorded steps (UseGraph) as without, through many steps, and the device records the step once", BFloat16RowsBatch),
        ("generation: a measured prompt piece is never fewer than PrefillMinChunk positions (256 by default) however little memory is free, never more than the prompt, and as many as the room holds between", PrefillPieceFloor),
        ("generation: chat end to end yields a final message and stats", ChatEndToEnd),
        ("generation: a kept KV cache reuses shared prompt prefixes without changing the output", PromptCacheReuse),
        ("generation: a windowed decoder generating far past its window with sliding windows' caches gives the texts of caches holding the whole reply (one row through recorded steps, a batch of two)", SlidingWindowGeneration),
        ("generation: ReuseDecoding keeps the decoding context, sampler and recorded step between generations (the same objects while the size and sampling match, new ones otherwise, no sampler without a seed) and every text is a fresh generator's; ReleaseCache frees them", DecodingReuse),
        ("generation: the prompt's pass (its rows, pieces and piece size) and each decoding chunk reach telemetry's Stages level from one prompt and from a batch (GenerateBatch), nothing waited for", GenerationStagesReported),
        ("generation: a long prompt's pass in pieces (PrefillChunk) gives one pass's last logits within float rounding, its cache length and its greedy tokens: one row, after a kept cache's prefix, with sliding windows' caches, and rows of different lengths padded on the left (row starts)", PrefillInPieces),
    ];

    // The stages a generation reports (InferenceStage at TelemetryLevel.Stages): the prompt's pass with its rows and the
    // pieces it went through (PrefillChunk), then the decoding chunks, every row's tokens counted; one prompt and a batch.
    internal static void GenerationStagesReported(Device device)
    {
        var (tiny, tokenizer) = TinyLanguageModel(device, context: 64);
        tiny.Dispose();
        var spec = new DecoderSpec
        {
            Vocabulary = tokenizer.VocabularySize, Dim = 16, Layers = 2, Heads = 4, KvHeads = 2, HeadDim = 6, FfDim = 24, MaxPositions = 64,
            Rope = new RopeSettings(500f, null, false, null),
        };
        using var decoder = spec.Build(null, new DecoderBuildOptions { Device = device, Seed = 3 });
        var generator = new TextGenerator(decoder, tokenizer, 64) { KeepCache = false };
        var options = new GenerationOptions { Seed = 0, NumPredict = 10, Temperature = 1f, TopK = 1, RepeatPenalty = 1f, RepeatStop = 0, ChunkSize = 4, NumCtx = 64, PrefillChunk = 3 };
        string[] prompts = generator.SupportsBatches ? ["the quick brown fox", "a short one"] : ["one two three", "four five six"];
        var hook = new StageList();
        using (Telemetry.Subscribe(hook))
        {
            generator.Generate(prompts[0], options);
            var single = hook.Take();
            int length = tokenizer.Encode(prompts[0]).Count;
            Check(single.Count(e => e.Stage == InferenceStageKind.Prefill) == 1
                  && single.Single(e => e.Stage == InferenceStageKind.Prefill) is { Rows: 1, Count: var fed, Pieces: var pieces, PieceSize: > 0 and <= 3 } && pieces == (length + 2) / 3 && fed == length,
                $"one prompt of {length} tokens in pieces of 3: {string.Join("; ", single)}");
            Check(single.Where(e => e.Stage == InferenceStageKind.DecodeChunk).Sum(e => e.Count) is > 0 and <= 10, $"one prompt's decoding chunks: {string.Join("; ", single)}");

            generator.GenerateBatch(prompts, options);
            var batch = hook.Take();
            int longest = prompts.Max(p => tokenizer.Encode(p).Count);
            var prefill = batch.Where(e => e.Stage == InferenceStageKind.Prefill).ToList();
            var chunks = batch.Where(e => e.Stage == InferenceStageKind.DecodeChunk).ToList();
            Check(prefill is [{ Rows: 2, PieceSize: > 0 and <= 3 } p] && p.Count == 2L * longest && p.Pieces == (longest + 2) / 3 && p.Duration > TimeSpan.Zero,
                $"the batch's prompt pass, 2 rows of {longest} positions in pieces of 3: {string.Join("; ", prefill)}");
            Check(chunks.Count >= 2 && chunks.All(c => c.Rows == 2 && c.Count > 0 && c.Count % 2 == 0) && chunks.Sum(c => c.Count) == 2L * (options.NumPredict - 1),
                $"the batch's decoding chunks, every row's tokens: {string.Join("; ", chunks)}");
        }

        generator.GenerateBatch(prompts, options);
        Check(hook.Take().Count == 0, "nothing reported once nobody listens");
    }

    // The stage events seen, taken in order.
    internal sealed class StageList : ITelemetryHook
    {
        private readonly List<InferenceStage> _seen = [];

        public TelemetryLevel Levels => TelemetryLevel.Stages;

        public void OnInferenceStage(in InferenceStage e)
        {
            lock (_seen)
            {
                _seen.Add(e);
            }
        }

        public List<InferenceStage> Take()
        {
            lock (_seen)
            {
                var taken = _seen.ToList();
                _seen.Clear();
                return taken;
            }
        }
    }

    // A long prompt's pass in pieces (PrefillChunk, ForwardPieces): each piece cached before the next attends to it, so
    // the last position's logits are one pass's within float rounding, the cache as long, and greedy generation gives the
    // same tokens, one row (fresh, after a kept cache's shared prefix, with sliding windows' caches) and a batch (rows of
    // different lengths padded on the left where the model attends from row starts, rows of one length elsewhere).
    internal static void PrefillInPieces(Device device)
    {
        var (tiny, tokenizer) = TinyLanguageModel(device, context: 64);
        using var _ = tiny;
        var plainSpec = new DecoderSpec
        {
            Vocabulary = tokenizer.VocabularySize, Dim = 16, Layers = 2, Heads = 4, KvHeads = 2, HeadDim = 6, FfDim = 24, MaxPositions = 64,
            QkNorm = true, Rope = new RopeSettings(500f, null, false, null),
        };
        using var decoder = plainSpec.Build(null, new DecoderBuildOptions { Device = device, Seed = 3 });
        using var windowed = (plainSpec with { SlidingWindow = 6, SlidingWindowLayers = [true, false] }).Build(null, new DecoderBuildOptions { Device = device, Seed = 8 });
        string[] texts = ["the quick brown fox jumps over the lazy dog", "a short one", "pack my box with five dozen jugs"];
        var greedy = new GenerationOptions { Seed = 0, NumPredict = 12, Temperature = 1f, TopK = 1, RepeatPenalty = 1f, ChunkSize = 4, NumCtx = 64 };
        int least = TextGenerator.MinSlidingSteps;
        try
        {
            TextGenerator.MinSlidingSteps = 4;                                 // the windowed layers' caches hold their window
            foreach (var (model, name) in new[] { (tiny, "encoder layers"), (decoder, "decoder"), (windowed, "windowed decoder") })
            {
                var generator = new TextGenerator(model, tokenizer, 64) { KeepCache = false };
                bool rowStarts = model != tiny && generator.SupportsBatches;
                Check(model != decoder || rowStarts, $"{name}: rows of different lengths decode together on {device}");

                // The last position's logits and the cache's length after the prompt's pass, one pass or in pieces.
                (float[] Logits, int Length) Pass(int[] rows, int piece)
                {
                    var ids = rows.Select(r => tokenizer.Encode(texts[r]).ToArray()).ToArray();
                    int length = ids.Max(x => x.Length);
                    var starts = ids.Select(x => length - x.Length).ToArray();
                    var input = new float[ids.Length * length];
                    for (int r = 0; r < ids.Length; r++)
                    {
                        for (int i = 0; i < ids[r].Length; i++)
                        {
                            input[r * length + starts[r] + i] = ids[r][i];
                        }
                    }

                    using var decoding = new DecodingContext(device, ids.Length, length + 4) { LastPositionOnly = true };
                    if (starts.Any(s => s > 0))
                    {
                        decoding.SetRowStarts(starts);
                    }

                    using var noGrad = Autograd.NoGrad();
                    using var scope = new TensorScope();
                    var logits = piece >= length ? model.ForwardCached(Tensor.From(input, [ids.Length, length], device), decoding)
                        : generator.ForwardPieces(input, ids.Length, length, decoding, piece);
                    return (logits.ToArray(), decoding.Length);
                }

                int[][] sets = rowStarts ? [[0], [0, 1, 2]] : [[0]];
                foreach (int[] rows in sets)
                {
                    var (one, length) = Pass(rows, int.MaxValue);
                    foreach (int piece in new[] { 5, 4 })
                    {
                        var (pieced, piecedLength) = Pass(rows, piece);
                        float worst = MaxDifference(one, pieced), peak = one.Max(MathF.Abs);
                        Check(piecedLength == length && worst <= 1e-4f * Math.Max(1f, peak),
                            $"{name}, {rows.Length} row(s) in pieces of {piece}: the last logits {worst} from one pass's (largest {peak}), cache length {piecedLength} for {length}");
                    }
                }

                var pieces = greedy with { PrefillChunk = 5 };
                string expected = generator.Generate(texts[0], greedy).Text, actual = generator.Generate(texts[0], pieces).Text;
                Check(actual == expected, $"{name}: '{actual}' with the prompt in pieces, '{expected}' in one pass");

                // A kept cache: the second prompt shares the first's start, the rest fed in pieces after it.
                var keeping = new TextGenerator(model, tokenizer, 64);
                keeping.Generate(texts[1], pieces);
                string extended = texts[1] + " and then " + texts[2];
                expected = generator.Generate(extended, greedy).Text;
                actual = keeping.Generate(extended, pieces).Text;
                Check(actual == expected, $"{name}: '{actual}' after a kept cache's prefix with the rest in pieces, '{expected}' in one pass");
                keeping.ReleaseCache();

                string[] prompts = rowStarts ? texts : ["one two three", "four five six"];
                var batch = generator.GenerateBatch(prompts, greedy).Select(r => r.Text).ToArray();
                var batchPieces = generator.GenerateBatch(prompts, greedy with { PrefillChunk = 3 }).Select(r => r.Text).ToArray();
                Check(batchPieces.SequenceEqual(batch), $"{name}, a batch: '{string.Join("' '", batchPieces)}' in pieces, '{string.Join("' '", batch)}' in one pass");
            }
        }
        finally
        {
            TextGenerator.MinSlidingSteps = least;
        }
    }

    // A windowed decoder generating far past its window: the windowed layers' caches hold the window (moved as they fill,
    // also between recorded steps), every text as with caches of the whole reply, one row and a batch of two.
    internal static void SlidingWindowGeneration(Device device)
    {
        var (_, tokenizer) = TinyLanguageModel(device);
        var spec = new DecoderSpec
        {
            Vocabulary = tokenizer.VocabularySize, Dim = 16, Layers = 2, Heads = 4, KvHeads = 2, HeadDim = 6, FfDim = 24, MaxPositions = 96,
            Rope = new RopeSettings(500f, null, false, null), SlidingWindow = 6, SlidingWindowLayers = [true, false],
        };
        using var model = spec.Build(null, new DecoderBuildOptions { Device = device, Seed = 8 });
        var options = new GenerationOptions { Seed = 4, NumPredict = 60, Temperature = 0.9f, TopK = 8, RepeatPenalty = 1f, ChunkSize = 7, NumCtx = 96 };
        string[] prompts = ["once upon a time", "the river ran"];
        int least = TextGenerator.MinSlidingSteps;
        try
        {
            (string[] Single, string[] Batch) Generate(bool slide)
            {
                TextGenerator.SlideWindows = slide;
                var generator = new TextGenerator(model, tokenizer, 96) { KeepCache = false };
                var single = prompts.Select(p => generator.Generate(p, options).Text).ToArray();
                var batch = generator.GenerateBatch(["one two three", "four five six"], options).Select(r => r.Text).ToArray();
                return (single, batch);
            }

            TextGenerator.MinSlidingSteps = 4;
            var (slidSingle, slidBatch) = Generate(slide: true);
            var (single, batch) = Generate(slide: false);
            Check(slidSingle.SequenceEqual(single), $"one row: '{string.Join("' '", slidSingle)}' with sliding windows, '{string.Join("' '", single)}' without");
            Check(slidBatch.SequenceEqual(batch), $"a batch: '{string.Join("' '", slidBatch)}' with sliding windows, '{string.Join("' '", batch)}' without");
        }
        finally
        {
            TextGenerator.SlideWindows = true;
            TextGenerator.MinSlidingSteps = least;
        }
    }

    internal static void PrefillPieceFloor(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                              // arithmetic only: once
        }

        Check(GenerationOptions.DefaultPrefillMinChunk == 256 && new GenerationOptions().PrefillMinChunk == 0, "the floor's default and the option's");
        Check(TextGenerator.PieceOf(0, 1000, 256, 3000) == 256 && TextGenerator.PieceOf(-5_000_000, 1000, 256, 3000) == 256, "no room (or less than none): the floor");
        Check(TextGenerator.PieceOf(200_000, 1000, 256, 3000) == 256, "room for 200 positions: the floor");
        Check(TextGenerator.PieceOf(1_000_000, 1000, 256, 3000) == 1000, "room for 1000 positions: 1000");
        Check(TextGenerator.PieceOf(long.MaxValue, 1000, 256, 3000) == 3000, "room for more than the prompt: the prompt");
        Check(TextGenerator.PieceOf(0, 1000, 256, 100) == 100, "a prompt shorter than the floor: one pass");
        Check(TextGenerator.PieceOf(0, 1000, 0, 3000) == 1, "a floor of 0 counts as 1");
        ThrowsNaming<ArgumentOutOfRangeException>(() => _ = new GenerationOptions { PrefillMinChunk = -1 }, "a negative floor", "PrefillMinChunk");
    }

    internal static void BFloat16RowsBatch(Device device)
    {
        var (tiny, tokenizer) = TinyLanguageModel(device, context: 64);
        tiny.Dispose();
        var spec = new DecoderSpec
        {
            Vocabulary = tokenizer.VocabularySize, Dim = 16, Layers = 2, Heads = 4, KvHeads = 2, HeadDim = 8, FfDim = 24, MaxPositions = 64,
            QkNorm = true, Rope = new RopeSettings(500f, null, false, null),
        };
        using var model = spec.Build(null, new DecoderBuildOptions { Device = device, Seed = 5 });
        var layout = KeyValueLayouts.For(KeyValueFormat.BFloat16);
        var generator = new TextGenerator(model, tokenizer, 64) { CacheFormat = KeyValueFormat.BFloat16, KeepCache = false };
        Check(layout is IRowStartsLayout && generator.SupportsBatches, $"bfloat16 attends from per-row starts ({layout is IRowStartsLayout}) and the model batches rows ({generator.SupportsBatches})");
        var options = new GenerationOptions { Temperature = 0f, TopK = 1, RepeatPenalty = 1f, NumPredict = 14, ChunkSize = 3 };
        string[] prompts = ["abc", "hello there", "a much longer prompt of words"];
        var alone = prompts.Select(p => generator.Generate(p, options).Text).ToArray();
        var together = generator.GenerateBatch(prompts, options).Select(r => r.Text).ToArray();
        Check(together.SequenceEqual(alone), $"a padded batch with a bfloat16 cache: '{string.Join("' '", together)}', alone: '{string.Join("' '", alone)}'");

        // Recorded steps over rows with starts (their positions advance on the device): the texts of steps run as they are,
        // through many steps (the cache filling), float32 and bfloat16 caches; the device records the step once.
        var many = options with { NumPredict = 40, RepeatStop = 0, ChunkSize = 5 };
        var hook = new StageList();
        foreach (var format in new[] { KeyValueFormat.Float32, KeyValueFormat.BFloat16 })
        {
            var rows = new TextGenerator(model, tokenizer, 64) { CacheFormat = format, KeepCache = false };
            using (var context = new DecodingContext(device, prompts.Length, 64, KeyValueLayouts.For(format)))
            {
                context.SetRowStarts([3, 1, 0]);
                Check(context.Recordable, $"{format}: rows with starts are recordable");
                context.SetRotaryOffsets([2, 2, 2], 0);
                Check(!context.Recordable, $"{format}: rows with starts and rotary offsets are not (their rotary positions come from the host)");
            }

            string[] plain, recorded;
            int plainCaptures, recordedCaptures;
            using (Telemetry.Subscribe(hook))
            {
                hook.Take();
                plain = [.. rows.GenerateBatch(prompts, many with { UseGraph = false }).Select(r => r.Text)];
                plainCaptures = hook.Take().Count(e => e.Stage == InferenceStageKind.GraphCapture);
                recorded = [.. rows.GenerateBatch(prompts, many with { UseGraph = true }).Select(r => r.Text)];
                recordedCaptures = hook.Take().Count(e => e.Stage == InferenceStageKind.GraphCapture);
            }

            // Alone, a prompt has more room (the batch's context holds the longest prompt), so its text begins with the batch's.
            var aloneMany = prompts.Select(p => rows.Generate(p, many with { UseGraph = false }).Text).ToArray();
            Check(recorded.SequenceEqual(plain) && plain.Zip(aloneMany).All(x => x.Second.StartsWith(x.First, StringComparison.Ordinal)),
                $"{format}, a padded batch of up to 40 steps: '{string.Join("' '", recorded)}' recorded, '{string.Join("' '", plain)}' step by step, '{string.Join("' '", aloneMany)}' alone");
            int expected = device.Backend.SupportsGraphs ? 1 : 0;
            Check(plainCaptures == 0 && recordedCaptures == expected,
                $"{format}: the padded batch's step recorded {recordedCaptures} time(s) with UseGraph ({expected} expected on {device}), {plainCaptures} without");
        }
    }

    internal static void DecodingReuse(Device device)
    {
        var (tiny, tokenizer) = TinyLanguageModel(device, context: 64);
        var spec = new DecoderSpec
        {
            Vocabulary = tokenizer.VocabularySize, Dim = 16, Layers = 2, Heads = 4, KvHeads = 2, HeadDim = 6, FfDim = 24, MaxPositions = 64,
            QkNorm = true, Rope = new RopeSettings(500f, null, false, null), SlidingWindow = 8, SlidingWindowLayers = [true, false],
        };
        using var decoder = spec.Build(null, new DecoderBuildOptions { Device = device, Seed = 3 });
        using var _ = tiny;
        var greedy = new GenerationOptions { Seed = 0, NumPredict = 14, Temperature = 1f, TopK = 1, RepeatPenalty = 1f, ChunkSize = 4, NumCtx = 64 };
        var sampled = greedy with { Seed = 5, Temperature = 0.8f, TopK = 5, RepeatPenalty = 1.1f };
        foreach (var model in new[] { tiny, decoder })
        {
            var reusing = new TextGenerator(model, tokenizer, 64) { KeepCache = false, ReuseDecoding = true };
            DecodingContext? previous = null;
            Idrak.Abstraction.Generation.ITokenSampler? previousSampler = null;
            // Prompts of one length (the context and sampler kept), a longer one (a new context, the sampler kept), other
            // options, no seed (its sampler not kept), then a seed again.
            var runs = new (string Prompt, GenerationOptions Options, bool SameContext, bool SameSampler)[]
            {
                ("the quick brown fox", greedy, false, false),
                ("the lazy brown dogs", greedy, true, true),
                ("the lazy brown dogs", greedy with { Stop = ["zz"] }, true, true),
                ("a much longer prompt than the others", greedy, false, true),
                ("the quick brown fox", sampled, false, false),
                ("the quick brown cow", sampled, true, true),
                ("the quick brown cat", sampled with { Seed = null }, true, false),
                ("the quick brown cat", greedy, true, false),
            };
            foreach (var (prompt, options, sameContext, sameSampler) in runs)
            {
                string expected = new TextGenerator(model, tokenizer, 64) { KeepCache = false }.Generate(prompt, options).Text;
                string actual = reusing.Generate(prompt, options).Text;
                if (options.Seed is not null)
                {
                    Check(actual == expected, $"{model.Name}: '{prompt}' → '{actual}' reusing the decoding, '{expected}' fresh");
                }

                var (context, sampler, _) = reusing.HeldDecoding;
                Check(context is not null && ReferenceEquals(context, previous) == sameContext,
                    $"{model.Name}: '{prompt}': the context {(sameContext ? "kept" : "made anew")}");
                Check(options.Seed is null ? sampler is null : sampler is not null && ReferenceEquals(sampler, previousSampler) == sameSampler,
                    $"{model.Name}: '{prompt}': the sampler {(options.Seed is null ? "not kept (no seed)" : sameSampler ? "kept" : "made anew")}");
                (previous, previousSampler) = (context, sampler);
            }

            reusing.ReleaseCache();
            var (released, releasedSampler, step) = reusing.HeldDecoding;
            Check(released is null && releasedSampler is null && step is null, $"{model.Name}: ReleaseCache frees them");
        }
    }

    internal static void WordTokenizerBehaviour(Device device)
    {
        _ = device;
        var tokenizer = WordTokenizer.FromTexts(["the lions beat the eagles 3 to 1 .", "rain is likely , bring a coat !"], ["<pad>", "<sum>", "<end>"]);
        Check(tokenizer.Vocabulary[0] == "<pad>" && tokenizer.Vocabulary[2] == "<end>" && tokenizer.Vocabulary[3] == "the", "specials first, then by frequency");
        var ids = tokenizer.Encode("The Lions beat the wolves 3 to 1. <sum> <end>");
        Check(ids.Count == 11 && ids[4] == tokenizer["<unk>"] && ids[8] == tokenizer["."] && ids[9] == 1 && ids[10] == 2, $"encode {string.Join(",", ids)}");
        string whole = tokenizer.Decode(ids);
        Check(whole == " the lions beat the <unk> 3 to 1. <sum> <end>", $"decode '{whole}'");
        string pieces = string.Concat(Enumerable.Range(0, ids.Count).Select(i => tokenizer.Decode([ids[i]])));
        Check(pieces == whole, "piecewise decoding equals whole decoding");
    }

    internal static void ParserSplitsOutput(Device device)
    {
        _ = device;
        const string Raw = "<think>\nThe user wants the latest version.\n</think>\n\nLet me check.<tool_call>\n{\"name\": \"web_fetch\", \"arguments\": {\"url\": \"https://ollama.com\"}}\n</tool_call>";
        var template = new ChatMLTemplate();
        foreach (int size in new[] { 1, 2, 3, 7, 1000 })
        {
            var parser = new ChatOutputParser(template);
            string content = "", thinking = "";
            var calls = new List<ToolCall>();
            for (int i = 0; i < Raw.Length; i += size)
            {
                var d = parser.Feed(Raw.Substring(i, Math.Min(size, Raw.Length - i)));
                content += d.Content; thinking += d.Thinking; calls.AddRange(d.ToolCalls);
            }

            var f = parser.Finish();
            content += f.Content; thinking += f.Thinking; calls.AddRange(f.ToolCalls);
            Check(thinking.Trim() == "The user wants the latest version.", $"chunk {size}: thinking '{thinking}'");
            Check(content == "Let me check.", $"chunk {size}: content '{content}'");
            Check(calls.Count == 1 && calls[0].Name == "web_fetch" && calls[0].Arguments["url"]!.GetValue<string>() == "https://ollama.com",
                $"chunk {size}: tool call");
        }

        var plain = new ChatOutputParser(template);
        var a = plain.Feed("Hello <tool_call>not json</tool_call> bye");
        var b = plain.Finish();
        Check(a.Content + b.Content == "Hello <tool_call>not json</tool_call> bye" && a.ToolCalls.Count + b.ToolCalls.Count == 0, "invalid tool JSON stays text");

        var merged = new ChatOutputParser(template, separateThinking: false);
        var m = merged.Feed("<think>x</think>y");
        var mf = merged.Finish();
        Check(m.Thinking + mf.Thinking == "" && (m.Content + mf.Content) == "xy", "think=false folds reasoning into content");

        var open = new ChatOutputParser(template);
        var o = open.Feed("<think>unfinished");
        var of = open.Finish();
        Check(o.Thinking + of.Thinking == "unfinished", "unterminated reasoning is flushed as reasoning");
    }

    internal static void TemplateRendering(Device device)
    {
        _ = device;
        var template = new ChatMLTemplate();
        var tool = new ToolDefinition("web_fetch", "Fetch a page from the allowlisted search results.",
            JsonNode.Parse("""{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}"""));
        ChatMessage[] messages =
        [
            new("system", "You are a helpful assistant."),
            new("user", "What is the latest Ollama version?"),
            new("assistant", "", ToolCalls: [new ToolCall("web_fetch", new JsonObject { ["url"] = "https://ollama.com" })]),
            new("tool", "v0.12.3", ToolName: "web_fetch"),
        ];
        string prompt = template.Render(messages, [tool], think: null);
        Check(prompt.StartsWith("<|im_start|>system\nYou are a helpful assistant.\n\n# Tools"), "system turn with tools first");
        Check(prompt.Contains("\"name\":\"web_fetch\"") && prompt.Contains("<tools>"), "tool signature listed");
        Check(prompt.Contains("<|im_start|>user\nWhat is the latest Ollama version?<|im_end|>"), "user turn");
        Check(prompt.Contains("<tool_call>{\"name\":\"web_fetch\",\"arguments\":{\"url\":\"https://ollama.com\"}}</tool_call>"), "assistant tool call");
        Check(prompt.Contains("<tool_response>\nv0.12.3\n</tool_response>"), "tool result");
        Check(prompt.EndsWith("<|im_start|>assistant\n"), "open assistant turn");
        Check(template.Render(messages, [], think: false).EndsWith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "think=false closes an empty think block");
    }

    internal sealed class ManualClock : TimeProvider
    {
        public DateTimeOffset Now { get; set; } = new(2026, 1, 1, 0, 0, 0, TimeSpan.Zero);
        public override DateTimeOffset GetUtcNow() => Now;
    }

    internal sealed class Dummy : IDisposable
    {
        public bool Disposed { get; private set; }
        public void Dispose() => Disposed = true;
    }

    internal static void KeepAliveAndHost(Device device)
    {
        _ = device;
        Check(KeepAlive.Parse("30m") == TimeSpan.FromMinutes(30), "30m");
        Check(KeepAlive.Parse("1h30m") == TimeSpan.FromMinutes(90), "1h30m");
        Check(KeepAlive.Parse("500ms") == TimeSpan.FromMilliseconds(500), "500ms");
        Check(KeepAlive.Parse("0") == TimeSpan.Zero, "0");
        Check(KeepAlive.Parse("-1") is null && KeepAlive.Parse("-1m") is null, "negative keeps forever");
        Check(KeepAlive.Parse(JsonDocument.Parse("300").RootElement, null) == TimeSpan.FromMinutes(5), "number = seconds");
        Check(KeepAlive.Parse((JsonElement?)null, TimeSpan.FromMinutes(5)) == TimeSpan.FromMinutes(5), "missing uses the default");
        try { KeepAlive.Parse("soon"); Check(false, "invalid duration must throw"); } catch (FormatException) { }

        var clock = new ManualClock();
        var created = new List<Dummy>();
        using var host = new ModelHost<Dummy>(_ => { var d = new Dummy(); created.Add(d); return d; }, TimeSpan.FromMinutes(5), clock);
        using (var lease = host.Acquire("m", TimeSpan.FromMinutes(30)))
        {
            Check(host.Loaded.Single().ActiveUsers == 1 && host.Loaded.Single().ExpiresAt is null, "in use: no expiry");
        }

        Check(host.Loaded.Single().ExpiresAt == clock.Now.AddMinutes(30), "expiry = release time + keep-alive");
        clock.Now = clock.Now.AddMinutes(29);
        host.Sweep();
        Check(host.Loaded.Count == 1, "still loaded before expiry");
        using (host.Acquire("m")) { }
        Check(created.Count == 1, "reused, not reloaded");
        clock.Now = clock.Now.AddMinutes(6);
        host.Sweep();
        Check(host.Loaded.Count == 0 && created[0].Disposed, "unloaded and disposed after the default 5 minutes");

        using (host.Acquire("now", TimeSpan.Zero)) { }
        Check(host.Loaded.Count == 0 && created[1].Disposed, "keep-alive 0 unloads immediately");
        using (host.Acquire("forever", null)) { }
        clock.Now = clock.Now.AddYears(1);
        host.Sweep();
        Check(host.Loaded.Single().Name == "forever", "negative keep-alive never expires");
    }

    internal static (Sequential Model, CharTokenizer Tokenizer) TinyLanguageModel(Device device, int context = 32)
    {
        var tokenizer = new CharTokenizer("abcdefghijklmnopqrstuvwxyz .,<>|/_{}\":\n");
        var r = new Random(4);
        var model = new Sequential
        {
            new Embedding(tokenizer.VocabularySize, 16, device, r),
            new PositionalEncoding(context, 16, device),
            new TransformerEncoderLayer(16, 2, dropout: 0f, causal: true, device: device, random: r),
            new LayerNorm(16, device: device),
            new Linear(16, tokenizer.VocabularySize, device: device, random: r),
        };
        return (model, tokenizer);
    }

    internal static void PromptCacheReuse(Device device)
    {
        var (tiny, tokenizer) = TinyLanguageModel(device, context: 64);
        var spec = new DecoderSpec
        {
            Vocabulary = tokenizer.VocabularySize, Dim = 16, Layers = 2, Heads = 4, KvHeads = 2, HeadDim = 6, FfDim = 24, MaxPositions = 64,
            QkNorm = true, Rope = new RopeSettings(500f, null, false, null),
        };
        using var decoder = spec.Build(null, new DecoderBuildOptions { Device = device, Seed = 3 });
        using var _ = tiny;
        foreach (var model in new[] { tiny, decoder })
        {
            var options = new GenerationOptions { Seed = 5, NumPredict = 12, Temperature = 0.8f, TopK = 5, RepeatPenalty = 1.1f, ChunkSize = 3, NumCtx = 64 };
            string[] prompts = ["the quick brown fox", "the quick brown fox", "the quick brown fox jumps over", "the lazy dog", "the quick brown fox jumps over the"];
            var reused = new TextGenerator(model, tokenizer, 64);
            foreach (string prompt in prompts)
            {
                var fresh = new TextGenerator(model, tokenizer, 64) { KeepCache = false };
                string expected = fresh.Generate(prompt, options).Text;
                string actual = reused.Generate(prompt, options).Text;
                Check(actual == expected, $"{model.Name}: '{prompt}' → '{actual}' with the kept cache, '{expected}' without");
            }

            reused.ReleaseCache();
        }
    }

    internal static void GeneratorBehaviour(Device device)
    {
        var (model, tokenizer) = TinyLanguageModel(device);
        using var _ = model;
        var generator = new TextGenerator(model, tokenizer, contextLength: 32);
        var baseOptions = new GenerationOptions { Seed = 11, NumPredict = 24, Temperature = 1f, TopK = 0, TopP = 0.95f, RepeatPenalty = 1.2f, ChunkSize = 5 };

        var (text, reason, stats) = generator.Generate("hello", baseOptions);
        Check(reason == "length" && stats.GeneratedTokens == 24 && text.Length == 24, $"num_predict: {reason}, {stats.GeneratedTokens} tokens");
        Check(stats.PromptTokens == 5, "prompt tokens counted");

        var noGraph = generator.Generate("hello", baseOptions with { UseGraph = false }).Text;
        var recompute = generator.Generate("hello", baseOptions with { UseCache = false }).Text;
        Check(noGraph == text, "graph replay gives the same text");
        Check(recompute == text, $"full recompute gives the same text ('{recompute}' vs '{text}')");

        string stop = text.Substring(8, 3);
        int first = text.IndexOf(stop, StringComparison.Ordinal);
        var stopped = generator.Generate("hello", baseOptions with { Stop = [stop] });
        Check(stopped.DoneReason == "stop" && stopped.Text == text[..first], $"stop sequence ends the text before it ('{stopped.Text}')");

        var streamed = string.Concat(generator.Stream("hello", baseOptions with { Stop = [stop], ChunkSize = 1 }).Select(c => c.Text));
        Check(streamed == text[..first], "streaming never leaks part of a stop sequence");

        var small = generator.Generate("abcdefghij", baseOptions with { NumCtx = 8, NumPredict = 30 });
        Check(small.Stats.PromptTokens == 7 && small.Text.Length == 30 && small.Stats.ContextResets > 0, "num_ctx truncates the prompt and slides the window");
        foreach (bool cache in new[] { true, false })
        {
            var ended = generator.Generate("abcdefghij", baseOptions with { NumCtx = 8, NumPredict = 30, SlideContext = false, UseCache = cache });
            var room = generator.Generate("abc", baseOptions with { NumCtx = 8, NumPredict = 30, SlideContext = false, UseCache = cache });
            Check(ended.Stats.PromptTokens == 7 && ended.Stats.GeneratedTokens == 1 && ended.DoneReason == "length" && ended.Stats.ContextResets == 0
                  && room.Stats.PromptTokens == 3 && room.Stats.GeneratedTokens == 5 && room.DoneReason == "length" && room.Stats.ContextResets == 0,
                $"no slide{(cache ? "" : ", no cache")}: the reply ends where the window does ({ended.Stats.GeneratedTokens} and {room.Stats.GeneratedTokens} tokens, {ended.DoneReason} and {room.DoneReason})");
        }
    }

    internal static void ChatEndToEnd(Device device)
    {
        var (model, tokenizer) = TinyLanguageModel(device, context: 512);
        using var _ = model;
        var chat = new ChatGenerator(new TextGenerator(model, tokenizer, 512));
        var request = new ChatRequest([new ChatMessage("system", "be brief."), new ChatMessage("user", "hi")], Think: true,
            Options: new GenerationOptions { Seed = 3, NumPredict = 12, NumCtx = 4096 });
        var chunks = chat.Stream(request).ToList();
        var last = chunks[^1];
        Check(last.Done && last.Message is { Role: "assistant" } && last.Stats is { PromptTokens: > 0 }, "final chunk carries the message and stats");
        Check(chat.RenderPrompt(request).Contains("<|im_start|>user\nhi<|im_end|>"), "prompt rendered with the template");
    }
}
