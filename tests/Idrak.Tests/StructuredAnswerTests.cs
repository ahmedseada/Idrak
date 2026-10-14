// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Diagnostics;
using Idrak.Data.Abstractions;
using Idrak.Generation;
using Idrak.Models;
using Idrak.Nlp.Abstractions;
using Idrak.Nlp.Generation;

// A model's JSON answers read leniently (JsonRepairs), the text of an answer a metric compares (AnswerTexts), the model
// host's limit on loaded models, and the image reader (ImageReader over ModelImages) on the tiny Gemma 3 fixture.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] StructuredAnswerGroup =
    [
        ("json repair: valid JSON reads unchanged and is not marked repaired; text and a code fence around it are dropped; text without JSON has none", JsonRepairValid),
        ("json repair: an answer cut at its token limit, trailing and missing commas, single and curly quotes, bare keys, Python literals, quotes inside values and repeated keys read as json_repair reads them", JsonRepairBroken),
        ("json repair: the registry has the library's lenient repair; an app's repair shadows it and is removed again; an unknown name is not registered", JsonRepairRegistry),
        ("answer texts: raw keeps the answer, values takes its JSON's values in document order (repaired JSON too; an answer without JSON as it is); Normalize composes and folds whitespace", AnswerTextModes),
        ("model host: MaxLoaded unloads the longest unused idle model before loading another, and never one in use", ModelHostMaxLoaded),
        ("image reader: a vision checkpoint reads a page greedily (the same answer twice), streams it as Read returns it, prepares the page with its transforms or the read's own; no attention cache held between reads unless KeepCache; a sampled read with a seed repeats; pages with one instruction read as one batch, each answer its own read's", ImageReaderReads),
        ("image reader: pages without room together (a memory limit between one page's peak and four's) are read in halves until they fit, each reading saying how many it was read with, the answers the pages' own; later batches start where they fit; BatchFit.Fail raises; MaxBatch caps the batch", ImageReaderBatchFits),
        ("image reader: suggested settings (SettingAdvisors, TelemetryLevel.Suggestions): loading suggests bfloat16 weights for a bfloat16 checkpoint read as float32; the library's image reader advisor suggests bfloat16 encoder and prompt exactly where the device has matrix units, and after a measured read (the first warm one) the pages that fit together; each published once, printed and written as JSON; an app's advisor adds its own and is removed; no read measured when Suggest is false", ImageReaderSuggestions),
        ("generation: the repetition stop finds an answer ending in one run of tokens repeated N times (runs from the minimum to 64 tokens; shorter runs such as a row of dots are text); a read says why it ended", RepetitionStop),
        ("image reader: telemetry: a read publishes its stages at the Inference level (the page decoded and transformed, the encoder, each cache allocated, the prompt's pass, the decoding chunks) with their sizes; the console line and the JSON Lines record", ImageReaderTelemetry),
        ("image reader: a stopped read ends before its first token (the prompt's pass checks between layers; the step is withdrawn, the context's length unchanged) and the next read is unaffected", ImageReaderStops),
    ];

    private static JsonNode? Repaired(string text, bool expectRepaired = true)
    {
        var result = JsonRepairs.Parse(text);
        Check(result is not null, $"no JSON found in {text}");
        Check(result!.Repaired == expectRepaired, $"repaired {result.Repaired} for {text}");
        return result.Json;
    }

    private static void JsonRepairValid(Device device)
    {
        if (!FirstRun(nameof(JsonRepairValid)))
        {
            return;
        }

        var json = Repaired("""{"name": "محمد", "items": [1, 2.5, true, null]}""", expectRepaired: false);
        Check(json!["name"]!.GetValue<string>() == "محمد" && json["items"]!.AsArray().Count == 4, json.ToJsonString());
        json = Repaired("Here is the result:\n```json\n{\"a\": 1}\n```\nDone.", expectRepaired: false);
        Check(json!["a"]!.GetValue<int>() == 1, "fenced");
        json = Repaired("The answer is {\"a\": [1, 2]} as asked.");
        Check(json!["a"]!.AsArray().Count == 2, "text around the JSON");
        Check(JsonRepairs.Parse("no JSON here") is null, "text without JSON");
    }

    private static void JsonRepairBroken(Device device)
    {
        if (!FirstRun(nameof(JsonRepairBroken)))
        {
            return;
        }

        void Same(string broken, string expected)
        {
            var json = Repaired(broken);
            Check(JsonNode.DeepEquals(json, JsonNode.Parse(expected)), $"{broken} read as {json?.ToJsonString()}, expected {expected}");
        }

        Same("""{"a": "cut here""", """{"a": "cut here"}""");                                 // the token limit in a string
        Same("""{"a": [1, 2, {"b": "c""", """{"a": [1, 2, {"b": "c"}]}""");                   // and in nested brackets
        Same("""{"a": 1,}""", """{"a": 1}""");                                                 // a trailing comma
        Same("""[1, 2, ]""", """[1, 2]""");
        Same("{\"a\": 1\n\"b\": 2}", """{"a": 1, "b": 2}""");                                 // a missing comma
        Same("""{'a': 'b'}""", """{"a": "b"}""");                                             // single quotes
        Same("""{“a”: “ب”}""", """{"a": "ب"}""");                                             // curly quotes
        Same("""{a: 1, b c: "x"}""", """{"a": 1, "b c": "x"}""");                             // bare keys
        Same("""{"a": True, "b": False, "c": None}""", """{"a": true, "b": false, "c": null}"""); // Python's literals
        Same("""{"a": "he said "yes" today", "b": 1}""", """{"a": "he said \"yes\" today", "b": 1}""");   // quotes inside a value
        Same("{\"a\": \"line one\nline two\"}", """{"a": "line one\nline two"}""");                // a raw line break
        Same("""{"a": }""", """{"a": ""}""");                                                  // a key without its value
        Same("""{"output": "a", "rules": [{"k": 1, "k": 2}], "output": "c"}""", """{"output": "c", "rules": [{"k": 2}]}""");   // repeated keys: the last value, in the first one's place
    }

    private sealed class EmptyRepair : IJsonRepair
    {
        public string Name => JsonRepairs.Lenient;

        public string Summary => "always an empty object";

        public JsonRepairResult? Repair(string text) => new(new JsonObject(), Repaired: true);
    }

    private static void JsonRepairRegistry(Device device)
    {
        if (!FirstRun(nameof(JsonRepairRegistry)))
        {
            return;
        }

        Check(JsonRepairs.Names.Contains(JsonRepairs.Lenient) && JsonRepairs.Default(JsonRepairs.Lenient) is not null, "the library's repair is registered");
        JsonRepairs.Register(new EmptyRepair());
        try
        {
            Check(JsonRepairs.Parse("""{"a": 1}""")!.Json!.AsObject().Count == 0, "the app's repair answers under the library's name");
            Check(JsonRepairs.Origin(JsonRepairs.Lenient) != typeof(JsonRepairs).Assembly.GetName().Name, $"origin {JsonRepairs.Origin(JsonRepairs.Lenient)}");
        }
        finally
        {
            JsonRepairs.Unregister(JsonRepairs.Lenient);
        }

        Check(JsonRepairs.Parse("""{"a": 1}""")!.Json!["a"]!.GetValue<int>() == 1, "the library's repair answers again");
        try
        {
            JsonRepairs.Get("nope");
            Check(false, "an unknown repair must throw");
        }
        catch (NotSupportedException ex)
        {
            Check(ex.Message.Contains("not registered", StringComparison.Ordinal), ex.Message);
        }
    }

    private static void AnswerTextModes(Device device)
    {
        if (!FirstRun(nameof(AnswerTextModes)))
        {
            return;
        }

        const string answer = "```json\n{\"court\": \"محكمة النقض\", \"year\": 2021, \"parties\": [\"أحمد\", \"\"], \"note\": null}\n```";
        Check(AnswerTexts.Get(AnswerTexts.Raw).Text(answer) == answer, "raw keeps the answer");
        Check(AnswerTexts.Get(AnswerTexts.Values).Text(answer) == "محكمة النقض\n2021\nأحمد", AnswerTexts.Get(AnswerTexts.Values).Text(answer));
        Check(AnswerTexts.Of("{\"a\": \"x\", \"b\": \"y", AnswerTexts.Values) == "x y", "repaired JSON's values, normalized");
        Check(AnswerTexts.Of("  plain\n\ttext ", AnswerTexts.Values) == "plain text", "an answer without JSON as it is, normalized");
        Check(AnswerTexts.Of("""{"output": "a", "rules": "b", "output": "c"}""", AnswerTexts.Values) == "c b", "a repeated key's last value (was an ArgumentException out of the metric)");
        Check(AnswerTexts.Normalize("á﻿  b\n") == "á b", "composed, the mark dropped, whitespace folded");
        Check(AnswerTexts.Names.Contains(AnswerTexts.Raw) && AnswerTexts.Names.Contains(AnswerTexts.Values), "both registered");
    }

    private static void ModelHostMaxLoaded(Device device)
    {
        if (!FirstRun(nameof(ModelHostMaxLoaded)))
        {
            return;
        }

        var clock = new ManualClock();
        var created = new Dictionary<string, Dummy>();
        using var host = new ModelHost<Dummy>(name => created[name] = new Dummy(), defaultKeepAlive: null, clock) { MaxLoaded = 2 };
        using (host.Acquire("a")) { }
        clock.Now = clock.Now.AddMinutes(1);
        using (host.Acquire("b")) { }
        clock.Now = clock.Now.AddMinutes(1);
        using (host.Acquire("a")) { }                                            // a is now the more recently used
        clock.Now = clock.Now.AddMinutes(1);
        using (host.Acquire("c")) { }
        Check(created["b"].Disposed && !created["a"].Disposed && host.Loaded.Select(m => m.Name).Order().SequenceEqual(["a", "c"]),
            $"the longest unused (b) went: {string.Join(", ", host.Loaded.Select(m => m.Name))}");

        using var busy = host.Acquire("a");
        clock.Now = clock.Now.AddMinutes(1);
        using (host.Acquire("d")) { }
        Check(!created["a"].Disposed && created["c"].Disposed, "the idle one went, the one in use stayed");
    }

    // Collects the stages of reading an answer (InferenceStage).
    private sealed class StageHook : ITelemetryHook
    {
        public TelemetryLevel Levels => TelemetryLevel.Inference;

        public List<InferenceStage> Stages { get; } = [];

        public void OnInferenceStage(in InferenceStage e)
        {
            lock (Stages)
            {
                Stages.Add(e);
            }
        }
    }

    // A read with a hook listening at TelemetryLevel.Inference: the page decoded and transformed, the encoder, every cache
    // allocated, the prompt's pass and the decoding chunks, each with its sizes; the console logger's line and the JSON
    // Lines record of one.
    private static void ImageReaderTelemetry(Device device)
    {
        RegisterGemma3Vision();
        using var model = PretrainedModel.Load(TestData("vlm/tiny-gemma3"), new PretrainedOptions { Device = device });
        using var reader = new ImageReader(model, new ImageReaderOptions { Transforms = ImageTransformPipeline.Parse("grayscale,max_width=8"), ContextLength = 256 });
        var pixels = new float[3 * 12 * 20];
        for (int i = 0; i < pixels.Length; i++)
        {
            pixels[i] = (i % 89) / 88f;
        }

        var page = ChatImage.FromBytes(ImageEncoders.Get("png").Encode(new ImageData(pixels, 3, 12, 20)), "image/png");
        var request = new ImageReadRequest(page, "Extract details to JSON.") { MaxTokens = 6 };
        var hook = new StageHook();
        var console = new StringWriter();
        ImageReading result;
        using (Telemetry.Subscribe(hook))
        using (Telemetry.Subscribe(new ConsoleLogger(TelemetryLevel.Inference, output: console)))
        {
            result = reader.Read(request);
        }

        InferenceStage[] Of(InferenceStageKind kind) => [.. hook.Stages.Where(s => s.Stage == kind)];
        string seen = string.Join(", ", hook.Stages.Select(s => $"{s.Stage} {s.Count}/{s.Bytes}"));
        Check(Of(InferenceStageKind.ImageDecode) is [{ Count: 3 * 12 * 20, Device: "cpu" } decode] && decode.Bytes == page.Data.Length, $"the page decoded: {seen}");
        Check(Of(InferenceStageKind.ImageTransforms) is [{ Count: 1 * 4 * 8 }], $"its transforms (grey, 8 wide): {seen}");
        Check(Of(InferenceStageKind.Encoder) is [{ Count: 1, Bytes: > 0 } encoder] && encoder.Device == device.Backend.Kind, $"one image encoded on {device}: {seen}");
        Check(Of(InferenceStageKind.CacheAllocation) is { Length: > 0 } caches && caches.All(c => c.Count > 0 && c.Bytes > 0), $"each layer's cache with its size: {seen}");
        Check(Of(InferenceStageKind.Prefill) is [{ } prefill] && prefill.Count == result.Stats!.PromptTokens, $"the prompt's pass over {result.Stats!.PromptTokens} tokens: {seen}");
        Check(Of(InferenceStageKind.DecodeChunk).Sum(c => c.Count) == result.Stats.GeneratedTokens - 1, $"the decoding chunks' tokens, {result.Stats.GeneratedTokens} generated: {seen}");
        Check(hook.Stages.All(s => s.Duration >= TimeSpan.Zero), "durations");
        Check(console.ToString().Contains("Read Prefill on ", StringComparison.Ordinal), $"the console logger's line: {console}");

        var file = Path.Combine(Path.GetTempPath(), $"idrak-stages-{Guid.NewGuid():N}.jsonl");
        try
        {
            using (var json = new JsonLinesLogger(file, TelemetryLevel.Inference))
            using (Telemetry.Subscribe(json))
            {
                Telemetry.InferenceStage(InferenceStageKind.Encoder, device, Telemetry.Start(TelemetryLevel.Inference), 2, 4096);
            }

            string written = File.ReadAllText(file);
            Check(written.Contains("\"event\":\"inference_stage\"", StringComparison.Ordinal) && written.Contains("\"stage\":\"Encoder\"", StringComparison.Ordinal)
                  && written.Contains("\"bytes\":4096", StringComparison.Ordinal), $"the JSON Lines record: {written}");
        }
        finally
        {
            File.Delete(file);
        }
    }

    private static void ImageReaderReads(Device device)
    {
        RegisterGemma3Vision();
        using var model = PretrainedModel.Load(TestData("vlm/tiny-gemma3"), new PretrainedOptions { Device = device });
        using var reader = new ImageReader(model, new ImageReaderOptions { Transforms = ImageTransformPipeline.Parse("grayscale,max_width=8"), ContextLength = 256 });

        // A small colour page: 12 × 20, a gradient.
        var pixels = new float[3 * 12 * 20];
        for (int i = 0; i < pixels.Length; i++)
        {
            pixels[i] = (i % 97) / 96f;
        }

        var page = ChatImage.FromBytes(ImageEncoders.Get("png").Encode(new ImageData(pixels, 3, 12, 20)), "image/png");
        var request = new ImageReadRequest(page, "Extract details to JSON.") { MaxTokens = 6 };
        var prepared = reader.Prepared(request);
        Check(prepared.Channels == 1 && prepared.Width == 8 && prepared.Height == 4, $"prepared {prepared.Channels}×{prepared.Height}×{prepared.Width}");
        var own = reader.Prepared(request with { Transforms = ImageTransformPipeline.Empty });
        Check(own.Channels == 3 && own.Width == 20, $"the read's own transforms: {own.Channels}×{own.Height}×{own.Width}");

        var streamed = new System.Text.StringBuilder();
        var first = reader.Read(request, text => streamed.Append(text));
        var second = reader.Read(request);
        Check(first.Text == second.Text, $"greedy: \"{first.Text}\" then \"{second.Text}\"");
        Check(!reader.Chat.Generator.HoldsCache, "no attention cache held between reads");

        // The tokenizer's special tokens added (a transformers pipeline calling the processor on the rendered template):
        // Gemma's second <bos>, one prompt token more.
        Check(reader.Chat.Template.AddedPrefix == "<bos>", $"added prefix '{reader.Chat.Template.AddedPrefix}'");
        var withSpecial = reader.Read(request with { AddSpecialTokens = true });
        Check(withSpecial.Stats!.PromptTokens == first.Stats!.PromptTokens + 1, $"{withSpecial.Stats.PromptTokens} prompt tokens, {first.Stats.PromptTokens} without");
        var rendered = reader.Chat.RenderPrompt(new ChatRequest([new ChatMessage("user", "hi")]) { AddSpecialTokens = true });
        Check(rendered.StartsWith("<bos><bos>", StringComparison.Ordinal), rendered);

        // Sampling: a seed repeats a read exactly; impossible settings are refused.
        var sampled = request with { Sample = true, Temperature = 1.5f, TopK = 0, TopP = 0.95f, Seed = 11 };
        Check(reader.Read(sampled).Text == reader.Read(sampled).Text, "a sampled read with a seed repeats");
        try
        {
            reader.Read(sampled with { Temperature = 0f });
            Check(false, "sampling at temperature 0 must be refused");
        }
        catch (ArgumentException ex)
        {
            Check(ex.Message.Contains("temperature", StringComparison.Ordinal), ex.Message);
        }

        using (var keeping = new ImageReader(model, new ImageReaderOptions { Transforms = ImageTransformPipeline.Parse("grayscale,max_width=8"), ContextLength = 256, KeepCache = true }))
        {
            Check(keeping.Read(request).Text == first.Text && keeping.Chat.Generator.HoldsCache, "KeepCache keeps it, the same answer");
        }

        // ReuseDecoding: the second read takes the first's context and sampler (the same answers), and disposing frees them.
        var reusing = new ImageReader(model, new ImageReaderOptions { Transforms = ImageTransformPipeline.Parse("grayscale,max_width=8"), ContextLength = 256, ReuseDecoding = true });
        var reuseGenerator = reusing.Chat.Generator;
        Check(reusing.Read(request).Text == first.Text, "ReuseDecoding: the first read's answer");
        var (heldContext, heldSampler, _) = reuseGenerator.HeldDecoding;
        Check(reusing.Read(request).Text == first.Text && !reuseGenerator.HoldsCache, "ReuseDecoding: the second read's answer, no attention cache kept as KeepCache keeps it");
        Check(heldContext is not null && ReferenceEquals(reuseGenerator.HeldDecoding.Context, heldContext) && ReferenceEquals(reuseGenerator.HeldDecoding.Sampler, heldSampler),
            "ReuseDecoding: the second read took the first's context and sampler");
        Check(reusing.Read(sampled).Text == reader.Read(sampled).Text, "ReuseDecoding: a sampled read with a seed as a fresh reader's");
        reusing.Dispose();
        Check(reuseGenerator.HeldDecoding.Context is null, "disposing the reader frees the context");

        // The encoder's precision: a scope around the encoder alone (the CPU computes float32 under it: the same answer).
        var precisions = new List<MatMulPrecision>();
        using (var bf16 = new ImageReader(model, new ImageReaderOptions
        {
            Transforms = ImageTransformPipeline.Parse("grayscale,max_width=8"), ContextLength = 256, EncoderPrecision = MatMulPrecision.BFloat16,
            EncoderFactory = (m, o) => new PrecisionProbe(m.CreateVisionEncoder(o), precisions),
        }))
        {
            var read = bf16.Read(request);
            Check(read.Text == first.Text && precisions.Count > 0 && precisions.All(p => p == MatMulPrecision.BFloat16) && MixedPrecision.Current == MixedPrecision.Default,
                $"EncoderPrecision: the encoder under {string.Join(", ", precisions)}, the scope ended after it; \"{read.Text}\"");
        }

        // PrefillPrecision: the prompt's pass (after the generator made its sampler, before its first sample) under it; the
        // decoding steps (after) and the vision encoder (before the sampler) under the caller's.
        var passes = new LayerPrecisions();
        using (var prefill = new ImageReader(model, new ImageReaderOptions
        {
            Transforms = ImageTransformPipeline.Parse("grayscale,max_width=8"), ContextLength = 256, PrefillPrecision = MatMulPrecision.BFloat16,
        }))
        {
            var generator = prefill.Chat.Generator;
            Check(generator.PrefillPrecision == MatMulPrecision.BFloat16, "the reader's prefill precision reaches its generator");
            var create = generator.CreateSampler;
            generator.CreateSampler = request => new PhaseSampler(create(request), passes);
            using (Telemetry.Subscribe(passes))
            {
                _ = prefill.Read(request);
            }
        }

        var byPhase = passes.Seen.GroupBy(p => Math.Sign(p.Phase)).ToDictionary(g => g.Key, g => g.Select(p => p.Precision).ToList());
        Check(byPhase.TryGetValue(0, out var prompt) && prompt.All(p => p == MatMulPrecision.BFloat16)
              && byPhase.TryGetValue(1, out var steps) && steps.All(p => p == MixedPrecision.Default)
              && (!byPhase.TryGetValue(-1, out var encoder) || encoder.All(p => p == MixedPrecision.Default)) && MixedPrecision.Current == MixedPrecision.Default,
            $"PrefillPrecision: {string.Join("; ", byPhase.OrderBy(g => g.Key).Select(g => $"{(g.Key < 0 ? "encoder" : g.Key == 0 ? "prompt" : "decoding")} {g.Value.Count} layer passes under {string.Join(", ", g.Value.Distinct())}"))}");

        // ReadBatch: two pages with one instruction decode as one batch of two rows, and a third with another instruction
        // (another length) with them where the model attends from per-row starts (SupportsBatches), else alone; each answer,
        // its end and its count of tokens as the page's own read gives them.
        ChatImage Page(int shift)
        {
            var values = new float[3 * 12 * 20];
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = ((i * (shift + 1) + shift * 31) % 97) / 96f;
            }

            return ChatImage.FromBytes(ImageEncoders.Get("png").Encode(new ImageData(values, 3, 12, 20)), "image/png");
        }

        ImageReadRequest[] pages =
        [
            new(Page(1), "Extract details to JSON.") { MaxTokens = 12 }, new(Page(2), "Extract details to JSON.") { MaxTokens = 12 },
            new(Page(3), "Read every line of this page.") { MaxTokens = 12 },
        ];
        var alone = pages.Select(p => reader.Read(p)).ToList();
        var rows = new List<int>();
        var made = reader.Chat.Generator.CreateSampler;
        reader.Chat.Generator.CreateSampler = request =>
        {
            rows.Add(request.Rows);
            return made(request);
        };
        var pieces = new System.Text.StringBuilder[pages.Length].Select(_ => new System.Text.StringBuilder()).ToArray();
        IReadOnlyList<ImageReading> together;
        try
        {
            together = reader.ReadBatch(pages, (index, text) => pieces[index].Append(text));
        }
        finally
        {
            reader.Chat.Generator.CreateSampler = made;
        }

        for (int i = 0; i < pages.Length; i++)
        {
            Check(together[i].Text == alone[i].Text && together[i].DoneReason == alone[i].DoneReason && together[i].Stats!.GeneratedTokens == alone[i].Stats!.GeneratedTokens
                  && together[i].Stats!.PromptTokens == alone[i].Stats!.PromptTokens && pieces[i].ToString().Trim() == together[i].Text,
                $"page {i}: batched \"{together[i].Text}\" ({together[i].DoneReason}, {together[i].Stats!.GeneratedTokens} tokens), "
                + $"alone \"{alone[i].Text}\" ({alone[i].DoneReason}, {alone[i].Stats!.GeneratedTokens}); streamed \"{pieces[i]}\"");
        }

        bool together3 = reader.Chat.Generator.SupportsBatches;
        Check(rows.Order().SequenceEqual(together3 ? [3] : [1, 2]),
            $"{(together3 ? "one batch of three rows (row starts)" : "one batch of two rows and one page alone")}: samplers of {string.Join(", ", rows)} rows");

        Check(streamed.ToString().Trim() == first.Text, $"streamed \"{streamed}\", read \"{first.Text}\"");
        Check(first.Stats is { GeneratedTokens: > 0 and <= 6 } && first.Truncated == (first.Stats.GeneratedTokens >= 6), $"{first.Stats?.GeneratedTokens} tokens, truncated {first.Truncated}");
        Check(reader.Vision.Family.Length > 0 && model.Vision is not null, "the family");
    }

    private static void ImageReaderStops(Device device)
    {
        RegisterGemma3Vision();
        using var model = PretrainedModel.Load(TestData("vlm/tiny-gemma3"), new PretrainedOptions { Device = device });

        // The layer loop: a cancelled pass throws, withdraws its step and counts nothing; the same context then runs.
        using (var context = new DecodingContext(device, 1, 64) { LastPositionOnly = true })
        using (var stopped = new CancellationTokenSource())
        using (Autograd.NoGrad())
        {
            stopped.Cancel();
            context.Cancellation = stopped.Token;
            using var ids = Tensor.From([2f, 10f, 11f, 12f], [1, 4], device);
            try
            {
                model.Network.ForwardCached(ids, context).Dispose();
                Check(false, "a cancelled pass must throw");
            }
            catch (OperationCanceledException)
            {
            }

            Check(context.Length == 0 && !context.InStep, $"the stopped step withdrawn: length {context.Length}, in step {context.InStep}");
            context.Cancellation = default;
            model.Network.ForwardCached(ids, context).Dispose();
            Check(context.Length == 4, $"the context runs after: length {context.Length}");
        }

        // A read stopped before it starts gives no answer; the next read is the one an unstopped reader gives.
        using var reader = new ImageReader(model, new ImageReaderOptions { Transforms = ImageTransformPipeline.Parse("grayscale"), ContextLength = 256 });
        var pixels = new float[3 * 12 * 20];
        for (int i = 0; i < pixels.Length; i++)
        {
            pixels[i] = (i % 89) / 88f;
        }

        var request = new ImageReadRequest(ChatImage.FromBytes(ImageEncoders.Get("png").Encode(new ImageData(pixels, 3, 12, 20)), "image/png"), "Read.") { MaxTokens = 5 };
        using (var stopped = new CancellationTokenSource())
        {
            stopped.Cancel();
            int pieces = 0;
            try
            {
                reader.Read(request, _ => pieces++, stopped.Token);
                Check(false, "a stopped read must throw");
            }
            catch (OperationCanceledException)
            {
            }

            Check(pieces == 0, $"{pieces} pieces from a stopped read");
        }

        var after = reader.Read(request);
        using var fresh = new ImageReader(model, new ImageReaderOptions { Transforms = ImageTransformPipeline.Parse("grayscale"), ContextLength = 256 });
        Check(after.Text == fresh.Read(request).Text, $"after a stopped read: \"{after.Text}\"");
    }

    private static void RepetitionStop(Device device)
    {
        int[] phrase = [11, 12, 13, 14, 15];
        int[] text = [1, 2, 3, 4, 5, 6, 7, 8];
        int[] Times(int[] run, int n) => [.. Enumerable.Repeat(run, n).SelectMany(r => r)];
        Check(TextGenerator.Looping([.. text, .. Times(phrase, 4)], 4, 4), "a 5-token phrase four times");
        Check(!TextGenerator.Looping([.. text, .. Times(phrase, 3)], 4, 4), "three times is not yet four");
        Check(!TextGenerator.Looping([.. text, .. Times([9], 40)], 4, 4), "a row of one token (dots) is text");
        Check(!TextGenerator.Looping([.. text, .. Times([9, 10], 20)], 4, 4), "a two-token rule is text");
        Check(TextGenerator.Looping([.. text, .. Times([.. Enumerable.Range(100, 64)], 2)], 2, 4), "a 64-token run twice");
        Check(!TextGenerator.Looping([.. text, .. Times([.. Enumerable.Range(100, 65)], 2)], 2, 4), "runs longer than 64 tokens are not checked");
        Check(!TextGenerator.Looping([.. Times(phrase, 3), 99], 2, 4), "the repeats must end the answer");

        // A read says why its answer ended (the tiny model's answers stop at the limit or at its end token).
        RegisterGemma3Vision();
        using var model = PretrainedModel.Load(TestData("vlm/tiny-gemma3"), new PretrainedOptions { Device = device });
        using var reader = new ImageReader(model, new ImageReaderOptions { ContextLength = 256 });
        var pixels = new float[3 * 10 * 10];
        Array.Fill(pixels, 0.5f);
        var page = ChatImage.FromBytes(ImageEncoders.Get("png").Encode(new ImageData(pixels, 3, 10, 10)), "image/png");
        var reading = reader.Read(new ImageReadRequest(page, "Read.") { MaxTokens = 3, RepeatPenalty = 1.3f, FrequencyPenalty = 0.5f, RepeatStop = 2 });
        Check(reading.DoneReason is "length" or "stop" or "repetition" && reading.Truncated == (reading.DoneReason == "length"),
            $"done reason {reading.DoneReason}, truncated {reading.Truncated}");
    }
    // The precision each layer's forward pass ran under, by phase: -1 before the generator made its sampler (the vision
    // encoder), 0 from then to its first sample (the prompt's pass), then the decoding step's number.
    private sealed class LayerPrecisions : ITelemetryHook
    {
        public int Phase = -1;

        public List<(int Phase, MatMulPrecision Precision)> Seen { get; } = [];

        public TelemetryLevel Levels => TelemetryLevel.Layers;

        public void OnLayerForward(in LayerForward e)
        {
            lock (Seen)
            {
                Seen.Add((Phase, MixedPrecision.Current));
            }
        }
    }

    // A sampler that moves LayerPrecisions' phase: made (the prompt's pass follows), then each sample (a decoding step follows).
    private sealed class PhaseSampler : ITokenSampler
    {
        private readonly ITokenSampler _inner;
        private readonly LayerPrecisions _phases;

        public PhaseSampler(ITokenSampler inner, LayerPrecisions phases)
        {
            (_inner, _phases) = (inner, phases);
            phases.Phase = 0;
        }

        public int Rows => _inner.Rows;

        public int Vocabulary => _inner.Vocabulary;

        public Tensor Ids => _inner.Ids;

        public bool Recordable => _inner.Recordable;

        public void Sample(Tensor logits)
        {
            _inner.Sample(logits);
            _phases.Phase++;
        }

        public void SetHistory(IReadOnlyList<int> tokens) => _inner.SetHistory(tokens);

        public void Reset() => _inner.Reset();

        public SampledToken[][] Read(int fromStep, int toStep) => _inner.Read(fromStep, toStep);

        public void Dispose() => _inner.Dispose();
    }

    // An encoder that records the matrix-product precision it encodes under.
    private sealed class PrecisionProbe(IVisionEncoder inner, List<MatMulPrecision> seen) : IVisionEncoder
    {
        public int Width => inner.Width;

        public Device Device => inner.Device;

        public IReadOnlyList<ImageTokenLayout> Blocks(ImageData image, VisionOptions? options = null) => inner.Blocks(image, options);

        public IReadOnlyList<ImageFeatures> Encode(IReadOnlyList<ImageData> images, VisionOptions? options = null)
        {
            seen.Add(MixedPrecision.Current);
            return inner.Encode(images, options);
        }

        public void Dispose() => inner.Dispose();
    }

    private static void ImageReaderBatchFits(Device device)
    {
        RegisterGemma3Vision();
        using var model = PretrainedModel.Load(TestData("vlm/tiny-gemma3"), new PretrainedOptions { Device = device });
        ImageReader Reader(BatchFit fit, int? most = null) => new(model, new ImageReaderOptions
        {
            Transforms = ImageTransformPipeline.Parse("grayscale,max_width=8"), ContextLength = 256, BatchFit = fit, MaxBatch = most,
        });
        ChatImage Page(int shift)
        {
            var values = new float[3 * 12 * 20];
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = ((i * (shift + 1) + shift * 31) % 97) / 96f;
            }

            return ChatImage.FromBytes(ImageEncoders.Get("png").Encode(new ImageData(values, 3, 12, 20)), "image/png");
        }

        ImageReadRequest[] pages = [.. Enumerable.Range(1, 4).Select(i => new ImageReadRequest(Page(i), "Extract details to JSON.") { MaxTokens = 12 })];
        long InUse() => ComputeResources.GetMemoryUsage(device).InUse;
        long Above(Action read)
        {
            ComputeResources.ReleaseCachedMemory(device);
            long start = InUse();
            device.Backend.ResetPeakMemoryUsage();
            read();
            return ComputeResources.GetMemoryUsage(device).Peak - start;
        }

        void Limit(long? bytes)
        {
            if (device.Type == DeviceType.Cpu)
            {
                ComputeResources.CpuMemoryLimit = bytes;
            }
            else
            {
                ComputeResources.GpuMemoryLimit = bytes;
            }
        }

        // What one page and four together take above what the model holds (the encoder loaded by a first read).
        List<ImageReading> alone;
        long one, four;
        using (var measured = Reader(BatchFit.Split))
        {
            alone = [.. pages.Select(p => measured.Read(p))];
            one = Above(() => measured.Read(pages[0]));
            four = Above(() => measured.ReadBatch(pages));
        }

        Check(four > one, $"four pages together take more than one: {four:N0} and {one:N0} bytes above the model");
        string Same(IReadOnlyList<ImageReading> read) => string.Join("; ", read.Select((r, i) => r.Text == alone[i].Text ? null : $"page {i}: \"{r.Text}\", alone \"{alone[i].Text}\"").OfType<string>());
        try
        {
            using (var failing = Reader(BatchFit.Fail))
            {
                failing.Read(pages[0]);
                Limit(InUse() + (one + four) / 2);
                Expect<ResourceLimitExceededException>(() => failing.ReadBatch(pages), "limit", "BatchFit.Fail: four pages without room");
                Limit(null);
            }

            using var splitting = Reader(BatchFit.Split);
            splitting.Read(pages[0]);
            Limit(InUse() + (one + four) / 2);
            var read = splitting.ReadBatch(pages);
            int fit = read.Max(r => r.BatchSize);
            Check(read.All(r => r.BatchSize < pages.Length) && Same(read).Length == 0,
                $"BatchFit.Split: pages read {string.Join(", ", read.Select(r => r.BatchSize))} together; {Same(read)}");
            var again = splitting.ReadBatch(pages);
            Check(again.All(r => r.BatchSize <= fit) && Same(again).Length == 0, $"a later batch starts where the pages fit ({fit}): {string.Join(", ", again.Select(r => r.BatchSize))}");
        }
        finally
        {
            Limit(null);
        }

        using var capped = Reader(BatchFit.Split, most: 2);
        var two = capped.ReadBatch(pages);
        Check(two.All(r => r.BatchSize == 2) && Same(two).Length == 0, $"MaxBatch 2: {string.Join(", ", two.Select(r => r.BatchSize))}; {Same(two)}");
    }

    private sealed class SuggestionRecorder : ITelemetryHook
    {
        public List<SettingSuggestion> Seen { get; } = [];

        public TelemetryLevel Levels => TelemetryLevel.Suggestions;

        public void OnSuggestion(in SettingSuggestion e) => Seen.Add(e);
    }

    private sealed class FixedAdvisor : ISettingAdvisor
    {
        public string Name => "test pages";

        public string Summary => "a test's own suggestion for image readers";

        public IEnumerable<SettingSuggestion> Suggest(AdviceSituation situation) =>
            situation.Subject is ImageReader ? [new SettingSuggestion("Test.Pages", "2", "a test", situation.Device.ToString(), Name)] : [];
    }

    private static void ImageReaderSuggestions(Device device)
    {
        RegisterGemma3Vision();
        Check(SettingAdvisors.Names.Contains("image reader") && SettingAdvisors.Origin("image reader") == Overrides.Library
              && SettingAdvisors.Names.Contains("pretrained model") && SettingAdvisors.Origin("pretrained model") == Overrides.Library,
            $"the library's advisors: {string.Join(", ", SettingAdvisors.Names)}");

        // Loading publishes the loaded model's: bfloat16 weights exactly when float32 ones were read from a bfloat16 checkpoint.
        var loading = new SuggestionRecorder();
        using (Telemetry.Subscribe(loading))
        {
            using var loaded = PretrainedModel.Load(TestData("vlm/tiny-gemma3"), new PretrainedOptions { Device = device });
            bool bfloat16 = loading.Seen.Any(s => s.Setting == "PretrainedOptions.BFloat16" && s.Source == "pretrained model");
            Check(bfloat16 == (loaded.CheckpointFormat == Idrak.Abstraction.Formats.WeightFormat.BFloat16),
                $"checkpoint {loaded.CheckpointFormat?.ToString() ?? "not reported"}, read as float32: {string.Join("; ", loading.Seen)}");
        }

        using var model = PretrainedModel.Load(TestData("vlm/tiny-gemma3"), new PretrainedOptions { Device = device });
        var page = ChatImage.FromBytes(ImageEncoders.Get("png").Encode(new ImageData([.. Enumerable.Range(0, 3 * 12 * 20).Select(i => (i % 97) / 96f)], 3, 12, 20)), "image/png");
        var request = new ImageReadRequest(page, "Extract details to JSON.") { MaxTokens = 6 };
        var recorder = new SuggestionRecorder();
        using (Telemetry.Subscribe(recorder))
        {
            using var reader = new ImageReader(model, new ImageReaderOptions { Transforms = ImageTransformPipeline.Parse("grayscale,max_width=8"), ContextLength = 256 });
            bool matrixUnits = device.Backend.TensorCoresUnavailable() is null;
            bool precision = recorder.Seen.Any(s => s.Setting == "ImageReaderOptions.EncoderPrecision" && s.Value == "BFloat16")
                             && recorder.Seen.Any(s => s.Setting == "ImageReaderOptions.PrefillPrecision");
            Check(precision == matrixUnits, $"bfloat16 encoder and prompt suggested {precision}, matrix units {matrixUnits}: {string.Join("; ", recorder.Seen)}");
            Check(recorder.Seen.All(s => s.Device == device.ToString() && s.Source == "image reader" && s.Reason.Length > 0), "each names its device, its advisor and why");

            reader.Read(request);
            Check(recorder.Seen.All(s => s.Setting != "ImageReaderOptions.MaxBatch"), "the first (cold) read is not measured");
            reader.Read(request);
            bool measurable = device.Backend.AvailableMemory() is not null && device.Backend.Hardware.MemoryBytes is not null;
            var batch = recorder.Seen.Where(s => s.Setting == "ImageReaderOptions.MaxBatch").ToList();
            Check(measurable ? batch.Count == 1 && int.Parse(batch[0].Value, System.Globalization.CultureInfo.InvariantCulture) >= 2 : batch.Count == 0,
                $"after the first warm read (device memory {(measurable ? "reported" : "not reported")}): {string.Join("; ", recorder.Seen)}");
            int published = recorder.Seen.Count;
            reader.Read(request);
            Check(recorder.Seen.Count == published, "each suggestion published once");
            Check(reader.Suggestions().Select(s => (s.Setting, s.Value)).ToHashSet().SetEquals(recorder.Seen.Select(s => (s.Setting, s.Value))),
                "Suggestions() is what was published");

            // An app's advisor: asked for every subject, it answers for the ones it knows.
            SettingAdvisors.Register(new FixedAdvisor());
            try
            {
                Check(SettingAdvisors.Origin("test pages") != Overrides.Library && reader.Suggestions().Any(s => s.Setting == "Test.Pages" && s.Source == "test pages"),
                    "the app's advisor answers for the reader");
                Check(SettingAdvisors.Suggest(new AdviceSituation(device, "another subject", null)).Count == 0, "no advisor answers for a subject it does not know");
            }
            finally
            {
                SettingAdvisors.Unregister("test pages");
            }

            Check(reader.Suggestions().All(s => s.Source != "test pages"), "the app's advisor removed");

            // Printed, and written as JSON.
            var text = new StringWriter();
            var sample = new SettingSuggestion("ImageReaderOptions.MaxBatch", "3", "3 pages fit", "cuda:0", "image reader");
            new ConsoleLogger(TelemetryLevel.Suggestions, output: text).OnSuggestion(in sample);
            Check(text.ToString().Contains("ImageReaderOptions.MaxBatch = 3 (3 pages fit)", StringComparison.Ordinal), $"printed: {text}");
            using var stream = new MemoryStream();
            using (var writer = new System.Text.Json.Utf8JsonWriter(stream))
            {
                TelemetryJson.Write(writer, new TelemetryRecord(DateTimeOffset.UnixEpoch, sample));
            }

            string json = System.Text.Encoding.UTF8.GetString(stream.ToArray());
            Check(json.Contains("\"event\":\"setting_suggestion\"", StringComparison.Ordinal) && json.Contains("\"value\":\"3\"", StringComparison.Ordinal), $"JSON: {json}");
        }

        // Not measured: with Suggest false the first read is not measured, so nothing from its figures.
        using var quiet = new ImageReader(model, new ImageReaderOptions { Transforms = ImageTransformPipeline.Parse("grayscale,max_width=8"), ContextLength = 256, Suggest = false });
        quiet.Read(request);
        quiet.Read(request);
        Check(quiet.Suggestions().All(s => s.Setting != "ImageReaderOptions.MaxBatch"), "Suggest = false: no read measured");
    }
}
