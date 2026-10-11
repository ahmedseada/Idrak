// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Data.Abstractions;
using Idrak.Generation;
using Idrak.Generation.Abstractions;
using Idrak.Models;
using Idrak.Nlp.Abstractions;

// A model's JSON answers read leniently (JsonRepairs), the text of an answer a metric compares (AnswerTexts), the model
// host's limit on loaded models, and the image reader (ImageReader over ModelImages) on the tiny Gemma 3 fixture.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] StructuredAnswerGroup =
    [
        ("json repair: valid JSON reads unchanged and is not marked repaired; text and a code fence around it are dropped; text without JSON has none", JsonRepairValid),
        ("json repair: an answer cut at its token limit, trailing and missing commas, single and curly quotes, bare keys, Python literals and quotes inside values read as json_repair reads them", JsonRepairBroken),
        ("json repair: the registry has the library's lenient repair; an app's repair shadows it and is removed again; an unknown name is not registered", JsonRepairRegistry),
        ("answer texts: raw keeps the answer, values takes its JSON's values in document order (repaired JSON too; an answer without JSON as it is); Normalize composes and folds whitespace", AnswerTextModes),
        ("model host: MaxLoaded unloads the longest unused idle model before loading another, and never one in use", ModelHostMaxLoaded),
        ("image reader: a vision checkpoint reads a page greedily (the same answer twice), streams it as Read returns it, prepares the page with its transforms or the read's own; no attention cache held between reads unless KeepCache; a sampled read with a seed repeats; pages with one instruction read as one batch, each answer its own read's", ImageReaderReads),
        ("generation: the repetition stop finds an answer ending in one run of tokens repeated N times (runs from the minimum to 64 tokens; shorter runs such as a row of dots are text); a read says why it ended", RepetitionStop),
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

        // ReadBatch: two pages with one instruction decode as one batch of two rows, a third with another instruction (another
        // length) alone; each answer, its end and its count of tokens as the page's own read gives them.
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

        Check(rows.Order().SequenceEqual([1, 2]), $"one batch of two rows and one page alone: samplers of {string.Join(", ", rows)} rows");

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
}
