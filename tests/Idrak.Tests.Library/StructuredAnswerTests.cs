// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Diagnostics;
using Idrak.Data.Abstractions;
using Idrak.Generation;
using Idrak.Imaging;
using Idrak.Models;
using Idrak.Text.Abstractions;
using Idrak.Text.Generation;

// A model's JSON answers read leniently (JsonRepairs), the text of an answer a metric compares (AnswerTexts), the model
// host's limit on loaded models, and the image reader (ImageReader over ModelImages) on the tiny Gemma 3 fixture.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] StructuredAnswerGroup =
    [
        ("json repair: valid JSON reads unchanged and is not marked repaired; text and a code fence around it are dropped; text without JSON has none", JsonRepairValid),
        ("json repair: an answer cut at its token limit, trailing and missing commas, single and curly quotes, bare keys, Python literals, quotes inside values and repeated keys read as json_repair reads them", JsonRepairBroken),
        ("json repair: the registry has the library's lenient repair; an app's repair shadows it and is removed again; an unknown name is not registered", JsonRepairRegistry),
        ("answer texts: raw keeps the answer, values takes its JSON's values in document order (repaired JSON too; an answer without JSON as it is); Normalize composes and folds whitespace", AnswerTextModes),
        ("model host: MaxLoaded unloads the longest unused idle model before loading another, and never one in use", ModelHostMaxLoaded),
        ("image reader: pages without room together (a memory limit between one page's peak and four's) are read in halves until they fit, each reading saying how many it was read with, the answers the pages' own; later batches start where they fit; BatchFit.Fail raises; MaxBatch caps the batch", ImageReaderBatchFits),
        ("generation: the repetition stop finds an answer ending in one run of tokens repeated N times (runs from the minimum to 64 tokens; shorter runs such as a row of dots are text); a read says why it ended", RepetitionStop),
        ("image reader: telemetry: a read publishes its stages at the Inference level (the page decoded and transformed, the encoder, each cache allocated, the prompt's pass, the decoding chunks) with their sizes; the console line and the JSON Lines record", ImageReaderTelemetry),
        ("image reader: a stopped read ends before its first token (the prompt's pass checks between layers; the step is withdrawn, the context's length unchanged) and the next read is unaffected", ImageReaderStops),
    ];

    internal static JsonNode? Repaired(string text, bool expectRepaired = true)
    {
        var result = JsonRepairs.Parse(text);
        Check(result is not null, $"no JSON found in {text}");
        Check(result!.Repaired == expectRepaired, $"repaired {result.Repaired} for {text}");
        return result.Json;
    }

    internal static void JsonRepairValid(Device device)
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

    internal static void JsonRepairBroken(Device device)
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

    internal sealed class EmptyRepair : IJsonRepair
    {
        public string Name => JsonRepairs.Lenient;

        public string Summary => "always an empty object";

        public JsonRepairResult? Repair(string text) => new(new JsonObject(), Repaired: true);
    }

    internal static void JsonRepairRegistry(Device device)
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

    internal static void AnswerTextModes(Device device)
    {
        if (!FirstRun(nameof(AnswerTextModes)))
        {
            return;
        }

        const string answer = "```json\n{\"office\": \"مكتب الرياض\", \"year\": 2021, \"members\": [\"أحمد\", \"\"], \"note\": null}\n```";
        Check(AnswerTexts.Get(AnswerTexts.Raw).Text(answer) == answer, "raw keeps the answer");
        Check(AnswerTexts.Get(AnswerTexts.Values).Text(answer) == "مكتب الرياض\n2021\nأحمد", AnswerTexts.Get(AnswerTexts.Values).Text(answer));
        Check(AnswerTexts.Of("{\"a\": \"x\", \"b\": \"y", AnswerTexts.Values) == "x y", "repaired JSON's values, normalized");
        Check(AnswerTexts.Of("  plain\n\ttext ", AnswerTexts.Values) == "plain text", "an answer without JSON as it is, normalized");
        Check(AnswerTexts.Of("""{"output": "a", "rules": "b", "output": "c"}""", AnswerTexts.Values) == "c b", "a repeated key's last value (was an ArgumentException out of the metric)");
        Check(AnswerTexts.Normalize("á﻿  b\n") == "á b", "composed, the mark dropped, whitespace folded");
        Check(AnswerTexts.Names.Contains(AnswerTexts.Raw) && AnswerTexts.Names.Contains(AnswerTexts.Values), "both registered");
    }

    internal static void ModelHostMaxLoaded(Device device)
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
    internal sealed class StageHook : ITelemetryHook
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
    internal static void ImageReaderTelemetry(Device device)
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

    internal static void ImageReaderStops(Device device)
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

    internal static void RepetitionStop(Device device)
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
    internal static void ImageReaderBatchFits(Device device)
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

    internal sealed class SuggestionRecorder : ITelemetryHook
    {
        public List<SettingSuggestion> Seen { get; } = [];

        public TelemetryLevel Levels => TelemetryLevel.Suggestions;

        public void OnSuggestion(in SettingSuggestion e) => Seen.Add(e);
    }

    internal sealed class FixedAdvisor : ISettingAdvisor
    {
        public string Name => "test pages";

        public string Summary => "a test's own suggestion for image readers";

        public IEnumerable<SettingSuggestion> Suggest(AdviceSituation situation) =>
            situation.Subject is ImageReader ? [new SettingSuggestion("Test.Pages", "2", "a test", situation.Device.ToString(), Name)] : [];
    }

}
