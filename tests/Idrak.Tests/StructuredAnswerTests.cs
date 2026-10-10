// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Data.Abstractions;
using Idrak.Generation;
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
        ("image reader: a vision checkpoint reads a page greedily (the same answer twice), streams it as Read returns it, prepares the page with its transforms or the read's own", ImageReaderReads),
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
        Check(streamed.ToString().Trim() == first.Text, $"streamed \"{streamed}\", read \"{first.Text}\"");
        Check(first.Stats is { GeneratedTokens: > 0 and <= 6 } && first.Truncated == (first.Stats.GeneratedTokens >= 6), $"{first.Stats?.GeneratedTokens} tokens, truncated {first.Truncated}");
        Check(reader.Vision.Family.Length > 0 && model.Vision is not null, "the family");
    }
}
