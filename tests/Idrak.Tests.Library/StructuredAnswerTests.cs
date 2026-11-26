// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Diagnostics;
using Idrak.Data.Abstractions;
using Idrak.Generation;
using Idrak.Imaging;
using Idrak.Models;
using Idrak.Text;
using Idrak.Text.Abstractions;
using Idrak.Text.Generation;

// A model's JSON answers read leniently (JsonRepairs), the text of an answer a metric compares (AnswerTexts), the model
// host's limit on loaded models, and the repetition stop.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] StructuredAnswerGroup =
    [
        ("json repair: valid JSON reads unchanged and is not marked repaired; text and a code fence around it are dropped; text without JSON has none", JsonRepairValid),
        ("json repair: an answer cut at its token limit, trailing and missing commas, single and curly quotes, bare keys, Python literals, quotes inside values and repeated keys read as json_repair reads them", JsonRepairBroken),
        ("answer texts: raw keeps the answer, values takes its JSON's values in document order (repaired JSON too; an answer without JSON as it is); Normalize composes and folds whitespace", AnswerTextModes),
        ("model host: MaxLoaded unloads the longest unused idle model before loading another, and never one in use", ModelHostMaxLoaded),
        ("generation: the repetition stop finds an answer ending in one run of tokens repeated N times (runs from the minimum to 64 tokens; shorter runs such as a row of dots are text)", RepetitionStop),
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

    internal static void AnswerTextModes(Device device)
    {
        if (!FirstRun(nameof(AnswerTextModes)))
        {
            return;
        }

        const string answer = "```json\n{\"office\": \"مكتب الرياض\", \"year\": 2021, \"members\": [\"أحمد\", \"\"], \"note\": null}\n```";
        Check(new RawAnswerText().Text(answer) == answer, "raw keeps the answer");
        Check(new JsonValuesAnswerText().Text(answer) == "مكتب الرياض\n2021\nأحمد", new JsonValuesAnswerText().Text(answer));
        Check(AnswerTexts.Normalize(new JsonValuesAnswerText().Text("{\"a\": \"x\", \"b\": \"y")) == "x y", "repaired JSON's values, normalized");
        Check(AnswerTexts.Normalize(new JsonValuesAnswerText().Text("  plain\n\ttext ")) == "plain text", "an answer without JSON as it is, normalized");
        Check(AnswerTexts.Normalize(new JsonValuesAnswerText().Text("""{"output": "a", "rules": "b", "output": "c"}""")) == "c b", "a repeated key's last value (was an ArgumentException out of the metric)");
        Check(AnswerTexts.Normalize("á﻿  b\n") == "á b", "composed, the mark dropped, whitespace folded");
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
