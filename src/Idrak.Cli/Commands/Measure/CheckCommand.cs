// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.LanguageModels;

namespace Idrak.Cli.Commands.Measure;

/// <summary>
/// <c>idrak check MODEL --reference FILE</c>: compares a model with a reference made by transformers
/// (tools/pytorch/pretrained_reference.py): token ids, chat templates, logits and greedy output, as the Chat sample's
/// check does.
/// </summary>
internal sealed class CheckCommand : Command
{
    public override string Name => "check";

    public override string Summary => "Compare token ids, chat templates, logits and greedy output with a transformers reference";

    public override string Usage =>
        "MODEL --reference FILE [options]\n\n" +
        "FILE is the JSON that tools/pytorch/pretrained_reference.py writes: \"texts\" (text and ids), \"chats\" (messages,\n" +
        "tools and the rendered template), \"tools\", and \"runs\" (prompt ids, last-position logits, greedy ids and text).\n" +
        "Logits must agree within 2e-3 of the largest (float32 weights) and greedy text exactly; with packed weights\n" +
        "or a packed cache only the top token is required. Exit code 1 when anything differs.\n\n" +
        "Options:\n" +
        "      --reference FILE  the reference JSON (required)\n" +
        "  -w, --weights FORMAT  int8, int4, bf16 or a registered packed format\n" +
        "  -k, --kv FORMAT       the key/value cache format\n" +
        "      --context N       the context length; --adapter DIR: merge an adapter (default: the reference's)\n\n" +
        "Examples:\n" +
        "  idrak check org/model --reference reference.json\n" +
        "  idrak check ./tuned --reference ref.json -d vulkan:0 -j\n\n" +
        "Environment: IDRAK_CACHE (models), HF_TOKEN (gated downloads); products stay float32 here whatever IDRAK_MATMUL says";

    public override IReadOnlyCollection<string> ValueOptions => [.. Models.ValueOptions, "--reference"];

    public override IReadOnlyDictionary<string, string> ShortForms => Models.ShortForms;

    public override int Run(CommandContext context)
    {
        string modelName = context.Argument(0, "MODEL");
        string referencePath = context.Option("--reference") ?? throw new UsageException("Missing --reference FILE (the JSON tools/pytorch/pretrained_reference.py writes).");
        if (!File.Exists(referencePath))
        {
            throw new UsageException($"No reference file {referencePath}: make one with tools/pytorch/pretrained_reference.py.");
        }

        var reference = JsonNode.Parse(File.ReadAllText(referencePath)) as JsonObject ?? throw new InvalidDataException($"{referencePath} is not a JSON object.");
        var choice = Models.Choose(context, modelName);
        if (choice.Adapter is null && (string?)reference["adapter"] is { } adapter)
        {
            choice = choice with { Adapter = adapter };
        }

        // The reference's logits are float32: products stay float32 unless the weights are packed anyway.
        var precision = MixedPrecision.Default;
        MixedPrecision.Default = MatMulPrecision.Float32;
        try
        {
            using var model = Models.Load(context, choice);
            return Check(context, model, choice, reference);
        }
        finally
        {
            MixedPrecision.Default = precision;
        }
    }

    private static int Check(CommandContext context, PretrainedModel model, Models.ModelChoice choice, JsonObject reference)
    {
        var device = context.Device;
        var tokenizer = model.Tokenizer ?? throw new InvalidOperationException($"{choice.Model} has no tokenizer (tokenizer.json).");
        string TokenOf(int id) => tokenizer is BpeTokenizer bpe ? bpe.TokenOf(id) : id.ToString();
        int failures = 0;
        string kv = choice.Kv ?? "float32";
        bool approximate = choice.Weights is not null and not ("float32" or "f32") || kv != "float32";
        bool packedWeights = choice.Weights is not null and not ("float32" or "f32");

        // 1. Tokenizer: the same ids, and decoding gives the text back.
        context.Write("Tokenizer");
        var texts = new JsonArray();
        foreach (var item in Array(reference, "texts"))
        {
            string text = (string)item["text"]!;
            int[] expected = [.. item["ids"]!.AsArray().Select(i => (int)i!)];
            var actual = tokenizer.Encode(text);
            bool same = actual.SequenceEqual(expected);
            string decoded = tokenizer.Decode(expected), expectedDecoded = (string?)item["decoded"] ?? text;
            bool decodes = decoded == expectedDecoded;
            failures += same && decodes ? 0 : 1;
            context.Write($"  {(same && decodes ? "ok  " : "DIFF")} {expected.Length,4} ids{(decodes ? "" : $"  decoded differently: {Shorten(decoded)}")}  {Shorten(text)}");
            int? at = null;
            if (!same)
            {
                int common = Math.Min(actual.Count, expected.Length);
                at = Enumerable.Range(0, common).FirstOrDefault(i => actual[i] != expected[i], common);
                context.Write($"       first difference at {at}: expected [{string.Join(", ", expected.Skip(at.Value).Take(4).Select(TokenOf))}], "
                              + $"got [{string.Join(", ", actual.Skip(at.Value).Take(4).Select(TokenOf))}]");
            }

            texts.Add(new JsonObject { ["text"] = text, ["ok"] = same && decodes, ["ids_match"] = same, ["decodes"] = decodes, ["first_difference"] = at });
        }

        // 2. Chat templates: the model's own template renders each conversation identically.
        context.Write("\nChat template");
        var tools = Array(reference, "tools").Select(t => new ToolDefinition((string)t["name"]!, (string?)t["description"], t["parameters"]?.DeepClone())).ToList();
        var chats = new JsonArray();
        foreach (var chat in Array(reference, "chats"))
        {
            var messages = chat["messages"]!.AsArray().Select(m => new ChatMessage((string)m!["role"]!, (string?)m["content"] ?? "", (string?)m["thinking"],
                m["tool_calls"]?.AsArray().Select(c => new ToolCall((string)c!["name"]!, (JsonObject)c["arguments"]!.DeepClone())).ToList(),
                (string?)m["tool_name"])).ToList();
            var chatTools = chat["tools"] is JsonArray { Count: > 0 } ? tools : [];
            bool? think = chat["think"] is JsonValue v ? (bool)v : null;
            bool generationPrompt = (bool?)chat["add_generation_prompt"] ?? true;
            string expected = (string)chat["rendered"]!;
            string actual;
            try
            {
                actual = model.ChatTemplate?.Render(messages, chatTools, think, generationPrompt) ?? "ERROR: no chat template";
            }
            catch (InvalidOperationException ex)
            {
                actual = "ERROR: " + ex.Message;
            }

            bool same = actual == expected || actual.StartsWith("ERROR", StringComparison.Ordinal) && expected.StartsWith("ERROR", StringComparison.Ordinal);
            failures += same ? 0 : 1;
            context.Write($"  {(same ? "ok  " : "DIFF")} {messages.Count} messages, {chatTools.Count} tools, think {think?.ToString() ?? "default"}, "
                          + $"generation prompt {generationPrompt}: {expected.Length} characters");
            int? at = null;
            if (!same)
            {
                int common = Math.Min(actual.Length, expected.Length);
                at = Enumerable.Range(0, common).FirstOrDefault(i => actual[i] != expected[i], common);
                context.Write($"       first difference at {at}:\n       expected …{Shorten(expected[Math.Max(0, at.Value - 30)..])}\n       got      …{Shorten(actual[Math.Max(0, at.Value - 30)..])}");
            }

            chats.Add(new JsonObject { ["messages"] = messages.Count, ["tools"] = chatTools.Count, ["ok"] = same, ["first_difference"] = at });
        }

        // 3. Logits for the same ids, and 4. greedy continuations through the generator and its key/value cache.
        context.Write("\nLogits and greedy output");
        var runs = new JsonArray();
        foreach (var run in Array(reference, "runs"))
        {
            int[] ids = [.. run["ids"]!.AsArray().Select(i => (int)i!)];
            float[] expected = [.. run["logits"]!.AsArray().Select(x => (float)x!)];
            var watch = Stopwatch.StartNew();
            float[] all;
            using (var input = Tensor.From([.. ids.Select(i => (float)i)], [1, ids.Length], device))
            using (var logits = model.Network.Predict(input))
            {
                all = logits.ToArray();
            }

            double forwardMs = watch.Elapsed.TotalMilliseconds;
            var actual = all.AsSpan(all.Length - expected.Length).ToArray();
            float maxDiff = 0, scale = 0;
            for (int i = 0; i < expected.Length; i++)
            {
                maxDiff = Math.Max(maxDiff, Math.Abs(actual[i] - expected[i]));
                scale = Math.Max(scale, Math.Abs(expected[i]));
            }

            var topExpected = TopK(expected, 10);
            var topActual = TopK(actual, 10);
            bool top1 = topExpected[0] == topActual[0];
            int overlap = topExpected.Intersect(topActual).Count();
            string prompt = (string)run["prompt"]!;
            bool encodes = tokenizer.Encode(prompt).SequenceEqual(ids);
            int greedyLength = run["generated"]?.AsArray().Count ?? 0;
            string expectedText = (string?)run["generated_text"] ?? "";
            var generator = model.CreateGenerator(KeyValueLayouts.Get(kv), model.MaxPositions);
            var options = new GenerationOptions { TopK = 1, Temperature = 1f, RepeatPenalty = 1f, NumPredict = Math.Max(1, greedyLength), NumCtx = model.MaxPositions, Seed = 0 };
            var (text, _, stats) = generator.Generate(prompt, options);
            int agree = 0;
            while (agree < Math.Min(text.Length, expectedText.Length) && text[agree] == expectedText[agree])
            {
                agree++;
            }

            bool greedySame = text == expectedText;
            bool logitsClose = packedWeights || maxDiff <= 2e-3f * Math.Max(1f, scale);
            bool ok = encodes && top1 && logitsClose && (greedySame || approximate);
            failures += ok ? 0 : 1;
            context.Write($"  {(ok ? "ok  " : "DIFF")} {ids.Length,4} ids: max |Δlogit| {maxDiff:G3} (largest logit {scale:F1}), top-1 {(top1 ? "same" : "DIFFERENT")}, "
                          + $"top-10 overlap {overlap}/10, forward {forwardMs:F0} ms; greedy {(greedySame ? "identical" : $"first {agree} of {expectedText.Length} characters agree")} "
                          + $"({stats.TokensPerSecond:F1} tokens/s){(encodes ? "" : "; the prompt text encodes to different ids")}");
            if (!greedySame)
            {
                context.Write($"       reference: {Shorten(expectedText)}\n       Idrak:     {Shorten(text)}");
            }

            runs.Add(new JsonObject
            {
                ["ids"] = ids.Length,
                ["ok"] = ok,
                ["max_logit_difference"] = maxDiff,
                ["largest_logit"] = scale,
                ["top1_same"] = top1,
                ["top10_overlap"] = overlap,
                ["prompt_encodes"] = encodes,
                ["greedy_same"] = greedySame,
                ["greedy_agreeing_characters"] = agree,
                ["greedy_text"] = text,
            });
        }

        context.Write(failures == 0 ? "\nEverything matches the reference." : $"\n{failures} check(s) differ.");
        context.WriteJson(new JsonObject
        {
            ["model"] = choice.Model,
            ["device"] = device.ToString(),
            ["weights"] = choice.Weights,
            ["kv"] = kv,
            ["ok"] = failures == 0,
            ["failures"] = failures,
            ["texts"] = texts,
            ["chats"] = chats,
            ["runs"] = runs,
        });
        return failures == 0 ? ExitCodes.Ok : ExitCodes.Failed;
    }

    private static IEnumerable<JsonObject> Array(JsonObject reference, string name) =>
        reference[name] is JsonArray array ? array.OfType<JsonObject>() : [];

    private static int[] TopK(float[] values, int k) =>
        [.. values.Select((v, i) => (v, i)).OrderByDescending(p => p.v).Take(k).Select(p => p.i)];

    private static string Shorten(string text)
    {
        string flat = text.Replace("\r", "\\r").Replace("\n", "\\n").Replace("\t", "\\t");
        return flat.Length > 100 ? flat[..100] + "…" : flat;
    }
}
