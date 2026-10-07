// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Generation;
using Idrak.Layers;

internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] Generation =
    [
        ("generation: parser splits thinking, content and tool calls from any chunking", ParserSplitsOutput),
        ("generation: word tokenizer splits, maps unknown words and decodes piecewise", WordTokenizerBehaviour),
        ("generation: ChatML template renders system, tools, turns and think modes", TemplateRendering),
        ("generation: keep-alive durations and model host expiry", KeepAliveAndHost),
        ("generation: num_predict, stop sequences, cache/graph/recompute agree", GeneratorBehaviour),
        ("generation: chat end to end yields a final message and stats", ChatEndToEnd),
        ("generation: a kept KV cache reuses shared prompt prefixes without changing the output", PromptCacheReuse),
    ];

    private static void WordTokenizerBehaviour(Device device)
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

    private static void ParserSplitsOutput(Device device)
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

    private static void TemplateRendering(Device device)
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

    private sealed class ManualClock : TimeProvider
    {
        public DateTimeOffset Now { get; set; } = new(2026, 1, 1, 0, 0, 0, TimeSpan.Zero);
        public override DateTimeOffset GetUtcNow() => Now;
    }

    private sealed class Dummy : IDisposable
    {
        public bool Disposed { get; private set; }
        public void Dispose() => Disposed = true;
    }

    private static void KeepAliveAndHost(Device device)
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

    private static (Sequential Model, CharTokenizer Tokenizer) TinyLanguageModel(Device device, int context = 32)
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

    private static void PromptCacheReuse(Device device)
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

    private static void GeneratorBehaviour(Device device)
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
    }

    private static void ChatEndToEnd(Device device)
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
