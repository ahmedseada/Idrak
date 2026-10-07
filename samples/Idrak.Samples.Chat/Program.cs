// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Diagnostics;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.LanguageModels;

// Chat with a language model in the Hugging Face layout (Llama, Qwen, Mistral, Gemma …), run by Idrak's own engine
// (CPU, CUDA or any Vulkan GPU):
//
//   info  <folder>                  what the loader made of the model (architecture, parameters, notes)
//   chat  <folder>                  chat with the model in its own chat template (reasoning and tool calls shown)
//   profile <folder>                time each operation and layer of a prompt pass and of decoding
//   check <reference.json>          compare with transformers: token ids, chat templates, logits, greedy output
//                                   (make the reference with tools/pytorch/pretrained_reference.py)
//
//   (<folder> may also be a Hugging Face model id, for example Qwen/Qwen3-0.6B: taken from the Hugging Face cache or
//   Idrak's, else downloaded once; HF_TOKEN or huggingface-cli login for gated models; a .gguf file; or a model
//   of the local model store such as store:qwen3:8b, read in place)
//
//   A coding agent on the same models: samples/Idrak.Samples.CodingAgent. Fine-tuning, evaluating, exporting and
//   downloading models: the idrak tool (idrak tune, idrak pull; src/Idrak.Cli).
//
// Options: --offload (when the GPU is full, keep tensors in system memory: slower, but larger models and batches fit),
//          --gpu-memory GiB (cap the GPU memory used), --adapter <dir> (load a PEFT adapter), --cuda / --cpu / --vulkan / --device NAME (cpu, cuda:N, vulkan:N), --int8 (int8 weights), --int4 (4-bit weights), --bf16 (bfloat16 weights), --kv8 (int8 KV cache), --kv16 (bfloat16 KV cache), --context N (default 4096),
//          --folder F (check: read the model from F instead of the folder named in the reference), --no-think,
//          --matmul fp32|bf16|fp8 (precision of the larger matrix products: bf16 tensor cores by default, fp32 for check).
var positional = new List<string>();
bool int8 = false, bf16 = false, int4 = false, kv8 = false, kv16 = false, noThink = false;
int context = 4096;
string? folderOverride = null, adapterFolder = null;
MatMulPrecision? matmul = null;
Device device = Device.Default;                                       // CUDA, else a Vulkan GPU with IDRAK_VULKAN_DEFAULT=1, else the CPU
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--cuda" or "--gpu": device = Device.Cuda(); break;
        case "--cpu": device = Device.Cpu; break;
        case "--vulkan": device = Device.Parse("vulkan:0"); break;
        case "--device": device = Device.Parse(args[++i]); break;     // any name --list-devices shows: cpu, cuda:N, vulkan:N
        case "--int8": int8 = true; break;
        case "--bf16": bf16 = true; break;
        case "--int4": int4 = true; break;
        case "--kv8": kv8 = true; break;
        case "--kv16": kv16 = true; break;
        case "--no-think": noThink = true; break;
        case "--context": context = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
        case "--folder": folderOverride = args[++i]; break;
        case "--adapter": adapterFolder = args[++i]; break;
        case "--offload": ComputeResources.OffloadToHostMemory = true; break;
        case "--matmul":
            matmul = args[++i] switch
            {
                "fp32" or "float32" => MatMulPrecision.Float32,
                "bf16" or "bfloat16" => MatMulPrecision.BFloat16,
                "fp8" or "float8" => MatMulPrecision.Float8,
                var other => throw new ArgumentException($"--matmul {other}: use fp32, bf16 or fp8"),
            };
            break;
        case "--gpu-memory": ComputeResources.GpuMemoryLimit = (long)(double.Parse(args[++i], CultureInfo.InvariantCulture) * (1L << 30)); break;
        case ['-', '-', ..]:
            Console.Error.WriteLine($"Unknown option {args[i]}.");
            return 1;
        default: positional.Add(args[i]); break;
    }
}

if (positional.Count < 2 || positional[0] is not ("info" or "chat" or "check" or "profile"))
{
    Console.WriteLine("usage: info <folder> | chat <folder> | profile <folder> | check <reference.json>");
    Console.WriteLine("       [--cuda|--cpu|--vulkan|--device NAME] [--int8|--int4|--bf16] [--kv8|--kv16] [--context N] [--adapter DIR] [--folder F] [--no-think] [--matmul fp32|bf16|fp8]");
    Console.WriteLine("fine-tuning, evaluating, exporting and downloading models: idrak tune, idrak pull (src/Idrak.Cli)");
    return 1;
}

// Tensor cores for prompts and larger products (bfloat16) unless asked otherwise; check compares with transformers'
// float32 logits, so it stays float32 by default.
MixedPrecision.Default = matmul ?? (positional[0] == "check" ? MatMulPrecision.Float32 : MatMulPrecision.BFloat16);

Device.Default = device;
var cacheFormat = kv8 ? KeyValueFormat.Int8 : kv16 ? KeyValueFormat.BFloat16 : KeyValueFormat.Float32;

// A model folder, or a Hugging Face model id (owner/name): found in a cache or downloaded (with progress) on first use.
static string ResolveModel(string model) =>
    ModelSource.Resolve(model, downloader: new Idrak.Data.ConsoleStatus().CreateDownloader());

// The sampling chat uses (Qwen3's recommended settings for thinking mode).
GenerationOptions ChatSampling() => new() { Temperature = 0.6f, TopK = 20, TopP = 0.95f, RepeatPenalty = 1f, NumCtx = context };

PretrainedModel Load(string folder)
{
    folder = ResolveModel(folder);
    var watch = Stopwatch.StartNew();
    // The adapter is merged as the weights are read (full speed).
    bool merge = adapterFolder is not null;
    var model = PretrainedModel.Load(folder, new PretrainedOptions
    {
        Device = device, Int8 = int8, BFloat16 = bf16, Int4 = int4, MaxPositions = context, MergeAdapter = merge ? adapterFolder : null,
    });
    Console.WriteLine($"Loaded {model.Config["architectures"]?[0]} from {folder} in {watch.Elapsed.TotalSeconds:F1} s on {device}{(int8 ? ", int8 weights" : int4 ? ", int4 weights" : bf16 ? ", bf16 weights" : "")}");
    Console.WriteLine($"  {model.Spec.ParameterCount / 1e6:F0}M parameters, {model.Spec.Layers} layers, dim {model.Spec.Dim}, heads {model.Spec.Heads}/{model.Spec.KvHeads}, "
                      + $"vocabulary {model.Spec.Vocabulary}, context {model.MaxPositions}");
    foreach (var note in model.Notes)
    {
        Console.WriteLine($"  note: {note}");
    }

    if (adapterFolder is not null && !merge)
    {
        Console.WriteLine($"  adapter: {model.LoadAdapter(adapterFolder)} layers from {adapterFolder}");
    }

    return model;
}

switch (positional[0])
{
    case "info":
    {
        using var model = Load(positional[1]);
        Console.WriteLine(model.Spec.ToJson());
        Console.WriteLine(model.ChatTemplate is { } t ? $"chat template: {t.Source.Length} characters, stops [{string.Join(", ", t.StopSequences)}]" : "no chat template");
        return 0;
    }

    case "chat":
    {
        using var model = Load(positional[1]);
        var chat = model.CreateChat(cacheFormat, context);
        var messages = new List<ChatMessage>();
        Console.WriteLine("Type a message (empty line to quit, /system <text> to set the system prompt).");
        while (Console.ReadLine() is { Length: > 0 } line)
        {
            if (line.StartsWith("/system ", StringComparison.Ordinal))
            {
                messages.RemoveAll(m => m.Role == "system");
                messages.Insert(0, new ChatMessage("system", line[8..]));
                continue;
            }

            messages.Add(new ChatMessage("user", line));
            var options = ChatSampling();
            ChatMessage? reply = null;
            bool thinking = false;
            foreach (var chunk in chat.Stream(new ChatRequest(messages, Think: noThink ? false : null, Options: options)))
            {
                if (chunk.Delta.Thinking.Length > 0)
                {
                    if (!thinking)
                    {
                        Console.ForegroundColor = ConsoleColor.DarkGray;
                        thinking = true;
                    }

                    Console.Write(chunk.Delta.Thinking);
                }

                if (chunk.Delta.Content.Length > 0)
                {
                    if (thinking)
                    {
                        Console.ResetColor();
                        Console.WriteLine();
                        thinking = false;
                    }

                    Console.Write(chunk.Delta.Content);
                }

                foreach (var call in chunk.Delta.ToolCalls)
                {
                    Console.Write($"\n[tool call] {call.Name}({call.Arguments.ToJsonString()})");
                }

                if (chunk.Done)
                {
                    reply = chunk.Message;
                    Console.ResetColor();
                    Console.WriteLine($"\n[{chunk.Stats?.GeneratedTokens} tokens, {chunk.Stats?.TokensPerSecond:F1} tok/s]");
                }
            }

            if (reply is not null)
            {
                messages.Add(reply);
            }
        }

        return 0;
    }

    case "profile":
    {
        using var model = Load(positional[1]);
        return Profile(model);
    }

    default:
        return Check(positional[1]);
}

// Times a prompt pass and a short generation: wall time, then the operations and layers that took it.
int Profile(PretrainedModel model)
{
    string prompt = string.Concat(Enumerable.Repeat("public static int Add(int a, int b) => a + b;\n", 12));
    var ids = model.Tokenizer!.Encode(prompt);
    using var input = Tensor.From([.. ids.Select(i => (float)i)], [1, ids.Count], device);
    for (int i = 0; i < 2; i++)
    {
        var warm = Stopwatch.StartNew();
        using (var y = model.Network.Predict(input))
        {
            device.Synchronize();
        }

        Console.WriteLine($"prompt pass {i + 1} ({ids.Count} tokens): {warm.Elapsed.TotalMilliseconds:F1} ms");
    }

    var generator = model.CreateGenerator(cacheFormat, context);
    var greedy = new GenerationOptions { TopK = 1, Temperature = 1f, RepeatPenalty = 1f, NumPredict = 32, NumCtx = context, Seed = 0 };
    var (_, _, stats) = generator.Generate(prompt, greedy);
    Console.WriteLine($"generation: prompt {stats.PromptDuration.TotalMilliseconds:F0} ms, {stats.GeneratedTokens} tokens at {stats.TokensPerSecond:F1} tok/s");
    var sampled = ChatSampling() with { NumPredict = 128, Seed = 0 };
    var (_, _, sampledStats) = generator.Generate(prompt, sampled);
    Console.WriteLine($"generation with chat sampling (temperature {sampled.Temperature}, top-k {sampled.TopK}, top-p {sampled.TopP}): "
                      + $"{sampledStats.GeneratedTokens} tokens at {sampledStats.TokensPerSecond:F1} tok/s");

    var recorder = new ProfileRecorder();
    Telemetry.SynchronizeForTiming = true;
    using (Telemetry.Subscribe(recorder))
    {
        var watch = Stopwatch.StartNew();
        using (var y = model.Network.Predict(input))
        {
            device.Synchronize();
        }

        recorder.Report($"prompt pass, {ids.Count} tokens, timed per operation: {watch.Elapsed.TotalMilliseconds:F1} ms wall");
        recorder.Clear();
        watch.Restart();
        var (_, _, timed) = generator.Generate(prompt, greedy with { UseGraph = false });
        recorder.Report($"generation of {timed.GeneratedTokens} tokens (with the prompt), timed per operation: {watch.Elapsed.TotalMilliseconds:F1} ms wall");
    }

    Telemetry.SynchronizeForTiming = false;

    // GPU time per kernel (CUDA events around each launch; host launch costs excluded).
    if (device.Type == DeviceType.Cuda)
    {
        GpuProfiler.Start(device);
        using (var y = model.Network.Predict(input))
        {
            device.Synchronize();
        }

        Console.WriteLine($"\nprompt pass, {ids.Count} tokens, GPU time per kernel:");
        Console.Write(GpuProfiler.Format(GpuProfiler.Stop(device), rows: 25));
        var decodeOnly = greedy with { NumPredict = 1, UseGraph = false };
        generator.Generate(prompt, decodeOnly);
        GpuProfiler.Start(device);
        var (_, _, profiled) = generator.Generate(prompt, greedy with { UseGraph = false });
        var entries = GpuProfiler.Stop(device);
        Console.WriteLine($"\ngeneration of {profiled.GeneratedTokens} tokens (with the prompt), GPU time per kernel:");
        Console.Write(GpuProfiler.Format(entries, rows: 25));
        Console.WriteLine($"(decode steps are {entries.Sum(e => e.Calls) / Math.Max(1, profiled.GeneratedTokens)} launches per token)");
    }

    return 0;
}

int Check(string referencePath)
{
    var reference = JsonNode.Parse(File.ReadAllText(referencePath))!.AsObject();
    adapterFolder ??= (string?)reference["adapter"];
    using var model = Load(folderOverride ?? (string)reference["folder"]!);
    var tokenizer = (BpeTokenizer)(model.Tokenizer ?? throw new InvalidOperationException("The model has no tokenizer.json."));
    int failures = 0;

    // 1. Tokenizer: the same ids as transformers, and decoding gives the text back.
    Console.WriteLine("\nTokenizer");
    foreach (var item in reference["texts"]!.AsArray())
    {
        string text = (string)item!["text"]!;
        int[] expected = [.. item["ids"]!.AsArray().Select(i => (int)i!)];
        var actual = tokenizer.Encode(text);
        bool same = actual.SequenceEqual(expected);
        string decoded = tokenizer.Decode(expected), expectedDecoded = (string?)item["decoded"] ?? text;
        bool decodes = decoded == expectedDecoded;
        Console.WriteLine($"  {(same && decodes ? "ok  " : "DIFF")} {expected.Length,4} ids{(decodes ? "" : $"  decoded differently: {Shorten(decoded)}")}  {Shorten(text)}");
        failures += same && decodes ? 0 : 1;
        if (!same)
        {
            int at = Enumerable.Range(0, Math.Min(actual.Count, expected.Length)).FirstOrDefault(i => actual[i] != expected[i], Math.Min(actual.Count, expected.Length));
            Console.WriteLine($"       first difference at {at}: expected [{string.Join(", ", expected.Skip(at).Take(4).Select(tokenizer.TokenOf))}], "
                              + $"got [{string.Join(", ", actual.Skip(at).Take(4).Select(tokenizer.TokenOf))}]");
        }
    }

    // 2. Chat templates: the model's own template renders conversations (tools, tool calls, reasoning) identically.
    Console.WriteLine("\nChat template");
    var tools = reference["tools"]!.AsArray().Select(t => new ToolDefinition((string)t!["name"]!, (string?)t["description"], t["parameters"]?.DeepClone())).ToList();
    foreach (var chat in reference["chats"]!.AsArray())
    {
        var messages = chat!["messages"]!.AsArray().Select(m => new ChatMessage((string)m!["role"]!, (string)m["content"]!, (string?)m["thinking"],
            m["tool_calls"]?.AsArray().Select(c => new ToolCall((string)c!["name"]!, (JsonObject)c["arguments"]!.DeepClone())).ToList(),
            (string?)m["tool_name"])).ToList();
        var chatTools = chat["tools"]!.AsArray().Count > 0 ? tools : [];
        bool? think = chat["think"] is JsonValue v ? (bool)v : null;
        bool generationPrompt = (bool)chat["add_generation_prompt"]!;
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
        Console.WriteLine($"  {(same ? "ok  " : "DIFF")} {messages.Count} messages, {chatTools.Count} tools, think {think?.ToString() ?? "default"}, "
                          + $"generation prompt {generationPrompt}: {expected.Length} characters");
        if (!same)
        {
            failures++;
            int at = Enumerable.Range(0, Math.Min(actual.Length, expected.Length)).FirstOrDefault(i => actual[i] != expected[i], Math.Min(actual.Length, expected.Length));
            Console.WriteLine($"       first difference at {at}:\n       expected …{Shorten(expected[Math.Max(0, at - 30)..])}\n       got      …{Shorten(actual[Math.Max(0, at - 30)..])}");
        }
    }

    // 3. Logits for the same ids, and 4. greedy continuations (through the generator and its KV cache).
    Console.WriteLine("\nLogits and greedy output");
    foreach (var run in reference["runs"]!.AsArray())
    {
        int[] ids = [.. run!["ids"]!.AsArray().Select(i => (int)i!)];
        float[] expected = [.. run["logits"]!.AsArray().Select(x => (float)x!)];
        using var input = Tensor.From([.. ids.Select(i => (float)i)], [1, ids.Length], device);
        var watch = Stopwatch.StartNew();
        float[] all;
        using (var logits = model.Network.Predict(input))
        {
            all = logits.ToArray();
        }

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

        bool encodes = tokenizer.Encode((string)run["prompt"]!).SequenceEqual(ids);
        int[] greedyExpected = [.. run["generated"]!.AsArray().Select(i => (int)i!)];
        var generator = model.CreateGenerator(cacheFormat, context);
        var options = new GenerationOptions { TopK = 1, Temperature = 1f, RepeatPenalty = 1f, NumPredict = greedyExpected.Length, NumCtx = context, Seed = 0 };
        var (text, _, stats) = generator.Generate((string)run["prompt"]!, options);
        string expectedText = (string)run["generated_text"]!;
        int agree = 0;
        while (agree < Math.Min(text.Length, expectedText.Length) && text[agree] == expectedText[agree])
        {
            agree++;
        }

        bool greedySame = text == expectedText;
        bool approximate = int8 || int4 || bf16 || kv8 || kv16;
        bool ok = encodes && top1 && (int8 || int4 || bf16 || maxDiff <= 2e-3f * Math.Max(1f, scale)) && (greedySame || approximate);
        failures += ok ? 0 : 1;
        Console.WriteLine($"  {(ok ? "ok  " : "DIFF")} {ids.Length,4} ids: max |Δlogit| {maxDiff:G3} (largest logit {scale:F1}), top-1 {(top1 ? "same" : "DIFFERENT")}, "
                          + $"top-10 overlap {overlap}/10, forward {watch.Elapsed.TotalMilliseconds:F0} ms; greedy {(greedySame ? "identical" : $"first {agree} of {expectedText.Length} characters agree")} "
                          + $"({stats.TokensPerSecond:F1} tok/s){(encodes ? "" : "; the prompt text encodes to different ids")}");
        if (!greedySame)
        {
            Console.WriteLine($"       transformers: {Shorten(expectedText)}\n       Idrak:  {Shorten(text)}");
        }
    }

    Console.WriteLine(failures == 0 ? "\nEverything matches transformers." : $"\n{failures} check(s) differ.");
    return failures == 0 ? 0 : 2;
}

static int[] TopK(float[] values, int k) =>
    [.. values.Select((v, i) => (v, i)).OrderByDescending(p => p.v).Take(k).Select(p => p.i)];

static string Shorten(string text)
{
    string flat = text.Replace("\r", "\\r").Replace("\n", "\\n").Replace("\t", "\\t");
    return flat.Length > 100 ? flat[..100] + "…" : flat;
}

// Sums the time of each operation (by name and shape) and each layer type.
sealed class ProfileRecorder : ITelemetryHook
{
    private readonly Dictionary<string, (int Count, double Ms)> _operations = [];
    private readonly Dictionary<string, (int Count, double Ms)> _layers = [];

    public TelemetryLevel Levels => TelemetryLevel.Operations | TelemetryLevel.Layers;

    public void OnOperation(in OperationCompleted e) =>
        Add(_operations, $"{e.Operation}{(e.Backward ? " (backward)" : "")} [{string.Join("x", e.Shape)}]", e.Duration);

    public void OnLayerForward(in LayerForward e) => Add(_layers, $"{e.LayerType} (depth {e.Depth})", e.Duration);

    private static void Add(Dictionary<string, (int Count, double Ms)> table, string key, TimeSpan duration)
    {
        var (count, ms) = table.GetValueOrDefault(key);
        table[key] = (count + 1, ms + duration.TotalMilliseconds);
    }

    public void Clear()
    {
        _operations.Clear();
        _layers.Clear();
    }

    public void Report(string title)
    {
        Console.WriteLine($"\n{title}");
        Console.WriteLine($"  operations: {_operations.Values.Sum(v => v.Ms):F1} ms in {_operations.Values.Sum(v => v.Count)} calls; the slowest:");
        foreach (var (name, (count, ms)) in _operations.OrderByDescending(p => p.Value.Ms).Take(20))
        {
            Console.WriteLine($"    {ms,9:F1} ms {count,6}×  {name}");
        }

        Console.WriteLine("  layers:");
        foreach (var (name, (count, ms)) in _layers.OrderByDescending(p => p.Value.Ms).Take(12))
        {
            Console.WriteLine($"    {ms,9:F1} ms {count,6}×  {name}");
        }
    }
}
