// A small GPT: a decoder-only, character-level transformer with causal self-attention that learns to
// continue text. It trains on sentences from a small grammar (or your own text via --corpus), saves the
// model, then generates new text one character at a time.
//
//   dotnet run -c Release --project samples/Idrak.Samples.TextGeneration                       train, save, generate
//   dotnet run -c Release --project samples/Idrak.Samples.TextGeneration -- --corpus book.txt  train on your own text
//   dotnet run -c Release --project samples/Idrak.Samples.TextGeneration -- --predict --input "the old wizard" --temperature 0.8 --length 400
//                                                                                                  generate from the saved model
//   dotnet run -c Release --project samples/Idrak.Samples.TextGeneration -- --predict --benchmark true
//                                                          compare full recompute, KV cache, CUDA graph and batched sampling
// The Web API sample (Idrak.Samples.GptApi) serves the same model with a browser UI.

using Idrak.Diagnostics;
using Idrak.Samples;
using Idrak.Samples.Gpt;

if (SampleOptions.Parse(args,
        ("corpus", "text file to train on (default: sentences from a built-in grammar)"),
        ("temperature", "sampling temperature: lower is safer, higher is more creative (default 0.7)"),
        ("top-k", "sample only among the k most likely characters (default: all)"),
        ("length", "characters to generate (default 400)"),
        ("samples", "continuations generated together as one batch (default 1)"),
        ("cache", "on|off: incremental decoding with a KV cache (default on)"),
        ("graph", "on|off: record the decoding step as a CUDA graph on GPU (default on)"),
        ("benchmark", "true: compare decoding modes and batch sizes"),
        ("chat", "true: train (or load) a chat model on synthetic ChatML transcripts with reasoning and web_fetch tool calls")) is not { } options)
{
    return 0;
}

var device = options.Device;
Console.WriteLine($"Device: {device} - {device.Name}\n");
using var logger = Telemetry.Subscribe(new ConsoleLogger(options.LogLevels));
bool chatMode = options.Get("chat") is "true" or "1";
string modelPath = options.ModelPath(chatMode ? "chat.weights" : "transformer.weights");

CharGpt gpt;
if (options.PredictOnly)
{
    if (!options.RequireModel(modelPath))
    {
        return 1;
    }

    gpt = CharGpt.Load(modelPath, device);
    Console.WriteLine($"Loaded {gpt.Model.ParameterCount:N0} parameters from {modelPath}" +
                      $" (trained {gpt.Config.TrainedEpochs} epochs on {gpt.Config.Corpus}, validation accuracy {gpt.Config.ValidationAccuracy:P1})\n");
}
else
{
    string? corpusPath = options.Get("corpus");
    string corpus = corpusPath is not null ? File.ReadAllText(corpusPath)
        : chatMode ? ChatCorpus.Generate(count: 4000, seed: 1)
        : Grammar.Corpus(sentences: 6000, seed: 1);
    string vocabulary = new([.. corpus.Distinct().Order()]);
    Console.WriteLine($"Corpus: {corpus.Length:N0} characters, vocabulary of {vocabulary.Length}: \"{vocabulary.Replace("\n", "\\n")}\"");
    Console.WriteLine($"Sample: \"{corpus[..160].Replace("\n", " ")}...\"\n");

    // Small enough to train in a couple of minutes on a CPU; raise Dim / Layers / Context on a GPU.
    gpt = CharGpt.Create(chatMode ? new GptConfig(vocabulary, Context: 256, Dim: 96, Heads: 4, Layers: 3)
                                  : new GptConfig(vocabulary, Context: 64, Dim: 96, Heads: 4, Layers: 3), device);
    var trained = GptTraining.Train(gpt, corpus, options.Epochs ?? (chatMode ? 6 : 2), options.BatchSize ?? 32,
        corpusPath ?? (chatMode ? "synthetic ChatML transcripts" : "built-in grammar"));
    gpt.Save(modelPath, trained);
    Console.WriteLine($"\nSaved the model to {modelPath}\nTest it later with: --predict --input \"the little robot\"\n");
}

if (chatMode)
{
    using (gpt)
    {
        ChatDemo(gpt);
    }

    return 0;
}

using (gpt)
{
    bool On(string name) => options.Get(name, "on") is "on" or "true" or "1";
    var settings = new GenerationSettings(
        Length: int.Parse(options.Get("length", "400")!),
        Temperature: float.Parse(options.Get("temperature", "0.7")!, System.Globalization.CultureInfo.InvariantCulture),
        TopK: int.Parse(options.Get("top-k", "0")!),
        Seed: 6,
        Samples: int.Parse(options.Get("samples", "1")!),
        UseCache: On("cache"),
        UseGraph: On("graph"));

    if (options.Get("benchmark") is "true" or "1")
    {
        Benchmark(gpt, settings);
        return 0;
    }

    string[] prompts = options.Input is { } input ? [input] : ["the little robot ", "every morning, ", "a curious cat "];
    foreach (var (prompt, i) in prompts.Select((p, i) => (p, i)))
    {
        var result = gpt.Generate(prompt.ToLowerInvariant(), settings with { Seed = settings.Seed + i });
        var m = result.Metrics;
        Console.WriteLine($"Prompt \"{prompt}\" (temperature {settings.Temperature}, {m.Mode}):");
        foreach (var sample in result.Samples)
        {
            Console.WriteLine((result.Samples.Count > 1 ? $"  [{sample.Sample + 1}] " : "  ") + (prompt + sample.Text).Replace("\n", "\n  "));
        }

        Console.WriteLine($"  -> {m.TotalTokens} characters in {m.TotalMs:F0} ms ({m.TokensPerSecond:F0}/s, {m.MsPerStep:F2} ms per step), " +
                          $"first token {m.FirstTokenMs:F1} ms, average confidence {m.AverageProbability:P0}, perplexity {m.Perplexity:F2}");
        if (gpt.Config.Corpus == "built-in grammar")
        {
            var (valid, total) = Grammar.Score(string.Join(" ", result.Samples.Select(s => s.Text)));
            Console.WriteLine($"  -> {valid}/{total} generated words are real vocabulary words ({valid / (double)Math.Max(total, 1):P0})");
        }

        Console.WriteLine();
    }
}

return 0;

// Times each decoding mode on the same prompt: characters per second summed over all samples.
static void Benchmark(CharGpt gpt, GenerationSettings settings)
{
    var modes = new (string Name, GenerationSettings Settings)[]
    {
        ("full recompute, 1 sample", settings with { UseCache = false, UseGraph = false, Samples = 1 }),
        ("KV cache, 1 sample", settings with { UseCache = true, UseGraph = false, Samples = 1 }),
        ("KV cache + graph, 1 sample", settings with { UseCache = true, UseGraph = true, Samples = 1 }),
        ("KV cache + graph, 8 samples", settings with { UseCache = true, UseGraph = true, Samples = 8 }),
        ("KV cache + graph, 32 samples", settings with { UseCache = true, UseGraph = true, Samples = 32 }),
    };
    Console.WriteLine($"Benchmark on {gpt.Device} ({gpt.Device.Name}), {settings.Length} characters per sample\n");
    Console.WriteLine("  mode                            chars/s   ms/step   first token   note");
    double? baseline = null;
    foreach (var (name, s) in modes)
    {
        gpt.Generate("the little robot ", s with { Length = 40 });   // warm-up (JIT, memory pool, graph)
        var m = gpt.Generate("the little robot ", s).Metrics;
        baseline ??= m.TokensPerSecond;
        Console.WriteLine($"  {name,-30} {m.TokensPerSecond,8:F0}  {m.MsPerStep,8:F2}  {m.FirstTokenMs,9:F1} ms   {m.TokensPerSecond / baseline:F1}x{(m.GraphNote is { } note ? "  " + note : "")}");
    }
}

// The chat model on a request like an Ollama /api/chat call: reasoning, a tool call, then an answer from the tool result.
static void ChatDemo(CharGpt gpt)
{
    var chat = new Idrak.Generation.ChatGenerator(new Idrak.Generation.TextGenerator(gpt.Model,
        new Idrak.Generation.CharTokenizer(gpt.Config.Vocabulary), gpt.Config.Context));
    var options = new Idrak.Generation.GenerationOptions { Temperature = 1f, TopK = 20, TopP = 0.95f, MinP = 0f, RepeatPenalty = 1f, NumCtx = 4096, NumPredict = 300, Seed = 7 };
    var messages = new List<Idrak.Generation.ChatMessage>
    {
        new("system", "You are a helpful assistant. Cite sources as [1], [2] when a research pack is present."),
        new("user", "What is the latest Ollama version?"),
    };
    Idrak.Generation.ToolDefinition[] tools = [ChatCorpus.WebFetch];

    void Turn(string title)
    {
        var reply = chat.Chat(new Idrak.Generation.ChatRequest(messages, tools, Think: true, Options: options));
        var m = reply.Message!;
        Console.WriteLine($"{title}");
        Console.WriteLine($"  thinking:   {m.Thinking}");
        Console.WriteLine($"  content:    {m.Content}");
        foreach (var call in m.ToolCalls ?? [])
        {
            Console.WriteLine($"  tool call:  {call.Name}({call.Arguments.ToJsonString()})");
        }

        Console.WriteLine($"  done: {reply.DoneReason}, {reply.Stats!.PromptTokens} prompt tokens, {reply.Stats.GeneratedTokens} generated, {reply.Stats.TokensPerSecond:F0} tokens/s\n");
        messages.Add(m);
    }

    Turn("Turn 1 (tools offered, think = true):");
    if (messages[^1].ToolCalls is { Count: > 0 } calls)
    {
        string url = calls[0].Arguments["url"]?.GetValue<string>() ?? "";
        messages.Add(new("tool", $"[1] {url}: Ollama 0.12.3 is the latest release.", ToolName: "web_fetch"));
        Turn("Turn 2 (after the tool result):");
    }
}
