using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Diagnostics;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Pretrained;

// Pretrained models in the Hugging Face layout, run by Idrak's own engine (CPU or CUDA):
//
//   info  <folder>                  what the loader made of the model (architecture, parameters, notes)
//   chat  <folder>                  chat with the model in its own chat template (reasoning and tool calls shown)
//   profile <folder>                time each operation and layer of a prompt pass and of decoding
//   check <reference.json>          compare with transformers: token ids, chat templates, logits, greedy output
//                                   (make the reference with tools/pytorch/pretrained_reference.py)
//
//   (<folder> may also be a Hugging Face model id, for example Qwen/Qwen3-0.6B: taken from the Hugging Face cache or
//   Idrak's, else downloaded once; HF_TOKEN or huggingface-cli login for gated models; a .gguf file; or an Ollama
//   model such as ollama:qwen3:8b, read from Ollama's own store)
//   agent <folder> <task…> --workspace <dir>
//                                   a coding agent (read, search, edit, write, run dotnet / npm …) working in <dir>
//   agent-run <folder> <suite> --out <runs.jsonl>
//                                   evaluates a model on a task suite (folders with task.json, workspace/, verify/): each
//                                   task runs in a fresh copy and is verified by its own commands; one record per run
//                                   (--attempts N, --filter S, --work DIR, --rounds N, --temperature T)
//   agent-check <suite>             checks every task without a model: verification fails on the starting files and
//                                   passes with the task's solution/ folder (--filter S, --work DIR)
//
//   Fine-tuning, evaluating, exporting and downloading models: the idrak-tune tool (src/Idrak.FineTuning.Cli).
//
// Options: --offload (when the GPU is full, keep tensors in system memory: slower, but larger models and batches fit),
//          --gpu-memory GiB (cap the GPU memory used), --adapter <dir> (chat, check, profile: load a PEFT adapter), --cuda / --cpu, --int8 (int8 weights), --int4 (4-bit weights), --bf16 (bfloat16 weights), --kv8 (int8 KV cache), --kv16 (bfloat16 KV cache), --context N (default 4096),
//          --folder F (check: read the model from F instead of the folder named in the reference), --no-think,
//          --matmul fp32|bf16|fp8 (precision of the larger matrix products: bf16 tensor cores by default, fp32 for check).
var positional = new List<string>();
bool int8 = false, bf16 = false, int4 = false, kv8 = false, kv16 = false, noThink = false;
int context = 4096;
string? folderOverride = null, output = null, adapterFolder = null;
string? workspace = null, workRoot = null, filter = null;
int attempts = 1, maxRounds = 40;
float? temperature = null;
MatMulPrecision? matmul = null;
Device device = Device.IsCudaAvailable ? Device.Cuda() : Device.Cpu;
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--cuda" or "--gpu": device = Device.Cuda(); break;
        case "--cpu": device = Device.Cpu; break;
        case "--int8": int8 = true; break;
        case "--bf16": bf16 = true; break;
        case "--int4": int4 = true; break;
        case "--kv8": kv8 = true; break;
        case "--kv16": kv16 = true; break;
        case "--no-think": noThink = true; break;
        case "--context": context = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
        case "--folder": folderOverride = args[++i]; break;
        case "--out": output = args[++i]; break;
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
        case "--workspace": workspace = args[++i]; break;
        case "--work": workRoot = args[++i]; break;
        case "--filter": filter = args[++i]; break;
        case "--attempts": attempts = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
        case "--rounds": maxRounds = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
        case "--temperature": temperature = float.Parse(args[++i], CultureInfo.InvariantCulture); break;
        case ['-', '-', ..]:
            Console.Error.WriteLine($"Unknown option {args[i]}.");
            return 1;
        default: positional.Add(args[i]); break;
    }
}

if (positional.Count < 2 || positional[0] is not ("info" or "chat" or "check" or "profile" or "agent" or "agent-run" or "agent-check")
    || positional[0] is "agent" && (positional.Count < 3 || workspace is null) || positional[0] is "agent-run" && (positional.Count < 3 || output is null))
{
    Console.WriteLine("usage: info <folder> | chat <folder> | profile <folder> | check <reference.json>");
    Console.WriteLine("       agent <folder> <task…> --workspace <dir> | agent-run <folder> <suite> --out <runs.jsonl> [--attempts N] | agent-check <suite>");
    Console.WriteLine("       [--cuda|--cpu] [--int8|--int4|--bf16] [--kv8|--kv16] [--context N] [--adapter DIR] [--folder F] [--no-think] [--matmul fp32|bf16|fp8]");
    Console.WriteLine("fine-tuning, evaluating, exporting and downloading models: idrak-tune (src/Idrak.FineTuning.Cli)");
    return 1;
}

// Tensor cores for prompts, fine-tuning and larger products (bfloat16) unless asked otherwise; check compares with
// transformers' float32 logits, so it stays float32 by default.
MixedPrecision.Default = matmul ?? (positional[0] == "check" ? MatMulPrecision.Float32 : MatMulPrecision.BFloat16);

Device.Default = device;
var cacheFormat = kv8 ? KeyValueFormat.Int8 : kv16 ? KeyValueFormat.BFloat16 : KeyValueFormat.Float32;

// A model folder, or a Hugging Face model id (owner/name): found in a cache or downloaded (with progress) on first use.
static string ResolveModel(string model) =>
    ModelSource.Resolve(model, downloader: new Idrak.Datasets.ConsoleStatus().CreateDownloader());

// npm packages of the tasks' shared projects, installed once (each run links them instead of installing).
static async Task InstallDependencies(IEnumerable<AgentTask> tasks)
{
    foreach (var folder in tasks.SelectMany(t => new[] { t.Base, t.Workspace }).OfType<string>().Distinct())
    {
        if (File.Exists(Path.Combine(folder, "package.json")) && !Directory.Exists(Path.Combine(folder, "node_modules")))
        {
            Console.WriteLine($"installing npm packages in {folder} (once)…");
            var npm = new CodingTools(folder, new CodingToolOptions { CommandTimeout = TimeSpan.FromMinutes(15) });
            string command = File.Exists(Path.Combine(folder, "package-lock.json")) ? "npm ci" : "npm install";
            var result = await npm.ExecuteAsync(command);
            Console.WriteLine(result.Succeeded ? "  done" : $"  {command} failed:\n{result.Output}");
        }
    }
}

// The sampling chat uses (Qwen3's recommended settings for thinking mode).
GenerationOptions ChatSampling() => new() { Temperature = 0.6f, TopK = 20, TopP = 0.95f, RepeatPenalty = 1f, NumCtx = context };

PretrainedModel Load(string folder)
{
    folder = ResolveModel(folder);
    var watch = Stopwatch.StartNew();
    // Chat, profile and check merge the adapter as the weights are read (full speed); finetune and export keep it separate.
    bool merge = adapterFolder is not null && positional[0] is not ("finetune" or "export");
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

var status = new Idrak.Datasets.ConsoleStatus();
var downloads = status.CreateDownloader();

switch (positional[0])
{
    case "agent-check":
    {
        var suite = AgentTask.LoadSuite(positional[1]).Where(t => filter is null || t.Id.Contains(filter, StringComparison.OrdinalIgnoreCase)).ToList();
        string work = Path.GetFullPath(workRoot ?? Path.Combine(Path.GetTempPath(), "idrak-agent-check", DateTime.Now.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture)));
        var checker = new CodingAgent(FakeChatModel.Script());
        await InstallDependencies(suite);
        int good = 0;
        foreach (var task in suite)
        {
            var watch = Stopwatch.StartNew();
            var (startFails, solutionPasses, report) = await checker.CheckTaskAsync(task, Path.Combine(work, task.Id.Replace('/', Path.DirectorySeparatorChar)));
            bool ok = startFails && solutionPasses;
            good += ok ? 1 : 0;
            Console.WriteLine($"  {(ok ? "ok  " : "BAD ")} {task.Id} ({watch.Elapsed.TotalSeconds:F0} s){(startFails ? "" : "; verification passes without any change")}{(solutionPasses ? "" : "; the solution does not pass")}");
            if (!ok)
            {
                Console.WriteLine("       " + string.Join("\n       ", report.Trim().Split('\n').TakeLast(40)));
            }
        }

        Console.WriteLine($"{good}/{suite.Count} tasks check out; work folders under {work}");
        return good == suite.Count ? 0 : 2;
    }

    case "agent" or "agent-run":
    {
        // Any supported model family, in its own chat template.
        using var model = Load(positional[1]);
        IChatModel chatModel = model.CreateChat(cacheFormat, context);
        var sampling = ChatSampling() with { Temperature = temperature ?? 0.6f };
        var agentOptions = new AgentOptions { Think = noThink ? false : null, Sampling = sampling, MaxRounds = maxRounds };
        bool live = positional[0] == "agent";
        bool inThinking = false;
        void Show(ChatDelta delta)
        {
            if (delta.Thinking.Length > 0)
            {
                Console.ForegroundColor = ConsoleColor.DarkGray;
                inThinking = true;
                Console.Write(delta.Thinking);
            }

            if (delta.Content.Length > 0)
            {
                if (inThinking)
                {
                    Console.ResetColor();
                    Console.WriteLine();
                    inThinking = false;
                }

                Console.Write(delta.Content);
            }
        }

        void ShowTool(ToolResult result)
        {
            Console.ResetColor();
            inThinking = false;
            string arguments = result.Call.Arguments.ToJsonString();
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine($"\n> {result.Call.Name} {(arguments.Length > 160 ? arguments[..160] + "…" : arguments)}");
            Console.ForegroundColor = ConsoleColor.DarkGray;
            var lines = (result.Succeeded ? result.Content : "Error: " + result.Error).Split('\n');
            Console.WriteLine(string.Join('\n', lines.Take(8)) + (lines.Length > 8 ? $"\n… {lines.Length - 8} more lines" : ""));
            Console.ResetColor();
        }

        using var cancel = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) =>
        {
            e.Cancel = !cancel.IsCancellationRequested;
            cancel.Cancel();
        };

        if (live)
        {
            var task = new AgentTask("interactive", string.Join(' ', positional.Skip(2)));
            var agent = new CodingAgent(chatModel, agentOptions) { OnDelta = Show, OnToolResult = ShowTool };
            var run = await agent.RunAsync(task, new CodingTools(workspace!, agentOptions.Tools), cancel.Token);
            Console.ResetColor();
            Console.WriteLine($"\n[{run.Outcome}{(run.Outcome == AgentOutcome.Passed ? "" : ": " + run.VerifyOutput)}; {run.Rounds} replies, {run.ToolCalls} tool calls ({run.ToolErrors} errors), "
                              + $"{run.GeneratedTokens} tokens; model {run.ModelTime.TotalSeconds:F1} s, tools {run.ToolTime.TotalSeconds:F1} s]");
            if (output is not null)
            {
                File.AppendAllText(output, run.ToJson().ToJsonString() + "\n");
                Console.WriteLine($"transcript appended to {output}");
            }

            return 0;
        }

        var suite = AgentTask.LoadSuite(positional[2]).Where(t => filter is null || t.Id.Contains(filter, StringComparison.OrdinalIgnoreCase)).ToList();
        await InstallDependencies(suite);
        string work = Path.GetFullPath(workRoot ?? Path.Combine(Path.GetTempPath(), "idrak-agent", DateTime.Now.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture)));
        Console.WriteLine($"{suite.Count} tasks × {attempts} attempts; work folders under {work}");
        var outcomes = new List<(AgentTask Task, AgentRun Run)>();
        var total = Stopwatch.StartNew();
        foreach (var task in suite)
        {
            for (int attempt = 1; attempt <= attempts && !cancel.IsCancellationRequested; attempt++)
            {
                var agent = new CodingAgent(chatModel, agentOptions with { Sampling = sampling with { Seed = attempt } })
                {
                    OnStatus = s => Console.WriteLine($"    {s}"),
                };
                var watch = Stopwatch.StartNew();
                AgentRun run;
                try
                {
                    run = await agent.RunAsync(task, Path.Combine(work, task.Id.Replace('/', Path.DirectorySeparatorChar), $"attempt-{attempt}"), cancel.Token);
                }
                catch (OperationCanceledException)
                {
                    break;
                }

                outcomes.Add((task, run));
                Console.WriteLine($"  {task.Id} #{attempt}: {run.Outcome} — {run.Rounds} replies, {run.ToolCalls} tool calls ({run.ToolErrors} errors), {run.GeneratedTokens} tokens, "
                                  + $"{watch.Elapsed.TotalSeconds:F0} s (model {run.ModelTime.TotalSeconds:F0} s, tools {run.ToolTime.TotalSeconds:F0} s)");
                if (run.Outcome is AgentOutcome.SetupFailed or AgentOutcome.Error)
                {
                    Console.WriteLine("    " + run.VerifyOutput.Trim().Replace("\n", "\n    ", StringComparison.Ordinal));
                }

                File.AppendAllText(output!, run.ToJson().ToJsonString() + "\n");
            }
        }

        var scored = outcomes.Where(o => o.Run.Outcome is not (AgentOutcome.SetupFailed or AgentOutcome.Error)).ToList();
        int passed = scored.Count(o => o.Run.Outcome == AgentOutcome.Passed);
        Console.WriteLine($"\n{passed}/{scored.Count} runs passed ({(scored.Count == 0 ? 0 : 100.0 * passed / scored.Count):F1}%), "
                          + $"{outcomes.Count - scored.Count} not scored (setup failed or model error), {total.Elapsed:hh\\:mm\\:ss}");
        foreach (var group in scored.GroupBy(o => o.Task.Language ?? "other").OrderBy(g => g.Key, StringComparer.Ordinal))
        {
            int p = group.Count(o => o.Run.Outcome == AgentOutcome.Passed);
            Console.WriteLine($"  {group.Key}: {p}/{group.Count()} ({100.0 * p / group.Count():F1}%), "
                              + $"median {group.Select(o => o.Run.Rounds).Order().ElementAt(group.Count() / 2)} replies, {group.Average(o => o.Run.ToolErrors):F1} tool errors per run");
        }

        int tasksSolved = scored.GroupBy(o => o.Task.Id).Count(g => g.Any(o => o.Run.Outcome == AgentOutcome.Passed));
        Console.WriteLine($"  tasks solved at least once: {tasksSolved}/{scored.Select(o => o.Task.Id).Distinct().Count()}; runs written to {output}");
        return passed == scored.Count ? 0 : 2;
    }

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

// Reports on the calling thread, at once (Progress<T> posts to the thread pool, so lines can arrive late or out of order).
internal sealed class ConsoleProgress<T>(Action<T> report) : IProgress<T>
{
    public void Report(T value) => report(value);
}
