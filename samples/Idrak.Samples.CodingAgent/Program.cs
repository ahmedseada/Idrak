// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Abstraction.Reinforcement;
using Idrak.Diagnostics;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Reinforcement;
using Idrak.Reinforcement.Plugins;
using Idrak.Reinforcement.Plugins.Coding;
using Idrak.Text;
using Idrak.Text.Generation;
using Idrak.Text.Plugins;
using Idrak.Text.Reinforcement;

// A coding agent: a language model (any family Idrak loads, in its own chat template) reads, searches, edits and writes
// files and runs allowlisted commands (dotnet, npm …) in a workspace, run by Idrak's own engine (CPU or CUDA):
//
//   agent <folder> <task…> --workspace <dir>
//                                   works on the task in <dir>, streaming its reasoning and tool calls
//                                   (--out F: append the transcript to F)
//   agent-run <folder> <suite> --out <runs.jsonl>
//                                   evaluates a model on a task suite (folders with task.json, workspace/, verify/;
//                                   Idrak.Reinforcement.Plugins' coding task layer): each task runs in a fresh copy and is
//                                   verified by its own commands; one record per run
//                                   (--attempts N, --filter S, --work DIR, --rounds N, --temperature T)
//   agent-check <suite>             checks every task without a model: verification fails on the starting files and
//                                   passes with the task's solution/ folder (--filter S, --work DIR)
//   agent-train <folder> <suite> --out <dir>
//                                   trains the model's LoRA adapters on the suite by reinforcement learning: the library's
//                                   trainer (Idrak.Reinforcement) in the coding plug-in's environment, the tasks' hidden tests
//                                   the reward; checkpoints in <dir>/checkpoints, the adapter in <dir> (--algorithm grpo|ppo|
//                                   reinforce, --iterations N, --group N, --episodes N, --lr R, --rank N, --max-new N,
//                                   --rounds N, --filter S, --work DIR, --temperature T, --resume; every other setting from
//                                   its IDRAK_RL_* variable). suite/ holds a small C# demo suite.
//
//   (<folder> may also be a Hugging Face model id, for example Qwen/Qwen3-0.6B, a .gguf file or store:name.)
//   data/agent-demo-*.jsonl: tool-calling transcripts in this layout, to fine-tune a model for the agent with idrak tune.
//
// Options: --cuda / --cpu / --vulkan / --device NAME (cpu, cuda:N, vulkan:N), --int8 | --int4 | --bf16 (base weights), --kv8 | --kv16 (KV cache), --context N (default
//          4096), --adapter <dir> (load a PEFT adapter), --no-think, --offload, --gpu-memory GiB, --matmul fp32|bf16|fp8.
TextPlugins.RegisterAll();                          // the model families (Llama, Qwen, Mistral, Gemma, …) are Idrak.Text.Plugins'
ReinforcementPlugins.RegisterAll();                 // the coding plug-in: the container sandbox (IDRAK_SANDBOX=container) among it

var positional = new List<string>();
bool int8 = false, bf16 = false, int4 = false, kv8 = false, kv16 = false, noThink = false;
int context = 4096;
string? output = null, adapterFolder = null;
string? workspace = null, workRoot = null, filter = null;
int attempts = 1, maxRounds = 40;
int? iterations = null, groupSize = null, episodes = null, rank = null, maxNew = null;
float? learningRate = null;
string? algorithm = null;
bool resume = false;
float? temperature = null;
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
        case "--algorithm": algorithm = args[++i]; break;
        case "--iterations": iterations = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
        case "--group": groupSize = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
        case "--episodes": episodes = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
        case "--lr": learningRate = float.Parse(args[++i], CultureInfo.InvariantCulture); break;
        case "--rank": rank = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
        case "--max-new": maxNew = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
        case "--resume": resume = true; break;
        case ['-', '-', ..]:
            Console.Error.WriteLine($"Unknown option {args[i]}.");
            return 1;
        default: positional.Add(args[i]); break;
    }
}

if (positional.Count < 2 || positional[0] is not ("agent" or "agent-run" or "agent-check" or "agent-train")
    || positional[0] is "agent" && (positional.Count < 3 || workspace is null) || positional[0] is "agent-run" or "agent-train" && (positional.Count < 3 || output is null))
{
    Console.WriteLine("usage: agent <folder> <task…> --workspace <dir> | agent-run <folder> <suite> --out <runs.jsonl> [--attempts N] | agent-check <suite>");
    Console.WriteLine("       | agent-train <folder> <suite> --out <dir> [--algorithm grpo|ppo|reinforce] [--iterations N] [--group N] [--episodes N] [--resume]");
    Console.WriteLine("       [--cuda|--cpu|--vulkan|--device NAME] [--int8|--int4|--bf16] [--kv8|--kv16] [--context N] [--adapter DIR] [--no-think] [--matmul fp32|bf16|fp8]");
    return 1;
}

MixedPrecision.Default = matmul ?? MatMulPrecision.BFloat16;
Device.Default = device;
var cacheFormat = kv8 ? KeyValueFormat.Int8 : kv16 ? KeyValueFormat.BFloat16 : KeyValueFormat.Float32;

// A model folder, or a Hugging Face model id (owner/name): found in a cache or downloaded (with progress) on first use.
static string ResolveModel(string model) =>
    ModelSource.Resolve(model, downloader: new Idrak.Data.ConsoleStatus().CreateDownloader());

// npm packages of the tasks' shared projects, installed once (each run links them instead of installing).
static async Task InstallDependencies(IEnumerable<CodingTask> tasks)
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

PretrainedModel Load(string folder, bool mergeAdapter = true)
{
    folder = ResolveModel(folder);
    var watch = Stopwatch.StartNew();
    // The adapter is merged as the weights are read (full speed), unless it is to be trained further (agent-train).
    bool merge = mergeAdapter && adapterFolder is not null;
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
    case "agent-train":
    {
        // Reinforcement learning on the suite, with no trainer of its own: the library's trainer (Idrak.Reinforcement),
        // Idrak.Text's language-model policy over a fine-tuning session, the coding plug-in's environment. The run's
        // settings come from the IDRAK_RL_* variables; the options given here win.
        var options = ReinforcementOptions.FromEnvironment();
        options = options with
        {
            Algorithm = algorithm ?? options.Algorithm,
            Iterations = iterations ?? options.Iterations,
            GroupSize = groupSize ?? options.GroupSize,
            EpisodesPerIteration = episodes ?? options.EpisodesPerIteration,
            Temperature = temperature ?? options.Temperature,
            CheckpointFolder = options.CheckpointFolder ?? Path.Combine(output!, "checkpoints"),
        };
        options = options with { Resume = resume ? options.CheckpointFolder : options.Resume };
        var environment = new Dictionary<string, string> { ["suite"] = Path.GetFullPath(positional[2]), ["rounds"] = maxRounds.ToString(CultureInfo.InvariantCulture) };
        if (filter is not null)
        {
            environment["filter"] = filter;
        }

        if (workRoot is not null)
        {
            environment["work"] = workRoot;
        }

        if (noThink)
        {
            environment["think"] = "false";
        }

        options = options with { EnvironmentOptions = environment };
        await InstallDependencies(CodingSuite.Load(positional[2], filter));

        using var model = Load(positional[1], mergeAdapter: false);       // an --adapter goes on training
        // The session's schedule spans the run's updates: an update per mini-batch of every epoch of every iteration (the
        // trainer's defaults: ppo 4 epochs, else 1; the whole rollout a mini-batch).
        int epochs = options.Epochs > 0 ? options.Epochs : options.Algorithm.Equals(ReinforcementNames.Ppo, StringComparison.OrdinalIgnoreCase) ? 4 : 1;
        int miniBatch = options.MiniBatch > 0 ? Math.Min(options.MiniBatch, options.EpisodesPerIteration) : options.EpisodesPerIteration;
        int updates = Math.Max(1, options.Iterations * epochs * ((options.EpisodesPerIteration + miniBatch - 1) / miniBatch));
        var defaults = new FineTuningOptions();
        var tuning = defaults with
        {
            Rank = rank ?? defaults.Rank,
            LearningRate = learningRate ?? defaults.LearningRate,
            Scheduler = LearningRateSchedules.Factory(options.Schedule, options.ScheduleSettings),
        };
        var session = new FineTuningSession(model, tuning, updates);
        using var policy = new LanguageModelPolicy(session, new LanguageModelPolicyOptions
        {
            MaxNewTokens = maxNew ?? 1024, ContextLength = context, Think = noThink ? false : null,
        });

        using var cancel = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) =>
        {
            e.Cancel = !cancel.IsCancellationRequested;              // the first Ctrl+C stops after the current iteration
            cancel.Cancel();
        };

        Console.WriteLine($"{options.Algorithm} on {positional[2]}: {options.Iterations} iterations of {options.EpisodesPerIteration} episodes, checkpoints in {options.CheckpointFolder}");
        using var progress = ProgressReporters.Start("agent-train", options.Iterations);
        var trainer = new ReinforcementTrainer<ChatObservation, ChatMessage>(EnvironmentKinds.Get("coding"), policy, options)
        {
            Progress = progress,
            OnIteration = s =>
            {
                progress.Erase();
                Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                    $"  iteration {s.Iteration}: pass rate {s.RewardMean:P0} (± {s.RewardSpread:0.##}), loss {s.Loss:0.####}, rollout {s.RolloutTime.TotalSeconds:F0} s, update {s.UpdateTime.TotalSeconds:F0} s"));
            },
        };
        var result = trainer.Train(cancel.Token);
        Directory.CreateDirectory(output!);
        session.Save(output!);
        Console.WriteLine($"{(result.Cancelled ? "stopped after" : "trained")} {result.Iterations} iterations; adapter written to {Path.GetFullPath(output!)}"
                          + (result.Checkpoint is { } c ? $", checkpoint {c}" : "") + $"; try it: agent-run {positional[1]} {positional[2]} --adapter {output} --out runs.jsonl");
        return 0;
    }

    case "agent-check":
    {
        var suite = CodingSuite.Load(positional[1], filter);
        string work = Path.GetFullPath(workRoot ?? Path.Combine(Path.GetTempPath(), "idrak-agent-check", DateTime.Now.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture)));
        var checker = new CodingVerifier();
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
            var task = new CodingTask("interactive", string.Join(' ', positional.Skip(2)));
            var agent = new CodingAgent(chatModel, agentOptions) { OnDelta = Show, OnToolResult = ShowTool };
            var run = await agent.RunAsync(task.Prompt, new CodingTools(workspace!, agentOptions.Tools), cancel.Token);
            Console.ResetColor();
            // A typed task has no verification commands: finishing is all "Passed" can mean, so say that.
            var typed = new CodingRun(task, run, run.End switch
            {
                CodingAgentEnd.Answered => CodingOutcome.Passed,
                CodingAgentEnd.OutOfBudget => CodingOutcome.OutOfBudget,
                _ => CodingOutcome.Error,
            }, run.Note);
            string outcome = typed.Outcome == CodingOutcome.Passed ? "Done (the task has no checks)" : $"{typed.Outcome}: {typed.VerifyOutput}";
            Console.WriteLine($"\n[{outcome}; {run.Rounds} replies, {run.ToolCalls} tool calls ({run.ToolErrors} errors), "
                              + $"{run.GeneratedTokens} tokens; model {run.ModelTime.TotalSeconds:F1} s, tools {run.ToolTime.TotalSeconds:F1} s]");
            if (output is not null)
            {
                File.AppendAllText(output, typed.ToJson().ToJsonString() + "\n");
                Console.WriteLine($"transcript appended to {output}");
            }

            return 0;
        }

        var suite = CodingSuite.Load(positional[2], filter);
        await InstallDependencies(suite);
        string work = Path.GetFullPath(workRoot ?? Path.Combine(Path.GetTempPath(), "idrak-agent", DateTime.Now.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture)));
        Console.WriteLine($"{suite.Count} tasks × {attempts} attempts; work folders under {work}");
        var outcomes = new List<(CodingTask Task, CodingRun Run)>();
        var verifier = new CodingVerifier(new CodingVerifierOptions { Tools = agentOptions.Tools }) { OnStatus = s => Console.WriteLine($"    {s}") };
        var total = Stopwatch.StartNew();
        foreach (var task in suite)
        {
            for (int attempt = 1; attempt <= attempts && !cancel.IsCancellationRequested; attempt++)
            {
                var agent = new CodingAgent(chatModel, agentOptions with { Sampling = sampling with { Seed = attempt } });
                var watch = Stopwatch.StartNew();
                CodingRun run;
                try
                {
                    run = await verifier.RunAsync(agent, task, Path.Combine(work, task.Id.Replace('/', Path.DirectorySeparatorChar), $"attempt-{attempt}"), cancel.Token);
                }
                catch (OperationCanceledException)
                {
                    break;
                }

                outcomes.Add((task, run));
                Console.WriteLine($"  {task.Id} #{attempt}: {run.Outcome} — {run.Agent.Rounds} replies, {run.Agent.ToolCalls} tool calls ({run.Agent.ToolErrors} errors), {run.Agent.GeneratedTokens} tokens, "
                                  + $"{watch.Elapsed.TotalSeconds:F0} s (model {run.Agent.ModelTime.TotalSeconds:F0} s, tools {run.Agent.ToolTime.TotalSeconds:F0} s)");
                if (run.Outcome is CodingOutcome.SetupFailed or CodingOutcome.Error)
                {
                    Console.WriteLine("    " + run.VerifyOutput.Trim().Replace("\n", "\n    ", StringComparison.Ordinal));
                }

                File.AppendAllText(output!, run.ToJson().ToJsonString() + "\n");
            }
        }

        var scored = outcomes.Where(o => o.Run.Outcome is not (CodingOutcome.SetupFailed or CodingOutcome.Error)).ToList();
        int passed = scored.Count(o => o.Run.Outcome == CodingOutcome.Passed);
        Console.WriteLine($"\n{passed}/{scored.Count} runs passed ({(scored.Count == 0 ? 0 : 100.0 * passed / scored.Count):F1}%), "
                          + $"{outcomes.Count - scored.Count} not scored (setup failed or model error), {total.Elapsed:hh\\:mm\\:ss}");
        foreach (var group in scored.GroupBy(o => o.Task.Language ?? "other").OrderBy(g => g.Key, StringComparer.Ordinal))
        {
            int p = group.Count(o => o.Run.Outcome == CodingOutcome.Passed);
            Console.WriteLine($"  {group.Key}: {p}/{group.Count()} ({100.0 * p / group.Count():F1}%), "
                              + $"median {group.Select(o => o.Run.Agent.Rounds).Order().ElementAt(group.Count() / 2)} replies, {group.Average(o => o.Run.Agent.ToolErrors):F1} tool errors per run");
        }

        int tasksSolved = scored.GroupBy(o => o.Task.Id).Count(g => g.Any(o => o.Run.Outcome == CodingOutcome.Passed));
        Console.WriteLine($"  tasks solved at least once: {tasksSolved}/{scored.Select(o => o.Task.Id).Distinct().Count()}; runs written to {output}");
        return passed == scored.Count ? 0 : 2;
    }

    default:
        return 1;
}
