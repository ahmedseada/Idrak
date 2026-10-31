// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Reinforcement;
using Idrak.Cli.Commands.Cluster;
using Idrak.Cli.Shared;
using Idrak.Models;
using Idrak.Reinforcement;
using Idrak.Text;
using Idrak.Text.Abstractions;
using Idrak.Text.Reinforcement;
using Idrak.Text.Reinforcement.Abstractions;

namespace Idrak.Cli.Commands.Train;

/// <summary>
/// <c>idrak reinforce MODEL</c> (Plan 27, R5): reinforcement learning of a language model's adapters in a registered
/// environment (<c>EnvironmentKinds</c>: <c>prompts</c>, the coding plug-in's <c>coding</c>, a plug-in's), by the library's
/// trainer (<see cref="ReinforcementTrainer{TObservation, TAction}"/>) over Idrak.Text's language-model policy
/// (<see cref="LanguageModelPolicy"/>), with a teacher term beside the reward when asked; <c>--algorithm distill</c> is
/// on-policy distillation alone (Idrak.Text's <see cref="OnPolicyDistiller"/>, as <c>idrak distill --on-policy</c>). The
/// model, the adapters, the optimizer and the device are idrak tune's (its options, parsed by <see cref="TuneTool"/>).
/// Every run setting reads its <c>IDRAK_RL_*</c> variable (<see cref="ReinforcementOptions.FromEnvironment"/>); the
/// command line wins. On a cluster (a rank of <c>idrak cluster sim</c>, or <c>--cluster FILE</c>; Plan 27, R3c) every rank
/// runs the command: the session trains on the rank's device with its collectives (as idrak tune's), each rank plays its
/// shards' episodes (R3b), and rank 0 alone writes the adapter, its manifest, the checkpoints and the log and shows the
/// progress.
/// </summary>
internal sealed class ReinforceCommand : Command
{
    /// <summary>The algorithm that is on-policy distillation alone (no reward): not a policy loss.</summary>
    public const string Distill = "distill";

    /// <summary>The variable of <c>--env</c>.</summary>
    public const string EnvironmentVariable = "IDRAK_RL_ENV";

    /// <summary>The variable of <c>--reward</c>.</summary>
    public const string RewardVariable = "IDRAK_RL_REWARD";

    /// <summary>The variable of <c>--max-new</c>.</summary>
    public const string MaxNewVariable = "IDRAK_RL_MAX_NEW";

    /// <summary>The variable of <c>--teacher</c>.</summary>
    public const string TeacherVariable = "IDRAK_RL_TEACHER";

    /// <summary>The variable of <c>--teacher-device</c>.</summary>
    public const string TeacherDeviceVariable = "IDRAK_RL_TEACHER_DEVICE";

    // The run's own options (those of a fine-tuning session come from TuneTool).
    private static readonly string[] OwnValues =
    [
        "--base", "--env", "--data", "--suite", "--reward", "--rounds", "--env-option", "--sandbox", "--algorithm", "--estimator", "--group", "--episodes", "--kl",
        "--clip", "--epochs", "--minibatch", "--iterations", "--environments", "--temperature", "--max-new", "--save-every", "--evaluate-every", "--teacher",
        "--teacher-device", "--teacher-weights", "--teacher-weight", "--divergence", "--beta", "--mix", "--out", "--weights", "--shards", "--spread", "--kv",
        "--pairs", "--loss", "--pair-option", "--loss-option", "--reference",
    ];

    public override string Name => "reinforce";

    public override string Summary => "Reinforcement learning of a model's adapters: GRPO, PPO, on-policy distillation, DPO, ORPO, SimPO";

    public override string Usage => $$"""
        MODEL --env NAME [--data FILE | --suite DIR] -o DIR [options]
        --env NAME -o DIR [options]   (an environment of numbers: bandit, cartpole, pendulum, mountain-car; no MODEL)
        --list

        The model's adapters (LoRA, as idrak tune trains them) learn from what the model does in an environment: each
        iteration plays episodes with the model's own sampled replies, scores them, estimates each one's advantage and
        updates the adapters with the algorithm's loss, by the library's trainer. Checkpoints go to DIR/checkpoints after
        every --save-every iterations and at the end (or on Ctrl+C, after the current iteration); --resume goes on from the
        latest. The adapter is written to DIR (idrak tune chat DIR, idrak run MODEL --adapter DIR use it). Every iteration
        is a control point: the cluster dashboard (idrak cluster ui, cluster sim --dashboard) or a launcher's control file
        (IDRAK_CLUSTER_CONTROL, in one process too) pauses the run there, resumes it as if nothing happened, or stops it
        there with its checkpoint; the dashboard shows the iterations, rewards and pass rate from DIR/log.jsonl.

        Environments (--list shows the registered ones, a plug-in's among them):
          prompts     chat prompts with references (--data, conversations as idrak tune reads them: the last assistant
                      message is the reference), the reply scored by --reward (exact by default: a reward or metric name,
                      or weighted terms such as exact:1,length:0.1; a term's options with --env-option term.key=value)
          coding      coding tasks (--suite DIR: folders with task.json, workspace/, hidden verify/ tests); the model works
                      with the coding tools (read, edit, run commands in --sandbox), rewarded by the hidden tests
                      (tests-passed by default; verify-fraction, rounds, tool-errors)

        Options:
          -b, --base MODEL        the model as an option (a Hugging Face id, a folder, a .gguf file or an alias)
              --env NAME          the environment kind (default prompts; IDRAK_RL_ENV)
              --data FILE         the prompts (repeatable): files, folders, hf: specs, as idrak tune reads them
              --suite DIR         the coding environment's task suite
              --reward EXPR       the reward expression (IDRAK_RL_REWARD; default: the environment's)
              --rounds N          the most replies of an episode (prompts 1, coding 40)
              --sandbox NAME      where coding commands run: process (default) or container (CommandSandboxes; IDRAK_SANDBOX)
              --env-option K=V    any option of the environment (repeatable), e.g. filter=csharp, refuse-test-edits=true,
                                  metric.metric=exact
              --algorithm NAME    grpo, ppo, reinforce or a plug-in's (PolicyLosses), or distill: on-policy distillation
                                  alone, with --teacher, on prompts (default ppo; IDRAK_RL_ALGORITHM)
              --estimator NAME    the advantage estimator: gae, group, leave-one-out, returns (default: the algorithm's)
              --group N           episodes of a group, each the same situation played again (default: 8, 4 or 2)
              --episodes N        episodes of an iteration, a multiple of the group (default 16)
              --kl B              the weight of the KL penalty to the base model (ppo, grpo; default 0)
              --clip E            the ratio's clip (ppo, grpo; default 0.2)
              --epochs N          passes over each iteration's episodes (default: ppo 4, else 1)
              --minibatch N       episodes of each update (default: all of the iteration)
              --iterations N      the iterations of the run (default 100)
              --environments N    environments stepped at once (default: the CPU threads)
              --temperature T     the temperature replies are sampled at (default 1)
              --max-new N         the longest reply in tokens (default 256; IDRAK_RL_MAX_NEW)
              --kv FORMAT         the key/value cache replies are generated with: float32, bfloat16 (half the memory),
                                  int8 (default float32; IDRAK_RL_KV)
              --no-think          replies without the template's reasoning
              --save-every N      a checkpoint after every N iterations (default: at the end only)
              --evaluate-every N  greedy evaluation episodes after every N iterations (default: none)
              --teacher MODEL     a teacher model of the same vocabulary: its next-token distributions on the model's own
                                  replies, a term beside the reward (on-policy distillation; IDRAK_RL_TEACHER)
              --teacher-device D  the teacher's device (default: the model's; IDRAK_RL_TEACHER_DEVICE)
              --teacher-weights F the teacher's weights: int8, int4 or bf16 (default: as stored)
              --teacher-weight W  the teacher term's weight (default 1 with --teacher; IDRAK_RL_TEACHER_WEIGHT)
              --divergence NAME   forward, reverse (default) or jsd (TeacherDivergences; IDRAK_RL_DIVERGENCE)
              --beta B            jsd's weight, in (0, 1) (default 0.5)
              --mix L             with --algorithm distill: the share of a step's prompts trained on the data's answers
              --pairs NAME        preference rounds instead of a policy loss: each iteration's episodes become pairs by
                                  the source NAME (PreferencePairSources: group, the best and the worst answer of each
                                  group by reward; verdict, an answer the episode records as rejected against the text
                                  kept instead) and the adapters train on them with --loss (IDRAK_RL_PAIRS)
              --loss NAME         the preference loss of --pairs: dpo (default), orpo, simpo or an app's
                                  (PreferenceLosses; IDRAK_RL_PREFERENCE_LOSS)
              --pair-option K=V   an option of the pair source (repeatable), e.g. margin=0.5 (IDRAK_RL_PAIR_OPTIONS)
              --loss-option K=V   an option of the preference loss (repeatable), e.g. beta=0.2 (IDRAK_RL_PREFERENCE_LOSS_OPTIONS)
              --reference NAME    what the preference loss trains against: previous (the adapter as the iteration began,
                                  the default) or base (IDRAK_RL_PREFERENCE_REFERENCE)
              --resume            go on from the latest checkpoint in DIR/checkpoints (IDRAK_RL_RESUME: another one)
          -o, --out DIR           where the adapter and the checkpoints go
              --list              the registered environments, rewards, algorithms, estimators, sandboxes, divergences,
                                  pair sources and preference losses
          -w, --weights F         the model's base weights: int8, int4 (QLoRA) or bf16
              --shards N          the shards a rollout is cut into: a shard's episodes act together, and on a cluster each
                                  rank plays its shards' episodes; the same count gives the same episodes on any number
                                  of ranks (default 0: one per rank, so one in one process; IDRAK_RL_SHARDS)
              --spread NAME       how the episodes are cut into shards: blocks (runs in number order, a group's together;
                                  the default) or interleaved (IDRAK_RL_SPREAD)
              --cluster FILE      train as one rank of a real cluster of the topology file (idrak cluster init writes
                                  one); on a node of several workers IDRAK_RANK names this process's (--rank stays the
                                  adapter's rank); --node NAME, --coordinator H:P and --world-size N as idrak tune has them
          and these options of idrak tune train: --rank, --alpha, --lr, --max-length, --batch-tokens, --accumulate,
          --targets, --optimizer, --weight-decay, --momentum, --schedule (default: IDRAK_RL_SCHEDULE's, constant),
          --warmup (default: IDRAK_RL_WARMUP's, 0), --min-lr, --decay, --adapter-type, --adapter, --context, --matmul,
          --gpu-memory, --no-packing, --checkpointing, --no-checkpointing, --recompute, --bf16-activations, --no-activation-estimate, --no-graphs,
          --fp8, --cpu-optimizer, --offload. Every run setting also reads its IDRAK_RL_* variable (idrak help env); the
          command line wins. A JSON line per iteration goes to DIR/log.jsonl (IDRAK_RL_LOG: another file; the pass rate
          counts the episodes whose return reaches IDRAK_RL_PASS_RETURN, 1). --json prints one JSON document at the end.
        {{ReinforceNumeric.UsageOptions}}

        On a cluster (idrak cluster sim --topology FILE -- reinforce ..., or --cluster FILE) every rank runs the command on
        its device: each rollout's episodes are spread over the ranks by shards and gathered, every rank makes the same
        updates (the gradients summed across the ranks), and rank 0 alone writes the adapter, its manifest, the checkpoints
        and the log and shows the progress; --resume goes on from the checkpoint on every rank. --algorithm distill cuts
        each step's prompts into the same shards (each rank samples its shards', the samples are exchanged), as idrak
        distill --on-policy does on a cluster.

        Examples:
          idrak reinforce Qwen/Qwen3-0.6B --env prompts --data math.jsonl --reward exact --algorithm grpo --group 8 -o runs/math
          idrak reinforce Qwen/Qwen3-0.6B --env coding --suite tasks --algorithm grpo --group 8 --iterations 20 -o runs/rl-demo
          idrak reinforce Qwen/Qwen3-0.6B --env coding --suite tasks --sandbox container -o runs/rl --resume
          idrak reinforce Qwen/Qwen3-0.6B --env coding --suite tasks --algorithm grpo --teacher Qwen/Qwen3-8B -w int4 -o runs/t
          idrak reinforce Qwen/Qwen3-0.6B --env prompts --data chats.jsonl --algorithm distill --teacher Qwen/Qwen3-8B -o runs/d
          idrak cluster sim --topology two.json -- reinforce Qwen/Qwen3-0.6B --data m.jsonl --algorithm grpo --shards 4 -o rl
          idrak reinforce Qwen/Qwen3-0.6B --env prompts --data math.jsonl --reward exact --pairs group --loss dpo --group 4 -o runs/dpo
          idrak reinforce --list
        {{ReinforceNumeric.UsageExamples}}
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } = [.. OwnValues, .. TuneTool.SessionValueOptions, .. ClusterJoin.Options, .. ReinforceNumeric.ValueOptions];

    public override IReadOnlyCollection<string> Flags { get; } = ["--list", "--resume", "--no-think", .. TuneTool.SessionFlags, .. ReinforceNumeric.Flags];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string>
    {
        ["-b"] = "--base", ["-o"] = "--out", ["-w"] = "--weights",
    };

    public override int Run(CommandContext context)
    {
        using var cluster = ClusterJoin.Join(context, ownsRank: true);    // --cluster FILE: one rank of a real cluster; --rank is the adapter's
        if (context.Flag("--list"))
        {
            return List(context);
        }

        if (ReinforceNumeric.Kind(context) is { } numeric)
        {
            return ReinforceNumeric.Run(context, numeric);                   // an environment of numbers (R9): no model, a module policy or DQN
        }

        // The model and the output.
        var positional = context.Positional.ToList();
        string model = context.Option("--base") ?? (positional.Count > 0 ? positional[0] : null)
            ?? throw new UsageException("Give the model: idrak reinforce MODEL --env NAME -o DIR (or -b, --base MODEL); --list shows what can be chosen.");
        if (positional.Count > (context.Option("--base") is null ? 1 : 0))
        {
            throw new UsageException($"reinforce takes one model ('{positional[^1]}' is extra): give the data with --data, a suite with --suite.");
        }

        string output = context.Option("--out") ?? throw new UsageException("Give -o, --out DIR for the adapter and the checkpoints.");

        // The run's settings: the variables, then the command line.
        var options = Settings(context, output);
        bool distill = options.Algorithm.Equals(Distill, StringComparison.OrdinalIgnoreCase);

        // The environment and its options.
        string envName = context.Option("--env") ?? Text(EnvironmentVariable) ?? PromptsEnvironmentKind.KindName;
        IEnvironmentKind kind;
        try
        {
            kind = EnvironmentKinds.Get(envName);
        }
        catch (NotSupportedException e)
        {
            throw new UsageException($"--env {envName}: {e.Message}");
        }

        if (kind.ObservationType != typeof(ChatObservation) || kind.ActionType != typeof(ChatMessage))
        {
            throw new UsageException($"--env {kind.Name}: its environments observe {kind.ObservationType.Name} and act with {kind.ActionType.Name}; a language model observes "
                                     + "conversations and acts with replies (ChatObservation, ChatMessage).");
        }

        var promptData = kind as IPromptDataEnvironmentKind;                // its tasks are the prompts of --data
        bool prompts = promptData is not null;
        var data = context.Options("--data");
        var environmentOptions = EnvironmentOptions(context, kind, prompts, data);
        if (prompts && data.Count == 0)
        {
            throw new UsageException($"The {kind.Name} environment answers the prompts of --data FILE (conversations, as idrak tune reads them).");
        }

        if (distill && options.Pairs is not null)
        {
            throw new UsageException($"--algorithm {Distill} learns from a teacher, --pairs from preferences: give one of them.");
        }

        if (distill && !prompts)
        {
            throw new UsageException($"--algorithm {Distill} is on-policy distillation alone, over the prompts of --data (--env prompts); in the {kind.Name} environment "
                                     + "give a policy loss (grpo, ppo, reinforce) with --teacher: the teacher's term then goes beside the reward.");
        }

        // The teacher, its term and the policy's settings.
        string? teacher = context.Option("--teacher") ?? Text(TeacherVariable);
        string? teacherDeviceName = context.Option("--teacher-device") ?? Text(TeacherDeviceVariable);
        Device? teacherDevice = teacherDeviceName is null ? null : ParseDevice(teacherDeviceName);
        var teacherChoice = teacher is null ? null : new ModelChoices.ModelChoice(teacher, context.Option("--teacher-weights"), null, null, null);
        if (teacherChoice?.Weights?.ToLowerInvariant() is not (null or "int8" or "int4" or "bf16" or "bfloat16"))
        {
            throw new UsageException($"--teacher-weights {teacherChoice!.Weights}: use int8, int4 or bf16.");
        }

        var policyOptions = PolicyOptions(context, teacher is not null);
        if (distill && teacher is null)
        {
            throw new UsageException($"--algorithm {Distill} learns from a teacher: give --teacher MODEL.");
        }

        if (distill && (context.Flag("--resume") || options.Resume is not null))
        {
            throw new UsageException($"--algorithm {Distill} runs as idrak distill --on-policy does, without checkpoints to resume from: give a policy loss to --resume.");
        }

        var onPolicy = distill ? DistillCommand.OnPolicyOptions(context) : null;

        // The model, the adapters, the optimizer and the device: idrak tune's options, parsed by its tool.
        var captured = context.Json ? new StringWriter() : null;
        var console = ToolHost.Console(context, captured ?? (context.Quiet ? TextWriter.Null : context.Output));
        var choice = ModelChoices.Choose(context, model);
        var args = TuneArguments.Train(context, choice, prompts ? data : [], output, "the model");
        TuneArguments.Pass(context, args, TuneTool.SessionValueOptions, TuneTool.SessionFlags);
        var tool = new TuneTool(console) { Verbose = context.Verbose };
        try
        {
            tool.Parse(args);
        }
        catch (Exception e) when (e is ArgumentException or FormatException)
        {
            throw new UsageException(e.Message);
        }

        PretrainedModel? teacherModel = null;
        IDistillationTeacher? source = null;
        ReinforcementResult? result = null;
        IReadOnlyList<float> distilled = [];
        tool.Session = (loaded, tuning) =>
        {
            tuning = RunSchedule(context, options, tuning);
            if (teacherChoice is not null)
            {
                (teacherModel, source) = DistillCommand.LoadTeacher(context, console, teacherChoice, teacherDevice ?? tool.Device, 0, tuning.BatchTokens);
            }

            int code;
            if (distill)
            {
                (code, distilled) = DistillCommand.TrainOnPolicy(context, console, loaded, tuning, source!, tool.ReadChats(), onPolicy! with
                {
                    Steps = options.Iterations, PromptsPerStep = options.EpisodesPerIteration, MaxLength = tuning.MaxLength, MaxNewTokens = policyOptions.MaxNewTokens,
                    SamplingTemperature = options.Temperature, Seed = options.Seed, Divergence = policyOptions.Divergence, Beta = policyOptions.Beta,
                    Shards = options.Shards, Spread = options.Spread,              // R10: a step's prompts in R3b's shards, on a cluster too
                }, output);
                return code;
            }

            (code, result) = Train(context, console, tool, loaded, tuning, source, kind, prompts, options with { EnvironmentOptions = environmentOptions }, policyOptions, output);
            return code;
        };

        if (tool.Problem() is { } problem)
        {
            throw new UsageException(problem);
        }

        int exit;
        try
        {
            context.Detail($"tune {string.Join(' ', args)}");
            exit = tool.Execute();
        }
        finally
        {
            source?.Dispose();
            teacherModel?.Dispose();
        }

        if (captured is not null)
        {
            var json = new JsonObject
            {
                ["model"] = choice.Model,
                ["environment"] = kind.Name,
                ["algorithm"] = options.Algorithm,
                ["pairs"] = options.Pairs,
                ["loss"] = options.Pairs is null ? null : options.PreferenceLoss,
                ["out"] = Path.GetFullPath(output),
                ["device"] = tool.Device.ToString(),
                ["exitCode"] = exit,
            };
            if (result is not null)
            {
                json["iterations"] = result.Iterations;
                json["cancelled"] = result.Cancelled;
                json["checkpoint"] = result.Checkpoint;
                json["history"] = new JsonArray([.. result.History.Select(h => (JsonNode)new JsonObject
                {
                    ["iteration"] = h.Iteration,
                    ["episodes"] = h.Episodes,
                    ["rewardMean"] = h.RewardMean,
                    ["rewardSpread"] = h.RewardSpread,
                    ["loss"] = h.Loss,
                    ["updates"] = h.Updates,
                    ["evaluation"] = h.EvaluationReward,
                    ["returns"] = new JsonArray([.. h.Returns.Select(r => (JsonNode)r)]),
                })]);
            }

            if (distill)
            {
                json["losses"] = new JsonArray([.. distilled.Select(l => (JsonNode)l)]);
            }

            json["output"] = new JsonArray([.. captured.ToString().Split('\n').Select(l => l.TrimEnd('\r')).Where(l => l.Length > 0).Select(l => (JsonNode)l)]);
            context.WriteJson(json);
        }

        return exit;
    }

    // The run: a session on the loaded model (with the teacher), the policy over it, the trainer in the environment; the
    // adapter written to `output` at the end, also after a stop.
    private static (int Code, ReinforcementResult Result) Train(CommandContext context, ToolConsole console, TuneTool tool, PretrainedModel model, FineTuningOptions tuning,
        IDistillationTeacher? teacher, IEnvironmentKind kind, bool prompts, ReinforcementOptions options, LanguageModelPolicyOptions policyOptions, string output)
    {
        // The prompts environment over the data as idrak tune reads it, counting each answer's tokens for a length reward.
        string tasks = "";
        if (prompts)
        {
            var chats = tool.ReadChats();
            if (chats.Count == 0)
            {
                throw new InvalidOperationException("The data holds no conversation to take prompts from.");
            }

            kind = ((IPromptDataEnvironmentKind)kind).WithTasks(chats, model.Tokenizer);
            tasks = $" ({chats.Count:N0} prompts)";
        }

        bool pairs = options.Pairs is not null;
        int epochs = options.Epochs > 0 ? options.Epochs : !pairs && options.Algorithm.Equals(ReinforcementNames.Ppo, StringComparison.OrdinalIgnoreCase) ? 4 : 1;
        int miniBatch = options.MiniBatch > 0 ? Math.Min(options.MiniBatch, options.EpisodesPerIteration) : options.EpisodesPerIteration;
        int totalSteps = Math.Max(1, options.Iterations * epochs * (pairs ? 1 : (options.EpisodesPerIteration + miniBatch - 1) / miniBatch));
        string algorithm = pairs ? $"{options.PreferenceLoss} on {options.Pairs} pairs against the {options.PreferenceReference} adapter" : options.Algorithm;
        console.Out.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"reinforce: {algorithm} in {kind.Name}{tasks}, {options.EpisodesPerIteration} episodes an iteration{(options.GroupSize > 0 ? $" in groups of {options.GroupSize}" : "")}, "
            + $"{options.Iterations} iterations, replies of at most {policyOptions.MaxNewTokens} tokens at temperature {options.Temperature}"
            + $"{(teacher is null ? "" : FormattableString.Invariant($"; teacher term {policyOptions.TeacherWeight} ({policyOptions.Divergence})"))}"
            + $"{(options.Resume is { } resume ? $"; resuming from {resume}" : "")}"));
        var session = new FineTuningSession(model, teacher is null ? tuning : tuning with { Teacher = teacher }, totalSteps, trace: context.Verbose ? console.Log : null);
        LanguageModelPolicy policy;
        try
        {
            policy = new LanguageModelPolicy(session, policyOptions);
        }
        catch
        {
            session.Dispose();
            throw;
        }

        using (policy)
        {
            using var stop = new StopKey(context, console.Log, "stopping after the current iteration and saving the adapter (Ctrl+C again to quit now)…");
            // On a cluster rank 0 shows the progress (the other ranks' lines would only be kept as their error text).
            using var progress = new ProgressLine(context, $"reinforce {kind.Name}", options.Iterations, enabled: ClusterSession.Current is { Rank: > 0 } ? false : null);
            var trainer = new ReinforcementTrainer<ChatObservation, ChatMessage>(kind, policy, options)
            {
                Progress = progress,
                Steering = Steering(),                                         // the dashboard's pause, resume and stop at each iteration (R11)
                OnIteration = s =>
                {
                    progress.Erase();
                    console.Out.WriteLine(string.Create(CultureInfo.InvariantCulture,
                        $"iteration {s.Iteration}/{options.Iterations}: reward {s.RewardMean:0.###} ± {s.RewardSpread:0.###} ({s.RewardMin:0.###} to {s.RewardMax:0.###}), "
                        + $"loss {s.Loss:0.####}, {s.Updates} update{(s.Updates == 1 ? "" : "s")}"
                        + $"{(s.Metrics.TryGetValue("pairs", out double made) ? FormattableString.Invariant($" on {made:0} pairs") : "")}"
                        + $"{(s.Metrics.TryGetValue("margin", out double margin) ? FormattableString.Invariant($" (margin {margin:0.###})") : "")}{(s.EvaluationReward is { } e ? FormattableString.Invariant($", evaluation {e:0.###}") : "")}, "
                        + $"rollout {s.RolloutTime.TotalSeconds:0.0} s, update {s.UpdateTime.TotalSeconds:0.0} s"));
                },
            };
            var clock = System.Diagnostics.Stopwatch.StartNew();
            var result = trainer.Train(stop.Token);
            bool wrote = session.SaveOnce(output);                             // every rank takes part; rank 0 writes on a cluster
            var history = result.History;
            console.Out.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"{(result.Cancelled ? "stopped after" : "trained")} {result.Iterations} iterations in {clock.Elapsed.TotalSeconds:0.0} s"
                + $"{(history.Count > 0 ? FormattableString.Invariant($": reward {history[0].RewardMean:0.###} → {history[^1].RewardMean:0.###}") : "")}"
                + $"; adapter {(wrote ? "written" : "written by rank 0")} to {Path.GetFullPath(output)}{(result.Checkpoint is { } c ? $", checkpoint {c}" : "")}"));
            return (ExitCodes.Ok, result);
        }
    }

    /// <summary>The run's JSON log in DIR (a line per iteration; IDRAK_RL_LOG names another file): the dashboard shows the run from it.</summary>
    public const string LogFile = "log.jsonl";

    /// <summary>
    /// The run's control point at its boundaries (Plan 27, R11): the dashboard's in-process control of this cluster rank, or
    /// the control file a launcher set (IDRAK_CLUSTER_CONTROL; in one process too); null when nothing steers the run.
    /// </summary>
    internal static global::Idrak.Abstraction.Cluster.IRunSteering? Steering()
    {
        var rank = ClusterSession.Current;
        try
        {
            return global::Idrak.Abstraction.Cluster.RunSteerings.For(rank?.Collectives, rank?.Context.Options ?? global::Idrak.Abstraction.Cluster.ClusterOptions.FromEnvironment());
        }
        catch (ArgumentException e)
        {
            throw new UsageException(e.Message);
        }
    }

    // The run's settings: ReinforcementOptions.FromEnvironment, then each option given on the command line in its place.
    internal static ReinforcementOptions Settings(CommandContext context, string output)
    {
        ReinforcementOptions o;
        try
        {
            o = ReinforcementOptions.FromEnvironment();
        }
        catch (ArgumentException e)
        {
            throw new UsageException(e.Message);
        }

        o = context.Option("--algorithm") is { } algorithm ? o with { Algorithm = algorithm } : o;
        o = context.Option("--estimator") is { } estimator ? o with { Estimator = estimator } : o;
        o = Int(context, "--group", 1) is { } group ? o with { GroupSize = group } : o;
        o = Int(context, "--episodes", 1) is { } episodes ? o with { EpisodesPerIteration = episodes } : o;
        o = Float(context, "--kl", 0f) is { } kl ? o with { KlWeight = kl } : o;
        o = Float(context, "--clip", 0f, 1f) is { } clip ? o with { Clip = clip } : o;
        o = Int(context, "--epochs", 1) is { } epochs ? o with { Epochs = epochs } : o;
        o = Int(context, "--minibatch", 1) is { } miniBatch ? o with { MiniBatch = miniBatch } : o;
        o = Int(context, "--iterations", 1) is { } iterations ? o with { Iterations = iterations } : o;
        o = Int(context, "--environments", 1) is { } environments ? o with { Environments = environments } : o;
        o = Float(context, "--temperature", float.Epsilon) is { } temperature ? o with { Temperature = temperature } : o;
        o = Int(context, "--save-every", 1) is { } saveEvery ? o with { SaveEvery = saveEvery } : o;
        o = Int(context, "--evaluate-every", 1) is { } evaluateEvery ? o with { EvaluateEvery = evaluateEvery } : o;
        o = context.Seed is { } seed ? o with { Seed = seed } : o;
        o = Int(context, "--shards", 0) is { } shards ? o with { Shards = shards } : o;
        o = context.Option("--spread") is { } spread ? o with { Spread = Spread(spread) } : o;
        o = o with { CheckpointFolder = o.CheckpointFolder ?? Path.Combine(output, "checkpoints"), Log = o.Log ?? Path.Combine(output, LogFile) };
        o = context.Flag("--resume") ? o with { Resume = o.CheckpointFolder } : o;
        o = context.Option("--pairs") is { } pairs ? o with { Pairs = pairs } : o;
        o = context.Option("--loss") is { } preferenceLoss ? o with { PreferenceLoss = preferenceLoss } : o;
        o = context.Option("--reference") is { } reference ? o with { PreferenceReference = Reference(reference) } : o;
        o = KeyValues(context, "--pair-option") is { } pairOptions ? o with { PairOptions = pairOptions } : o;
        o = KeyValues(context, "--loss-option") is { } lossOptions ? o with { PreferenceLossOptions = lossOptions } : o;
        if (o.Pairs is null && (context.Option("--loss") ?? context.Option("--reference")) is not null || o.Pairs is null && context.Options("--loss-option").Count + context.Options("--pair-option").Count > 0)
        {
            throw new UsageException("--loss, --reference, --pair-option and --loss-option set the preference rounds of --pairs NAME: give --pairs (group, verdict).");
        }

        if (o.Pairs is { } source)
        {
            var sourceKind = PreferencePairSources.Find(source) ?? throw new UsageException(
                $"{(context.Option("--pairs") is null ? "IDRAK_RL_PAIRS=" : "--pairs ")}{source}: use {string.Join(", ", PreferencePairSources.Names)} (PreferencePairSources).");
            var loss = PreferenceLosses.Find(o.PreferenceLoss) ?? throw new UsageException(
                $"{(context.Option("--loss") is null ? "IDRAK_RL_PREFERENCE_LOSS=" : "--loss ")}{o.PreferenceLoss}: use {string.Join(", ", PreferenceLosses.Names)} (PreferenceLosses).");
            try
            {
                _ = sourceKind.Options.Parse(o.PairOptions);
                _ = loss.Options.Parse(o.PreferenceLossOptions);
            }
            catch (ArgumentException e)
            {
                throw new UsageException(e.Message);
            }
        }

        // The names, checked by their registries before anything loads.
        if (o.Pairs is null && !o.Algorithm.Equals(Distill, StringComparison.OrdinalIgnoreCase) && PolicyLosses.Find(o.Algorithm) is null)
        {
            throw new UsageException($"{(context.Option("--algorithm") is null ? "IDRAK_RL_ALGORITHM=" : "--algorithm ")}{o.Algorithm}: use {string.Join(", ", PolicyLosses.Names)} (PolicyLosses), or {Distill} (on-policy distillation alone).");
        }

        if (o.Estimator is { } named && AdvantageEstimators.Find(named) is null)
        {
            throw new UsageException($"{(context.Option("--estimator") is null ? "IDRAK_RL_ESTIMATOR=" : "--estimator ")}{named}: use {string.Join(", ", AdvantageEstimators.Names)} (AdvantageEstimators).");
        }

        if (LearningRateSchedules.Find(o.Schedule) is null)
        {
            throw new UsageException($"IDRAK_RL_SCHEDULE={o.Schedule}: use {string.Join(", ", LearningRateSchedules.Names)}.");
        }

        if (o.GroupSize > 0 && o.EpisodesPerIteration % o.GroupSize != 0)
        {
            throw new UsageException(string.Create(CultureInfo.InvariantCulture,
                $"--episodes {o.EpisodesPerIteration} is not a multiple of --group {o.GroupSize}: each group plays one situation, so the episodes fill whole groups."));
        }

        if (o.Resume is { } resume && !File.Exists(Path.Combine(resume, "state.json"))
            && !(Directory.Exists(resume) && Directory.EnumerateDirectories(resume, "iteration-*").Any(d => File.Exists(Path.Combine(d, "state.json")))))
        {
            throw new UsageException($"Nothing to resume: no checkpoint in {Path.GetFullPath(resume)}.");
        }

        return o;
    }

    // The environment's options: --reward, --rounds and --no-think (under the keys a rewarded kind declares,
    // IRewardedEnvironmentKind), --suite and --sandbox (a suite kind's, ISuiteEnvironmentKind), --data (for a kind that
    // reads a data file itself) and every --env-option, each checked against the kind's keys (a reward term's options hold
    // a dot). A kind that does not declare a key gets the flag under its usual name, which its keys then refuse.
    private static Dictionary<string, string> EnvironmentOptions(CommandContext context, IEnvironmentKind kind, bool prompts, IReadOnlyList<string> data)
    {
        var rewarded = kind as IRewardedEnvironmentKind;
        var suites = kind as ISuiteEnvironmentKind;
        var options = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (string item in context.Options("--env-option"))
        {
            int equals = item.IndexOf('=', StringComparison.Ordinal);
            if (equals <= 0)
            {
                throw new UsageException($"--env-option {item}: give KEY=VALUE (the {kind.Name} environment's keys: {string.Join(", ", kind.Keys)}).");
            }

            options[item[..equals].Trim()] = item[(equals + 1)..];
        }

        if ((context.Option("--reward") ?? Text(RewardVariable)) is { } reward)
        {
            options[rewarded?.RewardKey ?? "reward"] = reward;
        }

        if (context.Option("--suite") is { } suite)
        {
            options[suites?.SuiteKey ?? "suite"] = suite;
        }

        if (context.Option("--rounds") is not null)
        {
            options[rewarded?.RoundsKey ?? "rounds"] = context.IntOption("--rounds", 1) is > 0 and var rounds ? rounds.ToString(CultureInfo.InvariantCulture)
                : throw new UsageException("--rounds is a positive whole number.");
        }

        if (context.Option("--sandbox") is { } sandbox)
        {
            options[suites?.SandboxKey ?? "sandbox"] = CommandSandboxes.Find(sandbox)?.Name ?? throw new UsageException($"--sandbox {sandbox}: use {string.Join(", ", CommandSandboxes.Names)} (CommandSandboxes).");
        }

        if (context.Flag("--no-think") && rewarded?.ThinkKey is { } think)
        {
            options[think] = "false";
        }

        if (!prompts && data.Count > 0)
        {
            options["data"] = data.Count == 1 ? data[0] : throw new UsageException($"The {kind.Name} environment reads one data file: give --data once.");
        }

        // The prompts environment's tasks are the data as idrak tune reads it (not its own data option).
        string? ownData = (kind as IPromptDataEnvironmentKind)?.DataKey;
        var keys = prompts ? kind.Keys.Where(k => k != ownData).ToList() : [.. kind.Keys];
        foreach (string key in options.Keys)
        {
            if (!key.Contains('.', StringComparison.Ordinal) && !keys.Contains(key))
            {
                throw new UsageException($"The {kind.Name} environment does not take '{key}'{(prompts && key == ownData ? " as an option: give --data FILE" : "")}; it takes: "
                                         + $"{string.Join(", ", keys)}, and a reward term's options as term.key.");
            }
        }

        if (suites?.SuiteKey is { } suiteKey && !options.ContainsKey(suiteKey))
        {
            throw new UsageException($"The {kind.Name} environment plays the tasks of a suite: give --suite DIR (folders with task.json).");
        }

        return options;
    }

    // The language-model policy's settings: the longest reply, the reasoning mode, the teacher term. The library's defaults
    // read their variables (IDRAK_RL_TURNS, IDRAK_RL_BEHAVIOR, IDRAK_RL_TEACHER_WEIGHT, IDRAK_RL_DIVERGENCE).
    private static LanguageModelPolicyOptions PolicyOptions(CommandContext context, bool teacher)
    {
        LanguageModelPolicyOptions o;
        try
        {
            o = new LanguageModelPolicyOptions();
        }
        catch (ArgumentException e)
        {
            throw new UsageException(e.Message);
        }

        int maxNew = context.Option("--max-new") is not null ? context.IntOption("--max-new", 0)
            : Text(MaxNewVariable) is { } fromVariable ? int.TryParse(fromVariable, NumberStyles.Integer, CultureInfo.InvariantCulture, out int n) ? n
                : throw new UsageException($"{MaxNewVariable}={fromVariable} is not a whole number.")
            : o.MaxNewTokens;
        if (maxNew < 1)
        {
            throw new UsageException("--max-new is a positive whole number.");
        }

        float weight = Float(context, "--teacher-weight", 0f) ?? (teacher && o.TeacherWeight == 0f ? 1f : o.TeacherWeight);
        if (weight > 0f && !teacher)
        {
            throw new UsageException("A teacher weight needs a teacher: give --teacher MODEL (or leave out --teacher-weight and IDRAK_RL_TEACHER_WEIGHT).");
        }

        string divergence = context.Option("--divergence") ?? o.Divergence;
        float beta = Float(context, "--beta", 0f, 1f) ?? o.Beta;
        if (beta is <= 0f or >= 1f)
        {
            throw new UsageException("--beta is between 0 and 1 (both left out).");
        }

        string cacheFormat = context.Option("--kv") ?? o.CacheFormat;
        if (!KeyValueLayouts.Names.Contains(cacheFormat, StringComparer.OrdinalIgnoreCase))
        {
            throw new UsageException($"--kv {cacheFormat}: unknown key/value cache format ({string.Join(", ", KeyValueLayouts.Names)}; IDRAK_RL_KV).");
        }

        return o with
        {
            CacheFormat = cacheFormat,
            MaxNewTokens = maxNew,
            Think = context.Flag("--no-think") ? false : o.Think,
            TeacherWeight = weight,
            Divergence = TeacherDivergences.Find(divergence)?.Name ?? throw new UsageException($"--divergence {divergence}: use {string.Join(", ", TeacherDivergences.Names)}."),
            Beta = beta,
        };
    }

    // The session's schedule: --schedule as idrak tune reads it, else the run's (IDRAK_RL_SCHEDULE, constant by default,
    // with IDRAK_RL_WARMUP's warm-up unless --warmup is given).
    private static FineTuningOptions RunSchedule(CommandContext context, ReinforcementOptions options, FineTuningOptions tuning)
    {
        if (context.Option("--schedule") is not null)
        {
            return tuning;
        }

        float warmup = context.Option("--warmup") is null ? options.ScheduleSettings.WarmupFraction : tuning.WarmupFraction;
        float decay = Float(context, "--decay", 0f, 1f) ?? options.ScheduleSettings.DecayFraction;
        return tuning with
        {
            WarmupFraction = warmup,
            Scheduler = LearningRateSchedules.Factory(options.Schedule, new LearningRateScheduleSettings(warmup, tuning.MinLearningRate, decay)),
        };
    }

    // --list: every registry's entries, with their summaries (and the options each takes).
    private static int List(CommandContext context)
    {
        var sections = new (string Name, string Registry, IReadOnlyList<(string Name, string Summary, IReadOnlyCollection<string> Keys)> Entries)[]
        {
            ("environments", nameof(EnvironmentKinds), [.. EnvironmentKinds.Names.Select(n => EnvironmentKinds.Get(n)).Select(k => (k.Name, k.Summary, k.Keys))]),
            ("rewards", nameof(RewardFunctions), [.. RewardFunctions.Names.Select(n => RewardFunctions.Get(n)).Select(r => (r.Name, r.Summary, r.Keys))]),
            ("algorithms", nameof(PolicyLosses), [.. PolicyLosses.Names.Select(n => PolicyLosses.Get(n)).Select(l => (l.Name, l.Summary, l.Keys)),
                (Distill, "on-policy distillation alone: the model's own replies to the prompts, the teacher's distributions on them (--teacher; --env prompts)", (IReadOnlyCollection<string>)[])]),
            ("off-policy", nameof(Idrak.Reinforcement.Abstractions.QLosses), [.. Idrak.Reinforcement.Abstractions.QLosses.Names.Select(n => Idrak.Reinforcement.Abstractions.QLosses.Get(n)).Select(l => (l.Name, l.Summary, l.Keys))]),
            ("estimators", nameof(AdvantageEstimators), [.. AdvantageEstimators.Names.Select(n => AdvantageEstimators.Get(n)).Select(e => (e.Name, e.Summary, e.Keys))]),
            ("sandboxes", nameof(CommandSandboxes), [.. CommandSandboxes.Names.Select(n => CommandSandboxes.Get(n)).Select(s => (s.Name, s.Summary, (IReadOnlyCollection<string>)[]))]),
            ("divergences", nameof(TeacherDivergences), [.. TeacherDivergences.Names.Select(n => TeacherDivergences.Get(n)).Select(d => (d.Name, d.Summary, (IReadOnlyCollection<string>)[]))]),
            ("pairs", nameof(PreferencePairSources), [.. PreferencePairSources.Names.Select(n => PreferencePairSources.Get(n)).Select(p => (p.Name, p.Summary, p.Keys))]),
            ("preference-losses", nameof(PreferenceLosses), [.. PreferenceLosses.Names.Select(n => PreferenceLosses.Get(n)).Select(l => (l.Name, l.Summary, l.Keys))]),
        };
        if (context.Json)
        {
            var json = new JsonObject();
            foreach (var (name, _, entries) in sections)
            {
                json[name] = new JsonArray([.. entries.Select(e => (JsonNode)new JsonObject
                {
                    ["name"] = e.Name,
                    ["summary"] = e.Summary,
                    ["keys"] = new JsonArray([.. e.Keys.Select(k => (JsonNode)k)]),
                })]);
            }

            context.WriteJson(json);
            return ExitCodes.Ok;
        }

        foreach (var (name, registry, entries) in sections)
        {
            context.Write($"{name} ({registry}):");
            int width = entries.Count == 0 ? 0 : entries.Max(e => e.Name.Length);
            foreach (var e in entries)
            {
                context.Write($"  {e.Name.PadRight(width)}  {e.Summary}{(e.Keys.Count > 0 ? $" [{string.Join(", ", e.Keys)}]" : "")}");
            }

            context.Write("");
        }

        return ExitCodes.Ok;
    }

    // --reference: a PreferenceReferences name.
    private static string Reference(string text)
    {
        try
        {
            return PreferenceReferences.Parse(text);
        }
        catch (ArgumentException)
        {
            throw new UsageException($"--reference {text}: use {string.Join(" or ", PreferenceReferences.Names)}.");
        }
    }

    // A repeatable K=V option as a dictionary; null when not given.
    private static Dictionary<string, string>? KeyValues(CommandContext context, string option)
    {
        var items = context.Options(option);
        if (items.Count == 0)
        {
            return null;
        }

        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (string item in items)
        {
            int equals = item.IndexOf('=', StringComparison.Ordinal);
            if (equals <= 0)
            {
                throw new UsageException($"{option} {item}: give KEY=VALUE.");
            }

            values[item[..equals].Trim()] = item[(equals + 1)..].Trim();
        }

        return values;
    }

    // --spread: a RolloutSpread name.
    private static string Spread(string text)
    {
        try
        {
            return RolloutSpread.Parse(text);
        }
        catch (ArgumentException)
        {
            throw new UsageException($"--spread {text}: use {string.Join(" or ", RolloutSpread.Names)}.");
        }
    }

    internal static string? Text(string variable) => Environment.GetEnvironmentVariable(variable) is { Length: > 0 } value && value.Trim().Length > 0 ? value.Trim() : null;

    internal static int? Int(CommandContext context, string option, int min)
    {
        if (context.Option(option) is null)
        {
            return null;
        }

        int value = context.IntOption(option, min);
        return value >= min ? value : throw new UsageException(string.Create(CultureInfo.InvariantCulture, $"{option} is a whole number of at least {min}."));
    }

    internal static float? Float(CommandContext context, string option, float min, float max = float.PositiveInfinity)
    {
        if (context.Option(option) is not { } text)
        {
            return null;
        }

        return float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out float value) && float.IsFinite(value) && value >= min && value <= max ? value
            : throw new UsageException(string.Create(CultureInfo.InvariantCulture, $"{option} needs a number from {min} to {max}, not '{text}'."));
    }

    private static Device ParseDevice(string name)
    {
        try
        {
            return Device.Parse(name);
        }
        catch (Exception e) when (e is InvalidOperationException or FormatException or ArgumentException)
        {
            throw new UsageException($"--teacher-device {name}: {e.Message}");
        }
    }
}

/// <summary>
/// Ctrl+C for a training run of a command (idrak reinforce, idrak distill --on-policy): the first stops the run after its
/// current step or iteration, which then keeps what it trained; a second quits at once. <c>--timeout</c> stops it the same way.
/// </summary>
internal sealed class StopKey : IDisposable
{
    private readonly CancellationTokenSource _source;
    private readonly ConsoleCancelEventHandler _handler;

    /// <param name="context">The command (its <c>--timeout</c>).</param>
    /// <param name="say">Where the stop is announced.</param>
    /// <param name="message">What the first Ctrl+C says.</param>
    public StopKey(CommandContext context, Action<string> say, string message)
    {
        _source = CancellationTokenSource.CreateLinkedTokenSource(context.TimeoutToken);
        _handler = (_, e) =>
        {
            if (_source.IsCancellationRequested)
            {
                return;                                                           // a second Ctrl+C quits at once
            }

            e.Cancel = true;
            _source.Cancel();
            say(message);
        };
        Idrak.Cli.Shared.Terminal.AddCancelKey(_handler);
    }

    /// <summary>Cancelled by Ctrl+C or the timeout.</summary>
    public CancellationToken Token => _source.Token;

    public void Dispose()
    {
        Idrak.Cli.Shared.Terminal.RemoveCancelKey(_handler);
        _source.Dispose();
    }
}
