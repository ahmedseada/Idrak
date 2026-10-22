// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Reinforcement;
using Idrak.Cli.Commands.Cluster;
using Idrak.Cli.Shared;
using Idrak.Reinforcement;
using Idrak.Reinforcement.Abstractions;

namespace Idrak.Cli.Commands.Train;

/// <summary>
/// <c>idrak reinforce --env cartpole</c> (Plan 27, R9): reinforcement learning in an environment whose observations are
/// numbers (the classic plug-in's bandit, cartpole, pendulum, mountain-car, or any registered kind of float observations),
/// with no model to load. On-policy (<c>--algorithm reinforce|ppo|grpo</c>, a <c>PolicyLosses</c> name) the library's
/// <see cref="ReinforcementTrainer{TObservation, TAction}"/> trains a <see cref="ModulePolicy{TAction}"/> that
/// <see cref="PolicyNetworks"/> builds from the options (categorical for discrete actions, Gaussian for a box); off-policy
/// (<c>--algorithm dqn</c>, a <c>QLosses</c> name) <see cref="OffPolicyTrainer"/> trains a Q-network built the same way
/// (discrete actions only). Every option reads its <c>IDRAK_RL_*</c> or <c>IDRAK_RL_DQN_*</c> variable underneath
/// (<see cref="ReinforcementOptions.FromEnvironment"/>, <see cref="OffPolicyOptions.FromEnvironment"/>,
/// <see cref="ModulePolicyOptions.FromEnvironment"/>); the command line wins. <c>--out DIR</c> receives the policy (or the
/// Q-network) and <c>idrak-reinforce.json</c> (its settings), checkpoints go to DIR/checkpoints, and <c>--resume</c> goes on
/// from the latest. <see cref="ReinforceCommand"/> sends a run here when its environment kind does not observe
/// conversations.
/// </summary>
internal static class ReinforceNumeric
{
    /// <summary>The file in the output folder that holds the run's settings (the network's shape, the environment, the algorithm).</summary>
    public const string SettingsFile = "idrak-reinforce.json";

    /// <summary>The online Q-network's file in the output folder of a DQN run.</summary>
    public const string NetworkFile = "network.ikw";

    // The options of this path alone (the networks' shape, the off-policy run, settings the language-model path leaves to its session).
    public static readonly string[] ValueOptions =
    [
        "--hidden", "--activation", "--steps", "--gamma", "--lambda", "--entropy", "--max-grad-norm", "--max-steps", "--evaluation-episodes", "--replay", "--batch",
        "--learning-starts", "--train-every", "--gradient-steps", "--target-every", "--tau", "--epsilon-start", "--epsilon-end", "--epsilon-fraction", "--huber",
        "--report-every",
    ];

    public static readonly string[] Flags = ["--value-head", "--no-value-head", "--double"];

    // The session options of idrak tune that mean the same here: the optimizer and the schedule (declared before
    // LanguageModelOnly, whose initializer reads it).
    private static readonly string[] SharedSession = ["--lr", "--optimizer", "--weight-decay", "--schedule", "--warmup", "--min-lr", "--decay"];

    // A language model's options: refused here, naming them.
    private static readonly string[] LanguageModelOnly =
    [
        "--base", "--data", "--suite", "--reward", "--rounds", "--sandbox", "--max-new", "--no-think", "--teacher", "--teacher-device", "--teacher-weights", "--teacher-weight",
        "--divergence", "--beta", "--mix", "--weights",
        .. TuneTool.SessionValueOptions.Where(o => !SharedSession.Contains(o)), .. TuneTool.SessionFlags,
    ];

    // The on-policy trainer's options, refused for an off-policy run (and the other way round).
    private static readonly string[] OnPolicyOnly =
        ["--estimator", "--group", "--episodes", "--kl", "--clip", "--epochs", "--minibatch", "--iterations", "--temperature", "--shards", "--spread", "--lambda", "--entropy",
         "--value-head", "--no-value-head"];

    private static readonly string[] OffPolicyOnly =
        ["--steps", "--replay", "--batch", "--learning-starts", "--train-every", "--gradient-steps", "--target-every", "--tau", "--epsilon-start", "--epsilon-end",
         "--epsilon-fraction", "--huber", "--report-every", "--double"];

    /// <summary>The usage's lines of this path, in the command's Options section.</summary>
    public const string UsageOptions = """

        Options of an environment of numbers (--env bandit, cartpole, pendulum, mountain-car or a plug-in's: no MODEL):
              --algorithm NAME    on-policy: ppo (default), reinforce, grpo (PolicyLosses: a categorical policy for
                                  discrete actions, a Gaussian one for a box); off-policy: dqn (QLosses: a Q-network,
                                  discrete actions only)
              --hidden W,W        the hidden widths of the networks (default 64,64; none: one linear layer; IDRAK_RL_HIDDEN)
              --activation NAME   tanh (default), relu, gelu or sigmoid (IDRAK_RL_ACTIVATION)
              --value-head        a critic beside the actor; --no-value-head: none (default: ppo, gae or a value baseline
                                  have one; IDRAK_RL_VALUE_HEAD)
              --lr, --optimizer, --weight-decay, --schedule, --warmup, --min-lr, --decay
                                  the optimizer and its schedule (IDRAK_RL_LEARNING_RATE, or IDRAK_RL_DQN_LEARNING_RATE
                                  for dqn; IDRAK_RL_OPTIMIZER, IDRAK_RL_SCHEDULE, IDRAK_RL_WARMUP)
              --gamma G           the discount (default 0.99; IDRAK_RL_GAMMA); --lambda L: GAE's λ (IDRAK_RL_LAMBDA)
              --entropy W         the entropy bonus's weight (IDRAK_RL_ENTROPY)
              --max-grad-norm N   the gradient-norm clip (default 1, dqn 10; IDRAK_RL_MAX_GRAD_NORM, IDRAK_RL_DQN_*)
              --max-steps N       the most steps of an episode (default: the environment's; IDRAK_RL_MAX_STEPS)
              --evaluation-episodes N  greedy episodes of an evaluation (default 8; IDRAK_RL_EVALUATION_EPISODES)
              --steps N           dqn: the environment steps of the run (default 100000; IDRAK_RL_DQN_STEPS)
              --environments N    dqn: environments played side by side, a round steps each once (default 1)
              --replay N          dqn: the transitions the replay buffer keeps (default 100000)
              --batch N           dqn: the transitions of an update (default 32)
              --learning-starts N dqn: random steps before the first update (default 100)
              --train-every N     dqn: rounds between two trainings (default 4); --gradient-steps N: updates each (1)
              --target-every N    dqn: steps between two target-network updates (default 10000); --tau T: their reach (1)
              --epsilon-start E   dqn: ε-greedy's ε at the start (default 1); --epsilon-end E: at the end (0.05);
                                  --epsilon-fraction F: the share of the run over which it falls (0.1)
              --double            dqn: double Q-learning; --huber D: the Huber loss's δ (default 1, 0: squared error)
              --report-every N    dqn: steps between two report lines (default 1000); --save-every and --evaluate-every
                                  count steps too
        Each reads its IDRAK_RL_* variable (dqn's: IDRAK_RL_DQN_*); the command line wins. DIR receives the policy (or
        the Q-network, network.ikw) and idrak-reinforce.json (its settings); checkpoints go to DIR/checkpoints.
        """;

    /// <summary>The usage's examples of this path.</summary>
    public const string UsageExamples = """
          idrak reinforce --env cartpole --algorithm ppo --iterations 50 -o runs/cartpole
          idrak reinforce --env cartpole --algorithm dqn --steps 50000 -o runs/cartpole-dqn
          idrak reinforce --env pendulum --algorithm ppo --hidden 64,64 --iterations 200 -o runs/pendulum --resume
        """;

    /// <summary>The kind <c>--env</c> (or <c>IDRAK_RL_ENV</c>) names when it is registered and does not observe conversations; null otherwise (the language-model path).</summary>
    public static IEnvironmentKind? Kind(CommandContext context)
    {
        string? name = context.Option("--env") ?? ReinforceCommand.Text(ReinforceCommand.EnvironmentVariable);
        return name is not null && EnvironmentKinds.Find(name) is { } kind && kind.ObservationType != typeof(ChatObservation) ? kind : null;
    }

    /// <summary>Trains in <paramref name="kind"/>'s environments, on-policy or off-policy as <c>--algorithm</c> says.</summary>
    public static int Run(CommandContext context, IEnvironmentKind kind)
    {
        if (kind.ObservationType != typeof(float[]) || (kind.ActionType != typeof(int) && kind.ActionType != typeof(float[])))
        {
            throw new UsageException($"--env {kind.Name}: its environments observe {kind.ObservationType.Name} and act with {kind.ActionType.Name}; idrak reinforce trains a "
                                     + "language model on conversations, or a policy of its own on float observations (float[]) with discrete (int) or box (float[]) actions.");
        }

        if (ClusterSession.Current is not null)
        {
            throw new UsageException($"--env {kind.Name} trains in one process: a policy of numbers is not spread over a cluster's ranks (only a language model's "
                                     + "rollouts are). Run it without cluster sim or --cluster.");
        }

        if (context.Positional.Count > 0)
        {
            throw new UsageException($"--env {kind.Name} observes numbers: no model is loaded ('{context.Positional[0]}' is not wanted); the policy is built from --hidden, "
                                     + "--activation and --value-head.");
        }

        Refuse(context, LanguageModelOnly, $"--env {kind.Name} trains a policy of numbers, not a language model's adapters");
        string output = context.Option("--out") ?? throw new UsageException("Give -o, --out DIR for the policy and the checkpoints.");
        var environmentOptions = EnvironmentOptions(context, kind);
        Space observations, actions;
        try
        {
            (observations, actions) = PolicyNetworks.Spaces(kind, environmentOptions);
        }
        catch (ArgumentException e)
        {
            throw new UsageException($"--env {kind.Name}: {e.Message}");
        }

        if (observations.Kind != SpaceKind.Box)
        {
            throw new UsageException($"--env {kind.Name}: its observations are {observations}; a policy of numbers reads a box of floats.");
        }

        // The algorithm: a policy loss (on-policy) or a Q-learning loss (off-policy), by its registry.
        string algorithm = context.Option("--algorithm") ?? ReinforceCommand.Text("IDRAK_RL_ALGORITHM") ?? ReinforcementNames.Ppo;
        bool offPolicy = PolicyLosses.Find(algorithm) is null && QLosses.Find(algorithm) is not null;
        if (!offPolicy && PolicyLosses.Find(algorithm) is null)
        {
            throw new UsageException($"{(context.Option("--algorithm") is null ? "IDRAK_RL_ALGORITHM=" : "--algorithm ")}{algorithm}: in the {kind.Name} environment use "
                                     + $"{string.Join(", ", PolicyLosses.Names)} (PolicyLosses, on-policy) or {string.Join(", ", QLosses.Names)} (QLosses, off-policy)"
                                     + $"{(algorithm.Equals(ReinforceCommand.Distill, StringComparison.OrdinalIgnoreCase) ? "; distill is a language model's" : "")}.");
        }

        var network = NetworkOptions(context);
        var lines = context.Json ? new StringWriter() : null;
        var writer = lines ?? (context.Quiet ? TextWriter.Null : context.Output);
        var json = new JsonObject
        {
            ["environment"] = kind.Name,
            ["algorithm"] = algorithm,
            ["observations"] = observations.ToString(),
            ["actions"] = actions.ToString(),
            ["out"] = Path.GetFullPath(output),
            ["device"] = context.Device.ToString(),
        };
        int exit;
        if (offPolicy)
        {
            Refuse(context, OnPolicyOnly, $"--algorithm {algorithm} trains off-policy (it runs --steps)");
            if (actions.Kind != SpaceKind.Discrete)
            {
                throw new UsageException($"--algorithm {algorithm} values each of a discrete set of actions; the {kind.Name} environment acts in {actions} (a box): train it "
                                         + $"on-policy with {string.Join(", ", PolicyLosses.Names)} (a Gaussian policy).");
            }

            exit = OffPolicy(context, writer, kind, observations, actions, OffPolicySettings(context, output, algorithm, environmentOptions), network, output, json);
        }
        else
        {
            Refuse(context, OffPolicyOnly, $"--algorithm {algorithm} trains on-policy (it runs --iterations)");
            var options = ReinforceCommand.Settings(context, output);
            options = OnPolicySettings(context, options) with { EnvironmentOptions = environmentOptions };
            exit = actions.Kind == SpaceKind.Discrete
                ? OnPolicy(context, writer, kind, observations, actions, options, network, output, json,
                    () => PolicyNetworks.Categorical(observations, actions, network, network.ValueHeadFor(options), context.Device, options.Seed, options.KlWeight > 0))
                : OnPolicy(context, writer, kind, observations, actions, options, network, output, json,
                    () => PolicyNetworks.Gaussian(observations, actions, network, network.ValueHeadFor(options), context.Device, options.Seed, options.KlWeight > 0));
        }

        if (lines is not null)
        {
            json["exitCode"] = exit;
            json["output"] = new JsonArray([.. lines.ToString().Split('\n').Select(l => l.TrimEnd('\r')).Where(l => l.Length > 0).Select(l => (JsonNode)l)]);
            context.WriteJson(json);
        }

        return exit;
    }

    // An on-policy run: the module policy (made, or loaded from the checkpoint by the trainer), the trainer, the policy and
    // its settings written to `output` at the end (also after a stop).
    private static int OnPolicy<TAction>(CommandContext context, TextWriter lines, IEnvironmentKind kind, Space observations, Space actions, ReinforcementOptions options,
        ModulePolicyOptions network, string output, JsonObject json, Func<ModulePolicy<TAction>> make)
    {
        ModulePolicy<TAction> policy;
        try
        {
            policy = make();
        }
        catch (ArgumentException e)
        {
            throw new UsageException(e.Message);
        }

        using (policy)
        {
            bool gaussian = actions.Kind == SpaceKind.Box;
            lines.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"reinforce: {options.Algorithm} in {kind.Name} (observations {observations}, actions {actions}), a {(gaussian ? "Gaussian" : "categorical")} policy of "
                + $"{Shape(network)}{(policy.Critic is null ? "" : " with a value head")}, {options.EpisodesPerIteration} episodes an iteration, {options.Iterations} iterations"
                + $"{(options.Resume is { } resume ? $"; resuming from {resume}" : "")}"));
            using var stop = new StopKey(context, context.ErrorOutput.WriteLine, "stopping after the current iteration and saving the policy (Ctrl+C again to quit now)…");
            using var progress = new ProgressLine(context, $"reinforce {kind.Name}", options.Iterations);
            var trainer = new ReinforcementTrainer<float[], TAction>(kind, policy, options)
            {
                Progress = progress,
                OnIteration = s =>
                {
                    progress.Erase();
                    lines.WriteLine(string.Create(CultureInfo.InvariantCulture,
                        $"iteration {s.Iteration}/{options.Iterations}: reward {s.RewardMean:0.###} ± {s.RewardSpread:0.###} ({s.RewardMin:0.###} to {s.RewardMax:0.###}), "
                        + $"loss {s.Loss:0.####}, {s.Updates} update{(s.Updates == 1 ? "" : "s")}{(s.EvaluationReward is { } e ? FormattableString.Invariant($", evaluation {e:0.###}") : "")}, "
                        + $"rollout {s.RolloutTime.TotalSeconds:0.00} s, update {s.UpdateTime.TotalSeconds:0.00} s"));
                },
            };
            var clock = Stopwatch.StartNew();
            ReinforcementResult result;
            try
            {
                result = trainer.Train(stop.Token);
            }
            catch (Exception failure) when (failure is InvalidDataException or FileNotFoundException)
            {
                throw new UsageException($"--resume: {failure.Message}");
            }

            policy.Save(output);
            WriteSettings(output, kind, options.EnvironmentOptions, options.Algorithm, gaussian ? "gaussian" : "categorical", observations, actions, network,
                policy.Critic is not null, options.Seed, new JsonObject { ["iterations"] = result.Iterations, ["checkpoint"] = result.Checkpoint });
            var history = result.History;
            lines.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"{(result.Cancelled ? "stopped after" : "trained")} {result.Iterations} iterations in {clock.Elapsed.TotalSeconds:0.0} s"
                + $"{(history.Count > 0 ? FormattableString.Invariant($": reward {history[0].RewardMean:0.###} → {history[^1].RewardMean:0.###}") : "")}"
                + $"; policy written to {Path.GetFullPath(output)}{(result.Checkpoint is { } c ? $", checkpoint {c}" : "")}"));
            json["iterations"] = result.Iterations;
            json["cancelled"] = result.Cancelled;
            json["checkpoint"] = result.Checkpoint;
            json["history"] = new JsonArray([.. history.Select(h => (JsonNode)new JsonObject
            {
                ["iteration"] = h.Iteration,
                ["episodes"] = h.Episodes,
                ["steps"] = h.Steps,
                ["rewardMean"] = Finite(h.RewardMean),
                ["rewardSpread"] = Finite(h.RewardSpread),
                ["loss"] = Finite(h.Loss),
                ["updates"] = h.Updates,
                ["evaluation"] = h.EvaluationReward is { } e ? Finite(e) : null,
                ["returns"] = new JsonArray([.. h.Returns.Select(r => (JsonNode)r)]),
            })]);
            return ExitCodes.Ok;
        }
    }

    // An off-policy run: the Q-network and its target, the trainer (which owns them), the online network and its settings
    // written to `output` at the end (also after a stop).
    private static int OffPolicy(CommandContext context, TextWriter lines, IEnvironmentKind kind, Space observations, Space actions, OffPolicyOptions options,
        ModulePolicyOptions network, string output, JsonObject json)
    {
        lines.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"reinforce: {options.Algorithm} in {kind.Name} (observations {observations}, actions {actions}), a Q-network of {Shape(network)}, {options.Steps} steps, "
            + $"{options.Environments} environment{(options.Environments == 1 ? "" : "s")} side by side, a replay of {options.ReplayCapacity} transitions"
            + $"{(options.Resume is { } resume ? $"; resuming from {resume}" : "")}"));
        using var stop = new StopKey(context, context.ErrorOutput.WriteLine, "stopping after the current round and saving the network (Ctrl+C again to quit now)…");
        using var progress = new ProgressLine(context, $"reinforce {kind.Name}", options.Steps);
        OffPolicyTrainer trainer;
        try
        {
            var (online, target) = PolicyNetworks.QNetworks(observations, actions, network, context.Device, options.Seed);
            trainer = new OffPolicyTrainer(kind, online, target, options)
            {
                Progress = progress,
                OnReport = s =>
                {
                    progress.Erase();
                    lines.WriteLine(string.Create(CultureInfo.InvariantCulture,
                        $"step {s.Step}/{options.Steps}: return {(s.Returns.Length > 0 ? FormattableString.Invariant($"{s.RewardMean:0.###}") : "-")} "
                        + $"({s.Returns.Length} episode{(s.Returns.Length == 1 ? "" : "s")}), loss {(double.IsFinite(s.Loss) ? FormattableString.Invariant($"{s.Loss:0.####}") : "-")}, "
                        + $"ε {s.Epsilon:0.###}, {s.Updates} updates{(s.EvaluationReward is { } e ? FormattableString.Invariant($", evaluation {e:0.###}") : "")}"));
                },
            };
        }
        catch (ArgumentException e)
        {
            throw new UsageException(e.Message);
        }

        using (trainer)
        {
            var clock = Stopwatch.StartNew();
            OffPolicyResult result;
            try
            {
                result = trainer.Train(stop.Token);
            }
            catch (Exception failure) when (failure is InvalidDataException or FileNotFoundException)
            {
                throw new UsageException($"--resume: {failure.Message}");
            }

            Directory.CreateDirectory(output);
            trainer.Network.Save(Path.Combine(output, NetworkFile));
            WriteSettings(output, kind, options.EnvironmentOptions, options.Algorithm, "q-network", observations, actions, network, false, options.Seed,
                new JsonObject { ["steps"] = result.Steps, ["updates"] = result.Updates, ["checkpoint"] = result.Checkpoint });
            var last = result.History.Count > 0 ? result.History[^1] : null;
            lines.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"{(result.Cancelled ? "stopped after" : "trained")} {result.Steps} steps ({result.Updates} updates) in {clock.Elapsed.TotalSeconds:0.0} s"
                + $"{(last?.EvaluationReward is { } e ? FormattableString.Invariant($": evaluation {e:0.###}") : "")}"
                + $"; network written to {Path.GetFullPath(Path.Combine(output, NetworkFile))}{(result.Checkpoint is { } c ? $", checkpoint {c}" : "")}"));
            json["steps"] = result.Steps;
            json["updates"] = result.Updates;
            json["cancelled"] = result.Cancelled;
            json["checkpoint"] = result.Checkpoint;
            json["history"] = new JsonArray([.. result.History.Select(h => (JsonNode)new JsonObject
            {
                ["step"] = h.Step,
                ["updates"] = h.Updates,
                ["episodes"] = h.Episodes,
                ["rewardMean"] = Finite(h.RewardMean),
                ["loss"] = Finite(h.Loss),
                ["epsilon"] = h.Epsilon,
                ["evaluation"] = h.EvaluationReward is { } e ? Finite(e) : null,
                ["returns"] = new JsonArray([.. h.Returns.Select(r => (JsonNode)r)]),
            })]);
            return ExitCodes.Ok;
        }
    }

    // The on-policy settings this path adds to ReinforceCommand.Settings (the variables, then the command line).
    private static ReinforcementOptions OnPolicySettings(CommandContext context, ReinforcementOptions o)
    {
        o = ReinforceCommand.Float(context, "--gamma", 0f, 1f) is { } gamma ? o with { Gamma = gamma } : o;
        o = ReinforceCommand.Float(context, "--lambda", 0f, 1f) is { } lambda ? o with { Lambda = lambda } : o;
        o = ReinforceCommand.Float(context, "--entropy", 0f) is { } entropy ? o with { EntropyWeight = entropy } : o;
        o = ReinforceCommand.Float(context, "--max-grad-norm", 0f) is { } norm ? o with { MaxGradientNorm = norm } : o;
        o = ReinforceCommand.Int(context, "--max-steps", 0) is { } maxSteps ? o with { MaxEpisodeSteps = maxSteps } : o;
        o = ReinforceCommand.Int(context, "--evaluation-episodes", 1) is { } evaluationEpisodes ? o with { EvaluationEpisodes = evaluationEpisodes } : o;
        o = context.Option("--schedule") is { } schedule ? o with { Schedule = Schedule(schedule) } : o;
        return o with { ScheduleSettings = ScheduleSettings(context, o.ScheduleSettings) };
    }

    // The off-policy settings: OffPolicyOptions.FromEnvironment, then each option given on the command line in its place.
    private static OffPolicyOptions OffPolicySettings(CommandContext context, string output, string algorithm, IReadOnlyDictionary<string, string> environmentOptions)
    {
        OffPolicyOptions o;
        try
        {
            o = OffPolicyOptions.FromEnvironment();
        }
        catch (ArgumentException e)
        {
            throw new UsageException(e.Message);
        }

        o = o with { Algorithm = QLosses.Find(algorithm)!.Name, EnvironmentOptions = environmentOptions };
        o = Long(context, "--steps", 1) is { } steps ? o with { Steps = steps } : o;
        o = ReinforceCommand.Int(context, "--environments", 1) is { } environments ? o with { Environments = environments } : o;
        o = ReinforceCommand.Int(context, "--replay", 1) is { } replay ? o with { ReplayCapacity = replay } : o;
        o = ReinforceCommand.Int(context, "--batch", 1) is { } batch ? o with { BatchSize = batch } : o;
        o = Long(context, "--learning-starts", 0) is { } learningStarts ? o with { LearningStarts = learningStarts } : o;
        o = ReinforceCommand.Int(context, "--train-every", 1) is { } trainEvery ? o with { TrainEvery = trainEvery } : o;
        o = ReinforceCommand.Int(context, "--gradient-steps", 1) is { } gradientSteps ? o with { GradientSteps = gradientSteps } : o;
        o = Long(context, "--target-every", 1) is { } targetEvery ? o with { TargetEvery = targetEvery } : o;
        o = ReinforceCommand.Float(context, "--tau", float.Epsilon, 1f) is { } tau ? o with { Tau = tau } : o;
        o = ReinforceCommand.Float(context, "--gamma", 0f, 1f) is { } gamma ? o with { Gamma = gamma } : o;
        o = ReinforceCommand.Float(context, "--epsilon-start", 0f, 1f) is { } epsilonStart ? o with { EpsilonStart = epsilonStart } : o;
        o = ReinforceCommand.Float(context, "--epsilon-end", 0f, 1f) is { } epsilonEnd ? o with { EpsilonEnd = epsilonEnd } : o;
        o = ReinforceCommand.Float(context, "--epsilon-fraction", 0f, 1f) is { } epsilonFraction ? o with { EpsilonFraction = epsilonFraction } : o;
        o = context.Flag("--double") ? o with { DoubleQ = true } : o;
        o = ReinforceCommand.Float(context, "--huber", 0f) is { } huber ? o with { Huber = huber } : o;
        o = context.Option("--optimizer") is { } optimizer ? o with { Optimizer = Optimizer(optimizer) } : o with { Optimizer = Optimizer(o.Optimizer) };
        o = ReinforceCommand.Float(context, "--lr", 0f) is { } rate ? o with { LearningRate = rate } : o;
        o = ReinforceCommand.Float(context, "--weight-decay", 0f) is { } decay ? o with { WeightDecay = decay } : o;
        o = ReinforceCommand.Float(context, "--max-grad-norm", 0f) is { } norm ? o with { MaxGradientNorm = norm } : o;
        o = context.Option("--schedule") is { } schedule ? o with { Schedule = Schedule(schedule) } : o with { Schedule = Schedule(o.Schedule) };
        o = o with { ScheduleSettings = ScheduleSettings(context, o.ScheduleSettings) };
        o = ReinforceCommand.Int(context, "--max-steps", 0) is { } maxSteps ? o with { MaxEpisodeSteps = maxSteps } : o;
        o = context.Seed is { } seed ? o with { Seed = seed } : o;
        o = Long(context, "--report-every", 1) is { } reportEvery ? o with { ReportEvery = reportEvery } : o;
        o = Long(context, "--evaluate-every", 1) is { } evaluateEvery ? o with { EvaluateEvery = evaluateEvery } : o;
        o = ReinforceCommand.Int(context, "--evaluation-episodes", 1) is { } evaluationEpisodes ? o with { EvaluationEpisodes = evaluationEpisodes } : o;
        o = Long(context, "--save-every", 1) is { } saveEvery ? o with { SaveEvery = saveEvery } : o;
        o = o with { CheckpointFolder = o.CheckpointFolder ?? Path.Combine(output, "checkpoints") };
        o = context.Flag("--resume") ? o with { Resume = o.CheckpointFolder } : o;
        if (o.Resume is { } resume && !File.Exists(Path.Combine(resume, "state.json"))
            && !(Directory.Exists(resume) && Directory.EnumerateDirectories(resume, "step-*").Any(d => File.Exists(Path.Combine(d, "state.json")))))
        {
            throw new UsageException($"Nothing to resume: no checkpoint in {Path.GetFullPath(resume)}.");
        }

        return o;
    }

    // The networks' shape and the module policy's optimizer: ModulePolicyOptions.FromEnvironment, then the command line.
    private static ModulePolicyOptions NetworkOptions(CommandContext context)
    {
        ModulePolicyOptions o;
        try
        {
            o = ModulePolicyOptions.FromEnvironment();
        }
        catch (ArgumentException e)
        {
            throw new UsageException(e.Message);
        }

        if (context.Option("--hidden") is { } hidden)
        {
            try
            {
                o = o with { Hidden = ModulePolicyOptions.ParseHidden(hidden) };
            }
            catch (ArgumentException e)
            {
                throw new UsageException($"--hidden {e.Message}");
            }
        }

        if (context.Option("--activation") is { } activation)
        {
            o = o with
            {
                Activation = PolicyNetworks.Activations.FirstOrDefault(a => a.Equals(activation, StringComparison.OrdinalIgnoreCase))
                             ?? throw new UsageException($"--activation {activation}: use {string.Join(", ", PolicyNetworks.Activations)}."),
            };
        }

        if (context.Flag("--value-head") && context.Flag("--no-value-head"))
        {
            throw new UsageException("Give --value-head or --no-value-head, not both.");
        }

        o = context.Flag("--value-head") ? o with { ValueHead = true } : context.Flag("--no-value-head") ? o with { ValueHead = false } : o;
        o = o with { Optimizer = Optimizer(context.Option("--optimizer") ?? o.Optimizer) };
        o = ReinforceCommand.Float(context, "--lr", 0f) is { } rate ? o with { LearningRate = rate } : o;
        o = ReinforceCommand.Float(context, "--weight-decay", 0f) is { } decay ? o with { WeightDecay = decay } : o;
        return o;
    }

    // --env-option KEY=VALUE, each key checked against the kind's before anything is made.
    private static Dictionary<string, string> EnvironmentOptions(CommandContext context, IEnvironmentKind kind)
    {
        var options = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (string item in context.Options("--env-option"))
        {
            int equals = item.IndexOf('=', StringComparison.Ordinal);
            if (equals <= 0)
            {
                throw new UsageException($"--env-option {item}: give KEY=VALUE (the {kind.Name} environment's keys: {string.Join(", ", kind.Keys)}).");
            }

            string key = item[..equals].Trim();
            if (!kind.Keys.Contains(key))
            {
                throw new UsageException($"The {kind.Name} environment does not take '{key}'; it takes: {string.Join(", ", kind.Keys)}.");
            }

            options[key] = item[(equals + 1)..];
        }

        return options;
    }

    // The schedule's settings: --warmup, --min-lr and --decay over the run's.
    private static LearningRateScheduleSettings ScheduleSettings(CommandContext context, LearningRateScheduleSettings settings)
    {
        settings = ReinforceCommand.Float(context, "--warmup", 0f, 1f) is { } warmup ? settings with { WarmupFraction = warmup } : settings;
        settings = ReinforceCommand.Float(context, "--min-lr", 0f) is { } minimum ? settings with { MinLearningRate = minimum } : settings;
        return ReinforceCommand.Float(context, "--decay", 0f, 1f) is { } decay ? settings with { DecayFraction = decay } : settings;
    }

    private static string Schedule(string name) => LearningRateSchedules.Find(name) is not null ? name
        : throw new UsageException($"The schedule {name}: use {string.Join(", ", LearningRateSchedules.Names)} (LearningRateSchedules).");

    private static string Optimizer(string name) => OptimizerKinds.Find(name) is not null ? name
        : throw new UsageException($"The optimizer {name}: use {string.Join(", ", OptimizerKinds.Names)} (OptimizerKinds).");

    // Options of the other path (or of a language model), refused naming them.
    private static void Refuse(CommandContext context, IEnumerable<string> options, string why)
    {
        var given = options.Where(o => context.Option(o) is not null || context.Flag(o)).Distinct().ToList();
        if (given.Count > 0)
        {
            throw new UsageException($"{string.Join(", ", given)}: {why}; {(given.Count == 1 ? "it does" : "they do")} not apply here.");
        }
    }

    private static long? Long(CommandContext context, string option, long min)
    {
        if (context.Option(option) is not { } text)
        {
            return null;
        }

        return long.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out long value) && value >= min ? value
            : throw new UsageException(string.Create(CultureInfo.InvariantCulture, $"{option} is a whole number of at least {min}."));
    }

    // The networks' shape as the lines say it: "64, 64 tanh", or "one linear layer".
    private static string Shape(ModulePolicyOptions network) => network.Hidden.Count == 0 ? "one linear layer"
        : $"{string.Join(", ", network.Hidden)} {network.Activation}";

    private static double? Finite(double value) => double.IsFinite(value) ? value : null;

    // idrak-reinforce.json in the output folder: what the networks are and how they were trained, to build them again.
    private static void WriteSettings(string output, IEnvironmentKind kind, IReadOnlyDictionary<string, string>? environmentOptions, string algorithm, string policy,
        Space observations, Space actions, ModulePolicyOptions network, bool valueHead, int seed, JsonObject done)
    {
        Directory.CreateDirectory(output);
        var settings = new JsonObject
        {
            ["format"] = "idrak-reinforce/1",
            ["environment"] = kind.Name,
            ["environmentOptions"] = new JsonObject([.. (environmentOptions ?? new Dictionary<string, string>()).OrderBy(p => p.Key, StringComparer.Ordinal)
                .Select(p => KeyValuePair.Create(p.Key, (JsonNode?)p.Value))]),
            ["algorithm"] = algorithm,
            ["policy"] = policy,
            ["observations"] = observations.ToString(),
            ["observationSize"] = observations.Size,
            ["actions"] = actions.ToString(),
            ["actionCount"] = actions.Kind == SpaceKind.Discrete ? actions.Count : actions.Size,
            ["hidden"] = new JsonArray([.. network.Hidden.Select(w => (JsonNode)w)]),
            ["activation"] = network.Activation,
            ["valueHead"] = valueHead,
            ["seed"] = seed,
        };
        foreach (var (key, value) in done)
        {
            settings[key] = value?.DeepClone();
        }

        File.WriteAllText(Path.Combine(output, SettingsFile), settings.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }) + "\n");
    }
}
