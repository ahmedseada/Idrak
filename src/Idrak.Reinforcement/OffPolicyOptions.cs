// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Reinforcement.Abstractions;

namespace Idrak.Reinforcement;

/// <summary>
/// The settings of an off-policy run (<see cref="OffPolicyTrainer"/>, DQN). Every one has a default (Stable-Baselines3's
/// DQN defaults where they exist, with a replay of 100,000 transitions in place of its million) and a variable
/// <see cref="FromEnvironment"/> reads: <c>IDRAK_RL_DQN_*</c> for those of this trainer alone, and the shared
/// <c>IDRAK_RL_SEED</c>, <c>IDRAK_RL_GAMMA</c>, <c>IDRAK_RL_OPTIMIZER</c>, <c>IDRAK_RL_SCHEDULE</c>, <c>IDRAK_RL_WARMUP</c>,
/// <c>IDRAK_RL_MAX_STEPS</c>, <c>IDRAK_RL_PARALLEL_STEPS</c>, <c>IDRAK_RL_EVALUATION_EPISODES</c>,
/// <c>IDRAK_RL_CHECKPOINTS</c>, <c>IDRAK_RL_RESUME</c> and <c>IDRAK_RL_LOG</c>, which mean here what they mean for the
/// on-policy trainer (an app's explicit options win: they read no variable). Steps count environment steps, all
/// environments together.
/// </summary>
public sealed record OffPolicyOptions
{
    /// <summary>The algorithm: a <see cref="QLosses"/> name (dqn, or an app's). <c>IDRAK_RL_DQN_LOSS</c>.</summary>
    public string Algorithm { get; init; } = QLosses.Dqn;

    /// <summary>Double Q-learning (van Hasselt et al. 2016): the loss's option <c>double</c>; null for the loss's own (false). <c>IDRAK_RL_DQN_DOUBLE</c>.</summary>
    public bool? DoubleQ { get; init; }

    /// <summary>The Huber loss's δ (0: the squared error): the loss's option <c>huber</c>; null for the loss's own (1). <c>IDRAK_RL_DQN_HUBER</c>.</summary>
    public float? Huber { get; init; }

    /// <summary>The environment steps of the run (every environment's together). <c>IDRAK_RL_DQN_STEPS</c>.</summary>
    public long Steps { get; init; } = 100_000;

    /// <summary>The environments played side by side: each round steps every one once (in order), so a round is this many steps. <c>IDRAK_RL_DQN_ENVIRONMENTS</c>.</summary>
    public int Environments { get; init; } = 1;

    /// <summary>Whether the environments of a round step in parallel: true, false, or null to measure which is faster (rule 79). Results do not depend on it. <c>IDRAK_RL_PARALLEL_STEPS</c>.</summary>
    public bool? ParallelSteps { get; init; }

    /// <summary>The transitions the replay buffer keeps (the oldest overwritten). <c>IDRAK_RL_DQN_REPLAY</c>.</summary>
    public int ReplayCapacity { get; init; } = 100_000;

    /// <summary>The transitions of each update's mini-batch. <c>IDRAK_RL_DQN_BATCH</c>.</summary>
    public int BatchSize { get; init; } = 32;

    /// <summary>The steps taken with uniformly random actions, and before any update, to fill the buffer. <c>IDRAK_RL_DQN_LEARNING_STARTS</c>.</summary>
    public long LearningStarts { get; init; } = 100;

    /// <summary>The rounds between two trainings (each <see cref="GradientSteps"/> updates). <c>IDRAK_RL_DQN_TRAIN_EVERY</c>.</summary>
    public int TrainEvery { get; init; } = 4;

    /// <summary>The updates (optimizer steps) of each training. <c>IDRAK_RL_DQN_GRADIENT_STEPS</c>.</summary>
    public int GradientSteps { get; init; } = 1;

    /// <summary>The environment steps between two updates of the target network (rounded to whole rounds, at least one). <c>IDRAK_RL_DQN_TARGET_EVERY</c>.</summary>
    public long TargetEvery { get; init; } = 10_000;

    /// <summary>
    /// How far each update of the target network moves it to the online one: 1 copies it (Mnih et al. 2015), below 1 is
    /// Polyak averaging θ' ← τ θ + (1 − τ) θ' (Lillicrap et al. 2016). <c>IDRAK_RL_DQN_TAU</c>.
    /// </summary>
    public float Tau { get; init; } = 1f;

    /// <summary>The discount γ of the targets. <c>IDRAK_RL_GAMMA</c>.</summary>
    public float Gamma { get; init; } = 0.99f;

    /// <summary>ε at the start of the run: the share of actions drawn uniformly instead of greedily. <c>IDRAK_RL_DQN_EPSILON_START</c>.</summary>
    public float EpsilonStart { get; init; } = 1f;

    /// <summary>ε once its decay is over. <c>IDRAK_RL_DQN_EPSILON_END</c>.</summary>
    public float EpsilonEnd { get; init; } = 0.05f;

    /// <summary>The share of the run's steps over which ε falls linearly from its start to its end (0: the end at once). <c>IDRAK_RL_DQN_EPSILON_FRACTION</c>.</summary>
    public float EpsilonFraction { get; init; } = 0.1f;

    /// <summary>The optimizer, by name in <c>OptimizerKinds</c>. <c>IDRAK_RL_OPTIMIZER</c>.</summary>
    public string Optimizer { get; init; } = "adam";

    /// <summary>The optimizer's learning rate, the base of the schedule. <c>IDRAK_RL_DQN_LEARNING_RATE</c>.</summary>
    public float LearningRate { get; init; } = 1e-4f;

    /// <summary>The optimizer's weight decay.</summary>
    public float WeightDecay { get; init; }

    /// <summary>The largest global gradient norm an update applies (0: no clip). <c>IDRAK_RL_DQN_MAX_GRAD_NORM</c>.</summary>
    public float MaxGradientNorm { get; init; } = 10f;

    /// <summary>The learning-rate schedule over the run's updates, a <c>LearningRateSchedules</c> name. <c>IDRAK_RL_SCHEDULE</c>.</summary>
    public string Schedule { get; init; } = LearningRateSchedules.Constant;

    /// <summary>The schedule's settings; no warm-up by default. <c>IDRAK_RL_WARMUP</c> sets its warm-up share.</summary>
    public LearningRateScheduleSettings ScheduleSettings { get; init; } = new(WarmupFraction: 0f);

    /// <summary>The most steps of an episode before the trainer cuts it (truncated: bootstrapped); 0 for the environment's own limit. <c>IDRAK_RL_MAX_STEPS</c>.</summary>
    public int MaxEpisodeSteps { get; init; }

    /// <summary>The run's seed: every episode's seed, every exploration draw and every mini-batch derive from it. <c>IDRAK_RL_SEED</c>.</summary>
    public int Seed { get; init; }

    /// <summary>A report (progress, telemetry, the log's line, <see cref="OffPolicyTrainer.OnReport"/>) after every this many steps (rounded to whole rounds). <c>IDRAK_RL_DQN_REPORT_EVERY</c>.</summary>
    public long ReportEvery { get; init; } = 1_000;

    /// <summary>Greedy evaluation episodes at every report that this many steps have passed since the last; 0 for none. <c>IDRAK_RL_DQN_EVALUATE_EVERY</c>.</summary>
    public long EvaluateEvery { get; init; }

    /// <summary>The episodes of an evaluation. <c>IDRAK_RL_EVALUATION_EPISODES</c>.</summary>
    public int EvaluationEpisodes { get; init; } = 8;

    /// <summary>The evaluation episodes' seed; null for one derived from <see cref="Seed"/>. The same episodes every evaluation.</summary>
    public int? EvaluationSeed { get; init; }

    /// <summary>A checkpoint into <see cref="CheckpointFolder"/> at every report that this many steps have passed since the last (and at the end); 0 for the end only. <c>IDRAK_RL_DQN_SAVE_EVERY</c>.</summary>
    public long SaveEvery { get; init; }

    /// <summary>Where checkpoints are written (step-N folders); null for none. <c>IDRAK_RL_CHECKPOINTS</c>.</summary>
    public string? CheckpointFolder { get; init; }

    /// <summary>A checkpoint to resume from: a step-N folder, or a checkpoint folder (its latest); null to start afresh. <c>IDRAK_RL_RESUME</c>.</summary>
    public string? Resume { get; init; }

    /// <summary>A file each report is appended to as a JSON line; null for none. <c>IDRAK_RL_LOG</c>.</summary>
    public string? Log { get; init; }

    /// <summary>The environment kind's options.</summary>
    public IReadOnlyDictionary<string, string>? EnvironmentOptions { get; init; }

    /// <summary>More options for the loss, passed as they are after <see cref="DoubleQ"/> and <see cref="Huber"/>.</summary>
    public IReadOnlyDictionary<string, string>? LossOptions { get; init; }

    /// <summary>
    /// These options with each one a variable sets in its place; the variables not set keep <paramref name="start"/>'s (the
    /// defaults when null). The variables: IDRAK_RL_DQN_LOSS, _DOUBLE, _HUBER, _STEPS, _ENVIRONMENTS, _REPLAY, _BATCH,
    /// _LEARNING_STARTS, _TRAIN_EVERY, _GRADIENT_STEPS, _TARGET_EVERY, _TAU, _EPSILON_START, _EPSILON_END, _EPSILON_FRACTION,
    /// _LEARNING_RATE, _MAX_GRAD_NORM, _REPORT_EVERY, _EVALUATE_EVERY, _SAVE_EVERY; and IDRAK_RL_SEED, IDRAK_RL_GAMMA,
    /// IDRAK_RL_OPTIMIZER, IDRAK_RL_SCHEDULE, IDRAK_RL_WARMUP, IDRAK_RL_MAX_STEPS, IDRAK_RL_PARALLEL_STEPS,
    /// IDRAK_RL_EVALUATION_EPISODES, IDRAK_RL_CHECKPOINTS, IDRAK_RL_RESUME, IDRAK_RL_LOG.
    /// </summary>
    /// <exception cref="ArgumentException">A variable's value is not one its option takes (the message names the variable).</exception>
    public static OffPolicyOptions FromEnvironment(OffPolicyOptions? start = null)
    {
        var o = start ?? new OffPolicyOptions();
        o = Variables.Text("IDRAK_RL_DQN_LOSS") is { } loss ? o with { Algorithm = loss } : o;
        o = Variables.Switch("IDRAK_RL_DQN_DOUBLE") is var (doubleSet, doubleQ) && doubleSet ? o with { DoubleQ = doubleQ } : o;
        o = Variables.Float("IDRAK_RL_DQN_HUBER", 0f) is { } huber ? o with { Huber = huber } : o;
        o = Long("IDRAK_RL_DQN_STEPS", 1) is { } steps ? o with { Steps = steps } : o;
        o = Variables.Int("IDRAK_RL_DQN_ENVIRONMENTS", 1) is { } environments ? o with { Environments = environments } : o;
        o = Variables.Switch("IDRAK_RL_PARALLEL_STEPS") is var (parallelSet, parallel) && parallelSet ? o with { ParallelSteps = parallel } : o;
        o = Variables.Int("IDRAK_RL_DQN_REPLAY", 1) is { } replay ? o with { ReplayCapacity = replay } : o;
        o = Variables.Int("IDRAK_RL_DQN_BATCH", 1) is { } batch ? o with { BatchSize = batch } : o;
        o = Long("IDRAK_RL_DQN_LEARNING_STARTS", 0) is { } learningStarts ? o with { LearningStarts = learningStarts } : o;
        o = Variables.Int("IDRAK_RL_DQN_TRAIN_EVERY", 1) is { } trainEvery ? o with { TrainEvery = trainEvery } : o;
        o = Variables.Int("IDRAK_RL_DQN_GRADIENT_STEPS", 1) is { } gradientSteps ? o with { GradientSteps = gradientSteps } : o;
        o = Long("IDRAK_RL_DQN_TARGET_EVERY", 1) is { } targetEvery ? o with { TargetEvery = targetEvery } : o;
        o = Variables.Float("IDRAK_RL_DQN_TAU", float.Epsilon, 1f) is { } tau ? o with { Tau = tau } : o;
        o = Variables.Float("IDRAK_RL_GAMMA", 0f, 1f) is { } gamma ? o with { Gamma = gamma } : o;
        o = Variables.Float("IDRAK_RL_DQN_EPSILON_START", 0f, 1f) is { } epsilonStart ? o with { EpsilonStart = epsilonStart } : o;
        o = Variables.Float("IDRAK_RL_DQN_EPSILON_END", 0f, 1f) is { } epsilonEnd ? o with { EpsilonEnd = epsilonEnd } : o;
        o = Variables.Float("IDRAK_RL_DQN_EPSILON_FRACTION", 0f, 1f) is { } epsilonFraction ? o with { EpsilonFraction = epsilonFraction } : o;
        o = Variables.Text("IDRAK_RL_OPTIMIZER") is { } optimizer ? o with { Optimizer = optimizer } : o;
        o = Variables.Float("IDRAK_RL_DQN_LEARNING_RATE", 0f) is { } rate ? o with { LearningRate = rate } : o;
        o = Variables.Float("IDRAK_RL_DQN_MAX_GRAD_NORM", 0f) is { } norm ? o with { MaxGradientNorm = norm } : o;
        o = Variables.Text("IDRAK_RL_SCHEDULE") is { } schedule ? o with { Schedule = schedule } : o;
        o = Variables.Float("IDRAK_RL_WARMUP", 0f, 1f) is { } warmup ? o with { ScheduleSettings = o.ScheduleSettings with { WarmupFraction = warmup } } : o;
        o = Variables.Int("IDRAK_RL_MAX_STEPS", 0) is { } maxSteps ? o with { MaxEpisodeSteps = maxSteps } : o;
        o = Variables.Int("IDRAK_RL_SEED", int.MinValue) is { } seed ? o with { Seed = seed } : o;
        o = Long("IDRAK_RL_DQN_REPORT_EVERY", 1) is { } reportEvery ? o with { ReportEvery = reportEvery } : o;
        o = Long("IDRAK_RL_DQN_EVALUATE_EVERY", 0) is { } evaluateEvery ? o with { EvaluateEvery = evaluateEvery } : o;
        o = Variables.Int("IDRAK_RL_EVALUATION_EPISODES", 1) is { } evaluationEpisodes ? o with { EvaluationEpisodes = evaluationEpisodes } : o;
        o = Long("IDRAK_RL_DQN_SAVE_EVERY", 0) is { } saveEvery ? o with { SaveEvery = saveEvery } : o;
        o = Variables.Text("IDRAK_RL_CHECKPOINTS") is { } checkpoints ? o with { CheckpointFolder = checkpoints } : o;
        o = Variables.Text("IDRAK_RL_RESUME") is { } resume ? o with { Resume = resume } : o;
        o = Variables.Text("IDRAK_RL_LOG") is { } log ? o with { Log = log } : o;
        return o;
    }

    private static long? Long(string name, long min)
    {
        if (Variables.Text(name) is not { } text)
        {
            return null;
        }

        return long.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out long value) && value >= min
            ? value
            : throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"{name}={text} is not a whole number of at least {min}."));
    }

    // The settings that cannot work, refused before a run starts (the names are checked by their registries).
    internal void Validate()
    {
        static void Check(bool ok, string name, object value, string rule)
        {
            if (!ok)
            {
                throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"OffPolicyOptions.{name} is {value}: {rule}."));
            }
        }

        ArgumentException.ThrowIfNullOrWhiteSpace(Algorithm);
        Check(Steps > 0, nameof(Steps), Steps, "it is positive");
        Check(Environments > 0, nameof(Environments), Environments, "it is positive");
        Check(ReplayCapacity > 0, nameof(ReplayCapacity), ReplayCapacity, "it is positive");
        Check(BatchSize > 0, nameof(BatchSize), BatchSize, "it is positive");
        Check(LearningStarts >= 0, nameof(LearningStarts), LearningStarts, "it is 0 or positive");
        Check(TrainEvery > 0, nameof(TrainEvery), TrainEvery, "it is positive");
        Check(GradientSteps > 0, nameof(GradientSteps), GradientSteps, "it is positive");
        Check(TargetEvery > 0, nameof(TargetEvery), TargetEvery, "it is positive");
        Check(Tau > 0 && Tau <= 1, nameof(Tau), Tau, "it is in (0, 1]");
        Check(Gamma >= 0 && Gamma <= 1, nameof(Gamma), Gamma, "it is in [0, 1]");
        Check(EpsilonStart >= 0 && EpsilonStart <= 1, nameof(EpsilonStart), EpsilonStart, "it is in [0, 1]");
        Check(EpsilonEnd >= 0 && EpsilonEnd <= 1, nameof(EpsilonEnd), EpsilonEnd, "it is in [0, 1]");
        Check(EpsilonFraction >= 0 && EpsilonFraction <= 1, nameof(EpsilonFraction), EpsilonFraction, "it is in [0, 1]");
        Check(LearningRate >= 0 && float.IsFinite(LearningRate), nameof(LearningRate), LearningRate, "it is 0 or positive");
        Check(MaxGradientNorm >= 0 && float.IsFinite(MaxGradientNorm), nameof(MaxGradientNorm), MaxGradientNorm, "it is 0 (no clip) or positive");
        Check(MaxEpisodeSteps >= 0, nameof(MaxEpisodeSteps), MaxEpisodeSteps, "it is 0 or positive");
        Check(ReportEvery > 0, nameof(ReportEvery), ReportEvery, "it is positive");
        Check(EvaluateEvery >= 0, nameof(EvaluateEvery), EvaluateEvery, "it is 0 or positive");
        Check(EvaluationEpisodes > 0, nameof(EvaluationEpisodes), EvaluationEpisodes, "it is positive");
        Check(SaveEvery >= 0, nameof(SaveEvery), SaveEvery, "it is 0 or positive");
        Check(Huber is null || (Huber >= 0 && float.IsFinite(Huber.Value)), nameof(Huber), Huber ?? 0, "it is 0 or positive");
    }
}
