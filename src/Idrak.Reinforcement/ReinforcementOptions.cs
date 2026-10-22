// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Reinforcement;

/// <summary>
/// The settings of a reinforcement-learning run (<see cref="ReinforcementTrainer{TObservation, TAction}"/>). Every one has a
/// default and an <c>IDRAK_RL_*</c> variable <see cref="FromEnvironment"/> reads (an app's explicit options win: they read
/// no variable). The estimator's and the loss's options are passed to them only when they take the key
/// (<c>IAdvantageEstimator.Keys</c>, <c>IPolicyLoss.Keys</c>); <see cref="EstimatorOptions"/> and <see cref="LossOptions"/>
/// are passed as they are, after those, so an option one of them does not take is refused by name.
/// </summary>
public sealed record ReinforcementOptions
{
    /// <summary>The algorithm: a <c>PolicyLosses</c> name (reinforce, ppo, grpo, or an app's). <c>IDRAK_RL_ALGORITHM</c>.</summary>
    public string Algorithm { get; init; } = ReinforcementNames.Ppo;

    /// <summary>
    /// The advantage estimator, an <c>AdvantageEstimators</c> name; null for the algorithm's: grpo's group, ppo's gae (returns
    /// when the policy has no critic), returns for reinforce and an app's. <c>IDRAK_RL_ESTIMATOR</c>.
    /// </summary>
    public string? Estimator { get; init; }

    /// <summary>
    /// The episodes compared with each other (reset with the same seed: the same situation played again); 0 for automatic:
    /// for the group and leave-one-out estimators the largest of 8, 4, 2 that divides <see cref="EpisodesPerIteration"/>, else
    /// 1. <c>IDRAK_RL_GROUP</c>.
    /// </summary>
    public int GroupSize { get; init; }

    /// <summary>The episodes of each iteration's rollout, all played at once; a multiple of the group size. <c>IDRAK_RL_EPISODES</c>.</summary>
    public int EpisodesPerIteration { get; init; } = 16;

    /// <summary>The most environments stepped at once (threads); 0 for <c>ComputeResources.MaxCpuThreads</c>. Results do not depend on it. <c>IDRAK_RL_ENVIRONMENTS</c>.</summary>
    public int Environments { get; init; }

    /// <summary>Whether environments step in parallel: true, false, or null to measure which is faster (rule 79). <c>IDRAK_RL_PARALLEL_STEPS</c> (on, off, auto).</summary>
    public bool? ParallelSteps { get; init; }

    /// <summary>
    /// The shards each rollout is cut into (<see cref="RolloutSpread"/>): a shard's episodes act together, one call of the
    /// policy a round with a seed of the shard's own, and on a cluster each rank plays the episodes of its shards. 0 for one
    /// per rank (1 in one process: the rollout acts as one call, as before). A seeded run gives the same episodes on any
    /// number of ranks that share a shard count, so set it to a multiple of every rank count a run is compared on.
    /// <c>IDRAK_RL_SHARDS</c>.
    /// </summary>
    public int Shards { get; init; }

    /// <summary>How the episodes are cut into shards: <see cref="RolloutSpread.Blocks"/> (runs in number order, a group's together) or <see cref="RolloutSpread.Interleaved"/>. <c>IDRAK_RL_SPREAD</c>.</summary>
    public string Spread { get; init; } = RolloutSpread.Blocks;

    /// <summary>The iterations of the run (each a rollout, its advantages and the updates). <c>IDRAK_RL_ITERATIONS</c>.</summary>
    public int Iterations { get; init; } = 100;

    /// <summary>The passes over each rollout's episodes; 0 for the algorithm's (4 for ppo, else 1). <c>IDRAK_RL_EPOCHS</c>.</summary>
    public int Epochs { get; init; }

    /// <summary>The episodes of each update (an optimizer step); 0 for all of the rollout. <c>IDRAK_RL_MINIBATCH</c>.</summary>
    public int MiniBatch { get; init; }

    /// <summary>The discount γ (gae, returns). <c>IDRAK_RL_GAMMA</c>.</summary>
    public float Gamma { get; init; } = 0.99f;

    /// <summary>GAE's λ. <c>IDRAK_RL_LAMBDA</c>.</summary>
    public float Lambda { get; init; } = 0.95f;

    /// <summary>Whether the group estimator divides by the group's spread; null for the estimator's (true). <c>IDRAK_RL_GROUP_SCALE</c>.</summary>
    public bool? GroupScale { get; init; }

    /// <summary>Whether gae and returns normalize the advantages over the batch; null for the estimator's (false). <c>IDRAK_RL_NORMALIZE_ADVANTAGES</c>.</summary>
    public bool? NormalizeAdvantages { get; init; }

    /// <summary>The ratio's clip ε (ppo, grpo). <c>IDRAK_RL_CLIP</c>.</summary>
    public float Clip { get; init; } = 0.2f;

    /// <summary>The weight β of the KL penalty to the reference (ppo, grpo); above 0 needs a policy with a reference. <c>IDRAK_RL_KL</c>.</summary>
    public float KlWeight { get; init; }

    /// <summary>The KL estimator: k1, k2 or k3. <c>IDRAK_RL_KL_ESTIMATOR</c>.</summary>
    public string KlEstimator { get; init; } = "k3";

    /// <summary>The weight of the entropy bonus. <c>IDRAK_RL_ENTROPY</c>.</summary>
    public float EntropyWeight { get; init; }

    /// <summary>The weight of the value head's squared error (ppo, reinforce with a critic). <c>IDRAK_RL_VALUE</c>.</summary>
    public float ValueWeight { get; init; } = 0.5f;

    /// <summary>How the loss divides the step's sum: token, sequence or constant; null for the loss's own. <c>IDRAK_RL_NORMALIZE</c>.</summary>
    public string? Normalization { get; init; }

    /// <summary>The largest global gradient norm an update applies (0: no clip), for a policy that exposes its optimizer. <c>IDRAK_RL_MAX_GRAD_NORM</c>.</summary>
    public float MaxGradientNorm { get; init; } = 1f;

    /// <summary>The learning-rate schedule over the run's optimizer steps, a <c>LearningRateSchedules</c> name. <c>IDRAK_RL_SCHEDULE</c>.</summary>
    public string Schedule { get; init; } = LearningRateSchedules.Constant;

    /// <summary>The schedule's settings; no warm-up by default. <c>IDRAK_RL_WARMUP</c> sets its warm-up share.</summary>
    public LearningRateScheduleSettings ScheduleSettings { get; init; } = new(WarmupFraction: 0f);

    /// <summary>The temperature rollouts sample at (1: the policy as it is). <c>IDRAK_RL_TEMPERATURE</c>.</summary>
    public float Temperature { get; init; } = 1f;

    /// <summary>The most steps of an episode before the trainer cuts it (truncated); 0 for the environment's own limit. <c>IDRAK_RL_MAX_STEPS</c>.</summary>
    public int MaxEpisodeSteps { get; init; }

    /// <summary>The run's seed: every episode's seed, every draw of the policy and every shuffle derive from it. <c>IDRAK_RL_SEED</c>.</summary>
    public int Seed { get; init; }

    /// <summary>Evaluation episodes (greedy, seeds of their own) after every this many iterations; 0 for none. <c>IDRAK_RL_EVALUATE_EVERY</c>.</summary>
    public int EvaluateEvery { get; init; }

    /// <summary>The episodes of an evaluation. <c>IDRAK_RL_EVALUATION_EPISODES</c>.</summary>
    public int EvaluationEpisodes { get; init; } = 8;

    /// <summary>The evaluation episodes' seed; null for one derived from <see cref="Seed"/>. The same episodes every evaluation.</summary>
    public int? EvaluationSeed { get; init; }

    /// <summary>A checkpoint into <see cref="CheckpointFolder"/> after every this many iterations (and at the end); 0 for the end only. <c>IDRAK_RL_SAVE_EVERY</c>.</summary>
    public int SaveEvery { get; init; }

    /// <summary>Where checkpoints are written (iteration-N folders); null for none. <c>IDRAK_RL_CHECKPOINTS</c>.</summary>
    public string? CheckpointFolder { get; init; }

    /// <summary>A checkpoint to resume from: an iteration-N folder, or a checkpoint folder (its latest); null to start afresh. <c>IDRAK_RL_RESUME</c>.</summary>
    public string? Resume { get; init; }

    /// <summary>A file each iteration's summary is appended to as a JSON line; null for none. <c>IDRAK_RL_LOG</c>.</summary>
    public string? Log { get; init; }

    /// <summary>
    /// The return an episode passes at: an iteration's pass rate (its summary's "pass-rate", the log's pass_rate) is the share
    /// of its episodes whose return reaches it. 1 by default, the full score of the rewards that run from 0 to 1 (exact,
    /// tests-passed, a metric); set it for an environment whose returns are on another scale; null for no pass rate.
    /// <c>IDRAK_RL_PASS_RETURN</c> (a number, or none).
    /// </summary>
    public float? PassReturn { get; init; } = 1f;

    /// <summary>The environment kind's options.</summary>
    public IReadOnlyDictionary<string, string>? EnvironmentOptions { get; init; }

    /// <summary>More options for the estimator, passed as they are (a baseline for returns, say).</summary>
    public IReadOnlyDictionary<string, string>? EstimatorOptions { get; init; }

    /// <summary>More options for the loss, passed as they are (clip-high, constant).</summary>
    public IReadOnlyDictionary<string, string>? LossOptions { get; init; }

    /// <summary>
    /// These options with each one an <c>IDRAK_RL_*</c> variable sets in its place; the variables not set keep
    /// <paramref name="start"/>'s (the defaults when null). The variables: IDRAK_RL_ALGORITHM, _ESTIMATOR, _GROUP, _EPISODES,
    /// _ENVIRONMENTS, _PARALLEL_STEPS, _SHARDS, _SPREAD, _ITERATIONS, _EPOCHS, _MINIBATCH, _GAMMA, _LAMBDA, _GROUP_SCALE,
    /// _NORMALIZE_ADVANTAGES, _CLIP, _KL, _KL_ESTIMATOR, _ENTROPY, _VALUE, _NORMALIZE, _MAX_GRAD_NORM, _SCHEDULE, _WARMUP,
    /// _TEMPERATURE, _MAX_STEPS, _SEED, _EVALUATE_EVERY, _EVALUATION_EPISODES, _SAVE_EVERY, _CHECKPOINTS, _RESUME, _LOG,
    /// _PASS_RETURN.
    /// </summary>
    /// <exception cref="ArgumentException">A variable's value is not one its option takes (the message names the variable).</exception>
    public static ReinforcementOptions FromEnvironment(ReinforcementOptions? start = null)
    {
        var o = start ?? new ReinforcementOptions();
        o = Variables.Text("IDRAK_RL_ALGORITHM") is { } algorithm ? o with { Algorithm = algorithm } : o;
        o = Variables.Text("IDRAK_RL_ESTIMATOR") is { } estimator ? o with { Estimator = estimator } : o;
        o = Variables.Int("IDRAK_RL_GROUP", 0) is { } group ? o with { GroupSize = group } : o;
        o = Variables.Int("IDRAK_RL_EPISODES", 1) is { } episodes ? o with { EpisodesPerIteration = episodes } : o;
        o = Variables.Int("IDRAK_RL_ENVIRONMENTS", 0) is { } environments ? o with { Environments = environments } : o;
        o = Variables.Switch("IDRAK_RL_PARALLEL_STEPS") is var (set, parallel) && set ? o with { ParallelSteps = parallel } : o;
        o = Variables.Int(RolloutSpread.ShardsVariable, 0) is { } shards ? o with { Shards = shards } : o;
        o = Variables.Text(RolloutSpread.SpreadVariable) is { } spread ? o with { Spread = ParseSpread(spread) } : o;
        o = Variables.Int("IDRAK_RL_ITERATIONS", 0) is { } iterations ? o with { Iterations = iterations } : o;
        o = Variables.Int("IDRAK_RL_EPOCHS", 0) is { } epochs ? o with { Epochs = epochs } : o;
        o = Variables.Int("IDRAK_RL_MINIBATCH", 0) is { } miniBatch ? o with { MiniBatch = miniBatch } : o;
        o = Variables.Float("IDRAK_RL_GAMMA", 0f, 1f) is { } gamma ? o with { Gamma = gamma } : o;
        o = Variables.Float("IDRAK_RL_LAMBDA", 0f, 1f) is { } lambda ? o with { Lambda = lambda } : o;
        o = Variables.Switch("IDRAK_RL_GROUP_SCALE") is var (scaleSet, scale) && scaleSet ? o with { GroupScale = scale } : o;
        o = Variables.Switch("IDRAK_RL_NORMALIZE_ADVANTAGES") is var (normalizeSet, normalize) && normalizeSet ? o with { NormalizeAdvantages = normalize } : o;
        o = Variables.Float("IDRAK_RL_CLIP", 0f, 1f) is { } clip ? o with { Clip = clip } : o;
        o = Variables.Float("IDRAK_RL_KL", 0f) is { } kl ? o with { KlWeight = kl } : o;
        o = Variables.Text("IDRAK_RL_KL_ESTIMATOR") is { } klEstimator ? o with { KlEstimator = klEstimator } : o;
        o = Variables.Float("IDRAK_RL_ENTROPY") is { } entropy ? o with { EntropyWeight = entropy } : o;
        o = Variables.Float("IDRAK_RL_VALUE", 0f) is { } value ? o with { ValueWeight = value } : o;
        o = Variables.Text("IDRAK_RL_NORMALIZE") is { } normalization ? o with { Normalization = normalization } : o;
        o = Variables.Float("IDRAK_RL_MAX_GRAD_NORM", 0f) is { } norm ? o with { MaxGradientNorm = norm } : o;
        o = Variables.Text("IDRAK_RL_SCHEDULE") is { } schedule ? o with { Schedule = schedule } : o;
        o = Variables.Float("IDRAK_RL_WARMUP", 0f, 1f) is { } warmup ? o with { ScheduleSettings = o.ScheduleSettings with { WarmupFraction = warmup } } : o;
        o = Variables.Float("IDRAK_RL_TEMPERATURE", float.Epsilon) is { } temperature ? o with { Temperature = temperature } : o;
        o = Variables.Int("IDRAK_RL_MAX_STEPS", 0) is { } maxSteps ? o with { MaxEpisodeSteps = maxSteps } : o;
        o = Variables.Int("IDRAK_RL_SEED", int.MinValue) is { } seed ? o with { Seed = seed } : o;
        o = Variables.Int("IDRAK_RL_EVALUATE_EVERY", 0) is { } evaluateEvery ? o with { EvaluateEvery = evaluateEvery } : o;
        o = Variables.Int("IDRAK_RL_EVALUATION_EPISODES", 1) is { } evaluationEpisodes ? o with { EvaluationEpisodes = evaluationEpisodes } : o;
        o = Variables.Int("IDRAK_RL_SAVE_EVERY", 0) is { } saveEvery ? o with { SaveEvery = saveEvery } : o;
        o = Variables.Text("IDRAK_RL_CHECKPOINTS") is { } checkpoints ? o with { CheckpointFolder = checkpoints } : o;
        o = Variables.Text("IDRAK_RL_RESUME") is { } resume ? o with { Resume = resume } : o;
        o = Variables.Text("IDRAK_RL_LOG") is { } log ? o with { Log = log } : o;
        o = Variables.Text(PassReturnVariable) is { } pass ? o with { PassReturn = ParsePassReturn(pass) } : o;
        return o;
    }

    /// <summary>The variable of <see cref="PassReturn"/>.</summary>
    public const string PassReturnVariable = "IDRAK_RL_PASS_RETURN";

    // The pass return named by its variable: a finite number, or none.
    private static float? ParsePassReturn(string text) =>
        text.Equals("none", StringComparison.OrdinalIgnoreCase) ? null
        : float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out float value) && float.IsFinite(value) ? value
        : throw new ArgumentException($"{PassReturnVariable}={text} is not a number (the return an episode passes at) or none.");

    // A split named by its variable, refused naming the variable.
    private static string ParseSpread(string text)
    {
        try
        {
            return RolloutSpread.Parse(text);
        }
        catch (ArgumentException)
        {
            throw new ArgumentException($"{RolloutSpread.SpreadVariable}={text} is not {string.Join(" or ", RolloutSpread.Names)}.");
        }
    }

    // The settings that cannot work, refused before a run starts (the names are checked by their registries).
    internal void Validate()
    {
        static void Positive(int value, string name)
        {
            if (value <= 0)
            {
                throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"ReinforcementOptions.{name} is {value}: it is positive."));
            }
        }

        static void NotNegative(int value, string name)
        {
            if (value < 0)
            {
                throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"ReinforcementOptions.{name} is {value}: it is 0 or positive."));
            }
        }

        ArgumentException.ThrowIfNullOrWhiteSpace(Algorithm);
        Positive(EpisodesPerIteration, nameof(EpisodesPerIteration));
        Positive(EvaluationEpisodes, nameof(EvaluationEpisodes));
        NotNegative(Iterations, nameof(Iterations));
        NotNegative(GroupSize, nameof(GroupSize));
        NotNegative(Environments, nameof(Environments));
        NotNegative(Shards, nameof(Shards));
        _ = RolloutSpread.Parse(Spread);
        NotNegative(Epochs, nameof(Epochs));
        NotNegative(MiniBatch, nameof(MiniBatch));
        NotNegative(MaxEpisodeSteps, nameof(MaxEpisodeSteps));
        NotNegative(EvaluateEvery, nameof(EvaluateEvery));
        NotNegative(SaveEvery, nameof(SaveEvery));
        if (!(Temperature > 0) || !float.IsFinite(Temperature))
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"ReinforcementOptions.Temperature is {Temperature}: it is positive and finite."));
        }

        if (PassReturn is { } pass && !float.IsFinite(pass))
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"ReinforcementOptions.PassReturn is {pass}: a finite number, or null for no pass rate."));
        }

        if (!(MaxGradientNorm >= 0) || !float.IsFinite(MaxGradientNorm))
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"ReinforcementOptions.MaxGradientNorm is {MaxGradientNorm}: 0 (no clip) or positive."));
        }
    }
}

// The IDRAK_RL_* variables: unset or empty is null; a value that does not parse is refused naming the variable.
internal static class Variables
{
    public static string? Text(string name) => Environment.GetEnvironmentVariable(name) is { Length: > 0 } value && value.Trim().Length > 0 ? value.Trim() : null;

    public static int? Int(string name, int min)
    {
        if (Text(name) is not { } text)
        {
            return null;
        }

        return int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) && value >= min
            ? value
            : throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"{name}={text} is not a whole number of at least {min}."));
    }

    public static float? Float(string name, float min = float.NegativeInfinity, float max = float.PositiveInfinity)
    {
        if (Text(name) is not { } text)
        {
            return null;
        }

        return float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out float value) && float.IsFinite(value) && value >= min && value <= max
            ? value
            : throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"{name}={text} is not a number from {min} to {max}."));
    }

    // on / off / auto (and true, false, 1, 0): (set, value) with value null for auto.
    public static (bool Set, bool? Value) Switch(string name) => Text(name)?.ToLowerInvariant() switch
    {
        null => (false, null),
        "on" or "true" or "1" or "yes" => (true, true),
        "off" or "false" or "0" or "no" => (true, false),
        "auto" => (true, null),
        var other => throw new ArgumentException($"{name}={other} is not on, off or auto."),
    };
}
