// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Reinforcement;
using Idrak.Abstraction.Testing;
using Idrak.Reinforcement;

// Plan 27, phase R2: Idrak.Reinforcement, the general part: the advantage estimators and policy losses (their contract
// suites and direct computations on fixed inputs), the module policy, the rollout collector and the trainer, on fakes
// written here (not shipped): a two-armed bandit, a short corridor and a box with a rewarded point.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] ReinforcementGroup =
    [
        ("reinforcement: the estimators pass their contract suites and match direct computations on fixed episodes (gae, group with and without its scale, leave-one-out, returns with and without a baseline)", ReinforcementEstimators),
        ("reinforcement: the policy losses pass their contract suites and match direct computations on fixed units (reinforce, ppo with values, entropy and a k3 KL, grpo per sequence, with a k1 KL and by a constant)", ReinforcementLosses),
        ("reinforcement: the estimators and losses are the registries' library defaults from Idrak.Reinforcement; unknown names, options and group sizes are refused naming what is wrong", ReinforcementDefaults),
        ("reinforcement: REINFORCE, PPO and GRPO raise a two-armed bandit's reward to the best arm's within a seeded number of iterations", ReinforcementBandit),
        ("reinforcement: REINFORCE, PPO and GRPO walk a short corridor in the fewest steps (the optimum return) within a seeded number of iterations", ReinforcementCorridor),
        ("reinforcement: a Gaussian module policy over a box moves its mean to the rewarded action; its first ratio is 1", ReinforcementGaussian),
        ("reinforcement: a seeded run gives the same rewards, advantages and weights twice, and on 1 and 4 threads stepping in parallel or not", ReinforcementReproducible),
        ("reinforcement: a resumed run matches an uninterrupted one bit for bit (rewards, advantages, weights, the schedule's rate)", ReinforcementResume),
        ("reinforcement: memory held after iteration 2 does not grow by iteration 20 (device memory, managed heap, environments made)", ReinforcementMemory),
        ("reinforcement: the collector plays episodes at once in number order and measures parallel steps; a cancelled run stops between iterations with a checkpoint; progress, telemetry and the JSON log report each iteration", ReinforcementCollectorAndReports),
        ("reinforcement: options read their IDRAK_RL_* variables and refuse bad values naming the variable", ReinforcementOptionVariables),
    ];

    private const float CorridorOptimum = 0.97f;                                // 4 steps right in a corridor of 5: three of -0.01, then 1

    // A two-armed bandit: one step, arm a pays 1 when the episode's draw u (from its seed) is below p_a, else 0; episodes of
    // a group share u, so their arms are compared on the same luck.
    private sealed class RlBanditKind : IEnvironmentKind
    {
        public string Name => "reinforcement-test-bandit";

        public string Summary => "a two-armed bandit (the reinforcement tests)";

        public IReadOnlyCollection<string> Keys { get; } = ["low", "high"];

        public Type ObservationType => typeof(float[]);

        public Type ActionType => typeof(int);

        public IEnvironment Create(IReadOnlyDictionary<string, string>? options = null, int seed = 0)
        {
            float low = 0.2f, high = 0.8f;
            foreach (var (key, value) in options ?? new Dictionary<string, string>())
            {
                switch (key)
                {
                    case "low": low = float.Parse(value, CultureInfo.InvariantCulture); break;
                    case "high": high = float.Parse(value, CultureInfo.InvariantCulture); break;
                    default: throw new ArgumentException($"The environment kind {Name} does not take the option '{key}'; it takes: {string.Join(", ", Keys)}.");
                }
            }

            return new RlBandit(low, high);
        }
    }

    private sealed class RlBandit(float low, float high) : IEnvironment<float[], int>
    {
        private double _draw;
        private bool _running;

        public Space ObservationSpace { get; } = Space.Box([1], 0f, 0f);

        public Space ActionSpace { get; } = Space.Discrete(2);

        public float[] Reset(int seed)
        {
            _draw = new Random(seed).NextDouble();
            _running = true;
            return [0f];
        }

        public ValueTask<EnvironmentStep<float[]>> StepAsync(int action, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (!_running)
            {
                throw new InvalidOperationException("Reset the bandit first.");
            }

            if (!ActionSpace.Contains(action))
            {
                throw new ArgumentException($"{action} is not in {ActionSpace}.", nameof(action));
            }

            _running = false;
            return ValueTask.FromResult(new EnvironmentStep<float[]>([0f], _draw < (action == 1 ? high : low) ? 1f : 0f, true, false));
        }

        public void Dispose() => _running = false;
    }

    // A corridor of `length` cells walked from the left end: 0 steps left (against the wall: stays), 1 right; reaching the
    // right end pays 1 and terminates, every other step costs 0.01, and `limit` steps truncate. The observation is the
    // cell, one-hot. Deterministic: its optimum is length - 1 steps right.
    private sealed class RlCorridorKind(int length = 5, int limit = 20) : IEnvironmentKind
    {
        private int _created;

        public string Name => "reinforcement-test-walk";

        public string Summary => "a short corridor walked to its right end (the reinforcement tests)";

        public IReadOnlyCollection<string> Keys { get; } = [];

        public Type ObservationType => typeof(float[]);

        public Type ActionType => typeof(int);

        public int Created => Volatile.Read(ref _created);

        public IEnvironment Create(IReadOnlyDictionary<string, string>? options = null, int seed = 0)
        {
            Interlocked.Increment(ref _created);
            return new RlCorridor(length, limit);
        }
    }

    private sealed class RlCorridor(int length, int limit) : IEnvironment<float[], int>
    {
        private int _cell, _steps;
        private bool _running;

        public Space ObservationSpace { get; } = Space.Box([length], 0f, 1f);

        public Space ActionSpace { get; } = Space.Discrete(2);

        public float[] Reset(int seed)
        {
            _cell = 0;
            _steps = 0;
            _running = true;
            return Observe();
        }

        public ValueTask<EnvironmentStep<float[]>> StepAsync(int action, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (!_running)
            {
                throw new InvalidOperationException("Reset the corridor first.");
            }

            if (!ActionSpace.Contains(action))
            {
                throw new ArgumentException($"{action} is not in {ActionSpace}.", nameof(action));
            }

            _cell = Math.Clamp(_cell + (action == 1 ? 1 : -1), 0, length - 1);
            _steps++;
            bool terminated = _cell == length - 1, truncated = !terminated && _steps >= limit;
            _running = !(terminated || truncated);
            return ValueTask.FromResult(new EnvironmentStep<float[]>(Observe(), terminated ? 1f : -0.01f, terminated, truncated));
        }

        public void Dispose() => _running = false;

        private float[] Observe()
        {
            var observation = new float[length];
            observation[_cell] = 1f;
            return observation;
        }
    }

    // A box of one action in [-2, 2]: one step, rewarded 1 - (a - 1)².
    private sealed class RlTargetKind : IEnvironmentKind
    {
        public string Name => "reinforcement-test-target";

        public string Summary => "a rewarded point in a box (the reinforcement tests)";

        public IReadOnlyCollection<string> Keys { get; } = [];

        public Type ObservationType => typeof(float[]);

        public Type ActionType => typeof(float[]);

        public IEnvironment Create(IReadOnlyDictionary<string, string>? options = null, int seed = 0) => new RlTarget();
    }

    private sealed class RlTarget : IEnvironment<float[], float[]>
    {
        private bool _running;

        public Space ObservationSpace { get; } = Space.Box([1], 0f, 0f);

        public Space ActionSpace { get; } = Space.Box([1], -2f, 2f);

        public float[] Reset(int seed)
        {
            _running = true;
            return [0f];
        }

        public ValueTask<EnvironmentStep<float[]>> StepAsync(float[] action, CancellationToken cancellationToken = default)
        {
            if (!_running)
            {
                throw new InvalidOperationException("Reset the target first.");
            }

            if (!ActionSpace.Contains(action))
            {
                throw new ArgumentException("The action is not in the box.", nameof(action));
            }

            _running = false;
            return ValueTask.FromResult(new EnvironmentStep<float[]>([0f], 1f - (action[0] - 1f) * (action[0] - 1f), true, false));
        }

        public void Dispose() => _running = false;
    }

    // The telemetry of a run, counted.
    private sealed class RlHook : ITelemetryHook
    {
        public TelemetryLevel Levels => TelemetryLevel.Training | TelemetryLevel.Batches;

        public int Started { get; private set; }

        public int Batches { get; private set; }

        public int Completed { get; private set; }

        public List<EpochCompleted> Epochs { get; } = [];

        public void OnTrainingStarted(in TrainingStarted e) => Started++;

        public void OnBatchCompleted(in BatchCompleted e) => Batches++;

        public void OnEpochCompleted(in EpochCompleted e) => Epochs.Add(e);

        public void OnTrainingCompleted(in TrainingCompleted e) => Completed++;
    }

    private static (Space Observations, Space Actions) Spaces(IEnvironmentKind kind)
    {
        using var environment = kind.Create();
        return (environment.ObservationSpace, environment.ActionSpace);
    }

    // A categorical policy over the kind's spaces: a perceptron of one hidden layer of 16, a critic of the same shape when
    // asked, weights from `seed`.
    private static ModulePolicy<int> RlPolicy(IEnvironmentKind kind, Device device, int seed, float rate, bool critic = false, bool reference = false)
    {
        var (observations, actions) = Spaces(kind);
        var random = new Random(seed);
        var actor = ModulePolicy.Perceptron(observations.Size, actions.Count, [16], device, random);
        var value = critic ? ModulePolicy.Perceptron(observations.Size, 1, [16], device, random) : null;
        var frozen = reference ? ModulePolicy.Perceptron(observations.Size, actions.Count, [16], device, new Random(1)) : null;
        return ModulePolicy.Categorical(observations, actions, actor, value, frozen, new ModulePolicyOptions { LearningRate = rate });
    }

    private static float[] Weights<TAction>(ModulePolicy<TAction> policy) => [.. policy.Parameters.SelectMany(p => p.ToArray())];

    private static string? SameBits(float[] expected, float[] actual, string what)
    {
        if (expected.Length != actual.Length)
        {
            return $"{what}: {actual.Length} values, {expected.Length} expected";
        }

        for (int i = 0; i < expected.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
            {
                return FormattableString.Invariant($"{what}: element {i} is {actual[i]:R}, {expected[i]:R} expected");
            }
        }

        return null;
    }

    // Trains until `done` says so (the run is then cancelled before its next iteration); the iteration it was reached at, or 0.
    private static (ReinforcementResult Result, int Reached) TrainUntil<TAction>(IEnvironmentKind kind, IPolicy<float[], TAction> policy, ReinforcementOptions options,
        Func<IterationSummary, bool> done)
    {
        using var stop = new CancellationTokenSource();
        int reached = 0;
        var trainer = new ReinforcementTrainer<float[], TAction>(kind, policy, options)
        {
            OnIteration = s =>
            {
                if (reached == 0 && done(s))
                {
                    reached = s.Iteration;
                    stop.Cancel();
                }
            },
        };
        return (trainer.Train(stop.Token), reached);
    }

    // ---------------------------------------------------------------- estimators

    private static void ReinforcementEstimators(Device device)
    {
        _ = device;
        foreach (var (name, options) in new (string, Dictionary<string, string>?)[]
                 {
                     ("gae", null), ("gae", new() { ["gamma"] = "0.9", ["lambda"] = "0.8", ["normalize"] = "true" }), ("group", null), ("group", new() { ["scale"] = "false" }),
                     ("leave-one-out", null), ("returns", null), ("returns", new() { ["baseline"] = "mean", ["normalize"] = "true", ["gamma"] = "1" }),
                 })
        {
            var report = Conformance.CheckAdvantageEstimator(AdvantageEstimators.Get(name), options);
            Check(report.Passed, $"{name} ({string.Join(", ", options?.Select(o => $"{o.Key}={o.Value}") ?? [])}) passes the estimator suite:\n{report}");
        }

        // Fixed episodes: three of 2, 3 and 1 steps, the second truncated (bootstrapped from 0.5), the first two in a group.
        float[] rewards = [1f, 0f, 0.5f, -1f, 2f, 3f], values = [0.2f, 0.4f, 0.1f, -0.3f, 0.6f, 1f], finals = [0.7f, 0.5f, 0f];
        int[] lengths = [2, 3, 1], groups = [0, 0, 1];
        bool[] truncated = [false, true, false];
        (float[] Advantages, float[] Returns) Run(string name, Dictionary<string, string>? options)
        {
            var advantages = new float[6];
            var returns = new float[6];
            AdvantageEstimators.Get(name).Estimate(new AdvantageInput(rewards, values, lengths, truncated, finals, groups), advantages, returns, options);
            return (advantages, returns);
        }

        void Near(float[] actual, double[] expected, string what) =>
            Check(Comparisons.Difference(expected, [.. actual.Select(a => (double)a)], 1e-5) is null,
                $"{what}: {string.Join(", ", actual.Select(a => a.ToString("G6", CultureInfo.InvariantCulture)))}; expected {string.Join(", ", expected.Select(e => e.ToString("G6", CultureInfo.InvariantCulture)))}");

        // GAE as the sum of discounted TD errors forward from each step (not the backward recursion).
        const double gamma = 0.9, lambda = 0.8;
        var gae = new double[6];
        var gaeReturns = new double[6];
        int start = 0;
        for (int e = 0; e < 3; e++)
        {
            for (int t = start; t < start + lengths[e]; t++)
            {
                double sum = 0;
                for (int l = t; l < start + lengths[e]; l++)
                {
                    double next = l + 1 < start + lengths[e] ? values[l + 1] : truncated[e] ? finals[e] : 0.0;
                    sum += Math.Pow(gamma * lambda, l - t) * (rewards[l] + gamma * next - values[l]);
                }

                gae[t] = sum;
                gaeReturns[t] = sum + values[t];
            }

            start += lengths[e];
        }

        var (gaeA, gaeR) = Run("gae", new() { ["gamma"] = "0.9", ["lambda"] = "0.8" });
        Near(gaeA, gae, "gae advantages");
        Near(gaeR, gaeReturns, "gae returns");

        // Group: returns 1, 1.5 (episodes 0, 1: one group) and 3 (alone).
        double mean = 1.25, spread = Math.Sqrt((0.25 * 0.25 + 0.25 * 0.25) / 1);
        double a0 = (1 - mean) / (spread + 1e-4f), a1 = (1.5 - mean) / (spread + 1e-4f);
        double[] toGo = [1, 0, 1.5, 1, 2, 3];
        var (groupA, groupR) = Run("group", null);
        Near(groupA, [a0, a0, a1, a1, a1, 0], "group advantages");
        Near(groupR, toGo, "group returns (rewards to go)");
        Near(Run("group", new() { ["scale"] = "false" }).Advantages, [-0.25, -0.25, 0.25, 0.25, 0.25, 0], "group advantages without the scale");
        var (looA, looR) = Run("leave-one-out", null);
        Near(looA, [-0.5, -0.5, 0.5, 0.5, 0.5, 0], "leave-one-out advantages (1 - 1.5, 1.5 - 1; a group of one 0)");
        Near(looR, toGo, "leave-one-out returns");

        // Returns: G discounted by 0.9, the truncated episode bootstrapped from 0.5.
        double g4 = 2 + 0.9 * 0.5, g3 = -1 + 0.9 * g4, g2 = 0.5 + 0.9 * g3;
        double[] g = [1 + 0.9 * 0, 0, g2, g3, g4, 3];
        var (retA, retR) = Run("returns", new() { ["gamma"] = "0.9" });
        Near(retA, g, "returns advantages (no baseline)");
        Near(retR, g, "returns");
        double gMean = g.Sum() / 6;
        Near(Run("returns", new() { ["gamma"] = "0.9", ["baseline"] = "mean" }).Advantages, [.. g.Select(v => v - gMean)], "returns less the batch's mean");
        Near(Run("returns", new() { ["gamma"] = "0.9", ["baseline"] = "value" }).Advantages, [.. g.Select((v, i) => v - values[i])], "returns less the critic's values");
        Check(Throws<ArgumentException>(() => AdvantageEstimators.Get("returns").Estimate(new AdvantageInput(rewards, [], lengths, truncated, [], groups), new float[6], new float[6],
            new Dictionary<string, string> { ["baseline"] = "value" })), "a value baseline without values is refused");
    }

    // ---------------------------------------------------------------- losses

    private static void ReinforcementLosses(Device device)
    {
        foreach (var (name, options, needsReference, needsValues) in new (string, Dictionary<string, string>?, bool, bool)[]
                 {
                     ("reinforce", null, false, false), ("reinforce", new() { ["normalize"] = "constant", ["constant"] = "7", ["entropy"] = "0.01" }, false, true),
                     ("ppo", null, false, false), ("ppo", new() { ["entropy"] = "0.01", ["kl"] = "0.05", ["clip-high"] = "0.28" }, true, true),
                     ("grpo", null, false, false), ("grpo", new() { ["kl"] = "0.1", ["kl-estimator"] = "k1", ["normalize"] = "token" }, true, false),
                     ("grpo", new() { ["kl"] = "0.04" }, true, false),
                 })
        {
            var report = Conformance.CheckPolicyLoss(PolicyLosses.Get(name), options, needsReference: needsReference, needsValues: needsValues);
            Check(report.Passed, $"{name} ({string.Join(", ", options?.Select(o => $"{o.Key}={o.Value}") ?? [])}) passes the policy-loss suite:\n{report}");
        }

        // Fixed units: two sequences of 2 and 3.
        double[] logp = [-0.5, -1.2, -0.3, -2.0, -0.8], behavior = [-0.6, -1.0, -0.3, -2.5, -0.7], reference = [-0.4, -1.1, -0.5, -2.1, -0.9];
        double[] advantages = [1.0, -0.5, 0.3, 2.0, -1.5], predicted = [0.1, 0.2, -0.3, 0.4, 0.0], targets = [0.5, 0.0, 0.1, 0.2, -0.2], entropy = [0.6, 0.5, 0.7, 0.4, 0.3];
        int[] counts = [2, 3];
        float Loss(string name, Dictionary<string, string>? options, bool withValues)
        {
            using var scope = new TensorScope();
            Tensor Of(double[] v) => Tensor.From([.. v.Select(x => (float)x)], [v.Length], device);
            var input = new PolicyLossInput(Of(logp), counts, Of(advantages), Of(behavior), Of(reference), withValues ? Of(predicted) : null, withValues ? Of(targets) : null, Of(entropy));
            return PolicyLosses.Get(name).Loss(input, options).Item();
        }

        double Surrogate(int u, double low, double high)
        {
            double ratio = Math.Exp(logp[u] - behavior[u]);
            return Math.Min(ratio * advantages[u], Math.Clamp(ratio, 1 - low, 1 + high) * advantages[u]);
        }

        double K3(int u) => Math.Exp(reference[u] - logp[u]) - (reference[u] - logp[u]) - 1;

        void Near(float actual, double expected, string what) =>
            Check(Math.Abs(actual - expected) <= 1e-5 * Math.Max(1, Math.Abs(expected)), FormattableString.Invariant($"{what}: {actual:G7}, expected {expected:G7}"));

        double reinforce = 0, ppo = 0, grpoSequence = 0, grpoK1 = 0, grpoConstant = 0;
        for (int u = 0; u < 5; u++)
        {
            int sequence = u < 2 ? 0 : 1;
            reinforce += (-advantages[u] * logp[u] + 0.5 * Math.Pow(predicted[u] - targets[u], 2)) / 5;
            ppo += (-Surrogate(u, 0.2, 0.2) + 0.1 * K3(u) - 0.01 * entropy[u] + 0.5 * Math.Pow(predicted[u] - targets[u], 2)) / 5;
            grpoSequence += -Surrogate(u, 0.2, 0.2) / (counts[sequence] * 2.0);
            grpoK1 += (-Surrogate(u, 0.2, 0.2) + 0.04 * (logp[u] - reference[u])) / (counts[sequence] * 2.0);
            grpoConstant += -Surrogate(u, 0.2, 0.28) / 10;
        }

        Near(Loss("reinforce", null, withValues: true), reinforce, "reinforce: −A log p and the value term, per token");
        Near(Loss("ppo", new() { ["kl"] = "0.1", ["entropy"] = "0.01" }, withValues: true), ppo, "ppo: the clipped surrogate, a k3 KL, the entropy bonus and the value term, per token");
        Near(Loss("grpo", null, withValues: true), grpoSequence, "grpo: the clipped surrogate per sequence (no value term)");
        Near(Loss("grpo", new() { ["kl"] = "0.04", ["kl-estimator"] = "k1" }, withValues: false), grpoK1, "grpo: a k1 KL to the reference");
        Near(Loss("grpo", new() { ["normalize"] = "constant", ["constant"] = "10", ["clip-high"] = "0.28" }, withValues: false), grpoConstant, "grpo: by a constant, the upper clip apart");

        // A KL without a reference, and a constant normalization without its constant, are refused naming the option.
        ThrowsNaming<ArgumentException>(() =>
        {
            using var scope = new TensorScope();
            PolicyLosses.Get("grpo").Loss(new PolicyLossInput(Tensor.From([-1f], [1], device), [1], Tensor.From([1f], [1], device)), new Dictionary<string, string> { ["kl"] = "0.1" });
        }, "a KL without a reference", "reference", "'kl'");
        ThrowsNaming<ArgumentException>(() =>
        {
            using var scope = new TensorScope();
            PolicyLosses.Get("ppo").Loss(new PolicyLossInput(Tensor.From([-1f], [1], device), [1], Tensor.From([1f], [1], device)), new Dictionary<string, string> { ["normalize"] = "constant" });
        }, "a constant normalization without the constant", "'constant'");
    }

    // ---------------------------------------------------------------- registries

    private static void ReinforcementDefaults(Device device)
    {
        foreach (string name in ReinforcementNames.Estimators)
        {
            Check(AdvantageEstimators.Origin(name) == Overrides.Library && AdvantageEstimators.Default(name) is not null && AdvantageEstimators.Get(name).Name == name,
                $"{name} is a library default of AdvantageEstimators: {AdvantageEstimators.Origin(name)}");
        }

        foreach (string name in ReinforcementNames.Algorithms)
        {
            Check(PolicyLosses.Origin(name) == Overrides.Library && PolicyLosses.Default(name) is not null && PolicyLosses.Get(name.ToUpperInvariant()).Name == name,
                $"{name} is a library default of PolicyLosses: {PolicyLosses.Origin(name)}");
        }

        Check(AdvantageEstimators.Names.Take(4).SequenceEqual(ReinforcementNames.Estimators) && PolicyLosses.Names.Take(3).SequenceEqual(ReinforcementNames.Algorithms),
            $"registered in order: {string.Join(", ", AdvantageEstimators.Names)}; {string.Join(", ", PolicyLosses.Names)}");
        Check(AdvantageEstimators.Describe().Contains("gae: generalized advantage estimation", StringComparison.Ordinal)
              && PolicyLosses.Describe().Contains("grpo: GRPO", StringComparison.Ordinal), "the summaries are listed");

        var kind = new RlBanditKind();
        using var policy = RlPolicy(kind, device, 1, 0.01f);
        ThrowsNaming<NotSupportedException>(() => ReinforcementTrainer.Train(kind, policy, new ReinforcementOptions { Algorithm = "nope", Iterations = 1 }),
            "an unknown algorithm", "Policy loss 'nope' is not registered", "reinforce", "ppo", "grpo");
        ThrowsNaming<NotSupportedException>(() => ReinforcementTrainer.Train(kind, policy, new ReinforcementOptions { Estimator = "nope", Iterations = 1 }),
            "an unknown estimator", "Advantage estimator 'nope' is not registered", "gae", "leave-one-out");
        ThrowsNaming<ArgumentException>(() => ReinforcementTrainer.Train(kind, policy, new ReinforcementOptions { GroupSize = 3, EpisodesPerIteration = 8, Iterations = 1 }),
            "a group that does not divide the episodes", "EpisodesPerIteration (8)", "group size (3)");
        ThrowsNaming<ArgumentException>(() => ReinforcementTrainer.Train(kind, policy, new ReinforcementOptions { Iterations = 1, EpisodesPerIteration = 2, LossOptions = new Dictionary<string, string> { ["nope"] = "1" } }),
            "a loss option the loss does not take", "'nope'", "clip");
        ThrowsNaming<NotSupportedException>(() => ReinforcementTrainer.Train("reinforcement-test-nobody", policy, new ReinforcementOptions { Iterations = 1 }),
            "an environment kind not registered", "Environment kind 'reinforcement-test-nobody' is not registered");
        ThrowsNaming<ArgumentException>(() => _ = new ReinforcementTrainer<float[], float[]>(kind, null!), "the policy", "policy");
        ThrowsNaming<ArgumentException>(() => _ = new ReinforcementTrainer<float[], float[]>(kind, new DummyBoxPolicy()), "types that are not the kind's", "acts with Int32", "acts with Single[]");
    }

    // A policy of the wrong action type, refused before it is used.
    private sealed class DummyBoxPolicy : IPolicy<float[], float[]>
    {
        public ValueTask<PolicyActions<float[]>> ActAsync(IReadOnlyList<float[]> observations, ActionSampling sampling, CancellationToken cancellationToken = default) =>
            throw new NotSupportedException();

        public UpdateStats Update(RolloutBatch<float[], float[]> batch, IPolicyLoss loss, IReadOnlyList<int>? episodes = null, IReadOnlyDictionary<string, string>? lossOptions = null) =>
            throw new NotSupportedException();

        public void Save(string path) => throw new NotSupportedException();

        public void Load(string path) => throw new NotSupportedException();

        public void Dispose()
        {
        }
    }

    // ---------------------------------------------------------------- learning

    private static ReinforcementOptions LearningOptions(string algorithm, int budget) => new()
    {
        Algorithm = algorithm,
        EpisodesPerIteration = 16,
        Iterations = budget,
        Seed = 11,
        MiniBatch = algorithm == ReinforcementNames.Ppo ? 8 : 0,
        EstimatorOptions = algorithm == ReinforcementNames.Reinforce ? new Dictionary<string, string> { ["baseline"] = "mean" } : null,
        Environments = 1,
        ParallelSteps = false,
    };

    private static void ReinforcementBandit(Device device)
    {
        var kind = new RlBanditKind();
        const int budget = 100;
        var reached = new List<string>();
        foreach (string algorithm in ReinforcementNames.Algorithms)
        {
            using var policy = RlPolicy(kind, device, 3, 0.05f, critic: algorithm == ReinforcementNames.Ppo);
            double BestArm()
            {
                var acted = policy.ActAsync([[0f]], new ActionSampling { Greedy = true }).AsTask().GetAwaiter().GetResult();
                return acted.Actions[0] == 1 ? Math.Exp(acted.Records[0].LogProbabilities[0]) : 0;
            }

            double before = BestArm() * 0.8 + (1 - BestArm()) * 0.2;
            var (result, at) = TrainUntil(kind, policy, LearningOptions(algorithm, budget), _ => BestArm() >= 0.95);
            double expected = BestArm() * 0.8 + (1 - BestArm()) * 0.2;
            Check(at > 0 && at <= budget, $"{algorithm}: the best arm's probability reached 0.95 within {budget} iterations (it is {BestArm():0.###} after {result.Iterations})");
            Check(expected >= 0.95 * 0.8, $"{algorithm}: the expected reward {expected:0.###} is within 5% of the best arm's 0.8");
            Check(expected > before + 0.2, $"{algorithm}: the expected reward rose from {before:0.###} (untrained) to {expected:0.###}");
            reached.Add($"{algorithm} {at}");
        }

        Console.WriteLine($"    bandit: the best arm at 0.95 after {string.Join(", ", reached)} iterations");
    }

    private static void ReinforcementCorridor(Device device)
    {
        var kind = new RlCorridorKind();
        const int budget = 150;
        var reached = new List<string>();
        foreach (string algorithm in ReinforcementNames.Algorithms)
        {
            using var policy = RlPolicy(kind, device, 5, 0.02f, critic: algorithm == ReinforcementNames.Ppo);
            var options = LearningOptions(algorithm, budget) with { EvaluateEvery = 1, EvaluationEpisodes = 2 };
            var (result, at) = TrainUntil(kind, policy, options, s => s.EvaluationReward is { } e && Math.Abs(e - CorridorOptimum) < 1e-4);
            Check(at > 0 && at <= budget, $"{algorithm}: the greedy walk reached the optimum return {CorridorOptimum} within {budget} iterations "
                                          + $"(evaluation {result.History[^1].EvaluationReward:0.####}, mean reward {result.History[^1].RewardMean:0.###} after {result.Iterations})");
            Check(result.History[^1].RewardMean > result.History[0].RewardMean, $"{algorithm}: the mean reward rose from {result.History[0].RewardMean:0.###} to {result.History[^1].RewardMean:0.###}");
            Check(result.Cancelled && result.Iterations == at, $"{algorithm}: the run stopped after the iteration that reached it ({result.Iterations}, {at})");
            reached.Add($"{algorithm} {at}");
        }

        Console.WriteLine($"    corridor: the optimum after {string.Join(", ", reached)} iterations");
    }

    private static void ReinforcementGaussian(Device device)
    {
        var kind = new RlTargetKind();
        var (observations, actions) = Spaces(kind);
        using var policy = ModulePolicy.Gaussian(observations, actions, ModulePolicy.Perceptron(1, 1, [], device, new Random(2)), options: new ModulePolicyOptions { LearningRate = 0.05f, InitialLogStd = -0.5f });
        float Mean() => policy.ActAsync([[0f]], new ActionSampling { Greedy = true }).AsTask().GetAwaiter().GetResult().Actions[0][0];

        // Acted and updated log-probabilities agree: the first update's ratios are 1.
        var collector = new RolloutCollector<float[], float[]>(kind, environments: 1);
        using (collector)
        using (var batch = collector.CollectAsync(policy, [new EpisodeStart(0, 1, 0), new EpisodeStart(1, 2, 1)], new ActionSampling { Seed = 3 }).AsTask().GetAwaiter().GetResult())
        {
            Check(batch.Trajectories.All(t => actions.Contains(t.Actions[0])), "the Gaussian's actions lie in the box");
            batch.Estimate(AdvantageEstimators.Get("returns"));
            var stats = policy.Update(batch, PolicyLosses.Get("ppo"));
            Check(stats.Metrics!["kl"] < 1e-6 && stats.Metrics["clip-fraction"] == 0, $"the first update's ratios are 1: kl {stats.Metrics["kl"]}, clipped {stats.Metrics["clip-fraction"]}");
        }

        var options = LearningOptions(ReinforcementNames.Reinforce, 150) with { Algorithm = ReinforcementNames.Reinforce };
        var (result, at) = TrainUntil(kind, policy, options, _ => Math.Abs(Mean() - 1f) < 0.1f);
        Check(at > 0, $"the mean reached 1 ± 0.1 within 150 iterations: {Mean():0.###} after {result.Iterations}");
        Console.WriteLine($"    box: the mean at {Mean():0.###} after {at} iterations");
    }

    // ---------------------------------------------------------------- reproducible, resumed, memory

    // A run's episode returns and advantages per iteration, and the weights it ends with.
    private static (List<float[]> Returns, List<float[]> Advantages, float[] Weights, ReinforcementResult Result) Recorded(IEnvironmentKind kind, Device device, ReinforcementOptions options,
        int policySeed = 5, int stopAfter = 0)
    {
        using var policy = RlPolicy(kind, device, policySeed, 0.02f, critic: true);
        using var stop = new CancellationTokenSource();
        var advantages = new List<float[]>();
        var trainer = new ReinforcementTrainer<float[], int>(kind, policy, options)
        {
            OnBatch = (_, batch) => advantages.Add(batch.Advantages.ToArray()),
            OnIteration = s =>
            {
                if (s.Iteration == stopAfter)
                {
                    stop.Cancel();
                }
            },
        };
        var result = trainer.Train(stop.Token);
        return ([.. result.History.Select(s => s.Returns)], advantages, Weights(policy), result);
    }

    private static string? SameRun((List<float[]> Returns, List<float[]> Advantages, float[] Weights, ReinforcementResult Result) expected,
        (List<float[]> Returns, List<float[]> Advantages, float[] Weights, ReinforcementResult Result) actual, int skip = 0)
    {
        if (actual.Returns.Count + skip != expected.Returns.Count)
        {
            return $"{actual.Returns.Count} iterations, {expected.Returns.Count - skip} expected";
        }

        for (int i = 0; i < actual.Returns.Count; i++)
        {
            if ((SameBits(expected.Returns[i + skip], actual.Returns[i], $"iteration {i + skip + 1}'s rewards")
                 ?? SameBits(expected.Advantages[i + skip], actual.Advantages[i], $"iteration {i + skip + 1}'s advantages")) is { } difference)
            {
                return difference;
            }
        }

        return SameBits(expected.Weights, actual.Weights, "the weights");
    }

    private static void ReinforcementReproducible(Device device)
    {
        var kind = new RlCorridorKind();
        var options = new ReinforcementOptions
        {
            Algorithm = ReinforcementNames.Ppo, EpisodesPerIteration = 8, Iterations = 4, MiniBatch = 4, Epochs = 2, Seed = 21, Environments = 1, ParallelSteps = false,
        };
        int threads = ComputeResources.MaxCpuThreads;
        try
        {
            ComputeResources.MaxCpuThreads = 1;
            var first = Recorded(kind, device, options);
            var again = Recorded(kind, device, options);
            ComputeResources.MaxCpuThreads = 4;
            var parallel = Recorded(kind, device, options with { Environments = 4, ParallelSteps = true });
            var measured = Recorded(kind, device, options with { Environments = 4, ParallelSteps = null });
            Check(SameRun(first, again) is null, $"the same seed twice: {SameRun(first, again)}");
            Check(SameRun(first, parallel) is null, $"on 4 threads, stepping 4 environments in parallel: {SameRun(first, parallel)}");
            Check(SameRun(first, measured) is null, $"on 4 threads, stepping as measured: {SameRun(first, measured)}");
            var other = Recorded(kind, device, options with { Seed = 22 });
            Check(SameRun(first, other) is not null, "another seed plays other episodes");
            var group = Recorded(kind, device, options with { Algorithm = ReinforcementNames.Grpo, Epochs = 0, MiniBatch = 0, GroupSize = 4 });
            var groupAgain = Recorded(kind, device, options with { Algorithm = ReinforcementNames.Grpo, Epochs = 0, MiniBatch = 0, GroupSize = 4, Environments = 4, ParallelSteps = true });
            Check(SameRun(group, groupAgain) is null, $"grpo in groups of 4, sequential and parallel: {SameRun(group, groupAgain)}");
        }
        finally
        {
            ComputeResources.MaxCpuThreads = threads;
        }
    }

    private static void ReinforcementResume(Device device)
    {
        var kind = new RlCorridorKind();
        string root = Path.Combine(Path.GetTempPath(), $"idrak-rl-resume-{Environment.ProcessId}");
        try
        {
            var options = new ReinforcementOptions
            {
                Algorithm = ReinforcementNames.Ppo, EpisodesPerIteration = 8, Iterations = 6, MiniBatch = 4, Epochs = 2, Seed = 31, Environments = 1, ParallelSteps = false,
                Schedule = LearningRateSchedules.Linear, SaveEvery = 3,
            };
            var whole = Recorded(kind, device, options with { CheckpointFolder = Path.Combine(root, "whole") });
            var stopped = Recorded(kind, device, options with { CheckpointFolder = Path.Combine(root, "stopped") }, stopAfter: 3);
            Check(stopped.Result.Cancelled && stopped.Result.Iterations == 3, $"the run stopped after 3 of 6 iterations: {stopped.Result.Iterations}");
            Check(Directory.Exists(Path.Combine(root, "whole", "iteration-000003")) && Directory.Exists(Path.Combine(root, "whole", "iteration-000006"))
                  && stopped.Result.Checkpoint == Path.Combine(root, "stopped", "iteration-000003"), $"checkpoints after 3 and 6 iterations: {stopped.Result.Checkpoint}");
            Check(File.Exists(Path.Combine(root, "stopped", "iteration-000003", "policy", "optimizer.safetensors"))
                  && File.Exists(Path.Combine(root, "stopped", "iteration-000003", "state.json")), "a checkpoint holds the policy, its optimizer and the run's state");
            var resumed = Recorded(kind, device, options with { Resume = Path.Combine(root, "stopped"), CheckpointFolder = Path.Combine(root, "resumed") }, policySeed: 99);
            Check(resumed.Result.Iterations == 6 && resumed.Result.History.Count == 3 && resumed.Result.History[0].Iteration == 4, $"the resumed run continues at iteration 4: {resumed.Result.History.Count} run");
            Check(SameRun(whole, resumed, skip: 3) is null, $"resumed after 3 of 6 iterations, as the run that did not stop: {SameRun(whole, resumed, skip: 3)}");
            Check(BitConverter.SingleToInt32Bits(resumed.Result.History[^1].LearningRate ?? 0f) == BitConverter.SingleToInt32Bits(whole.Result.History[^1].LearningRate ?? 1f),
                $"the schedule's rate resumes where it was: {resumed.Result.History[^1].LearningRate}, {whole.Result.History[^1].LearningRate}");
            ThrowsNaming<FileNotFoundException>(() => Recorded(kind, device, options with { Resume = Path.Combine(root, "nothing") }), "a folder with no checkpoint", "no reinforcement checkpoint");
        }
        finally
        {
            if (Directory.Exists(root))
            {
                Directory.Delete(root, recursive: true);
            }
        }
    }

    private static void ReinforcementMemory(Device device)
    {
        var kind = new RlCorridorKind();
        using var policy = RlPolicy(kind, device, 5, 0.02f, critic: true);
        long heap2 = 0, heap20 = 0, device2 = 0, device20 = 0;
        int made2 = 0, madeBefore = kind.Created;
        var options = new ReinforcementOptions { Algorithm = ReinforcementNames.Ppo, EpisodesPerIteration = 16, Iterations = 20, MiniBatch = 8, Seed = 41, Environments = 1, ParallelSteps = false };
        var trainer = new ReinforcementTrainer<float[], int>(kind, policy, options)
        {
            OnIteration = s =>
            {
                if (s.Iteration is 2 or 20)
                {
                    long heap = GC.GetTotalMemory(forceFullCollection: true);
                    long held = ComputeResources.GetMemoryUsage(device).InUse;
                    if (s.Iteration == 2)
                    {
                        (heap2, device2, made2) = (heap, held, kind.Created);
                    }
                    else
                    {
                        (heap20, device20) = (heap, held);
                    }
                }
            },
        };
        var result = trainer.Train();
        Check(result.Iterations == 20 && device20 == device2, $"device memory in use after iteration 2: {device2} bytes; after 20: {device20}");
        Check(heap20 - heap2 < 512 * 1024, $"the managed heap after iteration 2: {heap2 / 1024} KB; after 20: {heap20 / 1024} KB (grew {(heap20 - heap2) / 1024} KB)");
        Check(made2 - madeBefore == 16 && kind.Created == made2, $"one environment per episode slot, made once: {made2 - madeBefore}, then {kind.Created - madeBefore}");
        Console.WriteLine($"    memory: device {device2} → {device20} bytes, heap {heap2 / 1024} → {heap20 / 1024} KB over 18 iterations");
    }

    // ---------------------------------------------------------------- collector, stopping, reports

    private static void ReinforcementCollectorAndReports(Device device)
    {
        var kind = new RlCorridorKind();
        using var policy = RlPolicy(kind, device, 5, 0.02f, critic: true);
        EpisodeStart[] starts = [.. Enumerable.Range(0, 6).Select(k => new EpisodeStart(5 - k, 100 + k, k / 2))];
        float[] Rewards(RolloutBatch<float[], int> batch) => batch.Rewards.ToArray();
        float[] sequential, parallel;
        using (var one = new RolloutCollector<float[], int>(kind, environments: 1))
        using (var batch = one.CollectAsync(policy, starts, new ActionSampling { Seed = 9 }, maxSteps: 7).AsTask().GetAwaiter().GetResult())
        {
            Check(batch.Trajectories.Select(t => t.Index).SequenceEqual([0, 1, 2, 3, 4, 5]) && batch.Trajectories.Select(t => t.Seed).SequenceEqual([105, 104, 103, 102, 101, 100])
                  && batch.Trajectories.Select(t => t.Group).SequenceEqual([2, 2, 1, 1, 0, 0]), "episodes in number order, each with its seed and group");
            Check(batch.Trajectories.All(t => t.Steps <= 7 && (t.Terminated || t.Truncated)) && batch.HasValues
                  && batch.Trajectories.Where(t => t.Truncated).All(t => float.IsFinite(t.FinalValue)), "cut at 7 steps, truncated ones with the value of their last observation");
            sequential = Rewards(batch);
            Check(one.Created == 6 && one.SteppingInParallel == false, $"an environment per episode: {one.Created}");
        }

        using (var four = new RolloutCollector<float[], int>(kind, environments: 4, parallelSteps: true))
        using (var batch = four.CollectAsync(policy, starts, new ActionSampling { Seed = 9 }, maxSteps: 7).AsTask().GetAwaiter().GetResult())
        {
            parallel = Rewards(batch);
            using var fewer = four.CollectAsync(policy, starts[..3], new ActionSampling { Seed = 9 }).AsTask().GetAwaiter().GetResult();
            Check(four.Created == 6, $"a smaller rollout reuses the environments: {four.Created}");
        }

        Check(SameBits(sequential, parallel, "rewards") is null, $"stepped in parallel, the same episodes: {SameBits(sequential, parallel, "rewards")}");
        using (var measuring = new RolloutCollector<float[], int>(kind, environments: 2))
        using (var batch = measuring.CollectAsync(policy, [.. Enumerable.Range(0, 16).Select(k => new EpisodeStart(k, k, k))], new ActionSampling { Seed = 4 }).AsTask().GetAwaiter().GetResult())
        {
            Check(measuring.SteppingInParallel.HasValue, "the collector measured sequential against parallel steps and chose");
        }

        // A cancelled run stops between iterations with a checkpoint of the iterations it finished.
        string root = Path.Combine(Path.GetTempPath(), $"idrak-rl-stop-{Environment.ProcessId}");
        try
        {
            using var stop = new CancellationTokenSource();
            var options = new ReinforcementOptions { Algorithm = ReinforcementNames.Ppo, EpisodesPerIteration = 4, Iterations = 10, Seed = 3, CheckpointFolder = root, Environments = 1 };
            var stopped = new ReinforcementTrainer<float[], int>(kind, policy, options) { OnIteration = s => { if (s.Iteration == 2) stop.Cancel(); } }.Train(stop.Token);
            Check(stopped.Cancelled && stopped.Iterations == 2 && stopped.Checkpoint == Path.Combine(root, "iteration-000002")
                  && File.Exists(Path.Combine(root, "iteration-000002", "policy", "actor.ikw")), $"stopped after 2 iterations with a checkpoint: {stopped.Checkpoint}");
            var state = JsonNode.Parse(File.ReadAllText(Path.Combine(root, "iteration-000002", "state.json")))!;
            Check((int)state["iteration"]! == 2 && (int)state["seed"]! == 3 && (string?)state["algorithm"] == "ppo", $"the run's state: {state.ToJsonString()}");

            // Progress, telemetry and the JSON log.
            var hook = new RlHook();
            var writer = new StringWriter();
            string log = Path.Combine(root, "run.jsonl");
            var progress = ProgressReporters.Start("reinforce", options: new ProgressOptions { Reporter = ProgressReporters.JsonReporter, Output = writer, Interval = TimeSpan.Zero });
            ReinforcementResult reported;
            using (Telemetry.Subscribe(hook))
            {
                reported = ReinforcementTrainer.Train(kind, policy, new ReinforcementOptions
                {
                    Algorithm = ReinforcementNames.Ppo, EpisodesPerIteration = 4, Iterations = 2, Epochs = 3, MiniBatch = 2, Seed = 5, Log = log, Environments = 1, EvaluateEvery = 1,
                }, progress);
            }

            Check(progress.Done == 2 && progress.Total == 2 && writer.ToString().Contains("\"detail\"", StringComparison.Ordinal) && writer.ToString().Contains("reward", StringComparison.Ordinal),
                $"progress: {progress.Done} of {progress.Total}, {writer}");
            Check(hook.Started == 1 && hook.Completed == 1 && hook.Batches == 12 && hook.Epochs.Count == 2, $"telemetry: {hook.Started} start, {hook.Batches} updates, {hook.Epochs.Count} iterations, {hook.Completed} end");
            var metrics = hook.Epochs[^1].Metrics;
            foreach (string name in new[] { "episodes", "steps", "reward-mean", "reward-spread", "kl", "clip-fraction", "entropy", "gradient-norm", "rollout-seconds", "update-seconds", "terminated" })
            {
                Check(metrics.ContainsKey(name), $"the iteration's telemetry has {name}: {string.Join(", ", metrics.Keys)}");
            }

            Check(hook.Epochs[^1].ValidationMetrics?["reward-mean"] == reported.History[^1].EvaluationReward, "the evaluation reward is the epoch's validation metric");
            var lines = File.ReadAllLines(log);
            Check(lines.Length == 2 && lines.All(l => JsonNode.Parse(l)!["reward_mean"] is not null) && (int)JsonNode.Parse(lines[1])!["iteration"]! == 2 && JsonNode.Parse(lines[1])!["evaluation_reward"] is not null,
                $"a JSON line per iteration: {string.Join(" | ", lines)}");
            Check(reported.History.All(s => s.Updates == 6 && s.Metrics["clip-fraction"] >= 0), "three passes of two mini-batches: 6 updates an iteration");
        }
        finally
        {
            if (Directory.Exists(root))
            {
                Directory.Delete(root, recursive: true);
            }
        }
    }

    // ---------------------------------------------------------------- settings

    private static void ReinforcementOptionVariables(Device device)
    {
        _ = device;
        string[] names = ["IDRAK_RL_ALGORITHM", "IDRAK_RL_GROUP", "IDRAK_RL_PARALLEL_STEPS", "IDRAK_RL_CLIP", "IDRAK_RL_WARMUP", "IDRAK_RL_GROUP_SCALE", "IDRAK_RL_EPISODES",
            "IDRAK_RL_LEARNING_RATE", "IDRAK_RL_OPTIMIZER", "IDRAK_RL_RESUME"];
        try
        {
            var defaults = ReinforcementOptions.FromEnvironment();
            Check(defaults == new ReinforcementOptions(), "with no variable set, the defaults");
            Environment.SetEnvironmentVariable("IDRAK_RL_ALGORITHM", "grpo");
            Environment.SetEnvironmentVariable("IDRAK_RL_GROUP", "4");
            Environment.SetEnvironmentVariable("IDRAK_RL_PARALLEL_STEPS", "off");
            Environment.SetEnvironmentVariable("IDRAK_RL_CLIP", "0.3");
            Environment.SetEnvironmentVariable("IDRAK_RL_WARMUP", "0.1");
            Environment.SetEnvironmentVariable("IDRAK_RL_GROUP_SCALE", "false");
            Environment.SetEnvironmentVariable("IDRAK_RL_RESUME", "runs/rl");
            Environment.SetEnvironmentVariable("IDRAK_RL_LEARNING_RATE", "0.01");
            Environment.SetEnvironmentVariable("IDRAK_RL_OPTIMIZER", "sgd");
            var read = ReinforcementOptions.FromEnvironment(new ReinforcementOptions { Iterations = 7 });
            Check(read.Algorithm == "grpo" && read.GroupSize == 4 && read.ParallelSteps == false && read.Clip == 0.3f && read.ScheduleSettings.WarmupFraction == 0.1f
                  && read.GroupScale == false && read.Resume == "runs/rl" && read.Iterations == 7, $"the variables set their options, the others keep the start's: {read}");
            var policyOptions = ModulePolicyOptions.FromEnvironment();
            Check(policyOptions.LearningRate == 0.01f && policyOptions.Optimizer == "sgd", $"the module policy's variables: {policyOptions}");
            Environment.SetEnvironmentVariable("IDRAK_RL_EPISODES", "zero");
            ThrowsNaming<ArgumentException>(() => ReinforcementOptions.FromEnvironment(), "a bad value", "IDRAK_RL_EPISODES=zero");
            Environment.SetEnvironmentVariable("IDRAK_RL_EPISODES", null);
            Environment.SetEnvironmentVariable("IDRAK_RL_PARALLEL_STEPS", "sometimes");
            ThrowsNaming<ArgumentException>(() => ReinforcementOptions.FromEnvironment(), "a bad switch", "IDRAK_RL_PARALLEL_STEPS=sometimes", "on, off or auto");
            Environment.SetEnvironmentVariable("IDRAK_RL_PARALLEL_STEPS", "auto");
            Check(ReinforcementOptions.FromEnvironment(new ReinforcementOptions { ParallelSteps = true }).ParallelSteps is null, "auto measures");
        }
        finally
        {
            foreach (string name in names)
            {
                Environment.SetEnvironmentVariable(name, null);
            }
        }

        var table = Idrak.Cli.Shared.EnvironmentVariables.All.Where(v => v.Name.StartsWith("IDRAK_RL_", StringComparison.Ordinal)).Select(v => v.Name).ToHashSet(StringComparer.Ordinal);
        Check(names.All(table.Contains) && table.Count >= 33, $"the IDRAK_RL_* variables are in the tool's table: {table.Count}");
    }
}
