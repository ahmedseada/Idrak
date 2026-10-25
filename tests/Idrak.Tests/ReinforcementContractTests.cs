// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Abstraction.Reinforcement;
using Idrak.Abstraction.Testing;

// Plan 27, phase R1: the reinforcement contracts (Idrak.Abstraction.Reinforcement): spaces, trajectories and rollout
// batches; the four registries (environment kinds, rewards, advantage estimators, policy losses) with their "not
// registered" messages, origins and, for the guarded three, the slot policies; and the contract suites of
// Idrak.Abstraction.Testing, each passed by a small fake written here (not shipped) and failed by a fake broken on purpose.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] ReinforcementContractGroup =
    [
        ("reinforcement contracts: spaces hold what they say, action records, trajectories and inputs check their lengths, and a rollout batch lays episodes out in number order in pooled arrays it returns on Dispose", ReinforcementDataTypes),
        ("reinforcement contracts: the environment suite passes a seeded corridor (reset and step rules, replays, cancellation) and fails one whose draws are shared, so its replay differs", ReinforcementEnvironmentSuite),
        ("reinforcement contracts: the reward suite passes a reward scoring each answer alone and fails one scoring an answer against its batch", ReinforcementRewardSuite),
        ("reinforcement contracts: the advantage-estimator suite passes discounted returns with and without a baseline (bit for bit, from several threads, against a reference) and fails a noisy estimator", ReinforcementEstimatorSuite),
        ("reinforcement contracts: the policy-loss suite passes a clipped-ratio loss (one finite element, its gradient against central differences past the clip's kinks) and fails one whose gradient does not reach the log-probabilities", ReinforcementLossSuite),
        ("reinforcement contracts: registries: every not-registered message names the registered ones and the package to reference; register, find ignoring case, origin and unregister; the guarded three throw, fall back and shadow by policy over a library default, and environment kinds have no policy", ReinforcementRegistries),
        ("reinforcement contracts: Idrak.Reinforcement references Idrak alone (not Idrak.Text) and registers through LibraryDefaults; Idrak.Reinforcement.Plugins is a plug-in package on Idrak.Reinforcement, Idrak.Text and Idrak.Text.Plugins", ReinforcementPackages),
    ];

    private const string CorridorName = "reinforcement-test-corridor";

    // The broken corridor's slips: one source shared by every environment, so an episode's draws depend on what ran before.
    private static readonly Random SharedSlips = new(1);

    // A corridor of `length` cells walked left (0) or right (1) from a cell drawn from the episode's seed; a step slips the
    // other way one time in ten (drawn from the episode's seed too, or from SharedSlips for the broken kind). The episode
    // terminates at the right end (reward 1) or is truncated after `limit` steps; every other step costs 0.01.
    private sealed class CorridorKind(string name = CorridorName, bool sharedSlips = false) : IEnvironmentKind
    {
        public string Name => name;

        public string Summary => "a corridor walked to its right end (the reinforcement contract tests)";

        public IReadOnlySet<string> Keys { get; } = new OptionKeys("length", "limit");

        public Type ObservationType => typeof(float[]);

        public Type ActionType => typeof(int);

        public IEnvironment Create(EnvironmentRequest request)
        {
            var options = request.Options;
            int length = 6, limit = 12;
            foreach (var (key, value) in options ?? new Dictionary<string, string>())
            {
                switch (key)
                {
                    case "length": length = int.Parse(value, CultureInfo.InvariantCulture); break;
                    case "limit": limit = int.Parse(value, CultureInfo.InvariantCulture); break;
                    default: throw new ArgumentException($"The environment kind {Name} does not take the option '{key}'; it takes: {string.Join(", ", Keys)}.");
                }
            }

            return new Corridor(length, limit, sharedSlips);
        }
    }

    private sealed class Corridor(int length, int limit, bool sharedSlips) : IEnvironment<float[], int>
    {
        private Random? _random;
        private int _position, _steps;
        private bool _running;

        public Space ObservationSpace { get; } = Space.Box([1], 0f, 1f);

        public Space ActionSpace { get; } = Space.Discrete(2);

        public float[] Reset(int seed)
        {
            _random = new Random(seed);
            _position = _random.Next(length - 1);
            _steps = 0;
            _running = true;
            return [_position / (float)(length - 1)];
        }

        public ValueTask<EnvironmentStep<float[]>> StepAsync(int action, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (!_running)
            {
                throw new InvalidOperationException("No episode is under way: reset the corridor first.");
            }

            if (!ActionSpace.Contains(action))
            {
                throw new ArgumentException($"{action} is not in {ActionSpace}.", nameof(action));
            }

            bool slip;
            var random = sharedSlips ? SharedSlips : _random!;
            lock (random)
            {
                slip = random.NextDouble() < 0.1;
            }

            _position = Math.Clamp(_position + ((action == 1) != slip ? 1 : -1), 0, length - 1);
            _steps++;
            bool terminated = _position == length - 1, truncated = !terminated && _steps >= limit;
            _running = !(terminated || truncated);
            var info = new Dictionary<string, string>(StringComparer.Ordinal) { ["position"] = _position.ToString(CultureInfo.InvariantCulture) };
            return ValueTask.FromResult(new EnvironmentStep<float[]>([_position / (float)(length - 1)], terminated ? 1f : -0.01f, terminated, truncated, info));
        }

        public void Dispose() => _running = false;
    }

    // 1 for the reference answer, else 0, less 0.1 a character past the budget; the broken one subtracts its batch's mean.
    private sealed class BudgetReward(string name = "reinforcement-test-budget", bool batchRelative = false, float bonus = 0f, bool broken = false) : IRewardFunction
    {
        public string Name => name;

        public string Summary => "the reference answer within a length budget (the reinforcement contract tests)";

        public IReadOnlyCollection<string> Keys { get; } = ["budget"];

        public ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (broken)
            {
                throw new InvalidOperationException("the app's reward broke");
            }

            int budget = 10;
            foreach (var (key, value) in options ?? new Dictionary<string, string>())
            {
                budget = key == "budget" ? int.Parse(value, CultureInfo.InvariantCulture)
                    : throw new ArgumentException($"The reward {Name} does not take the option '{key}'; it takes: {string.Join(", ", Keys)}.");
            }

            float[] scores = [.. inputs.Select(i => (i.Answer == i.Reference ? 1f : 0f) - 0.1f * Math.Max(0, i.Answer.Length - budget) + bonus)];
            if (batchRelative && scores.Length > 0)
            {
                float mean = scores.Average();
                for (int i = 0; i < scores.Length; i++)
                {
                    scores[i] -= mean;
                }
            }

            return ValueTask.FromResult(scores);
        }
    }

    // Discounted returns per episode (bootstrapped from a truncated episode's final value), the advantage the return less
    // the value estimate when there is one; the noisy one adds a draw of its own, the broken one throws.
    private sealed class ReturnsEstimator(string name = "reinforcement-test-returns", bool usesValues = false, bool noisy = false, bool broken = false, float shift = 0f)
        : IAdvantageEstimator
    {
        public string Name => name;

        public string Summary => "discounted returns, less a baseline when values are given (the reinforcement contract tests)";

        public IReadOnlySet<string> Keys { get; } = new OptionKeys("gamma");

        public bool UsesValues => usesValues;

        public void Estimate(AdvantageInput input, Span<float> advantages, Span<float> returns, IReadOnlyDictionary<string, string>? options = null)
        {
            if (broken)
            {
                throw new InvalidOperationException("the app's estimator broke");
            }

            if (advantages.Length != input.Steps || returns.Length != input.Steps)
            {
                throw new ArgumentException($"Outputs of {input.Steps} steps: {advantages.Length} advantages and {returns.Length} returns given.");
            }

            if (usesValues && !input.HasValues)
            {
                throw new ArgumentException($"The estimator {Name} needs value estimates.");
            }

            float gamma = 1f;
            foreach (var (key, value) in options ?? new Dictionary<string, string>())
            {
                gamma = key == "gamma" ? float.Parse(value, CultureInfo.InvariantCulture)
                    : throw new ArgumentException($"The estimator {Name} does not take the option '{key}'; it takes: {string.Join(", ", Keys)}.");
            }

            int end = 0;
            for (int e = 0; e < input.Episodes; e++)
            {
                int start = end;
                end += input.Lengths[e];
                float g = input.IsTruncated(e) && input.HasValues ? input.FinalValues[e] : 0f;
                for (int t = end - 1; t >= start; t--)
                {
                    g = input.Rewards[t] + gamma * g;
                    returns[t] = g;
                    advantages[t] = g - (input.HasValues ? input.Values[t] : 0f) + shift + (noisy ? Random.Shared.NextSingle() * 1e-3f : 0f);
                }
            }
        }
    }

    // PPO's clipped ratio (min of ratio·A and clip(ratio)·A), a k3 KL to the reference and a squared value error, averaged
    // over the step's units; the detached one cuts the gradient, the broken one throws.
    private sealed class ClippedLoss(string name = "reinforcement-test-clipped", bool detached = false, bool broken = false, float shift = 0f) : IPolicyLoss
    {
        public string Name => name;

        public string Summary => "a clipped ratio with a KL penalty (the reinforcement contract tests)";

        public IReadOnlySet<string> Keys { get; } = new OptionKeys("clip", "kl");

        public Tensor Loss(PolicyLossInput input, IReadOnlyDictionary<string, string>? options = null)
        {
            if (broken)
            {
                throw new InvalidOperationException("the app's loss broke");
            }

            float clip = 0.2f, kl = 0.05f;
            foreach (var (key, value) in options ?? new Dictionary<string, string>())
            {
                switch (key)
                {
                    case "clip": clip = float.Parse(value, CultureInfo.InvariantCulture); break;
                    case "kl": kl = float.Parse(value, CultureInfo.InvariantCulture); break;
                    default: throw new ArgumentException($"The policy loss {Name} does not take the option '{key}'; it takes: {string.Join(", ", Keys)}.");
                }
            }

            var logp = detached ? input.LogProbabilities.Detach() : input.LogProbabilities;
            var ratio = (logp - input.BehaviorLogProbabilities).Exp();
            var a = ratio * input.Advantages;
            var b = ratio.Clamp(1f - clip, 1f + clip) * input.Advantages;
            var objective = (a + b - (a - b).Abs()) * 0.5f;                            // min(a, b)
            var loss = -objective.Sum() / input.StepUnits;
            if (input.ReferenceLogProbabilities is { } reference)
            {
                var d = reference - logp;
                loss = loss + (d.Exp() - d - 1f).Sum() * (kl / input.StepUnits);
            }

            if (input.Values is { } values)
            {
                loss = loss + (values - input.Returns!).Square().Sum() * (0.5f / input.StepUnits);
            }

            return shift == 0f ? loss : loss + shift;
        }
    }

    private static Trajectory<float[], int> Episode(int index, float[] rewards, bool terminated = true, int? group = null, float value = float.NaN, float finalValue = float.NaN) =>
        new(index, index * 7, [.. Enumerable.Range(0, rewards.Length + 1).Select(i => new[] { i / 10f })], [.. rewards.Select(_ => 1)],
            [.. rewards.Select(_ => new ActionRecord([1], [-0.5f], value))], rewards, terminated, !terminated, group, finalValue);

    private static void ReinforcementDataTypes(Device device)
    {
        _ = device;
        // Spaces.
        var discrete = Space.Discrete(3);
        var box = Space.Box([2], -1f, 1f);
        Check(discrete.Contains(2) && !discrete.Contains(3) && !discrete.Contains(-1) && discrete.ContainsValue(0) && !discrete.ContainsValue(0f), "a discrete space holds 0 to n - 1");
        Check(box.Contains([0.5f, -1f]) && !box.Contains([2f, 0f]) && !box.Contains([float.NaN, 0f]) && !box.Contains([0f]) && box.ContainsValue(new[] { 1f, 1f }) && box.Size == 2,
            "a box holds arrays of its size within its bounds, never NaN");
        Check(Space.Text.ContainsValue("an answer") && !Space.Text.ContainsValue(1) && Space.Conversation.ContainsValue(new object()) && !Space.Conversation.ContainsValue(null),
            "text holds strings, a conversation anything but null");
        Check(discrete.ToString() == "discrete(3)" && box.ToString() == "box([2], -1 to 1)" && Space.Box([1, 2], [0f, -1f], [1f, 1f]).ToString() == "box([1, 2], bounds per element)"
              && Space.Text.ToString() == "text", $"spaces print as {discrete}, {box}");
        Check(box.Equals(Space.Box([2], -1f, 1f)) && !box.Equals(Space.Box([2], -1f, 2f)) && discrete.Equals(Space.Discrete(3)), "spaces are equal by value");
        Check(Throws<ArgumentException>(() => Space.Box([2], 1f, -1f)) && Throws<ArgumentException>(() => Space.Box([0], 0f, 1f))
              && Throws<ArgumentException>(() => Space.Box([2], [0f], [1f])) && Throws<ArgumentOutOfRangeException>(() => Space.Discrete(0)),
            "a box with bounds the wrong way round, an empty dimension or bounds of another length, and a discrete space of nothing, are refused");

        // Records, sampling, policy actions.
        Check(Throws<ArgumentException>(() => _ = new ActionRecord([1, 2], [-1f])) && Throws<ArgumentException>(() => _ = new ActionRecord([], [])),
            "an action record has a log-probability per id (or one for an action not made of ids)");
        var record = new ActionRecord([], [-0.25f], 0.5f);
        Check(record.Units == 1 && record.HasValue && !new ActionRecord([4, 5], [-1f, -2f]).HasValue, "a box action is one unit; NaN is no value");
        Check(Throws<ArgumentOutOfRangeException>(() => _ = new ActionSampling { Temperature = 0f }) && ActionSampling.Default.Temperature == 1f && !ActionSampling.Default.Greedy,
            "the temperature is positive; the default samples at 1");
        Check(Throws<ArgumentException>(() => _ = new PolicyActions<int>([1, 2], [record])), "a record per action");

        // Trajectories.
        Check(Throws<ArgumentException>(() => _ = new Trajectory<float[], int>(0, 0, [[0f]], [1], [record], [1f], true, false)), "one observation more than the steps");
        Check(Throws<ArgumentException>(() => Episode(0, [float.NaN])) && Throws<ArgumentException>(() => Episode(0, [])), "rewards are finite, and an episode has a step");
        var three = Episode(4, [0.5f, -0.25f, 1f]);
        Check(three.Steps == 3 && three.Units == 3 && three.Return == 1.25f && three.Group == 4 && three.Seed == 28 && three.Terminated && !three.Truncated,
            $"a trajectory's steps, units, return and group: {three.Steps}, {three.Units}, {three.Return}, {three.Group}");

        // Rollout batches: episode-number order, flat arrays, values only when every action has one, pooled and returned.
        var batch = new RolloutBatch<float[], int>([Episode(2, [3f]), Episode(0, [1f, 1f], group: 9), Episode(1, [2f, 2f, 2f], terminated: false)]);
        Check(batch.Trajectories.Select(t => t.Index).SequenceEqual([0, 1, 2]) && batch.Episodes == 3 && batch.Steps == 6, "episodes in number order");
        Check(batch.Rewards.SequenceEqual([1f, 1f, 2f, 2f, 2f, 3f]) && batch.Lengths.SequenceEqual([2, 3, 1]) && batch.Offsets.SequenceEqual([0, 2, 5])
              && batch.Groups.SequenceEqual([9, 1, 2]) && batch.Truncated.SequenceEqual([false, true, false]),
            $"flat rewards {string.Join(", ", batch.Rewards.ToArray())}, lengths, offsets, groups and truncation");
        Check(!batch.HasValues && batch.Values.IsEmpty && batch.FinalValues.IsEmpty && batch.Advantages.ToArray().All(v => v == 0f), "no values without a critic; advantages start at 0");
        batch.Estimate(new ReturnsEstimator());
        Check(batch.Returns.SequenceEqual([2f, 1f, 6f, 4f, 2f, 3f]) && batch.Advantages.SequenceEqual(batch.Returns), $"Estimate writes the batch: {string.Join(", ", batch.Returns.ToArray())}");
        batch.Dispose();
        batch.Dispose();
        Check(Throws<ObjectDisposedException>(() => _ = batch.Rewards.Length) && Throws<ObjectDisposedException>(() => _ = batch.Advantages.Length), "a disposed batch cannot be read");
        Check(Throws<ArgumentException>(() => _ = new RolloutBatch<float[], int>([Episode(0, [1f]), Episode(0, [1f])])) && Throws<ArgumentException>(() => _ = new RolloutBatch<float[], int>([])),
            "two episodes of one number, or none, are refused");
        Check(Throws<ArgumentException>(() => _ = new RolloutBatch<float[], int>([Episode(0, [1f], terminated: false, value: 0.5f)])), "a truncated episode with values needs its final value");
        using (var valued = new RolloutBatch<float[], int>([Episode(0, [1f, 1f], terminated: false, value: 0.5f, finalValue: 2f), Episode(1, [1f], value: 0.25f, finalValue: 7f)]))
        {
            Check(valued.HasValues && valued.Values.SequenceEqual([0.5f, 0.5f, 0.25f]) && valued.FinalValues.SequenceEqual([2f, 0f]), "values, and a final value for the truncated episode only");
            valued.Estimate(new ReturnsEstimator(usesValues: true));
            Check(valued.Returns.SequenceEqual([4f, 3f, 1f]) && valued.Advantages.SequenceEqual([3.5f, 2.5f, 0.75f]), $"bootstrapped returns: {string.Join(", ", valued.Returns.ToArray())}");
        }

        // Advantage and policy-loss inputs check their lengths.
        Check(Throws<ArgumentException>(() => _ = new AdvantageInput([1f, 2f], [], [3], [], [], [])) && Throws<ArgumentException>(() => _ = new AdvantageInput([1f], [], [1], [], [0f], []))
              && Throws<ArgumentException>(() => _ = new AdvantageInput([1f], [1f], [1], [], [], [])) && Throws<ArgumentException>(() => _ = new AdvantageInput([1f], [], [1], [], [], [0, 1]))
              && Throws<ArgumentException>(() => _ = new AdvantageInput([], [], [], [], [], [])),
            "an advantage input whose lengths do not agree is refused");
        var input = new AdvantageInput([1f, 2f, 3f], [], [1, 2], [false, true], [], []);
        Check(input.Steps == 3 && input.Episodes == 2 && input.Group(1) == 1 && input.IsTruncated(1) && !input.HasValues, "an advantage input's sizes, groups and ends");
        using (var scope = new TensorScope())
        {
            var logp = Tensor.From([-1f, -2f, -3f], [3], requiresGrad: true);
            var advantages = Tensor.From([1f, 1f, 1f], [3]);
            Check(Throws<ArgumentException>(() => _ = new PolicyLossInput(logp, [2], advantages)) && Throws<ArgumentException>(() => _ = new PolicyLossInput(logp, [3], advantages, values: logp))
                  && Throws<ArgumentException>(() => _ = new PolicyLossInput(logp, [3], Tensor.From([1f], [1]))), "a policy-loss input whose sizes do not agree is refused");
            var lossInput = new PolicyLossInput(logp, [1, 2], advantages, stepUnits: 12);
            Check(!lossInput.HasBehavior && !lossInput.BehaviorLogProbabilities.RequiresGrad && lossInput.BehaviorLogProbabilities.ToArray().SequenceEqual([-1f, -2f, -3f])
                  && lossInput.StepUnits == 12 && lossInput.StepSequences == 2 && lossInput.Units == 3, "on-policy, the behaviour log-probabilities are the current ones detached");
        }
    }

    private static void ReinforcementEnvironmentSuite(Device device)
    {
        _ = device;
        var report = Conformance.CheckEnvironmentKind<float[], int>(new CorridorKind(), (random, _) => random.Next(2));
        report.ThrowIfFailed();
        foreach (string check in new[] { "a step before the first reset is refused", "a finished episode must be reset: stepping it again is refused",
                     "the same environment reset with the same seed replays the episode", "a new environment of the kind replays the episode from its seed",
                     "environments run at once give the steps of one alone", "a cancelled step throws OperationCanceledException", "an option not in Keys is refused, naming it",
                     "observations are in the observation space" })
        {
            Check(report.Entries.Any(e => e.Name == check && e.Status == CheckStatus.Passed), $"{check}:\n{report}");
        }

        Conformance.CheckEnvironmentKind<float[], int>(new CorridorKind(), (random, _) => random.Next(2), new Dictionary<string, string> { ["length"] = "3", ["limit"] = "4" })
            .ThrowIfFailed();
        var shared = Conformance.CheckEnvironmentKind<float[], int>(new CorridorKind("reinforcement-test-shared", sharedSlips: true), (random, _) => random.Next(2));
        Check(!shared.Passed && shared.Failures.Any(f => f.Check == "the same environment reset with the same seed replays the episode"), $"shared draws fail the replay:\n{shared}");
        var wrongTypes = Conformance.CheckEnvironmentKind<float[], float>(new CorridorKind(), (random, _) => 0f, options: new ContractCheckOptions { RandomCases = 0 });
        Check(wrongTypes.Failures.Any(f => f.Check == "the observation and action types asked for") && wrongTypes.Failures.Any(f => f.Check == "creates an environment of the types asked for"),
            $"the wrong action type is reported:\n{wrongTypes}");
    }

    private static readonly RewardInput[] RewardSamples =
    [
        new("2 + 2?", "4", "4"),
        new("The capital of France?", "Paris, which is the capital", "Paris"),
        new("Say nothing.", "", null),
        new("Fix the test.", "done", "done", new Dictionary<string, string> { ["tests"] = "3/3" }),
    ];

    private static void ReinforcementRewardSuite(Device device)
    {
        _ = device;
        var report = Conformance.CheckRewardFunction(new BudgetReward(), RewardSamples);
        report.ThrowIfFailed();
        foreach (string check in new[] { "a batch scores what its inputs score one by one", "the same scores from two calls at once", "an option not in Keys is refused, naming it",
                     "a cancelled call throws OperationCanceledException", "no score for no input" })
        {
            Check(report.Entries.Any(e => e.Name == check && e.Status == CheckStatus.Passed), $"{check}:\n{report}");
        }

        Conformance.CheckRewardFunction(new BudgetReward(), RewardSamples, new Dictionary<string, string> { ["budget"] = "3" }, reference: new BudgetReward("another")).ThrowIfFailed();
        var relative = Conformance.CheckRewardFunction(new BudgetReward("reinforcement-test-relative", batchRelative: true), RewardSamples);
        Check(!relative.Passed && relative.Failures.Any(f => f.Check == "a batch scores what its inputs score one by one"), $"a batch-relative reward fails:\n{relative}");
        var differs = Conformance.CheckRewardFunction(new BudgetReward(), RewardSamples, reference: new BudgetReward(bonus: 0.5f));
        Check(differs.Failures.Any(f => f.Check == "agrees with the reference"), $"a reward that differs from its reference fails:\n{differs}");
    }

    private static void ReinforcementEstimatorSuite(Device device)
    {
        _ = device;
        var report = Conformance.CheckAdvantageEstimator(new ReturnsEstimator());
        report.ThrowIfFailed();
        foreach (string check in new[] { "every step gets a finite advantage and return", "the same input gives the same output, bit for bit", "the same output from several threads at once",
                     "outputs of the wrong length are refused", "an option not in Keys is refused, naming it" })
        {
            Check(report.Entries.Any(e => e.Name == check && e.Status == CheckStatus.Passed), $"{check}:\n{report}");
        }

        var baseline = Conformance.CheckAdvantageEstimator(new ReturnsEstimator("reinforcement-test-baseline", usesValues: true), new Dictionary<string, string> { ["gamma"] = "0.9" },
            reference: new ReturnsEstimator("another", usesValues: true));
        baseline.ThrowIfFailed();
        Check(baseline.Entries.Any(e => e.Name == "an input without values is refused (the estimator uses them)" && e.Status == CheckStatus.Passed)
              && baseline.Entries.Any(e => e.Name == "agrees with the reference" && e.Status == CheckStatus.Passed), $"with values and a reference:\n{baseline}");
        var noisy = Conformance.CheckAdvantageEstimator(new ReturnsEstimator("reinforcement-test-noisy", noisy: true));
        Check(!noisy.Passed && noisy.Failures.Any(f => f.Check == "the same input gives the same output, bit for bit"), $"a noisy estimator fails:\n{noisy}");
    }

    private static void ReinforcementLossSuite(Device device)
    {
        _ = device;
        var report = Conformance.CheckPolicyLoss(new ClippedLoss(), reference: new ClippedLoss("another"));
        report.ThrowIfFailed();
        foreach (string check in new[] { "a one-element loss", "the same loss every time", "the gradient reaches the log-probabilities",
                     "the gradient agrees with central differences where the loss is smooth", "an option not in Keys is refused, naming it", "agrees with the reference" })
        {
            Check(report.Entries.Any(e => e.Name == check && e.Status == CheckStatus.Passed), $"{check}:\n{report}");
        }

        Conformance.CheckPolicyLoss(new ClippedLoss(), new Dictionary<string, string> { ["clip"] = "0.1", ["kl"] = "0.5" }, needsReference: true, needsValues: true).ThrowIfFailed();
        var detached = Conformance.CheckPolicyLoss(new ClippedLoss("reinforcement-test-detached", detached: true));
        Check(!detached.Passed && detached.Failures.Any(f => f.Check == "the gradient reaches the log-probabilities"), $"a loss cut from the log-probabilities fails:\n{detached}");
        var differs = Conformance.CheckPolicyLoss(new ClippedLoss(), reference: new ClippedLoss(shift: 0.5f));
        Check(differs.Failures.Any(f => f.Check == "agrees with the reference"), $"a loss that differs from its reference fails:\n{differs}");
    }

    private static void ReinforcementRegistries(Device device)
    {
        _ = device;
        const string None = "ReinforcementTestNobody";
        ThrowsNaming<NotSupportedException>(() => EnvironmentKinds.Get(None), "EnvironmentKinds",
            $"Environment kind '{None}' is not registered", "EnvironmentKinds.Register", "Idrak.Reinforcement.Plugins", "ReinforcementPlugins.RegisterAll()");
        ThrowsNaming<NotSupportedException>(() => RewardFunctions.Get(None), "RewardFunctions",
            $"Reward function '{None}' is not registered", "RewardFunctions.Register", "Idrak.Text", "Idrak.Reinforcement.Plugins", "ReinforcementPlugins.RegisterAll()");
        ThrowsNaming<NotSupportedException>(() => AdvantageEstimators.Get(None), "AdvantageEstimators",
            $"Advantage estimator '{None}' is not registered", "AdvantageEstimators.Register", "Idrak.Reinforcement");
        ThrowsNaming<NotSupportedException>(() => PolicyLosses.Get(None), "PolicyLosses", $"Policy loss '{None}' is not registered", "PolicyLosses.Register", "Idrak.Reinforcement");

        // Environment kinds: an app's, by name ignoring case, typed creation, no policy.
        var corridor = new CorridorKind();
        try
        {
            EnvironmentKinds.Register(corridor);
            Check(ReferenceEquals(EnvironmentKinds.Get(CorridorName.ToUpperInvariant()), corridor) && EnvironmentKinds.Names.Contains(CorridorName) && EnvironmentKinds.Default(CorridorName) is null
                  && EnvironmentKinds.Origin(CorridorName) == "Idrak.Tests" && EnvironmentKinds.Describe().Contains($"{CorridorName}: a corridor", StringComparison.Ordinal),
                $"an app's kind is registered, found ignoring case, from the app: {EnvironmentKinds.Origin(CorridorName)}");
            ThrowsNaming<NotSupportedException>(() => EnvironmentKinds.Get(None), "the message lists the registered kinds", CorridorName);
            using (var environment = EnvironmentKinds.Create<float[], int>(CorridorName, seed: 3))
            {
                Check(environment.ActionSpace.Equals(Space.Discrete(2)) && environment.Reset(1).Length == 1, "a typed environment from the registry");
            }

            ThrowsNaming<ArgumentException>(() => EnvironmentKinds.Create<string, int>(CorridorName), "the wrong types", "observes Single[]", "not String and Int32");
            Check(Overrides.Report().Any(o => o.Registry == nameof(EnvironmentKinds) && o.Name == CorridorName && !o.Guarded), "environment kinds are unguarded");
            Check(typeof(EnvironmentKinds).GetMethod("SetPolicy") is null, "environment kinds have no policy to set");
        }
        finally
        {
            Check(EnvironmentKinds.Unregister(CorridorName) && !EnvironmentKinds.Unregister(CorridorName) && EnvironmentKinds.Find(CorridorName) is null, "Unregister takes the kind away, once");
        }

        // The guarded three: over a library default registered here (once a process: a default is never removed), an app's
        // throws (Throw), falls back to the library's (FallBack), or lets the library's answer and is compared (Shadow).
        const string Guarded = "reinforcement-test-guarded";
        var inputs = RewardSamples;
        GuardedRegistry(nameof(RewardFunctions), Guarded, RewardFunctions.Register, RewardFunctions.Unregister, RewardFunctions.Get, RewardFunctions.Default, RewardFunctions.Origin,
            RewardFunctions.SetPolicy, new BudgetReward(Guarded), new BudgetReward(Guarded, broken: true), new BudgetReward(Guarded, bonus: 1f),
            reward => reward.ScoreAsync(inputs).AsTask().GetAwaiter().GetResult());
        float[] Estimated(IAdvantageEstimator estimator)
        {
            var advantages = new float[3];
            estimator.Estimate(new AdvantageInput([1f, 2f, 3f], [], [2, 1], [], [], []), advantages, new float[3]);
            return advantages;
        }

        GuardedRegistry(nameof(AdvantageEstimators), Guarded, AdvantageEstimators.Register, AdvantageEstimators.Unregister, n => AdvantageEstimators.Get(n), AdvantageEstimators.Default,
            AdvantageEstimators.Origin, AdvantageEstimators.SetPolicy, new ReturnsEstimator(Guarded), new ReturnsEstimator(Guarded, broken: true), new ReturnsEstimator(Guarded, shift: 1f),
            Estimated);
        float[] Lost(IPolicyLoss loss)
        {
            using var scope = new TensorScope();
            return loss.Loss(new PolicyLossInput(Tensor.From([-1f, -2f], [2]), [2], Tensor.From([1f, -1f], [2]))).ToArray();
        }

        GuardedRegistry(nameof(PolicyLosses), Guarded, PolicyLosses.Register, PolicyLosses.Unregister, n => PolicyLosses.Get(n), PolicyLosses.Default, PolicyLosses.Origin,
            PolicyLosses.SetPolicy, new ClippedLoss(Guarded), new ClippedLoss(Guarded, broken: true), new ClippedLoss(Guarded, shift: 1f), Lost);
    }

    private static void GuardedRegistry<T>(string registry, string name, Action<T> register, Func<string, bool> unregister, Func<string, T> get, Func<string, T?> library,
        Func<string, string?> origin, Action<string, SlotPolicy, double> setPolicy, T builtIn, T broken, T differs, Func<T, float[]> use)
        where T : class
    {
        if (library(name) is null)
        {
            Overrides.AsLibraryDefaults(() => register(builtIn));
        }

        var expected = use(library(name)!);
        Check(origin(name) == Overrides.Library && ReferenceEquals(get(name), library(name)), $"{registry}: a library default is handed out as it is");
        try
        {
            register(broken);
            Check(origin(name) == "Idrak.Tests" && !ReferenceEquals(get(name), broken), $"{registry}: an app's entry over a default is from the app, and guarded: {origin(name)}");
            Check(Throws<InvalidOperationException>(() => use(get(name))), $"{registry}: under Throw the app's failure reaches the caller");
            setPolicy(name, SlotPolicy.FallBack, Slot.DefaultShadowRate);
            Check(use(get(name)).SequenceEqual(expected), $"{registry}: under FallBack the library's answers");
            register(differs);
            setPolicy(name, SlotPolicy.Shadow, 1.0);
            Check(use(get(name)).SequenceEqual(expected), $"{registry}: under Shadow the library's answers");
            var slot = Overrides.Report().Single(o => o.Registry == registry && o.Name == name);
            Check(slot.Guarded && slot.ReplacesDefault && slot.Compared >= 1 && slot.Differed >= 1, $"{registry}: the shadow compared the app's answer and found it differs: {slot}");
        }
        finally
        {
            setPolicy(name, SlotPolicy.Throw, Slot.DefaultShadowRate);
            Check(unregister(name) && !unregister(name) && origin(name) == Overrides.Library && use(get(name)).SequenceEqual(expected), $"{registry}: Unregister brings the library's back");
        }
    }

    private static void ReinforcementPackages(Device device)
    {
        _ = device;
        string root = RepositoryRoot();
        string library = File.ReadAllText(Path.Combine(root, "src", "Idrak.Reinforcement", "Idrak.Reinforcement.csproj"));
        var references = System.Text.RegularExpressions.Regex.Matches(library, "<ProjectReference Include=\"([^\"]+)\"").Select(m => Path.GetFileNameWithoutExtension(m.Groups[1].Value.Replace('\\', '/'))).ToList();
        Check(references.SequenceEqual(["Idrak"]), $"Idrak.Reinforcement references {string.Join(", ", references)}: Idrak alone, never Idrak.Text");
        string plugins = File.ReadAllText(Path.Combine(root, "src", "Idrak.Reinforcement.Plugins", "Idrak.Reinforcement.Plugins.csproj"));
        var pluginReferences = System.Text.RegularExpressions.Regex.Matches(plugins, "<ProjectReference Include=\"([^\"]+)\"").Select(m => Path.GetFileNameWithoutExtension(m.Groups[1].Value.Replace('\\', '/'))).ToList();
        Check(pluginReferences.Order(StringComparer.Ordinal).SequenceEqual(["Idrak.Reinforcement", "Idrak.Text", "Idrak.Text.Plugins"]), $"Idrak.Reinforcement.Plugins references {string.Join(", ", pluginReferences)}");
        var defaults = (string[])typeof(Idrak.Abstraction.Plugins.IdrakPlugins).Assembly.GetType("Idrak.Abstraction.LibraryDefaults", throwOnError: true)!
            .GetField("Assemblies", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!.GetValue(null)!;
        Check(defaults.Contains("Idrak.Reinforcement"), "LibraryDefaults lists Idrak.Reinforcement, so its estimators and losses become the slots' defaults");
        var registrations = System.Reflection.CustomAttributeExtensions.GetCustomAttribute<LibraryRegistrationsAttribute>(System.Reflection.Assembly.Load("Idrak.Reinforcement"))?.Type;
        Check(registrations is not null && typeof(ILibraryRegistrations).IsAssignableFrom(registrations),
            "Idrak.Reinforcement names its ILibraryRegistrations with the LibraryRegistrations attribute, which LibraryDefaults reads");
        Check(Idrak.Reinforcement.Plugins.ReinforcementPlugins.Package.Name == "Idrak.Reinforcement.Plugins"
              && Idrak.Reinforcement.Plugins.ReinforcementPlugins.All.Select(p => p.Name).SequenceEqual(["coding", "csharp", "classic"]),
            "the plug-in package's families: coding (R4), csharp (R4b), classic (R6)");
    }
}
