// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Abstraction.Reinforcement;
using Idrak.Abstraction.Testing;

// Plan 27, phase R1: the reinforcement contracts (Idrak.Abstraction.Reinforcement): spaces, trajectories and rollout
// batches; and the packages' references.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] ReinforcementContractGroup =
    [
        ("reinforcement contracts: spaces hold what they say, action records, trajectories and inputs check their lengths, and a rollout batch lays episodes out in number order in pooled arrays it returns on Dispose", ReinforcementDataTypes),
        ("reinforcement contracts: Idrak.Reinforcement references Idrak alone (not Idrak.Text); Idrak.Reinforcement.Plugins is a package on Idrak.Reinforcement, Idrak.Text and Idrak.Text.Plugins", ReinforcementPackages),
    ];

    // Discounted returns per episode (bootstrapped from a truncated episode's final value), the advantage the return less
    // the value estimate when there is one; the noisy one adds a draw of its own, the broken one throws.
    internal sealed class ReturnsEstimator(string name = "reinforcement-test-returns", bool usesValues = false, bool noisy = false, bool broken = false, float shift = 0f)
        : IAdvantageEstimator
    {
        public string Name => name;

        public string Summary => "discounted returns, less a baseline when values are given (the reinforcement contract tests)";

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
                    : throw new ArgumentException($"The estimator {Name} does not take the option '{key}'; it takes: gamma.");
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

    internal static Trajectory<float[], int> Episode(int index, float[] rewards, bool terminated = true, int? group = null, float value = float.NaN, float finalValue = float.NaN) =>
        new(index, index * 7, [.. Enumerable.Range(0, rewards.Length + 1).Select(i => new[] { i / 10f })], [.. rewards.Select(_ => 1)],
            [.. rewards.Select(_ => new ActionRecord([1], [-0.5f], value))], rewards, terminated, !terminated, group, finalValue);

    internal static void ReinforcementDataTypes(Device device)
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

    internal static void ReinforcementPackages(Device device)
    {
        _ = device;
        string root = RepositoryRoot();
        string library = File.ReadAllText(Path.Combine(root, "src", "Idrak.Reinforcement", "Idrak.Reinforcement.csproj"));
        var references = System.Text.RegularExpressions.Regex.Matches(library, "<ProjectReference Include=\"([^\"]+)\"").Select(m => Path.GetFileNameWithoutExtension(m.Groups[1].Value.Replace('\\', '/'))).ToList();
        Check(references.SequenceEqual(["Idrak"]), $"Idrak.Reinforcement references {string.Join(", ", references)}: Idrak alone, never Idrak.Text");
        string plugins = File.ReadAllText(Path.Combine(root, "src", "Idrak.Reinforcement.Plugins", "Idrak.Reinforcement.Plugins.csproj"));
        var pluginReferences = System.Text.RegularExpressions.Regex.Matches(plugins, "<ProjectReference Include=\"([^\"]+)\"").Select(m => Path.GetFileNameWithoutExtension(m.Groups[1].Value.Replace('\\', '/'))).ToList();
        Check(pluginReferences.Order(StringComparer.Ordinal).SequenceEqual(["Idrak.Reinforcement", "Idrak.Text", "Idrak.Text.Plugins"]), $"Idrak.Reinforcement.Plugins references {string.Join(", ", pluginReferences)}");
    }
}
