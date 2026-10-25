// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Reinforcement;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The checks of a reward function (<see cref="IRewardFunction"/>) on answers of its own (<c>samples</c>: a reward reads
/// what its task gives it, so the test supplies inputs it can score), in batches the suite draws from them:
/// <list type="bullet">
/// <item>a score per input, finite, and none for no input;</item>
/// <item>a batch scores what its inputs score one by one (within <see cref="Tolerance"/>), whatever else is in it;</item>
/// <item>the same scores every time, and from two calls at once;</item>
/// <item>an option not in its <see cref="IConfigurableKind.Keys"/> is refused, naming it; a cancelled call throws an
/// <see cref="OperationCanceledException"/>;</item>
/// <item>given a reference (the library's reward of that name), the same scores within <see cref="Tolerance"/>.</item>
/// </list>
/// </summary>
/// <param name="samples">The answers the reward is checked on; at least one.</param>
/// <param name="options">The reward's options for every call; null for its defaults.</param>
/// <param name="reference">The library's reward to agree with, or null.</param>
public sealed class RewardFunctionSuite(IReadOnlyList<RewardInput> samples, IReadOnlyDictionary<string, string>? options = null, IRewardFunction? reference = null)
    : ContractSuite<IRewardFunction>
{
    private const string UnknownOption = "conformance-unknown-option";

    private readonly IReadOnlyList<RewardInput> _samples = samples is { Count: > 0 } ? samples
        : throw new ArgumentException("A reward function is checked on answers of its own: give at least one sample.", nameof(samples));

    /// <inheritdoc />
    public override string Name => "reward-function";

    /// <summary>The agreement wanted between a batch and its inputs alone, and with the reference, relative to each score (at least 1).</summary>
    public float Tolerance { get; init; } = 1e-5f;

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        yield return Case("every sample", [.. Enumerable.Range(0, _samples.Count)]);
        yield return Case("the first sample", [0]);
        yield return Case("every sample, last first", [.. Enumerable.Range(0, _samples.Count).Reverse()]);
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        int count = random.Next(1, (large ? 8 : 2) * _samples.Count + 1);
        return Case("random", [.. Enumerable.Range(0, count).Select(_ => random.Next(_samples.Count))]);
    }

    /// <inheritdoc />
    public override void Run(IRewardFunction implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var reward = implementation;
        RewardInput[] inputs = [.. @case.Data["inputs"]!.AsArray().Select(n => _samples[(int)n!])];
        float[] Score(IReadOnlyList<RewardInput> batch) => Wait(reward.ScoreAsync(batch, options));

        checks.Check("a name and a summary", !string.IsNullOrWhiteSpace(reward.Name) && !string.IsNullOrWhiteSpace(reward.Summary), $"name \"{reward.Name}\", summary \"{reward.Summary}\"");
        var scores = Score(inputs);
        checks.Check("a score per input", scores.Length == inputs.Length, $"{scores.Length} scores for {inputs.Length} inputs");
        checks.Check("the scores are finite", scores.All(float.IsFinite), $"scores {string.Join(", ", scores.Take(8))}");
        checks.Check("no score for no input", Score([]).Length == 0, "scores for an empty batch");
        if (scores.Length != inputs.Length)
        {
            return;
        }

        float[] alone = [.. inputs.Select(i => Score([i]).Single())];
        checks.Compare("a batch scores what its inputs score one by one", Comparisons.Difference(alone, scores, Tolerance));
        checks.Compare("the same scores every time", Comparisons.Difference(scores, Score(inputs), 0f));
        var both = new float[2][];
        Parallel.For(0, 2, i => both[i] = Score(inputs));
        checks.Compare("the same scores from two calls at once", Comparisons.Difference(scores, both[0], 0f) ?? Comparisons.Difference(scores, both[1], 0f));

        var unknown = new Dictionary<string, string>(StringComparer.Ordinal) { [UnknownOption] = "1" };
        foreach (var (key, value) in options ?? new Dictionary<string, string>())
        {
            unknown[key] = value;
        }

        try
        {
            Wait(reward.ScoreAsync([inputs[0]], unknown));
            checks.Fail("an option not in Keys is refused, naming it", $"scored with the option '{UnknownOption}'");
        }
        catch (ArgumentException e)
        {
            checks.Check("an option not in Keys is refused, naming it", e.Message.Contains(UnknownOption, StringComparison.Ordinal), $"\"{e.Message}\" does not name '{UnknownOption}'");
        }

        using (var cancelled = new CancellationTokenSource())
        {
            cancelled.Cancel();
            try
            {
                Wait(reward.ScoreAsync(inputs, options, cancelled.Token));
                checks.Fail("a cancelled call throws OperationCanceledException", "the call scored with its token cancelled");
            }
            catch (OperationCanceledException)
            {
                checks.Pass("a cancelled call throws OperationCanceledException");
            }
        }

        if (reference is not null)
        {
            checks.Compare("agrees with the reference", Comparisons.Difference(Wait(reference.ScoreAsync(inputs, options)), scores, Tolerance));
        }
        else
        {
            checks.Skip("agrees with the reference", "no reference given");
        }
    }

    private static T Wait<T>(ValueTask<T> task) => task.IsCompletedSuccessfully ? task.Result : task.AsTask().GetAwaiter().GetResult();

    private static ContractCase Case(string name, int[] inputs) =>
        new($"{name} ({inputs.Length} inputs)", new JsonObject { ["inputs"] = new JsonArray(inputs.Select(i => (JsonNode?)i).ToArray()) });
}
