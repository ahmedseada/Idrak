// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Reinforcement;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The checks of an advantage estimator (<see cref="IAdvantageEstimator"/>) on batches the suite draws: episodes of
/// random lengths in groups, rewards of either sign (some exactly 0 and 1, as pass or fail rewards are), value estimates
/// and truncated episodes or not (always values when the estimator uses them):
/// <list type="bullet">
/// <item>every step gets a finite advantage and return;</item>
/// <item>the same input gives the same output bit for bit (a fixed summation order), also from several threads at
/// once;</item>
/// <item>outputs of the wrong length are refused, and so is an input without values when the estimator uses them
/// (<see cref="ArgumentException"/>);</item>
/// <item>an option not in its <see cref="IAdvantageEstimator.Keys"/> is refused, naming it;</item>
/// <item>given a reference (the library's estimator of that name), the same advantages and returns within
/// <see cref="Tolerance"/>.</item>
/// </list>
/// </summary>
/// <param name="options">The estimator's options for every call; null for its defaults.</param>
/// <param name="reference">The library's estimator to agree with, or null.</param>
public sealed class AdvantageEstimatorSuite(IReadOnlyDictionary<string, string>? options = null, IAdvantageEstimator? reference = null) : ContractSuite<IAdvantageEstimator>
{
    private const string UnknownOption = "conformance-unknown-option";

    /// <inheritdoc />
    public override string Name => "advantage-estimator";

    /// <summary>The agreement wanted with the reference, relative to each value (at least 1).</summary>
    public float Tolerance { get; init; } = 1e-5f;

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        yield return Case("one step", 1, 1, 1, 1, values: false, truncated: false);
        yield return Case("one episode", 2, 1, 6, 1, values: true, truncated: false);
        yield return Case("a group of four", 3, 4, 3, 1, values: false, truncated: false);
        yield return Case("two groups, values and truncated episodes", 4, 6, 5, 2, values: true, truncated: true);
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        int episodes = random.Next(1, large ? 65 : 9);
        return Case("random", random.Next(), episodes, random.Next(1, large ? 200 : 12), random.Next(1, episodes + 1), random.Next(2) == 0, random.Next(2) == 0);
    }

    /// <inheritdoc />
    public override void Run(IAdvantageEstimator implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var estimator = implementation;
        var batch = Batch(@case, estimator.UsesValues);
        int steps = batch.Rewards.Length;
        checks.Check("a name and a summary", !string.IsNullOrWhiteSpace(estimator.Name) && !string.IsNullOrWhiteSpace(estimator.Summary),
            $"name \"{estimator.Name}\", summary \"{estimator.Summary}\"");

        var (advantages, returns) = Estimate(estimator, batch, options);
        int unwritten = advantages.Concat(returns).Count(v => !float.IsFinite(v));
        checks.Check("every step gets a finite advantage and return", unwritten == 0, $"{unwritten} of {2 * steps} outputs are not finite (or not written)");
        var (again, againReturns) = Estimate(estimator, batch, options);
        checks.Compare("the same input gives the same output, bit for bit", Bits(advantages, again) ?? Bits(returns, againReturns));
        var parallel = new (float[] Advantages, float[] Returns)[4];
        Parallel.For(0, parallel.Length, i => parallel[i] = Estimate(estimator, batch, options));
        checks.Compare("the same output from several threads at once", parallel.Select(p => Bits(advantages, p.Advantages) ?? Bits(returns, p.Returns)).FirstOrDefault(d => d is not null));

        checks.Check("outputs of the wrong length are refused", Refused(() =>
            estimator.Estimate(batch.Input(), new float[steps + 1], new float[steps], options)) && Refused(() =>
            estimator.Estimate(batch.Input(), new float[steps], new float[Math.Max(0, steps - 1)], options)), "outputs one too long or one too short were taken");
        if (estimator.UsesValues)
        {
            checks.Check("an input without values is refused (the estimator uses them)",
                Refused(() => estimator.Estimate((batch with { Values = [], FinalValues = [] }).Input(), new float[steps], new float[steps], options)), "estimated without values");
        }
        else
        {
            checks.Skip("an input without values is refused (the estimator uses them)", "the estimator does not use values");
        }

        var unknown = new Dictionary<string, string>(StringComparer.Ordinal) { [UnknownOption] = "1" };
        foreach (var (key, value) in options ?? new Dictionary<string, string>())
        {
            unknown[key] = value;
        }

        try
        {
            estimator.Estimate(batch.Input(), new float[steps], new float[steps], unknown);
            checks.Fail("an option not in Keys is refused, naming it", $"estimated with the option '{UnknownOption}'");
        }
        catch (ArgumentException e)
        {
            checks.Check("an option not in Keys is refused, naming it", e.Message.Contains(UnknownOption, StringComparison.Ordinal), $"\"{e.Message}\" does not name '{UnknownOption}'");
        }

        if (reference is not null)
        {
            var (expected, expectedReturns) = Estimate(reference, batch, options);
            checks.Compare("agrees with the reference", Prefixed("advantages", Comparisons.Difference(expected, advantages, Tolerance))
                                                      ?? Prefixed("returns", Comparisons.Difference(expectedReturns, returns, Tolerance)));
        }
        else
        {
            checks.Skip("agrees with the reference", "no reference given");
        }
    }

    // A batch as arrays, made into an AdvantageInput for each call (a ref struct cannot be kept).
    private sealed record Arrays(float[] Rewards, float[] Values, int[] Lengths, bool[] Truncated, float[] FinalValues, int[] Groups)
    {
        public AdvantageInput Input() => new(Rewards, Values, Lengths, Truncated, FinalValues, Groups);
    }

    private static Arrays Batch(ContractCase @case, bool usesValues)
    {
        var data = @case.Data;
        var random = new Random((int)data["seed"]!);
        int episodes = (int)data["episodes"]!, longest = (int)data["longest"]!, groups = (int)data["groups"]!;
        bool values = usesValues || (bool)data["values"]!, truncated = (bool)data["truncated"]!;
        int[] lengths = [.. Enumerable.Range(0, episodes).Select(_ => random.Next(1, longest + 1))];
        int steps = lengths.Sum();
        float[] rewards = [.. Enumerable.Range(0, steps).Select(_ => random.Next(4) switch { 0 => 0f, 1 => 1f, _ => (random.NextSingle() - 0.5f) * 4 })];
        return new Arrays(rewards,
            values ? [.. Enumerable.Range(0, steps).Select(_ => (random.NextSingle() - 0.5f) * 2)] : [],
            lengths,
            truncated ? [.. Enumerable.Range(0, episodes).Select(_ => random.Next(2) == 0)] : [],
            values ? [.. Enumerable.Range(0, episodes).Select(_ => (random.NextSingle() - 0.5f) * 2)] : [],
            [.. Enumerable.Range(0, episodes).Select(e => (int)((long)e * groups / episodes))]);
    }

    // The outputs, filled with NaN first so a step left unwritten shows.
    private static (float[] Advantages, float[] Returns) Estimate(IAdvantageEstimator estimator, Arrays batch, IReadOnlyDictionary<string, string>? options)
    {
        var advantages = new float[batch.Rewards.Length];
        var returns = new float[batch.Rewards.Length];
        advantages.AsSpan().Fill(float.NaN);
        returns.AsSpan().Fill(float.NaN);
        estimator.Estimate(batch.Input(), advantages, returns, options);
        return (advantages, returns);
    }

    private static bool Refused(Action call)
    {
        try
        {
            call();
            return false;
        }
        catch (ArgumentException)
        {
            return true;
        }
    }

    // Null when the two hold the same bits; else the first that differs.
    private static string? Bits(float[] expected, float[] actual)
    {
        if (expected.Length != actual.Length)
        {
            return $"{actual.Length} values, {expected.Length} expected";
        }

        for (int i = 0; i < expected.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
            {
                return FormattableString.Invariant($"element {i} is {actual[i]:R}, {expected[i]:R} the first time");
            }
        }

        return null;
    }

    private static string? Prefixed(string what, string? difference) => difference is null ? null : $"{what}: {difference}";

    private static ContractCase Case(string name, int seed, int episodes, int longest, int groups, bool values, bool truncated) =>
        new($"{name} ({episodes} episodes of up to {longest} steps, {groups} groups{(values ? ", values" : "")}{(truncated ? ", truncated" : "")})",
            new JsonObject { ["seed"] = seed, ["episodes"] = episodes, ["longest"] = longest, ["groups"] = groups, ["values"] = values, ["truncated"] = truncated });
}
