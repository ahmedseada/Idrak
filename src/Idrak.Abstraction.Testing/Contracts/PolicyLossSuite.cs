// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Reinforcement;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The checks of a policy loss (<see cref="IPolicyLoss"/>) on inputs the suite draws: sequences of random unit counts,
/// log-probabilities with behaviour and reference ones near them (or none), advantages of either sign, a value head's
/// predictions and targets and the entropies or not:
/// <list type="bullet">
/// <item>the loss is one element, finite, and the same every time;</item>
/// <item>its gradient reaches the log-probabilities (finite, not all 0), and agrees with central differences wherever the
/// loss is smooth there (a clipped ratio has kinks, where the one-sided differences disagree; those coordinates are
/// left out);</item>
/// <item>an option not in its <see cref="IConfigurableKind.Keys"/> is refused, naming it;</item>
/// <item>given a reference (the library's loss of that name), the same loss within <see cref="Tolerance"/>.</item>
/// </list>
/// A loss that needs what a case does not give (a reference, values) may refuse that case with an
/// <see cref="ArgumentException"/>; set <see cref="NeedsReference"/> or <see cref="NeedsValues"/> so every case gives it.
/// </summary>
/// <param name="options">The loss's options for every call; null for its defaults.</param>
/// <param name="reference">The library's loss to agree with, or null.</param>
public sealed class PolicyLossSuite(IReadOnlyDictionary<string, string>? options = null, IPolicyLoss? reference = null) : ContractSuite<IPolicyLoss>
{
    private const string UnknownOption = "conformance-unknown-option";

    /// <inheritdoc />
    public override string Name => "policy-loss";

    /// <summary>The agreement wanted with the reference, relative to the loss (at least 1).</summary>
    public float Tolerance { get; init; } = 1e-5f;

    /// <summary>Every case gives reference log-probabilities (a loss with a KL penalty to the reference).</summary>
    public bool NeedsReference { get; init; }

    /// <summary>Every case gives a value head's predictions and targets (a loss with a value term).</summary>
    public bool NeedsValues { get; init; }

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        yield return Case("one unit, on-policy", 1, 1, 1, behavior: false, withReference: false, values: false);
        yield return Case("one sequence", 2, 1, 8, behavior: true, withReference: false, values: false);
        yield return Case("a few sequences with a reference", 3, 4, 6, behavior: true, withReference: true, values: false);
        yield return Case("values and entropies", 4, 3, 5, behavior: true, withReference: true, values: true);
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        return Case("random", random.Next(), random.Next(1, large ? 33 : 6), random.Next(1, large ? 64 : 10), random.Next(2) == 0, random.Next(2) == 0, random.Next(2) == 0);
    }

    /// <inheritdoc />
    public override void Run(IPolicyLoss implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var loss = implementation;
        var values = Draw(@case);
        using var scope = new TensorScope();
        checks.Check("a name and a summary", !string.IsNullOrWhiteSpace(loss.Name) && !string.IsNullOrWhiteSpace(loss.Summary), $"name \"{loss.Name}\", summary \"{loss.Summary}\"");

        float[] Value(float[] logProbabilities, IReadOnlyDictionary<string, string>? with = null)
        {
            using (Autograd.NoGrad())
            {
                return loss.Loss(values.Input(logProbabilities, requiresGrad: false), with ?? options).ToArray();
            }
        }

        var value = Value(values.LogProbabilities);
        checks.Check("a one-element loss", value.Length == 1, $"{value.Length} elements");
        checks.Check("the loss is finite", value.Length == 1 && float.IsFinite(value[0]), $"loss {string.Join(", ", value.Take(4))}");
        if (value.Length != 1 || !float.IsFinite(value[0]))
        {
            return;
        }

        checks.Compare("the same loss every time", Comparisons.Difference(value, Value(values.LogProbabilities), 0f));

        // The gradient: to the log-probabilities, against central differences where the loss is smooth.
        var input = values.Input(values.LogProbabilities, requiresGrad: true);
        var x = input.LogProbabilities;
        loss.Loss(input, options).Backward();
        var gradient = x.Grad?.ToArray();
        checks.Check("the gradient reaches the log-probabilities", gradient is not null && gradient.All(float.IsFinite) && gradient.Any(g => g != 0f),
            gradient is null ? "no gradient reached the log-probabilities" : $"gradient {string.Join(", ", gradient.Take(8))}");
        if (gradient is not null)
        {
            var random = new Random((int)@case.Data["seed"]! + 1);
            string? wrong = null;
            int compared = 0;
            for (int probe = 0; probe < Math.Min(12, gradient.Length) && wrong is null; probe++)
            {
                int i = random.Next(gradient.Length);
                const float h = 1e-3f;
                var plus = (float[])values.LogProbabilities.Clone();
                var minus = (float[])values.LogProbabilities.Clone();
                plus[i] += h;
                minus[i] -= h;
                double at = value[0], up = Value(plus)[0], down = Value(minus)[0];
                double forward = (up - at) / h, backward = (at - down) / h, central = (up - down) / (2 * h);
                if (Math.Abs(forward - backward) > 1e-2 * Math.Max(1, Math.Abs(central)))
                {
                    continue;                                                          // a kink (a clip's edge) lies within h
                }

                compared++;
                if (Math.Abs(central - gradient[i]) > 2e-2 * Math.Max(1, Math.Abs(central)))
                {
                    wrong = $"unit {i}: gradient {gradient[i]}, central differences {central:G6}";
                }
            }

            if (compared == 0 && wrong is null)
            {
                checks.Skip("the gradient agrees with central differences where the loss is smooth", "every probe lay at a kink");
            }
            else
            {
                checks.Compare("the gradient agrees with central differences where the loss is smooth", wrong);
            }
        }

        var unknown = new Dictionary<string, string>(StringComparer.Ordinal) { [UnknownOption] = "1" };
        foreach (var (key, option) in options ?? new Dictionary<string, string>())
        {
            unknown[key] = option;
        }

        try
        {
            Value(values.LogProbabilities, unknown);
            checks.Fail("an option not in Keys is refused, naming it", $"computed a loss with the option '{UnknownOption}'");
        }
        catch (ArgumentException e)
        {
            checks.Check("an option not in Keys is refused, naming it", e.Message.Contains(UnknownOption, StringComparison.Ordinal), $"\"{e.Message}\" does not name '{UnknownOption}'");
        }

        if (reference is not null)
        {
            float[] expected;
            using (Autograd.NoGrad())
            {
                expected = reference.Loss(values.Input(values.LogProbabilities, requiresGrad: false), options).ToArray();
            }

            checks.Compare("agrees with the reference", Comparisons.Difference(expected, value, Tolerance));
        }
        else
        {
            checks.Skip("agrees with the reference", "no reference given");
        }
    }

    // A case's values as arrays, made into an input with the log-probabilities given. On-policy (no behaviour given), the
    // drawn log-probabilities stay the behaviour of a perturbed input, so a difference quotient moves the ratio as the
    // gradient does.
    private sealed record Values(float[] LogProbabilities, int[] Counts, float[] Advantages, float[]? Behavior, float[]? Reference, float[]? Predictions,
        float[]? Returns, float[]? Entropies)
    {
        public PolicyLossInput Input(float[] logProbabilities, bool requiresGrad)
        {
            int n = logProbabilities.Length;
            Tensor? Of(float[]? v, bool grad = false) => v is null ? null : Tensor.From(v, [n], requiresGrad: grad);
            var behavior = Behavior ?? (ReferenceEquals(logProbabilities, LogProbabilities) ? null : LogProbabilities);
            return new PolicyLossInput(Tensor.From(logProbabilities, [n], requiresGrad: requiresGrad), Counts, Tensor.From(Advantages, [n]), Of(behavior), Of(Reference),
                Of(Predictions, requiresGrad), Of(Returns), Of(Entropies, requiresGrad));
        }
    }

    private Values Draw(ContractCase @case)
    {
        var data = @case.Data;
        var random = new Random((int)data["seed"]!);
        int sequences = (int)data["sequences"]!, longest = (int)data["longest"]!;
        bool behavior = (bool)data["behavior"]!, withReference = NeedsReference || (bool)data["reference"]!, values = NeedsValues || (bool)data["values"]!;
        int[] counts = [.. Enumerable.Range(0, sequences).Select(_ => random.Next(1, longest + 1))];
        int n = counts.Sum();
        float[] Near(float[] around) => [.. around.Select(v => v + (random.NextSingle() - 0.5f) * 0.6f)];
        float[] Uniform(float low, float high) => [.. Enumerable.Range(0, n).Select(_ => low + random.NextSingle() * (high - low))];
        var logProbabilities = Uniform(-3f, -0.1f);
        var advantages = Uniform(-1f, 1f);
        return new Values(logProbabilities, counts, advantages, behavior ? Near(logProbabilities) : null, withReference ? Near(logProbabilities) : null,
            values ? Uniform(-1f, 1f) : null, values ? Uniform(-1f, 1f) : null, values ? Uniform(0f, 2f) : null);
    }

    private static ContractCase Case(string name, int seed, int sequences, int longest, bool behavior, bool withReference, bool values) =>
        new($"{name} ({sequences} sequences of up to {longest} units{(behavior ? ", off-policy" : "")}{(withReference ? ", a reference" : "")}{(values ? ", values" : "")})",
            new JsonObject { ["seed"] = seed, ["sequences"] = sequences, ["longest"] = longest, ["behavior"] = behavior, ["reference"] = withReference, ["values"] = values });
}
