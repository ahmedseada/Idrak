// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using Idrak.Abstraction.Reinforcement;

namespace Idrak.Reinforcement;

// The library's advantage estimators, their options their constructors'. Every sum runs in a fixed order, in double precision, rounded
// to float once when written: episode by episode in the input's order, each episode's steps backwards from its last, a
// group's episodes in their order (rule 86). None keeps state between calls; the pooled arrays one rents are returned in
// `finally`.
internal static class Estimators
{
    // The options by key the contracts still pass: the estimators' and the losses' options are their constructors', so
    // every key is one they do not take.
    public static void NoOptions(string what, IReadOnlyDictionary<string, string>? options)
    {
        if (options is { Count: > 0 })
        {
            throw new ArgumentException($"The {what} takes its options as constructor parameters, not by key (given '{string.Join("', '", options.Keys)}').", nameof(options));
        }
    }

    // A number option within [min, max].
    public static float InRange(float value, float min, float max, string name) =>
        value >= min && value <= max ? value
            : throw new ArgumentOutOfRangeException(name, value, string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{name} is a number in [{min}, {max}]."));

    // A choice among `choices`.
    public static string OneOf(string value, string name, params ReadOnlySpan<string> choices)
    {
        ArgumentNullException.ThrowIfNull(value);
        foreach (string choice in choices)
        {
            if (string.Equals(choice, value, StringComparison.OrdinalIgnoreCase))
            {
                return choice;
            }
        }

        throw new ArgumentException($"{name} is one of {string.Join(", ", choices.ToArray())}, not '{value}'.", name);
    }

    // The outputs' lengths, and values when the estimator needs them.
    public static void CheckOutputs(string name, AdvantageInput input, Span<float> advantages, Span<float> returns, bool needsValues)
    {
        if (advantages.Length != input.Steps || returns.Length != input.Steps)
        {
            throw new ArgumentException($"The estimator {name} writes {input.Steps} advantages and returns (a step each): {advantages.Length} and {returns.Length} given.");
        }

        if (needsValues && !input.HasValues)
        {
            throw new ArgumentException($"The estimator {name} needs the critic's value estimates: the input has none (a policy without a value head; "
                                        + $"use {ReinforcementNames.Group}, {ReinforcementNames.LeaveOneOut} or {ReinforcementNames.Returns} instead).");
        }
    }

    // Each step's reward-to-go (the undiscounted sum of the rewards from it to the episode's end): the return the
    // group-relative estimators write for a value head.
    public static void RewardsToGo(AdvantageInput input, Span<float> returns)
    {
        int end = 0;
        for (int e = 0; e < input.Episodes; e++)
        {
            int start = end;
            end += input.Lengths[e];
            double sum = 0;
            for (int t = end - 1; t >= start; t--)
            {
                sum += input.Rewards[t];
                returns[t] = (float)sum;
            }
        }
    }

    // Each episode's return (its rewards summed in step order) into `totals` [episodes].
    public static void EpisodeReturns(AdvantageInput input, Span<double> totals)
    {
        int at = 0;
        for (int e = 0; e < input.Episodes; e++)
        {
            double sum = 0;
            for (int t = 0; t < input.Lengths[e]; t++)
            {
                sum += input.Rewards[at + t];
            }

            totals[e] = sum;
            at += input.Lengths[e];
        }
    }

    // The batch's advantages to mean 0 and spread 1 (the sample deviation, n - 1), both summed in step order; one step, or
    // all equal, gives 0.
    public static void Normalize(Span<float> advantages)
    {
        int n = advantages.Length;
        double sum = 0;
        foreach (float a in advantages)
        {
            sum += a;
        }

        double mean = sum / n, squares = 0;
        foreach (float a in advantages)
        {
            squares += (a - mean) * (a - mean);
        }

        double spread = n > 1 ? Math.Sqrt(squares / (n - 1)) : 0;
        double scale = 1.0 / (spread + 1e-8);
        for (int i = 0; i < n; i++)
        {
            advantages[i] = (float)((advantages[i] - mean) * scale);
        }
    }

    // The episodes ordered by (group, number), so a group's episodes come together in their order: keys rented, the
    // caller returns them. Sorting composite keys is the same whatever the sort's stability.
    public static long[] ByGroup(AdvantageInput input)
    {
        int episodes = input.Episodes;
        long[] keys = ArrayPool<long>.Shared.Rent(episodes);
        for (int e = 0; e < episodes; e++)
        {
            keys[e] = ((long)input.Group(e) << 32) | (uint)e;
        }

        Array.Sort(keys, 0, episodes);
        return keys;
    }

    public static int GroupOf(long key) => (int)(key >> 32);

    public static int EpisodeOf(long key) => (int)(key & 0xFFFF_FFFF);

    // The first step of each episode into `offsets` [episodes].
    public static void Offsets(AdvantageInput input, Span<int> offsets)
    {
        int at = 0;
        for (int e = 0; e < input.Episodes; e++)
        {
            offsets[e] = at;
            at += input.Lengths[e];
        }
    }
}

/// <summary>
/// Generalized advantage estimation (Schulman, Moritz, Levine, Jordan, Abbeel 2015, "High-Dimensional Continuous Control
/// Using Generalized Advantage Estimation", arXiv:1506.02438). With the critic's value V(s_t) of each step's observation,
/// the temporal-difference error δ_t = r_t + γ V(s_{t+1}) − V(s_t), where V after the last step is the episode's final value
/// when it was truncated and 0 when it terminated; the advantage A_t = δ_t + γλ A_{t+1} (A after the last step 0), summed
/// backwards from each episode's last step; the return R_t = A_t + V(s_t) (the value head's target, TD(λ)). Options (the
/// constructor's): gamma (γ in [0, 1], default 0.99), lambda (λ in [0, 1], default 0.95), normalize (the batch's advantages
/// to mean 0 and spread 1, default false).
/// </summary>
internal sealed class GeneralizedAdvantageEstimator : IAdvantageEstimator
{
    private readonly float _gamma, _lambda;
    private readonly bool _normalize;

    public GeneralizedAdvantageEstimator(float gamma = 0.99f, float lambda = 0.95f, bool normalize = false)
    {
        _gamma = Estimators.InRange(gamma, 0f, 1f, nameof(gamma));
        _lambda = Estimators.InRange(lambda, 0f, 1f, nameof(lambda));
        _normalize = normalize;
    }

    public string Name => ReinforcementNames.Gae;

    public bool UsesValues => true;

    public void Estimate(AdvantageInput input, Span<float> advantages, Span<float> returns, IReadOnlyDictionary<string, string>? options = null)
    {
        Estimators.NoOptions($"advantage estimator {Name}", options);
        double gamma = _gamma;
        double lambda = _lambda;
        bool normalize = _normalize;
        Estimators.CheckOutputs(Name, input, advantages, returns, needsValues: true);
        int end = 0;
        for (int e = 0; e < input.Episodes; e++)
        {
            int start = end;
            end += input.Lengths[e];
            double next = input.IsTruncated(e) ? input.FinalValues[e] : 0.0;
            double advantage = 0;
            for (int t = end - 1; t >= start; t--)
            {
                double value = input.Values[t];
                double delta = input.Rewards[t] + gamma * next - value;
                advantage = delta + gamma * lambda * advantage;
                advantages[t] = (float)advantage;
                returns[t] = (float)(advantage + value);
                next = value;
            }
        }

        if (normalize)
        {
            Estimators.Normalize(advantages);
        }
    }
}

/// <summary>
/// Group-relative advantages (GRPO: Shao et al. 2024, "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open
/// Language Models", arXiv:2402.03300, section 4.1.2, outcome supervision). Each episode's return R_i (its rewards summed)
/// is compared with its group's (the episodes that answered the same situation): A_i = (R_i − mean(R)) / (std(R) + ε), or
/// A_i = R_i − mean(R) without the scale (Liu et al. 2025, "Understanding R1-Zero-Like Training: A Critical Perspective",
/// arXiv:2503.20783, drop it as a bias towards easy and hard groups); std is the sample deviation (n − 1), so a group of one
/// gets 0. Every step of the episode gets A_i; the return of a step is its reward-to-go. Options (the constructor's):
/// scale (default true), epsilon (at least 0, default 1e-4). Needs no critic.
/// </summary>
internal sealed class GroupAdvantageEstimator : IAdvantageEstimator
{
    private readonly bool _scale;
    private readonly float _epsilon;

    public GroupAdvantageEstimator(bool scale = true, float epsilon = 1e-4f)
    {
        _scale = scale;
        _epsilon = Estimators.InRange(epsilon, 0f, float.MaxValue, nameof(epsilon));
    }

    public string Name => ReinforcementNames.Group;

    public bool UsesValues => false;

    public void Estimate(AdvantageInput input, Span<float> advantages, Span<float> returns, IReadOnlyDictionary<string, string>? options = null)
    {
        Estimators.NoOptions($"advantage estimator {Name}", options);
        bool scale = _scale;
        double epsilon = _epsilon;
        Estimators.CheckOutputs(Name, input, advantages, returns, needsValues: false);
        GroupEstimate(input, advantages, (totals, members, count, result) =>
        {
            double sum = 0;
            for (int m = 0; m < count; m++)
            {
                sum += totals[members[m]];
            }

            double mean = sum / count, squares = 0;
            for (int m = 0; m < count; m++)
            {
                double d = totals[members[m]] - mean;
                squares += d * d;
            }

            double divisor = scale ? (count > 1 ? Math.Sqrt(squares / (count - 1)) : 0) + epsilon : 1.0;
            for (int m = 0; m < count; m++)
            {
                result[m] = divisor > 0 ? (totals[members[m]] - mean) / divisor : 0.0;          // equal returns with no epsilon: 0, not 0 / 0
            }
        });
        Estimators.RewardsToGo(input, returns);
    }

    // The advantage of each episode from its group, written over its steps: `advantagesOf` gets the totals by episode, the
    // group's members (episode indices in order) and their count, and writes each member's advantage into the last array.
    internal static void GroupEstimate(AdvantageInput input, Span<float> advantages, Action<double[], int[], int, double[]> advantagesOf)
    {
        int episodes = input.Episodes;
        long[] keys = Estimators.ByGroup(input);
        double[] totals = ArrayPool<double>.Shared.Rent(episodes);
        int[] members = ArrayPool<int>.Shared.Rent(episodes);
        int[] offsets = ArrayPool<int>.Shared.Rent(episodes);
        double[] result = ArrayPool<double>.Shared.Rent(episodes);
        try
        {
            Estimators.EpisodeReturns(input, totals.AsSpan(0, episodes));
            Estimators.Offsets(input, offsets.AsSpan(0, episodes));
            int first = 0;
            while (first < episodes)
            {
                int group = Estimators.GroupOf(keys[first]), count = 0;
                while (first + count < episodes && Estimators.GroupOf(keys[first + count]) == group)
                {
                    members[count] = Estimators.EpisodeOf(keys[first + count]);
                    count++;
                }

                advantagesOf(totals, members, count, result);
                for (int i = 0; i < count; i++)
                {
                    int e = members[i];
                    advantages.Slice(offsets[e], input.Lengths[e]).Fill((float)result[i]);
                }

                first += count;
            }
        }
        finally
        {
            ArrayPool<long>.Shared.Return(keys);
            ArrayPool<double>.Shared.Return(totals);
            ArrayPool<int>.Shared.Return(members);
            ArrayPool<int>.Shared.Return(offsets);
            ArrayPool<double>.Shared.Return(result);
        }
    }
}

/// <summary>
/// Leave-one-out advantages (RLOO: Ahmadian et al. 2024, "Back to Basics: Revisiting REINFORCE Style Optimization for
/// Learning from Human Feedback in LLMs", arXiv:2402.14740; the baseline of Kool, van Hoof, Welling 2019, "Buy 4 REINFORCE
/// Samples, Get a Baseline for Free!"). Each episode's return R_i against the mean of the other k − 1 returns of its group:
/// A_i = R_i − (1 / (k − 1)) Σ_{j ≠ i} R_j, an unbiased baseline since it does not depend on episode i; a group of one gets 0.
/// Every step of the episode gets A_i; the return of a step is its reward-to-go. No options; needs no critic.
/// </summary>
internal sealed class LeaveOneOutEstimator : IAdvantageEstimator
{
    public string Name => ReinforcementNames.LeaveOneOut;

    public bool UsesValues => false;

    public void Estimate(AdvantageInput input, Span<float> advantages, Span<float> returns, IReadOnlyDictionary<string, string>? options = null)
    {
        Estimators.NoOptions($"advantage estimator {Name}", options);
        Estimators.CheckOutputs(Name, input, advantages, returns, needsValues: false);
        GroupAdvantageEstimator.GroupEstimate(input, advantages, static (totals, members, count, result) =>
        {
            double sum = 0;
            for (int m = 0; m < count; m++)
            {
                sum += totals[members[m]];
            }

            for (int m = 0; m < count; m++)
            {
                double own = totals[members[m]];
                result[m] = count < 2 ? 0.0 : own - (sum - own) / (count - 1);
            }
        });
        Estimators.RewardsToGo(input, returns);
    }
}

/// <summary>
/// Discounted returns, REINFORCE's weights (Williams 1992, "Simple Statistical Gradient-Following Algorithms for
/// Connectionist Reinforcement Learning"; Sutton and Barto 2018, section 13.3): G_t = r_t + γ G_{t+1}, summed backwards from
/// each episode's last step, where G after the last step is the episode's final value when it was truncated and the input
/// has values, else 0. The advantage is G_t less a baseline (section 13.4): none (A_t = G_t), mean (the batch's mean of G,
/// summed in step order) or value (the critic's V(s_t); the input must have values). The return is G_t. Options (the
/// constructor's): gamma (γ in [0, 1], default 0.99), baseline (none, mean or value; default none), normalize (the batch's
/// advantages to mean 0 and spread 1, default false).
/// </summary>
internal sealed class ReturnsEstimator : IAdvantageEstimator
{
    private readonly float _gamma;
    private readonly string _baseline;
    private readonly bool _normalize;

    public ReturnsEstimator(float gamma = 0.99f, string baseline = "none", bool normalize = false)
    {
        _gamma = Estimators.InRange(gamma, 0f, 1f, nameof(gamma));
        _baseline = Estimators.OneOf(baseline, nameof(baseline), "none", "mean", "value");
        _normalize = normalize;
    }

    public string Name => ReinforcementNames.Returns;

    public bool UsesValues => false;

    public void Estimate(AdvantageInput input, Span<float> advantages, Span<float> returns, IReadOnlyDictionary<string, string>? options = null)
    {
        Estimators.NoOptions($"advantage estimator {Name}", options);
        double gamma = _gamma;
        string baseline = _baseline;
        bool normalize = _normalize;
        Estimators.CheckOutputs(Name, input, advantages, returns, needsValues: baseline == "value");
        int end = 0;
        double sum = 0;
        for (int e = 0; e < input.Episodes; e++)
        {
            int start = end;
            end += input.Lengths[e];
            double g = input.IsTruncated(e) && input.HasValues ? input.FinalValues[e] : 0.0;
            for (int t = end - 1; t >= start; t--)
            {
                g = input.Rewards[t] + gamma * g;
                returns[t] = (float)g;
            }
        }

        for (int t = 0; t < input.Steps; t++)
        {
            sum += returns[t];
        }

        double mean = sum / input.Steps;
        for (int t = 0; t < input.Steps; t++)
        {
            advantages[t] = baseline switch
            {
                "mean" => (float)(returns[t] - mean),
                "value" => (float)((double)returns[t] - input.Values[t]),
                _ => returns[t],
            };
        }

        if (normalize)
        {
            Estimators.Normalize(advantages);
        }
    }
}
