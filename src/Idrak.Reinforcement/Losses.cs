// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Reinforcement;

namespace Idrak.Reinforcement;

// The library's policy losses (PolicyLosses): compositions of tensor operations, so autograd carries the gradient to the
// log-probabilities, values and entropies the policy recorded. Each unit's term is weighted by the step's normalization and
// the weighted terms summed by one reduction over the units in order (rule 86).
internal static class PolicyLossTerms
{
    public static void Register()
    {
        PolicyLosses.Register(new ReinforceLoss());
        PolicyLosses.Register(new ClippedRatioLoss(ReinforcementNames.Ppo, "token", values: true,
            "PPO (Schulman et al. 2017): a clipped probability ratio, a value loss and an entropy bonus; a KL penalty to a reference when weighted"));
        PolicyLosses.Register(new ClippedRatioLoss(ReinforcementNames.Grpo, "sequence", values: false,
            "GRPO (Shao et al. 2024): a clipped probability ratio and a KL penalty to the reference (k1, k2 or k3), normalized per sequence, per token or by a constant"));
    }

    // Each unit's weight in the step's sum [units], not recorded: token, 1 / the step's units; sequence, 1 / (its sequence's
    // units · the step's sequences); constant, 1 / the constant.
    public static Tensor Weights(PolicyLossInput input, string normalize, float constant, string loss)
    {
        var device = input.LogProbabilities.Device;
        int units = input.Units;
        switch (normalize)
        {
            case "token":
                return Tensor.Full([units], 1f / input.StepUnits, device);
            case "constant":
                if (!(constant > 0))
                {
                    throw new ArgumentException($"The policy loss {loss} normalizes by a constant: give the option '{ReinforcementNames.ConstantKey}' (the divisor of the step's sum, above 0).");
                }

                return Tensor.Full([units], 1f / constant, device);
            default:
                float[] weights = new float[units];
                int at = 0;
                for (int s = 0; s < input.Sequences; s++)
                {
                    int count = input.Counts[s];
                    weights.AsSpan(at, count).Fill(1f / ((float)count * input.StepSequences));
                    at += count;
                }

                return Tensor.From(weights, [units], device);
        }
    }

    // The per-unit KL of the current policy to the reference, estimated from the units sampled (Schulman 2020,
    // "Approximating KL divergence", http://joschu.net/blog/kl-approx.html), with d = log π_ref − log π_θ: k1 = −d (unbiased,
    // may be negative), k2 = d² / 2, k3 = e^d − d − 1 (unbiased and never negative; GRPO's, equation 4 of Shao et al. 2024).
    public static Tensor Kl(PolicyLossInput input, string estimator, string loss)
    {
        var reference = input.ReferenceLogProbabilities
                        ?? throw new ArgumentException($"The policy loss {loss} has a KL penalty (option '{ReinforcementNames.KlKey}' above 0) and needs the reference policy's log-probabilities: "
                                                       + $"the input has none (give a reference policy, or set '{ReinforcementNames.KlKey}' to 0).");
        var d = reference - input.LogProbabilities;
        return estimator switch
        {
            "k1" => -d,
            "k2" => d.Square() * 0.5f,
            _ => d.Exp() - d - 1f,
        };
    }

    // −(entropy weight) · H per unit, when a weight and entropies are given; null otherwise.
    public static Tensor? EntropyBonus(PolicyLossInput input, float weight) =>
        weight != 0f && input.Entropies is { } entropies ? entropies * -weight : null;
}

/// <summary>
/// REINFORCE (Williams 1992, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning";
/// Sutton and Barto 2018, section 13.3): L = −Σ_u w_u A_u log π_θ(a_u | s_u), the score-function gradient weighted by each
/// unit's advantage, with −c_H Σ_u w_u H_u for an entropy bonus and c_V Σ_u w_u (V_u − R_u)² for a value head when values are
/// given (a learned baseline). w_u is the step's normalization: token (1 / the step's units, the default), sequence
/// (1 / (|sequence| · the step's sequences)) or constant (1 / constant). It ignores the behaviour policy: an on-policy
/// estimator. Options: entropy (c_H, default 0), value (c_V, default 0.5), normalize, constant.
/// </summary>
internal sealed class ReinforceLoss : IPolicyLoss
{
    public string Name => ReinforcementNames.Reinforce;

    public string Summary => "REINFORCE (Williams 1992): log-probabilities weighted by the advantages, with an entropy bonus and a value loss when weighted";

    public IReadOnlyCollection<string> Keys { get; } =
        [ReinforcementNames.EntropyKey, ReinforcementNames.ValueKey, ReinforcementNames.NormalizeKey, ReinforcementNames.ConstantKey];

    public Tensor Loss(PolicyLossInput input, IReadOnlyDictionary<string, string>? options = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        var reader = new OptionReader($"policy loss {Name}", Keys, options);
        float entropy = reader.Float(ReinforcementNames.EntropyKey, 0f);
        float value = reader.Float(ReinforcementNames.ValueKey, 0.5f, 0f);
        string normalize = reader.Choice(ReinforcementNames.NormalizeKey, "token", "token", "sequence", "constant");
        float constant = reader.Float(ReinforcementNames.ConstantKey, 0f, 0f);
        var weights = PolicyLossTerms.Weights(input, normalize, constant, Name);
        var perUnit = -(input.Advantages * input.LogProbabilities);
        if (PolicyLossTerms.EntropyBonus(input, entropy) is { } bonus)
        {
            perUnit = perUnit + bonus;
        }

        if (input.Values is { } values && value > 0f)
        {
            perUnit = perUnit + (values - input.Returns!).Square() * value;
        }

        return (perUnit * weights).Sum();
    }
}

/// <summary>
/// A clipped probability ratio, as PPO and GRPO write it. With the ratio ρ_u = π_θ(a_u | s_u) / π_old(a_u | s_u) (the
/// behaviour policy that sampled the unit; 1 on-policy) and the unit's advantage A_u, the clipped surrogate
/// min(ρ_u A_u, clip(ρ_u, 1 − ε, 1 + ε_high) A_u), which takes no gradient from a ratio already past its clip in the
/// advantage's direction. The loss is L = Σ_u w_u [−min(…) + β KL_u − c_H H_u] + c_V Σ_u w_u (V_u − R_u)².
/// <list type="bullet">
/// <item>ppo (Schulman, Wolski, Dhariwal, Radford, Klimov 2017, "Proximal Policy Optimization Algorithms",
/// arXiv:1707.06347, equations 7 and 9): defaults ε = 0.2, c_V = 0.5 (the value term when values are given), c_H = 0,
/// β = 0, normalized by the step's units (token).</item>
/// <item>grpo (Shao et al. 2024, "DeepSeekMath", arXiv:2402.03300, equation 3): the same surrogate with the KL to the
/// reference inside the sum, estimated by k3 (or k1, k2), normalized per sequence by default (1/G Σ_i 1/|o_i| Σ_t); no value
/// term. β defaults to 0, as recent practice has it (DAPO, Yu et al. 2025, arXiv:2503.14476; TRL's GRPOTrainer); the
/// paper's 0.04 is the option kl=0.04, which needs the reference's log-probabilities.</item>
/// </list>
/// KL_u is the estimator's (see the kl-estimator option); a β above 0 refuses an input without a reference. Options: clip (ε),
/// clip-high (ε_high, default ε), kl (β), kl-estimator (k1, k2, k3; default k3), entropy (c_H), value (ppo only, c_V),
/// normalize (token, sequence, constant), constant.
/// </summary>
internal sealed class ClippedRatioLoss(string name, string normalization, bool values, string summary) : IPolicyLoss
{
    public string Name => name;

    public string Summary => summary;

    public IReadOnlyCollection<string> Keys { get; } = values
        ? [ReinforcementNames.ClipKey, ReinforcementNames.ClipHighKey, ReinforcementNames.ValueKey, ReinforcementNames.EntropyKey, ReinforcementNames.KlKey,
            ReinforcementNames.KlEstimatorKey, ReinforcementNames.NormalizeKey, ReinforcementNames.ConstantKey]
        : [ReinforcementNames.ClipKey, ReinforcementNames.ClipHighKey, ReinforcementNames.KlKey, ReinforcementNames.KlEstimatorKey, ReinforcementNames.EntropyKey,
            ReinforcementNames.NormalizeKey, ReinforcementNames.ConstantKey];

    public Tensor Loss(PolicyLossInput input, IReadOnlyDictionary<string, string>? options = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        var reader = new OptionReader($"policy loss {Name}", Keys, options);
        float clip = reader.Float(ReinforcementNames.ClipKey, 0.2f, 0f, 1f);
        float clipHigh = reader.Float(ReinforcementNames.ClipHighKey, clip, 0f);
        float kl = reader.Float(ReinforcementNames.KlKey, 0f, 0f);
        string estimator = reader.Choice(ReinforcementNames.KlEstimatorKey, "k3", "k1", "k2", "k3");
        float entropy = reader.Float(ReinforcementNames.EntropyKey, 0f);
        float value = values ? reader.Float(ReinforcementNames.ValueKey, 0.5f, 0f) : 0f;
        string normalize = reader.Choice(ReinforcementNames.NormalizeKey, normalization, "token", "sequence", "constant");
        float constant = reader.Float(ReinforcementNames.ConstantKey, 0f, 0f);
        var weights = PolicyLossTerms.Weights(input, normalize, constant, Name);

        var ratio = (input.LogProbabilities - input.BehaviorLogProbabilities).Exp();
        var surrogate = (ratio * input.Advantages).Minimum(ratio.Clamp(1f - clip, 1f + clipHigh) * input.Advantages);
        var perUnit = -surrogate;
        if (kl > 0f)
        {
            perUnit = perUnit + PolicyLossTerms.Kl(input, estimator, Name) * kl;
        }

        if (PolicyLossTerms.EntropyBonus(input, entropy) is { } bonus)
        {
            perUnit = perUnit + bonus;
        }

        if (input.Values is { } predicted && value > 0f)
        {
            perUnit = perUnit + (predicted - input.Returns!).Square() * value;
        }

        return (perUnit * weights).Sum();
    }
}
