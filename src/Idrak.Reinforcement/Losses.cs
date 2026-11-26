// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Reinforcement;

namespace Idrak.Reinforcement;

// The library's policy losses, their options their constructors': compositions of tensor operations, so autograd carries the gradient to the
// log-probabilities, values and entropies the policy recorded. Each unit's term is weighted by the step's normalization and
// the weighted terms summed by one reduction over the units in order (rule 86).
internal static class PolicyLossTerms
{
    // How a loss divides the step's sum: token, sequence or constant.
    public static string Normalization(string normalize) => Estimators.OneOf(normalize, nameof(normalize), "token", "sequence", "constant");

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
                    throw new ArgumentException($"The policy loss {loss} normalizes by a constant: give it the constant (the divisor of the step's sum, above 0).");
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
                        ?? throw new ArgumentException($"The policy loss {loss} has a KL penalty (its weight above 0) and needs the reference policy's log-probabilities: "
                                                       + "the input has none (give a reference policy, or weight the penalty 0).");
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
/// estimator. Options (the constructor's): entropy (c_H, default 0), value (c_V, at least 0, default 0.5), normalize,
/// constant (at least 0).
/// </summary>
internal sealed class ReinforceLoss : IPolicyLoss
{
    private readonly float _entropy, _value, _constant;
    private readonly string _normalize;

    public ReinforceLoss(float entropy = 0f, float value = 0.5f, string normalize = "token", float constant = 0f)
    {
        _entropy = entropy;
        _value = Estimators.InRange(value, 0f, float.MaxValue, nameof(value));
        _normalize = PolicyLossTerms.Normalization(normalize);
        _constant = Estimators.InRange(constant, 0f, float.MaxValue, nameof(constant));
    }

    public string Name => ReinforcementNames.Reinforce;

    // No KL term: the reference is never read.
    public bool NeedsReference(IReadOnlyDictionary<string, string>? options = null) => false;

    public Tensor Loss(PolicyLossInput input, IReadOnlyDictionary<string, string>? options = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        Estimators.NoOptions($"policy loss {Name}", options);
        float entropy = _entropy;
        float value = _value;
        string normalize = _normalize;
        float constant = _constant;
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
/// KL_u is the estimator's (see klEstimator); a β above 0 refuses an input without a reference. Options (the constructor's,
/// after the loss's name: ppo or grpo): normalize (token, sequence, constant; ppo's token, grpo's sequence), clip (ε in
/// [0, 1]), clipHigh (ε_high, at least 0, default ε), kl (β, at least 0), klEstimator (k1, k2, k3; default k3), entropy
/// (c_H), value (c_V, at least 0; ppo's 0.5, grpo's 0: no value term), constant (at least 0).
/// </summary>
internal sealed class ClippedRatioLoss : IPolicyLoss
{
    private readonly string _normalize, _klEstimator;
    private readonly float _clip, _clipHigh, _kl, _entropy, _value, _constant;

    public ClippedRatioLoss(string name, string normalize = "token", float clip = 0.2f, float? clipHigh = null, float kl = 0f, string klEstimator = "k3",
        float entropy = 0f, float value = 0.5f, float constant = 0f)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        Name = name;
        _normalize = PolicyLossTerms.Normalization(normalize);
        _clip = Estimators.InRange(clip, 0f, 1f, nameof(clip));
        _clipHigh = clipHigh is { } high ? Estimators.InRange(high, 0f, float.MaxValue, nameof(clipHigh)) : _clip;
        _kl = Estimators.InRange(kl, 0f, float.MaxValue, nameof(kl));
        _klEstimator = Estimators.OneOf(klEstimator, nameof(klEstimator), "k1", "k2", "k3");
        _entropy = entropy;
        _value = Estimators.InRange(value, 0f, float.MaxValue, nameof(value));
        _constant = Estimators.InRange(constant, 0f, float.MaxValue, nameof(constant));
    }

    public string Name { get; }

    // The reference is read by the KL term only, weighted by kl (0 by default).
    public bool NeedsReference(IReadOnlyDictionary<string, string>? options = null) => _kl != 0f;

    public Tensor Loss(PolicyLossInput input, IReadOnlyDictionary<string, string>? options = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        Estimators.NoOptions($"policy loss {Name}", options);
        float clip = _clip;
        float clipHigh = _clipHigh;
        float kl = _kl;
        string estimator = _klEstimator;
        float entropy = _entropy;
        float value = _value;
        string normalize = _normalize;
        float constant = _constant;
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
