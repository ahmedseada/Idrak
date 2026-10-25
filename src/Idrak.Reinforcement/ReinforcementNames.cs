// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Reinforcement;

/// <summary>
/// The names this package registers: the advantage estimators (<c>AdvantageEstimators</c>) and the policy losses
/// (<c>PolicyLosses</c>, a trainer's algorithm), and the keys of their options.
/// </summary>
public static class ReinforcementNames
{
    /// <summary>Generalized advantage estimation (Schulman et al. 2015, arXiv:1506.02438), with a critic.</summary>
    public const string Gae = "gae";

    /// <summary>Group-relative advantages (GRPO, Shao et al. 2024, arXiv:2402.03300): an episode's return against its group's.</summary>
    public const string Group = "group";

    /// <summary>Leave-one-out advantages (RLOO, Ahmadian et al. 2024, arXiv:2402.14740): an episode's return against the others' of its group.</summary>
    public const string LeaveOneOut = "leave-one-out";

    /// <summary>Discounted returns (REINFORCE, Williams 1992), less a baseline or not.</summary>
    public const string Returns = "returns";

    /// <summary>The REINFORCE policy gradient (Williams 1992): log-probabilities weighted by advantages.</summary>
    public const string Reinforce = "reinforce";

    /// <summary>Proximal policy optimization (Schulman et al. 2017, arXiv:1707.06347): a clipped ratio, a value loss, an entropy bonus.</summary>
    public const string Ppo = "ppo";

    /// <summary>Group relative policy optimization (Shao et al. 2024, arXiv:2402.03300): a clipped ratio and a KL penalty to the reference.</summary>
    public const string Grpo = "grpo";

    /// <summary>The discount γ of the estimators that discount (gae, returns).</summary>
    public const string GammaKey = "gamma";

    /// <summary>GAE's λ.</summary>
    public const string LambdaKey = "lambda";

    /// <summary>Whether gae and returns normalize the batch's advantages to mean 0 and spread 1 (true or false).</summary>
    public const string NormalizeAdvantagesKey = "normalize";

    /// <summary>The returns estimator's baseline: none, mean (the batch's mean return) or value (the critic's estimate).</summary>
    public const string BaselineKey = "baseline";

    /// <summary>Whether the group estimator divides by the group's spread (true or false).</summary>
    public const string ScaleKey = "scale";

    /// <summary>What the group estimator adds to the spread before dividing by it.</summary>
    public const string EpsilonKey = "epsilon";

    /// <summary>The ratio's clip ε of ppo and grpo: ratios outside [1 - ε, 1 + ε_high] stop the gradient.</summary>
    public const string ClipKey = "clip";

    /// <summary>The ratio's upper clip ε_high (DAPO's decoupled clip, Yu et al. 2025, arXiv:2503.14476); the clip when not given.</summary>
    public const string ClipHighKey = "clip-high";

    /// <summary>The weight of the KL penalty to the reference (β).</summary>
    public const string KlKey = "kl";

    /// <summary>The KL estimator: k1, k2 or k3 (Schulman, "Approximating KL divergence", 2020).</summary>
    public const string KlEstimatorKey = "kl-estimator";

    /// <summary>The weight of the entropy bonus.</summary>
    public const string EntropyKey = "entropy";

    /// <summary>The weight of the value head's squared error.</summary>
    public const string ValueKey = "value";

    /// <summary>How a loss divides the step's sum: token (by its units), sequence (each sequence's mean, then by the sequences) or constant.</summary>
    public const string NormalizeKey = "normalize";

    /// <summary>The divisor of the step's sum when the loss normalizes by a constant (Dr. GRPO, Liu et al. 2025, arXiv:2503.20783).</summary>
    public const string ConstantKey = "constant";

    /// <summary>The library's estimators, in their registration order.</summary>
    public static IReadOnlyList<string> Estimators { get; } = [Gae, Group, LeaveOneOut, Returns];

    /// <summary>The library's policy losses, in their registration order.</summary>
    public static IReadOnlyList<string> Algorithms { get; } = [Reinforce, Ppo, Grpo];
}

// Every draw of a run from its seed: a stream number and indices mixed by SplitMix64 (Steele et al. 2014), so a run's
// random state is its seed and where it is, and an episode's draws never depend on what ran before it or on threads.
internal static class Seeds
{
    public const int Episodes = 1, Sampling = 2, Shuffle = 3, Evaluation = 4, EvaluationSampling = 5, Creation = 6, FinalValues = 7;

    public static int Derive(int seed, int stream, long first = 0, long second = 0)
    {
        ulong x = Mix((uint)seed + 0x9E37_79B9_7F4A_7C15UL * (uint)stream);
        x = Mix(x ^ (ulong)first);
        x = Mix(x ^ (ulong)second);
        return (int)(x & 0x7FFF_FFFF);
    }

    private static ulong Mix(ulong z)
    {
        z += 0x9E37_79B9_7F4A_7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58_476D_1CE4_E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D0_49BB_1331_11EBUL;
        return z ^ (z >> 31);
    }
}
