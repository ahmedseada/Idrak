// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Reinforcement.Abstractions;

namespace Idrak.Reinforcement;

/// <summary>
/// DQN's loss (<see cref="QLosses.Dqn"/>; Mnih et al. 2015, Nature 518, "Human-level control through deep reinforcement
/// learning", Algorithm 1): the target y_i = r_i + d_i · max_a Q'(s'_i, a), with double Q-learning (van Hasselt, Guez and
/// Silver 2016, equation 4) y_i = r_i + d_i · Q'(s'_i, argmax_a Q(s'_i, a)) (the first of equal values on a tie), computed on
/// the host in double precision and rounded once; then the Huber loss of δ (option <c>huber</c>, 1 by default: the paper's
/// error clipped to [−1, 1], PyTorch's smooth L1 of β 1) of Q(s_i, a_i) − y_i, ½ x² for |x| ≤ δ and δ (|x| − ½ δ) beyond, or
/// ½ x² for δ = 0, averaged over the batch: L = (1 / B) Σ_i w_i ℓ_i, one reduction in transition order (rule 86). Q(s_i, a_i)
/// is taken from the recorded values by a one-hot mask, so the gradient reaches only the actions taken.
/// </summary>
internal sealed class DqnLoss : IQLoss
{
    public string Name => QLosses.Dqn;

    public string Summary => "DQN (Mnih et al. 2015): a Huber loss of Q(s, a) against r + γ max Q'(s', ·) of a target network; double Q-learning (van Hasselt et al. 2016) as an option";

    public IReadOnlyCollection<string> Keys { get; } = [QLosses.HuberKey, QLosses.DoubleKey];

    public bool NeedsNextValues(IReadOnlyDictionary<string, string>? options = null) => new OptionReader($"Q-learning loss {Name}", Keys, options).Bool(QLosses.DoubleKey, false);

    public Tensor Loss(QLossInput input, IReadOnlyDictionary<string, string>? options = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        var read = new OptionReader($"Q-learning loss {Name}", Keys, options);
        float delta = read.Float(QLosses.HuberKey, 1f, min: 0f);
        bool doubleQ = read.Bool(QLosses.DoubleKey, false);
        if (doubleQ && input.NextValues.IsEmpty)
        {
            throw new ArgumentException($"The Q-learning loss {Name} with '{QLosses.DoubleKey}' true chooses the next action by the online network: the input has no next values of it (QLossInput.NextValues).");
        }

        int batch = input.Batch, actions = input.ActionCount;
        var device = input.Values.Device;
        float[] targets = new float[batch], chosen = new float[batch * actions];
        var rewards = input.Rewards.Span;
        var discounts = input.Discounts.Span;
        var next = input.NextTargetValues.Span;
        var online = input.NextValues.Span;
        var taken = input.Actions.Span;
        for (int i = 0; i < batch; i++)
        {
            double bootstrap = 0;
            if (discounts[i] != 0f)
            {
                var row = next.Slice(i * actions, actions);
                bootstrap = doubleQ ? row[ArgMax(online.Slice(i * actions, actions))] : row[ArgMax(row)];
            }

            targets[i] = (float)(rewards[i] + (double)discounts[i] * bootstrap);
            chosen[i * actions + taken[i]] = 1f;
        }

        var q = (input.Values * Tensor.From(chosen, [batch, actions], device)).Sum(1);
        var error = q - Tensor.From(targets, [batch], device);
        Tensor perTransition;
        if (delta > 0f)
        {
            var size = error.Abs();
            var inside = size.Minimum(delta);
            perTransition = inside.Square() * 0.5f + (size - inside) * delta;
        }
        else
        {
            perTransition = error.Square() * 0.5f;
        }

        if (input.Weights.IsEmpty)
        {
            return perTransition.Mean();
        }

        return (perTransition * Tensor.From(input.Weights.Span, [batch], device)).Sum() * (1f / batch);
    }

    // The first index of the largest value.
    internal static int ArgMax(ReadOnlySpan<float> values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > values[best])
            {
                best = i;
            }
        }

        return best;
    }
}
