// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Reinforcement;

/// <summary>
/// What an <see cref="IPolicyLoss"/> receives for one mini-batch: the log-probability under the current weights of every
/// action unit taken (a token of a reply, a whole choice), the sequences after one another (an episode's actions, or a
/// reply's tokens), each in order; the same units under the policy that sampled them and under the reference; each unit's
/// advantage; a value head's predictions and targets when the policy has one; and the sizes of the whole optimizer step,
/// so a loss averaged over the step divides by <see cref="StepUnits"/> or <see cref="StepSequences"/>. Every tensor holds
/// one value per unit; only <see cref="LogProbabilities"/>, <see cref="Values"/> and <see cref="Entropies"/> are recorded
/// for the gradient.
/// </summary>
public sealed class PolicyLossInput
{
    private readonly Tensor? _behavior;
    private Tensor? _detached;

    /// <summary>The input for <paramref name="logProbabilities"/> divided into sequences of <paramref name="counts"/> units.</summary>
    /// <param name="logProbabilities">log p of each unit under the current weights [units], recorded.</param>
    /// <param name="counts">The units of each sequence, in order; they add up to the units.</param>
    /// <param name="advantages">The advantage of each unit [units] (an episode's or step's advantage repeated over its units).</param>
    /// <param name="behaviorLogProbabilities">log p of each unit under the policy that sampled it [units]; null when it is the current one (on-policy: every ratio is 1).</param>
    /// <param name="referenceLogProbabilities">log p of each unit under the reference policy [units] (for a KL penalty); null when there is none.</param>
    /// <param name="values">A value head's prediction for each unit [units], recorded; null without one (given with <paramref name="returns"/>).</param>
    /// <param name="returns">Each unit's return [units], the value head's target; null without one.</param>
    /// <param name="entropies">The entropy of the policy's distribution at each unit [units], recorded (for an entropy bonus); null when not computed.</param>
    /// <param name="stepUnits">Units of the whole optimizer step (0: this input's).</param>
    /// <param name="stepSequences">Sequences of the whole optimizer step (0: this input's).</param>
    /// <exception cref="ArgumentException">Counts that do not add up to the units, a tensor of another size, or values without returns (or returns without values).</exception>
    public PolicyLossInput(Tensor logProbabilities, IReadOnlyList<int> counts, Tensor advantages, Tensor? behaviorLogProbabilities = null,
        Tensor? referenceLogProbabilities = null, Tensor? values = null, Tensor? returns = null, Tensor? entropies = null, int stepUnits = 0, int stepSequences = 0)
    {
        ArgumentNullException.ThrowIfNull(logProbabilities);
        ArgumentNullException.ThrowIfNull(counts);
        ArgumentNullException.ThrowIfNull(advantages);
        long total = 0;
        foreach (int count in counts)
        {
            if (count < 0)
            {
                throw new ArgumentException("A sequence's unit count is not negative.", nameof(counts));
            }

            total += count;
        }

        int units = logProbabilities.Size;
        if (counts.Count == 0 || total != units)
        {
            throw new ArgumentException($"The unit counts ({total} in {counts.Count} sequences) do not match the {units} log-probabilities (at least one sequence).", nameof(counts));
        }

        if ((values is null) != (returns is null))
        {
            throw new ArgumentException("A value head's predictions come with their targets: give values and returns together, or neither.", values is null ? nameof(values) : nameof(returns));
        }

        SameSize(advantages, units, nameof(advantages));
        SameSize(behaviorLogProbabilities, units, nameof(behaviorLogProbabilities));
        SameSize(referenceLogProbabilities, units, nameof(referenceLogProbabilities));
        SameSize(values, units, nameof(values));
        SameSize(returns, units, nameof(returns));
        SameSize(entropies, units, nameof(entropies));
        ArgumentOutOfRangeException.ThrowIfNegative(stepUnits);
        ArgumentOutOfRangeException.ThrowIfNegative(stepSequences);
        LogProbabilities = logProbabilities;
        Counts = counts;
        Advantages = advantages;
        _behavior = behaviorLogProbabilities;
        ReferenceLogProbabilities = referenceLogProbabilities;
        Values = values;
        Returns = returns;
        Entropies = entropies;
        StepUnits = stepUnits > 0 ? stepUnits : units;
        StepSequences = stepSequences > 0 ? stepSequences : counts.Count;
    }

    /// <summary>log p of each unit under the current weights [units]: sequence 0's units, then sequence 1's, and so on. Recorded.</summary>
    public Tensor LogProbabilities { get; }

    /// <summary>The units of each sequence, in order.</summary>
    public IReadOnlyList<int> Counts { get; }

    /// <summary>The advantage of each unit [units]. Not recorded.</summary>
    public Tensor Advantages { get; }

    /// <summary>
    /// log p of each unit under the policy that sampled it [units], not recorded: the denominator of a ratio. When none was
    /// given (on-policy), <see cref="LogProbabilities"/> detached, so every ratio is 1 and its gradient is that of log p.
    /// </summary>
    public Tensor BehaviorLogProbabilities => _behavior ?? (_detached ??= LogProbabilities.Detach());

    /// <summary>Whether behaviour log-probabilities were given (off the current weights), rather than the current ones detached.</summary>
    public bool HasBehavior => _behavior is not null;

    /// <summary>log p of each unit under the reference policy [units], not recorded; null when there is none.</summary>
    public Tensor? ReferenceLogProbabilities { get; }

    /// <summary>A value head's prediction for each unit [units], recorded; null without one.</summary>
    public Tensor? Values { get; }

    /// <summary>Each unit's return [units], the value head's target, not recorded; null without one.</summary>
    public Tensor? Returns { get; }

    /// <summary>The entropy of the policy at each unit [units], recorded; null when not computed.</summary>
    public Tensor? Entropies { get; }

    /// <summary>The units of this input.</summary>
    public int Units => LogProbabilities.Size;

    /// <summary>The sequences of this input.</summary>
    public int Sequences => Counts.Count;

    /// <summary>Units of the whole optimizer step (this input's unless the step is split into several).</summary>
    public int StepUnits { get; }

    /// <summary>Sequences of the whole optimizer step.</summary>
    public int StepSequences { get; }

    private static void SameSize(Tensor? tensor, int units, string name)
    {
        if (tensor is not null && tensor.Size != units)
        {
            throw new ArgumentException($"{name} holds {tensor.Size} values for {units} units: one per unit.", name);
        }
    }
}

/// <summary>REINFORCE, PPO's clipped ratio with a value loss and an entropy bonus, GRPO's clipped ratio with a KL
/// penalty to the reference. It is a composition of tensor operations on a <see cref="PolicyLossInput"/>, so autograd
/// carries its gradient to the log-probabilities (and values) the policy computed; sums run in a fixed order. Its
/// option keys (<c>Keys</c>) are the ones <see cref="Loss"/> takes: clip, KL weight and
/// estimator, value and entropy weights, normalization. Thread-safe: it keeps no state between calls.
/// </summary>
public interface IPolicyLoss
{
    /// <summary>The mini-batch's loss, one element, finite; its gradient flows to <see cref="PolicyLossInput.LogProbabilities"/>.</summary>
    /// <param name="input">The log-probabilities, advantages and the rest, for the mini-batch.</param>
    /// <param name="options">The loss's options by key (<c>Keys</c>), values as text; null for
    /// its defaults.</param>
    /// <exception cref="ArgumentException">An option the loss does not take (the message names it and <c>Keys</c>), a bad value, or an input without what the loss needs (a reference,
    /// values).</exception>
    Tensor Loss(PolicyLossInput input, IReadOnlyDictionary<string, string>? options = null);

    /// <summary>
    /// Whether <see cref="Loss"/> with <paramref name="options"/> reads <see cref="PolicyLossInput.ReferenceLogProbabilities"/>
    /// (a KL penalty to the reference weighted above 0), so a policy computes the reference's log-probabilities only for a
    /// loss that reads them (a pass of the reference model per rollout). True unless the loss says otherwise, so a loss
    /// that does not answer is always given them.
    /// </summary>
    /// <param name="options">The options <see cref="Loss"/> will be called with; null for its defaults.</param>
    bool NeedsReference(IReadOnlyDictionary<string, string>? options = null) => true;
}

