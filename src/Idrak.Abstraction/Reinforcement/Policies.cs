// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Reinforcement;

/// <summary>How a policy chooses its actions (<see cref="IPolicy{TObservation, TAction}.ActAsync"/>).</summary>
public sealed record ActionSampling
{
    /// <summary>Sampling at temperature 1 from seed 0: what a rollout collects with unless it says otherwise.</summary>
    public static ActionSampling Default { get; } = new();

    /// <summary>The most likely action instead of a sampled one (an evaluation episode); <see cref="Temperature"/> and <see cref="Seed"/> are then unused.</summary>
    public bool Greedy { get; init; }

    /// <summary>Divides the policy's logits (or widens its spread) before sampling: 1 samples the policy as it is. Positive and finite.</summary>
    public float Temperature
    {
        get;
        init => field = value > 0 && float.IsFinite(value) ? value : throw new ArgumentOutOfRangeException(nameof(Temperature), value, "The temperature is positive and finite.");
    } = 1f;

    /// <summary>The draws' seed: the same observations, seed and weights give the same actions.</summary>
    public int Seed { get; init; }
}

/// <summary>
/// What an update needs to know about one action a policy took, recorded when it acted
/// (<see cref="PolicyActions{TAction}.Records"/>): the ids it sampled, their log-probabilities under the policy that
/// sampled them (the behaviour policy), and its value estimate. A language model's reply is many units (a token each); a
/// choice or a box action is one. An update trains on exactly these ids, never on a re-encoding of the action.
/// </summary>
public sealed class ActionRecord
{
    /// <summary>A record of an action of <paramref name="logProbabilities"/>' units.</summary>
    /// <param name="tokens">The ids sampled, one per unit (a reply's token ids, a choice's index); empty when the action is not made of ids (a box).</param>
    /// <param name="logProbabilities">log p of each unit under the behaviour policy (one for an action that is not made of ids); at least one.</param>
    /// <param name="value">The policy's estimate of the observation's value (a critic's output); NaN when it has no critic.</param>
    /// <param name="context">The ids the policy conditioned on (a reply's prompt as rendered), when the update needs them; null otherwise.</param>
    /// <exception cref="ArgumentException">No log-probability, or ids of another count than the log-probabilities.</exception>
    public ActionRecord(int[] tokens, float[] logProbabilities, float value = float.NaN, int[]? context = null)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        ArgumentNullException.ThrowIfNull(logProbabilities);
        if (logProbabilities.Length == 0 || (tokens.Length > 0 && tokens.Length != logProbabilities.Length))
        {
            throw new ArgumentException($"An action record has a log-probability per sampled id (or one for an action not made of ids): {tokens.Length} ids, {logProbabilities.Length} log-probabilities.");
        }

        Tokens = tokens;
        LogProbabilities = logProbabilities;
        Value = value;
        Context = context;
    }

    /// <summary>The ids sampled, one per unit; empty when the action is not made of ids.</summary>
    public int[] Tokens { get; }

    /// <summary>log p of each unit under the behaviour policy.</summary>
    public float[] LogProbabilities { get; }

    /// <summary>The policy's value estimate for the observation acted on; NaN when it has none.</summary>
    public float Value { get; }

    /// <summary>The ids the policy conditioned on, or null.</summary>
    public int[]? Context { get; }

    /// <summary>The units of the action (<see cref="LogProbabilities"/>' count): what a policy loss sums over.</summary>
    public int Units => LogProbabilities.Length;

    /// <summary>Whether <see cref="Value"/> is a finite estimate.</summary>
    public bool HasValue => float.IsFinite(Value);
}

/// <summary>What a policy chose for a batch of observations: an action each and its record, in the observations' order.</summary>
public sealed class PolicyActions<TAction>
{
    /// <summary>The actions and their records, one each, in the observations' order.</summary>
    /// <exception cref="ArgumentException">Not as many records as actions.</exception>
    public PolicyActions(IReadOnlyList<TAction> actions, IReadOnlyList<ActionRecord> records)
    {
        ArgumentNullException.ThrowIfNull(actions);
        ArgumentNullException.ThrowIfNull(records);
        if (actions.Count != records.Count)
        {
            throw new ArgumentException($"A record per action: {actions.Count} actions, {records.Count} records.");
        }

        Actions = actions;
        Records = records;
    }

    /// <summary>The action for each observation.</summary>
    public IReadOnlyList<TAction> Actions { get; }

    /// <summary>What the update needs of each action.</summary>
    public IReadOnlyList<ActionRecord> Records { get; }
}

/// <summary>What one update of a policy did (<see cref="IPolicy{TObservation, TAction}.Update"/>).</summary>
/// <param name="Loss">The policy loss of the update (its mean over the mini-batches when the policy split it).</param>
/// <param name="Episodes">The episodes trained on.</param>
/// <param name="Units">The action units trained on (tokens, actions).</param>
/// <param name="Metrics">What else the policy measured, by name (KL to the reference, the share of clipped ratios, entropy, gradient norm); null for nothing.</param>
public sealed record UpdateStats(float Loss, int Episodes, int Units, IReadOnlyDictionary<string, double>? Metrics = null);

/// <summary>
/// A policy: what chooses an agent's actions from its observations, and learns from the episodes it played. A module
/// policy over a network (Idrak.Reinforcement) and a language-model policy over a chat model (Idrak.Text) implement it; a
/// trainer collects episodes with <see cref="ActAsync"/>, estimates advantages (<see cref="IAdvantageEstimator"/>), then
/// calls <see cref="Update"/> with a policy loss (<see cref="IPolicyLoss"/>) for each mini-batch. A policy holds its
/// weights, optimizer and device memory, released by <see cref="IDisposable.Dispose"/>. Not thread-safe: one trainer
/// drives it.
/// </summary>
/// <typeparam name="TObservation">The environment's observations.</typeparam>
/// <typeparam name="TAction">The environment's actions.</typeparam>
public interface IPolicy<TObservation, TAction> : IDisposable
{
    /// <summary>
    /// An action for each of <paramref name="observations"/> (one per episode under way), with what the update will need
    /// of it. The same observations, sampling and weights give the same actions.
    /// </summary>
    /// <param name="observations">The observations to act on, one per episode.</param>
    /// <param name="sampling">Greedy or sampled, the temperature and the seed.</param>
    /// <param name="cancellationToken">Stops the call.</param>
    ValueTask<PolicyActions<TAction>> ActAsync(IReadOnlyList<TObservation> observations, ActionSampling sampling, CancellationToken cancellationToken = default);

    /// <summary>
    /// One optimizer step on the episodes <paramref name="episodes"/> of <paramref name="batch"/> (all when null), whose
    /// advantages and returns are estimated: the log-probabilities of the actions taken under the current weights
    /// (and the reference's, when <paramref name="loss"/> asks), given to <paramref name="loss"/>, its gradient applied.
    /// </summary>
    /// <param name="batch">The episodes played, with <see cref="RolloutBatch{TObservation, TAction}.Advantages"/> and <see cref="RolloutBatch{TObservation, TAction}.Returns"/> filled.</param>
    /// <param name="loss">The policy loss.</param>
    /// <param name="episodes">The mini-batch: indices into <see cref="RolloutBatch{TObservation, TAction}.Trajectories"/>; null for the whole batch.</param>
    /// <param name="lossOptions">The loss's options by key (<see cref="IPolicyLoss.Keys"/>), null for its defaults.</param>
    UpdateStats Update(RolloutBatch<TObservation, TAction> batch, IPolicyLoss loss, IReadOnlyList<int>? episodes = null, IReadOnlyDictionary<string, string>? lossOptions = null);

    /// <summary>Saves what the policy learned (its weights or adapters, and what resuming needs) to <paramref name="path"/>.</summary>
    void Save(string path);

    /// <summary>Loads what <see cref="Save"/> wrote at <paramref name="path"/>.</summary>
    void Load(string path);
}
