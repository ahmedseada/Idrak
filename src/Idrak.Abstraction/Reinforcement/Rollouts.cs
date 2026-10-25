// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;

namespace Idrak.Abstraction.Reinforcement;

/// <summary>
/// One episode as it was played: the observations, the actions and their records, the rewards, and how it ended. Made by
/// a rollout from the steps of an <see cref="IEnvironment{TObservation, TAction}"/> and the
/// <see cref="PolicyActions{TAction}"/> that chose them; immutable.
/// </summary>
public sealed class Trajectory<TObservation, TAction>
{
    private readonly float[] _rewards;

    /// <summary>An episode of <paramref name="actions"/>' steps.</summary>
    /// <param name="index">The episode's number in its run: batches order episodes by it, so the order threads finished in changes nothing.</param>
    /// <param name="seed">The seed it was reset with (<see cref="IEnvironment{TObservation, TAction}.Reset"/>).</param>
    /// <param name="observations">The first observation, then the one after each step: one more than the steps (the last is what a truncated episode bootstraps from).</param>
    /// <param name="actions">The action of each step; at least one.</param>
    /// <param name="records">The record of each action (<see cref="PolicyActions{TAction}.Records"/>).</param>
    /// <param name="rewards">The reward of each step, finite.</param>
    /// <param name="terminated">The last step ended the episode of itself (<see cref="EnvironmentStep{TObservation}.Terminated"/>).</param>
    /// <param name="truncated">The last step cut it by a limit (<see cref="EnvironmentStep{TObservation}.Truncated"/>).</param>
    /// <param name="group">
    /// The episodes compared with each other share a group (a prompt answered several times, for group-relative
    /// advantages); null puts the episode in a group of its own, numbered as <paramref name="index"/>.
    /// </param>
    /// <param name="finalValue">The policy's value estimate of the last observation (what a truncated episode bootstraps from); NaN when it has no critic.</param>
    /// <param name="info">What the environment reported at the last step; null for nothing.</param>
    /// <exception cref="ArgumentException">No step, lists of other lengths than the steps say, or a reward that is not finite.</exception>
    public Trajectory(int index, int seed, IReadOnlyList<TObservation> observations, IReadOnlyList<TAction> actions, IReadOnlyList<ActionRecord> records,
        IReadOnlyList<float> rewards, bool terminated, bool truncated, int? group = null, float finalValue = float.NaN, IReadOnlyDictionary<string, string>? info = null)
    {
        ArgumentNullException.ThrowIfNull(observations);
        ArgumentNullException.ThrowIfNull(actions);
        ArgumentNullException.ThrowIfNull(records);
        ArgumentNullException.ThrowIfNull(rewards);
        int steps = actions.Count;
        if (steps == 0 || observations.Count != steps + 1 || records.Count != steps || rewards.Count != steps)
        {
            throw new ArgumentException($"An episode of {steps} steps has {steps + 1} observations (the first, then one per step), a record and a reward per step: "
                                        + $"{observations.Count} observations, {records.Count} records and {rewards.Count} rewards given; at least one step.");
        }

        _rewards = [.. rewards];
        int bad = FirstNotFinite(_rewards);
        if (bad >= 0)
        {
            throw new ArgumentException(FormattableString.Invariant($"The reward of step {bad} is {_rewards[bad]}: rewards are finite."), nameof(rewards));
        }

        int units = 0;
        foreach (var record in records)
        {
            ArgumentNullException.ThrowIfNull(record, nameof(records));
            units += record.Units;
        }

        Index = index;
        Seed = seed;
        Group = group ?? index;
        Observations = observations;
        Actions = actions;
        Records = records;
        Terminated = terminated;
        Truncated = truncated;
        FinalValue = finalValue;
        Info = info;
        Units = units;
    }

    /// <summary>The episode's number in its run.</summary>
    public int Index { get; }

    /// <summary>The seed it was reset with.</summary>
    public int Seed { get; }

    /// <summary>Its group: the episodes compared with each other (the episode's own number when it was given none).</summary>
    public int Group { get; }

    /// <summary>The first observation, then the one after each step.</summary>
    public IReadOnlyList<TObservation> Observations { get; }

    /// <summary>The action of each step.</summary>
    public IReadOnlyList<TAction> Actions { get; }

    /// <summary>The record of each action.</summary>
    public IReadOnlyList<ActionRecord> Records { get; }

    /// <summary>The reward of each step.</summary>
    public ReadOnlySpan<float> Rewards => _rewards;

    /// <summary>The last step ended the episode of itself.</summary>
    public bool Terminated { get; }

    /// <summary>The last step cut it by a limit.</summary>
    public bool Truncated { get; }

    /// <summary>The value estimate of the last observation, or NaN.</summary>
    public float FinalValue { get; }

    /// <summary>What the environment reported at the last step, or null.</summary>
    public IReadOnlyDictionary<string, string>? Info { get; }

    /// <summary>The steps of the episode.</summary>
    public int Steps => _rewards.Length;

    /// <summary>The action units of the episode (every record's <see cref="ActionRecord.Units"/>).</summary>
    public int Units { get; }

    /// <summary>The sum of the rewards, added in step order (in double precision, then rounded once).</summary>
    public float Return
    {
        get
        {
            double sum = 0;
            foreach (float reward in _rewards)
            {
                sum += reward;
            }

            return (float)sum;
        }
    }

    private static int FirstNotFinite(ReadOnlySpan<float> values)
    {
        for (int i = 0; i < values.Length; i++)
        {
            if (!float.IsFinite(values[i]))
            {
                return i;
            }
        }

        return -1;
    }
}

/// <summary>
/// The episodes of one rollout, laid out flat for an <see cref="IAdvantageEstimator"/> and a policy's update: the steps of
/// every episode one after another, in episode-number order (<see cref="Trajectory{TObservation, TAction}.Index"/>, so the
/// order threads finished in changes nothing), with the rewards, the value estimates, and room for the advantages and
/// returns. Its arrays are rented from the shared pool and returned by <see cref="Dispose"/>: a trainer makes a batch per
/// iteration without the memory growing. Not thread-safe while <see cref="Advantages"/> and <see cref="Returns"/> are
/// written; reading them from several threads afterwards is safe.
/// </summary>
public sealed class RolloutBatch<TObservation, TAction> : IDisposable
{
    // Floats: rewards, values, advantages, returns ([steps] each), then the final values [episodes]. Integers: the
    // episodes' lengths, groups and first steps ([episodes] each).
    private float[]? _floats;
    private int[]? _integers;
    private bool[]? _truncated;

    /// <summary>A batch of <paramref name="trajectories"/>, ordered by their numbers; advantages and returns start at 0.</summary>
    /// <exception cref="ArgumentException">
    /// No episode, two episodes of one number, or (when every action has a value estimate) a truncated episode without
    /// the value of its last observation to bootstrap from.
    /// </exception>
    public RolloutBatch(IEnumerable<Trajectory<TObservation, TAction>> trajectories)
    {
        ArgumentNullException.ThrowIfNull(trajectories);
        Trajectory<TObservation, TAction>[] sorted = [.. trajectories.OrderBy(t => t?.Index ?? throw new ArgumentException("A rollout batch holds no null episode.", nameof(trajectories)))];
        if (sorted.Length == 0)
        {
            throw new ArgumentException("A rollout batch holds at least one episode.", nameof(trajectories));
        }

        long steps = 0;
        bool values = true;
        for (int e = 0; e < sorted.Length; e++)
        {
            if (e > 0 && sorted[e].Index == sorted[e - 1].Index)
            {
                throw new ArgumentException($"Two episodes are numbered {sorted[e].Index}: a batch numbers each episode once.", nameof(trajectories));
            }

            steps += sorted[e].Steps;
            foreach (var record in sorted[e].Records)
            {
                values &= record.HasValue;
            }
        }

        if (4 * steps + sorted.Length > Array.MaxLength)
        {
            throw new ArgumentException($"{steps} steps are more than a rollout batch holds.", nameof(trajectories));
        }

        Trajectories = sorted;
        Episodes = sorted.Length;
        Steps = (int)steps;
        HasValues = values;
        _floats = ArrayPool<float>.Shared.Rent(4 * Steps + Episodes);
        _integers = ArrayPool<int>.Shared.Rent(3 * Episodes);
        _truncated = ArrayPool<bool>.Shared.Rent(Episodes);
        try
        {
            Fill(sorted);
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>The episodes, in number order.</summary>
    public IReadOnlyList<Trajectory<TObservation, TAction>> Trajectories { get; }

    /// <summary>The episodes in the batch.</summary>
    public int Episodes { get; }

    /// <summary>The steps of every episode together.</summary>
    public int Steps { get; }

    /// <summary>Whether every action has a value estimate (a policy with a critic): <see cref="Values"/> and <see cref="FinalValues"/> are then filled, else empty.</summary>
    public bool HasValues { get; }

    /// <summary>The reward of each step [<see cref="Steps"/>], episodes one after another.</summary>
    public ReadOnlySpan<float> Rewards => Floats().AsSpan(0, Steps);

    /// <summary>The value estimate of each step's observation [<see cref="Steps"/>]; empty when <see cref="HasValues"/> is false.</summary>
    public ReadOnlySpan<float> Values => HasValues ? Floats().AsSpan(Steps, Steps) : default;

    /// <summary>The advantage of each step [<see cref="Steps"/>], written by <see cref="Estimate"/> (0 until then).</summary>
    public Span<float> Advantages => Floats().AsSpan(2 * Steps, Steps);

    /// <summary>The return of each step [<see cref="Steps"/>] (a value head's target), written by <see cref="Estimate"/> (0 until then).</summary>
    public Span<float> Returns => Floats().AsSpan(3 * Steps, Steps);

    /// <summary>The value estimate of each episode's last observation [<see cref="Episodes"/>] (0 for one that terminated); empty when <see cref="HasValues"/> is false.</summary>
    public ReadOnlySpan<float> FinalValues => HasValues ? Floats().AsSpan(4 * Steps, Episodes) : default;

    /// <summary>The steps of each episode [<see cref="Episodes"/>].</summary>
    public ReadOnlySpan<int> Lengths => Integers().AsSpan(0, Episodes);

    /// <summary>The group of each episode [<see cref="Episodes"/>].</summary>
    public ReadOnlySpan<int> Groups => Integers().AsSpan(Episodes, Episodes);

    /// <summary>The first step of each episode in the flat arrays [<see cref="Episodes"/>].</summary>
    public ReadOnlySpan<int> Offsets => Integers().AsSpan(2 * Episodes, Episodes);

    /// <summary>Whether each episode was cut by a limit rather than ended of itself [<see cref="Episodes"/>] (one both truncated and terminated counts as terminated).</summary>
    public ReadOnlySpan<bool> Truncated => (_truncated ?? throw Disposed()).AsSpan(0, Episodes);

    /// <summary>What an <see cref="IAdvantageEstimator"/> reads of the batch.</summary>
    public AdvantageInput Input => new(Rewards, Values, Lengths, Truncated, FinalValues, Groups);

    /// <summary>Writes <see cref="Advantages"/> and <see cref="Returns"/> with <paramref name="estimator"/>.</summary>
    /// <param name="estimator">The estimator (an <see cref="AdvantageEstimators"/> entry).</param>
    /// <param name="options">Its options by key (<see cref="IConfigurableKind.Keys"/>); null for its defaults.</param>
    public void Estimate(IAdvantageEstimator estimator, IReadOnlyDictionary<string, string>? options = null)
    {
        ArgumentNullException.ThrowIfNull(estimator);
        estimator.Estimate(Input, Advantages, Returns, options);
    }

    /// <summary>Returns the batch's arrays to the pool; the batch cannot be read afterwards. More than once is harmless.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _floats, null) is { } floats)
        {
            ArrayPool<float>.Shared.Return(floats);
        }

        if (Interlocked.Exchange(ref _integers, null) is { } integers)
        {
            ArrayPool<int>.Shared.Return(integers);
        }

        if (Interlocked.Exchange(ref _truncated, null) is { } truncated)
        {
            ArrayPool<bool>.Shared.Return(truncated);
        }
    }

    private void Fill(Trajectory<TObservation, TAction>[] sorted)
    {
        var floats = _floats!.AsSpan();
        var rewards = floats.Slice(0, Steps);
        var values = floats.Slice(Steps, Steps);
        floats.Slice(2 * Steps, 2 * Steps).Clear();                                   // advantages and returns start at 0
        var finals = floats.Slice(4 * Steps, Episodes);
        var integers = _integers!.AsSpan();
        int at = 0;
        for (int e = 0; e < sorted.Length; e++)
        {
            var trajectory = sorted[e];
            trajectory.Rewards.CopyTo(rewards[at..]);
            if (HasValues)
            {
                for (int s = 0; s < trajectory.Steps; s++)
                {
                    values[at + s] = trajectory.Records[s].Value;
                }
            }

            bool truncated = trajectory.Truncated && !trajectory.Terminated;
            if (HasValues && truncated && !float.IsFinite(trajectory.FinalValue))
            {
                throw new ArgumentException($"Episode {trajectory.Index} was truncated without the value of its last observation (FinalValue) to bootstrap from, "
                                            + "while its actions have value estimates.");
            }

            finals[e] = truncated && HasValues ? trajectory.FinalValue : 0f;
            integers[e] = trajectory.Steps;
            integers[Episodes + e] = trajectory.Group;
            integers[2 * Episodes + e] = at;
            _truncated![e] = truncated;
            at += trajectory.Steps;
        }
    }

    private float[] Floats() => _floats ?? throw Disposed();

    private int[] Integers() => _integers ?? throw Disposed();

    private static ObjectDisposedException Disposed() => new("RolloutBatch", "The rollout batch was disposed: its arrays are back in the pool.");
}
