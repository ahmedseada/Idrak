// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Abstraction.Reinforcement;

namespace Idrak.Reinforcement;

/// <summary>
/// Where an episode of a rollout starts (<see cref="RolloutCollector{TObservation, TAction}.CollectAsync"/>).
/// </summary>
/// <param name="Index">The episode's number in its run: the batch orders episodes by it.</param>
/// <param name="Seed">The seed its environment is reset with: episodes of one group share it (the same situation answered several times).</param>
/// <param name="Group">The episodes compared with each other (group-relative advantages) share a group.</param>
public readonly record struct EpisodeStart(int Index, int Seed, int Group);

/// <summary>
/// Plays a rollout's episodes, all of them at once: each episode in an environment of its own (made once from the kind and
/// reset for every episode, so a run keeps as many environments as its largest rollout), the policy asked for the actions of
/// every episode under way in one call (<see cref="IPolicy{TObservation, TAction}.ActAsync"/>), then those environments
/// stepped, one after another or <see cref="Environments"/> at once. What it records lands in scratch lists per episode slot
/// that are cleared and reused, never grown past the longest episode, so collecting allocates only the trajectories it
/// returns (and what the environments and the policy allocate).
/// <para>
/// <b>Reproducible:</b> the episodes under way are acted on in their order, each call's draws come from the sampling seed
/// and the call's number, and the batch is sorted by episode number; how many environments step at once, and whether they
/// step in parallel, changes nothing in the result. Stepping in parallel pays only where a step costs more than spreading it
/// over threads (rule 79): with <see cref="ParallelSteps"/> null the collector times its first rounds both ways (a step's time
/// sequential against in parallel) and keeps the faster.
/// </para>
/// Not thread-safe: one trainer drives a collector. Dispose it to dispose its environments.
/// </summary>
/// <typeparam name="TObservation">The environments' observations.</typeparam>
/// <typeparam name="TAction">Their actions.</typeparam>
public sealed class RolloutCollector<TObservation, TAction> : IDisposable
{
    private const int TrialRounds = 4;                                      // two sequential, two parallel, after a first one to warm up

    private readonly IEnvironmentKind _kind;
    private readonly IReadOnlyDictionary<string, string>? _options;
    private readonly int _seed;
    private readonly List<IEnvironment<TObservation, TAction>> _environments = [];
    private readonly List<Slot> _slots = [];
    private readonly List<int> _active = [], _still = [], _pending = [];
    private readonly List<TObservation> _observations = [];
    private EnvironmentStep<TObservation>[] _steps = [];
    private int _trials;
    private long _sequentialTicks, _sequentialSteps, _parallelTicks, _parallelSteps;
    private bool? _measured;
    private bool _disposed;

    /// <summary>A collector of episodes in environments of <paramref name="kind"/>.</summary>
    /// <param name="kind">The environment kind; its observations and actions must be <typeparamref name="TObservation"/> and <typeparamref name="TAction"/>.</param>
    /// <param name="environmentOptions">The kind's options (<see cref="IEnvironmentKind.Keys"/>); null for its defaults.</param>
    /// <param name="seed">What each environment is made with (<see cref="IEnvironmentKind.Create"/>), mixed with its slot.</param>
    /// <param name="environments">The most environments stepped at once; 0 for <see cref="ComputeResources.MaxCpuThreads"/>. 1 steps them one after another.</param>
    /// <param name="parallelSteps">true always steps in parallel, false never; null measures which is faster.</param>
    /// <exception cref="ArgumentException">The kind observes or acts with other types.</exception>
    public RolloutCollector(IEnvironmentKind kind, IReadOnlyDictionary<string, string>? environmentOptions = null, int seed = 0, int environments = 0, bool? parallelSteps = null)
    {
        ArgumentNullException.ThrowIfNull(kind);
        ArgumentOutOfRangeException.ThrowIfNegative(environments);
        if (kind.ObservationType != typeof(TObservation) || kind.ActionType != typeof(TAction))
        {
            throw new ArgumentException($"Environment kind '{kind.Name}' observes {kind.ObservationType.Name} and acts with {kind.ActionType.Name}, "
                                        + $"not {typeof(TObservation).Name} and {typeof(TAction).Name}.", nameof(kind));
        }

        _kind = kind;
        _options = environmentOptions;
        _seed = seed;
        Environments = environments > 0 ? environments : ComputeResources.MaxCpuThreads;
        ParallelSteps = parallelSteps;
    }

    /// <summary>The most environments stepped at once.</summary>
    public int Environments { get; }

    /// <summary>Whether environments step in parallel as asked: true, false, or null for measured.</summary>
    public bool? ParallelSteps { get; }

    /// <summary>Whether the environments step in parallel: as asked, or as measured (null until the measurement is done).</summary>
    public bool? SteppingInParallel => Environments <= 1 ? false : ParallelSteps ?? _measured;

    /// <summary>The environments made so far (one per episode slot of the largest rollout).</summary>
    public int Created => _environments.Count;

    /// <summary>
    /// Plays the episodes <paramref name="episodes"/> to their ends (or <paramref name="maxSteps"/>) with
    /// <paramref name="policy"/>, and returns them as a batch in episode-number order (the caller disposes it). For an episode
    /// cut short, by its environment or by <paramref name="maxSteps"/>, whose actions have value estimates, the policy is asked
    /// for the value of its last observation to bootstrap from.
    /// </summary>
    /// <param name="policy">The policy that acts.</param>
    /// <param name="episodes">Where each episode starts; at least one.</param>
    /// <param name="sampling">Greedy or sampled, the temperature, and the seed each round's seed is drawn from.</param>
    /// <param name="maxSteps">The most steps an episode takes before it is cut (truncated); 0 for the environment's own limit.</param>
    /// <param name="cancellationToken">Stops the rollout (between rounds, and in the steps of environments that observe it).</param>
    public async ValueTask<RolloutBatch<TObservation, TAction>> CollectAsync(IPolicy<TObservation, TAction> policy, IReadOnlyList<EpisodeStart> episodes,
        ActionSampling sampling, int maxSteps = 0, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(policy);
        ArgumentNullException.ThrowIfNull(episodes);
        ArgumentNullException.ThrowIfNull(sampling);
        ArgumentOutOfRangeException.ThrowIfNegative(maxSteps);
        if (episodes.Count == 0)
        {
            throw new ArgumentException("A rollout plays at least one episode.", nameof(episodes));
        }

        Prepare(episodes.Count);
        var trajectories = new Trajectory<TObservation, TAction>[episodes.Count];
        _active.Clear();
        for (int k = 0; k < episodes.Count; k++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var slot = _slots[k];
            slot.Clear();
            slot.Current = _environments[k].Reset(episodes[k].Seed);
            slot.Observations.Add(slot.Current);
            _active.Add(k);
        }

        for (int round = 0; _active.Count > 0; round++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            _observations.Clear();
            foreach (int k in _active)
            {
                _observations.Add(_slots[k].Current);
            }

            var acted = await policy.ActAsync(_observations, sampling with { Seed = Seeds.Derive(sampling.Seed, Seeds.Sampling, round) }, cancellationToken).ConfigureAwait(false);
            if (acted.Actions.Count != _active.Count)
            {
                throw new InvalidOperationException($"The policy acted {acted.Actions.Count} times for {_active.Count} observations: an action per observation.");
            }

            await StepAsync(acted.Actions, cancellationToken).ConfigureAwait(false);

            // Record the round in episode order; the episodes that ended wait for their trajectories.
            _still.Clear();
            _pending.Clear();
            for (int i = 0; i < _active.Count; i++)
            {
                int k = _active[i];
                var slot = _slots[k];
                var step = _steps[i];
                slot.Actions.Add(acted.Actions[i]);
                slot.Records.Add(acted.Records[i]);
                slot.Rewards.Add(step.Reward);
                slot.Observations.Add(step.Observation);
                slot.Current = step.Observation;
                slot.Info = step.Info;
                bool cut = !step.Done && maxSteps > 0 && slot.Actions.Count >= maxSteps;
                if (step.Done || cut)
                {
                    slot.Terminated = step.Terminated;
                    slot.Truncated = !step.Terminated && (step.Truncated || cut);
                    _pending.Add(k);
                }
                else
                {
                    _still.Add(k);
                }
            }

            await FinalValuesAsync(policy, sampling, round, cancellationToken).ConfigureAwait(false);
            foreach (int k in _pending)
            {
                var slot = _slots[k];
                var start = episodes[k];
                trajectories[k] = new Trajectory<TObservation, TAction>(start.Index, start.Seed, slot.Observations.ToArray(), slot.Actions.ToArray(), slot.Records.ToArray(),
                    slot.Rewards, slot.Terminated, slot.Truncated, start.Group, slot.FinalValue, slot.Info);
            }

            _active.Clear();
            _active.AddRange(_still);
        }

        return new RolloutBatch<TObservation, TAction>(trajectories);
    }

    /// <summary>Disposes the environments the collector made; more than once is harmless.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        foreach (var environment in _environments)
        {
            environment.Dispose();
        }

        _environments.Clear();
    }

    // An environment and scratch lists for each episode slot, made once.
    private void Prepare(int slots)
    {
        while (_environments.Count < slots)
        {
            var made = _kind.Create(_options, Seeds.Derive(_seed, Seeds.Creation, _environments.Count));
            if (made is not IEnvironment<TObservation, TAction> typed)
            {
                made.Dispose();
                throw new InvalidOperationException($"Environment kind '{_kind.Name}' made a {made.GetType().Name}, not an IEnvironment<{typeof(TObservation).Name}, {typeof(TAction).Name}>.");
            }

            _environments.Add(typed);
            _slots.Add(new Slot());
        }

        if (_steps.Length < slots)
        {
            _steps = new EnvironmentStep<TObservation>[slots];
        }
    }

    // Steps the active environments with their actions into _steps, in order: one after another, or in parallel.
    private async ValueTask StepAsync(IReadOnlyList<TAction> actions, CancellationToken cancellationToken)
    {
        int count = _active.Count;
        bool? parallel = count < 2 || Environments <= 1 ? false : ParallelSteps ?? _measured;
        bool trial = parallel is null;
        if (trial)
        {
            parallel = _trials > 0 && _trials % 2 == 0;                     // round 0 warms up sequentially; then sequential, parallel, ...
        }

        long start = Stopwatch.GetTimestamp();
        if (parallel == true)
        {
            var options = new ParallelOptions { MaxDegreeOfParallelism = Environments, CancellationToken = cancellationToken };
            await Parallel.ForEachAsync(Enumerable.Range(0, count), options, async (i, token) =>
                _steps[i] = await _environments[_active[i]].StepAsync(actions[i], token).ConfigureAwait(false)).ConfigureAwait(false);
        }
        else
        {
            for (int i = 0; i < count; i++)
            {
                _steps[i] = await _environments[_active[i]].StepAsync(actions[i], cancellationToken).ConfigureAwait(false);
            }
        }

        if (trial)
        {
            long elapsed = Stopwatch.GetTimestamp() - start;
            if (_trials > 0)
            {
                if (parallel == true)
                {
                    _parallelTicks += elapsed;
                    _parallelSteps += count;
                }
                else
                {
                    _sequentialTicks += elapsed;
                    _sequentialSteps += count;
                }
            }

            if (++_trials > TrialRounds)
            {
                _measured = (double)_parallelTicks / _parallelSteps < (double)_sequentialTicks / _sequentialSteps;
            }
        }
    }

    // The value of the last observation of each episode that ended this round cut short while its actions carry values,
    // asked of the policy in one call; the others get NaN.
    private async ValueTask FinalValuesAsync(IPolicy<TObservation, TAction> policy, ActionSampling sampling, int round, CancellationToken cancellationToken)
    {
        _observations.Clear();
        foreach (int k in _pending)
        {
            var slot = _slots[k];
            slot.FinalValue = float.NaN;
            if (slot.Truncated && slot.Records[0].HasValue)
            {
                _observations.Add(slot.Current);
            }
        }

        if (_observations.Count == 0)
        {
            return;
        }

        var valued = await policy.ActAsync(_observations, sampling with { Seed = Seeds.Derive(sampling.Seed, Seeds.FinalValues, round) }, cancellationToken).ConfigureAwait(false);
        int next = 0;
        foreach (int k in _pending)
        {
            var slot = _slots[k];
            if (slot.Truncated && slot.Records[0].HasValue)
            {
                slot.FinalValue = valued.Records[next++].Value;
            }
        }
    }

    // One episode slot's scratch: cleared for each episode, its capacity kept.
    private sealed class Slot
    {
        public List<TObservation> Observations { get; } = [];

        public List<TAction> Actions { get; } = [];

        public List<ActionRecord> Records { get; } = [];

        public List<float> Rewards { get; } = [];

        public TObservation Current { get; set; } = default!;

        public bool Terminated { get; set; }

        public bool Truncated { get; set; }

        public float FinalValue { get; set; } = float.NaN;

        public IReadOnlyDictionary<string, string>? Info { get; set; }

        public void Clear()
        {
            Observations.Clear();
            Actions.Clear();
            Records.Clear();
            Rewards.Clear();
            Current = default!;
            Terminated = Truncated = false;
            FinalValue = float.NaN;
            Info = null;
        }
    }
}
