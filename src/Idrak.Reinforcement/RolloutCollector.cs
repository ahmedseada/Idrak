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
/// Plays a rollout's episodes, all of them at once: each episode in an environment of its own (made once from the kind for
/// its episode slot and reset for every episode, so a run keeps as many environments as its largest rollout), the policy
/// asked for the actions of every episode under way in one call (<see cref="IPolicy{TObservation, TAction}.ActAsync"/>) per
/// shard, then those environments stepped, one after another or <see cref="Environments"/> at once. What it records lands in
/// scratch lists per episode slot that are cleared and reused, never grown past the longest episode, so collecting
/// allocates only the trajectories it returns (and what the environments and the policy allocate, and the split of a
/// rollout of several shards).
/// <para>
/// <b>Reproducible:</b> the episodes under way are acted on in their order, each call's draws come from the sampling seed,
/// the call's round and its shard, and the batch is sorted by episode number; how many environments step at once, and
/// whether they step in parallel, changes nothing in the result. Stepping in parallel pays only where a step costs more than
/// spreading it over threads (rule 79): with <see cref="ParallelSteps"/> null the collector times its first rounds both ways
/// (a step's time sequential against in parallel) and keeps the faster.
/// </para>
/// <para>
/// <b>On a cluster</b> (Plan 27, R3b: a policy that is an <see cref="IRolloutRanks{TObservation, TAction}"/> of several
/// ranks, every rank running its own trainer and collector with the same settings): the rollout is cut into
/// <see cref="Shards"/> shards (<see cref="RolloutSpread"/>; one per rank by default), each rank plays the episodes of its
/// shards only, in environments of its own made for those slots alone, then the policy gathers every rank's episodes in rank
/// order and the batch, sorted by number, is the same on every rank. A shard's episodes act together in one call, so a
/// seeded run with the same shard count gives the same episodes on any number of ranks (one process included).
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
    private readonly List<IEnvironment<TObservation, TAction>?> _environments = [];
    private readonly List<Slot> _slots = [];
    private readonly List<int> _active = [], _still = [], _pending = [], _valued = [], _members = [];
    private readonly List<TObservation> _observations = [];
    private EnvironmentStep<TObservation>[] _steps = [];
    private TAction[] _actions = [];
    private ActionRecord[] _records = [];
    private int _created;
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
    /// <param name="shards">The shards a rollout is cut into (<see cref="RolloutSpread"/>); 0 for one per rank (1 in one process).</param>
    /// <param name="spread">How the episodes are cut into shards: <see cref="RolloutSpread.Blocks"/> (the default) or <see cref="RolloutSpread.Interleaved"/>.</param>
    /// <exception cref="ArgumentException">The kind observes or acts with other types, or an unknown split.</exception>
    public RolloutCollector(IEnvironmentKind kind, IReadOnlyDictionary<string, string>? environmentOptions = null, int seed = 0, int environments = 0, bool? parallelSteps = null,
        int shards = 0, string spread = RolloutSpread.Blocks)
    {
        ArgumentNullException.ThrowIfNull(kind);
        ArgumentOutOfRangeException.ThrowIfNegative(environments);
        ArgumentOutOfRangeException.ThrowIfNegative(shards);
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
        Shards = shards;
        Spread = RolloutSpread.Parse(spread);
    }

    /// <summary>The most environments stepped at once.</summary>
    public int Environments { get; }

    /// <summary>Whether environments step in parallel as asked: true, false, or null for measured.</summary>
    public bool? ParallelSteps { get; }

    /// <summary>The shards a rollout is cut into; 0: one per rank of the policy (1 in one process).</summary>
    public int Shards { get; }

    /// <summary>How the episodes are cut into shards (<see cref="RolloutSpread.Names"/>).</summary>
    public string Spread { get; }

    /// <summary>Whether the environments step in parallel: as asked, or as measured (null until the measurement is done).</summary>
    public bool? SteppingInParallel => Environments <= 1 ? false : ParallelSteps ?? _measured;

    /// <summary>The environments made so far (one per episode slot this collector played, of the largest rollout).</summary>
    public int Created => _created;

    /// <summary>The episodes this collector played in its last rollout (on a cluster, its rank's share; every episode in one process).</summary>
    public int Played { get; private set; }

    /// <summary>
    /// Plays the episodes <paramref name="episodes"/> to their ends (or <paramref name="maxSteps"/>) with
    /// <paramref name="policy"/>, and returns them as a batch in episode-number order (the caller disposes it). For an episode
    /// cut short, by its environment or by <paramref name="maxSteps"/>, whose actions have value estimates, the policy is asked
    /// for the value of its last observation to bootstrap from. On a cluster (the policy is an
    /// <see cref="IRolloutRanks{TObservation, TAction}"/> of several ranks), this rank plays its shards' episodes and the batch
    /// holds every rank's, gathered by the policy; every rank calls it with the same episodes.
    /// </summary>
    /// <param name="policy">The policy that acts.</param>
    /// <param name="episodes">Where each episode starts; at least one.</param>
    /// <param name="sampling">Greedy or sampled, the temperature, and the seed each round's (and shard's) seed is drawn from.</param>
    /// <param name="maxSteps">The most steps an episode takes before it is cut (truncated); 0 for the environment's own limit.</param>
    /// <param name="cancellationToken">Stops the rollout (between rounds, and in the steps of environments that observe it).</param>
    /// <exception cref="InvalidOperationException">The policy acted for another number of observations, or its ranks gathered another number of episodes.</exception>
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

        var ranks = policy as IRolloutRanks<TObservation, TAction>;
        int world = ranks?.WorldSize ?? 1, rank = ranks?.Rank ?? 0;
        if (world < 1 || (uint)rank >= (uint)world)
        {
            throw new InvalidOperationException($"The policy is rank {rank} of {world}: a rank from 0 to the world size less one.");
        }

        // The split: this rank's slots, and each slot's shard (none to compute for one shard).
        int shards = Shards > 0 ? Shards : world;
        int[]? shardOf = shards == 1 ? null : ShardsOf(episodes, shards);
        Prepare(episodes.Count);
        var trajectories = new Trajectory<TObservation, TAction>?[episodes.Count];
        _active.Clear();
        for (int k = 0; k < episodes.Count; k++)
        {
            if (RolloutSpread.Rank(shardOf?[k] ?? 0, shards, world) == rank)
            {
                _active.Add(k);
            }
        }

        Played = _active.Count;
        foreach (int k in _active)
        {
            _ = EnvironmentOf(k);                                            // this rank's slots' environments, made once
        }

        foreach (int k in _active)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var slot = _slots[k];
            slot.Clear();
            slot.Current = _environments[k]!.Reset(episodes[k].Seed);
            slot.Observations.Add(slot.Current);
        }

        // This rank's shards: shard s is rank ⌊s·W/S⌋'s, so its shards are one contiguous run.
        int firstShard = (int)(((long)rank * shards + world - 1) / world), endShard = (int)(((long)(rank + 1) * shards + world - 1) / world);
        for (int round = 0; _active.Count > 0; round++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (int s = firstShard; s < endShard; s++)
            {
                // The shard's episodes under way act together, from the round's and the shard's seed (the round's alone for shard 0).
                Members(shardOf, s, _active);
                if (_members.Count == 0)
                {
                    continue;
                }

                var acted = await policy.ActAsync(_observations, sampling with { Seed = Seeds.Derive(sampling.Seed, Seeds.Sampling, round, s) }, cancellationToken).ConfigureAwait(false);
                if (acted.Actions.Count != _members.Count || acted.Records.Count != _members.Count)
                {
                    throw new InvalidOperationException($"The policy acted {acted.Actions.Count} times for {_members.Count} observations: an action per observation.");
                }

                for (int j = 0; j < _members.Count; j++)
                {
                    _actions[_members[j]] = acted.Actions[j];
                    _records[_members[j]] = acted.Records[j];
                }
            }

            await StepAsync(cancellationToken).ConfigureAwait(false);

            // Record the round in episode order; the episodes that ended wait for their trajectories.
            _still.Clear();
            _pending.Clear();
            for (int i = 0; i < _active.Count; i++)
            {
                int k = _active[i];
                var slot = _slots[k];
                var step = _steps[i];
                slot.Actions.Add(_actions[i]);
                slot.Records.Add(_records[i]);
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

            Array.Clear(_actions, 0, _active.Count);                          // nothing of the round kept past it
            Array.Clear(_records, 0, _active.Count);
            await FinalValuesAsync(policy, sampling, round, shardOf, firstShard, endShard, cancellationToken).ConfigureAwait(false);
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

        var mine = new List<Trajectory<TObservation, TAction>>(Played);
        foreach (var trajectory in trajectories)
        {
            if (trajectory is not null)
            {
                mine.Add(trajectory);
            }
        }

        if (world == 1)
        {
            return new RolloutBatch<TObservation, TAction>(mine);
        }

        // Every rank's episodes, in rank order; the batch sorts them by number, so it is the same on every rank.
        var all = ranks!.Gather(mine, cancellationToken);
        if (all.Count != episodes.Count)
        {
            throw new InvalidOperationException($"The ranks gathered {all.Count} episodes for a rollout of {episodes.Count}: every rank collects with the same episodes and settings.");
        }

        return new RolloutBatch<TObservation, TAction>(all);
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
            environment?.Dispose();
        }

        _environments.Clear();
    }

    // Each episode's shard, from the episodes' numbers (RolloutSpread).
    private int[] ShardsOf(IReadOnlyList<EpisodeStart> episodes, int shards)
    {
        var indices = new int[episodes.Count];
        for (int k = 0; k < indices.Length; k++)
        {
            indices[k] = episodes[k].Index;
        }

        return RolloutSpread.Shards(indices, shards, Spread);
    }

    // The members of shard `shard` among `active` (their places in it) into _members, their observations into _observations.
    private void Members(int[]? shardOf, int shard, List<int> active)
    {
        _members.Clear();
        _observations.Clear();
        for (int i = 0; i < active.Count; i++)
        {
            if ((shardOf?[active[i]] ?? 0) == shard)
            {
                _members.Add(i);
                _observations.Add(_slots[active[i]].Current);
            }
        }
    }

    // Room for `slots` episode slots: scratch lists and the round's arrays; each slot's environment is made when a rollout
    // first plays it (EnvironmentOf).
    private void Prepare(int slots)
    {
        while (_slots.Count < slots)
        {
            _environments.Add(null);
            _slots.Add(new Slot());
        }

        if (_steps.Length < slots)
        {
            _steps = new EnvironmentStep<TObservation>[slots];
            _actions = new TAction[slots];
            _records = new ActionRecord[slots];
        }
    }

    // Slot k's environment, made once with the slot's seed (the same seed whichever rank plays the slot).
    private IEnvironment<TObservation, TAction> EnvironmentOf(int k)
    {
        if (_environments[k] is { } made)
        {
            return made;
        }

        var created = _kind.Create(_options, Seeds.Derive(_seed, Seeds.Creation, k));
        if (created is not IEnvironment<TObservation, TAction> typed)
        {
            created.Dispose();
            throw new InvalidOperationException($"Environment kind '{_kind.Name}' made a {created.GetType().Name}, not an IEnvironment<{typeof(TObservation).Name}, {typeof(TAction).Name}>.");
        }

        _environments[k] = typed;
        _created++;
        return typed;
    }

    // Steps the active environments with the round's actions into _steps, in order: one after another, or in parallel.
    private async ValueTask StepAsync(CancellationToken cancellationToken)
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
                _steps[i] = await _environments[_active[i]]!.StepAsync(_actions[i], token).ConfigureAwait(false)).ConfigureAwait(false);
        }
        else
        {
            for (int i = 0; i < count; i++)
            {
                _steps[i] = await _environments[_active[i]]!.StepAsync(_actions[i], cancellationToken).ConfigureAwait(false);
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
    // asked of the policy in one call per shard; the others get NaN.
    private async ValueTask FinalValuesAsync(IPolicy<TObservation, TAction> policy, ActionSampling sampling, int round, int[]? shardOf, int firstShard, int endShard,
        CancellationToken cancellationToken)
    {
        _valued.Clear();
        foreach (int k in _pending)
        {
            var slot = _slots[k];
            slot.FinalValue = float.NaN;
            if (slot.Truncated && slot.Records[0].HasValue)
            {
                _valued.Add(k);
            }
        }

        if (_valued.Count == 0)
        {
            return;
        }

        for (int s = firstShard; s < endShard; s++)
        {
            Members(shardOf, s, _valued);
            if (_members.Count == 0)
            {
                continue;
            }

            var valued = await policy.ActAsync(_observations, sampling with { Seed = Seeds.Derive(sampling.Seed, Seeds.FinalValues, round, s) }, cancellationToken).ConfigureAwait(false);
            for (int j = 0; j < _members.Count; j++)
            {
                _slots[_valued[_members[j]]].FinalValue = valued.Records[j].Value;
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
