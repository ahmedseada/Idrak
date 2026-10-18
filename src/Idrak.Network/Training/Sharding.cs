// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Runtime.CompilerServices;
using Idrak.Networking.Abstractions;
using Idrak.Networking.Compression;

namespace Idrak.Networking.Training;

/// <summary>"sharded" in <see cref="GradientSyncs"/>: ZeRO-style data parallel, each rank owning its share of the trained parameters (plan 23, phase C6).</summary>
internal sealed class ShardedSyncProvider : IGradientSyncProvider
{
    public string Name => "sharded";

    public string Summary => "each rank owns a share of the trained parameters: their gradients reduce-scattered to it, their optimizer state kept there, "
                             + "the weights gathered (IDRAK_CLUSTER_SHARD: optimizer, gradients, parameters)";

    public IGradientSync Create(ICollectives collectives, Optimizer optimizer, ClusterOptions options) => new GradientReduction(collectives, optimizer, options, sharded: true);
}

/// <summary>
/// Which rank owns each trained parameter in a sharded run (plan 23, phase C6): the largest first, each to the rank that
/// owns the fewest values so far (the lowest rank on a tie), so the shares are as even as whole parameters allow. It
/// depends on the parameters' sizes and the world size alone, so every rank computes the same owners.
/// </summary>
internal static class ShardOwners
{
    public static int[] Assign(IReadOnlyList<Tensor> parameters, int world)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(world);
        var owners = new int[parameters.Count];
        var load = new long[world];
        foreach (int i in Enumerable.Range(0, parameters.Count).OrderByDescending(i => parameters[i].Size).ThenBy(i => i))
        {
            int least = 0;
            for (int r = 1; r < world; r++)
            {
                if (load[r] < load[least])
                {
                    least = r;
                }
            }

            owners[i] = least;
            load[least] += parameters[i].Size;
        }

        return owners;
    }
}

/// <summary>
/// The sharded part of <see cref="GradientReduction"/> (plan 23, phase C6), ZeRO-style. Each trained parameter has one
/// owner (<see cref="ShardOwners"/>, over the ranks of the step's group). The gradient buckets are the all-reduce's (the
/// same sizes, overlap and measured size), each grouped by owner and reduce-scattered, so each rank receives the
/// token-weighted sum of its own parameters' gradients only; it clips them by the global norm (the ranks' squares summed),
/// steps its own parameters (the optimizer's moments exist for those alone: an optimizer makes a parameter's state when it
/// first steps it), and the weights travel back by bucket (an all-gather of each bucket's pieces).
/// <list type="bullet">
/// <item><see cref="ShardMode.Optimizer"/>: as above; the other parameters' gradients are dropped once reduced.</item>
/// <item><see cref="ShardMode.Gradients"/>: the other parameters' gradients are dropped as soon as their bucket was taken
/// during the backward pass, so a rank holds a bucket's worth beyond its own.</item>
/// <item><see cref="ShardMode.Parameters"/>: each rank keeps only its own parameters' weights (a master copy); every
/// parameter's memory is given back between uses and gathered from its owner just before a layer's forward
/// (<see cref="ForwardHooks"/>) and when the backward pass reads it (the storage's recomputation), then given back again;
/// no weights travel after the update.</item>
/// </list>
/// A group that changes (a late joiner) moves each share to its new owner: the weights and each parameter's optimizer
/// entries (<see cref="Optimizer.StateParameter"/>) from the rank that held them. A checkpoint joins every rank's share
/// (<see cref="SaveOptimizerState"/>), and a resumed run keeps its own (<see cref="LoadOptimizerState"/>).
/// </summary>
internal sealed partial class GradientReduction
{
    private readonly ShardMode _mode;
    private readonly WireType _wire;
    private readonly bool[] _had = [];                        // whether each parameter had a gradient when its bucket was taken
    private readonly bool[] _hadKnown = [];
    private readonly int[] _gatheredParts = [];               // buckets of each parameter taken this step
    private readonly bool[] _released = [];                   // gradients given back this step
    private int[]? _owners;                                   // each parameter's owner, in the group's numbering
    private int _ownedRank = -1, _ownedWorld;
    private long _lastPresent = -1;
    private ShardedWeights? _weights;
    private IReadOnlyDictionary<string, OptimizerStateValue>? _pendingState;
    private int? _releasedThenChanged;

    /// <summary>Each parameter's owner, or null before the first step (sharded runs only).</summary>
    internal IReadOnlyList<int>? Owners => _owners;

    /// <summary>The values of the parameters this rank owns.</summary>
    internal long OwnedValues => _owners is null ? 0 : Enumerable.Range(0, Parameters.Count).Where(i => _owners[i] == Collectives.Rank).Sum(i => (long)Parameters[i].Size);

    public override bool Updates => _mode != ShardMode.None;

    private WireType CheckSharded(ClusterOptions options)
    {
        if (HostGradients is not null)
        {
            throw new ArgumentException("A sharded run (IDRAK_CLUSTER_SHARD) steps each rank's parameters on its device; the HostOptimizer keeps its state on the host already: "
                                        + "use one or the other.", nameof(options));
        }

        if (options.LocalSgdEvery > 0)
        {
            throw new ArgumentException("Local SGD (IDRAK_CLUSTER_LOCAL_SGD) steps every rank's whole weights; it does not shard (unset IDRAK_CLUSTER_SHARD).", nameof(options));
        }

        return options.CompressionName.ToLowerInvariant() switch
        {
            "none" => WireType.Float32,
            "bf16" => WireType.BFloat16,
            var other => throw new ArgumentException($"A sharded run reduce-scatters its buckets as float32 or bfloat16; the compression '{other}' is not one of them "
                                                     + "(IDRAK_CLUSTER_COMPRESSION=none or bf16).", nameof(options)),
        };
    }

    protected override void Began(bool present)
    {
        if (present && _mode != ShardMode.None)
        {
            EnsurePartitioned();
            _lastPresent = CurrentStep;
        }
    }

    // The owners of the group this rank trains in, made at its first step (until then every rank holds the whole state).
    private void EnsurePartitioned()
    {
        if (_mode == ShardMode.None || !Present || _owners is not null)
        {
            return;
        }

        Adopt(ShardOwners.Assign(Parameters, World));
    }

    private void Adopt(int[] owners)
    {
        (_owners, _ownedRank, _ownedWorld) = (owners, Collectives.Rank, World);
        _layout = NewLayout(_layout.BucketValues);
        ApplyPendingState();
        if (_mode == ShardMode.Parameters && World > 1)
        {
            _weights = new ShardedWeights(this, owners);           // masters of its own, every weight given back
        }
    }

    private void ResetShardStep()
    {
        if (_mode == ShardMode.None)
        {
            return;
        }

        Array.Clear(_hadKnown);
        Array.Clear(_gatheredParts);
        Array.Clear(_released);
        _releasedThenChanged = null;
    }

    // A bucket was taken (copied or queued to the host): each parameter's presence noted before any gradient is dropped;
    // with Shard = gradients or parameters, another rank's parameter whose every bucket was taken drops its gradient now.
    private void Gathered(int bucket)
    {
        if (!_layout.Sharded)
        {
            return;
        }

        foreach (int p in _layout.ParametersOf(bucket))
        {
            if (!_hadKnown[p])
            {
                (_had[p], _hadKnown[p]) = (HasGradient(p), true);
            }

            if (++_gatheredParts[p] == _layout.BucketsOf(p).Length && _mode >= ShardMode.Gradients && _owners![p] != Collectives.Rank && Parameters[p].Grad is not null)
            {
                Parameters[p].ReleaseGrad();
                _released[p] = true;
            }
        }
    }

    private bool HadGradient(int parameter) => _layout.Sharded && _hadKnown[parameter] ? _had[parameter] : HasGradient(parameter);

    private void CheckNotReleased(int bucket)
    {
        if (!_layout.Sharded)
        {
            return;
        }

        foreach (int p in _layout.ParametersOf(bucket))
        {
            if (_released[p] || _releasedThenChanged == p)
            {
                throw new InvalidOperationException($"Parameter {p} {Tensor.FormatShape(Parameters[p].Shape)} gained gradient after its bucket went and its gradient was dropped "
                    + $"(IDRAK_CLUSTER_SHARD={_mode.ToString().ToLowerInvariant()}): a weight shared into a checkpointed block is read again later in the backward pass. "
                    + "Shard the optimizer only (IDRAK_CLUSTER_SHARD=optimizer), or leave that block unchecked.");
            }
        }
    }

    // After the step's reduction: no rank keeps another's parameters' gradients.
    private void AfterShardedReduce()
    {
        if (!_layout.Sharded)
        {
            return;
        }

        for (int p = 0; p < Parameters.Count; p++)
        {
            if (_owners![p] != Collectives.Rank)
            {
                Parameters[p].ReleaseGrad();
            }
        }
    }

    public override double? Update(float maxNorm, bool readNorm = false)
    {
        if (_mode == ShardMode.None)
        {
            return base.Update(maxNorm, readNorm);
        }

        if (!Present)
        {
            throw new InvalidOperationException($"This rank sits out step {CurrentStep} (IGradientSync.BeginStep returned false): it updates nothing that step.");
        }

        EnsurePartitioned();
        double? norm = maxNorm > 0f || readNorm ? GlobalNorm() : null;
        if (maxNorm > 0f && norm is { } n && n > maxNorm)
        {
            float factor = (float)(maxNorm / n);
            foreach (var p in Owned())
            {
                if (p.Grad is { } g)
                {
                    g.Backend.Affine(g.Storage, g.Storage, g.Size, factor, 0f);
                }
            }
        }

        using (_weights?.Local())                                 // Shard = parameters: its own weights in memory for the step
        {
            Optimizer.Step();                                      // only its own parameters have gradients: only they move
        }

        foreach (var p in Parameters)
        {
            p.ReleaseGrad();
        }

        if (_mode != ShardMode.Parameters && World > 1)
        {
            GatherWeights();
        }

        return norm;
    }

    private IEnumerable<Tensor> Owned()
    {
        for (int i = 0; i < Parameters.Count; i++)
        {
            if (_owners is null || _owners[i] == Collectives.Rank)
            {
                yield return Parameters[i];
            }
        }
    }

    // The gradients' global L2 norm: each rank's squares of its own parameters' gradients, summed over the ranks.
    private double GlobalNorm()
    {
        float squares = 0f;
        var owned = Owned().Where(p => p.Grad is not null).ToList();
        if (owned.Count > 0)
        {
            using var total = Tensor.PersistentZeros([1], Collectives.Device);
            foreach (var p in owned)
            {
                total.Backend.SumSquares(p.Grad!.Storage, total.Storage, p.Size);
            }

            squares = total.Item();
        }

        Span<float> sum = [squares];
        Values.AllReduce(sum, WireType.Float32);
        return Math.Sqrt(sum[0]);
    }

    // Each bucket's weights from their owners: every rank's piece read from its own parameters, all-gathered, the others
    // written into place (the same values on every rank, as float32).
    private void GatherWeights()
    {
        var buffer = ArrayPool<float>.Shared.Rent(Math.Max(1, _layout.BucketValues));
        try
        {
            int rank = Collectives.Rank;
            for (int b = 0; b < _layout.Count; b++)
            {
                var bucket = buffer.AsSpan(0, _layout.Size(b));
                foreach (var (parameter, offset, n, at) in _layout.Pieces(b))
                {
                    if (_owners![parameter] == rank)
                    {
                        Read(Parameters[parameter], Parameters[parameter].Storage, offset, bucket.Slice(at, n));
                    }
                }

                Values.AllGather(bucket, _layout.Counts(b));
                foreach (var (parameter, offset, n, at) in _layout.Pieces(b))
                {
                    if (_owners![parameter] != rank)
                    {
                        Write(Parameters[parameter], Parameters[parameter].Storage, offset, bucket.Slice(at, n));
                    }
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
        }
    }

    public override IDisposable? Gather() => _weights is { } weights && Present ? weights.Gather() : null;

    public override void End()
    {
        if (_weights is { } weights && Present)
        {
            weights.End();                                         // every rank holds the whole trained weights again
            weights.Dispose();
            _weights = null;
        }
    }

    public override IReadOnlyDictionary<string, OptimizerStateValue>? SaveOptimizerState()
    {
        if (_mode == ShardMode.None)
        {
            return null;
        }

        if (_owners is null || !Present)
        {
            return _pendingState;                                  // nothing trained since it was loaded: the optimizer's own otherwise
        }

        if (TrySaveState() is not { } mine)
        {
            return null;
        }

        int rank = Collectives.Rank;
        var own = mine.Where(e => Optimizer.StateParameter(e.Key) is { } i ? _owners[i] == rank : rank == 0).ToDictionary(e => e.Key, e => e.Value, StringComparer.Ordinal);
        var merged = new Dictionary<string, OptimizerStateValue>(StringComparer.Ordinal);
        for (int r = 0; r < World; r++)
        {
            foreach (var (name, value) in BroadcastState(r == rank ? own : null, r))
            {
                merged.TryAdd(name, value);
            }
        }

        return merged;
    }

    // The rank owning `parameter` (C8b: a checkpoint streamed to rank 0 takes it from there), or null: every rank holds it.
    public override int? Owner(Tensor parameter)
    {
        ArgumentNullException.ThrowIfNull(parameter);
        if (_mode == ShardMode.None || _owners is not { } owners || !Present)
        {
            return null;
        }

        for (int i = 0; i < Parameters.Count; i++)
        {
            if (ReferenceEquals(Parameters[i], parameter))
            {
                return owners[i];
            }
        }

        return null;
    }

    public override IDisposable? Local() => _weights is { } weights && Present ? weights.Local() : null;

    // This rank's own entries (C8b): its parameters' and, on rank 0, the optimizer's own; no collective.
    public override IReadOnlyDictionary<string, OptimizerStateValue>? SaveOwnOptimizerState()
    {
        if (_mode == ShardMode.None)
        {
            return null;
        }

        int rank = Collectives.Rank;
        if (_owners is null || !Present)
        {
            // Nothing trained since it was loaded: the loaded state, rank 0's to give.
            return rank == 0 ? _pendingState ?? TrySaveState() : new Dictionary<string, OptimizerStateValue>(StringComparer.Ordinal);
        }

        if (TrySaveState() is not { } mine)
        {
            return null;
        }

        // In name order, as the joined state takes each rank's (OptimizerState), so a streamed checkpoint is the joined one.
        return mine.Where(e => Optimizer.StateParameter(e.Key) is { } i ? _owners[i] == rank : rank == 0).OrderBy(e => e.Key, StringComparer.Ordinal)
            .ToDictionary(e => e.Key, e => e.Value, StringComparer.Ordinal);
    }

    public override bool LoadOptimizerState(IReadOnlyDictionary<string, OptimizerStateValue> state)
    {
        ArgumentNullException.ThrowIfNull(state);
        if (_mode == ShardMode.None)
        {
            return false;
        }

        _pendingState = state;
        if (_owners is not null)
        {
            ApplyPendingState();
        }

        return true;
    }

    // A checkpoint's state kept for this rank's share: its own parameters' entries and the optimizer's own (its step count).
    private void ApplyPendingState()
    {
        if (_pendingState is not { } state || _owners is not { } owners)
        {
            return;
        }

        int rank = Collectives.Rank;
        Optimizer.LoadState(state.Where(e => Optimizer.StateParameter(e.Key) is not { } i || owners[i] == rank).ToDictionary(e => e.Key, e => e.Value, StringComparer.Ordinal));
        _pendingState = null;
    }

    private IReadOnlyDictionary<string, OptimizerStateValue>? TrySaveState()
    {
        try
        {
            return Optimizer.SaveState();
        }
        catch (NotSupportedException)
        {
            return null;                                           // an optimizer that cannot save keeps its state where it is
        }
    }

    // Rank `root`'s entries (given there, null elsewhere) on every rank: their length, then their values.
    private Dictionary<string, OptimizerStateValue> BroadcastState(IReadOnlyDictionary<string, OptimizerStateValue>? entries, int root)
    {
        var payload = entries is null ? null : OptimizerState.Write(entries);
        var (high, low) = Split(payload?.Length ?? 0);
        Span<float> length = [high, low];
        Values.Broadcast(length, root);
        payload ??= new float[Join(length[0], length[1])];
        Values.Broadcast(payload, root);
        return OptimizerState.Read(payload);
    }

    // The group changed (a late joiner): every rank says whether it trained the step before and as which rank, so each
    // share moves from its old owner to its new one; nothing moves while no rank has trained (every rank holds the start).
    protected override void ShareTrainedState(int holder)
    {
        if (_mode == ShardMode.None)
        {
            base.ShareTrainedState(holder);
            return;
        }

        bool was = _owners is not null && _lastPresent == CurrentStep - 1;
        var info = new float[2 * World];
        (info[2 * Collectives.Rank], info[(2 * Collectives.Rank) + 1]) = was ? (_ownedRank, _ownedWorld) : (-1f, 0f);
        Values.AllGather(info, 2);
        var newOf = new Dictionary<int, int>();
        int oldWorld = 0;
        for (int r = 0; r < World; r++)
        {
            if (info[2 * r] >= 0)
            {
                newOf[(int)info[2 * r]] = r;
                oldWorld = (int)info[(2 * r) + 1];
            }
        }

        var newOwners = ShardOwners.Assign(Parameters, World);
        if (newOf.Count == 0)
        {
            DropShares();
            base.ShareTrainedState(holder);                        // nobody trained yet: the start, then this group's owners
            Adopt(newOwners);
            return;
        }

        if (newOf.Count != oldWorld)
        {
            var missing = Enumerable.Range(0, oldWorld).Where(r => !newOf.ContainsKey(r));
            throw new InvalidOperationException($"Rank(s) {string.Join(", ", missing)} of the step before (of {oldWorld}) are not in step {CurrentStep}'s group: "
                                                + "their share of the trained state is gone (a sharded run keeps each share on one rank).");
        }

        var from = ShardOwners.Assign(Parameters, oldWorld).Select(old => newOf[old]).ToArray();
        int rank = Collectives.Rank;

        // The weights: whole on every rank of the old group unless only the owners hold them.
        if (_mode == ShardMode.Parameters)
        {
            var old = _weights;
            for (int i = 0; i < Parameters.Count; i++)
            {
                old?.Materialize(i);                               // its owner's values, the others' memory only
                Collectives.Broadcast(Parameters[i].Storage, Parameters[i].Size, from[i]);
            }

            old?.Dispose();
            _weights = null;
        }
        else
        {
            Broadcast(Parameters, holder);
        }

        // The optimizer's state: its own entries (the step count) from the holder, each moving parameter's from its old owner.
        if (TrySaveState() is { } mine)                           // every rank's optimizer is of one kind: all save, or none does
        {
            var kept = mine.Where(e => Optimizer.StateParameter(e.Key) is { } i && from[i] == rank && newOwners[i] == rank).ToDictionary(e => e.Key, e => e.Value, StringComparer.Ordinal);
            foreach (var (name, value) in BroadcastState(rank == holder ? mine.Where(e => Optimizer.StateParameter(e.Key) is null).ToDictionary() : null, holder))
            {
                kept[name] = value;
            }

            for (int i = 0; i < Parameters.Count; i++)
            {
                if (from[i] == newOwners[i])
                {
                    continue;
                }

                var moved = BroadcastState(rank == from[i] ? mine.Where(e => Optimizer.StateParameter(e.Key) == i).ToDictionary() : null, from[i]);
                if (newOwners[i] == rank)
                {
                    foreach (var (name, value) in moved)
                    {
                        kept[name] = value;
                    }
                }
            }

            if (kept.Count > 0 || mine.Count > 0)
            {
                Optimizer.LoadState(kept);
            }
        }

        _owners = null;
        Adopt(newOwners);
    }

    private void DropShares()
    {
        _weights?.Dispose();
        _weights = null;
        _owners = null;
    }

    private void DisposeShards() => DropShares();

    /// <summary>
    /// Shard = parameters (phase C6): the master copies of this rank's own weights and the rule that gathers a layer's.
    /// Every trained parameter's memory is given back (<see cref="Backend.Evict"/>) with a recomputation that brings it
    /// back from its owner (a broadcast of the owner's master; on the owner, the master copied in first), so the backward
    /// pass's reads gather what they need and give it back after. A module whose trained parameters fit
    /// <see cref="ClusterOptions.GatherBytes"/> (each rank's share by default), or a module without children, is gathered
    /// whole as its forward begins and given back as it ends; a larger one gathers its own parameters only and leaves the
    /// rest to its children. Every rank runs the same forward and backward, so the gathers meet in the same order.
    /// </summary>
    private sealed class ShardedWeights : IDisposable
    {
        private readonly GradientReduction _sync;
        private readonly int[] _owners;
        private readonly int _rank;
        private readonly Tensor?[] _masters;
        private readonly Dictionary<Tensor, int> _index = new(ReferenceEqualityComparer.Instance);
        private readonly ConditionalWeakTable<Module, int[]> _units = new();
        private readonly long _budget;
        private IDisposable? _listener;
        private bool _local;
        private bool _ended;

        public ShardedWeights(GradientReduction sync, int[] owners)
        {
            _sync = sync;
            _owners = owners;
            _rank = sync.Collectives.Rank;
            var parameters = sync.Parameters;
            _masters = new Tensor?[parameters.Count];
            long total = 0;
            for (int i = 0; i < parameters.Count; i++)
            {
                _index.TryAdd(parameters[i], i);
                total += parameters[i].Size;
            }

            _budget = sync.Options.GatherBytes ?? Math.Max(sizeof(float), total * sizeof(float) / sync.World);
            for (int i = 0; i < parameters.Count; i++)
            {
                var p = parameters[i];
                if (owners[i] == _rank)
                {
                    // A deferred load (plan 23, C8b) left every parameter unread on the device: this rank reads its own now;
                    // the others' stay unread, and from now on come from their owners.
                    if (p.Storage.Evicted)
                    {
                        p.Backend.Restore(p.Storage);
                    }

                    var master = Tensor.PersistentZeros([Math.Max(1, p.Size)], p.Device);
                    p.Backend.Copy(p.Storage, master.Storage, p.Size);
                    _masters[i] = master;
                }

                Evict(i);
            }

            _listener = ForwardHooks.OnEnter(Enter);
        }

        private IReadOnlyList<Tensor> Parameters => _sync.Parameters;

        // The values of the parameters this rank owns, in memory for the optimizer's step (no collective); put back after.
        public IDisposable Local()
        {
            var restored = new List<int>();
            _local = true;
            try
            {
                for (int i = 0; i < Parameters.Count; i++)
                {
                    if (_owners[i] == _rank && Restore(i))
                    {
                        restored.Add(i);
                    }
                }
            }
            finally
            {
                _local = false;
            }

            return new Scope(() =>
            {
                foreach (int i in restored)
                {
                    Keep(i);
                    Evict(i);
                }
            });
        }

        // Parameter i in memory without a collective (a regroup broadcasts it next): the owner's values, the others' memory.
        public void Materialize(int i)
        {
            _local = true;
            try
            {
                Restore(i);
            }
            finally
            {
                _local = false;
            }
        }

        // Every weight whole on every rank until disposed; what is written meanwhile into a rank's own goes to its master.
        public IDisposable Gather()
        {
            var restored = new List<int>();
            for (int i = 0; i < Parameters.Count; i++)
            {
                if (Restore(i))
                {
                    restored.Add(i);
                }
            }

            return new Scope(() =>
            {
                foreach (int i in restored)
                {
                    Keep(i);
                    Evict(i);
                }
            });
        }

        // The run ended: every weight whole again, for good.
        public void End()
        {
            for (int i = 0; i < Parameters.Count; i++)
            {
                Restore(i);
            }

            _ended = true;
        }

        public void Dispose()
        {
            _listener?.Dispose();
            _listener = null;
            for (int i = 0; i < _masters.Length; i++)
            {
                _masters[i]?.Dispose();
                _masters[i] = null;
            }
        }

        private bool Restore(int i)
        {
            var storage = Parameters[i].Storage;
            return storage.Evicted && storage.Backend.Restore(storage);
        }

        private void Evict(int i)
        {
            if (_ended)
            {
                return;
            }

            var p = Parameters[i];
            p.Backend.SetRecompute(p.Storage, storage => Recompute(i, storage));   // evicts it, or replaces a deferred load's read
        }

        // The owner's values into its master (what a step or a gathered scope wrote).
        private void Keep(int i)
        {
            if (_masters[i] is { } master)
            {
                Parameters[i].Backend.Copy(Parameters[i].Storage, master.Storage, Parameters[i].Size);
            }
        }

        // A parameter given memory again: the owner's master, sent from the owner to every rank (unless local).
        private void Recompute(int i, Storage storage)
        {
            var p = Parameters[i];
            if (_masters[i] is { } master)
            {
                p.Backend.Copy(master.Storage, storage, p.Size);
            }

            if (!_local)
            {
                // The buckets the backward pass sent so far finish first: the gather runs on these collectives alone, in
                // the same place of every rank's order.
                _sync._worker?.Wait();
                _sync.Collectives.Broadcast(storage, p.Size, _owners[i]);
            }
        }

        // A module's forward begins: its unit gathered (each parameter not in memory brought from its owner), given back as it ends.
        private IDisposable? Enter(Module module)
        {
            if (_ended)
            {
                return null;
            }

            var unit = _units.GetValue(module, Unit);
            List<int>? restored = null;
            foreach (int i in unit)
            {
                if (Restore(i))
                {
                    (restored ??= []).Add(i);
                }
            }

            return restored is null ? null : new Scope(() =>
            {
                foreach (int i in restored)
                {
                    Evict(i);
                }
            });
        }

        // The trained parameters a module's forward gathers: all of its subtree's when they fit the budget or it has no
        // children, else its own (those under no child).
        private int[] Unit(Module module)
        {
            var subtree = Indices(module.Parameters());
            long bytes = subtree.Sum(i => (long)Parameters[i].Size * sizeof(float));
            var children = module.Children().ToList();
            if (children.Count == 0 || bytes <= _budget)
            {
                return subtree;
            }

            var below = new HashSet<int>(children.SelectMany(c => Indices(c.Parameters())));
            return [.. subtree.Where(i => !below.Contains(i))];
        }

        private int[] Indices(IEnumerable<Tensor> tensors)
        {
            var seen = new List<int>();
            foreach (var t in tensors)
            {
                if (_index.TryGetValue(t, out int i) && !seen.Contains(i))
                {
                    seen.Add(i);
                }
            }

            return [.. seen];
        }

        private sealed class Scope(Action end) : IDisposable
        {
            private Action? _end = end;

            public void Dispose()
            {
                var end = Interlocked.Exchange(ref _end, null);
                end?.Invoke();
            }
        }
    }
}
