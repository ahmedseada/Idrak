// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using Idrak.Networking.Training;

namespace Idrak.Networking.Abstractions;

/// <summary>
/// What every gradient synchronisation of Idrak.Network shares (plan 23, phases C2 and C4): the starting state, the
/// steps of a group whose ranks can change (<see cref="IStepGroup"/>: a rank that joins takes the weights and optimizer
/// state of one that trained the step before, the shares go even over the new group, the speed measurement starts
/// again), rank 0's weights every <see cref="ClusterOptions.SyncWeightsEvery"/> steps, the split of a step's rows by the
/// ranks' measured speeds, and the small collectives (sums, broadcasts, barriers) a trainer needs. A rank that sits a step
/// out runs no collective: its sums, broadcasts and barriers are its own.
/// </summary>
internal abstract class SyncCore : IGradientSync
{
    private double[] _shares;
    private long _mark;
    private double _measuredTokens, _measuredSeconds;
    private long _nextExchange;

    protected SyncCore(ICollectives collectives, Optimizer optimizer, ClusterOptions options)
    {
        Collectives = collectives ?? throw new ArgumentNullException(nameof(collectives));
        Optimizer = optimizer ?? throw new ArgumentNullException(nameof(optimizer));
        Options = options ?? throw new ArgumentNullException(nameof(options));
        Parameters = optimizer.Parameters;
        HostGradients = optimizer as IHostGradients;
        Values = new HostValues(collectives);
        _shares = [.. DrawShares.Even(collectives.WorldSize)];
        _nextExchange = FirstExchange;
        for (int i = 0; i < Parameters.Count; i++)
        {
            if (!ReferenceEquals(Parameters[i].Backend, collectives.Device.Backend))
            {
                throw new ArgumentException($"Parameter {i} {Tensor.FormatShape(Parameters[i].Shape)} is on {Parameters[i].Device}; rank {collectives.Rank}'s collectives are on {collectives.Device}: "
                                            + "train each rank's model on its rank's device.", nameof(optimizer));
            }
        }
    }

    public abstract string Name { get; }

    public ICollectives Collectives { get; }

    public IReadOnlyList<Tensor> Parameters { get; }

    public IReadOnlyList<double> Shares => _shares;

    protected Optimizer Optimizer { get; }

    protected ClusterOptions Options { get; }

    protected IHostGradients? HostGradients { get; }

    protected HostValues Values { get; }

    /// <summary>The step the trainer last began (<see cref="BeginStep"/>), or -1 when it begins none.</summary>
    protected long CurrentStep { get; private set; } = -1;

    /// <summary>Steps this rank reduced since its group formed (the same count on every rank of the group).</summary>
    protected long StepsInGroup { get; private set; }

    /// <summary>Whether this rank takes part in the current step (always, in a fixed group).</summary>
    protected bool Present => Collectives is IStepGroup group ? group.Present : _fixedPresent;

    private bool _fixedPresent = true;

    protected int World => Collectives.WorldSize;

    private long FirstExchange => Options.RebalanceSteps > 0 ? 1 + Options.RebalanceSteps : long.MaxValue;

    public void Start()
    {
        if (World > 1 && Present)
        {
            if (Options.StartingState == StartingState.Check)
            {
                CheckSame();
            }
            else
            {
                Broadcast(Parameters);
                HostGradients?.ReloadParameters();
            }

            Started();
        }

        _mark = Stopwatch.GetTimestamp();
    }

    /// <summary>Every rank of the group started (a collective step every rank takes, after the starting state).</summary>
    protected virtual void Started()
    {
    }

    public bool BeginStep(long step)
    {
        if (Collectives is not IStepGroup group)
        {
            // A fixed group (a localhost or started cluster's processes, a real one): every rank takes part, until a replay's
            // last step (SimulationOptions.ReplayStep, every rank alike) as ClusterRank.BeginStep has it.
            CurrentStep = step;
            _fixedPresent = Options.Simulation.ReplayStep is not { } last || step <= last;
            if (_fixedPresent && Options.Simulation.Control is { } control)
            {
                Steer(control, step);                          // a launcher's pause, resume and stop (phase C9)
            }

            Began(_fixedPresent);
            return _fixedPresent;
        }

        if (!group.BeginStep(step))
        {
            CurrentStep = step;
            Began(false);
            return false;
        }

        CurrentStep = step;
        if (group.Changed)
        {
            Rejoin(group.Holder);
        }

        Began(true);
        return true;
    }

    // A group of processes steered through a launcher's control file (plan 23, phase C9): rank 0 reads it as the step begins
    // and broadcasts what it says, so every rank runs, waits or stops at this same step boundary. While paused rank 0 reads it
    // again every ControlPoll and broadcasts again, so no rank's wait reaches the timeout; a stop throws on every rank alike
    // (the trainer writes the last finished step's checkpoint first). One float a step travels, only when a control file is set.
    private void Steer(string path, long step)
    {
        Span<float> said = stackalloc float[1];
        bool paused = false;
        while (true)
        {
            if (Collectives.Rank == 0)
            {
                said[0] = (float)ClusterControlFile.Read(path);
            }

            if (World > 1)
            {
                Values.Broadcast(said, 0);
            }

            switch ((ClusterControlCommand)(int)said[0])
            {
                case ClusterControlCommand.Run:
                    if (paused && Collectives.Rank == 0)
                    {
                        ClusterControlFile.Report(path, "running", step);
                    }

                    return;
                case ClusterControlCommand.Stop:
                    if (Collectives.Rank == 0)
                    {
                        ClusterControlFile.Report(path, "stopped", step);
                    }

                    throw new ClusterRunStoppedException(step, $"the run was stopped before step {step} through its control file {path}");
                default:
                    if (Collectives.Rank == 0)
                    {
                        if (!paused)
                        {
                            ClusterControlFile.Report(path, "paused", step);
                        }

                        Thread.Sleep(Options.Simulation.ControlPoll);
                    }

                    paused = true;
                    break;
            }
        }
    }

    /// <summary>
    /// A step this rank took part in ended (<see cref="Stepped"/>, before its weight sync and speed exchange):
    /// <paramref name="wall"/> since the previous step ended (the whole step: its draws, passes, reduction and update).
    /// </summary>
    protected virtual void StepTimed(TimeSpan wall)
    {
    }

    /// <summary>A step began (<see cref="CurrentStep"/>), its group known: <paramref name="present"/> when this rank takes part.</summary>
    protected virtual void Began(bool present)
    {
    }

    public virtual bool Updates => false;

    public virtual double? Update(float maxNorm, bool readNorm = false) =>
        throw new NotSupportedException($"The {Name} synchronisation leaves the update to the optimizer (Updates is false).");

    public virtual IDisposable? Gather() => null;

    public virtual IReadOnlyDictionary<string, OptimizerStateValue>? SaveOptimizerState() => null;

    public virtual bool LoadOptimizerState(IReadOnlyDictionary<string, OptimizerStateValue> state) => false;

    public virtual void End()
    {
    }

    public abstract GradientSyncResult Reduce(double tokens, double loss, bool normalized = false);

    public virtual IDisposable? Overlap(double tokens, bool normalized = false) => null;

    public virtual void OverlapLost(string reason)
    {
    }

    public virtual void Stepped(long step)
    {
        if (World > 1 && Present)
        {
            StepTimed(_mark != 0 ? Stopwatch.GetElapsedTime(_mark) : TimeSpan.Zero);
            if (Options.SyncWeightsEvery > 0 && step > 0 && step % Options.SyncWeightsEvery == 0)
            {
                using (Gather())                                 // a sharded run's weights whole for it (each has one owner anyway)
                {
                    Broadcast(Parameters);                     // ranks of different device kinds round apart: rank 0's values
                }

                HostGradients?.ReloadParameters();
            }

            if (StepsInGroup >= _nextExchange)
            {
                Rebalance();
            }

            (Collectives as HostCollectives)?.ReportStep(step);   // separate processes: every rank's step goes to rank 0's log (C5)
        }

        _mark = Stopwatch.GetTimestamp();
    }

    public void Sum(Span<float> values)
    {
        if (World > 1 && Present && values.Length > 0)
        {
            Values.AllReduce(values, WireType.Float32);
        }
    }

    public void Broadcast(IReadOnlyList<Tensor> tensors, int root = 0)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        if (World == 1 || !Present)
        {
            return;
        }

        foreach (var tensor in tensors)
        {
            if (tensor.Size > 0)
            {
                Collectives.Broadcast(tensor.Storage, tensor.Size, root);
            }
        }
    }

    public void Barrier()
    {
        if (World > 1 && Present)
        {
            Collectives.Barrier();
        }
    }

    public virtual void Dispose() => Values.Dispose();

    /// <summary>
    /// Counts a reduced step for the speed measurement (the first, cold step aside, rule 76): call at the start of every
    /// <see cref="Reduce"/>.
    /// </summary>
    protected void Count(double tokens)
    {
        if (!(tokens >= 0) || double.IsInfinity(tokens))
        {
            throw new ArgumentOutOfRangeException(nameof(tokens), tokens, "A rank's tokens are 0 or more.");
        }

        long now = Stopwatch.GetTimestamp();
        if (++StepsInGroup > 1 && _mark != 0)
        {
            _measuredSeconds += Stopwatch.GetElapsedTime(_mark, now).TotalSeconds;
            _measuredTokens += tokens;
        }
    }

    /// <summary>
    /// The group changed and this rank takes part: every rank takes <paramref name="holder"/>'s trained state (the weights,
    /// the optimizer's state and the synchronisation's own, <see cref="ShareState"/>), the shares go even over the group
    /// and the speed measurement starts again.
    /// </summary>
    protected virtual void Rejoin(int holder)
    {
        _shares = [.. DrawShares.Even(World)];
        (StepsInGroup, _measuredTokens, _measuredSeconds, _mark, _nextExchange) = (0, 0, 0, 0, FirstExchange);
        if (World == 1)
        {
            return;
        }

        ShareState(holder);
        ShareTrainedState(holder);
    }

    /// <summary>
    /// The trained state every rank of a changed group takes (the weights and the optimizer's state): by default
    /// <paramref name="holder"/>'s, which holds it whole; a sharded synchronisation moves each share instead.
    /// </summary>
    protected virtual void ShareTrainedState(int holder)
    {
        Broadcast(Parameters, holder);
        HostGradients?.ReloadParameters();
        ShareOptimizerState(holder);
    }

    /// <summary>A synchronisation's own state <paramref name="holder"/> sends to every rank of a changed group (none by default).</summary>
    protected virtual void ShareState(int holder)
    {
    }

    /// <summary>A whole number exactly in two float32 values (16 bits each).</summary>
    protected static (float High, float Low) Split(long value) =>
        value is >= 0 and <= uint.MaxValue ? ((float)(value >> 16), (float)(value & 0xFFFF)) : throw new ArgumentOutOfRangeException(nameof(value), value, "A count from 0 to 2^32 - 1.");

    /// <summary>The whole number <see cref="Split"/> made.</summary>
    protected static long Join(float high, float low) => ((long)high << 16) | (long)low;

    // The holder's optimizer state (moments, step count), sent in two broadcasts: its length, then its values; a rank
    // whose optimizer cannot load a state keeps its own (the holder's cannot save one: it sends none).
    private void ShareOptimizerState(int holder)
    {
        float[]? payload = null;
        if (Collectives.Rank == holder)
        {
            try
            {
                payload = OptimizerState.Write(Optimizer.SaveState());
            }
            catch (NotSupportedException)
            {
                payload = [];
            }
        }

        var (high, low) = Split(payload?.Length ?? 0);
        Span<float> length = [high, low];
        Values.Broadcast(length, holder);
        long count = Join(length[0], length[1]);
        if (count == 0)
        {
            return;
        }

        payload ??= new float[count];
        Values.Broadcast(payload, holder);
        if (Collectives.Rank != holder)
        {
            try
            {
                Optimizer.LoadState(OptimizerState.Read(payload));
            }
            catch (NotSupportedException)
            {
                // An optimizer that keeps no state of its own (plain SGD) has nothing to take.
            }
        }
    }

    // Each rank's tokens per second over the steps since the last exchange, summed into one list every rank holds; the
    // shares follow them unless the ranks are within the tolerance of each other (then equal, so noise moves no rows).
    private void Rebalance()
    {
        var rates = new float[World];
        rates[Collectives.Rank] = _measuredTokens > 0 && _measuredSeconds > 0 ? (float)(_measuredTokens / _measuredSeconds) : 0f;
        Values.AllReduce(rates, WireType.Float32);
        (_measuredTokens, _measuredSeconds) = (0, 0);
        _nextExchange = Options.RebalanceEvery > 0 ? StepsInGroup + Options.RebalanceEvery : long.MaxValue;
        float slowest = rates.Min(), fastest = rates.Max();
        if (!(slowest > 0))
        {
            return;                                                // a rank had no rows to time: the shares stay
        }

        if (fastest / slowest - 1 <= Options.RebalanceTolerance)
        {
            _shares = [.. DrawShares.Even(World)];
            return;
        }

        double sum = rates.Sum(r => (double)r);
        _shares = [.. rates.Select(r => r / sum)];
    }

    // Starting values compared across the ranks: each parameter's SHA-256 (its first 64 bits, as four 16-bit whole numbers
    // that float32 carries exactly), gathered from every rank; the first rank and parameter that differ from rank 0's are named.
    private void CheckSame()
    {
        int count = Parameters.Count, per = Math.Max(1, count * 4);
        var all = new float[per * World];
        var mine = all.AsSpan(Collectives.Rank * per, per);
        Span<byte> digest = stackalloc byte[SHA256.HashSizeInBytes];
        for (int i = 0; i < count; i++)
        {
            var p = Parameters[i];
            var values = ArrayPool<float>.Shared.Rent(Math.Max(1, p.Size));
            try
            {
                p.Backend.Download(p.Storage, values.AsSpan(0, p.Size));
                SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan(0, p.Size)), digest);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(values);
            }

            for (int c = 0; c < 4; c++)
            {
                mine[(i * 4) + c] = (digest[2 * c] << 8) | digest[(2 * c) + 1];
            }
        }

        Values.AllGather(all, per);
        for (int r = 1; r < World; r++)
        {
            for (int i = 0; i < count; i++)
            {
                if (!all.AsSpan((r * per) + (i * 4), 4).SequenceEqual(all.AsSpan(i * 4, 4)))
                {
                    throw new InvalidOperationException($"Rank {r} starts from other values than rank 0 (parameter {i} of {count}, {Tensor.FormatShape(Parameters[i].Shape)}): "
                                                        + "give every rank the same seed and weights, or let rank 0's values be sent (IDRAK_CLUSTER_START=broadcast).");
                }
            }
        }
    }

    /// <summary>Publishes a step's <see cref="GradientsSynced"/> when someone listens.</summary>
    protected void Publish(string compression, int buckets, int overlapped, long bucketBytes, long bytes, TimeSpan backward, TimeSpan network, TimeSpan exposed, string? lost)
    {
        if (Telemetry.IsEnabled(TelemetryLevel.Network))
        {
            Telemetry.GradientSync(new GradientsSynced(Name, compression, Collectives.Rank, World, CurrentStep, buckets, overlapped, bucketBytes, bytes, backward, network, exposed, lost));
        }
    }
}

/// <summary>
/// An optimizer's saved state as float32 values, exactly (names as UTF-16 code units, counts as 16-bit halves), so one
/// broadcast carries it to a rank that joins; entries in name order, so every rank writes the same values.
/// </summary>
internal static class OptimizerState
{
    public static float[] Write(IReadOnlyDictionary<string, OptimizerStateValue> state)
    {
        var values = new List<float> { state.Count };
        foreach (var (name, value) in state.OrderBy(e => e.Key, StringComparer.Ordinal))
        {
            values.Add(name.Length);
            foreach (char c in name)
            {
                values.Add(c);
            }

            values.Add(value.Shape.Length);
            foreach (int dimension in value.Shape)
            {
                Whole(values, dimension);
            }

            Whole(values, value.Values.Length);
            values.AddRange(value.Values);
        }

        return [.. values];
    }

    public static Dictionary<string, OptimizerStateValue> Read(ReadOnlySpan<float> values)
    {
        int at = 0;
        int entries = (int)values[at++];
        var state = new Dictionary<string, OptimizerStateValue>(entries, StringComparer.Ordinal);
        for (int e = 0; e < entries; e++)
        {
            int length = (int)values[at++];
            var name = new char[length];
            for (int c = 0; c < length; c++)
            {
                name[c] = (char)values[at++];
            }

            var shape = new int[(int)values[at++]];
            for (int d = 0; d < shape.Length; d++)
            {
                shape[d] = Whole(values, ref at);
            }

            int count = Whole(values, ref at);
            state[new string(name)] = new OptimizerStateValue(shape, values.Slice(at, count).ToArray());
            at += count;
        }

        return state;
    }

    private static void Whole(List<float> values, int value)
    {
        values.Add(value >> 16);
        values.Add(value & 0xFFFF);
    }

    private static int Whole(ReadOnlySpan<float> values, ref int at)
    {
        int value = ((int)values[at] << 16) | (int)values[at + 1];
        at += 2;
        return value;
    }
}
