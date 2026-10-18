// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Diagnostics;
using System.Globalization;
using Idrak.Networking.Abstractions;
using Idrak.Networking.Compression;

namespace Idrak.Networking.Training;

/// <summary>"all-reduce" in <see cref="GradientSyncs"/>: every step's gradients summed across the ranks, a bucket at a time, each sent while the backward pass goes on.</summary>
internal sealed class AllReduceSyncProvider : IGradientSyncProvider
{
    public string Name => "all-reduce";

    public string Summary => "every step's gradients summed across the ranks, token-weighted, a bucket at a time, each sent while the backward pass goes on";

    public IGradientSync Create(ICollectives collectives, Optimizer optimizer, ClusterOptions options) =>
        options.Shard == ShardMode.None
            ? new GradientReduction(collectives, optimizer, options)
            : throw new ArgumentException($"all-reduce keeps every rank's state whole; IDRAK_CLUSTER_SHARD={options.Shard.ToString().ToLowerInvariant()} needs the sharded "
                                          + "synchronisation (leave IDRAK_CLUSTER_GRADIENT_SYNC unset, or name sharded).", nameof(options));
}

/// <summary>
/// The gradient all-reduce of data-parallel training (plan 23, phases C2 and C4). The gradients are addressed as
/// (parameter, offset, count) pieces packed into buckets in reverse parameter order (<see cref="BucketLayout"/>), never a
/// packed copy on the device. Each bucket is this rank's values times its tokens (its rows; 1 when the losses are already
/// the step's shares), summed across the ranks through the run's compression (<see cref="GradientCompressions"/>: none,
/// bf16, top-k), then divided by the step's tokens over every rank, which a small header collective of float32 values
/// gives (the tokens, the loss and which parameters have a gradient on any rank). So the sum is token-weighted: the
/// gradient one process would compute on every rank's rows.
/// <para>
/// With overlap (<see cref="ClusterOptions.Overlap"/>, the trainers' <see cref="IGradientSync.Overlap"/> scope around the
/// step's last backward pass) a hook per parameter (<see cref="Autograd.OnGradientReady"/>) notes each finished gradient,
/// and a bucket whose parameters are all finished is copied to the host and reduced on a thread of its own while the
/// backward pass goes on; <see cref="Reduce"/> sends the rest, the header last, and writes every bucket back. Without it
/// the header goes first and each bucket is reduced and written back in turn through one pooled buffer. Both send the same
/// buckets with the same values, so the result is the same bit for bit (rule 71); overlap holds the step's gradients on the
/// host until the header says the step's tokens (one host buffer, allocated once). A parameter whose gradient changes after
/// its bucket was sent (a weight shared into a checkpointed block) marks the bucket, every rank learns it in the header,
/// and the bucket is summed again, whole and uncompressed.
/// </para>
/// <para>
/// The bucket size is <see cref="ClusterOptions.BucketBytes"/>, or measured on the first step (<see cref="BucketSizing"/>).
/// An optimizer that holds the gradients on the host (<see cref="IHostGradients"/>, the HostOptimizer) is reduced there,
/// after the backward pass. Each step is published as <see cref="GradientsSynced"/>: the buckets, how many went during
/// the backward pass, the network time and the part of it the step waited for after the backward pass, and why overlap
/// was lost (a recorded step replayed as a graph runs no hook).
/// </para>
/// <para>
/// Phase C6: on a device with pinned staging slots (<see cref="Backend.CreateHostStaging"/>) a bucket finished during the
/// backward pass is queued to the host behind the backward pass's work and the reducing thread waits for it, so the
/// training thread never waits for the device (<see cref="ClusterOptions.AsyncStaging"/>); the values are the ones the
/// synchronous copy gives. A sharded run (the "sharded" synchronisation, <c>Sharding.cs</c>) uses the same buckets, each
/// grouped by owner and reduce-scattered, and updates itself (<see cref="Update"/>).
/// </para>
/// </summary>
internal sealed partial class GradientReduction : SyncCore
{
    /// <summary>
    /// Makes a device's pinned staging slots (the backend's own; a test puts in a staging of its own to check the
    /// background path without a GPU).
    /// </summary>
    internal static Func<Backend, int, int, IHostStaging?> CreateStaging = static (backend, slots, floats) => backend.CreateHostStaging(slots, floats);

    // The per-message cost measured on the simulator, per collective, world size and device kind, kept for the process.
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, float> MessageCosts = new(StringComparer.Ordinal);

    private const int HeaderFloats = 2;                       // tokens, loss; then a presence flag per parameter, a changed flag per bucket

    private readonly IGradientCompressor _compressor;
    private readonly Dictionary<Tensor, int> _index = new(ReferenceEqualityComparer.Instance);
    private readonly bool[] _present;
    private readonly List<(double ReadyAt, TimeSpan Simulated, TimeSpan Wall)> _sent = [];
    private BucketLayout _layout;
    private bool _sized;
    private Tensor? _scratch;                                 // device staging of a piece of a parameter larger than a bucket, made once
    private OverlapWorker? _worker;
    private float[]? _held;                                   // overlap: every bucket's values, then the header

    // The step's overlap (Overlap to Reduce).
    private bool _overlapping;
    private double _tokens;
    private bool _normalized;
    private float _weight;
    private long _backwardStart;
    private bool[] _ready = [];
    private int[] _pending = [];
    private bool[] _launched = [];
    private bool[] _changed = [];
    private (double ReadyAt, TimeSpan Simulated, TimeSpan Wall)[] _took = [];
    private int _next, _overlapped;
    private int _header;                                      // the header's place among the step's timed collectives
    private IDisposable? _hook;
    private string? _lost;

    // Background staging of finished buckets (phase C6): pinned slots, which slot each launched bucket's copy went to, and
    // the pieces of it that had no gradient (zeros, written on the host).
    private IHostStaging? _staging;
    private bool _stagingTried;
    private int _stagingSlots;
    private int[] _slotOf = [];
    private readonly Stack<int> _freeSlots = new();
    private readonly object _slotGate = new();
    private List<(int At, int Count)>?[] _zeros = [];
    private long _launchTicks, _stagingWaitTicks;

    public GradientReduction(ICollectives collectives, Optimizer optimizer, ClusterOptions options, bool sharded = false)
        : base(collectives, optimizer, options)
    {
        _present = new bool[Parameters.Count];
        for (int i = 0; i < Parameters.Count; i++)
        {
            _index.TryAdd(Parameters[i], i);
        }

        _mode = sharded ? (options.Shard == ShardMode.None ? ShardMode.Optimizer : options.Shard) : ShardMode.None;
        if (_mode != ShardMode.None)
        {
            _wire = CheckSharded(options);
            _compressor = new DenseCompressor("none", collectives, _wire);   // the changed buckets' path only; the rest reduce-scatters
            _had = new bool[Parameters.Count];
            _hadKnown = new bool[Parameters.Count];
            _gatheredParts = new int[Parameters.Count];
            _released = new bool[Parameters.Count];
        }
        else
        {
            _compressor = GradientCompressions.Create(collectives, Parameters, options);
        }

        _sized = options.BucketBytes is not null;
        _layout = NewLayout(ValuesOf(options.BucketBytes ?? BucketSizing.ProbeBytes));
        if (collectives.Transport?.Capabilities.CrossProcess != true)
        {
            StartAdvice(DecideInProcess());                    // ranks of one process: one decision for the run, whichever rank comes first
        }
    }

    public override string Name => _mode == ShardMode.None ? "all-reduce" : "sharded";

    /// <summary>The training thread's time queuing finished buckets during the backward pass (gathering or staging them).</summary>
    internal TimeSpan LaunchTime => TimeSpan.FromSeconds(_launchTicks / (double)Stopwatch.Frequency);

    /// <summary>The reducing thread's time waiting for staged buckets' copies (time a copy on the training thread would have held it).</summary>
    internal TimeSpan StagingWait => TimeSpan.FromSeconds(Interlocked.Read(ref _stagingWaitTicks) / (double)Stopwatch.Frequency);

    /// <summary>Whether finished buckets are copied to the host in the background.</summary>
    internal bool Staged => _staging is not null;

    /// <summary>The bucket layout in use (its size measured on the first step when unset).</summary>
    internal BucketLayout Layout => _layout;

    private int HeaderLength => HeaderFloats + Parameters.Count + _layout.Count;

    public override IDisposable? Overlap(double tokens, bool normalized = false)
    {
        if (World == 1 || !Present)
        {
            return null;
        }

        EndOverlap();                                          // a step that never reached Reduce (it threw) leaves nothing behind
        EnsurePartitioned();                                   // a sharded run's owners before its buckets are laid out
        _backwardStart = Stopwatch.GetTimestamp();
        (_tokens, _normalized) = (tokens, normalized);
        _lost = !Options.Overlap ? "overlap is off (IDRAK_CLUSTER_OVERLAP=0)"
            : HostGradients is not null ? "the optimizer brings the gradients to the host after the backward pass (HostOptimizer)"
            : !Values.OnHost ? "the collectives take device buffers only"
            : null;
        if (_lost is not null || _layout.Count == 1)
        {
            return null;                                       // one bucket: it is finished when the backward pass is
        }

        int buckets = _layout.Count;
        if (_ready.Length != Parameters.Count)
        {
            _ready = new bool[Parameters.Count];
        }

        if (_pending.Length != buckets)
        {
            (_pending, _launched, _changed, _took) = (new int[buckets], new bool[buckets], new bool[buckets], new (double, TimeSpan, TimeSpan)[buckets]);
        }

        Array.Clear(_ready);
        Array.Clear(_launched);
        Array.Clear(_changed);
        for (int b = 0; b < buckets; b++)
        {
            _pending[b] = _layout.ParametersOf(b).Length;
        }

        if (_held is null || _held.Length < _layout.Total + HeaderLength)
        {
            _held = GC.AllocateUninitializedArray<float>(_layout.Total + HeaderLength);
        }

        _weight = Weight(tokens, normalized);
        (_next, _overlapped, _overlapping) = (0, 0, true);
        PrepareStaging(buckets);
        ResetShardStep();
        _hook = Autograd.OnGradientReady(Finished);
        return new OverlapScope(this);
    }

    // Pinned slots for the buckets the backward pass finishes (a device that has them, background staging on), made once
    // for the bucket size and kept (rule 77); every slot free at the start of a step.
    private void PrepareStaging(int buckets)
    {
        if (_slotOf.Length != buckets)
        {
            (_slotOf, _zeros) = (new int[buckets], new List<(int, int)>?[buckets]);
        }

        Array.Fill(_slotOf, -1);
        if (_staging is { } current && (current.SlotFloats < _layout.BucketValues || _stagingSlots < buckets))
        {
            current.Dispose();
            (_staging, _stagingTried) = (null, false);             // the buckets changed (measured): slots of the new layout
        }

        if (_staging is null && !_stagingTried && Options.AsyncStaging && HostGradients is null)
        {
            // A slot per bucket (pinned host memory the size of the gradients, as the held buffer is), so the training
            // thread never waits for one while the buckets before are still on the network.
            _stagingTried = true;
            _stagingSlots = buckets;
            var staging = CreateStaging(Collectives.Device.Backend, buckets, _layout.BucketValues);
            if (staging is { PlacesCopies: true })
            {
                _staging = staging;
            }
            else
            {
                staging?.Dispose();                                // a slot per piece would be needed: the training thread copies
            }
        }

        lock (_slotGate)
        {
            _freeSlots.Clear();
            for (int slot = (_staging is null ? 0 : _stagingSlots) - 1; slot >= 0; slot--)
            {
                _freeSlots.Push(slot);
            }
        }
    }

    public override void OverlapLost(string reason) => _lost = reason;

    public override GradientSyncResult Reduce(double tokens, double loss, bool normalized = false)
    {
        Count(tokens);
        double backward = _backwardStart != 0 ? BucketSizing.Seconds(_backwardStart) : 0;
        _backwardStart = 0;
        if (World == 1)
        {
            EndOverlap();
            return new(tokens, loss);
        }

        if (!Present)
        {
            throw new InvalidOperationException($"This rank sits out step {CurrentStep} (IGradientSync.BeginStep returned false): it reduces nothing that step.");
        }

        EnsurePartitioned();
        if (!_overlapping)
        {
            ResetShardStep();
        }

        bool advising = AdviceStep;
        if (advising && StepsInGroup == 1L + Options.SuggestSteps && _meter!.SliceCount == 0)
        {
            _meter.Capture = true;                             // the slice from the last measured step's own values (unless taken during its backward pass)
        }

        HostGradients?.DownloadGradients();
        long started = Stopwatch.GetTimestamp();
        _sent.Clear();
        int overlapped = 0;
        GradientSyncResult result;
        try
        {
            if (_overlapping)
            {
                if (tokens != _tokens || normalized != _normalized)
                {
                    throw new ArgumentException($"Reduce was given {tokens} tokens (normalized: {normalized}), the step's Overlap {_tokens} (normalized: {_normalized}): give both the same.", nameof(tokens));
                }

                overlapped = _overlapped;
                result = ReduceOverlapped(tokens, loss, normalized, backward);
            }
            else
            {
                result = ReduceAfter(tokens, loss, normalized, backward);
            }
        }
        finally
        {
            EndOverlap();
        }

        AfterShardedReduce();
        var reduceWall = Stopwatch.GetElapsedTime(started);
        var network = TimeSpan.Zero;
        bool onClock = Values.SimulatedTime > TimeSpan.Zero;
        double end = 0;
        foreach (var (readyAt, simulated, wall) in _sent)
        {
            var took = onClock ? simulated : wall;
            network += took;
            end = Math.Max(end, readyAt) + took.TotalSeconds;
        }

        // On a simulated transport the wait after the backward pass is predicted from the buckets' simulated times and when
        // each was ready; on a real one it is the time Reduce took.
        var exposed = onClock ? TimeSpan.FromSeconds(Math.Max(0, end - backward)) : reduceWall;
        if (advising)
        {
            _meter!.Pending = (exposed, network, TimeSpan.FromSeconds(backward), reduceWall, onClock);
        }

        Publish(_compressor.Name, _layout.Count, overlapped, (long)_layout.BucketValues * sizeof(float), (long)_layout.Total * sizeof(float),
            TimeSpan.FromSeconds(backward), network, exposed, overlapped > 0 ? null : _lost ?? (backward > 0 && _layout.Count == 1 ? "one bucket holds every gradient" : null));
        _lost = null;
        if (!_sized && result.Tokens > 0)
        {
            Measure(backward);
        }

        return result;
    }

    public override void Dispose()
    {
        EndOverlap();
        _worker?.Dispose();
        _worker = null;
        _staging?.Dispose();
        _staging = null;
        _scratch?.Dispose();
        _scratch = null;
        _compressor.Dispose();
        StopAdvice();
        DisposeShards();
        base.Dispose();
    }

    protected override void ShareState(int holder)
    {
        StopAdvice();                                          // a group that changed measures nothing more (every member stops alike)
        // The holder's bucket size, so a rank that joins after the first step's measurement lays its buckets out alike.
        var (high, low) = Split(_layout.BucketValues);
        Span<float> state = [_sized ? 1f : 0f, high, low];
        Values.Broadcast(state, holder);
        _sized = state[0] > 0;
        int values = (int)Join(state[1], state[2]);
        if (values != _layout.BucketValues)
        {
            _layout = NewLayout(values);
        }
    }

    // A layout of `values` a bucket: grouped by owner in a sharded run that has its owners.
    private BucketLayout NewLayout(int values) => new(Parameters, values, _owners, _owners is null ? 1 : World);

    private static int ValuesOf(long bytes) => (int)Math.Clamp(bytes / sizeof(float), 1, Array.MaxLength / 2);

    private static float Weight(double tokens, bool normalized) => tokens > 0 ? (normalized ? 1f : (float)tokens) : 0f;

    // The header goes first, then each bucket is reduced, divided by the step's tokens and written back in turn through
    // one pooled buffer.
    private GradientSyncResult ReduceAfter(double tokens, double loss, bool normalized, double backward)
    {
        float weight = Weight(tokens, normalized);
        int header = HeaderLength;
        var buffer = ArrayPool<float>.Shared.Rent(Math.Max(_layout.BucketValues, header));
        try
        {
            var head = buffer.AsSpan(0, header);
            WriteHeader(head, tokens, loss, normalized, changed: null);
            var headMark = Mark();
            Values.AllReduce(head, WireType.Float32);
            _header = Sent(backward, headMark);
            var (total, losses) = ReadHeader(head);
            if (total <= 0)
            {
                return new(0, 0);
            }

            float divide = normalized ? 1f : (float)(1 / total);
            for (int b = 0; b < _layout.Count; b++)
            {
                var bucket = buffer.AsSpan(0, _layout.Size(b));
                Gather(b, bucket);
                Gathered(b);
                var mark = Mark();
                ReduceBucket(b, bucket, weight);
                Sent(backward, mark);
                Divide(b, bucket, divide);
                Scatter(b, bucket);
            }

            return new(total, normalized ? losses : losses / total);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
        }
    }

    // The buckets the backward pass did not finish are sent now, after those it did; the header goes last (with every
    // rank's changed buckets), a changed bucket is summed again, and every bucket is divided and written back.
    private GradientSyncResult ReduceOverlapped(double tokens, double loss, bool normalized, double backward)
    {
        _hook?.Dispose();
        _hook = null;
        _worker?.Wait();
        var held = _held!;
        foreach (var took in _took.AsSpan(0, _next))
        {
            _sent.Add(took);
        }

        for (int b = _next; b < _layout.Count; b++)
        {
            var bucket = held.AsSpan(_layout.Start(b), _layout.Size(b));
            Gather(b, bucket);
            Gathered(b);
            var mark = Mark();
            ReduceBucket(b, bucket, _weight);
            Sent(backward, mark);
        }

        var head = held.AsSpan(_layout.Total, HeaderLength);
        WriteHeader(head, tokens, loss, normalized, _changed);
        var headMark = Mark();
        Values.AllReduce(head, WireType.Float32);
        _header = Sent(backward, headMark);
        var (total, losses) = ReadHeader(head);
        for (int b = 0; b < _layout.Count; b++)
        {
            if (head[HeaderFloats + Parameters.Count + b] > 0f)
            {
                // A gradient changed after its bucket went: the bucket again, whole and uncompressed, on every rank.
                CheckNotReleased(b);
                var bucket = held.AsSpan(_layout.Start(b), _layout.Size(b));
                Gather(b, bucket);
                _compressor.Forget(_layout.Pieces(b));
                DenseCompressor.Scale(bucket, _weight);
                var mark = Mark();
                if (_layout.Sharded)
                {
                    Values.ReduceScatter(bucket, _layout.Counts(b), WireType.Float32);
                }
                else
                {
                    Values.AllReduce(bucket, WireType.Float32);
                }

                Sent(backward, mark);
            }
        }

        if (total <= 0)
        {
            return new(0, 0);
        }

        float divide = normalized ? 1f : (float)(1 / total);
        for (int b = 0; b < _layout.Count; b++)
        {
            var bucket = held.AsSpan(_layout.Start(b), _layout.Size(b));
            Divide(b, bucket, divide);
            Scatter(b, bucket);
        }

        return new(total, normalized ? losses : losses / total);
    }

    // A collective of the step is timed from its mark (the clock and the wall); it was ready at `readyAt`. Returns its place.
    private (TimeSpan Simulated, long Wall) Mark() => (Values.SimulatedTime, Stopwatch.GetTimestamp());

    private int Sent(double readyAt, (TimeSpan Simulated, long Wall) mark)
    {
        _sent.Add((readyAt, Values.SimulatedTime - mark.Simulated, Stopwatch.GetElapsedTime(mark.Wall)));
        return _sent.Count - 1;
    }

    // A parameter's gradient is finished in the backward pass (the training thread): a bucket whose parameters all are
    // goes to the worker, in bucket order. A gradient that changes after its bucket went marks the bucket; one finished in
    // a pass nested in another's (a checkpointed block's) is not final until the outer pass ends.
    private void Finished(Tensor leaf, bool outermost)
    {
        if (!_overlapping || !_index.TryGetValue(leaf, out int p))
        {
            return;
        }

        var buckets = _layout.BucketsOf(p);
        if (_ready[p] || !outermost)
        {
            foreach (int b in buckets)
            {
                _changed[b] |= _launched[b];
            }

            return;
        }

        if (_layout.Sharded && _released[p])
        {
            _releasedThenChanged ??= p;                            // its gradient was dropped when its bucket went (named at Reduce)
        }

        _ready[p] = true;
        foreach (int b in buckets)
        {
            _pending[b]--;
        }

        while (_next < _layout.Count && _pending[_next] == 0)
        {
            Launch(_next++);
        }
    }

    private void Launch(int bucket)
    {
        long start = Stopwatch.GetTimestamp();
        if (_staging is not null)
        {
            Stage(bucket);                                         // queued behind the backward pass; the worker waits for it
        }
        else
        {
            Gather(bucket, _held!.AsSpan(_layout.Start(bucket), _layout.Size(bucket)));
        }

        Gathered(bucket);
        _took[bucket] = (BucketSizing.Seconds(_backwardStart), TimeSpan.Zero, TimeSpan.Zero);
        _launched[bucket] = true;
        _overlapped++;
        _worker ??= new OverlapWorker(ReduceQueued, $"rank {Collectives.Rank} gradient buckets");
        _worker.Queue(bucket);
        _launchTicks += Stopwatch.GetTimestamp() - start;
    }

    // A finished bucket's copy to a pinned slot, queued on the device behind the work that wrote its gradients; the pieces
    // without a gradient are noted, to be zeros on the host. Waits for a free slot only when every slot is in flight.
    private void Stage(int bucket)
    {
        int slot;
        lock (_slotGate)
        {
            while (_freeSlots.Count == 0)
            {
                Monitor.Wait(_slotGate);
            }

            slot = _freeSlots.Pop();
        }

        var zeros = _zeros[bucket] ??= [];
        zeros.Clear();
        foreach (var (parameter, offset, n, at) in _layout.Pieces(bucket))
        {
            if (!HasGradient(parameter))
            {
                zeros.Add((at, n));
                continue;
            }

            _staging!.Download(Parameters[parameter].Grad!.Storage, offset, slot, at, n);
        }

        _slotOf[bucket] = slot;
    }

    // The worker's part: a staged bucket's copy waited for and taken from its slot, then the bucket through the
    // compression (or its reduce-scatter), timed.
    private void ReduceQueued(int bucket)
    {
        var values = _held!.AsSpan(_layout.Start(bucket), _layout.Size(bucket));
        if (_slotOf[bucket] is var slot and >= 0)
        {
            long waited = Stopwatch.GetTimestamp();
            _staging!.Wait(slot);
            Interlocked.Add(ref _stagingWaitTicks, Stopwatch.GetTimestamp() - waited);
            _staging.Slot(slot)[..values.Length].CopyTo(values);
            foreach (var (at, n) in _zeros[bucket]!)
            {
                values.Slice(at, n).Clear();
            }

            _slotOf[bucket] = -1;
            lock (_slotGate)
            {
                _freeSlots.Push(slot);
                Monitor.PulseAll(_slotGate);
            }
        }

        var simulated = Values.SimulatedTime;
        long wall = Stopwatch.GetTimestamp();
        ReduceBucket(bucket, values, _weight);
        _took[bucket] = (_took[bucket].ReadyAt, Values.SimulatedTime - simulated, Stopwatch.GetElapsedTime(wall));
    }

    // One bucket summed across the ranks: through the run's compression, or (sharded) weighted and reduce-scattered, each
    // rank's piece of it its own parameters' values.
    private void ReduceBucket(int bucket, Span<float> values, float weight)
    {
        if (bucket == 0 && _meter is { Capture: true } meter)
        {
            CaptureSlice(meter, values, weight);
        }

        if (!_layout.Sharded)
        {
            _compressor.Reduce(values, weight, _layout.Pieces(bucket));
            return;
        }

        DenseCompressor.Scale(values, weight);
        Values.ReduceScatter(values, _layout.Counts(bucket), _wire);
    }

    // values /= the step's tokens: the whole bucket, or (sharded) this rank's piece, the only one it writes back.
    private void Divide(int bucket, Span<float> values, float divide)
    {
        if (_layout.Sharded)
        {
            var (start, count) = _layout.Piece(bucket, Collectives.Rank);
            DenseCompressor.Scale(values.Slice(start, count), divide);
            return;
        }

        DenseCompressor.Scale(values, divide);
    }

    private void EndOverlap()
    {
        _hook?.Dispose();
        _hook = null;
        if (_overlapping)
        {
            _overlapping = false;
            try
            {
                _worker?.Wait();                               // nothing of this step is left running
            }
            catch (InvalidOperationException)
            {
                // Reported by the Reduce that failed.
            }
        }
    }

    /// <summary>The kind of entry the measured bucket size is kept under in the tuning cache.</summary>
    internal const string BucketKind = "bucket";

    // What a kept bucket size was measured for: the synchronisation, the algorithm, the ranks, the device kind, the gradients'
    // values and the compression (no spaces: the cache's lines are split on them).
    private string BucketKey => string.Create(CultureInfo.InvariantCulture,
        $"{Name}/{Collectives.Name}/{World}/{Collectives.Device.Type}/{_layout.Total}/{_compressor.Name}").Replace(' ', '_');

    // Over a real transport an unset bucket size kept by an earlier run of the same shape is used from the first step, when
    // every rank kept the same (each machine has its own cache); on a virtual clock it is measured each run (cheap and the
    // same every time, from the clock).
    protected override void Started()
    {
        // Ranks in processes of their own decide the advisor's measurement together (any rank's listener turns it on for
        // every rank): in the bucket size's agreement when there is one, else in one float of its own (real transports only).
        bool crossProcess = Collectives.Transport?.Capabilities.CrossProcess == true;
        if (_sized || Values.SimulatedTime > TimeSpan.Zero || Collectives.Transport?.SimulatedCriticalRank is not null)
        {
            if (crossProcess)
            {
                StartAdvice(AgreeAdvice());
            }

            return;
        }

        long kept = ClusterCache.FindTuning(BucketKind, BucketKey, Options.TuningFile) is { Value: > 0 and var bytes } ? ValuesOf(bytes) : 0;
        Span<float> agree = [kept >> 20, kept & ((1 << 20) - 1), -(kept >> 20), -(kept & ((1 << 20) - 1)), AdviceWanted ? 1f : 0f];   // each part exact in float32
        Values.AllReduce(agree, WireType.Float32, ReduceOp.Max);
        if (crossProcess)
        {
            StartAdvice(agree[4] > 0f);
        }

        if (kept > 0 && agree[0] == -agree[2] && agree[1] == -agree[3])
        {
            _sized = true;
            if (kept != _layout.BucketValues)
            {
                _layout = NewLayout((int)kept);
            }
        }
    }

    // The first step's figures, the largest of each over every rank, choose the bucket size from the next step on: the
    // backward pass's time, the latency (a tiny collective timed after another, which takes any transfer still queued on the
    // links) and a bucket's time (the slowest, the first collective after the backward pass aside unless it is the only one).
    private void Measure(double backward)
    {
        bool onClock = Values.SimulatedTime > TimeSpan.Zero;
        double Seconds(TimeSpan simulated, TimeSpan wall) => (onClock ? simulated : wall).TotalSeconds;
        var buckets = Enumerable.Range(0, _sent.Count).Where(i => i != _header).ToList();
        double bucketSeconds = buckets.Skip(buckets.Count > 1 ? 1 : 0).Select(i => Seconds(_sent[i].Simulated, _sent[i].Wall)).DefaultIfEmpty(0).Max();
        int bucketValues = 0;
        for (int b = 0; b < _layout.Count; b++)
        {
            bucketValues = Math.Max(bucketValues, _layout.Size(b));
        }

        Span<float> probe = [0f];
        Values.AllReduce(probe, WireType.Float32);
        var simulated = Values.SimulatedTime;
        long wall = Stopwatch.GetTimestamp();
        Values.AllReduce(probe, WireType.Float32);
        var probeWall = Stopwatch.GetElapsedTime(wall);
        double latency = Seconds(Values.SimulatedTime - simulated, probeWall);

        // What a collective's messages cost beyond their bytes: set, else on the simulator the wall time of the tiny probe
        // here (the transport's own work per message on this machine), agreed over the ranks and kept for the process, so a
        // seed's runs choose alike; a real transport's timed latency holds it already.
        string key = $"{Collectives.Name}/{World}/{Collectives.Device.Type}";
        bool known = MessageCosts.TryGetValue(key, out float kept);
        float messageCost = Options.MessageCost is { } cost ? (float)cost.TotalSeconds : !onClock ? 0f : known ? kept : (float)probeWall.TotalSeconds;
        Span<float> figures = [onClock ? -1f : (float)backward, (float)latency, (float)bucketSeconds, messageCost];
        Values.AllReduce(figures, WireType.Float32, ReduceOp.Max);
        if (onClock && Options.MessageCost is null && !known)
        {
            MessageCosts.TryAdd(key, figures[3]);
        }

        long headerValues = 2L * (World - 1) * WireFormat.HeaderBytes / sizeof(float);   // a ring's messages each carry a header
        int values = BucketSizing.Choose(_layout.Total, figures[0], figures[1], figures[2], bucketValues, figures[3], headerValues);
        _sized = true;
        if (!onClock)
        {
            // Kept for the next run of this shape (and shown by idrak tuning show) with the figures that chose it.
            var inv = CultureInfo.InvariantCulture;
            string detail = $"backward {(figures[0] * 1e3).ToString("F2", inv)} ms, latency {(figures[1] * 1e6).ToString("F0", inv)} us, "
                            + $"a bucket of {((long)bucketValues * sizeof(float)).ToString(inv)} B {(figures[2] * 1e3).ToString("F2", inv)} ms, "
                            + $"{((_layout.Total + values - 1) / values).ToString(inv)} bucket(s)";
            string bucketKey = BucketKey;
            try
            {
                ClusterCache.Keep(new ClusterTuning(BucketKind, bucketKey, (long)values * sizeof(float), DateTimeOffset.Now, detail), Options.TuningFile);
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                // A cache that cannot be written: the next run measures again.
            }
        }

        if (values != _layout.BucketValues)
        {
            _layout = NewLayout(values);
        }
    }

    private void WriteHeader(Span<float> head, double tokens, double loss, bool normalized, bool[]? changed)
    {
        bool own = tokens > 0;
        head[0] = (float)tokens;
        head[1] = own ? (float)(normalized ? loss : loss * tokens) : 0f;
        for (int i = 0; i < Parameters.Count; i++)
        {
            head[HeaderFloats + i] = HadGradient(i) ? 1f : 0f;
        }

        var flags = head[(HeaderFloats + Parameters.Count)..];
        for (int b = 0; b < flags.Length; b++)
        {
            flags[b] = changed is not null && b < changed.Length && changed[b] ? 1f : 0f;
        }
    }

    // The step's tokens and loss sum, and which parameters have a gradient on any rank.
    private (double Total, double Losses) ReadHeader(ReadOnlySpan<float> head)
    {
        for (int i = 0; i < _present.Length; i++)
        {
            _present[i] = head[HeaderFloats + i] > 0f;
        }

        return (head[0], head[1]);
    }

    private bool HasGradient(int parameter) => HostGradients is { } host ? !host.HostGradient(parameter).IsEmpty : Parameters[parameter].Grad is not null;

    // This rank's values of a bucket's pieces (zeros where it has no gradient).
    private void Gather(int bucket, Span<float> values)
    {
        foreach (var (parameter, offset, n, at) in _layout.Pieces(bucket))
        {
            var target = values.Slice(at, n);
            if (!HasGradient(parameter))
            {
                target.Clear();
                continue;
            }

            if (HostGradients is { } host)
            {
                host.HostGradient(parameter).Slice(offset, n).CopyTo(target);
                continue;
            }

            var gradient = Parameters[parameter].Grad!;
            if (offset == 0 && n == gradient.Size)
            {
                gradient.Backend.Download(gradient.Storage, target);
            }
            else
            {
                gradient.Backend.DownloadRange(gradient.Storage, offset, target);
            }
        }
    }

    // The reduced values back into every piece's gradient (made where this rank had none and another rank had one); in a
    // sharded run only the pieces of the parameters this rank owns.
    private void Scatter(int bucket, ReadOnlySpan<float> values)
    {
        foreach (var (parameter, offset, n, at) in _layout.Pieces(bucket))
        {
            var source = values.Slice(at, n);
            if (!_present[parameter] || _layout.Sharded && _owners![parameter] != Collectives.Rank)
            {
                continue;                                          // no rank has a gradient for it (the optimizer skips it everywhere), or another rank owns it
            }

            if (HostGradients is { } host)
            {
                var target = host.HostGradient(parameter);
                if (target.IsEmpty)
                {
                    target = host.CreateHostGradient(parameter);
                }

                source.CopyTo(target.Slice(offset, n));
                continue;
            }

            var p = Parameters[parameter];
            Write(p, p.GradientTarget(out _), offset, source);     // every piece of it is written: no zero fill needed
        }
    }

    // `source` into values [offset, offset + source.Length) of `storage` (a parameter's or its gradient's, on its device).
    private void Write(Tensor p, Storage storage, int offset, ReadOnlySpan<float> source)
    {
        int n = source.Length;
        var backend = p.Backend;
        if (offset == 0 && n == p.Size)
        {
            backend.Upload(source, storage);
        }
        else if (p.Device.Type == DeviceType.Cpu)
        {
            source.CopyTo(storage.HostMemory.Slice(offset, n));
        }
        else
        {
            // A piece of a parameter larger than a bucket: up into the staging buffer, then one strided copy into place.
            var scratch = Scratch(n);
            backend.Upload(source, scratch.Storage);
            backend.Copy2D(scratch.Storage, 0, n, storage, offset, n, 1, n, accumulate: false);
        }
    }

    // Values [offset, offset + target.Length) of `storage` (on `p`'s device) into `target`.
    private static void Read(Tensor p, Storage storage, int offset, Span<float> target)
    {
        if (offset == 0 && target.Length == p.Size)
        {
            p.Backend.Download(storage, target);
        }
        else
        {
            p.Backend.DownloadRange(storage, offset, target);
        }
    }

    private Tensor Scratch(int values)
    {
        if (_scratch is null || _scratch.Size < values)
        {
            _scratch?.Dispose();
            _scratch = Tensor.PersistentZeros([Math.Max(values, _layout.BucketValues)], Collectives.Device);
        }

        return _scratch;
    }

    // Ends the hook when the trainer's backward pass is over (Reduce does the rest).
    private sealed class OverlapScope(GradientReduction owner) : IDisposable
    {
        public void Dispose()
        {
            owner._hook?.Dispose();
            owner._hook = null;
        }
    }
}
