// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Diagnostics;
using Idrak.Networking.Abstractions;
using Idrak.Networking.Compression;

namespace Idrak.Networking.Training;

/// <summary>"all-reduce" in <see cref="GradientSyncs"/>: every step's gradients summed across the ranks, a bucket at a time, each sent while the backward pass goes on.</summary>
internal sealed class AllReduceSyncProvider : IGradientSyncProvider
{
    public string Name => "all-reduce";

    public string Summary => "every step's gradients summed across the ranks, token-weighted, a bucket at a time, each sent while the backward pass goes on";

    public IGradientSync Create(ICollectives collectives, Optimizer optimizer, ClusterOptions options) => new GradientReduction(collectives, optimizer, options);
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
/// </summary>
internal sealed class GradientReduction : SyncCore
{
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

    public GradientReduction(ICollectives collectives, Optimizer optimizer, ClusterOptions options)
        : base(collectives, optimizer, options)
    {
        _compressor = GradientCompressions.Create(collectives, Parameters, options);
        _present = new bool[Parameters.Count];
        for (int i = 0; i < Parameters.Count; i++)
        {
            _index.TryAdd(Parameters[i], i);
        }

        _sized = options.BucketBytes is not null;
        _layout = new BucketLayout(Parameters, ValuesOf(options.BucketBytes ?? BucketSizing.ProbeBytes));
    }

    public override string Name => "all-reduce";

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
        _hook = Autograd.OnGradientReady(Finished);
        return new OverlapScope(this);
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
        var exposed = onClock ? TimeSpan.FromSeconds(Math.Max(0, end - backward)) : Stopwatch.GetElapsedTime(started);
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
        _scratch?.Dispose();
        _scratch = null;
        _compressor.Dispose();
        base.Dispose();
    }

    protected override void ShareState(int holder)
    {
        // The holder's bucket size, so a rank that joins after the first step's measurement lays its buckets out alike.
        var (high, low) = Split(_layout.BucketValues);
        Span<float> state = [_sized ? 1f : 0f, high, low];
        Values.Broadcast(state, holder);
        _sized = state[0] > 0;
        int values = (int)Join(state[1], state[2]);
        if (values != _layout.BucketValues)
        {
            _layout = new BucketLayout(Parameters, values);
        }
    }

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
                var mark = Mark();
                _compressor.Reduce(bucket, weight, _layout.Pieces(b));
                Sent(backward, mark);
                DenseCompressor.Scale(bucket, divide);
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
            var mark = Mark();
            _compressor.Reduce(bucket, _weight, _layout.Pieces(b));
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
                var bucket = held.AsSpan(_layout.Start(b), _layout.Size(b));
                Gather(b, bucket);
                _compressor.Forget(_layout.Pieces(b));
                DenseCompressor.Scale(bucket, _weight);
                var mark = Mark();
                Values.AllReduce(bucket, WireType.Float32);
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
            DenseCompressor.Scale(bucket, divide);
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
        Gather(bucket, _held!.AsSpan(_layout.Start(bucket), _layout.Size(bucket)));
        _took[bucket] = (BucketSizing.Seconds(_backwardStart), TimeSpan.Zero, TimeSpan.Zero);
        _launched[bucket] = true;
        _overlapped++;
        _worker ??= new OverlapWorker(ReduceQueued, $"rank {Collectives.Rank} gradient buckets");
        _worker.Queue(bucket);
    }

    // The worker's part: one bucket through the compression, timed.
    private void ReduceQueued(int bucket)
    {
        var simulated = Values.SimulatedTime;
        long wall = Stopwatch.GetTimestamp();
        _compressor.Reduce(_held!.AsSpan(_layout.Start(bucket), _layout.Size(bucket)), _weight, _layout.Pieces(bucket));
        _took[bucket] = (_took[bucket].ReadyAt, Values.SimulatedTime - simulated, Stopwatch.GetElapsedTime(wall));
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
        double latency = Seconds(Values.SimulatedTime - simulated, Stopwatch.GetElapsedTime(wall));
        Span<float> figures = [onClock ? -1f : (float)backward, (float)latency, (float)bucketSeconds];
        Values.AllReduce(figures, WireType.Float32, ReduceOp.Max);
        int values = BucketSizing.Choose(_layout.Total, figures[0], figures[1], figures[2], bucketValues);
        _sized = true;
        if (values != _layout.BucketValues)
        {
            _layout = new BucketLayout(Parameters, values);
        }
    }

    private void WriteHeader(Span<float> head, double tokens, double loss, bool normalized, bool[]? changed)
    {
        bool own = tokens > 0;
        head[0] = (float)tokens;
        head[1] = own ? (float)(normalized ? loss : loss * tokens) : 0f;
        for (int i = 0; i < Parameters.Count; i++)
        {
            head[HeaderFloats + i] = HasGradient(i) ? 1f : 0f;
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

    // The reduced values back into every piece's gradient (made where this rank had none and another rank had one).
    private void Scatter(int bucket, ReadOnlySpan<float> values)
    {
        foreach (var (parameter, offset, n, at) in _layout.Pieces(bucket))
        {
            var source = values.Slice(at, n);
            if (!_present[parameter])
            {
                continue;                                          // no rank has a gradient for it: the optimizer skips it everywhere
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
            var storage = p.GradientTarget(out _);                 // every piece of it is written: no zero fill needed
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
