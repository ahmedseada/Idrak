// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using Idrak.Networking.Abstractions;

namespace Idrak.Networking.Training;

/// <summary>"all-reduce" in <see cref="GradientSyncs"/>: every step's gradients summed across the ranks after the backward pass.</summary>
internal sealed class AllReduceSyncProvider : IGradientSyncProvider
{
    public string Name => "all-reduce";

    public string Summary => "every step's gradients summed across the ranks after the backward pass, token-weighted, a bucket at a time";

    public IGradientSync Create(ICollectives collectives, Optimizer optimizer, ClusterOptions options) => new GradientReduction(collectives, optimizer, options);
}

/// <summary>
/// The gradient all-reduce of data-parallel training (plan 23, phase C2). The gradients are addressed as (parameter,
/// offset, count) pieces in parameter order and reduced a bucket at a time (<see cref="ClusterOptions.BucketBytes"/>)
/// through one pooled host buffer, returned after every step: no packed copy on the device, and device memory that does
/// not grow (rule 77). The step's header (this rank's tokens, its loss and which parameters have a gradient here) travels
/// in the first bucket on a float32 wire, so a step is one collective per bucket; on a bfloat16 wire the header goes
/// first on its own in float32, so no count is rounded. The sum is token-weighted: each rank's gradient, a mean over its
/// own tokens, is weighted by them, so the result is the gradient one process would compute on every rank's rows. An
/// optimizer that holds the gradients on the host (<see cref="IHostGradients"/>, the HostOptimizer) is reduced there, on
/// the pieces it downloaded.
/// </summary>
internal sealed class GradientReduction : IGradientSync
{
    private const int HeaderFloats = 2;                       // tokens, loss; then one presence flag per parameter
    private const long DefaultBucketBytes = 16L << 20;        // until phase C4 measures it (IDRAK_CLUSTER_BUCKET_BYTES)

    private readonly ClusterOptions _options;
    private readonly HostCollectives? _host;                  // the library's collectives: buckets reduced straight from the host buffer
    private readonly IHostGradients? _hostGradients;
    private readonly int _bucket;                             // values a bucket holds
    private readonly bool _piggyback;                         // the header rides in the first bucket
    private readonly List<(int Parameter, int Offset, int Count)> _pieces = [];
    private readonly List<(int First, int Count, int Values)> _buckets = [];
    private readonly bool[] _present;
    private readonly int _largest;                            // the most values one collective of a step carries
    private double[] _shares;
    private Tensor? _scratch;                                 // device staging, made on first need and kept

    private long _steps;
    private long _mark;
    private double _measuredTokens, _measuredSeconds;
    private long _nextExchange;

    public GradientReduction(ICollectives collectives, Optimizer optimizer, ClusterOptions options)
    {
        Collectives = collectives ?? throw new ArgumentNullException(nameof(collectives));
        ArgumentNullException.ThrowIfNull(optimizer);
        _options = options ?? throw new ArgumentNullException(nameof(options));
        Parameters = optimizer.Parameters;
        _host = collectives as HostCollectives;
        _hostGradients = optimizer as IHostGradients;
        _bucket = (int)Math.Clamp((options.BucketBytes ?? DefaultBucketBytes) / sizeof(float), 1, Array.MaxLength / 2);
        _present = new bool[Parameters.Count];
        _shares = [.. DrawShares.Even(collectives.WorldSize)];
        _nextExchange = options.RebalanceSteps > 0 ? 1 + options.RebalanceSteps : long.MaxValue;
        for (int i = 0; i < Parameters.Count; i++)
        {
            if (!ReferenceEquals(Parameters[i].Backend, collectives.Device.Backend))
            {
                throw new ArgumentException($"Parameter {i} {Tensor.FormatShape(Parameters[i].Shape)} is on {Parameters[i].Device}; rank {collectives.Rank}'s collectives are on {collectives.Device}: "
                                            + "train each rank's model on its rank's device.", nameof(optimizer));
            }
        }

        int header = HeaderFloats + Parameters.Count;
        _piggyback = options.Wire == WireType.Float32 && header < _bucket;
        Layout(_piggyback ? _bucket - header : _bucket);
        _largest = Math.Max(header + (_piggyback ? _buckets[0].Values : 0), _buckets.Max(b => b.Values));
    }

    public string Name => "all-reduce";

    public ICollectives Collectives { get; }

    public IReadOnlyList<Tensor> Parameters { get; }

    public IReadOnlyList<double> Shares => _shares;

    private int World => Collectives.WorldSize;

    public void Start()
    {
        if (World > 1)
        {
            if (_options.StartingState == StartingState.Check)
            {
                CheckSame();
            }
            else
            {
                Broadcast(Parameters);
                _hostGradients?.ReloadParameters();
            }
        }

        _mark = Stopwatch.GetTimestamp();
    }

    public GradientSyncResult Reduce(double tokens, double loss, bool normalized = false)
    {
        if (!(tokens >= 0) || double.IsInfinity(tokens))
        {
            throw new ArgumentOutOfRangeException(nameof(tokens), tokens, "A rank's tokens are 0 or more.");
        }

        long now = Stopwatch.GetTimestamp();
        if (++_steps > 1 && _mark != 0)
        {
            _measuredSeconds += Stopwatch.GetElapsedTime(_mark, now).TotalSeconds;   // the first, cold step aside (rule 76)
            _measuredTokens += tokens;
        }

        if (World == 1)
        {
            return new(tokens, loss);
        }

        _hostGradients?.DownloadGradients();
        bool own = tokens > 0;
        int count = Parameters.Count, header = HeaderFloats + count;
        var buffer = ArrayPool<float>.Shared.Rent(Math.Max(_bucket, header));
        try
        {
            buffer[0] = (float)tokens;
            buffer[1] = own ? (float)(normalized ? loss : loss * tokens) : 0f;
            for (int i = 0; i < count; i++)
            {
                buffer[HeaderFloats + i] = HasGradient(i) ? 1f : 0f;
            }

            int first = 0;
            double total, losses;
            if (_piggyback)
            {
                // The header and the first bucket in one collective: each rank's gradients weighted by its tokens now, the
                // sum divided by the step's tokens once they are known.
                var (_, _, values) = _buckets[0];
                var bucket = buffer.AsSpan(0, header + values);
                Gather(0, bucket[header..], own ? (normalized ? 1f : (float)tokens) : 0f);
                AllReduce(bucket, WireType.Float32);
                (total, losses) = Header(bucket);
                if (total <= 0)
                {
                    return new(0, 0);
                }

                Scale(bucket[header..], normalized ? 1f : (float)(1 / total));
                Scatter(0, bucket[header..]);
                first = 1;
            }
            else
            {
                var head = buffer.AsSpan(0, header);
                AllReduce(head, WireType.Float32);
                (total, losses) = Header(head);
                if (total <= 0)
                {
                    return new(0, 0);
                }
            }

            // Every other bucket weighted before it travels: this rank's share of the step's tokens.
            float weight = own ? (normalized ? 1f : (float)(tokens / total)) : 0f;
            for (int b = first; b < _buckets.Count; b++)
            {
                var bucket = buffer.AsSpan(0, _buckets[b].Values);
                Gather(b, bucket, weight);
                AllReduce(bucket, _options.Wire);
                Scatter(b, bucket);
            }

            return new(total, normalized ? losses : losses / total);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
        }
    }

    public void Stepped(long step)
    {
        if (World > 1)
        {
            if (_options.SyncWeightsEvery > 0 && step > 0 && step % _options.SyncWeightsEvery == 0)
            {
                Broadcast(Parameters);                         // ranks of different device kinds round apart: rank 0's values
                _hostGradients?.ReloadParameters();
            }

            if (_steps >= _nextExchange)
            {
                Rebalance();
            }
        }

        _mark = Stopwatch.GetTimestamp();
    }

    public void Sum(Span<float> values)
    {
        if (World > 1 && values.Length > 0)
        {
            AllReduce(values, WireType.Float32);
        }
    }

    public void Broadcast(IReadOnlyList<Tensor> tensors, int root = 0)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        if (World == 1)
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
        if (World > 1)
        {
            Collectives.Barrier();
        }
    }

    public void Dispose()
    {
        _scratch?.Dispose();
        _scratch = null;
    }

    // The pieces and buckets: every parameter's values in order, cut where a bucket fills (the first holds less beside the
    // header when it rides along). The same on every rank: it depends on the parameters' sizes alone.
    private void Layout(int firstCapacity)
    {
        int capacity = firstCapacity, used = 0, firstPiece = 0;
        for (int i = 0; i < Parameters.Count; i++)
        {
            int size = Parameters[i].Size;
            for (int offset = 0; offset < size;)
            {
                if (used == capacity)
                {
                    _buckets.Add((firstPiece, _pieces.Count - firstPiece, used));
                    (firstPiece, used, capacity) = (_pieces.Count, 0, _bucket);
                }

                int n = Math.Min(size - offset, capacity - used);
                _pieces.Add((i, offset, n));
                (used, offset) = (used + n, offset + n);
            }
        }

        if (_pieces.Count > firstPiece || _buckets.Count == 0)
        {
            _buckets.Add((firstPiece, _pieces.Count - firstPiece, used));
        }
    }

    // The step's tokens and loss sum, and which parameters have a gradient on any rank.
    private (double Total, double Losses) Header(ReadOnlySpan<float> header)
    {
        for (int i = 0; i < _present.Length; i++)
        {
            _present[i] = header[HeaderFloats + i] > 0f;
        }

        return (header[0], header[1]);
    }

    private bool HasGradient(int parameter) => _hostGradients is { } host ? !host.HostGradient(parameter).IsEmpty : Parameters[parameter].Grad is not null;

    // This rank's values of a bucket's pieces, times `weight` (zeros where it has no gradient or no rows).
    private void Gather(int bucket, Span<float> values, float weight)
    {
        var (first, count, _) = _buckets[bucket];
        int at = 0;
        for (int k = first; k < first + count; k++)
        {
            var (parameter, offset, n) = _pieces[k];
            var target = values.Slice(at, n);
            at += n;
            if (weight == 0f || !HasGradient(parameter))
            {
                target.Clear();
                continue;
            }

            if (_hostGradients is { } host)
            {
                host.HostGradient(parameter).Slice(offset, n).CopyTo(target);
            }
            else
            {
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

            Scale(target, weight);
        }
    }

    // The reduced values back into every piece's gradient (made where this rank had none and another rank had one).
    private void Scatter(int bucket, ReadOnlySpan<float> values)
    {
        var (first, count, _) = _buckets[bucket];
        int at = 0;
        for (int k = first; k < first + count; k++)
        {
            var (parameter, offset, n) = _pieces[k];
            var source = values.Slice(at, n);
            at += n;
            if (!_present[parameter])
            {
                continue;                                          // no rank has a gradient for it: the optimizer skips it everywhere
            }

            if (_hostGradients is { } host)
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

    // A sum across the ranks of host values: straight from the host buffer with the library's collectives, through a
    // device buffer with another algorithm (a plug-in's collectives work on device storage).
    private void AllReduce(Span<float> values, WireType wire)
    {
        if (_host is { } host)
        {
            host.AllReduceHostValues(values, ReduceOp.Sum, wire);
            return;
        }

        var scratch = Scratch(values.Length);
        var backend = Collectives.Device.Backend;
        backend.Upload(values, scratch.Storage);
        Collectives.AllReduce(scratch.Storage, values.Length, ReduceOp.Sum, wire);
        backend.Download(scratch.Storage, values);
    }

    private Tensor Scratch(int values)
    {
        if (_scratch is null || _scratch.Size < values)
        {
            _scratch?.Dispose();
            _scratch = Tensor.PersistentZeros([Math.Max(values, _largest)], Collectives.Device);
        }

        return _scratch;
    }

    // Each rank's tokens per second over the steps since the last exchange, summed into one list every rank holds; the
    // shares follow them unless the ranks are within the tolerance of each other (then equal, so noise moves no rows).
    private void Rebalance()
    {
        var rates = new float[World];
        rates[Collectives.Rank] = _measuredTokens > 0 && _measuredSeconds > 0 ? (float)(_measuredTokens / _measuredSeconds) : 0f;
        AllReduce(rates, WireType.Float32);
        (_measuredTokens, _measuredSeconds) = (0, 0);
        _nextExchange = _options.RebalanceEvery > 0 ? _steps + _options.RebalanceEvery : long.MaxValue;
        float slowest = rates.Min(), fastest = rates.Max();
        if (!(slowest > 0))
        {
            return;                                                // a rank had no rows to time: the shares stay
        }

        if (fastest / slowest - 1 <= _options.RebalanceTolerance)
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
        int count = Parameters.Count;
        var mine = new float[Math.Max(1, count * 4)];
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

        using var input = Tensor.Persistent(mine, [mine.Length], Collectives.Device);
        using var output = Tensor.PersistentZeros([mine.Length * World], Collectives.Device);
        Collectives.AllGather(input.Storage, mine.Length, output.Storage);
        var all = output.ToArray();
        for (int r = 1; r < World; r++)
        {
            for (int i = 0; i < count; i++)
            {
                if (!all.AsSpan((r * mine.Length) + (i * 4), 4).SequenceEqual(all.AsSpan(i * 4, 4)))
                {
                    throw new InvalidOperationException($"Rank {r} starts from other values than rank 0 (parameter {i} of {count}, {Tensor.FormatShape(Parameters[i].Shape)}): "
                                                        + "give every rank the same seed and weights, or let rank 0's values be sent (IDRAK_CLUSTER_START=broadcast).");
                }
            }
        }
    }

    // values *= factor: a vector loop and a scalar tail.
    private static void Scale(Span<float> values, float factor)
    {
        if (factor == 1f)
        {
            return;
        }

        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            var f = new Vector<float>(factor);
            for (; i <= values.Length - Vector<float>.Count; i += Vector<float>.Count)
            {
                (new Vector<float>(values[i..]) * f).CopyTo(values[i..]);
            }
        }

        for (; i < values.Length; i++)
        {
            values[i] *= factor;
        }
    }
}
