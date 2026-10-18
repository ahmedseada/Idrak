// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Diagnostics;
using Idrak.Networking.Abstractions;
using Idrak.Networking.Compression;

namespace Idrak.Networking.Training;

/// <summary>"local-sgd" in <see cref="GradientSyncs"/>: each rank steps on its own gradients; the weights are averaged every k steps.</summary>
internal sealed class LocalSgdProvider : IGradientSyncProvider
{
    public string Name => "local-sgd";

    public string Summary => "each rank steps on its own gradients; the weights (not the gradients) are averaged every IDRAK_CLUSTER_LOCAL_SGD steps, weighted by each rank's rows";

    public IGradientSync Create(ICollectives collectives, Optimizer optimizer, ClusterOptions options) => new LocalSgd(collectives, optimizer, options);
}

/// <summary>
/// Local SGD (plan 23, phase C4): every rank updates with the gradient of its own rows, and every
/// <see cref="ClusterOptions.LocalSgdEvery"/> steps (k) the ranks' weights are averaged, each weighted by the rows it
/// trained since the last average, so the weights travel once in k steps instead of the gradients every step. A step
/// still sums one small header (each rank's rows and loss), so every rank reports the step's loss over every rank's rows
/// and the ranks end an epoch together. At k = 1 with plain SGD the average of the ranks' updated weights is the update
/// one process makes with the global batch (within float rounding). Each rank keeps its own optimizer state; the weights
/// travel as float32 a bucket of <see cref="ClusterOptions.BucketBytes"/> at a time (every value in one when unset)
/// through one pooled buffer.
/// </summary>
internal sealed class LocalSgd : SyncCore
{
    private readonly int _every;
    private readonly BucketLayout _layout;
    private double _mine, _all;                               // rows since the last average: this rank's and every rank's

    public LocalSgd(ICollectives collectives, Optimizer optimizer, ClusterOptions options)
        : base(collectives, optimizer, options)
    {
        if (options.Shard != ShardMode.None)
        {
            throw new ArgumentException("Local SGD steps every rank's whole weights; it does not shard (unset IDRAK_CLUSTER_SHARD, or IDRAK_CLUSTER_LOCAL_SGD).", nameof(options));
        }

        _every = Math.Max(1, options.LocalSgdEvery);
        long total = Parameters.Sum(p => (long)p.Size);
        _layout = new BucketLayout(Parameters, (int)Math.Clamp((options.BucketBytes ?? total * sizeof(float)) / sizeof(float), 1, Array.MaxLength / 2));
    }

    public override string Name => "local-sgd";

    public override GradientSyncResult Reduce(double tokens, double loss, bool normalized = false)
    {
        Count(tokens);
        if (World == 1)
        {
            return new(tokens, loss);
        }

        if (!Present)
        {
            throw new InvalidOperationException($"This rank sits out step {CurrentStep} (IGradientSync.BeginStep returned false): it reduces nothing that step.");
        }

        var simulated = Values.SimulatedTime;
        long wall = Stopwatch.GetTimestamp();
        bool own = tokens > 0;
        Span<float> head = [(float)tokens, own ? (float)(normalized ? loss : loss * tokens) : 0f];
        Values.AllReduce(head, WireType.Float32);
        double total = head[0], losses = head[1];
        if (total <= 0)
        {
            return new(0, 0);
        }

        var took = Took(simulated, wall);
        Publish("none", 0, 0, 0, 0, TimeSpan.Zero, took, took, "local SGD: the weights travel every " + _every.ToString(System.Globalization.CultureInfo.InvariantCulture) + " steps");

        if (normalized && own)
        {
            // A loss of one's own is this rank's share of the step's (divided by every rank's counts): its own mean is the
            // share times the step's rows over its own.
            float scale = (float)(total / tokens);
            foreach (var p in Parameters)
            {
                p.Grad?.Scale(scale);
            }
        }

        (_mine, _all) = (_mine + tokens, _all + total);
        return new(total, normalized ? losses : losses / total);
    }

    public override void Stepped(long step)
    {
        if (World > 1 && Present && step > 0 && step % _every == 0 && _all > 0)
        {
            Average();
        }

        base.Stepped(step);
    }

    protected override void Rejoin(int holder)
    {
        (_mine, _all) = (0, 0);                                // the holder's weights are every rank's now
        base.Rejoin(holder);
    }

    // Every rank's weights times its share of the rows since the last average, summed bucket by bucket.
    private void Average()
    {
        float share = (float)(_mine / _all);
        var simulated = Values.SimulatedTime;
        long wall = Stopwatch.GetTimestamp();
        var buffer = ArrayPool<float>.Shared.Rent(_layout.BucketValues);
        try
        {
            for (int b = 0; b < _layout.Count; b++)
            {
                var bucket = buffer.AsSpan(0, _layout.Size(b));
                foreach (var (parameter, offset, n, at) in _layout.Pieces(b))
                {
                    var p = Parameters[parameter];
                    p.Backend.DownloadRange(p.Storage, offset, bucket.Slice(at, n));
                }

                DenseCompressor.Scale(bucket, share);
                Values.AllReduce(bucket, WireType.Float32);
                foreach (var (parameter, offset, n, at) in _layout.Pieces(b))
                {
                    var p = Parameters[parameter];
                    if (offset == 0 && n == p.Size)
                    {
                        p.Backend.Upload(bucket.Slice(at, n), p.Storage);
                    }
                    else
                    {
                        p.Load(Merge(p, offset, bucket.Slice(at, n)));
                    }
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
        }

        (_mine, _all) = (0, 0);
        HostGradients?.ReloadParameters();
        var took = Took(simulated, wall);
        Publish("none", _layout.Count, 0, (long)_layout.BucketValues * sizeof(float), (long)_layout.Total * sizeof(float), TimeSpan.Zero, took, took, null);
    }

    // A parameter larger than a bucket: its values with one piece replaced (rare: a bucket smaller than a parameter).
    private static float[] Merge(Tensor p, int offset, ReadOnlySpan<float> piece)
    {
        var all = p.ToArray();
        piece.CopyTo(all.AsSpan(offset));
        return all;
    }

    private TimeSpan Took(TimeSpan simulated, long wall) =>
        Values.SimulatedTime - simulated is { Ticks: > 0 } onClock ? onClock : Stopwatch.GetElapsedTime(wall);
}
