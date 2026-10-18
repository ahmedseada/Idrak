// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Diagnostics;
using Idrak.Networking.Abstractions;

namespace Idrak.Networking.Training;

/// <summary>
/// The gradient buckets of a step (plan 23, phase C4): every parameter's values cut into pieces and packed into buckets of
/// a fixed number of values, in reverse parameter order (the order a backward pass finishes them: the last layers' first),
/// so the first bucket fills first. The layout depends on the parameters' sizes and the bucket size alone, so it is the
/// same on every rank. Each bucket knows the parameters it holds, so it can be sent as soon as all of them are finished.
/// </summary>
internal sealed class BucketLayout
{
    private readonly GradientPiece[] _pieces;
    private readonly (int First, int Count, int Start, int Values)[] _buckets;
    private readonly int[][] _bucketParameters;                 // the distinct parameters each bucket holds
    private readonly int[][] _parameterBuckets;                 // the buckets each parameter's pieces are in

    public BucketLayout(IReadOnlyList<Tensor> parameters, int bucketValues)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(bucketValues, 1);
        BucketValues = bucketValues;
        var pieces = new List<GradientPiece>();
        var buckets = new List<(int, int, int, int)>();
        int used = 0, firstPiece = 0, start = 0;
        for (int i = parameters.Count - 1; i >= 0; i--)
        {
            int size = parameters[i].Size;
            for (int offset = 0; offset < size;)
            {
                if (used == bucketValues)
                {
                    buckets.Add((firstPiece, pieces.Count - firstPiece, start, used));
                    (firstPiece, start, used) = (pieces.Count, start + used, 0);
                }

                int n = Math.Min(size - offset, bucketValues - used);
                pieces.Add(new GradientPiece(i, offset, n, used));
                (used, offset) = (used + n, offset + n);
            }
        }

        if (pieces.Count > firstPiece || buckets.Count == 0)
        {
            buckets.Add((firstPiece, pieces.Count - firstPiece, start, used));
        }

        _pieces = [.. pieces];
        _buckets = [.. buckets];
        Total = start + used;
        var byParameter = new List<int>[parameters.Count];
        _bucketParameters = new int[_buckets.Length][];
        for (int b = 0; b < _buckets.Length; b++)
        {
            var held = new List<int>();
            foreach (var piece in Pieces(b))
            {
                if (held.Count == 0 || held[^1] != piece.Parameter)
                {
                    held.Add(piece.Parameter);
                    (byParameter[piece.Parameter] ??= []).Add(b);
                }
            }

            _bucketParameters[b] = [.. held];
        }

        _parameterBuckets = [.. byParameter.Select(l => l?.ToArray() ?? [])];
    }

    /// <summary>The values a bucket holds (the last may hold fewer).</summary>
    public int BucketValues { get; }

    /// <summary>The buckets.</summary>
    public int Count => _buckets.Length;

    /// <summary>The values of every bucket together: every parameter's.</summary>
    public int Total { get; }

    /// <summary>Bucket <paramref name="bucket"/>'s pieces, in bucket order (<see cref="GradientPiece.At"/> from 0).</summary>
    public ReadOnlySpan<GradientPiece> Pieces(int bucket) => _pieces.AsSpan(_buckets[bucket].First, _buckets[bucket].Count);

    /// <summary>Where bucket <paramref name="bucket"/>'s values start among every bucket's.</summary>
    public int Start(int bucket) => _buckets[bucket].Start;

    /// <summary>The values bucket <paramref name="bucket"/> holds.</summary>
    public int Size(int bucket) => _buckets[bucket].Values;

    /// <summary>The distinct parameters bucket <paramref name="bucket"/> holds.</summary>
    public ReadOnlySpan<int> ParametersOf(int bucket) => _bucketParameters[bucket];

    /// <summary>The buckets parameter <paramref name="parameter"/>'s pieces are in.</summary>
    public ReadOnlySpan<int> BucketsOf(int parameter) => _parameterBuckets[parameter];
}

/// <summary>
/// The thread that reduces a step's buckets while the backward pass goes on (plan 23, phase C4): buckets are queued in
/// bucket order by the training thread and reduced here one at a time, in that order (every rank runs the same collectives
/// in the same order); each records its network time (the virtual clock's on a simulated transport, else the wall time).
/// The training thread waits for the queue to empty before it runs any other collective. An error stops the thread's work
/// and is thrown to the training thread when it waits.
/// </summary>
internal sealed class OverlapWorker : IDisposable
{
    private readonly BlockingCollection<int> _queue = new();
    private readonly Action<int> _reduce;
    private readonly Thread _thread;
    private readonly object _gate = new();                     // Monitor.Wait and PulseAll: a Lock has no wait
    private int _queued, _done;
    private Exception? _error;

    public OverlapWorker(Action<int> reduce, string name)
    {
        _reduce = reduce;
        _thread = new Thread(Run) { IsBackground = true, Name = name };
        _thread.Start();
    }

    /// <summary>Queues bucket <paramref name="bucket"/>.</summary>
    public void Queue(int bucket)
    {
        lock (_gate)
        {
            _queued++;
        }

        _queue.Add(bucket);
    }

    /// <summary>Waits until every queued bucket is reduced; throws the error that stopped the work, if any.</summary>
    public void Wait()
    {
        lock (_gate)
        {
            while (_done < _queued && _error is null)
            {
                Monitor.Wait(_gate);
            }

            if (_error is { } error)
            {
                _error = null;
                (_queued, _done) = (0, 0);
                throw new InvalidOperationException($"A gradient bucket reduced during the backward pass failed: {error.Message}", error);
            }
        }
    }

    public void Dispose()
    {
        _queue.CompleteAdding();
        _thread.Join();
        _queue.Dispose();
    }

    private void Run()
    {
        foreach (int bucket in _queue.GetConsumingEnumerable())
        {
            bool failed;
            lock (_gate)
            {
                failed = _error is not null;
            }

            if (!failed)
            {
                try
                {
                    _reduce(bucket);
                }
                catch (Exception e)
                {
                    lock (_gate)
                    {
                        _error = e;
                        Monitor.PulseAll(_gate);
                    }

                    continue;
                }
            }

            lock (_gate)
            {
                _done++;
                Monitor.PulseAll(_gate);
            }
        }
    }
}

/// <summary>
/// How big a gradient bucket is when <see cref="ClusterOptions.BucketBytes"/> is unset (plan 23, phase C4): measured once,
/// on the first step, never a fixed size. The first step runs buckets of a first probe size (the C2 default, a probe and
/// not a constant: <see cref="ProbeBytes"/>) and measures the backward pass's time, one tiny collective's (the link's
/// latency) and the largest bucket's (its rate); the ranks take the largest of each figure, so every rank chooses alike.
/// On a simulated transport the link's figures are the virtual clock's and the backward pass is not timed (see
/// <see cref="Choose"/>), so the choice is the same on every run of a seed.
/// The choice is the bucket count whose predicted wait after the backward pass is least, buckets being finished evenly
/// through the backward pass and sent one after another: a bucket costs the latency plus its bytes over the rate. A
/// bucket is also kept to what the link moves in a hundred latencies (beyond that one latency costs under 1% of a bucket's
/// time and only host memory grows), though never cut into more than <see cref="MostBuckets"/>.
/// </summary>
internal static class BucketSizing
{
    /// <summary>The first step's bucket bytes: a probe the measurement starts from, then replaced.</summary>
    public const long ProbeBytes = 16L << 20;

    /// <summary>The most buckets a step is cut into by measurement (a count, not a size: each costs a latency).</summary>
    public const int MostBuckets = 256;

    /// <summary>
    /// The values of a bucket for <paramref name="total"/> gradient values, from the measured figures (seconds). A
    /// <paramref name="backward"/> below 0 means the backward pass was not measured (a simulated transport, whose clock holds
    /// network time only): it is taken to last as long as the gradients' transfer, the case overlap is for, so the choice
    /// depends on the virtual clock alone and a simulated run of a seed stays the same, bit for bit.
    /// </summary>
    public static int Choose(long total, double backward, double latency, double bucketSeconds, long bucketValues)
    {
        if (total <= 1)
        {
            return (int)Math.Max(1, total);
        }

        double perValue = bucketSeconds > latency && bucketValues > 0 ? (bucketSeconds - latency) / bucketValues : 0;   // seconds a value takes on the link
        if (backward < 0)
        {
            backward = total * perValue;
        }

        int best = 1;
        double bestWait = double.PositiveInfinity;
        int most = (int)Math.Min(MostBuckets, total);
        for (int n = 1; n <= most; n++)
        {
            double size = Math.Ceiling(total / (double)n), each = latency + (size * perValue), end = 0;
            for (int i = 1; i <= n; i++)
            {
                end = Math.Max(end, backward * i / n) + each;
            }

            double wait = end - backward;
            if (wait < bestWait - 1e-12)
            {
                (best, bestWait) = (n, wait);
            }
        }

        long values = (total + best - 1) / best;
        if (perValue > 0 && latency > 0)
        {
            values = Math.Min(values, Math.Max((total + MostBuckets - 1) / MostBuckets, (long)(100 * latency / perValue)));
        }

        return (int)Math.Clamp(values, 1, Array.MaxLength / 2);
    }

    /// <summary>The seconds since <paramref name="since"/> (a <see cref="Stopwatch"/> timestamp).</summary>
    public static double Seconds(long since) => Stopwatch.GetElapsedTime(since).TotalSeconds;
}
