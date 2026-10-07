// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections;
using Idrak.Data.Abstractions;

namespace Idrak.Data;

/// <summary>
/// Cuts an <see cref="ISampleSource"/> (a <see cref="Dataset"/>, a file source, a view) or an <see cref="ISampleStream"/>
/// into mini-batches and moves them to a device. Each enumeration is one epoch, reshuffled when <see cref="Shuffle"/> is
/// on, with the <see cref="Transforms"/> applied to every sample. While the model trains on batch n, batch n+1 is
/// gathered on a worker thread (for batches large enough for that to pay off).
/// </summary>
/// <example>
/// <code>
/// var loader = new DataLoader(train, batchSize: 64, shuffle: true);
/// foreach (var batch in loader)
/// {
///     using (batch) { /* batch.Features, batch.Targets */ }
/// }
/// </code>
/// </example>
public sealed class DataLoader : IBatchSource
{
    /// <summary>Batches with at least this many floats are gathered on a background thread.</summary>
    private const int PrefetchThreshold = 16 * 1024;

    private readonly Random _random;
    private readonly int _transformSeed;
    private readonly int[] _featureShape, _targetShape;
    private readonly int _featureSize, _targetSize;
    private int _epoch;
    private int? _streamBatches, _streamSamples;

    /// <summary>Creates a loader.</summary>
    /// <param name="source">The samples (a <see cref="Dataset"/> or any other source).</param>
    /// <param name="batchSize">Samples per batch.</param>
    /// <param name="shuffle">Reorder samples every epoch (recommended for training).</param>
    /// <param name="dropLast">Skip the final, smaller batch.</param>
    /// <param name="device">Where batches are placed; defaults to <see cref="Device.Default"/>.</param>
    /// <param name="seed">Seed of the shuffles and the transforms, for reproducible runs.</param>
    public DataLoader(ISampleSource source, int batchSize = 32, bool shuffle = false, bool dropLast = false, Device? device = null, int? seed = null)
        : this(batchSize, dropLast, device, seed, (source ?? throw new ArgumentNullException(nameof(source))).FeatureShape, source.TargetShape)
    {
        Source = source;
        Shuffle = shuffle;
    }

    /// <summary>Creates a loader over samples read in order.</summary>
    /// <param name="stream">The samples.</param>
    /// <param name="batchSize">Samples per batch.</param>
    /// <param name="shuffleBuffer">
    /// Shuffle within a buffer of this many samples (0, the default, keeps the order): each batch takes samples at random
    /// from the buffer, which refills from the stream. The larger the buffer, the closer to a full shuffle.
    /// </param>
    /// <param name="dropLast">Skip the final, smaller batch.</param>
    /// <param name="device">Where batches are placed; defaults to <see cref="Device.Default"/>.</param>
    /// <param name="seed">Seed of the shuffles and the transforms, for reproducible runs.</param>
    public DataLoader(ISampleStream stream, int batchSize = 32, int shuffleBuffer = 0, bool dropLast = false, Device? device = null, int? seed = null)
        : this(batchSize, dropLast, device, seed, (stream ?? throw new ArgumentNullException(nameof(stream))).FeatureShape, stream.TargetShape)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(shuffleBuffer);
        Stream = stream;
        ShuffleBuffer = shuffleBuffer;
        Shuffle = shuffleBuffer > 1;
    }

    private DataLoader(int batchSize, bool dropLast, Device? device, int? seed, IReadOnlyList<int> featureShape, IReadOnlyList<int> targetShape)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);
        BatchSize = batchSize;
        DropLast = dropLast;
        Device = device ?? Device.Default;
        _random = seed is { } s ? new Random(s) : new Random();
        _transformSeed = seed ?? Random.Shared.Next();
        _featureShape = [.. featureShape];
        _targetShape = [.. targetShape];
        _featureSize = SampleSourceExtensions.Size(_featureShape);
        _targetSize = SampleSourceExtensions.Size(_targetShape);
    }

    /// <summary>The samples, or null when the loader reads a <see cref="Stream"/>.</summary>
    public ISampleSource? Source { get; }

    /// <summary>The samples read in order, or null when the loader reads a <see cref="Source"/>.</summary>
    public ISampleStream? Stream { get; }

    /// <summary>The source when it is a <see cref="Data.Dataset"/>, else null.</summary>
    public Dataset? Dataset => Source as Dataset;

    /// <summary>Samples per batch.</summary>
    public int BatchSize { get; }

    /// <summary>Whether samples are reordered every epoch (within <see cref="ShuffleBuffer"/> for a stream).</summary>
    public bool Shuffle { get; }

    /// <summary>For a stream: the samples shuffled at a time (0 when the order is kept).</summary>
    public int ShuffleBuffer { get; }

    /// <summary>Whether a final partial batch is skipped.</summary>
    public bool DropLast { get; }

    /// <summary>Where batch tensors are created.</summary>
    public Device Device { get; }

    /// <summary>
    /// Changes made to every sample as it is batched (e.g. <see cref="RandomFlip"/>, <see cref="RandomShift"/>,
    /// <see cref="GaussianNoise"/>), in order; each sample's random numbers come from the loader's seed, the epoch and
    /// the sample, so a seeded run repeats exactly. Use them on the training loader, not the validation one.
    /// </summary>
    public IReadOnlyList<ISampleTransform> Transforms { get; init; } = [];

    /// <summary>Batches per epoch; for a stream, those of the last full epoch (0 before the first).</summary>
    public int BatchCount => Source is null ? _streamBatches ?? 0
        : DropLast ? Source.Count / BatchSize : (Source.Count + BatchSize - 1) / BatchSize;

    /// <summary>Samples per epoch; for a stream, those of the last full epoch (0 before the first).</summary>
    public int SampleCount => Source is null ? _streamSamples ?? 0 : DropLast ? BatchCount * BatchSize : Source.Count;

    int? IBatchSource.BatchCount => Source is null ? _streamBatches : BatchCount;

    int? IBatchSource.SampleCount => Source is null ? _streamSamples : SampleCount;

    int? IBatchSource.BatchSize => BatchSize;

    /// <inheritdoc />
    public IEnumerator<Batch> GetEnumerator()
    {
        int epoch = Interlocked.Increment(ref _epoch) - 1;
        return Source is not null ? FromSource(Source, epoch) : FromStream(Stream!, epoch);
    }

    private IEnumerator<Batch> FromSource(ISampleSource source, int epoch)
    {
        int[] order = [.. Enumerable.Range(0, source.Count)];
        if (Shuffle)
        {
            _random.Shuffle(order);
        }

        int batches = BatchCount;
        if (batches == 0)
        {
            yield break;
        }

        int f = _featureSize, t = _targetSize;
        bool prefetch = batches > 1 && BatchSize * (f + t) >= PrefetchThreshold && ComputeResources.AllowParallel;
        var buffers = Buffers(prefetch);
        Task<int>? pending = prefetch ? Task.Run(() => Gather(source, order, 0, buffers[0], epoch)) : null;
        try
        {
            for (int b = 0; b < batches; b++)
            {
                var buffer = buffers[prefetch ? b & 1 : 0];
                int size = prefetch ? pending!.GetAwaiter().GetResult() : Gather(source, order, b, buffer, epoch);
                pending = null;
                if (prefetch && b + 1 < batches)
                {
                    int next = b + 1;
                    pending = Task.Run(() => Gather(source, order, next, buffers[next & 1], epoch));
                }

                yield return MakeBatch(buffer, size, b);
            }
        }
        finally
        {
            Settle(pending);
        }
    }

    private IEnumerator<Batch> FromStream(ISampleStream stream, int epoch)
    {
        int f = _featureSize, t = _targetSize;
        bool prefetch = BatchSize * (f + t) >= PrefetchThreshold && ComputeResources.AllowParallel;
        var buffers = Buffers(prefetch);
        using var feed = new StreamFeed(stream.Open(), f, t, ShuffleBuffer, _random);
        int samples = 0, b = 0;
        Task<int>? pending = prefetch ? Task.Run(() => Gather(feed, buffers[0], 0, epoch)) : null;
        try
        {
            while (true)
            {
                var buffer = buffers[prefetch ? b & 1 : 0];
                int size = prefetch ? pending!.GetAwaiter().GetResult() : Gather(feed, buffer, samples, epoch);
                pending = null;
                bool last = size < BatchSize;
                if (size == 0 || last && DropLast)
                {
                    break;
                }

                samples += size;
                if (prefetch && !last)
                {
                    int next = b + 1, start = samples;
                    pending = Task.Run(() => Gather(feed, buffers[next & 1], start, epoch));
                }

                yield return MakeBatch(buffer, size, b);
                b++;
                if (last)
                {
                    break;
                }
            }

            _streamBatches = b;
            _streamSamples = samples;
        }
        finally
        {
            Settle(pending);
        }
    }

    // Two host buffers alternate when prefetching: one is uploaded while the other is being filled.
    private (float[] X, float[] Y)[] Buffers(bool prefetch)
    {
        var buffers = new (float[] X, float[] Y)[prefetch ? 2 : 1];
        for (int i = 0; i < buffers.Length; i++)
        {
            buffers[i] = (new float[BatchSize * _featureSize], new float[BatchSize * _targetSize]);
        }

        return buffers;
    }

    private Batch MakeBatch((float[] X, float[] Y) buffer, int size, int index)
    {
        var x = Tensor.From(buffer.X.AsSpan(0, size * _featureSize), [size, .. _featureShape], Device);
        var y = Tensor.From(buffer.Y.AsSpan(0, size * _targetSize), [size, .. _targetShape], Device);
        return new Batch(x, y, index);
    }

    // A batch gathered ahead and then abandoned (the epoch was left early) is waited for, so no read outlives the epoch.
    private static void Settle(Task<int>? pending)
    {
        try
        {
            pending?.Wait();
        }
        catch (AggregateException)
        {
        }
    }

    /// <summary>Copies the samples of batch <paramref name="batch"/> into the host buffers; returns the sample count.</summary>
    private int Gather(ISampleSource source, int[] order, int batch, (float[] X, float[] Y) buffer, int epoch)
    {
        int start = batch * BatchSize;
        int size = Math.Min(BatchSize, source.Count - start);
        int f = _featureSize, t = _targetSize;
        for (int r = 0; r < size; r++)
        {
            int index = order[start + r];
            var x = buffer.X.AsSpan(r * f, f);
            var y = buffer.Y.AsSpan(r * t, t);
            source.Read(index, x, y);
            Transform(x, y, epoch, index);
        }

        return size;
    }

    /// <summary>Reads up to a batch of samples from the stream; returns the count (smaller at the end).</summary>
    private int Gather(StreamFeed feed, (float[] X, float[] Y) buffer, int first, int epoch)
    {
        int f = _featureSize, t = _targetSize, size = 0;
        while (size < BatchSize)
        {
            var x = buffer.X.AsSpan(size * f, f);
            var y = buffer.Y.AsSpan(size * t, t);
            if (!feed.Next(x, y))
            {
                break;
            }

            Transform(x, y, epoch, first + size);
            size++;
        }

        return size;
    }

    private void Transform(Span<float> features, Span<float> targets, int epoch, int sample)
    {
        if (Transforms.Count == 0)
        {
            return;
        }

        var random = new Random(SampleSeed(_transformSeed, epoch, sample));
        foreach (var transform in Transforms)
        {
            transform.Apply(features, targets, _featureShape, random);
        }
    }

    // The seed of one sample's transforms: the loader's seed, the epoch and the sample mixed (SplitMix64's finalizer).
    private static int SampleSeed(int seed, int epoch, int sample)
    {
        ulong z = ((ulong)(uint)seed << 32 | (uint)epoch) * 0x9E3779B97F4A7C15UL ^ (ulong)(uint)sample * 0xD6E8FEB86659FD93UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return (int)((z ^ (z >> 31)) & int.MaxValue);
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    // Samples of one pass over a stream, in order or drawn at random from a buffer that refills from the stream.
    private sealed class StreamFeed(ISampleReader reader, int f, int t, int capacity, Random random) : IDisposable
    {
        private readonly float[] _x = new float[capacity * f], _y = new float[capacity * t];
        private int _filled;
        private bool _ended;

        public bool Next(Span<float> features, Span<float> targets)
        {
            if (capacity <= 1)
            {
                return reader.Read(features, targets);
            }

            while (_filled < capacity && !_ended)
            {
                if (reader.Read(_x.AsSpan(_filled * f, f), _y.AsSpan(_filled * t, t)))
                {
                    _filled++;
                }
                else
                {
                    _ended = true;
                }
            }

            if (_filled == 0)
            {
                return false;
            }

            int j = random.Next(_filled);
            _x.AsSpan(j * f, f).CopyTo(features);
            _y.AsSpan(j * t, t).CopyTo(targets);
            if (_ended || !reader.Read(_x.AsSpan(j * f, f), _y.AsSpan(j * t, t)))
            {
                _ended = true;
                _filled--;
                _x.AsSpan(_filled * f, f).CopyTo(_x.AsSpan(j * f, f));                 // the last one fills the gap
                _y.AsSpan(_filled * t, t).CopyTo(_y.AsSpan(j * t, t));
            }

            return true;
        }

        public void Dispose() => reader.Dispose();
    }
}
