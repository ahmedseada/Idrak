using System.Collections;

namespace Idrak.Data;

/// <summary>One mini-batch, already on the loader's device. Dispose it (the trainer does) to recycle its memory.</summary>
public sealed class Batch : IDisposable
{
    internal Batch(Tensor features, Tensor targets, int index)
    {
        Features = features;
        Targets = targets;
        Index = index;
    }

    /// <summary>[Size, FeatureCount] inputs.</summary>
    public Tensor Features { get; }

    /// <summary>[Size, TargetCount] expected outputs.</summary>
    public Tensor Targets { get; }

    /// <summary>Zero-based batch number within the epoch.</summary>
    public int Index { get; }

    /// <summary>Samples in this batch.</summary>
    public int Size => Features.Shape[0];

    /// <inheritdoc />
    public void Dispose()
    {
        Features.Dispose();
        Targets.Dispose();
    }
}

/// <summary>
/// Cuts a <see cref="Dataset"/> into mini-batches and moves them to a device. Each enumeration is one
/// epoch, reshuffled when <see cref="Shuffle"/> is on. While the model trains on batch n, batch n+1 is
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
public sealed class DataLoader : IEnumerable<Batch>
{
    /// <summary>Batches with at least this many floats are gathered on a background thread.</summary>
    private const int PrefetchThreshold = 16 * 1024;

    private readonly Random _random;

    /// <summary>Creates a loader.</summary>
    /// <param name="dataset">The samples.</param>
    /// <param name="batchSize">Samples per batch.</param>
    /// <param name="shuffle">Reorder samples every epoch (recommended for training).</param>
    /// <param name="dropLast">Skip the final, smaller batch.</param>
    /// <param name="device">Where batches are placed; defaults to <see cref="Device.Default"/>.</param>
    /// <param name="seed">Shuffle seed, for reproducible runs.</param>
    public DataLoader(Dataset dataset, int batchSize = 32, bool shuffle = false, bool dropLast = false, Device? device = null, int? seed = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);
        Dataset = dataset;
        BatchSize = batchSize;
        Shuffle = shuffle;
        DropLast = dropLast;
        Device = device ?? Device.Default;
        _random = seed is { } s ? new Random(s) : new Random();
    }

    /// <summary>The source samples.</summary>
    public Dataset Dataset { get; }

    /// <summary>Samples per batch.</summary>
    public int BatchSize { get; }

    /// <summary>Whether samples are reordered every epoch.</summary>
    public bool Shuffle { get; }

    /// <summary>Whether a final partial batch is skipped.</summary>
    public bool DropLast { get; }

    /// <summary>Where batch tensors are created.</summary>
    public Device Device { get; }

    /// <summary>Batches per epoch.</summary>
    public int BatchCount => DropLast ? Dataset.Count / BatchSize : (Dataset.Count + BatchSize - 1) / BatchSize;

    /// <summary>Samples per epoch.</summary>
    public int SampleCount => DropLast ? BatchCount * BatchSize : Dataset.Count;

    /// <inheritdoc />
    public IEnumerator<Batch> GetEnumerator()
    {
        int[] order = [.. Enumerable.Range(0, Dataset.Count)];
        if (Shuffle)
        {
            _random.Shuffle(order);
        }

        int batches = BatchCount;
        if (batches == 0)
        {
            yield break;
        }

        int f = Dataset.FeatureCount, t = Dataset.TargetCount;
        bool prefetch = batches > 1 && BatchSize * (f + t) >= PrefetchThreshold && ComputeResources.AllowParallel;

        // Two host buffers alternate: one is uploaded while the other is being filled.
        var buffers = new (float[] X, float[] Y)[prefetch ? 2 : 1];
        for (int i = 0; i < buffers.Length; i++)
        {
            buffers[i] = (new float[BatchSize * f], new float[BatchSize * t]);
        }

        Task<int>? pending = prefetch ? Task.Run(() => Gather(order, 0, buffers[0])) : null;
        for (int b = 0; b < batches; b++)
        {
            var buffer = buffers[prefetch ? b & 1 : 0];
            int size = prefetch ? pending!.GetAwaiter().GetResult() : Gather(order, b, buffer);
            if (prefetch && b + 1 < batches)
            {
                int next = b + 1;
                pending = Task.Run(() => Gather(order, next, buffers[next & 1]));
            }

            var x = Tensor.From(buffer.X.AsSpan(0, size * f), [size, .. Dataset.FeatureShape], Device);
            var y = Tensor.From(buffer.Y.AsSpan(0, size * t), [size, t], Device);
            yield return new Batch(x, y, b);
        }
    }

    /// <summary>Copies the rows of batch <paramref name="batch"/> into the host buffers; returns the row count.</summary>
    private int Gather(int[] order, int batch, (float[] X, float[] Y) buffer)
    {
        int start = batch * BatchSize;
        int size = Math.Min(BatchSize, Dataset.Count - start);
        int f = Dataset.FeatureCount, t = Dataset.TargetCount;
        for (int r = 0; r < size; r++)
        {
            int index = order[start + r];
            Dataset.GetFeatures(index).CopyTo(buffer.X.AsSpan(r * f, f));
            Dataset.GetTargets(index).CopyTo(buffer.Y.AsSpan(r * t, t));
        }

        return size;
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
