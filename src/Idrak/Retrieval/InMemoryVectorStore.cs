using System.Numerics;

namespace Idrak.Retrieval;

/// <summary>
/// An <see cref="IVectorStore"/> held in memory: exact search (every record is compared, SIMD dot products), records
/// replaced or removed by id, metadata filters. Thread-safe. For collections that fit in memory; implement
/// <see cref="IVectorStore"/> over a vector database for larger ones.
/// </summary>
public sealed class InMemoryVectorStore : IVectorStore
{
    private readonly Lock _gate = new();
    private readonly Dictionary<string, int> _slots = [];                  // id → slot
    private readonly List<VectorRecord?> _records = [];                     // null: a removed record's free slot
    private readonly Stack<int> _free = [];

    /// <summary>Creates an empty store for vectors of <paramref name="dimensions"/> values.</summary>
    /// <param name="dimensions">The length of every vector.</param>
    /// <param name="metric">Dot product (vectors already unit length) or cosine (vectors scaled to unit length when stored and searched).</param>
    public InMemoryVectorStore(int dimensions, VectorMetric metric = VectorMetric.Cosine)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dimensions);
        Dimensions = dimensions;
        Metric = metric;
    }

    /// <summary>The length of every vector.</summary>
    public int Dimensions { get; }

    /// <summary>How vectors are compared.</summary>
    public VectorMetric Metric { get; }

    /// <summary>The number of records.</summary>
    public int Count
    {
        get
        {
            lock (_gate)
            {
                return _slots.Count;
            }
        }
    }

    /// <inheritdoc />
    public ValueTask UpsertAsync(IReadOnlyList<VectorRecord> records, CancellationToken cancellationToken = default)
    {
        lock (_gate)
        {
            foreach (var record in records)
            {
                if (record.Vector.Length != Dimensions)
                {
                    throw new ArgumentException($"Record '{record.Id}' has {record.Vector.Length} values; the store holds {Dimensions}.", nameof(records));
                }

                var stored = record with { Vector = Metric == VectorMetric.Cosine ? Unit(record.Vector) : [.. record.Vector] };
                if (_slots.TryGetValue(record.Id, out int slot))
                {
                    _records[slot] = stored;
                }
                else if (_free.TryPop(out slot))
                {
                    _records[slot] = stored;
                    _slots[record.Id] = slot;
                }
                else
                {
                    _slots[record.Id] = _records.Count;
                    _records.Add(stored);
                }
            }
        }

        return ValueTask.CompletedTask;
    }

    /// <inheritdoc />
    public ValueTask<IReadOnlyList<VectorMatch>> SearchAsync(float[] query, int top, IReadOnlyDictionary<string, string>? where = null,
        CancellationToken cancellationToken = default)
    {
        if (query.Length != Dimensions)
        {
            throw new ArgumentException($"Expected {Dimensions} values, got {query.Length}.", nameof(query));
        }

        var q = Metric == VectorMetric.Cosine ? Unit(query) : query;
        lock (_gate)
        {
            // The best hits kept in a heap over slots (ties to the earlier slot), then read back as ids.
            var best = new TopHits(top);
            for (int slot = 0; slot < _records.Count; slot++)
            {
                if (_records[slot] is { } record && Matches(record.Metadata, where))
                {
                    best.Offer(slot, Dot(q, record.Vector));
                }
            }

            IReadOnlyList<VectorMatch> matches = [.. best.ToArray().Select(h => new VectorMatch(_records[h.Id]!.Id, h.Score, _records[h.Id]!.Metadata))];
            return ValueTask.FromResult(matches);
        }
    }

    /// <inheritdoc />
    public ValueTask DeleteAsync(IReadOnlyList<string> ids, CancellationToken cancellationToken = default)
    {
        lock (_gate)
        {
            foreach (var id in ids)
            {
                if (_slots.Remove(id, out int slot))
                {
                    _records[slot] = null;
                    _free.Push(slot);
                }
            }
        }

        return ValueTask.CompletedTask;
    }

    private static bool Matches(IReadOnlyDictionary<string, string>? metadata, IReadOnlyDictionary<string, string>? where)
    {
        if (where is null || where.Count == 0)
        {
            return true;
        }

        if (metadata is null)
        {
            return false;
        }

        foreach (var (key, value) in where)
        {
            if (!metadata.TryGetValue(key, out var actual) || actual != value)
            {
                return false;
            }
        }

        return true;
    }

    private static float[] Unit(float[] vector)
    {
        double norm = Math.Sqrt(Dot(vector, vector));
        return norm > 0 ? [.. vector.Select(v => (float)(v / norm))] : [.. vector];
    }

    private static double Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        var sum = Vector<float>.Zero;
        int i = 0;
        for (; i <= a.Length - Vector<float>.Count; i += Vector<float>.Count)
        {
            sum += new Vector<float>(a[i..]) * new Vector<float>(b[i..]);
        }

        float total = Vector.Sum(sum);
        for (; i < a.Length; i++)
        {
            total += a[i] * b[i];
        }

        return total;
    }
}
