using System.Buffers.Binary;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;

namespace Idrak.Retrieval;

/// <summary>How a <see cref="VectorIndex"/> compares vectors.</summary>
public enum VectorMetric
{
    /// <summary>Dot product (use for vectors that are already unit length, like <see cref="TextEncoder"/>'s).</summary>
    Dot,

    /// <summary>Cosine similarity: vectors are scaled to unit length when added and when searched.</summary>
    Cosine,
}

/// <summary>
/// Exact nearest-neighbour search over stored vectors (every vector is compared, with SIMD). Fast enough for tens of
/// thousands of chunks on one thread and millions in batches; ids are the order of addition.
/// </summary>
public sealed class VectorIndex
{
    private const uint Magic = 0x3156_4B49; // "IKV1"
    private readonly List<float> _data = [];

    /// <summary>Creates an empty index for vectors of <paramref name="dimensions"/> values compared with <paramref name="metric"/>.</summary>
    public VectorIndex(int dimensions, VectorMetric metric)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dimensions);
        Dimensions = dimensions;
        Metric = metric;
    }

    /// <summary>Values per vector.</summary>
    public int Dimensions { get; }

    /// <summary>The comparison.</summary>
    public VectorMetric Metric { get; }

    /// <summary>Number of vectors.</summary>
    public int Count => _data.Count / Dimensions;

    /// <summary>Adds a vector and returns its id.</summary>
    public int Add(ReadOnlySpan<float> vector)
    {
        if (vector.Length != Dimensions)
        {
            throw new ArgumentException($"Expected {Dimensions} values, got {vector.Length}.", nameof(vector));
        }

        int id = Count;
        var copy = vector.ToArray();
        if (Metric == VectorMetric.Cosine)
        {
            Normalize(copy);
        }

        _data.AddRange(copy);
        return id;
    }

    /// <summary>Adds several vectors.</summary>
    public void AddRange(IEnumerable<float[]> vectors)
    {
        foreach (var v in vectors)
        {
            Add(v);
        }
    }

    /// <summary>The <paramref name="top"/> most similar vectors to <paramref name="query"/>, best first (ties by id).</summary>
    public IReadOnlyList<SearchHit> Search(ReadOnlySpan<float> query, int top)
    {
        if (query.Length != Dimensions)
        {
            throw new ArgumentException($"Expected {Dimensions} values, got {query.Length}.", nameof(query));
        }

        var q = query.ToArray();
        if (Metric == VectorMetric.Cosine)
        {
            Normalize(q);
        }

        int keep = Math.Min(top, Count);
        if (keep <= 0)
        {
            return [];
        }

        var data = CollectionsMarshal.AsSpan(_data);
        var best = new TopHits(keep);
        for (int i = 0; i < Count; i++)
        {
            best.Offer(i, Dot(q, data.Slice(i * Dimensions, Dimensions)));
        }

        return best.ToArray();
    }

    /// <summary>The stored vector with id <paramref name="id"/>.</summary>
    public float[] this[int id] => CollectionsMarshal.AsSpan(_data).Slice(id * Dimensions, Dimensions).ToArray();

    /// <summary>Writes the index (dimensions, metric and vectors) to <paramref name="stream"/>; the stream stays open.</summary>
    public void Save(Stream stream)
    {
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        writer.Write(Magic);
        writer.Write(Dimensions);
        writer.Write((int)Metric);
        writer.Write(Count);
        var data = _data.ToArray();
        if (!BitConverter.IsLittleEndian)
        {
            var bits = MemoryMarshal.Cast<float, int>(data.AsSpan());
            BinaryPrimitives.ReverseEndianness(bits, bits);
        }

        writer.Write(MemoryMarshal.AsBytes(data.AsSpan()));
    }

    /// <summary>Reads an index written by <see cref="Save"/>.</summary>
    public static VectorIndex Load(Stream stream)
    {
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        if (reader.ReadUInt32() != Magic)
        {
            throw new InvalidDataException("Not a Idrak vector index.");
        }

        var index = new VectorIndex(reader.ReadInt32(), (VectorMetric)reader.ReadInt32());
        var data = new float[reader.ReadInt32() * index.Dimensions];
        reader.BaseStream.ReadExactly(MemoryMarshal.AsBytes(data.AsSpan()));
        if (!BitConverter.IsLittleEndian)
        {
            var bits = MemoryMarshal.Cast<float, int>(data.AsSpan());
            BinaryPrimitives.ReverseEndianness(bits, bits);
        }

        index._data.AddRange(data);
        return index;
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

    private static void Normalize(Span<float> v)
    {
        double norm = Math.Sqrt(Dot(v, v));
        if (norm > 0)
        {
            for (int i = 0; i < v.Length; i++)
            {
                v[i] = (float)(v[i] / norm);
            }
        }
    }
}
