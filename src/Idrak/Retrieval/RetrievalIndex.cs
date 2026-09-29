using System.IO.Compression;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Idrak.Retrieval;

/// <summary>A chunk found by a search.</summary>
/// <param name="Chunk">The chunk.</param>
/// <param name="Score">Its score in the final order: BM25, vector similarity, fused score or re-ranker score.</param>
/// <param name="KeywordRank">Its position (1 = best) in the keyword results, or null if keywords were not searched or did not find it.</param>
/// <param name="VectorRank">Its position (1 = best) in the vector results, or null if vectors were not searched or did not find it.</param>
public sealed record RetrievedChunk(Chunk Chunk, double Score, int? KeywordRank, int? VectorRank);

/// <summary>
/// Searchable chunks: keyword search (<see cref="Bm25Index"/>), vector search (<see cref="TextEncoder"/> +
/// <see cref="VectorIndex"/>), or both merged with reciprocal rank fusion. Create one with <see cref="Create"/>.
/// </summary>
public sealed class RetrievalIndex
{
    private const string Format = "idrak-retrieval/1";

    internal RetrievalIndex(IReadOnlyList<Chunk> chunks, (double K1, double B)? bm25, TextEncoder? encoder, VectorIndex? vectors, (int K, int Depth)? fusion)
    {
        Chunks = chunks;
        Bm25Settings = bm25;
        Keywords = bm25 is { } b ? new Bm25Index(chunks.Select(c => c.Text), b.K1, b.B) : null;
        Encoder = encoder;
        Vectors = vectors;
        Fusion = fusion;
    }

    /// <summary>Starts an index: add chunks, then choose keyword search, vector search, or both with fusion.</summary>
    public static RetrievalIndexBuilder Create() => new();

    /// <summary>The indexed chunks; a chunk's <see cref="Chunk.Id"/> is its position here.</summary>
    public IReadOnlyList<Chunk> Chunks { get; }

    /// <summary>The keyword index, if keyword search is on.</summary>
    public Bm25Index? Keywords { get; }

    /// <summary>The text encoder, if vector search is on.</summary>
    public TextEncoder? Encoder { get; }

    /// <summary>The chunk vectors, if vector search is on.</summary>
    public VectorIndex? Vectors { get; }

    /// <summary>Reciprocal rank fusion settings when both searches are on: score = Σ 1 / (K + rank) over the first Depth results of each.</summary>
    public (int K, int Depth)? Fusion { get; }

    private (double K1, double B)? Bm25Settings { get; }

    /// <summary>The <paramref name="top"/> best chunks for <paramref name="query"/>, best first (ties by chunk id).</summary>
    public IReadOnlyList<RetrievedChunk> Search(string query, int top)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(top);
        if (Chunks.Count == 0)
        {
            return [];
        }

        int depth = Fusion?.Depth ?? top;
        var keyword = Keywords?.Search(query, depth);
        var vector = Vectors is null ? null : Vectors.Search(Encoder!.Encode(query), depth);
        var keywordRank = Ranks(keyword);
        var vectorRank = Ranks(vector);
        RetrievedChunk Result(int id, double score) =>
            new(Chunks[id], score, keywordRank.TryGetValue(id, out int k) ? k : null, vectorRank.TryGetValue(id, out int v) ? v : null);

        if (keyword is null || vector is null)
        {
            return [.. (keyword ?? vector)!.Take(top).Select(h => Result(h.Id, h.Score))];
        }

        var (fk, _) = Fusion!.Value;
        var fused = new Dictionary<int, double>();
        foreach (var ranks in new[] { keywordRank, vectorRank })
        {
            foreach (var (id, rank) in ranks)
            {
                fused[id] = fused.GetValueOrDefault(id) + 1.0 / (fk + rank);
            }
        }

        return [.. fused.OrderByDescending(p => p.Value).ThenBy(p => p.Key).Take(top).Select(p => Result(p.Key, p.Value))];
    }

    /// <summary>Writes the chunks, settings and vectors to a file (the text encoder's model is saved separately).</summary>
    public void Save(string path)
    {
        using var stream = File.Create(path);
        Save(stream);
    }

    /// <summary>Writes the chunks, settings and vectors to <paramref name="stream"/> (a zip archive); the stream stays open.</summary>
    public void Save(Stream stream)
    {
        using var zip = new ZipArchive(stream, ZipArchiveMode.Create, leaveOpen: true);
        var settings = new JsonObject
        {
            ["format"] = Format,
            ["bm25"] = Bm25Settings is { } b ? new JsonObject { ["k1"] = b.K1, ["b"] = b.B } : null,
            ["vectors"] = Vectors is not null,
            ["fusion"] = Fusion is { } f ? new JsonObject { ["k"] = f.K, ["depth"] = f.Depth } : null,
            ["chunks"] = new JsonArray([.. Chunks.Select(c => (JsonNode)new JsonObject
            {
                ["document"] = c.DocumentId,
                ["position"] = c.Position,
                ["text"] = c.Text,
            })]),
        };
        using (var entry = zip.CreateEntry("index.json").Open())
        using (var writer = new Utf8JsonWriter(entry))
        {
            settings.WriteTo(writer);
        }

        if (Vectors is not null)
        {
            using var entry = zip.CreateEntry("vectors.nsv").Open();
            Vectors.Save(entry);
        }
    }

    /// <summary>Reads an index written by <see cref="Save(string)"/>. Pass the same text encoder if it uses vector search.</summary>
    public static RetrievalIndex Load(string path, TextEncoder? encoder)
    {
        using var stream = File.OpenRead(path);
        return Load(stream, encoder);
    }

    /// <summary>Reads an index written by <see cref="Save(Stream)"/>. Pass the same text encoder if it uses vector search.</summary>
    public static RetrievalIndex Load(Stream stream, TextEncoder? encoder)
    {
        using var zip = new ZipArchive(stream, ZipArchiveMode.Read, leaveOpen: true);
        JsonObject settings;
        using (var entry = (zip.GetEntry("index.json") ?? throw new InvalidDataException("Not a Idrak retrieval index (no index.json).")).Open())
        {
            settings = JsonNode.Parse(entry)?.AsObject() ?? throw new InvalidDataException("Empty index.json.");
        }

        if ((string?)settings["format"] != Format)
        {
            throw new InvalidDataException($"Unsupported retrieval index format '{settings["format"]}'.");
        }

        var chunks = settings["chunks"]!.AsArray().Select((c, i) =>
            new Chunk(i, (string)c!["document"]!, (int)c["position"]!, (string)c["text"]!)).ToList();
        (double, double)? bm25 = settings["bm25"] is JsonObject b ? ((double)b["k1"]!, (double)b["b"]!) : null;
        (int, int)? fusion = settings["fusion"] is JsonObject f ? ((int)f["k"]!, (int)f["depth"]!) : null;
        VectorIndex? vectors = null;
        if ((bool)settings["vectors"]!)
        {
            if (encoder is null)
            {
                throw new ArgumentNullException(nameof(encoder), "This index uses vector search; pass the text encoder it was built with.");
            }

            using var entry = (zip.GetEntry("vectors.nsv") ?? throw new InvalidDataException("The index has no vectors.nsv.")).Open();
            using var buffer = new MemoryStream();
            entry.CopyTo(buffer);
            buffer.Position = 0;
            vectors = VectorIndex.Load(buffer);
        }

        return new RetrievalIndex(chunks, bm25, vectors is null ? null : encoder, vectors, fusion);
    }

    private static Dictionary<int, int> Ranks(IReadOnlyList<SearchHit>? hits) =>
        hits is null ? [] : hits.Select((h, i) => (h.Id, Rank: i + 1)).ToDictionary(p => p.Id, p => p.Rank);
}

/// <summary>
/// Builds a <see cref="RetrievalIndex"/>: the chunks (<see cref="Add"/> or <see cref="Documents"/>), then keyword
/// search (<see cref="Bm25"/>), vector search (<see cref="Embeddings"/>), or both, which also needs <see cref="Fusion"/>.
/// </summary>
public sealed class RetrievalIndexBuilder
{
    private readonly List<Chunk> _chunks = [];
    private (double K1, double B)? _bm25;
    private TextEncoder? _encoder;
    private int _encodeBatch;
    private (int K, int Depth)? _fusion;

    internal RetrievalIndexBuilder()
    {
    }

    /// <summary>Adds chunks; their ids are renumbered to their positions in the index.</summary>
    public RetrievalIndexBuilder Add(IEnumerable<Chunk> chunks)
    {
        foreach (var chunk in chunks)
        {
            _chunks.Add(chunk with { Id = _chunks.Count });
        }

        return this;
    }

    /// <summary>Splits documents with <see cref="Chunker.Split"/> and adds the chunks.</summary>
    public RetrievalIndexBuilder Documents(IEnumerable<Document> documents, ChunkUnit unit, int size, int overlap) =>
        Add(Chunker.Split(documents, unit, size, overlap));

    /// <summary>Keyword search with BM25 (<paramref name="k1"/>: term-frequency saturation, <paramref name="b"/>: length normalization).</summary>
    public RetrievalIndexBuilder Bm25(double k1 = 1.2, double b = 0.75)
    {
        _bm25 = (k1, b);
        return this;
    }

    /// <summary>Vector search: every chunk is encoded with <paramref name="encoder"/> (in batches of <paramref name="batchSize"/>) when the index is built.</summary>
    public RetrievalIndexBuilder Embeddings(TextEncoder encoder, int batchSize = 256)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);
        _encoder = encoder;
        _encodeBatch = batchSize;
        return this;
    }

    /// <summary>
    /// Merges keyword and vector results with reciprocal rank fusion: each chunk scores Σ 1 / (<paramref name="k"/> + rank)
    /// over the first <paramref name="depth"/> results of each search (k = 60 is the usual choice). Required when both are on.
    /// </summary>
    public RetrievalIndexBuilder Fusion(int k, int depth)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(k);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(depth);
        _fusion = (k, depth);
        return this;
    }

    /// <summary>Builds the index (encodes the chunks if vector search is on).</summary>
    public RetrievalIndex Build()
    {
        if (_bm25 is null && _encoder is null)
        {
            throw new InvalidOperationException("Choose keyword search (Bm25), vector search (Embeddings), or both.");
        }

        if (_bm25 is not null && _encoder is not null && _fusion is null)
        {
            throw new InvalidOperationException("Keyword and vector search together need Fusion(k, depth) to merge their results.");
        }

        if (_fusion is not null && (_bm25 is null || _encoder is null))
        {
            throw new InvalidOperationException("Fusion merges keyword and vector results; it needs both Bm25 and Embeddings.");
        }

        VectorIndex? vectors = null;
        if (_encoder is not null)
        {
            var encoded = _encoder.Encode([.. _chunks.Select(c => c.Text)], _encodeBatch);
            vectors = new VectorIndex(encoded.Length > 0 ? encoded[0].Length : 1, VectorMetric.Dot);
            vectors.AddRange(encoded);
        }

        return new RetrievalIndex([.. _chunks], _bm25, _encoder, vectors, _fusion);
    }
}
