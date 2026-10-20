// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.IO.Compression;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Text.Abstractions;
using Idrak.Text.Retrieval.Abstractions;

namespace Idrak.Text.Retrieval;

/// <summary>
/// Searchable chunks: keyword search (any <see cref="IKeywordIndex"/> registered in <see cref="KeywordIndexes"/>, BM25 by
/// default), vector search (<see cref="TextEncoder"/> +
/// <see cref="VectorIndex"/>, or any <see cref="IEmbedder"/> with any <see cref="IVectorStore"/>), or both merged with
/// reciprocal rank fusion. Create one with <see cref="Create"/>.
/// </summary>
public sealed class RetrievalIndex : IRetriever
{
    private const string Format = "idrak-retrieval/1";

    private readonly KeywordChoice? _keywords;                     // the keyword index asked for, as saved

    internal RetrievalIndex(IReadOnlyList<Chunk> chunks, KeywordChoice? keywords, TextEncoder? encoder, VectorIndex? vectors, (int K, int Depth)? fusion,
        IEmbedder? embedder = null, IVectorStore? store = null)
    {
        Chunks = chunks;
        _keywords = keywords;
        Keywords = keywords is { } k ? KeywordIndexes.Create(k.Name, chunks.Select(c => c.Text), k.Options) : null;
        Encoder = encoder;
        Vectors = vectors;
        Fusion = fusion;
        Embedder = store is null ? encoder : embedder;
        Store = store;
    }

    /// <summary>Starts an index: add chunks, then choose keyword search, vector search, or both with fusion.</summary>
    public static RetrievalIndexBuilder Create() => new();

    /// <summary>The indexed chunks; a chunk's <see cref="Chunk.Id"/> is its position here.</summary>
    public IReadOnlyList<Chunk> Chunks { get; }

    /// <summary>The keyword index (from <see cref="KeywordIndexes"/>, BM25 by default), if keyword search is on.</summary>
    public IKeywordIndex? Keywords { get; }

    /// <summary>The text encoder, if vector search is on.</summary>
    public TextEncoder? Encoder { get; }

    /// <summary>The chunk vectors, if vector search uses the built-in <see cref="TextEncoder"/> and <see cref="VectorIndex"/>.</summary>
    public VectorIndex? Vectors { get; }

    /// <summary>What turns queries into vectors, if vector search is on (the <see cref="Encoder"/> when that is used).</summary>
    public IEmbedder? Embedder { get; }

    /// <summary>Where the chunk vectors live, if they are in an <see cref="IVectorStore"/> rather than <see cref="Vectors"/>.</summary>
    public IVectorStore? Store { get; }

    /// <summary>Reciprocal rank fusion settings when both searches are on: score = Σ 1 / (K + rank) over the first Depth results of each.</summary>
    public (int K, int Depth)? Fusion { get; }

    /// <summary>
    /// The <paramref name="top"/> best chunks for <paramref name="query"/>, best first (ties by chunk id). With an
    /// <see cref="IVectorStore"/> this waits for <see cref="SearchAsync"/>; prefer that one then.
    /// </summary>
    public IReadOnlyList<RetrievedChunk> Search(string query, int top)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(top);
        if (Store is not null)
        {
            return SearchAsync(query, top).AsTask().GetAwaiter().GetResult();
        }

        if (Chunks.Count == 0)
        {
            return [];
        }

        int depth = Fusion?.Depth ?? top;
        var keyword = Keywords?.Search(query, depth);
        var vector = Vectors is null ? null : Vectors.Search(Encoder!.Encode(query), depth);
        return Combine(keyword, vector, top);
    }

    /// <summary>The <paramref name="top"/> best chunks for <paramref name="query"/>, best first (ties by chunk id).</summary>
    public async ValueTask<IReadOnlyList<RetrievedChunk>> SearchAsync(string query, int top, CancellationToken cancellationToken = default)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(top);
        if (Store is null)
        {
            return Search(query, top);
        }

        if (Chunks.Count == 0)
        {
            return [];
        }

        int depth = Fusion?.Depth ?? top;
        var keyword = Keywords?.Search(query, depth);
        var embedded = await Embedder!.EmbedAsync([query], cancellationToken).ConfigureAwait(false);
        var matches = await Store.SearchAsync(embedded[0], depth, null, cancellationToken).ConfigureAwait(false);
        // Records are keyed by chunk id; any other record a shared store returns is not one of this index's chunks.
        IReadOnlyList<SearchHit> vector = [.. matches
            .Select(m => (Ok: int.TryParse(m.Id, System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out int id), Id: id, m.Score))
            .Where(m => m.Ok && m.Id < Chunks.Count)
            .Select(m => new SearchHit(m.Id, m.Score))];
        return Combine(keyword, vector, top);
    }

    /// <inheritdoc />
    ValueTask<IReadOnlyList<RetrievedChunk>> IRetriever.RetrieveAsync(string query, int top, CancellationToken cancellationToken) =>
        SearchAsync(query, top, cancellationToken);

    // One search's hits, or both merged by reciprocal rank fusion.
    private IReadOnlyList<RetrievedChunk> Combine(IReadOnlyList<SearchHit>? keyword, IReadOnlyList<SearchHit>? vector, int top)
    {
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
                System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrAddDefault(fused, id, out _) += 1.0 / (fk + rank);
            }
        }

        return [.. fused.OrderByDescending(p => p.Value).ThenBy(p => p.Key).Take(top).Select(p => Result(p.Key, p.Value))];
    }

    /// <summary>
    /// Writes the chunks, settings and vectors to a file (the text encoder's model is saved separately). Vectors kept in an
    /// <see cref="IVectorStore"/> stay there: pass the same store to <see cref="Load(string, IEmbedder, IVectorStore)"/>.
    /// </summary>
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
            // BM25 as before ("bm25": {k1, b}); another keyword index by name in "keywords", read back through KeywordIndexes.
            ["bm25"] = _keywords is { } bm25 && IsPlainBm25(bm25) ? new JsonObject { ["k1"] = bm25.Options.K1, ["b"] = bm25.Options.B } : null,
            ["vectors"] = Vectors is not null,
            ["store"] = Store is not null,
            ["fusion"] = Fusion is { } f ? new JsonObject { ["k"] = f.K, ["depth"] = f.Depth } : null,
            ["chunks"] = new JsonArray([.. Chunks.Select(c => (JsonNode)new JsonObject
            {
                ["document"] = c.DocumentId,
                ["position"] = c.Position,
                ["text"] = c.Text,
            })]),
        };
        if (_keywords is { } other && !IsPlainBm25(other))
        {
            settings["keywords"] = new JsonObject
            {
                ["name"] = other.Name, ["k1"] = other.Options.K1, ["b"] = other.Options.B,
                ["parameters"] = new JsonObject(other.Options.Parameters.Select(p => KeyValuePair.Create(p.Key, (JsonNode?)p.Value))),
            };
        }

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

    /// <summary>Reads an index written by <see cref="Save(string)"/> whose vectors are in <paramref name="store"/>, searched with <paramref name="embedder"/>.</summary>
    public static RetrievalIndex Load(string path, IEmbedder embedder, IVectorStore store)
    {
        using var stream = File.OpenRead(path);
        return Load(stream, embedder, store);
    }

    /// <summary>Reads an index written by <see cref="Save(Stream)"/> whose vectors are in <paramref name="store"/>, searched with <paramref name="embedder"/>.</summary>
    public static RetrievalIndex Load(Stream stream, IEmbedder embedder, IVectorStore store)
    {
        ArgumentNullException.ThrowIfNull(embedder);
        ArgumentNullException.ThrowIfNull(store);
        var (chunks, keywords, fusion, settings, _) = ReadSettings(stream);
        if (settings["store"] is not JsonValue saved || !(bool)saved)
        {
            throw new InvalidDataException("This index keeps its own vectors; load it with Load(path, encoder).");
        }

        return new RetrievalIndex(chunks, keywords, null, null, fusion, embedder, store);
    }

    /// <summary>Reads an index written by <see cref="Save(Stream)"/>. Pass the same text encoder if it uses vector search.</summary>
    public static RetrievalIndex Load(Stream stream, TextEncoder? encoder)
    {
        var (chunks, keywords, fusion, settings, zip) = ReadSettings(stream);
        using var _ = zip;
        if (settings["store"] is JsonValue saved && (bool)saved)
        {
            throw new InvalidDataException("This index's vectors are in a vector store; load it with Load(path, embedder, store).");
        }

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

        return new RetrievalIndex(chunks, keywords, vectors is null ? null : encoder, vectors, fusion);
    }

    // Whether the index is the library's BM25 with no settings beyond k1 and b (saved as "bm25", as before keyword indexes had names).
    private static bool IsPlainBm25(KeywordChoice choice) =>
        string.Equals(choice.Name, KeywordIndexes.Bm25, StringComparison.OrdinalIgnoreCase) && choice.Options.Parameters.Count == 0;

    private static (List<Chunk> Chunks, KeywordChoice? Keywords, (int, int)? Fusion, JsonObject Settings, ZipArchive Zip) ReadSettings(Stream stream)
    {
        var zip = new ZipArchive(stream, ZipArchiveMode.Read, leaveOpen: true);
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
        KeywordChoice? keywords = settings["keywords"] is JsonObject k
            ? new KeywordChoice((string)k["name"]!, new KeywordIndexOptions
            {
                K1 = (double)k["k1"]!, B = (double)k["b"]!,
                Parameters = (k["parameters"] as JsonObject)?.ToDictionary(p => p.Key, p => (double)p.Value!) ?? new Dictionary<string, double>(),
            })
            : settings["bm25"] is JsonObject b ? new KeywordChoice(KeywordIndexes.Bm25, new KeywordIndexOptions { K1 = (double)b["k1"]!, B = (double)b["b"]! }) : null;
        (int, int)? fusion = settings["fusion"] is JsonObject f ? ((int)f["k"]!, (int)f["depth"]!) : null;
        return (chunks, keywords, fusion, settings, zip);
    }

    private static Dictionary<int, int> Ranks(IReadOnlyList<SearchHit>? hits) =>
        hits is null ? [] : hits.Select((h, i) => (h.Id, Rank: i + 1)).ToDictionary(p => p.Id, p => p.Rank);
}

// The keyword index an index builds over its chunks: its name in KeywordIndexes and its settings.
internal readonly record struct KeywordChoice(string Name, KeywordIndexOptions Options);

/// <summary>
/// Builds a <see cref="RetrievalIndex"/>: the chunks (<see cref="Add"/> or <see cref="Documents"/>), then keyword
/// search (<see cref="Bm25"/>, or any registered index with <see cref="Keywords"/>), vector search (<see cref="Embeddings"/>), or both, which also needs <see cref="Fusion"/>.
/// </summary>
public sealed class RetrievalIndexBuilder
{
    private readonly List<Chunk> _chunks = [];
    private KeywordChoice? _keywords;
    private IEmbedder? _embedder;
    private IVectorStore? _store;
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
    public RetrievalIndexBuilder Bm25(double k1 = 1.2, double b = 0.75) => Keywords(KeywordIndexes.Bm25, new KeywordIndexOptions { K1 = k1, B = b });

    /// <summary>
    /// Keyword search with the index registered as <paramref name="name"/> in <see cref="KeywordIndexes"/> ("bm25", or one
    /// of your own), built with <paramref name="options"/> (the defaults when null).
    /// </summary>
    /// <exception cref="NotSupportedException">No keyword index has that name.</exception>
    public RetrievalIndexBuilder Keywords(string name, KeywordIndexOptions? options = null)
    {
        _ = KeywordIndexes.Get(name);                                            // an unknown name fails here, not at Build
        _keywords = new KeywordChoice(name, options ?? new KeywordIndexOptions());
        return this;
    }

    /// <summary>
    /// Vector search: every chunk is embedded with <paramref name="embedder"/> (<see cref="TextEncoder"/> or any
    /// <see cref="IEmbedder"/>, in batches of <paramref name="batchSize"/>) when the index is built. The vectors go into a
    /// <see cref="VectorIndex"/> for a <see cref="TextEncoder"/>, into an <see cref="InMemoryVectorStore"/> for another
    /// embedder, or into the store given with <see cref="VectorStore"/>.
    /// </summary>
    public RetrievalIndexBuilder Embeddings(IEmbedder embedder, int batchSize = 256)
    {
        ArgumentNullException.ThrowIfNull(embedder);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);
        _embedder = embedder;
        _encodeBatch = batchSize;
        return this;
    }

    /// <summary>
    /// Keeps the chunk vectors in <paramref name="store"/> (a vector database, or a shared <see cref="InMemoryVectorStore"/>)
    /// under the chunk ids ("0", "1", …) with the chunk's document and position as metadata. Needs <see cref="Embeddings"/>.
    /// </summary>
    public RetrievalIndexBuilder VectorStore(IVectorStore store)
    {
        ArgumentNullException.ThrowIfNull(store);
        _store = store;
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

    /// <summary>Builds the index (embeds the chunks if vector search is on). With an <see cref="IVectorStore"/>, prefer <see cref="BuildAsync"/>.</summary>
    public RetrievalIndex Build()
    {
        Validate();
        if (_embedder is null)
        {
            return new RetrievalIndex([.. _chunks], _keywords, null, null, _fusion);    // keyword search only
        }

        if (_embedder is TextEncoder encoder && _store is null)
        {
            var encoded = encoder.Encode([.. _chunks.Select(c => c.Text)], _encodeBatch);
            var vectors = new VectorIndex(encoded.Length > 0 ? encoded[0].Length : 1, VectorMetric.Dot);
            vectors.AddRange(encoded);
            return new RetrievalIndex([.. _chunks], _keywords, encoder, vectors, _fusion);
        }

        return BuildAsync().AsTask().GetAwaiter().GetResult();
    }

    /// <summary>Builds the index, embedding the chunks and storing their vectors if vector search is on.</summary>
    public async ValueTask<RetrievalIndex> BuildAsync(CancellationToken cancellationToken = default)
    {
        Validate();
        if (_embedder is null || _embedder is TextEncoder && _store is null)
        {
            return Build();
        }

        IVectorStore? store = _store;
        for (int start = 0; start < _chunks.Count; start += _encodeBatch)
        {
            var batch = _chunks.Skip(start).Take(_encodeBatch).ToList();
            var vectors = await _embedder.EmbedAsync([.. batch.Select(c => c.Text)], cancellationToken).ConfigureAwait(false);
            if (vectors.Length != batch.Count)
            {
                throw new InvalidOperationException($"The embedder returned {vectors.Length} vectors for {batch.Count} texts.");
            }

            store ??= new InMemoryVectorStore(vectors.Length > 0 ? vectors[0].Length : 1, VectorMetric.Cosine);
            await store.UpsertAsync([.. batch.Select((c, i) => new VectorRecord(c.Id.ToString(System.Globalization.CultureInfo.InvariantCulture), vectors[i],
                new Dictionary<string, string> { ["document"] = c.DocumentId, ["position"] = c.Position.ToString(System.Globalization.CultureInfo.InvariantCulture) }))],
                cancellationToken).ConfigureAwait(false);
        }

        return new RetrievalIndex([.. _chunks], _keywords, null, null, _fusion, _embedder, store ?? new InMemoryVectorStore(1));
    }

    private void Validate()
    {
        if (_keywords is null && _embedder is null)
        {
            throw new InvalidOperationException("Choose keyword search (Bm25 or Keywords), vector search (Embeddings), or both.");
        }

        if (_keywords is not null && _embedder is not null && _fusion is null)
        {
            throw new InvalidOperationException("Keyword and vector search together need Fusion(k, depth) to merge their results.");
        }

        if (_fusion is not null && (_keywords is null || _embedder is null))
        {
            throw new InvalidOperationException("Fusion merges keyword and vector results; it needs both keyword search (Bm25 or Keywords) and Embeddings.");
        }

        if (_store is not null && _embedder is null)
        {
            throw new InvalidOperationException("A vector store holds the chunks' vectors; choose how to make them with Embeddings.");
        }
    }
}
