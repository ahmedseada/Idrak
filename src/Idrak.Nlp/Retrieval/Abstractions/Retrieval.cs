// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Retrieval.Abstractions;

/// <summary>A passage of a document, the unit that is indexed and retrieved.</summary>
/// <param name="Id">Position in the index (0, 1, 2, …).</param>
/// <param name="DocumentId">The document it came from.</param>
/// <param name="Position">Its number within that document (0 for the first chunk).</param>
/// <param name="Text">The passage.</param>
public sealed record Chunk(int Id, string DocumentId, int Position, string Text);

/// <summary>A chunk found by a search.</summary>
/// <param name="Chunk">The chunk.</param>
/// <param name="Score">Its score in the final order: BM25, vector similarity, fused score or re-ranker score.</param>
/// <param name="KeywordRank">Its position (1 = best) in the keyword results, or null if keywords were not searched or did not find it.</param>
/// <param name="VectorRank">Its position (1 = best) in the vector results, or null if vectors were not searched or did not find it.</param>
public sealed record RetrievedChunk(Chunk Chunk, double Score, int? KeywordRank, int? VectorRank);

/// <summary>A vector with the id it is found by and optional metadata to filter on.</summary>
/// <param name="Id">The record's key in the store; storing an id again replaces the record.</param>
/// <param name="Vector">The vector.</param>
/// <param name="Metadata">Values a search can require (see <see cref="IVectorStore.SearchAsync"/>).</param>
public sealed record VectorRecord(string Id, float[] Vector, IReadOnlyDictionary<string, string>? Metadata = null);

/// <summary>A record found by <see cref="IVectorStore.SearchAsync"/>.</summary>
/// <param name="Id">The record's id.</param>
/// <param name="Score">Its similarity to the query (higher is closer).</param>
/// <param name="Metadata">Its metadata.</param>
public sealed record VectorMatch(string Id, double Score, IReadOnlyDictionary<string, string>? Metadata);

/// <summary>
/// Where vectors live and are searched: <c>InMemoryVectorStore</c> is the built-in one; implement this to keep
/// them in a vector database. The filter is a set of metadata values that must all match, so a store can apply it on
/// its side.
/// </summary>
public interface IVectorStore
{
    /// <summary>Adds records, replacing any with the same id.</summary>
    ValueTask UpsertAsync(IReadOnlyList<VectorRecord> records, CancellationToken cancellationToken = default);

    /// <summary>
    /// The <paramref name="top"/> records most similar to <paramref name="query"/>, best first, among those whose metadata
    /// holds every key and value of <paramref name="where"/> (all records when null).
    /// </summary>
    ValueTask<IReadOnlyList<VectorMatch>> SearchAsync(float[] query, int top, IReadOnlyDictionary<string, string>? where = null,
        CancellationToken cancellationToken = default);

    /// <summary>Removes the records with these ids (ids not in the store are ignored).</summary>
    ValueTask DeleteAsync(IReadOnlyList<string> ids, CancellationToken cancellationToken = default);
}

/// <summary>
/// Finds passages for a query. <c>RetrievalIndex</c> is the built-in one; implement this to answer from another
/// search (a database's full-text search, a web search, a hosted index).
/// </summary>
public interface IRetriever
{
    /// <summary>The <paramref name="top"/> best chunks for <paramref name="query"/>, best first.</summary>
    ValueTask<IReadOnlyList<RetrievedChunk>> RetrieveAsync(string query, int top, CancellationToken cancellationToken = default);
}

/// <summary>
/// Re-orders a first search's candidates by reading each with the query. <c>CrossEncoder</c> is the built-in one;
/// implement this to use a hosted re-ranking service.
/// </summary>
public interface IReranker
{
    /// <summary>The best <paramref name="keep"/> of <paramref name="candidates"/> for <paramref name="query"/>, best first, each with its new score.</summary>
    ValueTask<IReadOnlyList<RetrievedChunk>> RerankAsync(string query, IReadOnlyList<RetrievedChunk> candidates, int keep,
        CancellationToken cancellationToken = default);
}
