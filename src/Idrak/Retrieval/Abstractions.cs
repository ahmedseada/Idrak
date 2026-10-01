// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Retrieval;

/// <summary>
/// Turns texts into vectors whose dot product (or cosine) measures how related they are. <see cref="TextEncoder"/> is the
/// built-in one; implement this to index with another model or a hosted embedding service.
/// </summary>
public interface IEmbedder
{
    /// <summary>The vectors of <paramref name="texts"/>, in the same order, all of the same length.</summary>
    ValueTask<float[][]> EmbedAsync(IReadOnlyList<string> texts, CancellationToken cancellationToken = default);
}

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
/// Where vectors live and are searched: <see cref="InMemoryVectorStore"/> is the built-in one; implement this to keep
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
/// Finds passages for a query. <see cref="RetrievalIndex"/> is the built-in one; implement this to answer from another
/// search (a database's full-text search, a web search, a hosted index).
/// </summary>
public interface IRetriever
{
    /// <summary>The <paramref name="top"/> best chunks for <paramref name="query"/>, best first.</summary>
    ValueTask<IReadOnlyList<RetrievedChunk>> RetrieveAsync(string query, int top, CancellationToken cancellationToken = default);
}

/// <summary>
/// Re-orders a first search's candidates by reading each with the query. <see cref="CrossEncoder"/> is the built-in one;
/// implement this to use a hosted re-ranking service.
/// </summary>
public interface IReranker
{
    /// <summary>The best <paramref name="keep"/> of <paramref name="candidates"/> for <paramref name="query"/>, best first, each with its new score.</summary>
    ValueTask<IReadOnlyList<RetrievedChunk>> RerankAsync(string query, IReadOnlyList<RetrievedChunk> candidates, int keep,
        CancellationToken cancellationToken = default);
}
