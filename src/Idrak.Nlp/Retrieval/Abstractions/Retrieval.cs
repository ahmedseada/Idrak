// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Nlp.Retrieval.Abstractions;

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
