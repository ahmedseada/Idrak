// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Retrieval;

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
