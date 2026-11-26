// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Text.Retrieval;

namespace Idrak.Text.Abstractions;

/// <summary>The settings a keyword index is built with (see <c>KeywordIndexes.Create</c>).</summary>
public sealed record KeywordIndexOptions
{
    /// <summary>Term-frequency saturation (BM25's k1; typically 1.2 to 2.0).</summary>
    public double K1 { get; init; } = 1.2;

    /// <summary>Length normalization (BM25's b), 0 (none) to 1 (full); typically 0.75.</summary>
    public double B { get; init; } = 0.75;

    /// <summary>Settings of an index of one's own beyond <see cref="K1"/> and <see cref="B"/>, by name (empty for the library's).</summary>
    public IReadOnlyDictionary<string, double> Parameters { get; init; } = new Dictionary<string, double>();
}

/// <summary>
/// Keyword search over a fixed list of texts: the best texts for a query by the words they share, a text's id its
/// position in the list. The library's is BM25 (<see cref="Bm25Index"/>); another (a database's full-text ranking,
/// BM25 variants) is an app's. Picking one by its name is not on the new design yet.
/// </summary>
public interface IKeywordIndex
{
    /// <summary>The index's name ("bm25").</summary>
    string Name { get; }

    /// <summary>The settings it was built with.</summary>
    KeywordIndexOptions Options { get; }

    /// <summary>Number of indexed texts.</summary>
    int Count { get; }

    /// <summary>
    /// The <paramref name="top"/> best texts for <paramref name="query"/>, best first (ties by id), each with its score;
    /// texts sharing no word with the query are left out.
    /// </summary>
    IReadOnlyList<SearchHit> Search(string query, int top);
}

/// <summary>
/// The names of the keyword indexes. The library's: <see cref="Bm25"/> (Okapi BM25, <see cref="Bm25Index"/>, with
/// <see cref="KeywordIndexOptions.K1"/> and <see cref="KeywordIndexOptions.B"/>).
/// </summary>
public static class KeywordIndexes
{
    /// <summary>Okapi BM25, the library's keyword index.</summary>
    public const string Bm25 = "bm25";
}
