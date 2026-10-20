// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using Idrak.Text.Retrieval;

namespace Idrak.Text.Abstractions;

/// <summary>The settings a keyword index is built with (see <see cref="KeywordIndexes.Create"/>).</summary>
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
/// position in the list. The library's is BM25 (<see cref="Bm25Index"/>); register another with
/// <see cref="KeywordIndexes.Register"/> (a database's full-text ranking, BM25 variants), and retrieval
/// (<c>RetrievalIndex</c>, <c>idrak rag</c>) uses it by name.
/// </summary>
public interface IKeywordIndex
{
    /// <summary>The name the index is registered under in <see cref="KeywordIndexes"/> ("bm25").</summary>
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
/// The keyword indexes, by name (ignoring case). The library's: <see cref="Bm25"/> (Okapi BM25, <see cref="Bm25Index"/>,
/// with <see cref="KeywordIndexOptions.K1"/> and <see cref="KeywordIndexOptions.B"/>). Register another with
/// <see cref="Register"/>: a factory from the texts and the options to an index; one under the library's name shadows it,
/// which <see cref="Unregister"/> brings back. The registry has no guard: an index holds the texts it indexed, and falling
/// back would index them twice.
/// </summary>
public static class KeywordIndexes
{
    /// <summary>Okapi BM25, the library's keyword index.</summary>
    public const string Bm25 = "bm25";

    private sealed record Entry(Func<IEnumerable<string>, KeywordIndexOptions, IKeywordIndex> Create, string Summary);

    private static readonly SlotTable<string, Entry> Table = new(nameof(KeywordIndexes), comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "an index holds the texts it indexed, and falling back would index them twice");

    static KeywordIndexes() => Overrides.AsLibraryDefaults(() =>
        Register(Bm25, (texts, options) => new Bm25Index(texts, options.K1, options.B),
            "Okapi BM25: rare words weigh more, repeated words saturate (k1), long texts weigh slightly less (b)"));

    /// <summary>
    /// Registers the keyword index <paramref name="name"/>, built by <paramref name="factory"/> from the texts and the
    /// options (over the library's of that name, which stays behind it).
    /// </summary>
    /// <param name="name">Its name, as retrieval and <c>idrak rag index --keywords</c> give it.</param>
    /// <param name="factory">Builds an index over the texts (their ids are their positions).</param>
    /// <param name="summary">One line for listings.</param>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(string name, Func<IEnumerable<string>, KeywordIndexOptions, IKeywordIndex> factory, string? summary = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(factory);
        Table.Register(name, new Entry(factory, summary ?? name), System.Reflection.Assembly.GetCallingAssembly(), factory);
    }

    /// <summary>Removes the app's index <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The factory registered as <paramref name="name"/>, or null.</summary>
    public static Func<IEnumerable<string>, KeywordIndexOptions, IKeywordIndex>? Find(string name) => Table.Find(name)?.Create;

    /// <summary>The factory registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names the registered ones.</exception>
    public static Func<IEnumerable<string>, KeywordIndexOptions, IKeywordIndex> Get(string name) => Find(name)
        ?? throw new NotSupportedException($"Keyword index '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with KeywordIndexes.Register.");

    /// <summary>The library's factory <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static Func<IEnumerable<string>, KeywordIndexOptions, IKeywordIndex>? Default(string name) => Table.Default(name)?.Create;

    /// <summary>Who registered the index <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered index as "name: summary", one per line.</summary>
    public static string Describe() => string.Join("\n", Table.Keys.Select(k => $"{k}: {Table.Find(k)!.Summary}"));

    /// <summary>Builds the index <paramref name="name"/> over <paramref name="texts"/> (their ids are their positions).</summary>
    /// <exception cref="NotSupportedException">No index has that name.</exception>
    public static IKeywordIndex Create(string name, IEnumerable<string> texts, KeywordIndexOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(texts);
        return Get(name)(texts, options ?? new KeywordIndexOptions());
    }
}
