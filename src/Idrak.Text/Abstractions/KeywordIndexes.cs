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

/// <summary>What a keyword index is built from (<see cref="IKeywordIndexKind"/>): the texts and the settings.</summary>
/// <param name="Texts">The texts to index (their ids are their positions).</param>
/// <param name="Options">The settings (<see cref="KeywordIndexOptions.Parameters"/>' keys among the kind's <see cref="IConfigurableKind.Keys"/>).</param>
public sealed record KeywordIndexRequest(IEnumerable<string> Texts, KeywordIndexOptions Options);

/// <summary>
/// A kind of keyword index, registered in <see cref="KeywordIndexes"/> under its <see cref="IKind.Name"/>: builds an
/// index over a request's texts (<see cref="IKind{TRequest, T}.Create"/>), with settings of its own beyond k1 and b
/// among its <see cref="IConfigurableKind.Keys"/> (<see cref="KeywordIndexOptions.Parameters"/>).
/// </summary>
public interface IKeywordIndexKind : IKind<KeywordIndexRequest, IKeywordIndex>;

/// <summary>
/// The keyword indexes, by name (ignoring case). The library's: <see cref="Bm25"/> (Okapi BM25, <see cref="Bm25Index"/>,
/// with <see cref="KeywordIndexOptions.K1"/> and <see cref="KeywordIndexOptions.B"/>). Register another kind with
/// <see cref="Register"/>; one under the library's name shadows it, which <see cref="Unregister"/> brings back. The
/// registry has no guard: an index holds the texts it indexed, and falling back would index them twice.
/// </summary>
public static class KeywordIndexes
{
    /// <summary>Okapi BM25, the library's keyword index.</summary>
    public const string Bm25 = "bm25";

    private static readonly KindTable<IKeywordIndexKind> Table = new(nameof(KeywordIndexes), "Keyword index", "add it with KeywordIndexes.Register",
        comparer: StringComparer.OrdinalIgnoreCase, unguarded: "an index holds the texts it indexed, and falling back would index them twice");

    static KeywordIndexes() => Overrides.AsLibraryDefaults(() => Register(new Bm25Kind()));

    /// <summary>Registers <paramref name="kind"/> under its <see cref="IKind.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IKeywordIndexKind kind) => Table.Register(kind, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's index <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Names;

    /// <summary>The kind registered as <paramref name="name"/>, or null.</summary>
    public static IKeywordIndexKind? Find(ReadOnlySpan<char> name) => Table.Find(name);

    /// <summary>The kind registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names the registered ones.</exception>
    public static IKeywordIndexKind Get(ReadOnlySpan<char> name) => Table.Get(name);

    /// <summary>The library's kind <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IKeywordIndexKind? Default(string name) => Table.Default(name);

    /// <summary>Who registered the index <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered index as "name: summary", one per line.</summary>
    public static string Describe() => Table.Describe();

    /// <summary>Builds the index <paramref name="name"/> over <paramref name="texts"/> (their ids are their positions).</summary>
    /// <exception cref="NotSupportedException">No index has that name.</exception>
    /// <exception cref="ArgumentException">A setting in <see cref="KeywordIndexOptions.Parameters"/> the kind does not take.</exception>
    public static IKeywordIndex Create(string name, IEnumerable<string> texts, KeywordIndexOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(texts);
        var kind = Get(name);
        options ??= new KeywordIndexOptions();
        foreach (string key in options.Parameters.Keys)
        {
            if (!kind.Keys.Contains(key))
            {
                throw new ArgumentException($"The keyword index {kind.Name} does not take the setting '{key}'; it takes: {(kind.Keys.Count == 0 ? "none" : string.Join(", ", kind.Keys))}.",
                    nameof(options));
            }
        }

        return kind.Create(new KeywordIndexRequest(texts, options));
    }

    // Okapi BM25 with k1 and b.
    private sealed class Bm25Kind : IKeywordIndexKind
    {
        public string Name => Bm25;

        public string Summary => "Okapi BM25: rare words weigh more, repeated words saturate (k1), long texts weigh slightly less (b)";

        public IReadOnlySet<string> Keys => OptionKeys.None;

        public IKeywordIndex Create(KeywordIndexRequest request) => new Bm25Index(request.Texts, request.Options.K1, request.Options.B);
    }
}
