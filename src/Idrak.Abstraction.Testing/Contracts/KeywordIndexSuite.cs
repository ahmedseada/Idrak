// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Testing;

/// <summary>A built keyword index's search, as an index under test gives it: the best texts for a query, best first.</summary>
/// <param name="query">The query.</param>
/// <param name="top">How many texts at most.</param>
public delegate IReadOnlyList<SearchHit> KeywordSearch(string query, int top);

/// <summary>
/// A keyword index under test (Idrak.Text's <c>IKeywordIndex</c>, built by a factory of <c>KeywordIndexes</c>, given as a
/// delegate since this kit depends on Idrak.Abstraction alone): builds an index over the texts, their ids their positions.
/// </summary>
/// <param name="Build">Builds the index over the texts and returns its search.</param>
public sealed record KeywordIndexUnderTest(Func<IReadOnlyList<string>, KeywordSearch> Build);

/// <summary>
/// The checks of a keyword index on small corpora (an empty one, repeated words, mixed case and punctuation, Arabic) and
/// random ones: at most the number of texts asked for, ids in range and each once, finite positive scores, best first with
/// ties by id, a query of words no text has finds nothing, the same hits again and from several threads at once, and,
/// given a <see cref="Reference"/> (the library's index it replaces), the same ids in the same order and scores within
/// <see cref="Tolerance"/>.
/// </summary>
/// <param name="reference">The library's index to agree with, or null.</param>
public sealed class KeywordIndexSuite(KeywordIndexUnderTest? reference = null) : ContractSuite<KeywordIndexUnderTest>
{
    // A word none of the corpora (fixed or random) holds.
    private const string Absent = "qzxjvw";

    private static readonly string[] Words = ["river", "town", "bridge", "market", "people", "live", "north", "old", "the", "a", "in", "of", "mayor", "food", "rain"];

    /// <summary>The library's index to agree with, or null.</summary>
    public KeywordIndexUnderTest? Reference { get; } = reference;

    /// <summary>The relative agreement wanted with the reference's scores.</summary>
    public double Tolerance { get; init; } = 1e-9;

    /// <inheritdoc />
    public override string Name => "keyword-index";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        yield return Case("empty corpus", [], ["river", ""], 3);
        yield return Case("one text", ["the river runs north"], ["river", "north river", "south"], 5);
        yield return Case("repeated words", ["rain rain rain", "rain", "rain and sun", "sun sun"], ["rain", "sun", "rain sun"], 4);
        yield return Case("case and punctuation", ["The Old Bridge.", "an old, OLD market!", "bridge: market? town"], ["old bridge", "Market", "TOWN!"], 2);
        yield return Case("arabic", ["النهر الكبير", "المدينة القديمة", "النهر"],
            ["النهر", "المدينة"], 3);
        yield return Case("ties", ["a b", "b a", "a b", "c"], ["a", "b"], 10);
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        int count = random.Next(1, large ? 3000 : 60);
        string Text(int words) => string.Join(' ', Enumerable.Range(0, words).Select(_ => Words[random.Next(Words.Length)]));
        string[] texts = [.. Enumerable.Range(0, count).Select(_ => Text(random.Next(0, 12)))];
        string[] queries = [.. Enumerable.Range(0, 4).Select(_ => Text(random.Next(1, 4)))];
        return Case($"random corpus of {count}", texts, queries, random.Next(1, count + 3));
    }

    /// <inheritdoc />
    public override void Run(KeywordIndexUnderTest implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var data = @case.Data;
        string[] texts = [.. data["texts"]!.AsArray().Select(t => (string)t!)];
        string[] queries = [.. data["queries"]!.AsArray().Select(q => (string)q!)];
        int top = (int)data["top"]!;
        var search = implementation.Build(texts);
        var expected = Reference?.Build(texts);
        foreach (string query in queries.Append(Absent))
        {
            var hits = search(query, top);
            checks.Check("at most top hits", hits.Count <= top, $"'{query}': {hits.Count} hits for top {top}");
            checks.Check("ids in range, each once", hits.All(h => h.Id >= 0 && h.Id < texts.Length) && hits.Select(h => h.Id).Distinct().Count() == hits.Count,
                $"'{query}': ids {string.Join(", ", hits.Select(h => h.Id))} of {texts.Length} texts");
            checks.Check("scores finite and positive", hits.All(h => double.IsFinite(h.Score) && h.Score > 0), $"'{query}': scores {string.Join(", ", hits.Select(h => h.Score))}");
            checks.Check("best first, ties by id", hits.Zip(hits.Skip(1)).All(p => p.First.Score > p.Second.Score || p.First.Score == p.Second.Score && p.First.Id < p.Second.Id),
                $"'{query}': {string.Join(", ", hits.Select(h => $"{h.Id}:{h.Score}"))}");
            if (query == Absent)
            {
                checks.Check("words no text has find nothing", hits.Count == 0, $"'{query}' found {hits.Count}");
            }

            checks.Compare("the same hits again", Same(hits, search(query, top), 0));
            var parallel = new IReadOnlyList<SearchHit>[8];
            Parallel.For(0, parallel.Length, i => parallel[i] = search(query, top));
            checks.Compare("the same hits from several threads", parallel.Select(p => Same(hits, p, 0)).FirstOrDefault(d => d is not null));
            if (expected is not null)
            {
                checks.Compare("agrees with the reference", Same(expected(query, top), hits, Tolerance) is { } d ? $"'{query}': {d}" : null);
            }
            else
            {
                checks.Skip("agrees with the reference", "no reference index given");
            }
        }
    }

    // How two hit lists differ (ids and order exactly, scores within a relative tolerance), or null.
    private static string? Same(IReadOnlyList<SearchHit> expected, IReadOnlyList<SearchHit> actual, double tolerance)
    {
        if (expected.Count != actual.Count)
        {
            return $"{actual.Count} hits, expected {expected.Count}";
        }

        for (int i = 0; i < expected.Count; i++)
        {
            if (expected[i].Id != actual[i].Id || Math.Abs(expected[i].Score - actual[i].Score) > tolerance * Math.Max(1, Math.Abs(expected[i].Score)))
            {
                return $"hit {i}: {actual[i].Id} ({actual[i].Score}), expected {expected[i].Id} ({expected[i].Score})";
            }
        }

        return null;
    }

    private static ContractCase Case(string name, string[] texts, string[] queries, int top) =>
        new(name, new JsonObject
        {
            ["texts"] = new JsonArray([.. texts.Select(t => (JsonNode)t)]),
            ["queries"] = new JsonArray([.. queries.Select(q => (JsonNode)q)]),
            ["top"] = top,
        });
}
