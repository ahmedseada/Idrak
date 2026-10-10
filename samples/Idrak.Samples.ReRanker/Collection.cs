// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using Idrak.Generation;

namespace Idrak.Samples.ReRanker;

/// <summary>A passage of the collection: the entity (town) it is about and the aspect it answers (null = general text).</summary>
public sealed record Passage(int Id, int Entity, string? Aspect, string Text);

/// <summary>A question, the entity and aspect it asks about, and the id of the passage that answers it.</summary>
public sealed record Query(string Text, int Entity, string Aspect, int Answer);

/// <summary>
/// A synthetic collection about invented towns: for every town, one passage per aspect (population, founding,
/// food, river, climate, sport, mayor, transport) plus two general passages. Questions use different words from
/// their answers ("how many people live in X" is answered by "X has about 40 thousand residents"), while the general
/// passages ("many people visit X ...", "X is a popular place to live ...") share the question's words without
/// answering it, which is exactly where word-matching search goes wrong.
/// </summary>
public sealed class Collection
{
    private static readonly string[] Starts = ["ar", "bel", "cor", "dan", "el", "fen", "gar", "hol", "ir", "jas", "kel", "lor", "mar", "nor", "os", "pel", "quin", "ros", "sel", "tor", "ul", "ver", "wen", "yar", "zel"];
    private static readonly string[] Ends = ["den", "mor", "ton", "vale", "by", "wick", "ford", "ham"];
    private static readonly string[] Dishes = ["fish stew", "apple cake", "lamb pie", "cheese bread", "plum dumplings", "spiced sausage", "honey buns", "mushroom soup"];
    private static readonly string[] Rivers = ["silver", "amber", "willow", "stone", "ash", "copper", "reed", "falcon"];
    private static readonly string[] Sports = ["football", "rugby", "hockey", "cricket", "rowing", "basketball", "cycling", "tennis"];
    private static readonly string[] People = ["ana lund", "ben okafor", "carla diaz", "dev patel", "eva stone", "farid haddad", "gina rossi", "hugo berg"];

    /// <summary>Question templates per aspect; {e} is the town.</summary>
    public static readonly IReadOnlyDictionary<string, string[]> Questions = new Dictionary<string, string[]>
    {
        ["population"] = ["how many people live in {e}", "what is the population of {e}", "how big is {e}"],
        ["founded"] = ["when was {e} founded", "how old is {e}", "what year did {e} begin"],
        ["food"] = ["what food is {e} known for", "what should i eat in {e}", "which dish comes from {e}"],
        ["river"] = ["which river runs through {e}", "what water passes {e}", "is {e} near a river"],
        ["climate"] = ["what is the weather like in {e}", "is {e} warm in summer", "what is the climate of {e}"],
        ["sport"] = ["what sport is popular in {e}", "which game do fans in {e} watch", "what do locals play in {e}"],
        ["mayor"] = ["who runs {e}", "who is in charge of {e}", "who is the mayor of {e}"],
        ["transport"] = ["how do people get around {e}", "is there public transport in {e}", "how can i travel across {e}"],
    };

    public Collection(int towns, int seed)
    {
        var random = new Random(seed);
        Towns = [.. Starts.SelectMany(s => Ends.Select(e => s + e)).OrderBy(_ => random.Next()).Take(towns)];
        for (int t = 0; t < Towns.Count; t++)
        {
            string e = Towns[t];
            T Pick<T>(T[] items) => items[random.Next(items.Length)];
            bool first = random.Next(2) == 0;
            Add(t, "population", first ? $"{e} has about {random.Next(5, 95)} thousand residents ." : $"roughly {random.Next(5, 95)} thousand inhabitants call {e} home .");
            Add(t, "founded", first ? $"{e} was established in {random.Next(1100, 1900)} ." : $"{e} dates back to {random.Next(1100, 1900)} when settlers arrived .");
            Add(t, "food", random.Next(2) == 0 ? $"{e} is famous for its {Pick(Dishes)} ." : $"visitors to {e} should try the local {Pick(Dishes)} .");
            Add(t, "river", random.Next(2) == 0 ? $"the {Pick(Rivers)} river flows through {e} ." : $"{e} sits on the banks of the {Pick(Rivers)} .");
            Add(t, "climate", random.Next(2) == 0 ? $"summers in {e} are {Pick(new[] { "hot", "mild" })} and winters are {Pick(new[] { "cold", "wet" })} ."
                : $"{e} has a {Pick(new[] { "dry", "damp" })} climate with {Pick(new[] { "snowy", "rainy" })} winters .");
            Add(t, "sport", random.Next(2) == 0 ? $"the favourite sport in {e} is {Pick(Sports)} ." : $"most people in {e} follow the local {Pick(Sports)} team .");
            Add(t, "mayor", random.Next(2) == 0 ? $"{e} is led by mayor {Pick(People)} ." : $"{Pick(People)} was elected mayor of {e} .");
            Add(t, "transport", random.Next(2) == 0 ? $"a tram network connects the districts of {e} ." : $"{e} has a {Pick(new[] { "metro", "bus", "ferry" })} service across the city .");
            Add(t, null, $"many people visit {e} to see the old town .");
            Add(t, null, $"{e} is a popular place to live and work .");
        }
    }

    public List<string> Towns { get; }

    public List<Passage> Passages { get; } = [];

    /// <summary>Every question about the given towns (all templates of all aspects).</summary>
    public List<Query> QueriesFor(IEnumerable<int> towns) =>
        [.. towns.SelectMany(t => Questions.SelectMany(q => q.Value.Select(text =>
            new Query(text.Replace("{e}", Towns[t]), t, q.Key, Passages.First(p => p.Entity == t && p.Aspect == q.Key).Id))))];

    private void Add(int town, string? aspect, string text) => Passages.Add(new Passage(Passages.Count, town, aspect, text));
}

/// <summary>Okapi BM25: the standard word-matching ranking function used as the first stage of search.</summary>
public sealed class Bm25
{
    private readonly List<string[]> _documents;
    private readonly Dictionary<string, int>[] _counts;                                    // each document's term counts, made once
    private readonly Dictionary<string, double> _idf = [];
    private readonly double _averageLength;

    public Bm25(IEnumerable<string> documents, double k1 = 1.2, double b = 0.75)
    {
        K1 = k1;
        B = b;
        _documents = [.. documents.Select(d => WordTokenizer.Split(d).ToArray())];
        _averageLength = _documents.Average(d => d.Length);
        _counts = new Dictionary<string, int>[_documents.Count];
        for (int d = 0; d < _documents.Count; d++)
        {
            var counts = _counts[d] = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (string word in _documents[d])
            {
                CollectionsMarshal.GetValueRefOrAddDefault(counts, word, out _)++;
            }
        }

        foreach (var group in _documents.SelectMany(d => d.Distinct()).GroupBy(w => w))
        {
            int n = group.Count();
            _idf[group.Key] = Math.Log(1 + (_documents.Count - n + 0.5) / (n + 0.5));
        }
    }

    public double K1 { get; }

    public double B { get; }

    /// <summary>The ids of the <paramref name="k"/> best-scoring documents for <paramref name="query"/>, best first.</summary>
    public int[] Search(string query, int k)
    {
        var terms = WordTokenizer.Split(query).Where(_idf.ContainsKey).Distinct().ToArray();
        var scores = new double[_documents.Count];
        for (int d = 0; d < _documents.Count; d++)
        {
            var doc = _documents[d];
            foreach (var term in terms)
            {
                int tf = _counts[d].GetValueOrDefault(term);
                if (tf > 0)
                {
                    scores[d] += _idf[term] * tf * (K1 + 1) / (tf + K1 * (1 - B + B * doc.Length / _averageLength));
                }
            }
        }

        return [.. Enumerable.Range(0, scores.Length).OrderByDescending(d => scores[d]).ThenBy(d => d).Take(k)];
    }
}
