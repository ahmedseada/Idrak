// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Retrieval;

/// <summary>One search result: the found item's id (a chunk, an image, a clip) and its score (higher is better).</summary>
/// <param name="Id">The item's id in the searched collection.</param>
/// <param name="Score">Its score: similarity, BM25 or fused (higher is better).</param>
public readonly record struct SearchHit(int Id, double Score);

/// <summary>
/// The best <c>keep</c> hits offered so far (best first, ties to the lower id), kept in a heap whose root is the worst of
/// them: a search of any index offers every candidate once and reads the best at the end.
/// </summary>
/// <param name="keep">How many hits to keep (none when zero or less).</param>
public sealed class TopHits(int keep)
{
    private readonly PriorityQueue<int, (double Score, int Id)> _heap = new(Math.Max(keep, 0), WorstFirst.Instance);

    /// <summary>Offers a candidate: kept when it is among the best so far.</summary>
    public void Offer(int id, double score)
    {
        if (keep <= 0)
        {
            return;
        }

        if (_heap.Count < keep)
        {
            _heap.Enqueue(id, (score, id));
        }
        else if (_heap.TryPeek(out _, out var worst) && WorstFirst.Instance.Compare((score, id), worst) > 0)
        {
            _heap.DequeueEnqueue(id, (score, id));
        }
    }

    /// <summary>The hits kept, best first (ties to the lower id).</summary>
    public SearchHit[] ToArray()
    {
        var hits = new SearchHit[_heap.Count];
        for (int r = hits.Length - 1; _heap.TryDequeue(out int id, out var entry); r--)
        {
            hits[r] = new SearchHit(id, entry.Score);
        }

        return hits;
    }

    // A lower score, or the same score with a higher id, is worse.
    private sealed class WorstFirst : IComparer<(double Score, int Id)>
    {
        public static readonly WorstFirst Instance = new();

        public int Compare((double Score, int Id) x, (double Score, int Id) y) =>
            x.Score != y.Score ? x.Score.CompareTo(y.Score) : y.Id.CompareTo(x.Id);
    }
}
