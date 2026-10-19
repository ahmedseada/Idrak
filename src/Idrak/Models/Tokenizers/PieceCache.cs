// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>
/// Encoded pieces by their text, shared by concurrent encodes and looked up by span (a hit allocates nothing). An entry is
/// the piece's ids and, when the cache keeps them (<see cref="Ranges"/>), each token's characters in the piece, so a
/// detailed encode that hits aligns nothing. Entries count against a <see cref="Budget"/> (<see cref="TokenizerOptions.PieceCacheLimit"/>),
/// which several caches of one tokenizer share, so together they never hold more pieces than the limit.
/// </summary>
internal sealed class PieceCache
{
    /// <summary>Pieces longer than this are not cached (long pieces rarely repeat; their ranges fit 16 bits each).</summary>
    public const int MaxChars = 256;

    private readonly ConcurrentDictionary<string, int[]> _entries = new();
    private readonly ConcurrentDictionary<string, int[]>.AlternateLookup<ReadOnlySpan<char>> _spans;

    /// <summary>A cache of its own budget of <paramref name="limit"/> pieces.</summary>
    public PieceCache(int limit, bool ranges)
        : this(new Budget(limit), ranges)
    {
    }

    /// <summary>A cache whose pieces count against <paramref name="budget"/> (shared with the tokenizer's other caches).</summary>
    public PieceCache(Budget budget, bool ranges)
    {
        Shared = budget;
        Ranges = ranges;
        _spans = _entries.GetAlternateLookup<ReadOnlySpan<char>>();
    }

    /// <summary>The pieces this cache may add, counted with every cache that shares it.</summary>
    public Budget Shared { get; }

    /// <summary>Whether entries keep each token's range in the piece (and <see cref="Add"/> needs them).</summary>
    public bool Ranges { get; }

    /// <summary>The piece's ids, when cached.</summary>
    public bool TryGet(ReadOnlySpan<char> piece, out ReadOnlySpan<int> ids)
    {
        if (!_spans.TryGetValue(piece, out var entry))
        {
            ids = default;
            return false;
        }

        ids = Ranges ? entry.AsSpan(0, entry.Length >> 1) : entry;
        return true;
    }

    /// <summary>The piece's ids and each token's range in it (packed: <see cref="Range"/> reads one), when cached.</summary>
    public bool TryGet(ReadOnlySpan<char> piece, out ReadOnlySpan<int> ids, out ReadOnlySpan<int> ranges)
    {
        if (!Ranges)
        {
            throw new InvalidOperationException("This piece cache keeps no ranges.");
        }

        if (!_spans.TryGetValue(piece, out var entry))
        {
            ids = default;
            ranges = default;
            return false;
        }

        int count = entry.Length >> 1;
        ids = entry.AsSpan(0, count);
        ranges = entry.AsSpan(count);
        return true;
    }

    /// <summary>Whether a piece of this length would be added now (room left in the budget): when not, its ranges need not be found.</summary>
    public bool Accepts(int length) => length <= MaxChars && Volatile.Read(ref Shared.Count) < Shared.Limit;

    /// <summary>Caches the piece's ids (and, when the cache keeps them, each token's range in the piece) if the budget has room.</summary>
    public void Add(ReadOnlySpan<char> piece, ReadOnlySpan<int> ids, List<(int Start, int Length)>? ranges)
    {
        if (!Accepts(piece.Length))
        {
            return;
        }

        int[] entry;
        if (Ranges)
        {
            if (ranges is null || ranges.Count != ids.Length)
            {
                throw new ArgumentException("A piece cache with ranges needs one range per id.", nameof(ranges));
            }

            entry = new int[ids.Length * 2];
            ids.CopyTo(entry);
            for (int i = 0; i < ranges.Count; i++)
            {
                var (start, length) = ranges[i];
                entry[ids.Length + i] = start | (length << 16);              // both at most MaxChars
            }
        }
        else
        {
            entry = ids.ToArray();
        }

        if (_spans.TryAdd(piece, entry))
        {
            Interlocked.Increment(ref Shared.Count);
        }
    }

    /// <summary>A packed range of an entry: the token's first character in the piece and its character count.</summary>
    public static (int Start, int Length) Range(int packed) => (packed & 0xFFFF, packed >>> 16);

    /// <summary>The pieces the caches that share it may hold, and how many they hold.</summary>
    public sealed class Budget(int limit)
    {
        public readonly int Limit = limit;
        public int Count;
    }
}
