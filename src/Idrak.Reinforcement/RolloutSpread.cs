// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Reinforcement;

/// <summary>
/// How a rollout's episodes are cut into shards and the shards spread over a cluster's ranks (Plan 27, R3b;
/// <see cref="RolloutCollector{TObservation, TAction}"/>). A shard is the episodes that act together: one call of the policy
/// a round for the shard's episodes under way, with a seed of the shard's own. A policy's draws depend on what else is in its
/// call (a language model samples each row of a batch from the seed, the step and the row), so a shard's episodes replay
/// alike wherever the shard is played: with the same number of shards, a seeded run gives the same episodes on any number of
/// ranks. Everything here follows from the episodes' numbers, the shard count and the rank count alone, so every rank
/// computes the same split without a message.
/// </summary>
public static class RolloutSpread
{
    /// <summary>Contiguous runs of episodes in number order (a group's episodes, numbered together, stay together): the default.</summary>
    public const string Blocks = "blocks";

    /// <summary>Episode k (in number order) in shard k mod the shard count: each shard holds an episode of every group.</summary>
    public const string Interleaved = "interleaved";

    /// <summary>The variable of the shard count (<see cref="ReinforcementOptions.Shards"/>).</summary>
    public const string ShardsVariable = "IDRAK_RL_SHARDS";

    /// <summary>The variable of the split (<see cref="ReinforcementOptions.Spread"/>): blocks or interleaved.</summary>
    public const string SpreadVariable = "IDRAK_RL_SPREAD";

    /// <summary>The splits, in order.</summary>
    public static IReadOnlyList<string> Names { get; } = [Blocks, Interleaved];

    /// <summary>The split <paramref name="spread"/> names, in its canonical spelling.</summary>
    /// <exception cref="ArgumentException">Not a split.</exception>
    public static string Parse(string? spread)
    {
        foreach (string name in Names)
        {
            if (string.Equals(name, spread?.Trim(), StringComparison.OrdinalIgnoreCase))
            {
                return name;
            }
        }

        throw new ArgumentException($"A rollout's episodes are spread as {string.Join(" or ", Names)}, not '{spread}' ({SpreadVariable}).", nameof(spread));
    }

    /// <summary>
    /// The shard of each episode: <paramref name="indices"/> are the episodes' numbers (each once), and the result is in their
    /// order. The episodes are ranked by number, then cut as <paramref name="spread"/> says into <paramref name="shards"/>
    /// shards whose sizes differ by one at most.
    /// </summary>
    /// <exception cref="ArgumentException">An unknown split, or two episodes of one number.</exception>
    public static int[] Shards(ReadOnlySpan<int> indices, int shards, string spread = Blocks)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(shards);
        bool blocks = Parse(spread) == Blocks;
        int n = indices.Length;
        var order = new int[n];
        var keys = indices.ToArray();
        for (int i = 0; i < n; i++)
        {
            order[i] = i;
        }

        Array.Sort(keys, order);
        var result = new int[n];
        for (int p = 0; p < n; p++)
        {
            if (p > 0 && keys[p] == keys[p - 1])
            {
                throw new ArgumentException($"Two episodes are numbered {keys[p]}: a rollout numbers each episode once.", nameof(indices));
            }

            result[order[p]] = blocks ? (int)((long)p * shards / n) : p % shards;
        }

        return result;
    }

    /// <summary>
    /// The rank that plays shard <paramref name="shard"/> of <paramref name="shards"/> on <paramref name="worldSize"/> ranks:
    /// contiguous runs of shards, shard s going to rank ⌊s·W/S⌋. With as many shards as ranks, shard r is rank r's; with a
    /// multiple of them, each rank plays S/W shards one after another; with fewer, some ranks play none.
    /// </summary>
    public static int Rank(int shard, int shards, int worldSize)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(shards);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(worldSize);
        ArgumentOutOfRangeException.ThrowIfNegative(shard);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(shard, shards);
        return (int)((long)shard * worldSize / shards);
    }
}
