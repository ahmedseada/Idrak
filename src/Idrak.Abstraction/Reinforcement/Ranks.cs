// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Reinforcement;

/// <summary>
/// A policy that runs on every rank of a cluster (data parallel: each rank holds the same weights on a device of its own and
/// takes part in every update), so a rollout's episodes can be spread over the ranks (Plan 27, R3b): each rank plays its
/// share of them in environments of its own, then <see cref="Gather"/> gives every rank every rank's episodes, and the
/// batch, sorted by episode number, is the same on each, so every rank estimates the same advantages and takes the same
/// data-parallel update. A rollout collector that finds it on its policy spreads the episodes; a policy that is not one
/// plays every episode in its process.
/// </summary>
/// <typeparam name="TObservation">The environment's observations.</typeparam>
/// <typeparam name="TAction">Its actions.</typeparam>
public interface IRolloutRanks<TObservation, TAction>
{
    /// <summary>This rank, from 0.</summary>
    int Rank { get; }

    /// <summary>The number of ranks; 1 when the policy runs in one process (nothing is spread).</summary>
    int WorldSize { get; }

    /// <summary>
    /// Every rank's episodes, in rank order, this rank's <paramref name="mine"/> among them. A collective: every rank calls it
    /// once per rollout, at the same point, with the episodes it played (none is allowed).
    /// </summary>
    /// <param name="mine">The episodes this rank played.</param>
    /// <param name="cancellationToken">Stops the call before it begins (a collective under way runs to its end).</param>
    IReadOnlyList<Trajectory<TObservation, TAction>> Gather(IReadOnlyList<Trajectory<TObservation, TAction>> mine, CancellationToken cancellationToken = default);
}
