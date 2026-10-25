// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Reinforcement;

namespace Idrak.Text.Reinforcement.Abstractions;

/// <summary>
/// An environment kind whose replies are scored by a reward expression (a <see cref="RewardFunctions"/> name, or weighted
/// terms "exact:1,length:0.1", each term's options as "term.key"), over at most a number of replies: the option keys a
/// trainer's command line (<c>idrak reinforce --reward</c>, <c>--rounds</c>, <c>--no-think</c>) sets them by.
/// </summary>
public interface IRewardedEnvironmentKind : IEnvironmentKind
{
    /// <summary>The option of the reward expression ("reward").</summary>
    string RewardKey { get; }

    /// <summary>The option of the most replies an episode has ("rounds"); null when the kind has none.</summary>
    string? RoundsKey { get; }

    /// <summary>The option of the reasoning mode its prompts render with, true or false ("think"); null when the kind has none.</summary>
    string? ThinkKey { get; }
}
