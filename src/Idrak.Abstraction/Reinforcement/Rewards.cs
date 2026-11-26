// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Reinforcement;

/// <summary>What a reward function scores: an answer to a prompt, with the expected answer and what the environment reported.</summary>
/// <param name="Prompt">The prompt (or the task) the answer is to.</param>
/// <param name="Answer">The policy's answer: a reply's text, the end of an episode's transcript.</param>
/// <param name="Reference">The expected answer, when the task has one; null otherwise.</param>
/// <param name="Info">What the environment reported (tests passed, rounds used, files changed), read by its keys (<see cref="EpisodeInfoKey{T}"/>); null for nothing.</param>
public sealed record RewardInput(string Prompt, string Answer, string? Reference = null, EpisodeInfo? Info = null);

/// <summary>Each answer's score depends on that answer alone, so a batch scores what its inputs score one by one (a
/// judge batches them for speed, not to compare them); the same input scores the same every time. Thread-safe: one
/// reward function scores many episodes at once.
/// </summary>
public interface IRewardFunction
{
    /// <summary>
    /// The score of each of <paramref name="inputs"/>, in order, finite (higher is better). Completes synchronously
    /// when
    /// nothing is waited for (a metric); awaits when something is (a judge model): await the result once.
    /// </summary>
    /// <param name="inputs">The answers to score.</param>
    /// <param name="options">The reward's options by key (<c>Keys</c>), values as text; null
    /// for its defaults.</param>
    /// <param name="cancellationToken">Stops the call.</param>
    /// <exception cref="ArgumentException">An option the reward does not take (the message names it and <c>Keys</c>), or a bad value.</exception>
    ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null, CancellationToken cancellationToken = default);
}

