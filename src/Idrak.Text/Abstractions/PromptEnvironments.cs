// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Reinforcement;

namespace Idrak.Text.Abstractions;

/// <summary>
/// A chat environment kind whose tasks are the prompts of conversations a trainer reads as tuning data
/// (<c>idrak reinforce --data</c>, as <c>idrak tune</c> reads it): the trainer makes the kind of those tasks
/// (<see cref="WithTasks"/>), counting answers' tokens with the policy's tokenizer, instead of giving it a data option.
/// Idrak.Text's <c>prompts</c> kind is one.
/// </summary>
public interface IPromptDataEnvironmentKind : IEnvironmentKind
{
    /// <summary>The kind's own option of a data file, which the trainer's data takes the place of (left out of what it accepts); null when it has none.</summary>
    string? DataKey { get; }

    /// <summary>A kind of this one's name over the prompts of <paramref name="conversations"/>, counting each answer's tokens with <paramref name="tokenizer"/> (null: not counted).</summary>
    IEnvironmentKind WithTasks(IReadOnlyList<ChatTranscript> conversations, ITokenizer? tokenizer);
}
