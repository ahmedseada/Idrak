// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Generation;

/// <summary>One reading of a sequence: its labels (repeats collapsed, blanks dropped) and their log-probability.</summary>
/// <param name="Labels">The labels, in order.</param>
/// <param name="LogProbability">The decoder's score: the log-probability of the best path for "greedy", of the labels
/// summed over the paths the beam kept for "beam".</param>
public sealed record CtcHypothesis(int[] Labels, float LogProbability);

/// <summary>What a CTC decoder is asked for.</summary>
public sealed record CtcDecodeOptions
{
    /// <summary>The blank class (default 0).</summary>
    public int Blank { get; init; }

    /// <summary>How many readings to return, the most likely first (default 1).</summary>
    public int Results { get; init; } = 1;

    /// <summary>The prefixes a beam search keeps after each step (default 10).</summary>
    public int BeamWidth { get; init; } = 10;

    /// <summary>
    /// Per-step pruning for a beam search: only the <see cref="TopClasses"/> most likely classes of a step extend the
    /// prefixes (null: every class). With many classes and few likely ones per step it is much faster for the same result.
    /// </summary>
    public int? TopClasses { get; init; }

    /// <summary>
    /// Per-step pruning for a beam search: classes whose log-probability at a step is below this extend no prefix
    /// (null: none skipped). The step's most likely class is always kept.
    /// </summary>
    public float? MinLogProbability { get; init; }
}

/// <summary></summary>
public interface ICtcDecoder
{
    /// <summary>
    /// Reads one sequence's log-probabilities, <paramref name="steps"/> rows of <paramref name="classes"/> (a log-softmax per
    /// step, as a network trained with CTC gives them), as label sequences, the most likely first.
    /// </summary>
    IReadOnlyList<CtcHypothesis> Decode(ReadOnlyMemory<float> logProbs, int steps, int classes, CtcDecodeOptions options);
}

