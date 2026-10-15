// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;

namespace Idrak.Audio;

/// <summary>
/// The best path of one sequence through CTC's 2L + 1 states (each label with a blank before, between and after them):
/// which label (or the blank) each step reads.
/// </summary>
/// <param name="Frames">One class per step: a label or the blank (torchaudio's <c>forced_align</c> layout).</param>
/// <param name="Starts">Each label's first step.</param>
/// <param name="Lengths">Each label's steps (its duration in steps; the blanks around it are not counted).</param>
/// <param name="LogProbability">The path's log-probability: the sum of the steps' log-probabilities of their classes.</param>
public sealed record CtcPath(int[] Frames, int[] Starts, int[] Lengths, float LogProbability);

/// <summary>
/// Forced alignment of a sequence's CTC log-probabilities to its labels (the Viterbi path through the same states
/// <c>Losses.Ctc</c> sums over), on the host, one pass over the steps: where each label is spoken, the durations a
/// synthesizer trains on (plan 21, S6). The recursion and its ties are torchaudio's <c>forced_align</c>'s: a step stays in
/// its state unless the state before it (or, between two different labels, the one before that) scores strictly more.
/// </summary>
public static class CtcAlignment
{
    /// <summary>The steps a sequence needs for <paramref name="labels"/>: one per label and one more between two equal labels.</summary>
    public static int MinimumSteps(ReadOnlySpan<int> labels)
    {
        int steps = labels.Length;
        for (int i = 1; i < labels.Length; i++)
        {
            steps += labels[i] == labels[i - 1] ? 1 : 0;
        }

        return steps;
    }

    /// <summary>The best path of <paramref name="logProbs"/> (a [steps, classes] tensor of log-probabilities) through <paramref name="labels"/>.</summary>
    /// <exception cref="ArgumentException">The shape is not [steps, classes], a label is the blank or out of range, or the steps cannot hold the labels.</exception>
    public static CtcPath Viterbi(Tensor logProbs, ReadOnlySpan<int> labels, int blank = 0)
    {
        ArgumentNullException.ThrowIfNull(logProbs);
        if (logProbs.Rank != 2)
        {
            throw new ArgumentException($"CTC alignment takes log-probabilities [steps, classes], not {Tensor.FormatShape(logProbs.Shape)}.", nameof(logProbs));
        }

        return Viterbi(logProbs.ToArray(), logProbs.Shape[0], logProbs.Shape[1], labels, blank);
    }

    /// <summary>
    /// The best path of <paramref name="steps"/> rows of <paramref name="classes"/> log-probabilities (row after row)
    /// through <paramref name="labels"/>, with <paramref name="blank"/> the blank class.
    /// </summary>
    /// <exception cref="ArgumentException">The sizes disagree, a label is the blank or out of range, or the steps cannot hold the labels.</exception>
    public static CtcPath Viterbi(ReadOnlySpan<float> logProbs, int steps, int classes, ReadOnlySpan<int> labels, int blank = 0)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(steps);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(classes);
        if ((uint)blank >= (uint)classes)
        {
            throw new ArgumentOutOfRangeException(nameof(blank), $"The blank {blank} is not one of the {classes} classes.");
        }

        if (logProbs.Length < (long)steps * classes)
        {
            throw new ArgumentException($"{logProbs.Length} log-probabilities hold fewer than {steps} steps of {classes} classes.", nameof(logProbs));
        }

        foreach (int label in labels)
        {
            if ((uint)label >= (uint)classes || label == blank)
            {
                throw new ArgumentException($"The label {label} is the blank ({blank}) or not one of the {classes} classes.", nameof(labels));
            }
        }

        int needed = MinimumSteps(labels);
        if (steps < needed)
        {
            throw new ArgumentException($"{steps} steps cannot hold {labels.Length} labels (they need {needed}: one per label and one between two equal labels).", nameof(labels));
        }

        int states = 2 * labels.Length + 1;
        var alpha = ArrayPool<float>.Shared.Rent(2 * states);
        var back = ArrayPool<byte>.Shared.Rent(steps * states);           // 0 stay, 1 from the state before, 2 from two before
        try
        {
            Span<float> previous = alpha.AsSpan(0, states), current = alpha.AsSpan(states, states);
            previous.Fill(float.NegativeInfinity);
            previous[0] = logProbs[blank];
            if (labels.Length > 0)
            {
                previous[1] = logProbs[labels[0]];
            }

            back.AsSpan(0, states).Clear();
            for (int t = 1; t < steps; t++)
            {
                var row = logProbs.Slice(t * classes, classes);
                var pointers = back.AsSpan(t * states, states);
                current[0] = previous[0] + row[blank];
                pointers[0] = 0;
                for (int s = 1; s < states; s++)
                {
                    float stay = previous[s], before = previous[s - 1], skip = float.NegativeInfinity;
                    bool label = (s & 1) == 1;
                    if (label && s > 1 && labels[s >> 1] != labels[(s >> 1) - 1])
                    {
                        skip = previous[s - 2];
                    }

                    // torchaudio's order: two before when it beats both, the one before when it beats both, else stay.
                    float best;
                    byte from;
                    if (skip > before && skip > stay)
                    {
                        (best, from) = (skip, 2);
                    }
                    else if (before > stay && before > skip)
                    {
                        (best, from) = (before, 1);
                    }
                    else
                    {
                        (best, from) = (stay, 0);
                    }

                    current[s] = best + row[label ? labels[s >> 1] : blank];
                    pointers[s] = from;
                }

                var swap = previous;
                previous = current;
                current = swap;
            }

            int state = states == 1 || previous[states - 1] > previous[states - 2] ? states - 1 : states - 2;
            float score = previous[state];
            if (!float.IsFinite(score))
            {
                throw new ArgumentException($"No path of {steps} steps reads the {labels.Length} labels with a finite log-probability.", nameof(logProbs));
            }

            var frames = new int[steps];
            var starts = new int[labels.Length];
            var lengths = new int[labels.Length];
            for (int t = steps - 1; t >= 0; t--)
            {
                bool label = (state & 1) == 1;
                frames[t] = label ? labels[state >> 1] : blank;
                if (label)
                {
                    starts[state >> 1] = t;
                    lengths[state >> 1]++;
                }

                state -= back[t * states + state];
            }

            return new CtcPath(frames, starts, lengths, score);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(back);
            ArrayPool<float>.Shared.Return(alpha);
        }
    }
}
