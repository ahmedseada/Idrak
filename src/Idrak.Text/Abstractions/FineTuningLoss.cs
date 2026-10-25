// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Text.Abstractions;

/// <summary>
/// A fine-tuning loss (<see cref="FineTuningOptions.Loss"/>): the loss of one batch from the log-probabilities of its
/// trained tokens (<see cref="Compute"/>), as a single-element tensor built with ordinary tensor operations from
/// <see cref="FineTuningLossInput.LogProbabilities"/> (and, for losses against a reference model,
/// <see cref="FineTuningLossInput.ReferenceLogProbabilities"/>), and what it needs beside them: preference pairs, the
/// reference, a teacher. Training checks those before its first step (a loss that needs pairs or a teacher without them
/// is refused), and a trainer prepares the reference beforehand only for a loss that reads it. The losses of the batches
/// of one optimizer step are added, so a loss averaged over the step divides by <see cref="FineTuningLossInput.StepTokens"/>
/// or <see cref="FineTuningLossInput.StepSequences"/>. Built-in: <see cref="FineTuningLosses"/>; one of one's own from a
/// function: <see cref="FineTuningLosses.Of"/>.
/// </summary>
public interface IFineTuningLoss : IKind
{
    /// <summary>Whether it reads preference pairs (<see cref="FineTuningLossInput.Pairs"/>): trained only on <see cref="PreferencePair"/> data.</summary>
    bool NeedsPairs { get; }

    /// <summary>Whether it reads <see cref="FineTuningLossInput.ReferenceLogProbabilities"/>: a trainer computes them beforehand only then.</summary>
    bool NeedsReference { get; }

    /// <summary>Whether it reads a teacher's divergences (<see cref="FineTuningLossInput.TeacherDivergence(float)"/>): trained only with <see cref="FineTuningOptions.Teacher"/>.</summary>
    bool NeedsTeacher { get; }

    /// <summary>The batch's loss, one element.</summary>
    /// <param name="input">The batch's log-probabilities and how they divide into sequences.</param>
    Tensor Compute(FineTuningLossInput input);
}
