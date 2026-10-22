// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Reinforcement.Abstractions;

/// <summary>
/// A policy whose optimizer a trainer may reach (<see cref="ReinforcementTrainer{TObservation, TAction}"/>): the trainer
/// sets its learning rate by a schedule from <c>LearningRateSchedules</c>, once per optimizer step, and the gradient-norm
/// clip its updates apply. <see cref="ModulePolicy{TAction}"/> implements it. A policy that does not keeps its own rate and
/// clip (a language-model policy schedules through its fine-tuning session), and the trainer leaves them alone.
/// </summary>
public interface IScheduledPolicy
{
    /// <summary>The optimizer every <c>Update</c> steps; its <see cref="Optimizer.LearningRate"/> is the schedule's base.</summary>
    Optimizer Optimizer { get; }

    /// <summary>The largest global L2 norm of the gradients an update applies (larger ones are scaled down to it); 0 for no clip.</summary>
    float MaxGradientNorm { get; set; }

    /// <summary>Where the policy's weights live (the trainer reports its memory).</summary>
    Device Device { get; }
}
