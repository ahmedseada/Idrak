// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Training;

/// <summary>A kind reads what it uses and leaves the rest.
/// </summary>
/// <param name="LearningRate">The step size the optimizer starts with (the peak a schedule works from).</param>
/// <param name="WeightDecay">Weight decay: decoupled for AdamW, an L2 term on the gradients for Adam and SGD.</param>
/// <param name="Momentum">SGD's momentum.</param>
public sealed record OptimizerSettings(float LearningRate, float WeightDecay = 0f, float Momentum = 0.9f)
{
    /// <summary>The first moment's decay (the Adam kinds); null for the kind's own (Adam's 0.9).</summary>
    public float? Beta1 { get; init; }

    /// <summary>The second moment's decay (the Adam kinds); null for the kind's own (Adam's 0.999).</summary>
    public float? Beta2 { get; init; }

    /// <summary>The term added to the second moment's root (the Adam kinds); null for the kind's own (Adam's 1e-8).</summary>
    public float? Epsilon { get; init; }
}

/// <summary>What an optimizer is made with (<see cref="IOptimizerKind.Create"/>): the tensors it updates and its settings.</summary>
/// <param name="Parameters">The tensors it updates, in the order its state is saved.</param>
/// <param name="Settings">The rate, decay and the kind's own settings.</param>
public sealed record OptimizerRequest(IReadOnlyList<Tensor> Parameters, OptimizerSettings Settings);

/// <summary>
/// A kind of optimizer, made by name from an <see cref="OptimizerRequest"/>: the parameters it updates and its <see cref="OptimizerSettings"/>. Thread-safe: one kind makes the optimizers of several runs.
/// </summary>
public interface IOptimizerKind
{
    /// <summary>A new optimizer over the request's parameters, starting at its <see cref="OptimizerSettings.LearningRate"/>.</summary>
    /// <param name="request">The tensors it updates and its settings.</param>
    /// <exception cref="ArgumentException">A setting the kind cannot use (a negative rate, say).</exception>
    Optimizer Create(OptimizerRequest request);
}

