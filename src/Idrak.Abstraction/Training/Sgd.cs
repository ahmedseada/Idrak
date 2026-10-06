// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Training;

/// <summary>Stochastic gradient descent with optional momentum and L2 weight decay: g += λp; v = μv + g; p -= lr·v.</summary>
public sealed class Sgd(IEnumerable<Tensor> parameters, float learningRate = 0.01f, float momentum = 0f, float weightDecay = 0f) : Optimizer(parameters, learningRate)
{
    /// <summary>L2 penalty λ added to the gradients.</summary>
    public float WeightDecay { get; } = weightDecay;

    private Tensor?[]? _velocity;

    /// <summary>Momentum factor μ (0 disables momentum).</summary>
    public float Momentum { get; } = momentum;

    /// <inheritdoc />
    public override void Step()
    {
        _velocity ??= new Tensor?[Parameters.Count];
        ApplyCoupledWeightDecay(WeightDecay);
        for (int i = 0; i < Parameters.Count; i++)
        {
            var p = Parameters[i];
            if (p.Grad is null)
            {
                continue;
            }

            Tensor? v = null;
            if (Momentum != 0f)
            {
                v = _velocity[i] ??= CreateState(p);
            }

            p.Backend.SgdStep(p.Storage, p.Grad.Storage, v?.Storage, p.Size, LearningRate, Momentum);
        }
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        foreach (var v in _velocity ?? [])
        {
            v?.Dispose();
        }

        base.Dispose();
    }
}
