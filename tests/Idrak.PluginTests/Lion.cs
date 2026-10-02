// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Optimizers;

namespace Idrak.PluginTests;

/// <summary>
/// Lion (Chen et al., 2023), written as an outside package writes an optimizer: the update is the sign of an
/// interpolation of the momentum and the gradient, applied with public tensor operations and in-place writes.
/// c = β1·m + (1 - β1)·g; p -= lr·(sign(c) + λ·p); m = β2·m + (1 - β2)·g.
/// </summary>
public sealed class Lion(IEnumerable<Tensor> parameters, float learningRate = 1e-4f, float beta1 = 0.9f, float beta2 = 0.99f, float weightDecay = 0f)
    : Optimizer(parameters, learningRate)
{
    private Tensor?[]? _momentum;

    /// <summary>Interpolation factor of the update's direction.</summary>
    public float Beta1 { get; } = beta1;

    /// <summary>Decay rate of the momentum.</summary>
    public float Beta2 { get; } = beta2;

    /// <summary>Decoupled weight decay λ.</summary>
    public float WeightDecay { get; } = weightDecay;

    /// <inheritdoc />
    public override void Step()
    {
        _momentum ??= new Tensor?[Parameters.Count];
        ApplyDecoupledWeightDecay(WeightDecay);
        for (int i = 0; i < Parameters.Count; i++)
        {
            var p = Parameters[i];
            if (p.Grad is not { } g)
            {
                continue;
            }

            var m = _momentum[i] ??= CreateState(p);
            using var scope = new TensorScope();
            using (Autograd.NoGrad())
            {
                var direction = (m * Beta1 + g * (1f - Beta1)).Sign();
                p.AddScaled(direction, -LearningRate);
                m.Scale(Beta2);
                m.AddScaled(g, 1f - Beta2);
            }
        }
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        foreach (var m in _momentum ?? [])
        {
            m?.Dispose();
        }

        base.Dispose();
    }
}
