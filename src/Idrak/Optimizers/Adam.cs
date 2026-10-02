// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Backends;

namespace Idrak.Optimizers;

/// <summary>
/// Adam (Kingma &amp; Ba, 2015): per-parameter adaptive steps from running averages of the
/// gradient and its square, with bias correction. A good default for most networks.
/// </summary>
/// <remarks>
/// <paramref name="weightDecay"/> adds an L2 penalty to the gradients (classic Adam). For decoupled weight
/// decay, which usually generalizes better, use <see cref="AdamW"/>.
/// </remarks>
public class Adam(IEnumerable<Tensor> parameters, float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f, float weightDecay = 0f)
    : Optimizer(parameters, learningRate)
{
    /// <summary>Weight decay factor λ.</summary>
    public float WeightDecay { get; } = weightDecay;

    /// <summary>True for AdamW-style decay (p -= lr·λ·p before the step) instead of L2 on the gradients.</summary>
    protected virtual bool DecoupledWeightDecay => false;

    private Tensor?[]? _m;
    private Tensor?[]? _v;
    private int _step;
    private IDisposable? _fused;

    /// <summary>Decay rate of the first-moment (mean) estimate.</summary>
    public float Beta1 { get; } = beta1;

    /// <summary>Decay rate of the second-moment (uncentered variance) estimate.</summary>
    public float Beta2 { get; } = beta2;

    /// <summary>Term added to the denominator for numerical stability.</summary>
    public float Epsilon { get; } = epsilon;

    /// <inheritdoc />
    public override void Step()
    {
        _step++;
        _m ??= new Tensor?[Parameters.Count];
        _v ??= new Tensor?[Parameters.Count];
        if (WeightDecay != 0f)
        {
            if (DecoupledWeightDecay)
            {
                ApplyDecoupledWeightDecay(WeightDecay);
            }
            else
            {
                ApplyCoupledWeightDecay(WeightDecay);
            }
        }

        // Fold both bias corrections into the step size: lr * sqrt(1 - β2^t) / (1 - β1^t).
        float correctedLr = (float)(LearningRate * Math.Sqrt(1 - Math.Pow(Beta2, _step)) / (1 - Math.Pow(Beta1, _step)));
        for (int i = 0; i < Parameters.Count; i++)
        {
            var p = Parameters[i];
            if (p.Grad is null)
            {
                continue;
            }

            var m = _m[i] ??= CreateState(p);
            var v = _v[i] ??= CreateState(p);
            p.Backend.AdamStep(p.Storage, p.Grad.Storage, m.Storage, v.Storage, p.Size, correctedLr, Beta1, Beta2, Epsilon);
        }
    }

    /// <summary>
    /// Clipping and the update of every parameter in three device passes (global norm, clipping factor, update, which
    /// also zeroes the gradients) where the device has them; otherwise as <see cref="Optimizer.ClipAndStep"/>.
    /// </summary>
    public override void ClipAndStep(float maxNorm)
    {
        var withGrad = Enumerable.Range(0, Parameters.Count).Where(i => Parameters[i].Grad is not null).ToList();
        bool fusable = (GetType() == typeof(Adam) || GetType() == typeof(AdamW)) && (WeightDecay == 0f || DecoupledWeightDecay) && withGrad.Count > 0
                       && withGrad.All(i => Parameters[i].Device == Parameters[withGrad[0]].Device);
        if (!fusable)
        {
            base.ClipAndStep(maxNorm);
            return;
        }

        _m ??= new Tensor?[Parameters.Count];
        _v ??= new Tensor?[Parameters.Count];
        var tensors = new (Storage P, Storage G, Storage M, Storage V, int N)[withGrad.Count];
        for (int j = 0; j < withGrad.Count; j++)
        {
            var p = Parameters[withGrad[j]];
            var m = _m[withGrad[j]] ??= CreateState(p);
            var v = _v[withGrad[j]] ??= CreateState(p);
            tensors[j] = (p.Storage, p.Grad!.Storage, m.Storage, v.Storage, p.Size);
        }

        int step = _step + 1;
        float correctedLr = (float)(LearningRate * Math.Sqrt(1 - Math.Pow(Beta2, step)) / (1 - Math.Pow(Beta1, step)));
        float decay = WeightDecay != 0f ? 1f - LearningRate * WeightDecay : 1f;
        if (!Parameters[withGrad[0]].Backend.FusedAdamW(tensors, ref _fused, maxNorm, correctedLr, decay, Beta1, Beta2, Epsilon, zeroGradients: true))
        {
            base.ClipAndStep(maxNorm);
            return;
        }

        _step = step;
        GradientsZeroed = true;
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        _fused?.Dispose();
        _fused = null;
        foreach (var t in (_m ?? []).Concat(_v ?? []))
        {
            t?.Dispose();
        }

        base.Dispose();
    }
}

/// <summary>
/// AdamW (Loshchilov &amp; Hutter, 2019): Adam with decoupled weight decay, p -= lr·λ·p each step.
/// The usual choice for transformers and other large models.
/// </summary>
public sealed class AdamW(IEnumerable<Tensor> parameters, float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f, float weightDecay = 0.01f)
    : Adam(parameters, learningRate, beta1, beta2, epsilon, weightDecay)
{
    /// <inheritdoc />
    protected override bool DecoupledWeightDecay => true;
}
