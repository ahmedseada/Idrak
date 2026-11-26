// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices.Cpu;

// The element losses: the sigmoid focal loss (torchvision's formula) and its gradient, an element at a time.
internal sealed unsafe partial class CpuBackend
{
    // p = σ(x), ce = max(x, 0) − x·t + ln(1 + e^−|x|), p_t = p·t + (1 − p)(1 − t), α_t = α·t + (1 − α)(1 − t) (1 where α < 0):
    // loss = α_t · ce · (1 − p_t)^γ, and its derivative in x.
    private static (float Loss, float Slope) Focal(float x, float t, float alpha, float gamma)
    {
        float p = Fn<float, F1>.Sigmoid(x);
        float ce = MathF.Max(x, 0f) - x * t + Fn<float, F1>.Log(1f + Fn<float, F1>.Exp(-MathF.Abs(x)));
        float pt = p * t + (1f - p) * (1f - t);
        float at = alpha >= 0f ? alpha * t + (1f - alpha) * (1f - t) : 1f;
        float weight = gamma == 0f ? 1f : CpuMath.Pow(1f - pt, gamma);
        float slope = (p - t) * weight;
        if (gamma != 0f)
        {
            float lower = gamma == 1f ? 1f : CpuMath.Pow(1f - pt, gamma - 1f);
            slope -= ce * gamma * lower * (2f * t - 1f) * p * (1f - p);
        }

        return (at * ce * weight, at * slope);
    }

    public override void SigmoidFocalLossKernel(Storage logits, Storage targets, Storage losses, int count, float alpha, float gamma)
    {
        var x = V(logits);
        var t = V(targets);
        var l = V(losses);
        for (int i = 0; i < count; i++)
        {
            l[i] = Focal(x[i], t[i], alpha, gamma).Loss;
        }
    }

    public override void SigmoidFocalLossBackwardKernel(Storage logits, Storage targets, Storage lossGrads, Storage dLogits, int count, float alpha, float gamma)
    {
        var x = V(logits);
        var t = V(targets);
        var g = V(lossGrads);
        var d = V(dLogits);
        for (int i = 0; i < count; i++)
        {
            d[i] += g[i] * Focal(x[i], t[i], alpha, gamma).Slope;
        }
    }
}
