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

    // ------------------------------------------------------------------ box overlap losses

    public override void BoxIouLossKernel(Storage predicted, Storage target, Storage losses, int count, BoxOverlap overlap, float eps)
    {
        fixed (float* p = V(predicted), t = V(target), l = V(losses))
        {
            for (int i = 0; i < count; i++)
            {
                l[i] = BoxLoss(p + 4L * i, t + 4L * i, overlap, eps, null);
            }
        }
    }

    public override void BoxIouLossBackwardKernel(Storage predicted, Storage target, Storage lossGrads, Storage dPredicted, int count, BoxOverlap overlap, float eps)
    {
        fixed (float* p = V(predicted), t = V(target), g = V(lossGrads), d = V(dPredicted))
        {
            float* grad = stackalloc float[4];
            for (int i = 0; i < count; i++)
            {
                if (g[i] == 0f)
                {
                    continue;
                }

                BoxLoss(p + 4L * i, t + 4L * i, overlap, eps, grad);
                for (int k = 0; k < 4; k++)
                {
                    d[4L * i + k] += g[i] * grad[k];
                }
            }
        }
    }

    // One pair's loss (torchvision's formulas, float32 as its inputs); with `grad`, its derivative in the predicted corners
    // written there (a tie of a minimum or maximum sharing the gradient in halves, as PyTorch's).
    private static float BoxLoss(float* p, float* g, BoxOverlap overlap, float eps, float* grad)
    {
        float x1 = p[0], y1 = p[1], x2 = p[2], y2 = p[3], gx1 = g[0], gy1 = g[1], gx2 = g[2], gy2 = g[3];
        float ix1 = MathF.Max(x1, gx1), iy1 = MathF.Max(y1, gy1), ix2 = MathF.Min(x2, gx2), iy2 = MathF.Min(y2, gy2);
        float iw = ix2 - ix1, ih = iy2 - iy1;
        bool overlaps = iw > 0f && ih > 0f;
        float inter = overlaps ? iw * ih : 0f;
        float wp = x2 - x1, hp = y2 - y1, wg = gx2 - gx1, hg = gy2 - gy1;
        float union = wp * hp + wg * hg - inter;
        float iou = inter / (union + eps);
        float loss = 1f - iou;

        // The enclosing box.
        float cx1 = MathF.Min(x1, gx1), cy1 = MathF.Min(y1, gy1), cx2 = MathF.Max(x2, gx2), cy2 = MathF.Max(y2, gy2);
        float cw = cx2 - cx1, ch = cy2 - cy1;
        float dLdInter = -1f / (union + eps), dLdUnion = inter / ((union + eps) * (union + eps));
        float dLdCw = 0f, dLdCh = 0f;     // through the enclosing box's width and height
        float dLdPx = 0f, dLdPy = 0f;     // through the predicted centre
        float dLdW = 0f, dLdH = 0f;       // through the predicted width and height (CIoU's aspect term)
        switch (overlap)
        {
            case BoxOverlap.IoU:
                break;
            case BoxOverlap.GIoU:
            {
                float area = cw * ch;
                loss += (area - union) / (area + eps);
                dLdUnion += -1f / (area + eps);
                float dLdArea = (union + eps) / ((area + eps) * (area + eps));
                dLdCw += dLdArea * ch;
                dLdCh += dLdArea * cw;
                break;
            }

            case BoxOverlap.DIoU or BoxOverlap.CIoU:
            {
                float diagonal = cw * cw + ch * ch + eps;
                float dx = (x1 + x2) / 2f - (gx1 + gx2) / 2f, dy = (y1 + y2) / 2f - (gy1 + gy2) / 2f;
                float rho = dx * dx + dy * dy;
                loss += rho / diagonal;
                dLdPx += 2f * dx / diagonal;
                dLdPy += 2f * dy / diagonal;
                float dLdDiagonal = -rho / (diagonal * diagonal);
                dLdCw += dLdDiagonal * 2f * cw;
                dLdCh += dLdDiagonal * 2f * ch;
                if (overlap == BoxOverlap.CIoU)
                {
                    const float K = (float)(4 / (Math.PI * Math.PI));
                    float difference = Atan(wg / hg) - Atan(wp / hp);
                    float v = K * difference * difference;
                    float alpha = v / (1f - iou + v + eps);    // a constant (no gradient), as torchvision's
                    loss += alpha * v;
                    float r2 = wp * wp + hp * hp;
                    dLdW += alpha * -2f * K * difference * hp / r2;
                    dLdH += alpha * 2f * K * difference * wp / r2;
                }

                break;
            }

            default:
                throw new ArgumentOutOfRangeException(nameof(overlap), overlap, "Not a box overlap.");
        }

        if (grad == null)
        {
            return loss;
        }

        // The union holds the predicted area and minus the intersection.
        dLdInter -= dLdUnion;
        dLdW += dLdUnion * hp;
        dLdH += dLdUnion * wp;
        float dIx1 = 0f, dIx2 = 0f, dIy1 = 0f, dIy2 = 0f;
        if (overlaps)
        {
            (dIx1, dIx2, dIy1, dIy2) = (-ih * dLdInter, ih * dLdInter, -iw * dLdInter, iw * dLdInter);
        }

        // x1: the width, the centre, the intersection's left (a maximum), the enclosing box's left (a minimum).
        grad[0] = -dLdW + dLdPx / 2f + dIx1 * Above(x1, gx1) - dLdCw * Below(x1, gx1);
        grad[1] = -dLdH + dLdPy / 2f + dIy1 * Above(y1, gy1) - dLdCh * Below(y1, gy1);
        grad[2] = dLdW + dLdPx / 2f + dIx2 * Below(x2, gx2) + dLdCw * Above(x2, gx2);
        grad[3] = dLdH + dLdPy / 2f + dIy2 * Below(y2, gy2) + dLdCh * Above(y2, gy2);
        return loss;

        // The share of max(a, b)'s gradient a gets (min's: Below).
        static float Above(float a, float b) => a > b ? 1f : a == b ? 0.5f : 0f;
        static float Below(float a, float b) => a < b ? 1f : a == b ? 0.5f : 0f;

        // atan of a float32 value, rounded once (the library's own atan).
        static float Atan(float x) => (float)CpuMath.AtanDouble(x);
    }
}
