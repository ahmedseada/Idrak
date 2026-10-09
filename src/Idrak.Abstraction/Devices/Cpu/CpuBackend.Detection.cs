// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices.Cpu;

// The losses detection trains with: box overlap losses (IoU, GIoU, DIoU, CIoU) on corner boxes, and the sigmoid focal
// loss, each in one pass over its inputs (no intermediate tensors), forward and backward (box losses in double precision,
// the focal loss in single precision with one exponential and one logarithm an element). The formulas
// are torchvision's (ops/giou_loss.py, diou_loss.py, ciou_loss.py, focal_loss.py); the gradients are their derivatives,
// with a tie of a minimum or a maximum sharing the gradient in halves as PyTorch's do.
internal sealed partial class CpuBackend
{
    public override void BoxIouLossKernel(Storage predicted, Storage target, Storage losses, int count, BoxOverlap overlap, float eps)
    {
        float[] pv = D(predicted), tv = D(target), lv = D(losses);
        For(count, count * 40L, (start, end) =>
        {
            for (int i = start; i < end; i++)
            {
                lv[i] = (float)BoxLoss(pv.AsSpan(4 * i, 4), tv.AsSpan(4 * i, 4), overlap, eps, default);
            }
        });
    }

    public override void BoxIouLossBackwardKernel(Storage predicted, Storage target, Storage lossGrads, Storage dPredicted, int count, BoxOverlap overlap, float eps)
    {
        float[] pv = D(predicted), tv = D(target), gv = D(lossGrads), dv = D(dPredicted);
        For(count, count * 80L, (start, end) =>
        {
            Span<double> grad = stackalloc double[4];
            for (int i = start; i < end; i++)
            {
                if (gv[i] == 0f)
                {
                    continue;
                }

                grad.Clear();
                BoxLoss(pv.AsSpan(4 * i, 4), tv.AsSpan(4 * i, 4), overlap, eps, grad);
                for (int k = 0; k < 4; k++)
                {
                    dv[4 * i + k] += (float)(gv[i] * grad[k]);
                }
            }
        });
    }

    // One pair's loss; with a non-empty `grad`, its derivative with respect to the predicted corners is added there.
    private static double BoxLoss(ReadOnlySpan<float> p, ReadOnlySpan<float> g, BoxOverlap overlap, double eps, Span<double> grad)
    {
        double x1 = p[0], y1 = p[1], x2 = p[2], y2 = p[3], gx1 = g[0], gy1 = g[1], gx2 = g[2], gy2 = g[3];
        double ix1 = Math.Max(x1, gx1), iy1 = Math.Max(y1, gy1), ix2 = Math.Min(x2, gx2), iy2 = Math.Min(y2, gy2);
        double iw = ix2 - ix1, ih = iy2 - iy1;
        bool overlaps = iw > 0 && ih > 0;
        double inter = overlaps ? iw * ih : 0;
        double wp = x2 - x1, hp = y2 - y1, wg = gx2 - gx1, hg = gy2 - gy1;
        double union = wp * hp + wg * hg - inter;
        double iou = inter / (union + eps);
        double loss = 1 - iou;

        // The enclosing box.
        double cx1 = Math.Min(x1, gx1), cy1 = Math.Min(y1, gy1), cx2 = Math.Max(x2, gx2), cy2 = Math.Max(y2, gy2);
        double cw = cx2 - cx1, ch = cy2 - cy1;
        double dLdInter = -1 / (union + eps), dLdUnion = inter / ((union + eps) * (union + eps));
        double dLdCw = 0, dLdCh = 0;     // through the enclosing box's width and height
        double dLdPx = 0, dLdPy = 0;     // through the predicted centre
        double dLdW = 0, dLdH = 0;       // through the predicted width and height (CIoU's aspect term)
        switch (overlap)
        {
            case BoxOverlap.IoU:
                break;
            case BoxOverlap.GIoU:
            {
                double area = cw * ch;
                loss += (area - union) / (area + eps);
                dLdUnion += -1 / (area + eps);
                double dLdArea = (union + eps) / ((area + eps) * (area + eps));
                dLdCw += dLdArea * ch;
                dLdCh += dLdArea * cw;
                break;
            }

            case BoxOverlap.DIoU or BoxOverlap.CIoU:
            {
                double diagonal = cw * cw + ch * ch + eps;
                double dx = (x1 + x2) / 2 - (gx1 + gx2) / 2, dy = (y1 + y2) / 2 - (gy1 + gy2) / 2;
                double rho = dx * dx + dy * dy;
                loss += rho / diagonal;
                dLdPx += 2 * dx / diagonal;
                dLdPy += 2 * dy / diagonal;
                double dLdDiagonal = -rho / (diagonal * diagonal);
                dLdCw += dLdDiagonal * 2 * cw;
                dLdCh += dLdDiagonal * 2 * ch;
                if (overlap == BoxOverlap.CIoU)
                {
                    const double K = 4 / (Math.PI * Math.PI);
                    double difference = Math.Atan(wg / hg) - Math.Atan(wp / hp);
                    double v = K * difference * difference;
                    double alpha = v / (1 - iou + v + eps);    // a constant (no gradient), as torchvision's
                    loss += alpha * v;
                    double r2 = wp * wp + hp * hp;
                    dLdW += alpha * -2 * K * difference * hp / r2;
                    dLdH += alpha * 2 * K * difference * wp / r2;
                }

                break;
            }

            default:
                throw new ArgumentOutOfRangeException(nameof(overlap), overlap, "Not a box overlap.");
        }

        if (grad.IsEmpty)
        {
            return loss;
        }

        // The union holds the predicted area and minus the intersection.
        dLdInter -= dLdUnion;
        double dLdArea1 = dLdUnion;
        dLdW += dLdArea1 * hp;
        dLdH += dLdArea1 * wp;
        double dIx1 = 0, dIx2 = 0, dIy1 = 0, dIy2 = 0;
        if (overlaps)
        {
            (dIx1, dIx2, dIy1, dIy2) = (-ih * dLdInter, ih * dLdInter, -iw * dLdInter, iw * dLdInter);
        }

        // x1: the width, the centre, the intersection's left (a maximum), the enclosing box's left (a minimum).
        grad[0] += -dLdW + dLdPx / 2 + dIx1 * Above(x1, gx1) - dLdCw * Below(x1, gx1);
        grad[1] += -dLdH + dLdPy / 2 + dIy1 * Above(y1, gy1) - dLdCh * Below(y1, gy1);
        grad[2] += dLdW + dLdPx / 2 + dIx2 * Below(x2, gx2) + dLdCw * Above(x2, gx2);
        grad[3] += dLdH + dLdPy / 2 + dIy2 * Below(y2, gy2) + dLdCh * Above(y2, gy2);
        return loss;

        // The share of max(a, b)'s gradient a gets (min's: Below).
        static double Above(double a, double b) => a > b ? 1 : a == b ? 0.5 : 0;
        static double Below(double a, double b) => a < b ? 1 : a == b ? 0.5 : 0;
    }

    public override void SigmoidFocalLossKernel(Storage logits, Storage targets, Storage losses, int count, float alpha, float gamma)
    {
        float[] xv = D(logits), tv = D(targets), lv = D(losses);
        For(count, count * 20L, (start, end) =>
        {
            for (int i = start; i < end; i++)
            {
                lv[i] = Focal(xv[i], tv[i], alpha, gamma, out _);
            }
        });
    }

    public override void SigmoidFocalLossBackwardKernel(Storage logits, Storage targets, Storage lossGrads, Storage dLogits, int count, float alpha, float gamma)
    {
        float[] xv = D(logits), tv = D(targets), gv = D(lossGrads), dv = D(dLogits);
        For(count, count * 30L, (start, end) =>
        {
            for (int i = start; i < end; i++)
            {
                Focal(xv[i], tv[i], alpha, gamma, out float derivative);
                dv[i] += gv[i] * derivative;
            }
        });
    }

    // One element's focal loss and its derivative with respect to the logit, in single precision as PyTorch's: one
    // exponential and one logarithm (e = exp(-|x|) gives both the sigmoid and log(1 + e^-|x|)); γ = 1 and 2 without a power.
    private static float Focal(float x, float t, float alpha, float gamma, out float derivative)
    {
        float e = MathF.Exp(-MathF.Abs(x));
        float p = x >= 0 ? 1 / (1 + e) : e / (1 + e);
        float ce = MathF.Max(x, 0) - x * t + MathF.Log(1 + e);
        float pt = p * t + (1 - p) * (1 - t), q = 1 - pt;
        float lower = gamma switch { 0 => 0, 1 => 1, 2 => q, _ => q == 0 && gamma < 1 ? 0 : MathF.Pow(q, gamma - 1) };   // q^(γ - 1)
        float modulating = gamma == 0 ? 1 : gamma is 1 or 2 ? lower * q : MathF.Pow(q, gamma);
        float weight = alpha >= 0 ? alpha * t + (1 - alpha) * (1 - t) : 1;
        float dPt = (2 * t - 1) * p * (1 - p);
        derivative = weight * ((p - t) * modulating - ce * gamma * lower * dPt);
        return weight * ce * modulating;
    }
}
