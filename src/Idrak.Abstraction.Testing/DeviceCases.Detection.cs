// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;

namespace Idrak.Abstraction.Testing;

// The cases of detection losses: box overlap losses and the sigmoid focal loss against torchvision's formulas written out
// here as plain loops, and their gradients against central differences of those loops.
public static partial class DeviceCases
{
    private static void DetectionLosses(DeviceCaseContext c)
    {
        var b = c.Backend;
        int count = c.Size(1, 40);
        var (predicted, target) = (RandomBoxes(c, count), RandomBoxes(c, count));
        if (count > 2)
        {
            Array.Copy(target, 0, predicted, 0, 4);                                  // the same box: every loss is (nearly) 0
            (predicted[4], predicted[5], predicted[6], predicted[7]) = (target[4] + 50, target[5] + 50, target[6] + 50, target[7] + 50);   // apart
        }

        Storage pv = c.Storage(predicted), tv = c.Storage(target);
        var scales = c.Values(count, 2f);
        foreach (var overlap in Enum.GetValues<BoxOverlap>())
        {
            var losses = c.Zeros(count);
            b.BoxIouLoss(pv, tv, losses, count, overlap, 1e-7f);
            var expected = new float[count];
            var expectedGrad = new float[4 * count];
            for (int i = 0; i < count; i++)
            {
                expected[i] = (float)BoxLossReference(predicted.AsSpan(4 * i, 4), target.AsSpan(4 * i, 4), overlap, 1e-7);
                // CIoU's α is a constant to the gradient (torchvision computes it without one): held at its value here.
                double alpha = CiouAlpha(predicted.AsSpan(4 * i, 4), target.AsSpan(4 * i, 4), 1e-7);
                for (int k = 0; k < 4 && !(count > 2 && i == 0); k++)                // no derivative where the boxes tie
                {
                    var plus = predicted.AsSpan(4 * i, 4).ToArray();
                    var minus = (float[])plus.Clone();
                    const float H = 1e-3f;
                    plus[k] += H;
                    minus[k] -= H;
                    expectedGrad[4 * i + k] = (float)(scales[i] * (BoxLossReference(plus, target.AsSpan(4 * i, 4), overlap, 1e-7, alpha)
                        - BoxLossReference(minus, target.AsSpan(4 * i, 4), overlap, 1e-7, alpha)) / (2 * H));
                }
            }

            c.ExpectClose(expected, Read(losses), 1e-5f, $"{overlap} loss");
            var grad = c.Zeros(4 * count);
            b.BoxIouLossBackward(pv, tv, c.Storage(scales), grad, count, overlap, 1e-7f);
            var actualGrad = Read(grad);
            if (count > 2)
            {
                Array.Clear(actualGrad, 0, 4);
            }

            c.ExpectClose(expectedGrad, actualGrad, 2e-3f, $"{overlap} gradient");
        }

        // Focal loss: logits from very negative to very positive, hard and soft targets, with and without α.
        int n = c.Size(1, 300);
        var logits = c.Values(n, 6f);
        var targets = c.Values(n).Select((v, i) => i % 3 == 2 ? (v + 1) / 2 : v > 0 ? 1f : 0f).ToArray();
        Storage lv = c.Storage(logits), yv = c.Storage(targets);
        var focalScales = c.Values(n, 2f);
        foreach (var (alpha, gamma) in new[] { (0.25f, 2f), (-1f, 2f), (0.5f, 0f), (0.75f, 1.5f) })
        {
            var losses = c.Zeros(n);
            b.SigmoidFocalLoss(lv, yv, losses, n, alpha, gamma);
            var expected = new float[n];
            var expectedGrad = new float[n];
            for (int i = 0; i < n; i++)
            {
                expected[i] = (float)FocalReference(logits[i], targets[i], alpha, gamma);
                const double H = 1e-4;
                expectedGrad[i] = (float)(focalScales[i] * (FocalReference(logits[i] + H, targets[i], alpha, gamma) - FocalReference(logits[i] - H, targets[i], alpha, gamma)) / (2 * H));
            }

            c.ExpectClose(expected, Read(losses), 1e-5f, $"focal loss (α {alpha}, γ {gamma})");
            var grad = c.Zeros(n);
            b.SigmoidFocalLossBackward(lv, yv, c.Storage(focalScales), grad, n, alpha, gamma);
            c.ExpectClose(expectedGrad, Read(grad), 1e-3f, $"focal gradient (α {alpha}, γ {gamma})");
        }
    }

    private static float[] RandomBoxes(DeviceCaseContext c, int count)
    {
        var boxes = new float[4 * count];
        for (int i = 0; i < count; i++)
        {
            float x = c.Random.NextSingle() * 20, y = c.Random.NextSingle() * 20;
            (boxes[4 * i], boxes[4 * i + 1], boxes[4 * i + 2], boxes[4 * i + 3]) = (x, y, x + 0.5f + c.Random.NextSingle() * 15, y + 0.5f + c.Random.NextSingle() * 15);
        }

        return boxes;
    }

    // CIoU's α = v / (1 - IoU + v + eps) of a pair, from its CIoU and DIoU losses.
    private static double CiouAlpha(ReadOnlySpan<float> p, ReadOnlySpan<float> g, double eps)
    {
        double diou = BoxLossReference(p, g, BoxOverlap.DIoU, eps), ciou = BoxLossReference(p, g, BoxOverlap.CIoU, eps);
        double v = 4 / (Math.PI * Math.PI) * Math.Pow(Math.Atan((g[2] - (double)g[0]) / (g[3] - (double)g[1])) - Math.Atan((p[2] - (double)p[0]) / (p[3] - (double)p[1])), 2);
        return v == 0 ? 0 : (ciou - diou) / v;
    }

    // torchvision's _loss_inter_union, generalized_box_iou_loss, _diou_iou_loss and complete_box_iou_loss, line by line;
    // CIoU's α held at `fixedAlpha` when given.
    private static double BoxLossReference(ReadOnlySpan<float> p, ReadOnlySpan<float> g, BoxOverlap overlap, double eps, double? fixedAlpha = null)
    {
        double x1 = p[0], y1 = p[1], x2 = p[2], y2 = p[3], x1g = g[0], y1g = g[1], x2g = g[2], y2g = g[3];
        double xkis1 = Math.Max(x1, x1g), ykis1 = Math.Max(y1, y1g), xkis2 = Math.Min(x2, x2g), ykis2 = Math.Min(y2, y2g);
        double intsctk = ykis2 > ykis1 && xkis2 > xkis1 ? (xkis2 - xkis1) * (ykis2 - ykis1) : 0;
        double unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk;
        double iou = intsctk / (unionk + eps);
        double xc1 = Math.Min(x1, x1g), yc1 = Math.Min(y1, y1g), xc2 = Math.Max(x2, x2g), yc2 = Math.Max(y2, y2g);
        switch (overlap)
        {
            case BoxOverlap.IoU:
                return 1 - iou;
            case BoxOverlap.GIoU:
            {
                double areaC = (xc2 - xc1) * (yc2 - yc1);
                return 1 - (iou - (areaC - unionk) / (areaC + eps));
            }

            default:
            {
                double diagonal = (xc2 - xc1) * (xc2 - xc1) + (yc2 - yc1) * (yc2 - yc1) + eps;
                double xp = (x2 + x1) / 2, yp = (y2 + y1) / 2, xg = (x1g + x2g) / 2, yg = (y1g + y2g) / 2;
                double diou = 1 - iou + ((xp - xg) * (xp - xg) + (yp - yg) * (yp - yg)) / diagonal;
                if (overlap == BoxOverlap.DIoU)
                {
                    return diou;
                }

                double v = 4 / (Math.PI * Math.PI) * Math.Pow(Math.Atan((x2g - x1g) / (y2g - y1g)) - Math.Atan((x2 - x1) / (y2 - y1)), 2);
                double alpha = fixedAlpha ?? v / (1 - iou + v + eps);
                return diou + alpha * v;
            }
        }
    }

    // torchvision's sigmoid_focal_loss with reduction "none".
    private static double FocalReference(double x, double t, double alpha, double gamma)
    {
        double p = 1 / (1 + Math.Exp(-x));
        double ce = -(t * Math.Log(p) + (1 - t) * Math.Log(1 - p));
        if (!double.IsFinite(ce))
        {
            ce = Math.Max(x, 0) - x * t + Math.Log(1 + Math.Exp(-Math.Abs(x)));
        }

        double pt = p * t + (1 - p) * (1 - t);
        double loss = ce * Math.Pow(1 - pt, gamma);
        return alpha >= 0 ? (alpha * t + (1 - alpha) * (1 - t)) * loss : loss;
    }
}
