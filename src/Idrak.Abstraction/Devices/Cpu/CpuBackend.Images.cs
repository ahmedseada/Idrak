// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices.Cpu;

// The image operations: unfolding windows for the convolution's products (Im2Col) and folding them back (Col2Im), max,
// average and adaptive pooling, and interpolation, with their gradients. Each plane (an image's channel) is one thread's,
// and a gradient adds into each input position in window order, so the results do not depend on the threads.
internal sealed unsafe partial class CpuBackend
{
    // Runs body(first, last) over [0, count) on the threads where the work passes the cut-over.
    private static void Planes(int count, long workEach, Action<int, int> body)
    {
        if (count <= 0)
        {
            return;
        }

        if (ComputeResources.MaxCpuThreads <= 1 || count == 1 || count * workEach < ParallelThreshold)
        {
            body(0, count);
            return;
        }

        HostParallel.For(count, 1, body);
    }

    // Each kernel reads a typed input through its type where it reads it (TypedRanges): a plane when its thread takes it, the
    // input rows a window row reads, a window's column values; never the whole input.
    public override void Im2ColKernel(Storage x, Storage cols, in ConvGeometry g)
    {
        var px = D(x);
        var pc = V(cols);
        var geometry = g;
        var tx = new TypedRanges(x, 0, "Im2Col");
        int oh = g.OH, ow = g.OW, patch = g.PatchSize;
        Planes(g.N * oh, (long)ow * patch, (first, last) =>
        {
            var s = geometry;
            for (int row = first; row < last; row++)
            {
                int n = row / oh, y = row % oh;
                if (tx.Typed)
                {
                    // Each channel's input rows the window row reads, read through x's type once (the same values copied).
                    int top = Math.Max(0, y * s.SH - s.PH), bottom = Math.Min(s.H, y * s.SH - s.PH + (s.KH - 1) * s.DH + 1);
                    for (int c = 0; c < s.C; c++)
                    {
                        long plane = (long)(n * s.C + c) * s.H;
                        long shift = 0;
                        var values = px;
                        if (bottom > top)
                        {
                            values = tx.Values(px, (plane + top) * s.W, (long)(bottom - top) * s.W, out shift);
                        }

                        for (int xo = 0; xo < ow; xo++)
                        {
                            long at = ((long)row * ow + xo) * patch + (long)c * s.KH * s.KW;
                            for (int kh = 0; kh < s.KH; kh++)
                            {
                                int ih = y * s.SH - s.PH + kh * s.DH;
                                for (int kw = 0; kw < s.KW; kw++, at++)
                                {
                                    int iw = xo * s.SW - s.PW + kw * s.DW;
                                    pc[at] = ih >= 0 && ih < s.H && iw >= 0 && iw < s.W ? values[shift + (plane + ih) * s.W + iw] : 0f;
                                }
                            }
                        }
                    }

                    continue;
                }

                for (int xo = 0; xo < ow; xo++)
                {
                    long at = ((long)row * ow + xo) * patch;
                    for (int c = 0; c < s.C; c++)
                    {
                        for (int kh = 0; kh < s.KH; kh++)
                        {
                            int ih = y * s.SH - s.PH + kh * s.DH;
                            for (int kw = 0; kw < s.KW; kw++, at++)
                            {
                                int iw = xo * s.SW - s.PW + kw * s.DW;
                                pc[at] = ih >= 0 && ih < s.H && iw >= 0 && iw < s.W ? px[((long)(n * s.C + c) * s.H + ih) * s.W + iw] : 0f;
                            }
                        }
                    }
                }
            }
        });
    }

    public override void Col2ImKernel(Storage dcols, Storage dx, in ConvGeometry g)
    {
        var pc = D(dcols);
        var px = V(dx);
        var s = g;
        var tc = new TypedRanges(dcols, 0, "Col2Im");
        int oh = g.OH, ow = g.OW, patch = g.PatchSize;
        Planes(g.N * g.C, (long)oh * ow * g.KH * g.KW, (first, last) =>
        {
            for (int plane = first; plane < last; plane++)
            {
                int n = plane / s.C, c = plane % s.C;
                long image = (long)plane * s.H * s.W;
                for (int y = 0; y < oh; y++)
                {
                    for (int xo = 0; xo < ow; xo++)
                    {
                        long at = ((long)(n * oh + y) * ow + xo) * patch + (long)c * s.KH * s.KW;
                        var values = tc.Values(pc, at, (long)s.KH * s.KW, out long shift);   // the window's values of this plane
                        for (int kh = 0; kh < s.KH; kh++)
                        {
                            int ih = y * s.SH - s.PH + kh * s.DH;
                            for (int kw = 0; kw < s.KW; kw++, at++)
                            {
                                int iw = xo * s.SW - s.PW + kw * s.DW;
                                if (ih >= 0 && ih < s.H && iw >= 0 && iw < s.W)
                                {
                                    px[image + (long)ih * s.W + iw] += values[shift + at];
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    public override void MaxPoolKernel(Storage x, Storage y, Storage argmax, in ConvGeometry g)
    {
        var wx = D(x);
        var py = V(y);
        var pa = D(argmax);
        var s = g;
        var tx = new TypedRanges(x, 0, "MaxPool");
        int oh = g.OH, ow = g.OW;
        Planes(g.N * g.C, (long)oh * ow * g.KH * g.KW, (first, last) =>
        {
            for (int plane = first; plane < last; plane++)
            {
                long image = (long)plane * s.H * s.W;
                var px = tx.Values(wx, image, (long)s.H * s.W, out long shift);
                for (int yo = 0; yo < oh; yo++)
                {
                    for (int xo = 0; xo < ow; xo++)
                    {
                        float best = float.NegativeInfinity;
                        long index = -1;
                        for (int kh = 0; kh < s.KH; kh++)
                        {
                            int ih = yo * s.SH - s.PH + kh * s.DH;
                            if (ih < 0 || ih >= s.H)
                            {
                                continue;
                            }

                            for (int kw = 0; kw < s.KW; kw++)
                            {
                                int iw = xo * s.SW - s.PW + kw * s.DW;
                                if (iw < 0 || iw >= s.W)
                                {
                                    continue;
                                }

                                long at = image + (long)ih * s.W + iw;
                                float v = px[shift + at];
                                if (index < 0 || v > best || float.IsNaN(v) && !float.IsNaN(best))
                                {
                                    (best, index) = (v, at);
                                }
                            }
                        }

                        long o = ((long)plane * oh + yo) * ow + xo;
                        py[o] = best;
                        pa[o] = BitConverter.Int32BitsToSingle((int)index);
                    }
                }
            }
        });
    }

    public override void MaxPoolBackwardKernel(Storage dy, Storage argmax, Storage dx, int count)
    {
        var wd = D(dy);
        var pa = D(argmax);
        var px = V(dx);
        var td = new TypedRanges(dy, 0, "MaxPoolBackward");
        for (int i0 = 0; i0 < count; i0 += ChunkValues)
        {
            int i1 = Math.Min(count, i0 + ChunkValues);
            var pd = td.Values(wd, i0, i1 - i0, out long shift);
            for (int i = i0; i < i1; i++)
            {
                int at = BitConverter.SingleToInt32Bits(pa[i]);
                if (at >= 0)
                {
                    px[at] += pd[shift + i];
                }
            }
        }
    }

    public override void MaxPoolBackwardKernel(Storage dy, Storage argmax, Storage dx, in ConvGeometry g) =>
        ArgmaxBackward(dy, argmax, dx, g.N * g.C, g.OH * g.OW);

    public override void AdaptiveMaxPoolBackwardKernel(Storage dy, Storage argmax, Storage dx, int planes, int height, int width, int outHeight, int outWidth) =>
        ArgmaxBackward(dy, argmax, dx, planes, outHeight * outWidth);

    // dx[argmax[i]] += dy[i], each plane's outputs in order (a plane's maxima lie in its own plane).
    private static void ArgmaxBackward(Storage dy, Storage argmax, Storage dx, int planes, int outputs)
    {
        var wd = D(dy);
        var pa = D(argmax);
        var px = V(dx);
        var td = new TypedRanges(dy, 0, "MaxPoolBackward");
        Planes(planes, outputs, (first, last) =>
        {
            for (int plane = first; plane < last; plane++)
            {
                long from = (long)plane * outputs;
                var pd = td.Values(wd, from, outputs, out long shift);
                for (long i = from; i < from + outputs; i++)
                {
                    int at = BitConverter.SingleToInt32Bits(pa[i]);
                    if (at >= 0)
                    {
                        px[at] += pd[shift + i];
                    }
                }
            }
        });
    }

    // An average window's rows or columns: [start, end) inside the input, and the divisor's extent.
    private static (int Start, int End, int Counted) Window(int o, int stride, int pad, int k, int size, int padEnd, bool countIncludePad)
    {
        int start = o * stride - pad, end = Math.Min(start + k, size + padEnd);
        int counted = end - start;
        int s = Math.Max(start, 0), e = Math.Min(end, size);
        return (s, e, countIncludePad ? counted : e - s);
    }

    public override void AvgPoolKernel(Storage x, Storage y, in ConvGeometry g, bool countIncludePad, int padBottom, int padRight)
    {
        var wx = D(x);
        var py = V(y);
        var s = g;
        var tx = new TypedRanges(x, 0, "AvgPool");
        int oh = g.OH, ow = g.OW;
        Planes(g.N * g.C, (long)oh * ow * g.KH * g.KW, (first, last) =>
        {
            for (int plane = first; plane < last; plane++)
            {
                var px = tx.Values(wx, (long)plane * s.H * s.W, (long)s.H * s.W, out long shift);
                long image = shift + (long)plane * s.H * s.W;
                for (int yo = 0; yo < oh; yo++)
                {
                    var (h0, h1, hc) = Window(yo, s.SH, s.PH, s.KH, s.H, padBottom, countIncludePad);
                    for (int xo = 0; xo < ow; xo++)
                    {
                        var (w0, w1, wc) = Window(xo, s.SW, s.PW, s.KW, s.W, padRight, countIncludePad);
                        float sum = 0f;
                        for (int ih = h0; ih < h1; ih++)
                        {
                            for (int iw = w0; iw < w1; iw++)
                            {
                                sum += px[image + (long)ih * s.W + iw];
                            }
                        }

                        py[((long)plane * oh + yo) * ow + xo] = sum / Math.Max(1, hc * wc);
                    }
                }
            }
        });
    }

    public override void AvgPoolBackwardKernel(Storage dy, Storage dx, in ConvGeometry g, bool countIncludePad, int padBottom, int padRight)
    {
        var wd = D(dy);
        var px = V(dx);
        var s = g;
        var td = new TypedRanges(dy, 0, "AvgPoolBackward");
        int oh = g.OH, ow = g.OW;
        Planes(g.N * g.C, (long)oh * ow * g.KH * g.KW, (first, last) =>
        {
            for (int plane = first; plane < last; plane++)
            {
                long image = (long)plane * s.H * s.W;
                var pd = td.Values(wd, (long)plane * oh * ow, (long)oh * ow, out long shift);
                for (int yo = 0; yo < oh; yo++)
                {
                    var (h0, h1, hc) = Window(yo, s.SH, s.PH, s.KH, s.H, padBottom, countIncludePad);
                    for (int xo = 0; xo < ow; xo++)
                    {
                        var (w0, w1, wc) = Window(xo, s.SW, s.PW, s.KW, s.W, padRight, countIncludePad);
                        float share = pd[shift + ((long)plane * oh + yo) * ow + xo] / Math.Max(1, hc * wc);
                        for (int ih = h0; ih < h1; ih++)
                        {
                            for (int iw = w0; iw < w1; iw++)
                            {
                                px[image + (long)ih * s.W + iw] += share;
                            }
                        }
                    }
                }
            }
        });
    }

    // Adaptive windows: output o reads [floor(o · size / out), ceil((o + 1) · size / out)).
    private static (int Start, int End) Adaptive(int o, int size, int outSize) =>
        ((int)((long)o * size / outSize), (int)(((long)(o + 1) * size + outSize - 1) / outSize));

    public override void AdaptiveAvgPoolKernel(Storage x, Storage y, int planes, int height, int width, int outHeight, int outWidth)
    {
        var wx = D(x);
        var py = V(y);
        var tx = new TypedRanges(x, 0, "AdaptiveAvgPool");
        Planes(planes, (long)height * width, (first, last) =>
        {
            for (int plane = first; plane < last; plane++)
            {
                var px = tx.Values(wx, (long)plane * height * width, (long)height * width, out long shift);
                long image = shift + (long)plane * height * width;
                for (int yo = 0; yo < outHeight; yo++)
                {
                    var (h0, h1) = Adaptive(yo, height, outHeight);
                    for (int xo = 0; xo < outWidth; xo++)
                    {
                        var (w0, w1) = Adaptive(xo, width, outWidth);
                        float sum = 0f;
                        for (int ih = h0; ih < h1; ih++)
                        {
                            for (int iw = w0; iw < w1; iw++)
                            {
                                sum += px[image + (long)ih * width + iw];
                            }
                        }

                        py[((long)plane * outHeight + yo) * outWidth + xo] = sum / Math.Max(1, (h1 - h0) * (w1 - w0));
                    }
                }
            }
        });
    }

    public override void AdaptiveAvgPoolBackwardKernel(Storage dy, Storage dx, int planes, int height, int width, int outHeight, int outWidth)
    {
        var wd = D(dy);
        var px = V(dx);
        var td = new TypedRanges(dy, 0, "AdaptiveAvgPoolBackward");
        Planes(planes, (long)height * width, (first, last) =>
        {
            for (int plane = first; plane < last; plane++)
            {
                long image = (long)plane * height * width;
                var pd = td.Values(wd, (long)plane * outHeight * outWidth, (long)outHeight * outWidth, out long shift);
                for (int yo = 0; yo < outHeight; yo++)
                {
                    var (h0, h1) = Adaptive(yo, height, outHeight);
                    for (int xo = 0; xo < outWidth; xo++)
                    {
                        var (w0, w1) = Adaptive(xo, width, outWidth);
                        float share = pd[shift + ((long)plane * outHeight + yo) * outWidth + xo] / Math.Max(1, (h1 - h0) * (w1 - w0));
                        for (int ih = h0; ih < h1; ih++)
                        {
                            for (int iw = w0; iw < w1; iw++)
                            {
                                px[image + (long)ih * width + iw] += share;
                            }
                        }
                    }
                }
            }
        });
    }

    public override void AdaptiveMaxPoolKernel(Storage x, Storage y, Storage argmax, int planes, int height, int width, int outHeight, int outWidth)
    {
        var wx = D(x);
        var py = V(y);
        var pa = D(argmax);
        var tx = new TypedRanges(x, 0, "AdaptiveMaxPool");
        Planes(planes, (long)height * width, (first, last) =>
        {
            for (int plane = first; plane < last; plane++)
            {
                long image = (long)plane * height * width;
                var px = tx.Values(wx, image, (long)height * width, out long shift);
                for (int yo = 0; yo < outHeight; yo++)
                {
                    var (h0, h1) = Adaptive(yo, height, outHeight);
                    for (int xo = 0; xo < outWidth; xo++)
                    {
                        var (w0, w1) = Adaptive(xo, width, outWidth);
                        float best = float.NegativeInfinity;
                        long index = -1;
                        for (int ih = h0; ih < h1; ih++)
                        {
                            for (int iw = w0; iw < w1; iw++)
                            {
                                long at = image + (long)ih * width + iw;
                                float v = px[shift + at];
                                if (index < 0 || v > best || float.IsNaN(v) && !float.IsNaN(best))
                                {
                                    (best, index) = (v, at);
                                }
                            }
                        }

                        long o = ((long)plane * outHeight + yo) * outWidth + xo;
                        py[o] = best;
                        pa[o] = BitConverter.Int32BitsToSingle((int)index);
                    }
                }
            }
        });
    }

    // An interpolated position's two input positions and the second's weight (PyTorch's bilinear source index), or the
    // nearest position (weight 0).
    private static (int First, int Second, float Weight) Source(int o, int size, float scale, InterpolationMode mode, bool alignCorners)
    {
        if (mode == InterpolationMode.Nearest)
        {
            return (Math.Min((int)MathF.Floor(o * scale), size - 1), 0, 0f);
        }

        float src = alignCorners ? scale * o : MathF.Max(scale * (o + 0.5f) - 0.5f, 0f);
        int first = Math.Min((int)src, size - 1);
        int second = first < size - 1 ? first + 1 : first;
        return (first, second, MathF.Min(src - first, 1f));
    }

    public override void Interpolate2dKernel(Storage x, Storage y, int planes, int height, int width, int outHeight, int outWidth, InterpolationMode mode,
        bool alignCorners, float scaleHeight, float scaleWidth)
    {
        var wx = D(x);
        var py = V(y);
        var tx = new TypedRanges(x, 0, "Interpolate2d");
        Planes(planes, (long)outHeight * outWidth * 4, (first, last) =>
        {
            for (int plane = first; plane < last; plane++)
            {
                var px = tx.Values(wx, (long)plane * height * width, (long)height * width, out long shift);
                long image = shift + (long)plane * height * width;
                for (int yo = 0; yo < outHeight; yo++)
                {
                    var (h0, h1, hl) = Source(yo, height, scaleHeight, mode, alignCorners);
                    for (int xo = 0; xo < outWidth; xo++)
                    {
                        var (w0, w1, wl) = Source(xo, width, scaleWidth, mode, alignCorners);
                        float value;
                        if (mode == InterpolationMode.Nearest)
                        {
                            value = px[image + (long)h0 * width + w0];
                        }
                        else
                        {
                            float top = (1f - wl) * px[image + (long)h0 * width + w0] + wl * px[image + (long)h0 * width + w1];
                            float bottom = (1f - wl) * px[image + (long)h1 * width + w0] + wl * px[image + (long)h1 * width + w1];
                            value = (1f - hl) * top + hl * bottom;
                        }

                        py[((long)plane * outHeight + yo) * outWidth + xo] = value;
                    }
                }
            }
        });
    }

    public override void Interpolate2dBackwardKernel(Storage dy, Storage dx, int planes, int height, int width, int outHeight, int outWidth, InterpolationMode mode,
        bool alignCorners, float scaleHeight, float scaleWidth)
    {
        var wd = D(dy);
        var px = V(dx);
        var td = new TypedRanges(dy, 0, "Interpolate2dBackward");
        Planes(planes, (long)outHeight * outWidth * 4, (first, last) =>
        {
            for (int plane = first; plane < last; plane++)
            {
                long image = (long)plane * height * width;
                var pd = td.Values(wd, (long)plane * outHeight * outWidth, (long)outHeight * outWidth, out long shift);
                for (int yo = 0; yo < outHeight; yo++)
                {
                    var (h0, h1, hl) = Source(yo, height, scaleHeight, mode, alignCorners);
                    for (int xo = 0; xo < outWidth; xo++)
                    {
                        var (w0, w1, wl) = Source(xo, width, scaleWidth, mode, alignCorners);
                        float g = pd[shift + ((long)plane * outHeight + yo) * outWidth + xo];
                        if (mode == InterpolationMode.Nearest)
                        {
                            px[image + (long)h0 * width + w0] += g;
                            continue;
                        }

                        px[image + (long)h0 * width + w0] += (1f - hl) * (1f - wl) * g;
                        px[image + (long)h0 * width + w1] += (1f - hl) * wl * g;
                        px[image + (long)h1 * width + w0] += hl * (1f - wl) * g;
                        px[image + (long)h1 * width + w1] += hl * wl * g;
                    }
                }
            }
        });
    }

    // ------------------------------------------------------------------ connectionist temporal classification

    public override void CtcLossKernel(Storage logProbs, Storage targets, Storage losses, ReadOnlySpan<int> inputLengths, ReadOnlySpan<int> targetLengths,
        ReadOnlySpan<int> targetOffsets, int steps, int batch, int classes, int blank, bool batchFirst, bool zeroInfinity) =>
        Ctc(logProbs, targets, losses, null, null, inputLengths, targetLengths, targetOffsets, steps, batch, classes, blank, batchFirst, zeroInfinity);

    public override void CtcLossBackwardKernel(Storage logProbs, Storage targets, Storage lossGrads, Storage dLogProbs, ReadOnlySpan<int> inputLengths,
        ReadOnlySpan<int> targetLengths, ReadOnlySpan<int> targetOffsets, int steps, int batch, int classes, int blank, bool batchFirst, bool zeroInfinity) =>
        Ctc(logProbs, targets, null, lossGrads, dLogProbs, inputLengths, targetLengths, targetOffsets, steps, batch, classes, blank, batchFirst, zeroInfinity);

    // The forward (α) and, for the gradient, backward (β) variables in log space (float32, the log-probabilities' own kind:
    // nothing widened), a sequence a thread: the loss
    // -log p, or dLogProbs += lossGrad · ∂(-log p)/∂lp = -lossGrad · e^(log Σ_s α β - lp - log p) over each class's places.
    private static void Ctc(Storage logProbs, Storage targets, Storage? losses, Storage? lossGrads, Storage? dLogProbs, ReadOnlySpan<int> inputLengths,
        ReadOnlySpan<int> targetLengths, ReadOnlySpan<int> targetOffsets, int steps, int batch, int classes, int blank, bool batchFirst, bool zeroInfinity)
    {
        var lp = D(logProbs);
        var tlp = new TypedRanges(logProbs, 0, "CtcLoss");
        var tw = D(targets);
        var loss = losses is null ? null : V(losses);
        var grads = lossGrads is null ? null : V(lossGrads);
        var dlp = dLogProbs is null ? null : V(dLogProbs);
        int[] inLengths = inputLengths.ToArray(), tLengths = targetLengths.ToArray(), tOffsets = targetOffsets.ToArray();
        bool intTargets = IntIndices(targets);
        for (int n = 0; n < batch; n++)
        {
            for (int j = 0; j < tLengths[n]; j++)
            {
                float word = tw[tOffsets[n] + j];
                int label = intTargets ? BitConverter.SingleToInt32Bits(word) : (int)word;
                if (label < 0 || label >= classes || label == blank)
                {
                    throw new ArgumentException($"CTC: label {label} of sequence {n} (place {j}) is the blank ({blank}) or outside the {classes} classes.", nameof(targets));
                }
            }
        }

        Planes(batch, (long)steps * 64, (first, last) =>
        {
            for (int n = first; n < last; n++)
            {
                int length = Math.Clamp(inLengths[n], 0, steps), labels = tLengths[n], states = 2 * labels + 1;
                int Label(int s) => s % 2 == 0 ? blank : intTargets ? BitConverter.SingleToInt32Bits(tw[tOffsets[n] + s / 2]) : (int)tw[tOffsets[n] + s / 2];
                long At(int t, int c) => batchFirst ? ((long)n * steps + t) * classes + c : ((long)t * batch + n) * classes + c;

                // A typed log-probability is read through its type a row (a step's classes) at a time, the row the passes are on.
                float[] rowValues = lp;
                long rowShift = 0;
                int rowStep = -1;
                float Lp(int t, int c)
                {
                    if (tlp.Typed && t != rowStep)
                    {
                        rowValues = tlp.Values(lp, At(t, 0), classes, out rowShift);
                        rowStep = t;
                    }

                    return rowValues[rowShift + At(t, c)];
                }

                var alpha = new float[Math.Max(1, length) * states];
                for (int i = 0; i < alpha.Length; i++)
                {
                    alpha[i] = float.NegativeInfinity;
                }

                if (length > 0)
                {
                    alpha[0] = Lp(0, blank);
                    if (states > 1)
                    {
                        alpha[1] = Lp(0, Label(1));
                    }

                    for (int t = 1; t < length; t++)
                    {
                        for (int s = 0; s < states; s++)
                        {
                            float a = alpha[(t - 1) * states + s];
                            if (s > 0)
                            {
                                a = LogAdd(a, alpha[(t - 1) * states + s - 1]);
                            }

                            if (s > 1 && s % 2 == 1 && Label(s) != Label(s - 2))
                            {
                                a = LogAdd(a, alpha[(t - 1) * states + s - 2]);
                            }

                            alpha[t * states + s] = a + Lp(t, Label(s));
                        }
                    }
                }

                float logP = length == 0 ? (labels == 0 ? 0 : float.NegativeInfinity)
                    : states > 1 ? LogAdd(alpha[(length - 1) * states + states - 1], alpha[(length - 1) * states + states - 2]) : alpha[(length - 1) * states];
                bool infinite = float.IsNegativeInfinity(logP);
                if (loss is not null)
                {
                    loss[n] = infinite && zeroInfinity ? 0f : -logP;
                    continue;
                }

                if (infinite || length == 0 || dlp is null || grads is null)
                {
                    continue;
                }

                var beta = new float[length * states];
                for (int i = 0; i < beta.Length; i++)
                {
                    beta[i] = float.NegativeInfinity;
                }

                beta[(length - 1) * states + states - 1] = Lp(length - 1, Label(states - 1));
                if (states > 1)
                {
                    beta[(length - 1) * states + states - 2] = Lp(length - 1, Label(states - 2));
                }

                for (int t = length - 2; t >= 0; t--)
                {
                    for (int s = 0; s < states; s++)
                    {
                        float b = beta[(t + 1) * states + s];
                        if (s < states - 1)
                        {
                            b = LogAdd(b, beta[(t + 1) * states + s + 1]);
                        }

                        if (s < states - 2 && s % 2 == 1 && Label(s) != Label(s + 2))
                        {
                            b = LogAdd(b, beta[(t + 1) * states + s + 2]);
                        }

                        beta[t * states + s] = b + Lp(t, Label(s));
                    }
                }

                float g = grads[n];
                var byClass = new float[classes];
                for (int t = 0; t < length; t++)
                {
                    Array.Fill(byClass, float.NegativeInfinity);
                    for (int s = 0; s < states; s++)
                    {
                        int c = Label(s);
                        byClass[c] = LogAdd(byClass[c], alpha[t * states + s] + beta[t * states + s]);
                    }

                    for (int c = 0; c < classes; c++)
                    {
                        if (!float.IsNegativeInfinity(byClass[c]))
                        {
                            long at = At(t, c);
                            dlp[at] -= g * Fn<float, F1>.Exp(byClass[c] - Lp(t, c) - logP);
                        }
                    }
                }
            }
        });
    }

    // ln(e^a + e^b) in float32 (the lanes' own exp and log).
    private static float LogAdd(float a, float b)
    {
        if (float.IsNegativeInfinity(a))
        {
            return b;
        }

        if (float.IsNegativeInfinity(b))
        {
            return a;
        }

        float max = MathF.Max(a, b);
        return max + Fn<float, F1>.Log(Fn<float, F1>.Exp(a - max) + Fn<float, F1>.Exp(b - max));
    }
}
