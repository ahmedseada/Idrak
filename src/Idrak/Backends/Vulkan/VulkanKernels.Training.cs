// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Backends.Vulkan;

// Training kernels: the layer-norm gradient, grouped normalization (batch norm) and its reductions, column sums, the
// fused AdamW and 8-bit Adam updates, and the gradient of tiled causal attention. Every sum is taken in a fixed order
// (partial sums per split, then added split by split), so results are the same run after run; nothing uses atomics.
internal static partial class VulkanKernels
{
    /// <summary>Words of 8-bit moments per block of <see cref="Optimizers.AdamW8Bit.BlockSize"/> elements (four bytes each).</summary>
    public const int AdamBlockWords = Optimizers.AdamW8Bit.BlockSize / 4;

    /// <summary>Values each partial of the two-pass reductions keeps (count, mean, M2; or two sums and an unused one).</summary>
    public const int PartialStride = 3;

    private static IEnumerable<(string, Func<SpirvKernel>)> TrainingKernels()
    {
        // dx += rstd · (g - mean(g) - x̂ · mean(g · x̂)) per row, g = dy · gamma, x̂ = (x - mean) · rstd from the stats the
        // forward pass kept (stats[r] = mean, stats[rows + r] = rstd); a workgroup per row, or an invocation ("_narrow").
        foreach (bool narrow in new[] { false, true })
        {
            string name = narrow ? "layer_norm_backward_narrow" : "layer_norm_backward";
            yield return (name, () =>
            {
                var k = new KernelBuilder(name, Block);
                var (x, gamma, dy, stats, dx) = (k.Buffer("x"), k.Buffer("gamma"), k.Buffer("dy"), k.Buffer("stats"), k.Buffer("dx"));
                var (rows, cols) = (k.PushInt("rows"), k.PushInt("cols"));
                var w = new RowWork(k, narrow);
                w.Each(rows, r =>
                {
                    var o = r * cols;
                    var (mean, rstd) = (stats[r], stats[rows + r]);
                    var (s1, s2) = (k.Local(0f), k.Local(0f));
                    w.Columns(cols, j =>
                    {
                        var g = dy[o + j] * gamma[j];
                        s1.V = s1.V + g;
                        s2.V = s2.V + g * ((x[o + j] - mean) * rstd);
                    });
                    var count = cols.ToFloat();
                    var m1 = w.Sum(s1.V) / count;
                    var m2 = w.Sum(s2.V) / count;
                    w.Columns(cols, j =>
                    {
                        var xhat = (x[o + j] - mean) * rstd;
                        dx[o + j] = dx[o + j] + rstd * (dy[o + j] * gamma[j] - m1 - xhat * m2);
                    });
                });
                return k.Build();
            });
        }

        // First pass of the column reductions: one invocation per (split s, column j) takes rows [s·chunk, (s+1)·chunk) of
        // the block a[offset + r·ld + j] in order and writes part[(s·cols + j)·3 …]:
        //   mode 0: count, mean and Σ(a - mean)² (the moments, for normalization statistics);
        //   mode 1: Σ a and Σ a·b (b may be a again);
        //   mode 2: Σ b·x̂ and Σ b with x̂ = (a - stats[r]) · stats[rows + r] (layer-norm gamma and beta gradients).
        yield return ("group_partials_columns", () =>
        {
            var k = new KernelBuilder("group_partials_columns", Block);
            var (a, b, stats, part) = (k.Buffer("a"), k.Buffer("b"), k.Buffer("stats"), k.Buffer("part"));
            var (rows, cols, ld, offset, splits, mode) = (k.PushInt("rows"), k.PushInt("cols"), k.PushInt("ld"), k.PushInt("offset"), k.PushInt("splits"), k.PushInt("mode"));
            Grid(k, splits * cols, idx =>
            {
                var (s, j) = (idx / cols, idx % cols);
                var chunk = (rows + splits - 1) / splits;
                var start = s * chunk;
                var end = k.Min(rows, start + chunk);
                var at = offset + j;
                var o = idx * PartialStride;
                var (s1, s2) = (k.Local(0f), k.Local(0f));
                k.If(mode.Eq(0), () =>
                {
                    k.For(start, end, 1, r => s1.V = s1.V + a[at + r * ld]);
                    var n = k.Max(end - start, k.Int(0)).ToFloat();
                    var mean = k.Select(n > 0f, s1.V / k.Max(n, k.Float(1f)), k.Float(0f));
                    k.For(start, end, 1, r =>
                    {
                        var d = a[at + r * ld] - mean;
                        s2.V = k.Fma(d, d, s2.V);
                    });
                    part[o] = n;
                    part[o + 1] = mean;
                    part[o + 2] = s2.V;
                }, () =>
                {
                    k.If(mode.Eq(1), () => k.For(start, end, 1, r =>
                    {
                        var v = a[at + r * ld];
                        s1.V = s1.V + v;
                        s2.V = k.Fma(v, b[at + r * ld], s2.V);
                    }), () => k.For(start, end, 1, r =>
                    {
                        var g = b[at + r * ld];
                        s1.V = k.Fma(g, (a[at + r * ld] - stats[r]) * stats[rows + r], s1.V);
                        s2.V = s2.V + g;
                    }));
                    part[o] = s1.V;
                    part[o + 1] = s2.V;
                    part[o + 2] = k.Float(0f);
                });
            });
            return k.Build();
        });

        // First pass of the group reductions over long contiguous runs: a workgroup per (group g, split s) of the
        // [outer, groups, inner] view (group g holds M = outer · inner elements, element m at ((m / inner) · groups + g) ·
        // inner + m % inner) takes elements [s·chunk, (s+1)·chunk) and writes part[(g·splits + s)·3 …]: mode 0 the
        // moments (count, mean, M2: a sum, then the squares about the mean), mode 1 Σ a and Σ a·b.
        yield return ("group_partials_rows", () =>
        {
            var k = new KernelBuilder("group_partials_rows", Block);
            var (a, b, part) = (k.Buffer("a"), k.Buffer("b"), k.Buffer("part"));
            var (outer, groups, inner, splits, mode) = (k.PushInt("outer"), k.PushInt("groups"), k.PushInt("inner"), k.PushInt("splits"), k.PushInt("mode"));
            var scratch = k.Shared("scratch", Block);
            EachRow(k, groups * splits, w =>
            {
                var (g, s) = (w / splits, w % splits);
                var total = outer * inner;
                var chunk = (total + splits - 1) / splits;
                var start = s * chunk;
                var end = k.Min(total, start + chunk);
                Val At(Val m) => (m / inner * groups + g) * inner + m % inner;
                var o = w * PartialStride;
                var (s1, s2) = (k.Local(0f), k.Local(0f));
                k.If(mode.Eq(0), () =>
                {
                    k.For(start + k.LocalX, end, Block, m => s1.V = s1.V + a[At(m)]);
                    var sum = k.ReduceSum(scratch, s1.V);
                    var n = k.Max(end - start, k.Int(0)).ToFloat();
                    var mean = k.Select(n > 0f, sum / k.Max(n, k.Float(1f)), k.Float(0f));
                    k.For(start + k.LocalX, end, Block, m =>
                    {
                        var d = a[At(m)] - mean;
                        s2.V = k.Fma(d, d, s2.V);
                    });
                    var m2 = k.ReduceSum(scratch, s2.V);
                    k.If(k.LocalX.Eq(0), () =>
                    {
                        part[o] = n;
                        part[o + 1] = mean;
                        part[o + 2] = m2;
                    });
                }, () =>
                {
                    k.For(start + k.LocalX, end, Block, m =>
                    {
                        var at = At(m);
                        var v = a[at];
                        s1.V = s1.V + v;
                        s2.V = k.Fma(v, b[at], s2.V);
                    });
                    var sumA = k.ReduceSum(scratch, s1.V);
                    var sumAB = k.ReduceSum(scratch, s2.V);
                    k.If(k.LocalX.Eq(0), () =>
                    {
                        part[o] = sumA;
                        part[o + 1] = sumAB;
                        part[o + 2] = k.Float(0f);
                    });
                });
            });
            return k.Build();
        });

        // Second pass: one invocation per group g adds its partials p = s·strideS + g·strideG + i (s < splits, i < count)
        // in that order. Mode 0 merges moments (Chan et al.) into mean (out0), biased variance (out1) and
        // 1 / sqrt(variance + eps) (out2); mode 1 adds the first sums to out0 (flags bit 0) and the second to out1 (bit 1).
        yield return ("group_finish", () =>
        {
            var k = new KernelBuilder("group_finish", Block);
            var (part, out0, out1, out2) = (k.Buffer("part"), k.Buffer("out0"), k.Buffer("out1"), k.Buffer("out2"));
            var (groups, splits, strideS, strideG, count) = (k.PushInt("groups"), k.PushInt("splits"), k.PushInt("strideS"), k.PushInt("strideG"), k.PushInt("count"));
            var (mode, flags, eps) = (k.PushInt("mode"), k.PushInt("flags"), k.PushFloat("eps"));
            Grid(k, groups, g =>
            {
                var (n, mean, m2) = (k.Local(0f), k.Local(0f), k.Local(0f));
                k.For(k.Int(0), splits, 1, s => k.For(k.Int(0), count, 1, i =>
                {
                    var o = (s * strideS + g * strideG + i) * PartialStride;
                    k.If(mode.Eq(0), () =>
                    {
                        var nb = part[o];
                        k.If(nb > 0f, () =>
                        {
                            var na = n.V;
                            var total = na + nb;
                            var delta = part[o + 1] - mean.V;
                            mean.V = mean.V + delta * (nb / total);
                            m2.V = m2.V + part[o + 2] + delta * delta * (na * nb / total);
                            n.V = total;
                        });
                    }, () =>
                    {
                        mean.V = mean.V + part[o];
                        m2.V = m2.V + part[o + 1];
                    });
                }));
                k.If(mode.Eq(0), () =>
                {
                    var variance = k.Select(n.V > 0f, k.Max(m2.V / k.Max(n.V, k.Float(1f)), k.Float(0f)), k.Float(0f));
                    out0[g] = mean.V;
                    out1[g] = variance;
                    out2[g] = 1f / k.Sqrt(variance + eps);
                }, () =>
                {
                    k.If((flags & 1).Ne(0), () => out0[g] = out0[g] + mean.V);
                    k.If((flags & 2).Ne(0), () => out1[g] = out1[g] + m2.V);
                });
            });
            return k.Build();
        });

        // y = (x - mean[g]) · invStd[g] as the CPU computes it (x · invStd + (-mean · invStd)), g = (i / inner) % groups.
        yield return ("norm_apply", () =>
        {
            var k = new KernelBuilder("norm_apply", Block);
            var (x, mean, invStd, y) = (k.Buffer("x"), k.Buffer("mean"), k.Buffer("invStd"), k.Buffer("y"));
            var (n, groups, inner) = (k.PushInt("n"), k.PushInt("groups"), k.PushInt("inner"));
            Grid(k, n, i =>
            {
                var g = i / inner % groups;
                var s = invStd[g];
                y[i] = k.Fma(x[i], s, -mean[g] * s);
            });
            return k.Build();
        });

        // dx += dxhat · invStd + (xhat · (-invStd · sum2 / M) - invStd · sum1 / M), the CPU's two passes in one.
        yield return ("norm_backward", () =>
        {
            var k = new KernelBuilder("norm_backward", Block);
            var (dxhat, xhat, sum1, sum2, invStd, dx) = (k.Buffer("dxhat"), k.Buffer("xhat"), k.Buffer("sum1"), k.Buffer("sum2"), k.Buffer("invStd"), k.Buffer("dx"));
            var (n, groups, inner, count) = (k.PushInt("n"), k.PushInt("groups"), k.PushInt("inner"), k.PushFloat("count"));
            Grid(k, n, i =>
            {
                var g = i / inner % groups;
                var s = invStd[g];
                var first = dx[i] + dxhat[i] * s;
                dx[i] = first + k.Fma(xhat[i], -s * sum2[g] / count, -s * sum1[g] / count);
            });
            return k.Build();
        });

        // y (+)= x · scale[g] + shift[g]; flags bit 0: a scale (else 1), bit 1: a shift (else 0), bit 2: accumulate.
        yield return ("group_scale_shift", () =>
        {
            var k = new KernelBuilder("group_scale_shift", Block);
            var (x, scale, shift, y) = (k.Buffer("x"), k.Buffer("scale"), k.Buffer("shift"), k.Buffer("y"));
            var (n, groups, inner, flags) = (k.PushInt("n"), k.PushInt("groups"), k.PushInt("inner"), k.PushInt("flags"));
            var (scaled, shifted, accumulate) = ((flags & 1).Ne(0), (flags & 2).Ne(0), (flags & 4).Ne(0));
            Grid(k, n, i =>
            {
                var g = i / inner % groups;
                var sc = k.Local(1f);
                var sh = k.Local(0f);
                k.If(scaled, () => sc.V = scale[g]);
                k.If(shifted, () => sh.V = shift[g]);
                var v = k.Fma(x[i], sc.V, sh.V);
                k.If(accumulate, () => y[i] = y[i] + v, () => y[i] = v);
            });
            return k.Build();
        });

        // c[i] = bias[i % cols] (the bias rows a product then adds itself to).
        yield return ("broadcast_rows", () =>
        {
            var k = new KernelBuilder("broadcast_rows", Block);
            var (bias, c) = (k.Buffer("bias"), k.Buffer("c"));
            var (n, cols) = (k.PushInt("n"), k.PushInt("cols"));
            Grid(k, n, i => c[i] = bias[i % cols]);
            return k.Build();
        });

        // part[offset + group] = Σ x² over the group's elements (i ≡ group · width + lane, step groups · width): the first
        // pass of the global gradient norm, one dispatch per tensor.
        yield return ("sum_squares_partials", () =>
        {
            var k = new KernelBuilder("sum_squares_partials", Block);
            var (x, part) = (k.Buffer("x"), k.Buffer("part"));
            var (n, offset) = (k.PushInt("n"), k.PushInt("offset"));
            var scratch = k.Shared("scratch", Block);
            var acc = k.Local(0f);
            k.For(k.GroupX * Block + k.LocalX, n, i => acc.V = k.Fma(x[i], x[i], acc.V), k.GridStrideX);
            var sum = k.ReduceSum(scratch, acc.V);
            k.If(k.LocalX.Eq(0), () => part[offset + k.GroupX] = sum);
            return k.Build();
        });

        // factor[0] = min(1, maxNorm / √Σ part) (1 when the sum is 0): one workgroup adds the partials, lane-strided and
        // then through a fixed tree.
        yield return ("clip_factor_partials", () =>
        {
            var k = new KernelBuilder("clip_factor_partials", Block);
            var (part, factor) = (k.Buffer("part"), k.Buffer("factor"));
            var (count, maxNorm) = (k.PushInt("count"), k.PushFloat("maxNorm"));
            var scratch = k.Shared("scratch", Block);
            var acc = k.Local(0f);
            k.For(k.LocalX, count, Block, i => acc.V = acc.V + part[i]);
            var sum = k.ReduceSum(scratch, acc.V);
            k.If(k.LocalX.Eq(0), () => factor[k.Int(0)] = k.Select(sum > 0f, k.Min(k.Float(1f), maxNorm / k.Sqrt(sum)), k.Float(1f)));
            return k.Build();
        });

        // AdamW with the gradient scaled by factor[0]: p = p · decay - lr · m / (√v + eps) with m, v updated in place; zero ≠
        // 0 also clears the gradient.
        yield return ("fused_adamw", () =>
        {
            var k = new KernelBuilder("fused_adamw", Block);
            var (p, g, m, v, factor) = (k.Buffer("p"), k.Buffer("g"), k.Buffer("m"), k.Buffer("v"), k.Buffer("factor"));
            var (n, lr, decay, beta1, beta2, eps, zero) = (k.PushInt("n"), k.PushFloat("lr"), k.PushFloat("decay"), k.PushFloat("beta1"), k.PushFloat("beta2"),
                k.PushFloat("eps"), k.PushInt("zero"));
            var f = factor[k.Int(0)];
            var (c1, c2) = (1f - beta1, 1f - beta2);
            Grid(k, n, i =>
            {
                var grad = g[i] * f;
                var mom = k.Fma(beta1, m[i], c1 * grad);
                var vel = k.Fma(beta2, v[i], c2 * grad * grad);
                m[i] = mom;
                v[i] = vel;
                p[i] = p[i] * decay - lr * mom / (k.Sqrt(vel) + eps);
                k.If(zero.Ne(0), () => g[i] = k.Float(0f));
            });
            return k.Build();
        });

        yield return ("adam_8bit", Adam8Bit);
        yield return ("attention_backward_dq", AttentionBackwardQueries);
        yield return ("attention_backward_dkv", AttentionBackwardKeys);
    }

    // n / d rounded as the CPU divides: the quotient corrected by its residual (the device's division may be a few ulp off).
    private static Val Quotient(KernelBuilder k, Val n, Val d)
    {
        var q = n / d;
        return k.Fma(k.Fma(-q, d, n), 1f / d, q);
    }

    // The code of the sorted 256-entry map at map[base …] nearest to x, as Optimizers.AdamW8Bit.Nearest finds it.
    private static Val NearestCode(KernelBuilder k, Buf map, int start, Val x)
    {
        var lo = k.Local(0);
        for (int step = 128; step > 0; step >>= 1)
        {
            int s = step;
            k.If(map[lo.V + (start + s)] <= x, () => lo.V = lo.V + s);
        }

        var at = lo.V;
        k.If(at < 255, () => k.If(map[at + (start + 1)] - x < x - map[at + start], () => lo.V = at + 1));
        return lo.V;
    }

    // 8-bit Adam (see Backend.AdamStep8Bit): a workgroup per block of 256 elements; invocation w < 64 takes the block's
    // word w (elements 4w … 4w + 3): it decodes the moments, updates them and the parameters, the workgroup finds the
    // new block scales (largest |m| and v), and it encodes its four moments to the nearest codes. Bytes past n keep their
    // bits. Invocations beyond the 64 words take part in the reductions only.
    private static SpirvKernel Adam8Bit()
    {
        var k = new KernelBuilder("adam_8bit", Block);
        var (p, g, m, v, absMax, map) = (k.Buffer("p"), k.Buffer("g"), k.Buffer("m"), k.Buffer("v"), k.Buffer("absMax"), k.Buffer("map"));
        var (n, lr, beta1, beta2, eps) = (k.PushInt("n"), k.PushFloat("lr"), k.PushFloat("beta1"), k.PushFloat("beta2"), k.PushFloat("eps"));
        var (gradientScale, decay, blocks) = (k.PushFloat("gradientScale"), k.PushFloat("decay"), k.PushInt("blocks"));
        var scratch = k.Shared("scratch", Block);
        var lane = k.LocalX;
        var (c1, c2) = (1f - beta1, 1f - beta2);
        var mom = new Var[4];
        var vel = new Var[4];
        for (int e = 0; e < 4; e++)
        {
            (mom[e], vel[e]) = (k.Local(ScalarKind.Float), k.Local(ScalarKind.Float));
        }

        EachRow(k, blocks, block =>
        {
            var word = block * AdamBlockWords + lane;
            var first = word * 4;
            var mine = (lane < AdamBlockWords) & (first < n);
            var (mScale, vScale) = (absMax[block], absMax[blocks + block]);
            var (mWord, vWord) = (k.Local(k.UInt(0)), k.Local(k.UInt(0)));
            var (mMax, vMax) = (k.Local(0f), k.Local(0f));
            k.If(mine, () =>
            {
                mWord.V = m.UInt(word);
                vWord.V = v.UInt(word);
                for (int e = 0; e < 4; e++)
                {
                    int shift = 8 * e;
                    var i = first + e;
                    k.If(i < n, () =>
                    {
                        var grad = g[i] * gradientScale;
                        var mCode = (mWord.V >> shift & 255).ToInt();
                        var vCode = (vWord.V >> shift & 255).ToInt();
                        var mo = k.Fma(beta1, map[mCode] * mScale, c1 * grad);
                        var ve = k.Fma(beta2, map[vCode + 256] * vScale, grad * grad * c2);
                        p[i] = p[i] * decay - mo / (k.Sqrt(ve) + eps) * lr;
                        mom[e].V = mo;
                        vel[e].V = ve;
                        mMax.V = k.Max(mMax.V, k.Abs(mo));
                        vMax.V = k.Max(vMax.V, ve);
                    });
                }
            });
            var mTop = k.ReduceMax(scratch, mMax.V);
            var vTop = k.ReduceMax(scratch, vMax.V);
            k.If(mine, () =>
            {
                var (mOut, vOut) = (k.Local(mWord.V), k.Local(vWord.V));
                for (int e = 0; e < 4; e++)
                {
                    int shift = 8 * e;
                    var keep = k.UInt(~(255u << shift));
                    k.If(first + e < n, () =>
                    {
                        var mq = NearestCode(k, map, 0, k.Select(mTop > 0f, Quotient(k, mom[e].V, k.Max(mTop, k.Float(float.Epsilon))), k.Float(0f)));
                        var vq = NearestCode(k, map, 256, k.Select(vTop > 0f, Quotient(k, vel[e].V, k.Max(vTop, k.Float(float.Epsilon))), k.Float(0f)));
                        mOut.V = (mOut.V & keep) | mq.ToUInt() << shift;
                        vOut.V = (vOut.V & keep) | vq.ToUInt() << shift;
                    });
                }

                m[word] = mOut.V.AsFloat();
                v[word] = vOut.V.AsFloat();
            });
            k.If(lane.Eq(0), () =>
            {
                absMax[block] = mTop;
                absMax[blocks + block] = vTop;
            });
        });
        return k.Build();
    }

    // Dimensions of a head each invocation of the attention gradients keeps (the head size up to AttentionMaxDim over the width).
    private static int AttentionDimsPerLane => Math.Max(1, AttentionMaxDim / Block);

    // The gradient of tiled causal attention for the queries (see Backend.AttentionTiledBackward): a workgroup per query
    // row (h, i), which sees keys c ≤ min(i % steps, capacity - 1) from its window's start (0 without a window). It stages
    // the row's query and dOutput, writes delta = dOutput · output for the keys' pass, then takes the keys a width at a
    // time: invocation j computes the score s of key c0 + j (soft-capped with a cap), its weight p = exp(s - lse) and
    // ds = p · (dOutput · v_c - delta) · s' into workgroup memory (s' the cap's slope, 1 without one); then every
    // invocation adds Σ_j ds_j · k_(c0+j) to the dimensions it keeps, keys in order. dq += scale · that sum.
    //
    // Workgroup memory: qs and gs (the row's query and dOutput) are written before a barrier and only read after it;
    // ds is written between the barrier ending the previous chunk's reads and the barrier before this chunk's reads.
    private static SpirvKernel AttentionBackwardQueries()
    {
        var k = new KernelBuilder("attention_backward_dq", Block);
        var (q, keys, values, output, lse, dOutput) = (k.Buffer("q"), k.Buffer("keys"), k.Buffer("values"), k.Buffer("output"), k.Buffer("lse"), k.Buffer("dOutput"));
        var (dq, delta) = (k.Buffer("dq"), k.Buffer("delta"));
        var (heads, rowsPerHead, steps, capacity, dim, scale) = (k.PushInt("heads"), k.PushInt("rowsPerHead"), k.PushInt("steps"), k.PushInt("capacity"),
            k.PushInt("dim"), k.PushFloat("scale"));
        var (window, softcap) = (k.PushInt("window"), k.PushFloat("softcap"));
        var qs = k.Shared("qs", AttentionMaxDim);
        var gs = k.Shared("gs", AttentionMaxDim);
        var ds = k.Shared("ds", Block);
        var scratch = k.Shared("scratch", Block);
        var lane = k.LocalX;
        int per = AttentionDimsPerLane;
        var acc = new Var[per];
        for (int t = 0; t < per; t++)
        {
            acc[t] = k.Local(ScalarKind.Float);
        }

        EachRow(k, heads * rowsPerHead, row =>
        {
            var h = row / rowsPerHead;
            var count = k.Min(row % rowsPerHead % steps, capacity - 1) + 1;
            var o = row * dim;
            var partial = k.Local(0f);
            k.For(lane, dim, Block, d =>
            {
                var g = dOutput[o + d];
                qs[d] = q[o + d];
                gs[d] = g;
                partial.V = k.Fma(g, output[o + d], partial.V);
            });
            var rowDelta = k.ReduceSum(scratch, partial.V);             // its barriers also publish qs and gs
            k.If(lane.Eq(0), () => delta[row] = rowDelta);
            var rowLse = lse[row];
            foreach (var a in acc)
            {
                a.V = k.Float(0f);
            }

            k.For(WindowStart(k, count, window), count, Block, c0 =>
            {
                var c = c0 + lane;
                var weight = k.Local(0f);
                k.If(c < count, () =>
                {
                    var key = (h * capacity + c) * dim;
                    var (dot, dp) = (k.Local(0f), k.Local(0f));
                    k.For(k.Int(0), dim, 1, d =>
                    {
                        dot.V = k.Fma(qs[d], keys[key + d], dot.V);
                        dp.V = k.Fma(gs[d], values[key + d], dp.V);
                    });
                    var score = Capped(k, dot.V * scale, softcap);
                    var pr = k.Exp(score - rowLse);
                    weight.V = pr * (dp.V - rowDelta) * CapSlope(k, score, softcap);
                });
                ds[lane] = weight.V;
                k.Barrier();
                var used = k.Min(count - c0, k.Int(Block));
                for (int t = 0; t < per; t++)
                {
                    var d = lane + t * Block;
                    var a = acc[t];
                    k.If(d < dim, () => k.For(k.Int(0), used, 1, j => a.V = k.Fma(ds[j], keys[(h * capacity + c0 + j) * dim + d], a.V)));
                }

                k.Barrier();
            });

            for (int t = 0; t < per; t++)
            {
                var d = lane + t * Block;
                var a = acc[t];
                k.If(d < dim, () => dq[o + d] = k.Fma(scale, a.V, dq[o + d]));
            }
        });
        return k.Build();
    }

    // The gradient for the keys and values: a workgroup per key position (h, c) with c < min(capacity, steps), seen by
    // the rows i of head h with min(i % steps, capacity - 1) ≥ c (and, with a window, less than the window past c; the
    // scores soft-capped as in the queries' pass). It stages k_c and v_c, then takes the rows a width at
    // a time: invocation j computes row i0 + j's weight p and ds (with the delta the queries' pass wrote) into workgroup
    // memory, zero for rows that do not see c; then every invocation adds Σ_j ds_j · q_(i0+j) and Σ_j p_j · dOutput_(i0+j)
    // to the dimensions it keeps, rows in order. dk += scale · the first sum, dv += the second.
    //
    // Workgroup memory: ks and vs are written before a barrier and only read after it; ps and ds as in the queries' pass.
    private static SpirvKernel AttentionBackwardKeys()
    {
        var k = new KernelBuilder("attention_backward_dkv", Block);
        var (q, keys, values, lse, dOutput, delta) = (k.Buffer("q"), k.Buffer("keys"), k.Buffer("values"), k.Buffer("lse"), k.Buffer("dOutput"), k.Buffer("delta"));
        var (dkeys, dvalues) = (k.Buffer("dkeys"), k.Buffer("dvalues"));
        var (heads, rowsPerHead, steps, capacity, dim, scale) = (k.PushInt("heads"), k.PushInt("rowsPerHead"), k.PushInt("steps"), k.PushInt("capacity"),
            k.PushInt("dim"), k.PushFloat("scale"));
        var (window, softcap) = (k.PushInt("window"), k.PushFloat("softcap"));
        var ks = k.Shared("ks", AttentionMaxDim);
        var vs = k.Shared("vs", AttentionMaxDim);
        var ps = k.Shared("ps", Block);
        var ds = k.Shared("ds", Block);
        var lane = k.LocalX;
        int per = AttentionDimsPerLane;
        var dk = new Var[per];
        var dv = new Var[per];
        for (int t = 0; t < per; t++)
        {
            (dk[t], dv[t]) = (k.Local(ScalarKind.Float), k.Local(ScalarKind.Float));
        }

        var used = k.Min(capacity, steps);
        EachRow(k, heads * used, slot =>
        {
            var (h, c) = (slot / used, slot % used);
            var key = (h * capacity + c) * dim;
            k.For(lane, dim, Block, d =>
            {
                ks[d] = keys[key + d];
                vs[d] = values[key + d];
            });
            k.Barrier();
            for (int t = 0; t < per; t++)
            {
                dk[t].V = k.Float(0f);
                dv[t].V = k.Float(0f);
            }

            k.For(k.Int(0), rowsPerHead, Block, i0 =>
            {
                var i = i0 + lane;
                var (weight, slope) = (k.Local(0f), k.Local(0f));
                var limit = k.Min(i % steps, capacity - 1);
                k.If((i < rowsPerHead) & (limit >= c) & ((window <= 0) | (limit - c < window)), () =>
                {
                    var row = h * rowsPerHead + i;
                    var o = row * dim;
                    var (dot, dp) = (k.Local(0f), k.Local(0f));
                    k.For(k.Int(0), dim, 1, d =>
                    {
                        dot.V = k.Fma(q[o + d], ks[d], dot.V);
                        dp.V = k.Fma(dOutput[o + d], vs[d], dp.V);
                    });
                    var score = Capped(k, dot.V * scale, softcap);
                    var pr = k.Exp(score - lse[row]);
                    weight.V = pr;
                    slope.V = pr * (dp.V - delta[row]) * CapSlope(k, score, softcap);
                });

                ps[lane] = weight.V;
                ds[lane] = slope.V;
                k.Barrier();
                var rows = k.Min(rowsPerHead - i0, k.Int(Block));
                for (int t = 0; t < per; t++)
                {
                    var d = lane + t * Block;
                    var (a, b) = (dk[t], dv[t]);
                    k.If(d < dim, () => k.For(k.Int(0), rows, 1, j =>
                    {
                        var o = (h * rowsPerHead + i0 + j) * dim + d;
                        a.V = k.Fma(ds[j], q[o], a.V);
                        b.V = k.Fma(ps[j], dOutput[o], b.V);
                    }));
                }

                k.Barrier();
            });

            for (int t = 0; t < per; t++)
            {
                var d = lane + t * Block;
                var (a, b) = (dk[t], dv[t]);
                k.If(d < dim, () =>
                {
                    dkeys[key + d] = k.Fma(scale, a.V, dkeys[key + d]);
                    dvalues[key + d] = dvalues[key + d] + b.V;
                });
            }
        });
        return k.Build();
    }
}
