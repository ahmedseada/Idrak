// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Vulkan;

// Row kernels: one workgroup per row (rows looped with a stride of the group count), each invocation taking the
// columns j ≡ its lane; per-row sums and maxima through workgroup reductions. An invocation always reads and writes the
// same columns in every pass, so the output may be the input (in place).
internal static partial class VulkanKernels
{
    // Every row kernel twice: a workgroup per row, and an invocation per row ("_narrow") for short rows.
    private static IEnumerable<(string, Func<SpirvKernel>)> RowKernels() => RowKernels(narrow: false).Concat(RowKernels(narrow: true));

    private static IEnumerable<(string, Func<SpirvKernel>)> RowKernels(bool narrow)
    {
        string Named(string name) => narrow ? name + "_narrow" : name;

        // y = softmax(x) per row, or log-softmax with log ≠ 0.
        yield return (Named("softmax"), () =>
        {
            var k = new KernelBuilder(Named("softmax"), Block);
            var (x, y) = (k.Buffer("x"), k.Buffer("y"));
            var (rows, cols, log) = (k.PushInt("rows"), k.PushInt("cols"), k.PushInt("log"));
            var w = new RowWork(k, narrow);
            w.Each(rows, r =>
            {
                var o = r * cols;
                var max = w.RowMax(cols, j => x[o + j]);
                k.If(log.Ne(0), () =>
                {
                    var logSum = k.Log(w.RowSum(cols, j => k.Exp(x[o + j] - max))) + max;
                    w.Columns(cols, j => y[o + j] = x[o + j] - logSum);
                }, () =>
                {
                    var inverse = 1f / w.RowSum(cols, j =>
                    {
                        var e = k.Exp(x[o + j] - max);
                        y[o + j] = e;
                        return e;
                    });
                    w.Columns(cols, j => y[o + j] = y[o + j] * inverse);
                });
            });
            return k.Build();
        });

        // y[r] (+)= scale · Σ_j x[r, j]: the axis sum over a row (inner 1; sum_rows in VulkanKernels.cs sums the columns); sum_axis in VulkanKernels.Shapes.cs takes the
        // rest. The row's partial sums go through the workgroup reduction in a fixed order (rule 86).
        yield return (Named("sum_axis_rows"), () =>
        {
            var k = new KernelBuilder(Named("sum_axis_rows"), Block);
            var (x, y) = (k.Buffer("x"), k.Buffer("y"));
            var (rows, cols, scale, accumulate) = (k.PushInt("rows"), k.PushInt("cols"), k.PushFloat("scale"), k.PushInt("accumulate"));
            var w = new RowWork(k, narrow);
            w.Each(rows, r =>
            {
                var o = r * cols;
                var s = scale * w.RowSum(cols, j => x[o + j]);
                w.Leader(() => k.If(accumulate.Ne(0), () => y[r] = y[r] + s, () => y[r] = s));
            });
            return k.Build();
        });

        // Softmax: dx += y · (dy - Σ dy·y). Log-softmax: dx += dy - exp(y) · Σ dy.
        yield return (Named("softmax_backward"), () =>
        {
            var k = new KernelBuilder(Named("softmax_backward"), Block);
            var (y, dy, dx) = (k.Buffer("y"), k.Buffer("dy"), k.Buffer("dx"));
            var (rows, cols, log) = (k.PushInt("rows"), k.PushInt("cols"), k.PushInt("log"));
            var w = new RowWork(k, narrow);
            w.Each(rows, r =>
            {
                var o = r * cols;
                var isLog = log.Ne(0);
                var s = w.RowSum(cols, j => k.Select(isLog, dy[o + j], dy[o + j] * y[o + j]));
                w.Columns(cols, j =>
                {
                    var g = dy[o + j];
                    dx[o + j] = k.AddRounded(dx[o + j], k.Select(isLog, g - k.MulRounded(k.Exp(y[o + j]), s), k.MulRounded(y[o + j], g - s)));
                });
            });
            return k.Build();
        });

        // y[r, :] = softmax(scale · x[r, :] + mask[r % maskRows, :]); hasMask = 0: no mask (bind any storage there).
        yield return (Named("scale_mask_softmax"), () =>
        {
            var k = new KernelBuilder(Named("scale_mask_softmax"), Block);
            var (x, mask, y) = (k.Buffer("x"), k.Buffer("mask"), k.Buffer("y"));
            var (rows, cols, maskRows, scale, hasMask) = (k.PushInt("rows"), k.PushInt("cols"), k.PushInt("maskRows"), k.PushFloat("scale"), k.PushInt("hasMask"));
            var w = new RowWork(k, narrow);
            w.Each(rows, r =>
            {
                var o = r * cols;
                var mo = r % k.Max(maskRows, k.Int(1)) * cols;
                var masked = hasMask.Ne(0);
                var max = k.Local(float.NegativeInfinity);
                w.Columns(cols, j =>
                {
                    var add = k.Local(0f);
                    k.If(masked, () => add.V = mask[mo + j]);
                    var v = k.AddRounded(k.MulRounded(x[o + j], scale), add.V);
                    y[o + j] = v;
                    max.V = k.Max(max.V, v);
                });
                var rowMax = w.Max(max.V);
                var inverse = 1f / w.RowSum(cols, j =>
                {
                    var e = k.Exp(y[o + j] - rowMax);
                    y[o + j] = e;
                    return e;
                });
                w.Columns(cols, j => y[o + j] = y[o + j] * inverse);
            });
            return k.Build();
        });

        // y = x · inv per row, inv[r] = 1 / sqrt(mean(x²) + eps).
        yield return (Named("rms_norm"), () =>
        {
            var k = new KernelBuilder(Named("rms_norm"), Block);
            var (x, y, inv) = (k.Buffer("x"), k.Buffer("y"), k.Buffer("inv"));
            var (rows, cols, eps) = (k.PushInt("rows"), k.PushInt("cols"), k.PushFloat("eps"));
            var w = new RowWork(k, narrow);
            w.Each(rows, r =>
            {
                var o = r * cols;
                var sum = w.RowSumOfSquares(cols, j => x[o + j]);
                var scale = 1f / k.Sqrt(sum / cols.ToFloat() + eps);
                w.Leader(() => inv[r] = scale);
                w.Columns(cols, j => y[o + j] = x[o + j] * scale);
            });
            return k.Build();
        });

        // dx += inv[r] · (dy - y · mean(dy · y)).
        yield return (Named("rms_norm_backward"), () =>
        {
            var k = new KernelBuilder(Named("rms_norm_backward"), Block);
            var (dy, y, inv, dx) = (k.Buffer("dy"), k.Buffer("y"), k.Buffer("inv"), k.Buffer("dx"));
            var (rows, cols) = (k.PushInt("rows"), k.PushInt("cols"));
            var w = new RowWork(k, narrow);
            w.Each(rows, r =>
            {
                var o = r * cols;
                var mean = w.RowSum(cols, j => dy[o + j] * y[o + j]) / cols.ToFloat();
                var scale = inv[r];
                w.Columns(cols, j => dx[o + j] = k.AddRounded(dx[o + j], k.MulRounded(scale, dy[o + j] - k.MulRounded(y[o + j], mean))));   // never fused: the same bits at every width
            });
            return k.Build();
        });

        // w = gain[r] · x / ‖x‖ per row (weight normalization along the first axis).
        yield return (Named("weight_norm"), () =>
        {
            var k = new KernelBuilder(Named("weight_norm"), Block);
            var (gain, x, w) = (k.Buffer("gain"), k.Buffer("x"), k.Buffer("w"));
            var (rows, cols) = (k.PushInt("rows"), k.PushInt("cols"));
            var work = new RowWork(k, narrow);
            work.Each(rows, r =>
            {
                var o = r * cols;
                var sum = work.RowSumOfSquares(cols, j => x[o + j]);
                var scale = gain[r] / k.Sqrt(sum);
                work.Columns(cols, j => w[o + j] = x[o + j] * scale);
            });
            return k.Build();
        });

        // With n = ‖x‖ and dot = Σ dw · x per row: dgain[r] += dot / n (flags & 1), dx += gain[r] / n · (dw − x · dot / n²) (flags & 2).
        yield return (Named("weight_norm_backward"), () =>
        {
            var k = new KernelBuilder(Named("weight_norm_backward"), Block);
            var (gain, x, dw, dgain, dx) = (k.Buffer("gain"), k.Buffer("x"), k.Buffer("dw"), k.Buffer("dgain"), k.Buffer("dx"));
            var (rows, cols, flags) = (k.PushInt("rows"), k.PushInt("cols"), k.PushInt("flags"));
            var work = new RowWork(k, narrow);
            work.Each(rows, r =>
            {
                var o = r * cols;
                var sums = work.RowSums(2, cols, j => [x[o + j] * x[o + j], dw[o + j] * x[o + j]]);
                var (sumSquares, dot) = (sums[0], sums[1]);
                var inverse = 1f / k.Sqrt(sumSquares);
                k.If((flags & 1).Ne(0), () => work.Leader(() => dgain[r] = k.AddRounded(dgain[r], k.MulRounded(dot, inverse))));
                k.If((flags & 2).Ne(0), () =>
                {
                    var scale = gain[r] * inverse;
                    var along = scale * dot / sumSquares;
                    work.Columns(cols, j => dx[o + j] = k.AddRounded(dx[o + j], k.MulRounded(dw[o + j], scale) - k.MulRounded(x[o + j], along)));
                });
            });
            return k.Build();
        });

        // y = x · inv · (gain[c] + offset) per row.
        yield return (Named("rms_norm_affine"), () =>
        {
            var k = new KernelBuilder(Named("rms_norm_affine"), Block);
            var (x, gain, y) = (k.Buffer("x"), k.Buffer("gain"), k.Buffer("y"));
            var (rows, cols, eps, offset) = (k.PushInt("rows"), k.PushInt("cols"), k.PushFloat("eps"), k.PushFloat("offset"));
            var w = new RowWork(k, narrow);
            w.Each(rows, r =>
            {
                var o = r * cols;
                var sum = w.RowSumOfSquares(cols, j => x[o + j]);
                var scale = 1f / k.Sqrt(sum / cols.ToFloat() + eps);
                w.Columns(cols, j => y[o + j] = x[o + j] * scale * (gain[j] + offset));
            });
            return k.Build();
        });

        // sum = a + b; y = RMS-normalized sum · (gain[c] + offset) per row.
        yield return (Named("add_rms_norm_affine"), () =>
        {
            var k = new KernelBuilder(Named("add_rms_norm_affine"), Block);
            var (a, b, sum, gain, y) = (k.Buffer("a"), k.Buffer("b"), k.Buffer("sum"), k.Buffer("gain"), k.Buffer("y"));
            var (rows, cols, eps, offset) = (k.PushInt("rows"), k.PushInt("cols"), k.PushFloat("eps"), k.PushFloat("offset"));
            var w = new RowWork(k, narrow);
            w.Each(rows, r =>
            {
                var o = r * cols;
                var squares = w.RowSumOfSquares(cols, j =>
                {
                    var s = a[o + j] + b[o + j];
                    sum[o + j] = s;
                    return s;
                });
                var scale = 1f / k.Sqrt(squares / cols.ToFloat() + eps);
                w.Columns(cols, j => y[o + j] = sum[o + j] * scale * (gain[j] + offset));
            });
            return k.Build();
        });

        // y[r] = index of the largest element of row r (the first one on ties), as a float.
        yield return (Named("arg_max"), () =>
        {
            var k = new KernelBuilder(Named("arg_max"), Block);
            var (x, y) = (k.Buffer("x"), k.Buffer("y"));
            var (rows, cols) = (k.PushInt("rows"), k.PushInt("cols"));
            var w = new RowWork(k, narrow);
            var values = narrow ? null : k.Shared("values", Block);
            var indices = narrow ? null : k.Shared("indices", Block);
            var lane = k.LocalX;
            w.Each(rows, r =>
            {
                var o = r * cols;
                var best = k.Local(float.NegativeInfinity);
                var at = k.Local(k.Int(0x7FFFFFFF));
                w.Columns(cols, j => k.If((x[o + j] > best.V) | at.V.Eq(0x7FFFFFFF), () =>
                {
                    best.V = x[o + j];
                    at.V = j;
                }));
                if (values is null || indices is null)
                {
                    y[r] = k.Max(at.V, k.Int(0)).ToFloat();
                    return;
                }

                values[lane] = best.V;
                indices[lane] = at.V.AsFloat();                              // the index's bits, exact for any width
                k.Barrier();
                for (int stride = Block / 2; stride > 0; stride /= 2)
                {
                    int s = stride;
                    k.If(lane < s, () =>
                    {
                        var (va, vb) = (values[lane], values[lane + s]);
                        var (ia, ib) = (indices[lane].AsInt(), indices[lane + s].AsInt());
                        k.If((vb > va) | (vb.Eq(va) & (ib < ia)) | ia.Eq(0x7FFFFFFF), () =>
                        {
                            values[lane] = vb;
                            indices[lane] = ib.AsFloat();
                        });
                    });
                    k.Barrier();
                }

                k.If(lane.Eq(0), () => y[r] = k.Max(indices[k.Int(0)].AsInt(), k.Int(0)).ToFloat());
                k.Barrier();
            });
            return k.Build();
        });

        // Token cross-entropy per row r (target t_r, weight w_r): losses[r] = w_r · (logsumexp(x_r) - x_r[t_r]); the logits
        // become scale · w_r · (softmax(x_r) - onehot(t_r)).
        yield return (Named("softmax_cross_entropy_rows"), () =>
        {
            var k = new KernelBuilder(Named("softmax_cross_entropy_rows"), Block);
            var (logits, targets, weights, losses) = (k.Buffer("logits"), k.Buffer("targets"), k.Buffer("weights"), k.Buffer("losses"));
            var (rows, vocabulary, scale) = (k.PushInt("rows"), k.PushInt("vocabulary"), k.PushFloat("scale"));
            var w = new RowWork(k, narrow);
            w.Each(rows, r =>
            {
                var o = r * vocabulary;
                var target = targets[r].ToInt();
                var picked = logits[o + target];                             // read before any invocation overwrites it
                var weight = weights[r];
                var max = w.RowMax(vocabulary, j => logits[o + j]);
                var lse = max + k.Log(w.RowSum(vocabulary, j => k.Exp(logits[o + j] - max)));
                w.Leader(() => losses[r] = weight * (lse - picked));
                var g = scale * weight;
                w.Columns(vocabulary, j =>
                    logits[o + j] = g * (k.Exp(logits[o + j] - lse) - k.Select(j.Eq(target), k.Float(1f), k.Float(0f))));
            });
            return k.Build();
        });

        yield return (Named("layer_norm"), () => LayerNorm(Named("layer_norm"), stats: false, narrow));

        // layer_norm that also stores each row's mean (stats[r]) and 1 / sqrt(var + eps) (stats[rows + r]).
        yield return (Named("layer_norm_train"), () => LayerNorm(Named("layer_norm_train"), stats: true, narrow));
    }

    // y = (x - mean) / sqrt(var + eps) · gamma + beta per row (mean first, then the variance about it, as the CPU).
    private static SpirvKernel LayerNorm(string name, bool stats, bool narrow)
    {
        var k = new KernelBuilder(name, Block);
        var (x, gamma, beta, y) = (k.Buffer("x"), k.Buffer("gamma"), k.Buffer("beta"), k.Buffer("y"));
        var statistics = stats ? k.Buffer("stats") : null;
        var (rows, cols, eps) = (k.PushInt("rows"), k.PushInt("cols"), k.PushFloat("eps"));
        var w = new RowWork(k, narrow);
        w.Each(rows, r =>
        {
            var o = r * cols;
            var count = cols.ToFloat();
            var mean = w.RowSum(cols, j => x[o + j]) / count;
            var variance = w.RowSumOfSquares(cols, j => x[o + j] - mean) / count;
            var inv = 1f / k.Sqrt(variance + eps);
            if (statistics is not null)
            {
                w.Leader(() =>
                {
                    statistics[r] = mean;
                    statistics[rows + r] = inv;
                });
            }

            w.Columns(cols, j => y[o + j] = k.AddRounded(k.MulRounded(k.MulRounded(x[o + j] - mean, inv), gamma[j]), beta[j]));
        });
        return k.Build();
    }

    // How a row kernel spreads a row: over a workgroup (lane-strided columns, workgroup reductions), or over one
    // invocation (narrow: every column in turn, no barriers; rows over the whole grid).
    //
    // A row's sums have one order, fixed by its length alone (docs/optimization.md rule 86), so the narrow kernel, a
    // workgroup and any width give the same bits, and the tuning that picks among them changes no result: the columns
    // fall in P lanes (the power of two at or above the row's length, at most MaxWidth), lane l adding columns l, l + P,
    // l + 2P, … in turn from -0 (the one exact identity), each add rounded on its own (AddRounded, or Fma for squares);
    // the lanes then combine in KernelBuilder.Reduce's tree, pairing lane l with l + stride as the strides halve from
    // P / 2. A workgroup keeps an accumulator per lane it covers and folds the strides at or above its width in
    // registers before the reduction; one invocation takes the lanes in bit-reversed order and sums them pairwise (a
    // stack of partial sums), which pairs them exactly as the tree does. The columns' own results round every product on
    // its own too (MulRounded, AddRounded): the driver compiles each width apart and may fuse a product into the add it
    // feeds at one width and not another (one driver's layer-norm gradient differed by one bit between widths 64 and 256).
    private sealed class RowWork(KernelBuilder k, bool narrow)
    {
        private const int LaneBits = 10;                                 // log2 of MaxWidth: at most MaxWidth lanes

        private SharedArray? _scratch;                                   // workgroup reductions' memory, made on first use

        public void Each(Val rows, Action<Val> body)
        {
            if (narrow)
            {
                Grid(k, rows, body);
            }
            else
            {
                EachRow(k, rows, body);
            }
        }

        // This invocation's columns of the row: all of them (narrow), or j = lane, lane + width, … (writes, and maxima,
        // whose order does not change them; sums go through RowSum).
        public void Columns(Val cols, Action<Val> body)
        {
            if (narrow)
            {
                k.For(k.Int(0), cols, 1, body);
            }
            else
            {
                k.For(k.LocalX, cols, Block, body);
            }
        }

        public Val Max(Val partial) => narrow ? partial : k.ReduceMax(_scratch ??= k.Shared("scratch", Block), partial);

        // Code one invocation of the row runs (a per-row result).
        public void Leader(Action body)
        {
            if (narrow)
            {
                body();
            }
            else
            {
                k.If(k.LocalX.Eq(0), body);
            }
        }

        // Σ_j term(j) over the row in the row's one order, returned to every invocation of it. term runs once for each
        // column (it may write the column too) and returns -0 for a column it leaves out.
        public Val RowSum(Val cols, Func<Val, Val> term) => Accumulate(1, cols, (acc, j) => acc[0].V = k.AddRounded(acc[0].V, term(j)))[0];

        // Several sums over the row in one pass, each in the row's one order (as RowSum).
        public Val[] RowSums(int count, Val cols, Func<Val, Val[]> terms) => Accumulate(count, cols, (acc, j) =>
        {
            var t = terms(j);
            for (int i = 0; i < count; i++)
            {
                acc[i].V = k.AddRounded(acc[i].V, t[i]);
            }
        });

        // Σ_j value(j)² over the row in the row's one order (each square fused into its add: Fma), returned to every
        // invocation of it. value runs once for each column (it may write the column too).
        public Val RowSumOfSquares(Val cols, Func<Val, Val> value) => Accumulate(1, cols, (acc, j) =>
        {
            var v = value(j);
            acc[0].V = k.Fma(v, v, acc[0].V);
        })[0];

        // The sum of each invocation's own partial (a statistic its kernel accumulates in its Columns loop: the order then
        // follows the variant; a value any result depends on goes through RowSum).
        public Val Sum(Val partial) => narrow ? partial : k.ReduceSum(_scratch ??= k.Shared("scratch", Block), partial);

        // max_j term(j) over the row, returned to every invocation of it.
        public Val RowMax(Val cols, Func<Val, Val> term)
        {
            var acc = k.Local(float.NegativeInfinity);
            Columns(cols, j => acc.V = k.Max(acc.V, term(j)));
            return Max(acc.V);
        }

        // `count` sums over the row in its one order: `add` adds column j to the accumulators (from -0).
        private Val[] Accumulate(int count, Val cols, Action<Var[], Val> add)
        {
            var bits = k.Min(k.FindMsb(k.Max(cols, k.Int(1)) - 1) + 1, k.Int(LaneBits));
            var lanes = k.Int(1).ShiftLeft(bits);
            if (narrow)
            {
                var stack = k.LocalArray((LaneBits + 1) * count);
                var depth = k.Local(k.Int(0));
                k.For(k.Int(0), lanes, 1, position =>
                {
                    var lane = k.Select(bits.Eq(0), k.Int(0), k.BitReverse(position).AsUInt().ShiftRight((32 - bits).AsUInt()).AsInt());
                    var partial = Enumerable.Range(0, count).Select(_ => k.Local(k.Float(-0f))).ToArray();
                    k.For(lane, cols, j => add(partial, j), lanes);
                    var merging = k.Local(k.Bool(true));
                    for (int bit = 0; bit < LaneBits; bit++)
                    {
                        int b = bit;
                        k.If(merging.V & ((position >> b) & 1).Eq(1), () =>
                        {
                            depth.V = depth.V - 1;
                            for (int i = 0; i < count; i++)
                            {
                                partial[i].V = stack[depth.V * count + i] + partial[i].V;   // the earlier half first, as the tree adds
                            }
                        }, () => merging.V = k.Bool(false));
                    }

                    for (int i = 0; i < count; i++)
                    {
                        stack[depth.V * count + i] = partial[i].V;
                    }

                    depth.V = depth.V + 1;
                });
                return [.. Enumerable.Range(0, count).Select(i => stack[k.Int(i)])];
            }

            int each = MaxWidth / Block;                                 // lanes an invocation covers, at most
            var acc = Enumerable.Range(0, each).Select(_ => Enumerable.Range(0, count).Select(_ => k.Local(k.Float(-0f))).ToArray()).ToArray();
            k.For(k.Int(0), cols, start =>
            {
                for (int m = 0; m < each; m++)
                {
                    var lane = k.LocalX + m * Block;
                    var j = start + lane;
                    var mine = acc[m];
                    k.If((lane < lanes) & (j < cols), () => add(mine, j));
                }
            }, lanes);
            for (int stride = MaxWidth / 2; stride >= Block; stride /= 2)
            {
                int pairs = stride / Block;
                k.If(lanes > stride, () =>
                {
                    for (int m = 0; m < pairs; m++)
                    {
                        for (int i = 0; i < count; i++)
                        {
                            acc[m][i].V = acc[m][i].V + acc[m + pairs][i].V;
                        }
                    }
                });
            }

            var scratch = _scratch ??= k.Shared("scratch", Block);
            return [.. Enumerable.Range(0, count).Select(i => k.ReduceSum(scratch, acc[0][i].V))];
        }
    }
}
