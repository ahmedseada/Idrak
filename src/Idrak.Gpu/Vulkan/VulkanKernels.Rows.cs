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
                var sum = k.Local(0f);
                k.If(log.Ne(0), () =>
                {
                    w.Columns(cols, j => sum.V = sum.V + k.Exp(x[o + j] - max));
                    var logSum = k.Log(w.Sum(sum.V)) + max;
                    w.Columns(cols, j => y[o + j] = x[o + j] - logSum);
                }, () =>
                {
                    w.Columns(cols, j =>
                    {
                        var e = k.Exp(x[o + j] - max);
                        y[o + j] = e;
                        sum.V = sum.V + e;
                    });
                    var inverse = 1f / w.Sum(sum.V);
                    w.Columns(cols, j => y[o + j] = y[o + j] * inverse);
                });
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
                    dx[o + j] = dx[o + j] + k.Select(isLog, g - k.Exp(y[o + j]) * s, y[o + j] * (g - s));
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
                    var v = x[o + j] * scale + add.V;
                    y[o + j] = v;
                    max.V = k.Max(max.V, v);
                });
                var rowMax = w.Max(max.V);
                var sum = k.Local(0f);
                w.Columns(cols, j =>
                {
                    var e = k.Exp(y[o + j] - rowMax);
                    y[o + j] = e;
                    sum.V = sum.V + e;
                });
                var inverse = 1f / w.Sum(sum.V);
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
                var sum = w.RowSum(cols, j => x[o + j] * x[o + j]);
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
                w.Columns(cols, j => dx[o + j] = dx[o + j] + scale * (dy[o + j] - y[o + j] * mean));
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
                var sum = w.RowSum(cols, j => x[o + j] * x[o + j]);
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
                var squares = k.Local(0f);
                w.Columns(cols, j =>
                {
                    var s = a[o + j] + b[o + j];
                    sum[o + j] = s;
                    squares.V = k.Fma(s, s, squares.V);
                });
                var scale = 1f / k.Sqrt(w.Sum(squares.V) / cols.ToFloat() + eps);
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
            var variance = w.RowSum(cols, j => (x[o + j] - mean) * (x[o + j] - mean)) / count;
            var inv = 1f / k.Sqrt(variance + eps);
            if (statistics is not null)
            {
                w.Leader(() =>
                {
                    statistics[r] = mean;
                    statistics[rows + r] = inv;
                });
            }

            w.Columns(cols, j => y[o + j] = (x[o + j] - mean) * inv * gamma[j] + beta[j]);
        });
        return k.Build();
    }

    // How a row kernel spreads a row: over a workgroup (lane-strided columns, workgroup reductions), or over one
    // invocation (narrow: every column in turn, no barriers; rows over the whole grid).
    private sealed class RowWork(KernelBuilder k, bool narrow)
    {
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

        // This invocation's columns of the row: all of them (narrow), or j = lane, lane + width, …
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

        public Val Sum(Val partial) => narrow ? partial : k.ReduceSum(_scratch ??= k.Shared("scratch", Block), partial);

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

        // Σ_j term(j) over the row, returned to every invocation of it.
        public Val RowSum(Val cols, Func<Val, Val> term)
        {
            var acc = k.Local(0f);
            Columns(cols, j => acc.V = acc.V + term(j));
            return Sum(acc.V);
        }

        // max_j term(j) over the row, returned to every invocation of it.
        public Val RowMax(Val cols, Func<Val, Val> term)
        {
            var acc = k.Local(float.NegativeInfinity);
            Columns(cols, j => acc.V = k.Max(acc.V, term(j)));
            return Max(acc.V);
        }
    }
}
