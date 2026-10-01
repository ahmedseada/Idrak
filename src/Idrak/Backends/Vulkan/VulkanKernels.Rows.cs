// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Backends.Vulkan;

// Row kernels: one workgroup of 256 per row (rows looped with a stride of the group count), each invocation taking the
// columns j ≡ its lane; per-row sums and maxima through workgroup reductions. An invocation always reads and writes the
// same columns in every pass, so the output may be the input (in place).
internal static partial class VulkanKernels
{
    private static IEnumerable<(string, Func<SpirvKernel>)> RowKernels()
    {
        // y = softmax(x) per row, or log-softmax with log ≠ 0.
        yield return ("softmax", () =>
        {
            var k = new KernelBuilder("softmax", Block);
            var (x, y) = (k.Buffer("x"), k.Buffer("y"));
            var (rows, cols, log) = (k.PushInt("rows"), k.PushInt("cols"), k.PushInt("log"));
            var scratch = k.Shared("scratch", Block);
            EachRow(k, rows, r =>
            {
                var o = r * cols;
                var max = RowMax(k, scratch, cols, j => x[o + j]);
                var sum = k.Local(0f);
                k.If(log.Ne(0), () =>
                {
                    Columns(k, cols, j => sum.V = sum.V + k.Exp(x[o + j] - max));
                    var logSum = k.Log(k.ReduceSum(scratch, sum.V)) + max;
                    Columns(k, cols, j => y[o + j] = x[o + j] - logSum);
                }, () =>
                {
                    Columns(k, cols, j =>
                    {
                        var e = k.Exp(x[o + j] - max);
                        y[o + j] = e;
                        sum.V = sum.V + e;
                    });
                    var inverse = 1f / k.ReduceSum(scratch, sum.V);
                    Columns(k, cols, j => y[o + j] = y[o + j] * inverse);
                });
            });
            return k.Build();
        });

        // Softmax: dx += y · (dy - Σ dy·y). Log-softmax: dx += dy - exp(y) · Σ dy.
        yield return ("softmax_backward", () =>
        {
            var k = new KernelBuilder("softmax_backward", Block);
            var (y, dy, dx) = (k.Buffer("y"), k.Buffer("dy"), k.Buffer("dx"));
            var (rows, cols, log) = (k.PushInt("rows"), k.PushInt("cols"), k.PushInt("log"));
            var scratch = k.Shared("scratch", Block);
            EachRow(k, rows, r =>
            {
                var o = r * cols;
                var isLog = log.Ne(0);
                var s = RowSum(k, scratch, cols, j => k.Select(isLog, dy[o + j], dy[o + j] * y[o + j]));
                Columns(k, cols, j =>
                {
                    var g = dy[o + j];
                    dx[o + j] = dx[o + j] + k.Select(isLog, g - k.Exp(y[o + j]) * s, y[o + j] * (g - s));
                });
            });
            return k.Build();
        });

        // y[r, :] = softmax(scale · x[r, :] + mask[r % maskRows, :]); hasMask = 0: no mask (bind any storage there).
        yield return ("scale_mask_softmax", () =>
        {
            var k = new KernelBuilder("scale_mask_softmax", Block);
            var (x, mask, y) = (k.Buffer("x"), k.Buffer("mask"), k.Buffer("y"));
            var (rows, cols, maskRows, scale, hasMask) = (k.PushInt("rows"), k.PushInt("cols"), k.PushInt("maskRows"), k.PushFloat("scale"), k.PushInt("hasMask"));
            var scratch = k.Shared("scratch", Block);
            EachRow(k, rows, r =>
            {
                var o = r * cols;
                var mo = r % k.Max(maskRows, k.Int(1)) * cols;
                var masked = hasMask.Ne(0);
                var max = k.Local(float.NegativeInfinity);
                Columns(k, cols, j =>
                {
                    var add = k.Local(0f);
                    k.If(masked, () => add.V = mask[mo + j]);
                    var v = x[o + j] * scale + add.V;
                    y[o + j] = v;
                    max.V = k.Max(max.V, v);
                });
                var rowMax = k.ReduceMax(scratch, max.V);
                var sum = k.Local(0f);
                Columns(k, cols, j =>
                {
                    var e = k.Exp(y[o + j] - rowMax);
                    y[o + j] = e;
                    sum.V = sum.V + e;
                });
                var inverse = 1f / k.ReduceSum(scratch, sum.V);
                Columns(k, cols, j => y[o + j] = y[o + j] * inverse);
            });
            return k.Build();
        });

        // y = x · inv per row, inv[r] = 1 / sqrt(mean(x²) + eps).
        yield return ("rms_norm", () =>
        {
            var k = new KernelBuilder("rms_norm", Block);
            var (x, y, inv) = (k.Buffer("x"), k.Buffer("y"), k.Buffer("inv"));
            var (rows, cols, eps) = (k.PushInt("rows"), k.PushInt("cols"), k.PushFloat("eps"));
            var scratch = k.Shared("scratch", Block);
            EachRow(k, rows, r =>
            {
                var o = r * cols;
                var sum = RowSum(k, scratch, cols, j => x[o + j] * x[o + j]);
                var scale = 1f / k.Sqrt(sum / cols.ToFloat() + eps);
                k.If(k.LocalX.Eq(0), () => inv[r] = scale);
                Columns(k, cols, j => y[o + j] = x[o + j] * scale);
            });
            return k.Build();
        });

        // dx += inv[r] · (dy - y · mean(dy · y)).
        yield return ("rms_norm_backward", () =>
        {
            var k = new KernelBuilder("rms_norm_backward", Block);
            var (dy, y, inv, dx) = (k.Buffer("dy"), k.Buffer("y"), k.Buffer("inv"), k.Buffer("dx"));
            var (rows, cols) = (k.PushInt("rows"), k.PushInt("cols"));
            var scratch = k.Shared("scratch", Block);
            EachRow(k, rows, r =>
            {
                var o = r * cols;
                var mean = RowSum(k, scratch, cols, j => dy[o + j] * y[o + j]) / cols.ToFloat();
                var scale = inv[r];
                Columns(k, cols, j => dx[o + j] = dx[o + j] + scale * (dy[o + j] - y[o + j] * mean));
            });
            return k.Build();
        });

        // y = x · inv · (gain[c] + offset) per row.
        yield return ("rms_norm_affine", () =>
        {
            var k = new KernelBuilder("rms_norm_affine", Block);
            var (x, gain, y) = (k.Buffer("x"), k.Buffer("gain"), k.Buffer("y"));
            var (rows, cols, eps, offset) = (k.PushInt("rows"), k.PushInt("cols"), k.PushFloat("eps"), k.PushFloat("offset"));
            var scratch = k.Shared("scratch", Block);
            EachRow(k, rows, r =>
            {
                var o = r * cols;
                var sum = RowSum(k, scratch, cols, j => x[o + j] * x[o + j]);
                var scale = 1f / k.Sqrt(sum / cols.ToFloat() + eps);
                Columns(k, cols, j => y[o + j] = x[o + j] * scale * (gain[j] + offset));
            });
            return k.Build();
        });

        // sum = a + b; y = RMS-normalized sum · (gain[c] + offset) per row.
        yield return ("add_rms_norm_affine", () =>
        {
            var k = new KernelBuilder("add_rms_norm_affine", Block);
            var (a, b, sum, gain, y) = (k.Buffer("a"), k.Buffer("b"), k.Buffer("sum"), k.Buffer("gain"), k.Buffer("y"));
            var (rows, cols, eps, offset) = (k.PushInt("rows"), k.PushInt("cols"), k.PushFloat("eps"), k.PushFloat("offset"));
            var scratch = k.Shared("scratch", Block);
            EachRow(k, rows, r =>
            {
                var o = r * cols;
                var squares = k.Local(0f);
                Columns(k, cols, j =>
                {
                    var s = a[o + j] + b[o + j];
                    sum[o + j] = s;
                    squares.V = k.Fma(s, s, squares.V);
                });
                var scale = 1f / k.Sqrt(k.ReduceSum(scratch, squares.V) / cols.ToFloat() + eps);
                Columns(k, cols, j => y[o + j] = sum[o + j] * scale * (gain[j] + offset));
            });
            return k.Build();
        });

        // y[r] = index of the largest element of row r (the first one on ties), as a float.
        yield return ("arg_max", () =>
        {
            var k = new KernelBuilder("arg_max", Block);
            var (x, y) = (k.Buffer("x"), k.Buffer("y"));
            var (rows, cols) = (k.PushInt("rows"), k.PushInt("cols"));
            var values = k.Shared("values", Block);
            var indices = k.Shared("indices", Block);
            var lane = k.LocalX;
            EachRow(k, rows, r =>
            {
                var o = r * cols;
                var best = k.Local(float.NegativeInfinity);
                var at = k.Local(k.Int(0x7FFFFFFF));
                Columns(k, cols, j => k.If((x[o + j] > best.V) | at.V.Eq(0x7FFFFFFF), () =>
                {
                    best.V = x[o + j];
                    at.V = j;
                }));
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
        yield return ("softmax_cross_entropy_rows", () =>
        {
            var k = new KernelBuilder("softmax_cross_entropy_rows", Block);
            var (logits, targets, weights, losses) = (k.Buffer("logits"), k.Buffer("targets"), k.Buffer("weights"), k.Buffer("losses"));
            var (rows, vocabulary, scale) = (k.PushInt("rows"), k.PushInt("vocabulary"), k.PushFloat("scale"));
            var scratch = k.Shared("scratch", Block);
            EachRow(k, rows, r =>
            {
                var o = r * vocabulary;
                var target = targets[r].ToInt();
                var picked = logits[o + target];                             // read before any invocation overwrites it
                var weight = weights[r];
                var max = RowMax(k, scratch, vocabulary, j => logits[o + j]);
                var lse = max + k.Log(RowSum(k, scratch, vocabulary, j => k.Exp(logits[o + j] - max)));
                k.If(k.LocalX.Eq(0), () => losses[r] = weight * (lse - picked));
                var g = scale * weight;
                Columns(k, vocabulary, j =>
                    logits[o + j] = g * (k.Exp(logits[o + j] - lse) - k.Select(j.Eq(target), k.Float(1f), k.Float(0f))));
            });
            return k.Build();
        });

        yield return ("layer_norm", () => LayerNorm("layer_norm", stats: false));

        // layer_norm that also stores each row's mean (stats[r]) and 1 / sqrt(var + eps) (stats[rows + r]).
        yield return ("layer_norm_train", () => LayerNorm("layer_norm_train", stats: true));
    }

    // y = (x - mean) / sqrt(var + eps) · gamma + beta per row (mean first, then the variance about it, as the CPU).
    private static SpirvKernel LayerNorm(string name, bool stats)
    {
        var k = new KernelBuilder(name, Block);
        var (x, gamma, beta, y) = (k.Buffer("x"), k.Buffer("gamma"), k.Buffer("beta"), k.Buffer("y"));
        var statistics = stats ? k.Buffer("stats") : null;
        var (rows, cols, eps) = (k.PushInt("rows"), k.PushInt("cols"), k.PushFloat("eps"));
        var scratch = k.Shared("scratch", Block);
        EachRow(k, rows, r =>
        {
            var o = r * cols;
            var count = cols.ToFloat();
            var mean = RowSum(k, scratch, cols, j => x[o + j]) / count;
            var variance = RowSum(k, scratch, cols, j => (x[o + j] - mean) * (x[o + j] - mean)) / count;
            var inv = 1f / k.Sqrt(variance + eps);
            if (statistics is not null)
            {
                k.If(k.LocalX.Eq(0), () =>
                {
                    statistics[r] = mean;
                    statistics[rows + r] = inv;
                });
            }

            Columns(k, cols, j => y[o + j] = (x[o + j] - mean) * inv * gamma[j] + beta[j]);
        });
        return k.Build();
    }

    // Each invocation's columns of the row: j = lane, lane + 256, …
    private static void Columns(KernelBuilder k, Val cols, Action<Val> body) => k.For(k.LocalX, cols, Block, body);

    // Σ_j term(j) over the row, returned to every invocation.
    private static Val RowSum(KernelBuilder k, SharedArray scratch, Val cols, Func<Val, Val> term)
    {
        var acc = k.Local(0f);
        Columns(k, cols, j => acc.V = acc.V + term(j));
        return k.ReduceSum(scratch, acc.V);
    }

    // max_j term(j) over the row, returned to every invocation.
    private static Val RowMax(KernelBuilder k, SharedArray scratch, Val cols, Func<Val, Val> term)
    {
        var acc = k.Local(float.NegativeInfinity);
        Columns(k, cols, j => acc.V = k.Max(acc.V, term(j)));
        return k.ReduceMax(scratch, acc.V);
    }
}
