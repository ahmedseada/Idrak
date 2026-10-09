// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Idrak.Abstraction.Devices.Cpu;

// Kernels for classification, normalization, embeddings, convolution, pooling and N-D shape operations.
internal sealed partial class CpuBackend
{
    /// <summary>
    /// Runs body(start, end) over [0, count), splitting across cores when work (≈ element operations) passes the measured
    /// cut-over. A body whose chunks are summed together (<paramref name="chunksSummed"/>) splits by the numerical
    /// contract instead (<see cref="CpuTuning.ReductionElements"/>), so a timing never changes its result.
    /// </summary>
    private static void For(int count, long work, Action<int, int> body, bool chunksSummed = false)
    {
        bool split = chunksSummed ? work >= CpuTuning.ReductionElements && ComputeResources.AllowParallel : CpuTuning.SplitElements(work);
        if (count <= 1 || !split)
        {
            body(0, count);
            return;
        }

        int chunks = Math.Min(count, ComputeResources.MaxCpuThreads * 4);
        int size = (count + chunks - 1) / chunks;
        Parallel.For(0, chunks, ComputeResources.ParallelOptions, c =>
        {
            int start = c * size;
            if (start < count)
            {
                body(start, Math.Min(start + size, count));
            }
        });
    }

    public override void BatchedMatMulKernel(Storage a, Storage b, Storage c, int batch, int m, int n, int k, bool transA, bool transB, float beta)
    {
        float[] av = D(a), bv = D(b), cv = D(c);
        if (batch == 1)
        {
            CpuMatMul.Multiply(av, bv, cv, m, n, k, transA, transB, beta);
            return;
        }

        // Many small products (attention heads): parallelize across the batch, each product single-threaded.
        int mk = m * k, kn = k * n, mn = m * n;
        For(batch, (long)batch * m * n * k, (start, end) =>
        {
            for (int i = start; i < end; i++)
            {
                CpuMatMul.Multiply(av, i * mk, bv, i * kn, cv, i * mn, m, n, k, transA, transB, beta);
            }
        });
    }

    public override void SoftmaxKernel(Storage x, Storage y, int rows, int cols, bool log)
    {
        float[] xv = D(x), yv = D(y);
        For(rows, (long)rows * cols * 8, (start, end) =>
        {
            for (int r = start; r < end; r++)
            {
                var xs = xv.AsSpan(r * cols, cols);
                var ys = yv.AsSpan(r * cols, cols);
                float max = CpuMath.Max(xs);
                if (log && xs.Overlaps(ys))
                {
                    // In place: the exponentials go to scratch, so the log-probabilities still read the inputs.
                    var scratch = System.Buffers.ArrayPool<float>.Shared.Rent(cols);
                    xs.CopyTo(scratch);
                    float inPlace = (float)Math.Log(CpuMath.ExpShifted(scratch.AsSpan(0, cols), max)) + max;
                    System.Buffers.ArrayPool<float>.Shared.Return(scratch);
                    for (int j = 0; j < cols; j++)
                    {
                        ys[j] = xs[j] - inPlace;
                    }

                    continue;
                }

                xs.CopyTo(ys);
                double sum = CpuMath.ExpShifted(ys, max);

                if (log)
                {
                    float logSum = (float)Math.Log(sum) + max;
                    for (int j = 0; j < cols; j++)
                    {
                        ys[j] = xs[j] - logSum;
                    }
                }
                else
                {
                    CpuMath.Scale(ys, (float)(1.0 / sum));
                }
            }
        });
    }

    public override void SoftmaxBackwardKernel(Storage y, Storage dy, Storage dx, int rows, int cols, bool log)
    {
        float[] yv = D(y), gv = D(dy), dv = D(dx);
        For(rows, (long)rows * cols * 4, (start, end) =>
        {
            for (int r = start; r < end; r++)
            {
                int o = r * cols;
                double dot = 0;
                for (int j = 0; j < cols; j++)
                {
                    dot += log ? gv[o + j] : gv[o + j] * yv[o + j];
                }

                float s = (float)dot;
                for (int j = 0; j < cols; j++)
                {
                    dv[o + j] += log ? gv[o + j] - MathF.Exp(yv[o + j]) * s : yv[o + j] * (gv[o + j] - s);
                }
            }
        });
    }

    private static int ArgMaxRow(ReadOnlySpan<float> row)
    {
        int best = 0;
        for (int j = 1; j < row.Length; j++)
        {
            if (row[j] > row[best])
            {
                best = j;
            }
        }

        return best;
    }

    public override void ArgMaxKernel(Storage x, Storage y, int rows, int cols)
    {
        float[] xv = D(x), yv = D(y);
        for (int r = 0; r < rows; r++)
        {
            yv[r] = ArgMaxRow(xv.AsSpan(r * cols, cols));
        }
    }

    public override void ClassMatchKernel(Storage predictions, Storage targets, Storage y, int rows, int cols, float threshold)
    {
        float[] pv = D(predictions), tv = D(targets), yv = D(y);
        for (int r = 0; r < rows; r++)
        {
            bool match = cols == 1
                ? (pv[r] >= threshold) == (tv[r] >= 0.5f)
                : ArgMaxRow(pv.AsSpan(r * cols, cols)) == ArgMaxRow(tv.AsSpan(r * cols, cols));
            yv[r] = match ? 1f : 0f;
        }
    }

    public override void NormStatsKernel(Storage x, Storage mean, Storage variance, Storage invStd, int outer, int groups, int inner, float eps)
    {
        // Two passes: the mean, then the sum of squares around it. E[x²] − mean² in one pass cancels when the values are
        // large next to their spread (a one-element group lost 0.2% of 1/√eps).
        float[] xv = D(x);
        var s1 = new double[groups];
        var s2 = new double[groups];
        GroupMoments(xv, null, outer, groups, inner, s1, s2);
        float[] mv = D(mean), vv = D(variance), sv = D(invStd);
        double m = Math.Max(outer * inner, 1);
        For(groups, (long)groups * outer * inner * 2, (start, end) =>
        {
            for (int g = start; g < end; g++)
            {
                double mu = s1[g] / m, sum = 0;
                for (int o = 0; o < outer; o++)
                {
                    int offset = (o * groups + g) * inner;
                    for (int i = 0; i < inner; i++)
                    {
                        double d = xv[offset + i] - mu;
                        sum += d * d;
                    }
                }

                s2[g] = sum;
            }
        });

        for (int g = 0; g < groups; g++)
        {
            double mu = s1[g] / m;
            double var = s2[g] / m;
            mv[g] = (float)mu;
            vv[g] = (float)var;
            sv[g] = (float)(1.0 / Math.Sqrt(var + eps));
        }
    }

    /// <summary>
    /// s1[g] += Σ a and s2[g] += Σ a·(b ?? a) over each group of the [outer, groups, inner] view, SIMD-vectorized.
    /// Long contiguous runs (inner ≥ vector width) are summed per group; short ones (e.g. inner = 1, where a
    /// group is a column) are accumulated row by row across all groups at once.
    /// </summary>
    private static void GroupMoments(float[] a, float[]? b, int outer, int groups, int inner, double[] s1, double[] s2)
    {
        int w = Vector<float>.Count;
        if (inner >= w)
        {
            For(groups, (long)groups * outer * inner * 2, (start, end) =>
            {
                for (int g = start; g < end; g++)
                {
                    double t1 = 0, t2 = 0;
                    for (int o = 0; o < outer; o++)
                    {
                        int offset = (o * groups + g) * inner;
                        var (p1, p2) = Moments(a.AsSpan(offset, inner), b is null ? a.AsSpan(offset, inner) : b.AsSpan(offset, inner));
                        t1 += p1;
                        t2 += p2;
                    }

                    s1[g] += t1;
                    s2[g] += t2;
                }
            });
            return;
        }

        int row = groups * inner;
        var gate = new Lock();
        For(outer, (long)outer * row * 2, (start, end) =>
        {
            var acc1 = new float[row];
            var acc2 = new float[row];
            for (int o = start; o < end; o++)
            {
                var ra = a.AsSpan(o * row, row);
                var rb = b is null ? ra : b.AsSpan(o * row, row);
                AddInPlace(acc1, ra);
                MultiplyAddInPlace(acc2, ra, rb);
            }

            lock (gate)
            {
                for (int i = 0; i < row; i++)
                {
                    s1[i / inner] += acc1[i];
                    s2[i / inner] += acc2[i];
                }
            }
        }, chunksSummed: true);
    }

    private static (double Sum, double SumProduct) Moments(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        var av = MemoryMarshal.Cast<float, Vector<float>>(a);
        var bv = MemoryMarshal.Cast<float, Vector<float>>(b);
        Vector<float> v1 = default, v2 = default;
        for (int i = 0; i < av.Length; i++)
        {
            v1 += av[i];
            v2 = Vector.FusedMultiplyAdd(av[i], bv[i], v2);
        }

        double t1 = Vector.Sum(v1), t2 = Vector.Sum(v2);
        for (int i = av.Length * Vector<float>.Count; i < a.Length; i++)
        {
            t1 += a[i];
            t2 += a[i] * b[i];
        }

        return (t1, t2);
    }

    private static void MultiplyAddInPlace(Span<float> target, ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        var tv = MemoryMarshal.Cast<float, Vector<float>>(target);
        var av = MemoryMarshal.Cast<float, Vector<float>>(a);
        var bv = MemoryMarshal.Cast<float, Vector<float>>(b);
        for (int i = 0; i < tv.Length; i++)
        {
            tv[i] = Vector.FusedMultiplyAdd(av[i], bv[i], tv[i]);
        }

        for (int i = tv.Length * Vector<float>.Count; i < target.Length; i++)
        {
            target[i] += a[i] * b[i];
        }
    }

    /// <summary>
    /// Applies y (+)= x * scale + shift where scale/shift are per group of the [outer, groups, inner] view.
    /// For inner = 1 the per-group values form a contiguous vector matching each row; otherwise they are
    /// broadcast over each contiguous inner run.
    /// </summary>
    private static void GroupAffineCore(float[] x, float[] y, int n, int groups, int inner, Func<int, float> scaleOf, Func<int, float> shiftOf,
        float[]? scaleRow, float[]? shiftRow, bool accumulate)
    {
        int w = Vector<float>.Count;
        if (inner == 1 && groups >= w && scaleRow is not null && shiftRow is not null)
        {
            int rows = n / groups;
            For(rows, n, (start, end) =>
            {
                var sv = MemoryMarshal.Cast<float, Vector<float>>(scaleRow.AsSpan(0, groups));
                var hv = MemoryMarshal.Cast<float, Vector<float>>(shiftRow.AsSpan(0, groups));
                for (int r = start; r < end; r++)
                {
                    var xs = x.AsSpan(r * groups, groups);
                    var ys = y.AsSpan(r * groups, groups);
                    var xv = MemoryMarshal.Cast<float, Vector<float>>(xs);
                    var yv = MemoryMarshal.Cast<float, Vector<float>>(ys);
                    for (int i = 0; i < xv.Length; i++)
                    {
                        var v = Vector.FusedMultiplyAdd(xv[i], sv[i], hv[i]);
                        yv[i] = accumulate ? yv[i] + v : v;
                    }

                    for (int i = xv.Length * w; i < groups; i++)
                    {
                        float v = xs[i] * scaleRow[i] + shiftRow[i];
                        ys[i] = accumulate ? ys[i] + v : v;
                    }
                }
            });
            return;
        }

        int blocks = n / inner;
        For(blocks, n, (start, end) =>
        {
            for (int blk = start; blk < end; blk++)
            {
                int g = blk % groups;
                float sc = scaleOf(g), sh = shiftOf(g);
                var xs = x.AsSpan(blk * inner, inner);
                var ys = y.AsSpan(blk * inner, inner);
                var xv = MemoryMarshal.Cast<float, Vector<float>>(xs);
                var yv = MemoryMarshal.Cast<float, Vector<float>>(ys);
                var vs = new Vector<float>(sc);
                var vh = new Vector<float>(sh);
                for (int i = 0; i < xv.Length; i++)
                {
                    var v = Vector.FusedMultiplyAdd(xv[i], vs, vh);
                    yv[i] = accumulate ? yv[i] + v : v;
                }

                for (int i = xv.Length * w; i < inner; i++)
                {
                    float v = xs[i] * sc + sh;
                    ys[i] = accumulate ? ys[i] + v : v;
                }
            }
        });
    }


    public override void NormApplyKernel(Storage x, Storage mean, Storage invStd, Storage y, int outer, int groups, int inner)
    {
        // (x - mean) * invStd == x * invStd + (-mean * invStd)
        float[] mv = D(mean), sv = D(invStd);
        var scale = sv.AsSpan(0, groups).ToArray();
        var shift = new float[groups];
        for (int g = 0; g < groups; g++)
        {
            shift[g] = -mv[g] * sv[g];
        }

        GroupAffineCore(D(x), D(y), outer * groups * inner, groups, inner, g => scale[g], g => shift[g], scale, shift, accumulate: false);
    }


    public override void NormBackwardKernel(Storage dxhat, Storage xhat, Storage sum1, Storage sum2, Storage invStd, Storage dx, int outer, int groups, int inner)
    {
        // dx += invStd/M * (M*dxhat - s1 - xhat*s2): two fused affine passes, dx += dxhat*invStd, then dx += xhat*(-invStd*s2/M) - invStd*s1/M.
        float[] s1 = D(sum1), s2 = D(sum2), sv = D(invStd);
        float m = outer * inner;
        var scale1 = sv.AsSpan(0, groups).ToArray();
        var zero = new float[groups];
        var scale2 = new float[groups];
        var shift2 = new float[groups];
        for (int g = 0; g < groups; g++)
        {
            scale2[g] = -sv[g] * s2[g] / m;
            shift2[g] = -sv[g] * s1[g] / m;
        }

        int n = outer * groups * inner;
        GroupAffineCore(D(dxhat), D(dx), n, groups, inner, g => scale1[g], _ => 0f, scale1, zero, accumulate: true);
        GroupAffineCore(D(xhat), D(dx), n, groups, inner, g => scale2[g], g => shift2[g], scale2, shift2, accumulate: true);
    }


    public override void GroupScaleShiftKernel(Storage x, Storage? scale, Storage? shift, Storage y, int n, int groups, int inner, bool accumulate)
    {
        var sc = scale is null ? Enumerable.Repeat(1f, groups).ToArray() : D(scale).AsSpan(0, groups).ToArray();
        var sh = shift is null ? new float[groups] : D(shift).AsSpan(0, groups).ToArray();
        GroupAffineCore(D(x), D(y), n, groups, inner, g => sc[g], g => sh[g], sc, sh, accumulate);
    }


    public override void GroupReduceKernel(Storage a, Storage? b, Storage sumA, Storage? sumAB, int outer, int groups, int inner)
    {
        var s1 = new double[groups];
        var s2 = new double[groups];
        GroupMoments(D(a), b is null ? null : D(b), outer, groups, inner, s1, s2);
        float[] r1 = D(sumA);
        float[]? r2 = sumAB is null ? null : D(sumAB);
        for (int g = 0; g < groups; g++)
        {
            r1[g] += (float)s1[g];
            if (r2 is not null)
            {
                r2[g] += (float)s2[g];
            }
        }
    }


    public override void InvSqrtKernel(Storage x, Storage y, int n, float eps)
    {
        float[] xv = D(x), yv = D(y);
        for (int i = 0; i < n; i++)
        {
            yv[i] = 1f / MathF.Sqrt(xv[i] + eps);
        }
    }

    public override void GatherKernel(Storage table, Storage indices, Storage y, int count, int dim, int vocabulary)
    {
        float[] tv = D(table), iv = D(indices), yv = D(y);
        for (int i = 0; i < count; i++)
        {
            int index = CheckIndex(iv[i], vocabulary);
            tv.AsSpan(index * dim, dim).CopyTo(yv.AsSpan(i * dim, dim));
        }
    }

    public override void OneHotKernel(Storage indices, Storage y, int count, int classes)
    {
        float[] iv = D(indices), yv = D(y);
        yv.AsSpan(0, count * classes).Clear();
        for (int i = 0; i < count; i++)
        {
            yv[i * classes + CheckIndex(iv[i], classes)] = 1f;
        }
    }

    public override void GatherBFloat16Kernel(Storage packed, Storage indices, Storage y, int count, int dim, int vocabulary)
    {
        var halves = System.Runtime.InteropServices.MemoryMarshal.Cast<float, ushort>(D(packed).AsSpan());
        float[] iv = D(indices), yv = D(y);
        int stride = (dim + 1) / 2 * 2;
        for (int i = 0; i < count; i++)
        {
            var row = halves.Slice(CheckIndex(iv[i], vocabulary) * stride, dim);
            var target = yv.AsSpan(i * dim, dim);
            for (int j = 0; j < dim; j++)
            {
                target[j] = BitConverter.Int32BitsToSingle(row[j] << 16);
            }
        }
    }

    public override void GatherBFloat16ColumnsKernel(Storage packed, Storage indices, Storage y, int count, int dim, int vocabulary)
    {
        var halves = System.Runtime.InteropServices.MemoryMarshal.Cast<float, ushort>(D(packed).AsSpan());
        float[] iv = D(indices), yv = D(y);
        int stride = (vocabulary + 1) / 2 * 2;
        for (int i = 0; i < count; i++)
        {
            int column = CheckIndex(iv[i], vocabulary);
            var target = yv.AsSpan(i * dim, dim);
            for (int j = 0; j < dim; j++)
            {
                target[j] = BitConverter.Int32BitsToSingle(halves[j * stride + column] << 16);
            }
        }
    }

    public override void ScatterAddKernel(Storage dy, Storage indices, Storage dtable, int count, int dim, int vocabulary)
    {
        float[] gv = D(dy), iv = D(indices), tv = D(dtable);
        for (int i = 0; i < count; i++)
        {
            int index = CheckIndex(iv[i], vocabulary);
            var target = tv.AsSpan(index * dim, dim);
            AddInPlace(target, gv.AsSpan(i * dim, dim));
        }
    }

    private static int CheckIndex(float value, int vocabulary)
    {
        int index = (int)value;
        if ((uint)index >= (uint)vocabulary || index != value)
        {
            throw new IndexOutOfRangeException($"Embedding index {value} is not an integer in [0, {vocabulary}).");
        }

        return index;
    }

    private static void AddInPlace(Span<float> target, ReadOnlySpan<float> source)
    {
        var tv = MemoryMarshal.Cast<float, Vector<float>>(target);
        var sv = MemoryMarshal.Cast<float, Vector<float>>(source);
        for (int i = 0; i < tv.Length; i++)
        {
            tv[i] += sv[i];
        }

        for (int i = tv.Length * Vector<float>.Count; i < target.Length; i++)
        {
            target[i] += source[i];
        }
    }

    public override void Im2ColKernel(Storage x, Storage cols, in ConvGeometry g)
    {
        float[] xv = D(x), cv = D(cols);
        var geo = g;
        For(g.N * g.OH, (long)g.Positions * g.PatchSize, (start, end) => Im2ColRows(xv, cv, geo, start, end));
    }

    /// <summary>Fills the column rows of output rows [start, end) of (n, oh); each patch row is copied as contiguous runs.</summary>
    private static void Im2ColRows(float[] xv, float[] cv, ConvGeometry g, int start, int end)
    {
        int c = g.C, h = g.H, w = g.W, kh = g.KH, kw = g.KW, sh = g.SH, sw = g.SW, ph = g.PH, pw = g.PW, oh = g.OH, ow = g.OW, patch = g.PatchSize;
        int dh = g.DH, dw = g.DW;
        for (int noh = start; noh < end; noh++)
        {
            int n = noh / oh, y = noh % oh;
            for (int x0 = 0; x0 < ow; x0++)
            {
                var row = cv.AsSpan((noh * ow + x0) * patch, patch);
                int col = 0;
                int iw0 = x0 * sw - pw;
                for (int ch = 0; ch < c; ch++)
                {
                    int plane = (n * c + ch) * h;
                    for (int ki = 0; ki < kh; ki++, col += kw)
                    {
                        int ih = y * sh - ph + ki * dh;
                        var target = row.Slice(col, kw);
                        if ((uint)ih >= (uint)h)
                        {
                            target.Clear();
                            continue;
                        }

                        if (dw == 1 && iw0 >= 0 && iw0 + kw <= w)
                        {
                            xv.AsSpan((plane + ih) * w + iw0, kw).CopyTo(target);
                            continue;
                        }

                        for (int kj = 0; kj < kw; kj++)
                        {
                            int iw = iw0 + kj * dw;
                            target[kj] = (uint)iw < (uint)w ? xv[(plane + ih) * w + iw] : 0f;
                        }
                    }
                }
            }
        }
    }


    public override void Col2ImKernel(Storage dcols, Storage dx, in ConvGeometry g)
    {
        float[] cv = D(dcols), dv = D(dx);
        var geo = g;

        // Images write to disjoint parts of dx, so they can run in parallel without atomics.
        For(g.N, (long)g.Positions * g.PatchSize, (start, end) => Col2ImImages(cv, dv, geo, start, end));
    }

    private static void Col2ImImages(float[] cv, float[] dv, ConvGeometry g, int start, int end)
    {
        int c = g.C, h = g.H, w = g.W, kh = g.KH, kw = g.KW, sh = g.SH, sw = g.SW, ph = g.PH, pw = g.PW, oh = g.OH, ow = g.OW, patch = g.PatchSize;
        int dh = g.DH, dw = g.DW;
        for (int n = start; n < end; n++)
        {
            for (int y = 0; y < oh; y++)
            {
                for (int x0 = 0; x0 < ow; x0++)
                {
                    var row = cv.AsSpan(((n * oh + y) * ow + x0) * patch, patch);
                    int col = 0;
                    int iw0 = x0 * sw - pw;
                    for (int ch = 0; ch < c; ch++)
                    {
                        int plane = (n * c + ch) * h;
                        for (int ki = 0; ki < kh; ki++, col += kw)
                        {
                            int ih = y * sh - ph + ki * dh;
                            if ((uint)ih >= (uint)h)
                            {
                                continue;
                            }

                            var source = row.Slice(col, kw);
                            if (dw == 1 && iw0 >= 0 && iw0 + kw <= w)
                            {
                                AddInPlace(dv.AsSpan((plane + ih) * w + iw0, kw), source);
                                continue;
                            }

                            for (int kj = 0; kj < kw; kj++)
                            {
                                int iw = iw0 + kj * dw;
                                if ((uint)iw < (uint)w)
                                {
                                    dv[(plane + ih) * w + iw] += source[kj];
                                }
                            }
                        }
                    }
                }
            }
        }
    }


    public override void MaxPoolKernel(Storage x, Storage y, Storage argmax, in ConvGeometry g)
    {
        float[] xv = D(x), yv = D(y), av = D(argmax);
        var geo = g;
        For(g.N * g.C, (long)g.N * g.C * g.OH * g.OW * g.KH * g.KW, (start, end) => MaxPoolPlanes(xv, yv, av, geo, start, end));
    }

    // Kernel bodies live in static methods so loop bounds are register locals, not closure fields.
    private static void MaxPoolPlanes(float[] xv, float[] yv, float[] av, ConvGeometry g, int start, int end)
    {
        int h = g.H, w = g.W, kh = g.KH, kw = g.KW, sh = g.SH, sw = g.SW, ph = g.PH, pw = g.PW, oh = g.OH, ow = g.OW;
        var indices = MemoryMarshal.Cast<float, int>(av.AsSpan());
        for (int nc = start; nc < end; nc++)
        {
            int planeOffset = nc * h * w;
            var plane = xv.AsSpan(planeOffset, h * w);
            int o = nc * oh * ow;
            for (int y0 = 0; y0 < oh; y0++)
            {
                int r0 = y0 * sh - ph;
                int rStart = Math.Max(r0, 0), rEnd = Math.Min(r0 + kh, h);
                for (int x0 = 0; x0 < ow; x0++, o++)
                {
                    int c0 = x0 * sw - pw;
                    int cStart = Math.Max(c0, 0), cEnd = Math.Min(c0 + kw, w);
                    float best = float.NegativeInfinity;
                    int bestIndex = 0;
                    for (int r = rStart; r < rEnd; r++)
                    {
                        int rowOffset = r * w;
                        for (int c = cStart; c < cEnd; c++)
                        {
                            // Branch-free arg-max: a data-dependent branch here mispredicts about half the
                            // time on real activations (measured 3x slower), so select the index with a bit mask.
                            float v = plane[rowOffset + c];
                            bool greater = v > best;
                            int mask = -Unsafe.As<bool, byte>(ref greater);
                            bestIndex = (bestIndex & ~mask) | ((rowOffset + c) & mask);
                            best = MathF.Max(best, v);
                        }
                    }

                    yv[o] = best;
                    indices[o] = planeOffset + bestIndex;
                }
            }
        }
    }



    public override void MaxPoolBackwardKernel(Storage dy, Storage argmax, Storage dx, int count)
    {
        float[] gv = D(dy), dv = D(dx);
        var indices = MemoryMarshal.Cast<float, int>(D(argmax).AsSpan(0, count));
        for (int i = 0; i < count; i++)
        {
            dv[indices[i]] += gv[i];
        }
    }

    public override void PermuteKernel(Storage x, Storage y, ReadOnlySpan<int> outShape, ReadOnlySpan<int> inStrides, bool accumulate)
    {
        float[] xv = D(x), yv = D(y);
        int rank = outShape.Length;
        int[] shape = outShape.ToArray(), strides = inStrides.ToArray();
        int total = 1;
        foreach (int d in shape)
        {
            total *= d;
        }

        // The innermost output dimension is walked with a fixed input stride; outer coordinates are decoded once per row.
        int last = shape[rank - 1], lastStride = strides[rank - 1];
        int rows = total / Math.Max(last, 1);
        For(rows, total, (start, end) =>
        {
            for (int r = start; r < end; r++)
            {
                int rem = r, offset = 0;
                for (int d = rank - 2; d >= 0; d--)
                {
                    offset += rem % shape[d] * strides[d];
                    rem /= shape[d];
                }

                int o = r * last;
                if (accumulate)
                {
                    for (int j = 0; j < last; j++)
                    {
                        yv[o + j] += xv[offset + j * lastStride];
                    }
                }
                else
                {
                    for (int j = 0; j < last; j++)
                    {
                        yv[o + j] = xv[offset + j * lastStride];
                    }
                }
            }
        });
    }

    public override void Copy2D(Storage src, int srcOffset, int srcStride, Storage dst, int dstOffset, int dstStride, int rows, int cols, bool accumulate)
    {
        float[] sv = D(src), dv = D(dst);
        for (int r = 0; r < rows; r++)
        {
            var from = sv.AsSpan(srcOffset + r * srcStride, cols);
            var to = dv.AsSpan(dstOffset + r * dstStride, cols);
            if (accumulate)
            {
                AddInPlace(to, from);
            }
            else
            {
                from.CopyTo(to);
            }
        }
    }

    public override void SumAxisKernel(Storage x, Storage y, int outer, int dim, int inner, float scale, bool accumulate)
    {
        float[] xv = D(x), yv = D(y);
        For(outer, (long)outer * dim * inner, (start, end) =>
        {
            Span<float> acc = inner <= 4096 ? stackalloc float[inner] : new float[inner];
            for (int o = start; o < end; o++)
            {
                acc.Clear();
                for (int d = 0; d < dim; d++)
                {
                    AddInPlace(acc, xv.AsSpan((o * dim + d) * inner, inner));
                }

                var target = yv.AsSpan(o * inner, inner);
                for (int i = 0; i < inner; i++)
                {
                    target[i] = accumulate ? target[i] + scale * acc[i] : scale * acc[i];
                }
            }
        });
    }

    public override void BroadcastAxisKernel(Storage dy, Storage dx, int outer, int dim, int inner, float scale)
    {
        float[] gv = D(dy), dv = D(dx);
        For(outer, (long)outer * dim * inner, (start, end) =>
        {
            for (int o = start; o < end; o++)
            {
                var source = gv.AsSpan(o * inner, inner);
                for (int d = 0; d < dim; d++)
                {
                    var target = dv.AsSpan((o * dim + d) * inner, inner);
                    for (int i = 0; i < inner; i++)
                    {
                        target[i] += scale * source[i];
                    }
                }
            }
        });
    }

    // ------------------------------------------------------------ element-wise kernels

    private readonly struct ExpLoop(float[] x, float[] y) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var xs = x.AsSpan(start, end - start);
            var ys = y.AsSpan(start, end - start);
            var xv = MemoryMarshal.Cast<float, Vector<float>>(xs);
            var yv = MemoryMarshal.Cast<float, Vector<float>>(ys);
            for (int i = 0; i < xv.Length; i++)
            {
                yv[i] = Vector.Exp(xv[i]);
            }

            for (int i = xv.Length * Vector<float>.Count; i < xs.Length; i++)
            {
                ys[i] = MathF.Exp(xs[i]);
            }
        }
    }

    private readonly struct LogLoop(float[] x, float[] y) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var xs = x.AsSpan(start, end - start);
            var ys = y.AsSpan(start, end - start);
            var xv = MemoryMarshal.Cast<float, Vector<float>>(xs);
            var yv = MemoryMarshal.Cast<float, Vector<float>>(ys);
            for (int i = 0; i < xv.Length; i++)
            {
                yv[i] = Vector.Log(xv[i]);
            }

            for (int i = xv.Length * Vector<float>.Count; i < xs.Length; i++)
            {
                ys[i] = MathF.Log(xs[i]);
            }
        }
    }

    private readonly struct LogBackwardLoop(float[] x, float[] dy, float[] dx) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            for (int i = start; i < end; i++)
            {
                dx[i] += dy[i] / x[i];
            }
        }
    }

    // GELU, tanh approximation: 0.5 x (1 + tanh(k (x + 0.044715 x³))), k = sqrt(2/π).
    private const float GeluK = 0.7978845608f;
    private const float GeluC = 0.044715f;

    /// <summary>tanh(u) = 1 - 2 / (e^(2u) + 1), vectorized; saturates cleanly to ±1.</summary>
    private static Vector<float> TanhVector(Vector<float> u) =>
        Vector<float>.One - new Vector<float>(2f) / (Vector.Exp(u + u) + Vector<float>.One);

    private readonly struct GeluLoop(float[] x, float[] y) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var xs = x.AsSpan(start, end - start);
            var ys = y.AsSpan(start, end - start);
            var xv = MemoryMarshal.Cast<float, Vector<float>>(xs);
            var yv = MemoryMarshal.Cast<float, Vector<float>>(ys);
            var k = new Vector<float>(GeluK);
            var c = new Vector<float>(GeluC);
            var half = new Vector<float>(0.5f);
            for (int i = 0; i < xv.Length; i++)
            {
                var v = xv[i];
                var t = TanhVector(k * Vector.FusedMultiplyAdd(c * v * v, v, v));
                yv[i] = half * v * (Vector<float>.One + t);
            }

            for (int i = xv.Length * Vector<float>.Count; i < xs.Length; i++)
            {
                float v = xs[i];
                ys[i] = 0.5f * v * (1f + MathF.Tanh(GeluK * (v + GeluC * v * v * v)));
            }
        }
    }

    private readonly struct GeluBackwardLoop(float[] x, float[] dy, float[] dx) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var xs = x.AsSpan(start, end - start);
            var gs = dy.AsSpan(start, end - start);
            var ds = dx.AsSpan(start, end - start);
            var xv = MemoryMarshal.Cast<float, Vector<float>>(xs);
            var gv = MemoryMarshal.Cast<float, Vector<float>>(gs);
            var dv = MemoryMarshal.Cast<float, Vector<float>>(ds);
            var k = new Vector<float>(GeluK);
            var c = new Vector<float>(GeluC);
            var c3 = new Vector<float>(3f * GeluC);
            var half = new Vector<float>(0.5f);
            var one = Vector<float>.One;
            for (int i = 0; i < xv.Length; i++)
            {
                var v = xv[i];
                var v2 = v * v;
                var t = TanhVector(k * Vector.FusedMultiplyAdd(c * v2, v, v));
                var derivative = half * (one + t) + half * v * (one - t * t) * k * Vector.FusedMultiplyAdd(c3, v2, one);
                dv[i] = Vector.FusedMultiplyAdd(gv[i], derivative, dv[i]);
            }

            for (int i = xv.Length * Vector<float>.Count; i < xs.Length; i++)
            {
                float v = xs[i];
                float t = MathF.Tanh(GeluK * (v + GeluC * v * v * v));
                float derivative = 0.5f * (1f + t) + 0.5f * v * (1f - t * t) * GeluK * (1f + 3f * GeluC * v * v);
                ds[i] += gs[i] * derivative;
            }
        }
    }
}
